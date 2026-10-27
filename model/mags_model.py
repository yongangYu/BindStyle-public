#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


from typing import NamedTuple
import json
import numpy as np
import torch
from simple_knn._C import distCUDA2
from torch import nn
from torch.nn import functional as F

from utils.graphics_utils import uvw_to_value, uvw_to_xyz
from utils.general_utils import (
    get_expon_lr_func,
    inverse_sigmoid,
)
from utils.sh_utils import RGB2SH

import os
from plyfile import PlyData, PlyElement
from utils.graphics_utils import compute_face_orientation

from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from roma import quat_product, quat_wxyz_to_xyzw

from utils.general_utils import (
    build_scaling_rotation,
    strip_symmetric,
)


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class MaGSModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.device = config.dataset.data_device

        self.active_sh_degree = 0
        self.max_sh_degree = config.model.sh_degree
        self.setup_functions()

        # mesh related parameters
        self.refined_verts = torch.Tensor()
        self.refined_norms = torch.Tensor()
        self.mesh_faces = torch.Tensor()
        self.mesh_feat = torch.Tensor()

        # learnable parameters
        self._xyz = torch.Tensor()
        self._uvw = torch.Tensor()
        self._scaling = torch.Tensor()
        self._rotation = torch.Tensor()
        self._opacity = torch.Tensor()
        self._features_dc = torch.Tensor()
        self._features_rest = torch.Tensor()

        # unlearnable parameters
        self.d_uvw = torch.Tensor()
        self.xyz = torch.Tensor()
        self.faceid = torch.Tensor()
        self.d_xyz = None
        self.d_rotation = None
        self.d_scaling = None
        self.d_opacity = None
        self.d_color = None
        self.style_features_delta = None
        self.style_uvw_delta = None
        self.style_log_scaling_delta = None
        self.style_rotation_delta = None
        self.pretrained_exposures = None
        self.exposure_mapping = {}

        self.GS_PARAMS = [
            "_uvw",
            "_scaling",
            "_rotation",
            "_opacity",
            "_features_dc",
            "_features_rest",
            "xyz",
            "faceid",
        ]

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def apply_ct(self, color_ct: np.ndarray):
        sh_data = self.get_features.data.detach().cpu().numpy()
        cnt = sh_data.shape[0]
        sh_data = sh_data.reshape((-1, 3))
        color_ct = color_ct.astype(np.float32)
        sh_data = (
            sh_data @ color_ct[:3, :3].T + color_ct[:3, 3][np.newaxis, :]
        ).astype(np.float32)
        sh_data = torch.from_numpy(sh_data.reshape(cnt, -1, 3)).to(self.device)
        self._features_dc.data.copy_(sh_data[:, 0:1, :])
        self._features_rest.data.copy_(sh_data[:, 1:, :])

    def _set_xyz_cache(self, xyz):
        if "xyz" in self._parameters:
            del self._parameters["xyz"]
        self.__dict__["xyz"] = xyz

    @property
    def num_gauss(self):
        return self.get_xyz.shape[0]

    @property
    def get_xyz(self):
        return self.xyz if self.xyz.numel() > 0 else self._xyz

    @property
    def is_mesh_bound(self):
        return (
            self.faceid.numel() > 0
            and self.mesh_faces.numel() > 0
            and hasattr(self, "face_orien_quat")
            and hasattr(self, "face_scaling")
        )

    @property
    def get_cano_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation(self):
        rotation = self._rotation
        if self.d_rotation is not None:
            rotation = rotation + uvw_to_value(
                self.mesh_faces, self.d_rotation, self.faceid, self.get_uvw
            )
        rot = self.rotation_activation(rotation)
        if self.style_rotation_delta is not None:
            rot = quat_xyzw_to_wxyz(
                quat_product(
                    quat_wxyz_to_xyzw(rot),
                    quat_wxyz_to_xyzw(self.style_rotation_delta),
                )
            )
        if not self.is_mesh_bound:
            return rot
        face_orien_quat = self.rotation_activation(self.face_orien_quat[self.faceid])
        return quat_xyzw_to_wxyz(
            quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot))
        )

    @property
    def get_cano_scaling(self):
        return self.scaling_activation(self._scaling.clamp(-20.0, 8.0))

    @property
    def get_scaling(self):
        scaling = self._scaling
        if self.d_scaling is not None:
            scaling = scaling + uvw_to_value(
                self.mesh_faces, self.d_scaling, self.faceid, self.get_uvw
            )
        if self.style_log_scaling_delta is not None:
            scaling = scaling + self.style_log_scaling_delta
        scaling = scaling.clamp(-20.0, 8.0)
        scaling = self.scaling_activation(scaling)
        if not self.is_mesh_bound:
            return scaling
        return scaling * self.face_scaling[self.faceid]

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        if self.d_color is not None:
            delta = uvw_to_value(
                self.mesh_faces, self.d_color, self.faceid, self.get_uvw
            ).reshape(features_dc.shape)
            features_dc = features_dc + delta
        features = torch.cat((features_dc, features_rest), dim=1)
        if self.style_features_delta is not None:
            features = features + self.style_features_delta.to(
                device=features.device,
                dtype=features.dtype,
            )
        return features

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_ins_feat(self):
        ins_feat = uvw_to_value(
            self.mesh_faces, self.mesh_feat, self.faceid, self.get_uvw
        )
        return torch.nn.functional.normalize(ins_feat, dim=1)

    @property
    def get_uvw(self):
        uvw = self._uvw + (self.d_uvw if self.d_uvw is not None else 0)
        if self.style_uvw_delta is not None:
            uvw = uvw + self.style_uvw_delta
        return uvw

    @property
    def get_refined_verts(self):
        return self.refined_verts + (self.d_xyz if self.d_xyz is not None else 0)

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos=None, spatial_lr_scale=None):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        fused_color = RGB2SH(
            torch.tensor(np.asarray(pcd.colors)).float().to(self.device)
        )
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .to(self.device)
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._set_xyz_cache(torch.empty(0, device=self.device))
        if self._uvw.numel() == 0:
            self._uvw = nn.Parameter(
                torch.zeros((fused_point_cloud.shape[0], 3), device=self.device),
                requires_grad=True,
            )
        if self.faceid.numel() == 0:
            self.faceid = torch.zeros(
                (fused_point_cloud.shape[0],), dtype=torch.long, device=self.device
            )
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._uvw.shape[0]), device=self.device)

    def create_from_mesh(self, mesh_vertices, mesh_faces, mesh_normals):
        self.refined_verts = mesh_vertices.to(self.device)
        self.mesh_faces = mesh_faces.to(self.device)
        self.refined_norms = mesh_normals.to(self.device)
        mesh_feat = torch.rand((mesh_vertices.shape[0], 3), dtype=torch.float).to(
            self.device
        )
        self.mesh_feat = nn.Parameter(mesh_feat.requires_grad_(True))

        mesh_alpha = self.config.dataset.mesh_alpha
        num_samples = mesh_alpha * self.mesh_faces.shape[0]
        uvw = torch.zeros(num_samples, 3)
        uvw[:, 0] = torch.rand(num_samples)
        uvw[:, 1] = torch.rand(num_samples) * (1.0 - uvw[:, 0])
        uvw[:, 2] = 0
        self.faceid = (
            torch.arange(self.mesh_faces.shape[0]).repeat(mesh_alpha).to(self.device)
        )

        self._uvw = nn.Parameter(uvw.to(self.device), requires_grad=True)
        xyz = uvw_to_xyz(
            self.mesh_faces,
            self.refined_verts,
            self.refined_norms,
            self.faceid,
            self._uvw,
        )
        norms = self.refined_norms[self.faceid].float()
        norms = F.normalize(norms, dim=-1)

        pcd = BasicPointCloud(
            points=xyz.detach().cpu().numpy(),
            normals=norms.detach().cpu().numpy(),
            colors=torch.full_like(xyz, 0.5).float().cpu(),
        )
        self.create_from_pcd(pcd)
        self.d_uvw = torch.zeros_like(self._uvw)
        self._set_xyz_cache(xyz)

    def update_mesh(
        self,
        refined_faces,
        refined_verts,
        refined_norms,
        refined_cano_verts=None,
        coarse_cano_verts=None,
        coarse_deform_verts=None,
        deformer=None,
        detach_mesh=True,
        deformation_components=None,
    ):
        if detach_mesh:
            self.refined_verts = refined_verts.detach()
            self.refined_norms = refined_norms.detach()
        else:
            self.refined_verts = refined_verts
            self.refined_norms = refined_norms
        self.mesh_faces = refined_faces.detach()
        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            self.refined_verts.squeeze(0), self.mesh_faces.squeeze(0)
        )
        self.face_orien_quat = quat_xyzw_to_wxyz(
            rotmat_to_unitquat(self.face_orien_mat)
        )
        if deformer is not None:
            delta = deformer(
                refined_cano_verts,
                self.mesh_feat,
                self.mesh_faces,
                self.faceid,
                self._uvw,
                coarse_cano_verts,
                coarse_deform_verts,
            )
            component_names = {
                "d_uvw",
                "d_xyz",
                "d_rotation",
                "d_scaling",
                "d_opacity",
                "d_color",
            }
            enabled_components = (
                component_names
                if deformation_components is None
                else set(deformation_components)
            )
            unknown_components = enabled_components - component_names
            if unknown_components:
                raise ValueError(
                    "Unknown deformation components: "
                    + ", ".join(sorted(unknown_components))
                )
            self.d_uvw = delta["d_uvw"] if "d_uvw" in enabled_components else None
            self.d_xyz = delta["d_xyz"] if "d_xyz" in enabled_components else None
            self.d_rotation = (
                delta["d_rotation"]
                if "d_rotation" in enabled_components
                else None
            )
            self.d_scaling = (
                delta["d_scaling"]
                if "d_scaling" in enabled_components
                else None
            )
            self.d_opacity = (
                delta["d_opacity"]
                if "d_opacity" in enabled_components
                else None
            )
            self.d_color = (
                delta["d_color"] if "d_color" in enabled_components else None
            )
        else:
            self.d_uvw = self.d_xyz = self.d_rotation = self.d_scaling = (
                self.d_opacity
            ) = self.d_color = None

        self._set_xyz_cache(uvw_to_xyz(
            self.mesh_faces,
            self.get_refined_verts,
            self.refined_norms,
            self.faceid,
            self.get_uvw,
        ))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.num_gauss, 1), device=self.device)
        self.denom = torch.zeros((self.num_gauss, 1), device=self.device)

        self.spatial_lr_scale = training_args.spatial_lr_scale

        l = [
            {
                "params": [self._uvw],
                "lr": training_args.optim_uvw.lr * self.spatial_lr_scale,
                "name": "_uvw",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.optim_features.lr,
                "name": "_features_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.optim_features.lr / 20.0,
                "name": "_features_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.optim_opacity.lr,
                "name": "_opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.optim_scaling.lr * self.spatial_lr_scale,
                "name": "_scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.optim_rotation.lr,
                "name": "_rotation",
            },
        ]

        print("Learning rates: ", [group["lr"] for group in l])

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.optim_uvw.scheduler_args.lr_init
            * self.spatial_lr_scale,
            lr_final=training_args.optim_uvw.scheduler_args.lr_final
            * self.spatial_lr_scale,
            lr_delay_mult=training_args.optim_uvw.scheduler_args.lr_delay_mult,
            max_steps=training_args.optim_uvw.scheduler_args.lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "_uvw":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def load_ply(self, path, use_train_test_exp=False, reset_basis_dim=0):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(
                os.path.dirname(path), os.pardir, os.pardir, "exposure.json"
            )
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {
                    image_name: torch.FloatTensor(exposures[image_name])
                    .requires_grad_(False)
                    .to(self.device)
                    for image_name in exposures
                }
                print("Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self._set_xyz_cache(torch.empty(0, device=self.device))
        self._uvw = nn.Parameter(
            torch.zeros((xyz.shape[0], 3), dtype=torch.float, device=self.device),
            requires_grad=True,
        )
        self.faceid = torch.zeros((xyz.shape[0],), dtype=torch.long, device=self.device)
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self.max_radii2D = torch.zeros((xyz.shape[0]), device=self.device)
        self.active_sh_degree = reset_basis_dim if reset_basis_dim else self.max_sh_degree

        return xyz, opacities, scales

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(
            opacities_new, "_opacity"
        )
        self._opacity = optimizable_tensors["_opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state[group["params"][0]]
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        for key, value in optimizable_tensors.items():
            setattr(self, key, value)
        for key in ["xyz", "faceid", "xyz_gradient_accum", "denom", "max_radii2D"]:
            setattr(self, key, getattr(self, key)[valid_points_mask])

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] not in tensors_dict:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, d):
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        for key, value in optimizable_tensors.items():
            setattr(self, key, value)

        self.xyz_gradient_accum = torch.zeros((self.num_gauss, 1), device=self.device)
        self.denom = torch.zeros((self.num_gauss, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.num_gauss), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.num_gauss
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_cano_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        # TODO: new_uvw = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._uvw[selected_pts_mask].repeat(N, 1)
        new_uvw = self._uvw[selected_pts_mask].repeat(N, 1) + torch.normal(
            0.0, 0.01, size=(3,), device=self.device
        )
        new_scaling = self.scaling_inverse_activation(
            self.get_cano_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        d = {"_uvw": new_uvw, "_scaling": new_scaling}
        for param in self.GS_PARAMS:
            if param in ["_features_dc", "_features_rest"]:
                d[param] = getattr(self, param)[selected_pts_mask].repeat(N, 1, 1)
            elif param not in ["_uvw", "_scaling"]:
                d[param] = getattr(self, param)[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(d)

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device=self.device, dtype=bool
                ),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_cano_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        d = {param: getattr(self, param)[selected_pts_mask] for param in self.GS_PARAMS}

        self.densification_postfix(d)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_cano_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(
        self, viewspace_point_tensor, update_filter, width=0, height=0
    ):
        # self.xyz_gradient_accum[update_filter] += torch.norm(
        #     viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        # )
        grad = viewspace_point_tensor.grad.squeeze(0)  # [N, 2]
        # Normalize the gradient to [-1, 1] screen size
        # grad[:, 0] *= width * 0.5
        # grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[update_filter] += torch.norm(
            grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def save_ply(self, path):
        with torch.no_grad():
            os.makedirs(os.path.dirname(path), exist_ok=True)

            xyz = self.get_xyz.detach()
            normals = torch.zeros_like(xyz, device=self.device)
            features = self.get_features.detach()
            f_dc = (
                features[:, 0:1, :]
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
            )
            f_rest = (
                features[:, 1:, :]
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
            )
            opacities = self._opacity.detach()
            scale = self._scaling.detach()
            rotation = self._rotation.detach()

            if self.d_scaling is not None:
                delta = uvw_to_value(
                    self.mesh_faces, self.d_scaling, self.faceid, self.get_uvw
                )
                scale = scale + delta.detach()

            if self.d_rotation is not None:
                delta = uvw_to_value(
                    self.mesh_faces, self.d_rotation, self.faceid, self.get_uvw
                )
                rotation = rotation + delta.detach()

            attributes = torch.cat(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation), dim=1
            )
            attributes_np = attributes.cpu().numpy()

            dtype_full = [
                (attribute, "f4") for attribute in self.construct_list_of_attributes()
            ]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            elements[:] = list(map(tuple, attributes_np))
            el = PlyElement.describe(elements, "vertex")
            PlyData([el]).write(path)
