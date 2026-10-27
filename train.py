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

import datetime
import os
import random
import sys

import cv2

# import lpips
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from scene import Scene
from dataset import get_dataset, getNerfppNorm
from gaussian_renderer import inria_render
from model.deform_model import DeformModel
from model.mags_model import MaGSModel
from model.style_gaussian_geometry import CanonicalGaussianGeometryResidual
from utils.general_utils import safe_state
from utils.loss_utils import arap_loss, l1_loss, ssim, psnr
from utils.mesh_style_poisson_aligner import MeshStylePoissonAligner
from utils.sds_guidance import build_sds_guidance
from utils.style_numerics import check_backward_finite, require_finite
from utils.direct_style_io import assert_separate_style_output, create_direct_style_output

lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()

best_psnr = 0.0
best_l1 = 1e10
best_lpips = 1e10
best_iter = 0

cnt = 0


def limit_dataset_frames(config, dataset, train=True):
    split_name = "train" if train else "test"
    split_limit = config.dataset.get(f"{split_name}_frame_limit", None)
    frame_limit = split_limit
    if frame_limit is None:
        frame_limit = config.dataset.get("frame_limit", 0)
    frame_limit = int(frame_limit)
    if frame_limit <= 0:
        return dataset

    limited_dataset = dataset[: min(frame_limit, len(dataset))]
    print(
        f"[MaGS] Using first {len(limited_dataset)}/{len(dataset)} "
        f"{split_name} frames."
    )
    return limited_dataset


def load_training_config(argv):
    if not argv:
        raise ValueError(
            "Usage: python train.py config/base.yaml,config/scene.yaml "
            "[optim.mesh_style_update.enabled=true ...]"
        )
    configs = [OmegaConf.load(path) for path in argv[0].split(",")]
    config = OmegaConf.merge(*configs)
    if len(argv) > 1:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(argv[1:]))
    return config


class MeshStyleUpdateController:
    def __init__(
        self,
        config,
        base_vertices,
        faces,
        device,
        gaussians=None,
        deformer=None,
        checkpoint=None,
        train_dataset=None,
        shared_sds_guidance=None,
    ):
        mesh_style_cfg = config.optim.get("mesh_style_update", {})
        if not mesh_style_cfg or not mesh_style_cfg.get("enabled", False):
            self.enabled = False
            print("[MaGS] Mesh-style Poisson update disabled.")
            return

        self.enabled = True
        self.cfg = mesh_style_cfg
        self.device = device
        self.gaussians = gaussians
        self.deformer = deformer

        self.start_iter = max(1, int(mesh_style_cfg.get("start_iter", 0)))
        self.end_iter = int(mesh_style_cfg.get("end_iter", 0))
        if self.end_iter <= 0:
            self.end_iter = int(config.optim.iterations)

        self.style_weight = float(
            mesh_style_cfg.get(
                "style_weight", mesh_style_cfg.get("sds_weight", 1.0)
            )
        )
        self.jacobian_weight = float(mesh_style_cfg.get("jacobian_weight", 10.0))
        self.poisson_weight = float(mesh_style_cfg.get("poisson_weight", 1.0))
        self.flip_weight = float(mesh_style_cfg.get("flip_weight", 100.0))
        self.delta_weight = float(mesh_style_cfg.get("delta_weight", 0.0))
        self.grad_clip = float(mesh_style_cfg.get("grad_clip", 1.0))
        self.apply_to_gaussian = bool(mesh_style_cfg.get("apply_to_gaussian", True))
        self.freeze_gaussians = bool(mesh_style_cfg.get("freeze_gaussians", True))
        self.freeze_deformer = bool(mesh_style_cfg.get("freeze_deformer", True))
        self.use_deformer = bool(mesh_style_cfg.get("use_deformer", True))
        self.window_size = max(1, int(mesh_style_cfg.get("window_size", 4)))
        self.shared_noise_seed = int(mesh_style_cfg.get("shared_noise_seed", 4321))
        if not self.apply_to_gaussian:
            raise ValueError(
                "Dynamic Jacobian style training requires apply_to_gaussian=true "
                "so CSD is evaluated on the final Gaussian render."
            )
        if not self.freeze_gaussians or not self.freeze_deformer:
            raise ValueError(
                "Dynamic Jacobian style training requires freeze_gaussians=true "
                "and freeze_deformer=true; style_log_jac is the only trainable state."
            )

        self.aligner = MeshStylePoissonAligner(
            base_vertices,
            faces,
            device=device,
            laplacian_jitter=float(mesh_style_cfg.get("laplacian_jitter", 1e-8)),
        )
        self.sds_guidance = shared_sds_guidance
        if self.sds_guidance is None:
            self.sds_guidance = build_sds_guidance(mesh_style_cfg, device).to(device)
        if not self.sds_guidance.enabled:
            raise ValueError(
                "mesh_style_update requires an enabled diffusion guidance model."
            )
        if getattr(self.sds_guidance, "gradient_mode", "sds") != "csd":
            raise ValueError(
                "Dynamic Jacobian style training requires sds.gradient_mode=csd."
            )
        if train_dataset is None:
            raise ValueError("Mesh CSD requires the training content frames.")
        if shared_sds_guidance is None:
            self.sds_guidance.cache_source_images(
                {cam.uid: cam.original_image for cam in train_dataset}
            )
        self.optimizer = torch.optim.Adam(
            [self.aligner.style_log_jac],
            lr=float(mesh_style_cfg.get("lr", 0.0005)),
        )

        checkpoint = checkpoint or {}
        saved_log_jac = checkpoint.get("mesh_style_log_jac", None)
        self.loop_start_iteration = 1
        if saved_log_jac is not None:
            if saved_log_jac.shape != self.aligner.style_log_jac.shape:
                raise ValueError("Checkpoint mesh style Jacobian shape mismatch.")
            self.aligner.style_log_jac.data.copy_(
                saved_log_jac.to(
                    device=self.aligner.style_log_jac.device,
                    dtype=self.aligner.style_log_jac.dtype,
                )
            )
            if bool(mesh_style_cfg.get("resume_state", True)):
                self.loop_start_iteration = int(
                    checkpoint.get("iteration", 0)
                ) + 1
            saved_gradient = checkpoint.get("mesh_style_log_jac_grad", None)
            if saved_gradient is not None:
                self.aligner.style_log_jac.grad = saved_gradient.to(
                    device=self.aligner.style_log_jac.device,
                    dtype=self.aligner.style_log_jac.dtype,
                )
        if "mesh_style_optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["mesh_style_optimizer"])
                print("[MaGS] Restored dynamic mesh-style optimizer.")
            except ValueError as exc:
                print(f"[WARN] Could not restore mesh-style optimizer: {exc}")

        if self.freeze_gaussians and gaussians is not None:
            for param in gaussians.parameters():
                param.requires_grad_(False)
            gaussians.optimizer.zero_grad(set_to_none=True)
        if self.freeze_deformer and deformer is not None:
            deformer.eval()
            for param in deformer.parameters():
                param.requires_grad_(False)
            deformer.optimizer.zero_grad(set_to_none=True)
        self._gradient_path_verified = False

        print(
            "[MaGS] Dynamic canonical Jacobian style update enabled "
            f"(start={self.start_iter}, end={self.end_iter}, "
            f"lr={mesh_style_cfg.get('lr', 0.0005)}, "
            f"window={self.window_size}, "
            f"guidance={getattr(self.sds_guidance, 'gradient_mode', 'sds')}, "
            f"apply_to_gaussian={self.apply_to_gaussian}, "
            f"deformer_forward={self.use_deformer})."
        )

    def should_align(self, iteration):
        return self.enabled and iteration >= self.start_iter

    def should_apply_to_gaussian(self, iteration):
        return self.should_align(iteration) and self.apply_to_gaussian

    def is_training_step(self, iteration):
        return self.enabled and self.start_iter <= iteration <= self.end_iter

    def align(self, verts):
        return self.aligner(verts)

    def guidance_kwargs(self, iteration):
        if getattr(self.sds_guidance, "gradient_mode", "sds") != "csd":
            return {}
        window_index = (iteration - self.start_iter) // self.window_size
        rng = random.Random(self.shared_noise_seed + window_index)
        timestep = rng.randint(
            int(self.sds_guidance.min_step),
            int(self.sds_guidance.max_step),
        )
        return {
            "timesteps": timestep,
            "noise_seed": rng.randint(0, 2**31 - 1),
        }

    def loss(self, align_pkg, gaussian_image, source_key, iteration):
        image_batch = (
            gaussian_image.unsqueeze(0)
            if gaussian_image.dim() == 3
            else gaussian_image
        )
        style_raw_loss, diagnostics = self.sds_guidance(
            image_batch,
            source_key=source_key,
            return_diagnostics=True,
            **self.guidance_kwargs(iteration),
        )
        style_loss = self.style_weight * style_raw_loss
        jacobian_loss = (
            self.jacobian_weight
            * self.aligner.style_log_jac.square().mean()
        )
        poisson_loss = self.poisson_weight * (
            align_pkg["reconstructed_jac"] - align_pkg["target_jac"]
        ).square().mean()
        determinant = torch.linalg.det(align_pkg["style_jac"])
        flip_loss = self.flip_weight * torch.relu(1e-4 - determinant).mean()
        delta_loss = self.delta_weight * (align_pkg["delta"] ** 2).mean()
        total_loss = (
            style_loss
            + jacobian_loss
            + poisson_loss
            + flip_loss
            + delta_loss
        )
        losses = {
            "mesh_style_update_loss": total_loss,
            "mesh_style_update_sds_loss": style_loss,
            "mesh_style_update_sds_raw_loss": style_raw_loss,
            "mesh_style_update_csd_loss": style_loss,
            "mesh_style_update_csd_raw_loss": style_raw_loss,
            "mesh_style_update_jacobian_loss": jacobian_loss,
            "mesh_style_update_poisson_loss": poisson_loss,
            "mesh_style_update_flip_loss": flip_loss,
            "mesh_style_update_delta_loss": delta_loss,
            "mesh_style_update_train_r_mean": gaussian_image.detach().mean(),
            "mesh_style_update_jacobian_rms": self.aligner.style_log_jac
            .detach()
            .float()
            .pow(2)
            .mean()
            .sqrt(),
            "mesh_style_update_min_determinant": determinant.detach().min(),
        }
        losses.update(
            {
                f"mesh_style_update_{name}": value.detach().float()
                for name, value in diagnostics.items()
            }
        )
        return losses

    def is_window_end(self, iteration):
        if not self.enabled:
            return False
        local_step = iteration - self.start_iter + 1
        return (
            self.is_training_step(iteration)
            and (
                local_step % self.window_size == 0
                or iteration == self.end_iter
            )
        )

    def window_divisor(self, iteration):
        window_index = (iteration - self.start_iter) // self.window_size
        window_start = self.start_iter + window_index * self.window_size
        return min(self.window_size, self.end_iter - window_start + 1)

    def verify_gradient_boundary(self):
        gradient = self.aligner.style_log_jac.grad
        if gradient is None:
            raise RuntimeError(
                "Style loss did not reach canonical style_log_jac."
            )
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(
                "Canonical style_log_jac received a non-finite gradient."
            )
        if self._gradient_path_verified:
            return
        leaked = []
        if self.gaussians is not None:
            leaked.extend(
                f"gaussians.{name}"
                for name, param in self.gaussians.named_parameters()
                if param.grad is not None
            )
        if self.deformer is not None:
            leaked.extend(
                f"deformer.{name}"
                for name, param in self.deformer.named_parameters()
                if param.grad is not None
            )
        if leaked:
            raise RuntimeError(
                "Mesh-style gradients escaped style_log_jac: "
                + ", ".join(leaked)
            )
        self._gradient_path_verified = True
        print(
            "[MaGS] Verified geometry style gradient: Gaussian RGB CSD -> "
            "mesh binding -> Poisson -> canonical style_log_jac only."
        )

    def gradient_norm(self):
        gradient = self.aligner.style_log_jac.grad
        if gradient is None:
            return self.aligner.style_log_jac.detach().new_tensor(0.0)
        return gradient.detach().float().norm()

    def step(self):
        if not self.enabled:
            return
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [self.aligner.style_log_jac], self.grad_clip
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def zero_grad(self):
        if self.enabled:
            self.optimizer.zero_grad(set_to_none=True)


def build_mesh_style_update(
    config,
    base_vertices,
    faces,
    device,
    gaussians=None,
    deformer=None,
    checkpoint=None,
    train_dataset=None,
    shared_sds_guidance=None,
):
    return MeshStyleUpdateController(
        config,
        base_vertices,
        faces,
        device,
        gaussians=gaussians,
        deformer=deformer,
        checkpoint=checkpoint,
        train_dataset=train_dataset,
        shared_sds_guidance=shared_sds_guidance,
    )


def _torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_matching_state_dict(module, state_dict, module_name):
    current = module.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    skipped = sorted(set(state_dict) - set(compatible))
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"[WARN] Skipped incompatible {module_name} keys: {skipped}")
    if missing:
        print(f"[WARN] Missing {module_name} keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected {module_name} keys: {unexpected}")


def _restore_gaussian_parameters(gaussians, state_dict):
    variable_params = [
        "_xyz",
        "_uvw",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
        "mesh_feat",
    ]
    for name in variable_params:
        if name not in state_dict:
            continue
        tensor = state_dict[name].detach().clone().to(gaussians.device)
        setattr(gaussians, name, torch.nn.Parameter(tensor.requires_grad_(True)))


def load_pretrained_mags_checkpoint(path, gaussians, deformer, device):
    if not path:
        raise ValueError(
            "A pretrained MaGS start_checkpoint is required when a style update "
            "stage is enabled."
        )
    checkpoint = _torch_load_checkpoint(path, device)
    if not isinstance(checkpoint, dict) or "gaussians" not in checkpoint:
        raise ValueError(
            "The style start checkpoint must contain a MaGS 'gaussians' state_dict."
        )
    if "deformer" not in checkpoint:
        raise ValueError(
            "The style start checkpoint must contain the pretrained MaGS deformer."
        )

    gaussian_state = dict(checkpoint["gaussians"])
    if "_xyz" not in gaussian_state and "xyz" in checkpoint:
        gaussian_state["_xyz"] = checkpoint["xyz"]
    _restore_gaussian_parameters(gaussians, gaussian_state)
    _load_matching_state_dict(gaussians, gaussian_state, "Gaussian")
    _load_matching_state_dict(deformer, checkpoint["deformer"], "Deformer")
    if "faceid" in checkpoint:
        gaussians.faceid = checkpoint["faceid"].long().to(device)
    if gaussians.faceid.shape[0] != gaussians._uvw.shape[0]:
        raise ValueError(
            "Checkpoint Gaussian binding is inconsistent: "
            f"faceid={gaussians.faceid.shape[0]}, uvw={gaussians._uvw.shape[0]}."
        )
    if "xyz" in checkpoint:
        gaussians._set_xyz_cache(checkpoint["xyz"].float().to(device))
    gaussians.active_sh_degree = int(
        checkpoint.get("active_sh_degree", gaussians.max_sh_degree)
    )
    print(
        f"[MaGS] Loaded pretrained dynamic checkpoint: {path} "
        f"(iteration={checkpoint.get('iteration', 'unknown')}, "
        f"gaussians={gaussians._uvw.shape[0]})."
    )
    return checkpoint


class GaussianStyleDelta(torch.nn.Module):
    def __init__(self, features_dc, features_rest, dc_init=None, rest_init=None):
        super().__init__()
        if dc_init is None:
            dc_init = torch.zeros_like(features_dc)
        if rest_init is None:
            rest_init = torch.zeros_like(features_rest)
        if dc_init.shape != features_dc.shape:
            raise ValueError(
                "Style DC delta shape mismatch: "
                f"expected {tuple(features_dc.shape)}, got {tuple(dc_init.shape)}."
            )
        if rest_init.shape != features_rest.shape:
            raise ValueError(
                "Style rest delta shape mismatch: "
                f"expected {tuple(features_rest.shape)}, got {tuple(rest_init.shape)}."
            )
        self.dc_delta = torch.nn.Parameter(dc_init.detach().clone())
        self.rest_delta = torch.nn.Parameter(rest_init.detach().clone())

    def combined(self):
        return torch.cat((self.dc_delta, self.rest_delta), dim=1)


class GaussianSHStyleController:
    def __init__(
        self,
        config,
        gaussians,
        deformer,
        checkpoint,
        device,
        train_dataset=None,
        shared_sds_guidance=None,
        source_reference=None,
    ):
        self.video_guidance = False
        self.split_supervision = False
        self._split_pending = None
        style_cfg = config.optim.get("gaussian_style_update", {})
        if not style_cfg or not style_cfg.get("enabled", False):
            self.enabled = False
            print("[MaGS] Gaussian SH style update disabled.")
            return

        self.enabled = True
        self.cfg = style_cfg
        self.gaussians = gaussians
        self.deformer = deformer
        self.parameterization = str(
            style_cfg.get("parameterization", "direct")
        ).lower()
        if self.parameterization not in {"delta", "direct"}:
            raise ValueError(
                "Gaussian style parameterization must be direct or delta."
            )
        self.direct_mode = self.parameterization == "direct"
        self.supervision_mode = str(style_cfg.get("supervision_mode", "diffusion")).lower()
        if self.supervision_mode not in {"diffusion", "split_vgg_diffusion"}:
            raise ValueError("supervision_mode must be diffusion or split_vgg_diffusion.")
        self.split_supervision = self.supervision_mode == "split_vgg_diffusion"
        if self.split_supervision and not self.direct_mode:
            raise ValueError("split_vgg_diffusion requires parameterization=direct.")
        saved_mode = checkpoint.get("gaussian_style", {}).get("supervision_mode", "diffusion")
        if checkpoint.get("gaussian_style_parameterization") == "direct" and saved_mode != self.supervision_mode:
            raise ValueError("Cannot resume with a different supervision_mode. Start the new experiment from the original MaGS checkpoint.")
        self.source_reference = source_reference
        self.appearance_cfg = style_cfg.get("appearance", {})
        self.appearance_weight = float(self.appearance_cfg.get("weight", 1.0))
        self.appearance_grad_clip = float(self.appearance_cfg.get("grad_clip", 1.0))
        self.source_checkpoint = str(style_cfg.get("start_checkpoint", ""))
        self.pretrained_checkpoint = str(
            checkpoint.get("pretrained_checkpoint", self.source_checkpoint)
        )
        if self.direct_mode:
            assert_separate_style_output(config.model.model_path, self.source_checkpoint)
            assert_separate_style_output(config.model.model_path, self.pretrained_checkpoint)
            if any(key in checkpoint for key in (
                "gaussian_style_delta", "gaussian_style_geometry_residual",
                "gaussian_style_base_dc", "mesh_style_log_jac",
            )) or (
                checkpoint.get("gaussian_style")
                and checkpoint["gaussian_style"].get("parameterization") != "direct"
            ):
                raise ValueError(
                    "Direct mode cannot silently load a residual-style checkpoint. "
                    "Use the original MaGS checkpoint or a direct-style checkpoint."
                )
        elif checkpoint.get("gaussian_style_parameterization") == "direct":
            raise ValueError("A direct-style checkpoint requires parameterization=direct.")
        self.guidance_mode = str(style_cfg.get("guidance_mode", "csd")).lower()
        if self.guidance_mode != "csd":
            raise ValueError(
                "Gaussian style training currently supports guidance_mode=csd only."
            )
        self.start_iter = max(1, int(style_cfg.get("start_iter", 1)))
        self.dc_only_iterations = max(0, int(style_cfg.get("dc_only_iterations", 0)))
        self.sds_weight = float(style_cfg.get("sds_weight", 1.0))
        self.current_sds_weight = self.sds_weight
        self.dc_reg_weight = float(style_cfg.get("dc_reg_weight", 0.0))
        self.rest_reg_weight = float(style_cfg.get("rest_reg_weight", 0.0))
        self.grad_clip = float(style_cfg.get("grad_clip", 1.0))
        self.use_deformer = bool(style_cfg.get("use_deformer", True))
        self.train_sh = bool(style_cfg.get("train_sh", True))
        if self.split_supervision and (not self.train_sh or self.appearance_weight <= 0):
            raise ValueError("Split supervision requires train_sh=true and appearance.weight>0.")
        self.base_lr_dc = float(style_cfg.get("lr_dc", 0.001))
        self.base_lr_rest = float(style_cfg.get("lr_rest", 0.00005))
        geometry_cfg = style_cfg.get(
            "direct_geometry" if self.direct_mode else "canonical_geometry", {}
        )
        self.geometry_cfg = geometry_cfg
        self.geometry_grad_clip = float(geometry_cfg.get("grad_clip", 1.0))
        self.geometry_enabled = bool(
            geometry_cfg and geometry_cfg.get("enabled", False)
        )
        if self.split_supervision and checkpoint.get("gaussian_style_parameterization") == "direct":
            required = {"gaussian_style_optimizer", "split_style_reference", "direct_style_state"}
            if self.geometry_enabled:
                required.add("gaussian_geometry_optimizer")
            missing = required.difference(checkpoint)
            if missing:
                raise ValueError("Incomplete split style checkpoint: " + ", ".join(sorted(missing)))
        self.geometry_start_iter = max(
            self.start_iter,
            int(geometry_cfg.get("start_iter", self.start_iter + 1000)),
        )
        if self.geometry_enabled:
            legacy_schedule_keys = [
                f"{name}_start_iter"
                for name in ("uv", "normal", "scaling", "rotation")
                if f"{name}_start_iter" in geometry_cfg
            ]
            if legacy_schedule_keys:
                raise ValueError(
                    "Per-component geometry schedules have been replaced by "
                    f"{'direct_geometry' if self.direct_mode else 'canonical_geometry'}.start_iter. Remove these old keys: "
                    + ", ".join(legacy_schedule_keys)
                )
        self.geometry_lrs = {
            "uv": float(geometry_cfg.get("lr_uv", 0.0001)),
            "normal": float(geometry_cfg.get("lr_normal", 0.0002)),
            "scaling": float(geometry_cfg.get("lr_scaling", 0.00005)),
            "rotation": float(geometry_cfg.get("lr_rotation", 0.00005)),
        }
        if self.direct_mode:
            self.geometry_lrs = {
                "uvw": float(geometry_cfg.get("lr_uvw", 0.0001)),
                "scaling": float(geometry_cfg.get("lr_scaling", 0.00005)),
                "rotation": float(geometry_cfg.get("lr_rotation", 0.00005)),
            }
            if style_cfg.get("canonical_geometry", {}).get("enabled", False):
                raise ValueError(
                    "Direct mode uses direct_geometry, not canonical_geometry. "
                    "Set canonical_geometry.enabled=false."
                )
            if self.dc_reg_weight != 0 or self.rest_reg_weight != 0:
                raise ValueError("Direct CSD mode requires dc_reg_weight=rest_reg_weight=0.")
        self.geometry_reg_weights = {
            "uv": float(geometry_cfg.get("uv_weight", 1.0)),
            "normal": float(geometry_cfg.get("normal_weight", 1.0)),
            "scaling": float(geometry_cfg.get("scaling_weight", 1.0)),
            "rotation": float(geometry_cfg.get("rotation_weight", 1.0)),
            "smooth": float(geometry_cfg.get("smooth_weight", 0.1)),
        }
        if not self.train_sh and not self.geometry_enabled:
            raise ValueError(
                "gaussian_style_update has no trainable target: enable train_sh "
                "or geometry training."
            )
        if (
            not self.train_sh
            and self.geometry_enabled
            and self.geometry_start_iter > self.start_iter
        ):
            raise ValueError(
                "SH-first geometry training requires train_sh=true. For a "
                "geometry-only run, set geometry start_iter equal to "
                "gaussian_style_update.start_iter."
            )

        trajectory_cfg = style_cfg.get("gt_mesh_trajectory", {})
        self.gt_mesh_trajectory_enabled = bool(
            trajectory_cfg and trajectory_cfg.get("enabled", False)
        )
        self.trajectory_cfg = trajectory_cfg
        self.trajectory_window_size = max(
            1, int(trajectory_cfg.get("window_size", 4))
        )
        self.trajectory_shared_noise_seed = int(
            trajectory_cfg.get("shared_noise_seed", 2468)
        )
        raw_deformer_mode = trajectory_cfg.get("deformer_mode", "off")
        if isinstance(raw_deformer_mode, bool):
            self.deformer_mode = "frozen" if raw_deformer_mode else "off"
        else:
            self.deformer_mode = str(raw_deformer_mode).lower()
        valid_deformer_modes = {"off", "local_shape", "frozen"}
        if self.deformer_mode not in valid_deformer_modes:
            raise ValueError(
                "gaussian_style_update.gt_mesh_trajectory.deformer_mode must "
                "be one of: off, local_shape, frozen."
            )
        self.deformation_components = None
        if self.direct_mode:
            self.deformer_mode = "off"
            self.use_deformer = False
            self.cfg.use_deformer = False
            if trajectory_cfg:
                trajectory_cfg["deformer_mode"] = "off"
            config.optim.disable_deformer = True
            if not self.gt_mesh_trajectory_enabled:
                raise ValueError("Direct mesh-bound style mode requires gt_mesh_trajectory.enabled=true.")
        if self.gt_mesh_trajectory_enabled:
            mesh_style_cfg = config.optim.get("mesh_style_update", {})
            if mesh_style_cfg and mesh_style_cfg.get("enabled", False):
                raise ValueError(
                    "GT-mesh trajectory style training requires "
                    "mesh_style_update.enabled=false; geometry is supplied "
                    "directly by the known mesh sequence."
                )
            if self.deformer_mode == "off":
                self.use_deformer = False
            elif self.deformer_mode == "local_shape":
                self.use_deformer = True
                self.deformation_components = (
                    "d_rotation",
                    "d_scaling",
                )
            else:
                self.use_deformer = True
        if self.geometry_enabled:
            if not self.gt_mesh_trajectory_enabled:
                raise ValueError(
                    "Geometry style training requires gt_mesh_trajectory.enabled=true "
                    "so shared local parameters follow the known mesh sequence."
                )
            mesh_style_cfg = config.optim.get("mesh_style_update", {})
            if mesh_style_cfg and mesh_style_cfg.get("enabled", False):
                raise ValueError(
                    "Gaussian geometry style replaces the Poisson/Jacobian branch; "
                    "set mesh_style_update.enabled=false."
                )

        outer_cfg = style_cfg.get("outer_loop", {})
        self.outer_enabled = bool(
            outer_cfg and outer_cfg.get("enabled", False)
        )
        if self.gt_mesh_trajectory_enabled and self.outer_enabled:
            raise ValueError(
                "GT-mesh trajectory style training currently requires "
                "outer_loop.enabled=false so every temporal window can share "
                "one CSD timestep and noise sample."
            )
        self.outer_cfg = outer_cfg
        self.outer_iterations = max(
            1, int(outer_cfg.get("outer_iterations", 1))
        )
        self.inner_iterations = max(
            1, int(outer_cfg.get("inner_iterations", 1))
        )
        self.outer_lr_decay = float(outer_cfg.get("lr_decay", 1.0))
        if not 0.0 < self.outer_lr_decay <= 1.0:
            raise ValueError("outer_loop.lr_decay must lie in (0, 1].")
        self.reset_optimizer_each_outer = bool(
            outer_cfg.get("reset_optimizer_each_outer", False)
        )
        self.noise_bank_size = max(
            1, int(outer_cfg.get("noise_bank_size", 4))
        )
        self.resample_noise_bank_each_outer = bool(
            outer_cfg.get("resample_noise_bank_each_outer", True)
        )
        self.outer_base_seed = int(outer_cfg.get("base_seed", 1234))
        self.dc_only_outer_iterations = max(
            0, int(outer_cfg.get("dc_only_outer_iterations", 1))
        )
        self.resume_outer_state = bool(
            outer_cfg.get("resume_outer_state", True)
        )
        self.current_outer = -1
        self.current_inner = -1
        self.current_timestep_range = None
        self.current_noise_slot = -1
        self.noise_bank = []
        self._last_outer = -1
        self._resume_noise_bank = None
        self._resume_outer_index = None
        self._current_iteration = 0

        configured_end_iter = int(style_cfg.get("end_iter", 0))
        if self.outer_enabled:
            batch_size = int(config.optim.batch_size)
            if self.inner_iterations % batch_size != 0:
                raise ValueError(
                    "outer_loop.inner_iterations must be divisible by "
                    f"optim.batch_size ({batch_size}) so gradients do not cross "
                    "outer-loop boundaries."
                )
            self.total_style_iterations = (
                self.outer_iterations * self.inner_iterations
            )
            scheduled_end_iter = (
                self.start_iter + self.total_style_iterations - 1
            )
            if (
                configured_end_iter > 0
                and configured_end_iter != scheduled_end_iter
            ):
                raise ValueError(
                    "With outer_loop enabled, gaussian_style_update.end_iter "
                    "must be 0 or equal start_iter + outer_iterations * "
                    f"inner_iterations - 1 ({scheduled_end_iter})."
                )
            self.end_iter = scheduled_end_iter
            if self.end_iter > int(config.optim.iterations):
                raise ValueError(
                    "The outer-loop style schedule ends at iteration "
                    f"{self.end_iter}, beyond optim.iterations="
                    f"{config.optim.iterations}."
                )
        else:
            self.end_iter = configured_end_iter
            if self.end_iter <= 0:
                self.end_iter = int(config.optim.iterations)
            self.total_style_iterations = self.end_iter - self.start_iter + 1
        self._verified_dc_grad_boundary = False
        self._verified_rest_grad_boundary = False
        self._verified_geometry_grad_boundary = {
            name: False for name in self.geometry_lrs
        }
        self._geometry_training_logged = False

        for param in gaussians.parameters():
            param.requires_grad_(False)
        gaussians.style_features_delta = None
        gaussians.optimizer.zero_grad(set_to_none=True)
        deformer.eval()
        for param in deformer.parameters():
            param.requires_grad_(False)
        deformer.optimizer.zero_grad(set_to_none=True)

        dc_init = None
        rest_init = None
        style_delta_state = checkpoint.get("gaussian_style_delta", None)
        if style_delta_state is not None:
            dc_init = style_delta_state.get("dc_delta", None)
            rest_init = style_delta_state.get("rest_delta", None)
            if dc_init is None or rest_init is None:
                raise ValueError(
                    "gaussian_style_delta must contain dc_delta and rest_delta."
                )
            dc_init = dc_init.to(device)
            rest_init = rest_init.to(device)
        elif "gaussian_style_base_dc" in checkpoint:
            base_dc = checkpoint["gaussian_style_base_dc"].to(device)
            base_rest = checkpoint["gaussian_style_base_rest"].to(device)
            if base_dc.shape != gaussians._features_dc.shape:
                raise ValueError("Legacy Gaussian style DC base shape mismatch.")
            if base_rest.shape != gaussians._features_rest.shape:
                raise ValueError("Legacy Gaussian style rest base shape mismatch.")
            dc_init = gaussians._features_dc.detach() - base_dc
            rest_init = gaussians._features_rest.detach() - base_rest
            gaussians._features_dc.data.copy_(base_dc)
            gaussians._features_rest.data.copy_(base_rest)
            print(
                "[MaGS] Migrated legacy in-place Gaussian SH style parameters "
                "to canonical style deltas."
            )

        self.style_delta = GaussianStyleDelta(
            gaussians._features_dc,
            gaussians._features_rest,
            dc_init=dc_init,
            rest_init=rest_init,
        ).to(device) if not self.direct_mode else None
        self.sh_dc = gaussians._features_dc if self.direct_mode else self.style_delta.dc_delta
        self.sh_rest = gaussians._features_rest if self.direct_mode else self.style_delta.rest_delta
        self.sh_dc.requires_grad_(False)
        self.sh_rest.requires_grad_(False)

        self.geometry_residual = None
        if self.geometry_enabled and not self.direct_mode:
            self.geometry_residual = CanonicalGaussianGeometryResidual(
                gaussians._uvw,
                gaussians.faceid,
                gaussians.refined_verts.squeeze(0),
                gaussians.mesh_faces.squeeze(0),
                geometry_cfg,
            ).to(device)
            geometry_state = checkpoint.get(
                "gaussian_style_geometry_residual", None
            )
            if geometry_state is not None:
                self.geometry_residual.load_state_dict(geometry_state, strict=True)
                print("[MaGS] Restored canonical Gaussian geometry style residual.")
            for param in self.geometry_residual.parameters():
                param.requires_grad_(False)

        self.sds_guidance = shared_sds_guidance
        if self.sds_guidance is None:
            self.sds_guidance = build_sds_guidance(style_cfg, device)
            if not getattr(
                self.sds_guidance, "manages_device_placement", False
            ):
                self.sds_guidance = self.sds_guidance.to(device)
        if not self.sds_guidance.enabled:
            raise ValueError(
                "optim.gaussian_style_update.sds.enabled must be true for "
                "Gaussian SH style training."
            )
        if getattr(self.sds_guidance, "gradient_mode", "sds") != "csd":
            raise ValueError(
                "Single-step Gaussian style training requires sds.gradient_mode=csd."
            )
        self.video_guidance = bool(
            getattr(self.sds_guidance, "expects_video", False)
        )
        if self.split_supervision and not self.video_guidance:
            raise ValueError("split_vgg_diffusion requires the video AnimateDiff/ControlNet teacher.")
        if self.video_guidance:
            if not self.gt_mesh_trajectory_enabled:
                raise ValueError(
                    "SDXL AnimateDiff/ControlNet guidance requires "
                    "gaussian_style_update.gt_mesh_trajectory.enabled=true."
                )
            if self.outer_enabled:
                raise ValueError(
                    "Video CSD manages a real temporal window and cannot be "
                    "combined with gaussian_style_update.outer_loop."
                )
            video_window_size = int(self.sds_guidance.window_size)
            if self.trajectory_window_size != video_window_size:
                raise ValueError(
                    "gt_mesh_trajectory.window_size and "
                    "sds.video_window_size must match for video CSD "
                    f"({self.trajectory_window_size} != {video_window_size})."
                )
            if self.total_style_iterations % video_window_size != 0:
                raise ValueError(
                    "Video CSD currently requires the number of style "
                    "iterations to be divisible by video_window_size."
                )
            if (
                self.dc_only_iterations > 0
                and self.dc_only_iterations % video_window_size != 0
            ):
                raise ValueError(
                    "dc_only_iterations must be divisible by "
                    "video_window_size so one video window uses one SH phase."
                )
            if (
                self.geometry_enabled
                and self.geometry_start_iter <= self.end_iter
                and (self.geometry_start_iter - self.start_iter) % video_window_size != 0
            ):
                raise ValueError(
                    "Geometry start_iter must begin at a video-window "
                    "boundary: "
                    f"(start_iter - {self.start_iter}) must be divisible "
                    f"by {video_window_size}."
                )
        if train_dataset is None:
            raise ValueError("CSD requires the training content frames.")
        if shared_sds_guidance is None:
            if self.split_supervision and source_reference is None:
                raise ValueError("Split supervision requires the fixed no-deformer render reference.")
            self.sds_guidance.cache_source_images(
                source_reference["images"] if self.split_supervision
                else {cam.uid: cam.original_image for cam in train_dataset}
            )
            if self.video_guidance:
                self.sds_guidance.cache_control_frames(train_dataset)
                if self.split_supervision:
                    directory = os.path.join(config.model.model_path, "style_progress", "control_reference")
                    os.makedirs(directory, exist_ok=True)
                    for uid, depth in self.sds_guidance.control_cache.items():
                        rgb = source_reference["images"][uid].unsqueeze(0)
                        rgb = torch.nn.functional.interpolate(rgb, size=depth.shape[-2:], mode="bilinear", align_corners=False)
                        torchvision.utils.save_image(depth, os.path.join(directory, f"view_{uid}_depth.png"))
                        torchvision.utils.save_image(0.7 * rgb + 0.3 * depth, os.path.join(directory, f"view_{uid}_overlay.png"))
        if self.split_supervision:
            self.sds_guidance.park_diffusion_backbone()
            from utils.vgg_appearance_loss import VGGAppearanceLoss
            self.appearance = VGGAppearanceLoss(
                self.appearance_cfg, str(style_cfg.get("style_path", "")), train_dataset, device,
            )
        self._video_render_buffer = []
        self._video_source_keys = []
        self.last_video_preview = None
        self.timestep_curriculum = self._parse_timestep_curriculum()
        self.style_strength_schedule = self._parse_outer_scalar_schedule(
            "style_strength_schedule",
            1.0,
            minimum=0.0,
        )
        self.guidance_scale_schedule = self._parse_outer_scalar_schedule(
            "guidance_scale_schedule",
            float(self.sds_guidance.guidance_scale),
            minimum=0.0,
        )
        self.ip_adapter_scale_schedule = self._parse_outer_scalar_schedule(
            "ip_adapter_scale_schedule",
            float(self.sds_guidance.ip_adapter_scale),
            minimum=0.0,
        )
        optimizer_groups = [
                {
                    "params": [self.sh_dc],
                    "lr": self.base_lr_dc,
                    "name": "_features_dc" if self.direct_mode else "style_dc_delta",
                },
                {
                    "params": [self.sh_rest],
                    "lr": self.base_lr_rest,
                    "name": "_features_rest" if self.direct_mode else "style_rest_delta",
                },
            ]
        if self.geometry_enabled:
            for name, parameter in self.geometry_parameters().items():
                optimizer_groups.append(
                    {
                        "params": [parameter],
                        "lr": self.geometry_lrs[name],
                        "name": f"_{name}" if self.direct_mode else f"style_geometry_{name}",
                    }
                )
        self.optimizer = torch.optim.Adam(
            optimizer_groups[:2] if self.split_supervision else optimizer_groups,
            lr=0.0,
            eps=1e-15,
        )
        self.geometry_optimizer = None
        if self.split_supervision and optimizer_groups[2:]:
            self.geometry_optimizer = torch.optim.Adam(optimizer_groups[2:], lr=0.0, eps=1e-15)
            if "gaussian_geometry_optimizer" in checkpoint:
                self.geometry_optimizer.load_state_dict(checkpoint["gaussian_geometry_optimizer"])
        if "gaussian_style_optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["gaussian_style_optimizer"])
                print("[MaGS] Restored Gaussian style optimizer.")
            except ValueError as exc:
                if self.direct_mode:
                    raise ValueError("Direct style optimizer state does not match the parameter groups.") from exc
                print(f"[WARN] Could not restore Gaussian style optimizer: {exc}")

        self.loop_start_iteration = 1
        if self.direct_mode and checkpoint.get("gaussian_style_parameterization") == "direct":
            self._restore_direct_schedule(checkpoint)
        outer_state = checkpoint.get("gaussian_style_outer_state", None)
        if self.outer_enabled and self.resume_outer_state and outer_state:
            checkpoint_iteration = int(checkpoint.get("iteration", 0))
            completed = int(outer_state.get("style_steps_completed", 0))
            if completed > 0:
                saved_style_cfg = checkpoint.get("gaussian_style", {})
                saved_outer_cfg = saved_style_cfg.get("outer_loop", {})
                saved_outer_iterations = int(
                    outer_state.get(
                        "outer_iterations",
                        saved_outer_cfg.get(
                            "outer_iterations", self.outer_iterations
                        ),
                    )
                )
                saved_inner_iterations = int(
                    outer_state.get(
                        "inner_iterations",
                        saved_outer_cfg.get(
                            "inner_iterations", self.inner_iterations
                        ),
                    )
                )
                if (
                    saved_outer_iterations != self.outer_iterations
                    or saved_inner_iterations != self.inner_iterations
                ):
                    raise ValueError(
                        "Cannot resume Gaussian style outer state with a "
                        "different outer/inner schedule. Match the checkpoint "
                        "configuration or set outer_loop.resume_outer_state=false."
                    )
                self.loop_start_iteration = checkpoint_iteration + 1
                self._resume_outer_index = int(
                    outer_state.get("outer_index", -1)
                )
                resume_bank = outer_state.get("noise_bank", None)
                if resume_bank:
                    self._resume_noise_bank = [
                        {
                            "seed": int(item["seed"]),
                            "timestep": int(item["timestep"]),
                        }
                        for item in resume_bank
                    ]
                print(
                    "[MaGS] Restoring Gaussian style outer schedule from "
                    f"iteration {checkpoint_iteration} "
                    f"(completed_style_steps={completed})."
                )
        if self.loop_start_iteration > int(config.optim.iterations) + 1:
            raise ValueError(
                "The restored Gaussian style checkpoint is newer than "
                "optim.iterations. Increase optim.iterations or disable "
                "outer_loop.resume_outer_state."
            )

        print(
            "[MaGS] Gaussian style update enabled "
            f"(start={self.start_iter}, end={self.end_iter}, "
            f"parameterization={self.parameterization}, "
            f"train_sh={self.train_sh}, "
            f"dc_only={self.dc_only_iterations}, "
            f"lr_dc={style_cfg.get('lr_dc', 0.001)}, "
            f"lr_rest={style_cfg.get('lr_rest', 0.00005)}, "
            f"guidance={getattr(self.sds_guidance, 'gradient_mode', 'sds')}, "
            f"deformer_forward={self.use_deformer})."
        )
        if self.gt_mesh_trajectory_enabled:
            print(
                "[MaGS] GT-mesh trajectory style mode enabled "
                f"(window={self.trajectory_window_size}, "
                f"deformer_mode={self.deformer_mode}, "
                "geometry=hard GT mesh)."
            )
        if self.geometry_enabled:
            print(
                "[MaGS] Gaussian geometry style training enabled "
                "(shared across frames; all geometry components start together "
                f"at iteration {self.geometry_start_iter}; "
                f"continue_SH_training={self.train_sh})."
            )
        if self.direct_mode:
            print(
                "[MaGS] Direct Gaussian parameters only; no residual modules, "
                "no deformer forward, opacity/densification/pruning disabled. "
                f"Output: {config.model.model_path}"
            )
        if self.video_guidance:
            print(
                "[MaGS] Window-level video CSD enabled "
                f"(frames={self.trajectory_window_size}, "
                "teacher=SDXL+IP-Adapter+AnimateDiff+depth ControlNet)."
            )
        if self.split_supervision:
            print("[MaGS] Split supervision: VGG -> SH; video CSD -> UVW/scale/rotation only. No diffusion prediction during the SH-only stage.")
        if self.outer_enabled:
            print(
                "[MaGS] Gaussian style outer loop enabled "
                f"(outer={self.outer_iterations}, "
                f"inner={self.inner_iterations}, "
                f"total={self.total_style_iterations}, "
                f"noise_bank={self.noise_bank_size}, "
                f"lr_decay={self.outer_lr_decay})."
            )

    def geometry_parameters(self):
        if not self.geometry_enabled:
            return {}
        if self.direct_mode:
            return {
                "uvw": self.gaussians._uvw,
                "scaling": self.gaussians._scaling,
                "rotation": self.gaussians._rotation,
            }
        return self.geometry_residual.parameter_items()

    def _restore_direct_schedule(self, checkpoint):
        saved = checkpoint["direct_style_state"]
        current = self.direct_state_dict()
        if saved != current:
            raise ValueError(
                "Direct style resume must preserve the SH/geometry schedule and "
                f"video window: saved={saved}, current={current}."
            )
        iteration = int(checkpoint["iteration"])
        if self.video_guidance and (iteration - self.start_iter + 1) % self.trajectory_window_size:
            raise ValueError("Resume a direct-style checkpoint at a completed video window.")
        self.loop_start_iteration = iteration + 1

    def direct_state_dict(self):
        state = {
            "start_iter": self.start_iter,
            "geometry_start_iter": self.geometry_start_iter,
            "geometry_enabled": self.geometry_enabled,
            "train_sh": self.train_sh,
            "dc_only_iterations": self.dc_only_iterations,
            "window_size": self.trajectory_window_size,
            "video_guidance": self.video_guidance,
        }
        if self.split_supervision:
            state["split_supervision"] = {
                "version": 1,
                "style_fingerprint": self.appearance.style_fingerprint,
                "crop": self.appearance.crop,
                "loss_type": self.appearance.loss_type,
                "layers": list(self.appearance.layers),
                "image_size": self.appearance.image_size,
                "reference_signature": self.source_reference["signature"],
            }
        return state

    def _parse_timestep_curriculum(self):
        default_range = (
            float(self.sds_guidance.t_min),
            float(self.sds_guidance.t_max),
        )
        if not self.outer_enabled:
            return [default_range]

        raw_curriculum = self.outer_cfg.get("timestep_curriculum", [])
        curriculum = []
        for item in raw_curriculum:
            if len(item) != 2:
                raise ValueError(
                    "Each outer_loop.timestep_curriculum entry must be "
                    "[t_min, t_max]."
                )
            t_min, t_max = float(item[0]), float(item[1])
            if not 0.0 <= t_min < t_max <= 1.0:
                raise ValueError(
                    "Outer-loop timestep ranges must satisfy "
                    "0 <= t_min < t_max <= 1."
                )
            curriculum.append((t_min, t_max))

        if not curriculum:
            curriculum = [default_range] * self.outer_iterations
        elif len(curriculum) == 1:
            curriculum = curriculum * self.outer_iterations
        elif len(curriculum) != self.outer_iterations:
            raise ValueError(
                "outer_loop.timestep_curriculum must contain one range or "
                f"exactly {self.outer_iterations} ranges."
            )
        return curriculum

    def _parse_outer_scalar_schedule(self, name, default_value, minimum=None):
        if not self.outer_enabled:
            return [float(default_value)]

        raw_schedule = self.outer_cfg.get(name, [])
        schedule = [float(value) for value in raw_schedule]
        if not schedule:
            schedule = [float(default_value)] * self.outer_iterations
        elif len(schedule) == 1:
            schedule = schedule * self.outer_iterations
        elif len(schedule) != self.outer_iterations:
            raise ValueError(
                f"outer_loop.{name} must contain one value or exactly "
                f"{self.outer_iterations} values."
            )
        if minimum is not None and any(value < minimum for value in schedule):
            raise ValueError(
                f"outer_loop.{name} values must be >= {minimum}."
            )
        return schedule

    def _make_noise_bank(self, outer_index):
        t_min, t_max = self.timestep_curriculum[outer_index]
        num_steps = self.sds_guidance.num_train_timesteps
        min_step = max(0, int(num_steps * t_min))
        max_step = min(num_steps - 1, int(num_steps * t_max))
        if max_step <= min_step:
            raise ValueError(
                f"Outer {outer_index + 1} has an empty timestep range."
            )

        timestep_rng = random.Random(
            self.outer_base_seed + 104729 * (outer_index + 1)
        )
        noise_outer = outer_index if self.resample_noise_bank_each_outer else 0
        noise_rng = random.Random(
            self.outer_base_seed + 130363 * (noise_outer + 1)
        )
        return [
            {
                "seed": noise_rng.randint(0, 2**31 - 1),
                "timestep": timestep_rng.randint(min_step, max_step),
            }
            for _ in range(self.noise_bank_size)
        ]

    def _enter_outer(self, outer_index):
        if self._last_outer == outer_index:
            return
        if self.reset_optimizer_each_outer and self._last_outer >= 0:
            self.optimizer.state.clear()

        lr_scale = self.outer_lr_decay**outer_index
        self.optimizer.param_groups[0]["lr"] = self.base_lr_dc * lr_scale
        self.optimizer.param_groups[1]["lr"] = self.base_lr_rest * lr_scale
        self.current_timestep_range = self.timestep_curriculum[outer_index]
        self.current_sds_weight = (
            self.sds_weight * self.style_strength_schedule[outer_index]
        )
        self.sds_guidance.guidance_scale = self.guidance_scale_schedule[
            outer_index
        ]
        self.sds_guidance.ip_adapter_scale = self.ip_adapter_scale_schedule[
            outer_index
        ]
        self.sds_guidance.pipe.set_ip_adapter_scale(
            self.sds_guidance.ip_adapter_scale
        )

        if (
            self._resume_noise_bank is not None
            and self._resume_outer_index == outer_index
        ):
            if len(self._resume_noise_bank) != self.noise_bank_size:
                raise ValueError(
                    "Restored outer-loop noise bank size does not match the "
                    "current configuration."
                )
            self.noise_bank = self._resume_noise_bank
            self._resume_noise_bank = None
        else:
            self.noise_bank = self._make_noise_bank(outer_index)

        self._last_outer = outer_index
        print(
            f"[MaGS] Entering Gaussian style outer "
            f"{outer_index + 1}/{self.outer_iterations} "
            f"(t={self.current_timestep_range[0]:.2f}-"
            f"{self.current_timestep_range[1]:.2f}, "
            f"lr_dc={self.optimizer.param_groups[0]['lr']:.6g}, "
            f"lr_rest={self.optimizer.param_groups[1]['lr']:.6g}, "
            f"csd_weight={self.current_sds_weight:.4g}, "
            f"guidance={self.sds_guidance.guidance_scale:.3g}, "
            f"ip_scale={self.sds_guidance.ip_adapter_scale:.3g})."
        )

    def is_training_step(self, iteration):
        return self.enabled and self.start_iter <= iteration <= self.end_iter

    def frame_index(self, iteration, frame_count):
        if frame_count <= 0:
            raise ValueError("GT-mesh trajectory training requires at least one frame.")
        if self.trajectory_window_size > frame_count:
            raise ValueError(
                "GT-mesh trajectory window_size cannot exceed the number of "
                f"training frames ({frame_count})."
            )
        local_step = iteration - self.start_iter
        window_index = local_step // self.trajectory_window_size
        slot = local_step % self.trajectory_window_size
        max_window_start = frame_count - self.trajectory_window_size
        window_start = (
            window_index * self.trajectory_window_size
        ) % (max_window_start + 1)
        return window_start + slot

    def guidance_kwargs(self, iteration):
        if not self.gt_mesh_trajectory_enabled or not self.is_training_step(iteration):
            return {}
        window_index = (
            iteration - self.start_iter
        ) // self.trajectory_window_size
        rng = random.Random(self.trajectory_shared_noise_seed + window_index)
        return {
            "timesteps": rng.randint(
                int(self.sds_guidance.min_step),
                int(self.sds_guidance.max_step),
            ),
            "noise_seed": rng.randint(0, 2**31 - 1),
        }

    def accumulation_divisor(self, iteration, default_batch_size):
        if self.video_guidance:
            return 1
        if not self.gt_mesh_trajectory_enabled:
            return max(1, int(default_batch_size))
        window_index = (
            iteration - self.start_iter
        ) // self.trajectory_window_size
        window_start = self.start_iter + window_index * self.trajectory_window_size
        return min(
            self.trajectory_window_size,
            self.end_iter - window_start + 1,
        )

    def is_accumulation_end(self, iteration, default_batch_size):
        if not self.is_training_step(iteration):
            return False
        accumulation_size = (
            self.trajectory_window_size
            if self.gt_mesh_trajectory_enabled
            else max(1, int(default_batch_size))
        )
        local_step = iteration - self.start_iter + 1
        return (
            local_step % accumulation_size == 0
            or iteration == self.end_iter
        )

    def prepare_iteration(self, iteration):
        if not self.enabled:
            return
        self._current_iteration = iteration
        style_active = self.is_training_step(iteration)
        train_dc = style_active and self.train_sh
        if self.outer_enabled and style_active:
            style_step = iteration - self.start_iter
            self.current_outer = style_step // self.inner_iterations
            self.current_inner = style_step % self.inner_iterations
            self._enter_outer(self.current_outer)
            self.current_noise_slot = (
                self.current_inner % self.noise_bank_size
            )
            train_rest = self.train_sh and (
                self.current_outer >= self.dc_only_outer_iterations
            )
        else:
            train_rest = (
                style_active
                and self.train_sh
                and iteration - self.start_iter >= self.dc_only_iterations
            )
        self.sh_dc.requires_grad_(train_dc)
        self.sh_rest.requires_grad_(train_rest)
        if self.geometry_enabled:
            geometry_active = style_active and iteration >= self.geometry_start_iter
            for parameter in self.geometry_parameters().values():
                parameter.requires_grad_(geometry_active)
            if geometry_active and not self._geometry_training_logged:
                target = (
                    "direct Gaussian geometry parameters" if self.direct_mode
                    else "canonical geometry style residuals"
                )
                print(
                    f"[MaGS] Enabling all {target} at "
                    f"iteration {iteration}: {', '.join(self.geometry_parameters())} "
                    f"(SH training continues={self.train_sh})."
                )
                self._geometry_training_logged = True

    def apply_geometry_to_gaussians(self, detach=False):
        if not self.enabled or self.direct_mode or not self.geometry_enabled:
            self.gaussians.style_uvw_delta = None
            self.gaussians.style_log_scaling_delta = None
            self.gaussians.style_rotation_delta = None
            return
        package = self.geometry_residual()
        value = lambda tensor: tensor.detach() if detach else tensor
        self.gaussians.style_uvw_delta = value(package["uvw_delta"])
        self.gaussians.style_log_scaling_delta = value(
            package["log_scaling_delta"]
        )
        self.gaussians.style_rotation_delta = value(package["rotation_delta"])

    def apply_to_gaussians(self, detach=False):
        if not self.enabled or self.direct_mode:
            self.gaussians.style_features_delta = None
            return
        style_delta = self.style_delta.combined()
        self.gaussians.style_features_delta = (
            style_delta.detach() if detach else style_delta
        )

    def stage_name(self, iteration):
        if not self.is_training_step(iteration):
            return "frozen"
        if self.outer_enabled:
            phase = "dc" if not self.sh_rest.requires_grad else "sh"
            return f"style-o{self.current_outer + 1}-{phase}"
        sh_active = self.sh_dc.requires_grad
        rest_active = self.sh_rest.requires_grad
        geometry_active = self.geometry_enabled and any(
            parameter.requires_grad
            for parameter in self.geometry_parameters().values()
        )
        if geometry_active and not sh_active:
            phase = "style-geometry"
        elif geometry_active:
            phase = "style-sh+geometry" if rest_active else "style-dc+geometry"
        elif rest_active:
            phase = "style-sh"
        else:
            phase = "style-dc"
        prefix = "direct-" if self.direct_mode else ""
        return prefix + (f"gtmesh-{phase}" if self.gt_mesh_trajectory_enabled else phase)

    def _appearance_regularization(self, reference):
        zero = reference.new_zeros(())
        if self.direct_mode:
            return zero, zero
        return (
            self.dc_reg_weight * self.sh_dc.square().mean() if self.dc_reg_weight else zero,
            self.rest_reg_weight * self.sh_rest.square().mean() if self.rest_reg_weight else zero,
        )

    def _geometry_regularization(self, reference):
        zero = reference.new_zeros(())
        names = ("uv", "normal", "scaling", "rotation", "smooth")
        if not self.geometry_enabled or self.direct_mode:
            diagnostics = {
                f"gaussian_style_geometry_{name}_reg": zero for name in names
            } | {
                f"gaussian_style_geometry_{name}_rms": zero
                for name in ("uv", "normal", "scaling", "rotation")
            }
            if self.direct_mode:
                diagnostics.update({
                    f"gaussian_style_geometry_{name}_parameter_rms":
                    parameter.detach().float().square().mean().sqrt()
                    for name, parameter in self.geometry_parameters().items()
                })
            return zero, diagnostics

        raw_regularizers = self.geometry_residual.regularization()
        weighted = {
            name: self.geometry_reg_weights[name] * raw_regularizers[name]
            for name in names
        }
        total = sum(weighted.values(), zero)
        diagnostics = {
            f"gaussian_style_geometry_{name}_reg": value
            for name, value in weighted.items()
        }
        diagnostics.update(
            {
                f"gaussian_style_geometry_{name}": value.detach()
                for name, value in self.geometry_residual.rms_diagnostics().items()
            }
        )
        return total, diagnostics

    def _video_loss(self, image_batch, source_key, guidance_kwargs):
        if image_batch.shape[0] != 1:
            raise ValueError("Window-level video CSD requires render batch size 1.")
        expected_slot = (
            self._current_iteration - self.start_iter
        ) % self.trajectory_window_size
        if len(self._video_render_buffer) != expected_slot:
            raise RuntimeError(
                "Video CSD must start or resume at the first frame of a "
                "temporal window. The current window expects slot "
                f"{expected_slot}, but {len(self._video_render_buffer)} "
                "frames are buffered."
            )
        if source_key is None:
            raise ValueError("Video CSD requires a source key for every frame.")
        self._video_render_buffer.append(image_batch[0])
        self._video_source_keys.append(source_key)
        if len(self._video_render_buffer) > self.trajectory_window_size:
            raise RuntimeError("Video CSD render buffer exceeded its window size.")

        ready = len(self._video_render_buffer) == self.trajectory_window_size
        if not ready:
            zero = image_batch.detach().new_zeros(())
            _, geometry_diagnostics = self._geometry_regularization(image_batch)
            geometry_diagnostics = {
                name: zero if name.endswith("_reg") else value
                for name, value in geometry_diagnostics.items()
            }
            losses = {
                "gaussian_style_loss": zero,
                "gaussian_style_csd_loss": zero,
                "gaussian_style_csd_raw_loss": zero,
                "gaussian_style_dc_reg": zero,
                "gaussian_style_rest_reg": zero,
                "gaussian_style_dc_parameter_rms": self.sh_dc.float()
                .pow(2)
                .mean()
                .sqrt()
                .detach(),
                "gaussian_style_rest_parameter_rms": self.sh_rest.float()
                .pow(2)
                .mean()
                .sqrt()
                .detach(),
                "gaussian_style_csd_grad_norm": zero,
                "gaussian_style_video_window_ready": zero,
                "gaussian_style_video_buffered_frames": zero.new_tensor(
                    float(len(self._video_render_buffer))
                ),
                "gaussian_style_trajectory_window": zero.new_tensor(
                    float(
                        (self._current_iteration - self.start_iter)
                        // self.trajectory_window_size
                    )
                ),
                "gaussian_style_trajectory_slot": zero.new_tensor(
                    float(
                        (self._current_iteration - self.start_iter)
                        % self.trajectory_window_size
                    )
                ),
            }
            losses.update(geometry_diagnostics)
            return losses

        video = torch.stack(self._video_render_buffer, dim=0).unsqueeze(0)
        self.last_video_preview = video.detach().cpu()
        source_keys = list(self._video_source_keys)
        self._video_render_buffer.clear()
        self._video_source_keys.clear()
        if self.split_supervision:
            if self._split_pending is not None:
                raise RuntimeError("The previous split style window has not been differentiated.")
            self._split_pending = (video, source_keys, guidance_kwargs)
            zero = video.detach().new_zeros(())
            _, geometry_losses = self._geometry_regularization(video)
            return {
                **geometry_losses,
                "gaussian_style_loss": zero,
                "gaussian_style_appearance_loss": zero,
                "gaussian_style_csd_loss": zero,
                "gaussian_style_csd_raw_loss": zero,
                "gaussian_style_dc_reg": zero,
                "gaussian_style_rest_reg": zero,
                "gaussian_style_dc_parameter_rms": self.sh_dc.detach().float().square().mean().sqrt(),
                "gaussian_style_rest_parameter_rms": self.sh_rest.detach().float().square().mean().sqrt(),
                "gaussian_style_csd_grad_norm": zero,
                "gaussian_style_video_window_ready": zero.new_tensor(1.0),
                "gaussian_style_video_buffered_frames": zero.new_tensor(float(video.shape[1])),
                "gaussian_style_trajectory_window": zero.new_tensor(float((self._current_iteration - self.start_iter) // self.trajectory_window_size)),
                "gaussian_style_trajectory_slot": zero.new_tensor(float(self.trajectory_window_size - 1)),
            }
        try:
            csd_raw, diagnostics = self.sds_guidance(
                video,
                source_keys=source_keys,
                return_diagnostics=True,
                **guidance_kwargs,
            )
        except FloatingPointError as exc:
            raise FloatingPointError(
                f"Video CSD failed at iteration {self._current_iteration}, "
                f"frames={source_keys}: {exc}"
            ) from exc
        csd_loss = self.current_sds_weight * csd_raw
        dc_reg, rest_reg = self._appearance_regularization(video)
        geometry_reg, geometry_losses = self._geometry_regularization(video)
        total = csd_loss + dc_reg + rest_reg + geometry_reg
        require_finite(
            total,
            f"style loss at iteration {self._current_iteration}, frames={source_keys}",
        )
        losses = {
            "gaussian_style_loss": total,
            "gaussian_style_csd_loss": csd_loss,
            "gaussian_style_csd_raw_loss": csd_raw,
            "gaussian_style_dc_reg": dc_reg,
            "gaussian_style_rest_reg": rest_reg,
            "gaussian_style_dc_parameter_rms": self.sh_dc.float()
            .pow(2)
            .mean()
            .sqrt()
            .detach(),
            "gaussian_style_rest_parameter_rms": self.sh_rest.float()
            .pow(2)
            .mean()
            .sqrt()
            .detach(),
            "gaussian_style_video_window_ready": total.detach().new_tensor(1.0),
            "gaussian_style_video_buffered_frames": total.detach().new_tensor(
                float(video.shape[1])
            ),
            "gaussian_style_trajectory_window": total.detach().new_tensor(
                float(
                    (self._current_iteration - self.start_iter)
                    // self.trajectory_window_size
                )
            ),
            "gaussian_style_trajectory_slot": total.detach().new_tensor(
                float(self.trajectory_window_size - 1)
            ),
        }
        losses.update(geometry_losses)
        losses.update(
            {
                f"gaussian_style_{name}": value.detach().float()
                for name, value in diagnostics.items()
            }
        )
        return losses

    def loss(self, image, source_key=None, guidance_kwargs=None):
        context = f"iteration {self._current_iteration}, frame={source_key}"
        require_finite(image, f"Gaussian render at {context}")
        check_backward_finite(image, f"Gaussian render gradient at {context}")
        image_batch = image.unsqueeze(0) if image.dim() == 3 else image
        guidance_kwargs = dict(guidance_kwargs or {})
        if self.video_guidance:
            return self._video_loss(image_batch, source_key, guidance_kwargs)
        if self.outer_enabled:
            if guidance_kwargs:
                raise ValueError(
                    "External CSD sampling cannot be combined with the Gaussian "
                    "style outer-loop noise bank."
                )
            if not self.noise_bank or self.current_noise_slot < 0:
                raise RuntimeError(
                    "prepare_iteration() must initialize the outer-loop "
                    "sampling state before computing CSD loss."
                )
            sample = self.noise_bank[self.current_noise_slot]
            guidance_kwargs = {
                "timesteps": sample["timestep"],
                "noise_seed": sample["seed"],
            }
        csd_raw, diagnostics = self.sds_guidance(
            image_batch,
            source_key=source_key,
            return_diagnostics=True,
            **guidance_kwargs,
        )
        csd_loss = self.current_sds_weight * csd_raw
        dc_reg, rest_reg = self._appearance_regularization(image_batch)
        geometry_reg, geometry_losses = self._geometry_regularization(image_batch)
        total = csd_loss + dc_reg + rest_reg + geometry_reg
        require_finite(total, f"style loss at {context}")
        losses = {
            "gaussian_style_loss": total,
            "gaussian_style_csd_loss": csd_loss,
            "gaussian_style_csd_raw_loss": csd_raw,
            "gaussian_style_dc_reg": dc_reg,
            "gaussian_style_rest_reg": rest_reg,
            "gaussian_style_dc_parameter_rms": self.sh_dc.float()
            .pow(2)
            .mean()
            .sqrt()
            .detach(),
            "gaussian_style_rest_parameter_rms": self.sh_rest.float()
            .pow(2)
            .mean()
            .sqrt()
            .detach(),
        }
        losses.update(geometry_losses)
        if self.gt_mesh_trajectory_enabled:
            losses.update(
                {
                    "gaussian_style_trajectory_window": image_batch.new_tensor(
                        float(
                            (self._current_iteration - self.start_iter)
                            // self.trajectory_window_size
                        )
                    ),
                    "gaussian_style_trajectory_slot": image_batch.new_tensor(
                        float(
                            (self._current_iteration - self.start_iter)
                            % self.trajectory_window_size
                        )
                    ),
                }
            )
        if self.outer_enabled:
            losses.update(
                {
                    "gaussian_style_outer_index": image_batch.new_tensor(
                        float(self.current_outer + 1)
                    ),
                    "gaussian_style_inner_index": image_batch.new_tensor(
                        float(self.current_inner + 1)
                    ),
                    "gaussian_style_noise_slot": image_batch.new_tensor(
                        float(self.current_noise_slot)
                    ),
                    "gaussian_style_strength": image_batch.new_tensor(
                        float(
                            self.style_strength_schedule[self.current_outer]
                        )
                    ),
                    "gaussian_style_effective_csd_weight": image_batch.new_tensor(
                        float(self.current_sds_weight)
                    ),
                    "gaussian_style_guidance_scale": image_batch.new_tensor(
                        float(self.sds_guidance.guidance_scale)
                    ),
                    "gaussian_style_ip_adapter_scale": image_batch.new_tensor(
                        float(self.sds_guidance.ip_adapter_scale)
                    ),
                }
            )
        losses.update(
            {
                f"gaussian_style_{name}": value.detach().float()
                for name, value in diagnostics.items()
            }
        )
        return losses

    def outer_state_dict(self, iteration):
        if not self.outer_enabled:
            return None
        completed = min(
            max(iteration - self.start_iter + 1, 0),
            self.total_style_iterations,
        )
        return {
            "style_steps_completed": completed,
            "outer_iterations": self.outer_iterations,
            "inner_iterations": self.inner_iterations,
            "outer_index": self.current_outer,
            "inner_index": self.current_inner,
            "noise_slot": self.current_noise_slot,
            "noise_bank": [dict(item) for item in self.noise_bank],
            "base_seed": self.outer_base_seed,
            "timestep_range": self.current_timestep_range,
            "style_strength": (
                self.style_strength_schedule[self.current_outer]
                if self.current_outer >= 0
                else 1.0
            ),
            "effective_csd_weight": self.current_sds_weight,
            "guidance_scale": self.sds_guidance.guidance_scale,
            "ip_adapter_scale": self.sds_guidance.ip_adapter_scale,
        }

    def _check_finite_style_state(self, stage, optimizer_state=False):
        for optimizer in self.style_optimizers():
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    name = f"{group['name']} at iteration {self._current_iteration} ({stage})"
                    require_finite(parameter, name)
                    if parameter.grad is not None:
                        require_finite(parameter.grad, f"gradient of {name}")
                    if optimizer_state:
                        for key, value in optimizer.state.get(parameter, {}).items():
                            if torch.is_tensor(value):
                                require_finite(value, f"Adam {key} for {name}")

    def verify_gradient_boundary(self):
        # Finiteness must be checked on every backward, even after the parameter
        # ownership boundary has been verified for the current training phase.
        self._check_finite_style_state("after backward")
        dc_active = self.sh_dc.requires_grad
        rest_active = self.sh_rest.requires_grad
        active_geometry = {}
        if self.geometry_enabled:
            active_geometry = {
                name: parameter
                for name, parameter in self.geometry_parameters().items()
                if parameter.requires_grad
            }
        already_verified = (
            (not dc_active or self._verified_dc_grad_boundary)
            and (not rest_active or self._verified_rest_grad_boundary)
            and all(
                self._verified_geometry_grad_boundary[name]
                for name in active_geometry
            )
        )
        if already_verified and not self.direct_mode:
            return
        allowed = (
            {id(self.sh_dc), id(self.sh_rest)}
            | {id(parameter) for parameter in active_geometry.values()}
            if self.direct_mode else set()
        )
        leaked = [
            name
            for name, param in self.gaussians.named_parameters()
            if param.grad is not None and (id(param) not in allowed or not param.requires_grad)
        ]
        leaked.extend(
            f"deformer.{name}"
            for name, param in self.deformer.named_parameters()
            if param.grad is not None
        )
        if leaked:
            raise RuntimeError(
                "Gaussian style gradients escaped the independent style residuals: "
                + ", ".join(leaked)
            )
        if already_verified:
            return
        if dc_active and self.sh_dc.grad is None:
            raise RuntimeError("CSD did not reach style_dc_delta.")
        if rest_active and self.sh_rest.grad is None:
            raise RuntimeError("CSD did not reach style_rest_delta.")
        missing_geometry = [
            name
            for name, parameter in active_geometry.items()
            if parameter.grad is None
        ]
        if missing_geometry:
            raise RuntimeError(
                "CSD did not reach canonical Gaussian geometry residuals: "
                + ", ".join(missing_geometry)
            )
        if dc_active:
            self._verified_dc_grad_boundary = True
        if rest_active:
            self._verified_rest_grad_boundary = True
        for name in active_geometry:
            self._verified_geometry_grad_boundary[name] = True
        active_names = []
        if dc_active:
            active_names.append("DC")
        if rest_active:
            active_names.append("SH-rest")
        active_names.extend(f"geometry-{name}" for name in active_geometry)
        target = (
            "Gaussian parameters directly" if self.direct_mode
            else "independent canonical style residuals only"
        )
        route = "VGG -> SH; video CSD -> geometry" if self.split_supervision else "SDXL CSD -> render"
        print(f"[MaGS] Verified style gradient path: {route} -> {target} (active={'+'.join(active_names)}).")

    def backward_style_loss(self, losses, divisor=1):
        if not self.split_supervision:
            (losses["gaussian_style_loss"] / divisor).backward()
            return
        if divisor != 1:
            raise ValueError("Split video supervision requires one update per complete window.")
        from utils.split_style_training import routed_style_gradients
        if self._split_pending is None:
            raise RuntimeError("No complete split style window to differentiate.")
        video, source_keys, kwargs = self._split_pending
        try:
            gradients, metrics = routed_style_gradients(
                video, source_keys, self.appearance, self.sds_guidance,
                [self.sh_dc, self.sh_rest], list(self.geometry_parameters().values()),
                self.appearance_weight, self.current_sds_weight, kwargs,
                offload_teacher=bool(self.appearance_cfg.get("offload_teacher", True)),
            )
            for parameter, gradient in gradients:
                if parameter.grad is not None:
                    raise RuntimeError("Split gradients must start from an empty optimizer window.")
            for parameter, gradient in gradients:
                parameter.grad = gradient.detach()
            losses.update(metrics)
        finally:
            self._split_pending = None

    def style_optimizers(self):
        result = [self.optimizer]
        if getattr(self, "geometry_optimizer", None) is not None:
            result.append(self.geometry_optimizer)
        return result

    def gradient_diagnostics(self, reference):
        zero = reference.detach().new_zeros(())

        def norm_or_zero(parameter):
            if parameter.grad is None:
                return zero
            return parameter.grad.detach().float().norm()

        diagnostics = {
            "gaussian_style_dc_parameter_grad_norm": norm_or_zero(
                self.sh_dc
            ),
            "gaussian_style_rest_parameter_grad_norm": norm_or_zero(
                self.sh_rest
            ),
        }
        geometry_norms = []
        if self.geometry_enabled:
            for name, parameter in self.geometry_parameters().items():
                grad_norm = norm_or_zero(parameter)
                diagnostics[
                    f"gaussian_style_geometry_{name}_grad_norm"
                ] = grad_norm
                if parameter.grad is not None:
                    geometry_norms.append(grad_norm)
        diagnostics["gaussian_style_geometry_grad_norm"] = (
            torch.stack(geometry_norms).square().sum().sqrt()
            if geometry_norms
            else zero
        )
        return diagnostics

    def step(self):
        self._check_finite_style_state("before optimizer step", optimizer_state=True)
        if self.split_supervision:
            if self._split_pending is not None or self._video_render_buffer:
                raise RuntimeError("Cannot step inside a split style video window.")
            snapshots = {}
            for optimizer, clip in zip(self.style_optimizers(), (self.appearance_grad_clip, self.geometry_grad_clip)):
                active = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
                if clip > 0 and active:
                    torch.nn.utils.clip_grad_norm_(active, clip, error_if_nonfinite=True)
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            snapshots[group["name"]] = (parameter, parameter.detach().clone())
            for optimizer in self.style_optimizers():
                optimizer.step()
            self.last_update_metrics = {
                f"gaussian_style_update_rms/{name}": (p.detach() - before).float().square().mean().sqrt()
                for name, (p, before) in snapshots.items()
            }
            self._check_finite_style_state("after optimizer step", optimizer_state=True)
            self.zero_grad()
            return
        params = [
            parameter
            for parameter in (self.sh_dc, self.sh_rest)
            if parameter.requires_grad
        ]
        if self.geometry_enabled:
            params.extend(
                parameter
                for parameter in self.geometry_parameters().values()
                if parameter.requires_grad
            )
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                params, self.grad_clip, error_if_nonfinite=True
            )
        self.optimizer.step()
        self._check_finite_style_state("after optimizer step", optimizer_state=True)
        self.optimizer.zero_grad(set_to_none=True)

    def zero_grad(self):
        if self.enabled:
            for optimizer in self.style_optimizers():
                optimizer.zero_grad(set_to_none=True)


def save_mags_checkpoint(
    config,
    iteration,
    gaussians,
    deformer,
    gaussian_style_update=None,
    mesh_style_update=None,
):
    direct_style = (
        gaussian_style_update is not None
        and gaussian_style_update.enabled
        and gaussian_style_update.direct_mode
    )
    if direct_style:
        if gaussian_style_update.video_guidance and (gaussian_style_update._video_render_buffer or gaussian_style_update._split_pending is not None):
            raise RuntimeError("Cannot save direct parameters with an unfinished video window.")
        assert_separate_style_output(config.model.model_path, gaussian_style_update.source_checkpoint)
        assert_separate_style_output(config.model.model_path, gaussian_style_update.pretrained_checkpoint)
    checkpoint_dir = os.path.join(config.model.model_path, "checkpoints")
    if direct_style:
        assert_separate_style_output(checkpoint_dir, gaussian_style_update.source_checkpoint)
        assert_separate_style_output(checkpoint_dir, gaussian_style_update.pretrained_checkpoint)
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "iteration": iteration,
        "gaussians": gaussians.state_dict(),
        "active_sh_degree": gaussians.active_sh_degree,
        "faceid": gaussians.faceid.detach().cpu(),
        "xyz": gaussians.get_xyz.detach().cpu(),
        "max_radii2D": gaussians.max_radii2D.detach().cpu(),
        "xyz_gradient_accum": gaussians.xyz_gradient_accum.detach().cpu(),
        "denom": gaussians.denom.detach().cpu(),
        "deformer": deformer.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    style_state_saved = False
    if gaussian_style_update is not None and gaussian_style_update.enabled:
        payload["gaussian_style_optimizer"] = (
            gaussian_style_update.optimizer.state_dict()
        )
        if gaussian_style_update.split_supervision:
            payload["split_style_reference"] = gaussian_style_update.source_reference
            if gaussian_style_update.geometry_optimizer is not None:
                payload["gaussian_geometry_optimizer"] = gaussian_style_update.geometry_optimizer.state_dict()
        payload["gaussian_style_parameterization"] = gaussian_style_update.parameterization
        if direct_style:
            payload["direct_style_state"] = gaussian_style_update.direct_state_dict()
            payload["source_checkpoint"] = gaussian_style_update.source_checkpoint
            payload["pretrained_checkpoint"] = gaussian_style_update.pretrained_checkpoint
        else:
            payload["gaussian_style_delta"] = {
                name: tensor.detach().cpu()
                for name, tensor in gaussian_style_update.style_delta.state_dict().items()
            }
        if gaussian_style_update.geometry_enabled and not direct_style:
            payload["gaussian_style_geometry_residual"] = {
                name: tensor.detach().cpu()
                for name, tensor in (
                    gaussian_style_update.geometry_residual.state_dict().items()
                )
            }
        payload["gaussian_style"] = OmegaConf.to_container(
            gaussian_style_update.cfg, resolve=True
        )
        outer_state = gaussian_style_update.outer_state_dict(iteration)
        if outer_state is not None:
            payload["gaussian_style_outer_state"] = outer_state
        style_state_saved = True
    if mesh_style_update is not None and mesh_style_update.enabled:
        payload["mesh_style_log_jac"] = (
            mesh_style_update.aligner.style_log_jac.detach().cpu()
        )
        if mesh_style_update.aligner.style_log_jac.grad is not None:
            payload["mesh_style_log_jac_grad"] = (
                mesh_style_update.aligner.style_log_jac.grad.detach().cpu()
            )
        payload["mesh_style_optimizer"] = (
            mesh_style_update.optimizer.state_dict()
        )
        payload["mesh_style"] = OmegaConf.to_container(
            mesh_style_update.cfg, resolve=True
        )
        style_state_saved = True
    if not style_state_saved:
        payload["deformer_optimizer"] = deformer.optimizer.state_dict()

    checkpoint_path = os.path.join(
        checkpoint_dir, f"iteration_{iteration}.pth"
    )
    if direct_style:
        # Never replace an existing checkpoint, even inside a new direct run.
        with open(checkpoint_path, "xb") as checkpoint_file:
            torch.save(payload, checkpoint_file)
    else:
        torch.save(payload, checkpoint_path)
    print(f"[MaGS] Saved checkpoint: {checkpoint_path}")


def collect_loss(config, gt_image, image):
    Ll1 = l1_loss(image, gt_image)

    if config.loss.lambda_ssim > 0:
        Lssim = 1.0 - ssim(image, gt_image)
        loss = (1.0 - config.loss.lambda_ssim) * Ll1 + config.loss.lambda_ssim * Lssim
    else:
        loss = Ll1
    return {"loss": loss}


@torch.no_grad()
def prepare_split_style_reference(config, checkpoint, gaussians, cameras, faces, meshes):
    from utils.split_style_training import reference_signature
    signature = reference_signature(cameras, faces, meshes["train_verts"])
    saved = checkpoint.get("split_style_reference")
    if saved is not None:
        if (saved["signature"] != signature
                or saved.get("white_background") != bool(config.dataset.white_background)
                or set(saved["images"]) != {cam.uid for cam in cameras}):
            raise ValueError("Split style reference does not match the current cameras, meshes, or foreground masks.")
        return {**saved, "images": {uid: image.detach().float().cpu() for uid, image in saved["images"].items()}}
    if checkpoint.get("gaussian_style_parameterization") == "direct":
        raise ValueError("A styled checkpoint without a fixed split reference cannot initialize this mode. Use the original MaGS checkpoint.")
    background = torch.tensor([1, 1, 1] if config.dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device=gaussians._uvw.device)
    images = {}
    directory = os.path.join(config.model.model_path, "style_progress", "source_reference")
    os.makedirs(directory, exist_ok=True)
    for cam in cameras:
        gaussians.update_mesh(faces, meshes["train_verts"][cam.uid],
                              meshes["train_normals"][cam.uid], deformer=None)
        image = inria_render(cam, gaussians, config.pipe, background)["render"]
        require_finite(image, f"fixed no-deformer reference frame {cam.uid}")
        images[cam.uid] = image.detach().float().clamp(0, 1).cpu()
        torchvision.utils.save_image(images[cam.uid], os.path.join(directory, f"view_{cam.uid}.png"))
    return {"version": 1, "signature": signature, "images": images,
            "white_background": bool(config.dataset.white_background)}


def training(config):
    train_dataset = limit_dataset_frames(config, get_dataset(config, train=True), True)
    test_dataset = limit_dataset_frames(config, get_dataset(config, train=False), False)
    if config.get("evaluation", {}).get("enabled", False):
        from utils.paper_eval_export import validate_paper_evaluation
        validate_paper_evaluation(config, train_dataset)
    refined_faces = train_dataset[0].mesh_faces.clone().cuda()
    refined_meshes = {
        "train_verts": torch.stack(
            [i.mesh_vertices.clone().cuda() for i in train_dataset]
        ),
        "train_normals": torch.stack(
            [i.mesh_normals.clone().cuda() for i in train_dataset]
        ),
        "test_verts": torch.stack(
            [i.mesh_vertices.clone().cuda() for i in test_dataset]
        ),
        "test_normals": torch.stack(
            [i.mesh_normals.clone().cuda() for i in test_dataset]
        ),
    }

    gaussian_style_cfg = config.optim.get("gaussian_style_update", {})
    gaussian_style_enabled = bool(
        gaussian_style_cfg and gaussian_style_cfg.get("enabled", False)
    )
    mesh_style_cfg = config.optim.get("mesh_style_update", {})
    mesh_style_enabled = bool(
        mesh_style_cfg and mesh_style_cfg.get("enabled", False)
    )
    joint_style_cfg = config.optim.get("joint_style_update", {})
    joint_style_enabled = bool(
        joint_style_cfg and joint_style_cfg.get("enabled", False)
    )
    if gaussian_style_enabled and mesh_style_enabled and not joint_style_enabled:
        raise ValueError(
            "gaussian_style_update and mesh_style_update cannot be enabled in "
            "the same run unless optim.joint_style_update.enabled=true."
        )
    if joint_style_enabled and not (
        gaussian_style_enabled and mesh_style_enabled
    ):
        raise ValueError(
            "joint_style_update requires both gaussian_style_update.enabled=true "
            "and mesh_style_update.enabled=true."
        )
    if joint_style_enabled and bool(
        gaussian_style_cfg.get("outer_loop", {}).get("enabled", False)
    ):
        raise ValueError(
            "joint_style_update currently requires "
            "gaussian_style_update.outer_loop.enabled=false so both branches "
            "share exactly the same CSD timestep and noise."
        )
    if joint_style_enabled:
        gaussian_style_path = str(
            gaussian_style_cfg.get("style_path", "")
            or gaussian_style_cfg.get("sds", {}).get("style_path", "")
        )
        mesh_style_path = str(
            mesh_style_cfg.get("style_path", "")
            or mesh_style_cfg.get("sds", {}).get("style_path", "")
        )
        if (
            gaussian_style_path
            and mesh_style_path
            and gaussian_style_path != mesh_style_path
        ):
            raise ValueError(
                "Joint SH/Jacobian style training must use the same style_path "
                "for both branches."
            )
        shared_style_path = gaussian_style_path or mesh_style_path
        if shared_style_path:
            gaussian_style_cfg.style_path = shared_style_path
            mesh_style_cfg.style_path = shared_style_path

    gaussians = MaGSModel(config)
    gaussians.create_from_mesh(
        refined_meshes["train_verts"][0],
        refined_faces,
        refined_meshes["train_normals"][0],
    )
    config.deform.control_points = train_dataset[0].mesh_vertices.shape[0]
    deformer = DeformModel(config.deform).cuda()
    checkpoint_data = {}
    if gaussian_style_enabled or mesh_style_enabled:
        gaussian_checkpoint = str(
            gaussian_style_cfg.get("start_checkpoint", "")
        )
        mesh_checkpoint = str(mesh_style_cfg.get("start_checkpoint", ""))
        if (
            joint_style_enabled
            and gaussian_checkpoint
            and mesh_checkpoint
            and gaussian_checkpoint != mesh_checkpoint
        ):
            raise ValueError(
                "Joint SH/Jacobian style training must load one shared MaGS "
                "start_checkpoint."
            )
        start_checkpoint = (
            gaussian_checkpoint
            if gaussian_style_enabled and gaussian_checkpoint
            else mesh_checkpoint
        )
        checkpoint_data = load_pretrained_mags_checkpoint(
            start_checkpoint,
            gaussians,
            deformer,
            "cuda",
        )
    gaussians.training_setup(config.optim)
    nerf_normalization = getNerfppNorm(train_dataset)
    cameras_extent = (
        nerf_normalization["radius"]
        if nerf_normalization["radius"] > 0
        else 2.0
    )

    source_reference = None
    if gaussian_style_enabled and gaussian_style_cfg.get("supervision_mode", "diffusion") == "split_vgg_diffusion":
        if config.optim.random_bg:
            raise ValueError("Split supervision requires random_bg=false for a fixed source reference.")
        source_reference = prepare_split_style_reference(
            config, checkpoint_data, gaussians, train_dataset, refined_faces, refined_meshes,
        )
    gaussian_style_update = GaussianSHStyleController(
        config,
        gaussians,
        deformer,
        checkpoint_data,
        "cuda",
        train_dataset,
        source_reference=source_reference,
    )
    if gaussian_style_update.video_guidance and joint_style_enabled:
        raise ValueError(
            "Window-level SDXL AnimateDiff/ControlNet guidance currently "
            "supports the Gaussian SH branch only. Disable mesh_style_update "
            "and joint_style_update for this experiment."
        )
    shared_sds_guidance = (
        gaussian_style_update.sds_guidance
        if joint_style_enabled
        else None
    )
    mesh_style_update = build_mesh_style_update(
        config,
        refined_meshes["train_verts"][0],
        refined_faces,
        "cuda",
        gaussians=gaussians,
        deformer=deformer,
        checkpoint=checkpoint_data,
        train_dataset=train_dataset,
        shared_sds_guidance=shared_sds_guidance,
    )
    if joint_style_enabled:
        print(
            "[MaGS] Joint style update enabled: CSD -> SH for appearance, "
            "CSD/geometry regularization -> canonical Jacobian for shape."
        )

    bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.model.model_path, "tensorboard"))
    if joint_style_enabled:
        loop_start_iteration = max(
            gaussian_style_update.loop_start_iteration,
            mesh_style_update.loop_start_iteration,
        )
    else:
        loop_start_iteration = (
            gaussian_style_update.loop_start_iteration
            if gaussian_style_update.enabled
            else (
                mesh_style_update.loop_start_iteration
                if mesh_style_update.enabled
                else 1
            )
        )
    progress_bar = tqdm(
        range(loop_start_iteration, config.optim.iterations + 1),
        desc="Training progress",
        dynamic_ncols=True,
    )
    BATCH_SIZE = config.optim.batch_size
    batch_gaussian_loss = 0.0
    disable_deformer = bool(config.optim.get("disable_deformer", False))
    gt_mesh_trajectory_enabled = (
        gaussian_style_update.enabled
        and gaussian_style_update.gt_mesh_trajectory_enabled
    )
    trajectory_train_dataset = (
        sorted(train_dataset, key=lambda camera: int(camera.uid))
        if gt_mesh_trajectory_enabled
        else train_dataset
    )
    mesh_style_entry_logged = False
    joint_appearance_boundary_verified = False
    joint_geometry_boundary_verified = False
    if mesh_style_update.enabled:
        print(
            "[MaGS] Gaussian parameters and deformer are frozen; the deformer "
            "is used for dynamic forward only."
        )
    elif gt_mesh_trajectory_enabled:
        print(
            "[MaGS] Dynamic geometry is driven directly by the ordered GT-mesh "
            f"sequence (deformer_mode={gaussian_style_update.deformer_mode})."
        )
    elif disable_deformer and not gaussian_style_update.enabled:
        print("[MaGS] Deformer disabled for this training run.")
    elif disable_deformer and gaussian_style_update.enabled:
        print(
            "[MaGS] gaussian_style_update.use_deformer overrides "
            "optim.disable_deformer for frozen dynamic forward only."
        )

    if gaussian_style_update.enabled and gaussian_style_update.direct_mode:
        with torch.no_grad():
            gaussian_style_update.apply_geometry_to_gaussians(detach=True)
            gaussian_style_update.apply_to_gaussians(detach=True)
            initial_cam = trajectory_train_dataset[0]
            gaussians.update_mesh(
                refined_faces,
                refined_meshes["train_verts"][initial_cam.uid],
                refined_meshes["train_normals"][initial_cam.uid],
                deformer=None,
            )
            initial_image = inria_render(initial_cam, gaussians, config.pipe, background)["render"]
            require_finite(initial_image, "direct style no-deformer baseline")
            preview_dir = os.path.join(config.model.model_path, "style_progress")
            os.makedirs(preview_dir, exist_ok=True)
            baseline_path = os.path.join(preview_dir, "initial_no_deformer.png")
            torchvision.utils.save_image(initial_image.clamp(0, 1), baseline_path)
            print(f"[MaGS] Saved no-deformer baseline: {baseline_path}")

    pending_checkpoint = False
    paper_optimized_uids = set()
    for iteration in progress_bar:
        if (
            not gaussian_style_update.enabled
            and not mesh_style_update.enabled
            and iteration % 100 == 0
        ):
            gaussians.oneupSHdegree()
        if (
            not mesh_style_update.enabled
            and not gt_mesh_trajectory_enabled
            and iteration % len(train_dataset) == 0
        ):
            random.shuffle(train_dataset)

        if (
            gt_mesh_trajectory_enabled
            and iteration >= gaussian_style_update.start_iter
        ):
            frame_index = gaussian_style_update.frame_index(
                iteration, len(trajectory_train_dataset)
            )
            cam = trajectory_train_dataset[frame_index]
        elif mesh_style_update.enabled and iteration >= mesh_style_update.start_iter:
            frame_index = (
                iteration - mesh_style_update.start_iter
            ) % len(train_dataset)
            cam = train_dataset[frame_index]
        else:
            cam = train_dataset[iteration % len(train_dataset)]
        gaussian_style_update.prepare_iteration(iteration)
        gaussian_style_train_active = gaussian_style_update.is_training_step(iteration)
        if gaussian_style_train_active:
            paper_optimized_uids.add(int(cam.uid))
        mesh_style_train_active = mesh_style_update.is_training_step(iteration)
        mesh_style_apply_active = mesh_style_update.should_apply_to_gaussian(iteration)
        if mesh_style_train_active and not mesh_style_entry_logged:
            if joint_style_enabled:
                print(
                    f"[MaGS] Entering joint SH/Jacobian style update at "
                    f"iteration {iteration} (end={mesh_style_update.end_iter}, "
                    f"window={mesh_style_update.window_size}, "
                    "appearance=CSD->SH, shape=CSD+regularizers->Jacobian)."
                )
            else:
                print(
                    f"[MaGS] Entering dynamic Jacobian style update at iteration "
                    f"{iteration} (end={mesh_style_update.end_iter}, "
                    f"window={mesh_style_update.window_size}, "
                    "loss=CSD+Jacobian+Poisson+Flip+Delta)."
                )
            mesh_style_entry_logged = True
        if mesh_style_update.enabled:
            use_deformer = mesh_style_update.use_deformer
        elif gaussian_style_update.enabled:
            use_deformer = gaussian_style_update.use_deformer
        else:
            use_deformer = not disable_deformer and iteration > config.optim.warm_up
        deformation_components = (
            gaussian_style_update.deformation_components
            if gaussian_style_update.enabled
            else None
        )
        if config.optim.random_bg:
            background = torch.rand(3, dtype=torch.float32, device="cuda")
        gt_image = cam.original_image.cuda()
        align_pkg = None
        loss_dict = {}
        mesh_style_loss = None
        gaussian_style_loss = None
        video_window_ready = False
        gaussian_train_active = (
            not gaussian_style_update.enabled and not mesh_style_update.enabled
        )
        if gaussian_style_update.enabled:
            # Geometry style is mesh-local and must be visible while update_mesh
            # builds the current frame's Gaussian centers.
            gaussian_style_update.apply_geometry_to_gaussians(
                detach=not gaussian_style_train_active
            )
        else:
            gaussians.style_uvw_delta = None
            gaussians.style_log_scaling_delta = None
            gaussians.style_rotation_delta = None
        if joint_style_enabled:
            shared_guidance_kwargs = None
            if mesh_style_update.should_align(iteration):
                shared_guidance_kwargs = mesh_style_update.guidance_kwargs(iteration)

            # Appearance branch: the original dynamic mesh is fixed and CSD can
            # reach only the trainable SH delta.
            if gaussian_style_train_active:
                jacobian_leak = []
                jacobian_hook = None
                if not joint_appearance_boundary_verified:
                    jacobian_hook = (
                        mesh_style_update.aligner.style_log_jac.register_hook(
                            lambda grad: jacobian_leak.append(True)
                        )
                    )
                gaussians.update_mesh(
                    refined_faces,
                    refined_meshes["train_verts"][cam.uid],
                    refined_meshes["train_normals"][cam.uid],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][cam.uid],
                    deformer if use_deformer else None,
                    deformation_components=deformation_components,
                )
                gaussian_style_update.apply_to_gaussians(detach=False)
                appearance_render_pkg = inria_render(
                    cam,
                    gaussians,
                    config.pipe,
                    background,
                    iteration=iteration,
                )
                appearance_image = appearance_render_pkg["render"]
                appearance_losses = gaussian_style_update.loss(
                    appearance_image,
                    source_key=cam.uid,
                    guidance_kwargs=shared_guidance_kwargs,
                )
                gaussian_style_loss = appearance_losses["gaussian_style_loss"]
                (
                    gaussian_style_loss
                    / gaussian_style_update.accumulation_divisor(
                        iteration, BATCH_SIZE
                    )
                ).backward()
                if jacobian_hook is not None:
                    jacobian_hook.remove()
                    if jacobian_leak:
                        raise RuntimeError(
                            "Appearance CSD leaked into canonical style_log_jac."
                        )
                    joint_appearance_boundary_verified = True
                    print(
                        "[MaGS] Verified appearance boundary: CSD -> SH only; "
                        "canonical Jacobian is disconnected."
                    )
                gaussian_style_update.verify_gradient_boundary()
                loss_dict.update(
                    {
                        name: value.detach()
                        for name, value in appearance_losses.items()
                    }
                )
                loss_dict.update(
                    gaussian_style_update.gradient_diagnostics(appearance_image)
                )
                del appearance_render_pkg, appearance_image, appearance_losses

            # Geometry branch: the current SH style is visible to the teacher,
            # but detached so this loss can reach only the canonical Jacobian.
            if mesh_style_train_active:
                align_pkg = mesh_style_update.align(
                    refined_meshes["train_verts"][cam.uid]
                )
            elif mesh_style_apply_active:
                with torch.no_grad():
                    align_pkg = mesh_style_update.align(
                        refined_meshes["train_verts"][cam.uid]
                    )
            if mesh_style_apply_active:
                gaussians.update_mesh(
                    refined_faces,
                    align_pkg["aligned_verts"],
                    align_pkg["aligned_normals"],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][cam.uid],
                    deformer if use_deformer else None,
                    detach_mesh=not mesh_style_train_active,
                    deformation_components=deformation_components,
                )
            else:
                gaussians.update_mesh(
                    refined_faces,
                    refined_meshes["train_verts"][cam.uid],
                    refined_meshes["train_normals"][cam.uid],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][cam.uid],
                    deformer if use_deformer else None,
                    deformation_components=deformation_components,
                )
            gaussian_style_update.apply_to_gaussians(detach=True)
            render_pkg = inria_render(
                cam,
                gaussians,
                config.pipe,
                background,
                iteration=iteration,
            )
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            if mesh_style_train_active:
                sh_leak = []
                sh_hooks = []
                if not joint_geometry_boundary_verified:
                    for parameter in (
                        gaussian_style_update.style_delta.dc_delta,
                        gaussian_style_update.style_delta.rest_delta,
                    ):
                        if parameter.requires_grad:
                            sh_hooks.append(
                                parameter.register_hook(
                                    lambda grad: sh_leak.append(True)
                                )
                            )
                geometry_losses = mesh_style_update.loss(
                    align_pkg,
                    image,
                    cam.uid,
                    iteration,
                )
                mesh_style_loss = geometry_losses["mesh_style_update_loss"]
                (
                    mesh_style_loss / mesh_style_update.window_divisor(iteration)
                ).backward()
                for hook in sh_hooks:
                    hook.remove()
                if sh_hooks:
                    if sh_leak:
                        raise RuntimeError(
                            "Geometry CSD leaked into the Gaussian SH style delta."
                        )
                    joint_geometry_boundary_verified = True
                    print(
                        "[MaGS] Verified geometry boundary: CSD/regularizers -> "
                        "canonical Jacobian only; SH delta is detached."
                    )
                mesh_style_update.verify_gradient_boundary()
                geometry_losses["mesh_style_update_jacobian_grad_norm"] = (
                    mesh_style_update.gradient_norm()
                )
                loss_dict.update(
                    {
                        name: value.detach()
                        for name, value in geometry_losses.items()
                    }
                )
            zero = image.detach().new_tensor(0.0)
            appearance_value = loss_dict.get("gaussian_style_loss", zero)
            geometry_value = loss_dict.get("mesh_style_update_loss", zero)
            loss = appearance_value + geometry_value
            gaussian_style_update.apply_to_gaussians(detach=False)
        else:
            if mesh_style_train_active or mesh_style_apply_active:
                if mesh_style_train_active:
                    align_pkg = mesh_style_update.align(
                        refined_meshes["train_verts"][cam.uid]
                    )
                else:
                    with torch.no_grad():
                        align_pkg = mesh_style_update.align(
                            refined_meshes["train_verts"][cam.uid]
                        )
            if mesh_style_apply_active:
                gaussians.update_mesh(
                    refined_faces,
                    align_pkg["aligned_verts"],
                    align_pkg["aligned_normals"],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][cam.uid],
                    deformer if use_deformer else None,
                    detach_mesh=not mesh_style_train_active,
                    deformation_components=deformation_components,
                )
            else:
                gaussians.update_mesh(
                    refined_faces,
                    refined_meshes["train_verts"][cam.uid],
                    refined_meshes["train_normals"][cam.uid],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][cam.uid],
                    deformer if use_deformer else None,
                    deformation_components=deformation_components,
                )
            if gaussian_style_update.enabled:
                gaussian_style_update.apply_to_gaussians()
            else:
                gaussians.style_features_delta = None
            render_pkg = inria_render(
                cam,
                gaussians,
                config.pipe,
                background,
                iteration=iteration,
            )
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            if gaussian_style_train_active:
                loss_dict = gaussian_style_update.loss(
                    image,
                    source_key=cam.uid,
                    guidance_kwargs=gaussian_style_update.guidance_kwargs(iteration),
                )
                video_window_ready = bool(
                    loss_dict.get(
                        "gaussian_style_video_window_ready",
                        image.detach().new_tensor(1.0),
                    ).item()
                )
                if video_window_ready:
                    gaussian_style_loss = loss_dict["gaussian_style_loss"]
                    loss = gaussian_style_loss
                else:
                    gaussian_style_loss = None
                    loss = image.detach().new_zeros(())
            elif gaussian_style_update.enabled:
                loss = image.detach().new_tensor(0.0)
                loss_dict = {"gaussian_style_loss": loss}
            elif mesh_style_train_active:
                loss_dict = mesh_style_update.loss(
                    align_pkg,
                    image,
                    cam.uid,
                    iteration,
                )
                mesh_style_loss = loss_dict["mesh_style_update_loss"]
                loss = mesh_style_loss
            elif mesh_style_update.enabled:
                loss = image.detach().new_tensor(0.0)
                loss_dict = {"mesh_style_update_loss": loss}
            else:
                loss_dict = collect_loss(config, gt_image, image)
                loss = loss_dict["loss"]
            if gaussian_train_active:
                batch_gaussian_loss = batch_gaussian_loss + loss_dict["loss"]
            if mesh_style_loss is not None:
                (
                    mesh_style_loss / mesh_style_update.window_divisor(iteration)
                ).backward()
                mesh_style_update.verify_gradient_boundary()
                loss_dict["mesh_style_update_jacobian_grad_norm"] = (
                    mesh_style_update.gradient_norm()
                )
            if gaussian_style_loss is not None:
                gaussian_style_update.backward_style_loss(
                    loss_dict, gaussian_style_update.accumulation_divisor(iteration, BATCH_SIZE),
                )
                loss = loss_dict["gaussian_style_loss"]
                gaussian_style_update.verify_gradient_boundary()
                loss_dict.update(
                    gaussian_style_update.gradient_diagnostics(image)
                )

            if iteration % BATCH_SIZE == 0:
                if gaussian_train_active:
                    (batch_gaussian_loss / BATCH_SIZE).backward()
                batch_gaussian_loss = 0.0

        current_psnr = psnr(
            image.unsqueeze(0).contiguous(), gt_image.unsqueeze(0)
        ).item()
        if joint_style_enabled:
            stage_name = (
                "joint-style"
                if gaussian_style_train_active or mesh_style_train_active
                else "joint-frozen"
            )
        elif gaussian_style_update.enabled:
            stage_name = gaussian_style_update.stage_name(iteration)
        elif mesh_style_train_active:
            stage_name = "mesh-style"
        elif mesh_style_update.enabled and iteration < mesh_style_update.start_iter:
            stage_name = "mesh-pre"
        elif mesh_style_update.enabled:
            stage_name = "mesh-post"
        else:
            stage_name = "gaussian"
        postfix = {
            "#gauss": gaussians.xyz.shape[0],
            "#faces": refined_faces.shape[0],
            "#verts": refined_meshes["train_verts"][cam.uid].shape[0],
            "stage": stage_name,
            "loss": f"{loss.detach().float().item():.4f}",
            "psnr": current_psnr,
        }
        if mesh_style_train_active:
            postfix.update(
                {
                    "mesh_style": f"{loss_dict['mesh_style_update_loss'].detach().float().item():.4f}",
                    "geometry_csd" if joint_style_enabled else "csd": f"{loss_dict['mesh_style_update_csd_loss'].detach().float().item():.4f}",
                    "jac": f"{loss_dict['mesh_style_update_jacobian_loss'].detach().float().item():.4f}",
                    "poisson": f"{loss_dict['mesh_style_update_poisson_loss'].detach().float().item():.4f}",
                    "flip": f"{loss_dict['mesh_style_update_flip_loss'].detach().float().item():.4f}",
                    "delta": f"{loss_dict['mesh_style_update_delta_loss'].detach().float().item():.4f}",
                    "jac_grad": f"{loss_dict['mesh_style_update_jacobian_grad_norm'].detach().float().item():.4f}",
                }
            )
        if gaussian_style_train_active:
            postfix.update(
                {
                    "appearance_csd" if joint_style_enabled else "csd": f"{loss_dict['gaussian_style_csd_loss'].detach().float().item():.4f}",
                    "dc_rms": f"{loss_dict['gaussian_style_dc_parameter_rms'].item():.4f}",
                    "rest_rms": f"{loss_dict['gaussian_style_rest_parameter_rms'].item():.4f}",
                    "csd_grad": f"{loss_dict['gaussian_style_csd_grad_norm'].item():.4f}",
                }
            )
            if gaussian_style_update.split_supervision:
                postfix["vgg"] = f"{loss_dict.get('gaussian_style_appearance_loss', image.new_zeros(())).item():.4f}"
            if gaussian_style_update.gt_mesh_trajectory_enabled:
                postfix.update(
                    {
                        "window": int(
                            loss_dict["gaussian_style_trajectory_window"].item()
                        ),
                        "slot": (
                            f"{int(loss_dict['gaussian_style_trajectory_slot'].item()) + 1}/"
                            f"{gaussian_style_update.trajectory_window_size}"
                        ),
                        "frame": cam.uid,
                    }
                )
            if gaussian_style_update.geometry_enabled and gaussian_style_update.direct_mode:
                postfix.update({
                    name: f"{loss_dict[f'gaussian_style_geometry_{name}_parameter_rms'].item():.5f}"
                    for name in ("uvw", "scaling", "rotation")
                })
                postfix["geo_grad"] = f"{loss_dict.get('gaussian_style_geometry_grad_norm', image.new_zeros(())).item():.4f}"
            elif gaussian_style_update.geometry_enabled:
                postfix.update(
                    {
                        "geo_uv": f"{loss_dict['gaussian_style_geometry_uv_rms'].item():.5f}",
                        "geo_n": f"{loss_dict['gaussian_style_geometry_normal_rms'].item():.5f}",
                        "geo_s": f"{loss_dict['gaussian_style_geometry_scaling_rms'].item():.5f}",
                        "geo_r": f"{loss_dict['gaussian_style_geometry_rotation_rms'].item():.5f}",
                        "geo_grad": f"{loss_dict.get('gaussian_style_geometry_grad_norm', image.new_zeros(())).item():.4f}",
                    }
                )
            if gaussian_style_update.outer_enabled:
                postfix.update(
                    {
                        "outer": (
                            f"{gaussian_style_update.current_outer + 1}/"
                            f"{gaussian_style_update.outer_iterations}"
                        ),
                        "inner": (
                            f"{gaussian_style_update.current_inner + 1}/"
                            f"{gaussian_style_update.inner_iterations}"
                        ),
                        "noise": gaussian_style_update.current_noise_slot,
                        "strength": (
                            f"{gaussian_style_update.style_strength_schedule[gaussian_style_update.current_outer]:.2f}"
                        ),
                    }
                )
        progress_bar.set_postfix(postfix)

        writer.add_scalar("Loss/train", loss.item(), iteration)
        for loss_name, loss_value in loss_dict.items():
            writer.add_scalar(
                f"Loss/{loss_name}",
                loss_value.detach().float().item(),
                iteration,
            )
        writer.add_scalar("PSNR/train", current_psnr, iteration)
        writer.add_scalar("Gaussians/count", gaussians.xyz.shape[0], iteration)
        writer.add_scalar("Faces/count", refined_faces.shape[0], iteration)
        writer.add_scalar("Stage/gaussian_train_active", float(gaussian_train_active), iteration)
        writer.add_scalar(
            "Stage/mesh_style_train_active",
            float(mesh_style_train_active),
            iteration,
        )
        writer.add_scalar(
            "Stage/gaussian_style_train_active",
            float(gaussian_style_train_active),
            iteration,
        )
        writer.add_scalar("Stage/deformer_active", float(use_deformer), iteration)
        if mesh_style_update.enabled:
            writer.add_scalar(
                "LearningRate/style_log_jac",
                mesh_style_update.optimizer.param_groups[0]["lr"],
                iteration,
            )
        if gaussian_style_update.enabled:
            writer.add_scalar(
                "LearningRate/style_features_dc",
                gaussian_style_update.optimizer.param_groups[0]["lr"],
                iteration,
            )
            writer.add_scalar(
                "LearningRate/style_features_rest",
                gaussian_style_update.optimizer.param_groups[1]["lr"],
                iteration,
            )
            for group in gaussian_style_update.optimizer.param_groups[2:]:
                writer.add_scalar(
                    f"LearningRate/{group['name']}", group["lr"], iteration
                )
        if not mesh_style_update.enabled and not gaussian_style_update.enabled:
            writer.add_scalar(
                "LearningRate/optim_uvw",
                gaussians.optimizer.param_groups[0]["lr"],
                iteration,
            )

        outer_iteration_end = (
            gaussian_style_train_active
            and gaussian_style_update.outer_enabled
            and gaussian_style_update.current_inner + 1
            == gaussian_style_update.inner_iterations
        )

        if mesh_style_update.is_window_end(iteration):
            mesh_style_update.step()
        elif mesh_style_update.enabled and not mesh_style_train_active:
            mesh_style_update.zero_grad()

        with torch.no_grad():
            if iteration % config.dataset.test_per_iter == 0:
                render_test(
                    config,
                    gaussians,
                    deformer,
                    refined_meshes,
                    refined_faces,
                    iteration,
                    train_dataset,
                    test_dataset,
                    background,
                    writer,
                    disable_deformer,
                    mesh_style_update,
                    gaussian_style_update,
                )


            if gaussian_train_active and iteration < config.optim.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )

                if (
                    iteration > config.optim.densify_from_iter
                    and iteration % config.optim.densification_interval == 0
                ):
                    size_threshold = (
                        config.optim.size_threshold
                        if iteration > config.optim.opacity_reset_interval
                        else None
                    )
                    gaussians.densify_and_prune(
                        config.optim.densify_grad_threshold,
                        config.optim.min_opacity,
                        cameras_extent,
                        size_threshold,
                    )

                if (
                    iteration > config.optim.densify_from_iter
                    and iteration % config.optim.opacity_reset_interval == 0
                ):
                    gaussians.reset_opacity()


            if iteration < config.optim.iterations:
                if gaussian_train_active:
                    gaussians.optimizer.step()
                    gaussians.update_learning_rate(iteration)
                    gaussians.optimizer.zero_grad(set_to_none=True)
                if use_deformer and gaussian_train_active:
                    deformer.optimizer.step()
                    deformer.update_learning_rate(iteration)
                    deformer.optimizer.zero_grad(set_to_none=True)
                else:
                    deformer.optimizer.zero_grad(set_to_none=True)

            style_batch_end = gaussian_style_update.is_accumulation_end(
                iteration, BATCH_SIZE
            )
            if style_batch_end:
                gaussian_style_update.step()
                for name, value in getattr(gaussian_style_update, "last_update_metrics", {}).items():
                    writer.add_scalar(name, value.item(), iteration)

            render_interval = int(config.dataset.get("render_per_iter", 0))
            if (
                (gaussian_style_update.enabled or mesh_style_update.enabled)
                and (
                    (
                        render_interval > 0
                        and iteration % render_interval == 0
                    )
                    or outer_iteration_end
                )
            ):
                style_progress_dir = os.path.join(
                    config.model.model_path, "style_progress"
                )
                os.makedirs(style_progress_dir, exist_ok=True)
                render_path = os.path.join(
                    style_progress_dir,
                    f"iter_{iteration:05d}_view_{cam.uid}_styled.png",
                )
                torchvision.utils.save_image(image.clamp(0.0, 1.0), render_path)
                print(f"[MaGS] Saved style render: {render_path}")
                if (
                    gaussian_style_update.video_guidance
                    and video_window_ready
                    and gaussian_style_update.last_video_preview is not None
                ):
                    video_grid = torchvision.utils.make_grid(
                        gaussian_style_update.last_video_preview[0],
                        nrow=gaussian_style_update.trajectory_window_size,
                        padding=2,
                    )
                    video_path = os.path.join(
                        style_progress_dir,
                        f"iter_{iteration:05d}_video_window_styled.png",
                    )
                    torchvision.utils.save_image(video_grid, video_path)
                    print(f"[MaGS] Saved styled video window: {video_path}")

            save_interval = int(config.dataset.get("save_per_iter", 0))
            pending_checkpoint = pending_checkpoint or (
                save_interval > 0
                and iteration % save_interval == 0
            ) or outer_iteration_end or iteration == config.optim.iterations
            checkpoint_ready = not (
                gaussian_style_update.enabled
                and gaussian_style_update.video_guidance
                and gaussian_style_update._video_render_buffer
            )
            if pending_checkpoint and checkpoint_ready:
                if gaussian_style_update.enabled and gaussian_style_update.direct_mode:
                    # Evaluation and optimizer.step may have invalidated the xyz
                    # cache. Save a snapshot matching the updated local parameters.
                    gaussians.update_mesh(
                        refined_faces,
                        refined_meshes["train_verts"][cam.uid],
                        refined_meshes["train_normals"][cam.uid],
                        deformer=None,
                    )
                save_mags_checkpoint(
                    config,
                    iteration,
                    gaussians,
                    deformer,
                    gaussian_style_update,
                    mesh_style_update,
                )
                pending_checkpoint = False

    writer.close()
    if config.get("evaluation", {}).get("enabled", False):
        # Evaluation runs only after the final optimizer step and checkpoint.
        # It never contributes a loss or changes the training parameter routing.
        from utils.paper_eval_export import export_paper_bundle
        from utils.paper_eval_protocol import write_json
        try:
            if pending_checkpoint or gaussian_style_update._video_render_buffer or gaussian_style_update._split_pending is not None:
                raise RuntimeError("Cannot evaluate an unfinished optimizer window; use an iteration count divisible by window_size")
            bundle = export_paper_bundle(config, gaussians, gaussian_style_update, train_dataset,
                                         inria_render, config.optim.iterations, paper_optimized_uids)
            if config.evaluation.get("compute_metrics", True):
                from tools.evaluate_paper import evaluate_bundle
                evaluate_bundle(bundle)
        except Exception as exc:
            import traceback
            error_dir = os.path.join(config.model.model_path, "paper_eval")
            os.makedirs(error_dir, exist_ok=True)
            write_json(os.path.join(error_dir, "failure.json"), {"status": "failed", "error": str(exc)})
            with open(os.path.join(error_dir, "failure_traceback.txt"), "w", encoding="utf-8") as stream:
                stream.write(traceback.format_exc())
            print(f"[Paper Eval] Evaluation failed; training checkpoint retained. Details: {error_dir}")
            if config.evaluation.get("fail_on_error", False):
                raise


def colorizeWeightsMap(
    weights, colormap=cv2.COLORMAP_JET, min_val=None, max_val=None, to_rgb=False
):
    if min_val is None:
        min_val = weights.min()
    if max_val is None:
        max_val = weights.max()

    vals = (weights - min_val) / (max_val - min_val)
    vals = (vals.clip(0, 1) * 255).astype(np.uint8)
    canvas = cv2.applyColorMap(vals, colormap=colormap)
    if to_rgb:
        return canvas[..., [2, 1, 0]]
    else:
        return canvas


def visualize_compare(gt_image, image, psnr, ssim, lpips):
    compare = torch.concat([gt_image, image], dim=2)
    compare = (compare.permute([1, 2, 0]) * 255)[:, :, [2, 1, 0]].detach().cpu().numpy()
    compare = cv2.putText(
        compare, "psnr/ssim/lpips", (20, compare.shape[0] - 50), 0, 1, (0, 0, 255)
    )
    compare = cv2.putText(
        compare,
        f"{psnr:.4f}/{ssim:.4f}/{lpips:.4f}",
        (20, compare.shape[0] - 10),
        0,
        1,
        (0, 0, 255),
    )

    err = (image - gt_image).abs().max(dim=0)[0].clip(0, 1)
    err_map = colorizeWeightsMap(err.detach().cpu().numpy(), min_val=0, max_val=1)
    compare = np.concatenate([compare, err_map], axis=1)
    return compare


def render_test(
    config,
    gaussians,
    deformer,
    refined_meshes,
    refined_faces,
    iteration,
    train_dataset,
    test_dataset,
    background,
    writer,
    disable_deformer=False,
    mesh_style_update=None,
    gaussian_style_update=None,
):
    global best_psnr, best_l1, best_lpips, best_ssim, best_iter
    torch.cuda.empty_cache()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    if config.dataset.white_background or config.optim.random_bg:
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    for name, case in {"test": test_dataset, "train": train_dataset}.items():
        psnrs = []
        l1s = []
        lpipss = []
        ssims = []
        output_path = f"{config.model.model_path}/{iteration}"
        os.makedirs(output_path, exist_ok=True)
        sum_time = 0
        for idx, cam in enumerate(case):
            start_time = datetime.datetime.now()
            if mesh_style_update is not None and mesh_style_update.enabled:
                use_deformer = mesh_style_update.use_deformer
            elif gaussian_style_update is not None and gaussian_style_update.enabled:
                use_deformer = gaussian_style_update.use_deformer
            else:
                use_deformer = (
                    not disable_deformer and iteration > config.optim.warm_up
                )
            deformation_components = (
                gaussian_style_update.deformation_components
                if gaussian_style_update is not None
                and gaussian_style_update.enabled
                else None
            )
            if gaussian_style_update is not None and gaussian_style_update.enabled:
                gaussian_style_update.apply_geometry_to_gaussians(detach=True)
            else:
                gaussians.style_uvw_delta = None
                gaussians.style_log_scaling_delta = None
                gaussians.style_rotation_delta = None
            if (
                mesh_style_update is not None
                and mesh_style_update.should_apply_to_gaussian(iteration)
            ):
                align_pkg = mesh_style_update.align(refined_meshes[f"{name}_verts"][cam.uid])
                gaussians.update_mesh(
                    refined_faces,
                    align_pkg["aligned_verts"].detach(),
                    align_pkg["aligned_normals"].detach(),
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][0],
                    refined_meshes[f"{name}_verts"][cam.uid],
                    deformer if use_deformer else None,
                    detach_mesh=True,
                    deformation_components=deformation_components,
                )
            else:
                gaussians.update_mesh(
                    refined_faces,
                    refined_meshes[f"{name}_verts"][cam.uid],
                    refined_meshes[f"{name}_normals"][cam.uid],
                    refined_meshes["train_verts"][0],
                    refined_meshes["train_verts"][0],
                    refined_meshes[f"{name}_verts"][cam.uid],
                    deformer if use_deformer else None,
                    deformation_components=deformation_components,
                )
            if gaussian_style_update is not None and gaussian_style_update.enabled:
                gaussian_style_update.apply_to_gaussians()
            else:
                gaussians.style_features_delta = None
            render_pkg = inria_render(
                cam,
                gaussians,
                config.pipe,
                background,
            )
            end_time = datetime.datetime.now()
            sum_time += (end_time - start_time).total_seconds()
            require_finite(
                render_pkg["render"],
                f"{name} evaluation render at iteration {iteration}, frame={cam.uid}",
            )
            image = torch.clamp(
                render_pkg["render"],
                0.0,
                1.0,
            )
            gt_image = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
            if config.optim.random_bg or config.dataset.white_background:
                H, W = gt_image.shape[1:]
                mask = cam.gt_mask.repeat(3, 1, 1)
                bg = background.reshape(3, 1, 1).repeat(1, H, W)
                gt_image = gt_image * mask + bg * (1 - mask)

            image = image.contiguous()
            psnr_val = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
            l1_val = l1_loss(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
            lpips_val = lpips(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
            ssim_val = ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()

            psnrs.append(psnr_val)
            l1s.append(l1_val)
            lpipss.append(lpips_val)
            ssims.append(ssim_val)

            if name == "test":
                torchvision.utils.save_image(
                    image, f"{output_path}/test_{idx}_render.png"
                )
                torchvision.utils.save_image(
                    gt_image, f"{output_path}/test_{idx}_gt.png"
                )
                compare = visualize_compare(
                    gt_image, image, psnr_val, ssim_val, lpips_val
                )
                cv2.imwrite(f"{output_path}/test_{idx}_compare.png", compare)

        l1_test = sum(l1s) / len(l1s)
        psnr_test = sum(psnrs) / len(psnrs)
        lpips_test = sum(lpipss) / len(lpipss)
        ssim_test = sum(ssims) / len(ssims)
        fps = len(case) / sum_time

        print(
            "\n[ITER {}] Evaluating {}: L1 {:.7f} PSNR {:.7f} LPIPS {:.7f} SSIM {:.7f}".format(
                iteration, name, l1_test, psnr_test, lpips_test, ssim_test
            )
        )
        print(f"FPS: {fps:.2f}")
        if psnr_test > best_psnr and name == "test":
            best_psnr = psnr_test
            best_l1 = l1_test
            best_lpips = lpips_test
            best_ssim = ssim_test
            best_iter = iteration

        # Log test metrics to TensorBoard
        writer.add_scalar(f"Loss/{name}", l1_test, iteration)
        writer.add_scalar(f"PSNR/{name}", psnr_test, iteration)
        writer.add_scalar(f"LPIPS/{name}", lpips_test, iteration)
        writer.add_scalar(f"SSIM/{name}", ssim_test, iteration)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    config = load_training_config(sys.argv[1:])
    print("Optimizing " + config.model.model_path)
    style_cfg = config.optim.get("gaussian_style_update", {})
    if style_cfg.get("enabled", False) and str(style_cfg.get("parameterization", "direct")).lower() == "direct":
        config.model.model_path = create_direct_style_output(
            config.model.model_path, str(style_cfg.get("start_checkpoint", ""))
        )
        style_cfg.use_deformer = False
        style_cfg.gt_mesh_trajectory.deformer_mode = "off"
        config.optim.disable_deformer = True
    else:
        experiment_id = datetime.datetime.now().strftime("%d%H%M%S")
        config.model.model_path = os.path.join(config.model.model_path, experiment_id)
        os.makedirs(config.model.model_path, exist_ok=True)
    with open(os.path.join(config.model.model_path, "cfg_args.yaml"), "w") as cfg_log_f:
        OmegaConf.save(config, cfg_log_f)

    # Initialize system state (RNG)
    safe_state(config)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(bool(config.get("detect_anomaly", False)))
    training(config)
    print(
        f"Best PSNR: {best_psnr:.4f} L1: {best_l1:.4f} LPIPS: {best_lpips:.4f} SSIM: {best_ssim:.4f} at iteration {best_iter}"
    )
    # All done
    print("\nTraining complete.")
