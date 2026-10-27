import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def axis_angle_to_quaternion_wxyz(axis_angle):
    """Convert bounded axis-angle vectors to unit quaternions in wxyz order."""
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    sin_half_over_angle = torch.where(
        angle > 1e-8,
        torch.sin(half_angle) / angle.clamp_min(1e-8),
        0.5 - angle.square() / 48.0,
    )
    xyz = axis_angle * sin_half_over_angle
    return F.normalize(torch.cat((torch.cos(half_angle), xyz), dim=-1), dim=-1)


class CanonicalGaussianGeometryResidual(nn.Module):
    """Time-invariant Gaussian geometry residuals in mesh-local coordinates."""

    def __init__(self, base_uvw, face_ids, mesh_vertices, mesh_faces, cfg):
        super().__init__()
        if base_uvw.dim() != 2 or base_uvw.shape[-1] != 3:
            raise ValueError("base_uvw must have shape [N, 3].")

        gaussian_count = base_uvw.shape[0]
        self.raw_barycentric = nn.Parameter(
            torch.zeros(gaussian_count, 3, device=base_uvw.device)
        )
        self.raw_normal = nn.Parameter(
            torch.zeros(gaussian_count, 1, device=base_uvw.device)
        )
        self.raw_log_scaling = nn.Parameter(
            torch.zeros(gaussian_count, 3, device=base_uvw.device)
        )
        self.raw_axis_angle = nn.Parameter(
            torch.zeros(gaussian_count, 3, device=base_uvw.device)
        )

        base_barycentric = torch.stack(
            (
                base_uvw[:, 0],
                base_uvw[:, 1],
                1.0 - base_uvw[:, 0] - base_uvw[:, 1],
            ),
            dim=-1,
        )
        base_barycentric = base_barycentric.clamp_min(1e-6)
        base_barycentric = base_barycentric / base_barycentric.sum(
            dim=-1, keepdim=True
        )
        self.register_buffer(
            "base_barycentric", base_barycentric.detach(), persistent=False
        )
        self.register_buffer("face_ids", face_ids.long().detach(), persistent=False)
        self.num_faces = int(mesh_faces.shape[0])

        faces = mesh_faces.long()
        triangles = mesh_vertices[faces]
        edge_lengths = torch.stack(
            (
                torch.linalg.vector_norm(triangles[:, 1] - triangles[:, 0], dim=-1),
                torch.linalg.vector_norm(triangles[:, 2] - triangles[:, 1], dim=-1),
                torch.linalg.vector_norm(triangles[:, 0] - triangles[:, 2], dim=-1),
            ),
            dim=-1,
        )
        face_edge_length = edge_lengths.mean(dim=-1).detach().clamp_min(1e-8)
        mean_edge_length = face_edge_length.mean()
        self.register_buffer(
            "mean_edge_length", mean_edge_length, persistent=False
        )
        self.register_buffer(
            "normal_offset_scale",
            face_edge_length[self.face_ids].unsqueeze(-1),
            persistent=False,
        )

        self.max_barycentric_logit = float(
            cfg.get("max_barycentric_logit", 0.25)
        )
        self.max_normal_edge_ratio = float(
            cfg.get("max_normal_edge_ratio", 0.10)
        )
        self.max_log_scaling = float(cfg.get("max_log_scaling", 0.10))
        self.max_rotation_radians = math.radians(
            float(cfg.get("max_rotation_degrees", 5.0))
        )

    def _bounded_axis_angle(self):
        # Component-wise bounding keeps a non-zero derivative at zero while
        # guaranteeing the full axis-angle norm cannot exceed the configured cap.
        component_limit = self.max_rotation_radians / math.sqrt(3.0)
        return component_limit * torch.tanh(self.raw_axis_angle)

    def forward(self):
        barycentric_logits = (
            torch.log(self.base_barycentric)
            + self.max_barycentric_logit * torch.tanh(self.raw_barycentric)
        )
        styled_barycentric = torch.softmax(barycentric_logits, dim=-1)
        uv_delta = styled_barycentric[:, :2] - self.base_barycentric[:, :2]
        normal_delta = (
            self.max_normal_edge_ratio
            * self.normal_offset_scale
            * torch.tanh(self.raw_normal)
        )
        uvw_delta = torch.cat((uv_delta, normal_delta), dim=-1)

        log_scaling_delta = self.max_log_scaling * torch.tanh(
            self.raw_log_scaling
        )
        axis_angle_delta = self._bounded_axis_angle()
        rotation_delta = axis_angle_to_quaternion_wxyz(axis_angle_delta)
        return {
            "uvw_delta": uvw_delta,
            "log_scaling_delta": log_scaling_delta,
            "axis_angle_delta": axis_angle_delta,
            "rotation_delta": rotation_delta,
        }

    def _same_face_smoothness(self, value):
        flat_value = value.reshape(value.shape[0], -1)
        face_sum = torch.zeros(
            self.num_faces,
            flat_value.shape[-1],
            device=value.device,
            dtype=value.dtype,
        )
        face_sum.index_add_(0, self.face_ids, flat_value)
        counts = torch.zeros(
            self.num_faces, 1, device=value.device, dtype=value.dtype
        )
        counts.index_add_(
            0,
            self.face_ids,
            torch.ones(value.shape[0], 1, device=value.device, dtype=value.dtype),
        )
        face_mean = face_sum / counts.clamp_min(1.0)
        return (flat_value - face_mean[self.face_ids]).square().mean()

    def regularization(self):
        package = self()
        smooth_values = torch.cat(
            (
                package["uvw_delta"],
                package["log_scaling_delta"],
                package["axis_angle_delta"],
            ),
            dim=-1,
        )
        return {
            "uv": package["uvw_delta"][:, :2].square().mean(),
            "normal": package["uvw_delta"][:, 2:].square().mean(),
            "scaling": package["log_scaling_delta"].square().mean(),
            "rotation": package["axis_angle_delta"].square().mean(),
            "smooth": self._same_face_smoothness(smooth_values),
        }

    def rms_diagnostics(self):
        package = self()
        return {
            "uv_rms": package["uvw_delta"][:, :2].float().square().mean().sqrt(),
            "normal_rms": package["uvw_delta"][:, 2:].float().square().mean().sqrt(),
            "scaling_rms": package["log_scaling_delta"].float().square().mean().sqrt(),
            "rotation_rms": package["axis_angle_delta"].float().square().mean().sqrt(),
        }

    def parameter_items(self):
        return {
            "uv": self.raw_barycentric,
            "normal": self.raw_normal,
            "scaling": self.raw_log_scaling,
            "rotation": self.raw_axis_angle,
        }
