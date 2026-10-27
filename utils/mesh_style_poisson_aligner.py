import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_unbatched_vertices(vertices):
    if vertices.dim() == 2:
        return vertices
    if vertices.dim() == 3 and vertices.shape[0] == 1:
        return vertices.squeeze(0)
    raise ValueError(f"Expected vertices with shape [V, 3] or [1, V, 3], got {vertices.shape}.")


def _face_normals(vertices, faces):
    faces = faces.to(device=vertices.device, dtype=torch.long)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normal_len = normals.norm(dim=-1, keepdim=True)
    normals = F.normalize(normals, dim=-1, eps=1e-12)
    fallback = torch.tensor([0.0, 0.0, 1.0], device=vertices.device, dtype=vertices.dtype)
    valid = torch.isfinite(normals).all(dim=-1, keepdim=True) & (normal_len > 1e-12)
    return torch.where(valid, normals, fallback.view(1, 3).expand_as(normals))


def _face_frames(vertices, faces):
    """Build invertible per-face frames with two edges and an area-scaled normal."""
    faces = faces.to(device=vertices.device, dtype=torch.long)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge_1 = v1 - v0
    edge_2 = v2 - v0
    cross = torch.cross(edge_1, edge_2, dim=-1)
    area_scale = cross.norm(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    normal = F.normalize(cross, dim=-1, eps=1e-12) * area_scale
    return torch.stack((edge_1, edge_2, normal), dim=-1)


class MeshStylePoissonAligner(nn.Module):
    """Compose known GT motion with one shared canonical style Jacobian."""

    def __init__(
        self,
        base_vertices,
        faces,
        device="cuda",
        poisson_dtype=torch.float64,
        laplacian_jitter=1e-8,
    ):
        super().__init__()
        device = torch.device(device)
        base_vertices = _as_unbatched_vertices(base_vertices.detach()).to(
            device=device,
            dtype=poisson_dtype,
        )
        faces = faces.detach().long().to(device)

        identity = torch.eye(3, device=device, dtype=poisson_dtype).view(1, 1, 3, 3)
        identity = identity.repeat(1, faces.shape[0], 1, 1).contiguous()
        self.style_log_jac = nn.Parameter(torch.zeros_like(identity))
        self.register_buffer("identity_jac", identity.clone())
        self.register_buffer("base_vertices", base_vertices.detach().clone())
        self.register_buffer("faces", faces)
        base_frames = _face_frames(base_vertices, faces)
        self.register_buffer(
            "base_frame_inverse",
            torch.linalg.pinv(base_frames).detach(),
        )

        edge_src, edge_dst, edge_face, base_edges = self._build_edges(
            base_vertices,
            faces,
        )
        lap_cholesky = self._build_laplacian_cholesky(
            base_vertices.shape[0],
            edge_src,
            edge_dst,
            device,
            poisson_dtype,
            laplacian_jitter,
        )
        self.register_buffer("edge_src", edge_src)
        self.register_buffer("edge_dst", edge_dst)
        self.register_buffer("edge_face", edge_face)
        self.register_buffer("base_edges", base_edges)
        self.register_buffer("lap_cholesky", lap_cholesky)

        with torch.no_grad():
            identity_recon = self.vertices_from_jacobians(identity).detach()
        self.register_buffer("identity_recon", identity_recon)

    @staticmethod
    def _build_edges(base_vertices, faces):
        edge_src = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]], dim=0)
        edge_dst = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]], dim=0)
        edge_face = torch.arange(
            faces.shape[0],
            device=faces.device,
            dtype=torch.long,
        ).repeat(3)
        base_edges = base_vertices[edge_dst] - base_vertices[edge_src]
        return edge_src, edge_dst, edge_face, base_edges

    @staticmethod
    def _build_laplacian_cholesky(
        num_vertices,
        edge_src,
        edge_dst,
        device,
        dtype,
        jitter,
    ):
        lap = torch.zeros((num_vertices, num_vertices), device=device, dtype=dtype)
        ones = torch.ones(edge_src.shape[0], device=device, dtype=dtype)
        lap.index_put_((edge_src, edge_src), ones, accumulate=True)
        lap.index_put_((edge_dst, edge_dst), ones, accumulate=True)
        lap.index_put_((edge_src, edge_dst), -ones, accumulate=True)
        lap.index_put_((edge_dst, edge_src), -ones, accumulate=True)

        reduced = lap[1:, 1:]
        eye = torch.eye(reduced.shape[0], device=device, dtype=dtype)
        return torch.linalg.cholesky(reduced + float(jitter) * eye)

    def vertices_from_jacobians(self, jacobians):
        jac = jacobians
        batch_size = jac.shape[0]
        edge_jac = jac[:, self.edge_face]
        edge_vec = self.base_edges.view(1, -1, 3, 1).to(
            device=jac.device,
            dtype=jac.dtype,
        )
        pred_edges = torch.matmul(edge_jac, edge_vec).squeeze(-1)

        rhs = torch.zeros(
            (batch_size, self.base_vertices.shape[0], 3),
            device=jac.device,
            dtype=jac.dtype,
        )
        src_idx = self.edge_src.view(1, -1, 1).expand(batch_size, -1, 3)
        dst_idx = self.edge_dst.view(1, -1, 1).expand(batch_size, -1, 3)
        rhs.scatter_add_(1, src_idx, -pred_edges)
        rhs.scatter_add_(1, dst_idx, pred_edges)

        rhs_reduced = rhs[:, 1:, :]
        num_unknowns = rhs_reduced.shape[1]
        flat_rhs = rhs_reduced.permute(1, 0, 2).reshape(num_unknowns, -1)
        flat_sol = torch.cholesky_solve(flat_rhs, self.lap_cholesky)
        sol_reduced = flat_sol.view(num_unknowns, batch_size, 3).permute(1, 0, 2)

        anchor = torch.zeros(
            (batch_size, 1, 3),
            device=jac.device,
            dtype=jac.dtype,
        )
        return torch.cat([anchor, sol_reduced], dim=1)

    def motion_jacobians(self, vertices):
        vertices = _as_unbatched_vertices(vertices).to(
            device=self.base_vertices.device,
            dtype=self.base_vertices.dtype,
        )
        current_frames = _face_frames(vertices, self.faces)
        return torch.matmul(current_frames, self.base_frame_inverse).unsqueeze(0)

    def style_jacobians(self):
        return torch.matrix_exp(self.style_log_jac)

    def forward(self, verts):
        verts = _as_unbatched_vertices(verts)
        motion_jac = self.motion_jacobians(verts)
        style_jac = self.style_jacobians()
        target_jac = torch.matmul(motion_jac, style_jac)

        motion_recon = self.vertices_from_jacobians(motion_jac)
        styled_recon = self.vertices_from_jacobians(target_jac)
        delta = (styled_recon - motion_recon).squeeze(0).to(
            device=verts.device,
            dtype=verts.dtype,
        )

        aligned_verts = verts + delta
        aligned_normals = _face_normals(aligned_verts, self.faces)
        canonical_recon = self.vertices_from_jacobians(style_jac)
        canonical_delta = (canonical_recon - self.identity_recon).squeeze(0)
        aligned_cano_verts = self.base_vertices.to(
            device=verts.device,
            dtype=verts.dtype,
        ) + canonical_delta.to(device=verts.device, dtype=verts.dtype)
        reconstructed_jac = self.motion_jacobians(aligned_verts)

        return {
            "aligned_verts": aligned_verts,
            "aligned_normals": aligned_normals,
            "aligned_cano_verts": aligned_cano_verts,
            "delta": delta,
            "style_jac": style_jac,
            "motion_jac": motion_jac,
            "target_jac": target_jac,
            "reconstructed_jac": reconstructed_jac,
        }
