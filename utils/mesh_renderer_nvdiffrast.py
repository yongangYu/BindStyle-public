import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_nvdiffrast():
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise ImportError(
            "nvdiffrast is required for mesh_style_update mesh rendering. "
            "Install nvdiffrast in the training environment before enabling SDS."
        ) from exc
    return dr


def _as_hw(resolution, cam):
    if isinstance(resolution, (tuple, list)):
        return int(resolution[0]), int(resolution[1])
    resolution = int(resolution)
    if resolution > 0:
        return resolution, resolution
    return int(cam.image_height), int(cam.image_width)


def _vertex_normals(vertices, faces):
    faces = faces.to(device=vertices.device, dtype=torch.long)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    normals = torch.zeros_like(vertices)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    normals = F.normalize(normals, dim=-1, eps=1e-8)
    fallback = torch.tensor([0.0, 0.0, 1.0], device=vertices.device, dtype=vertices.dtype)
    valid = torch.isfinite(normals).all(dim=-1, keepdim=True)
    return torch.where(valid, normals, fallback.view(1, 3).expand_as(normals))


def sequence_inverse_depth_range(cameras):
    """A single near-white/far-black calibration for every frame in a sequence."""
    ranges = []
    for cam in cameras:
        verts = cam.mesh_vertices.detach().float()
        homogeneous = torch.cat([verts, torch.ones_like(verts[:, :1])], -1)
        z = (homogeneous @ cam.full_proj_transform.to(verts))[:, 3]
        valid = (z > float(cam.znear)) & (z < float(cam.zfar)) & torch.isfinite(z)
        if valid.any():
            inverse = z[valid].reciprocal()
            ranges.append((inverse.min().item(), inverse.max().item()))
    if not ranges:
        raise ValueError("No valid GT-mesh camera depths for ControlNet normalization.")
    lower, upper = min(x[0] for x in ranges), max(x[1] for x in ranges)
    if upper - lower < 1e-6:
        lower, upper = lower - 1e-6, upper + 1e-6
    return lower, upper


class NVDiffMeshRenderer(nn.Module):
    def __init__(
        self,
        resolution=256,
        mode="normal",
        background=(0.0, 0.0, 0.0),
        antialias=True,
        depth_mode="ndc",
        depth_range=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.mode = str(mode).lower()
        self.antialias = bool(antialias)
        self.depth_mode = depth_mode
        self.depth_range = depth_range
        if depth_mode not in {"ndc", "relative_inverse"}:
            raise ValueError("depth_mode must be ndc or relative_inverse.")
        if depth_mode == "relative_inverse" and (depth_range is None or depth_range[1] <= depth_range[0]):
            raise ValueError("Relative inverse depth requires a shared increasing depth_range.")
        self.register_buffer(
            "background",
            torch.tensor(background, dtype=torch.float32).view(1, 1, 1, 3),
        )
        self._dr = None
        self._ctx = None

    def _context(self, device):
        if self._dr is None:
            self._dr = _load_nvdiffrast()
        if self._ctx is None:
            if device.type != "cuda":
                raise RuntimeError("nvdiffrast rasterization requires CUDA tensors.")
            try:
                self._ctx = self._dr.RasterizeCudaContext(device=device)
            except TypeError:
                self._ctx = self._dr.RasterizeCudaContext()
        return self._dr, self._ctx

    def _clip_vertices(self, vertices, cam):
        transform = cam.full_proj_transform.to(
            device=vertices.device,
            dtype=vertices.dtype,
        )
        ones = torch.ones(
            (vertices.shape[0], 1),
            device=vertices.device,
            dtype=vertices.dtype,
        )
        vertices_h = torch.cat([vertices, ones], dim=-1)
        return vertices_h @ transform

    def _attributes(self, vertices, faces, clip):
        if self.mode == "normal":
            return (_vertex_normals(vertices, faces) + 1.0) * 0.5
        if self.mode == "position":
            vmin = vertices.detach().amin(dim=0, keepdim=True)
            vmax = vertices.detach().amax(dim=0, keepdim=True)
            return (vertices - vmin) / (vmax - vmin).clamp_min(1e-6)
        if self.mode == "depth":
            if self.depth_mode == "relative_inverse":
                # Interpolate camera z first, then invert; avoid interpolating
                # an already nonlinear depth attribute with perspective weights.
                return clip[:, 3:4].repeat(1, 3)
            ndc_z = clip[:, 2:3] / clip[:, 3:4].clamp_min(1e-6)
            depth = (ndc_z + 1.0) * 0.5
            return depth.repeat(1, 3)
        raise ValueError(f"Unsupported mesh render mode: {self.mode}")

    def forward(self, vertices, faces, cam):
        dr, ctx = self._context(vertices.device)
        height, width = _as_hw(self.resolution, cam)

        faces_i32 = faces.to(device=vertices.device, dtype=torch.int32).contiguous()
        clip = self._clip_vertices(vertices, cam).contiguous()
        attrs = self._attributes(vertices, faces, clip).contiguous()

        clip_b = clip.unsqueeze(0)
        attrs_b = attrs.unsqueeze(0)
        rast, _ = dr.rasterize(ctx, clip_b, faces_i32, resolution=[height, width])
        rendered, _ = dr.interpolate(attrs_b, rast, faces_i32)
        mask = (rast[..., 3:4] > 0).to(rendered.dtype)
        if self.mode == "depth" and self.depth_mode == "relative_inverse":
            lower, upper = self.depth_range
            rendered = ((rendered.clamp_min(1e-6).reciprocal() - lower) / (upper - lower)).clamp(0, 1)

        background = self.background.to(device=vertices.device, dtype=rendered.dtype)
        rendered = rendered * mask + background * (1.0 - mask)
        if self.antialias:
            rendered = dr.antialias(rendered, rast, clip_b, faces_i32)

        image = rendered.permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)
        return {
            "image": image,
            "mask": mask.permute(0, 3, 1, 2).contiguous(),
            "rast": rast,
        }
