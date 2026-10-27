"""Selective derivatives for the two independent style parameter groups."""

import hashlib
import torch
from utils.style_numerics import require_finite


def reference_signature(cameras, faces, meshes):
    digest = hashlib.sha256()
    for tensor in [faces] + [value for cam in cameras for value in
                            (torch.tensor([cam.uid]), cam.full_proj_transform,
                             cam.gt_mask, meshes[cam.uid])]:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def routed_style_gradients(video, source_keys, appearance, teacher, sh_params,
                           geometry_params, appearance_weight, geometry_weight,
                           guidance_kwargs=None, offload_teacher=False,
                           geometry_supervision="diffusion"):
    """Return gradients without accumulating into any unrelated .grad fields.

    Finish VGG differentiation and release its feature activations before the
    large video teacher forward. Both groups still use the same render snapshot.
    """
    sh_params = [p for p in sh_params if p.requires_grad]
    geometry_params = [p for p in geometry_params if p.requires_grad]
    if geometry_supervision not in {"diffusion", "vgg"}:
        raise ValueError("geometry_supervision must be diffusion or vgg")
    if not sh_params and not geometry_params:
        raise ValueError("No active style parameters")
    park = getattr(teacher, "park_diffusion_backbone", None)
    if offload_teacher and park is not None:
        park()
    rgb_metrics = {}
    branch = "appearance"

    def record_rgb_gradient(gradient):
        require_finite(gradient, f"split {branch} RGB gradient")
        rgb_metrics[f"gaussian_style_{branch}_rgb_grad_norm"] = gradient.detach().float().norm()

    hook = video.register_hook(record_rgb_gradient)
    diagnostics = {}
    geo_value = video.new_zeros(())
    raw_value = geo_value
    geo_grads = ()
    sh_grads = ()
    app_value = geo_value
    base_app_loss = None
    try:
        if sh_params or (geometry_params and geometry_supervision == "vgg"):
            base_app_loss = appearance(video.flatten(0, 1))
        if sh_params:
            app_loss = base_app_loss * appearance_weight
            app_value = require_finite(app_loss, "split appearance loss").detach()
            sh_grads = torch.autograd.grad(app_loss, sh_params, retain_graph=bool(geometry_params))
            del app_loss
        if geometry_supervision == "diffusion":
            del base_app_loss
        if geometry_params:
            branch = "geometry"
            if geometry_supervision == "vgg":
                raw = base_app_loss
            else:
                raw, diagnostics = teacher(video, source_keys=source_keys,
                                           return_diagnostics=True, **(guidance_kwargs or {}))
            geo_loss = geometry_weight * raw
            raw_value = raw.detach()
            geo_value = require_finite(geo_loss, "split geometry loss").detach()
            geo_grads = torch.autograd.grad(geo_loss, geometry_params)
    finally:
        hook.remove()
        if offload_teacher and park is not None:
            park()
    # Do not commit a partial update if either branch failed numerically.
    for name, gradients in (("SH", sh_grads), ("geometry", geo_grads)):
        for gradient in gradients:
            require_finite(gradient, f"split {name} gradient")
    metrics = {f"gaussian_style_{key}": value.detach().float() for key, value in diagnostics.items()}
    metrics.update({
        "gaussian_style_loss": app_value + geo_value,
        "gaussian_style_appearance_loss": app_value,
        "gaussian_style_geometry_loss": geo_value,
        "gaussian_style_csd_loss": geo_value if geometry_supervision == "diffusion" else video.new_zeros(()),
        "gaussian_style_csd_raw_loss": raw_value if geometry_supervision == "diffusion" else video.new_zeros(()),
        "gaussian_style_csd_grad_norm": diagnostics.get("csd_grad_norm", video.new_zeros(())).detach(),
        "gaussian_style_rgb_saturation": ((video.detach() < 0) | (video.detach() > 1)).float().mean(),
    })
    metrics.update(rgb_metrics)
    return list(zip(sh_params, sh_grads)) + list(zip(geometry_params, geo_grads)), metrics
