import torch


def require_finite(tensor, name):
    values = tensor.detach()
    finite = torch.isfinite(values)
    if not finite.all().item():
        invalid_count = int((~finite).sum().item())
        raise FloatingPointError(
            f"Non-finite {name}: {invalid_count}/{values.numel()} values, "
            f"shape={tuple(values.shape)}, dtype={values.dtype}. "
            "Training stopped; do not resume from a NaN checkpoint."
        )
    return tensor


def check_backward_finite(tensor, name):
    if tensor.requires_grad:
        tensor.register_hook(lambda gradient: require_finite(gradient, name))
    return tensor


def video_csd_surrogate(
    target_latents,
    source_uncond,
    source_cond,
    target_uncond,
    target_cond,
    weight,
    guidance_scale,
    grad_clip,
):
    # Keep the score difference and reduction in float32 even when the frozen
    # denoisers use float16. Only the rendered latents retain an autograd path.
    branches = {}
    for name, prediction in (
        ("source_uncond", source_uncond),
        ("source_cond", source_cond),
        ("target_uncond", target_uncond),
        ("target_cond", target_cond),
    ):
        branches[name] = require_finite(
            prediction.detach().float(), f"video CSD {name} prediction"
        )
    latents = require_finite(target_latents.float(), "video CSD target latents")
    source_direction = guidance_scale * (
        branches["source_cond"] - branches["source_uncond"]
    )
    target_direction = guidance_scale * (
        branches["target_cond"] - branches["target_uncond"]
    )
    gradient = weight.detach().float() * (target_direction - source_direction)
    require_finite(gradient, "video CSD score gradient before clipping")
    if grad_clip > 0:
        gradient = gradient.clamp(-grad_clip, grad_clip)
    divisor = max(latents.shape[0] * latents.shape[2], 1)
    loss = (gradient.detach() * latents).sum(dtype=torch.float32) / divisor
    require_finite(loss, "video CSD surrogate loss")
    return loss, gradient, source_direction, target_direction
