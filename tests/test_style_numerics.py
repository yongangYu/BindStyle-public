import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from utils.style_numerics import (
    check_backward_finite,
    require_finite,
    video_csd_surrogate,
)


ROOT = Path(__file__).resolve().parents[1]


def load_methods(relative_path, class_name, method_names):
    # The real modules load CUDA rasterizers or models at import time; load the
    # production methods while using small CPU tensors and stand-in networks.
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node.body = [
        n for n in node.body
        if isinstance(n, ast.FunctionDef) and n.name in method_names
    ]
    namespace = {
        "torch": torch, "nn": torch.nn, "F": F,
        "require_finite": require_finite, "check_backward_finite": check_backward_finite,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[class_name]


Guidance = load_methods(
    "utils/sdxl_animatediff_controlnet_csd_guidance.py",
    "SDXLAnimateDiffControlNetCSDGuidance",
    {"_configure_vae_precision", "_encode_frame_batch"},
)
Controller = load_methods(
    "train.py", "GaussianSHStyleController",
    {"step", "_check_finite_style_state", "verify_gradient_boundary", "style_optimizers"},
)


class VideoCSDNumericsTests(unittest.TestCase):
    def test_float32_reduction_preserves_finite_loss_and_exact_gradient(self):
        latents = torch.full((1, 4, 2, 2, 2), 60000.0, dtype=torch.float16, requires_grad=True)
        zeros = torch.zeros_like(latents)
        ones = torch.ones_like(latents)
        self.assertFalse(torch.isfinite((ones * latents).sum()).item())
        loss, gradient, _, _ = video_csd_surrogate(
            latents, zeros, zeros, zeros, ones,
            torch.ones(1, 1, 1, 1, 1), 1.0, 1.0,
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 960000.0)
        loss.backward()
        torch.testing.assert_close(latents.grad.float(), gradient / 2)

    def test_half_precision_score_subtraction_does_not_overflow(self):
        latents = torch.ones((1, 4, 2, 1, 1), requires_grad=True)
        high = torch.full_like(latents, 60000.0, dtype=torch.float16)
        low = -high
        self.assertFalse(torch.isfinite(high - low).all().item())
        loss, gradient, _, target_direction = video_csd_surrogate(
            latents, low, low, low, high,
            torch.ones(1, 1, 1, 1, 1), 7.0, 1.0,
        )
        self.assertTrue(torch.isfinite(target_direction).all().item())
        self.assertTrue(torch.isfinite(loss).item())
        torch.testing.assert_close(gradient, torch.ones_like(latents))

    def test_score_is_detached_but_rendered_latents_receive_gradient(self):
        latents = torch.ones((1, 4, 2, 2, 2), requires_grad=True)
        predictions = [torch.full_like(latents, value, requires_grad=True) for value in (1, 3, 2, 5)]
        loss, gradient, _, _ = video_csd_surrogate(
            latents, *predictions, torch.ones(1, 1, 1, 1, 1) * 0.5, 2.0, 0.0
        )
        loss.backward()
        torch.testing.assert_close(gradient, torch.ones_like(latents))
        torch.testing.assert_close(latents.grad, gradient / 2)
        self.assertTrue(all(value.grad is None for value in predictions))

    def test_invalid_prediction_is_reported_before_clipping(self):
        latents = torch.ones((1, 4, 2, 1, 1), requires_grad=True)
        zeros = torch.zeros_like(latents)
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(FloatingPointError, "target_cond prediction"):
                video_csd_surrogate(
                    latents, zeros, zeros, zeros, torch.full_like(latents, value),
                    torch.ones(1, 1, 1, 1, 1), 7.0, 1.0,
                )

    def test_invalid_latents_are_not_hidden_by_zero_score(self):
        latents = torch.full((1, 4, 2, 1, 1), float("nan"))
        zeros = torch.zeros_like(latents)
        with self.assertRaisesRegex(FloatingPointError, "target latents"):
            video_csd_surrogate(
                latents, zeros, zeros, zeros, zeros,
                torch.ones(1, 1, 1, 1, 1), 7.0, 1.0,
            )


class BadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class SmallVAE(torch.nn.Module):
    def __init__(self, force_upcast=True, bad_backward=False):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, 4, 1, bias=False)
        self.config = SimpleNamespace(force_upcast=force_upcast, scaling_factor=0.13025)
        self.bad_backward = bad_backward
        self.seen_dtype = None

    def encode(self, image):
        self.seen_dtype = image.dtype
        value = BadBackward.apply(image) if self.bad_backward else image
        encoded = self.proj(value)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: encoded))


def make_guidance(force_upcast=True, half_precision=True, bad_backward=False):
    guidance = Guidance()
    guidance.device = torch.device("cpu")
    guidance.image_size = 8
    guidance.vae_latent_mode = "mode"
    vae = SmallVAE(force_upcast, bad_backward).half().requires_grad_(False)
    guidance.pipe = SimpleNamespace(vae=vae)
    with contextlib.redirect_stdout(io.StringIO()):
        guidance._configure_vae_precision(torch.float16, half_precision)
    return guidance


class VAEPrecisionTests(unittest.TestCase):
    def test_force_upcast_overrides_half_precision_option(self):
        guidance = make_guidance()
        self.assertEqual(next(guidance.pipe.vae.parameters()).dtype, torch.float32)
        images = torch.rand(2, 3, 8, 8, requires_grad=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            latents = guidance._encode_frame_batch(images)
        self.assertEqual(guidance.pipe.vae.seen_dtype, torch.float32)
        self.assertEqual(latents.dtype, torch.float32)
        latents.square().mean().backward()
        self.assertTrue(torch.isfinite(images.grad).all().item())
        self.assertGreater(images.grad.norm().item(), 0.0)
        self.assertTrue(all(p.grad is None for p in guidance.pipe.vae.parameters()))

    def test_explicit_float32_is_honored_for_other_vae(self):
        guidance = make_guidance(force_upcast=False, half_precision=False)
        self.assertEqual(next(guidance.pipe.vae.parameters()).dtype, torch.float32)

    def test_fp16_compatible_vae_may_stay_half(self):
        guidance = make_guidance(force_upcast=False)
        self.assertEqual(next(guidance.pipe.vae.parameters()).dtype, torch.float16)

    def test_invalid_image_fails_before_clamp(self):
        guidance = make_guidance()
        with self.assertRaisesRegex(FloatingPointError, "VAE input image"):
            guidance._encode_frame_batch(torch.full((1, 3, 8, 8), float("inf")))

    def test_backward_nan_is_localized_at_vae_input(self):
        guidance = make_guidance(bad_backward=True)
        image = torch.rand(1, 3, 8, 8, requires_grad=True)
        loss = guidance._encode_frame_batch(image).sum()
        with self.assertRaisesRegex(FloatingPointError, "VAE backward image gradient"):
            loss.backward()


def make_controller():
    controller = Controller()
    controller.style_delta = torch.nn.Module()
    controller.style_delta.register_parameter("dc_delta", torch.nn.Parameter(torch.zeros(2, 1, 3)))
    controller.style_delta.register_parameter("rest_delta", torch.nn.Parameter(torch.zeros(2, 3, 3)))
    controller.direct_mode = False
    controller.split_supervision = False
    controller.sh_dc = controller.style_delta.dc_delta
    controller.sh_rest = controller.style_delta.rest_delta
    controller.optimizer = torch.optim.Adam([
        {"params": [controller.style_delta.dc_delta], "name": "style_dc_delta"},
        {"params": [controller.style_delta.rest_delta], "name": "style_rest_delta"},
    ], lr=0.002, eps=1e-15)
    controller.geometry_enabled = False
    controller.grad_clip = 1.0
    controller._current_iteration = 498
    controller._verified_dc_grad_boundary = True
    controller._verified_rest_grad_boundary = True
    return controller


class StyleOptimizerNumericsTests(unittest.TestCase):
    def test_invalid_gradient_never_reaches_adam(self):
        controller = make_controller()
        parameter = controller.style_delta.dc_delta
        original = parameter.detach().clone()
        parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "gradient of style_dc_delta"):
            controller.step()
        torch.testing.assert_close(parameter, original)
        self.assertFalse(controller.optimizer.state)

    def test_boundary_check_still_checks_finiteness_after_first_iteration(self):
        controller = make_controller()
        parameter = controller.style_delta.rest_delta
        parameter.grad = torch.full_like(parameter, float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "gradient of style_rest_delta"):
            controller.verify_gradient_boundary()

    def test_invalid_adam_state_is_rejected_before_update(self):
        controller = make_controller()
        parameter = controller.style_delta.dc_delta
        controller.optimizer.state[parameter]["exp_avg_sq"] = torch.full_like(parameter, float("inf"))
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(FloatingPointError, "Adam exp_avg_sq"):
            controller.step()
        torch.testing.assert_close(parameter, torch.zeros_like(parameter))

    def test_overflowed_total_gradient_norm_is_not_silently_clipped(self):
        controller = make_controller()
        parameter = controller.style_delta.dc_delta
        parameter.grad = torch.full_like(parameter, 1e30)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            controller.step()
        self.assertFalse(controller.optimizer.state)

    def test_finite_gradients_update_sh_and_clear_gradients(self):
        controller = make_controller()
        for parameter in controller.style_delta.parameters():
            parameter.grad = torch.ones_like(parameter)
        controller.step()
        for parameter in controller.style_delta.parameters():
            self.assertTrue(torch.isfinite(parameter).all().item())
            self.assertGreater(parameter.detach().abs().max().item(), 0.0)
            self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
