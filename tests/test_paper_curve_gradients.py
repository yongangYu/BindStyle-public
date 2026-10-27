from types import SimpleNamespace
import unittest

import torch
from torch import nn
from utils.split_style_training import routed_style_gradients
from utils.sdxl_animatediff_controlnet_csd_guidance import SDXLAnimateDiffControlNetCSDGuidance as Guidance


class Appearance(nn.Module):
    def forward(self, video):
        return (video - .25).square().mean()


class Teacher:
    def __call__(self, video, **kwargs):
        return -video.mean(), {}


class GradientTests(unittest.TestCase):
    def run_route(self, sh_enabled, geo_enabled, kind):
        sh = nn.Parameter(torch.tensor(.7), requires_grad=sh_enabled)
        geo = nn.Parameter(torch.tensor(.2), requires_grad=geo_enabled)
        opacity = nn.Parameter(torch.tensor(.3), requires_grad=False)
        video = (sh + 2 * geo + opacity).expand(1, 2, 3, 2, 2)
        teacher = None if kind == "vgg" else Teacher()
        pairs, metrics = routed_style_gradients(video, [0, 1], Appearance(), teacher, [sh], [geo], 1., 1.,
                                                geometry_supervision=kind)
        self.assertTrue(all(p.grad is None for p in (sh, geo, opacity)))
        return pairs, metrics

    def test_geometry_first_has_no_sh_gradients(self):
        pairs, metrics = self.run_route(False, True, "diffusion")
        self.assertEqual(len(pairs), 1)
        torch.testing.assert_close(pairs[0][1], torch.tensor(-2.))
        self.assertEqual(float(metrics["gaussian_style_appearance_loss"]), 0.)

    def test_vgg_geometry_uses_same_loss_and_both_parameters(self):
        pairs, metrics = self.run_route(True, True, "vgg")
        self.assertEqual(len(pairs), 2)
        torch.testing.assert_close(pairs[1][1], 2 * pairs[0][1])
        self.assertEqual(float(metrics["gaussian_style_csd_loss"]), 0.)
        self.assertGreater(float(metrics["gaussian_style_geometry_loss"]), 0.)

    def test_geometry_only_vgg_and_sh_only_are_supported(self):
        self.assertEqual(len(self.run_route(False, True, "vgg")[0]), 1)
        self.assertEqual(len(self.run_route(True, False, "diffusion")[0]), 1)

    def test_motion_ablation_preserves_frame_condition_order(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = nn.Parameter(torch.zeros(()))
                self.received = None

            def forward(self, value, timesteps, **kwargs):
                self.received = (value, timesteps, kwargs)
                return (value,)

        class Control(Net):
            def forward(self, value, timesteps, **kwargs):
                self.received = (value, timesteps, kwargs)
                return (value,), value

        model = Guidance.__new__(Guidance)
        nn.Module.__init__(model)
        model.temporal_mode = "framewise"
        model.controlnet_conditioning_scale = .35
        model.pipe = SimpleNamespace(unet=Net())
        model.controlnet = Control()
        model.scheduler = SimpleNamespace(scale_model_input=lambda x, t: x)
        prompt = torch.tensor([[[10.]], [[20.]]])
        pooled = torch.tensor([[30.], [40.]])
        time_ids = torch.tensor([[50.], [60.]])
        embeds = [torch.tensor([[[70.]], [[80.]]])]
        model._branch_conditions = lambda *args: (prompt, pooled, time_ids, embeds)
        model._prediction_to_epsilon = lambda prediction, *args: prediction
        noisy = torch.arange(2 * 4 * 3 * 2 * 2).reshape(2, 4, 3, 2, 2).float()
        t = torch.tensor([100, 200])
        result = model._predict_video_epsilon(noisy, t, torch.zeros(1, 3, 3, 2, 2), True, embeds)
        torch.testing.assert_close(result, noisy)
        value, actual_t, kwargs = model.pipe.unet.received
        self.assertEqual(value.shape, (6, 4, 2, 2))
        self.assertEqual(actual_t.tolist(), [100] * 3 + [200] * 3)
        self.assertEqual(kwargs["encoder_hidden_states"].flatten().tolist(), [10.] * 3 + [20.] * 3)
        self.assertEqual(kwargs["added_cond_kwargs"]["image_embeds"][0].flatten().tolist(), [70.] * 3 + [80.] * 3)
        self.assertIn("down_block_additional_residuals", kwargs)


if __name__ == "__main__":
    unittest.main()
