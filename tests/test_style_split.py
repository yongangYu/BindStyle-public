import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from test_style_direct import Controller, RUNTIME, TinyGaussians, ForbiddenDeformer, make_config, render_proxy, ROOT, OmegaConf
from utils.vgg_appearance_loss import VGGAppearanceLoss, foreground_box, nearest_feature_loss
from utils.split_style_training import routed_style_gradients, reference_signature
from utils.mesh_renderer_nvdiffrast import NVDiffMeshRenderer, sequence_inverse_depth_range


class Appearance(nn.Module):
    style_fingerprint = "test-style"
    crop = [0, 0, 1, 1]
    loss_type = "nnfm"
    layers = (11, 13, 15)
    image_size = 256

    def forward(self, image):
        return (image - 0.25).square().mean()


class Teacher:
    enabled = True
    gradient_mode = "csd"
    expects_video = True
    window_size = 2
    t_min, t_max = 0.2, 0.65
    min_step, max_step = 200, 650
    guidance_scale, ip_adapter_scale = 7.0, 1.0

    def __init__(self):
        self.calls = 0

    def __call__(self, video, **kwargs):
        self.calls += 1
        return -video.mean(), {"csd_grad_norm": video.new_tensor(1.0)}

    def park_diffusion_backbone(self):
        pass


class SplitStyleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = make_config(self.root)
        self.cfg.optim.gaussian_style_update.supervision_mode = "split_vgg_diffusion"
        self.reference = {"signature": "fixed-reference", "images": {0: torch.rand(3, 2, 2)}}

    def controller(self, checkpoint=None, teacher=None, reference=None):
        with contextlib.redirect_stdout(io.StringIO()), patch("utils.vgg_appearance_loss.VGGAppearanceLoss", return_value=Appearance()):
            gaussian = TinyGaussians()
            if checkpoint:
                gaussian.load_state_dict(checkpoint["gaussians"])
            return Controller(self.cfg, gaussian, ForbiddenDeformer(), checkpoint or {}, "cpu",
                              train_dataset=[], shared_sds_guidance=teacher or Teacher(),
                              source_reference=reference or self.reference)

    def window(self, controller, start):
        controller.prepare_iteration(start)
        first = controller.loss(render_proxy(controller), source_key=0)
        self.assertFalse(first["gaussian_style_video_window_ready"].item())
        controller.prepare_iteration(start + 1)
        losses = controller.loss(render_proxy(controller, 0.1), source_key=1)
        controller.backward_style_loss(losses)
        controller.verify_gradient_boundary()
        controller.step()
        return losses

    def test_sh_phase_does_not_call_teacher_or_change_geometry(self):
        teacher = Teacher()
        controller = self.controller(teacher=teacher)
        before = {n: p.clone() for n, p in controller.gaussians.named_parameters()}
        losses = self.window(controller, 1)
        self.assertEqual(teacher.calls, 0)
        self.assertEqual(losses["gaussian_style_csd_loss"].item(), 0)
        self.assertGreater(losses["gaussian_style_appearance_loss"].item(), 0)
        for name, p in controller.gaussians.named_parameters():
            self.assertEqual(not torch.equal(before[name], p), name in {"_features_dc", "_features_rest"})
        self.assertFalse(controller.geometry_optimizer.state)

    def test_route_matches_selected_partial_derivatives_without_cross_accumulation(self):
        controller = self.controller()
        controller.prepare_iteration(1001)
        sh = [controller.sh_dc, controller.sh_rest]
        geom = list(controller.geometry_parameters().values())
        video = torch.stack([render_proxy(controller), render_proxy(controller, 0.1)])[None]
        expected_sh = torch.autograd.grad(Appearance()(video.flatten(0, 1)), sh, retain_graph=True)
        expected_geo = torch.autograd.grad(-video.mean(), geom, retain_graph=True)
        pairs, metrics = routed_style_gradients(video, [0, 1], Appearance(), Teacher(), sh, geom, 1., 1.)
        for (_, actual), expected in zip(pairs, expected_sh + expected_geo):
            torch.testing.assert_close(actual, expected)
        self.assertTrue(all(p.grad is None for p in controller.gaussians.parameters()))
        self.assertTrue(all(p.grad is None for p in controller.deformer.parameters()))

    def test_teacher_strength_cannot_change_sh_gradient(self):
        results = []
        for weight in (0.1, 10.0):
            controller = self.controller()
            controller.prepare_iteration(1001)
            self.window_gradients(controller, weight)
            results.append((controller.sh_dc.grad.clone(), controller.gaussians._uvw.grad.clone()))
        torch.testing.assert_close(results[0][0], results[1][0])
        torch.testing.assert_close(results[1][1], 100 * results[0][1])

    def test_geometry_clipping_cannot_shrink_the_sh_optimizer_step(self):
        values = []
        for weight in (1., 1e8):
            controller = self.controller()
            controller.current_sds_weight = weight
            self.window(controller, 1001)
            values.append(controller.sh_dc.detach().clone())
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)

    def window_gradients(self, controller, weight):
        controller.current_sds_weight = weight
        controller.prepare_iteration(1001)
        controller.loss(render_proxy(controller), source_key=0)
        controller.prepare_iteration(1002)
        losses = controller.loss(render_proxy(controller, 0.1), source_key=1)
        controller.backward_style_loss(losses)

    def test_bad_teacher_does_not_commit_partial_sh_gradients(self):
        class BadTeacher(Teacher):
            def __call__(self, video, **kwargs):
                return video.sum() * float("nan"), {}
        controller = self.controller(teacher=BadTeacher())
        with self.assertRaises(FloatingPointError):
            self.window_gradients(controller, 1.)
        self.assertTrue(all(p.grad is None for p in controller.gaussians.parameters()))
        self.assertIsNone(controller._split_pending)
        self.assertFalse(controller.optimizer.state)

    def test_second_phase_uses_two_optimizers_and_saves_reference_for_resume(self):
        controller = self.controller()
        self.window(controller, 999)
        self.window(controller, 1001)
        self.assertEqual(controller.optimizer.state[controller.sh_dc]["step"].item(), 2)
        for p in controller.geometry_parameters().values():
            self.assertEqual(controller.geometry_optimizer.state[p]["step"].item(), 1)
        self.assertTrue(all(p.grad is None for p in controller.gaussians.parameters()))
        with contextlib.redirect_stdout(io.StringIO()):
            RUNTIME["save_mags_checkpoint"](self.cfg, 1002, controller.gaussians, controller.deformer, controller)
        path = Path(self.cfg.model.model_path) / "checkpoints/iteration_1002.pth"
        saved = torch.load(path, weights_only=False)
        self.assertIn("gaussian_geometry_optimizer", saved)
        torch.testing.assert_close(saved["split_style_reference"]["images"][0], self.reference["images"][0])
        self.cfg.optim.gaussian_style_update.start_checkpoint = str(path)
        self.cfg.model.model_path = str(self.root / "another-run")
        restored = self.controller(saved, reference=saved["split_style_reference"])
        self.assertEqual(restored.loop_start_iteration, 1003)
        self.assertEqual(restored.geometry_optimizer.state[restored.gaussians._uvw]["step"].item(), 1)
        self.window(controller, 1003)
        self.window(restored, 1003)
        for name, p in controller.gaussians.named_parameters():
            torch.testing.assert_close(p, dict(restored.gaussians.named_parameters())[name])

    def test_mid_window_and_missing_backward_cannot_step(self):
        controller = self.controller()
        controller.prepare_iteration(1)
        controller.loss(render_proxy(controller), source_key=0)
        with self.assertRaisesRegex(RuntimeError, "inside"):
            controller.step()
        controller.prepare_iteration(2)
        controller.loss(render_proxy(controller), source_key=1)
        with self.assertRaisesRegex(RuntimeError, "inside"):
            controller.step()
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            RUNTIME["save_mags_checkpoint"](self.cfg, 2, controller.gaussians, controller.deformer, controller)

    def test_vgg_strength_cannot_change_geometry_gradient(self):
        gradients = []
        for weight in (1., 5.):
            controller = self.controller()
            controller.appearance_weight = weight
            self.window_gradients(controller, 1.)
            gradients.append((controller.sh_dc.grad.clone(), controller.gaussians._uvw.grad.clone()))
        torch.testing.assert_close(gradients[1][0], gradients[0][0] * 5)
        torch.testing.assert_close(gradients[1][1], gradients[0][1])

    def test_pixel_gradient_diagnostics_and_parameter_update_metrics_exist(self):
        controller = self.controller()
        losses = self.window(controller, 1001)
        for name in ("appearance", "geometry"):
            self.assertGreater(losses[f"gaussian_style_{name}_rgb_grad_norm"].item(), 0)
        self.assertEqual(len(controller.last_update_metrics), 5)
        self.assertTrue(all(torch.isfinite(value) for value in controller.last_update_metrics.values()))

    def test_geometry_disabled_runs_vgg_only(self):
        self.cfg.optim.gaussian_style_update.direct_geometry.enabled = False
        teacher = Teacher()
        controller = self.controller(teacher=teacher)
        self.window(controller, 1001)
        self.assertEqual(teacher.calls, 0)
        self.assertIsNone(controller.geometry_optimizer)

    def test_legacy_objective_checkpoint_cannot_silently_change_objective(self):
        with self.assertRaisesRegex(ValueError, "different supervision_mode"):
            self.controller({"gaussian_style_parameterization": "direct", "gaussians": TinyGaussians().state_dict()})

    def test_missing_split_optimizer_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete split style checkpoint"):
            self.controller({"gaussian_style_parameterization": "direct",
                             "gaussian_style": {"parameterization": "direct", "supervision_mode": "split_vgg_diffusion"},
                             "gaussians": TinyGaussians().state_dict()})

    def test_fixed_source_render_is_reused_after_gaussian_changes(self):
        self.cfg.dataset = SimpleNamespace(white_background=False)
        self.cfg.pipe = SimpleNamespace()
        gaussians = TinyGaussians()
        gaussians.style_uvw_delta = None
        cam = SimpleNamespace(uid=0, full_proj_transform=torch.eye(4), gt_mask=torch.ones(1, 4, 4))
        faces = torch.tensor([[0, 1, 2]])
        meshes = {"train_verts": {0: torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])},
                  "train_normals": {0: torch.tensor([[0., 0., 1.]])}}
        calls = []

        def renderer(camera, gaussian, pipe, background):
            calls.append(camera.uid)
            return {"render": gaussian._features_dc.reshape(3, 1, 1).expand(3, 4, 4)}

        with patch.dict(RUNTIME, {"inria_render": renderer,
                                  "torchvision": SimpleNamespace(utils=SimpleNamespace(save_image=lambda *a: None))}):
            prepare = RUNTIME["prepare_split_style_reference"]
            initial = prepare(self.cfg, {}, gaussians, [cam], faces, meshes)
            saved_pixels = initial["images"][0].clone()
            with torch.no_grad():
                gaussians._features_dc.add_(.1)
            restored = prepare(self.cfg, {"split_style_reference": initial}, gaussians, [cam], faces, meshes)
            self.assertEqual(calls, [0])
            self.assertEqual(restored["images"][0].device.type, "cpu")
            self.assertFalse(restored["images"][0].requires_grad)
            torch.testing.assert_close(saved_pixels, restored["images"][0])
            self.cfg.dataset.white_background = True
            with self.assertRaisesRegex(ValueError, "reference does not match"):
                prepare(self.cfg, {"split_style_reference": initial}, gaussians, [cam], faces, meshes)

    @unittest.skipIf(OmegaConf is None, "OmegaConf is not installed")
    def test_new_yaml_profile_merges_and_constructs_split_controller(self):
        self.cfg = OmegaConf.merge(*[OmegaConf.load(ROOT / path) for path in (
            "config/3dgs.yaml", "config/dnerf/jumpingjacks.yaml", "config/style/split_vgg_geometry.yaml",
        )])
        self.cfg.model.model_path = str(self.root / "styled")
        self.cfg.optim.gaussian_style_update.start_checkpoint = str(self.root / "source/checkpoints/iteration_40000.pth")
        self.assertEqual(self.cfg.optim.gaussian_style_update.sds.control_depth_mode, "relative_inverse")
        self.assertFalse(self.cfg.optim.gaussian_style_update.appearance.allow_download)
        controller = self.controller()
        self.assertTrue(controller.split_supervision)
        self.window(controller, 1)


class AppearanceTests(unittest.TestCase):
    def test_chunked_nnfm_value_and_gradient_match_dense_cosine_matching(self):
        torch.manual_seed(12)
        output = torch.rand(2, 4, 3, 3, requires_grad=True)
        style = torch.rand(1, 4, 2, 3)
        value = nearest_feature_loss(output, style, chunk_size=2)
        query = F.normalize(output.flatten(2).transpose(1, 2), dim=-1).reshape(-1, 4)
        keys = F.normalize(style.flatten(2).transpose(1, 2), dim=-1).reshape(-1, 4)
        expected = (1 - (query @ keys.T).max(-1).values).mean()
        torch.testing.assert_close(value, expected)
        torch.testing.assert_close(torch.autograd.grad(value, output, retain_graph=True)[0],
                                   torch.autograd.grad(expected, output)[0])

    def test_fixed_crop_uses_union_of_gt_masks(self):
        a, b = torch.zeros(1, 10, 10), torch.zeros(1, 10, 10)
        a[:, 2:4, 3:5] = 1
        b[:, 4:8, 4:9] = 1
        torch.testing.assert_close(torch.tensor(foreground_box([a, b], 0)), torch.tensor([.2, .3, .8, .9]))

    def test_cached_features_are_frozen_but_rgb_receives_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "style.png"
            Image.fromarray(np.full((32, 32, 3), [70, 100, 200], dtype=np.uint8)).save(path)
            cfg = {"layers": [0], "image_size": 32, "chunk_size": 64}
            for loss_type in ("nnfm", "gram", "adain"):
                cfg["loss_type"] = loss_type
                features = nn.Sequential(nn.Conv2d(3, 4, 1))
                loss = VGGAppearanceLoss(cfg, path, [SimpleNamespace(gt_mask=torch.ones(1, 32, 32))], "cpu", features)
                image = torch.rand(2, 3, 32, 32, requires_grad=True)
                loss(image).backward()
                self.assertTrue(torch.isfinite(image.grad).all())
                self.assertGreater(image.grad.norm().item(), 0)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in loss.parameters()))
                self.assertFalse(loss.style_0.requires_grad)


class DepthTests(unittest.TestCase):
    def camera(self, depths):
        projection = torch.eye(4)
        projection[3, 3] = 0
        projection[2, 3] = 1
        return SimpleNamespace(uid=0, znear=.1, zfar=100., full_proj_transform=projection,
                               gt_mask=torch.ones(1, 4, 4), mesh_vertices=torch.tensor([[0., 0., z] for z in depths]))

    def test_range_is_shared_across_frames_and_nearer_depth_is_brighter(self):
        a, b = self.camera([2., 3.]), self.camera([4., 6.])
        low, high = sequence_inverse_depth_range([a, b])
        self.assertAlmostEqual(low, 1 / 6)
        self.assertAlmostEqual(high, 1 / 2)
        values = (torch.tensor([2., 3., 6.]).reciprocal() - low) / (high - low)
        self.assertTrue(torch.all(values[:-1] > values[1:]))
        self.assertAlmostEqual(values[0].item(), 1.)
        self.assertAlmostEqual(values[-1].item(), 0.)

    def test_flat_depth_has_nonzero_normalization_range(self):
        low, high = sequence_inverse_depth_range([self.camera([2., 2.])])
        self.assertGreater(high, low)

    def test_legacy_ndc_and_new_camera_depth_attributes(self):
        clip = torch.tensor([[0., 0., 1.8, 2.], [0., 0., 5.8, 6.]])
        legacy = NVDiffMeshRenderer(mode="depth")
        torch.testing.assert_close(legacy._attributes(None, None, clip)[:, 0], (clip[:, 2] / clip[:, 3] + 1) / 2)
        updated = NVDiffMeshRenderer(mode="depth", depth_mode="relative_inverse", depth_range=(.1, .5))
        torch.testing.assert_close(updated._attributes(None, None, clip)[:, 0], clip[:, 3])

    def test_reference_identity_changes_when_camera_mesh_or_mask_changes(self):
        cam = self.camera([2., 3., 4.])
        faces = torch.tensor([[0, 1, 2]])
        meshes = {0: cam.mesh_vertices}
        before = reference_signature([cam], faces, meshes)
        cam.gt_mask[0, 0, 0] = 0
        self.assertNotEqual(before, reference_signature([cam], faces, meshes))


if __name__ == "__main__":
    unittest.main()
