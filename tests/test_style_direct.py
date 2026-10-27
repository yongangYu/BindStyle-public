import ast
import contextlib
import io
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
import torch.nn as nn

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

from utils.direct_style_io import assert_separate_style_output, create_direct_style_output
from utils.style_numerics import check_backward_finite, require_finite


ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def config(value):
    if isinstance(value, dict):
        return Config({key: config(item) for key, item in value.items()})
    return value


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return value


def load_runtime():
    # Exercise production controller/checkpoint logic without importing CUDA
    # rasterizers or downloading diffusion weights on the CPU test host.
    path = ROOT / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "GaussianStyleDelta", "GaussianSHStyleController", "save_mags_checkpoint",
        "_torch_load_checkpoint", "_load_matching_state_dict",
        "_restore_gaussian_parameters", "load_pretrained_mags_checkpoint",
        "prepare_split_style_reference",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = {
        "torch": torch, "os": os, "random": random,
        "require_finite": require_finite,
        "check_backward_finite": check_backward_finite,
        "assert_separate_style_output": assert_separate_style_output,
        "OmegaConf": SimpleNamespace(to_container=lambda value, **kwargs: plain(value)),
        "CanonicalGaussianGeometryResidual": Mock(side_effect=AssertionError("Unexpected residual module")),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    namespace["GaussianStyleDelta"] = Mock(side_effect=AssertionError("Unexpected SH delta module"))
    return namespace


RUNTIME = load_runtime()
Controller = RUNTIME["GaussianSHStyleController"]


def load_mesh_binding():
    namespace = {"torch": torch, "nn": nn}
    graphics = ast.parse((ROOT / "utils/graphics_utils.py").read_text(encoding="utf-8"))
    nodes = [node for node in graphics.body if getattr(node, "name", None) == "uvw_to_xyz"]
    model = ast.parse((ROOT / "model/mags_model.py").read_text(encoding="utf-8"))
    cls = next(node for node in model.body if getattr(node, "name", None) == "MaGSModel")
    methods = {"get_uvw", "get_refined_verts", "get_xyz", "_set_xyz_cache", "update_mesh"}
    cls.body = [node for node in cls.body if getattr(node, "name", None) in methods]
    # Face orientation is fixed and independent of the optimized UVW. The
    # binding function and update_mesh below are the actual production methods.
    namespace.update({
        "compute_face_orientation": lambda verts, faces: (torch.eye(3)[None], torch.ones(1, 1)),
        "rotmat_to_unitquat": lambda matrix: torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "quat_xyzw_to_wxyz": lambda quat: quat[:, [3, 0, 1, 2]],
    })
    exec(compile(ast.Module(body=nodes + [cls], type_ignores=[]), str(ROOT / "model/mags_model.py"), "exec"), namespace)
    return namespace["MaGSModel"]


MeshBinding = load_mesh_binding()


class TinyGaussians(MeshBinding):
    def __init__(self):
        super().__init__()
        for name, value in {
            "_uvw": [[0.2, 0.3, 0.05]],
            "_features_dc": [[[0.1, 0.2, 0.3]]],
            "_features_rest": [[[0.2, 0.1, 0.3]]],
            "_scaling": [[-1.0, -1.2, -1.3]],
            "_rotation": [[1.0, 0.1, 0.2, 0.3]],
            "_opacity": [[0.2]],
            "mesh_feat": [[0.1, 0.2, 0.3]] * 3,
        }.items():
            setattr(self, name, nn.Parameter(torch.tensor(value)))
        self.device = "cpu"
        self.faceid = torch.tensor([0])
        self.active_sh_degree = self.max_sh_degree = 1
        self.xyz = torch.zeros(1, 3)
        self.max_radii2D = torch.zeros(1)
        self.xyz_gradient_accum = torch.zeros(1, 1)
        self.denom = torch.zeros(1, 1)
        for name in ("style_features_delta", "style_uvw_delta", "style_log_scaling_delta", "style_rotation_delta"):
            setattr(self, name, torch.ones(1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)


class ForbiddenDeformer(nn.Linear):
    def __init__(self):
        super().__init__(3, 3)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

    def forward(self, *args, **kwargs):
        raise AssertionError("Deformer must not run in direct mode")


class TinyTeacher:
    enabled = True
    gradient_mode = "csd"
    expects_video = True
    window_size = 2
    t_min = 0.2
    t_max = 0.65
    min_step = 200
    max_step = 650
    guidance_scale = 7.0
    ip_adapter_scale = 1.0

    def __call__(self, video, **kwargs):
        # A deterministic stand-in for the distilled image gradient. This test
        # verifies routing and optimizer ownership, not diffusion model quality.
        return video.square().mean(), {"csd_grad_norm": video.new_tensor(1.0)}


def make_config(root):
    return config({
        "model": {"model_path": str(root / "styled" / "new-run")},
        "optim": {
            "iterations": 5000, "batch_size": 1,
            "mesh_style_update": {"enabled": False},
            "gaussian_style_update": {
                "enabled": True, "parameterization": "direct",
                "start_checkpoint": str(root / "original" / "checkpoints" / "iteration_40000.pth"),
                "use_deformer": True, "dc_only_iterations": 0,
                "gt_mesh_trajectory": {"enabled": True, "window_size": 2, "deformer_mode": "frozen"},
                "direct_geometry": {"enabled": True, "start_iter": 1001},
            },
        },
    })


def make_controller(cfg, checkpoint=None, gaussians=None, deformer=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return Controller(
            cfg, gaussians or TinyGaussians(), deformer or ForbiddenDeformer(),
            checkpoint or {}, "cpu", train_dataset=[], shared_sds_guidance=TinyTeacher(),
        )


def render_proxy(controller, shift=0.0):
    gaussian = controller.gaussians
    controller.apply_geometry_to_gaussians()
    controller.apply_to_gaussians()
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]) + shift
    gaussian.update_mesh(
        torch.tensor([[0, 1, 2]]), vertices, torch.tensor([[0., 0., 1.]]),
        deformer=controller.deformer if controller.use_deformer else None,
    )
    value = (
        gaussian.get_xyz.square().sum()
        + gaussian._features_dc.square().sum()
        + gaussian._features_rest.square().sum()
        + gaussian._scaling.exp().sum()
        + gaussian._rotation.square().sum()
    )
    return value.expand(3, 2, 2)


class DirectStyleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = make_config(self.root)

    def train_window(self, controller, start):
        for slot in range(2):
            controller.prepare_iteration(start + slot)
            image = render_proxy(controller, shift=slot * 0.1)
            losses = controller.loss(image, source_key=slot)
            if not slot:
                self.assertFalse(losses["gaussian_style_video_window_ready"].item())
                self.assertFalse(losses["gaussian_style_loss"].requires_grad)
            else:
                self.assertTrue(controller.is_accumulation_end(start + slot, 1))
                losses["gaussian_style_loss"].backward()
                controller.verify_gradient_boundary()
                controller.step()
        return losses

    def test_initialization_owns_actual_parameters_and_disables_deformer(self):
        gaussian = TinyGaussians()
        identities = {name: id(param) for name, param in gaussian.named_parameters()}
        controller = make_controller(self.cfg, gaussians=gaussian)
        self.assertIsNone(controller.style_delta)
        self.assertIsNone(controller.geometry_residual)
        self.assertFalse(controller.use_deformer)
        self.assertEqual(controller.deformer_mode, "off")
        self.assertTrue(self.cfg.optim.disable_deformer)
        self.assertFalse(controller.deformer.training)
        self.assertTrue(all(not p.requires_grad for p in controller.deformer.parameters()))
        self.assertEqual(identities, {name: id(param) for name, param in gaussian.named_parameters()})
        for group in controller.optimizer.param_groups:
            self.assertIs(group["params"][0], getattr(gaussian, group["name"]))
        render_proxy(controller)
        for name in ("style_features_delta", "style_uvw_delta", "style_log_scaling_delta", "style_rotation_delta", "d_uvw", "d_xyz", "d_color"):
            self.assertIsNone(getattr(gaussian, name))

    def test_first_phase_changes_only_sh(self):
        controller = make_controller(self.cfg)
        original = {name: p.detach().clone() for name, p in controller.gaussians.named_parameters()}
        self.train_window(controller, 1)
        for name, param in controller.gaussians.named_parameters():
            changed = not torch.equal(param, original[name])
            self.assertEqual(changed, name in {"_features_dc", "_features_rest"}, name)
        self.assertEqual(controller.stage_name(2), "direct-gtmesh-style-sh")
        self.assertTrue(all(p.grad is None for p in controller.deformer.parameters()))

    def test_second_phase_changes_all_geometry_and_preserves_sh_momentum(self):
        controller = make_controller(self.cfg)
        self.train_window(controller, 999)
        dc_step = controller.optimizer.state[controller.sh_dc]["step"].item()
        identities = [id(p) for g in controller.optimizer.param_groups for p in g["params"]]
        original = {name: p.detach().clone() for name, p in controller.gaussians.named_parameters()}
        losses = self.train_window(controller, 1001)
        for name, param in controller.gaussians.named_parameters():
            self.assertEqual(not torch.equal(param, original[name]), name not in {"_opacity", "mesh_feat"}, name)
        self.assertEqual(controller.optimizer.state[controller.sh_dc]["step"].item(), dc_step + 1)
        self.assertEqual(identities, [id(p) for g in controller.optimizer.param_groups for p in g["params"]])
        self.assertEqual(controller.stage_name(1002), "direct-gtmesh-style-sh+geometry")
        torch.testing.assert_close(losses["gaussian_style_loss"], losses["gaussian_style_csd_loss"])
        for name, value in losses.items():
            if name.endswith("_reg"):
                self.assertEqual(value.item(), 0.0, name)

    def test_gt_mesh_binding_preserves_direct_uvw_gradient(self):
        controller = make_controller(self.cfg)
        controller.prepare_iteration(1001)
        render_proxy(controller).sum().backward()
        gradient = controller.gaussians._uvw.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient != 0).all())
        self.assertFalse(controller.gaussians.refined_verts.requires_grad)

    def test_geometry_disabled_and_outside_interval_stay_frozen(self):
        self.cfg.optim.gaussian_style_update.direct_geometry.enabled = False
        controller = make_controller(self.cfg)
        self.train_window(controller, 1001)
        self.assertFalse(controller.gaussians._uvw.requires_grad)
        controller.prepare_iteration(5001)
        self.assertTrue(all(not p.requires_grad for p in controller.gaussians.parameters()))

    def test_mid_window_geometry_switch_is_rejected(self):
        self.cfg.optim.gaussian_style_update.direct_geometry.start_iter = 1002
        with self.assertRaisesRegex(ValueError, "video-window"):
            make_controller(self.cfg)

    def test_residual_checkpoint_and_legacy_branch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "residual-style checkpoint"):
            make_controller(self.cfg, {"gaussian_style_delta": {}})
        self.cfg.optim.gaussian_style_update.canonical_geometry = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "direct_geometry"):
            make_controller(self.cfg)

    def test_nonzero_residual_regularizer_is_rejected(self):
        self.cfg.optim.gaussian_style_update.dc_reg_weight = 0.1
        with self.assertRaisesRegex(ValueError, "dc_reg_weight"):
            make_controller(self.cfg)

    def test_nan_gradient_is_rejected_before_direct_optimizer(self):
        controller = make_controller(self.cfg)
        controller.prepare_iteration(1)
        controller.sh_dc.grad = torch.full_like(controller.sh_dc, float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "gradient of _features_dc"):
            controller.step()
        self.assertFalse(controller.optimizer.state)

    def test_opacity_gradient_is_rejected(self):
        controller = make_controller(self.cfg)
        controller.prepare_iteration(1)
        controller.gaussians._opacity.grad = torch.ones_like(controller.gaussians._opacity)
        with self.assertRaisesRegex(RuntimeError, "_opacity"):
            controller.verify_gradient_boundary()

    def test_frozen_gradient_is_rejected_after_initial_boundary_check(self):
        controller = make_controller(self.cfg)
        self.train_window(controller, 1)
        controller.gaussians._opacity.grad = torch.ones_like(controller.gaussians._opacity)
        with self.assertRaisesRegex(RuntimeError, "_opacity"):
            controller.verify_gradient_boundary()

    def test_incomplete_window_cannot_be_saved(self):
        controller = make_controller(self.cfg)
        controller.prepare_iteration(1)
        controller.loss(render_proxy(controller), source_key=0)
        with self.assertRaisesRegex(RuntimeError, "unfinished video window"):
            RUNTIME["save_mags_checkpoint"](self.cfg, 1, controller.gaussians, controller.deformer, controller)
        self.assertFalse((Path(self.cfg.model.model_path) / "checkpoints").exists())

    def test_save_load_resume_contains_absolute_parameters_and_preserves_source(self):
        source = Path(self.cfg.optim.gaussian_style_update.start_checkpoint)
        source.parent.mkdir(parents=True)
        source.write_bytes(b"original model must remain untouched")
        controller = make_controller(self.cfg)
        self.train_window(controller, 999)
        self.train_window(controller, 1001)
        with contextlib.redirect_stdout(io.StringIO()):
            RUNTIME["save_mags_checkpoint"](self.cfg, 1002, controller.gaussians, controller.deformer, controller)
        saved_path = Path(self.cfg.model.model_path) / "checkpoints/iteration_1002.pth"
        saved_bytes = saved_path.read_bytes()
        payload = torch.load(saved_path, weights_only=False)
        self.assertNotIn("gaussian_style_delta", payload)
        self.assertNotIn("gaussian_style_geometry_residual", payload)
        self.assertEqual(payload["gaussian_style_parameterization"], "direct")
        for name, param in controller.gaussians.named_parameters():
            torch.testing.assert_close(payload["gaussians"][name], param)
        with self.assertRaises(FileExistsError):
            RUNTIME["save_mags_checkpoint"](self.cfg, 1002, controller.gaussians, controller.deformer, controller)
        self.assertEqual(saved_bytes, saved_path.read_bytes())
        self.assertEqual(source.read_bytes(), b"original model must remain untouched")

        restored_gaussians, restored_deformer = TinyGaussians(), ForbiddenDeformer()
        with contextlib.redirect_stdout(io.StringIO()):
            payload = RUNTIME["load_pretrained_mags_checkpoint"](str(saved_path), restored_gaussians, restored_deformer, "cpu")
        self.cfg.optim.gaussian_style_update.start_checkpoint = str(saved_path)
        self.cfg.model.model_path = str(self.root / "resumed" / "new-run")
        restored = make_controller(self.cfg, payload, restored_gaussians, restored_deformer)
        self.assertEqual(restored.loop_start_iteration, 1003)
        restored.prepare_iteration(1003)
        self.assertTrue(restored.gaussians._uvw.requires_grad)
        torch.testing.assert_close(restored.sh_dc, controller.sh_dc)
        self.assertEqual(restored.optimizer.state[restored.sh_dc]["step"].item(), 2)

    def test_resume_schedule_change_is_rejected(self):
        controller = make_controller(self.cfg)
        state = controller.direct_state_dict()
        state["geometry_start_iter"] = 2001
        with self.assertRaisesRegex(ValueError, "preserve the SH/geometry schedule"):
            make_controller(self.cfg, {"gaussian_style_parameterization": "direct", "direct_style_state": state, "iteration": 1000})

    def test_resume_mid_window_is_rejected(self):
        controller = make_controller(self.cfg)
        with self.assertRaisesRegex(ValueError, "completed video window"):
            make_controller(self.cfg, {"gaussian_style_parameterization": "direct", "direct_style_state": controller.direct_state_dict(), "iteration": 999})


class DirectOutputTests(unittest.TestCase):
    def test_rejects_original_run_and_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original/checkpoints/iteration_40000.pth"
            for output in (root / "original", root / "original/new-run", root / "x/../original"):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside the source"):
                    create_direct_style_output(output, source)
            self.assertFalse((root / "original").exists())

    def test_allows_separate_sibling_and_allocates_unique_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original/checkpoints/iteration_40000.pth"
            first = create_direct_style_output(root / "styled", source)
            second = create_direct_style_output(root / "styled", source)
            self.assertNotEqual(first, second)
            self.assertTrue(Path(first).is_dir())
            self.assertTrue(Path(first).name.startswith("direct-"))

    def test_missing_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "start_checkpoint"):
            assert_separate_style_output("output", "")


@unittest.skipIf(OmegaConf is None, "OmegaConf is not installed")
class DirectConfigTests(unittest.TestCase):
    def test_example_uses_real_project_config_merge_and_initializes_controller(self):
        tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
        node = next(node for node in tree.body if getattr(node, "name", None) == "load_training_config")
        namespace = {"OmegaConf": OmegaConf}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "train.py"), "exec"), namespace)
        paths = ",".join(str(ROOT / name) for name in (
            "config/3dgs.yaml", "config/dnerf/jumpingjacks.yaml", "config/style/direct_gaussian.yaml",
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = namespace["load_training_config"]([
                paths, f"model.model_path={root / 'styled'}",
                f"optim.gaussian_style_update.start_checkpoint={root / 'original/checkpoints/iteration_40000.pth'}",
            ])
            self.assertEqual(cfg.optim.gaussian_style_update.parameterization, "direct")
            self.assertEqual(cfg.optim.gaussian_style_update.gt_mesh_trajectory.deformer_mode, "off")
            self.assertEqual(cfg.optim.gaussian_style_update.direct_geometry.start_iter, 1001)
            self.assertFalse(cfg.optim.gaussian_style_update.sds.vae_half_precision)
            self.assertEqual(cfg.optim.gaussian_style_update.sds.provider, "sdxl_animatediff_controlnet")
            controller = make_controller(cfg)
            controller.prepare_iteration(1001)
            self.assertTrue(controller.gaussians._uvw.requires_grad)
            self.assertFalse(controller.use_deformer)


if __name__ == "__main__":
    unittest.main()
