import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from utils.paper_eval_protocol import (aggregate, bilinear_sample, erode_mask, flow_correspondence,
                                      geometry_statistics, histogram_distance, masked_rmse,
                                      mesh_correspondence, pixel_grid, record, rgbuv_histogram,
                                      temporal_pairs, view_pairs, write_json)


class ProtocolTests(unittest.TestCase):
    def test_rmse_range_and_empty_mask(self):
        a = np.zeros((8, 8, 3), np.float32)
        b = np.ones_like(a)
        mask = np.ones((8, 8), bool)
        self.assertEqual(masked_rmse(a, a, mask)["value"], 0)
        self.assertEqual(masked_rmse(a, b, mask)["value"], 1)
        self.assertIsNone(masked_rmse(a, b, ~mask)["value"])
        with self.assertRaises(ValueError):
            masked_rmse(a, b * 255, mask)

    def test_mask_does_not_reward_background(self):
        a = np.zeros((10, 10, 3), np.float32)
        b = a.copy()
        b[:2, :2] = 1
        mask = np.zeros((10, 10), bool)
        mask[:2, :2] = True
        self.assertEqual(masked_rmse(a, b, mask)["value"], 1)

    def test_histogram_probability_and_distance(self):
        a = np.full((10, 10, 3), 0.5, np.float32)
        b = a.copy()
        b[..., 0] = 1
        ha, hb = rgbuv_histogram(a, bins=16), rgbuv_histogram(b, bins=16)
        self.assertAlmostEqual(ha.sum(), 1)
        self.assertEqual(histogram_distance(ha, ha), 0)
        self.assertTrue(0 < histogram_distance(ha, hb) < 1)
        self.assertAlmostEqual(histogram_distance([1, 0], [0, 1]), 1)
        self.assertTrue(np.isfinite(rgbuv_histogram(np.zeros_like(a))).all())

    def test_temporal_protocol_not_playback_seconds(self):
        times = np.linspace(0, 1, 200)
        short, long = temporal_pairs(times, 2), temporal_pairs(times, 20)
        self.assertEqual((len(short), len(long)), (198, 180))
        self.assertAlmostEqual(short[0]["delta_tau"], 2 / 199)
        self.assertAlmostEqual(long[0]["delta_tau"], 20 / 199)
        self.assertEqual(temporal_pairs(np.linspace(0, 1, 10), 20), [])
        with self.assertRaises(ValueError):
            temporal_pairs([0, 0, 1], 2)

    def test_view_angle_bins(self):
        angles = np.deg2rad([0, 5, 35, 90])
        centers = np.stack([np.sin(angles), np.zeros(4), np.cos(angles)], axis=-1)
        result = view_pairs(centers, [0, 0, 0])
        self.assertEqual(next(r["group"] for r in result if (r["a"], r["b"]) == (0, 1)), "narrow")
        self.assertEqual(next(r["group"] for r in result if (r["a"], r["b"]) == (0, 2)), "wide")
        self.assertFalse(any((r["a"], r["b"]) == (0, 3) for r in result))

    def test_flow_translation_and_occlusion(self):
        forward = np.zeros((8, 8, 2), np.float32)
        forward[..., 0] = 1
        mask = np.ones((8, 8), bool)
        xy, valid = flow_correspondence(forward, -forward, mask, mask)
        self.assertEqual(valid.sum(), 56)
        self.assertFalse(valid[:, -1].any())
        np.testing.assert_allclose(xy[..., 0], pixel_grid(8, 8)[..., 0] + 1)
        _, invalid = flow_correspondence(forward, forward, mask, mask)
        self.assertFalse(invalid.any())
        self.assertEqual(erode_mask(mask, 1).sum(), 36)

    def test_mesh_identity_and_depth_rejection(self):
        grid = pixel_grid(8, 8)
        points = np.concatenate([(grid + 0.5) / 4 - 1, np.zeros((8, 8, 1))], -1)
        data = {"points": points, "mask": np.ones((8, 8), bool), "depth": np.ones((8, 8))}
        xy, mask = mesh_correspondence(data, data, np.eye(4))
        np.testing.assert_allclose(xy, grid, atol=1e-6)
        self.assertTrue(mask.all())
        _, mask = mesh_correspondence(data, {**data, "depth": np.full((8, 8), 0.5)}, np.eye(4))
        self.assertFalse(mask.any())
        np.testing.assert_allclose(bilinear_sample(points, grid), points)

    def test_geometry_units_and_scale_clip(self):
        a = np.zeros((4, 3))
        b = a.copy()
        b[:, 0] = 0.1
        zeros = np.zeros_like(a)
        result = geometry_statistics(a, b, 10, zeros, zeros + np.log(2))
        self.assertAlmostEqual(result["delta_pos"], 0.01)
        self.assertAlmostEqual(result["delta_scale"], np.log(2))
        self.assertEqual(geometry_statistics(a, a, 1, zeros - 100, zeros + 100)["delta_scale"], 28)
        with self.assertRaises(ValueError):
            geometry_statistics(a, b[:2], 10, zeros, zeros)

    def test_missing_not_zero_and_strict_json(self):
        self.assertEqual(aggregate([record(0.2), record(reason="missing")])["value"], 0.2)
        self.assertIsNone(aggregate([record(reason="missing")])["value"])
        self.assertEqual(record(float("nan"))["status"], "nonfinite")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            write_json(path, {"metric": record(float("inf"))})
            self.assertNotIn("Infinity", path.read_text())
            self.assertEqual(json.loads(path.read_text())["metric"]["value"], None)

    def test_hook_after_final_save_not_inside_training_loop(self):
        root = ast.parse((Path(__file__).resolve().parents[1] / "train.py").read_text(encoding="utf-8"))
        training = next(node for node in root.body if isinstance(node, ast.FunctionDef) and node.name == "training")
        loops = [node for node in training.body if isinstance(node, ast.For)]
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                             node.func.id == "export_paper_bundle" for loop in loops for node in ast.walk(loop)))
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                            node.func.id == "export_paper_bundle" for node in ast.walk(training)))


class RunnerTests(unittest.TestCase):
    def test_bundle_outputs_and_missing_models(self):
        from tools.evaluate_paper import evaluate_bundle
        fake = ModuleType("utils.paper_eval_metrics")

        class Models:
            def __init__(self, cfg):
                self.weights, self.errors = {}, {}

            def lpips(self, a, b, mask=None):
                self.errors["lpips"] = "test missing weights"
                raise FileNotFoundError("test missing weights")

            def artscore(self, image):
                raise FileNotFoundError("test missing ArtScore")

            def cfsd(self, a, b):
                raise ValueError("disabled")

            def flow(self, a, b):
                return np.zeros((*a.shape[:2], 2), np.float32)

        fake.PaperModels = Models
        fake.resize_histogram_input = lambda image, size: image
        fake.ssim_content = lambda a, b: float(np.array_equal(a, b))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            Image.new("RGB", (8, 8), (100, 60, 30)).save(path / "style.png")
            samples = []
            for i in range(3):
                name = f"time_{i:04d}"
                a = np.ones((8, 8, 3), np.float32) * 0.4
                np.savez(path / f"{name}.npz", source=a, styled=a, mask=np.ones((8, 8), bool))
                samples.append({"id": name, "crop": [0, 8, 0, 8], "frame_index": i})
            manifest = {"protocol": "test", "scope": "pilot", "scene": "test", "style_sha256": "test",
                        "settings": {"mask_erosion": 0, "histogram": {"bins": 4}}, "samples": samples,
                        "view_pairs": [], "temporal_pairs": [{"a": "time_0000", "b": "time_0002", "group": "short", "delta_tau": 1}],
                        "geometry": [{"delta_pos": 0, "delta_pos_p50": 0, "delta_pos_p95": 0, "delta_scale": 0}],
                        "coverage_status": "not evaluated", "optimizer_steps": {}}
            write_json(path / "manifest.json", manifest)
            human_file = path / "user_study_template.csv"
            human_file.write_text("existing human ratings\n", encoding="utf-8")
            with patch.dict("sys.modules", {"utils.paper_eval_metrics": fake}), contextlib.redirect_stdout(io.StringIO()):
                summary = evaluate_bundle(path)
            self.assertEqual(summary["metrics"]["temporal_short"]["rmse"]["value"], 0)
            self.assertIsNone(summary["metrics"]["temporal_short"]["lpips"]["value"])
            self.assertIsNone(summary["metrics"]["temporal_long"]["rmse"]["value"])
            self.assertIsNone(summary["user_study"]["style"]["value"])
            self.assertEqual(json.loads((path / "status.json").read_text())["status"], "partial")
            self.assertTrue((path / "pair_directions.csv").is_file())
            self.assertTrue((path / "correspondence/temporal_00000_0.npz").is_file())
            self.assertEqual(human_file.read_text(encoding="utf-8"), "existing human ratings\n")
            self.assertIn("artfid", (path / "summary.csv").read_text())


class CollectionTests(unittest.TestCase):
    def make_bundle(self, path, style, value, scope="full_sequence", method="Ours"):
        write_json(path / "manifest.json", {
            "scope": scope, "scene": "scene", "source_checkpoint_sha256": "source",
            "mesh_sequence_sha256": "mesh", "style_sha256": style,
            "samples": [], "view_pairs": [], "temporal_pairs": []})
        write_json(path / "summary.json", {
            "settings": {}, "weights": {}, "method": method, "variant": "full",
            "metrics": {"foreground": {"cm": record(value)}}, "geometry": {}})
        return path

    def test_equal_combination_weighting_and_single_style_artfid(self):
        from tools.collect_paper_results import collect
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [self.make_bundle(root / "a", "style-a", 0.1),
                    self.make_bundle(root / "b", "style-a", 0.3),
                    self.make_bundle(root / "c", "style-b", 0.9)]
            rows = collect(runs, root / "result")
            self.assertAlmostEqual(next(r["value"] for r in rows if r["metric"] == "cm"), 0.55)
            self.assertIsNone(next(r["value"] for r in rows if r["metric"] == "artfid"))
            self.assertTrue((root / "result/collection.json").is_file())

    def test_reject_pilot_duplicate_and_unmatched_methods(self):
        from tools.collect_paper_results import collect
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot = self.make_bundle(root / "pilot", "style-a", 0.1, scope="pilot")
            with self.assertRaisesRegex(ValueError, "pilot"):
                collect([pilot], root / "result")
            a = self.make_bundle(root / "a", "style-a", 0.1)
            b = self.make_bundle(root / "b", "style-b", 0.2, method="Baseline")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                collect([a, a], root / "result")
            with self.assertRaisesRegex(ValueError, "same scene-style"):
                collect([a, b], root / "result")


if __name__ == "__main__":
    unittest.main()
