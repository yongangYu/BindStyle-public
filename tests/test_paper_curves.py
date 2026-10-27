import csv
from pathlib import Path
import tempfile
import unittest

from utils.paper_curve_protocol import coverage_windows, heldout_indices, heldout_pairs, make_jobs, schedule
from tools.plot_paper_curves import summarize
from tools.run_paper_curves import collect, save_json


def config():
    return dict(frame_count=200, window_size=2, resolution=256, sh_updates=2500, geometry_updates=2000,
                schedule_warmup=500, evaluate_every_percent=5, checkpoint_every_updates=500, probe_frames=20,
                A=dict(windows=[2, 4, 8, 12, 16], resolutions=[256, 384, 512], warmup_updates=10, measured_updates=50),
                C=dict(coverage=[1, 5, 10, 20, 40], heldout_ranges=[[80, 100], [160, 180]], pair_gap=2))


class CurveProtocolTests(unittest.TestCase):
    def test_exact_parameter_budgets(self):
        for name, length in (("simultaneous", 2500), ("geometry_first", 3000), ("appearance_first", 2500)):
            steps = schedule(name, 2500, 2000, 500)
            self.assertEqual(len(steps), length)
            self.assertEqual(sum(p["sh"] for p in steps), 2500)
            self.assertEqual(sum(p["geometry"] for p in steps), 2000)
            self.assertTrue(all(p["sh"] or p["geometry"] for p in steps))
        self.assertEqual(schedule("geometry_first", 2500, 2000, 500)[0], dict(sh=False, geometry=True))

    def test_nested_coverage_and_no_leakage(self):
        previous = set()
        ranges = config()["C"]["heldout_ranges"]
        heldout = set(heldout_indices(200, ranges))
        for coverage in (1, 5, 10, 20, 40):
            windows = coverage_windows(200, 2, coverage, ranges, 1234)
            selected = {i for window in windows for i in window}
            self.assertEqual(len(selected), coverage * 2)
            self.assertTrue(previous.issubset(selected))
            self.assertFalse(selected & heldout)
            self.assertTrue(all(b == a + 1 for a, b in windows))
            previous = selected
        self.assertEqual(len(heldout_pairs(200, ranges)), 36)

    def test_invalid_configs_fail(self):
        with self.assertRaises(ValueError):
            coverage_windows(199, 2, 1, [[80, 100]], 1)
        with self.assertRaises(ValueError):
            heldout_indices(200, [[80, 100], [90, 110]])
        with self.assertRaises(ValueError):
            schedule("unknown", 5, 4, 1)
        with self.assertRaises(ValueError):
            schedule("appearance_first", 5, 4, 6)
        self.assertEqual(heldout_pairs(10, [[0, 2], [3, 5]], 2), [])
        duplicate = config()
        duplicate["A"]["windows"] = [2, 2]
        with self.assertRaises(ValueError):
            make_jobs(duplicate, ["A"], [1234])

    def test_full_plan_and_identical_variant_frames(self):
        jobs = make_jobs(config(), ["A", "B", "C"], [1234, 2345, 3456])
        self.assertEqual(len(jobs), 99)
        self.assertEqual(len({job["id"] for job in jobs}), 99)
        chosen = [j for j in jobs if j["group"] == "C" and j["seed"] == 1234 and j["coverage"] == 10]
        self.assertEqual(len(chosen), 3)
        self.assertTrue(all(j["windows"] == chosen[0]["windows"] for j in chosen))

    def test_missing_jobs_are_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            jobs = make_jobs(config(), ["B"], [1234])
            folder = output / jobs[0]["id"]
            folder.mkdir()
            save_json(folder / "status.json", {"status": "oom", "error": "test OOM"})
            collect(output, {"jobs": jobs})
            with (output / "results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["value"] == "" for row in rows))
            self.assertEqual(summarize(rows), [])

    def test_seed_not_update_is_replication_unit(self):
        rows = []
        for seed, values in ((1, [1., 3.]), (2, [10.])):
            for v in values:
                rows.append(dict(group="A", resolution="256", window="2", seed=seed, status="ok",
                                 job_status="complete", metric="seconds_per_update", value=v))
        rows.append({**rows[-1], "value": 1e6, "job_status": "oom"})
        summary = summarize(rows)
        self.assertEqual(summary[0]["mean"], 6.)
        self.assertEqual(summary[0]["n_seeds"], 2)
        one = summarize(rows[:2])[0]
        self.assertIsNone(one["std"])


if __name__ == "__main__":
    unittest.main()
