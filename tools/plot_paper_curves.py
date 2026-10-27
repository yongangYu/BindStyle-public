"""Plot measured A/B/C results; missing or OOM trials stay missing."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_paper_curves import collect, save_json

LABELS = {
    "simultaneous": "Simultaneous", "geometry_first": "Geometry-first",
    "appearance_first": "Appearance-first", "vgg_geometry": "VGG on SH and geometry",
    "no_motion": "w/o MotionAdapter", "ours": "Ours",
}
COLORS = ["#2878B5", "#D95319", "#309864"]


def summarize(rows):
    # Aggregate within a seed first. Updates/frames are not independent trials.
    trials = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("job_status") != "complete":
            continue
        try:
            value = float(row["value"])
        except (ValueError, TypeError):
            continue
        if not math.isfinite(value):
            continue
        group = row["group"]
        curve = row["resolution"] if group == "A" else row["schedule"] if group == "B" else row["variant"]
        x = float(row["window"] if group == "A" else row["progress"] if group == "B" else row["coverage"])
        key = group, row["metric"], str(curve), x, int(row["seed"])
        trials.setdefault(key, []).append(value)
    buckets = {}
    for (group, metric, curve, x, seed), values in trials.items():
        value = max(values) if metric.startswith("peak_") else statistics.mean(values)
        buckets.setdefault((group, metric, curve, x), []).append((seed, value))
    result = []
    for (group, metric, curve, x), trials in sorted(buckets.items()):
        values = [v for _, v in trials]
        result.append(dict(group=group, metric=metric, curve=curve, x=x,
                           mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else None,
                           n_seeds=len(values), seeds=sorted(seed for seed, _ in trials)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Sweep output containing plan.json")
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    collect(directory, plan)
    with (directory / "results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pair_sets = {row.get("pair_set_sha256") for row in rows if row["group"] == "C" and row["metric"] == "lpips"
                 and row["status"] == "ok" and row["job_status"] == "complete"}
    if len(pair_sets) > 1:
        raise ValueError("C runs used different valid temporal pairs. Resolve this before comparing coverage curves.")
    summary = summarize(rows)
    output = directory / "plots"
    output.mkdir(exist_ok=True)
    save_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "metric", "curve", "x", "mean", "std", "n_seeds", "seeds"])
        writer.writeheader()
        writer.writerows(summary)
    statuses = {row["id"]: row["job_status"] for row in rows}
    warnings = [f"{job}: {status}" for job, status in statuses.items() if status != "complete"]
    if any(row["n_seeds"] < 3 for row in summary):
        warnings.append("Fewer than three seeds at some points; single-seed points have no uncertainty band.")
    if plan["smoke"]:
        warnings.append("SMOKE TEST: reduced budgets; figures are not paper results.")
    save_json(output / "warnings.json", warnings)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
                         "axes.titlesize": 11, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

    def draw(ax, group, metric, curves, xs, xlabel, ylabel):
        any_data = False
        for color, curve in zip(COLORS, curves):
            measurements = {row["x"]: row for row in summary if row["group"] == group
                            and row["metric"] == metric and row["curve"] == str(curve)}
            # B samples record the actual matched-budget progress, not interpolation.
            x = sorted(measurements) if group == "B" else list(xs)
            if not x:
                continue
            y = np.array([measurements[v]["mean"] if v in measurements else np.nan for v in x])
            std = np.array([measurements[v]["std"] if v in measurements and measurements[v]["std"] is not None
                            else np.nan for v in x])
            label = f"{curve} x {curve}" if group == "A" else LABELS[curve]
            if np.isfinite(y).any():
                ax.plot(x, y, marker="o", markersize=4, linewidth=1.6, color=color, label=label)
                ax.fill_between(x, y - std, y + std, alpha=.12, color=color)
                any_data = True
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="both", alpha=.18)
        if group != "B":
            ax.set_xticks(xs)
        else:
            ax.set_xlim(0, 100)
        if any_data:
            ax.legend(frameon=False)
        else:
            ax.text(.5, .5, "No complete measurements", transform=ax.transAxes, ha="center", va="center")

    def save(fig, name):
        if plan["smoke"]:
            fig.suptitle("SMOKE TEST - NOT EXPERIMENTAL RESULTS", fontsize=10, color="#AA2222")
        fig.tight_layout()
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    present = {job["group"] for job in plan["jobs"]}
    cfg = plan["config"]["curves"]
    if "A" in present:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
        for ax, metric, label in zip(axes, ("seconds_per_update", "peak_allocated_gb"),
                                     ("Time per Update (s)", "Peak GPU Memory (GB)")):
            draw(ax, "A", metric, cfg["A"]["resolutions"], cfg["A"]["windows"], "Frames per Window", label)
        save(fig, "A_temporal_window_scaling")
    if "B" in present:
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        draw(ax, "B", "cm", ("simultaneous", "geometry_first", "appearance_first"), [],
             "Optimization Progress (%)", "CM Loss (lower is better)")
        save(fig, "B_optimization_order")
    if "C" in present:
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        draw(ax, "C", "lpips", ("vgg_geometry", "no_motion", "ours"), cfg["C"]["coverage"],
             "Temporal Coverage (%)", "Held-out LPIPS (lower is better)")
        save(fig, "C_temporal_coverage")
    print(f"Figures and measured summaries: {output}")
    for warning in warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
