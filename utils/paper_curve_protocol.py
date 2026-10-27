"""CPU-only schedules and manifests for the A/B/C curve experiments."""

import hashlib
import json
import random


PROTOCOL = "mags-curves-v1"
VARIANTS = ("vgg_geometry", "no_motion", "ours")
SCHEDULES = ("simultaneous", "geometry_first", "appearance_first")


def schedule(name, sh_updates, geometry_updates, warmup):
    if name not in SCHEDULES or min(sh_updates, geometry_updates) < 1:
        raise ValueError("Invalid optimization schedule or update budget")
    if not 0 <= warmup <= min(sh_updates, geometry_updates):
        raise ValueError("Schedule warmup exceeds a parameter budget")
    phases = []
    sh, geo = sh_updates, geometry_updates
    if name == "appearance_first" and warmup:
        phases.append((warmup, True, False))
        sh -= warmup
    elif name == "geometry_first" and warmup:
        phases.append((warmup, False, True))
        geo -= warmup
    joint = min(sh, geo)
    if joint:
        phases.append((joint, True, True))
    if sh > joint:
        phases.append((sh - joint, True, False))
    if geo > joint:
        phases.append((geo - joint, False, True))
    return [dict(sh=a, geometry=g) for count, a, g in phases for _ in range(count)]


def heldout_indices(frame_count, ranges):
    result = set()
    for start, stop in ranges:
        if not 0 <= start < stop <= frame_count:
            raise ValueError("Heldout ranges must be valid half-open frame intervals")
        if result.intersection(range(start, stop)):
            raise ValueError("Heldout ranges overlap")
        result.update(range(start, stop))
    return sorted(result)


def coverage_windows(frame_count, window_size, coverage, ranges, seed):
    heldout = set(heldout_indices(frame_count, ranges))
    count = frame_count * float(coverage) / 100 / window_size
    if abs(count - round(count)) > 1e-8 or count < 1:
        raise ValueError("Coverage must represent an integer number of disjoint windows")
    candidates = [list(range(i, i + window_size)) for i in range(0, frame_count - window_size + 1, window_size)
                  if not heldout.intersection(range(i, i + window_size))]
    random.Random(seed).shuffle(candidates)
    if round(count) > len(candidates):
        raise ValueError("Not enough non-heldout windows for requested coverage")
    return candidates[:round(count)]


def heldout_pairs(frame_count, ranges, gap=2):
    if gap < 1:
        raise ValueError("Pair gap must be positive")
    selected = set(heldout_indices(frame_count, ranges))
    return [(i, i + gap) for i in sorted(selected) if i + gap in selected
            and all(j in selected for j in range(i, i + gap + 1))]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def make_jobs(cfg, groups, seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    n = int(cfg["frame_count"])
    f = int(cfg["window_size"])
    if n < f or f < 2:
        raise ValueError("Invalid frame/window count")
    if int(cfg["resolution"]) < 128 or int(cfg["resolution"]) % 8:
        raise ValueError("Resolution must be >=128 and divisible by 8")
    if not 0 < float(cfg["evaluate_every_percent"]) <= 100:
        raise ValueError("Evaluation interval must be in (0,100]")
    if min(int(cfg["checkpoint_every_updates"]), int(cfg["probe_frames"])) < 1:
        raise ValueError("Checkpoint interval and probe count must be positive")
    if int(cfg["A"]["warmup_updates"]) < 1 or int(cfg["A"]["measured_updates"]) < 1:
        raise ValueError("A needs at least one warmup and one measured update")
    if "C" in groups and not heldout_pairs(n, cfg["C"]["heldout_ranges"], int(cfg["C"]["pair_gap"])):
        raise ValueError("No held-out temporal pairs")
    jobs = []
    for group in groups:
        if group not in {"A", "B", "C"}:
            raise ValueError("Unknown curve group")
        for seed in seeds:
            common = dict(group=group, seed=int(seed), frame_count=n, window=f,
                          resolution=int(cfg["resolution"]), variant="ours", coverage=100,
                          schedule="appearance_first")
            if group == "A":
                for resolution in cfg["A"]["resolutions"]:
                    if resolution < 128 or resolution % 8:
                        raise ValueError("Benchmark resolution must be >=128 and divisible by 8")
                    for window in cfg["A"]["windows"]:
                        if not 2 <= window <= n:
                            raise ValueError("Benchmark window outside sequence")
                        jobs.append(dict(common, resolution=resolution, window=window, schedule="joint_benchmark"))
            elif group == "B":
                for order in SCHEDULES:
                    schedule(order, cfg["sh_updates"], cfg["geometry_updates"], cfg["schedule_warmup"])
                    jobs.append(dict(common, schedule=order))
            else:
                for coverage in cfg["C"]["coverage"]:
                    windows = coverage_windows(n, f, coverage, cfg["C"]["heldout_ranges"], seed)
                    if len(windows) > len(schedule("appearance_first", cfg["sh_updates"], cfg["geometry_updates"], cfg["schedule_warmup"])):
                        raise ValueError("Budget cannot visit every coverage window")
                    for variant in VARIANTS:
                        jobs.append(dict(common, variant=variant, coverage=coverage, windows=windows))
    for job in jobs:
        job["id"] = (f"{job['group']}-{job['variant']}-{job['schedule']}-f{job['window']}"
                     f"-r{job['resolution']}-c{job['coverage']}-seed{job['seed']}")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate experiment settings would overwrite a job")
    return jobs
