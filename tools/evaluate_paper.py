"""Recompute paper metrics from a final-render bundle without retraining."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paper_eval_protocol import (aggregate, bilinear_sample, erode_mask, flow_correspondence,
                                      histogram_distance, masked_rmse, mesh_correspondence,
                                      record, rgbuv_histogram, write_csv, write_json)


def attempt(function):
    try:
        return record(function())
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        return record(reason=f"{type(exc).__name__}: {exc}")


def evaluate_bundle(directory, overrides=None):
    from utils.paper_eval_metrics import PaperModels, resize_histogram_input, ssim_content
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cfg = {**manifest["settings"], **(overrides or {})}
    write_json(directory / "status.json", {"status": "evaluating"})
    models = PaperModels(cfg)
    samples = {sample["id"]: sample for sample in manifest["samples"]}
    rows, pair_rows = [], []
    buckets = {}

    def put(bucket, metric, measurement, **identity):
        buckets.setdefault(bucket, {}).setdefault(metric, []).append(measurement)
        rows.append({"group": bucket, "metric": metric, **identity, **measurement})

    style = np.array(Image.open(directory / "style.png").convert("RGB"), dtype=np.float32) / 255
    hist_options = cfg.get("histogram", {})
    def histogram(image):
        return rgbuv_histogram(resize_histogram_input(image, int(hist_options.get("size", 150))),
                               int(hist_options.get("bins", 64)), float(hist_options.get("sigma", 0.02)))
    style_histogram = histogram(style)
    for sample in manifest["samples"]:
        if not sample["id"].startswith("time_"):
            continue
        with np.load(directory / (sample["id"] + ".npz"), allow_pickle=False) as data:
            source, styled = data["source"], data["styled"]
        for region in ("full", "foreground"):
            a, b = source, styled
            if region == "foreground":
                if sample["crop"] is None:
                    for metric in ("cm", "lpips_gray", "content_lpips", "artscore", "content_psnr", "content_ssim", "cfsd"):
                        put(region, metric, record(reason="Empty source foreground"), sample=sample["id"])
                    continue
                y0, y1, x0, x1 = sample["crop"]
                a, b = source[y0:y1, x0:x1], styled[y0:y1, x0:x1]
            gray_a = np.repeat((a @ np.array([0.299, 0.587, 0.114], np.float32))[..., None], 3, -1)
            gray_b = np.repeat((b @ np.array([0.299, 0.587, 0.114], np.float32))[..., None], 3, -1)
            mse = float(np.square(a.astype(np.float64) - b).mean())
            metrics = {"cm": attempt(lambda: histogram_distance(histogram(b), style_histogram)),
                       "lpips_gray": attempt(lambda: models.lpips(gray_a, gray_b)),
                       "content_lpips": attempt(lambda: models.lpips(a, b)),
                       "artscore": attempt(lambda: models.artscore(b)),
                       "content_ssim": attempt(lambda: ssim_content(a, b)),
                       "content_psnr": record(-10 * np.log10(mse)) if mse > 0 else record(status="perfect", reason="PSNR=+inf for identical images"),
                       "cfsd": (attempt(lambda: models.cfsd(a, b)) if cfg.get("cfsd", {}).get("enabled", False)
                                else record(status="disabled", reason="Optional CFSD diagnostic disabled"))}
            for name, measurement in metrics.items():
                put(region, name, measurement, sample=sample["id"])
        if sample["frame_index"] % 20 == 0:
            print(f"[Paper Eval] Image metrics frame {sample['frame_index']}")

    for pair_type, key in (("view", "view_pairs"), ("temporal", "temporal_pairs")):
        for pair_index, pair in enumerate(manifest[key]):
            data = []
            for name in (pair["a"], pair["b"]):
                with np.load(directory / f"{name}.npz", allow_pickle=False) as archive:
                    data.append(dict(archive))
            error, flows = None, None
            if pair_type == "temporal":
                try:
                    flows = (models.flow(data[0]["source"], data[1]["source"]),
                             models.flow(data[1]["source"], data[0]["source"]))
                except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            directions = []
            for reverse in (False, True):
                i, j = (1, 0) if reverse else (0, 1)
                a, b = data[i], data[j]
                direction = {"type": pair_type, "group": pair["group"], "a": pair["a"], "b": pair["b"],
                             "reverse": reverse, "angle_degrees": pair.get("angle_degrees"),
                             "delta_tau": pair.get("delta_tau"), "valid_pixels": 0, "coverage": 0}
                try:
                    if error:
                        raise RuntimeError(error)
                    if pair_type == "view":
                        xy, valid = mesh_correspondence(a, b, samples[pair["a"] if reverse else pair["b"]]["full_proj_transform"],
                                                        float(cfg.get("depth_relative_tolerance", 0.01)),
                                                        float(cfg.get("depth_absolute_tolerance", 0.0001)))
                    else:
                        xy, valid = flow_correspondence(flows[i], flows[j], a["mask"], b["mask"],
                                                        float(cfg.get("fb_alpha", 0.01)), float(cfg.get("fb_beta", 0.5)))
                    valid = erode_mask(valid, int(cfg.get("mask_erosion", 3)))
                    coverage = float(valid.sum() / max(1, a["mask"].sum()))
                    direction.update(valid_pixels=int(valid.sum()), coverage=coverage)
                    if valid.sum() == 0 or coverage < float(cfg.get("minimum_coverage", 0.05)):
                        raise ValueError("Insufficient valid source-foreground coverage")
                    warped = bilinear_sample(b["styled"], xy).clip(0, 1)
                    direction["rmse"] = masked_rmse(a["styled"], warped, valid)
                    direction["lpips"] = attempt(lambda: models.lpips(a["styled"], warped, valid))
                    direction["source_rmse"] = masked_rmse(a["source"], bilinear_sample(b["source"], xy).clip(0, 1), valid)
                    maps = directory / "correspondence"
                    maps.mkdir(exist_ok=True)
                    np.savez_compressed(maps / f"{pair_type}_{pair_index:05d}_{int(reverse)}.npz", xy=xy, mask=valid)
                except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
                    direction.update({metric: record(reason=str(exc)) for metric in ("lpips", "rmse", "source_rmse")})
                directions.append(direction)
                for metric in ("lpips", "rmse", "source_rmse"):
                    pair_rows.append({**{k: v for k, v in direction.items() if k not in {"lpips", "rmse", "source_rmse"}},
                                      "metric": metric, **direction[metric]})
            for metric in ("lpips", "rmse", "source_rmse"):
                measurements = [d[metric] for d in directions]
                # Do not average a single valid direction as a bidirectional pair.
                value = aggregate(measurements) if all(r["status"] == "ok" for r in measurements) else record(reason="One or both directions unavailable")
                put(f"{pair_type}_{pair['group']}", metric, value, sample=f"{pair['a']}:{pair['b']}")
            if pair_index % 20 == 0:
                print(f"[Paper Eval] {pair_type} pair {pair_index + 1}/{len(manifest[key])}")

    summary = {"protocol": manifest["protocol"], "scope": manifest["scope"], "scene": manifest["scene"],
               "style_sha256": manifest["style_sha256"], "method": cfg.get("method", "Ours"),
               "variant": cfg.get("variant", "full"), "metrics": {},
               "weights": models.weights, "model_errors": models.errors, "settings": cfg,
               "artfid": record(reason="Single-run data. Pool balanced scene-style bundles with collect_paper_results.py"),
               "user_study": {name: record(status="requires_human_study", reason="Five-point blinded ratings required")
                              for name in ("style", "geometry", "temporal")},
               "coverage": record(reason=manifest["coverage_status"]),
               "optimizer_steps": manifest["optimizer_steps"], "versions": {}}
    for name in ("torch", "torchvision", "lpips", "numpy", "scikit-image"):
        try:
            summary["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            summary["versions"][name] = None
    for bucket in ("full", "foreground", "view_narrow", "view_wide", "temporal_short", "temporal_long"):
        metrics = buckets.get(bucket, {})
        summary["metrics"][bucket] = {
            name: (record(status="disabled", reason="Optional diagnostic disabled")
                   if all(item["status"] == "disabled" for item in items) else aggregate(items))
            for name, items in metrics.items()}
        if bucket.startswith(("view_", "temporal_")) and not metrics:
            summary["metrics"][bucket] = {name: record(reason="No eligible pairs in this run") for name in ("lpips", "rmse")}
    summary["geometry"] = {name: aggregate([record(row[name]) for row in manifest["geometry"]])
                           for name in ("delta_pos", "delta_pos_p50", "delta_pos_p95", "delta_scale")}
    summary["geometry"]["percentile_aggregation"] = "mean of per-frame primitive percentiles"
    write_json(directory / "summary.json", summary)
    write_csv(directory / "per_image_and_pair_metrics.csv", rows, ["group", "sample", "metric", "value", "status", "reason", "count", "excluded"])
    write_csv(directory / "pair_directions.csv", pair_rows,
              ["type", "group", "a", "b", "reverse", "angle_degrees", "delta_tau", "coverage", "valid_pixels", "metric", "value", "status", "reason"])
    flat = [{"group": group, "metric": metric, **item} for group, metrics in summary["metrics"].items() for metric, item in metrics.items()]
    flat.extend({"group": "geometry", "metric": metric, **item} for metric, item in summary["geometry"].items() if isinstance(item, dict))
    incomplete = bool(models.errors) or any(item["status"] not in {"ok", "disabled", "perfect"} for item in flat)
    flat.append({"group": "collection", "metric": "artfid", **summary["artfid"]})
    flat.append({"group": "coverage", "metric": "heldout_protocol", **summary["coverage"]})
    flat.extend({"group": "user_study", "metric": name, **item} for name, item in summary["user_study"].items())
    write_csv(directory / "summary.csv", flat, ["group", "metric", "value", "status", "reason", "count", "excluded"])
    if not (directory / "user_study_template.csv").exists():
        write_csv(directory / "user_study_template.csv", [], ["participant", "scene", "style_sha256", "method", "style", "geometry", "temporal"])
    write_json(directory / "status.json", {"status": "partial" if incomplete else "computed", "summary": "summary.json",
                                           "note": "ArtFID collection and human study remain separate"})
    print(f"[Paper Eval] Metrics saved: {directory / 'summary.csv'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--artscore-weights")
    parser.add_argument("--alexnet-weights")
    parser.add_argument("--raft-weights")
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    options = manifest["settings"]
    if args.device:
        options["device"] = args.device
    for key, value, field in (("artscore", args.artscore_weights, "weights"), ("lpips", args.alexnet_weights, "alexnet_weights"),
                              ("raft", args.raft_weights, "weights")):
        if value:
            options[key] = {**options.get(key, {}), field: value}
    evaluate_bundle(args.bundle, options)


if __name__ == "__main__":
    main()
