"""Aggregate matched evaluation runs and compute pooled ArtFID offline."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.paper_eval_protocol import aggregate, fingerprint, record, write_csv, write_json


def collect(bundles, output, art_inception_weights="", device="cpu", min_styles=20, images_per_run=20):
    if min_styles < 2 or images_per_run < 2:
        raise ValueError("ArtFID collection needs >=2 distinct styles and >=2 images per run")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = {}
    source_signatures = {}
    metric_settings = None
    seen = set()
    for path in bundles:
        path = Path(path).resolve()
        if path in seen:
            raise ValueError(f"Duplicate evaluation bundle: {path}")
        seen.add(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if manifest["scope"] != "full_sequence":
            raise ValueError(f"Refusing to mix pilot results into paper collection: {path}")
        scene = manifest["scene"]
        source_signature = (manifest["source_checkpoint_sha256"], manifest["mesh_sequence_sha256"],
                            [(s["id"], s["time"], s["full_proj_transform"], s["crop"]) for s in manifest["samples"]],
                            manifest["view_pairs"], manifest["temporal_pairs"])
        if scene in source_signatures and source_signatures[scene] != source_signature:
            raise ValueError(f"Unmatched source geometry, cameras, crops or pairs for {scene}")
        source_signatures[scene] = source_signature
        settings = summary["settings"]
        signature = {k: settings.get(k) for k in (
            "image_size", "short_gap", "long_gap", "mask_erosion", "minimum_coverage", "fb_alpha", "fb_beta",
            "depth_relative_tolerance", "depth_absolute_tolerance", "histogram", "cfsd", "crop_padding")}
        signature["raft_updates"] = settings.get("raft", {}).get("updates", 12)
        signature["versions"] = summary.get("versions", {})
        signature["weights"] = {k: (v.get("sha256") if isinstance(v, dict) else v) for k, v in summary["weights"].items()}
        if metric_settings is not None and metric_settings != signature:
            raise ValueError("Evaluator settings/weights differ across runs")
        metric_settings = signature
        method = f"{summary['method']} / {summary['variant']}"
        methods.setdefault(method, {}).setdefault((scene, manifest["style_sha256"]), []).append((path, manifest, summary))
    if not methods:
        raise ValueError("No evaluation bundles")
    combinations = set(next(iter(methods.values())))
    if any(set(group) != combinations for group in methods.values()):
        raise ValueError("All methods must have exactly the same scene-style combinations")
    rows, artfid_results = [], {}
    for method, group in methods.items():
        combination_metrics = {}
        for runs in group.values():
            values = {}
            for _, _, summary in runs:
                for category, metrics in {**summary["metrics"], "geometry": summary["geometry"]}.items():
                    for metric, item in metrics.items():
                        if isinstance(item, dict):
                            values.setdefault((category, metric), []).append(item)
            for key, items in values.items():
                result = aggregate(items) if all(x["status"] == "ok" for x in items) else record(reason="Incomplete seed measurements")
                combination_metrics.setdefault(key, []).append(result)
        for (category, metric), items in combination_metrics.items():
            result = aggregate(items) if all(x["status"] == "ok" for x in items) else record(reason="Incomplete scene-style measurements")
            rows.append({"method": method, "group": category, "metric": metric, **result})
        distinct_styles = len({key[1] for key in group})
        result = record(reason=f"Need >= {min_styles} distinct styles and local Art-Inception weights")
        if distinct_styles >= min_styles and art_inception_weights:
            try:
                result = pooled_artfid(group, art_inception_weights, device, images_per_run, output)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                result = record(reason=f"{type(exc).__name__}: {exc}")
        artfid_results[method] = {**result, "distinct_styles": distinct_styles,
                                 "scene_style_combinations": len(group), "minimum_styles_policy": min_styles}
        rows.append({"method": method, "group": "collection", "metric": "artfid", **result})
    write_csv(output / "paper_results.csv", rows, ["method", "group", "metric", "value", "status", "reason", "count", "excluded"])
    write_json(output / "collection.json", {"bundles": [str(Path(p).resolve()) for p in bundles],
                                           "aggregation": "equal scene-style means after equal seed means",
                                           "settings": metric_settings, "artfid": artfid_results,
                                           "uncertainty": "Not estimated; do not interpret frames as independent replicates"})
    return rows


def pooled_artfid(group, weights, device, count, output):
    import torch
    from art_fid import art_fid, inception
    from utils.paper_eval_metrics import PaperModels

    if not Path(weights).is_file():
        raise FileNotFoundError(weights)
    seed_counts = {len(runs) for runs in group.values()}
    if len(seed_counts) != 1:
        raise ValueError("Pooled ArtFID requires equal seed counts per scene-style combination")
    lengths = [sum(s["id"].startswith("time_") for s in manifest["samples"])
               for runs in group.values() for _, manifest, _ in runs]
    if min(lengths) < count:
        raise ValueError("Not enough frames for the declared ArtFID sampling count")
    settings = next(iter(group.values()))[0][2]["settings"]
    models = PaperModels({**settings, "device": device})
    with tempfile.TemporaryDirectory(prefix="artfid-", dir=output) as tmp:
        tmp = Path(tmp)
        for name in ("styled", "source", "style"):
            (tmp / name).mkdir()
        content_values, selected = [], []
        for key in sorted(group):
            for path, manifest, _ in group[key]:
                samples = [s for s in manifest["samples"] if s["id"].startswith("time_")]
                reference = Image.open(path / "style.png").convert("RGB")
                for index in np.linspace(0, len(samples) - 1, count).round().astype(int):
                    sample = samples[index]
                    if sample["crop"] is None:
                        raise ValueError("Empty crop in ArtFID manifest; cannot silently drop it")
                    y0, y1, x0, x1 = sample["crop"]
                    with np.load(path / (sample["id"] + ".npz"), allow_pickle=False) as data:
                        source = data["source"][y0:y1, x0:x1]
                        styled = data["styled"][y0:y1, x0:x1]
                    name = f"{len(selected):07d}.png"
                    for label, image in (("source", source), ("styled", styled)):
                        Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(tmp / label / name)
                    reference.save(tmp / "style" / name)
                    # Official ArtFID content distance is AlexNet LPIPS on paired
                    # images; use its preprocessing below, not the warp metric.
                    selected.append({"bundle": str(path), "sample": sample["id"], "name": name})
        # Reuse the authors' artwork-trained Inception and FID implementation.
        model = inception.Inception3().to(device).eval().requires_grad_(False)
        state = torch.load(weights, map_location=device, weights_only=True)
        # The official evaluator consumes features, not either classifier head.
        # Accept omitted heads but never an incomplete feature backbone.
        heads = ("fc1.", "fc2.", "AuxLogits1.", "AuxLogits2.")
        state = {key: value for key, value in state.items() if not key.startswith(heads)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith(heads) for key in missing):
            raise ValueError(f"Incomplete Art-Inception backbone: missing={missing}, unexpected={unexpected}")
        a = art_fid.get_image_paths(str(tmp / "styled"), sort=True)
        b = art_fid.get_image_paths(str(tmp / "style"), sort=True)
        mu0, cov0 = art_fid.compute_activation_statistics(a, model, batch_size=1, device=device, num_workers=0)
        mu1, cov1 = art_fid.compute_activation_statistics(b, model, batch_size=1, device=device, num_workers=0)
        fid = float(art_fid.compute_frechet_distance(mu0, cov0, mu1, cov1))
        # Official LPIPS content preprocessing is ToTensor without resizing.
        from torchvision import transforms
        transform = transforms.ToTensor()
        for entry in selected:
            images = []
            for label in ("source", "styled"):
                tensor = transform(Image.open(tmp / label / entry["name"]).convert("RGB"))
                images.append(tensor.permute(1, 2, 0).numpy())
            content_values.append(models.lpips(*images))
        content = float(np.mean(content_values))
    result = record((1 + fid) * (1 + content), fid=fid, content_lpips=content, samples=len(selected),
                    mode="art_fid", feature_weights_sha256=fingerprint(weights), sampling_manifest=selected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--art-inception-weights", default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--min-styles", type=int, default=20)
    parser.add_argument("--images-per-run", type=int, default=20)
    args = parser.parse_args()
    collect(args.bundles, args.output, args.art_inception_weights, args.device, args.min_styles, args.images_per_run)


if __name__ == "__main__":
    main()
