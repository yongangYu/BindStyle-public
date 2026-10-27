"""One isolated A/B/C experiment. Launch via run_paper_curves.py."""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_paper_curves import save_json
from utils.paper_curve_protocol import digest, heldout_indices, heldout_pairs, schedule


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def run(folder, payload):
    import numpy as np
    from PIL import Image
    import torch
    from omegaconf import OmegaConf
    # Resolve the existing scene/camera import cycle before importing dataset.
    from scene import Scene  # noqa: F401
    from dataset import get_dataset
    from gaussian_renderer import inria_render
    from utils.paper_eval_export import _camera, export_paper_bundle, load_base_gaussians
    from utils.paper_eval_protocol import fingerprint, foreground_box, histogram_distance, rgbuv_histogram
    from utils.paper_eval_metrics import PaperModels, resize_histogram_input
    from utils.sds_guidance import build_sds_guidance
    from utils.split_style_training import routed_style_gradients
    from utils.style_numerics import require_finite
    from utils.vgg_appearance_loss import VGGAppearanceLoss

    if not torch.cuda.is_available():
        raise RuntimeError("The curve worker requires CUDA and the MaGS CUDA extensions")
    job = payload["job"]
    cfg = OmegaConf.create(payload["config"])
    curves, style = cfg.curves, cfg.optim.gaussian_style_update
    if style.parameterization != "direct" or style.use_deformer:
        raise ValueError("Curves require direct Gaussian parameters without deformer forward")
    if cfg.optim.mesh_style_update.enabled or style.supervision_mode != "split_vgg_diffusion":
        raise ValueError("Curves require known GT meshes and split_vgg_diffusion supervision")
    if style.sds.provider != "sdxl_animatediff_controlnet":
        raise ValueError("This curve protocol requires the video ControlNet guidance provider")
    checkpoint = Path(style.start_checkpoint).resolve()
    style_path = Path(style.style_path).resolve()
    if not checkpoint.is_file() or not style_path.is_file():
        raise FileNotFoundError("Original checkpoint or reference style image missing")
    seed = job["seed"]

    def seed_all():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_all()
    torch.backends.cudnn.benchmark = False
    cfg.dataset.frame_limit = 0
    cfg.model.model_path = str(folder)
    cfg.optim.random_bg = False
    cfg.optim.disable_deformer = True
    resolution, window_size = job["resolution"], job["window"]
    style.appearance.image_size = resolution
    style.sds.style_path = str(style_path)
    style.sds.image_size = resolution
    style.sds.video_window_size = window_size
    style.sds.temporal_mode = "framewise" if job["variant"] == "no_motion" else "motion"
    if job["variant"] != "vgg_geometry" and not style.sds.enabled:
        raise ValueError("The diffusion variants require sds.enabled=true")
    if job["group"] == "C":
        # Fail before an expensive run when the two required metric models are missing.
        probe = PaperModels({**OmegaConf.to_container(cfg.evaluation), "device": "cpu"})
        probe.get("lpips")
        probe.get("raft")
        del probe
        gc.collect()

    frames = sorted(get_dataset(cfg, train=True), key=lambda cam: float(cam.fid))
    if len(frames) != job["frame_count"]:
        raise ValueError(f"Expected {job['frame_count']} training times; found {len(frames)}")
    if len({int(c.uid) for c in frames}) != len(frames):
        raise ValueError("Duplicate camera UIDs")
    if any(float(b.fid) <= float(a.fid) for a, b in zip(frames, frames[1:])):
        raise ValueError("Training times must be strictly increasing")
    faces = frames[0].mesh_faces.cuda().long()
    if not all(torch.equal(c.mesh_faces.to(faces), faces) for c in frames):
        raise ValueError("All frames must share the same mesh topology and vertex indexing")
    center = (frames[0].mesh_vertices.amin(0) + frames[0].mesh_vertices.amax(0)) / 2
    cameras = [_camera(c, c, resolution, 0, center.to(c.world_view_transform.device)) for c in frames]
    if any((c.image_height, c.image_width) != (resolution, resolution) for c in cameras):
        raise ValueError("This R x R benchmark requires square source cameras; do not distort intrinsics")
    model = load_base_gaussians(cfg, str(checkpoint), "cuda")
    if model.faceid.min() < 0 or model.faceid.max() >= len(faces):
        raise ValueError("Checkpoint face bindings do not match this dataset")
    frozen = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
              if name not in {"_features_dc", "_features_rest", "_uvw", "_scaling", "_rotation"}}
    frozen_faces = model.faceid.detach().cpu().clone()
    background = torch.tensor([1., 1., 1.] if cfg.dataset.white_background else [0., 0., 0.], device="cuda")

    def render(cam):
        model.update_mesh(faces, cam.mesh_vertices.cuda(), cam.mesh_normals.cuda(), deformer=None)
        return require_finite(inria_render(cam, model, cfg.pipe, background)["render"], "curve Gaussian RGB")

    sh_params = [model._features_dc, model._features_rest]
    geo_params = [model._uvw, model._scaling, model._rotation]
    geo_cfg = style.direct_geometry
    for value in (style.lr_dc, style.lr_rest, geo_cfg.lr_uvw, geo_cfg.lr_scaling, geo_cfg.lr_rotation,
                  style.appearance.weight, style.appearance.grad_clip, geo_cfg.grad_clip, style.sds_weight):
        if float(value) <= 0:
            raise ValueError("Curve learning rates, branch weights, and gradient clips must be positive")
    sh_optimizer = torch.optim.Adam([
        {"name": "f_dc", "params": sh_params[:1], "lr": float(style.lr_dc)},
        {"name": "f_rest", "params": sh_params[1:], "lr": float(style.lr_rest)}], eps=1e-15)
    geo_optimizer = torch.optim.Adam([
        {"name": name, "params": [param], "lr": float(geo_cfg[key])}
        for name, param, key in zip(("uvw", "scaling", "rotation"), geo_params,
                                    ("lr_uvw", "lr_scaling", "lr_rotation"))], eps=1e-15)
    if job["group"] == "A":
        steps = [dict(sh=True, geometry=True)] * (int(curves.A.warmup_updates) + int(curves.A.measured_updates))
    else:
        steps = schedule(job["schedule"], int(curves.sh_updates), int(curves.geometry_updates), int(curves.schedule_warmup))
    windows = job.get("windows") or [list(range(i, i + window_size)) for i in range(len(frames) - window_size + 1)]
    used = sorted({i for window in windows for i in window})
    if job["group"] == "C":
        heldout = heldout_indices(len(frames), curves.C.heldout_ranges)
        if set(used).intersection(heldout):
            raise ValueError("Style optimization overlaps held-out times")
    appearance = VGGAppearanceLoss(style.appearance, str(style_path), frames, "cuda")
    teacher = None
    if job["variant"] != "vgg_geometry":
        # Conditions are frozen original renders, never GT RGB from held-out times.
        with torch.no_grad():
            source_images = {int(frames[i].uid): render(cameras[i]).detach().cpu() for i in used}
        teacher = build_sds_guidance(style, "cuda")
        teacher.cache_source_images(source_images)
        del source_images
        # Known full-sequence geometry fixes the depth normalization across C coverages.
        teacher.cache_control_frames(cameras)
        teacher.park_diffusion_backbone()
    save_json(folder / "effective_config.json", OmegaConf.to_container(cfg, resolve=True))
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "numpy"):
        versions[name] = importlib.metadata.version(name)
    provenance = {
        "job": job, "smoke": payload["smoke"], "plan_signature": payload["plan_signature"],
        "checkpoint_sha256": fingerprint(checkpoint), "style_sha256": fingerprint(style_path),
        "gpu": torch.cuda.get_device_name(), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "versions": versions, "training_windows": windows, "frame_uids": [int(c.uid) for c in frames],
        "teacher_unet_class": type(teacher.pipe.unet).__name__ if teacher is not None else None,
        "torch_cuda": torch.version.cuda, "python": sys.version,
        "frame_times": [float(c.fid) for c in frames], "train_camera_mode": "original_per_frame",
        "deformer": "off", "trainable": ["_features_dc", "_features_rest", "_uvw", "_scaling", "_rotation"],
        "appearance_crop": appearance.crop, "time_unit": "seconds_per_whole_window_update",
        "memory_unit": "decimal_GB", "offload_teacher": bool(style.appearance.offload_teacher),
        "nvdiffrast_role": "fixed_GT_depth_and_evaluation_masks_only",
        "heldout_scope": "style optimization only; pretraining and GT motion may include all times",
        "code_sha256": {path: fingerprint(ROOT / path) for path in (
            "tools/train_paper_curve.py", "utils/split_style_training.py", "utils/paper_curve_protocol.py",
            "utils/sdxl_animatediff_controlnet_csd_guidance.py", "utils/paper_eval_export.py")}}
    sequence_hash = hashlib.sha256(faces.cpu().numpy().tobytes())
    for frame in frames:
        for tensor in (frame.mesh_vertices, frame.mesh_normals, frame.full_proj_transform):
            sequence_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    provenance["geometry_camera_sha256"] = sequence_hash.hexdigest()
    save_json(folder / "manifest.json", provenance)

    probe_cameras, probe_boxes = [], []
    hist_cfg = cfg.evaluation.get("histogram", {})

    def histogram(image):
        return rgbuv_histogram(resize_histogram_input(image, int(hist_cfg.get("size", 150))),
                               int(hist_cfg.get("bins", 64)), float(hist_cfg.get("sigma", .02)))

    if job["group"] == "B":
        from utils.mesh_renderer_nvdiffrast import NVDiffMeshRenderer
        rasterizer = NVDiffMeshRenderer(resolution=0, antialias=False)
        indices = np.unique(np.linspace(0, len(frames) - 1, int(curves.probe_frames)).round().astype(int))
        template = frames[int(cfg.evaluation.get("fixed_camera_index", 0))]
        with torch.no_grad():
            for i in indices:
                cam = _camera(template, frames[i], resolution, 0, center.to(template.world_view_transform.device))
                mask = rasterizer(cam.mesh_vertices.cuda(), faces, cam)["mask"][0, 0].cpu().numpy() > .5
                probe_cameras.append(cam)
                probe_boxes.append(foreground_box(mask, float(cfg.evaluation.get("crop_padding", .1))))
        del rasterizer
        with Image.open(style_path) as image:
            style_hist = histogram(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)
        provenance.update(probe_frame_indices=indices.tolist(), probe_boxes=probe_boxes)
        save_json(folder / "manifest.json", provenance)

    measurements = folder / "measurements.jsonl"
    counts = dict(sh=0, geometry=0)
    optimized = set()

    @torch.no_grad()
    def measure_cm(progress, requested_progress, step):
        values, clipped = [], []
        for cam, box in zip(probe_cameras, probe_boxes):
            image = render(cam)
            clipped.append(float(((image < 0) | (image > 1)).float().mean()))
            array = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            y0, y1, x0, x1 = box
            value = histogram_distance(histogram(array[y0:y1, x0:x1]), style_hist)
            if isinstance(value, dict):
                value = value["value"]
            values.append(float(value))
        append(measurements, dict(metric="cm", value=float(np.mean(values)), status="ok", step=step,
                                 progress=progress, requested_progress=requested_progress, **counts,
                                 probe_count=len(values), clipped_fraction=float(np.mean(clipped))))

    def save_checkpoint(step):
        target = folder / "checkpoints"
        target.mkdir(exist_ok=True)
        path = target / f"update_{step:06d}.pth"
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        # Export only to this experiment directory. The source is never overwritten.
        torch.save({"gaussians": state, "faceid": model.faceid.detach().cpu(),
                    "active_sh_degree": model.active_sh_degree, "iteration": step,
                    "gaussian_style_parameterization": "direct", "curve_job": job,
                    "update_counts": counts.copy(), "optimized_uids": sorted(optimized),
                    "source_checkpoint": str(checkpoint)}, path)

    def update(window, phase, step):
        for params, enabled in ((sh_params, phase["sh"]), (geo_params, phase["geometry"])):
            for p in params:
                p.requires_grad_(enabled)
        sh_optimizer.zero_grad(set_to_none=True)
        geo_optimizer.zero_grad(set_to_none=True)
        video = torch.stack([render(cameras[i]) for i in window])[None]
        rng = random.Random(seed + 100003 * step)
        guidance_kwargs = {}
        if teacher is not None:
            guidance_kwargs = {"timesteps": rng.randint(teacher.min_step, teacher.max_step),
                               "noise_seed": rng.randrange(2**31)}
        gradients, metrics = routed_style_gradients(
            video, [int(frames[i].uid) for i in window], appearance, teacher, sh_params, geo_params,
            float(style.appearance.weight),
            float(style.appearance.weight if job["variant"] == "vgg_geometry" else style.sds_weight),
            guidance_kwargs=guidance_kwargs, offload_teacher=bool(style.appearance.offload_teacher),
            geometry_supervision="vgg" if job["variant"] == "vgg_geometry" else "diffusion")
        for param, gradient in gradients:
            param.grad = gradient.detach()
        for enabled, params, clip in ((phase["sh"], sh_params, float(style.appearance.grad_clip)),
                                      (phase["geometry"], geo_params, float(geo_cfg.grad_clip))):
            if enabled:
                torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
        for name, optimizer in (("sh", sh_optimizer), ("geometry", geo_optimizer)):
            if phase[name]:
                optimizer.step()
                counts[name] += 1
        for param in sh_params + geo_params:
            require_finite(param, "curve optimized parameter")
        optimized.update(int(frames[i].uid) for i in window)
        result = {key: float(value) for key, value in metrics.items()}
        # Drop render graphs before resetting CUDA peak measurements next update.
        model.xyz = model.xyz.detach()
        return result

    seed_all()
    order_rng = random.Random(seed)
    order = []
    next_progress = float(curves.evaluate_every_percent)
    if job["group"] == "B":
        measure_cm(0., 0., 0)
    for step, phase in enumerate(steps, 1):
        if not order:
            order = list(range(len(windows)))
            order_rng.shuffle(order)
        window = windows[order.pop()]
        benchmark = job["group"] == "A" and step > int(curves.A.warmup_updates)
        if benchmark:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
        metrics = update(window, phase, step)
        if benchmark:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            for name, value in (("seconds_per_update", elapsed),
                                ("peak_allocated_gb", torch.cuda.max_memory_allocated() / 1e9),
                                ("peak_reserved_gb", torch.cuda.max_memory_reserved() / 1e9)):
                append(measurements, dict(metric=name, value=value, status="ok", step=step))
        append(folder / "updates.jsonl", dict(step=step, frames=window, **counts, **metrics))
        if step == 1 or step % 50 == 0 or step == len(steps):
            print(f"[Curves {job['group']}] {step}/{len(steps)} SH={counts['sh']} geometry={counts['geometry']} "
                  f"loss={metrics['gaussian_style_loss']:.6g}", flush=True)
        if job["group"] == "B":
            progress = 100. * sum(counts.values()) / (int(curves.sh_updates) + int(curves.geometry_updates))
            if progress + 1e-8 >= next_progress or step == len(steps):
                measure_cm(progress, min(100., next_progress), step)
                next_progress += float(curves.evaluate_every_percent)
        if job["group"] != "A" and (step % int(curves.checkpoint_every_updates) == 0 or step == len(steps)):
            save_checkpoint(step)
    if job["group"] != "A" and counts != dict(sh=int(curves.sh_updates), geometry=int(curves.geometry_updates)):
        raise AssertionError("Parameter update budgets were not matched")
    for name, optimizer in (("sh", sh_optimizer), ("geometry", geo_optimizer)):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if int(optimizer.state[parameter]["step"]) != counts[name]:
                    raise AssertionError(f"Adam step counter mismatch for {group['name']}")
    for name, value in model.named_parameters():
        if name in frozen and not torch.equal(value.detach().cpu(), frozen[name]):
            raise AssertionError(f"Frozen parameter changed: {name}")
    if not torch.equal(frozen_faces, model.faceid.detach().cpu()):
        raise AssertionError("Primitive identities changed")
    if job["group"] == "C":
        if optimized != {int(frames[i].uid) for i in used}:
            raise AssertionError("Not every declared coverage time was style optimized")
        if teacher is not None:
            teacher.park_diffusion_backbone()
        del teacher, appearance
        sh_optimizer.zero_grad(set_to_none=True)
        geo_optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        cfg.evaluation.temporal_frame_indices = heldout
        cfg.evaluation.require_heldout = True
        cfg.evaluation.view_times = 0
        cfg.evaluation.image_size = resolution
        cfg.evaluation.short_gap = int(curves.C.pair_gap)
        cfg.evaluation.long_gap = len(frames) + 1
        cfg.evaluation.variant = job["variant"]
        controller = SimpleNamespace(cfg=style, source_checkpoint=str(checkpoint), pretrained_checkpoint=str(checkpoint),
                                     style_optimizers=lambda: [sh_optimizer, geo_optimizer])
        bundle = export_paper_bundle(cfg, model, controller, frames, inria_render, len(steps), optimized)
        from tools.evaluate_paper import evaluate_bundle
        summary = evaluate_bundle(bundle)
        with (bundle / "per_image_and_pair_metrics.csv").open(encoding="utf-8", newline="") as stream:
            pair_ids = sorted(row["sample"] for row in csv.DictReader(stream)
                              if row["group"] == "temporal_short" and row["metric"] == "lpips" and row["status"] == "ok")
        for bucket, metric in (("foreground", "cm"), ("temporal_short", "lpips")):
            row = summary["metrics"][bucket][metric]
            append(measurements, dict(metric=metric, **row, bundle=str(bundle), **counts,
                                     pair_set_sha256=digest(pair_ids) if metric == "lpips" else ""))
        result = summary["metrics"]["temporal_short"]["lpips"]
        if result["status"] != "ok":
            raise RuntimeError(f"Held-out LPIPS unavailable; retained evaluation bundle: {bundle}")
        expected = len(heldout_pairs(len(frames), curves.C.heldout_ranges, int(curves.C.pair_gap)))
        provenance.update(heldout_pair_count=expected, measured_pairs=result.get("count"),
                          excluded_pairs=result.get("excluded", 0), valid_pair_ids=pair_ids,
                          evaluation_bundle=str(bundle))
    provenance.update(update_counts=counts, optimized_uids=sorted(optimized))
    save_json(folder / "manifest.json", provenance)
    return {"status": "complete", "update_counts": counts, "optimized_frames": len(optimized)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_directory", type=Path)
    args = parser.parse_args()
    folder = args.job_directory.resolve()
    payload = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    os.chdir(ROOT)
    save_json(folder / "status.json", {"status": "running"})
    try:
        status = run(folder, payload)
    except Exception as exc:
        traceback.print_exc()
        message = f"{type(exc).__name__}: {exc}"
        status = {"status": "oom" if "out of memory" in message.lower() else "failed", "error": message}
    save_json(folder / "status.json", status)
    if status["status"] != "complete":
        sys.exit(2)


if __name__ == "__main__":
    main()
