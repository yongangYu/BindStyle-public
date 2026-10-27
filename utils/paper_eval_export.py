"""Export final Gaussian renders and source-only correspondence data."""
import copy
import datetime
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.paper_eval_protocol import (VERSION, fingerprint, foreground_box,
                                      geometry_statistics, temporal_pairs, view_pairs, write_json)


def validate_paper_evaluation(config, cameras):
    cfg = config.get("evaluation", {})
    if not cfg.get("enabled", False):
        return
    style = config.optim.gaussian_style_update
    if not style.get("enabled", False) or style.get("parameterization") != "direct" or style.get("use_deformer", False):
        raise ValueError("Paper evaluation currently requires direct Gaussian style with no deformer forward")
    if config.optim.mesh_style_update.get("enabled", False):
        raise ValueError("Paper evaluation requires the known, unedited GT-mesh sequence")
    required = int(cfg.get("required_frames", 200))
    if not cameras or required < 2:
        raise ValueError("Paper evaluation needs cameras and required_frames >= 2")
    if cfg.get("require_full_sequence", False) and len(cameras) != required:
        raise ValueError(f"Paper evaluation requires {required} frames, found {len(cameras)}. Set dataset.frame_limit=0.")
    temporal_pairs(sorted(float(c.fid) for c in cameras), 2)
    for key in ("short_gap", "long_gap", "view_times", "max_pairs_per_bin"):
        if int(cfg.get(key, {"short_gap": 2, "long_gap": 20, "view_times": 10, "max_pairs_per_bin": 20}[key])) < 1:
            raise ValueError(f"evaluation.{key} must be positive")
    if not 0 <= int(cfg.get("fixed_camera_index", 0)) < len(cameras):
        raise ValueError("evaluation.fixed_camera_index is out of range")
    if int(cfg.get("image_size", 256)) < 128:
        raise ValueError("evaluation.image_size must be >=128")
    if int(cfg.get("mask_erosion", 3)) < 0 or not 0 <= float(cfg.get("minimum_coverage", 0.05)) <= 1:
        raise ValueError("Invalid evaluation mask settings")
    if len(cfg.get("view_yaws", [0, 5, 35, 60, 90, 95, 125, 150])) < 2:
        raise ValueError("Paper evaluation needs at least two orbit cameras")


def _camera(template, frame, resolution, yaw, center):
    cam = copy.copy(template)
    changes = {"uid": frame.uid, "fid": frame.fid, "mesh_vertices": frame.mesh_vertices,
               "mesh_faces": frame.mesh_faces, "mesh_normals": frame.mesh_normals}
    if resolution > 0:
        ratio = resolution / max(template.image_height, template.image_width)
        changes.update(image_height=max(1, round(template.image_height * ratio)),
                       image_width=max(1, round(template.image_width * ratio)))
    # Rotate the entire reference camera about a common center, retaining its
    # radius, intrinsics and orientation relative to the object.
    c2w = template.world_view_transform.T.inverse().clone()
    angle = np.deg2rad(yaw)
    rotation = c2w.new_tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                              [-np.sin(angle), 0, np.cos(angle)]])
    c2w[:3, :3] = rotation @ c2w[:3, :3]
    c2w[:3, 3] = center + rotation @ (c2w[:3, 3] - center)
    world = c2w.inverse().T.contiguous()
    changes.update(world_view_transform=world,
                   full_proj_transform=(world @ template.projection_matrix).contiguous(),
                   camera_center=c2w[:3, 3].contiguous(),
                   R=world[:3, :3].cpu().numpy(), T=world[3, :3].cpu().numpy())
    if hasattr(cam, "_replace"):
        return cam._replace(**changes)
    for name, value in changes.items():
        setattr(cam, name, value)
    return cam


def load_base_gaussians(config, checkpoint_path, device):
    # Only read an explicitly configured, trusted MaGS checkpoint. This is not
    # a diffusion weight loader and never downloads or rewrites the source.
    from model.mags_model import MaGSModel
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("gaussian_style_parameterization"):
        raise ValueError("Evaluation source must be the original unstyled MaGS checkpoint")
    model = MaGSModel(config)
    for name in ("_uvw", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest", "mesh_feat"):
        value = payload["gaussians"][name].to(device)
        setattr(model, name, torch.nn.Parameter(value, requires_grad=False))
    model.faceid = payload["faceid"].long().to(device)
    model.active_sh_degree = int(payload.get("active_sh_degree", model.max_sh_degree))
    return model


def _source_model(gaussians, config, checkpoint_path):
    model = load_base_gaussians(config, checkpoint_path, gaussians._uvw.device)
    if not torch.equal(model.faceid, gaussians.faceid):
        raise ValueError("Evaluation requires identical primitive identities and face bindings")
    return model


def _numpy(tensor):
    return tensor.detach().float().cpu().numpy()


@torch.no_grad()
def export_paper_bundle(config, gaussians, controller, cameras, render, iteration, optimized_uids):
    from omegaconf import OmegaConf
    from utils.mesh_renderer_nvdiffrast import NVDiffMeshRenderer

    cfg = OmegaConf.to_container(config.evaluation, resolve=True)
    directory = Path(config.model.model_path) / "paper_eval" / f"iteration_{iteration:06d}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "status.json", {"status": "exporting", "protocol": VERSION})
    frames = sorted(cameras, key=lambda c: float(c.fid))
    source_path = controller.pretrained_checkpoint
    source = _source_model(gaussians, config, source_path)
    source.requires_grad_(False)
    source_faces = frames[0].mesh_faces.to(gaussians._uvw.device).long()
    if not all(torch.equal(c.mesh_faces.to(source_faces), source_faces) for c in frames):
        raise ValueError("Paper mesh correspondence requires a shared mesh topology")
    mesh_digest = hashlib.sha256(source_faces.cpu().numpy().astype("<i8").tobytes())
    for frame in frames:
        mesh_digest.update(_numpy(frame.mesh_vertices).astype("<f4").tobytes())
        mesh_digest.update(_numpy(frame.mesh_normals).astype("<f4").tobytes())
    style_path = str(controller.cfg.get("style_path", ""))
    style = Image.open(style_path).convert("RGB")
    style.save(directory / "style.png")
    manifest = {"protocol": VERSION, "iteration": iteration, "settings": cfg,
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "scene": str(config.dataset.source_path), "style_sha256": fingerprint(style_path),
                "source_checkpoint": source_path, "source_checkpoint_sha256": fingerprint(source_path),
                "mesh_sequence_sha256": mesh_digest.hexdigest(),
                "style_path": style_path, "deformer_forward": False,
                "training_frame_count": len(frames), "optimized_uids": sorted(optimized_uids),
                "resumed_style": controller.source_checkpoint != controller.pretrained_checkpoint,
                "scope": "full_sequence" if len(frames) == int(cfg.get("required_frames", 200)) else "pilot",
                "samples": [], "view_pairs": [], "temporal_pairs": [], "geometry": [],
                "coverage_status": "not_evaluated_no_heldout_optimization_protocol",
                "config_file": str(Path(config.model.model_path) / "cfg_args.yaml")}
    write_json(directory / "training_config.json", OmegaConf.to_container(config, resolve=True))
    center = (frames[0].mesh_vertices.amin(0) + frames[0].mesh_vertices.amax(0)) * 0.5
    template = frames[int(cfg.get("fixed_camera_index", 0))]
    background = gaussians._uvw.new_tensor([1, 1, 1] if config.dataset.white_background else [0, 0, 0])
    renderer = NVDiffMeshRenderer(resolution=0, antialias=False)
    geometry_initial = _numpy(source._scaling)
    geometry_final = _numpy(gaussians._scaling)
    times = [float(c.fid) for c in frames]
    selected_indices = cfg.get("temporal_frame_indices", list(range(len(frames))))
    selected_indices = sorted(set(int(i) for i in selected_indices))
    if not selected_indices or min(selected_indices) < 0 or max(selected_indices) >= len(frames):
        raise ValueError("Invalid evaluation temporal_frame_indices")
    if cfg.get("require_heldout", False):
        heldout_uids = {int(frames[i].uid) for i in selected_indices}
        if heldout_uids.intersection(optimized_uids):
            raise ValueError("Held-out evaluation contains style-optimized frames")
        manifest["coverage_status"] = "verified_heldout_style_optimization"
        manifest["heldout_indices"] = selected_indices

    def save_sample(frame_index, yaw, sample_id):
        frame = frames[frame_index]
        cam = _camera(template, frame, int(cfg.get("image_size", 256)), yaw, center)
        verts, norms = frame.mesh_vertices.to(source_faces.device), frame.mesh_normals.to(source_faces.device)
        source.update_mesh(source_faces, verts, norms, deformer=None)
        gaussians.update_mesh(source_faces, verts, norms, deformer=None)
        a = render(cam, source, config.pipe, background)["render"]
        b = render(cam, gaussians, config.pipe, background)["render"]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Non-finite final Gaussian render {sample_id}")
        clipped = float(((b < 0) | (b > 1)).float().mean())
        a, b = a.clamp(0, 1), b.clamp(0, 1)
        raster = renderer(verts, source_faces, cam)
        points, _ = renderer._dr.interpolate(verts[None].contiguous(), raster["rast"], source_faces.int().contiguous())
        world = points[0]
        depth = (torch.cat([world, torch.ones_like(world[..., :1])], -1) @ cam.full_proj_transform)[..., 3]
        mask = raster["mask"][0, 0] > 0.5
        a, b, mask = _numpy(a.permute(1, 2, 0)), _numpy(b.permute(1, 2, 0)), mask.cpu().numpy()
        np.savez_compressed(directory / f"{sample_id}.npz", source=a, styled=b, mask=mask,
                            points=_numpy(world), depth=_numpy(depth))
        try:
            box = foreground_box(mask, float(cfg.get("crop_padding", 0.1)))
        except ValueError:
            box = None
        for label, array in (("source", a), ("styled", b)):
            folder = directory / label
            folder.mkdir(exist_ok=True)
            Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(folder / f"{sample_id}.png")
        manifest["samples"].append({"id": sample_id, "frame_index": frame_index,
                                    "uid": int(frame.uid), "time": times[frame_index], "yaw": yaw,
                                    "crop": box, "clipped_fraction": clipped,
                                    "full_proj_transform": _numpy(cam.full_proj_transform).tolist(),
                                    "camera_center": _numpy(cam.camera_center).tolist(),
                                    "world_view_transform": _numpy(cam.world_view_transform).tolist(),
                                    "fov_radians": [float(cam.FoVx), float(cam.FoVy)],
                                    "resolution": [cam.image_height, cam.image_width]})
        return cam

    for index in selected_indices:
        frame = frames[index]
        name = f"time_{index:04d}"
        save_sample(index, 0, name)
        extent = float((frame.mesh_vertices.amax(0) - frame.mesh_vertices.amin(0)).norm())
        stats = geometry_statistics(_numpy(source.get_xyz), _numpy(gaussians.get_xyz), extent,
                                    geometry_initial, geometry_final)
        manifest["geometry"].append({"frame_index": index, "time": times[index], **stats})
        if index % 20 == 0:
            print(f"[Paper Eval] Exported fixed-camera frame {index + 1}/{len(frames)}")
    for group, gap in (("short", cfg.get("short_gap", 2)), ("long", cfg.get("long_gap", 20))):
        for pair in temporal_pairs(times, int(gap)):
            if pair["a"] not in selected_indices or pair["b"] not in selected_indices:
                continue
            if cfg.get("require_heldout", False) and any(i not in selected_indices for i in range(pair["a"], pair["b"] + 1)):
                continue
            manifest["temporal_pairs"].append({**pair, "group": group,
                                               "a": f"time_{pair['a']:04d}", "b": f"time_{pair['b']:04d}"})
    indices = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), int(cfg.get("view_times", 10)))).round().astype(int))
    yaws = cfg.get("view_yaws", [0, 5, 35, 60, 90, 95, 125, 150])
    rng = np.random.default_rng(int(cfg.get("seed", 2026)))
    for index in indices:
        names, poses = [], []
        for view_index, yaw in enumerate(yaws):
            name = f"view_{index:04d}_{view_index:02d}"
            cam = save_sample(int(index), float(yaw), name)
            names.append(name)
            poses.append(_numpy(cam.camera_center))
        target = _numpy((frames[index].mesh_vertices.amin(0) + frames[index].mesh_vertices.amax(0)) * 0.5)
        extent = float((frames[index].mesh_vertices.amax(0) - frames[index].mesh_vertices.amin(0)).norm())
        pairs = view_pairs(poses, target, cfg.get("narrow_degrees", [0, 10]), cfg.get("wide_degrees", [30, 60]))
        for group in ("narrow", "wide"):
            candidates = [pair for pair in pairs if pair["group"] == group]
            for offset in rng.permutation(len(candidates))[:int(cfg.get("max_pairs_per_bin", 20))]:
                pair = candidates[offset]
                manifest["view_pairs"].append({**pair, "a": names[pair["a"]], "b": names[pair["b"]],
                                               "normalized_baseline": float(np.linalg.norm(poses[pair["a"]] - poses[pair["b"]]) / extent),
                                               "frame_index": int(index)})
    manifest["optimizer_steps"] = {
        group["name"]: max([int(optimizer.state.get(p, {}).get("step", 0)) for p in group["params"]] or [0])
        for optimizer in controller.style_optimizers() for group in optimizer.param_groups}
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "status.json", {"status": "exported", "protocol": VERSION})
    print(f"[Paper Eval] Saved evaluation inputs: {directory}")
    return directory
