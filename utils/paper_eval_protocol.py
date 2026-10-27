"""Pure array utilities for the paper's evaluation protocol (no training loss)."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


VERSION = "mags-paper-v1"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(value=None, reason="", status=None, **metadata):
    if value is not None:
        value = float(value)
        if not np.isfinite(value):
            return {"value": None, "status": "nonfinite", "reason": reason or "Non-finite metric", **metadata}
    return {"value": value, "status": status or ("ok" if value is not None else "unavailable"),
            "reason": reason, **metadata}


def aggregate(records):
    values = [r["value"] for r in records if r["status"] == "ok"]
    return record(np.mean(values) if values else None,
                  reason="" if values else "No valid measurements",
                  count=len(values), excluded=len(records) - len(values))


def rgb_array(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
        raise ValueError("Expected finite HWC RGB")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Metrics require RGB in [0,1]; do not rescale individual images")
    return image


def masked_rmse(a, b, mask):
    a, b = rgb_array(a), rgb_array(b)
    mask = np.asarray(mask, dtype=bool)
    if a.shape != b.shape or mask.shape != a.shape[:2]:
        raise ValueError("Image/mask shape mismatch")
    if not mask.any():
        return record(reason="No valid pixels")
    return record(np.sqrt(np.square(a[mask].astype(np.float64) - b[mask]).mean()))


def bilinear_sample(array, xy):
    """Sample pixel-center coordinates; callers must mask points out of bounds."""
    array = np.asarray(array)
    h, w = array.shape[:2]
    x, y = np.clip(xy[..., 0], 0, w - 1), np.clip(xy[..., 1], 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    if array.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    return ((1 - wy) * ((1 - wx) * array[y0, x0] + wx * array[y0, x1])
            + wy * ((1 - wx) * array[y1, x0] + wx * array[y1, x1]))


def pixel_grid(height, width):
    y, x = np.mgrid[:height, :width]
    return np.stack((x, y), axis=-1).astype(np.float32)


def flow_correspondence(forward, backward, mask_a, mask_b, alpha=0.01, beta=0.5):
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise ValueError("Non-finite optical flow")
    h, w = forward.shape[:2]
    xy = pixel_grid(h, w) + forward
    reverse = bilinear_sample(backward, xy)
    valid = ((xy[..., 0] >= 0) & (xy[..., 0] <= w - 1)
             & (xy[..., 1] >= 0) & (xy[..., 1] <= h - 1))
    error = np.square(forward + reverse).sum(-1)
    threshold = alpha * (np.square(forward).sum(-1) + np.square(reverse).sum(-1)) + beta
    valid &= (error <= threshold) & mask_a & (bilinear_sample(mask_b.astype(float), xy) >= 0.999)
    return xy, valid


def mesh_correspondence(source, destination, projection, relative_tolerance=0.01, absolute_tolerance=1e-4):
    points = source["points"]
    clip = np.concatenate([points, np.ones((*points.shape[:2], 1))], -1) @ np.asarray(projection)
    depth = clip[..., 3]
    h, w = destination["mask"].shape
    xy = (clip[..., :2] / np.maximum(depth[..., None], 1e-12) + 1) * np.array([w, h]) / 2 - 0.5
    visible = ((depth > 0) & source["mask"] & (xy[..., 0] >= 0) & (xy[..., 0] <= w - 1)
               & (xy[..., 1] >= 0) & (xy[..., 1] <= h - 1))
    destination_depth = bilinear_sample(destination["depth"], xy)
    visible &= bilinear_sample(destination["mask"].astype(float), xy) >= 0.999
    visible &= np.abs(destination_depth - depth) <= absolute_tolerance + relative_tolerance * np.abs(depth)
    return xy.astype(np.float32), visible


def erode_mask(mask, radius):
    mask = np.asarray(mask, dtype=bool)
    if radius < 0:
        raise ValueError("Mask erosion radius must be nonnegative")
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask, radius, constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))
    return windows.all(axis=(-2, -1))


def foreground_box(mask, padding=0.1):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Empty source foreground")
    h, w = mask.shape
    pad = int(np.ceil(max(x.max() - x.min() + 1, y.max() - y.min() + 1) * padding))
    return [max(0, int(y.min()) - pad), min(h, int(y.max()) + 1 + pad),
            max(0, int(x.min()) - pad), min(w, int(x.max()) + 1 + pad)]


def rgbuv_histogram(image, bins=64, sigma=0.02):
    """HistoGAN RGB-uv inverse-quadratic histogram; caller handles resizing.

    Matches the equations and defaults of Afifi et al., CVPR 2021. All three
    histograms are normalized jointly, as required by the paper's CM metric.
    """
    pixels = rgb_array(image).reshape(-1, 3).astype(np.float64)
    if bins < 2 or sigma <= 0:
        raise ValueError("Invalid histogram configuration")
    centers = np.linspace(-3, 3, bins)
    logs = np.log(pixels + 1e-6)
    intensity = np.sqrt(np.square(pixels).sum(1) + 1e-6)
    histograms = []
    for primary, other1, other2 in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        u, v = logs[:, primary] - logs[:, other1], logs[:, primary] - logs[:, other2]
        ku = 1 / (1 + np.square((u[:, None] - centers) / sigma))
        kv = 1 / (1 + np.square((v[:, None] - centers) / sigma))
        histograms.append((ku * intensity[:, None]).T @ kv)
    result = np.stack(histograms)
    return result / result.sum()


def histogram_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or (a < 0).any() or (b < 0).any() or a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("Invalid histograms")
    return float(np.linalg.norm(np.sqrt(a / a.sum()) - np.sqrt(b / b.sum())) / np.sqrt(2))


def temporal_pairs(times, gap):
    times = np.asarray(times, dtype=float)
    if gap <= 0 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("Temporal protocol requires strictly increasing timestamps and positive gaps")
    duration = times[-1] - times[0] if len(times) > 1 else 0
    return [{"a": i, "b": i + gap, "gap": gap,
             "delta_tau": float((times[i + gap] - times[i]) / duration)}
            for i in range(max(0, len(times) - gap))]


def view_pairs(centers, target, narrow=(0, 10), wide=(30, 60)):
    directions = np.asarray(centers, dtype=float) - np.asarray(target, dtype=float)
    lengths = np.linalg.norm(directions, axis=-1)
    if (lengths < 1e-8).any():
        raise ValueError("Camera at mesh center")
    directions /= lengths[:, None]
    result = []
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            angle = float(np.degrees(np.arccos(np.clip(directions[i] @ directions[j], -1, 1))))
            group = "narrow" if narrow[0] < angle <= narrow[1] + 1e-8 else (
                "wide" if wide[0] - 1e-8 <= angle <= wide[1] + 1e-8 else None)
            if group:
                result.append({"a": i, "b": j, "angle_degrees": angle, "group": group})
    return result


def geometry_statistics(initial_xyz, edited_xyz, extent, initial_log_scale, edited_log_scale):
    a, b = np.asarray(initial_xyz), np.asarray(edited_xyz)
    s0, s1 = np.asarray(initial_log_scale), np.asarray(edited_log_scale)
    if a.shape != b.shape or s0.shape != s1.shape or len(a) != len(s0) or extent <= 0:
        raise ValueError("Geometry evaluation needs matched primitives and positive source extent")
    if not all(np.isfinite(x).all() for x in (a, b, s0, s1)):
        raise ValueError("Non-finite geometry")
    displacement = np.linalg.norm(b - a, axis=-1) / extent
    return {"delta_pos": float(displacement.mean()), "delta_pos_p50": float(np.median(displacement)),
            "delta_pos_p95": float(np.percentile(displacement, 95)),
            "delta_scale": float(np.abs(np.clip(s1, -20, 8) - np.clip(s0, -20, 8)).mean())}
