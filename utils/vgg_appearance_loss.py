"""Frozen VGG16 appearance supervision with cached, spatial style features."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from utils.style_numerics import require_finite


def image_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def foreground_box(masks, padding=0.1):
    """One fixed crop for a sequence, derived only from its GT masks."""
    points = []
    for mask in masks:
        mask = mask.detach().cpu().squeeze()
        if mask.ndim != 2:
            raise ValueError("GT foreground mask must have shape [1,H,W] or [H,W].")
        indices = (mask > 0.5).nonzero()
        if indices.numel():
            size = torch.tensor(mask.shape, dtype=torch.float32)
            points.extend([indices.amin(0).float() / size,
                           (indices.amax(0).float() + 1) / size])
    if not points:
        raise ValueError("Cannot construct an appearance crop from empty GT masks.")
    points = torch.stack(points)
    lower, upper = points.amin(0), points.amax(0)
    margin = (upper - lower) * float(padding)
    return torch.cat([(lower - margin).clamp(0, 1),
                      (upper + margin).clamp(0, 1)]).tolist()


def nearest_feature_loss(output, style, chunk_size=512):
    """Same cosine nearest-feature objective as NNFMLoss, with bounded memory."""
    query = F.normalize(output.flatten(2).transpose(1, 2).float(), dim=-1, eps=1e-8)
    keys = F.normalize(style.flatten(2).transpose(1, 2).float(), dim=-1, eps=1e-8)
    keys = keys.reshape(-1, keys.shape[-1]).detach()
    query = query.reshape(-1, query.shape[-1])
    losses = []
    for block in query.split(chunk_size):
        # Matching is discrete; gradients flow through the selected cosine pairs.
        with torch.no_grad():
            best = block.new_full((len(block),), -float("inf"))
            selected = torch.zeros(len(block), dtype=torch.long, device=block.device)
            for start in range(0, len(keys), chunk_size):
                values, indices = (block.detach() @ keys[start:start + chunk_size].T).max(1)
                better = values > best
                selected = torch.where(better, indices + start, selected)
                best = torch.maximum(best, values)
        losses.append((1 - (block * keys[selected]).sum(-1)).sum())
    return torch.stack(losses).sum() / len(query)


class VGGAppearanceLoss(nn.Module):
    def __init__(self, cfg, style_path, cameras, device, features=None):
        super().__init__()
        self.loss_type = str(cfg.get("loss_type", "nnfm"))
        if self.loss_type not in {"nnfm", "gram", "adain"}:
            raise ValueError("appearance.loss_type must be nnfm, gram, or adain.")
        self.image_size = int(cfg.get("image_size", 256))
        self.chunk_size = int(cfg.get("chunk_size", 512))
        if self.image_size < 32 or self.chunk_size < 1:
            raise ValueError("appearance image_size must be >=32 and chunk_size positive.")
        # Block 2 matches the existing NNFMLoss default: relu3_1/2/3.
        self.layers = tuple(int(x) for x in cfg.get("layers", [11, 13, 15]))
        if not self.layers or list(self.layers) != sorted(set(self.layers)) or min(self.layers) < 0:
            raise ValueError("appearance.layers must be sorted unique VGG feature indices.")
        self.crop = foreground_box([cam.gt_mask for cam in cameras], cfg.get("crop_padding", 0.1))
        self.style_fingerprint = image_fingerprint(style_path)
        if features is None:
            from torchvision.models import VGG16_Weights, vgg16

            weights_path = str(cfg.get("vgg_weights_path", ""))
            if not weights_path and not bool(cfg.get("allow_download", False)):
                weights_path = str(Path(torch.hub.get_dir()) / "checkpoints" / Path(VGG16_Weights.DEFAULT.url).name)
                if not Path(weights_path).is_file():
                    raise FileNotFoundError(
                        "VGG16 weights are not cached. Set appearance.vgg_weights_path "
                        "to the torchvision VGG16 state dict, or appearance.allow_download=true. "
                        f"Expected cache file: {weights_path}"
                    )
            model = vgg16(weights=None if weights_path else VGG16_Weights.DEFAULT)
            if weights_path:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
            features = model.features
            del model
        if max(self.layers) >= len(features):
            raise ValueError("appearance.layers exceeds the VGG feature network.")
        self.features = features[:max(self.layers) + 1].to(device).float().eval()
        self.features.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        with Image.open(style_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
        style = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            for index, value in enumerate(self.get_features(self.resize(style))):
                self.register_buffer(f"style_{index}", value.detach())

    def resize(self, image):
        h, w = image.shape[-2:]
        scale = self.image_size / max(h, w)
        return F.interpolate(image.float(), size=(max(32, round(h * scale)), max(32, round(w * scale))),
                             mode="bilinear", align_corners=False)

    def get_features(self, image):
        value = (image.float().clamp(0, 1) - self.mean) / self.std
        result = []
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in self.layers:
                result.append(value)
        return result

    def forward(self, images):
        require_finite(images, "VGG appearance RGB")
        h, w = images.shape[-2:]
        y0, x0, y1, x1 = self.crop
        crop = images[..., int(y0 * h):max(int(y0 * h) + 1, round(y1 * h)),
                      int(x0 * w):max(int(x0 * w) + 1, round(x1 * w))]
        features = self.get_features(self.resize(crop))
        loss = images.new_zeros((), dtype=torch.float32)
        for index, output in enumerate(features):
            style = getattr(self, f"style_{index}")
            if self.loss_type == "nnfm":
                loss = loss + nearest_feature_loss(output, style, self.chunk_size)
            elif self.loss_type == "gram":
                def gram(value):
                    flat = value.float().flatten(2)
                    return flat @ flat.transpose(1, 2) / (flat.shape[1] * flat.shape[2])
                loss = loss + (gram(output) - gram(style)).square().mean()
            else:
                def stats(value):
                    flat = value.float().flatten(2)
                    return flat.mean(-1), (flat.var(-1, unbiased=False) + 1e-5).sqrt()
                out_mean, out_std = stats(output)
                sty_mean, sty_std = stats(style)
                loss = loss + (out_mean - sty_mean).square().mean() + (out_std - sty_std).square().mean()
        return require_finite(loss / len(features), "VGG appearance loss")
