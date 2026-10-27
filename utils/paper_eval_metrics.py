"""Offline, frozen evaluation models. No downloads and no training gradients."""
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from utils.paper_eval_protocol import fingerprint


def _weights(path, default_url=None):
    if not path and default_url:
        path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(default_url).path).name
    path = Path(path) if path else None
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Local evaluator weights missing: {path}. Evaluation never auto-downloads.")
    return path


def image_tensor(image, device):
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float().to(device)


def resize_histogram_input(image, size=150):
    if max(image.shape[:2]) <= size:
        return image
    x = F.interpolate(image_tensor(image, "cpu"), size=(size, size), mode="bilinear", align_corners=False)
    return x[0].permute(1, 2, 0).numpy()


class PaperModels:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cpu"))
        self.models, self.errors, self.weights = {}, {}, {}

    def get(self, name):
        if name in self.errors:
            raise RuntimeError(self.errors[name])
        if name not in self.models:
            try:
                self.models[name] = self._load(name).eval().requires_grad_(False).to(self.device)
            except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                raise
        return self.models[name]

    def _load(self, name):
        options = self.cfg.get(name, {})
        if name == "lpips":
            import lpips
            path = _weights(options.get("alexnet_weights", ""), torchvision.models.AlexNet_Weights.IMAGENET1K_V1.url)
            # Random initialization avoids torchvision's implicit download; all
            # feature weights are then replaced strictly by the local checkpoint.
            model = lpips.LPIPS(net="alex", version="0.1", spatial=True, pnet_rand=True, verbose=False)
            state = torch.load(path, map_location="cpu", weights_only=True)
            feature_state = {key: state["features." + key.split(".", 1)[1]] for key in model.net.state_dict()}
            model.net.load_state_dict(feature_state, strict=True)
            calibration = Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"
            self.weights["lpips_calibration"] = fingerprint(calibration)
        elif name == "raft":
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
            path = _weights(options.get("weights", ""), Raft_Large_Weights.C_T_SKHT_V2.url)
            model = raft_large(weights=None, progress=False)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        elif name == "artscore":
            path = _weights(options.get("weights", ""))
            model = torchvision.models.resnet50(weights=None)
            model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1000), torch.nn.ReLU(),
                                           torch.nn.Dropout(0.5), torch.nn.Linear(1000, 1))
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        elif name == "cfsd":
            path = _weights(options.get("weights", ""), torchvision.models.VGG19_Weights.IMAGENET1K_V1.url)
            model = torchvision.models.vgg19(weights=None)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model = model.features[:int(options.get("feature_end", 12))]
        else:
            raise ValueError(f"Unknown metric model {name}")
        self.weights[name] = {"path": str(path), "sha256": fingerprint(path)}
        return model

    @torch.no_grad()
    def lpips(self, a, b, mask=None):
        model = self.get("lpips")
        if min(a.shape[:2]) < 64:
            raise ValueError("LPIPS crop must have both sides >=64 pixels")
        model.spatial = mask is not None
        distance = model(image_tensor(a, self.device) * 2 - 1, image_tensor(b, self.device) * 2 - 1)
        if mask is None:
            return float(distance.mean())
        mask_tensor = torch.as_tensor(mask.copy(), device=self.device).float()[None, None]
        mask_tensor = F.interpolate(mask_tensor, distance.shape[-2:], mode="nearest")
        if mask_tensor.sum() == 0:
            raise ValueError("No valid LPIPS pixels")
        return float((distance * mask_tensor).sum() / mask_tensor.sum())

    @torch.no_grad()
    def artscore(self, image):
        # Official ArtScore preprocessing uses PIL Resize(224) and ImageNet normalization.
        pil = Image.fromarray(np.rint(image * 255).astype(np.uint8))
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224), torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return float(self.get("artscore")(transform(pil)[None].to(self.device)).reshape(-1)[0])

    @torch.no_grad()
    def flow(self, a, b):
        model = self.get("raft")
        x, y = image_tensor(a, self.device), image_tensor(b, self.device)
        h, w = x.shape[-2:]
        # RAFT needs a correlation pyramid at least 128x128 and multiples of 8.
        pad = (0, max(128, (w + 7) // 8 * 8) - w, 0, max(128, (h + 7) // 8 * 8) - h)
        x, y = F.pad(x, pad, mode="replicate") * 2 - 1, F.pad(y, pad, mode="replicate") * 2 - 1
        result = model(x.contiguous(), y.contiguous(), num_flow_updates=int(self.cfg.get("raft", {}).get("updates", 12)))[-1]
        return result[0, :, :h, :w].permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def cfsd(self, a, b):
        options = self.cfg.get("cfsd", {})
        if not options.get("enabled", False):
            raise ValueError("CFSD optional diagnostic disabled")
        features = []
        model = self.get("cfsd")
        for image in (a, b):
            x = F.interpolate(image_tensor(image, self.device), size=(int(options.get("size", 128)),) * 2,
                              mode="bilinear", align_corners=False)
            x = torchvision.transforms.functional.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            features.append(model(x)[0].flatten(1).T)
        f0, f1 = features
        total = f0.new_zeros(())
        for start in range(0, len(f0), 128):
            p = F.log_softmax(f0[start:start + 128] @ f0.T, dim=-1)
            q = F.log_softmax(f1[start:start + 128] @ f1.T, dim=-1)
            total += (p.exp() * (p - q)).sum()
        return float(total / len(f0))


def ssim_content(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0,
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
