#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import VGG16_Weights, vgg16


def gram_matrix(feat):
    b, c, h, w = feat.shape
    feat = feat.view(b, c, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2))
    return gram / max(c * h * w, 1)


def cos_loss(feat, target):
    feat = F.normalize(feat, dim=1, eps=1e-6)
    target = F.normalize(target.detach(), dim=1, eps=1e-6)
    return 1.0 - F.cosine_similarity(feat, target, dim=1, eps=1e-6).mean()


def nn_feat_replace(feat, style_feat):
    b, c, h, w = feat.shape
    feat_flat = feat.reshape(b, c, -1)
    style_flat = style_feat.reshape(style_feat.shape[0], style_feat.shape[1], -1)
    targets = []
    for batch_idx in range(b):
        style_idx = min(batch_idx, style_flat.shape[0] - 1)
        x = F.normalize(feat_flat[batch_idx], dim=0, eps=1e-6)
        s = F.normalize(style_flat[style_idx], dim=0, eps=1e-6)
        nearest = torch.matmul(x.transpose(0, 1), s).argmax(dim=1)
        targets.append(style_flat[style_idx].index_select(1, nearest).view(c, h, w))
    return torch.stack(targets, dim=0)


def lum_transform(image):
    if image.shape[1] == 1:
        return image.repeat(1, 3, 1, 1)
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    lum = (image[:, :3] * weights).sum(dim=1, keepdim=True)
    return lum.repeat(1, 3, 1, 1)


def crop_nonzero_region(image):
    mask = image.detach().abs().sum(dim=1, keepdim=True) > 0
    if not mask.any():
        return image
    coords = mask.nonzero(as_tuple=False)
    y0, y1 = coords[:, 2].min().item(), coords[:, 2].max().item() + 1
    x0, x1 = coords[:, 3].min().item(), coords[:, 3].max().item() + 1
    return image[:, :, y0:y1, x0:x1]


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5))
        + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5))
    )


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class NNFMLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
        )
        self.scale_coefs = [
            [0, 0, 0, 5, 3, 0],  # 0
            [0, 0, 5, 3, 0, 0],  # 1
            [8, 0, 0, 0, 0, 0],  # 2
        ]

    def get_feats(self, x, layers=[]):
        x = x.float().clamp(0.0, 1.0)
        x = (x - self.mean) / self.std
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs

    def forward(
            self,
            outputs,
            styles,
            blocks=[
                2,
            ],
            loss_names=["nnfm_loss"],  # can also include 'gram_loss', 'content_loss'
            contents=None,
            scale_level=None,
            x_mask=None,
            s_mask=None,
            styles2=None,
            spatial_weight=20,
    ):
        for x in loss_names:
            assert x in ['nnfm_loss', 'content_loss', 'gram_loss', 'lum_nnfm_loss', "spatial_loss", "scale_loss"]

        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        blocks.sort()
        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]
        # print('mask', x_mask.shape, outputs.shape,x_mask)
        if x_mask is not None:
            x_feats_all = self.get_feats(outputs * (1 - x_mask), all_layers)
        else:
            x_feats_all = self.get_feats(outputs, all_layers)
        with torch.no_grad():
            s_feats_all = self.get_feats(styles, all_layers)
            if "content_loss" in loss_names:
                content_feats_all = self.get_feats(contents, all_layers)

        # For spatial control
        if "spatial_loss" in loss_names:
            masked_x = crop_nonzero_region(outputs * x_mask)
            x_feats_mask = self.get_feats(masked_x, all_layers)

            if styles2 is not None:
                s_feats_mask = self.get_feats(styles2, all_layers)
            elif s_mask:
                masked_s = crop_nonzero_region(styles * s_mask)
                s_feats_mask = self.get_feats(masked_s, all_layers)
            else:
                s_feats_mask = self.get_feats(styles, all_layers)

        if 'lum_nnfm_loss' in loss_names:
            lum_outputs = lum_transform(outputs)
            lum_styles = lum_transform(styles)
            lum_x_feats = self.get_feats(lum_outputs, all_layers)
            lum_s_feats = self.get_feats(lum_styles, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        loss_dict = dict([(x, 0.) for x in loss_names])
        for block in blocks:
            layers = block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

            if 'lum_nnfm_loss' in loss_names:
                x_feats = torch.cat([lum_x_feats[ix_map[ix]] for ix in layers], 1)
                s_feats = torch.cat([lum_s_feats[ix_map[ix]] for ix in layers], 1)
                target_feats = nn_feat_replace(x_feats, s_feats)
                loss_dict["lum_nnfm_loss"] += cos_loss(x_feats, target_feats)

            if "nnfm_loss" in loss_names and scale_level is None:
                target_feats = nn_feat_replace(x_feats, s_feats)
                loss_dict["nnfm_loss"] += cos_loss(x_feats, target_feats)

            if "gram_loss" in loss_names:
                loss_dict["gram_loss"] += F.mse_loss(gram_matrix(x_feats), gram_matrix(s_feats))

            if "content_loss" in loss_names:
                content_feats = torch.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
                loss_dict["content_loss"] += F.mse_loss(x_feats, content_feats)

            if "spatial_loss" in loss_names:
                x_mask_feats = torch.cat([x_feats_mask[ix_map[ix]] for ix in layers], 1)
                s_mask_feats = torch.cat([s_feats_mask[ix_map[ix]] for ix in layers], 1)
                loss_dict['spatial_loss'] += cos_loss(x_mask_feats,
                                                      nn_feat_replace(x_mask_feats, s_mask_feats)) * spatial_weight

        if scale_level is not None:
            layer_coef = self.scale_coefs[scale_level]
            for layerindex in all_layers:
                x_feats = x_feats_all[ix_map[layerindex]]
                s_feats = s_feats_all[ix_map[layerindex]]
                if layer_coef[ix_map[layerindex]] != 0:
                    target_feats = nn_feat_replace(x_feats, s_feats)
                    loss_dict["scale_loss"] += cos_loss(x_feats, target_feats) * layer_coef[ix_map[layerindex]]

        return loss_dict


def arap_loss(
        predict_verts: torch.Tensor,
        ref_verts: torch.Tensor,
        faces: torch.LongTensor,
        eps: float = 1e-8,
) -> torch.Tensor:
    edges = torch.cat(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        dim=0,
    )
    edges = torch.sort(edges, dim=1)[0]
    edges = torch.unique(edges, dim=0)
    v0_pred = predict_verts[edges[:, 0]]
    v1_pred = predict_verts[edges[:, 1]]
    v0_ref = ref_verts[edges[:, 0]]
    v1_ref = ref_verts[edges[:, 1]]
    pred_lengths = torch.norm(v0_pred - v1_pred, dim=1, p=2)
    ref_lengths = torch.norm(v0_ref - v1_ref, dim=1, p=2)
    sq_diff = (pred_lengths - ref_lengths).pow(2)
    rel_error = sq_diff / (ref_lengths.pow(2) + eps)
    normalized_loss = rel_error.clamp(max=1.0).mean()

    return normalized_loss
