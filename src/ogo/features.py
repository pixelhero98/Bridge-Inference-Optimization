"""Tensor-only feature adapters. Model loading and pinned preprocessing are explicit."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def freeze(module):
    module.eval().requires_grad_(False)
    return module


def unit_features(features):
    norm = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
    if not torch.isfinite(features).all() or (norm == 0).any():
        raise ValueError("Cannot normalize nonfinite or zero feature vectors")
    return features / norm


class ImagePreprocess(nn.Module):
    """Explicit resize-short-edge / center-crop on decoded NCHW images in [-1,1].

    Parameters must match the pinned encoder processor. No quantization/PIL.
    """
    def __init__(self, *, resize_short, crop_size, mean, std, interpolation, clamp, antialias=True):
        super().__init__()
        if (type(resize_short) is not int or type(crop_size) is not int or
                resize_short < crop_size or crop_size <= 0 or interpolation not in ("bilinear", "bicubic") or
                len(mean) != 3 or len(std) != 3 or type(clamp) is not bool or type(antialias) is not bool or
                any(not math.isfinite(m) for m in mean) or
                any(not math.isfinite(s) or s <= 0 for s in std)):
            raise ValueError("Invalid image preprocessing specification")
        self.resize_short, self.crop_size = resize_short, crop_size
        self.mean, self.std = tuple(mean), tuple(std)
        self.interpolation, self.clamp = interpolation, clamp
        self.antialias = antialias

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected NCHW RGB tensors")
        prefix = images.shape[:-3]
        h, w = images.shape[-2:]
        images = images.reshape(-1, 3, h, w).add(1).div(2)
        if self.clamp:
            images = images.clamp(0, 1)
        size = (self.resize_short, int(w * self.resize_short / h)) if h <= w else (int(h * self.resize_short / w), self.resize_short)
        # TorchVision also returns the input when the spatial size is unchanged.
        # Avoid invoking a redundant CUDA antialias backward, which is not
        # deterministic even though this particular resize is the identity.
        if (h, w) != size:
            images = F.interpolate(images, size=size, mode=self.interpolation, align_corners=False, antialias=self.antialias)
        top, left = ((d - self.crop_size) // 2 for d in size)
        images = images[:, :, top:top+self.crop_size, left:left+self.crop_size]
        images = (images - images.new_tensor(self.mean)[None, :, None, None]) / images.new_tensor(self.std)[None, :, None, None]
        return images.reshape(*prefix, 3, self.crop_size, self.crop_size)


class LatentFeatures(nn.Module):
    def forward(self, states, context=None):
        return states.flatten(1)


class DinoFeatures(nn.Module):
    """Unit-normalized official DINOv2 image class-token features."""
    def __init__(self, decoder, preprocess, encoder):
        super().__init__()
        self.decoder, self.preprocess, self.encoder = decoder, preprocess, encoder
        freeze(self)

    def forward(self, states, context=None):
        images = self.decoder(states)
        if images.ndim != 4:
            raise ValueError("Image DINO requires decoded NCHW images")
        return unit_features(self.encoder.forward_features(self.preprocess(images))["x_norm_clstoken"])
