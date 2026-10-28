"""Differentiable frozen reward adapters; no model downloads or scoring services."""
import math

import torch
from torch import nn

from .features import freeze, unit_features
from .objectives import UtilityValues


class PairedLPIPS(nn.Module):
    """LPIPS v0.1 VGG supplied by the caller, decoded RGB range [-1,1].

    baseline is the SAME-NOISE frozen teacher endpoint, not an independent
    matching target. E1 uses image_size=256 and eval-mode LPIPS (no dropout).
    """
    def __init__(self, decoder, lpips_model, *, image_size=256):
        super().__init__()
        self.decoder, self.lpips_model, self.image_size = decoder, lpips_model, image_size
        freeze(self)

    def forward(self, states, context, baseline):
        if baseline is None or baseline.shape != states.shape:
            raise ValueError("Paired LPIPS requires same-noise teacher endpoints with matching shape")
        images = self.decoder(states)
        with torch.no_grad():
            reference = self.decoder(baseline.detach())
        expected = (len(states), 3, self.image_size, self.image_size)
        if images.shape != expected or reference.shape != expected:
            raise ValueError("LPIPS requires decoded RGB images at the declared resolution")
        # Clamping is an explicit part of the decoded LPIPS objective, not uint8 conversion.
        values = self.lpips_model(images.clamp(-1, 1), reference.clamp(-1, 1))
        if values.numel() != len(states) or values.shape[0] != len(states):
            raise ValueError("LPIPS must return exactly one distance per draw")
        costs = values.reshape(len(states))
        return UtilityValues(costs, {"lpips_vgg": costs, "reward": -costs})
