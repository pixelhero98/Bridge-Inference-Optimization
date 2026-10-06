"""Paired reward and conditional Laplace objectives, independent of the task."""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .precision import check_tensor_tree


@dataclass(frozen=True)
class TaskLoss:
    kind: str
    tau: float
    alpha: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be positive and finite")
        if self.kind not in ("softplus", "softplus_linear", "regret"):
            raise ValueError("Unknown task loss")
        if self.kind == "softplus_linear":
            if self.alpha is None or not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
                raise ValueError("Mixed loss requires finite 0 < alpha < 1")
        elif self.alpha is not None:
            raise ValueError("Pure softplus/regret requires alpha=None")

    def __call__(self, advantage):
        check_tensor_tree(advantage)
        if advantage.ndim != 1 or advantage.numel() == 0:
            raise ValueError("Expected one advantage per draw")
        if self.kind == "regret":
            return (self.tau * F.softplus(-advantage / self.tau)).mean()
        utility = self.tau * F.softplus(advantage / self.tau)
        if self.kind == "softplus_linear":
            utility = (1 - self.alpha) * utility + self.alpha * advantage
        return -utility.mean()


def paired_advantage(corrected_scores, base_scores, *, scale, direction="maximize"):
    check_tensor_tree((corrected_scores, base_scores))
    if corrected_scores.ndim != 1 or corrected_scores.shape != base_scores.shape:
        raise ValueError("Expected paired score vectors, one scalar per draw")
    scale = torch.as_tensor(scale, dtype=torch.float32, device=corrected_scores.device).detach()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale <= 0:
        raise ValueError("Frozen reward scale must be positive and finite")
    delta = corrected_scores - base_scores.detach()
    if direction == "minimize":
        delta = -delta
    elif direction != "maximize":
        raise ValueError("direction must be maximize or minimize")
    return delta / scale


def laplace_terms(generated, target, *, bandwidth, mode="paired", repulsion=True):
    """Single context; target-only constants are omitted.

    Repulsion is 0.5 times the mean off-diagonal generated similarity. Paired
    attraction uses matching draw indices; ensemble uses all pairs; observed
    uses exactly one observed target rather than duplicating it into draws.
    """
    check_tensor_tree((generated, target))
    if generated.ndim != 2 or target.ndim != 2 or generated.shape[1] != target.shape[1]:
        raise ValueError("Expected [draws, features] and [targets, features]")
    if len(generated) == 0 or len(target) == 0 or not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Nonempty features and a positive finite bandwidth required")
    if mode == "paired" and target.shape != generated.shape:
        raise ValueError("Paired targets must match generated draws one-to-one")
    if mode == "observed" and len(target) != 1:
        raise ValueError("Observed mode requires exactly one target")
    if mode not in ("paired", "ensemble", "observed"):
        raise ValueError("Unknown attraction mode")
    target = target.detach()
    if mode in ("paired", "observed"):
        attraction = torch.exp(-torch.linalg.vector_norm(generated - target, dim=-1) / bandwidth).mean()
    else:
        distance = torch.linalg.vector_norm(generated[:, None] - target[None], dim=-1)
        attraction = torch.exp(-distance / bandwidth).mean()
    if repulsion:
        if len(generated) < 2:
            raise ValueError("Repulsion requires at least two generated draws")
        if len(generated) == 2:
            repel = .5 * torch.exp(-torch.linalg.vector_norm(generated[0] - generated[1]) / bandwidth)
        else:
            repel = .5 * torch.exp(-torch.pdist(generated, p=2) / bandwidth).mean()
    else:
        repel = generated.new_zeros(())
    return {"repulsion": repel, "attraction": attraction, "loss": repel - attraction}


class _OfficialForwardProxyBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proxy, official):
        return official.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def official_value_proxy_gradient(proxy, official):
    """Exact supplied forward score, with derivative only through its proxy."""
    check_tensor_tree((proxy, official))
    if proxy.shape != official.shape or proxy.device != official.device:
        raise ValueError("Official and proxy scores must have the same shape/device")
    return _OfficialForwardProxyBackward.apply(proxy, official.detach())


@dataclass(frozen=True)
class LinearRamp:
    warmup_updates: int
    ramp_updates: int

    def __post_init__(self):
        if self.warmup_updates < 0 or self.ramp_updates <= 0:
            raise ValueError("Invalid ramp schedule")

    def __call__(self, update):
        return min(1., max(0., (update - self.warmup_updates) / self.ramp_updates))
