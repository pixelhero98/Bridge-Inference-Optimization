"""Conditional kernel score in any fixed differentiable feature spaces."""
from dataclasses import dataclass
from collections.abc import Callable
import math

import torch


def nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class GaussianKernel:
    bandwidth: float

    def __post_init__(self):
        if not math.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive and finite")

    def __call__(self, left, right):
        return torch.exp(-0.5 * ((left - right) / self.bandwidth).square().sum(-1))


@dataclass(frozen=True)
class FeatureKernel:
    """feature(states, context) -> [draws, features]; kernel is symmetric.

    A branch may operate on latent states, decoded pixels, DINO embeddings, or
    another fixed geometry. Parameters must be frozen, input derivatives kept.
    Weights are applied as supplied, without implicit normalization.
    """
    name: str
    feature: Callable
    kernel: Callable
    weight: float

    def __post_init__(self):
        if not self.name:
            raise ValueError("Feature branch needs a name")
        nonnegative(self.weight, "feature weight")


def kernel_score(generated, target, kernel):
    """Eq. (21), generalized to M independent targets; leading groups supported.

    Shapes [..., K, F], [..., M, F]. No cross-context pairs or self-pairs.
    The target-only constant is omitted, so scores can be negative.
    """
    if (generated.ndim < 2 or target.ndim != generated.ndim or
            generated.shape[:-2] != target.shape[:-2] or generated.shape[-1] != target.shape[-1]):
        raise ValueError("Expected matching context dimensions and feature width")
    if generated.device != target.device or generated.dtype != target.dtype:
        raise ValueError("Generated and target features must share dtype/device")
    k, m = generated.shape[-2], target.shape[-2]
    if k < 2:
        raise ValueError("Conditional kernel matching requires K >= 2")
    if m < 1:
        raise ValueError("Matching needs at least one independent target")
    indices = torch.arange(k, device=generated.device)
    left, right = torch.meshgrid(indices, indices, indexing="ij")
    mask = left != right
    # Index off-diagonal pairs before calling potentially singular kernels.
    gg = kernel(generated[..., left[mask], :], generated[..., right[mask], :])
    gt = kernel(generated[..., :, None, :], target.detach()[..., None, :, :])
    if gg.shape != (*generated.shape[:-2], k * (k - 1)) or gt.shape != (*generated.shape[:-2], k, m):
        raise ValueError("Kernel must return one scalar per pair")
    return gg.sum(-1) / (2 * k * (k - 1)) - gt.sum((-2, -1)) / (k * m)


def conditional_loss(endpoint, target, context, utility, branches, *,
                     utility_weight=1.0, matching_weight=1.0, utility_baseline=None):
    """One context. Caller averages context losses, not all particles pooled.

    utility(endpoint, context, utility_baseline) returns K losses to MINIMIZE.
    A maximized reward is represented by U=-reward or a declared transform.
    """
    nonnegative(utility_weight, "utility_weight")
    nonnegative(matching_weight, "matching_weight")
    if endpoint.ndim < 2 or len(endpoint) == 0:
        raise ValueError("Endpoint must contain a nonempty draw dimension")
    if len({b.name for b in branches}) != len(branches):
        raise ValueError("Feature branch names must be unique")
    loss, terms = endpoint.sum() * 0.0, {}
    if utility_weight:
        if utility is None:
            raise ValueError("Active utility requires a utility adapter")
        baseline = utility_baseline.detach() if utility_baseline is not None else None
        values = utility(endpoint, context, baseline)
        if values.shape != (len(endpoint),) or values.dtype != endpoint.dtype or values.device != endpoint.device:
            raise ValueError("Utility must return one loss per draw on the endpoint dtype/device")
        terms["utility"] = values.mean()
        loss = loss + utility_weight * terms["utility"]
    if matching_weight:
        if target is None or not any(b.weight > 0 for b in branches):
            raise ValueError("Active matching requires targets and positive feature weight")
        matching = endpoint.sum() * 0.0
        for branch in branches:
            if not branch.weight:
                continue
            generated_features = branch.feature(endpoint, context)
            with torch.no_grad():
                target_features = branch.feature(target.detach(), context)
            if generated_features.ndim != 2 or generated_features.shape[0] != len(endpoint):
                raise ValueError("Feature adapter must return [draws, features]")
            if target_features.ndim != 2 or target_features.shape[0] != len(target):
                raise ValueError("Target feature adapter must preserve draws")
            terms[f"matching/{branch.name}"] = kernel_score(generated_features, target_features, branch.kernel)
            matching = matching + branch.weight * terms[f"matching/{branch.name}"]
        terms["matching"] = matching
        loss = loss + matching_weight * matching
    if not utility_weight and not matching_weight:
        raise ValueError("At least one objective must be active")
    return loss, terms
