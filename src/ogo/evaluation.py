"""Held-out conditional statistics, distinct from the half-scaled training score."""
import torch
from .objectives import validate_mixture_weights


@torch.no_grad()
def mixture_mmd2(generated, sources, weights, kernel):
    """Full MMD against independent target sources, including cross-source terms."""
    validate_mixture_weights(weights)
    a, b = _pairs(generated)
    result = kernel(a, b).mean()
    active = [(name, w) for name, w in weights.items() if w > 0]
    for i, (name, weight) in enumerate(active):
        target = sources[name]
        c, d = _pairs(target)
        if target.shape[-1] != generated.shape[-1] or target.dtype != generated.dtype or target.device != generated.device:
            raise ValueError("All mixture sources must share generated feature geometry")
        result = result + weight**2 * kernel(c, d).mean()
        result = result - 2*weight*kernel(generated[:, None], target[None]).mean()
        for other, other_weight in active[:i]:
            result = result + 2*weight*other_weight*kernel(target[:, None], sources[other][None]).mean()
    return result


def _pairs(features):
    if features.ndim != 2 or len(features) < 2 or not torch.isfinite(features).all():
        raise ValueError("Evaluation requires >=2 finite draws per condition")
    i, j = torch.triu_indices(len(features), len(features), offset=1, device=features.device)
    return features[i], features[j]


@torch.no_grad()
def unbiased_mmd2(generated, target, kernel):
    """Full U-statistic: self_x + self_y - 2*cross; negative estimates are retained."""
    a, b = _pairs(generated)
    c, d = _pairs(target)
    if (generated.shape[1] != target.shape[1] or generated.device != target.device or
            generated.dtype != target.dtype):
        raise ValueError("Generated and teacher evaluation features must share geometry")
    return kernel(a, b).mean() + kernel(c, d).mean() - 2 * kernel(generated[:, None], target[None, :]).mean()


@torch.no_grad()
def conditional_mmd2(generated, target, kernel):
    """Mappings use exact class/prompt/image+prompt identities, averaged equally."""
    if not generated or generated.keys() != target.keys():
        raise ValueError("Generated and teacher groups must have identical nonempty condition IDs")
    per_condition = {key: unbiased_mmd2(generated[key], target[key], kernel) for key in generated}
    return torch.stack(list(per_condition.values())).mean(), per_condition


@torch.no_grad()
def conditional_diversity(groups, distance):
    """Mean unordered off-diagonal distance, then equal averaging over conditions.

    Use LPIPS-AlexNet for images, squared Euclidean distance of unit I3D pooled
    embeddings for video. Higher is not automatically better generation.
    """
    if not groups:
        raise ValueError("Diversity needs condition groups")
    per_condition = {}
    for key, samples in groups.items():
        if samples.ndim < 2 or len(samples) < 2:
            raise ValueError("Diversity needs >=2 independent outputs per condition")
        i, j = torch.triu_indices(len(samples), len(samples), offset=1, device=samples.device)
        values = distance(samples[i], samples[j])
        if values.numel() != len(i) or not torch.isfinite(values).all():
            raise ValueError("Distance must return a finite scalar for each distinct pair")
        per_condition[key] = values.mean()
    return torch.stack(list(per_condition.values())).mean(), per_condition
