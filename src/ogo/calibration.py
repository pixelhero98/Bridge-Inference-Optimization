"""Immutable training-only reward moments and within-condition kernel scales."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch


def condition_id(*, prompt=None, image_sha256=None, class_id=None):
    """Hash exact conditioning, including image CONTENT identity for I2V."""
    if class_id is not None:
        if type(class_id) is not int or class_id < 0 or prompt is not None or image_sha256 is not None:
            raise ValueError("Class conditions cannot contain prompt/image fields")
    elif not isinstance(prompt, str) or not prompt:
        raise ValueError("Text/video conditions require a nonempty prompt")
    if image_sha256 is not None:
        if len(image_sha256) != 64 or any(c not in "0123456789abcdef" for c in image_sha256):
            raise ValueError("image_sha256 must identify the exact conditioning image")
    value = {"prompt": prompt, "image_sha256": image_sha256, "class_id": class_id}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_condition_splits(training, calibration, evaluation):
    train, calib, evaluate = map(set, (training, calibration, evaluation))
    if not train or not calib or not evaluate or not calib <= train or train & evaluate:
        raise ValueError("Calibration must be training-only; evaluation conditions must be disjoint")


@dataclass(frozen=True)
class RewardCalibration:
    names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    sample_count: int
    teacher_identity: str
    conditions_sha256: str

    def __post_init__(self):
        for field in ("names", "means", "stds"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if (not self.names or len(set(self.names)) != len(self.names) or
                len(self.means) != len(self.names) or len(self.stds) != len(self.names) or
                any(not isinstance(n, str) or not n for n in self.names) or
                any(not math.isfinite(m) for m in self.means) or
                any(not math.isfinite(s) or s <= 0 for s in self.stds) or
                type(self.sample_count) is not int or self.sample_count < 2 or
                not self.teacher_identity or not self.conditions_sha256):
            raise ValueError("Invalid fixed reward calibration; zero-variance heads require investigation")

    def normalize(self, scores):
        if scores.ndim != 2 or scores.shape[1] != len(self.names):
            raise ValueError("Scores must have shape [draws, declared heads]")
        return (scores - scores.new_tensor(self.means)) / scores.new_tensor(self.stds)

    def identity(self):
        return asdict(self)

    @classmethod
    def fit(cls, scores, names, *, condition_ids, training_ids, evaluation_ids, teacher_identity):
        """Fit population moments once on frozen teacher scores, never per batch."""
        check_condition_splits(training_ids, condition_ids, evaluation_ids)
        if (scores.ndim != 2 or len(scores) < 2 or len(scores) != len(condition_ids) or
                scores.shape[1] != len(names) or scores.requires_grad or not torch.isfinite(scores).all()):
            raise ValueError("Calibration requires finite detached teacher scores and their conditions")
        values = scores.detach().to(device="cpu", dtype=torch.float64)
        digest = hashlib.sha256(json.dumps(list(condition_ids), separators=(",", ":")).encode()).hexdigest()
        return cls(tuple(names), tuple(values.mean(0).tolist()), tuple(values.std(0, correction=0).tolist()),
                   len(values), teacher_identity, digest)


def teacher_bandwidth(groups):
    """sigma = sqrt(median within-condition squared distance / 2).

    groups maps exact condition IDs to detached [independent draws, features].
    No pooled cross-condition distances and no silent zero-distance fallback.
    Provenance/split IDs and this result belong in the adapter identity.
    """
    if not groups:
        raise ValueError("Bandwidth calibration needs teacher groups")
    distances = []
    for features in groups.values():
        if (features.ndim != 2 or len(features) < 2 or features.requires_grad or
                not torch.isfinite(features).all()):
            raise ValueError("Each calibration condition needs >=2 finite frozen teacher draws")
        i, j = torch.triu_indices(len(features), len(features), offset=1, device=features.device)
        distances.append((features[i].double() - features[j].double()).square().sum(-1).cpu())
    median = torch.quantile(torch.cat(distances), 0.5).item()
    if not math.isfinite(median) or median <= 0:
        raise ValueError("Degenerate teacher distances; do not invent a bandwidth floor")
    return math.sqrt(median / 2)


def laplace_bandwidth(groups):
    """Median within-condition Euclidean distance; immutable after calibration."""
    if not groups:
        raise ValueError("Bandwidth calibration needs teacher groups")
    distances = []
    for features in groups.values():
        if (features.ndim != 2 or len(features) < 2 or features.requires_grad or not torch.isfinite(features).all()):
            raise ValueError("Each calibration condition needs >=2 finite frozen teacher draws")
        i, j = torch.triu_indices(len(features), len(features), offset=1, device=features.device)
        distances.append(torch.linalg.vector_norm(features[i].double()-features[j].double(), dim=-1).cpu())
    median = torch.quantile(torch.cat(distances), 0.5).item()
    if not math.isfinite(median) or median <= 0:
        raise ValueError("Degenerate teacher distances; do not invent a bandwidth floor")
    return median
