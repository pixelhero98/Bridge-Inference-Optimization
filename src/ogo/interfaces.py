"""Task boundaries: sampling semantics are distinct from utility baselines."""
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from .objectives import FeatureKernel


@dataclass
class TargetSource:
    """Independent target draws, represented separately in every fixed branch."""
    features: dict[str, torch.Tensor]
    ids: tuple[str, ...]

    def validate(self, initial):
        if not self.features or not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Target source requires unique draw identities and feature branches")
        for value in self.features.values():
            if (value.ndim != 2 or len(value) != len(self.ids) or value.requires_grad or
                    value.dtype != initial.dtype or value.device != initial.device or not torch.isfinite(value).all()):
                raise ValueError("Target features must be finite detached [draws, features] with state dtype/device")


@dataclass
class ContextBatch:
    context: Any
    initial: torch.Tensor
    matching_targets: torch.Tensor | None
    student_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    utility_baseline: torch.Tensor | None = None
    baseline_ids: tuple[str, ...] | None = None
    target_sources: dict[str, TargetSource] | None = None

    def validate(self):
        if self.initial.ndim < 2:
            raise ValueError("States need a draw dimension and matching state shapes")
        if self.initial.dtype not in (torch.float32, torch.float64):
            raise TypeError("Expected float32 or float64 states")
        draws = [(self.student_ids, self.initial)]
        if self.matching_targets is not None:
            if self.target_sources is not None:
                raise ValueError("Select raw targets or named feature sources")
            if self.matching_targets.shape[1:] != self.initial.shape[1:]:
                raise ValueError("Student and matching target state shapes differ")
            if self.matching_targets.dtype != self.initial.dtype or self.matching_targets.device != self.initial.device:
                raise ValueError("Targets must share state dtype/device")
            draws.append((self.target_ids, self.matching_targets))
        elif self.target_ids:
            raise ValueError("Raw target identities require raw targets")
        seen = set(self.student_ids)
        for source in (self.target_sources or {}).values():
            source.validate(self.initial)
            if seen.intersection(source.ids):
                raise ValueError("Matching targets must use independent student/teacher/data streams")
            seen.update(source.ids)
        for ids, tensor in draws:
            if len(ids) != len(tensor) or not ids or len(set(ids)) != len(ids):
                raise ValueError("Every draw needs a unique nonempty role-qualified identity")
            if tensor.requires_grad:
                raise ValueError("Noise and matching targets must be parameter-independent data")
        if set(self.student_ids) & set(self.target_ids):
            raise ValueError("Matching targets must use independent student/teacher streams")
        if self.utility_baseline is not None:
            if self.baseline_ids != self.student_ids or len(self.utility_baseline) != len(self.initial):
                raise ValueError("Paired utility baseline identities must match student draws")
            if self.utility_baseline.requires_grad:
                raise ValueError("Utility baseline must be frozen")
            if (self.utility_baseline.shape != self.initial.shape or self.utility_baseline.dtype != self.initial.dtype
                    or self.utility_baseline.device != self.initial.device):
                raise ValueError("Paired reference must have the student's shape, dtype, and device")
        elif self.baseline_ids is not None:
            raise ValueError("Baseline identities require a baseline")


class TaskAdapter(Protocol):
    """Factories are resolved explicitly as module:callable by ogo.train.

    Frozen modules includes every backbone, decoder, perceptual encoder and
    utility module, even when hidden in callables. identity() must identify
    their weights, preprocessing, dataset, kernels, and sampling conventions.
    """
    correction: nn.Module
    dynamics: Any
    branches: tuple[FeatureKernel, ...]
    frozen_modules: tuple[nn.Module, ...]

    def batch(self, cursor: int, *, target_weights=None, utility_active=True) -> tuple[ContextBatch, ...]: ...
    # Tensor[K] costs, or UtilityValues with per-draw costs and component diagnostics.
    def utility(self, endpoint, context, baseline): ...
    def identity(self) -> dict: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
