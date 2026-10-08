"""Task boundaries: sampling semantics are distinct from utility baselines."""
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from .objectives import FeatureKernel


@dataclass
class ContextBatch:
    context: Any
    initial: torch.Tensor
    matching_targets: torch.Tensor
    student_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    utility_baseline: torch.Tensor | None = None
    baseline_ids: tuple[str, ...] | None = None

    def validate(self):
        if self.initial.ndim < 2 or self.matching_targets.ndim != self.initial.ndim:
            raise ValueError("States need a draw dimension and matching state shapes")
        if self.matching_targets.shape[1:] != self.initial.shape[1:]:
            raise ValueError("Student and matching target state shapes differ")
        if self.initial.dtype not in (torch.float32, torch.float64):
            raise TypeError("Expected float32 or float64 states")
        if self.matching_targets.dtype != self.initial.dtype or self.matching_targets.device != self.initial.device:
            raise ValueError("Targets must share state dtype/device")
        for ids, tensor in ((self.student_ids, self.initial), (self.target_ids, self.matching_targets)):
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
        elif self.baseline_ids is not None:
            raise ValueError("Baseline identities require a baseline")


class TaskAdapter(Protocol):
    """Factories are resolved explicitly as module:callable by fct.train.

    Frozen modules includes every backbone, decoder, perceptual encoder and
    utility module, even when hidden in callables. identity() must identify
    their weights, preprocessing, dataset, kernels, and sampling conventions.
    """
    correction: nn.Module
    dynamics: Any
    branches: tuple[FeatureKernel, ...]
    frozen_modules: tuple[nn.Module, ...]

    def batch(self, cursor: int) -> tuple[ContextBatch, ...]: ...
    def utility(self, endpoint, context, baseline): ...
    def identity(self) -> dict: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
