"""Finite-step velocity correction; no displacement or standalone-student API."""
from .rollout import OPERATOR, euler_step, frozen_rollout, uniform_euler
from .objectives import FeatureKernel, GaussianKernel, conditional_loss, kernel_score
from .interfaces import ContextBatch, TaskAdapter

__version__ = "0.2.0"
__all__ = ["OPERATOR", "euler_step", "frozen_rollout", "uniform_euler",
           "FeatureKernel", "GaussianKernel", "conditional_loss", "kernel_score",
           "ContextBatch", "TaskAdapter"]
