"""Strict-FP32 finite-step corrective optimization and operator distillation."""
from .dit import DiTConfig, ResidualDiT
from .noise import keyed_noise
from .objectives import TaskLoss, LinearRamp, paired_advantage, laplace_terms, official_value_proxy_gradient
from .precision import freeze, fp32_context
from .rollout import TimeGrid, Rollout, base_rollout, corrective_rollout, direct_rollout
from .training import Reward, Kernel, TrainingConfig, TrainingContext, TrainingBatch, Trainer

__all__ = ["DiTConfig", "ResidualDiT", "keyed_noise", "TaskLoss", "LinearRamp", "paired_advantage",
           "laplace_terms", "official_value_proxy_gradient", "freeze", "fp32_context", "TimeGrid",
           "Rollout", "base_rollout", "corrective_rollout", "direct_rollout", "Reward", "Kernel",
           "TrainingConfig", "TrainingContext", "TrainingBatch", "Trainer"]
