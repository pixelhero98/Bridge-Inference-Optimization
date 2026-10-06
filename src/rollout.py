"""Finite-step Euler correction and standalone next-state inference."""
from contextlib import nullcontext
from dataclasses import dataclass
from collections.abc import Callable
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .precision import check_tensor_tree, checkpoint_contexts, fp32_context, freeze


@dataclass(frozen=True)
class TimeGrid:
    points: tuple[float, ...]

    def __post_init__(self):
        points = tuple(float(t) for t in self.points)
        object.__setattr__(self, "points", points)
        if len(points) < 2 or not all(math.isfinite(t) for t in points):
            raise ValueError("TimeGrid needs at least two finite points")
        deltas = [b - a for a, b in zip(points, points[1:])]
        if not (all(d > 0 for d in deltas) or all(d < 0 for d in deltas)):
            raise ValueError("TimeGrid must be strictly monotone with nonzero steps")

    @property
    def steps(self):
        return len(self.points) - 1

    def intervals(self):
        return zip(self.points, self.points[1:])


@dataclass
class Rollout:
    states: tuple[torch.Tensor, ...]
    base_evaluations: int
    operator_evaluations: int

    @property
    def final(self):
        return self.states[-1]


def _prepare(state, conditioning, dynamics=None):
    check_tensor_tree((state, conditioning))
    if not state.is_floating_point() or state.ndim < 2 or len(state) == 0:
        raise ValueError("State must be a nonempty batch-first FP32 tensor")
    if isinstance(dynamics, nn.Module):
        freeze(dynamics)


def _velocity(dynamics, state, start, conditioning):
    value = dynamics(state, start, conditioning)
    check_tensor_tree(value)
    if value.shape != state.shape or value.device != state.device:
        raise ValueError("Dynamics must return an FP32 velocity with the state's shape/device")
    return value


def base_rollout(dynamics: Callable, initial, conditioning, grid: TimeGrid):
    """Reference-only path. No parameters or states receive gradients."""
    with fp32_context(), torch.no_grad():
        _prepare(initial, conditioning, dynamics)
        state = initial.detach().clone()
        states = [state]
        for start, end in grid.intervals():
            state = state + (end - start) * _velocity(dynamics, state, start, conditioning)
            check_tensor_tree(state)
            states.append(state)
        return Rollout(tuple(states), grid.steps, 0)


def corrective_rollout(dynamics: Callable, corrector, initial, conditioning, grid: TimeGrid,
                       *, strength=1.0, checkpoint_dynamics=True, cpu_offload=False):
    """Each DiT residual acts on the recursively generated base next state.

    Corrector implements delta(base_next, start, end, conditioning). No velocity
    correction, teacher forcing, detached state, or division by timestep.
    """
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    with fp32_context():
        _prepare(initial, conditioning, dynamics)
        offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if cpu_offload else nullcontext()
        with offload:
            state, states = initial, [initial]
            for start, end in grid.intervals():
                # Bind the step now: backward recomputes after the loop finishes.
                def velocity(x, step_start=start):
                    return _velocity(dynamics, x, step_start, conditioning)
                v = (checkpoint(velocity, state, use_reentrant=False, context_fn=checkpoint_contexts)
                     if checkpoint_dynamics and torch.is_grad_enabled() else velocity(state))
                base_next = state + (end - start) * v
                residual = corrector.delta(base_next, start, end, conditioning)
                check_tensor_tree(residual)
                if residual.shape != state.shape or residual.device != state.device:
                    raise ValueError("Corrector residual must have the state's shape/device")
                state = base_next + strength * residual
                check_tensor_tree(state)
                states.append(state)
            return Rollout(tuple(states), grid.steps, grid.steps)


def direct_rollout(student, initial, conditioning, grid: TimeGrid, *, cpu_offload=False):
    """Student returns next state, including its identity skip. No teacher argument."""
    with fp32_context():
        _prepare(initial, conditioning)
        offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if cpu_offload else nullcontext()
        with offload:
            state, states = initial, [initial]
            for start, end in grid.intervals():
                next_state = student(state, start, end, conditioning)
                check_tensor_tree(next_state)
                if next_state.shape != state.shape or next_state.device != state.device:
                    raise ValueError("Student must return the state's shape/device")
                state = next_state
                states.append(state)
            return Rollout(tuple(states), 0, grid.steps)
