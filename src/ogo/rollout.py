"""The prescribed uniform Euler composition on normalized time [0, 1]."""
from contextlib import nullcontext

import torch
from torch.utils.checkpoint import checkpoint

OPERATOR = "uniform-euler-velocity-v1"


def check_steps(steps):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")


def check_state(value, reference=None):
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise TypeError("States and velocities must be float32 or float64 tensors")
    if value.numel() == 0:
        raise ValueError("Empty state")
    if reference is not None and (value.shape != reference.shape or value.dtype != reference.dtype
                                  or value.device != reference.device):
        raise ValueError("Velocity must preserve state shape, dtype, and device")


def euler_step(dynamics, correction, state, time, step_size, context, *, checkpoint_dynamics=False):
    """Both velocities are evaluated at state, then multiplied by h exactly once."""
    def base(x):
        return dynamics(x, time, context)
    velocity = (checkpoint(base, state, use_reentrant=False)
                if checkpoint_dynamics and torch.is_grad_enabled() else base(state))
    residual_velocity = correction(state, time, context)
    check_state(velocity, state)
    check_state(residual_velocity, state)
    return state + step_size * (velocity + residual_velocity)


def uniform_euler(dynamics, correction, initial, context, steps, *,
                  checkpoint_dynamics=False, cpu_offload=False, return_states=False):
    """Differentiate the full trajectory, including frozen input Jacobians.

    Adapters provide deterministic normalized-time velocities. Freeze their
    parameters before calling; never put generated backbone calls in no_grad.
    """
    check_steps(steps)
    check_state(initial)
    offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if cpu_offload else nullcontext()
    with offload:
        state, states = initial, [initial]
        for k in range(steps):
            state = euler_step(dynamics, correction, state, k / steps, 1.0 / steps,
                               context, checkpoint_dynamics=checkpoint_dynamics)
            if return_states:
                states.append(state)
    return (state, states) if return_states else state


def frozen_rollout(dynamics, initial, context, steps):
    """Parameter-independent teacher or utility reference; same uniform operator."""
    with torch.no_grad():
        return uniform_euler(dynamics, lambda x, t, c: torch.zeros_like(x),
                             initial.detach(), context, steps)
