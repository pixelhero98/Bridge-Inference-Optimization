"""Paper equations (3), (21), (24), (25), and (41)-(43), in float64.

Shapes: states [context, particle, 2], targets [context, 2], theta [58].
Backbone and feature coefficients are constants, not optimized parameters.
"""

import math

import numpy as np
import torch

from . import rollout as canonical
from .objectives import conditional_loss, kernel_score
from .synthetic import PARAMETERS, backbone, correction, features, feature_branches, utility as utility_adapter


REGIMES = {"utility": (1.0, 0.0), "matching": (0.0, 1.0), "combined": (1.0, 1.0)}
CONTROLS = ("no_backbone_input", "last_step_only", "one_sided_kernel", "extra_ensemble_mean")


def euler_step(theta, x, t, h, contexts, control="full"):
    dynamics = (lambda state, time, c: backbone(state, time, c).detach()) if control == "no_backbone_input" else backbone
    return canonical.euler_step(dynamics, lambda state, time, c: correction(theta, state, time, c),
                                x, t, h, contexts)


def rollout(theta, initial, contexts, steps, control="full", return_states=False):
    canonical.check_steps(steps)
    if control not in ("full",) + CONTROLS:
        raise ValueError(f"Unknown control: {control}")
    dynamics = (lambda x, t, c: backbone(x, t, c).detach()) if control == "no_backbone_input" else backbone
    if control != "last_step_only":
        return canonical.uniform_euler(dynamics, lambda x, t, c: correction(theta, x, t, c),
                                       initial, contexts, steps, return_states=return_states)
    x, states = initial, [initial]
    for k in range(steps):
        if k == steps - 1:
            x = x.detach()
        x = euler_step(theta, x, k / steps, 1.0 / steps, contexts)
        if return_states:
            states.append(x)
    return (x, states) if return_states else x


def frozen_rollout(initial, contexts, steps):
    return canonical.frozen_rollout(backbone, initial, contexts, steps)


def gaussian(a, b):
    """Bandwidth sigma=1; exp(-||a-b||^2 / (2 sigma^2))."""
    return torch.exp(-0.5 * (a - b).square().sum(dim=-1))


def context_components(endpoint, target, contexts, matching=True, one_sided=False):
    """Return utility and score per context, before averaging contexts."""
    k = endpoint.shape[1]
    y = torch.stack((contexts, -0.5 * contexts), dim=-1)
    utility = 0.5 * (endpoint - y[:, None, :]).square().sum(-1).mean(-1)
    score = utility * 0.0
    if matching:
        if k < 2:
            raise ValueError("Conditional kernel matching requires K >= 2")
        kernel = (lambda a, b: gaussian(a, b.detach())) if one_sided else gaussian
        for u, v in zip(features(endpoint), features(target)):
            score = score + 0.5 * kernel_score(u, v[:, None, :], kernel)
    return utility, score


def terminal_objective(endpoint, target, contexts, regime, control="full"):
    utility_weight, matching_weight = REGIMES[regime]
    branches = feature_branches()
    if control == "one_sided_kernel":
        from dataclasses import replace
        branches = tuple(replace(b, kernel=lambda a, v: gaussian(a, v.detach())) for b in branches)
    losses = [conditional_loss(endpoint[i], target[i:i+1], contexts[i], utility_adapter, branches,
                               utility_weight=utility_weight, matching_weight=matching_weight)[0]
              for i in range(len(contexts))]
    return torch.stack(losses).mean()


def objective(theta, initial, target, contexts, steps, regime, control="full"):
    endpoint = rollout(theta, initial, contexts, steps, control)
    return terminal_objective(endpoint, target, contexts, regime, control)


def autograd_gradient(theta, initial, target, contexts, steps, regime, control="full"):
    parameter = theta.detach().clone().requires_grad_(True)
    loss = objective(parameter, initial, target, contexts, steps, regime, control)
    gradient, = torch.autograd.grad(loss, parameter)
    if control == "extra_ensemble_mean":
        gradient = gradient / initial.shape[1]
    return loss.detach(), gradient.detach()


def terminal_covector(endpoint, target, contexts, regime):
    """Analytic Eq. (43), including exactly one particle and context mean.

    This uses analytic Gaussian/tanh derivatives, not autograd on the loss.
    """
    utility_weight, matching_weight = REGIMES[regime]
    groups, k, _ = endpoint.shape
    y = torch.stack((contexts, -0.5 * contexts), dim=-1)
    covector = utility_weight * (endpoint - y[:, None, :])
    if matching_weight:
        if k < 2:
            raise ValueError("Conditional kernel matching requires K >= 2")
        mask = 1.0 - torch.eye(k, dtype=endpoint.dtype, device=endpoint.device)
        m = endpoint.new_tensor([[1.0, 0.5], [-0.25, 1.0]])
        for branch, (u, v) in enumerate(zip(features(endpoint), features(target))):
            diff = u[:, :, None, :] - u[:, None, :, :]
            grad_gg = -diff * torch.exp(-0.5 * diff.square().sum(-1))[..., None]
            cross_diff = u - v[:, None, :]
            grad_gt = -cross_diff * torch.exp(-0.5 * cross_diff.square().sum(-1))[..., None]
            b = (grad_gg * mask[..., None]).sum(-2) / (k - 1) - grad_gt
            pulled = b if branch == 0 else (b * (1.0 - u.square())) @ m
            covector = covector + matching_weight * 0.5 * pulled
    return covector / (k * groups)


def discrete_adjoint(theta, initial, target, contexts, steps, regime):
    """Explicit reverse recursion; each local partial holds its input fixed."""
    with torch.no_grad():
        endpoint, states = rollout(theta, initial, contexts, steps, return_states=True)
        a = terminal_covector(endpoint, target, contexts, regime)
    parameter = theta.detach().clone().requires_grad_(True)
    gradient = torch.zeros_like(parameter)
    for k in reversed(range(steps)):
        x = states[k].detach().requires_grad_(True)
        updated = euler_step(parameter, x, k / steps, 1.0 / steps, contexts)
        a, contribution = torch.autograd.grad(updated, (x, parameter), grad_outputs=a)
        gradient = gradient + contribution
    return gradient.detach()


def loop_objective(endpoint, target, contexts, regime):
    """Independent NumPy scalar loops implementing the ordered Eq. (21) sum."""
    x, v, c = (a.detach().cpu().numpy() for a in (endpoint, target, contexts))
    uw, mw = REGIMES[regime]
    groups, k, _ = x.shape
    if mw and k < 2:
        raise ValueError("Conditional kernel matching requires K >= 2")

    def psi(a, r):
        if r == 0:
            return a
        return np.array([np.tanh(a[0] + 0.5 * a[1]), np.tanh(a[1] - 0.25 * a[0])])

    def kernel(a, b):
        delta = a - b
        return math.exp(-float(np.dot(delta, delta)) / 2.0)

    total = 0.0
    for group in range(groups):
        uterm = sum(0.5 * ((x[group, i, 0] - c[group]) ** 2
                          + (x[group, i, 1] + 0.5 * c[group]) ** 2) for i in range(k)) / k
        score = 0.0
        if mw:
            for r in range(2):
                repulsion = sum(kernel(psi(x[group, i], r), psi(x[group, j], r))
                                for i in range(k) for j in range(k) if i != j)
                attraction = sum(kernel(psi(x[group, i], r), psi(v[group], r)) for i in range(k))
                score += 0.5 * (repulsion / (2 * k * (k - 1)) - attraction / k)
        total += uw * uterm + mw * score
    return total / groups


def scalar_rollout(phi, steps):
    return canonical.uniform_euler(lambda x, t, c: 0.4 * x, lambda x, t, c: phi * x,
                                   torch.ones_like(phi), None, steps)


def scalar_oracle(phi, steps):
    r = 1.0 + (0.4 + phi) / steps
    return r ** steps, (r ** steps - 2.0) * r ** (steps - 1)


def scalar_adjoint(phi, steps):
    states = [1.0]
    for _ in range(steps):
        states.append(states[-1] + (0.4 + phi) * states[-1] / steps)
    a, gradient = states[-1] - 2.0, 0.0
    for k in reversed(range(steps)):
        gradient += states[k] * a / steps
        a *= 1.0 + (0.4 + phi) / steps
    return gradient
