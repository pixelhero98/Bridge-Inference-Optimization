"""Independent analytic/loop oracles for the new Laplace mixture regression.

Historical Gaussian E0 fixtures and oracles remain in verification_cases.py.
"""
import math
import numpy as np
import torch

from .interfaces import TargetSource
from .objectives import FeatureKernel, LaplaceKernel, conditional_loss
from .synthetic import features, utility
from .verification_cases import REGIMES, rollout, euler_step


def branches(control="full"):
    kernel = LaplaceKernel(1.)
    if control == "one_sided_kernel":
        kernel = lambda a, b: LaplaceKernel(1.)(a, b.detach())
    return tuple(FeatureKernel(str(r), lambda x, c, r=r: features(x)[r], kernel, .5) for r in range(2))


def terminal(endpoint, targets, contexts, regime, rho, control="full"):
    uw, mw = REGIMES[regime]
    losses = []
    for g, c in enumerate(contexts):
        sources = {name: TargetSource({str(r): f for r, f in enumerate(features(y[g]))}, ())
                   for name, y in targets.items()}
        losses.append(conditional_loss(endpoint[g], None, c, utility, branches(control),
            utility_weight=uw, matching_weight=mw, target_sources=sources,
            target_weights={"teacher": 1-rho, "ground_truth": rho})[0])
    return torch.stack(losses).mean()


def objective(theta, initial, targets, contexts, steps, regime, rho, control="full"):
    return terminal(rollout(theta, initial, contexts, steps, control), targets, contexts, regime, rho, control)


def gradient(theta, initial, targets, contexts, steps, regime, rho, control="full"):
    p = theta.detach().clone().requires_grad_(True)
    loss = objective(p, initial, targets, contexts, steps, regime, rho, control)
    grad, = torch.autograd.grad(loss, p)
    if control == "extra_ensemble_mean":
        grad = grad / initial.shape[1]
    return loss.detach(), grad.detach()


def covector(endpoint, targets, contexts, regime, rho):
    """Analytic Laplace particle covectors, including one K and context mean."""
    uw, mw = REGIMES[regime]
    groups, k, _ = endpoint.shape
    desired = torch.stack((contexts, -.5*contexts), -1)
    result = uw * (endpoint-desired[:, None])
    if mw:
        matrix = endpoint.new_tensor([[1., .5], [-.25, 1.]])
        mask = 1-torch.eye(k, dtype=endpoint.dtype, device=endpoint.device)
        def derivative(delta):
            radius = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            # Explicit convention, used only in the analytic oracle, never smoothed.
            unit = torch.where(radius > 0, delta / torch.where(radius > 0, radius, 1.), 0.)
            return -torch.exp(-radius)*unit
        for r, a in enumerate(features(endpoint)):
            force = (derivative(a[:, :, None]-a[:, None])*mask[None, :, :, None]).sum(-2)/(k-1)
            for name, weight in (("teacher", 1-rho), ("ground_truth", rho)):
                b = features(targets[name])[r]
                force = force - weight*derivative(a[:, :, None]-b[:, None]).mean(-2)
            pulled = force if r == 0 else (force*(1-a.square())) @ matrix
            result = result + .5*mw*pulled
    return result/(groups*k)


def adjoint(theta, initial, targets, contexts, steps, regime, rho):
    with torch.no_grad():
        endpoint, states = rollout(theta, initial, contexts, steps, return_states=True)
        a = covector(endpoint, targets, contexts, regime, rho)
    p = theta.detach().clone().requires_grad_(True)
    result = torch.zeros_like(p)
    for k in reversed(range(steps)):
        x = states[k].detach().requires_grad_(True)
        updated = euler_step(p, x, k/steps, 1/steps, contexts)
        a, contribution = torch.autograd.grad(updated, (x, p), grad_outputs=a)
        result = result + contribution
    return result.detach()


def loop_value(endpoint, targets, contexts, regime, rho):
    """NumPy scalar loops: unequal teacher/data counts, no tensor loss reuse."""
    x = endpoint.detach().cpu().numpy()
    target = {n: y.detach().cpu().numpy() for n, y in targets.items()}
    c = contexts.detach().cpu().numpy()
    uw, mw = REGIMES[regime]
    def feature(a, r):
        return a if r == 0 else np.array([np.tanh(a[0]+.5*a[1]), np.tanh(a[1]-.25*a[0])])
    def kernel(a, b):
        return math.exp(-math.sqrt(sum(float(v)**2 for v in a-b)))
    values = []
    for g in range(len(c)):
        k = len(x[g])
        value = uw*sum(.5*((a[0]-c[g])**2+(a[1]+.5*c[g])**2) for a in x[g])/k
        if mw:
            for r in range(2):
                score = sum(kernel(feature(a,r),feature(b,r)) for i,a in enumerate(x[g])
                            for j,b in enumerate(x[g]) if i != j)/(2*k*(k-1))
                for name, w in (("teacher", 1-rho), ("ground_truth", rho)):
                    score -= w*sum(kernel(feature(a,r),feature(b,r)) for a in x[g]
                                   for b in target[name][g])/(k*len(target[name][g]))
                value += .5*mw*score
        values.append(value)
    return float(sum(values)/len(values))
