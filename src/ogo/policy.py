"""Uniform Euler deployment and explicit Gaussian Euler training transitions."""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint


class LowRankWeight(nn.Module):
    """W + BA with frozen W; works for official linear and convolution layers."""
    def __init__(self, weight, rank, scale=1.):
        super().__init__()
        self.shape = weight.shape
        self.a = nn.Parameter(torch.empty(rank, weight[0].numel(), device=weight.device, dtype=weight.dtype))
        self.b = nn.Parameter(torch.zeros(weight.shape[0], rank, device=weight.device, dtype=weight.dtype))
        nn.init.normal_(self.a, std=1 / math.sqrt(self.a.shape[1]))
        self.enabled = True
        self.scale = scale

    def forward(self, weight):
        return weight + self.scale*(self.b @ self.a).reshape(self.shape) if self.enabled else weight


def install_lora(model, rank, scope, scale=1.):
    if scope not in ('attention', 'all_linear_conv'):
        raise ValueError('Unknown LoRA scope')
    model.requires_grad_(False)
    installed = []
    for name, module in list(model.named_modules()):
        weight = module._parameters.get('weight')
        if weight is None or weight.ndim not in (2, 4):
            continue
        if scope == 'attention' and name.rsplit('.', 1)[-1] not in ('qkv', 'proj'):
            continue
        # Embeddings and normalization are not linear/convolution weight adapters.
        if isinstance(module, nn.Embedding):
            continue
        parametrize.register_parametrization(module, 'weight', LowRankWeight(weight, rank, scale))
        installed.append(name)
    if not installed:
        raise ValueError('No declared LoRA targets found')
    return installed


@contextmanager
def reference_policy(model):
    adapters = [m for m in model.modules() if isinstance(m, LowRankWeight)]
    previous = [m.enabled for m in adapters]
    try:
        for m in adapters:
            m.enabled = False
        with torch.no_grad():
            yield
    finally:
        for m, state in zip(adapters, previous):
            m.enabled = state


def euler(policy, initial, labels, steps, *, backward_steps=None, rematerialize=True):
    """DRaFT-K truncates backward depth; the forward still uses all N steps."""
    if steps not in (16, 32) and not (isinstance(steps, int) and steps > 0):
        raise ValueError('Positive integer Euler horizon required')
    depth = steps if backward_steps is None else backward_steps
    if not 0 <= depth <= steps:
        raise ValueError('Backward depth outside trajectory')
    x = initial
    for k in range(steps):
        active = torch.is_grad_enabled() and k >= steps-depth
        if k == steps-depth:
            x = x.detach()
        def step(z, t=k/steps):
            return z + policy(z, t, labels)/steps
        if active:
            x = checkpoint(step, x, use_reentrant=False) if rematerialize else step(x)
        else:
            with torch.no_grad():
                x = step(x)
    return x


def transition_mean(state, velocity, time, steps, backbone, noise_level):
    """Noise-to-data time. EDM: a=(1+80t)/81; SiT: a=t; b=1-a.

    For a linear interpolant v=a'*(x_data-z), score=(a*v/a'-x)/b.
    Forward marginal-preserving drift is v+g²*score/2. At SiT t=0
    g² uses the next grid value in the denominator, as Flow-GRPO does.
    The exact t=0 score is -x; no division by zero is hidden by a floor.
    """
    if not 0 <= time < 1 or steps <= 0 or noise_level < 0:
        raise ValueError('Invalid Euler transition')
    if backbone == 'edm64':
        speed, a = 80/81, (1+80*time)/81
    elif backbone == 'sit256':
        speed, a = 1., time
    else:
        raise ValueError('Unknown interpolant')
    b = 1-a
    g2 = noise_level**2 * b / (a if a > 0 else 1/steps)
    score = (a*velocity/speed-state)/b
    return state+(velocity+.5*g2*score)/steps, math.sqrt(g2/steps)


def gaussian_log_prob(sample, mean, std, reduction='mean'):
    if not math.isfinite(std) or std <= 0:
        raise ValueError('Deterministic transitions have no Gaussian likelihood')
    value = -.5*((sample.detach()-mean)/std).square()-math.log(std)-.5*math.log(2*math.pi)
    if reduction == 'mean':
        return value.flatten(1).mean(1)
    if reduction == 'sum':
        return value.flatten(1).sum(1)
    raise ValueError('Declare likelihood reduction')


@torch.no_grad()
def collect(policy, initial, labels, steps, backbone, noise_level, generator):
    x = initial
    states, means, logps = [x.detach()], [], []
    for k in range(steps):
        mean, std = transition_mean(x, policy(x, k/steps, labels), k/steps, steps, backbone, noise_level)
        x = mean + std*torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        states.append(x.detach())
        means.append(mean.detach())
        logps.append(gaussian_log_prob(x, mean, std))
    return {'states': states, 'means': means, 'logps': torch.stack(logps, 1), 'labels': labels.detach()}
