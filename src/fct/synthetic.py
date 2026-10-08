"""Small smooth adapter for correctness and training smoke tests, not a benchmark."""
import hashlib
import json
import math

import torch
from torch import nn

from .interfaces import ContextBatch
from .objectives import FeatureKernel, GaussianKernel
from .rollout import frozen_rollout

PARAMETERS = 58


def unpack(theta):
    if theta.shape != (PARAMETERS,):
        raise ValueError("theta must contain exactly 58 correction parameters")
    return theta[:32].reshape(8, 4), theta[32:40], theta[40:56].reshape(2, 8), theta[56:]


def backbone(x, t, contexts):
    a = x.new_tensor([[-0.3, -0.5], [0.4, -0.2]])
    forcing = x.new_tensor([math.cos(math.pi * t), math.sin(math.pi * t)])
    return x @ a.T + 0.2 * torch.tanh(x) + 0.1 * contexts[:, None, None] * forcing


def correction(theta, x, t, contexts):
    w1, b1, w2, b2 = unpack(theta)
    inputs = torch.cat((x, torch.full_like(x[..., :1], t),
                        contexts[:, None, None].expand_as(x[..., :1])), dim=-1)
    return torch.tanh(inputs @ w1.T + b1) @ w2.T + b2


def features(x):
    m = x.new_tensor([[1.0, 0.5], [-0.25, 1.0]])
    return x, torch.tanh(x @ m.T)


def utility(x, context, baseline=None):
    target = torch.stack((context, -0.5 * context))
    return 0.5 * (x - target).square().sum(-1)


def feature_branches():
    return (FeatureKernel("identity", lambda x, c: x, GaussianKernel(1.0), 0.5),
            FeatureKernel("nonlinear", lambda x, c: features(x)[1], GaussianKernel(1.0), 0.5))


def keyed_noise(shape, *, seed, role, cursor, context, draw, dtype, device):
    """Role-separated deterministic streams, independent of batching/resume."""
    key = json.dumps([seed, role, cursor, context, draw], separators=(",", ":"))
    derived = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(derived)
    return torch.randn(shape, generator=generator, dtype=dtype).to(device), key


class Correction(nn.Module):
    def __init__(self, seed, dtype, device, nonzero=False):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        theta = torch.zeros(PARAMETERS, dtype=dtype)
        theta[:32] = torch.randn(32, generator=generator, dtype=dtype) * 0.5
        if nonzero:
            theta[40:] = torch.randn(18, generator=generator, dtype=dtype) * 0.1
        self.theta = nn.Parameter(theta.to(device))

    def forward(self, x, t, context):
        return correction(self.theta, x[None], t, context.reshape(1))[0]


class SyntheticTask:
    def __init__(self, options, *, seed, dtype, device, steps, teacher_steps):
        allowed = {"particles", "targets", "contexts", "paired_baseline", "nonzero"}
        if set(options) - allowed:
            raise ValueError(f"Unknown synthetic options: {set(options) - allowed}")
        self.particles, self.targets = options.get("particles", 4), options.get("targets", 1)
        if any(type(n) is not int or n < 1 for n in (self.particles, self.targets)):
            raise ValueError("Synthetic draw counts must be positive integers")
        self.contexts = options.get("contexts", [-1.0, 1.0])
        if not self.contexts or len(set(self.contexts)) != len(self.contexts) or not all(math.isfinite(c) for c in self.contexts):
            raise ValueError("Use distinct finite synthetic contexts")
        self.options, self.seed, self.dtype, self.device = dict(options), seed, dtype, device
        self.steps, self.teacher_steps = steps, teacher_steps
        self.correction = Correction(seed, dtype, device, options.get("nonzero", False))
        self.branches, self.frozen_modules = feature_branches(), ()

    @staticmethod
    def dynamics(x, t, context):
        return backbone(x[None], t, context.reshape(1))[0]

    utility = staticmethod(utility)

    def identity(self):
        return {"adapter": "synthetic-smooth2d-v1", "options": self.options,
                "features": ["identity", "tanh-linear"], "bandwidths": [1., 1.],
                "feature_weights": [0.5, 0.5], "teacher": "frozen-smooth2d",
                "streams": "sha256(seed,role,cursor,context,draw)",
                "utility": "half-squared-distance-to-(c,-c/2)"}

    def batch(self, cursor):
        batches = []
        for context in self.contexts:
            c = torch.tensor(context, dtype=self.dtype, device=self.device)
            def draws(role, count):
                pairs = [keyed_noise((2,), seed=self.seed, role=role, cursor=cursor,
                                    context=context, draw=i, dtype=self.dtype, device=self.device)
                         for i in range(count)]
                return torch.stack([p[0] for p in pairs]), tuple(p[1] for p in pairs)
            initial, student_ids = draws("student", self.particles)
            teacher_initial, target_ids = draws("matching-teacher", self.targets)
            target = frozen_rollout(self.dynamics, teacher_initial, c, self.teacher_steps)
            baseline = (frozen_rollout(self.dynamics, initial, c, self.teacher_steps)
                        if self.options.get("paired_baseline", False) else None)
            batches.append(ContextBatch(c, initial, target, student_ids, target_ids,
                                        baseline, student_ids if baseline is not None else None))
        return tuple(batches)

    def state_dict(self):
        return {}  # Sampling is a pure function of the committed cursor and identities.

    def load_state_dict(self, state):
        if state:
            raise ValueError("Synthetic adapter has no mutable sampler state")


def make_task(options, **kwargs):
    return SyntheticTask(options, **kwargs)
