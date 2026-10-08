"""The 159,196,832-parameter DiT, with domain-independent state layouts."""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .precision import check_module, check_tensor_tree, checkpoint_contexts, fp32_context


@dataclass(frozen=True)
class DiTConfig:
    channels: int = 32
    conditioning_dim: int = 2304
    layers: int = 9
    hidden: int = 1024
    heads: int = 16
    mlp: int = 3200
    layout: str = "spatial"  # spatial [B,C,H,W], tokens [B,N,C], vector [B,C]
    grid_shape: tuple[int, ...] = (16, 16)
    activation_checkpointing: bool = True

    def __post_init__(self):
        for name in ("channels", "layers", "hidden", "heads", "mlp"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.conditioning_dim < 0 or self.hidden % self.heads or self.hidden % 4:
            raise ValueError("Invalid conditioning dimension or attention/position width")
        expected = {"spatial": 2, "tokens": 1, "vector": 0}
        if self.layout not in expected or len(self.grid_shape) != expected[self.layout]:
            raise ValueError("grid_shape must match the explicitly selected layout")
        if any(not isinstance(n, int) or n <= 0 for n in self.grid_shape):
            raise ValueError("grid_shape entries must be positive integers")


def position_embedding(width, grid_shape):
    """Static constants: original FP64 construction, stored as FP32 buffers.

    This preserves the original 159M position table exactly. No forward or
    backward operation uses FP64. Spatial axis order is width, then height.
    """
    if not grid_shape:
        return torch.zeros(1, 1, width, dtype=torch.float32)
    if len(grid_shape) == 2:
        y, x = torch.meshgrid(*(torch.arange(n) for n in grid_shape), indexing="ij")
        axes = (x, y)
    else:
        axes = (torch.arange(grid_shape[0]),)
    part_width = width // (2 * len(axes))
    frequency = 1 / 10000 ** (torch.arange(part_width, dtype=torch.float64) / part_width)
    parts = []
    for axis in axes:
        angle = axis.flatten().double()[:, None] * frequency[None]
        parts.extend((angle.sin(), angle.cos()))
    return torch.cat(parts, -1).float()[None]


def sinusoid(value):
    frequency = torch.exp(-math.log(10000) * torch.arange(32, device=value.device, dtype=torch.float32) / 31)
    phase = value.reshape(-1, 1) * frequency
    return torch.cat((phase.sin(), phase.cos()), -1)


class Block(nn.Module):
    def __init__(self, width=1024, mlp_width=3200, heads=16):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_width), nn.GELU(approximate="tanh"), nn.Linear(mlp_width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x, condition):
        with fp32_context():
            s1, a1, g1, s2, a2, g2 = self.modulation(condition).chunk(6, -1)
            q, k, v = self.qkv(self.norm1(x) * (1 + a1[:, None]) + s1[:, None]).reshape(
                x.shape[0], x.shape[1], 3, self.heads, self.width // self.heads
            ).permute(2, 0, 3, 1, 4)
            attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0)
            x = x + g1[:, None] * self.proj(attention.transpose(1, 2).reshape_as(x))
            return x + g2[:, None] * self.mlp(self.norm2(x) * (1 + a2[:, None]) + s2[:, None])


class ResidualDiT(nn.Module):
    """delta(state, start, end, condition) returns a state displacement.

    forward returns state + delta, as in the original DirectDiT. Corrective
    rollout supplies the base's next state; direct rollout supplies current state.
    """
    def __init__(self, config: DiTConfig = DiTConfig()):
        super().__init__()
        if torch.get_default_dtype() != torch.float32:
            raise TypeError("Construct DiT with the FP32 default dtype")
        self.config = config
        width = config.hidden
        self.input = nn.Linear(config.channels, width, dtype=torch.float32)
        self.time = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
        self.conditioning = (nn.Sequential(nn.LayerNorm(config.conditioning_dim), nn.Linear(config.conditioning_dim, width))
                             if config.conditioning_dim else None)
        self.blocks = nn.ModuleList(Block(width, config.mlp, config.heads) for _ in range(config.layers))
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, config.channels)
        self.register_buffer("position", position_embedding(width, config.grid_shape))
        self.float()
        self.apply(self._initialize)
        for module in [*(b.modulation[-1] for b in self.blocks), self.modulation[-1], self.output]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def output_head_is_zero(self):
        return bool(torch.count_nonzero(self.output.weight) == 0 and torch.count_nonzero(self.output.bias) == 0)

    def identity(self):
        return asdict(self.config)

    def _tokens(self, state):
        cfg = self.config
        shape = ((cfg.channels, *cfg.grid_shape) if cfg.layout == "spatial" else
                 (*cfg.grid_shape, cfg.channels) if cfg.layout == "tokens" else (cfg.channels,))
        if tuple(state.shape[1:]) != shape or state.ndim != len(shape) + 1:
            raise ValueError(f"Expected batch-first {cfg.layout} state with trailing shape {shape}")
        if cfg.layout == "spatial":
            return state.flatten(2).transpose(1, 2)
        return state[:, None] if cfg.layout == "vector" else state

    def _time(self, value, state):
        if isinstance(value, torch.Tensor):
            check_tensor_tree(value)
            if value.device != state.device:
                raise ValueError("Time and state must be on the same device")
            value = value.reshape(-1)
            if value.numel() not in (1, len(state)):
                raise ValueError("Time must be scalar or one value per batch entry")
            return value.expand(len(state))
        if not math.isfinite(value):
            raise ValueError("Nonfinite time")
        return torch.full((len(state),), value, device=state.device, dtype=torch.float32)

    def delta(self, state, start, end, conditioning=None):
        with fp32_context():
            check_module(self, finite=False)
            check_tensor_tree((state, conditioning))
            if not state.is_floating_point():
                raise TypeError("State must be FP32")
            tokens = self._tokens(state)
            start, end = self._time(start, state), self._time(end, state)
            condition = self.time(torch.cat((sinusoid(start), sinusoid(end)), -1))
            if self.conditioning is not None:
                if conditioning is None or conditioning.shape != (len(state), self.config.conditioning_dim):
                    raise ValueError("Expected one conditioning vector per batch entry")
                condition = condition + self.conditioning(conditioning)
            elif conditioning is not None:
                raise ValueError("conditioning_dim=0 expects conditioning=None")
            x = self.input(tokens) + self.position
            for block in self.blocks:
                x = (checkpoint(block, x, condition, use_reentrant=False, context_fn=checkpoint_contexts)
                     if torch.is_grad_enabled() and self.config.activation_checkpointing else block(x, condition))
            shift, scale = self.modulation(condition).chunk(2, -1)
            delta = self.output(self.norm(x) * (1 + scale[:, None]) + shift[:, None])
            if self.config.layout == "spatial":
                delta = delta.transpose(1, 2).reshape_as(state)
            elif self.config.layout == "vector":
                delta = delta[:, 0]
            check_tensor_tree(delta)
            return delta

    def forward(self, state, start, end, conditioning=None):
        result = state + self.delta(state, start, end, conditioning)
        check_tensor_tree(result)
        return result
