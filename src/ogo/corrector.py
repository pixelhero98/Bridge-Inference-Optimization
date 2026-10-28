"""Small conditional RoPE transformer that returns current-state velocity."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(t, dimension=256, max_period=10000):
    frequencies = torch.exp(-math.log(max_period) * torch.arange(dimension//2, device=t.device, dtype=t.dtype) / (dimension//2))
    angles = t[:, None] * frequencies[None]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def rotate_pairs(value, cosine, sine):
    even, odd = value[..., ::2], value[..., 1::2]
    return torch.stack((even*cosine-odd*sine, even*sine+odd*cosine), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, width, heads, mlp_width):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3*width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_width), nn.GELU(approximate="tanh"), nn.Linear(mlp_width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6*width))

    def forward(self, tokens, condition, cosine, sine):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, dim=-1)
        value = self.norm1(tokens)*(1+scale1[:, None])+shift1[:, None]
        b, n, width = value.shape
        q, k, v = self.qkv(value).reshape(b, n, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rotate_pairs(q, cosine, sine), rotate_pairs(k, cosine, sine)
        value = F.scaled_dot_product_attention(q, k, v, dropout_p=0.).transpose(1, 2).reshape(b, n, width)
        tokens = tokens + gate1[:, None]*self.proj(value)
        return tokens + gate2[:, None]*self.mlp(self.norm2(tokens)*(1+scale2[:, None])+shift2[:, None])


class RoPECorrector(nn.Module):
    """E1 architecture: 16,012,816 trainable parameters at the declared sizes."""
    def __init__(self, *, input_size, channels, classes, patch_size, width, depth, heads, mlp_width):
        super().__init__()
        if input_size % patch_size or width % heads or (width//heads) % 4:
            raise ValueError("Patches must tile the latent; head width must be divisible by four")
        self.input_size, self.channels, self.patch_size = input_size, channels, patch_size
        self.sample_evaluations = 0
        self.patch = nn.Conv2d(channels, width, patch_size, stride=patch_size)
        self.label = nn.Embedding(classes, width)
        self.time = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([Block(width, heads, mlp_width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2*width))
        self.output = nn.Linear(width, patch_size*patch_size*channels)
        size, axis_pairs = input_size//patch_size, (width//heads)//4
        rows, columns = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        frequency = torch.exp(-math.log(10000)*torch.arange(axis_pairs)/axis_pairs)
        angles = torch.cat((rows.flatten()[:, None]*frequency, columns.flatten()[:, None]*frequency), dim=-1)
        self.register_buffer("rope_cosine", angles.cos()[None, None], persistent=False)
        self.register_buffer("rope_sine", angles.sin()[None, None], persistent=False)
        self.apply(self._initialize)
        nn.init.normal_(self.label.weight, std=.02)
        for layer in (self.time[0], self.time[2]):
            nn.init.normal_(layer.weight, std=.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight.reshape(module.weight.shape[0], -1))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state, time, context):
        self.sample_evaluations += len(state)
        if state.shape[1:] != (self.channels, self.input_size, self.input_size):
            raise ValueError("Corrector input shape does not match its declared latent geometry")
        labels = torch.as_tensor(context, device=state.device, dtype=torch.long).reshape(-1)
        if len(labels) == 1:
            labels = labels.expand(len(state))
        if len(labels) != len(state):
            raise ValueError("One class per generated state is required")
        times = torch.full((len(state),), time, dtype=state.dtype, device=state.device)
        condition = self.label(labels) + self.time(timestep_embedding(times))
        tokens = self.patch(state).flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, condition, self.rope_cosine, self.rope_sine)
        shift, scale = self.final_modulation(condition).chunk(2, -1)
        patches = self.output(self.norm(tokens)*(1+scale[:, None])+shift[:, None])
        p, n = self.patch_size, self.input_size//self.patch_size
        return patches.reshape(len(state), n, n, p, p, self.channels).permute(0, 5, 1, 3, 2, 4).reshape_as(state)
