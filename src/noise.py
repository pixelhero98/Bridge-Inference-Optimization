"""Stable noise identities independent of batching or resume position."""
import hashlib
import torch


def keyed_noise(shape, key: str, *, seed=42, device="cpu"):
    derived = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(derived)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)
