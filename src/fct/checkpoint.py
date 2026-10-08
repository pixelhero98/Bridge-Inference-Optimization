"""Atomic canonical checkpoints, explicit provenance, and reproducible RNG state."""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .rollout import OPERATOR


def source_hashes():
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device set changed")
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_snapshot(item) for item in value)
    return value


def atomic_save(path, payload):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        torch.save(cpu_snapshot(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def load_checkpoint(path, identity):
    # Only load trusted local training output, never arbitrary downloaded pickles.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("operator") != OPERATOR or state.get("schema") != 1:
        raise ValueError("Incompatible operator/checkpoint; legacy displacement weights are not velocity weights")
    if state["identity"] != identity:
        raise ValueError("Changed configuration, adapter, or source identity; start a distinct run")
    return state
