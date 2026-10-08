"""Atomic update commits and a journal that refuses incomplete-update replay."""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_digest():
    root = Path(__file__).parent
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    return hashlib.sha256(canonical(files).encode()).hexdigest()


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG device set differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    # A failed write leaves the temp file as evidence; never replay silently.
    with os.fdopen(fd, "wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    _sync_directory(path.parent)


class CheckpointStore:
    def __init__(self, directory, identity, *, resume=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = json.loads(canonical(identity))
        self.identity_sha = hashlib.sha256(canonical(identity).encode()).hexdigest()
        self.journal = self.directory / "journal.jsonl"
        manifest = self.directory / "identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != self.identity:
                raise ValueError("Changed model, method, adapter, source or optimizer identity")
        else:
            if any(self.directory.iterdir()):
                raise RuntimeError("Nonempty output directory without an identity manifest")
            _atomic(manifest, lambda f: f.write((canonical(identity) + "\n").encode()))
        if not resume and (self.checkpoints() or self.journal.exists()):
            raise RuntimeError("Existing run requires explicit resume=True")
        self.state = self._reconcile() if resume else None

    def checkpoints(self):
        return sorted(self.directory.glob("update-*.pt"))

    def event(self, kind, **values):
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(canonical({"event": kind, **values}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _reconcile(self):
        # These files are trusted, locally produced training checkpoints.
        paths = self.checkpoints()
        state = torch.load(paths[-1], map_location="cpu", weights_only=False) if paths else None
        if state is not None and state["identity_sha"] != self.identity_sha:
            raise ValueError("Checkpoint identity mismatch")
        rows = [json.loads(line) for line in self.journal.read_text().splitlines()] if self.journal.exists() else []
        if any(row["event"] == "failed" for row in rows):
            raise RuntimeError("Run contains a failed attempt; automatic replay is forbidden")
        started = [r["update"] for r in rows if r["event"] == "started"]
        committed = [r["update"] for r in rows if r["event"] == "committed"]
        updates = state["updates"] if state else 0
        if started != list(range(1, updates + 1)):
            raise RuntimeError("Uncommitted attempt or inconsistent journal; stop without replay")
        if committed == list(range(1, updates)) and updates > 0:
            # The atomic checkpoint is the commit boundary, even if logging was interrupted.
            self.event("committed", update=updates, recovered=True, metrics=state["metrics"])
        elif committed != list(range(1, updates + 1)):
            raise RuntimeError("Journal does not match committed checkpoints")
        return state

    def commit(self, payload):
        path = self.directory / f"update-{payload['updates']:08d}.pt"
        if path.exists():
            raise RuntimeError("Update already committed")
        _atomic(path, lambda f: torch.save(cpu_snapshot(payload), f))
        self.event("committed", update=payload["updates"], metrics=payload["metrics"])
        for old in self.checkpoints()[:-2]:
            old.unlink()

    def final(self, payload):
        _atomic(self.directory / "final.pt", lambda f: torch.save(cpu_snapshot(payload), f))
