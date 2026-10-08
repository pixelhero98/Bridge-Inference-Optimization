"""One optimizer and full-rollout objective across predeclared training stages."""
from dataclasses import asdict, dataclass, field
import importlib
import inspect
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from .checkpoint import (atomic_save, canonical_json, load_checkpoint, restore_rng,
                         rng_state, seed_all, source_hashes)
from .objectives import conditional_loss, nonnegative
from .rollout import OPERATOR, check_steps, uniform_euler


@dataclass(frozen=True)
class Stage:
    updates: int
    utility_weight: float
    ramp_updates: int = 0

    def __post_init__(self):
        check_steps(self.updates)
        nonnegative(self.utility_weight, "utility_weight")
        if type(self.ramp_updates) is not int or not 0 <= self.ramp_updates <= self.updates:
            raise ValueError("ramp_updates must be an integer between zero and stage length")
        if not self.utility_weight and self.ramp_updates:
            raise ValueError("A matching-only stage cannot have a utility ramp")


@dataclass(frozen=True)
class TrainingConfig:
    regime: str
    steps: int
    teacher_steps: int
    stages: tuple[Stage, ...]
    adapter: str
    adapter_config: dict = field(default_factory=dict)
    matching_weight: float = 1.0
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    gradient_clip: float | None = None
    dtype: str = "float64"
    device: str = "cpu"
    seed: int = 0
    checkpoint_dynamics: bool = False
    cpu_offload: bool = False
    checkpoint_every: int = 1

    def __post_init__(self):
        check_steps(self.steps)
        check_steps(self.teacher_steps)
        check_steps(self.checkpoint_every)
        if self.dtype not in ("float32", "float64") or self.device not in ("cpu", "cuda"):
            raise ValueError("Select explicit float32/float64 and cpu/cuda")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not self.stages or not isinstance(self.adapter, str) or self.adapter.count(":") != 1:
            raise ValueError("Require stages and an explicit module:factory adapter")
        if self.regime == "same_step":
            if self.steps != self.teacher_steps or len(self.stages) != 1 or self.stages[0].utility_weight <= 0:
                raise ValueError("same_step requires N=N_T and one joint stage")
        elif self.regime == "reduced_step":
            if (self.steps >= self.teacher_steps or len(self.stages) != 2 or
                    self.stages[0].utility_weight != 0 or self.stages[1].utility_weight <= 0):
                raise ValueError("reduced_step requires N<N_T, matching then joint stages")
        else:
            raise ValueError("Unknown training regime")
        if not math.isfinite(self.matching_weight) or self.matching_weight <= 0:
            raise ValueError("Both canonical regimes retain positive matching weight")
        for name in ("lr", "eps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        nonnegative(self.weight_decay, "weight_decay")
        if self.gradient_clip is not None and (not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0):
            raise ValueError("gradient_clip must be positive or null")
        if len(self.betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in self.betas):
            raise ValueError("Invalid AdamW betas")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["stages"] = tuple(Stage(**stage) for stage in data["stages"])
        if "betas" in data:
            data["betas"] = tuple(data["betas"])
        return cls(**data)

    @property
    def total_updates(self):
        return sum(stage.updates for stage in self.stages)

    def stage_at(self, update):
        if not 1 <= update <= self.total_updates:
            raise ValueError("Update outside stage schedule")
        offset = 0
        for index, stage in enumerate(self.stages):
            local = update - offset
            if local <= stage.updates:
                multiplier = min(1.0, local / stage.ramp_updates) if stage.ramp_updates else 1.0
                return index, local, stage.utility_weight * multiplier
            offset += stage.updates


def create_adapter(config):
    module_name, name = config.adapter.split(":")
    factory = getattr(importlib.import_module(module_name), name)
    return factory(config.adapter_config, seed=config.seed, dtype=getattr(torch, config.dtype),
                   device=config.device, steps=config.steps, teacher_steps=config.teacher_steps)


def freeze(module):
    module.eval().requires_grad_(False)
    for p in module.parameters():
        p.grad = None


def finite_tensor(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {label}")


class Trainer:
    def __init__(self, config, output, *, resume=False, adapter=None):
        self.config, self.output = config, Path(output)
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
            raise RuntimeError("Canonical reference execution requires autocast disabled")
        seed_all(config.seed)
        self.adapter = create_adapter(config) if adapter is None else adapter
        self.model = self.adapter.correction
        self.frozen = list(self.adapter.frozen_modules)
        for item in (self.adapter.dynamics, self.adapter.utility,
                     *(branch.feature for branch in self.adapter.branches),
                     *(branch.kernel for branch in self.adapter.branches)):
            if isinstance(item, nn.Module) and all(item is not m for m in self.frozen):
                self.frozen.append(item)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("Correction needs trainable parameters")
        for module in self.frozen:
            if {id(p) for p in trainable} & {id(p) for p in module.parameters()}:
                raise ValueError("Frozen and correction parameters overlap")
            freeze(module)
        for module in [self.model, *self.frozen]:
            for p in (*module.parameters(), *module.buffers()):
                if p.is_floating_point() and (p.dtype != getattr(torch, config.dtype) or p.device.type != config.device):
                    raise ValueError("Adapter modules must use configured dtype/device")
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr, betas=config.betas,
                                          eps=config.eps, weight_decay=config.weight_decay, foreach=False)
        adapter_source = inspect.getsourcefile(type(self.adapter))
        self.identity = json.loads(canonical_json({"operator": OPERATOR, "config": asdict(config),
            "adapter": self.adapter.identity(), "source": source_hashes(),
            "adapter_source_sha256": hashlib.sha256(Path(adapter_source).read_bytes()).hexdigest(),
            "parameters": {n: list(p.shape) for n, p in self.model.named_parameters()}}))
        self.updates = self.cursor = 0
        self.history = []
        self.poisoned = False
        if resume:
            state = load_checkpoint(self.output / "latest.pt", self.identity)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.adapter.load_state_dict(state["adapter_state"])
            self.updates, self.cursor, self.history = state["updates"], state["cursor"], state["history"]
            restore_rng(state["rng"])
            self.event("resume", from_update=self.updates)
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            manifest = {"identity": self.identity, "python": sys.version, "torch": torch.__version__,
                        "numpy": np.__version__, "cuda": torch.version.cuda, "platform": platform.platform(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_unix": time.time()}
            (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            # Initial state permits explicit deterministic recovery even before update 1.
            self.save()

    def event(self, event, **data):
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json({"event": event, **data}) + "\n")
            stream.flush()

    def payload(self):
        return {"schema": 1, "operator": OPERATOR, "identity": self.identity,
                "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "rng": rng_state(), "adapter_state": self.adapter.state_dict(),
                "updates": self.updates, "cursor": self.cursor, "history": self.history}

    def save(self):
        atomic_save(self.output / "latest.pt", self.payload())

    def step(self):
        if self.poisoned:
            raise RuntimeError("Failed trainer; explicitly resume its last checkpoint in a new process")
        update = self.updates + 1
        stage, local, utility_weight = self.config.stage_at(update)
        self.event("attempt", update=update, stage=stage, cursor=self.cursor)
        try:
            batches = self.adapter.batch(self.cursor)
            if not batches:
                raise ValueError("Empty context batch")
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            metrics = {"update": update, "stage": stage, "stage_update": local,
                       "utility_weight": utility_weight, "matching_weight": self.config.matching_weight,
                       "student_draws": sum(len(b.initial) for b in batches)}
            for batch in batches:
                batch.validate()
                endpoint = uniform_euler(self.adapter.dynamics, self.model, batch.initial, batch.context,
                    self.config.steps, checkpoint_dynamics=self.config.checkpoint_dynamics,
                    cpu_offload=self.config.cpu_offload)
                loss, terms = conditional_loss(endpoint, batch.matching_targets, batch.context,
                    self.adapter.utility, self.adapter.branches, utility_weight=utility_weight,
                    matching_weight=self.config.matching_weight, utility_baseline=batch.utility_baseline)
                finite_tensor(loss, "loss")
                (loss / len(batches)).backward()
                for name, value in {"loss": loss, **terms}.items():
                    metrics[name] = metrics.get(name, 0.0) + float(value.detach()) / len(batches)
            for p in self.model.parameters():
                if p.requires_grad:
                    if p.grad is None:
                        raise RuntimeError("Disconnected correction parameter")
                    finite_tensor(p.grad, "gradient")
            for module in self.frozen:
                if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
                    raise RuntimeError("Frozen module invariant violated")
            if self.config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
            self.optimizer.step()
            for p in self.model.parameters():
                finite_tensor(p, "updated parameter")
            metrics["student_backbone_evaluations"] = metrics["student_draws"] * self.config.steps
            metrics["correction_evaluations"] = metrics["student_backbone_evaluations"]
            self.updates, self.cursor = update, self.cursor + 1
            self.history.append(metrics)
            boundary = local == self.config.stages[stage].updates
            if update % self.config.checkpoint_every == 0 or boundary:
                self.save()
            self.event("completed", **metrics)
            return metrics
        except BaseException:
            self.poisoned = True
            failure = traceback.format_exc()
            (self.output / f"failure-{update}-{time.time_ns()}.txt").write_text(failure)
            self.event("failed", update=update)
            raise

    def run(self, until_update=None):
        limit = self.config.total_updates if until_update is None else until_update
        if type(limit) is not int or not self.updates <= limit <= self.config.total_updates:
            raise ValueError("Stop update outside the remaining schedule")
        while self.updates < limit:
            self.step()
        self.save()
        summary = {"status": "COMPLETE" if self.updates == self.config.total_updates else "STOPPED",
                   "updates": self.updates, "cursor": self.cursor, "history": self.history}
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary
