"""Online grouped-draw training shared by corrective and direct distillation."""
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from collections.abc import Callable, Iterable, Mapping
import math

import torch
from torch import nn

from .checkpoint import CheckpointStore, rng_state, restore_rng, source_digest
from .objectives import TaskLoss, LinearRamp, paired_advantage, laplace_terms
from .precision import check_module, check_tensor_tree, freeze, fp32_context
from .rollout import TimeGrid, base_rollout, corrective_rollout, direct_rollout


@dataclass(frozen=True)
class Reward:
    name: str
    function: Callable
    scale: float
    weight: float = 1.
    direction: str = "maximize"

    def __post_init__(self):
        if not self.name or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Reward needs a name and a positive finite frozen scale")
        if not math.isfinite(self.weight) or self.weight < 0 or self.direction not in ("maximize", "minimize"):
            raise ValueError("Invalid reward weight/direction")

    def identity(self):
        return {"name": self.name, "scale": self.scale, "weight": self.weight, "direction": self.direction}


@dataclass(frozen=True)
class Kernel:
    name: str
    feature: Callable
    bandwidth: float
    weight: float
    mode: str = "paired"
    repulsion: bool = True
    state_index: int = -1  # final, or an explicit post-step state index (1..K)
    ramp: LinearRamp | None = None

    def __post_init__(self):
        if not self.name or not math.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("Kernel needs a name and positive finite bandwidth")
        if not math.isfinite(self.weight) or self.weight < 0 or self.mode not in ("paired", "ensemble", "observed"):
            raise ValueError("Invalid kernel weight/mode")
        if self.state_index == 0 or self.state_index < -1:
            raise ValueError("Kernel state_index must be -1 or a positive post-step index")

    def identity(self):
        return {"name": self.name, "bandwidth": self.bandwidth, "weight": self.weight,
                "mode": self.mode, "repulsion": self.repulsion, "state_index": self.state_index,
                "ramp": asdict(self.ramp) if self.ramp else None}


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    grid: TimeGrid
    reference_grid: TimeGrid
    task: TaskLoss | None = None
    task_weight: float = 1.
    lr: float = 1e-4
    betas: tuple[float, float] = (.9, .999)
    eps: float = 1e-8
    weight_decay: float = 1e-4
    gradient_clip: float = 1.
    strength: float = 1.
    checkpoint_dynamics: bool = True
    cpu_offload: bool = False

    def __post_init__(self):
        if self.method not in ("corrective", "distillation"):
            raise ValueError("method must be corrective or distillation")
        if (self.grid.points[0], self.grid.points[-1]) != (self.reference_grid.points[0], self.reference_grid.points[-1]):
            raise ValueError("Student/reference grids must share start and end times")
        for key in ("lr", "eps", "gradient_clip"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("task_weight", "weight_decay", "strength"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if len(self.betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in self.betas):
            raise ValueError("Invalid AdamW betas")
        if self.method == "distillation" and self.strength != 1:
            raise ValueError("strength applies only to corrective rollout")


@dataclass
class TrainingContext:
    initial: torch.Tensor  # [draws, ...state_shape]
    conditioning: torch.Tensor | None  # one vector [1, conditioning_dim]
    draw_ids: tuple[str, ...]
    base_scores: Mapping[str, torch.Tensor] = field(default_factory=dict)
    base_draw_ids: tuple[str, ...] | None = None
    reference_states: torch.Tensor | None = None
    reference_draw_ids: tuple[str, ...] | None = None
    reference_path: Mapping[int, torch.Tensor] = field(default_factory=dict)
    observed_targets: Mapping[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class TrainingBatch:
    contexts: tuple[TrainingContext, ...]
    cursor: int  # ordered batch index; checkpoint records cursor + 1


def _conditioning(context, count):
    if context.conditioning is None:
        return None
    if context.conditioning.ndim != 2 or len(context.conditioning) != 1:
        raise ValueError("One conditioning vector [1,C] is required per context")
    return context.conditioning.detach().expand(count, -1)


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, *, directory,
                 adapter_identity: Mapping, dynamics=None, teacher=None,
                 rewards: tuple[Reward, ...] = (), kernels: tuple[Kernel, ...] = (),
                 frozen_modules: tuple[nn.Module, ...] = (), resume=False):
        if not adapter_identity:
            raise ValueError("Explicit adapter/reference/data identity is required for safe resume")
        if config.method == "corrective" and dynamics is None:
            raise ValueError("Corrective training requires frozen base dynamics")
        if config.task is not None and config.task_weight > 0 and not any(r.weight > 0 for r in rewards):
            raise ValueError("An active task objective requires rewards")
        if not (config.task is not None and config.task_weight > 0) and not any(k.weight > 0 for k in kernels):
            raise ValueError("At least one objective must be enabled")
        if len({r.name for r in rewards}) != len(rewards) or len({k.name for k in kernels}) != len(kernels):
            raise ValueError("Reward/kernel names must be unique")
        if any(k.state_index > config.grid.steps for k in kernels):
            raise ValueError("Kernel state index is outside the student grid")
        self.model, self.config = model, config
        self.dynamics = dynamics
        self.teacher = teacher if teacher is not None else dynamics
        self.rewards, self.kernels = rewards, kernels
        self.frozen = list(frozen_modules)
        for item in (dynamics, self.teacher, *(r.function for r in rewards), *(k.feature for k in kernels)):
            if isinstance(item, nn.Module) and all(item is not m for m in self.frozen):
                self.frozen.append(item)
        model_params = {id(p) for p in model.parameters()}
        for module in self.frozen:
            if model_params.intersection(id(p) for p in module.parameters()):
                raise ValueError("Trainable and frozen modules share parameters")
            freeze(module)
        check_module(model)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas,
                                          eps=config.eps, weight_decay=config.weight_decay, foreach=False)
        architecture = model.identity() if hasattr(model, "identity") else {
            "class": type(model).__qualname__, "parameters": {n: list(p.shape) for n, p in model.named_parameters()}}
        identity = {"schema": 1, "architecture": architecture, "config": asdict(config),
                    "rewards": [r.identity() for r in rewards], "kernels": [k.identity() for k in kernels],
                    "adapters": dict(adapter_identity), "source": source_digest(), "precision": "strict_fp32_math_sdpa"}
        self.store = CheckpointStore(directory, identity, resume=resume)
        self.updates = self.cursor = self.used_trajectories = 0
        self.cost = {"generated": 0, "base_evaluations": 0, "operator_evaluations": 0, "reference_evaluations": 0}
        self.last_metrics = {}
        if self.store.state is not None:
            state = self.store.state
            # load_state_dict would silently promote corrupt lower-precision state.
            check_tensor_tree((state["model"], state["optimizer"]))
            model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.updates, self.cursor = state["updates"], state["cursor"]
            self.used_trajectories, self.cost = state["used_trajectories"], state["cost"]
            self.last_metrics = state["metrics"]
            check_module(model)
            check_tensor_tree(self.optimizer.state_dict())
            restore_rng(state["rng"])
        self.poisoned = False

    def _score(self, reward, state, condition):
        score = reward.function(state, condition)
        check_tensor_tree(score)
        if score.shape != (len(state),) or score.device != state.device:
            raise ValueError("Reward must return one FP32 scalar per draw on the state device")
        return score

    def _features(self, kernel, states, context):
        values = []
        condition = _conditioning(context, 1)
        for index in range(len(states)):
            value = kernel.feature(states[index:index + 1], condition)
            check_tensor_tree(value)
            if value.ndim < 2 or len(value) != 1 or value.device != states.device:
                raise ValueError("Feature adapter must preserve the batch dimension and state device")
            values.append(value.flatten(1))
        return torch.cat(values)

    def _validate_context(self, context):
        check_tensor_tree((context.initial, context.conditioning, context.base_scores,
                           context.reference_states, context.reference_path, context.observed_targets))
        draws = len(context.initial)
        if draws == 0 or len(context.draw_ids) != draws or len(set(context.draw_ids)) != draws:
            raise ValueError("Every initial draw needs a unique stable identity")
        if context.initial.requires_grad:
            raise ValueError("Initial noise is data, not a trainable tensor")
        _conditioning(context, draws)
        if context.base_scores and context.base_draw_ids != context.draw_ids:
            raise ValueError("Cached baseline scores do not match prompt/draw/noise identities")
        for score in context.base_scores.values():
            if score.shape != (draws,):
                raise ValueError("Cached baseline scores must have one value per draw")
        active = [k for k in self.kernels if k.weight > 0]
        if context.reference_states is not None or context.reference_path:
            if context.reference_draw_ids is None:
                raise ValueError("Cached teacher states need explicit draw identities")
            if any(k.mode == "paired" for k in active) and context.reference_draw_ids != context.draw_ids:
                raise ValueError("Paired teacher states have different draw/noise identities")
            for value in (context.reference_states, *context.reference_path.values()):
                if value is not None and (len(value) != len(context.reference_draw_ids) or value.shape[1:] != context.initial.shape[1:]):
                    raise ValueError("Cached teacher state shape/identity mismatch")
        for target in context.observed_targets.values():
            if target.shape != (1, *context.initial.shape[1:]):
                raise ValueError("Observed target must be a single state per context")

    def _references(self, context, active_kernels, task_enabled):
        scores = dict(context.base_scores)
        targets = dict(context.reference_path)
        if context.reference_states is not None:
            targets[-1] = context.reference_states
        missing_scores = [r for r in self.rewards if task_enabled and r.weight > 0 and r.name not in scores]
        needed = {k.state_index for k in active_kernels if k.mode != "observed" and k.state_index not in targets}
        if not missing_scores and not needed:
            return scores, targets
        if self.teacher is None:
            raise ValueError("Missing reference cache and frozen teacher dynamics")
        index_map = {-1: self.config.reference_grid.steps}
        for index in needed - {-1}:
            time = self.config.grid.points[index]
            matches = [i for i, t in enumerate(self.config.reference_grid.points) if abs(t - time) < 1e-7]
            if len(matches) != 1:
                raise ValueError("Teacher grid lacks requested intermediate target time; supply reference_path")
            index_map[index] = matches[0]
        generated_targets = {index: [] for index in needed}
        generated_scores = {r.name: [] for r in missing_scores}
        condition = _conditioning(context, 1)
        with torch.no_grad():
            for draw in range(len(context.initial)):
                reference = base_rollout(self.teacher, context.initial[draw:draw + 1], condition, self.config.reference_grid)
                self.cost["reference_evaluations"] += reference.base_evaluations
                for index in needed:
                    generated_targets[index].append(reference.states[index_map[index]])
                for reward in missing_scores:
                    generated_scores[reward.name].append(self._score(reward, reference.final, condition))
        targets.update({index: torch.cat(values) for index, values in generated_targets.items()})
        scores.update({name: torch.cat(values) for name, values in generated_scores.items()})
        return scores, targets

    def _context_loss(self, context, update):
        task_enabled = self.config.task is not None and self.config.task_weight > 0
        active = [(k, k.weight * (k.ramp(update) if k.ramp else 1.)) for k in self.kernels if k.weight > 0]
        active = [(k, w) for k, w in active if w > 0]
        if not task_enabled and not active:
            raise ValueError("No active objective at this update")
        scores, targets = self._references(context, [k for k, _ in active], task_enabled)
        trajectories = []
        condition = _conditioning(context, 1)
        for draw, identity in enumerate(context.draw_ids):
            self.store.event("draw_started", update=update, draw_id=identity)
            initial = context.initial[draw:draw + 1]
            if self.config.method == "corrective":
                rollout = corrective_rollout(self.dynamics, self.model, initial, condition, self.config.grid,
                    strength=self.config.strength, checkpoint_dynamics=self.config.checkpoint_dynamics)
            else:
                rollout = direct_rollout(self.model, initial, condition, self.config.grid)
            trajectories.append(rollout)
            self.cost["generated"] += 1
            self.cost["base_evaluations"] += rollout.base_evaluations
            self.cost["operator_evaluations"] += rollout.operator_evaluations
            self.store.event("draw_generated", update=update, draw_id=identity,
                             base_evaluations=rollout.base_evaluations, operator_evaluations=rollout.operator_evaluations)
        losses, metrics = [], {}
        if task_enabled:
            single_condition = _conditioning(context, 1)
            advantages = [r.weight * paired_advantage(torch.cat([self._score(r, t.final, single_condition) for t in trajectories]), scores[r.name],
                                                     scale=r.scale, direction=r.direction)
                          for r in self.rewards if r.weight > 0]
            advantage = sum(advantages[1:], advantages[0])
            task = self.config.task(advantage)
            losses.append(self.config.task_weight * task)
            metrics.update(task=float(task.detach()), advantage=float(advantage.detach().mean()))
        for kernel, weight in active:
            states = torch.cat([r.states[kernel.state_index] for r in trajectories])
            if kernel.mode == "observed":
                target = context.observed_targets.get(kernel.state_index)
                if target is None:
                    raise ValueError("Missing observed target at requested state index")
            else:
                target = targets[kernel.state_index]
            generated_features = self._features(kernel, states, context)
            with torch.no_grad():
                target_features = self._features(kernel, target.detach(), context)
            terms = laplace_terms(generated_features, target_features, bandwidth=kernel.bandwidth,
                                  mode=kernel.mode, repulsion=kernel.repulsion)
            losses.append(weight * terms["loss"])
            for name, value in terms.items():
                metrics[f"kernel/{kernel.name}/{name}"] = float(value.detach())
        loss = sum(losses[1:], losses[0])
        check_tensor_tree(loss)
        metrics["loss"] = float(loss.detach())
        return loss, metrics

    def _payload(self, metrics):
        return {"identity_sha": self.store.identity_sha, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "rng": rng_state(), "updates": self.updates,
                "cursor": self.cursor, "used_trajectories": self.used_trajectories,
                "cost": dict(self.cost), "metrics": metrics}

    def train_step(self, batch: TrainingBatch):
        if self.poisoned:
            raise RuntimeError("This trainer had a failed attempt; create a separate run")
        if batch.cursor != self.cursor or not batch.contexts:
            raise ValueError("Batch must start at the committed data cursor and contain contexts")
        for context in batch.contexts:
            self._validate_context(context)
        ids = [key for context in batch.contexts for key in context.draw_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("Draw identities must be unique within the update")
        update = self.updates + 1
        self.store.event("started", update=update, cursor=self.cursor, draw_ids=ids)
        committed = False
        try:
            with fp32_context():
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                aggregate = {}
                for context in batch.contexts:
                    offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if self.config.cpu_offload else nullcontext()
                    with offload:
                        loss, metrics = self._context_loss(context, update)
                    (loss / len(batch.contexts)).backward()
                    for key, value in metrics.items():
                        aggregate[key] = aggregate.get(key, 0.) + value / len(batch.contexts)
                for parameter in self.model.parameters():
                    if parameter.requires_grad and parameter.grad is None:
                        raise RuntimeError("Disconnected trainable parameter")
                    check_tensor_tree(parameter.grad)
                for module in self.frozen:
                    check_module(module)
                    if any(p.grad is not None or p.requires_grad for p in module.parameters()):
                        raise RuntimeError("Frozen model received parameter gradients")
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
                self.optimizer.step()
                check_module(self.model)
                check_tensor_tree(self.optimizer.state_dict())
                self.updates, self.cursor = update, batch.cursor + 1
                self.used_trajectories += len(ids)
                aggregate.update(update=update, used_trajectories=self.used_trajectories, gradient_norm=float(norm))
                self.store.commit(self._payload(aggregate))
                committed = True
                self.last_metrics = aggregate
                return aggregate
        except BaseException as error:
            self.poisoned = True
            # A checkpoint rename may have succeeded before post-commit logging failed.
            path = self.store.directory / f"update-{update:08d}.pt"
            if not committed and not path.exists():
                self.store.event("failed", update=update, error=type(error).__name__ + ": " + str(error), cost=self.cost)
            raise

    def fit(self, batches: Iterable[TrainingBatch], *, total_updates: int):
        if total_updates < self.updates:
            raise ValueError("total_updates precedes the committed checkpoint")
        iterator = iter(batches)
        history = []
        while self.updates < total_updates:
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise ValueError("Data iterator ended before total_updates") from error
            history.append(self.train_step(batch))
        if self.updates:
            self.store.final(self._payload(self.last_metrics))
        return history
