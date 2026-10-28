"""Disposable full-update timing, with phase and network-call accounting."""
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import time

import torch
from torch.utils.checkpoint import checkpoint

from .baselines import advantages, ppo_loss, lambda_loss, trajectory_dpo_loss
from .checkpoint import atomic_save, cpu_snapshot, rng_state, restore_rng
from .objectives import kernel_score, LaplaceKernel
from .policy import collect, euler, gaussian_log_prob, reference_policy, transition_mean
from .study_io import identity, write


class Meter:
    def __init__(self, velocity):
        self.velocity, self.seconds, self.calls, self.evaluations = velocity, {}, {}, {}

    @contextmanager
    def phase(self, name):
        torch.cuda.synchronize()
        start = time.perf_counter()
        calls, evaluations = self.velocity.calls, self.velocity.state_evaluations
        yield
        torch.cuda.synchronize()
        self.seconds[name] = self.seconds.get(name, 0.)+time.perf_counter()-start
        self.calls[name] = self.calls.get(name, 0)+self.velocity.calls-calls
        self.evaluations[name] = self.evaluations.get(name, 0)+self.velocity.state_evaluations-evaluations


def trainable_state(policy):
    return {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}


def restore_trainable(policy, state):
    current = dict(policy.named_parameters())
    if set(state) != {n for n, p in current.items() if p.requires_grad}:
        raise ValueError('Trainable parameter identity changed')
    with torch.no_grad():
        for n, value in state.items():
            current[n].copy_(value)


def finite_update(policy, optimizer, meter, max_norm):
    with meter.phase('optimizer'):
        gradients = [p.grad for p in policy.trainable if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Missing or nonfinite gradients')
        norm = torch.nn.utils.clip_grad_norm_(policy.trainable, max_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(norm)


class Profiler:
    def __init__(self, policy, dino, reward, pool, config, method, steps, output):
        self.policy, self.dino, self.reward, self.pool = policy, dino, reward, pool
        self.config, self.method, self.steps = config, method, steps
        self.settings = config['methods'][method]
        self.optimizer = torch.optim.AdamW(policy.trainable, lr=self.settings['lr'],
            betas=(.9,.999), eps=1e-8, weight_decay=self.settings['weight_decay'])
        self.output = Path(output)
        self.offline = None
        self.offline_ids = None
        self.bandwidths = None
        self.check_identity = identity({'config': config, 'method': method, 'steps': steps, 'pool': pool.identity})

    def batch(self, cursor, role='paired'):
        return self.pool.batch(cursor, self.config['batch_size'], self.config['group_size'], role=role)

    def prepare_matching(self):
        # Calibration references are separate from every timed update and paired reward.
        data = self.batch(10000, 'matching')
        with torch.no_grad():
            features = [data['reference'].flatten(1), torch.cat([self.dino(x) for x in data['reference'].split(self.config['microbatch'])])]
            widths = []
            for f in features:
                values = []
                for group in f.split(self.config['group_size']):
                    values.append(torch.pdist(group))
                width = torch.cat(values).median().item()
                if not math.isfinite(width) or width <= 0:
                    raise ValueError('Degenerate calibration features')
                widths.append(width)
        self.bandwidths = widths
        write(self.output/'bandwidths.json', {'values': widths, 'ids': data['ids'],
              'purpose': 'disposable timing calibration only; freeze before measured updates'}, immutable=True)

    def direct(self, data, meter, cursor):
        B, M = self.config['batch_size'], self.config['microbatch']
        matching = self.batch(cursor, 'matching') if self.method == 'ogo' else None
        if matching is not None:
            if set(data['ids']) & set(matching['ids']) or not torch.equal(data['labels'], matching['labels']):
                raise ValueError('Matching independence or class pairing violated')
        losses = []
        self.optimizer.zero_grad(set_to_none=True)
        for offset in range(0, B, M):
            x, labels, ref = (data[k][offset:offset+M] for k in ('initial', 'labels', 'reference'))
            with meter.phase('rollout'):
                y = euler(self.policy, x, labels, self.steps,
                          backward_steps=self.steps if self.method == 'ogo' else self.settings['backward_steps'])
            with meter.phase('reward'):
                costs = checkpoint(lambda z: self.reward(z, labels, ref).costs, y, use_reentrant=False)
                loss = costs.mean()
            if matching is not None:
                with meter.phase('matching'):
                    target = matching['reference'][offset:offset+M]
                    with torch.no_grad():
                        ft = [target.flatten(1), self.dino(target)]
                    fg = [y.flatten(1), checkpoint(self.dino, y, use_reentrant=False)]
                    loss = loss+self.settings['matching_weight']*sum(.5*kernel_score(g, t, LaplaceKernel(w))
                               for g, t, w in zip(fg, ft, self.bandwidths))
            with meter.phase('backward'):
                (loss*M/B).backward()
            losses.append(float(loss.detach()))
        norm = finite_update(self.policy, self.optimizer, meter, self.settings['max_grad_norm'])
        return {'loss': sum(losses)/len(losses), 'gradient_norm': norm, 'optimizer_steps': 1, 'new_rollouts': B}

    def trajectories(self, data, meter, cursor):
        B, M = self.config['batch_size'], self.config['microbatch']
        batches, rewards = [], []
        for offset in range(0, B, M):
            x, labels, ref = (data[k][offset:offset+M] for k in ('initial', 'labels', 'reference'))
            if self.method == 'offline_d3po':
                x, labels, ref = (v[::2].repeat_interleave(2, dim=0) for v in (x, labels, ref))
            generator = torch.Generator(device=x.device).manual_seed(710000+cursor*B+offset)
            with meter.phase('preference_collection' if self.method == 'offline_d3po' else 'rollout'):
                trajectory = collect(self.policy, x, labels, self.steps, self.pool.backbone,
                                     self.settings['noise_level'], generator)
            with meter.phase('reward'):
                with torch.no_grad():
                    rewards.append(-self.reward(trajectory['states'][-1], labels, ref).costs)
            batches.append(trajectory)
        rewards = torch.cat(rewards)
        if self.method == 'offline_d3po':
            winner = torch.sign(rewards.reshape(-1, 2)[:, 0]-rewards.reshape(-1, 2)[:, 1])
            # Strict tie metadata retained; no reward-based rejection/screening.
            for j, batch in enumerate(batches):
                batch['winner'] = winner[j*(M//2):(j+1)*(M//2)]
                batch['reference_logps'] = batch['logps'].clone()
            self.offline = batches
            self.offline_ids = list(data['ids'])
            atomic_save(self.output/'offline-preferences.pt', {'identity': self.check_identity,
                'trajectories': batches, 'rewards': rewards, 'input_ids': data['ids'], 'frozen': True})
        else:
            adv = advantages(rewards, None if self.method == 'ddpo_is' else self.config['group_size']).clamp(-5,5)
            for j, batch in enumerate(batches):
                batch['advantage'] = adv[j*M:(j+1)*M]
        return batches, float(rewards.mean())

    def rl(self, data, meter, cursor):
        B, M = self.config['batch_size'], self.config['microbatch']
        fresh = self.method != 'offline_d3po' or self.offline is None
        if fresh:
            batches, reward = self.trajectories(data, meter, cursor)
        else:
            batches, reward = self.offline, None
        losses, norms, ratio_errors = [], [], []
        for epoch in range(self.settings['replay_passes']):
            self.optimizer.zero_grad(set_to_none=True)
            for batch in batches:
                for k in range(self.steps):
                    x, following, labels = batch['states'][k], batch['states'][k+1], batch['labels']
                    with meter.phase('update_forward'):
                        velocity = self.policy(x, k/self.steps, labels)
                        mean, std = transition_mean(x, velocity, k/self.steps, self.steps, self.pool.backbone, self.settings['noise_level'])
                        logp = gaussian_log_prob(following, mean, std)
                        if epoch == 0 and fresh:
                            ratio_errors.append(float((logp-batch['logps'][:, k]).abs().max().detach()))
                    if self.method == 'offline_d3po':
                        loss = trajectory_dpo_loss(logp, batch['reference_logps'][:, k], batch['winner'], self.settings['beta'], self.settings['q_clip'])
                    elif self.method == 'lambda_grpo':
                        loss, _ = lambda_loss(logp, batch['logps'][:, k], mean, batch['means'][k], std, batch['advantage'],
                            clip=self.settings['clip'], retained_fraction=self.settings['retained_fraction'],
                            late_steps=self.steps//2, floor=self.settings['lambda_floor'])
                    else:
                        loss = ppo_loss(logp, batch['logps'][:, k], batch['advantage'], self.settings['clip'])
                    if self.method == 'flow_grpo' and self.settings['kl_weight'] > 0:
                        with meter.phase('reference_policy'):
                            with reference_policy(self.policy):
                                refmean, _ = transition_mean(x, self.policy(x, k/self.steps, labels), k/self.steps,
                                    self.steps, self.pool.backbone, self.settings['noise_level'])
                        loss = loss+self.settings['kl_weight']*((mean-refmean)/std).square().mean()/2
                    with meter.phase('backward'):
                        (loss*M/B/self.steps).backward()
                    losses.append(float(loss.detach()))
            norms.append(finite_update(self.policy, self.optimizer, meter, self.settings['max_grad_norm']))
        error = max(ratio_errors, default=0.)
        if error > self.config['thresholds']['on_policy_logp_atol']:
            raise ValueError('On-policy replay changed likelihood: '+str(error))
        return {'loss': sum(losses)/len(losses), 'gradient_norm': max(norms), 'reward': reward,
                'optimizer_steps': self.settings['replay_passes'], 'new_rollouts': B if fresh else 0,
                'on_policy_max_logp_error': error}

    def step(self, cursor):
        meter = Meter(self.policy.velocity)
        start = time.perf_counter()
        if self.method == 'offline_d3po' and self.offline is not None:
            data = {'ids': self.offline_ids}
        else:
            with meter.phase('reference_io'):
                data = self.batch(cursor)
        if self.method in ('ogo', 'draft_k'):
            result = self.direct(data, meter, cursor)
        else:
            result = self.rl(data, meter, cursor)
        torch.cuda.synchronize()
        return {**result, 'wall_seconds': time.perf_counter()-start, 'phase_seconds': meter.seconds,
                'network_calls': meter.calls, 'network_state_evaluations': meter.evaluations,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'cursor': cursor, 'input_ids': data['ids']}

    def checkpoint(self, path, cursor):
        atomic_save(path, {'identity': self.check_identity, 'cursor': cursor, 'trainable': trainable_state(self.policy),
                         'optimizer': self.optimizer.state_dict(), 'rng': rng_state(), 'offline': self.offline,
                         'bandwidths': self.bandwidths, 'offline_ids': self.offline_ids})

    def resume(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['identity'] != self.check_identity:
            raise ValueError('Changed study identity cannot resume')
        restore_trainable(self.policy, state['trainable'])
        self.optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
        self.bandwidths = state['bandwidths']
        self.offline_ids = state['offline_ids']
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.cuda()
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            if isinstance(value, list):
                return [move(v) for v in value]
            return value
        self.offline = move(state['offline'])
        return state['cursor']
