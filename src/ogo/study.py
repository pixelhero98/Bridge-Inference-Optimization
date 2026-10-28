"""Separate CLI for verification, bounded profiling, audit and cost reporting."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .study_io import digest, identity, read, sources, write


def matrix(config):
    return [dict(backbone=b, method=m, steps=n, key=f'{b}-{m}-n{n}')
            for b, m, n in itertools.product(config['backbones'], config['methods'], config['steps'])]


def validate(config):
    from .baselines import METHODS
    if (tuple(config['methods']) != METHODS or config['backbones'] != ['edm64', 'sit256'] or
            config['steps'] != [16,32] or len(set(config['class_ids'])) != 100 or
            any(not 0 <= c < 1000 for c in config['class_ids']) or
            config['evaluation'] != 'deterministic_uniform_euler' or
            config['microbatch'] != config['group_size'] or config['batch_size'] % config['microbatch']):
        raise ValueError('Agreed matrix, sampler or conditional batch geometry changed')
    if config['warmup_updates'] != 3 or config['timed_updates'] != 5:
        raise ValueError('Profiling coverage cannot silently change')


def allocation_required():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run numerical verification/profiling in an allocated job')


def warm_resume(profiler, checkpoint_path, cursor, folder, repetitions):
    """Warm a new process, restoring the committed state after every probe."""
    measurements = []
    for _ in range(repetitions):
        # Reuse a previously visited input batch; no extra scientific cases.
        measurements.append(profiler.step(max(0, cursor-1)))
        if profiler.resume(checkpoint_path) != cursor:
            raise ValueError('Warmup restore changed the committed cursor')
    write(folder/f'resume-warmup-{os.environ["SLURM_JOB_ID"]}.json',
          {'committed_cursor': cursor, 'committed_updates': 0, 'probes': measurements})


def case_run(args, config, case, run_identity):
    allocation_required()
    import torch
    from .backbones import load, Policy
    from .checkpoint import cpu_snapshot, seed_all
    from .policy import euler, reference_policy
    from .profiling import Profiler, Meter, trainable_state
    from .reference_pool import ReferencePool
    from .study_verify import equal_tree
    seed_all(config['seed'])
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    folder = Path(args.output)/case['key']
    folder.mkdir(exist_ok=True)
    write(folder/'identity.json', run_identity, immutable=True)
    start = time.perf_counter()
    assets = read(args.assets)
    velocity, dino, reward, provenance = load(case['backbone'], assets, 'cuda')
    pool = ReferencePool(case['backbone'], assets, config['class_ids'], args.output)
    if case['method'] == 'ogo' and case['steps'] == 16 and not (Path(args.output)/f"{case['backbone']}-reference-profile.json").exists():
        from .reference_profile import profile_references
        profile_references(velocity, pool, assets, args.output)
    seed_all(config['seed'])
    policy = Policy(velocity, case['method'], config['methods'][case['method']])
    profiler = Profiler(policy, dino, reward, pool, config, case['method'], case['steps'], folder)
    profiler.check_identity = identity({'profiler': profiler.check_identity, 'release': run_identity})
    if case['method'] == 'ogo':
        profiler.prepare_matching()
    torch.cuda.synchronize()
    startup = time.perf_counter()-start
    model_record = {'provenance': provenance, 'trainable_parameters': sum(p.numel() for p in policy.trainable),
                    'lora_targets': policy.lora_targets, 'pool_identity': pool.identity}
    write(folder/'model.json', model_record, immutable=True)
    write(folder/f"startup-{os.environ['SLURM_JOB_ID']}.json", {'startup_seconds': startup})
    if args.resume and (folder/'checkpoint.pt').exists():
        proof = read(folder/'verification.json')
        if proof['run_identity'] != identity(run_identity) or not proof['passed']:
            raise ValueError('Resume verification proof differs')
        cursor = profiler.resume(folder/'checkpoint.pt')
        warm_resume(profiler, folder/'checkpoint.pt', cursor, folder, config['warmup_updates'])
        raw = [read(folder/f'update-{k:02d}.json') for k in range(3,cursor)]
        warm = proof['warmup']
        # A measurement written immediately before an interrupted checkpoint is
        # uncommitted. Preserve it separately before repeating that exact update.
        dangling = folder/f'update-{cursor:02d}.json'
        if dangling.exists():
            dangling.rename(folder/f'precommit-{cursor:02d}-{os.environ["SLURM_JOB_ID"]}.json')
        finish_case(args, config, case, run_identity, profiler, folder, raw, warm, cursor, startup)
        return
    data = profiler.batch(0)
    M = config['microbatch']
    x, labels, ref = (data[k][:M] for k in ('initial', 'labels', 'reference'))
    verify_start = time.perf_counter()
    with torch.no_grad():
        actual = euler(policy, x, labels, case['steps'])
        base = x.clone()
        with reference_policy(policy):
            for k in range(case['steps']):
                base = base+velocity(base,k/case['steps'],labels)/case['steps']
        difference = float((actual-base).abs().max())
        if difference > config['thresholds']['zero_correction_atol']:
            raise ValueError('Initial corrective/LoRA policy differs from frozen Euler')
        cost = reward(actual, labels, ref).costs
        expected = reward.lpips_model(reward.decoder(actual).clamp(-1,1), reward.decoder(ref).clamp(-1,1)).flatten()
        torch.testing.assert_close(cost, expected, rtol=0, atol=0)
    warm = []
    checkpoint_path = folder/'resume-check.pt'
    warm.append(profiler.step(0))
    profiler.checkpoint(checkpoint_path, 1)
    warm.append(profiler.step(1))
    expected_state = trainable_state(policy)
    expected_optimizer = cpu_snapshot(profiler.optimizer.state_dict())
    cursor = profiler.resume(checkpoint_path)
    replay = profiler.step(cursor)
    if not equal_tree(expected_state, trainable_state(policy)) or not equal_tree(expected_optimizer, cpu_snapshot(profiler.optimizer.state_dict())):
        raise ValueError('Interrupted/resumed execution differs from uninterrupted training')
    warm.append(profiler.step(2))
    if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
        raise ValueError('Frozen backbone acquired parameter gradients')
    proof = {'passed': True, 'zero_initial_euler_max_error': difference, 'reward_parity': True,
             'exact_optimizer_resume': True, 'verification_seconds': time.perf_counter()-verify_start,
             'replay_measurement': replay, 'warmup': warm, 'run_identity': identity(run_identity)}
    write(folder/'verification.json', proof)
    # Save a committed starting point; measured rounds are identity checked and atomic.
    profiler.checkpoint(folder/'checkpoint.pt', 3)
    finish_case(args, config, case, run_identity, profiler, folder, [], warm, 3, startup)


def finish_case(args, config, case, run_identity, profiler, folder, raw, warm, cursor, startup):
    import torch
    from .profiling import Meter
    from .policy import euler
    policy, velocity, reward = profiler.policy, profiler.policy.velocity, profiler.reward
    M = config['microbatch']
    for cursor in range(cursor,8):
        if time.time()+max(120, 2*max(w['wall_seconds'] for w in warm)) > args.deadline:
            write(folder/'status.json', {'status': 'BUDGET_STOP', 'completed_timed_updates': len(raw)})
            return
        torch.cuda.reset_peak_memory_stats()
        row = profiler.step(cursor)
        raw.append(row)
        write(folder/f'update-{cursor:02d}.json', row, immutable=True)
        profiler.checkpoint(folder/'checkpoint.pt', cursor+1)
    development = []
    for cursor in range(config['development_batches']):
        batch = profiler.batch(cursor, 'development')
        meter = Meter(velocity)
        with meter.phase('deployment_euler'):
            with torch.no_grad():
                ys = [euler(policy, batch['initial'][i:i+M], batch['labels'][i:i+M], case['steps'])
                      for i in range(0,config['batch_size'],M)]
        with meter.phase('development_reward'):
            with torch.no_grad():
                costs = torch.cat([reward(y,batch['labels'][i:i+M],batch['reference'][i:i+M]).costs
                                   for y,i in zip(ys,range(0,config['batch_size'],M))])
        development.append({'seconds': meter.seconds, 'network_calls': meter.calls,
            'network_state_evaluations': meter.evaluations, 'lpips': float(costs.mean()), 'input_ids': batch['ids']})
    result = {'status': 'COMPLETE', **case, 'identity': identity(run_identity), 'startup_seconds': startup,
              'warmup_updates': 3, 'timed_updates': 5, 'measurements': raw, 'development': development,
              'batch_size': config['batch_size'], 'microbatch': M, 'settings': profiler.settings,
              'trainable_parameters': sum(p.numel() for p in policy.trainable)}
    write(folder/'profile.json', result)
    write(folder/'status.json', {'status': 'COMPLETE'})
    print(json.dumps({'case': case['key'], 'status': 'COMPLETE',
                      'seconds_per_round': sum(r['wall_seconds'] for r in raw)/len(raw)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify','profile','case','audit','report'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--assets')
    parser.add_argument('--case')
    parser.add_argument('--deadline', type=float, default=float('inf'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    validate(config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {'config': config, 'source': sources(), 'assets_sha256': digest(args.assets) if args.assets else None}
    if args.action in ('audit','report'):
        from .study_report import audit, report
        (audit if args.action == 'audit' else report)(output, config)
        return
    allocation_required()
    if args.action == 'verify':
        from .study_verify import verify
        write(output/'verification-identity.json', run_identity, immutable=True)
        verify(output, config['thresholds'])
        print('Numerical verification passed', flush=True)
        return
    if not args.assets:
        raise ValueError('Profiling requires an explicit asset manifest')
    if not math.isfinite(args.deadline) or args.deadline <= time.time():
        raise ValueError('Profiling requires a finite future --deadline within the approved allocation')
    proof = read(output/'numerical-verification.json')
    verified = read(output/'verification-identity.json')
    if not proof['passed'] or verified != run_identity:
        raise ValueError('Verification identity differs from the profiling release')
    write(output/'identity.json', run_identity, immutable=True)
    if args.action == 'case':
        case = next((c for c in matrix(config) if c['key'] == args.case), None)
        if case is None:
            raise ValueError('Unknown matrix case')
        try:
            case_run(args, config, case, run_identity)
        except Exception as exc:
            write(output/case['key']/'status.json', {'status': 'FAILED', 'error': str(exc), 'traceback': traceback.format_exc()})
            raise
        return
    statuses = []
    for case in matrix(config):
        path = output/case['key']/'profile.json'
        if path.exists():
            if not args.resume or read(path)['identity'] != identity(run_identity):
                raise ValueError('Existing results require matching --resume')
            statuses.append({'case': case['key'], 'status':'COMPLETE'})
            continue
        remaining = args.deadline-time.time()
        if remaining < 600:
            statuses.append({'case': case['key'], 'status':'UNCOVERED_BUDGET'})
            continue
        command = [sys.executable,'-B','-m','ogo.study','case','--config',str(Path(args.config).resolve()),
                   '--assets',str(Path(args.assets).resolve()),'--output',str(output.resolve()),
                   '--case',case['key'],'--deadline',str(args.deadline)]
        if args.resume:
            command.append('--resume')
        with (output/(case['key']+'.log')).open('w',encoding='utf-8') as log:
            try:
                child = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,remaining-60))
                status = read(output/case['key']/'status.json')['status']
                if child.returncode and status == 'COMPLETE':
                    raise RuntimeError('Child status contradicts process exit')
            except subprocess.TimeoutExpired:
                status = 'BUDGET_STOP'
                write(output/case['key']/'status.json', {'status':status})
        statuses.append({'case':case['key'],'status':status})
        write(output/'progress.json', {'matrix':statuses, 'job_id':os.environ['SLURM_JOB_ID']})
        print(json.dumps(statuses[-1]),flush=True)
    write(output/'coverage.json', {'matrix':statuses, 'expected':24,'complete':sum(r['status']=='COMPLETE' for r in statuses)})
    from .study_report import report
    report(output, config)


if __name__ == '__main__':
    main()
