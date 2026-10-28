"""Audit coverage and price explicit rollout/replay and reference-pool budgets."""
import csv
from pathlib import Path
import statistics

from .study_io import identity, read, write


def audit(output, config):
    from .study import matrix
    output = Path(output)
    provenance = read(output/'identity.json')
    if provenance['config'] != config:
        raise ValueError('Audit configuration differs from the recorded experiment')
    expected_identity = identity(provenance)
    if (read(output/'verification-identity.json') != provenance or
            not read(output/'numerical-verification.json')['passed']):
        raise ValueError('Numerical verification does not match the recorded experiment')
    checks = []
    for case in matrix(config):
        folder = output/case['key']
        if not (folder/'profile.json').exists():
            checks.append({'case':case['key'],'passed':False,'reason':'no complete profile'})
            continue
        profile, verification = read(folder/'profile.json'), read(folder/'verification.json')
        rows = profile['measurements']
        passed = (profile['identity'] == expected_identity and verification['run_identity'] == expected_identity
                  and verification['passed'] and len(rows) == config['timed_updates']
                  and len(profile['development']) == config['development_batches']
                  and all(row['wall_seconds'] > 0 and row['optimizer_steps'] > 0 for row in rows))
        passed = passed and all(read(folder/f'update-{row["cursor"]:02d}.json') == row for row in rows)
        timed_ids = {x for row in rows for x in row['input_ids']}
        dev_ids = {x for row in profile['development'] for x in row['input_ids']}
        passed = passed and not bool(timed_ids & dev_ids)
        checks.append({'case':case['key'],'passed':passed,'independent_development':not bool(timed_ids & dev_ids)})
    value = {'passed': all(c['passed'] for c in checks), 'covered':sum(c['passed'] for c in checks), 'expected':24,'checks':checks}
    write(output/'audit.json',value)
    return value


def report(output, config):
    output = Path(output)
    quality = audit(output, config)
    rows, prices, pools = [], [], []
    for item in quality['checks']:
        if not item['passed']:
            continue
        value = read(output/item['case']/'profile.json')
        measurements = value['measurements']
        seconds = statistics.mean(r['wall_seconds'] for r in measurements)
        phases = {k:statistics.mean(r['phase_seconds'].get(k,0.) for r in measurements)
                  for k in {k for r in measurements for k in r['phase_seconds']}}
        deployment = statistics.mean(r['seconds']['deployment_euler'] for r in value['development'])
        optimizer_steps = measurements[0]['optimizer_steps']
        proof = read(output/item['case']/'verification.json')
        collection = 0.
        if value['method'] == 'offline_d3po':
            initial = proof['warmup'][0]['phase_seconds']
            collection = initial.get('preference_collection',0.)+initial.get('reward',0.)
        row = {'case':item['case'],'backbone':value['backbone'],'method':value['method'],'steps':value['steps'],
               'seconds_per_round':seconds, 'minimum_round_seconds':min(r['wall_seconds'] for r in measurements),
               'maximum_round_seconds':max(r['wall_seconds'] for r in measurements),
               'optimizer_steps_per_round':optimizer_steps,'seconds_per_optimizer_step':seconds/optimizer_steps,
               'deployment_seconds_per_image':deployment/config['batch_size'],
               'generation_50k_gpu_hours':deployment/config['batch_size']*50000/3600,
               'peak_allocated_gb':max(r['peak_allocated_bytes'] for r in measurements)/1e9,
               'trainable_parameters':value['trainable_parameters'],'phase_seconds':phases,
               'preference_collection_seconds_per_pair':collection/(config['batch_size']/2)}
        rows.append(row)
        for updates in (128,512,1024):
            # Fixed-pool D3PO collection is separately priced once; online methods collect per round.
            hours = seconds/optimizer_steps*updates/3600
            prices.append({'case':item['case'],'optimizer_updates':updates,'training_gpu_hours':hours,
                           'new_trajectories':0 if value['method']=='offline_d3po' else config['batch_size']*updates/optimizer_steps,
                           'scope':'same update budget for 100-class screen or 1000-class run; class coverage alone does not set update cost'})
    for backbone in config['backbones']:
        path = output/f'{backbone}-reference-profile.json'
        if not path.exists():
            continue
        speed = read(path)['seconds_per_image']
        for classes in (100,1000):
            for paired in (8,16,32):
                matching, development = 16,10
                count = classes*(paired+matching+development)
                pools.append({'backbone':backbone,'classes':classes,'paired_per_class':paired,
                              'independent_matching_per_class':matching,'development_per_class':development,
                              'total_reference_images':count,'gross_preparation_gpu_hours':count*speed/3600,
                              'endpoint_storage_gb':count*(3*64*64 if backbone=='edm64' else 4*32*32)*4/1e9,
                              'reuse_discount':'Subtract only identity-checked unreserved records; no discount assumed in this gross figure'})
    write(output/'costs.json',{'coverage':quality,'measurements':rows,'update_budgets':prices,'reference_pool_budgets':pools,
          'limitations':['Disposable profile weights are not scientific results.','Five steady-state observations per case; no accuracy ranking.',
                        'Dataset class count does not determine update count; offline replay passes and preference collection must be priced separately.',
                        'Reference/evaluator setup and Slurm overhead are accounted separately from steady-state update costs.',
                        'Deployment entries are unmerged Euler endpoints; SiT VAE decoding and RGB output are not included. Use the separate RGB deployment audit for complete image-generation cost.']})
    if rows:
        with (output/'profiles.csv').open('w',encoding='utf-8',newline='') as f:
            flat = [{k:v for k,v in row.items() if k!='phase_seconds'} for row in rows]
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# OGO profiling and cost assessment','',f"Verified coverage: **{quality['covered']}/24 configurations**.",
           '','| Configuration | Seconds / update | Peak GB | 1,024 updates (GPU h) | 50K unmerged Euler endpoints (GPU h) |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case']} | {row['seconds_per_optimizer_step']:.2f} | {row['peak_allocated_gb']:.2f} | {row['seconds_per_optimizer_step']*1024/3600:.2f} | {row['generation_50k_gpu_hours']:.2f} |")
    lines += ['','All timings use the declared microbatch and accumulation. OGO and DRaFT take one optimizer update per fresh batch; DDPO/Flow/lambda take two. Offline D3PO replays a frozen preference batch.',
              '','Endpoint timing excludes SiT VAE decoding and retains unmerged LoRA. Use the separately parity-checked RGB deployment measurements for complete image-generation costs.',
              '','Reducing cached cases reduces preparation and storage. Fewer rollouts, updates or replay passes are required to reduce training compute.',
              '','Uncovered configurations: '+', '.join(c['case'] for c in quality['checks'] if not c['passed'])+'.',
              '','Reference-pool preparation and preference collection must be added; this table does not include final metric computation.']
    if pools:
        lines += ['','| Backbone | Classes | Paired/class | Matching/class | Development/class | Gross reference GPU h | Endpoint GB |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for p in pools:
            lines.append(f"| {p['backbone']} | {p['classes']} | {p['paired_per_class']} | 16 | 10 | {p['gross_preparation_gpu_hours']:.2f} | {p['endpoint_storage_gb']:.3f} |")
        lines += ['','These pools are shared across methods and horizons. Identity-checked reuse reduces the gross preparation costs; it does not reduce policy updates.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows
