"""Audit complete coverage before producing single-seed scientific comparisons."""
from pathlib import Path
import csv

from .campaign_runtime import METHODS,read,write,sha,key,protected,storage_bytes


def evaluation_check(folder,identity,count):
    folder=Path(folder)
    if not (folder/'complete.json').exists():return False
    v=read(folder/'complete.json');samples=read(folder/'samples.json');m=read(folder/'metrics.json')
    if (v['campaign_identity']!=identity or v['count']!=count or m['count']!=count or
        sha(folder/'metrics.json')!=v['metrics_sha256'] or sha(folder/'samples.json')!=m['samples_sha256'] or
        len(m['per_class'])!=1000 or [r['class_id'] for r in m['per_class']]!=list(range(1000))):return False
    ids=[i for shard in samples['shards'] for i in shard['ids']]
    if len(ids)!=count or len(set(ids))!=count:return False
    for shard in samples['shards']:
        if sha(folder/shard['metrics_file'])!=shard['metrics_sha256']:return False
    if count==10000:
        fid=read(folder/'fid.json')
        if (fid['campaign_identity']!=identity or fid['generated_count']!=10000 or fid['reference_count']!=50000 or
            fid['class_counts']!=[10]*1000 or sha(folder/'fid.json')!=v['fid_sha256'] or
            sha(folder/'statistics.npz')!=fid['statistics_sha256']):return False
        for i,record in enumerate(fid['features']):
            if sha(folder/'adm'/f'block-{i:04d}.npz')!=record['sha256']:return False
    return True


def audit(config,assets,output,identity):
    output=Path(output);checks=[];protected(assets)
    reference=output/'references/ready.json'
    reference_good=reference.exists()
    if reference_good:
        r=read(reference)
        reference_good=(r['campaign_identity']==identity and r['count']==20000 and
                        sha(output/'references/index.json')==r['index_sha256'])
        reference_good=reference_good and all(sha(name)==h for name,h in r['legacy'].items())
        reference_good=reference_good and all(sha((output/'references/shards'/name).with_suffix('.pt'))==h for name,h in r['shards'].items())
    checks.append(dict(name='reference_pool',passed=reference_good))
    real=output/'evaluation/real';real_good=(real/'ready.json').exists()
    if real_good:
        r=read(real/'ready.json')
        real_good=r['campaign_identity']==identity and r['count']==50000 and r['class_counts']==[50]*1000 and sha(real/'statistics.npz')==r['statistics_sha256']
        real_good=real_good and all(sha(real/'adm'/f'block-{i:04d}.npz')==v['sha256'] for i,v in enumerate(r['features']))
    checks.append(dict(name='official_validation',passed=real_good))
    rows=[]
    for method in METHODS:
        for steps in (16,32):
            name=key(method,steps);folder=output/'training'/name;done=folder/'complete.json'
            good=done.exists()
            if good:
                value=read(done);cp=value['checkpoint'];path=folder/'checkpoints'/cp['file']
                good=value['campaign']==identity and value['optimizer_updates']==512 and sha(path)==cp['sha256']
                expected_rounds=256 if method in ('ddpo_is','flow_grpo','lambda_grpo') else 512
                good=good and value['rounds']==expected_rounds
                for round_id in range(1,expected_rounds+1):
                    row=read(folder/f'round-{round_id:04d}.json')
                    good=good and row['cursor']==round_id-1 and row['optimizer_updates']==round_id*(512//expected_rounds)
            checks.append(dict(name='training/'+name,passed=bool(good)))
            for update in (128,256,384):
                f=output/'evaluation'/name/f'development-u{update:04d}';good=evaluation_check(f,identity,2000)
                checks.append(dict(name=name+f'/development/{update}',passed=bool(good)))
    for method in (*METHODS,'control'):
        for steps in (16,32):
            name=key(method,steps);update=0 if method=='control' else 512
            f=output/'evaluation'/name/f'final-u{update:04d}';done=f/'complete.json';good=evaluation_check(f,identity,10000)
            if good:
                v=read(done);fid=read(f/'fid.json');m=read(f/'metrics.json')
                good=(v['campaign_identity']==identity and v['count']==10000 and fid['reference_count']==50000 and
                      sha(f/'metrics.json')==v['metrics_sha256'] and sha(f/'fid.json')==v['fid_sha256'])
                rows.append(dict(method=method,steps=steps,updates=update,FID10K=fid['FID10K'],lpips=m['lpips'],
                    top1=m['top1'],top5=m['top5'],dino_diversity=m['dino_diversity']))
            checks.append(dict(name=name+'/final',passed=bool(good)))
            if method=='control':
                f=output/'evaluation'/name/'development-u0000'
                checks.append(dict(name=name+'/development',passed=evaluation_check(f,identity,2000)))
    for steps in (16,32):
        folder=output/'preferences'/f'n{steps}';p=folder/'ready.json';good=p.exists()
        if good:
            value=read(p);good=value['campaign_identity']==identity and value['pairs']==8000 and len(value['shards'])==500
            good=good and all(sha(folder/name)==h for name,h in value['shards'].items())
        checks.append(dict(name=f'preferences/n{steps}',passed=bool(good)))
    result=dict(campaign_identity=identity,passed=all(c['passed'] for c in checks),checks=checks,
                metrics=rows,storage_bytes=storage_bytes(output),protected_inputs_unchanged=True)
    if 'campaign_ledger' in assets and Path(assets['campaign_ledger']).exists():
        result['allocation_ledger']=read(assets['campaign_ledger'])
        if result['allocation_ledger']['committed_seconds']>config['max_aggregate_gpu_seconds']:
            raise ValueError('Consumed plus reserved allocation exceeds budget')
    if result['storage_bytes']>config['storage_bytes_cap']:raise ValueError('Storage ceiling exceeded')
    write(output/'audit.json',result);return result


def report(config,assets,output,identity):
    output=Path(output);value=audit(config,assets,output,identity);rows=value['metrics']
    if rows:
        with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    development=[];compute=[]
    for method in (*METHODS,'control'):
        for steps in (16,32):
            name=key(method,steps)
            for update in ((0,) if method=='control' else (128,256,384)):
                p=output/'evaluation'/name/f'development-u{update:04d}'/'metrics.json'
                if p.exists():
                    m=read(p);development.append(dict(method=method,steps=steps,updates=update,
                        lpips=m['lpips'],top1=m['top1'],top5=m['top5'],dino_diversity=m['dino_diversity']))
            records=[read(p) for p in sorted((output/'training'/name).glob('round-*.json'))]
            if records:
                compute.append(dict(method=method,steps=steps,optimizer_updates=records[-1]['optimizer_updates'],
                    fresh_rollout_trajectories=sum(r['new_rollouts'] for r in records),
                    pair_presentations=records[-1].get('pair_presentations',0),
                    accepted_training_seconds=sum(r['wall_seconds'] for r in records),
                    backbone_calls=sum(sum(r['network_calls'].values()) for r in records),
                    state_evaluations=sum(sum(r['network_state_evaluations'].values()) for r in records)))
    for filename,values in [('development.csv',development),('compute.csv',compute)]:
        if values:
            with (output/filename).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    lines=['# ImageNet-256 baseline campaign','',f'Complete scientific audit: **{value["passed"]}**.',
        '', 'One training seed (0), all 1,000 classes, fixed update-512 endpoints. FID uses 10,000 generated images and 50,000 official validation images. These are FID-10K results, not FID-50K benchmarks.',
        '', '| Method | Euler steps | Updates | FID-10K | Paired LPIPS | Top-1 | Top-5 | DINO diversity |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["steps"]} | {r["updates"]} | {r["FID10K"]:.4f} | {r["lpips"]:.6f} | {r["top1"]:.4f} | {r["top5"]:.4f} | {r["dino_diversity"]:.6f} |')
    lines+=['','DRaFT uses 16,384 fresh trajectories; each two-pass online RL method uses 8,192. D3PO collects 8,000 pairs per horizon and makes 8,192 pair presentations. Equal optimizer counts do not imply equal data exposure or compute.',
        '', 'No checkpoint selection or hyperparameter tuning was performed. Development readouts at 128/256/384 use fixed independent cases. Frozen controls share the same evaluation inputs.',
        '', 'Resource accounting, source/configuration identities, replay evidence, reference/preference manifests and per-case phase timings are retained in this campaign. Unfinished cases are listed in audit.json; an incomplete audit is not a completed comparison.']
    if 'allocation_ledger' in value:
        ledger=value['allocation_ledger']
        lines+=['',f'Finalized allocations: {ledger["consumed_seconds"]/3600:.6f} GH200-hours; open allocation reservations: {ledger["reserved_seconds"]/3600:.6f} hours. These include verification, preparation, probes, failures, retries and evaluation. Accepted-round compute in compute.csv excludes repeated work; scheduler accounting is authoritative.',
                f'Current run storage: {value["storage_bytes"]:,} bytes. The controller separately counts all campaign releases and failure evidence against the 64 GB ceiling.']
    (output/'readout.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return value
