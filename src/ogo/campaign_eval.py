"""Deterministic Euler evaluation with shared cases and streamed metric evidence."""
import gc
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn.utils import parametrize

from .backbones import load,Policy,checked_weight
from .campaign_data import Pool
from .campaign_runtime import read,write,sha,key,configure_torch,BudgetStop
from .profiling import restore_trainable
from .policy import euler


def save_npz(path,**values):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f,**values);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def measurement_batch(policy,dino,reward,classifier,preprocess,data,steps):
    seconds={}
    def measured(name,fn):
        torch.cuda.synchronize();began=time.perf_counter();value=fn();torch.cuda.synchronize()
        seconds[name]=time.perf_counter()-began;return value
    with torch.no_grad():
        latent=measured('euler',lambda:euler(policy,data['initial'],data['labels'],steps,backward_steps=0))
        decoded,reference=measured('decode',lambda:(reward.decoder(latent),reward.decoder(data['reference'])))
        rgb=(decoded+1).div(2).clamp(0,1)
        lpips=measured('lpips',lambda:reward.lpips_model(decoded.clamp(-1,1),reference.clamp(-1,1)).reshape(-1))
        logits=measured('classifier',lambda:classifier(preprocess(rgb)))
        if logits.shape!=(len(rgb),1000) or not torch.isfinite(logits).all():
            raise ValueError('Classifier output does not cover all 1,000 classes')
        prediction=logits.topk(5,dim=1).indices
        features=measured('dino',lambda:dino.encoder.forward_features(dino.preprocess(decoded))['x_norm_clstoken'])
        features=features/torch.linalg.vector_norm(features,dim=1,keepdim=True)
    if not torch.isfinite(latent).all() or not torch.isfinite(lpips).all() or not torch.isfinite(features).all():
        raise FloatingPointError('Nonfinite evaluation output')
    pixels=(rgb*255+.5).clamp(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
    return dict(images=pixels,labels=data['labels'].cpu().numpy(),lpips=lpips.cpu().numpy(),
        top1=(prediction[:,0]==data['labels']).cpu().numpy(),top5=(prediction==data['labels'][:,None]).any(1).cpu().numpy(),
        dino=features.cpu().numpy()),seconds


def fid_process(config_path,assets_path,output,assets,guard,mode,samples=None):
    command=[assets['evaluation_python'],'-B',str(Path(__file__).with_name('campaign_fid.py')),
        '--config',str(config_path),'--assets',str(assets_path),'--output',str(output),
        '--deadline',str(guard.deadline),'--mode',mode]
    if samples is not None:command+=['--samples',str(samples)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',MKL_NUM_THREADS='16',
             PYTHONPATH=assets['evaluation_overlay'])
    log=Path(samples)/'fid.log' if samples else Path(output)/('fid-'+mode+'.log')
    with log.open('a',encoding='utf-8') as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
    if result.returncode==75:raise BudgetStop('FID extraction checkpointed at deadline')
    if result.returncode:raise RuntimeError('FID worker failed; inspect '+str(log))


def summarize(folder,role,identity):
    folder=Path(folder);manifest=read(folder/'samples.json')
    values=[]
    for shard in manifest['shards']:
        path=folder/shard['metrics_file']
        if sha(path)!=shard['metrics_sha256']:
            raise ValueError('Evaluation measurements changed')
        with np.load(path) as data:values.append({k:data[k] for k in data.files})
    arrays={k:np.concatenate([r[k] for r in values]) for k in values[0]}
    labels=arrays['labels'];n=2 if role=='development' else 10
    if len(labels)!=1000*n or not np.all(np.bincount(labels,minlength=1000)==n):
        raise ValueError('Incomplete evaluation class coverage')
    per_class=[]
    for label in range(1000):
        ix=labels==label;features=arrays['dino'][ix].astype(np.float64)
        i,j=np.triu_indices(n,1)
        diversity=np.square(features[i]-features[j]).sum(1).mean()
        per_class.append(dict(class_id=label,lpips=float(arrays['lpips'][ix].mean()),
            top1=float(arrays['top1'][ix].mean()),top5=float(arrays['top5'][ix].mean()),dino_diversity=float(diversity)))
    result=dict(campaign_identity=identity,role=role,count=len(labels),
        lpips=float(arrays['lpips'].mean()),reward=-float(arrays['lpips'].mean()),
        top1=float(arrays['top1'].mean()),top5=float(arrays['top5'].mean()),
        dino_diversity=float(np.mean([r['dino_diversity'] for r in per_class])),per_class=per_class,
        samples_sha256=sha(folder/'samples.json'))
    write(folder/'metrics.json',result,immutable=True)
    return result


def committed_snapshot(folder,update):
    """A per-snapshot receipt is not published until the atomic latest pointer advances."""
    folder=Path(folder);latest=read(folder/'latest.json')
    if latest['optimizer_updates']<update:
        raise RuntimeError('Requested snapshot has not been committed')
    path=folder/'checkpoints'/f'update-{update:04d}.pt';record=read(path.with_suffix('.json'))
    digest=sha(path)
    if (record['optimizer_updates']!=update or record['file']!=path.name or record['identity']!=latest['identity'] or
        record['identity']!=read(folder/'identity.json')['identity'] or digest!=record['sha256'] or
        (latest['optimizer_updates']==update and record!=latest)):
        raise ValueError('Committed snapshot identity differs')
    return path,record,digest


def evaluate(config,assets,output,identity,guard,method,steps,role,update,config_path,assets_path):
    output=Path(output);name=key(method,steps)
    folder=output/'evaluation'/name/f'{role}-u{update:04d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        if read(folder/'complete.json')['campaign_identity']!=identity:
            raise ValueError('Evaluation completion identity changed')
        return
    pool=Pool(config,output);count=2000 if role=='development' else 10000
    if method=='control':
        if update!=0:raise ValueError('Frozen control has no optimizer updates')
        checkpoint_sha='frozen-reference-weights'
    else:
        path,checkpoint_record,checkpoint_sha=committed_snapshot(output/'training'/name,update)
    descriptor=dict(campaign_identity=identity,method=method,steps=steps,role=role,update=update,
                    checkpoint_sha256=checkpoint_sha,pool_identity=pool.identity,count=count)
    write(folder/'identity.json',descriptor,immutable=True)
    configure_torch(config['seed'])
    policy=dino=reward=classifier=preprocess=None
    shards=[]
    for start in range(0,count,128):
        guard.check();index=start//128;receipt=folder/f'part-{index:04d}.json'
        end=min(start+128,count)
        if receipt.exists():
            record=read(receipt)
            if record['identity']!=descriptor or sha(folder/record['metrics_file'])!=record['metrics_sha256']:
                raise ValueError('Evaluation shard identity differs')
            if role=='final' and not (folder/'fid.json').exists() and sha(folder/record['image_file'])!=record['image_sha256']:
                raise ValueError('Unprocessed image shard changed')
            shards.append(record);continue
        if policy is None:
            velocity,dino,reward,_=load('sit256',assets,'cuda')
            configure_torch(config['seed'])
            if method=='control':policy=velocity
            else:
                policy=Policy(velocity,method,config['methods'][method])
                state=torch.load(path,map_location='cpu',weights_only=False)
                if state['identity']!=checkpoint_record['identity'] or state['optimizer_updates']!=update:
                    raise ValueError('Checkpoint counters/identity differ')
                restore_trainable(policy,state['trainable']);del state
            first=pool.materialize(pool.rows(0,role)[:4])
            with torch.no_grad():
                unmerged=euler(policy,first['initial'],first['labels'],steps,backward_steps=0)
                for module in list(policy.modules()):
                    if parametrize.is_parametrized(module,'weight'):
                        expected=module.weight.detach().clone()
                        parametrize.remove_parametrizations(module,'weight',leave_parametrized=True)
                        if not torch.equal(module.weight,expected):
                            raise ValueError('Materialized LoRA weight differs')
                merged=euler(policy,first['initial'],first['labels'],steps,backward_steps=0)
                error=float((merged-unmerged).abs().max())
            if error>config['thresholds']['zero_correction_atol']:
                raise ValueError('Merged deployment parity failed')
            write(folder/f'parity-{os.environ["SLURM_JOB_ID"]}.json',dict(identity=descriptor,passed=True,
                max_error=error,ids=first['ids'],optimization_updates=0),immutable=True)
            from torchvision.models import resnet50,ResNet50_Weights
            shared=read(assets['shared_manifest']);classifier=resnet50(weights=None).cuda().eval().requires_grad_(False)
            classifier.load_state_dict(torch.load(checked_weight(shared['assets']['resnet50']),map_location='cpu',weights_only=True))
            preprocess=ResNet50_Weights.IMAGENET1K_V2.transforms()
        parts={k:[] for k in ('labels','lpips','top1','top5','dino')};images=[];ids=[];representatives=[]
        seconds={k:0. for k in ('euler','decode','lpips','classifier','dino')}
        n=2 if role=='development' else 10
        for offset in range(start,end,4):
            rows=[pool.records[i//n][role][i%n] for i in range(offset,min(offset+4,end))]
            data=pool.materialize(rows)
            measurements,elapsed=measurement_batch(policy,dino,reward,classifier,preprocess,data,steps)
            for k,v in elapsed.items():seconds[k]+=v
            pixels=measurements['images']
            for j,row in enumerate(rows):
                if row['slot']==0:representatives.append((row['class_id'],pixels[j]))
            if role=='final':images.extend(pixels)
            for k in parts:parts[k].append(measurements[k])
            ids+=data['ids']
        metric_file=folder/f'metrics-{index:04d}.npz'
        save_npz(metric_file,**{k:np.concatenate(v) for k,v in parts.items()})
        if representatives:
            save_npz(folder/f'examples-{index:04d}.npz',labels=np.array([r[0] for r in representatives]),images=np.stack([r[1] for r in representatives]))
        record=dict(identity=descriptor,ids=ids,start=start,end=end,metrics_file=metric_file.name,
                    metrics_sha256=sha(metric_file),seconds=seconds,backbone_state_evaluations=(end-start)*steps*2)
        if role=='final':
            image_file=folder/f'images-{index:04d}.npz'
            save_npz(image_file,images=np.stack(images),labels=np.concatenate(parts['labels']))
            record.update(image_file=image_file.name,image_sha256=sha(image_file))
        write(receipt,record,immutable=True);shards.append(record)
    write(folder/'samples.json',dict(**descriptor,shards=shards),immutable=True)
    metrics=summarize(folder,role,identity)
    del policy,dino,reward,classifier;gc.collect();torch.cuda.empty_cache()
    if role=='final':
        if not (folder/'fid.json').exists():
            fid_process(config_path,assets_path,output,assets,guard,'generated',folder)
        fid=read(folder/'fid.json')
        if fid['campaign_identity']!=identity:raise ValueError('FID identity changed')
        for record in shards:
            image_file=folder/record['image_file']
            if image_file.exists():
                if image_file.parent.resolve()!=folder.resolve() or sha(image_file)!=record['image_sha256']:
                    raise ValueError('Temporary image retirement checksum/path differs')
                image_file.unlink()
        write(folder/'image-retirement.json',dict(preserved='metrics, ADM features, statistics, seeds, image hashes and fixed per-class examples',
            files=[r['image_file'] for r in shards]),immutable=True)
    write(folder/'complete.json',dict(**descriptor,metrics_sha256=sha(folder/'metrics.json'),
        fid_sha256=sha(folder/'fid.json') if role=='final' else None),immutable=True)
