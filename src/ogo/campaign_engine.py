"""Production recurrence with sharded preferences and committed update counters."""
import copy
import os
from pathlib import Path
import time

import numpy as np
import torch

from .backbones import load, Policy
from .campaign_data import Pool
from .campaign_runtime import BudgetStop, read,write,sha,token,key,lock,configure_torch
from .checkpoint import atomic_save,cpu_snapshot,rng_state,restore_rng
from .policy import collect
from .profiling import Profiler,Meter,trainable_state,restore_trainable


def equal_state(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def cuda_tree(value):
    if isinstance(value,torch.Tensor):
        return value.cuda()
    if isinstance(value,dict):
        return {k:cuda_tree(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cuda_tree(v) for v in value]
    return value


def preference_batch(policy,reward,pool,config,steps,cursor):
    """Both members identify the same actual paired reference and initial noise."""
    data=pool.batch(cursor,pairs=True)
    batches=[];rewards=[];B,M=config['batch_size'],config['microbatch']
    meter=Meter(policy.velocity)
    for offset in range(0,B,M):
        x,labels,ref=(data[k][offset:offset+M] for k in ('initial','labels','reference'))
        if not torch.equal(x[::2],x[1::2]) or not torch.equal(ref[::2],ref[1::2]):
            raise ValueError('Preference pair lost its shared initial/reference')
        generator=torch.Generator(device=x.device).manual_seed(710000+cursor*B+offset)
        with meter.phase('preference_collection'):
            trajectory=collect(policy,x,labels,steps,'sit256',config['methods']['offline_d3po']['noise_level'],generator)
        with meter.phase('reward'):
            with torch.no_grad():
                values=-reward(trajectory['states'][-1],labels,ref).costs
        trajectory['winner']=torch.sign(values[::2]-values[1::2])
        trajectory['reference_logps']=trajectory['logps'].clone()
        batches.append(trajectory);rewards.append(values)
    ids=data['ids']
    if any(ids[i]!=ids[i+1] for i in range(0,B,2)):
        raise ValueError('Preference member identity differs from its true input')
    return dict(trajectories=cpu_snapshot(batches),rewards=torch.cat(rewards).cpu(),input_ids=ids,
        pair_ids=[f'n{steps}/pair-{cursor*16+i:05d}' for i in range(16)],
        phase_seconds=meter.seconds,network_calls=meter.calls,network_state_evaluations=meter.evaluations)


def collect_preferences(config,assets,output,identity,guard,steps):
    folder=Path(output)/'preferences'/f'n{steps}';folder.mkdir(parents=True,exist_ok=True)
    pool=Pool(config,output)
    data_identity=token(dict(campaign=identity,pool=pool.identity,steps=steps,pairs=8000))
    policy=reward=None;checks=[]
    for cursor in range(500):
        guard.check();target=folder/f'part-{cursor:04d}.pt'
        receipt=target.with_suffix('.json')
        if receipt.exists():
            row=read(receipt)
            if row['identity']!=data_identity or sha(target)!=row['sha256']:
                raise ValueError('Preference shard changed during resume')
        else:
            if policy is None:
                configure_torch(config['seed'])
                velocity,_,reward,_=load('sit256',assets,'cuda')
                configure_torch(config['seed'])
                policy=Policy(velocity,'offline_d3po',config['methods']['offline_d3po'])
            value=preference_batch(policy,reward,pool,config,steps,cursor)
            value.update(identity=data_identity,cursor=cursor,frozen=True)
            atomic_save(target,value)
            row=dict(identity=data_identity,sha256=sha(target),pairs=16,cursor=cursor,
                     input_ids=value['input_ids'],pair_ids=value['pair_ids'],phase_seconds=value['phase_seconds'])
            write(receipt,row,immutable=True)
        checks.append(row)
    ids=[i for row in checks for i in row['pair_ids']]
    if len(ids)!=8000 or len(set(ids))!=8000:
        raise ValueError('Preference coverage differs')
    write(folder/'ready.json',dict(identity=data_identity,campaign_identity=identity,pool=pool.identity,
          steps=steps,pairs=8000,shards={f'part-{i:04d}.pt':r['sha256'] for i,r in enumerate(checks)}),immutable=True)


class Trainer(Profiler):
    def __init__(self,policy,dino,reward,pool,config,method,steps,output,campaign_identity,preferences=None):
        super().__init__(policy,dino,reward,pool,config,method,steps,output)
        self.updates_per_round=2 if method in ('ddpo_is','flow_grpo','lambda_grpo') else 1
        self.preference_root=Path(preferences) if preferences else None
        self.preference_manifest=read(self.preference_root/'ready.json') if preferences else None
        if self.preference_manifest and self.preference_manifest['identity']!='verification-only':
            if (self.preference_manifest['campaign_identity']!=campaign_identity or
                self.preference_manifest['pool']!=pool.identity or self.preference_manifest['steps']!=steps or
                self.preference_manifest['pairs']!=8000):
                raise ValueError('Frozen preference dataset belongs to another campaign or horizon')
        self.check_identity=token(dict(campaign=campaign_identity,pool=pool.identity,method=method,steps=steps,
            preferences=self.preference_manifest))
        self.verified_shards=set()

    def batch(self,cursor,role='training'):
        return self.pool.batch(cursor,role=role)

    def step(self,cursor):
        started=time.perf_counter();io_seconds=0.
        if self.method=='offline_d3po':
            shard=cursor%500;name=f'part-{shard:04d}.pt';path=self.preference_root/name
            if name not in self.verified_shards:
                if sha(path)!=self.preference_manifest['shards'][name]:
                    raise ValueError('Offline preference tensor changed')
                self.verified_shards.add(name)
            value=torch.load(path,map_location='cpu',weights_only=True)
            if value['identity']!=self.preference_manifest['identity'] or value['cursor']!=shard or not value['frozen']:
                raise ValueError('Offline shard metadata changed')
            self.offline=cuda_tree(value['trajectories']);self.offline_ids=value['input_ids']
            torch.cuda.synchronize();io_seconds=time.perf_counter()-started
        row=super().step(cursor)
        if io_seconds:
            row['phase_seconds']['preference_io']=io_seconds
            row['wall_seconds']+=io_seconds
        if row['optimizer_steps']!=self.updates_per_round:
            raise ValueError('Optimizer counter disagrees with native recurrence')
        row.update(optimizer_updates=(cursor+1)*self.updates_per_round,
            rollout_batches=0 if self.method=='offline_d3po' else cursor+1,
            replay_batches=(cursor+1)*self.updates_per_round if self.method!='draft_k' else 0)
        if self.method=='offline_d3po':
            row['pair_presentations']=(cursor+1)*16
            row['preference_shard']=cursor%500
        return row

    def state(self,cursor):
        return dict(identity=self.check_identity,cursor=cursor,optimizer_updates=cursor*self.updates_per_round,
            trainable=trainable_state(self.policy),optimizer=cpu_snapshot(self.optimizer.state_dict()),rng=rng_state())

    def restore(self,state):
        if state['identity']!=self.check_identity or state['optimizer_updates']!=state['cursor']*self.updates_per_round:
            raise ValueError('Training resume identity/counters changed')
        restore_trainable(self.policy,state['trainable']);self.optimizer.load_state_dict(cpu_snapshot(state['optimizer']))
        restore_rng(state['rng']);self.offline=None;self.offline_ids=None
        return state['cursor']


def checkpoint(trainer,folder,cursor):
    folder=Path(folder);root=folder/'checkpoints';root.mkdir(exist_ok=True)
    updates=cursor*trainer.updates_per_round;name=f'update-{updates:04d}.pt'
    old=read(folder/'latest.json') if (folder/'latest.json').exists() else None
    if old and old['cursor']==cursor:
        if sha(root/old['file'])!=old['sha256']:
            raise ValueError('Existing committed checkpoint changed')
        return old
    path=root/name
    atomic_save(path,trainer.state(cursor))
    receipt=dict(identity=trainer.check_identity,cursor=cursor,optimizer_updates=updates,file=name,sha256=sha(path))
    write(path.with_suffix('.json'),receipt)
    write(folder/'latest.json',receipt)
    if old and old['file']!=name and old['optimizer_updates'] not in (0,128,256,384,512):
        retired=root/old['file']
        if sha(retired)!=old['sha256']:
            raise ValueError('Previous rolling checkpoint changed')
        retired.unlink();retired.with_suffix('.json').unlink()
    return receipt


def warm(trainer,committed,cursor,count):
    rows=[]
    for _ in range(count):
        rows.append(trainer.step(max(0,cursor-1)))
        trainer.restore(committed)
        if not equal_state(committed,trainer.state(cursor)):
            raise ValueError('Warmup altered the committed training state')
    return rows


def train(config,assets,output,identity,guard,method,steps):
    folder=Path(output)/'training'/key(method,steps);folder.mkdir(parents=True,exist_ok=True)
    with lock(folder/'.lock') as owned:
        if not owned:
            raise RuntimeError('Duplicate training process')
        configure_torch(config['seed']);pool=Pool(config,output)
        velocity,dino,reward,provenance=load('sit256',assets,'cuda')
        configure_torch(config['seed']);policy=Policy(velocity,method,config['methods'][method])
        trainer=Trainer(policy,dino,reward,pool,config,method,steps,folder,identity,
            Path(output)/'preferences'/f'n{steps}' if method=='offline_d3po' else None)
        write(folder/'identity.json',dict(identity=trainer.check_identity,campaign=identity,model=provenance,
            trainable_parameters=sum(p.numel() for p in policy.trainable),targets=policy.lora_targets),immutable=True)
        if (folder/'complete.json').exists():
            complete=read(folder/'complete.json')
            if complete['identity']!=trainer.check_identity or complete['optimizer_updates']!=512:
                raise ValueError('Completed training identity changed')
            return
        if (folder/'latest.json').exists():
            latest=read(folder/'latest.json');path=folder/'checkpoints'/latest['file']
            if sha(path)!=latest['sha256']:
                raise ValueError('Committed checkpoint checksum changed')
            cursor=trainer.restore(torch.load(path,map_location='cpu',weights_only=False))
        else:
            cursor=0;checkpoint(trainer,folder,cursor)
        rounds=config['optimizer_updates']//trainer.updates_per_round
        if cursor<rounds:
            guard.check(600)
            committed=trainer.state(cursor)
            probes=warm(trainer,committed,cursor,config['profile_warmups'])
            write(folder/f'warmup-{os.environ["SLURM_JOB_ID"]}.json',dict(committed_updates=cursor*trainer.updates_per_round,probes=probes))
        last_seconds=120.
        try:
            while cursor<rounds:
                guard.check(max(180,2*last_seconds))
                row=trainer.step(cursor);cursor+=1;last_seconds=row['wall_seconds']
                write(folder/f'round-{cursor:04d}.json',row)
                updates=cursor*trainer.updates_per_round
                if updates%config['checkpoint_every']==0 or cursor==rounds:
                    checkpoint(trainer,folder,cursor)
                write(folder/'progress.json',dict(campaign_identity=identity,method=method,steps=steps,
                    optimizer_updates=updates,rounds=cursor,job_id=os.environ['SLURM_JOB_ID'],unix=time.time()))
                print(method,steps,updates,flush=True)
        except BudgetStop:
            checkpoint(trainer,folder,cursor)
            raise
        final=checkpoint(trainer,folder,cursor)
        write(folder/'complete.json',dict(identity=trainer.check_identity,campaign=identity,optimizer_updates=512,
            rounds=rounds,checkpoint=final,rollout_trajectories=0 if method=='offline_d3po' else rounds*32,
            pair_presentations=8192 if method=='offline_d3po' else 0),immutable=True)
