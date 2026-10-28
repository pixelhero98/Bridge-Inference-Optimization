"""Real-model recurrence parity, interrupted replay and production feasibility."""
import gc
from pathlib import Path
import time
from unittest.mock import patch

import torch
from torch.nn.utils import parametrize

from .backbones import load,Policy,checked_weight
from .campaign_data import build_index,Pool,reference_attention
from .campaign_engine import Trainer,preference_batch,equal_state,warm
from .campaign_eval import fid_process,measurement_batch,save_npz
from .campaign_runtime import METHODS,configure_torch,read,write,sha,token,protected
from .checkpoint import atomic_save,cpu_snapshot,rng_state
from .policy import euler
from .profiling import Profiler,trainable_state,restore_trainable
from .reference_pool import ReferencePool
from .study_verify import verify as numerical_verify


class FixturePool:
    backbone='sit256'
    def __init__(self,source):self.source,self.identity=source,source.identity
    def batch(self,cursor,batch_size=32,group_size=4,role='training',device='cuda',pairs=False):
        value=self.source.batch(cursor,batch_size,group_size,role='paired',device=device)
        if pairs:
            for k in ('initial','reference','labels'):value[k]=value[k][::2].repeat_interleave(2,dim=0)
            value['ids']=[v for v in value['ids'][::2] for _ in range(2)]
        return value


class InterruptedReplay(RuntimeError):pass


def verify(config,assets,output,identity,guard,config_path,assets_path):
    output=Path(output);folder=output/'verification';folder.mkdir(exist_ok=True)
    if (folder/'verified.json').exists():
        complete=read(folder/'verified.json')
        if complete['campaign_identity']!=identity:raise ValueError('Completed verification identity changed')
        if not complete['feasible_with_reserve']:raise RuntimeError('Complete matrix does not fit the measured budget')
        return
    numerical_verify(folder,config['thresholds']);configure_torch(config['seed'])
    index=build_index(config,assets,output,identity)
    if index['missing']!=18000:
        raise ValueError('Reference preparation differs from the approved reuse inventory')
    future=Pool(config,output,require_ready=False)
    for nrounds,pairs,min_count in [(256,False,8),(512,False,16),(500,True,16)]:
        counts=[0]*1000;ids=[]
        for k in range(nrounds):
            for row in future.rows(k,pairs=pairs):counts[row['class_id']]+=1;ids.append(row['id'])
        if min(counts)<min_count or len(ids)!=nrounds*32:
            raise ValueError('Class-balanced production traversal failed')
        if pairs and len(set(ids))!=8000:raise ValueError('Preference traversal does not cover all 8,000 cases')
    if not (folder/'fid/verified.json').exists():
        fid_process(config_path,assets_path,output,assets,guard,'verify')
    fid=read(folder/'fid/verified.json')
    if not fid['passed'] or fid['campaign_identity']!=identity:raise ValueError('Evaluator verification identity differs')
    profile_config=read(Path(__file__).parents[2]/'configs/ogo-profile.json')
    fixture_root=folder/'fixture';fixture_root.mkdir(exist_ok=True)
    legacy=ReferencePool('sit256',assets,profile_config['class_ids'],fixture_root);pool=FixturePool(legacy)
    velocity,dino,reward,model_provenance=load('sit256',assets,'cuda')
    from torchvision.models import resnet50,ResNet50_Weights
    shared=read(assets['shared_manifest']);classifier=resnet50(weights=None).cuda().eval().requires_grad_(False)
    classifier.load_state_dict(torch.load(checked_weight(shared['assets']['resnet50']),map_location='cpu',weights_only=True))
    preprocess=ResNet50_Weights.IMAGENET1K_V2.transforms()
    reference_probe=legacy.batch(0,32,32)
    if reference_probe['ids']!=config['reference_reconstruction_probe_ids']:
        raise ValueError('Pinned reconstruction probe identity differs')
    if (folder/'reference.json').exists():
        refproof=read(folder/'reference.json')
        if refproof['campaign_identity']!=identity:raise ValueError('Reference verification identity differs')
        reference_times,error=refproof['seconds'],refproof['max_error']
    else:
        reference_times=[]
        with torch.no_grad(),reference_attention():
            for i in range(3):
                guard.check(240);torch.cuda.synchronize();started=time.perf_counter()
                target=euler(velocity,reference_probe['initial'],reference_probe['labels'],256,backward_steps=0)
                torch.cuda.synchronize();reference_times.append(time.perf_counter()-started)
                error=float((target-reference_probe['reference']).abs().max())
                if error>config['thresholds']['zero_correction_atol']:
                    raise ValueError('Pinned high-step reference reconstruction failed: '+str(error))
        write(folder/'reference.json',dict(campaign_identity=identity,seconds=reference_times,max_error=error,
              ids=reference_probe['ids']),immutable=True)
    checks=[]
    for method in METHODS:
        for steps in (16,32):
            case=folder/f'{method}-n{steps}'
            if (case/'verified.json').exists():
                previous=read(case/'verified.json')
                if not previous['passed'] or previous['campaign_identity']!=identity:
                    raise ValueError('Native verification resume identity changed')
                checks.append(previous);continue
            print('verify',method,steps,flush=True)
            guard.check(900);case=folder/f'{method}-n{steps}';case.mkdir(exist_ok=True)
            configure_torch(config['seed']);policy=Policy(velocity,method,config['methods'][method])
            preferences=None
            if method=='offline_d3po':
                preferences=case/'preferences';preferences.mkdir(exist_ok=True);hashes={}
                for cursor in (0,1):
                    path=preferences/f'part-{cursor:04d}.pt'
                    if path.exists():
                        value=torch.load(path,map_location='cpu',weights_only=True)
                        if value['identity']!='verification-only' or value['cursor']!=cursor or not value['frozen']:
                            raise ValueError('Interrupted preference verification shard changed')
                    else:
                        value=preference_batch(policy,reward,pool,config,steps,cursor)
                        value.update(identity='verification-only',cursor=cursor,frozen=True)
                        atomic_save(path,value)
                    hashes[path.name]=sha(path)
                write(preferences/'ready.json',dict(identity='verification-only',shards=hashes),immutable=True)
            trainer=Trainer(policy,dino,reward,pool,config,method,steps,case,identity,preferences)
            initial=trainer.state(0)
            oracle=Profiler(policy,dino,reward,pool,config,method,steps,case)
            oracle_row=oracle.step(0)
            expected_weights=trainable_state(policy);expected_optimizer=cpu_snapshot(oracle.optimizer.state_dict())
            expected_rng=rng_state()
            trainer.restore(initial);actual=trainer.step(0)
            if not (equal_state(trainable_state(policy),expected_weights) and
                    equal_state(cpu_snapshot(trainer.optimizer.state_dict()),expected_optimizer) and equal_state(rng_state(),expected_rng)):
                raise ValueError('Production recurrence differs from verified profiling recurrence: '+method)
            committed=trainer.state(1);saved=case/'resume.pt';atomic_save(saved,committed)
            trainer.step(1);uninterrupted=trainer.state(2)
            trainer.restore(torch.load(saved,map_location='cpu',weights_only=False))
            import ogo.profiling as profiling
            original_update=profiling.finite_update
            def interrupt(*a,**kw):
                original_update(*a,**kw)
                raise InterruptedReplay('Injected interruption after first optimizer step')
            try:
                with patch.object(profiling,'finite_update',interrupt):trainer.step(1)
            except InterruptedReplay:pass
            else:raise AssertionError('Interruption fixture did not execute')
            trainer.restore(torch.load(saved,map_location='cpu',weights_only=False));trainer.step(1)
            if not equal_state(trainer.state(2),uninterrupted):
                raise ValueError('Interrupted recurrence/shard-boundary replay differs: '+method)
            trainer.restore(committed);probes=warm(trainer,committed,1,3)
            data=pool.batch(0)
            with torch.no_grad():
                before=euler(policy,data['initial'][:4],data['labels'][:4],steps,backward_steps=0)
                direct=reward.lpips_model(reward.decoder(before).clamp(-1,1),reward.decoder(data['reference'][:4]).clamp(-1,1)).reshape(-1)
                wrapped=reward(before,data['labels'][:4],data['reference'][:4]).costs
                if not torch.equal(direct,wrapped):raise ValueError('Evaluation reward adapter differs')
                for module in list(policy.modules()):
                    if parametrize.is_parametrized(module,'weight'):
                        parametrize.remove_parametrizations(module,'weight',leave_parametrized=True)
                after=euler(policy,data['initial'][:4],data['labels'][:4],steps,backward_steps=0)
                merge_error=float((after-before).abs().max())
            if merge_error>config['thresholds']['zero_correction_atol']:
                raise ValueError('Merged Euler deployment differs')
            evaluation_probes=[]
            for repeat in range(3):
                measured=[];torch.cuda.synchronize();started=time.perf_counter()
                for offset in range(0,32,4):
                    batch={k:v[offset:offset+4] for k,v in data.items()}
                    metrics,phases=measurement_batch(policy,dino,reward,classifier,preprocess,batch,steps)
                    measured.append(phases)
                save_npz(case/'metric-probe.npz',**metrics)
                evaluation_probes.append(dict(seconds=time.perf_counter()-started,phase_seconds=measured,images=32))
            record=dict(passed=True,campaign_identity=identity,method=method,steps=steps,oracle=oracle_row,actual=actual,
                exact_optimizer_rng_resume=True,interrupted_after_first_pass=True,shard_boundary=method=='offline_d3po',
                warmup_state_preserved=True,merge_max_error=merge_error,probes=probes,evaluation_probes=evaluation_probes)
            write(case/'verified.json',record,immutable=True);checks.append(record)
            print('verified',method,steps,flush=True)
            # Merging consumes the parametrization. Reload immutable backbone before the next case.
            del trainer,oracle,policy,velocity,dino,reward;gc.collect();torch.cuda.empty_cache()
            if (method,steps)!=(METHODS[-1],32):velocity,dino,reward,_=load('sit256',assets,'cuda')
    historical=read(assets['cost_assessment'])
    profiles=[r for r in historical['profiles'] if r['backbone']=='sit256' and r['method']!='ogo']
    train_hours=sum(sum(p['wall_seconds'] for p in r['probes'][1:])/2*512/
        (2 if r['method'] in ('ddpo_is','flow_grpo','lambda_grpo') else 1)/3600 for r in checks)
    generation=sum(sum(p['seconds'] for p in r['evaluation_probes'][1:])/64*10000/3600 for r in checks)
    reference_hours=index['missing']*sum(reference_times[1:])/(64*3600)
    feature_hours=170000*fid['seconds_per_image']/3600
    # Includes both controls, three development checks and reference target decoding reserve.
    modeled=train_hours+generation*1.84+reference_hours+3.784+feature_hours+5.
    value=dict(passed=True,campaign_identity=identity,cases=checks,model=model_provenance,
        reference_seconds_per_image=sum(reference_times[1:])/64,reference_reconstruction_max_error=error,
        modeled_remaining_gpu_hours=modeled,feasible_with_reserve=modeled+2<=110,
        cost_components=dict(training=train_hours,generation_and_GPU_metrics=generation*1.84,references=reference_hours,
                             preferences=3.784,ADM_features=feature_hours,setup_and_metric_reserve=5.),
        protected_files=len(protected(assets)))
    write(folder/'verified.json',value,immutable=True)
    if not value['feasible_with_reserve']:
        raise RuntimeError('Complete campaign does not fit measured remaining budget; retain all evidence without reducing coverage')
