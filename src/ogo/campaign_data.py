"""Frozen reference indices, balanced schedules and independently committed shards."""
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import time

from .campaign_runtime import read,write,sha,token,lock


def pool_seed(label, draw):
    return (1664525*(800_000_000+label*100000+draw)+1013904223)&0xffffffff


@contextmanager
def reference_attention():
    """Reproduce the pinned teacher runtime, restoring the policy runtime afterward."""
    import torch
    backend=torch.backends.cuda
    previous=(backend.flash_sdp_enabled(),backend.mem_efficient_sdp_enabled(),backend.math_sdp_enabled())
    backend.enable_flash_sdp(False);backend.enable_mem_efficient_sdp(False);backend.enable_math_sdp(True)
    try:yield
    finally:
        backend.enable_flash_sdp(previous[0]);backend.enable_mem_efficient_sdp(previous[1]);backend.enable_math_sdp(previous[2])


def build_index(config, assets, output, campaign_identity):
    output=Path(output);folder=output/'references';folder.mkdir(exist_ok=True)
    reserved=set(read(assets['reuse_audit'])['backbones']['sit256']['reserved_profile_ids'])
    reserved.update(config['reference_reconstruction_probe_ids'])
    records=[];missing=[];sources={}
    for label in config['class_ids']:
        meta_path=Path(assets['sit_pool'])/f'class-{label:04d}.json'
        available={r:[] for r in ('training','development','final')}
        if meta_path.exists():
            meta=read(meta_path)
            if meta['identity']!=assets['sit_pool_identity'] or meta['class_id']!=label:
                raise ValueError('Legacy reference source changed')
            sources[str(meta_path)]=sha(meta_path)
            for j,seed in enumerate(meta['seeds']):
                rid=f'sit/teacher-training/{label}/{j}'
                if rid in reserved:
                    continue
                if j<64:
                    available['training'].append((j,seed,rid))
                elif j>=128:
                    available['development'].append((j,seed,rid))
        available['final']=available['development'][2:12]
        available['development']=available['development'][:2]
        for role,number,start in [('training',8,0),('development',2,128),('final',10,130)]:
            selected=available[role][:number]
            if selected and len(selected)!=number:
                raise ValueError('Partial existing role coverage needs an explicit reviewed index')
            for j in range(number):
                if selected:
                    draw,seed,rid=selected[j]
                    record=dict(id=rid,class_id=label,role=role,slot=j,seed=seed,
                        file=str(meta_path.with_suffix('.pt')),row=draw,sha256=meta['sha256'],kind='legacy')
                else:
                    draw=start+j;rid=f'sit/teacher-training/{label}/{draw}'
                    if rid in reserved:
                        raise ValueError('New reference collides with reserved evidence')
                    record=dict(id=rid,class_id=label,role=role,slot=j,seed=pool_seed(label,draw),kind='new')
                    missing.append(record)
                records.append(record)
    if len({r['id'] for r in records})!=20000 or len({r['seed'] for r in records})!=20000:
        raise ValueError('Reference roles/initial seeds overlap')
    for i,row in enumerate(missing):
        row.update(file=f'shards/part-{i//32:05d}.pt',row=i%32)
    value=dict(campaign_identity=campaign_identity,records=records,legacy_metadata=sources,
               missing=len(missing),new_shards=(len(missing)+31)//32,counts=dict(training=8000,development=2000,final=10000))
    write(folder/'index.json',value,immutable=True)
    return value


def seal_references(output, identity):
    output=Path(output);folder=output/'references';index=read(folder/'index.json')
    hashes={}
    for i in range(index['new_shards']):
        p=folder/f'shards/part-{i:05d}.json'
        if not p.exists():
            return False
        row=read(p)
        if row['campaign_identity']!=identity or sha(p.with_suffix('.pt'))!=row['sha256']:
            raise ValueError('Prepared reference shard differs')
        hashes[p.name]=row['sha256']
    legacy={r['file']:r['sha256'] for r in index['records'] if r['kind']=='legacy'}
    for name,expected in legacy.items():
        if sha(name)!=expected:
            raise ValueError('Reused reference bytes changed')
    write(folder/'ready.json',dict(campaign_identity=identity,index_sha256=sha(folder/'index.json'),
          shards=hashes,legacy=legacy,count=20000),immutable=True)
    return True


def prepare(config, assets, output, identity, guard, part, parts=2):
    import torch
    from .backbones import load
    from .checkpoint import atomic_save
    from .policy import euler
    folder=Path(output)/'references';index=read(folder/'index.json')
    missing=[r for r in index['records'] if r['kind']=='new']
    velocity=None
    for i in range(part,index['new_shards'],parts):
        guard.check()
        rows=missing[i*32:(i+1)*32];target=folder/f'shards/part-{i:05d}.pt'
        target.parent.mkdir(exist_ok=True)
        with lock(target.with_suffix('.lock')) as owned:
            if not owned:
                raise RuntimeError('Duplicate reference preparation worker')
            if target.with_suffix('.json').exists():
                receipt=read(target.with_suffix('.json'))
                if receipt['ids']!=[r['id'] for r in rows] or receipt['campaign_identity']!=identity or sha(target)!=receipt['sha256']:
                    raise ValueError('Reference shard resume identity changed')
                continue
            if velocity is None:
                velocity,_,_=load('sit256',assets,'cuda')[:3]
            initial=torch.stack([torch.randn(4,32,32,generator=torch.Generator().manual_seed(r['seed'])) for r in rows]).cuda()
            labels=torch.tensor([r['class_id'] for r in rows],device='cuda')
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.no_grad(),reference_attention():
                endpoint=euler(velocity,initial,labels,256,backward_steps=0)
            torch.cuda.synchronize()
            if not torch.isfinite(endpoint).all():
                raise FloatingPointError('Nonfinite frozen reference endpoint')
            atomic_save(target,dict(latent=endpoint.cpu(),ids=[r['id'] for r in rows]))
            write(target.with_suffix('.json'),dict(campaign_identity=identity,ids=[r['id'] for r in rows],
                  sha256=sha(target),seconds=time.perf_counter()-started,nfe=256,cfg_state_evaluations=len(rows)*512),immutable=True)
    write(folder/f'part-{part}-complete.json',dict(campaign_identity=identity,part=part,parts=parts),immutable=True)
    seal_references(output,identity)


class Pool:
    backbone='sit256'

    def __init__(self,config,output,require_ready=True):
        self.config,self.output=config,Path(output)
        self.folder=self.output/'references';index=read(self.folder/'index.json')
        self.records={c:{r:[] for r in ('training','development','final')} for c in range(1000)}
        for row in index['records']:
            self.records[row['class_id']][row['role']].append(row)
        self.index=index;self.cache=OrderedDict();self.checked=set();self.ready=None
        if (len(index['records'])!=20000 or len({r['id'] for r in index['records']})!=20000 or
            any(len(self.records[c][role])!=number for c in range(1000)
                for role,number in [('training',8),('development',2),('final',10)])):
            raise ValueError('Frozen reference population is incomplete or overlapping')
        if require_ready:
            ready=read(self.folder/'ready.json')
            if ready['index_sha256']!=sha(self.folder/'index.json'):
                raise ValueError('Reference index changed after preparation')
            self.ready=ready;self.identity=token(ready)
        else:
            self.identity=token(index)

    def rows(self,cursor,role='training',pairs=False,batch_size=32):
        if role!='training':
            n=self.config['reference'][role+'_per_class']
            start=cursor*batch_size
            return [self.records[i//n][role][i%n] for i in range(start,min(start+batch_size,1000*n))]
        rows=[];groups=batch_size//4
        for g in range(groups):
            visit,offset=divmod(cursor*groups+g,1000)
            label=self.config['class_order'][offset]
            count=2 if pairs else 4
            for j in range(count):
                record=self.records[label]['training'][(visit*count+j)%8]
                rows.extend([record,record] if pairs else [record])
        return rows

    def materialize(self, rows, device='cuda'):
        import torch
        tensors=[]
        for row in rows:
            p=Path(row['file']) if row['kind']=='legacy' else self.folder/row['file']
            key=str(p)
            if key not in self.checked:
                expected=row['sha256'] if row['kind']=='legacy' else self.ready['shards'][p.with_suffix('.json').name]
                if sha(p)!=expected:
                    raise ValueError('Reference tensor changed')
                self.checked.add(key)
            if key not in self.cache:
                self.cache[key]=torch.load(p,map_location='cpu',weights_only=True)
                while len(self.cache)>16:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            if row['kind']=='new' and self.cache[key]['ids'][row['row']]!=row['id']:
                raise ValueError('Reference tensor row identity differs')
            tensors.append(self.cache[key]['latent'][row['row']].reshape(4,32,32))
        initial=torch.stack([torch.randn(4,32,32,generator=torch.Generator().manual_seed(r['seed'])) for r in rows])
        return dict(initial=initial.to(device),reference=torch.stack(tensors).to(device),
                    labels=torch.tensor([r['class_id'] for r in rows],device=device),ids=[r['id'] for r in rows])

    def batch(self,cursor,batch_size=32,group_size=4,role='training',device='cuda',pairs=False):
        if (batch_size,group_size)!=(32,4):
            raise ValueError('Campaign batch geometry changed')
        return self.materialize(self.rows(cursor,role,pairs,batch_size),device)
