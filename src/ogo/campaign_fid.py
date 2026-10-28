"""Standalone pinned ADM extraction; compatible with the isolated TF interpreter."""
import argparse
import bisect
import importlib.metadata
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

# Direct execution avoids importing the torch-dependent ogo package in TF's env.
sys.path.insert(0,str(Path(__file__).parent))
from campaign_runtime import BudgetStop,Guard,read,write,sha,token,allocated


def load(assets):
    graph=assets['assets']['adm_inception'];source=Path(assets['sources']['adm']['path'])/'evaluations/evaluator.py'
    if sha(graph['path'])!=graph['sha256']:
        raise ValueError('Pinned ADM graph changed')
    revision=subprocess.check_output(['git','-C',assets['sources']['adm']['path'],'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',assets['sources']['adm']['path'],'status','--porcelain','--untracked-files=no'],text=True).strip()
    if revision!=assets['sources']['adm']['revision'] or dirty:
        raise ValueError('Pinned ADM source changed')
    spec=importlib.util.spec_from_file_location('campaign_adm',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.INCEPTION_V3_PATH=graph['path'];module.tf.disable_eager_execution()
    cfg=module.tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=16,inter_op_parallelism_threads=2)
    evaluator=module.Evaluator(module.tf.Session(config=cfg),batch_size=32)
    identity=dict(revision=revision,source_sha256=sha(source),graph_sha256=graph['sha256'],worker_sha256=sha(__file__),
                  versions={k:importlib.metadata.version(k) for k in ('tensorflow','numpy','scipy','pyarrow','Pillow')},batch_size=32)
    return module,evaluator,identity


def crop(raw):
    with Image.open(io.BytesIO(raw)) as im:
        im=im.convert('RGB')
        while min(im.size)>=512:
            im=im.resize(tuple(d//2 for d in im.size),Image.Resampling.BOX)
        scale=256/min(im.size)
        im=im.resize(tuple(round(d*scale) for d in im.size),Image.Resampling.BICUBIC)
        pixels=np.asarray(im);y,x=((pixels.shape[i]-256)//2 for i in (0,1))
        return np.array(pixels[y:y+256,x:x+256],copy=True)


def save_npz(path,**values):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f,**values);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


class Validation:
    def __init__(self,assets,folder,guard):
        import pyarrow.parquet as pq
        self.root=Path(assets['imagenet']['path']);self.files=sorted(self.root.glob('data/validation-*.parquet'))
        if len(self.files)!=14:
            raise ValueError('Incomplete official ImageNet validation snapshot')
        proof=Path(folder)/'dataset.json'
        checks={str(p.relative_to(self.root)):assets['imagenet']['files'][str(p.relative_to(self.root))] for p in self.files}
        if proof.exists() and read(proof)['files']!=checks:
            raise ValueError('Validation dataset identity changed')
        for p in self.files:
            guard.check()
            if sha(p)!=checks[str(p.relative_to(self.root))]:
                raise ValueError('Validation parquet checksum differs')
        write(proof,dict(files=checks),immutable=True)
        self.labels=np.concatenate([pq.read_table(p,columns=['label'])['label'].to_numpy() for p in self.files]).astype(np.int16)
        if len(self.labels)!=50000 or not np.all(np.bincount(self.labels,minlength=1000)==50):
            raise ValueError('Official validation class counts differ')
        sizes=[pq.ParquetFile(p).metadata.num_rows for p in self.files]
        self.starts=np.concatenate(([0],np.cumsum(sizes))).tolist();self.cached=None

    def get(self,row):
        import pyarrow.parquet as pq
        shard=bisect.bisect_right(self.starts,row)-1;offset=row-self.starts[shard]
        f=pq.ParquetFile(self.files[shard]);group=0
        while offset>=f.metadata.row_group(group).num_rows:
            offset-=f.metadata.row_group(group).num_rows;group+=1
        if self.cached is None or self.cached[0]!=(shard,group):
            self.cached=((shard,group),f.read_row_group(group,columns=['image'])['image'])
        return crop(self.cached[1][int(offset)].as_py()['bytes'])


def extract_blocks(evaluator,evaluator_id,output,campaign_identity,blocks,guard):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[];total_seconds=0.
    for i,descriptor,get_images in blocks:
        guard.check(240)
        target=output/f'block-{i:04d}.npz';meta=target.with_suffix('.json')
        expected=dict(campaign=campaign_identity,evaluator=evaluator_id,input=descriptor)
        if meta.exists():
            saved=read(meta)
            if saved['identity']!=token(expected) or sha(target)!=saved['sha256']:
                raise ValueError('Cached ADM feature identity differs')
        else:
            images,labels=get_images()
            if images.dtype!=np.uint8 or images.shape[1:]!=(256,256,3):
                raise ValueError('ADM requires explicitly quantized uint8 RGB')
            started=time.perf_counter()
            pool,spatial=evaluator.compute_activations(images[j:j+32] for j in range(0,len(images),32))
            save_npz(target,pool=pool,spatial=spatial,labels=labels)
            saved=dict(identity=token(expected),sha256=sha(target),count=len(images),seconds=time.perf_counter()-started)
            write(meta,saved,immutable=True)
        records.append(saved);total_seconds+=saved['seconds']
    arrays=[np.load(output/f'block-{i:04d}.npz') for i in range(len(records))]
    pool=np.concatenate([a['pool'] for a in arrays]);labels=np.concatenate([a['labels'] for a in arrays])
    for a in arrays:a.close()
    return pool,labels,records,total_seconds


def run(args):
    allocated();config=read(args.config);assets=read(args.assets)
    root=Path(args.output);guard=Guard(root,config,args.deadline)
    identity=read(root/'identity.json')['identity'];shared=read(assets['shared_manifest'])
    module,evaluator,evaluator_id=load(shared)
    if args.mode=='verify':
        folder=root/'verification/fid';folder.mkdir(parents=True,exist_ok=True)
        dataset=Validation(shared,folder,guard)
        indices=sorted(int(np.flatnonzero(dataset.labels==c)[0]) for c in range(1000))
        data_started=time.perf_counter();images=np.stack([dataset.get(i) for i in indices])
        data_seconds=time.perf_counter()-data_started
        if not np.all(np.bincount(dataset.labels[indices],minlength=1000)==1):
            raise ValueError('Evaluator verification does not cover all 1,000 classes')
        rng=np.random.default_rng(835)
        evaluator.compute_activations([images[:32]])
        started=time.perf_counter();streamed,_=evaluator.compute_activations(images[i:i+32] for i in range(0,len(images),32))
        elapsed=time.perf_counter()-started
        fixture=folder/'official.npz';np.savez(fixture,images)
        native,_=evaluator.read_activations(str(fixture));error=float(np.max(np.abs(streamed-native)))
        fixture.unlink()
        if error>config['thresholds']['zero_correction_atol']:
            raise ValueError('Streaming ADM differs from official NPZ extraction')
        vectors=rng.normal(size=(64,8));stats=evaluator.compute_statistics(vectors)
        same=float(stats.frechet_distance(stats))
        if abs(same)>1e-8:
            raise ValueError('Identical-statistics FID check failed')
        write(folder/'verified.json',dict(passed=True,campaign_identity=identity,evaluator=evaluator_id,feature_max_abs=error,
            identical_statistics_fid=same,seconds_per_image=elapsed/len(images),images=len(images),
            classes=1000,validation_count=50000,data_seconds=data_seconds),immutable=True)
        return
    if args.mode=='real':
        folder=root/'evaluation/real';folder.mkdir(parents=True,exist_ok=True)
        dataset=Validation(shared,folder,guard)
        blocks=[]
        for i,start in enumerate(range(0,50000,128)):
            end=min(start+128,50000)
            def images(a=start,b=end):
                return np.stack([dataset.get(j) for j in range(a,b)]),dataset.labels[a:b]
            blocks.append((i,dict(dataset=token(read(folder/'dataset.json')),start=start,end=end),images))
        expected_count=50000;per_class=50
    else:
        folder=Path(args.samples);samples=read(folder/'samples.json')
        if samples['campaign_identity']!=identity or samples['role']!='final' or samples['count']!=10000:
            raise ValueError('Final generation population differs')
        blocks=[]
        for i,row in enumerate(samples['shards']):
            def images(record=row):
                path=folder/record['image_file']
                if sha(path)!=record['image_sha256']:
                    raise ValueError('Generated image shard changed')
                with np.load(path) as value:
                    return value['images'],value['labels']
            blocks.append((i,dict(shard=row['image_sha256'],samples_sha256=sha(folder/'samples.json')),images))
        expected_count=10000;per_class=10
    pool,labels,records,seconds=extract_blocks(evaluator,evaluator_id,folder/'adm',identity,blocks,guard)
    if len(pool)!=expected_count or not np.all(np.bincount(labels,minlength=1000)==per_class):
        raise ValueError('FID class balance or image count differs')
    guard.check(300)
    stats=evaluator.compute_statistics(pool)
    save_npz(folder/'statistics.npz',mu=stats.mu,sigma=stats.sigma)
    result=dict(campaign_identity=identity,evaluator=evaluator_id,count=len(pool),class_counts=np.bincount(labels,minlength=1000).tolist(),
        statistics_sha256=sha(folder/'statistics.npz'),features=records,feature_seconds=seconds)
    if args.mode=='real':
        write(folder/'ready.json',result,immutable=True)
    else:
        real=root/'evaluation/real';reference=read(real/'ready.json')
        if reference['campaign_identity']!=identity or reference['evaluator']!=evaluator_id or sha(real/'statistics.npz')!=reference['statistics_sha256']:
            raise ValueError('FID reference cache is incompatible')
        with np.load(real/'statistics.npz') as data:
            reference_stats=module.FIDStatistics(data['mu'],data['sigma'])
        fid=float(stats.frechet_distance(reference_stats))
        if not np.isfinite(fid):raise FloatingPointError('Nonfinite FID')
        result.update(FID10K=fid,generated_count=10000,reference_count=50000,reference_sha256=sha(real/'ready.json'))
        write(folder/'fid.json',result,immutable=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','assets','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--mode',choices=('verify','real','generated'),required=True)
    p.add_argument('--samples')
    args=p.parse_args()
    try:run(args)
    except BudgetStop as error:
        print(str(error),flush=True);raise SystemExit(75)


if __name__=='__main__':main()
