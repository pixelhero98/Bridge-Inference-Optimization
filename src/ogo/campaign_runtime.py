"""Campaign identities, atomic records, allocation guards and process locks."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import sys

METHODS = ('draft_k','ddpo_is','flow_grpo','lambda_grpo','offline_d3po')
STOP = False


class BudgetStop(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def token(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, immutable=False):
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():
        # Metadata must survive a fresh-process JSON round trip (tuple -> list).
        if read(path)!=json.loads(payload):
            raise ValueError('Immutable campaign record changed: '+str(path))
        return
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    with os.fdopen(fd,'w',encoding='utf-8') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(name,path)
    if os.name=='posix':
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


def validate(config):
    if (config['schema']!='ogo-campaign-v1' or config['backbone']!='sit256' or config['seed']!=0 or
        config['class_ids']!=list(range(1000)) or sorted(config['class_order'])!=list(range(1000)) or
        tuple(config['methods'])!=METHODS or config['steps']!=[16,32] or
        config['optimizer_updates']!=512 or config['final_update']!=512 or
        config['development_updates']!=[128,256,384] or config['checkpoint_every']!=32 or
        (config['batch_size'],config['microbatch'],config['group_size'])!=(32,4,4) or
        config['precision']!='FP32' or config['evaluation']!='deterministic_uniform_euler' or
        config['reference_attention']!='pinned_teacher_math_sdp' or
        config['max_aggregate_gpu_seconds']!=396000 or config['max_campaign_gpus']!=2 or
        config['storage_bytes_cap']!=64000000000):
        raise ValueError('Campaign differs from the agreed production protocol')
    expected=dict(steps=256,cfg=1.5,cfg_channels=3,latent_scale=.18215,
                  training_per_class=8,development_per_class=2,final_per_class=10)
    if config['reference']!=expected or config['preference_pairs_per_class']!=8 or config['preference_shard_pairs']!=16:
        raise ValueError('Reference or preference coverage changed')
    for method in METHODS:
        expected_passes=2 if method in ('ddpo_is','flow_grpo','lambda_grpo') else 1
        if method!='draft_k' and config['methods'][method]['replay_passes']!=expected_passes:
            raise ValueError('Native replay-pass count changed')


def initialize(config_path, assets_path, output):
    config=read(config_path);validate(config)
    assets=read(assets_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    contract=read(Path(__file__).parents[2]/'configs/ogo-profile.json')
    if sha(Path(__file__).parents[2]/'configs/ogo-profile.json')!=config['baseline_contract_sha256']:
        raise ValueError('Verified baseline contract bytes changed')
    if config['methods']!={k:contract['methods'][k] for k in METHODS} or config['thresholds']!=contract['thresholds']:
        raise ValueError('Baseline hyperparameters or thresholds changed')
    manifests={k:sha(assets[k]) for k in ('shared_manifest','reuse_audit','protection_receipt')}
    manifests['evaluation_overlay']=sha(Path(assets['evaluation_overlay'])/'ready.json')
    runtime=dict(python=sys.version,packages={name:importlib.metadata.version(name) for name in
                 ('torch','numpy','torchvision','diffusers','lpips','timm')})
    value=dict(config=config,source=source,runtime=runtime,assets_sha256=sha(assets_path),input_manifests=manifests)
    identity=token(value);write(output/'identity.json',dict(identity=identity,**value),immutable=True)
    return config,assets,identity


def protected(assets):
    root=Path(assets['protected_root'])
    values=read(assets['protection_receipt'])['protected_hashes']
    for name,expected in values.items():
        if sha(root/name)!=expected:
            raise ValueError('Protected input or source dependency changed: '+name)
    return values


def allocated():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Campaign numerical work requires a scheduler allocation')


def storage_bytes(root):
    seen=set();total=0
    for p in Path(root).rglob('*'):
        try:
            if p.is_symlink():
                raise ValueError('Campaign output must not contain symbolic links')
            if not p.is_file():
                continue
            s=p.stat()
        except FileNotFoundError:
            continue  # Another worker may atomically replace its own temporary file.
        key=(s.st_dev,s.st_ino)
        if key not in seen:
            seen.add(key);total+=s.st_size
    return total


class Guard:
    def __init__(self, output, config, deadline):
        if not isinstance(deadline,(int,float)) or not time.time()<deadline<float('inf'):
            raise ValueError('A finite future allocation deadline is required')
        self.output,self.config,self.deadline=Path(output),config,deadline
        self.last_storage=0.
        def stop(_signum,_frame):
            global STOP
            STOP=True
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,stop)
        if hasattr(signal,'SIGUSR1'):
            signal.signal(signal.SIGUSR1,stop)

    def check(self, seconds=180, storage=False):
        if STOP or time.time()+seconds>=self.deadline:
            raise BudgetStop('Checkpoint before the allocation deadline')
        if storage or time.monotonic()-self.last_storage>120:
            size=storage_bytes(self.output)
            if size+256_000_000>self.config['storage_bytes_cap']:
                write(self.output/'storage-stop.json',dict(bytes=size,cap=self.config['storage_bytes_cap']))
                raise BudgetStop('Campaign storage ceiling reached')
            self.last_storage=time.monotonic()


@contextmanager
def lock(path, blocking=False):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f,fcntl.LOCK_UN)


def configure_torch(seed=0):
    import torch
    from .checkpoint import seed_all
    seed_all(seed);torch.set_default_dtype(torch.float32);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def key(method, steps):
    return method+'-n'+str(steps)
