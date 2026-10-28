"""Disposable high-step reference preparation timings; never overwrite caches."""
from pathlib import Path
import time
import torch
from .backbones import import_source
from .policy import euler
from .reference_pool import edm_seed
from .study_io import read, write


def profile_references(velocity, pool, assets, output):
    B = 32
    data = pool.batch(12345, B, 4)
    if pool.backbone == 'edm64':
        base = read(assets['edm_manifest'])
        native = import_source('ogo_reference_edm_sampler', Path(base['sources']['edm']['path'])/'generate.py')
        z = data['initial']*(81/80)
        labels = torch.nn.functional.one_hot(data['labels'],1000).float()
        def generate():
            generators = [torch.Generator().manual_seed(910000000+i) for i in range(B)]
            def extra(x):
                return torch.stack([torch.randn(x.shape[1:],generator=g,dtype=x.dtype) for g in generators]).to(x.device)
            # Preserve pinned native FP16/internal and FP64 state semantics.
            deterministic = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(False)
                with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=False):
                    return native.edm_sampler(velocity.model,z,labels,randn_like=extra,num_steps=256,
                        sigma_min=.002,sigma_max=80.,rho=7.,S_churn=40.,S_min=.05,S_max=50.,S_noise=1.003).float()
            finally:
                torch.use_deterministic_algorithms(deterministic)
        nfe = 511
    else:
        def generate():
            return euler(velocity,data['initial'],data['labels'],256,backward_steps=0)
        nfe = 256
    rows = []
    with torch.no_grad():
        previous = None
        for i in range(3):
            torch.cuda.synchronize(); start=time.perf_counter()
            y=generate()
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            if not torch.isfinite(y).all():
                raise FloatingPointError('Nonfinite high-step reference')
            difference = None if previous is None else float((y-previous).abs().max())
            if difference is not None and difference != 0:
                raise ValueError('Fixed reference sampler does not replay exactly')
            previous=y.clone()
            rows.append({'warmup':i==0,'batch_seconds':elapsed,'batch_size':B,'nfe':nfe,
                         'network_state_evaluations':B*nfe*(1 if pool.backbone=='edm64' else 2),
                         'repeat_max_error':difference})
    result={'backbone':pool.backbone,'measurements':rows,'purpose':'reference preparation cost; disposable new auxiliary noise, no cache writes',
            'seconds_per_image':sum(r['batch_seconds'] for r in rows[1:])/(2*B)}
    write(Path(output)/f'{pool.backbone}-reference-profile.json',result,immutable=True)
    return result
