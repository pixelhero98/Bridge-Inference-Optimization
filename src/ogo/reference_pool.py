"""Read-only reuse of pinned paired endpoints; matching draws remain disjoint."""
from pathlib import Path
import torch
from .study_io import digest, identity, read, write


def edm_seed(role, index):
    offsets = {'student': 1_000_000, 'matching': 2_000_000, 'calibration': 3_000_000,
               'development': 4_000_000, 'profile': 5_000_000, 'verify': 6_000_000}
    return (0x9E3779B1*(offsets[role]+int(index))+0xA5A5A5A5) & 0xffffffff


class ReferencePool:
    def __init__(self, backbone, paths, classes, output):
        self.backbone, self.classes = backbone, classes
        self.path = Path(paths['edm_cache' if backbone == 'edm64' else 'sit_pool'])
        self.checked, self.loaded = set(), {}
        self.records = {c: {'paired': [], 'matching': [], 'development': []} for c in classes}
        manifest_files = {}
        if backbone == 'edm64':
            for path in sorted(self.path.glob('*.json')):
                meta = read(path)
                if 'descriptor' not in meta:
                    continue
                d = meta['descriptor']
                if d['role'] not in ('student', 'matching', 'development'):
                    continue
                role = {'student': 'paired', 'matching': 'matching', 'development': 'development'}[d['role']]
                teacher = dict(num_steps=256,sigma_min=.002,sigma_max=80.,rho=7.,S_churn=40.,S_min=.05,S_max=50.,S_noise=1.003)
                if (d['cache']['model'] != paths['edm_weight_sha256'] or d['cache']['teacher'] != teacher or
                        d['cache']['sampler'] != paths['edm_sampler_sha256'] or
                        d['cache']['noise'] != 'bijective32-cpu-initial-fp32-aux-fp64'):
                    raise ValueError('EDM reference provenance changed')
                for j, (index, label) in enumerate(zip(d['indices'], d['labels'])):
                    if label in self.records:
                        self.records[label][role].append({'file': str(path.with_suffix('.pt')), 'row': j,
                           'seed': edm_seed(d['role'], index), 'sha256': meta['sha256'],
                           'id': f"edm/{d['role']}/{index}", 'order': index})
                manifest_files[path.name] = digest(path)
        else:
            for label in classes:
                path = self.path/f'class-{label:04d}.json'
                meta = read(path)
                if meta['class_id'] != label or meta['identity'] != paths['sit_pool_identity']:
                    raise ValueError('SiT reference identity mismatch')
                for j, seed in enumerate(meta['seeds']):
                    # Disposable partitions; never include these in later training/development.
                    role = 'paired' if j < 64 else ('matching' if j < 128 else 'development')
                    self.records[label][role].append({'file': str(path.with_suffix('.pt')), 'row': j,
                        'seed': seed, 'sha256': meta['sha256'], 'id': f'sit/teacher-training/{label}/{j}', 'order': j})
                manifest_files[path.name] = digest(path)
        for c in classes:
            for role in self.records[c]:
                self.records[c][role].sort(key=lambda r: r['order'])
                if len(self.records[c][role]) < 4:
                    raise ValueError(f'Insufficient fixed references for class {c}, {role}')
            ids = [r['id'] for rows in self.records[c].values() for r in rows]
            if len(ids) != len(set(ids)):
                raise ValueError('Reference partitions overlap')
        self.identity = identity({'backbone': backbone, 'classes': classes, 'metadata': manifest_files})
        write(Path(output)/f'{backbone}-reuse.json', {'identity': self.identity,
              'metadata': manifest_files, 'roles': {str(c): {r: len(v) for r, v in roles.items()} for c, roles in self.records.items()},
              'purpose': 'disposable engineering profile; disjoint matching and independent development'}, immutable=True)

    def batch(self, cursor, batch_size, group_size, role='paired', device='cuda'):
        if batch_size % group_size:
            raise ValueError('Whole conditional groups required')
        groups = batch_size//group_size
        rows, labels = [], []
        for g in range(groups):
            c = self.classes[(cursor*groups+g) % len(self.classes)]
            records = self.records[c][role]
            for j in range(group_size):
                rows.append(records[(cursor*group_size+j) % len(records)])
                labels.append(c)
        tensors = []
        for row in rows:
            path = Path(row['file'])
            if row['file'] not in self.checked:
                if digest(path) != row['sha256']:
                    raise ValueError('Reference tensor changed')
                self.checked.add(row['file'])
            if row['file'] not in self.loaded:
                self.loaded[row['file']] = torch.load(path, map_location='cpu', weights_only=True)
            value = self.loaded[row['file']]
            tensors.append(value[row['row']] if self.backbone == 'edm64' else value['latent'][row['row']].reshape(4,32,32))
        shape = (3,64,64) if self.backbone == 'edm64' else (4,32,32)
        initial = torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(row['seed'])) for row in rows])
        if self.backbone == 'edm64':
            initial *= 80/81
        return {'initial': initial.to(device), 'reference': torch.stack(tensors).to(device),
                'labels': torch.tensor(labels, device=device), 'ids': [r['id'] for r in rows]}
