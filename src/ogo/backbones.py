"""Pinned EDM/SiT loading, continuous-image rewards and corrective policies."""
import importlib.util
import os
from pathlib import Path
import pickle
import subprocess
import sys

import torch
from torch import nn

from .corrector import RoPECorrector, timestep_embedding
from .features import DinoFeatures, ImagePreprocess, freeze
from .policy import install_lora
from .rewards import PairedLPIPS
from .study_io import digest, read

PREPROCESS = dict(resize_short=256, crop_size=224, mean=(.485,.456,.406), std=(.229,.224,.225),
                  interpolation='bicubic', clamp=True, antialias=True)


class CPUPreprocess(ImagePreprocess):
    def forward(self, images):
        return super().forward(images.cpu()).to(images.device)


class Decoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = freeze(vae)

    def forward(self, states):
        return self.vae.decode(states/.18215).sample


class Velocity(nn.Module):
    def __init__(self, model, backbone):
        super().__init__()
        self.model, self.backbone = model, backbone
        self.calls = 0
        self.state_evaluations = 0

    def forward(self, x, t, labels):
        self.calls += 1
        if self.backbone == 'edm64':
            self.state_evaluations += len(x)
            a, sigma = (1+80*t)/81, 80*(1-t)/(1+80*t)
            onehot = torch.nn.functional.one_hot(labels, 1000).to(x)
            denoised = self.model(x/a, x.new_full((len(x),), sigma), onehot, force_fp32=True)
            return (denoised-x)/(1-t)
        self.state_evaluations += 2*len(x)
        c = torch.cat((labels, torch.full_like(labels, 1000)))
        times = x.new_full((2*len(x),), t)
        return self.model.forward_with_cfg(torch.cat((x, x)), times, c, 1.5)[:len(x), :4]


class Policy(nn.Module):
    def __init__(self, velocity, method, settings):
        super().__init__()
        self.velocity, self.method = velocity, method
        velocity.model.requires_grad_(False).eval()
        self.correction = None
        self.lora_targets = []
        if method == 'ogo':
            edm = velocity.backbone == 'edm64'
            self.correction = RoPECorrector(input_size=64 if edm else 32, channels=3 if edm else 4,
                classes=1000, patch_size=4 if edm else 2, width=384, depth=6, heads=6, mlp_width=1344)
            self.correction.to(next(velocity.parameters())).eval()
        else:
            self.lora_targets = install_lora(velocity.model, settings['rank'], settings['lora_scope'], settings['lora_scale'])
        self.trainable = [p for p in self.parameters() if p.requires_grad]
        if not self.trainable:
            raise ValueError('Policy has no trainable parameters')

    def forward(self, x, t, labels):
        v = self.velocity(x, t, labels)
        return v if self.correction is None else v+self.correction(x, t, labels)


def import_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source(source):
    revision = subprocess.check_output(['git', '-C', source['path'], 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', source['path'], 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if revision != source['revision'] or dirty:
        raise ValueError('Pinned backbone/feature source changed')


def checked_weight(record):
    if digest(record['path']) != record['sha256']:
        raise ValueError('Weight changed: '+record['path'])
    return record['path']


def load(backbone, assets, device):
    shared = read(assets['shared_manifest'])
    for name in ('dino', 'vgg16'):
        checked_weight(shared['assets'][name])
    check_source(shared['sources']['dino'])
    if backbone == 'edm64':
        base = read(assets['edm_manifest'])
        check_source(base['sources']['edm'])
        sys.path.insert(0, base['sources']['edm']['path'])
        with open(checked_weight(base['assets']['edm']), 'rb') as f:
            model = pickle.load(f)['ema']
        if (model.img_resolution, model.img_channels, model.label_dim) != (64, 3, 1000):
            raise ValueError('EDM geometry changed')
        decoder = nn.Identity()
        weights = {'edm': base['assets']['edm']['sha256']}
    elif backbone == 'sit256':
        check_source(shared['sources']['sit'])
        module = import_source('ogo_pinned_sit', Path(shared['sources']['sit']['path'])/'models.py')
        module.TimestepEmbedder.timestep_embedding = staticmethod(timestep_embedding)
        state = torch.load(checked_weight(shared['assets']['sit']), map_location='cpu', weights_only=True)
        if 'ema' in state:
            state = state['ema']
        channels = state['final_layer.linear.weight'].shape[0]//4
        if channels not in (4, 8):
            raise ValueError('Unsupported SiT output head')
        model = module.SiT_models['SiT-XL/2'](input_size=32, num_classes=1000, learn_sigma=channels == 8)
        model.load_state_dict(state)
        for name, value in shared['vae']['files'].items():
            if digest(Path(shared['vae']['path'])/name) != value:
                raise ValueError('VAE changed')
        from diffusers import AutoencoderKL
        decoder = Decoder(AutoencoderKL.from_pretrained(shared['vae']['path'], local_files_only=True).to(device))
        weights = {'sit': shared['assets']['sit']['sha256'], 'vae': shared['vae']['files']}
    else:
        raise ValueError('Unknown backbone')
    model = freeze(model.to(device=device, dtype=torch.float32))
    encoder = torch.hub.load(shared['sources']['dino']['path'], 'dinov2_vitb14', source='local', pretrained=False)
    encoder.load_state_dict(torch.load(shared['assets']['dino']['path'], map_location='cpu', weights_only=True))
    dino = DinoFeatures(decoder, CPUPreprocess(**PREPROCESS), freeze(encoder.to(device)))
    import lpips
    if digest(Path(lpips.__file__).parent/'weights/v0.1/vgg.pth') != shared['lpips']['sha256']:
        raise ValueError('LPIPS calibration changed')
    cache = Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        os.link(shared['assets']['vgg16']['path'], cache)
    if digest(cache) != shared['assets']['vgg16']['sha256']:
        raise ValueError('VGG cache changed')
    reward = PairedLPIPS(decoder, freeze(lpips.LPIPS(net='vgg', version='0.1', verbose=False).to(device)),
                         image_size=64 if backbone == 'edm64' else 256)
    provenance = {'weights': weights, 'dino': shared['assets']['dino']['sha256'], 'vgg': shared['assets']['vgg16']['sha256'],
                  'lpips': shared['lpips']['sha256'], 'precision': 'FP32', 'preprocess': PREPROCESS,
                  'cfg': None if backbone == 'edm64' else {'scale': 1.5, 'channels': 3}}
    return Velocity(model, backbone), dino, reward, provenance
