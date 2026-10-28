"""Scientific population, resumable state and incompatible-cache invariants."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ogo.campaign_data import Pool,build_index,pool_seed,reference_attention
from ogo.campaign_engine import Trainer,equal_state,warm,checkpoint
from ogo.campaign_eval import summarize,save_npz,committed_snapshot
from ogo.campaign import command_tag
from ogo.campaign_runtime import validate,write,read,sha,lock,Guard,BudgetStop

CONFIG=Path(__file__).parents[1]/'configs/imagenet256-baselines-512.json'


class Populations(unittest.TestCase):
    def test_reference_backend_restores_policy_backend(self):
        b=torch.backends.cuda
        previous=(b.flash_sdp_enabled(),b.mem_efficient_sdp_enabled(),b.math_sdp_enabled())
        with self.assertRaises(InterruptedError):
            with reference_attention():
                self.assertFalse(b.flash_sdp_enabled());self.assertFalse(b.mem_efficient_sdp_enabled())
                self.assertTrue(b.math_sdp_enabled());raise InterruptedError()
        self.assertEqual(previous,(b.flash_sdp_enabled(),b.mem_efficient_sdp_enabled(),b.math_sdp_enabled()))

    def test_roles_all_classes_traversal_and_exclusions(self):
        config=read(CONFIG)
        config['reference_reconstruction_probe_ids']=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'legacy').mkdir()
            write(root/'audit.json',{'backbones':{'sit256':{'reserved_profile_ids':['sit/teacher-training/0/0','sit/teacher-training/0/128']}}})
            write(root/'legacy/class-0000.json',dict(identity='fixture',class_id=0,sha256='fixture',
                seeds=[pool_seed(0,i) for i in range(256)]))
            assets=dict(reuse_audit=str(root/'audit.json'),sit_pool=str(root/'legacy'),sit_pool_identity='fixture')
            index=build_index(config,assets,root,'fixture');pool=Pool(config,root,require_ready=False)
            self.assertEqual(index['missing'],19980)
            ids={r['id'] for r in index['records']};self.assertEqual(len(ids),20000)
            self.assertNotIn('sit/teacher-training/0/0',ids);self.assertNotIn('sit/teacher-training/0/128',ids)
            for role,n in [('training',8),('development',2),('final',10)]:
                self.assertTrue(all(len(pool.records[c][role])==n for c in range(1000)))
            rows=[r for cursor in range(500) for r in pool.rows(cursor,pairs=True)]
            self.assertEqual(len({r['id'] for r in rows}),8000)
            self.assertEqual(np.bincount([r['class_id'] for r in rows]).tolist(),[16]*1000)
            self.assertTrue(all(rows[i]['id']==rows[i+1]['id'] for i in range(0,16000,2)))
            self.assertEqual([pool.rows(i)[0]['class_id'] for i in range(5)],config['class_order'][0:40:8])
            wrong=copy.deepcopy(config);wrong['optimizer_updates']=256
            with self.assertRaises(ValueError):validate(wrong)
            write(root/'references/ready.json',dict(index_sha256='wrong'))
            with self.assertRaises(ValueError):Pool(config,root)


class OptimizerState(unittest.TestCase):
    def test_interrupted_two_pass_adam_and_warmup_are_exact(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.1,-.2]));self.trainable=[self.weight]
        class Fixture(Trainer):
            def step(self,cursor,interrupt=False):
                for i in range(2):
                    self.optimizer.zero_grad();loss=(self.policy.weight-torch.randn_like(self.policy.weight)).square().sum()
                    loss.backward();self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
                    if interrupt and i==0:raise InterruptedError()
                return dict(cursor=cursor,loss=float(loss.detach()))
        with tempfile.TemporaryDirectory() as tmp:
            p=Policy();t=Fixture(p,None,None,SimpleNamespace(identity='fixture'),read(CONFIG),'ddpo_is',16,tmp,'fixture')
            t.step(0);saved=t.state(1);t.step(1);expected=t.state(2)
            t.restore(saved)
            with self.assertRaises(InterruptedError):t.step(1,interrupt=True)
            t.restore(saved);t.step(1);self.assertTrue(equal_state(t.state(2),expected))
            t.restore(saved);warm(t,saved,1,3);self.assertTrue(equal_state(saved,t.state(1)))
            cp=checkpoint(t,tmp,1);again=checkpoint(t,tmp,1)
            self.assertEqual(cp,again);self.assertEqual(cp['optimizer_updates'],2)
            bad=copy.deepcopy(saved);bad['identity']='changed'
            with self.assertRaises(ValueError):t.restore(bad)


class Evaluation(unittest.TestCase):
    def test_unpublished_checkpoint_is_not_visible_to_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir();p=root/'checkpoints/update-0128.pt';p.write_bytes(b'fixture')
            record=dict(identity='fixture',optimizer_updates=128,cursor=64,file=p.name,sha256=sha(p))
            write(root/'identity.json',dict(identity='fixture'));write(p.with_suffix('.json'),record)
            write(root/'latest.json',dict(identity='fixture',optimizer_updates=96))
            with self.assertRaises(RuntimeError):committed_snapshot(root,128)
            write(root/'latest.json',record)
            self.assertEqual(committed_snapshot(root,128)[2],sha(p))
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):committed_snapshot(root,128)

    def test_irrelevant_flags_cannot_bypass_duplicate_lock(self):
        a=SimpleNamespace(command='collect',steps=16,part='references',partition=0)
        expected=command_tag(a);a.part='index';a.partition=1
        self.assertEqual(command_tag(a),expected)

    def test_exact_thousand_class_counts_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);labels=np.repeat(np.arange(1000),2)
            save_npz(root/'metrics.npz',labels=labels,lpips=np.full(2000,.2),top1=np.ones(2000),
                top5=np.ones(2000),dino=np.tile([[1.,0.],[0.,1.]],(1000,1)))
            manifest=dict(shards=[dict(metrics_file='metrics.npz',metrics_sha256=sha(root/'metrics.npz'))])
            write(root/'samples.json',manifest)
            value=summarize(root,'development','fixture')
            self.assertEqual(value['count'],2000);self.assertEqual(value['dino_diversity'],2.)
            self.assertEqual(len(value['per_class']),1000)
            save_npz(root/'metrics.npz',labels=np.zeros(2000))
            with self.assertRaises(ValueError):summarize(root,'development','fixture')

    def test_process_lock_and_expired_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            with lock(Path(tmp)/'.lock') as a:
                with lock(Path(tmp)/'.lock') as b:self.assertTrue(a);self.assertFalse(b)
            with self.assertRaises(ValueError):Guard(tmp,read(CONFIG),0)


if __name__=='__main__':unittest.main()
