"""A cold-process warmup must preserve Adam state, parameters, RNG and cursor."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from ogo.study import warm_resume
from ogo.study_verify import equal_tree


class ResumeWarmupTests(unittest.TestCase):
    def test_probe_updates_leave_committed_state_exactly_unchanged(self):
        class OptimizerFixture:
            def __init__(self):
                self.weight=torch.nn.Parameter(torch.tensor([.3,-.2],dtype=torch.float64))
                self.optimizer=torch.optim.AdamW([self.weight],lr=.01)
                self.calls=[]

            def step(self,cursor):
                self.calls.append(cursor)
                self.optimizer.zero_grad()
                loss=((self.weight-torch.randn_like(self.weight))**2).sum()
                loss.backward();self.optimizer.step()
                return {'loss':float(loss.detach()),'cursor':cursor}

            def resume(self,path):
                with torch.no_grad():self.weight.copy_(self.saved_weight)
                self.optimizer.load_state_dict(copy.deepcopy(self.saved_optimizer))
                torch.set_rng_state(self.saved_rng)
                return 4

        torch.manual_seed(18)
        fixture=OptimizerFixture();fixture.step(3)
        fixture.saved_weight=fixture.weight.detach().clone()
        fixture.saved_optimizer=copy.deepcopy(fixture.optimizer.state_dict())
        fixture.saved_rng=torch.get_rng_state().clone()
        fixture.calls=[]
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ,{'SLURM_JOB_ID':'fixture'}):
            folder=Path(temporary)
            warm_resume(fixture,folder/'checkpoint.pt',4,folder,3)
            receipt=json.loads((folder/'resume-warmup-fixture.json').read_text())
        self.assertEqual(fixture.calls,[3,3,3])
        self.assertTrue(torch.equal(fixture.weight,fixture.saved_weight))
        self.assertTrue(equal_tree(fixture.optimizer.state_dict(),fixture.saved_optimizer))
        self.assertTrue(torch.equal(torch.get_rng_state(),fixture.saved_rng))
        self.assertEqual(receipt['committed_updates'],0)
        self.assertEqual(receipt['committed_cursor'],4)


if __name__=='__main__':unittest.main()
