"""Behavioral checks for packaging-era regressions and the staged training contract."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from fct.checkpoint import load_checkpoint, rng_state
from fct.objectives import FeatureKernel, GaussianKernel, conditional_loss, kernel_score
from fct.rollout import frozen_rollout, uniform_euler
from fct.synthetic import SyntheticTask
from fct.training import Stage, Trainer, TrainingConfig


def config(**changes):
    values = dict(regime="reduced_step", steps=2, teacher_steps=8,
                  stages=(Stage(3, 0.), Stage(3, 1., 2)), adapter="fct.synthetic:make_task",
                  adapter_config={"particles": 4, "targets": 2, "paired_baseline": True},
                  lr=1e-3, dtype="float64", device="cpu", seed=17)
    values.update(changes)
    return TrainingConfig(**values)


class CanonicalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(9)

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_current_state_velocity_and_uniform_time(self):
        seen = []
        def correction(x, t, c):
            seen.append(t)
            return 0.2 * x
        x = torch.tensor([[1.]])
        result = uniform_euler(lambda x, t, c: 0.4 * x, correction, x, None, 4)
        torch.testing.assert_close(result, x * 1.15**4, atol=1e-14, rtol=1e-14)
        self.assertEqual(seen, [0., 0.25, 0.5, 0.75])

    def test_multiple_targets_against_independent_pair_loops(self):
        kernel = GaussianKernel(0.7)
        for k, m in [(2, 1), (4, 3), (8, 2)]:
            x = torch.randn(k, 2, requires_grad=True)
            y = torch.randn(m, 2, requires_grad=True)
            result = kernel_score(x, y, kernel)
            expected = sum(torch.exp(-((x[i] - x[j])**2).sum() / (2 * 0.7**2))
                           for i in range(k) for j in range(k) if i != j) / (2*k*(k-1))
            expected -= sum(torch.exp(-((x[i] - y[j].detach())**2).sum() / (2 * 0.7**2))
                            for i in range(k) for j in range(m)) / (k*m)
            torch.testing.assert_close(result, expected, atol=1e-13, rtol=1e-13)
            gx, gy = torch.autograd.grad(result, (x, y), allow_unused=True)
            self.assertIsNone(gy)
            torch.testing.assert_close(gx, torch.autograd.grad(expected, x)[0], atol=1e-13, rtol=1e-13)

    def test_latent_and_decoded_perceptual_branches_keep_input_gradients(self):
        # Small decoder/encoder stand-ins verify the DINO adapter contract without weights/downloads.
        decoder, encoder = nn.Linear(2, 3), nn.Linear(3, 4)
        for module in (decoder, encoder):
            module.eval().requires_grad_(False)
        latent = FeatureKernel("latent", lambda z, c: z, GaussianKernel(1.), 0.3)
        perceptual = FeatureKernel("perceptual", lambda z, c: encoder(torch.tanh(decoder(z))), GaussianKernel(0.8), 0.7)
        x, y = torch.randn(4, 2, requires_grad=True), torch.randn(3, 2)
        def loss(z, branches):
            return conditional_loss(z, y, None, None, branches, utility_weight=0.)[0]
        combined = torch.autograd.grad(loss(x, (latent, perceptual)), x)[0]
        latent_grad = torch.autograd.grad(loss(x, (latent,)), x)[0]
        perceptual_grad = torch.autograd.grad(loss(x, (perceptual,)), x)[0]
        torch.testing.assert_close(combined, latent_grad + perceptual_grad, atol=1e-13, rtol=1e-13)
        self.assertGreater(float(perceptual_grad.norm()), 1e-8)
        direction = torch.randn_like(x)
        direction /= direction.norm()
        epsilon = 1e-5
        fd = (loss(x.detach()+epsilon*direction, (latent, perceptual)) -
              loss(x.detach()-epsilon*direction, (latent, perceptual))) / (2*epsilon)
        torch.testing.assert_close(fd, (combined*direction).sum(), atol=1e-9, rtol=1e-6)
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in (decoder, encoder) for p in m.parameters()))

    def test_sampling_streams_and_paired_baseline_roles(self):
        task = SyntheticTask({"particles": 4, "targets": 4, "paired_baseline": True},
                             seed=1, dtype=torch.float64, device="cpu", steps=2, teacher_steps=8)
        first, repeated, following = task.batch(0), task.batch(0), task.batch(1)
        for a, b, c in zip(first, repeated, following):
            a.validate()
            self.assertTrue(torch.equal(a.initial, b.initial))
            self.assertTrue(torch.equal(a.matching_targets, b.matching_targets))
            self.assertFalse(torch.equal(a.initial, c.initial))
            self.assertFalse(set(a.student_ids) & set(a.target_ids))
            expected = frozen_rollout(task.dynamics, a.initial, a.context, 8)
            self.assertTrue(torch.equal(a.utility_baseline, expected))
            self.assertFalse(torch.equal(a.matching_targets, a.utility_baseline))
            with self.assertRaisesRegex(ValueError, "independent"):
                replace(a, target_ids=a.student_ids).validate()
            with self.assertRaisesRegex(ValueError, "baseline identities"):
                replace(a, baseline_ids=a.target_ids).validate()

    def test_checkpoint_and_offload_gradients_cpu_and_gpu(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            base = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)).to(device)
            base.eval().requires_grad_(False)
            initial = torch.randn(4, 2, device=device)
            parameter = torch.tensor(0.2, device=device, requires_grad=True)
            def gradient(checkpoint, offload):
                endpoint = uniform_euler(lambda x, t, c: base(x), lambda x, t, c: parameter*x,
                                         initial, None, 4, checkpoint_dynamics=checkpoint, cpu_offload=offload)
                return torch.autograd.grad(endpoint.square().mean(), parameter)[0]
            reference = gradient(False, False)
            for flags in [(True, False), (False, True), (True, True)]:
                torch.testing.assert_close(gradient(*flags), reference, atol=1e-10, rtol=1e-8)
            self.assertTrue(all(p.grad is None for p in base.parameters()))

    def test_resume_before_at_and_after_stage_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = Trainer(config(), root / "full")
            full.run()
            expected, expected_rng = full.payload(), rng_state()
            for stop in (2, 3, 4):
                part = Trainer(config(), root / str(stop))
                part.run(stop)
                resumed = Trainer(config(), root / str(stop), resume=True)
                resumed.run()
                self.assert_tree_equal(expected, resumed.payload())
                self.assert_tree_equal(expected_rng, rng_state())
            self.assertEqual([r["utility_weight"] for r in full.history], [0., 0., 0., 0.5, 1., 1.])
            self.assertTrue(all("utility" not in row for row in full.history[:3]))
            self.assertTrue(all(row["matching_weight"] == 1. for row in full.history))

    def test_same_step_fp32_smoke(self):
        cfg = config(regime="same_step", steps=2, teacher_steps=2, stages=(Stage(2, 1.),), dtype="float32")
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(cfg, Path(directory) / "fp32")
            self.assertEqual(trainer.run()["status"], "COMPLETE")
            self.assertTrue(all(p.dtype == torch.float32 for p in trainer.model.parameters()))

    def test_stage_and_config_validation(self):
        for changes in [{"steps": 8}, {"regime": "standalone"}, {"matching_weight": 0.},
                        {"stages": (Stage(3, 1.), Stage(3, 1.))}, {"dtype": "float16"}]:
            with self.assertRaises(ValueError):
                config(**changes)
        with self.assertRaises(TypeError):
            TrainingConfig.from_dict({"unknown": True, "stages": []})

    def test_reject_legacy_and_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.save({"model": {}}, path / "legacy.pt")
            with self.assertRaisesRegex(ValueError, "Incompatible operator"):
                load_checkpoint(path / "legacy.pt", {})
            trainer = Trainer(config(), path / "run")
            trainer.run(1)
            with self.assertRaisesRegex(ValueError, "Changed configuration"):
                Trainer(config(lr=0.01), path / "run", resume=True)
            with self.assertRaises(FileExistsError):
                Trainer(config(), path / "run")

    def test_failure_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(config(), Path(directory) / "run")
            trainer.adapter.utility = lambda x, c, b: x[:, 0] * float("nan")
            trainer.run(3)
            with self.assertRaises(FloatingPointError):
                trainer.step()
            self.assertEqual(len(list(trainer.output.glob("failure-*.txt"))), 1)
            with self.assertRaisesRegex(RuntimeError, "Failed trainer"):
                trainer.step()
            self.assertTrue((trainer.output / "latest.pt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
