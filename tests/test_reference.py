"""Structural tests complementing, rather than duplicating, the FD campaign."""

import unittest

import numpy as np
import torch

from fct.verification_cases import (CONTROLS, autograd_gradient, backbone, context_components,
                      discrete_adjoint, frozen_rollout, loop_objective, objective,
                      rollout, terminal_covector, terminal_objective)
from fct.verify import close, stable_window


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        generator = torch.Generator().manual_seed(123)
        self.theta = torch.randn(58, generator=generator) * 0.15
        self.z = torch.randn(2, 4, 2, generator=generator)
        self.target = torch.tensor([[0.3, -0.2], [1.1, 0.6]])
        self.context = torch.tensor([-1.0, 1.0])

    def test_analytic_backbone_jacobian_keeps_frozen_input_path(self):
        x = self.z.clone().requires_grad_(True)
        v = backbone(x, 0.25, self.context)
        for output in range(2):
            grad, = torch.autograd.grad(v[..., output].sum(), x, retain_graph=True)
            expected = torch.tensor([[-0.3, -0.5], [0.4, -0.2]])[output].expand_as(x).clone()
            expected[..., output] += 0.2 * (1.0 - torch.tanh(x[..., output]).square())
            torch.testing.assert_close(grad, expected, atol=1e-14, rtol=1e-14)
        self.assertFalse(self.target.requires_grad)
        self.assertFalse(self.context.requires_grad)

    def test_zero_correction_is_frozen_euler_at_every_budget(self):
        theta = self.theta.clone()
        theta[40:] = 0
        for n in (1, 2, 4, 8, 16):
            torch.testing.assert_close(rollout(theta, self.z, self.context, n),
                                       frozen_rollout(self.z, self.context, n), atol=1e-14, rtol=1e-14)

    def test_context_groups_are_separate_and_then_averaged(self):
        loss = terminal_objective(self.z, self.target, self.context, "combined")
        separate = sum(terminal_objective(self.z[i:i + 1], self.target[i:i + 1],
                                          self.context[i:i + 1], "combined") for i in range(2)) / 2
        torch.testing.assert_close(loss, separate, atol=1e-14, rtol=1e-14)
        changed_target = self.target.clone()
        changed_target[0] += 0.7
        _, before = context_components(self.z, self.target, self.context)
        _, after = context_components(self.z, changed_target, self.context)
        self.assertEqual(float(before[1]), float(after[1]))
        self.assertNotEqual(float(before[0]), float(after[0]))

    def test_k_one_requires_utility_only(self):
        z = self.z[:, :1]
        for regime in ("matching", "combined"):
            with self.assertRaisesRegex(ValueError, "K >= 2"):
                objective(self.theta, z, self.target, self.context, 2, regime)
            with self.assertRaisesRegex(ValueError, "K >= 2"):
                terminal_covector(z, self.target, self.context, regime)
        self.assertTrue(torch.isfinite(objective(self.theta, z, self.target, self.context, 2, "utility")))

    def test_ordered_pairs_and_covectors_against_independent_loop(self):
        for k in (2, 4):
            x = self.z[:, :k].clone().requires_grad_(True)
            for regime in ("utility", "matching", "combined"):
                loss = terminal_objective(x, self.target, self.context, regime)
                self.assertAlmostEqual(float(loss.detach()), loop_objective(x, self.target, self.context, regime), places=13)
                gradient, = torch.autograd.grad(loss, x)
                expected = terminal_covector(x.detach(), self.target, self.context, regime)
                torch.testing.assert_close(gradient, expected, atol=1e-13, rtol=1e-13)

    def test_gaussian_is_smooth_at_coincident_particles(self):
        x = torch.zeros_like(self.z).requires_grad_(True)
        target = torch.zeros_like(self.target)
        loss = terminal_objective(x, target, self.context, "matching")
        grad, = torch.autograd.grad(loss, x)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(grad, torch.zeros_like(x))
        self.assertLess(float(loss.detach()), 0)  # Target-only constant is intentionally omitted.

    def test_additivity_and_explicit_adjoint(self):
        grads = {}
        for regime in ("utility", "matching", "combined"):
            _, grads[regime] = autograd_gradient(self.theta, self.z, self.target, self.context, 4, regime)
            adjoint = discrete_adjoint(self.theta, self.z, self.target, self.context, 4, regime)
            torch.testing.assert_close(grads[regime], adjoint, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(grads["combined"], grads["utility"] + grads["matching"], atol=1e-12, rtol=1e-12)

    def test_controls_change_gradients_but_preserve_forward_values(self):
        loss, full = autograd_gradient(self.theta, self.z, self.target, self.context, 8, "combined")
        for control in CONTROLS:
            changed_loss, changed_grad = autograd_gradient(self.theta, self.z, self.target, self.context, 8, "combined", control)
            self.assertTrue(torch.equal(loss, changed_loss))
            self.assertGreater(float(torch.max(torch.abs(full - changed_grad))), 1e-7)
        for control in ("no_backbone_input", "last_step_only"):
            _, full1 = autograd_gradient(self.theta, self.z, self.target, self.context, 1, "combined")
            _, control1 = autograd_gradient(self.theta, self.z, self.target, self.context, 1, "combined", control)
            torch.testing.assert_close(full1, control1, atol=1e-14, rtol=1e-14)

    def test_both_kernel_arguments_supply_equal_self_interaction_derivatives(self):
        _, full = autograd_gradient(self.theta, self.z, self.target, self.context, 4, "matching")
        _, one_side = autograd_gradient(self.theta, self.z, self.target, self.context, 4, "matching", "one_sided_kernel")
        self.assertGreater(float(torch.linalg.vector_norm(full - one_side)), 1e-8)
        # The one-sided result should NOT be exactly half the full matching gradient:
        # the attraction derivative remains complete.
        self.assertGreater(float(torch.linalg.vector_norm(one_side - 0.5 * full)), 1e-8)

    def test_three_consecutive_epsilons_and_nonfinite_rejection(self):
        self.assertGreater(stable_window([0.1, 2.0, 0.1, 2.0, 0.1], 3), 1)
        self.assertLessEqual(stable_window([2.0, 0.1, 0.2, 0.3, 2.0], 3), 1)
        self.assertFalse(close(np.nan, np.nan, 1e-9, 1e-6))

    def test_invalid_step_budgets_fail(self):
        for n in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                rollout(self.theta, self.z, self.context, n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
