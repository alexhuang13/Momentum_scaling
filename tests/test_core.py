import unittest

import torch

from momentum_scaling.core import build_problem, excess_loss, lookahead, update


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.problem = build_problem(beta=2.0, s=1.0, sigma=0.1, dim=8,
                                     device=torch.device("cpu"), dtype=torch.float64)

    def test_power_law_problem(self):
        self.assertAlmostEqual(float(self.problem.lam[0]), 1.0)
        self.assertAlmostEqual(float(self.problem.lam[1]), 0.25)
        self.assertGreater(float(excess_loss(-self.problem.theta_star, self.problem.lam)), 0.0)

    def test_nesterov_update_uses_lookahead_step(self):
        current = torch.tensor([2.0], dtype=torch.float64)
        previous = torch.tensor([1.0], dtype=torch.float64)
        rho, eta = 0.5, 0.1
        point = lookahead(current, previous, "nesterov", rho)
        gradient = 3.0 * point
        actual = update(current, previous, gradient, algorithm="nesterov", eta=eta, rho=rho)
        expected = point - eta * gradient
        torch.testing.assert_close(actual, expected)

    def test_sgd_ignores_momentum(self):
        current = torch.tensor([2.0])
        previous = torch.tensor([-9.0])
        gradient = torch.tensor([3.0])
        actual = update(current, previous, gradient, algorithm="sgd", eta=0.1, rho=0.99)
        torch.testing.assert_close(actual, torch.tensor([1.7]))


class ScalingScheduleTests(unittest.TestCase):
    def test_focused_experiment_modules_import(self):
        from momentum_scaling.scaling_experiments import (
            D_PAPER, nesterov_scaling_experiment, polyak_scaling_experiment,
        )

        self.assertEqual(D_PAPER[0], 2000)
        self.assertTrue(callable(polyak_scaling_experiment))
        self.assertTrue(callable(nesterov_scaling_experiment))


if __name__ == "__main__":
    unittest.main()
