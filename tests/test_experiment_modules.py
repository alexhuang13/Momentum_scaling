import unittest

import torch

from momentum_scaling.blowup import (
    BlowupRuntime,
    blowup_experiment,
    momentum_at,
    simulate_increasing_momentum,
)
from momentum_scaling.core import build_problem
from momentum_scaling.loss_dynamics import LossDynamicsRuntime, loss_dynamics_experiment
from momentum_scaling.scaling import (
    D_PAPER,
    ScalingRuntime,
    nesterov_scaling_experiment,
    polyak_scaling_experiment,
    scaling_experiment,
)
from momentum_scaling.stability import StabilityRuntime, stability_experiment


class ExperimentModuleTests(unittest.TestCase):
    def test_public_experiment_entry_points(self):
        self.assertEqual(D_PAPER[0], 2000)
        self.assertTrue(callable(stability_experiment))
        self.assertTrue(callable(loss_dynamics_experiment))
        self.assertTrue(callable(polyak_scaling_experiment))
        self.assertTrue(callable(nesterov_scaling_experiment))
        self.assertTrue(callable(scaling_experiment))
        self.assertTrue(callable(blowup_experiment))
        self.assertTrue(StabilityRuntime)
        self.assertTrue(ScalingRuntime)
        self.assertTrue(LossDynamicsRuntime)
        self.assertTrue(BlowupRuntime)

    def test_increasing_momentum_schedule(self):
        self.assertAlmostEqual(momentum_at(1), 0.5)
        self.assertAlmostEqual(momentum_at(9), 0.9)
        with self.assertRaises(ValueError):
            momentum_at(0)

    def test_short_blowup_trajectory(self):
        problem = build_problem(
            beta=2.0,
            s=1.0,
            sigma=0.0,
            dim=4,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        result = simulate_increasing_momentum(
            problem,
            algorithm="nesterov",
            batch_size=2,
            eta=0.01,
            steps=3,
            replicas=2,
            record_every=1,
            seed=0,
            blowup_loss=10.0,
        )
        self.assertEqual(result["steps"].tolist(), [0, 1, 2, 3])
        self.assertEqual(result["median_loss"].shape, (4,))
        self.assertIsNone(result["blowup_step"])


if __name__ == "__main__":
    unittest.main()
