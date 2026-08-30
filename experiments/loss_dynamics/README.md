# Loss-dynamics experiments

Loss dynamics means loss as a function of optimization step $k$ at fixed
hyperparameters. These experiments separate transient decay, signal learning,
and the stochastic noise floor; they are not data-budget scaling sweeps.

| Notebook | Purpose | Output directory |
|---|---|---|
| [`scaling_law_fits.ipynb`](scaling_law_fits.ipynb) | Fit the transient, signal, and noise terms across six $(s,\beta)$ settings | `results/loss_dynamics/scaling_law_fits/` |
| [`phase_transitions.ipynb`](phase_transitions.ipynb) | Locate transient-to-scaling and noise-floor transitions | `results/loss_dynamics/phase_transitions/` |

Reusable simulation and fitting code lives in
[`src/momentum_scaling/loss_dynamics.py`](../../src/momentum_scaling/loss_dynamics.py).
