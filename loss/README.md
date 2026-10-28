# Loss

This folder studies the expected population excess risk
$\mathbb{E}[\mathcal{E}(\boldsymbol{\theta}_k)]$ as a function of optimization
step $k$ at fixed hyperparameters, separating transient decay, signal learning,
and the noise floor. Population risk is
$\mathcal{R}(\boldsymbol{\theta})=\mathcal{E}(\boldsymbol{\theta})+\sigma^2/2$.
See the [risk-dynamics summary](../README.md#2-excess-risk-scaling-laws) for
the damping regimes and conditions of Theorems 5.1–5.2.

| File | What to read or run |
|---|---|
| [`scaling_law_fits.ipynb`](scaling_law_fits.ipynb) | Fits across six $(s,q)$ settings |
| [`phase_transitions.ipynb`](phase_transitions.ipynb) | Transient-to-scaling and noise-floor transitions |
| [`experiment.py`](experiment.py) | Reusable long-time simulation and fitting code |
| [`core.py`](core.py) | Model, optimizer updates, and shared numerical utilities |

Saved paper-scale outputs are under [`results/`](results/). New notebook runs
write to `loss/results/generated/`, which is ignored by Git.

Main result: the fitted curves recover transient, algebraic, and noise-floor
regimes. In the mixed-damping regime, the transient envelopes are bounded by
$\beta^k$ for Polyak and $\min\{1,(\eta k)^{-s}\}\beta^k$ for Nesterov.
Polyak's crossover is
$\tau_\beta\eqsim(1-\beta)^{-1}\log[\eta/(1-\beta)^2]$; Nesterov has a
lower noise floor when the effective learning rate $\eta_\beta=\eta/(1-\beta)$
is large.

Notation follows the latest paper: $q$ is the spectral exponent and $\beta$ is
momentum. Historical kernel code and saved artifacts may use `beta` for $q$,
`rho` for $\beta$, and `K` for the update horizon $N$.
