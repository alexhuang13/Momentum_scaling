# Momentum: Scaling Laws for SGD, Heavy Ball, and Nesterov

This repository studies three first-order optimizers — **SGD**, **Heavy Ball
(HB)**, and **Nesterov (NAG)** — from two complementary angles: an exactly
solvable theory on synthetic linear regression, and an empirical study on real
image classifiers (MNIST / CIFAR-10). The two use the *same* optimizer
definitions, so the theory and the experiments line up.

## Repository layout

```
linear_regression/       Theory + exactly-solvable synthetic experiments
  linear_regression_scaling.ipynb   scaling-law notebook (numpy + matplotlib)

cifar_mnist/             Empirical study on real networks (PyTorch)
  experiments/           models, optimizers, data, training, run scripts, plotting
  configs/               per-(model, dataset) hyperparameters
  requirements.txt
  README.md              detailed setup, model specs, and run instructions
```

## `linear_regression/` — theory

A diagonal linear-regression model with power-law spectrum
`λ_j ≍ j^{-β}` and source condition `θ*_j² ≍ j^{-1-(s-1)β}`, where the population
excess loss is computed **analytically**. The notebook runs HB / NAG / SGD under
the hyper-parameter scaling prescribed by the theory and produces the key
deliverable: **excess loss vs. data budget `D`** on log-log axes, with fitted
slopes matching the predicted exponent

```
E[E_K] ≍ D^{-s/(s+1)}.
```

The notebook also states the two-sided heavy-ball scaling law it is verifying,

```
E[E_t] ≍ ρ^t + ((1-ρ)/(ηt))^s + ησ²/(B(1-ρ)) + (self-induced noise terms).
```

Self-contained: only `numpy` + `matplotlib`.

## `cifar_mnist/` — experiments

Ports the study to real networks. The three optimizers are **hand-written** to
match the notebook's update rules exactly (not `torch.optim`'s Nesterov, which
uses a different LR convention):

```
SGD        θ_{t+1} = θ_t - η·g(θ_t)
HeavyBall  θ_{t+1} = θ_t - η·g(θ_t)          + ρ·(θ_t - θ_{t-1})
Nesterov   θ_{t+1} = θ_t - η·g(θ_t + ρ·m_t)  + ρ·(θ_t - θ_{t-1})
```

Two experiments:

* **A — baseline comparison**: fixed dataset, multi-epoch, all three optimizers
  under a **shared fixed learning rate** and `ρ=0.9`; compares train-loss /
  test-loss / test-accuracy curves.
* **B — compute-budget scaling**: fixed batch size `B`, sweep the step count `K`,
  plot final **test loss vs. `D = B·K`** on log-log axes and fit a slope — the
  real-network analogue of the notebook's *excess loss vs. D* curve.

Models (from the experimental setting): fully-connected net for MNIST (~99k
params), a 3-layer CNN (~576k) and ResNet-32 (~467k) for CIFAR-10. See
[`cifar_mnist/README.md`](cifar_mnist/README.md) for full architecture specs,
install steps, and exact run commands.

> **Honesty note on B.** On real data the spectral / source exponents `(s, β)`
> are unknown, so — unlike the notebook — `B, K, ρ` are **not** scaled with `D`
> (that scaling requires the exponents). Experiment B is therefore an *empirical*
> power-law of loss vs. compute, not a reproduction of the exact `D^{-s/(s+1)}`
> rate.

## Quick start

Theory (no GPU needed):

```bash
cd linear_regression
jupyter notebook linear_regression_scaling.ipynb
```

Experiments (on a GPU machine; Python 3.10–3.12 for torch wheels):

```bash
cd cifar_mnist
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m experiments.run_baseline --config configs/mnist_fc.yaml --seeds 0 --epochs 3   # smoke test
```

See [`cifar_mnist/README.md`](cifar_mnist/README.md) for the full run matrix.
