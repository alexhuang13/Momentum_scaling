# Scaling experiments

Scaling experiments compare final or tail-averaged loss across data budgets and
batch-size choices. They do not estimate a stability boundary or analyze the
step-by-step shape of one training trajectory.

| Notebook | Purpose | Output directory |
|---|---|---|
| [`batch_regimes.ipynb`](batch_regimes.ipynb) | Compare small-, large-, and ultra-large-batch data-scaling exponents | `results/scaling/batch_regimes/<setting>/` |
| [`critical_batch_size.ipynb`](critical_batch_size.ipynb) | Find the best batch size at fixed $D=2^{14}$ | `results/scaling/critical_batch_size/<setting>/` |

Reusable schedules and CLI orchestration live in
[`src/momentum_scaling/scaling.py`](../../src/momentum_scaling/scaling.py).
