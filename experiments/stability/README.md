# Stability experiments

Stability experiments classify a run by whether its final moving-average loss
remains finite and below a fixed threshold. The reported boundary is the largest
stable learning rate.

| Notebook | Swept variables | Fixed variable | Output directory |
|---|---|---|---|
| [`batch_size.ipynb`](batch_size.ipynb) | $B$ and $\eta$ | $1-\rho=2^{-24}$ | `results/stability/batch_size/` |
| [`momentum_gap.ipynb`](momentum_gap.ipynb) | $1-\rho$ and $\eta$ | $B=32$ | `results/stability/momentum_gap/` |

The automation-friendly implementation is in
[`src/momentum_scaling/stability.py`](../../src/momentum_scaling/stability.py).
