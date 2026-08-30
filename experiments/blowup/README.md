# Blowup diagnostic

[`increasing_momentum.ipynb`](increasing_momentum.ipynb) studies the explicit
schedule $\rho_k=1-1/(k+1)$. It is separated from stability because the optimizer
changes over time instead of scanning a fixed $(B,\eta,\rho)$ configuration.

The notebook defaults to a quick smoke test. Set `PAPER_RUN=True` for the saved
paper configuration. Outputs are written to
`results/blowup/increasing_momentum/`.

The reusable implementation is in
[`src/momentum_scaling/blowup.py`](../../src/momentum_scaling/blowup.py).
