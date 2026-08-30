# Experiment map

Run notebooks from the repository root after installing the notebook dependencies
with `python -m pip install -e ".[notebooks]"`. Every new run writes to `results/<category>/...`;
saved paper outputs live separately under `paper_artifacts/`.

| Category | Scientific question | Independent axis | Entry points |
|---|---|---|---|
| [Scaling](scaling/) | How does final loss change with data budget or batch-size regime? | $D$ or $B$ | `batch_regimes.ipynb`, `critical_batch_size.ipynb` |
| [Stability](stability/) | Which learning rates remain stable? | $B$, $1-\rho$, and $\eta$ | `batch_size.ipynb`, `momentum_gap.ipynb` |
| [Loss dynamics](loss_dynamics/) | How does loss evolve during a stable run? | Optimization step $k$ | `scaling_law_fits.ipynb`, `phase_transitions.ipynb` |
| [Blowup](blowup/) | What fails under an increasingly aggressive momentum schedule? | Optimization step $k$ with time-varying $\rho_k$ | `increasing_momentum.ipynb` |

These categories are intentionally disjoint:

- **Scaling** compares endpoint performance across budgets or batch sizes.
- **Stability** estimates the boundary between bounded and divergent runs.
- **Loss dynamics** resolves the transient, signal, and noise regimes inside stable runs.
- **Blowup** is a targeted failure diagnostic with a time-varying optimizer schedule.

The reusable numerical implementations are in `src/momentum_scaling/`. Notebooks
contain paper-specific grids, presentation, and interactive controls; generated
files do not belong beside notebook source.
