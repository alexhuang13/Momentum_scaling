# Loss

This folder studies loss as a function of optimization step $k$ at fixed
hyperparameters, separating transient decay, signal learning, and the noise
floor.

| File | What to read or run |
|---|---|
| [`scaling_law_fits.ipynb`](scaling_law_fits.ipynb) | Fits across six $(s,\beta)$ settings |
| [`phase_transitions.ipynb`](phase_transitions.ipynb) | Transient-to-scaling and noise-floor transitions |
| [`experiment.py`](experiment.py) | Reusable long-time simulation and fitting code |
| [`core.py`](core.py) | Model, optimizer updates, and shared numerical utilities |

Saved paper-scale outputs are under [`results/`](results/). New notebook runs
write to `loss/results/generated/`, which is ignored by Git.

Main result: the fitted curves recover the transient, algebraic, and noise-floor
regimes, with $k_{\rm tr}\propto(1-\rho)^{-1}$ and a lower Nesterov noise floor.
