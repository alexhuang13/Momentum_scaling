# Blowup

This folder studies the explicitly time-varying schedule
$`\beta_k=1-1/(k+1)`$.

| File | What to read or run |
|---|---|
| [`increasing_momentum.ipynb`](increasing_momentum.ipynb) | Quick and paper-scale experiment entry point |
| [`experiment.py`](experiment.py) | Increasing-momentum simulation |
| [`core.py`](core.py) | Model, optimizer updates, and shared numerical utilities |
| [`plotting.py`](plotting.py) | Result plotting used by the notebook |

Saved paper-scale outputs are under [`results/`](results/). New notebook runs
write to `blowup/results/generated/`, which is ignored by Git.

Main result: Polyak eventually becomes unstable under the increasing-momentum
schedule, while Nesterov remains stable over the saved 500,000-step horizon.
This diagnostic uses time-varying momentum; the constant-momentum theorems
are not applied to this schedule.

Notation follows the latest paper: $`q`$ is the spectral exponent and $`\beta`$ is
momentum. Historical kernel code and saved artifacts may use `beta` for $`q`$,
`rho` for $`\beta`$, and `K` for the update horizon $`N`$.
