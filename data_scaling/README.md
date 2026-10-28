# Data scaling

This folder groups endpoint-loss scaling across data budgets and batch sizes.
The notebooks below study the kernel model; [ImageNet / ResNet-18](imagenet/README.md)
adds the fixed-sample-budget neural-network experiment with 396 archived runs.

| File | What to read or run |
|---|---|
| [`batch_regimes.ipynb`](batch_regimes.ipynb) | Small-, large-, and ultra-large-batch scaling exponents |
| [`critical_batch_size.ipynb`](critical_batch_size.ipynb) | Best batch size at fixed $D=2^{14}$ |
| [`experiment.py`](experiment.py) | Reusable PyTorch implementation and schedules |
| [`core.py`](core.py) | Model, optimizer updates, and shared numerical utilities |
| [`plot_rate_diagram.py`](plot_rate_diagram.py) | Recreate the theoretical phase diagram |
| [`plot_critical_batch_size.py`](plot_critical_batch_size.py) | Replot the fixed-budget sweep |

Saved paper-scale outputs are under [`results/`](results/). New notebook runs
write to `data_scaling/results/generated/`, which is ignored by Git.

Main result: Theorem 6.3 optimizes the final-step expected excess risk under
$N=D/B$, stability, and the requirement that the transient decay within the
training horizon. Polyak enlarges the critical batch exponent from
$b_1=s/(s+1)$ to $b_3=(2s+1)/(2s+2)$ without improving the best small-batch
rate. Nesterov has the fastest large-batch exponent, attaining
$2sq/(2sq+q+1)$ at $b_2=(2s+1/q)/(2s+1+1/q)$. At fixed $D=2^{14}$,
the archived sweep also shows lower loss and a larger best batch size for Nesterov.
See the [phase diagram](../README.md#3-data-scaling).


Notation follows the latest paper: $q$ is the spectral exponent and $\beta$ is
momentum. Historical kernel code and saved artifacts may use `beta` for $q$,
`rho` for $\beta$, and `K` for the update horizon $N$.
