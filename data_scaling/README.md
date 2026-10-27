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

Main result: Nesterov has the fastest large-batch exponent. At fixed
$D=2^{14}$ it also reaches the lowest loss and shifts the best batch size upward.

