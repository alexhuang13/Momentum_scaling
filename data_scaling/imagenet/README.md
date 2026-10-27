# ImageNet: critical batch size at a fixed sample budget

[Experiment index](../README.md) · [Repository overview](../../README.md)

Compare SGD, Polyak and Nesterov on ResNet-18 with GroupNorm at a fixed sample
budget. This is the neural-network batch-size experiment in Section 7.2 /
Appendix E.2.1. Read `train.py`, `sweep.py` and `plot.py` first; the
[source guide](source/README.md) points to the original implementation.

## Recreate the figure

From the repository root, no GPU or PyTorch required:

```bash
python -m pip install numpy matplotlib pandas
python data_scaling/imagenet/plot.py
# Also recompute the best configurations from all 396 per-run summaries.
python data_scaling/imagenet/plot.py --reaggregate
```

Output: `data_scaling/imagenet/outputs/figures/`. The saved results are unchanged.
Use `--output` for another destination or `--data-dir` to plot a new run.

[Original PNG](results/paper_3epochs/fixed_D_optimal_loss_vs_batch.png) ·
[Original PDF](results/paper_3epochs/fixed_D_optimal_loss_vs_batch.pdf) ·
[Best configurations](results/paper_3epochs/best_by_batch.csv)

## Run the experiment

Install a matched CUDA PyTorch/torchvision build and `pip install -e '.[nn]'`
from the repository root. The ImageNet directory must contain `train/` and `val/`.

```bash
export IMAGENET_ROOT=/path/to/imagenet
# Generate 396 commands and one complete manifest; no training.
python data_scaling/imagenet/sweep.py
# Execute the grid, then aggregate and plot.
python data_scaling/imagenet/sweep.py --execute
```

New runs go to `outputs/paper_3epochs/`. Override `--data-root`, `--runs-root` or
`--workers` as needed. The checked-in [grid](configs/paper_grid.csv) contains all
recorded configurations, without selecting on outcome. A single configuration:

```bash
python data_scaling/imagenet/train.py --data-root /path/to/imagenet --algorithm polyak --global-batch 256 --lr 0.08 --momentum 0.9 --sample-budget 3843501 --precision bf16 --output-dir data_scaling/imagenet/outputs/single
```

The sweep keeps an existing run only after checking its configuration and
completion. Interrupted training starts again from its seed; model checkpoints
are not provided. The original environment was not fully locked, so these commands
reproduce the protocol and configurations, not guaranteed bitwise-identical results.

## Protocol

| Item | Setting |
| --- | --- |
| Model | ResNet-18, GroupNorm, trained from scratch |
| Budget | D = 3 × 1,281,167 = 3,843,501 samples; floor(D/B) updates |
| Batch sizes | 1, 2, 4, ..., 4096; micro-batches at most 128 |
| Optimization | Constant learning rate/momentum; no warmup, weight decay or label smoothing |
| Precision and seed | BF16; seed 0 |
| Evaluation | All 50,000 validation images after training |
| Selection | Lowest validation CE among completed stable configurations per method/batch |

Effective learning rate is gamma = eta for SGD and eta/(1-rho) for momentum.
Nesterov evaluates gradients at theta + rho*velocity, restores theta, then updates;
validation evaluates theta. No optimizer or training code was rewritten.

## Data and interpretation

The snapshot has 396 configurations: 49 SGD, 171 Polyak, 176 Nesterov. The search
contains 360 main-grid runs, 31 extra small-batch runs and 5 extra Nesterov runs.
All supplied summaries are marked stable. Per-run `summary.json` and
`train_metrics.csv` are included alongside the aggregate tables.

Results are single-seed and selected on validation loss. Zero standard-deviation
columns do not measure run-to-run certainty. Plot region dividers are display
annotations based on batch-point positions, not estimated critical-batch boundaries.

The archived `commands_pilot.sh` and `pilot_manifest.csv` contain only three
configurations because the historical driver overwrote them per invocation.
Use `sweep.py` and its complete manifest instead. The source driver remains
available for custom searches with explicit paths.

See [original instructions](docs/README.original.md) and
[import checksums](docs/import_manifest.csv) for provenance.
