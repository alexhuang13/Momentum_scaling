# ImageNet implementation guide

Use the parent `train.py`, `sweep.py` and `plot.py` for normal work. The Python
files here are byte-for-byte copies of the uploaded implementation.

| Read next | What to inspect |
| --- | --- |
| [imagenet_momentum_experiment.py](imagenet_momentum_experiment.py) | `optimizer_for`: SGD and explicit position/velocity momentum; `train_one`: training; `aggregate_configurations` and `select_best_by_batch`: selection |
| [plot_imagenet_cbs.py](plot_imagenet_cbs.py) | `load_optimal_losses` and `plot_cbs`: table validation and figure construction |
| [imagenet_pilot_driver.py](imagenet_pilot_driver.py) | Historical effective-learning-rate grid generation |

The historical driver and direct plotting default paths refer to the old layout.
Pass `--script`, `--data-root`, `--runs-root` or `--data-dir` explicitly when using
these files directly. The parent entrypoints supply portable paths.
