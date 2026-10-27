# Momentum in Large-Batch Training

**Polyak Enlarges the Critical Batch Size, Nesterov Improves Data Efficiency**

[Paper](https://arxiv.org/abs/2609.02728) · [Blog](https://alexhuang13.github.io/momentum_blog/index.html) · [Theory and numerical results](docs/kernel_results.md)

Jia-Nan Wang, Zixun Huang, Kairui Li, and Lei Wu

This repository studies how SGD, Polyak momentum and Nesterov momentum respond
to batch size, momentum and data budget. The kernel experiments connect stability
with data scaling in a tractable power-law model. The neural-network experiments
measure fixed-budget ImageNet performance and finite-horizon CIFAR-10 stability.
Their empirical protocols are documented separately from the kernel theory.

## Choose an experiment

| Question | Kernel experiments | Neural-network experiments |
| --- | --- | --- |
| How does loss scale with data budget or batch size? | [Data scaling](data_scaling/README.md) | [ImageNet / ResNet-18](data_scaling/imagenet/README.md) |
| Which learning rates remain stable? | [Stability](stability/README.md) | [CIFAR-10 / ResNet-18](stability/cifar10/README.md) |
| How do the transient and noise floor evolve? | [Loss dynamics](loss/README.md) | — |
| What happens as momentum increases over time? | [Blowup](blowup/README.md) | — |

The earlier [ImageNet stability notebook](momentum_nn/README.md) remains available
with its own protocol and implementation. It is distinct from the two archived
neural-network experiments above.

## Start with the saved results

Clone the repository and run these commands from its root. Redrawing requires
NumPy, Matplotlib and pandas; it does not train a model or require a GPU.

```bash
python -m pip install numpy matplotlib pandas
python data_scaling/imagenet/plot.py
python stability/cifar10/plot.py
```

New figures are written under each module's `outputs/` directory. Committed
result tables and original figures stay in `results/`.

| ImageNet: fixed sample budget | CIFAR-10: stability across batch size and damping |
| --- | --- |
| [Original figure (PDF)](data_scaling/imagenet/results/paper_3epochs/fixed_D_optimal_loss_vs_batch.pdf) | [Original figure (PDF)](stability/cifar10/results/paper_2000steps/cifar_stability_merged.pdf) |

ImageNet includes 396 configurations at seed 0 and all per-run summary/metric
files. CIFAR-10 includes 1,309 aggregate cells, boundary and fit tables, follow-up
audit tables and figures; its original raw trial histories are not in this release.
The module READMEs distinguish aggregate redrawing from raw-data auditing and
fresh training.

## Run an experiment

Use Python 3.10+. Install PyTorch and torchvision matched to your CUDA environment,
then choose the dependencies you need:

```bash
# Existing kernel notebooks
python -m pip install -e '.[notebooks]'
# Neural-network training and plotting
python -m pip install -e '.[nn]'
```

Prepare both neural-network grids without training:

```bash
python data_scaling/imagenet/sweep.py
python stability/cifar10/sweep.py
```

Each command saves a manifest in `outputs/`; add `--execute` to start training.
Set `IMAGENET_ROOT` or `CIFAR10_ROOT` to the corresponding dataset directory.
CIFAR uses `DEVICES=cuda:0,cuda:1` by default; use `--devices cuda:0` for one GPU.
See the module README before launching a full grid.

Kernel notebook entrypoints:

| Topic | Notebooks |
| --- | --- |
| Data scaling | [Batch regimes](data_scaling/batch_regimes.ipynb), [critical batch size](data_scaling/critical_batch_size.ipynb) |
| Stability | [Batch size](stability/batch_size.ipynb), [momentum gap](stability/momentum_gap.ipynb) |
| Loss dynamics | [Scaling-law fits](loss/scaling_law_fits.ipynb), [phase transitions](loss/phase_transitions.ipynb) |
| Blowup | [Increasing momentum](blowup/increasing_momentum.ipynb) |

Run `jupyter lab` from the repository root. Notebook quick settings check code
paths; use the documented paper settings for quantitative results. The kernel
notebooks automatically choose CUDA, Apple MPS, or CPU. The neural-network
training protocols have their own device and precision settings.

## How the code is organized

Each scientific category keeps its existing notebooks and reusable kernel code.
Within the two neural-network modules:

| Path | Purpose |
| --- | --- |
| `README.md` | Protocol, quick start, available data and limitations |
| `train.py` | Training entrypoint with the original optimizer implementation |
| `sweep.py` | Complete configuration grid; dry run by default |
| `plot.py` | Rebuild figures in a separate output directory |
| `source/` | Original implementations and historical tools, with a reading guide |
| `results/` | Published result snapshot |
| `outputs/` | New generated work, ignored by Git |
| `docs/` | Provenance, original instructions and file checksums |

The original training implementations are preserved byte-for-byte so historical
results and code audits remain traceable. The entrypoints provide consistent paths
and commands without replacing scientific update rules. The detailed kernel
exposition and saved figures are in [Theory and numerical results](docs/kernel_results.md).

## Validation

```bash
python -m unittest discover -s tests -v
```

The CPU-only checks cover source integrity, configuration generation, archived
aggregate results and plotting. They do not rerun ImageNet or CIFAR training.
See [validation notes](docs/neural_experiments_validation.md).

## Citation

```bibtex
@misc{wang2026momentumlargebatchtrainingpolyak,
  title={Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency},
  author={Jia-Nan Wang and Zixun Huang and Kairui Li and Lei Wu},
  year={2026},
  eprint={2609.02728},
  archivePrefix={arXiv},
  primaryClass={stat.ML},
  url={https://arxiv.org/abs/2609.02728},
}
```

