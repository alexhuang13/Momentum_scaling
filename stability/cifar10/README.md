# CIFAR-10: learning-rate stability boundaries

[Experiment index](../README.md) · [Repository overview](../../README.md)

Compare SGD, Polyak and Nesterov stability at a fixed 2,000-update horizon on
ResNet-18 with GroupNorm (Appendix E.2.2). Start with the three entrypoints below;
the [source guide](source/README.md) explains the original training and analysis files.

## Recreate the figures

From the repository root, no GPU or PyTorch required:

```bash
python -m pip install numpy matplotlib
python stability/cifar10/plot.py
```

This checks aggregate counts, recomputes boundaries and fits with the original
functions, and writes figures to `stability/cifar10/outputs/redrawn_summary/`.
Use `--individual` for separate panels, `--output` for another destination, or
`--stable-fraction 0.8` to examine a different empirical boundary threshold.
The default is 50%. Published tables and original figures are unchanged.

[Original figure](results/paper_2000steps/cifar_stability_merged.pdf) ·
[Seed coverage](results/paper_2000steps/seed_coverage.pdf) ·
[Aggregate counts](results/paper_2000steps/merged_cells.csv)

## Run a fresh scan

Use Python 3.10+, a compatible Unix environment (`fcntl` is used), and a matched
PyTorch/torchvision build. Install `pip install -e '.[nn]'` from the repository root.

```bash
export CIFAR10_ROOT=/path/to/cifar10
# Prepare the manifest only (4,945 jobs); no training.
python stability/cifar10/sweep.py
# Launch on two GPUs; use --devices cuda:0 for one GPU.
python stability/cifar10/sweep.py --execute --devices cuda:0,cuda:1
# Small actual training check using synthetic data.
python stability/cifar10/train.py --preset smoke --output stability/cifar10/outputs/smoke
```

`--dry-run` is also accepted explicitly. The sweep defaults to `outputs/new_scan/`,
which can be changed with `--output` or `CIFAR_OUTPUT`. Device defaults can be
set with `DEVICES`. The manifest and original trainer handle run resumption.

The fresh grid uses the revised record-mode protocol throughout. It is not an
exact reconstruction of the historical adaptive reruns or their varying seed
counts and safety modes. Historical tools remain in `source/` for complete raw
datasets and require explicit input paths.

## Protocol and boundaries

| Item | Setting |
| --- | --- |
| Model and horizon | ResNet-18, GroupNorm; 2,000 updates |
| Batch scan | B = 1, 2, 4, ..., 256; damping 1-beta = 1/64 |
| Damping scan | Nine log-spaced values from 1/64 to 1/4; B = 4 |
| Probe | Fixed 1,024 training images; evaluation every 50 updates |
| Stability | Last ten-loss mean <= 2 × initial loss and <= 1.25 × previous ten-loss mean |
| Optimization | Constant learning rate/momentum; no warmup or weight decay |

Numerical failures and applicable stopping checks count as failures. A per-run
classification differs from the 50% stable-seed boundary in the final figure.
Heatmaps show measured proportions. Seed-count-weighted decreasing isotonic
regression is used only for boundary extraction; fitted curves describe empirical
shapes and do not establish an asymptotic stability law for neural networks.

The original trainer stops when parameter norm exceeds its threshold. The revised
trainer defaults to `record`: it logs a crossing and continues, retaining other
divergence checks. Archived aggregates retain legacy parameter-stop failures under
the original merge policy. Both source versions and their audit hashes are preserved.

## Included data and reproduction limits

The snapshot contains 1,309 aggregate cells, 54 boundary rows, boundary-extraction
and fit tables, a 250-row follow-up audit, before/after counts, fit metadata and
two PDFs. Seed coverage varies; follow-up records use seeds 23–32. Final panels
exclude damping 1/128 and retain the 2,000-update horizon. SGD damping columns
reuse the same B=4 observations, rather than independent trials.

Raw trial manifests, configs, summaries, probe histories, baseline snapshots and
complete follow-up directories were not included in the upload. `plot.py` redraws
from aggregates and explicitly records that raw trials were not reaudited. Running
`source/plot_cifar_all.py --followup-dir ...` requires those original raw inputs.
The existing audit CSV is preserved as supplied evidence.

See [original instructions](docs/README.original.md) and
[import checksums](docs/import_manifest.csv).
