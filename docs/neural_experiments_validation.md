# Neural-network experiment validation

Validated on 2026-10-27 with Python 3.12.14, using CPU-only analysis.
Run the regression suite from a checkout:

```bash
python -m unittest discover -s tests -v
```

All eight tests passed (9.3 seconds in the preparation environment):

- Checked 826 imported source/result files against their SHA-256 manifests.
- Preserved both CIFAR trainers' original training-function audit hashes.
- Parsed all 396 generated ImageNet commands with the original trainer and
  compared every recorded training hyperparameter with its source summary.
- Recomputed ImageNet's 396 configuration aggregates and 39 selected best rows,
  matching the published tables within numeric tolerance 1e-12.
- Recomputed CIFAR's 1,309 boundary-extraction rows, 54 empirical boundary rows
  and three fit rows, matching the published data within tolerance 1e-12.
- Exercised all six train/sweep/plot CLI help commands. Both sweeps default to
  configuration generation, with no training: 396 ImageNet jobs and 4,945 fresh
  CIFAR jobs. The fresh CIFAR grid is separate from historical adaptive reruns.
- Ran both plotting entrypoints and checked their outputs. CIFAR redraw metadata
  explicitly records that raw trial histories were not reaudited.
- Confirmed that plot entrypoints reject published result directories as outputs.

All Python source parsed successfully. The main ImageNet and CIFAR figures were
also visually inspected during the initial aggregate reconstruction. Original
training/optimizer source, result CSV/JSON files and supplied figures are retained
byte-for-byte. Module import manifests include original filenames and checksums.

No ImageNet or CIFAR training was rerun. PyTorch and torchvision were unavailable
in the preparation runtime, so GPU behavior and training throughput have not been
newly validated. CIFAR raw trial histories and baseline snapshots were not supplied;
the available data support aggregate reconstruction, not a new raw-data audit.
