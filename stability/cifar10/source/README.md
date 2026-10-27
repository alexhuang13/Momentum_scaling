# CIFAR implementation guide

Use the parent `train.py`, `sweep.py` and `plot.py` for normal work. The files here
preserve the uploaded source byte-for-byte, including training-function hashes
used by historical audits. They are grouped by role below.

| Role | Files and functions |
| --- | --- |
| Current trainer | [cifar_stability_revised.py](cifar_stability_revised.py): `PaperMomentum`, `classify`, `run_trial`, `make_manifest` |
| Current figure analysis | [plot_cifar_all.py](plot_cifar_all.py): `load_cells`, `boundaries`, `fit_models`, `plot_all` |
| Original stop-mode protocol | [cifar_stability.py](cifar_stability.py) |
| Earlier raw-data audit | [plot_cifar_summary.py](plot_cifar_summary.py) |
| Historical staged reruns | [run_targeted_round2.py](run_targeted_round2.py), [run_gap9_midpoints.py](run_gap9_midpoints.py), [run_nesterov_followup.py](run_nesterov_followup.py) |
| Raw-data export | [export_cifar_final_results.py](export_cifar_final_results.py) |
| Historical shell helpers | `setup.sh`, `start.sh`, `rerun_cifar_outliers_minimal_home.sh` |

The audit and rerun tools need their original complete manifests, histories and
baseline snapshots; the final aggregate table is not a substitute for those inputs.
Original server paths remain in these historical tools. Supply explicit paths and
use a new output directory. `start.sh` still selects the original stop-mode trainer.
