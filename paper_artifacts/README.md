# Saved paper artifacts

This directory contains committed outputs from paper-scale runs. It is read-only
reference material, not an experiment entry point.

The directory names mirror `experiments/`:

- `scaling/`: batch-regime, critical-batch-size, and phase-diagram outputs;
- `stability/`: saved stability figures, grids, and summaries;
- `loss_dynamics/`: fitted loss curves and transition data;
- `blowup/`: the increasing-momentum diagnostic figure.

`style_reference.png` is the visual reference image used while matching the
paper figures; it is not loaded by experiment code.

New runs must write to the Git-ignored `results/` directory. Copy a result into
`paper_artifacts/` only when intentionally updating a committed paper artifact.
