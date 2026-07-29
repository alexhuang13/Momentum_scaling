# From Polyak to Nesterov: Momentum Scaling in Power-Law Kernel Regression

This repository provides a streamlined implementation for reproducing the experiments in **“From Polyak to Nesterov: Landscape-Aware Momentum Methods May Improve Stability and Data Scaling.”** It studies stochastic optimization in high-dimensional power-law kernel regression and compares **SGD**, **Polyak heavy-ball momentum**, and **Nesterov momentum** in terms of stability, critical batch size, and data-scaling behavior.

> The former duplicated CPU and GPU notebooks have been replaced by a single PyTorch implementation. The `--device` option only changes the compute device; the model, random process, optimizer updates, and experiment configurations all share the same code.

## Main Results

The feature covariance spectrum and target satisfy

$$
\lambda_j \asymp j^{-\beta}, \qquad
\lambda_j |\theta_j^\star|^2 \asymp j^{-(1+s\beta)},
$$

where $\beta>1$ is the capacity exponent and $s>0$ is the source exponent. The data budget is $D=BK$, where $B$ is the batch size and $K$ is the number of optimization steps.

### 1. Stochastic Stability Boundaries

| Method | Critical learning rate |
|---|---|
| SGD | $\eta_{\mathrm{stab}}\asymp 1$ |
| Polyak | $\eta_{\mathrm{stab}}\asymp \min\{1,B(1-\rho)\}$ |
| Nesterov | $\eta_{\mathrm{stab}}\asymp \min\{1,B^\beta(1-\rho)\}$ |

Nesterov's look-ahead correction makes its stable region grow faster with batch size, allowing it to reach the deterministic stability ceiling earlier than Polyak momentum.

### 2. Data Scaling

In the presence of label noise:

- SGD achieves the optimal data-scaling rate $D^{-s/(s+1)}$.
- Polyak momentum achieves the same optimal rate but preserves it over a substantially larger range of batch sizes.
- Nesterov momentum achieves the faster rate
  $$
  D^{-\frac{2s\beta}{2s\beta+\beta+1}+o(1)}.
  $$

![Batch-size phase diagram](assets/figures/rate_diagram.png)

## Key Numerical Results from the Paper

The left panel validates the distinct stability boundaries of Polyak and Nesterov momentum. The middle panel shows that Polyak momentum preserves the optimal scaling rate beyond SGD's critical batch size. The right panel shows that Nesterov momentum achieves a faster data-scaling rate in the large-batch regime.

| Stability boundary | Polyak critical batch size | Nesterov large-batch acceleration |
|---|---|---|
| ![Stability boundary](assets/figures/beta1_5s2_plot.png) | ![Polyak scaling](assets/figures/mom_large_batch.png) | ![Nesterov scaling](assets/figures/nag_acc.png) |

## Repository Structure

```text
.
├── README.md
├── pyproject.toml              # Editable installation and CLI entry point
├── requirements.txt
├── src/momentum_scaling/
│   ├── core.py                 # Power-law problem, shared backend, and optimizers
│   ├── experiments.py          # Stability and data-scaling experiments
│   ├── plotting.py             # Generates figures from saved results
│   └── cli.py                  # Unified command-line interface
├── tests/test_core.py          # Tests for the model and update equations
├── paper/                      # LaTeX source, bibliography, and original figures
├── assets/figures/             # Key PNG figures displayed in this README
└── results/                    # Runtime outputs, excluded from version control
```

The old CPU/GPU notebooks, duplicated plotting scripts, caches, temporary binaries, and redundant PDF files have been removed. The paper source is retained under `paper/` for direct comparison with the theorems, experimental configurations, and original figures.

## Installation

Python 3.10 or later is required. Creating an isolated environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`requirements.txt` contains only NumPy, PyTorch, and Matplotlib. For NVIDIA GPUs, first install the PyTorch build that matches the local CUDA version, then run `python -m pip install -e .`.

Verify the installation with:

```bash
python -m unittest discover -s tests -v
momentum-scaling --help
```

## Quick Reproduction

The quick preset checks the environment, device selection, and complete experiment pipeline. It normally finishes quickly, but it is **not a quantitative replacement for the paper configuration**.

```bash
momentum-scaling all --preset quick --device auto --output-dir results/quick
```

The experiments can also be run without installing the command-line entry point:

```bash
PYTHONPATH=src python -m momentum_scaling.cli all \
  --preset quick --device auto --output-dir results/quick
```

The generated files are:

```text
results/quick/
├── stability.json
├── stability.npz
├── stability.png
├── stability.pdf
├── data_scaling.json
├── data_scaling.png
└── data_scaling.pdf
```

The JSON files record the configuration, device, random seed, means, and standard deviations for every experiment point. The NPZ file additionally stores the complete stability grid and per-seed stability boundaries for further inspection.

## CPU, CUDA, and Apple Silicon

```bash
# Automatic selection: CUDA, then MPS, then CPU
momentum-scaling all --preset quick --device auto

# Force CPU execution
momentum-scaling all --preset quick --device cpu --dtype float64

# Select an NVIDIA GPU
momentum-scaling all --preset quick --device cuda:0 --dtype float64

# Apple Silicon
momentum-scaling all --preset quick --device mps --dtype float32
```

MPS has limited support for `float64`. Therefore, `--dtype auto` selects `float32` on MPS and `float64` on CPU and CUDA. The paper-scale stability scan should be run on a CPU or CUDA GPU with double-precision support.

## Reproducibility Notes

- All random numbers are controlled by `--seed`.
- Deterministic seed offsets are assigned to algorithms, experiment points, and independent repetitions.
- CPU and GPU runs share the same mathematical implementation, although small floating-point differences may remain across hardware, BLAS, and CUDA versions.
- The stability experiment uses fresh Gaussian mini-batches and computes gradients in chunks to limit memory use.
- Following the paper appendix, the data-scaling experiment directly samples curvature-aligned Gaussian gradient noise.
- Each run prints the Python version, PyTorch version, device, and dtype. The same information is stored in the result JSON files.
- The `quick` preset is intended only as a smoke test. Paper-level conclusions should use the `paper` preset and 100 repetitions where specified.
