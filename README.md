<div align="center">

# Momentum in Large-Batch Training: Polyak Enlarges the Critical Batch Size, Nesterov Improves Data Efficiency

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2609.02728) [![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/alexhuang13/Momentum_scaling) [![Blog](https://img.shields.io/badge/blog-2563EB?style=for-the-badge&logo=githubpages&logoColor=white)](https://alexhuang13.github.io/momentum_blog/index.html)

[📖 Introduction](#introduction) · [📊 Main Results](#main-results) · [🧪 Experiments](#experiments) · [🚀 Getting Started](#getting-started) · [📚 Citation](#citation)

</div>

---

<a id="introduction"></a>

## 📖 Introduction

Momentum is a central ingredient in modern optimization, but its interaction with stochastic gradient noise and spectral structure is still not fully understood. Classical theory explains acceleration in deterministic settings, while offering a less complete picture of why Polyak and Nesterov momentum can behave differently in mini-batch training.

In this work, we study **SGD**, **Polyak heavy-ball momentum**, and **Nesterov momentum** in power-law kernel regression with Gaussian features. This model provides a tractable multi-scale learning problem in which the spectrum, target smoothness, gradient noise, and momentum dynamics can be analyzed together. The feature covariance spectrum and target obey

$$
\lambda_j \asymp j^{-\beta}, \qquad
\lambda_j |\theta_j^*|^2 \asymp j^{-(1+s\beta)},
$$

where $\beta>1$ is the capacity exponent and $s>0$ is the source exponent. Smaller $s$ places more target energy in low-curvature directions and therefore corresponds to a harder learning problem.

Our analysis connects mean-square stability with data scaling under a fixed budget $D=BK$, where $B$ is the batch size and $K$ is the number of optimization steps. It shows that Polyak momentum mainly enlarges the useful batch-size range, whereas Nesterov momentum can improve the optimal data-scaling exponent through curvature-dependent damping of stochastic noise.

<p align="center">
  <a href="data_scaling/results/overview/intro_figure.pdf"><img src="data_scaling/results/overview/intro_figure.png" alt="Theory and simulation of data-scaling regimes" width="800"></a>
</p>

This repository contains self-contained PyTorch implementations of the numerical
experiments. The notebooks share the same update rules and automatically select
CUDA, Apple Silicon, or CPU.

Neural-network experiments complement the kernel analysis: **ImageNet / ResNet-18**
measures performance across batch sizes at a fixed sample budget, and
**CIFAR-10 / ResNet-18** measures learning-rate stability at a fixed update horizon.
Their empirical protocols and saved results are organized alongside the
corresponding kernel experiments below.

<a id="main-results"></a>

## 📊 Main Results

### 1. Stochastic Stability Boundaries

<table align="center">
  <thead>
    <tr>
      <th align="center">Method</th>
      <th align="center">Critical learning rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">SGD</td>
      <td align="center">η<sup>crit</sup> ≍ 1</td>
    </tr>
    <tr>
      <td align="center">Polyak</td>
      <td align="center">η<sup>crit</sup> ≍ min{1, B(1−ρ)}</td>
    </tr>
    <tr>
      <td align="center">Nesterov</td>
      <td align="center">η<sup>crit</sup> ≍ min{1, B<sup>β</sup>(1−ρ)}</td>
    </tr>
  </tbody>
</table>

Nesterov's look-ahead correction makes its stable region grow faster with batch size, allowing it to reach the deterministic stability ceiling earlier than Polyak momentum.

| Stability versus batch size | Stability versus momentum damping $1-\beta$ |
|---|---|
| <a href="stability/results/overview/batch_size_boundary.pdf"><img src="stability/results/overview/batch_size_boundary.png" alt="Critical learning rates versus batch size" width="390"></a> | <a href="stability/results/momentum_gap/momentum_gap.pdf"><img src="stability/results/momentum_gap/momentum_gap.png" alt="Critical learning rates versus momentum damping 1−β" width="390"></a> |

The batch-size and momentum-damping sweeps are the canonical stability results
for this repository. Their PDF versions are linked from the displayed figures.

### 2. Excess-Risk Scaling Laws

The expected excess risk admits a signal-learning and noise-accumulation decomposition:

$$
\mathbb{E}[\mathcal E(\theta_k)]
\asymp
\mathcal D(k)
+
\frac{\eta^2}{B}
\sum_{m=0}^{k-1}
\mathcal K(k-1-m)
(\mathbb{E}[\mathcal E(\theta_m)]+\sigma^2),
$$

where $\mathcal D(k)$ describes how the target signal is learned and $\mathcal K(\tau)$ describes how long stochastic noise remains in the dynamics. In the noisy setting, the leading long-time terms are

$$
\begin{aligned}
\text{SGD:}\qquad
\mathbb{E}[\mathcal E(\theta_k)]
&\asymp
(\eta k)^{-s}
+
\frac{\eta\sigma^2}{B},\\
\text{Polyak:}\qquad
\mathbb{E}[\mathcal E(\theta_k)]
&\asymp
(\frac{1-\rho}{\eta k})^s
+
\frac{\eta\sigma^2}{B(1-\rho)},\\
\text{Nesterov:}\qquad
\mathbb{E}[\mathcal E(\theta_k)]
&\asymp
(\frac{1-\rho}{\eta k})^s
+
\frac{\sigma^2}{B}
\min\{
\frac{\eta}{1-\rho},
(\frac{\eta}{1-\rho})^{1/\beta}
\},
\end{aligned}
$$

up to exponentially decaying transient and lower-order multiplicative-noise terms. Both momentum methods accelerate signal learning through the effective learning rate $\eta/(1-\rho)$. Polyak momentum, however, amplifies label noise by the same factor, whereas Nesterov's curvature-dependent damping clips noise accumulation in sharper spectral directions. This difference is the mechanism behind Nesterov's improved data-scaling exponent.

### 3. Data Scaling

In the presence of label noise:

- SGD achieves the optimal data-scaling rate $D^{-s/(s+1)}$.
- Polyak momentum achieves the same optimal rate but preserves it over a substantially larger range of batch sizes.
- Nesterov momentum achieves the faster rate $D^{-2s\beta/(2s\beta+\beta+1)+o(1)}$.

| Small batch | Large batch | Ultra-large batch |
|---|---|---|
| <a href="data_scaling/results/batch_regimes/s0.3beta4/small_batch.pdf"><img src="data_scaling/results/batch_regimes/s0.3beta4/small_batch.png" alt="Small-batch data scaling" width="270"></a> | <a href="data_scaling/results/batch_regimes/s0.3beta4/large_batch.pdf"><img src="data_scaling/results/batch_regimes/s0.3beta4/large_batch.png" alt="Large-batch data scaling" width="270"></a> | <a href="data_scaling/results/batch_regimes/s0.3beta4/super_large_batch.pdf"><img src="data_scaling/results/batch_regimes/s0.3beta4/super_large_batch.png" alt="Ultra-large-batch data scaling" width="270"></a> |

### Key Numerical Results

The numerical evidence is organized into four distinct experiment types:

1. **Stability** scans recover the different dependence on $B$ and $1-\rho$.
2. **Loss-dynamics** runs recover the transient-to-scaling transition and the distinct Polyak and Nesterov noise floors across six $(s,\beta)$ settings.
3. **Scaling** runs recover the predicted small-, large-, and ultra-large-batch exponents, while a fixed-$D$ sweep shows the shift in the best batch size.
4. **Blowup** runs stress-test the optimizers under an explicitly time-varying momentum schedule.

#### Scaling-law fits across six PLK settings

The $2\times3$ sweep uses $s\in\{0.3,0.5\}$ and $\beta\in\{2,4,6\}$. Pale curves show the empirical population loss and solid curves show the fitted theoretical total. The time dependence of the transient, signal, and noise terms is fixed by theory; only nonnegative $O(1)$ coefficients are fitted.

The fits track both the oscillatory transient and the algebraic regime. For $s=0.5$, the late-time curves also expose Nesterov's lower noise floor increasingly clearly as $\beta$ grows.

#### Time-domain transitions

The transition from the oscillatory transient to algebraic scaling occurs at $k_{\rm tr}\propto(1-\rho)^{-1}$. Nesterov enters the scaling regime slightly earlier and reaches its lower noise-dominated regime later than Polyak.

#### Data scaling across three batch-size regimes

For $(s,\beta)=(0.3,4)$, all three methods share the same small-batch exponent. Nesterov is fastest in the large-batch regime, while Polyak and Nesterov share the same exponent once the batch is ultra-large. In the large-batch run, the theoretical/measured exponents are $0.1772/0.1787$ for SGD, $0.2308/0.2321$ for Polyak, and $0.3039/0.3074$ for Nesterov.

#### Fixed-budget batch size

At fixed $D=2^{14}$, Nesterov attains the lowest loss and shifts the best batch size upward.

#### Increasing-momentum blowup

<p align="center">
  <a href="blowup/results/blowup_time.pdf"><img src="blowup/results/blowup_time.png" alt="Loss under an increasingly aggressive momentum schedule" width="440"></a>
</p>

Under $\rho_k=1-1/(k+1)$, Polyak eventually becomes unstable, whereas Nesterov remains stable over the saved 500,000-step horizon.

### Neural-Network Experiments

Both experiments compare SGD, Polyak, and Nesterov on ResNet-18 with GroupNorm.
They use separate finite-budget protocols; the kernel scaling laws above are
not assumed to hold quantitatively for the neural networks.

| Experiment | Protocol and saved data | Results |
|---|---|---|
| **ImageNet: fixed-budget batch-size scaling** | 3,843,501 training samples; 396 configurations at seed 0, with per-run summaries and metrics | [Figure (PNG)](data_scaling/imagenet/results/paper_3epochs/fixed_D_optimal_loss_vs_batch.png) · [PDF](data_scaling/imagenet/results/paper_3epochs/fixed_D_optimal_loss_vs_batch.pdf) · [Best configurations](data_scaling/imagenet/results/paper_3epochs/best_by_batch.csv) |
| **CIFAR-10: stability boundaries** | 2,000 updates; 1,309 aggregate cells, boundary and fit tables, and follow-up audits | [Figure (PDF)](stability/cifar10/results/paper_2000steps/cifar_stability_merged.pdf) · [Seed coverage](stability/cifar10/results/paper_2000steps/seed_coverage.pdf) · [Aggregate counts](stability/cifar10/results/paper_2000steps/merged_cells.csv) |

ImageNet results are selected by validation loss within the recorded search and
come from one seed. CIFAR-10 raw trial histories are not included, so its saved
figures can be reconstructed from aggregates without repeating a raw-trial audit.
The experiment READMEs document selection, stability criteria, and reproduction limits.

<a id="experiments"></a>

## 🧪 Experiments

The repository uses four scientific categories. Each category contains its
kernel notebooks, reusable code, explanation, and saved results. Neural-network
experiments live within the category for the scientific question they address.

### Kernel experiments

| Category | What it measures | Notebook entry points | Reusable code |
|---|---|---|---|
| **Data scaling** | Endpoint loss across data budgets or batch-size regimes | [batch regimes](data_scaling/batch_regimes.ipynb), [critical batch size](data_scaling/critical_batch_size.ipynb) | [`data_scaling/experiment.py`](data_scaling/experiment.py) |
| **Stability** | Largest stable learning rate | [versus batch size](stability/batch_size.ipynb), [versus momentum gap](stability/momentum_gap.ipynb) | [`stability/experiment.py`](stability/experiment.py) |
| **Loss dynamics** | Loss as a function of optimization step in stable runs | [scaling-law fits](loss/scaling_law_fits.ipynb), [phase transitions](loss/phase_transitions.ipynb) | [`loss/experiment.py`](loss/experiment.py) |
| **Blowup** | Failure under the time-varying schedule $\rho_k=1-1/(k+1)$ | [increasing momentum](blowup/increasing_momentum.ipynb) | [`blowup/experiment.py`](blowup/experiment.py) |

Each folder has its own README. Committed numerical arrays and figures live
under that folder's `results/` directory; new notebook runs write to its ignored
`results/generated/` directory.

### Neural-network experiments

| Scientific question | Experiment guide | Entry points |
|---|---|---|
| Fixed-budget loss versus batch size | [ImageNet / ResNet-18](data_scaling/imagenet/README.md) | [Train](data_scaling/imagenet/train.py) · [Sweep](data_scaling/imagenet/sweep.py) · [Plot](data_scaling/imagenet/plot.py) |
| Learning-rate stability versus batch size and momentum damping | [CIFAR-10 / ResNet-18](stability/cifar10/README.md) | [Train](stability/cifar10/train.py) · [Sweep](stability/cifar10/sweep.py) · [Plot](stability/cifar10/plot.py) |
| Earlier fixed-horizon ImageNet stability protocol | [ImageNet stability notebook](momentum_nn/README.md) | [Interactive notebook](momentum_nn/stability.ipynb) |

### 1. Scaling

Use scaling experiments when the horizontal axis is data budget $D$ or batch
size $B$ and the quantity of interest is final or tail-averaged loss.

- `batch_regimes.ipynb` compares the small-, large-, and ultra-large-batch exponents.
- `critical_batch_size.ipynb` searches for the best $B$ at fixed $D=2^{14}$.
- [`data_scaling/README.md`](data_scaling/README.md) explains the saved settings and results.
- [`data_scaling/imagenet/`](data_scaling/imagenet/README.md) contains the neural-network
  fixed-budget experiment, the complete recorded grid, and its saved results.

### 2. Stability

Use stability experiments when the output is a largest stable learning rate.
For the kernel model:

- `batch_size.ipynb` sweeps $B=2^0,\ldots,2^8$ at fixed $1-\rho=2^{-6}$.
- `momentum_gap.ipynb` sweeps $1-\rho=2^{-6},\ldots,2^{-1.5}$ at fixed $B=4$.
- Both notebooks share $(s,\beta)=(0.3,1.5)$, the same $\eta$ grid, run length,
  stability rule, and seeds, and intersect at $(B,1-\rho)=(4,2^{-6})$.
- All quantitative points satisfy $K(1-\rho)\geq30$; the notebooks stop with
  an error if a future setting violates this finite-horizon guard.

The [CIFAR-10 experiment](stability/cifar10/README.md) scans batch size and momentum
damping at a fixed 2,000-update horizon. It includes the original and revised
trainers, aggregate result tables, empirical boundary fits, and plotting tools.

The earlier [ImageNet stability experiment](momentum_nn/README.md) remains
available with its own finite-time criteria, per-seed results, and censored
learning-rate boundaries. Its [interactive notebook](momentum_nn/stability.ipynb)
includes a FakeData smoke test, ImageNet pilot and full presets, resumable trials,
and inline diagnostics. The historical `autodl_stability/` folder referenced by
earlier instructions is not included in this checkout; use the linked notebook
and experiment guides for the available entry points.

### 3. Loss dynamics

Use loss dynamics for step-resolved curves at fixed $B$, $\eta$, and $\rho$.
These experiments fit and visualize the transient, signal-learning, and
noise-floor terms; they are not data-budget scaling experiments.

See [`loss/README.md`](loss/README.md) for the two notebook entry points and
their saved results.

### 4. Blowup

Blowup is a separate diagnostic because $\rho_k$ changes during training. The
notebook compares SGD, Polyak, and Nesterov under the same increasing momentum
schedule. See [`blowup/README.md`](blowup/README.md).

<a id="getting-started"></a>

## 🚀 Getting Started

### Repository Structure

| Path | Contents |
|---|---|
| [`data_scaling/`](data_scaling/README.md) | Kernel data-budget and batch-size experiments |
| [`data_scaling/imagenet/`](data_scaling/imagenet/README.md) | ImageNet fixed-budget batch-size experiment |
| [`stability/`](stability/README.md) | Kernel learning-rate stability experiments |
| [`stability/cifar10/`](stability/cifar10/README.md) | CIFAR-10 finite-horizon stability experiment |
| [`loss/`](loss/README.md) | Kernel loss dynamics and scaling-law fits |
| [`blowup/`](blowup/README.md) | Increasing-momentum diagnostic |
| [`momentum_nn/`](momentum_nn/README.md) | Earlier ImageNet stability notebook and implementation |
| [`docs/`](docs/) | Supplementary theory and experiment validation notes |
| [`tests/`](tests/) | Neural-network import, configuration, aggregate, and plotting checks |
| [`requirements.txt`](requirements.txt), [`pyproject.toml`](pyproject.toml) | Dependencies and package metadata |

Inside each kernel experiment folder, `README.md` explains the scientific question,
`*.ipynb` files are the readable experiment entry points, `experiment.py` and
`core.py` contain reusable code, and `results/` contains committed outputs.
The small amount of repeated core code keeps every experiment independently
readable without tracing imports through another directory.

The ImageNet and CIFAR-10 modules share a consistent reading order:

| Path | Purpose |
|---|---|
| `README.md` | Scientific protocol, quick start, and available data |
| `train.py`, `sweep.py`, `plot.py` | Single-run training, grid preparation/execution, and figure reconstruction |
| `source/` | Original implementations and historical tools, with a reading guide |
| `results/` | Committed result tables and original figures |
| `outputs/` | New runs and generated figures, ignored by Git |
| `docs/` | Original instructions and import checksums |

### Installation

Python 3.10 or later is required. Creating an isolated environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Launch Jupyter from the repository root so the relative result paths resolve
correctly. For NVIDIA GPUs, install the PyTorch build matching the local CUDA
version before launching the notebooks.

For neural-network training, install a compatible PyTorch/torchvision build and
the additional dependencies from the repository root:

```bash
python -m pip install -e '.[nn]'
```

### Redrawing saved neural-network results

Redrawing requires only NumPy, Matplotlib, and pandas; no GPU, PyTorch, or dataset
download is needed. From the repository root:

```bash
python -m pip install numpy matplotlib pandas
python data_scaling/imagenet/plot.py
python stability/cifar10/plot.py
```

Figures are written to the corresponding module's `outputs/` directory.
Committed tables and original figures remain in `results/`.

### Running the notebooks

Open any notebook in the four experiment folders and run it from top to bottom.
Where a `PAPER_RUN` switch is available, `False` checks the code path quickly
and `True` runs the quantitative setting. Generated JSON, NPZ, PNG, and PDF
files stay inside that experiment folder. Stability caches stay under
`results/generated/`, while the two validated figures overwrite the canonical
paths used above.

### Running neural-network experiments

Prepare both configuration grids without starting training:

```bash
python data_scaling/imagenet/sweep.py
python stability/cifar10/sweep.py
```

Each command writes a manifest under `outputs/`; add `--execute` to train.
Set `IMAGENET_ROOT` or `CIFAR10_ROOT` to the corresponding dataset directory.
CIFAR-10 defaults to `DEVICES=cuda:0,cuda:1`; use `--devices cuda:0` for one GPU.
Read the [ImageNet guide](data_scaling/imagenet/README.md) or
[CIFAR-10 guide](stability/cifar10/README.md) for single-run examples, full-grid
settings, and resumption behavior.

### CPU, CUDA, and Apple Silicon

The notebooks automatically select CUDA, then Apple MPS, then CPU. MPS uses
`float32`; CPU and CUDA use `float64`. Paper-scale stability scans are best run
on a CPU or CUDA GPU with double-precision support.

These automatic device and precision choices apply to the kernel notebooks.
Neural-network training has its own device and precision settings documented in
each experiment guide.

### Reproducibility Notes

For the kernel notebooks:

- All random numbers are controlled by explicit notebook seeds.
- Deterministic seed offsets are assigned to algorithms, experiment points, and independent repetitions.
- CPU and GPU runs share the same mathematical implementation, although small floating-point differences may remain across hardware, BLAS, and CUDA versions.
- The stability experiment uses fresh Gaussian mini-batches and computes gradients in chunks to limit memory use.
- Following the paper appendix, the data-scaling experiment directly samples curvature-aligned Gaussian gradient noise.
- Each run records its configuration, device, dtype, grids, and numerical summaries beside the generated figures.
- Quick settings are smoke tests only; quantitative conclusions should use the notebook's paper-scale setting.

For the neural-network experiments:

- Original training implementations and imported results are preserved byte-for-byte,
  with provenance and checksums in each module's `docs/` directory.
- ImageNet's grid contains all 396 recorded configurations. A fresh run reproduces
  the protocol and configuration, without a guarantee of bitwise-identical results.
- The fresh CIFAR-10 grid uses the revised record-mode protocol consistently;
  it does not reconstruct the historical adaptive reruns or their varying seed counts.
- CIFAR-10 plotting uses the supplied aggregates. Raw trial manifests and probe
  histories are not available in this release.

### Validation

```bash
python -m unittest discover -s tests -v
```

The CPU-only checks cover original-file integrity, configuration generation,
archived aggregates, and plotting. They do not rerun neural-network training.
See the [validation notes](docs/neural_experiments_validation.md).

<a id="citation"></a>

## 📚 Citation

If you use this code or build on these experiments, please cite:

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

