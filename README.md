<div align="center">

# Momentum in Large-Batch Training: Polyak Enlarges the Critical Batch Size, Nesterov Improves Data Efficiency

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2609.02728) [![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/alexhuang13/Momentum_scaling) [![Blog](https://img.shields.io/badge/blog-2563EB?style=for-the-badge&logo=githubpages&logoColor=white)](https://alexhuang13.github.io/momentum_blog/index.html)

[📖 Introduction](#introduction) · [📊 Main Results](#main-results) · [🧪 Experiments](#experiments) · [🚀 Getting Started](#getting-started) · [📚 Citation](#citation)

</div>

---

<a id="introduction"></a>

## 📖 Introduction

Momentum can help large-batch training in two different ways: it can preserve
data efficiency while reducing the number of sequential updates, or it can improve
data efficiency itself. We study **SGD**, **Polyak heavy-ball momentum**, and
**Nesterov momentum** in one-pass power-law kernel (PLK) regression to distinguish
these effects.

The Gaussian feature covariance and target satisfy the source and capacity conditions

$$
\lambda_j \eqsim j^{-q}, \qquad
\lambda_j |\theta_j^*|^2 \eqsim j^{-(1+sq)},
$$

where $q>1$ is the **capacity exponent** and $s>0$ is the **source exponent**.
Smaller $s$ places more target energy in low-curvature directions and makes the
problem harder. Following the paper, $\beta\in[0,1)$ denotes **momentum**,
$\eta$ is the learning rate, $B$ is the batch size, and $N=D/B$ is the number
of updates under data budget $D$. The symbol $\eqsim$ denotes equivalence up
to constant factors; the theory assumes constant-order label noise $\sigma\eqsim1$.

Our analysis follows the paper's sequence: **risk stability → risk dynamics →
optimal data scaling**. We first identify the admissible hyperparameter set
$\mathcal{A}_B$, then characterize the expected excess risk within that set,
and finally optimize the final-step risk subject to $N=D/B$. Polyak enlarges
the critical batch size while preserving the best small-batch exponent;
Nesterov's curvature-adaptive damping improves the exponent in the large-batch regime.

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

Risk stability means that the expected population risk remains bounded for
every admissible initialization with finite population risk:

$$
\sup_{k\geq0}\mathbb{E}[\mathcal{R}(\boldsymbol{\theta}_k)]<\infty.
$$

Theorem 4.3 gives the following critical-learning-rate scalings for sufficiently
large $B$ and sufficiently small $1-\beta$, with constants independent of
$B$ and $\beta$.

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
      <td align="center">η<sup>crit</sup> ≂ 1</td>
    </tr>
    <tr>
      <td align="center">Polyak</td>
      <td align="center">η<sup>crit</sup> ≂ min{1, B(1−β)}</td>
    </tr>
    <tr>
      <td align="center">Nesterov</td>
      <td align="center">η<sup>crit</sup> ≂ min{1, B<sup>q</sup>(1−β)}</td>
    </tr>
  </tbody>
</table>

Nesterov's look-ahead correction induces curvature-adaptive damping. Before
the order-one ceiling is reached, batch size compensates for stronger momentum
as $B$ for Polyak and $B^q$ for Nesterov. Multiplicative-noise feedback governs
stability, while additive label noise sets the noise floor; the critical-rate
scalings are independent of the label-noise scale $\sigma$.

| Stability versus batch size | Stability versus momentum damping $1-\beta$ |
|---|---|
| <a href="stability/results/overview/batch_size_boundary.pdf"><img src="stability/results/overview/batch_size_boundary.png" alt="Critical learning rates versus batch size" width="390"></a> | <a href="stability/results/momentum_gap/momentum_gap.pdf"><img src="stability/results/momentum_gap/momentum_gap.png" alt="Critical learning rates versus momentum damping 1−β" width="390"></a> |

The batch-size and momentum-damping sweeps are the canonical stability results
for this repository. Their PDF versions are linked from the displayed figures.

### 2. Excess-Risk Scaling Laws

Following Section 3.1 of the paper, distinguish the population risk
$\mathcal{R}$ from the excess risk $\mathcal{E}$:

$$
\begin{aligned}
\mathcal{R}(\boldsymbol{\theta})
&=\frac{1}{2}\mathbb{E}_{x,y}\left[(f(x;\boldsymbol{\theta})-y)^2\right],\\
\mathcal{E}(\boldsymbol{\theta})
&=\frac{1}{2}\left\langle\boldsymbol{\theta}-\boldsymbol{\theta}^{*},
\mathbf{H}(\boldsymbol{\theta}-\boldsymbol{\theta}^{*})\right\rangle,\\
\mathcal{R}(\boldsymbol{\theta})
&=\mathcal{E}(\boldsymbol{\theta})+\frac{\sigma^2}{2}.
\end{aligned}
$$

The scaling laws describe $\mathbb{E}[\mathcal{E}(\boldsymbol{\theta}_k)]$,
where the outer expectation averages over the training randomness. Write
$\eta_\beta:=\eta/(1-\beta)$ for the effective learning rate.

For **SGD**, Eq. (13) gives, within the stable region and for $k\eta\gtrsim1$,

$$
\mathbb{E}[\mathcal{E}(\boldsymbol{\theta}_k)]
\eqsim (\eta k)^{-s}+\frac{\sigma^2}{B}\eta.
$$

For **momentum**, Theorems 5.1–5.2 assume $\eta\lesssim1$ and
$k\gtrsim\max\{\eta_\beta^{-1},(1-\beta)^{-1}\}$ and distinguish two stable
damping regimes:

| Regime | Polyak | Nesterov |
|---|---|---|
| Fully overdamped | $1-\beta\gtrsim\sqrt{\eta}$ | $1-\beta\gtrsim\sqrt{\eta}$ |
| Mixed damping | $\eta/B\lesssim1-\beta\lesssim\sqrt{\eta}$ | $\eta/B^q\lesssim1-\beta\lesssim\sqrt{\eta}$ |

In the **fully overdamped regime**, both momentum methods follow the SGD
signal–noise tradeoff with the effective learning rate $\eta_\beta$:

$$
\mathbb{E}[\mathcal{E}(\boldsymbol{\theta}_k)]
\eqsim(\eta_\beta k)^{-s}+\frac{\sigma^2}{B}\eta_\beta.
$$

In the **mixed-damping regime**, the sharp modes also contribute an underdamped
transient:

$$
\begin{aligned}
\text{Polyak:}\qquad
\mathbb{E}[\mathcal{E}(\boldsymbol{\theta}_k)]
&\eqsim \mathbb{T}_{P}(k)+(\eta_\beta k)^{-s}
+\frac{\sigma^2}{B}\eta_\beta,\\
\text{Nesterov:}\qquad
\mathbb{E}[\mathcal{E}(\boldsymbol{\theta}_k)]
&\eqsim \mathbb{T}_{N}(k)+(\eta_\beta k)^{-s}
+\frac{\sigma^2}{B}\min\left\{\eta_\beta,\eta_\beta^{1/q}\right\}.
\end{aligned}
$$

The transient envelopes satisfy

$$
0\leq\mathbb{T}_{P}(k)\lesssim\beta^k,\qquad
0\leq\mathbb{T}_{N}(k)\lesssim\min\{1,(\eta k)^{-s}\}\beta^k.
$$

These terms can be dropped only after the transient has decayed. Polyak's
mixed-regime crossover is
$\tau_\beta\eqsim(1-\beta)^{-1}\log[\eta/(1-\beta)^2]$ (Eq. (15)), so
increasing momentum also prolongs relaxation. Below the respective stability
thresholds, $1-\beta\lesssim\eta/B$ for Polyak and
$1-\beta\lesssim\eta/B^q$ for Nesterov, the dynamics are not risk stable.
All regime boundaries are understood up to constant factors.

Both methods accelerate signal learning in flat modes. Polyak preserves the
SGD signal–noise tradeoff, while Nesterov's look-ahead damping reduces the noise
floor from $\sigma^2\eta_\beta/B$ to $\sigma^2\eta_\beta^{1/q}/B$ when
$\eta_\beta\gg1$. This reduced noise amplification is the mechanism behind
Nesterov's improved data efficiency.

### 3. Data Scaling

Section 6 optimizes the final-step expected excess risk over the admissible
hyperparameters while imposing the one-pass constraint. For $B=D^{b+o(1)}$,
$b\in[0,1)$, and $N=D/B$, define

$$
\inf_{(\eta,\beta)\in\mathcal{A}_B}
\mathbb{E}[\mathcal{E}(\boldsymbol{\theta}_N)]
=D^{-r(b)+o(1)},
$$

with $\beta$ omitted for SGD. A larger $r(b)$ means better data efficiency.
The transition exponents are

$$
b_1=\frac{s}{s+1},\qquad
b_2=\frac{2s+1/q}{2s+1+1/q},\qquad
b_3=\frac{2s+1}{2s+2},\qquad
0<b_1<b_2<b_3<1.
$$

Theorem 6.3 gives the three-regime comparison, with $r_0=s/(s+1)$:

| Batch-size regime | Optimal data-scaling exponents |
|---|---|
| Small: $b\leq b_1$ | $r_{\mathrm{Nesterov}}=r_{\mathrm{Polyak}}=r_{\mathrm{SGD}}=r_0$ |
| Large: $b_1<b<b_3$ | $r_{\mathrm{Nesterov}}>r_{\mathrm{Polyak}}=r_0>r_{\mathrm{SGD}}$ |
| Ultra-large: $b\geq b_3$ | $r_{\mathrm{Nesterov}}=r_{\mathrm{Polyak}}>r_{\mathrm{SGD}}$ |

<details>
<summary>Full batch-size-dependent exponents (Theorem 6.3)</summary>

$$
r_{\mathrm{SGD}}(b)=
\begin{cases}
s/(s+1), & 0\leq b<b_1,\\
s(1-b), & b_1\leq b<1,
\end{cases}
\qquad
r_{\mathrm{Polyak}}(b)=
\begin{cases}
s/(s+1), & 0\leq b<b_3,\\
2s(1-b), & b_3\leq b<1.
\end{cases}
$$

$$
r_{\mathrm{Nesterov}}(b)=
\begin{cases}
s/(s+1), & 0\leq b<b_1,\\
\dfrac{s(1+(q-1)b)}{sq+1}, & b_1\leq b<b_2,\\
2s(1-b), & b_2\leq b<1.
\end{cases}
$$

</details>

Polyak enlarges the critical batch size from $D^{b_1+o(1)}$ to
$D^{b_3+o(1)}$ while retaining the best small-batch rate $D^{-s/(s+1)+o(1)}$.
Nesterov attains its best rate $D^{-2sq/(2sq+q+1)+o(1)}$ at $b=b_2$.
The finite training horizon matters: increasing momentum must still leave time
for the transient to decay within $N=D/B$ updates. This is why stability alone
does not determine the critical batch size.

| Small batch | Large batch | Ultra-large batch |
|---|---|---|
| <a href="data_scaling/results/batch_regimes/s0.3beta4/small_batch.pdf"><img src="data_scaling/results/batch_regimes/s0.3beta4/small_batch.png" alt="Small-batch data scaling" width="270"></a> | <a href="data_scaling/results/batch_regimes/s0.3beta4/large_batch.pdf"><img src="data_scaling/results/batch_regimes/s0.3beta4/large_batch.png" alt="Large-batch data scaling" width="270"></a> | <a href="data_scaling/results/batch_regimes/s0.3beta4/super_large_batch.pdf"><img src="data_scaling/results/batch_regimes/s0.3beta4/super_large_batch.png" alt="Ultra-large-batch data scaling" width="270"></a> |

### Key Numerical Results

The archived kernel experiments are organized into four experiment types:

1. **Stability** scans recover the different dependence on $B$ and $1-\beta$.
2. **Loss-dynamics** runs recover the transient-to-scaling transition and the distinct Polyak and Nesterov noise floors across six $(s,q)$ settings.
3. **Scaling** runs recover the predicted small-, large-, and ultra-large-batch exponents, while a fixed-$D$ sweep shows the shift in the best batch size.
4. **Blowup** runs stress-test the optimizers under an explicitly time-varying momentum schedule.

#### Scaling-law fits across six PLK settings

The $2\times3$ sweep uses $s\in\{0.3,0.5\}$ and $q\in\{2,4,6\}$. Pale curves show the mean population excess risk and solid curves show the fitted theoretical total. The time dependence of the transient, signal, and noise terms is fixed by theory; only nonnegative $O(1)$ coefficients are fitted.

The fits track both the oscillatory transient and the algebraic regime. For $s=0.5$, the late-time curves also expose Nesterov's lower noise floor increasingly clearly as $q$ grows.

#### Time-domain transitions

The saved scans track the transition from underdamped transients to algebraic
decay. The dominant damping timescale is $(1-\beta)^{-1}$, with the logarithmic
factor in $\tau_\beta$ retained above. Nesterov's curvature-adaptive damping
suppresses its transient envelope and lowers the noise floor in the mixed regime.

#### Data scaling across three batch-size regimes

For $(s,q)=(0.3,4)$, all three methods share the same small-batch exponent. Nesterov is fastest in the large-batch regime, while Polyak and Nesterov share the same exponent once the batch is ultra-large. In the large-batch run, the theoretical/measured exponents are $0.1772/0.1787$ for SGD, $0.2308/0.2321$ for Polyak, and $0.3039/0.3074$ for Nesterov.

#### Fixed-budget batch size

At fixed $D=2^{14}$, Nesterov attains the lowest loss and shifts the best batch size upward.

#### Increasing-momentum blowup

<p align="center">
  <a href="blowup/results/blowup_time.pdf"><img src="blowup/results/blowup_time.png" alt="Loss under an increasingly aggressive momentum schedule" width="440"></a>
</p>

Under $\beta_k=1-1/(k+1)$, Polyak eventually becomes unstable, whereas Nesterov remains stable over the saved 500,000-step horizon.

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

The prose and formulas use the latest paper's notation. Historical kernel
code, notebook parameters, filenames, and saved figures may retain `beta` for
the spectral exponent $q$ and `rho` for momentum $\beta$; `K` denotes the update
horizon $N$. These stored identifiers remain unchanged for reproducibility.

### Kernel experiments

| Category | What it measures | Notebook entry points | Reusable code |
|---|---|---|---|
| **Data scaling** | Endpoint loss across data budgets or batch-size regimes | [batch regimes](data_scaling/batch_regimes.ipynb), [critical batch size](data_scaling/critical_batch_size.ipynb) | [`data_scaling/experiment.py`](data_scaling/experiment.py) |
| **Stability** | Largest stable learning rate | [versus batch size](stability/batch_size.ipynb), [versus momentum gap](stability/momentum_gap.ipynb) | [`stability/experiment.py`](stability/experiment.py) |
| **Loss dynamics** | Loss as a function of optimization step in stable runs | [scaling-law fits](loss/scaling_law_fits.ipynb), [phase transitions](loss/phase_transitions.ipynb) | [`loss/experiment.py`](loss/experiment.py) |
| **Blowup** | Failure under the time-varying schedule $\beta_k=1-1/(k+1)$ | [increasing momentum](blowup/increasing_momentum.ipynb) | [`blowup/experiment.py`](blowup/experiment.py) |

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

- `batch_size.ipynb` sweeps $B=2^0,\ldots,2^8$ at fixed $1-\beta=2^{-6}$.
- `momentum_gap.ipynb` sweeps $1-\beta=2^{-6},\ldots,2^{-1.5}$ at fixed $B=4$.
- Both notebooks share $(s,q)=(0.3,1.5)$, the same $\eta$ grid, run length,
  stability rule, and seeds, and intersect at $(B,1-\beta)=(4,2^{-6})$.
- All quantitative points satisfy $N(1-\beta)\geq30$; the notebooks stop with
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

Use loss dynamics for step-resolved curves at fixed $B$, $\eta$, and $\beta$.
These experiments fit and visualize the transient, signal-learning, and
noise-floor terms; they are not data-budget scaling experiments.

See [`loss/README.md`](loss/README.md) for the two notebook entry points and
their saved results.

### 4. Blowup

Blowup is a separate diagnostic because $\beta_k$ changes during training. The
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
  author={Jia-Nan Wang and Zixun Huang and Kairui Li and Zilin Wang and Lei Wu},
  year={2026},
  eprint={2609.02728},
  archivePrefix={arXiv},
  primaryClass={stat.ML},
  url={https://arxiv.org/abs/2609.02728},
}
```
