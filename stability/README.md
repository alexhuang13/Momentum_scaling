# Stability

This folder estimates the largest stable learning rate from bounded/divergent
training runs.

| File | What to read or run |
|---|---|
| [`batch_size.ipynb`](batch_size.ipynb) | Sweep $B=2^0,\ldots,2^8$ at fixed $1-\rho=2^{-6}$ |
| [`momentum_gap.ipynb`](momentum_gap.ipynb) | Sweep $1-\rho=2^{-6},\ldots,2^{-1.5}$ at fixed $B=4$ |
| [`experiment.py`](experiment.py) | Reusable stability scan implementation |
| [`core.py`](core.py) | Model, optimizer updates, and shared numerical utilities |

## Results

| Stability versus batch size | Stability versus momentum damping |
|---|---|
| <a href="results/overview/batch_size_boundary.pdf"><img src="results/overview/batch_size_boundary.png" alt="Critical learning rates versus batch size" width="390"></a> | <a href="results/momentum_gap/momentum_gap.pdf"><img src="results/momentum_gap/momentum_gap.png" alt="Critical learning rates versus momentum damping" width="390"></a> |

The two notebooks use the same $(s,\beta)=(0.3,1.5)$ problem, dimension,
$\eta$ grid, 2,000-step horizon, stability rule, five seeds, and device policy.
Their grids intersect at $(B,1-\rho)=(4,2^{-6})$, where the seed mapping is
also identical. This gives a direct numerical consistency check between the
two scans.

Every quantitative point satisfies

$$
K(1-\rho)\geq 2000\,2^{-6}=31.25.
$$

This guard is essential. If $K(1-\rho)\ll1$, a run only measures whether the
optimizer survives for $K$ steps. The apparent boundaries then scale as
$B/K$ for Polyak and $B^\beta/K$ for Nesterov. Fitting those data to the
asymptotic formulas creates a spurious prefactor of order
$1/[K(1-\rho)]$.

Both notebooks fit the same one-kink model after normalizing the horizontal
variable:

$$
\eta_{\rm crit}=\min\{L,cx\},\qquad
x_{\rm Polyak}=B(1-\rho),\quad
x_{\rm Nesterov}=B^\beta(1-\rho).
$$

The fitted, dimensionless $c$ is therefore comparable across notebooks. Raw
slopes against $B$ or $1-\rho$ depend on the fixed value of the other variable
and are not compared as theoretical constants. More explicitly, if $g_0$ and
$B_0$ denote the fixed values in the two scans, then

$$
c_{B,\rm Polyak}=cg_0,\quad c_{g,\rm Polyak}=cB_0,
\qquad
c_{B^\beta,\rm Nesterov}=cg_0,\quad
c_{g,\rm Nesterov}=cB_0^\beta.
$$

The normalized $c$, not either raw slope, is the cross-experiment quantity.

Both notebooks are safe to resume on a preemptible GPU instance. The batch
scan checkpoints after every `(algorithm, B)` block, while the momentum-gap
scan checkpoints after every `(algorithm, 1-rho)` block. Checkpoint replacement
is atomic, and rerunning the notebook skips completed blocks.

New caches write to `results/generated/aligned/`. Running the notebooks writes
the canonical figures to `results/overview/` and `results/momentum_gap/`.

Main result: the scans recover
$\eta^{\rm crit}_{\rm Polyak}\asymp\min\{1,B(1-\rho)\}$ and
$\eta^{\rm crit}_{\rm Nesterov}\asymp\min\{1,B^\beta(1-\rho)\}$.
