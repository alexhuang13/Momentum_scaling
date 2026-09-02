# Stability

This folder estimates the largest stable learning rate from bounded/divergent
training runs.

| File | What to read or run |
|---|---|
| [`batch_size.ipynb`](batch_size.ipynb) | Sweep $B=2^2,\ldots,2^{10}$ at fixed $1-\rho=2^{-8}$ |
| [`momentum_gap.ipynb`](momentum_gap.ipynb) | Sweep $1-\rho=2^{-8},\ldots,2^{-3.5}$ at fixed $B=16$ |
| [`experiment.py`](experiment.py) | Reusable stability scan implementation |
| [`core.py`](core.py) | Model, optimizer updates, and shared numerical utilities |
| [`plot_reference.py`](plot_reference.py) | Render the eight saved reference grids |
| [`plot_schematic.py`](plot_schematic.py) | Recreate the theoretical boundary schematic |

The two notebooks now use the same $(s,\beta)=(0.3,1.5)$ problem, dimension,
$\eta$ grid, 2,000-step horizon, stability rule, five seeds, and device policy.
Their grids intersect at $(B,1-\rho)=(16,2^{-8})$, where the seed mapping is
also identical. This gives a direct numerical consistency check between the
two scans.

Both notebooks fit the same one-kink model after normalizing the horizontal
variable:

$$
\eta_{\rm crit}=\min\{L,cx\},\qquad
x_{\rm Polyak}=B(1-\rho),\quad
x_{\rm Nesterov}=B^\beta(1-\rho).
$$

The fitted $c$ is therefore comparable across notebooks. A coefficient fitted
directly against $1-\rho$ is larger by $B$ for Polyak and by $B^\beta$ for
Nesterov; it is not a different theoretical constant. More explicitly, if
$g_0=2^{-8}$ and $B_0=16$, the unnormalized slopes obey

$$
c_{B,\rm Polyak}=cg_0,\quad c_{g,\rm Polyak}=cB_0,
\qquad
c_{B^\beta,\rm Nesterov}=cg_0,\quad
c_{g,\rm Nesterov}=cB_0^\beta.
$$

So the two raw slopes are expected to differ by $B_0/g_0=4096$ for Polyak
and by $B_0^\beta/g_0=16384$ for Nesterov. The normalized $c$, not either raw
slope, is the cross-experiment quantity.

Saved paper-scale outputs and older reference arrays are under
[`results/`](results/). They predate this cross-design and should not be used
to compare the two fitted constants directly. New aligned runs write to
`stability/results/generated/aligned/`, which is ignored by Git.

Main result: the scans recover
$\eta^{\rm crit}_{\rm Polyak}\asymp\min\{1,B(1-\rho)\}$ and
$\eta^{\rm crit}_{\rm Nesterov}\asymp\min\{1,B^\beta(1-\rho)\}$.
