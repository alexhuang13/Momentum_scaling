"""Plot data-optimal decay rate versus source exponent s (noiseless case).

The formulas are taken from the section on data-optimal scaling in the
strictly noiseless setting (sigma = 0):

    r_SGD(s) = r_Polyak(s) = min(s, 2 - 1/beta),

and

                         2 s beta / (beta + 1),
      0 < s <= (beta - 1)/(2 beta),

    r_Nesterov(s) =       s(3 beta - 1)
                         -------------------------------,
                         s(beta - 1) + beta + 2 - 1/beta
      (beta - 1)/(2 beta) < s <= 2 - 1/beta,

                         2 - 1/beta,
      s > 2 - 1/beta.

The output style follows rate_diagram.py in the same directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "lines.linewidth": 2.2,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (5.2, 3.75),
    "legend.fontsize": 10,
    "legend.handlelength": 2.3,
})

# Match the beta value used by rate_diagram.py. Change this value and rerun
# the script to produce the diagram for another capacity exponent.
BETA = 8.0
S_MAX = 2.5

OUTPUT_PDF = Path(__file__).with_name("noiseless_rate_vs_s.pdf")
OUTPUT_PNG = Path(__file__).with_name("noiseless_rate_vs_s.png")

# Same muted, print-friendly method palette as rate_diagram.py.
BLUE = "#2F5DA8"       # Nesterov
RED = "#D94B3D"        # Polyak
YELLOW = "#E3A018"     # SGD
GUIDE = "#B8B8B8"
AXIS = "#222222"


def critical_s(beta: float) -> tuple[float, float]:
    """Return the two Nesterov transition points in the source exponent."""
    if beta <= 1.0:
        raise ValueError(f"beta must be greater than 1; got beta={beta:g}.")
    s_1 = (beta - 1.0) / (2.0 * beta)
    s_2 = 2.0 - 1.0 / beta
    return s_1, s_2


def rates(s, beta: float = BETA):
    """Evaluate noiseless data-optimal rates for all three algorithms."""
    s = np.asarray(s, dtype=float)
    if np.any(s < 0.0):
        raise ValueError("The source exponent s must be nonnegative.")

    s_1, s_2 = critical_s(beta)
    baseline = np.minimum(s, s_2)

    nesterov = np.piecewise(
        s,
        [s <= s_1, (s > s_1) & (s <= s_2), s > s_2],
        [
            lambda x: 2.0 * beta * x / (beta + 1.0),
            lambda x: x * (3.0 * beta - 1.0)
            / (x * (beta - 1.0) + beta + 2.0 - 1.0 / beta),
            s_2,
        ],
    )
    return baseline, baseline.copy(), nesterov


def make_figure(beta: float = BETA, s_max: float = S_MAX):
    """Create the rate-versus-s comparison figure."""
    s_1, s_2 = critical_s(beta)
    if s_max <= s_2:
        raise ValueError(
            "s_max must exceed 2 - 1/beta so the saturation regime is shown; "
            f"got s_max={s_max:g}, threshold={s_2:g}."
        )

    s = np.linspace(0.0, s_max, 1200)
    sgd, polyak, nesterov = rates(s, beta)

    # Fixed panel rectangle shared by the noisy and noiseless figures.
    # Explicit positioning prevents tick-label differences from changing the
    # physical width or height of the coordinate axes.
    fig = plt.figure(figsize=(5.2, 3.75))
    ax = fig.add_axes([0.15, 0.19, 0.77, 0.70])

    # Color the full lower region, including the lower-right saturation area,
    # and use blue only for the additional Nesterov acceleration region.
    ax.fill_between(
        s, 0.0, sgd,
        color="#FFF1B8", alpha=0.58, linewidth=0, zorder=-2,
    )
    ax.fill_between(
        s, sgd, nesterov,
        where=nesterov >= sgd,
        color="#D8E6F5", alpha=0.68, linewidth=0, zorder=-1,
    )

    # SGD and Polyak have exactly the same noiseless data-optimal rate.
    # Draw both curves with different styles so the overlap is explicit.
    ax.plot(s, sgd, color=YELLOW, linestyle="-", label="SGD", zorder=2)
    ax.plot(
        s, polyak, color=RED, linestyle=(0, (5, 3)),
        label="Polyak", zorder=3,
    )
    ax.plot(s, nesterov, color=BLUE, linestyle="-", label="Nesterov", zorder=4)

    r_1 = 2.0 * beta * s_1 / (beta + 1.0)
    for x_value, y_value in ((s_1, r_1), (s_2, s_2)):
        ax.plot(
            [x_value, x_value], [0.0, y_value],
            color=GUIDE, linestyle=(0, (3, 3)), linewidth=0.9, zorder=1,
        )
    # Horizontal guides connect each symbolic y tick to its critical point.
    ax.plot(
        [0.0, s_1], [r_1, r_1],
        color=GUIDE, linestyle=(0, (3, 3)), linewidth=0.9, zorder=1,
    )
    ax.plot(
        [0.0, s_max], [s_2, s_2],
        color=GUIDE, linestyle=(0, (3, 3)), linewidth=0.9, zorder=1,
    )

    ax.scatter(
        [s_1, s_2], [r_1, s_2], s=28,
        color=BLUE, edgecolor="white", linewidth=0.7, zorder=5,
    )

    # Only show the symbolic transition points; the right endpoint is an
    # arrow tip rather than a numerical tick.
    ax.set_xticks([s_1, s_2])
    ax.set_xticklabels([
        r"$\frac{\beta-1}{2\beta}$",
        r"$2-\frac{1}{\beta}$",
    ])
    ax.set_yticks([r_1, s_2])
    ax.set_yticklabels([
        r"$\frac{\beta-1}{\beta+1}$",
        r"$2-\frac{1}{\beta}$",
    ])

    # Use arrow annotations to label the two colored regions, following the
    # composition of rate_diagram.py.
    blue_x = 0.57 * (s_1 + s_2)
    blue_bottom = np.interp(blue_x, s, sgd)
    blue_top = np.interp(blue_x, s, nesterov)
    ax.annotate(
        "Nesterov > SGD = Polyak",
        xy=(blue_x, blue_bottom + 0.60 * (blue_top - blue_bottom)),
        xytext=(s_2 + 0.43, 0.72 * s_2),
        color=BLUE, fontsize=8.5, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="none", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=0.9,
                        shrinkA=3, shrinkB=2,
                        connectionstyle="arc3,rad=0.22"),
        zorder=7,
    )

    x_right = s_max
    y_top = 1.12 * s_2
    ax.set_xlim(0.0, x_right)
    ax.set_ylim(0.0, y_top)

    # Arrowed coordinate axes and compact labels near the arrow tips.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="major", direction="in", length=3,
                   width=0.8, colors=AXIS, pad=5)
    ax.annotate(
        "", xy=(x_right, 0.0), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color=AXIS, linewidth=1.0,
                        mutation_scale=10, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        "", xy=(0.0, y_top), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color=AXIS, linewidth=1.0,
                        mutation_scale=10, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        r"$s$", xy=(x_right, 0.0), xycoords="data",
        xytext=(6, -1), textcoords="offset points",
        ha="left", va="center", fontsize=13, annotation_clip=False,
    )
    ax.annotate(
        r"$-\log_D \mathcal{E}(D)$", xy=(0.0, y_top), xycoords="data",
        xytext=(5, 1), textcoords="offset points",
        ha="left", va="bottom", fontsize=13, annotation_clip=False,
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.03, 0.97),
              borderaxespad=0.0, handlelength=2.0)
    ax.grid(False)

    return fig


if __name__ == "__main__":
    figure = make_figure()
    figure.savefig(OUTPUT_PDF)
    figure.savefig(OUTPUT_PNG, dpi=240)
    print(f"saved {OUTPUT_PDF}")
    print(f"saved {OUTPUT_PNG}")
