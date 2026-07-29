"""Plot globally data-optimal decay rate versus s in the noisy setting.

From the section "Data and Batch Size Scaling" (sigma > 0):

    r_SGD(s) = r_Polyak(s) = s / (s + 1),
    r_Nesterov(s) = 2 s beta / (2 s beta + beta + 1).

The visual style follows rate_diagram.py and noiseless_rate_vs_s.py.
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

BETA = 8.0
S_MAX = 2.5

OUTPUT_PDF = Path(__file__).with_name("noisy_rate_vs_s.pdf")
OUTPUT_PNG = Path(__file__).with_name("noisy_rate_vs_s.png")

BLUE = "#2F5DA8"
RED = "#D94B3D"
YELLOW = "#E3A018"
GUIDE = "#B8B8B8"
AXIS = "#222222"


def rates(s, beta: float = BETA):
    """Evaluate the three noisy globally data-optimal decay rates."""
    if beta <= 1.0:
        raise ValueError(f"beta must be greater than 1; got beta={beta:g}.")
    s = np.asarray(s, dtype=float)
    if np.any(s < 0.0):
        raise ValueError("The source exponent s must be nonnegative.")

    baseline = s / (s + 1.0)
    nesterov = 2.0 * s * beta / (2.0 * s * beta + beta + 1.0)
    return baseline, baseline.copy(), nesterov


def make_figure(beta: float = BETA, s_max: float = S_MAX):
    """Create the noisy data-optimal rate-versus-s comparison figure."""
    if s_max <= 0.0:
        raise ValueError(f"s_max must be positive; got {s_max:g}.")

    s = np.linspace(0.0, s_max, 1400)
    sgd, polyak, nesterov = rates(s, beta)

    # Fixed panel rectangle shared by the noisy and noiseless figures.
    # Explicit positioning prevents tick-label differences from changing the
    # physical width or height of the coordinate axes.
    fig = plt.figure(figsize=(5.2, 3.75))
    ax = fig.add_axes([0.15, 0.19, 0.77, 0.70])

    # Fill the common achievable region and the extra Nesterov gain.
    ax.fill_between(
        s, 0.0, sgd,
        color="#FFF1B8", alpha=0.58, linewidth=0, zorder=-2,
    )
    ax.fill_between(
        s, sgd, nesterov,
        color="#D8E6F5", alpha=0.68, linewidth=0, zorder=-1,
    )

    ax.plot(s, sgd, color=YELLOW, linestyle="-", label="SGD", zorder=2)
    ax.plot(
        s, polyak, color=RED, linestyle=(0, (5, 3)),
        label="Polyak", zorder=3,
    )
    ax.plot(s, nesterov, color=BLUE, linestyle="-", label="Nesterov", zorder=4)

    # All three rates approach one as s tends to infinity.
    ax.plot(
        [0.0, s_max], [1.0, 1.0],
        color=GUIDE, linestyle=(0, (3, 3)), linewidth=0.9, zorder=1,
    )

    # Arrow annotation for the acceleration region.
    target_x = 0.42 * s_max
    target_bottom = np.interp(target_x, s, sgd)
    target_top = np.interp(target_x, s, nesterov)
    ax.annotate(
        "Nesterov > SGD = Polyak",
        xy=(target_x, target_bottom + 0.58 * (target_top - target_bottom)),
        xytext=(0.91 * s_max, 0.49),
        color=BLUE, fontsize=8.5, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="none", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=0.9,
                        shrinkA=3, shrinkB=2,
                        connectionstyle="arc3,rad=0.22"),
        zorder=7,
    )

    # Keep only the meaningful symbolic y tick; the x-axis ends in an arrow.
    ax.set_xticks([])
    ax.set_yticks([1.0])
    ax.set_yticklabels([r"$1$"])

    x_right = s_max
    y_top = 1.10
    ax.set_xlim(0.0, x_right)
    ax.set_ylim(0.0, y_top)

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
