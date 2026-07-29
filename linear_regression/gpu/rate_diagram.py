"""Reproduce Fig.png with symbolic rate ticks and renamed axes."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (5, 3),
    "legend.fontsize": 10,
    "legend.handlelength": 1,
})

# Numerical values only determine the locations of the symbolic ticks.
S = 0.3
BETA = 8

OUTPUT_PDF = Path(__file__).with_name("rate_diagram.pdf")
OUTPUT_PNG = Path(__file__).with_name("rate_diagram.png")

# Muted, print-friendly palette inspired by Nature-style scientific figures.
# Red-yellow-blue method palette; purple marks the Polyak/Nesterov overlap.
BLUE = "#2F5DA8"
RED = "#D94B3D"
YELLOW = "#E3A018"
OVERLAP = "#7A4E9D"
GUIDE = "#B8B8B8"
AXIS = "#222222"


def critical_points(s, beta):
    a_0 = s / (s + 1.0)
    a_n = (2.0 * s * beta + 1.0) / (2.0 * s * beta + beta + 1.0)
    a_h = (2.0 * s + 1.0) / (2.0 * s + 2.0)
    r_0 = s / (s + 1.0)
    r_n = 2.0 * s * beta / (2.0 * s * beta + beta + 1.0)
    return a_0, a_n, a_h, r_0, r_n


def rates(a, s=S, beta=BETA):
    """Evaluate SGD, Polyak heavy-ball, and Nesterov rate functions."""
    a = np.asarray(a, dtype=float)
    a_0, a_n, _, r_0, _ = critical_points(s, beta)
    sgd = np.minimum(r_0, s * (1.0 - a))
    polyak = np.minimum(r_0, 2.0 * s * (1.0 - a))
    nesterov = np.piecewise(
        a,
        [a <= a_0, (a > a_0) & (a <= a_n), a > a_n],
        [
            r_0,
            lambda x: s * (1.0 + (beta - 1.0) * x) / (beta * s + 1.0),
            lambda x: 2.0 * s * (1.0 - x),
        ],
    )
    return sgd, polyak, nesterov


def make_figure(s=S, beta=BETA):
    # Use the true numerical coordinates implied by s and beta.
    a_0, a_n, a_h, r_0, r_n = critical_points(s, beta)
    if not (0.0 < a_0 < a_n < a_h < 1.0):
        raise ValueError(
            "The figure requires 0 < s/(s+1) < a_N < a_H < 1; "
            f"got s={s:g}, beta={beta:g}."
        )

    fig, ax = plt.subplots()

    # Match the original composition: leave room before the first breakpoint.
    x_left = 0.0
    x_start = 0.0
    x_right = 1.08
    y_top = max(r_n * 1.27, r_0 * 1.45)

    # Soft pastel fills between the three rate boundaries.
    x_fill = np.linspace(x_start, 1.0, 800)
    sgd_fill = np.where(
        x_fill <= a_0,
        r_0,
        r_0 * (1.0 - x_fill) / (1.0 - a_0),
    )
    polyak_fill = np.where(
        x_fill <= a_h,
        r_0,
        r_0 * (1.0 - x_fill) / (1.0 - a_h),
    )
    nesterov_fill = np.piecewise(
        x_fill,
        [x_fill <= a_0, (x_fill > a_0) & (x_fill <= a_n),
         (x_fill > a_n) & (x_fill <= a_h), x_fill > a_h],
        [
            r_0,
            lambda x: r_0 + (r_n - r_0) * (x - a_0) / (a_n - a_0),
            lambda x: r_n + (r_0 - r_n) * (x - a_n) / (a_h - a_n),
            lambda x: r_0 * (1.0 - x) / (1.0 - a_h),
        ],
    )

    ax.fill_between(x_fill, 0.0, sgd_fill, color="#FFF1B8", alpha=0.55,
                    linewidth=0, zorder=-4)
    ax.fill_between(x_fill, sgd_fill, polyak_fill, color="#FAD4D0", alpha=0.55,
                    linewidth=0, zorder=-3)
    ax.fill_between(x_fill, polyak_fill, nesterov_fill, color="#D8E6F5", alpha=0.62,
                    linewidth=0, zorder=-2)

    # Arrow annotations keep the text out of the narrow colored regions.
    blue_target_x = a_n + 0.18 * (a_h - a_n)
    blue_target_top = r_n + (r_0 - r_n) * (blue_target_x - a_n) / (a_h - a_n)
    blue_target_y = r_0 + 0.62 * (blue_target_top - r_0)
    ax.annotate(
        "Nesterov > Polyak",
        xy=(blue_target_x, blue_target_y), xycoords="data",
        xytext=(a_h + 0.20, r_n * 1.16), textcoords="data",
        color=BLUE, fontsize=8.5, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="none", alpha=0.90),
        arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=0.9,
                        shrinkA=3, shrinkB=2,
                        connectionstyle="arc3,rad=0.28"),
        zorder=7,
    )

    red_target_x = a_n + 0.58 * (a_h - a_n)
    red_target_bottom = r_0 * (1.0 - red_target_x) / (1.0 - a_0)
    red_target_y = red_target_bottom + 0.52 * (r_0 - red_target_bottom)
    ax.annotate(
        "Polyak > SGD",
        xy=(red_target_x, red_target_y), xycoords="data",
        xytext=(46, 26), textcoords="offset points",
        color=RED, fontsize=8.5, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="none", alpha=0.90),
        arrowprops=dict(arrowstyle="->", color=RED, linewidth=0.9,
                        shrinkA=3, shrinkB=2,
                        connectionstyle="arc3,rad=0.18"),
        zorder=7,
    )

    # Gray reference guides from the original figure.
    # Extend each y-axis tick horizontally to its corresponding critical point.
    ax.plot([x_left, a_h], [r_0, r_0], color=GUIDE,
            linestyle=(0, (3, 3)), linewidth=0.8, zorder=0)
    ax.plot([x_left, a_n], [r_n, r_n], color=GUIDE,
            linestyle=(0, (3, 3)), linewidth=0.8, zorder=0)

    # Drop a vertical dashed guide from every nonzero critical point to the x-axis.
    for x_value, y_value in ((a_0, r_0), (a_n, r_n), (a_h, r_0)):
        ax.plot([x_value, x_value], [0.0, y_value], color=GUIDE,
                linestyle=(0, (3, 3)), linewidth=0.8, zorder=0)

    # Common initial plateau. Blue is drawn last, as in the source figure.
    ax.plot([x_start, a_0], [r_0, r_0], color=RED, zorder=2)
    ax.plot([x_start, a_0], [r_0, r_0], color=BLUE, zorder=4)

    # SGD: horizontal before a_0 is hidden under the shared blue plateau;
    # its decreasing branch is visible as a gray dashed line.
    ax.plot([a_0, 1.0], [r_0, 0.0], color=YELLOW, zorder=1)

    # Polyak: flat until a_H, then coincides with Nesterov.
    ax.plot([a_0, a_h], [r_0, r_0], color=RED, zorder=2)

    # Nesterov: increasing branch followed by the 2s(1-a) branch.
    ax.plot([a_0, a_n], [r_0, r_n], color=BLUE, zorder=4)
    ax.plot([a_n, a_h], [r_n, r_0], color=BLUE, zorder=4)

    # Use a distinct blended color where Polyak and Nesterov coincide.
    ax.plot([a_h, 1.0], [r_0, 0.0], color=OVERLAP, zorder=5)
    ax.plot([a_n], [r_n], marker="o", linestyle="None", color=BLUE,
            markeredgecolor="white", markeredgewidth=0.7, zorder=6)

    # Use a compact legend rather than placing method names on the curves.
    legend_handles = [
        Line2D([0], [0], color=YELLOW, label="SGD"),
        Line2D([0], [0], color=RED, label="Polyak"),
        Line2D([0], [0], color=BLUE, label="Nesterov"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left",
              bbox_to_anchor=(0.82, 0.98), borderaxespad=0.0,
              handlelength=1.8)

    # Exact expressions are written directly on the ticks.
    ax.set_xticks([a_0, a_n, a_h, 1.0])
    ax.set_xticklabels([
        r"$\frac{s}{s+1}$",
        r"$\frac{2s\beta+1}{2s\beta+\beta+1}$",
        r"$\frac{2s+1}{2s+2}$",
        r"$1$",
    ])
    ax.set_yticks([r_0, r_n])
    ax.set_yticklabels([
        r"$\frac{s}{s+1}$",
        r"$\frac{2s\beta}{2s\beta+\beta+1}$",
    ])

    # Keep all exact x-axis expressions on the same baseline.
    for label in ax.get_xticklabels():
        label.set_y(-0.015)

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(0.0, y_top)
    # Arrowed coordinate axes with compact labels next to the arrow tips.
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="major", direction="in", length=3,
                   width=0.8, colors=AXIS, pad=5)

    ax.annotate(
        "", xy=(x_right, 0.0), xytext=(x_left, 0.0),
        arrowprops=dict(arrowstyle="->", color=AXIS, linewidth=1.0,
                        mutation_scale=10, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        "", xy=(x_left, y_top), xytext=(x_left, 0.0),
        arrowprops=dict(arrowstyle="->", color=AXIS, linewidth=1.0,
                        mutation_scale=10, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.annotate(
        r"$\log_D B$", xy=(x_right, 0.0), xycoords="data",
        xytext=(5, -1), textcoords="offset points",
        ha="left", va="center", fontsize=11, annotation_clip=False,
    )
    ax.annotate(
        r"$-\log_D \mathcal{E}(D)$", xy=(x_left, y_top), xycoords="data",
        xytext=(5, 1), textcoords="offset points",
        ha="left", va="bottom", fontsize=11, annotation_clip=False,
    )

    ax.grid(False)
    fig.subplots_adjust(left=0.22, right=0.88, bottom=0.24, top=0.86)
    return fig


if __name__ == "__main__":
    figure = make_figure()
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    print(f"saved {OUTPUT_PDF}")
    print(f"saved {OUTPUT_PNG}")
