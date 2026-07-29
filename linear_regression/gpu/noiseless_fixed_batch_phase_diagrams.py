"""Plot the three noiseless fixed-batch phase diagrams.

The panels correspond to the hard, intermediate, and easy source regimes in
Appendix: "Data-optimal scaling for fixed batch size in the noiseless setting".
The visual style follows rate_diagram.py in the same directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 14,
    "lines.linewidth": 2.2,
    "lines.markersize": 5.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (5.2, 3.75),
    "legend.fontsize": 9.5,
    "legend.handlelength": 2.0,
})

BETA = 8.0
# Representative values for the three theorem regimes at beta=8:
# hard: s <= 7/16; intermediate: 7/16 < s < 15/8; easy: s >= 15/8.
S_HARD = 0.30
S_INTERMEDIATE = 1.00
S_EASY = 2.50

OUTPUT_DIR = Path(__file__).parent

BLUE = "#2F5DA8"
RED = "#D94B3D"
YELLOW = "#E3A018"
GUIDE = "#B8B8B8"
AXIS = "#222222"
FILL_SGD = "#FFF1B8"
FILL_POL = "#FAD4D0"
FILL_NES = "#D8E6F5"


def sgd_rate(a, s, beta):
    a = np.asarray(a, dtype=float)
    return np.minimum(
        s * (1.0 - a),
        (2.0 - 1.0 / beta) - (1.0 - 1.0 / beta) * a,
    )


def polyak_rate(a, s, beta):
    """The three-regime Polyak formula from the fixed-batch theorem."""
    a = np.asarray(a, dtype=float)
    low = 1.0 - 1.0 / beta
    high = 2.0 - 1.0 / beta
    balance = s * beta * (3.0 - 2.0 * a) / (s * beta + beta + 1.0)
    transient = 2.0 * s * (1.0 - a)

    if s <= low:
        return np.where(a <= 0.5, s, transient)
    if s < high:
        first = (high - s) / 2.0
        second = (2.0 * s * beta + 2.0 - beta) / (2.0 * (s * beta + 1.0))
        return np.select(
            [a <= first, a <= second],
            [s, balance],
            default=transient,
        )

    first = (s - high) / (s - low)
    second = (2.0 * s * beta + 2.0 - beta) / (2.0 * (s * beta + 1.0))
    initial = high - low * a
    return np.select(
        [a <= first, a <= second],
        [initial, balance],
        default=transient,
    )


def nesterov_rate(a, s, beta):
    """The three-regime Nesterov formula from the fixed-batch theorem."""
    a = np.asarray(a, dtype=float)
    hard = (beta - 1.0) / (2.0 * beta)
    high = 2.0 - 1.0 / beta
    low = 1.0 - 1.0 / beta
    balance = s * beta * (3.0 - 2.0 * a) / (s * beta + beta + 1.0)
    transient = 2.0 * s * (1.0 - a)

    if s <= hard:
        return np.where(
            a <= 1.0 / (beta + 1.0),
            s * (1.0 + (beta - 1.0) * a),
            transient,
        )
    if s < high:
        first = (2.0 * beta - 1.0 - s * beta) / (
            s * beta * (beta - 1.0) + beta**2 + 2.0 * beta - 1.0
        )
        second = (2.0 * s * beta + 2.0 - beta) / (2.0 * (s * beta + 1.0))
        initial = s * (1.0 + (beta - 1.0) * a)
        return np.select(
            [a <= first, a <= second],
            [initial, balance],
            default=transient,
        )

    first = (s - high) / (s - low)
    second = (2.0 * s * beta + 2.0 - beta) / (2.0 * (s * beta + 1.0))
    initial = high - low * a
    return np.select(
        [a <= first, a <= second],
        [initial, balance],
        default=transient,
    )


def critical_points(s, beta):
    """Return theorem breakpoints and their curve heights for guide lines."""
    hard = (beta - 1.0) / (2.0 * beta)
    low = 1.0 - 1.0 / beta
    high = 2.0 - 1.0 / beta

    if s <= hard:
        points = [
            (1.0 / (beta + 1.0),
             s * (1.0 + (beta - 1.0) / (beta + 1.0))),
            (0.5, s),
        ]
        ticks = [
            (1.0 / (beta + 1.0), r"$\frac{1}{\beta+1}$"),
            (0.5, r"$\frac{1}{2}$"),
        ]
        return points, ticks

    if s < high:
        nes_first = (2.0 * beta - 1.0 - s * beta) / (
            s * beta * (beta - 1.0) + beta**2 + 2.0 * beta - 1.0
        )
        final = (2.0 * s * beta + 2.0 - beta) / (2.0 * (s * beta + 1.0))
        if s <= low:
            points = [
                (nes_first, nesterov_rate(nes_first, s, beta)),
                (0.5, s),
                (final, 2.0 * s * (1.0 - final)),
            ]
            ticks = [
                (nes_first,
                 r"$\frac{2\beta-1-s\beta}"
                 r"{s\beta(\beta-1)+\beta^2+2\beta-1}$"),
                (0.5, r"$\frac{1}{2}$"),
                (final,
                 r"$\frac{2s\beta+2-\beta}{2(s\beta+1)}$"),
            ]
        else:
            pol_first = (high - s) / 2.0
            points = [
                (nes_first, nesterov_rate(nes_first, s, beta)),
                (pol_first, s),
                (final, 2.0 * s * (1.0 - final)),
            ]
            ticks = [
                (nes_first,
                 r"$\frac{2\beta-1-s\beta}"
                 r"{s\beta(\beta-1)+\beta^2+2\beta-1}$"),
                (pol_first,
                 r"$\frac{2-\frac{1}{\beta}-s}{2}$"),
                (final,
                 r"$\frac{2s\beta+2-\beta}{2(s\beta+1)}$"),
            ]
        return points, ticks

    first = (s - high) / (s - low)
    final = (2.0 * s * beta + 2.0 - beta) / (2.0 * (s * beta + 1.0))
    points = [
        (first, sgd_rate(first, s, beta)),
        (final, 2.0 * s * (1.0 - final)),
    ]
    ticks = [
        (first,
             r"$\frac{s-2+\frac{1}{\beta}}"
             r"{s-1+\frac{1}{\beta}}$"),
        (final,
                 r"$\frac{2s\beta+2-\beta}{2(s\beta+1)}$"),
    ]
    return points, ticks


def add_arrow_axes(ax, y_top):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="major", direction="in", length=3,
                   width=0.8, colors=AXIS, pad=5)
    ax.annotate(
        "", xy=(1.03, 0.0), xytext=(0.0, 0.0),
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
        r"$a$", xy=(1.03, 0.0), xycoords="data",
        xytext=(5, -1), textcoords="offset points",
        ha="left", va="center", fontsize=11, annotation_clip=False,
    )
    ax.annotate(
        r"$-\log_D \mathcal{E}(D)$", xy=(0.0, y_top), xycoords="data",
        xytext=(4, 1), textcoords="offset points",
        ha="left", va="bottom", fontsize=11, annotation_clip=False,
    )


def make_figure(s, beta, regime, output_stem):
    a = np.linspace(0.0, 1.0, 1600)
    sgd = sgd_rate(a, s, beta)
    pol = polyak_rate(a, s, beta)
    nes = nesterov_rate(a, s, beta)

    fig = plt.figure(figsize=(5.2, 3.75))
    ax = fig.add_axes([0.15, 0.19, 0.77, 0.70])

    # Layered fills show which method improves on the previous boundary.
    lower_pol = np.maximum(pol, sgd)
    upper_nes = np.maximum(nes, lower_pol)
    ax.fill_between(a, 0.0, sgd, color=FILL_SGD, alpha=0.58,
                    linewidth=0, zorder=-3)
    ax.fill_between(a, sgd, lower_pol, color=FILL_POL, alpha=0.58,
                    linewidth=0, zorder=-2)
    ax.fill_between(a, lower_pol, upper_nes, color=FILL_NES, alpha=0.68,
                    linewidth=0, zorder=-1)

    ax.plot(a, sgd, color=YELLOW, label="SGD", zorder=2)
    ax.plot(a, pol, color=RED, label="Polyak", zorder=3)
    ax.plot(a, nes, color=BLUE, label="Nesterov", zorder=4)

    points, ticks = critical_points(s, beta)
    for x, y in points:
        ax.plot([x, x], [0.0, y], color=GUIDE,
                linestyle=(0, (3, 3)), linewidth=0.8, zorder=0)
        ax.plot([0.0, x], [y, y], color=GUIDE,
                linestyle=(0, (3, 3)), linewidth=0.8, zorder=0)

    # Mark the Nesterov peak in the hard regime and all shared transition
    # points in the other regimes.
    for x, y in points:
        ax.plot([x], [y], marker="o", linestyle="None", color=BLUE,
                markeredgecolor="white", markeredgewidth=0.7, zorder=6)

    ax.set_xticks([x for x, _ in ticks] + [1.0])
    ax.set_xticklabels([label for _, label in ticks] + [r"$1$"])
    for label in ax.get_xticklabels():
        label.set_fontsize(8.2)

    # Use exact symbolic y ticks where they are compact; numerical positions
    # are determined by the selected representative s and beta.
    if regime == "hard":
        y_peak = 2.0 * s * beta / (beta + 1.0)
        ax.set_yticks([s, y_peak])
        ax.set_yticklabels([r"$s$", r"$\frac{2s\beta}{\beta+1}$"])
    elif regime == "intermediate":
        ax.set_yticks([s])
        ax.set_yticklabels([r"$s$"])
    else:
        cap = 2.0 - 1.0 / beta
        ax.set_yticks([cap])
        ax.set_yticklabels([r"$2-\frac{1}{\beta}$"])

    # Region annotations, placed away from the narrow curve gaps.
    if regime == "hard":
        x_blue = 0.08
        y_blue = 0.5 * (polyak_rate(x_blue, s, beta)
                        + nesterov_rate(x_blue, s, beta))
        ax.annotate(
            "Nesterov > Polyak",
            xy=(x_blue, y_blue), xytext=(0.56, 1.08 * np.max(nes)),
            color=BLUE, fontsize=8.2, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=0.85,
                            connectionstyle="arc3,rad=0.22"), zorder=7,
        )
        x_red = 0.31
        y_red = 0.5 * (sgd_rate(x_red, s, beta)
                       + polyak_rate(x_red, s, beta))
        ax.annotate(
            "Polyak > SGD",
            xy=(x_red, y_red), xytext=(0.78, 0.47 * np.max(nes)),
            color=RED, fontsize=8.2, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=RED, linewidth=0.85,
                            connectionstyle="arc3,rad=-0.18"), zorder=7,
        )
    elif regime == "intermediate":
        x_blue = 0.04
        y_blue = 0.5 * (polyak_rate(x_blue, s, beta)
                        + nesterov_rate(x_blue, s, beta))
        ax.annotate(
            "Nesterov > Polyak",
            xy=(x_blue, y_blue), xytext=(0.49, 1.02 * np.max(nes)),
            color=BLUE, fontsize=8.2, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=0.85,
                            connectionstyle="arc3,rad=0.22"), zorder=7,
        )
        x_red = 0.25
        y_red = 0.5 * (sgd_rate(x_red, s, beta)
                       + polyak_rate(x_red, s, beta))
        ax.annotate(
            "Polyak > SGD",
            xy=(x_red, y_red), xytext=(0.76, 0.48 * np.max(nes)),
            color=RED, fontsize=8.2, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=RED, linewidth=0.85,
                            connectionstyle="arc3,rad=-0.18"), zorder=7,
        )
    else:
        # The momentum advantage starts only after the first easy-regime
        # breakpoint. Point to the interior of the red region between the
        # first breakpoint and the transient boundary.
        easy_first = (s - (2.0 - 1.0 / beta)) / (s - (1.0 - 1.0 / beta))
        easy_final = (2.0 * s * beta + 2.0 - beta) / (2.0 * (s * beta + 1.0))
        x_red = easy_first + 0.48 * (easy_final - easy_first)
        y_red = 0.5 * (sgd_rate(x_red, s, beta)
                       + polyak_rate(x_red, s, beta))
        ax.annotate(
            "Polyak = Nesterov > SGD",
            xy=(x_red, y_red), xytext=(0.90, 0.43 * np.max(pol)),
            color=RED, fontsize=8.2, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="none", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=RED, linewidth=0.85,
                            connectionstyle="arc3,rad=-0.18"), zorder=7,
        )

    y_top = 1.18 * max(np.max(sgd), np.max(pol), np.max(nes))
    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(0.0, y_top)
    add_arrow_axes(ax, y_top)

    handles = [
        Line2D([0], [0], color=YELLOW, label="SGD"),
        Line2D([0], [0], color=RED, label="Polyak"),
        Line2D([0], [0], color=BLUE, label="Nesterov"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right",
              bbox_to_anchor=(0.98, 0.94), borderaxespad=0.0)
    ax.text(0.50, -0.18, rf"{regime.capitalize()} regime: $s={s:g},\ \beta={beta:g}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=10.5)
    ax.grid(False)

    pdf = OUTPUT_DIR / f"{output_stem}.pdf"
    png = OUTPUT_DIR / f"{output_stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=240)
    plt.close(fig)
    print(f"saved {pdf}")
    print(f"saved {png}")


def validate_parameters():
    hard = (BETA - 1.0) / (2.0 * BETA)
    easy = 2.0 - 1.0 / BETA
    if not (0.0 < S_HARD <= hard):
        raise ValueError("S_HARD is outside the hard regime.")
    if not (hard < S_INTERMEDIATE < easy):
        raise ValueError("S_INTERMEDIATE is outside the intermediate regime.")
    if not (S_EASY >= easy):
        raise ValueError("S_EASY is outside the easy regime.")


if __name__ == "__main__":
    validate_parameters()
    make_figure(S_HARD, BETA, "hard", "noiseless_fixed_batch_hard")
    make_figure(
        S_INTERMEDIATE, BETA, "intermediate",
        "noiseless_fixed_batch_intermediate",
    )
    make_figure(S_EASY, BETA, "easy", "noiseless_fixed_batch_easy")
