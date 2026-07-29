"""Side-by-side noisy/noiseless data-optimal rate diagrams.

Both panels use explicitly positioned axes with identical width and height,
so their horizontal and vertical axes align exactly regardless of tick labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.labelsize": 14,
    "lines.linewidth": 2.2,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "legend.handlelength": 2.1,
})

BETA = 8.0
S_MAX = 2.5

OUTPUT_PDF = Path(__file__).with_name("noisy_noiseless_rate_vs_s.pdf")
OUTPUT_PNG = Path(__file__).with_name("noisy_noiseless_rate_vs_s.png")

BLUE = "#2F5DA8"
RED = "#D94B3D"
YELLOW = "#E3A018"
GUIDE = "#B8B8B8"
AXIS = "#222222"
COMMON_FILL = "#FFF1B8"
GAIN_FILL = "#D8E6F5"


def noisy_rates(s, beta=BETA):
    s = np.asarray(s, dtype=float)
    baseline = s / (s + 1.0)
    nesterov = 2.0 * s * beta / (2.0 * s * beta + beta + 1.0)
    return baseline, baseline.copy(), nesterov


def noiseless_critical_s(beta=BETA):
    return (beta - 1.0) / (2.0 * beta), 2.0 - 1.0 / beta


def noiseless_rates(s, beta=BETA):
    s = np.asarray(s, dtype=float)
    s_1, s_2 = noiseless_critical_s(beta)
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


def arrow_axes(ax, x_right, y_top):
    """Apply identical arrow-axis styling to one panel."""
    ax.set_xlim(0.0, x_right)
    ax.set_ylim(0.0, y_top)
    ax.set_xlabel("")
    ax.set_ylabel("")
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
        xytext=(5, -1), textcoords="offset points",
        ha="left", va="center", fontsize=12, annotation_clip=False,
    )
    ax.annotate(
        r"$-\log_D \mathcal{E}(D)$", xy=(0.0, y_top), xycoords="data",
        xytext=(4, 1), textcoords="offset points",
        ha="left", va="bottom", fontsize=11.5, annotation_clip=False,
    )
    ax.grid(False)


def plot_noisy(ax, beta=BETA, s_max=S_MAX):
    s = np.linspace(0.0, s_max, 1400)
    sgd, polyak, nesterov = noisy_rates(s, beta)
    y_top = 1.10

    ax.fill_between(s, 0.0, sgd, color=COMMON_FILL, alpha=0.58,
                    linewidth=0, zorder=-2)
    ax.fill_between(s, sgd, nesterov, color=GAIN_FILL, alpha=0.68,
                    linewidth=0, zorder=-1)
    ax.plot(s, sgd, color=YELLOW, label="SGD", zorder=2)
    ax.plot(s, polyak, color=RED, linestyle=(0, (5, 3)),
            label="Polyak", zorder=3)
    ax.plot(s, nesterov, color=BLUE, label="Nesterov", zorder=4)
    ax.plot([0.0, s_max], [1.0, 1.0], color=GUIDE,
            linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)

    target_x = 0.43 * s_max
    bottom = np.interp(target_x, s, sgd)
    top = np.interp(target_x, s, nesterov)
    ax.annotate(
        "Nesterov > SGD = Polyak",
        xy=(target_x, bottom + 0.57 * (top - bottom)),
        xytext=(0.93 * s_max, 0.47),
        color=BLUE, fontsize=7.8, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                  edgecolor="none", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=0.85,
                        shrinkA=3, shrinkB=2,
                        connectionstyle="arc3,rad=0.22"),
        zorder=7,
    )

    ax.set_xticks([])
    ax.set_yticks([1.0])
    ax.set_yticklabels([r"$1$"])
    arrow_axes(ax, s_max, y_top)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.04, 0.96),
              borderaxespad=0.0)
    ax.text(0.5, -0.13, r"(a) Noisy ($\sigma>0$)", transform=ax.transAxes,
            ha="center", va="top", fontsize=11)


def plot_noiseless(ax, beta=BETA, s_max=S_MAX):
    s = np.linspace(0.0, s_max, 1400)
    sgd, polyak, nesterov = noiseless_rates(s, beta)
    s_1, s_2 = noiseless_critical_s(beta)
    r_1 = (beta - 1.0) / (beta + 1.0)
    y_top = 1.12 * s_2

    ax.fill_between(s, 0.0, sgd, color=COMMON_FILL, alpha=0.58,
                    linewidth=0, zorder=-2)
    ax.fill_between(s, sgd, nesterov, color=GAIN_FILL, alpha=0.68,
                    linewidth=0, zorder=-1)
    ax.plot(s, sgd, color=YELLOW, label="SGD", zorder=2)
    ax.plot(s, polyak, color=RED, linestyle=(0, (5, 3)),
            label="Polyak", zorder=3)
    ax.plot(s, nesterov, color=BLUE, label="Nesterov", zorder=4)

    for x_value, y_value in ((s_1, r_1), (s_2, s_2)):
        ax.plot([x_value, x_value], [0.0, y_value], color=GUIDE,
                linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
    ax.plot([0.0, s_1], [r_1, r_1], color=GUIDE,
            linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
    ax.plot([0.0, s_max], [s_2, s_2], color=GUIDE,
            linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
    ax.scatter([s_1, s_2], [r_1, s_2], s=26, color=BLUE,
               edgecolor="white", linewidth=0.7, zorder=5)

    target_x = 0.57 * (s_1 + s_2)
    bottom = np.interp(target_x, s, sgd)
    top = np.interp(target_x, s, nesterov)
    ax.annotate(
        "Nesterov > SGD = Polyak",
        xy=(target_x, bottom + 0.60 * (top - bottom)),
        xytext=(s_2 + 0.43, 0.72 * s_2),
        color=BLUE, fontsize=7.8, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                  edgecolor="none", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=0.85,
                        shrinkA=3, shrinkB=2,
                        connectionstyle="arc3,rad=0.22"),
        zorder=7,
    )

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
    arrow_axes(ax, s_max, y_top)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.04, 0.96),
              borderaxespad=0.0)
    ax.text(0.5, -0.13, r"(b) Noiseless ($\sigma=0$)", transform=ax.transAxes,
            ha="center", va="top", fontsize=11)


def make_figure(beta=BETA, s_max=S_MAX):
    # Explicit axes rectangles: identical x, y, width, and height fractions.
    # Tick-label bounding boxes cannot alter these axes dimensions.
    fig = plt.figure(figsize=(10.4, 3.75))
    bottom = 0.19
    height = 0.70
    width = 0.385
    ax_left = fig.add_axes([0.075, bottom, width, height])
    ax_right = fig.add_axes([0.565, bottom, width, height])

    plot_noisy(ax_left, beta, s_max)
    plot_noiseless(ax_right, beta, s_max)
    return fig


if __name__ == "__main__":
    figure = make_figure()
    # Do not use bbox_inches='tight': fixed canvas margins preserve exact panel
    # geometry and therefore exact horizontal/vertical axis alignment.
    figure.savefig(OUTPUT_PDF)
    figure.savefig(OUTPUT_PNG, dpi=240)
    print(f"saved {OUTPUT_PDF}")
    print(f"saved {OUTPUT_PNG}")
