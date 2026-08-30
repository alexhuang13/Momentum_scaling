#!/usr/bin/env python3
"""Render the data-scaling phase diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# Global plotting settings
# Shared visual style for the scaling phase diagram.
# =============================================================================

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": (5, 4),
        "legend.fontsize": 8,
        "legend.handlelength": 1,
    }
)


COLORS = {
    "sgd": "C0",
    "polyak": "C1",
    "nesterov": "C2",
    # Match the critical-batch-size figure colors.
    "blue_fill": "lightskyblue",
    "red_fill": "lightcoral",
    "green_fill": "lightgreen",
}


# =============================================================================
# Coordinates
# =============================================================================


def coords(
    s: float,
    beta: float,
) -> dict[str, float]:

    y_base = s / (s + 1.0)

    y_peak = 2.0 * s * beta / (2.0 * s * beta + beta + 1.0)

    return {
        "x_s": (s / (s + 1.0)),
        "x_peak": ((2.0 * s * beta + 1.0) / (2.0 * s * beta + beta + 1.0)),
        "x_poly_end": ((2.0 * s + 1.0) / (2.0 * s + 2.0)),
        "y_base": y_base,
        "y_peak": y_peak,
    }


# =============================================================================
# Background regions
# Match the critical-batch-size figure boundaries.
# =============================================================================


def add_regions(
    ax,
    c,
) -> None:

    # Small-batch region
    ax.axvspan(
        0,
        c["x_s"],
        color="lightskyblue",
        alpha=0.18,
        zorder=0,
    )

    # Large-batch region
    ax.axvspan(
        c["x_s"],
        c["x_poly_end"],
        color="lightcoral",
        alpha=0.15,
        zorder=0,
    )

    # Ultra-large-batch region
    ax.axvspan(
        c["x_poly_end"],
        1,
        color="lightgreen",
        alpha=0.18,
        zorder=0,
    )

    # Use the same dashed boundary style as the critical-batch-size figure.
    ax.axvline(
        c["x_s"],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.6,
        zorder=1,
    )

    ax.axvline(
        c["x_poly_end"],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.6,
        zorder=1,
    )


# =============================================================================
# Region labels
# =============================================================================


def add_region_labels(
    ax,
    c,
) -> None:

    labels = [
        (
            0.5 * c["x_s"],
            0.93,
            "Small-batch",
        ),
        (
            0.5 * (c["x_s"] + c["x_poly_end"]),
            0.93,
            "Large-batch",
        ),
        (
            0.5 * (c["x_poly_end"] + 1.0),
            0.97,
            "Ultra\nlarge-batch",
        ),
    ]

    for x, y, label in labels:
        ax.text(
            x,
            y,
            label,
            transform=ax.get_xaxis_transform(),
            fontsize=12,
            fontweight="semibold",
            color="0.25",
            ha="center",
            va="top",
            zorder=30,
        )


# =============================================================================
# Theory curves
# =============================================================================


def draw_lines(
    ax,
    c,
) -> None:
    lw_main = 2.6

    # Nesterov: bottom layer
    ax.plot(
        [
            0,
            c["x_s"],
            c["x_peak"],
            c["x_poly_end"],
            1,
        ],
        [
            c["y_base"],
            c["y_base"],
            c["y_peak"],
            c["y_base"],
            0,
        ],
        color=COLORS["nesterov"],
        linewidth=lw_main,
        zorder=21,
    )

    # Polyak: middle layer
    ax.plot(
        [
            0,
            c["x_poly_end"],
            1,
        ],
        [
            c["y_base"],
            c["y_base"],
            0,
        ],
        color=COLORS["polyak"],
        linewidth=lw_main,
        zorder=22,
    )

    # SGD: top layer
    ax.plot(
        [
            0,
            c["x_s"],
            1,
        ],
        [
            c["y_base"],
            c["y_base"],
            0,
        ],
        color=COLORS["sgd"],
        linewidth=lw_main,
        zorder=23,
    )

    ax.scatter(
        [c["x_peak"]],
        [c["y_peak"]],
        s=42,
        color=COLORS["nesterov"],
        edgecolors="white",
        linewidths=0.8,
        clip_on=False,
        zorder=24,
    )

    # ax.annotate(
    #     "faster data scaling",
    #     xy=(
    #         c["x_peak"],
    #         c["y_peak"],
    #     ),
    #     xytext=(-12, 12),
    #     textcoords="offset points",
    #     fontsize=10,
    #     fontweight="semibold",
    #     color=COLORS["nesterov"],
    #     ha="right",
    #     va="bottom",
    #     arrowprops={
    #         "arrowstyle": "->",
    #         "color": COLORS["nesterov"],
    #         "linewidth": 1.1,
    #         "shrinkA": 2,
    #         "shrinkB": 5,
    #         "mutation_scale": 8,
    #     },
    #     zorder=25,
    # )


# =============================================================================
# Axes
# =============================================================================


def setup_axes(
    ax,
    c,
) -> None:

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        c["y_peak"] * 1.18,
    )

    # IMPORTANT:
    # Do NOT hide top/right spines.
    #
    # Keep the standard Matplotlib axes box.
    # so all four black spines remain visible.

    # -------------------------------------------------------------
    # Transition points
    # -------------------------------------------------------------

    ax.set_xticks(
        [
            c["x_s"],
            c["x_peak"],
            c["x_poly_end"],
        ]
    )

    ax.set_xticklabels(
        [
            r"$b_1$",
            r"$b_2$",
            r"$b_3$",
        ]
    )

    ax.set_yticks([])

    ax.tick_params(
        axis="x",
        labelsize=12,
        pad=3,
        length=3,
    )

    ax.tick_params(
        axis="y",
        length=0,
    )

    # -------------------------------------------------------------
    # Axis labels
    # -------------------------------------------------------------

    ax.set_xlabel(
        r"$\log B$",
        fontsize=16,
        labelpad=3,
    )

    ax.set_ylabel(
        r"Data-scaling rate $r_A(b)$",
        fontsize=13,
        labelpad=4,
    )

    # -------------------------------------------------------------
    # Legend
    #
    # Use the same framed legend style as the companion figures.
    # white face, grey border.
    # -------------------------------------------------------------

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["sgd"],
            linewidth=2.6,
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["polyak"],
            linewidth=2.6,
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["nesterov"],
            linewidth=2.6,
        ),
    ]

    # ax.legend(
    #     handles,
    #     [
    #         "SGD",
    #         "Polyak",
    #         "Nesterov",
    #     ],
    #     loc="lower left",
    #     bbox_to_anchor=(0.02, 0.03),
    #     borderaxespad=0,
    # )


# =============================================================================
# Standalone legend
# =============================================================================


def make_legend_figure(
    output_dir: Path,
    dpi: int = 300,
) -> None:

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["sgd"],
            linewidth=1.5,
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["polyak"],
            linewidth=1.5,
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["nesterov"],
            linewidth=1.5,
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(8.0, 0.55),
    )

    ax.axis("off")

    legend = ax.legend(
        handles,
        [
            "SGD",
            "Polyak",
            "Nesterov",
        ],
        loc="center",
        ncol=3,
        frameon=True,
        columnspacing=4.0,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.45,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.7")
    legend.get_frame().set_linewidth(0.8)

    pdf_path = output_dir / "legend.pdf"
    png_path = output_dir / "legend.png"

    fig.savefig(
        pdf_path,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        png_path,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=dpi,
    )

    plt.close(fig)

    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


# =============================================================================
# Figure
# =============================================================================


def make_figure(
    s: float,
    beta: float,
    out_dir: Path,
    dpi: int = 300,
) -> None:

    c = coords(
        s=s,
        beta=beta,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Match the companion figure layout.
    fig, ax = plt.subplots()

    add_regions(
        ax,
        c,
    )

    draw_lines(
        ax,
        c,
    )

    setup_axes(
        ax,
        c,
    )

    add_region_labels(
        ax,
        c,
    )

    # Match the companion figure style.
    fig.subplots_adjust(
        left=0.18,
        right=0.97,
        bottom=0.17,
        top=0.96,
    )

    pdf_path = out_dir / "rate_diagram.pdf"

    png_path = out_dir / "rate_diagram.png"

    # Match the companion figure export settings.
    fig.savefig(
        pdf_path,
        facecolor="white",
    )

    fig.savefig(
        png_path,
        facecolor="white",
        dpi=dpi,
    )

    plt.close(fig)

    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/figures/rate_diagram"),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    make_figure(
        s=args.s,
        beta=args.beta,
        out_dir=args.out_dir,
        dpi=args.dpi,
    )

    make_legend_figure(
        output_dir=args.out_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
