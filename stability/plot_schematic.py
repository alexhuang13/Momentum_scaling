#!/usr/bin/env python3
"""Render the theoretical batch-size stability schematic."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator


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
        "legend.fontsize": 10,
        "legend.handlelength": 1.8,
    }
)


COLORS = {
    "sgd": "C0",
    "polyak": "C1",
    "nesterov": "C2",
}


def stability_curves(
    B: np.ndarray,
    beta: float,
    one_minus_rho: float,
):
    eta_sgd = np.ones_like(B)

    eta_polyak = np.minimum(
        1.0,
        B * one_minus_rho,
    )

    eta_nag = np.minimum(
        1.0,
        B**beta * one_minus_rho,
    )

    return eta_sgd, eta_polyak, eta_nag


def draw_plot(
    ax,
    B,
    eta_sgd,
    eta_polyak,
    eta_nag,
) -> None:
    lw_main = 2.6

    # SGD: bottom layer
    ax.plot(
        B,
        eta_sgd,
        color=COLORS["sgd"],
        linestyle="-",
        linewidth=lw_main,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=21,
    )

    # Nesterov: middle layer
    ax.plot(
        B,
        eta_nag,
        color=COLORS["nesterov"],
        linestyle="-",
        linewidth=lw_main,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=22,
    )

    # Polyak: top layer
    ax.plot(
        B,
        eta_polyak,
        color=COLORS["polyak"],
        linestyle="-",
        linewidth=lw_main,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=23,
    )


def add_annotations(
    ax,
    beta: float,
) -> None:
    # Top horizontal line
    ax.text(
        5.0,
        1.1,
        r"$\eta_{\mathrm{crit}}\eqsim 1$",
        fontsize=16,
        ha="left",
        va="bottom",
        color="black",
    )

    # Polyak line
    ax.text(
        7.0,
        0.24,
        r"$\eta_{\mathrm{crit}}\eqsim B(1-\rho)$",
        fontsize=16,
        rotation=40,
        ha="center",
        va="center",
        color="black",
    )

    # Nesterov line
    ax.text(
        3.5,
        0.36,
        r"$\eta_{\mathrm{crit}}\eqsim B^\beta(1-\rho)$",
        fontsize=16,
        rotation=58,
        ha="center",
        va="center",
        color="black",
    )


def setup_axes(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1, 70)
    ax.set_ylim(0.05, 2)

    # Show a complete box around the plotting area.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")
        # Remove both major and minor ticks on the logarithmic axes.
        ax.xaxis.set_major_locator(NullLocator())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    ax.set_xlabel(
        r"$\log B$",
        labelpad=10,
    )

    ax.set_ylabel(
        r"$\log \eta_{\mathrm{crit}}$",
        labelpad=18,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["sgd"],
            linestyle="-",
            linewidth=2.6,
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["polyak"],
            linestyle="-",
            linewidth=2.6,
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["nesterov"],
            linestyle="-",
            linewidth=2.6,
        ),
    ]

    # legend = ax.legend(
    #     handles,
    #     [
    #         "SGD",
    #         "Polyak",
    #         "Nesterov",
    #     ],
    #     loc="lower right",
    #     frameon=True,
    #     fancybox=False,
    #     framealpha=1.0,
    #     facecolor="white",
    #     edgecolor="0.8",
    #     borderaxespad=0.25,
    #     borderpad=0.45,
    #     handlelength=2.0,
    # )

    # legend.get_frame().set_linewidth(0.8)


def make_figure(
    beta: float,
    one_minus_rho: float,
    out_dir: Path,
    dpi: int = 200,
) -> None:
    B = np.logspace(
        0,
        2,
        600,
    )

    eta_sgd, eta_polyak, eta_nag = stability_curves(
        B=B,
        beta=beta,
        one_minus_rho=one_minus_rho,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots()

    draw_plot(
        ax,
        B,
        eta_sgd,
        eta_polyak,
        eta_nag,
    )

    setup_axes(ax)
    add_annotations(ax, beta)

    fig.subplots_adjust(
        left=0.18,
        right=0.97,
        bottom=0.17,
        top=0.96,
    )

    pdf_path = out_dir / "stability_boundary.pdf"
    png_path = out_dir / "stability_boundary.png"

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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--one-minus-rho",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stability/results/generated/stability_schematic"),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    make_figure(
        beta=args.beta,
        one_minus_rho=args.one_minus_rho,
        out_dir=args.out_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
