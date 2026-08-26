#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. Configuration
# =============================================================================

D_FIXED = 2**14
S_VALUE = 0.3
BETA = 4.0

DEFAULT_DATA_DIR = Path.cwd() / "cbs" / "s_0.3_beta_4"
CSV_FILENAME = "optimal_loss.csv"

ALGORITHM_ORDER = [
    "SGD",
    "Polyak",
    "Nesterov",
]

MARKERS = {
    "SGD": "o",
    "Polyak": "^",
    "Nesterov": "s",
}

COLORS = {
    "SGD": "C0",
    "Polyak": "C1",
    "Nesterov": "C2",
}

plt.rcParams.update({
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
    "legend.handlelength": 1,
})


# =============================================================================
# 2. Read the saved optimal losses
# =============================================================================

def load_optimal_losses(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Could not find the saved loss CSV: {csv_path}"
        )

    loss_df = pd.read_csv(csv_path)

    required_columns = [
        "B",
        *ALGORITHM_ORDER,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in loss_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    loss_df = loss_df[required_columns].copy()

    for column in required_columns:
        loss_df[column] = pd.to_numeric(
            loss_df[column],
            errors="raise",
        )

    if loss_df.empty:
        raise ValueError(f"The loss CSV is empty: {csv_path}")

    if not np.isfinite(loss_df.to_numpy(dtype=np.float64)).all():
        raise ValueError(
            f"The loss CSV contains non-finite values: {csv_path}"
        )

    if (loss_df["B"] <= 0).any():
        raise ValueError("Every batch size B must be positive.")

    return (
        loss_df
        .sort_values("B")
        .reset_index(drop=True)
    )


# =============================================================================
# 3. Plot batch size versus optimal loss
# =============================================================================

def plot_cbs(
    loss_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    base_optimal_exponent = S_VALUE / (S_VALUE + 1.0)
    b_sgd_critical = D_FIXED**base_optimal_exponent

    b_values = loss_df["B"].to_numpy(dtype=np.float64)
    x_min = float(np.min(b_values))
    x_max = 1000.0

    # Keep the original three batch-size regimes unchanged.
    ultra_large_boundary = 100.0

    fig, ax = plt.subplots()

    # -------------------------------------------------------------------------
    # Batch-size regimes
    # -------------------------------------------------------------------------

    ax.axvspan(
        x_min,
        b_sgd_critical,
        color="lightskyblue",
        alpha=0.18,
        zorder=0,
    )

    ax.axvspan(
        b_sgd_critical,
        ultra_large_boundary,
        color="lightcoral",
        alpha=0.15,
        zorder=0,
    )

    ax.axvspan(
        ultra_large_boundary,
        x_max,
        color="lightgreen",
        alpha=0.18,
        zorder=0,
    )

    # -------------------------------------------------------------------------
    # Empirical curves
    # -------------------------------------------------------------------------

    for algorithm in ALGORITHM_ORDER:
        ax.plot(
            b_values,
            loss_df[algorithm].to_numpy(dtype=np.float64),
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            label=algorithm,
            zorder=3,
        )

    # -------------------------------------------------------------------------
    # Regime boundaries
    # -------------------------------------------------------------------------

    ax.axvline(
        b_sgd_critical,
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.6,
        zorder=1,
    )

    ax.axvline(
        ultra_large_boundary,
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.6,
        zorder=1,
    )

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------

    ax.set_xscale("log", base=10)
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel("$\log B$")
    ax.set_ylabel("Loss")

    ax.set_xticks(
        [1e1, 1e2, 1e3],
        [
            r"$1$",
            r"$2$",
            r"$3$",
        ],
    )

    # -------------------------------------------------------------------------
    # Region labels
    #
    # Since the x-axis is logarithmic, use geometric rather than arithmetic
    # centers for the three regions.
    # -------------------------------------------------------------------------

    region_labels = [
        (
            np.sqrt(x_min * b_sgd_critical),
            0.93,
            "Small-batch",
        ),
        (
            np.sqrt(
                b_sgd_critical
                * ultra_large_boundary
            ),
            0.93,
            "Large-batch",
        ),
        (
            np.sqrt(
                ultra_large_boundary
                * x_max
            ),
            0.97,
            "Ultra\nlarge-batch",
        ),
    ]

    for x_position, y_position, label in region_labels:
        ax.text(
            x_position,
            y_position,
            label,
            transform=ax.get_xaxis_transform(),
            fontsize=12,
            fontweight="semibold",
            color="0.25",
            ha="center",
            va="top",
            zorder=4,
        )

    # Move the legend below the SB label.
    # ax.legend(
    #     loc="upper left",
    #     bbox_to_anchor=(0.02, 0.84),
    #     borderaxespad=0,
    # )

    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = output_dir / "cbs.pdf"
    png_path = output_dir / "cbs.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=dpi,
    )

    plt.close(fig)

    return pdf_path, png_path


# =============================================================================
# 4. Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory containing optimal_loss.csv. "
            "The generated cbs.pdf and cbs.png are written "
            "back to this directory."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    csv_path = data_dir / CSV_FILENAME

    loss_df = load_optimal_losses(csv_path)

    pdf_path, png_path = plot_cbs(
        loss_df=loss_df,
        output_dir=data_dir,
        dpi=args.dpi,
    )

    print(f"read {csv_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()