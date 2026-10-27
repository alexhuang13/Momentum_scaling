from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "pilot_momentum_3epochs"
)
OUTPUT_STEM = "fixed_D_optimal_loss_vs_batch"

ALGORITHM_ORDER = ["SGD", "Polyak", "Nesterov"]

NAMES = {
    "sgd": "SGD",
    "polyak": "Polyak",
    "nesterov": "Nesterov",
}

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
    "figure.figsize": (5.6, 4.3),
    "legend.fontsize": 12,
    "legend.handlelength": 1,
    "axes.grid": False,
})


def load_optimal_losses(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing result CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "global_batch",
        "algorithm",
        "val_loss_mean",
        "status",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["algorithm"] = (
        df["algorithm"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["algorithm"].isin(NAMES)
    ].copy()

    rejected = df[
        df["status"] != "ok"
    ]

    if not rejected.empty:
        print("Excluded settings:")
        print(
            rejected[
                [
                    "global_batch",
                    "algorithm",
                    "status",
                ]
            ].to_string(index=False)
        )

    df = df[
        df["status"] == "ok"
    ].copy()

    if df.empty:
        raise ValueError(
            "No stable results available for plotting."
        )

    for column in [
        "global_batch",
        "val_loss_mean",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    values = df[
        [
            "global_batch",
            "val_loss_mean",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Batch sizes or losses contain non-finite values."
        )

    batches = df[
        "global_batch"
    ].to_numpy(dtype=float)

    if (
        (batches <= 0).any()
        or
        (batches != np.floor(batches)).any()
    ):
        raise ValueError(
            "Batch sizes must be positive integers."
        )

    if (
        df["val_loss_mean"] < 0
    ).any():
        raise ValueError(
            "Losses must be nonnegative."
        )

    df["global_batch"] = (
        df["global_batch"]
        .astype(int)
    )

    if df.duplicated(
        [
            "global_batch",
            "algorithm",
        ]
    ).any():
        raise ValueError(
            "Duplicate (batch, algorithm) entries."
        )

    df["algorithm"] = (
        df["algorithm"]
        .map(NAMES)
    )

    result = (
        df.pivot(
            index="global_batch",
            columns="algorithm",
            values="val_loss_mean",
        )
        .reindex(
            columns=ALGORITHM_ORDER
        )
        .sort_index()
    )

    if result.isna().any().any():
        print(
            "Missing results remain gaps; "
            "no values are filled in."
        )

    return (
        result
        .rename_axis("B")
        .reset_index()
    )


def automatic_boundaries(
    b_values: np.ndarray,
) -> tuple[float, float]:
    """
    Divide the measured batch sizes into:

        first 6 points  -> small batch
        next 4 points   -> large batch
        remaining       -> ultra-large batch

    Since the x-axis is logarithmic, boundaries are placed at
    geometric midpoints between points 6/7 and points 10/11.
    """

    if len(b_values) < 11:
        raise ValueError(
            "At least 11 batch-size points are required "
            "to place boundaries after the 6th and 10th points."
        )

    b_values = np.sort(
        np.asarray(
            b_values,
            dtype=float,
        )
    )

    first = np.sqrt(
        b_values[3]
        * b_values[4]
    )

    second = np.sqrt(
        b_values[8]
        * b_values[9]
    )

    return (
        float(first),
        float(second),
    )


def plot_cbs(
    loss_df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
    boundaries=None,
) -> tuple[Path, Path]:

    b_values = (
        loss_df["B"]
        .to_numpy(dtype=float)
    )

    data_min = float(
        b_values.min()
    )
    data_max = float(
        b_values.max()
    )

    x_min = data_min
    x_max = data_max

    if x_min == x_max:
        x_min = x_min / 2
        x_max = x_max * 2

    # Slight padding so endpoint markers are not clipped.
    margin = 10 ** (
        0.02
        * np.log10(
            x_max / x_min
        )
    )

    x_min /= margin
    x_max *= margin

    # ---------------------------------------------------------
    # Region boundaries
    # ---------------------------------------------------------
    if boundaries is None:
        first, second = automatic_boundaries(
            b_values
        )
    else:
        first, second = boundaries

    if not (
        data_min
        < first
        < second
        < data_max
    ):
        raise ValueError(
            "Require min(B) < first boundary "
            "< second boundary < max(B)."
        )

    print("Batch regions:")
    print(f"  small batch:       B <= {b_values[5]:g}")
    print(f"  first boundary:    {first:.6g}")
    print(f"  large batch:       {b_values[6]:g} <= B <= {b_values[9]:g}")
    print(f"  second boundary:   {second:.6g}")
    print(f"  ultra-large batch: B >= {b_values[10]:g}")

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------
    fig, ax = plt.subplots()

    # ---------------------------------------------------------
    # Boundary lines
    # ---------------------------------------------------------
    for boundary in (
        first,
        second,
    ):
        ax.axvline(
            boundary,
            linestyle=":",
            color="0.72",
            linewidth=1.2,
            alpha=0.95,
            zorder=1,
        )

    # ---------------------------------------------------------
    # Curves
    # ---------------------------------------------------------
    for algorithm in ALGORITHM_ORDER:
        ax.plot(
            b_values,
            loss_df[
                algorithm
            ].to_numpy(
                dtype=float
            ),
            color=COLORS[
                algorithm
            ],
            marker=MARKERS[
                algorithm
            ],
            label=algorithm,
            zorder=3,
        )

    # ---------------------------------------------------------
    # X axis
    # 改成自然的 log ticks，不再手动 set_xticks(...)
    # ---------------------------------------------------------
    ax.set_xscale(
        "log",
        base=10,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_xlabel(
        r"Batch size $B$"
    )

    ax.xaxis.set_major_locator(
        mticker.LogLocator(base=10)
    )
    ax.xaxis.set_major_formatter(
        mticker.LogFormatter(base=10, labelOnlyBase=True)
    )
    ax.xaxis.set_minor_locator(
        mticker.NullLocator()
    )

    ax.set_yscale(
        "log",
        base=10,
    )

    ax.set_ylim(
        bottom=2.2,
        top=5.7,
    )

    ax.set_ylabel(
        "Validation loss"
    )

    ax.yaxis.set_major_locator(
        mticker.LogLocator(
            base=10,
            subs=(1.0, 2.0, 3.0, 4.0, 5.0),
        )
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, _: f"{value:g}"
        )
    )
    ax.yaxis.set_minor_locator(
        mticker.NullLocator()
    )

    small_center = np.sqrt(
        x_min
        * first
    )

    large_center = np.sqrt(
        first
        * second
    )

    ultra_center = np.sqrt(
        second
        * x_max
    )

    region_labels = [
        (
            small_center,
            "small batch",
        ),
        (
            large_center,
            "large batch",
        ),
        (
            ultra_center,
            "ultra-large batch",
        ),
    ]

    for x_position, label in region_labels:
        ax.text(
            x_position,
            0.045,
            label,
            transform=ax.get_xaxis_transform(),
            fontsize=10.5,
            fontweight="semibold",
            color="0.05",
            ha="center",
            va="bottom",
            zorder=4,
        )

    # ---------------------------------------------------------
    # Appearance
    # ---------------------------------------------------------
    ax.grid(
        False,
        which="both",
    )

    ax.legend(
        loc="upper left",
        frameon=True,
    )

    fig.tight_layout()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        output_dir
        / f"{OUTPUT_STEM}.pdf"
    )

    png_path = (
        output_dir
        / f"{OUTPUT_STEM}.png"
    )

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

    return (
        pdf_path,
        png_path,
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--boundaries",
        type=float,
        nargs=2,
        metavar=(
            "SMALL_END",
            "ULTRA_START",
        ),
        default=None,
        help=(
            "Optional manual batch-size boundaries. "
            "If omitted, boundaries are automatically placed "
            "between the 6th/7th and 10th/11th batch points."
        ),
    )

    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error(
            "--dpi must be positive"
        )

    data_dir = (
        args.data_dir
        .expanduser()
        .resolve()
    )

    csv_path = (
        data_dir
        / "best_by_batch.csv"
    )

    loss_df = (
        load_optimal_losses(
            csv_path
        )
    )

    print(
        loss_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    pdf_path, png_path = plot_cbs(
        loss_df,
        data_dir,
        dpi=args.dpi,
        boundaries=args.boundaries,
    )

    print(
        f"read {csv_path}"
    )

    print(
        f"wrote {pdf_path}"
    )

    print(
        f"wrote {png_path}"
    )


if __name__ == "__main__":
    main()