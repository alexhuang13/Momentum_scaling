"""Empirical NN stability plots; censoring and missing trials remain visible."""

from __future__ import annotations

from pathlib import Path


def plot(root: Path, manifest: dict, cells: list[dict], boundaries: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"sgd": "#767676", "polyak": "#3775BA", "nesterov": "#B64342"}
    names = {"sgd": "SGD", "polyak": "Polyak", "nesterov": "Nesterov"}
    markers = {"sgd": "^", "polyak": "o", "nesterov": "s"}
    settings = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    label = (
        "SMOKE TEST — no scientific inference"
        if manifest["config"]["smoke_test"]
        else (
            f"Empirical stability, K={manifest['config']['steps']}, "
            f"required seed fraction={manifest['config']['required_fraction']:g}"
        )
    )
    with plt.rc_context(settings):
        for axis in dict.fromkeys(p["axis"] for p in manifest["points"]):
            points = [p for p in manifest["points"] if p["axis"] == axis]
            xs = [p["x"] for p in points]
            xlabel = "Batch size B" if axis == "batch" else r"Momentum gap $1-\rho$"
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            for method_index, algorithm in enumerate(manifest["algorithms"]):
                # Keep coincident brackets visible; tick labels retain true x values.
                offset = np.exp(
                    0.045 * (method_index - (len(manifest["algorithms"]) - 1) / 2)
                )
                rows = [
                    r
                    for r in boundaries
                    if r["axis"] == axis and r["algorithm"] == algorithm
                ]
                # Plot brackets as resolution intervals, never as confidence intervals.
                good = [r for r in rows if r["status"] == "bracketed"]
                if good:
                    x = np.array([r["x"] for r in good]) * offset
                    low = np.array([r["lower"] for r in good])
                    high = np.array([r["upper"] for r in good])
                    y = np.sqrt(low * high)
                    ax.errorbar(
                        x,
                        y,
                        yerr=np.stack((y - low, high - y)),
                        linestyle="none",
                        marker=markers[algorithm],
                        color=colors[algorithm],
                        capsize=4,
                    )
                for row in rows:
                    status = row["status"]
                    if status in ("left_censored", "right_censored"):
                        y = row["upper"] if status == "left_censored" else row["lower"]
                        ax.scatter(
                            row["x"] * offset,
                            y,
                            marker="v" if status == "left_censored" else "^",
                            facecolors="none",
                            edgecolors=colors[algorithm],
                            s=70,
                        )
                    elif status == "nonmonotone":
                        ax.scatter(
                            row["x"] * offset,
                            row["largest_observed_stable_lr"],
                            marker="x",
                            color=colors[algorithm],
                            s=65,
                        )
                ax.plot(
                    [],
                    [],
                    marker=markers[algorithm],
                    linestyle="none",
                    color=colors[algorithm],
                    label=names[algorithm],
                )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel=xlabel,
                ylabel="Empirical learning-rate boundary",
                title=label,
            )
            ax.set_xlim(min(xs) / 1.2, max(xs) * 1.2)
            # Empty/incomplete scans still need positive limits on the log axis.
            ax.set_ylim(manifest["lrs"][0] / 1.2, manifest["lrs"][-1] * 1.2)
            ax.set_xticks(xs, [f"{x:.4g}" for x in xs])
            ax.minorticks_off()
            ax.legend()
            ax.grid(alpha=0.15)
            fig.text(
                0.5,
                0.02,
                "Bars: LR grid brackets; open triangles: scan limits; ×: nonmonotone\n"
                "Markers: geometric bracket centers; methods offset horizontally.\n"
                "Incomplete grids omitted; see stability-fraction maps and CSVs.",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.12, 1, 1))
            for extension in ("png", "pdf"):
                fig.savefig(root / f"boundary_vs_{axis}.{extension}", dpi=300)
            plt.close(fig)

            fig, axes = plt.subplots(
                1,
                len(manifest["algorithms"]),
                figsize=(5 * len(manifest["algorithms"]), 5),
                squeeze=False,
                constrained_layout=True,
            )
            for ax, algorithm in zip(axes[0], manifest["algorithms"]):
                grid = np.full((len(manifest["lrs"]), len(xs)), np.nan)
                for cell in cells:
                    if (
                        cell["axis"] == axis
                        and cell["algorithm"] == algorithm
                        and cell["complete"]
                    ):
                        grid[manifest["lrs"].index(cell["lr"]), xs.index(cell["x"])] = (
                            cell["bounded_fraction"]
                        )
                cmap = plt.get_cmap("cividis").copy()
                cmap.set_bad("#dddddd")
                im = ax.imshow(
                    np.ma.masked_invalid(grid),
                    origin="lower",
                    aspect="auto",
                    vmin=0,
                    vmax=1,
                    cmap=cmap,
                    interpolation="nearest",
                )
                ax.set_xticks(
                    range(len(xs)), [f"{x:g}" for x in xs], rotation=45, ha="right"
                )
                ticks = np.unique(
                    np.linspace(
                        0, len(manifest["lrs"]) - 1, min(7, len(manifest["lrs"]))
                    ).astype(int)
                )
                ax.set_yticks(ticks, [f"{manifest['lrs'][i]:.3g}" for i in ticks])
                ax.set(
                    title=names[algorithm],
                    xlabel=xlabel,
                    ylabel="Learning rate (grid index spacing)",
                )
            fig.colorbar(
                im,
                ax=axes[0].tolist(),
                label="Bounded-run fraction (gray: incomplete)",
                shrink=0.8,
            )
            fig.suptitle(label)
            for extension in ("png", "pdf"):
                fig.savefig(root / f"stable_fraction_vs_{axis}.{extension}", dpi=300)
            plt.close(fig)
