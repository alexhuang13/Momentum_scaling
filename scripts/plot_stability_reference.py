#!/usr/bin/env python3
"""Plot the reference stability scans with colored curves and no background fills."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("SGD", "Polyak", "Nesterov")
COLORS = {"SGD": "tab:green", "Polyak": "tab:blue", "Nesterov": "tab:orange"}
MARKERS = {"SGD": "^", "Polyak": "o", "Nesterov": "s"}
AVAILABLE_SETTINGS = (
    (1.5, 2.0),
    (1.5, 3.0),
    (1.5, 5.0),
    (2.0, 2.0),
    (2.0, 3.0),
    (2.0, 5.0),
    (2.5, 3.0),
    (2.5, 5.0),
)


def number_tag(value: float) -> str:
    """Convert 1.5 to 1_5 and 2.0 to 2 for filenames."""
    return format(float(value), ".15g").replace(".", "_")


def paths_for(beta: float, s: float) -> tuple[Path, Path]:
    beta_tag = number_tag(beta)
    s_tag = number_tag(s)
    data_path = Path(f"assets/results/stability_beta{beta_tag}_s{s_tag}_reference.npz")
    output_stem = Path(f"assets/figures/beta{beta_tag}s{s_tag}_plot")
    return data_path, output_stem


def mean_max_stable_eta(stable: np.ndarray, eta_grid: np.ndarray) -> np.ndarray:
    """Average each seed's largest stable learning rate at every batch size."""
    per_seed = np.full((stable.shape[0], stable.shape[2]), np.nan)
    for seed_index in range(stable.shape[0]):
        for batch_index in range(stable.shape[2]):
            indices = np.flatnonzero(stable[seed_index, :, batch_index])
            if indices.size:
                per_seed[seed_index, batch_index] = eta_grid[indices[-1]]
    counts = np.sum(np.isfinite(per_seed), axis=0)
    totals = np.nansum(per_seed, axis=0)
    return np.divide(totals, counts, out=np.full(per_seed.shape[1], np.nan), where=counts > 0)


def fit_prefactor(
    boundary: np.ndarray,
    batch_sizes: np.ndarray,
    *,
    power: float,
    one_minus_rho: float,
    indices: np.ndarray | None = None,
) -> float:
    indices = np.arange(boundary.size) if indices is None else indices
    values = boundary[indices]
    batches = batch_sizes[indices]
    mask = np.isfinite(values) & (values > 0)
    if not np.any(mask):
        raise ValueError("cannot fit a prediction line without finite positive boundary values")
    denominator = batches[mask] ** power * one_minus_rho
    return float(np.exp(np.mean(np.log(values[mask] / denominator))))


def fit_horizontal_level(boundary: np.ndarray, indices: np.ndarray) -> float:
    values = boundary[indices]
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        raise ValueError("cannot fit a horizontal prediction without finite positive values")
    return float(np.exp(np.mean(np.log(values))))


def plot(data_path: Path, output_stem: Path) -> None:
    with np.load(data_path, allow_pickle=False) as data:
        stable = np.asarray(data["stable"], dtype=bool)
        algorithms = tuple(str(value) for value in data["algorithms"])
        batch_sizes = np.asarray(data["batch_sizes"], dtype=float)
        eta_grid = np.asarray(data["eta_grid"], dtype=float)
        beta = float(data["beta"])
        one_minus_rho = float(data["one_minus_rho"])

    if algorithms != ALGORITHMS:
        raise ValueError(f"expected algorithms {ALGORITHMS}, got {algorithms}")

    boundaries = {
        name: mean_max_stable_eta(stable[index], eta_grid)
        for index, name in enumerate(algorithms)
    }

    # SGD has a batch-size-independent deterministic stability ceiling.
    sgd_indices = np.flatnonzero(np.isfinite(boundaries["SGD"]))
    sgd_level = fit_horizontal_level(boundaries["SGD"], sgd_indices)
    sgd_prediction = np.full(batch_sizes.shape, sgd_level, dtype=float)

    # Polyak follows eta proportional to B(1-rho).
    polyak_coefficient = fit_prefactor(
        boundaries["Polyak"], batch_sizes,
        power=1.0, one_minus_rho=one_minus_rho,
    )
    polyak_prediction = polyak_coefficient * batch_sizes * one_minus_rho

    # Nesterov follows B**beta at small B and saturates at large B.
    power_indices = np.flatnonzero(batch_sizes <= 32)
    flat_indices = np.flatnonzero(batch_sizes >= 64)
    nesterov_coefficient = fit_prefactor(
        boundaries["Nesterov"], batch_sizes,
        power=beta, one_minus_rho=one_minus_rho, indices=power_indices,
    )
    nesterov_level = fit_horizontal_level(boundaries["Nesterov"], flat_indices)
    intersection = (
        nesterov_level / (nesterov_coefficient * one_minus_rho)
    ) ** (1.0 / beta)
    nesterov_batches = np.sort(np.unique(np.append(batch_sizes, intersection)))
    nesterov_prediction = np.minimum(
        nesterov_coefficient * nesterov_batches ** beta * one_minus_rho,
        nesterov_level,
    )

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    for name in algorithms:
        boundary = boundaries[name]
        mask = np.isfinite(boundary) & (boundary > 0)
        ax.plot(
            batch_sizes[mask], boundary[mask],
            linestyle="None",
            marker=MARKERS[name],
            markersize=6,
            color=COLORS[name],
            markerfacecolor=COLORS[name],
            markeredgecolor=COLORS[name],
            markeredgewidth=1.1,
            label="_nolegend_",
            zorder=3,
        )

    ax.plot(
        batch_sizes, sgd_prediction,
        color=COLORS["SGD"], linestyle="--", linewidth=1.8,
        label="SGD Prediction", zorder=2,
    )
    ax.plot(
        batch_sizes, polyak_prediction,
        color=COLORS["Polyak"], linestyle="--", linewidth=1.8,
        label="Polyak Prediction", zorder=2,
    )
    ax.plot(
        nesterov_batches, nesterov_prediction,
        color=COLORS["Nesterov"], linestyle="--", linewidth=1.8,
        label="Nesterov Prediction", zorder=2,
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(2**2, 2**10)
    exponents = np.arange(2, 11)
    ax.set_xticks(2.0**exponents)
    ax.set_xticklabels([rf"$2^{{{exponent}}}$" for exponent in exponents])
    ax.set_ylim(float(eta_grid.min()), float(eta_grid.max()))
    ax.set_xlabel(r"Batch Size  $B$")
    ax.set_ylabel(r"Stability Boundary  $\eta$")
    ax.grid(False)
    ax.legend(loc="lower right", handlelength=2.8)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_stem.with_suffix('.png')} and {output_stem.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--s", type=float, default=2.0)
    parser.add_argument("--all", action="store_true", help="plot every preserved reference scan")
    args = parser.parse_args()

    settings = AVAILABLE_SETTINGS if args.all else ((args.beta, args.s),)
    if not args.all and (args.beta, args.s) not in AVAILABLE_SETTINGS:
        choices = ", ".join(f"(beta={beta:g}, s={s:g})" for beta, s in AVAILABLE_SETTINGS)
        parser.error(f"no preserved data for beta={args.beta:g}, s={args.s:g}; available: {choices}")

    for beta, s in settings:
        data_path, output_stem = paths_for(beta, s)
        if not data_path.is_file():
            raise FileNotFoundError(f"missing reference data: {data_path}")
        plot(data_path, output_stem)


if __name__ == "__main__":
    main()
