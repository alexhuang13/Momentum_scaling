"""Small plotting layer kept separate from numerical experiments."""

from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile

_MPL_DIR = Path(tempfile.gettempdir()) / "momentum-scaling-matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"SGD": "#E3A018", "Polyak": "#D94B3D", "Nesterov": "#2F5DA8"}
MARKERS = {"SGD": "^", "Polyak": "o", "Nesterov": "s"}


def _save(fig, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_stability(payload: dict, output_dir: Path) -> None:
    b = np.asarray(payload["config"]["batch_sizes"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for name, values in payload["boundary"].items():
        ax.plot(b, values, marker=MARKERS[name], color=COLORS[name], label=name)
    ax.set(xscale="log", yscale="log", xlabel="batch size B", ylabel="largest stable learning rate")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    _save(fig, output_dir / "stability")


def _curve(ax, records, name):
    d = np.asarray([r["D"] for r in records], dtype=float)
    mean = np.asarray([r["mean"] for r in records])
    std = np.asarray([r["std"] for r in records])
    ax.errorbar(d, mean, yerr=std, marker=MARKERS[name], color=COLORS[name], label=name, capsize=2)


def plot_data_scaling(payload: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    polyak = payload["polyak_critical_batch"]
    for ax, regime, title in zip(axes[:2], ("small_batch", "large_batch"), ("Polyak: smaller batch", "Polyak: larger batch")):
        for name, records in polyak[regime].items(): _curve(ax, records, name)
        ax.set_title(title)
    for name, records in payload["nesterov_large_batch"]["results"].items(): _curve(axes[2], records, name)
    axes[2].set_title("Nesterov-optimal large batch")
    for ax in axes:
        ax.set(xscale="log", yscale="log", xlabel="data budget D", ylabel="final excess risk")
        ax.grid(True, which="both", alpha=0.25); ax.legend()
    _save(fig, output_dir / "data_scaling")


def plot_saved(output_dir: Path) -> None:
    stability = output_dir / "stability.json"
    scaling = output_dir / "data_scaling.json"
    if stability.exists(): plot_stability(json.loads(stability.read_text()), output_dir)
    if scaling.exists(): plot_data_scaling(json.loads(scaling.read_text()), output_dir)
