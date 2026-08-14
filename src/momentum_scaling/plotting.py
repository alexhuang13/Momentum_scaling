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
    """Plot colored stability boundaries without background region fills."""
    b = np.asarray(payload["config"]["batch_sizes"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for name, values in payload["boundary"].items():
        ax.plot(
            b, values,
            color=COLORS[name],
            marker=MARKERS[name],
            label=name,
        )
    ax.set(xscale="log", yscale="log", xlabel="batch size B", ylabel="largest stable learning rate")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    _save(fig, output_dir / "stability")



SCALING_MARKERS = {
    "sgd_large_batch": "o",
    "polyak_large_batch": "^",
    "sgd_small_batch": "s",
    "polyak_small_batch": "D",
    "sgd": "o",
    "polyak": "^",
    "nesterov": "s",
}


def plot_scaling_experiment(payload: dict, output_dir: Path) -> None:
    """Plot measured points and theory lines for a focused scaling experiment."""
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    anchor_index = min(3, len(payload["config"]["D"]) - 1)
    for name, records in payload["results"].items():
        group = payload["groups"][name]
        data = np.asarray([record["D"] for record in records], dtype=float)
        loss = np.asarray([record["mean"] for record in records], dtype=float)
        points, = ax.plot(
            data, loss, linestyle="None", marker=SCALING_MARKERS[name],
            label=group["label"], zorder=3,
        )
        exponent = float(group["theory_exponent"])
        reference = loss[anchor_index] * (data / data[anchor_index]) ** (-exponent)
        ax.plot(data, reference, "--", color=points.get_color(), zorder=2)
    ax.set(xscale="log", yscale="log", xlabel="Data budget D", ylabel="Excess loss")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    _save(fig, output_dir / payload["experiment"])


def plot_scaling_law(payload: dict, output_dir: Path) -> None:
    """Plot long-time empirical loss curves and fitted scaling-law predictions."""
    colors = {
        "polyak": {"empirical": "#f6bd83", "prediction": "#d55e00"},
        "nesterov": {"empirical": "#9ecae1", "prediction": "#0072b2"},
    }
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for algorithm in ("polyak", "nesterov"):
        if algorithm not in payload["results"]:
            continue
        result = payload["results"][algorithm]
        steps = np.asarray(result["steps"], dtype=float)
        empirical = np.asarray(result["empirical_loss"], dtype=float)
        prediction = np.asarray(result["theory_prediction"], dtype=float)
        valid = np.asarray(result["fit_valid_mask"], dtype=bool)
        positive = steps > 0
        ax.plot(
            steps[positive], empirical[positive],
            color=colors[algorithm]["empirical"], alpha=0.55, lw=1.4,
            label=f"{result['label']} simulation",
        )
        ax.plot(
            steps[valid], prediction[valid],
            color=colors[algorithm]["prediction"], lw=1.8,
            label=f"{result['label']} scaling law",
        )
    ax.set(xscale="log", yscale="log", xlabel="Optimization step k", ylabel="Excess loss")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    _save(fig, output_dir / "scaling_law")


def plot_saved(output_dir: Path) -> None:
    stability = output_dir / "stability.json"
    if stability.exists(): plot_stability(json.loads(stability.read_text()), output_dir)
    scaling_law = output_dir / "scaling_law.json"
    if scaling_law.exists(): plot_scaling_law(json.loads(scaling_law.read_text()), output_dir)
    for name in ("polyak_scaling", "nesterov_scaling"):
        path = output_dir / f"{name}.json"
        if path.exists():
            plot_scaling_experiment(json.loads(path.read_text()), output_dir)
