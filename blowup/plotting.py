"""Plot the increasing-momentum blowup experiment."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_MPL_DIR = Path(tempfile.gettempdir()) / "momentum-scaling-matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_blowup(payload: dict, output_dir: Path) -> None:
    """Plot median loss under the increasing-momentum schedule."""
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for algorithm in ("sgd", "polyak", "nesterov"):
        result = payload["trajectories"][algorithm]
        steps = np.asarray(result["steps"], dtype=float)
        loss = np.asarray(result["median_loss"], dtype=float)
        positive = (steps > 0) & np.isfinite(loss) & (loss > 0)
        label = {
            "sgd": "SGD",
            "polyak": "Polyak",
            "nesterov": "Nesterov",
        }[algorithm]
        ax.plot(steps[positive], loss[positive], label=label)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Optimization step k",
        ylabel="Median loss",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "blowup.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "blowup.pdf", bbox_inches="tight")
    plt.close(fig)
