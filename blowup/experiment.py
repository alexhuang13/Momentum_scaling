"""Increasing-momentum blowup diagnostic.

The experiment uses the schedule ``rho_k = 1 - 1 / (k + 1)`` and records
the median population loss across independent replicas. Polyak eventually
becomes unstable under this schedule, while Nesterov remains damped for much
longer in the paper configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch

from .core import (
    ALGORITHMS,
    DISPLAY_NAMES,
    Problem,
    aligned_noise_gradient,
    build_problem,
    excess_loss,
    lookahead,
    update,
)


@dataclass(frozen=True)
class BlowupRuntime:
    device: torch.device
    dtype: torch.dtype
    seed: int
    output_dir: Path


def momentum_at(step: int) -> float:
    """Return the increasing momentum coefficient used at a one-based step."""
    if step < 1:
        raise ValueError(f"step must be positive, got {step}")
    return 1.0 - 1.0 / (step + 1.0)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


@torch.no_grad()
def simulate_increasing_momentum(
    problem: Problem,
    *,
    algorithm: str,
    batch_size: int,
    eta: float,
    steps: int,
    replicas: int,
    record_every: int,
    seed: int,
    blowup_loss: float,
) -> dict[str, np.ndarray | int | None]:
    """Run one optimizer and return its recorded median-loss trajectory."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if min(batch_size, steps, replicas, record_every) < 1:
        raise ValueError(
            "batch_size, steps, replicas, and record_every must be positive"
        )

    generator = _generator(problem.lam.device, seed)
    delta = -problem.theta_star.expand(replicas, -1).clone()
    previous = delta.clone()
    initial = float(excess_loss(delta, problem.lam).median().item())
    recorded_steps = [0]
    median_losses = [initial]
    blowup_step: int | None = None

    for step in range(1, steps + 1):
        rho = momentum_at(step)
        evaluation = lookahead(delta, previous, algorithm, rho)
        gradient = aligned_noise_gradient(
            evaluation,
            problem,
            batch_size=batch_size,
            generator=generator,
        )
        new_delta = update(
            delta,
            previous,
            gradient,
            algorithm=algorithm,
            eta=eta,
            rho=rho,
        )
        previous, delta = delta, new_delta

        if step % record_every != 0 and step != steps:
            continue

        loss = float(excess_loss(delta, problem.lam).median().item())
        recorded_steps.append(step)
        median_losses.append(loss)
        if not math.isfinite(loss) or loss >= blowup_loss:
            blowup_step = step
            break

    return {
        "steps": np.asarray(recorded_steps, dtype=np.int64),
        "median_loss": np.asarray(median_losses, dtype=float),
        "blowup_step": blowup_step,
    }


def _config(quick: bool) -> dict[str, float | int]:
    return {
        "beta": 2.0,
        "s": 0.75,
        "sigma": 0.0,
        "dim": 256 if quick else 2048,
        "eta": 0.05,
        "batch_size": 32,
        "steps": 2_000 if quick else 500_000,
        "replicas": 4 if quick else 50,
        "record_every": 10 if quick else 1,
        "blowup_loss": 10.0,
    }


def _environment(runtime: BlowupRuntime) -> dict[str, str | int]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(runtime.device),
        "dtype": str(runtime.dtype),
        "seed": runtime.seed,
    }


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def blowup_experiment(runtime: BlowupRuntime, *, quick: bool) -> dict:
    """Run SGD, Polyak, and Nesterov under increasing momentum."""
    config = _config(quick)
    problem = build_problem(
        beta=float(config["beta"]),
        s=float(config["s"]),
        sigma=float(config["sigma"]),
        dim=int(config["dim"]),
        device=runtime.device,
        dtype=runtime.dtype,
    )
    results = {}
    for index, algorithm in enumerate(ALGORITHMS):
        print(f"[blowup] {DISPLAY_NAMES[algorithm]}", flush=True)
        result = simulate_increasing_momentum(
            problem,
            algorithm=algorithm,
            batch_size=int(config["batch_size"]),
            eta=float(config["eta"]),
            steps=int(config["steps"]),
            replicas=int(config["replicas"]),
            record_every=int(config["record_every"]),
            seed=runtime.seed + index * 10_000,
            blowup_loss=float(config["blowup_loss"]),
        )
        results[algorithm] = result
        print(
            f"  final recorded loss={result['median_loss'][-1]:.6e} "
            f"blowup_step={result['blowup_step']}",
            flush=True,
        )

    runtime.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for algorithm, result in results.items():
        arrays[f"{algorithm}_steps"] = result["steps"]
        arrays[f"{algorithm}_median_loss"] = result["median_loss"]
    np.savez_compressed(runtime.output_dir / "blowup.npz", **arrays)

    summary = {
        "experiment": "blowup",
        "preset": "quick" if quick else "paper",
        "environment": _environment(runtime),
        "config": config,
        "results": {
            algorithm: {
                "blowup_step": result["blowup_step"],
                "final_recorded_step": int(result["steps"][-1]),
                "final_median_loss": _finite_or_none(float(result["median_loss"][-1])),
            }
            for algorithm, result in results.items()
        },
    }
    (runtime.output_dir / "blowup.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return {**summary, "trajectories": results}
