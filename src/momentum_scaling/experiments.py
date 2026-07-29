"""Paper experiments and cache-friendly result serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
import torch
import platform

from .core import (
    ALGORITHMS, DISPLAY_NAMES, Problem, build_problem, exact_minibatch_gradient,
    excess_loss, lookahead, update,
)


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    dtype: torch.dtype
    seed: int
    output_dir: Path



def _environment(runtime: Runtime) -> dict[str, str | int]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(runtime.device),
        "dtype": str(runtime.dtype),
        "seed": runtime.seed,
    }


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _summary(values: np.ndarray) -> dict[str, float | list[float]]:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()), "values": values.tolist()}


def _generator(device: torch.device, seed: int) -> torch.Generator:
    out = torch.Generator(device=device)
    out.manual_seed(int(seed))
    return out


def run_stability_trial(
    problem: Problem, *, algorithm: str, batch_size: int, eta: float, rho: float,
    steps: int, window: int, threshold: float, divergence_loss: float,
    chunk: int, seed: int,
) -> tuple[bool, float]:
    generator = _generator(problem.lam.device, seed)
    delta = -problem.theta_star.clone()
    previous = delta.clone()
    tail: list[float] = []
    with torch.no_grad():
        for _ in range(steps):
            evaluation = lookahead(delta, previous, algorithm, rho)
            gradient = exact_minibatch_gradient(
                evaluation, problem, batch_size=batch_size, chunk=chunk, generator=generator
            )
            new_delta = update(
                delta, previous, gradient, algorithm=algorithm, eta=eta, rho=rho
            )
            previous, delta = delta, new_delta
            loss = float(excess_loss(delta, problem.lam).item())
            if not math.isfinite(loss) or loss > divergence_loss:
                return False, math.inf
            tail.append(loss)
            if len(tail) > window:
                tail.pop(0)
    final_average = float(np.mean(tail))
    return math.isfinite(final_average) and final_average < threshold, final_average


def stability_experiment(runtime: Runtime, *, quick: bool) -> dict:
    config = {
        "beta": 1.5, "s": 2.0, "sigma": 0.0,
        "dim": 256 if quick else 4096,
        "one_minus_rho": 2.0 ** -12 if quick else 2.0 ** -24,
        "batch_sizes": [1, 2, 4, 8] if quick else [2**i for i in range(13)],
        "eta_grid": np.geomspace(2**-8, 2.0, 8).tolist() if quick else np.geomspace(2**-10, 8.0, 50).tolist(),
        "steps": 30 if quick else 2000,
        "window": 10 if quick else 20,
        "threshold": 1.0,
        "divergence_loss": 1e12,
        "seeds": 2 if quick else 5,
        "chunk": 256 if quick else 2048,
    }
    problem = build_problem(
        beta=config["beta"], s=config["s"], sigma=config["sigma"], dim=config["dim"],
        device=runtime.device, dtype=runtime.dtype,
    )
    rho = 1.0 - config["one_minus_rho"]
    stable = np.zeros(
        (len(ALGORITHMS), config["seeds"], len(config["eta_grid"]), len(config["batch_sizes"])),
        dtype=bool,
    )
    final = np.full(stable.shape, np.nan, dtype=float)
    for a, algorithm in enumerate(ALGORITHMS):
        print(f"[stability] {DISPLAY_NAMES[algorithm]}", flush=True)
        for r in range(config["seeds"]):
            for j, batch_size in enumerate(config["batch_sizes"]):
                print(
                    f"  seed {r + 1}/{config['seeds']}  B={batch_size}",
                    flush=True,
                )
                for i, eta in enumerate(config["eta_grid"]):
                    trial_seed = runtime.seed + a * 1_000_000 + r * 100_000 + j * 1_000 + i
                    stable[a, r, i, j], final[a, r, i, j] = run_stability_trial(
                        problem, algorithm=algorithm, batch_size=batch_size, eta=float(eta),
                        rho=0.0 if algorithm == "sgd" else rho, steps=config["steps"],
                        window=config["window"], threshold=config["threshold"],
                        divergence_loss=config["divergence_loss"], chunk=config["chunk"],
                        seed=trial_seed,
                    )
    eta_grid = np.asarray(config["eta_grid"])
    per_seed = np.full((len(ALGORITHMS), config["seeds"], len(config["batch_sizes"])), np.nan)
    for a in range(len(ALGORITHMS)):
        for r in range(config["seeds"]):
            for j in range(len(config["batch_sizes"])):
                indices = np.flatnonzero(stable[a, r, :, j])
                if indices.size:
                    per_seed[a, r, j] = eta_grid[indices[-1]]
    boundary = np.nanmean(per_seed, axis=1)
    payload = {
        "experiment": "stability", "preset": "quick" if quick else "paper",
        "environment": _environment(runtime),
        "config": config,
        "boundary": {DISPLAY_NAMES[name]: boundary[i].tolist() for i, name in enumerate(ALGORITHMS)},
    }
    np.savez_compressed(
        runtime.output_dir / "stability.npz", stable=stable, final_average=final,
        per_seed_boundary=per_seed, mean_boundary=boundary,
        algorithms=np.asarray(ALGORITHMS), batch_sizes=np.asarray(config["batch_sizes"]),
        eta_grid=eta_grid, config=json.dumps(config, sort_keys=True),
    )
    _save_json(runtime.output_dir / "stability.json", payload)
    return payload
