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
    excess_loss, lookahead, run_aligned_replicas, update,
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


def _run_setting(
    problem: Problem, *, algorithm: str, D: int, batch_size: int,
    eta: float, one_minus_rho: float, seeds: int, seed: int, tail_fraction: float,
) -> dict:
    batch_size = max(1, min(int(batch_size), int(D)))
    steps = max(1, int(D) // batch_size)
    rho = 0.0 if algorithm == "sgd" else 1.0 - one_minus_rho
    values = run_aligned_replicas(
        problem, algorithm=algorithm, batch_size=batch_size, steps=steps,
        eta=eta, rho=rho, seeds=seeds, seed=seed, tail_fraction=tail_fraction,
    )
    return {
        "D": int(D), "B": batch_size, "K": steps, "eta": eta, "rho": rho,
        "one_minus_rho": 1.0 - rho, **_summary(values),
    }


def data_scaling_experiment(runtime: Runtime, *, quick: bool) -> dict:
    beta, s, sigma = 4.0, 0.3, 0.1
    D_values = [2000, 4000, 8000] if quick else [2000, 4000, 8000, 16000, 32000, 64000, 128000]
    seeds = 4 if quick else 100
    dim = 256 if quick else 4096
    problem = build_problem(
        beta=beta, s=s, sigma=sigma, dim=dim, device=runtime.device, dtype=runtime.dtype
    )
    p = s / (s + 1.0)
    polyak: dict[str, dict[str, list[dict]]] = {
        "small_batch": {"SGD": [], "Polyak": []},
        "large_batch": {"SGD": [], "Polyak": []},
    }
    for d_index, D in enumerate(D_values):
        print(f"[data-scaling: Polyak] D={D}", flush=True)
        settings = {
            "small_batch": {
                "batch_size": round(D**p),
                "sgd": (0.1, 1.0),
                "polyak": (0.1 * D ** (-p), D ** (-p)),
            },
            "large_batch": {
                "batch_size": round(D ** (2.0 * p)),
                "sgd": (0.5, 1.0),
                "polyak": (0.1, D ** (-p)),
            },
        }
        for regime, setting in settings.items():
            for a_index, algorithm in enumerate(("sgd", "polyak")):
                eta, omrho = setting[algorithm]
                polyak[regime][DISPLAY_NAMES[algorithm]].append(_run_setting(
                    problem, algorithm=algorithm, D=D, batch_size=setting["batch_size"],
                    eta=eta, one_minus_rho=omrho, seeds=seeds,
                    seed=runtime.seed + 10_000 * d_index + 100 * a_index,
                    tail_fraction=0.0,
                ))

    epsilon = 0.05
    a = (2.0 * s * beta + 1.0) / (2.0 * s * beta + beta + 1.0) - epsilon
    x = (s * beta - (beta - 1.0) * a) / (s * beta + 1.0)
    nesterov: dict[str, list[dict]] = {DISPLAY_NAMES[name]: [] for name in ALGORITHMS}
    for d_index, D in enumerate(D_values):
        print(f"[data-scaling: Nesterov] D={D}", flush=True)
        batch_size = round(0.15 * D**a)
        settings = {
            "sgd": (0.3, 1.0),
            "polyak": (0.1, D ** (-(a - p))),
            "nesterov": (0.1, 2.0 * D ** (-(a - x))),
        }
        for a_index, algorithm in enumerate(ALGORITHMS):
            eta, omrho = settings[algorithm]
            nesterov[DISPLAY_NAMES[algorithm]].append(_run_setting(
                problem, algorithm=algorithm, D=D, batch_size=batch_size,
                eta=eta, one_minus_rho=omrho, seeds=seeds,
                seed=runtime.seed + 1_000_000 + 10_000 * d_index + 100 * a_index,
                tail_fraction=0.01,
            ))

    payload = {
        "experiment": "data_scaling", "preset": "quick" if quick else "paper",
        "environment": _environment(runtime),
        "config": {"beta": beta, "s": s, "sigma": sigma, "dim": dim, "D": D_values, "seeds": seeds},
        "polyak_critical_batch": polyak,
        "nesterov_large_batch": {"batch_exponent": a, "x": x, "results": nesterov},
    }
    _save_json(runtime.output_dir / "data_scaling.json", payload)
    return payload
