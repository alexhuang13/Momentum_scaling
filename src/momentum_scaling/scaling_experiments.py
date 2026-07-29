"""Focused data-scaling experiments derived from the paper notebooks.

This module contains only schedules and numerical orchestration. The shared
power-law model and SGD/Polyak/Nesterov updates live in :mod:`core`.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import zlib

import numpy as np
import torch

from .core import Problem, build_problem, run_aligned_replicas

D_PAPER = [2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000]


def _environment(device: torch.device, dtype: torch.dtype, seed: int) -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "seed": seed,
    }


def _seed(base: int, *items: object) -> int:
    text = "|".join(str(item) for item in items)
    return int((base + zlib.crc32(text.encode("utf-8"))) % (2**31 - 1))


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fit_exponent(records: list[dict]) -> float:
    data = np.asarray([record["D"] for record in records], dtype=float)
    losses = np.asarray([record["mean"] for record in records], dtype=float)
    slope, _ = np.polyfit(np.log(data), np.log(losses), 1)
    return -float(slope)


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "values": values.tolist(),
    }


def _run(
    problem: Problem,
    *,
    algorithm: str,
    D: int,
    B: int,
    K: int,
    eta: float,
    one_minus_rho: float,
    runs: int,
    seed: int,
    tail_fraction: float,
) -> dict:
    B = max(1, int(B))
    K = max(1, min(int(K), int(D)))
    rho = 0.0 if algorithm == "sgd" else 1.0 - float(one_minus_rho)
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"Invalid rho={rho} for {algorithm}")
    values = run_aligned_replicas(
        problem,
        algorithm=algorithm,
        batch_size=B,
        steps=K,
        eta=float(eta),
        rho=rho,
        seeds=runs,
        seed=seed,
        tail_fraction=tail_fraction,
    )
    return {
        "D": int(D),
        "B": B,
        "K": K,
        "used_budget": B * K,
        "eta": float(eta),
        "rho": rho,
        "one_minus_rho": 1.0 - rho,
        **_summary(values),
    }


def _base_config(quick: bool) -> dict:
    return {
        "beta": 4.0,
        "s": 0.3,
        "sigma": 0.1,
        "dim": 256 if quick else 4096,
        "runs": 4 if quick else 100,
        "D": D_PAPER[:3] if quick else D_PAPER,
    }


def polyak_scaling_experiment(
    *, device: torch.device, dtype: torch.dtype, seed: int, output_dir: Path, quick: bool,
) -> dict:
    """Compare SGD and Polyak at large and data-optimal batch scalings.

    This is the canonical implementation of the four-configuration Polyak study.
    Each record reports the final excess loss of every independent trajectory.
    """
    config = _base_config(quick)
    problem = build_problem(
        beta=config["beta"], s=config["s"], sigma=config["sigma"], dim=config["dim"],
        device=device, dtype=dtype,
    )
    x = config["s"] / (config["s"] + 1.0)
    groups = {
        "sgd_large_batch": {
            "label": "SGD large batch", "algorithm": "sgd",
            "step_constant": 1.0, "step_exponent": 1.0 - 2.0 * x,
            "eta_constant": 0.5, "eta_exponent": 0.0,
            "gap_constant": 1.0, "gap_exponent": 0.0,
            "theory_exponent": config["s"] * (1.0 - 2.0 * x),
        },
        "polyak_large_batch": {
            "label": "Polyak large batch", "algorithm": "polyak",
            "step_constant": 1.0, "step_exponent": 1.0 - 2.0 * x,
            "eta_constant": 0.1, "eta_exponent": 0.0,
            "gap_constant": 1.0, "gap_exponent": x,
            "theory_exponent": x,
        },
        "sgd_small_batch": {
            "label": "SGD small batch", "algorithm": "sgd",
            "step_constant": 1.0, "step_exponent": 1.0 - x,
            "eta_constant": 0.1, "eta_exponent": 0.0,
            "gap_constant": 1.0, "gap_exponent": 0.0,
            "theory_exponent": x,
        },
        "polyak_small_batch": {
            "label": "Polyak small batch", "algorithm": "polyak",
            "step_constant": 1.05, "step_exponent": 1.0 - x,
            "eta_constant": 0.1, "eta_exponent": x,
            "gap_constant": 1.0, "gap_exponent": x,
            "theory_exponent": x,
        },
    }
    results: dict[str, list[dict]] = {name: [] for name in groups}
    for group_index, (name, schedule) in enumerate(groups.items()):
        print(f"[polyak-scaling] {schedule['label']}", flush=True)
        for d_index, D in enumerate(config["D"]):
            K = min(D, max(1, round(schedule["step_constant"] * D ** schedule["step_exponent"])))
            B = max(1, D // K)
            eta = schedule["eta_constant"] * D ** (-schedule["eta_exponent"])
            gap = min(1.0, schedule["gap_constant"] * D ** (-schedule["gap_exponent"]))
            record = _run(
                problem, algorithm=schedule["algorithm"], D=D, B=B, K=K, eta=eta,
                one_minus_rho=gap, runs=config["runs"],
                seed=_seed(seed, "polyak-scaling", group_index, d_index, D),
                tail_fraction=0.0,
            )
            results[name].append(record)
            print(f"  D={D:6d} B={B:4d} K={K:5d} loss={record['mean']:.6e}", flush=True)
    measured = {name: _fit_exponent(records) for name, records in results.items()}
    payload = {
        "experiment": "polyak_scaling",
        "preset": "quick" if quick else "paper",
        "environment": _environment(device, dtype, seed),
        "config": config,
        "groups": groups,
        "measured_exponents": measured,
        "results": results,
    }
    _save_json(output_dir / "polyak_scaling.json", payload)
    return payload


def nesterov_scaling_experiment(
    *, device: torch.device, dtype: torch.dtype, seed: int, output_dir: Path, quick: bool,
) -> dict:
    """Compare all methods at a shared Nesterov-optimal batch scaling.

    Each trajectory is evaluated by its mean loss over the final 1% of steps.
    This is the canonical implementation of the shared-batch Nesterov study.
    """
    config = _base_config(quick)
    config.update({"epsilon": 0.05, "tail_fraction": 0.01, "batch_constant": 0.15})
    problem = build_problem(
        beta=config["beta"], s=config["s"], sigma=config["sigma"], dim=config["dim"],
        device=device, dtype=dtype,
    )
    beta, s = config["beta"], config["s"]
    optimal = s / (s + 1.0)
    max_batch = (2.0 * s * beta + 1.0) / (2.0 * s * beta + beta + 1.0)
    a = max_batch - config["epsilon"]
    x = (s * beta - (beta - 1.0) * a) / (s * beta + 1.0)
    groups = {
        "sgd": {
            "label": "SGD", "algorithm": "sgd", "eta_constant": 0.3,
            "eta_exponent": max(optimal - a, 0.0), "gap_constant": 1.0,
            "gap_exponent": 0.0, "theory_exponent": s * (1.0 - a),
        },
        "polyak": {
            "label": "Polyak", "algorithm": "polyak", "eta_constant": 0.1,
            "eta_exponent": max(optimal - a, 0.0), "gap_constant": 1.0,
            "gap_exponent": a - optimal, "theory_exponent": optimal,
        },
        "nesterov": {
            "label": "Nesterov", "algorithm": "nesterov", "eta_constant": 0.1,
            "eta_exponent": 0.0, "gap_constant": 2.0,
            "gap_exponent": a - x, "theory_exponent": s * (1.0 - x),
        },
    }
    results: dict[str, list[dict]] = {name: [] for name in groups}
    for group_index, (name, schedule) in enumerate(groups.items()):
        print(f"[nesterov-scaling] {schedule['label']}", flush=True)
        for d_index, D in enumerate(config["D"]):
            B = max(1, round(config["batch_constant"] * D**a))
            K = max(1, D // B)
            eta = schedule["eta_constant"] * D ** (-schedule["eta_exponent"])
            gap = min(1.0, schedule["gap_constant"] * D ** (-schedule["gap_exponent"]))
            record = _run(
                problem, algorithm=schedule["algorithm"], D=D, B=B, K=K, eta=eta,
                one_minus_rho=gap, runs=config["runs"],
                seed=_seed(seed, "nesterov-scaling", group_index, d_index, D),
                tail_fraction=config["tail_fraction"],
            )
            results[name].append(record)
            print(f"  D={D:6d} B={B:4d} K={K:5d} loss={record['mean']:.6e}", flush=True)
    measured = {name: _fit_exponent(records) for name, records in results.items()}
    payload = {
        "experiment": "nesterov_scaling",
        "preset": "quick" if quick else "paper",
        "environment": _environment(device, dtype, seed),
        "config": {**config, "max_batch_exponent": max_batch, "shared_batch_exponent": a, "x": x},
        "groups": groups,
        "measured_exponents": measured,
        "results": results,
    }
    _save_json(output_dir / "nesterov_scaling.json", payload)
    return payload
