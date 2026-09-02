"""Long-time loss dynamics for Polyak and Nesterov momentum.

This module fixes optimizer hyperparameters and extends training time to show
how the excess loss moves from signal-dominated decay to a noise-dominated
plateau. It backs the notebooks in this directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform

import numpy as np
import torch

from .core import Problem, build_problem, excess_loss, lookahead, update


@dataclass(frozen=True)
class LossDynamicsRuntime:
    device: torch.device
    dtype: torch.dtype
    seed: int
    output_dir: Path


DEFAULT_SETTINGS = {
    "polyak": {"B": 64, "eta": 0.1, "rho": 0.995},
    "nesterov": {"B": 64, "eta": 0.1, "rho": 0.995},
}


def _environment(runtime: LossDynamicsRuntime) -> dict[str, str | int]:
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


def _generator(device: torch.device, seed: int) -> torch.Generator:
    out = torch.Generator(device=device)
    out.manual_seed(int(seed))
    return out


@torch.no_grad()
def simulate_long_time_curve(
    problem: Problem,
    *,
    algorithm: str,
    batch_size: int,
    eta: float,
    rho: float,
    steps: int,
    runs: int,
    seed: int,
    checkpoints: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate independent replicas and return mean excess loss by checkpoint."""
    if checkpoints is None:
        checkpoints = np.arange(steps + 1, dtype=int)
    else:
        checkpoints = np.unique(np.asarray(checkpoints, dtype=int))
        checkpoints = checkpoints[(0 <= checkpoints) & (checkpoints <= steps)]
        if checkpoints.size == 0 or checkpoints[0] != 0:
            checkpoints = np.concatenate(([0], checkpoints))
    checkpoint_set = set(int(k) for k in checkpoints.tolist())

    generator = _generator(problem.lam.device, seed)
    delta = -problem.theta_star.expand(runs, -1).clone()
    previous = delta.clone()

    saved_steps = [0]
    saved_losses = [float(excess_loss(delta, problem.lam).mean().item())]

    for k in range(1, steps + 1):
        evaluation = lookahead(delta, previous, algorithm, rho)
        current_loss = excess_loss(evaluation, problem.lam)
        scale = (
            ((current_loss + problem.sigma**2) / float(batch_size)).sqrt().unsqueeze(-1)
        )
        noise = torch.randn(
            evaluation.shape,
            device=evaluation.device,
            dtype=evaluation.dtype,
            generator=generator,
        )
        gradient = evaluation * problem.lam + noise * problem.lam.sqrt() * scale
        new_delta = update(
            delta, previous, gradient, algorithm=algorithm, eta=eta, rho=rho
        )
        previous, delta = delta, new_delta

        if k in checkpoint_set:
            saved_steps.append(k)
            saved_losses.append(float(excess_loss(delta, problem.lam).mean().item()))

    return np.asarray(saved_steps, dtype=int), np.asarray(saved_losses, dtype=float)


def theory_matrix(
    steps: np.ndarray,
    algorithm: str,
    batch_size: int,
    eta: float,
    rho: float,
    *,
    beta: float,
    s: float,
    sigma: float,
) -> np.ndarray:
    """Return transient, signal, and noise scaling-law columns."""
    k = np.maximum(np.asarray(steps, dtype=float), 1.0)
    eta_rho = eta / (1.0 - rho)
    signal = (eta_rho * k) ** (-s)

    if algorithm == "polyak":
        transient = rho**k
        noise = np.full_like(k, sigma**2 * eta_rho / float(batch_size))
    elif algorithm == "nesterov":
        r_nes = 4.0 * rho**2 / (1.0 + rho) ** 2
        transient = ((1.0 - rho) ** 2 / eta) ** (s + 1.0 / beta) * r_nes**k
        noise_floor = (
            sigma**2 / float(batch_size) * min(eta_rho, eta_rho ** (1.0 / beta))
        )
        noise = np.full_like(k, noise_floor)
    else:
        raise ValueError(
            f"Scaling-law curves are defined for Polyak/Nesterov, got {algorithm}."
        )

    return np.column_stack([transient, signal, noise])


def _nonnegative_lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve a tiny nonnegative least-squares problem by active-set enumeration."""
    best_coef = np.zeros(x.shape[1], dtype=float)
    best_error = np.inf
    for mask in range(1, 1 << x.shape[1]):
        active = [j for j in range(x.shape[1]) if mask & (1 << j)]
        coef_active, *_ = np.linalg.lstsq(x[:, active], y, rcond=None)
        if np.any(coef_active < 0):
            continue
        coef = np.zeros(x.shape[1], dtype=float)
        coef[active] = coef_active
        error = float(np.mean((x @ coef - y) ** 2))
        if error < best_error:
            best_coef, best_error = coef, error
    return best_coef


def fit_scaling_law_prediction(
    steps: np.ndarray,
    empirical: np.ndarray,
    algorithm: str,
    batch_size: int,
    eta: float,
    rho: float,
    *,
    beta: float,
    s: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit O(1) constants for transient, signal, and noise theory terms."""
    x = theory_matrix(
        steps,
        algorithm,
        batch_size,
        eta,
        rho,
        beta=beta,
        s=s,
        sigma=sigma,
    )
    eta_rho = eta / (1.0 - rho)
    valid = (np.asarray(steps) > 0) & (eta_rho * np.asarray(steps) >= 1.0)
    if valid.sum() < x.shape[1]:
        valid = np.asarray(steps) > 0
    if valid.sum() == 0:
        valid = np.ones_like(np.asarray(steps), dtype=bool)
    scale = np.maximum(empirical[valid], np.finfo(float).tiny)
    coefficients = _nonnegative_lstsq(x[valid] / scale[:, None], np.ones(valid.sum()))
    prediction = x @ coefficients
    return prediction, valid, coefficients, x


def _checkpoints(steps: int, quick: bool) -> np.ndarray:
    if quick:
        points = set(range(min(steps, 100) + 1))
        if steps > 0:
            points.update(np.geomspace(1, steps, 180).round().astype(int).tolist())
            points.add(steps)
        return np.asarray(sorted(points), dtype=int)
    return np.arange(steps + 1, dtype=int)


def _config(quick: bool) -> dict:
    return {
        "beta": 4.0,
        "s": 0.3,
        "sigma": 0.5,
        "dim": 256 if quick else 4096,
        "runs": 4 if quick else 100,
        "steps": 2_000 if quick else 500_000,
        "settings": DEFAULT_SETTINGS,
        "checkpoint_every_step": not quick,
    }


def loss_dynamics_experiment(runtime: LossDynamicsRuntime, *, quick: bool) -> dict:
    """Run directly executable long-time scaling-law experiments."""
    config = _config(quick)
    problem = build_problem(
        beta=config["beta"],
        s=config["s"],
        sigma=config["sigma"],
        dim=config["dim"],
        device=runtime.device,
        dtype=runtime.dtype,
    )
    checkpoints = _checkpoints(config["steps"], quick)
    results: dict[str, dict] = {}

    for index, algorithm in enumerate(("polyak", "nesterov")):
        q = config["settings"][algorithm]
        print(f"[loss-dynamics] {algorithm.capitalize()}", flush=True)
        steps, empirical = simulate_long_time_curve(
            problem,
            algorithm=algorithm,
            batch_size=int(q["B"]),
            eta=float(q["eta"]),
            rho=float(q["rho"]),
            steps=int(config["steps"]),
            runs=int(config["runs"]),
            seed=runtime.seed + index * 10_000,
            checkpoints=checkpoints,
        )
        prediction, valid, coefficients, terms = fit_scaling_law_prediction(
            steps,
            empirical,
            algorithm,
            int(q["B"]),
            float(q["eta"]),
            float(q["rho"]),
            beta=float(config["beta"]),
            s=float(config["s"]),
            sigma=float(config["sigma"]),
        )
        results[algorithm] = {
            "label": algorithm.capitalize(),
            "B": int(q["B"]),
            "eta": float(q["eta"]),
            "rho": float(q["rho"]),
            "steps": steps.tolist(),
            "empirical_loss": empirical.tolist(),
            "theory_prediction": prediction.tolist(),
            "fit_valid_mask": valid.tolist(),
            "fitted_coefficients": {
                "transient": float(coefficients[0]),
                "signal": float(coefficients[1]),
                "noise": float(coefficients[2]),
            },
            "theory_terms": {
                "transient": terms[:, 0].tolist(),
                "signal": terms[:, 1].tolist(),
                "noise": terms[:, 2].tolist(),
            },
        }
        np.savez_compressed(
            runtime.output_dir / f"loss_dynamics_{algorithm}.npz",
            steps=steps,
            empirical_loss=empirical,
            theory_prediction=prediction,
            fit_valid_mask=valid,
            fitted_coefficients=coefficients,
            theory_transient=terms[:, 0],
            theory_signal=terms[:, 1],
            theory_noise=terms[:, 2],
        )
        print(f"  final loss={empirical[-1]:.6e}", flush=True)

    payload = {
        "experiment": "loss_dynamics",
        "preset": "quick" if quick else "paper",
        "environment": _environment(runtime),
        "config": config,
        "results": results,
    }
    _save_json(runtime.output_dir / "loss_dynamics.json", payload)
    return payload
