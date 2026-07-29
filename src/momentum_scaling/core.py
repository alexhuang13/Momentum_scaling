"""Shared model, device, optimizer, and random-number utilities.

Both CPU and GPU runs use this module.  The only difference is the selected
PyTorch device; the mathematical update rules are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch

ALGORITHMS = ("sgd", "polyak", "nesterov")
DISPLAY_NAMES = {"sgd": "SGD", "polyak": "Polyak", "nesterov": "Nesterov"}


@dataclass(frozen=True)
class Problem:
    beta: float
    s: float
    sigma: float
    dim: int
    lam: torch.Tensor
    theta_star: torch.Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is unavailable.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float32 if device.type == "mps" else torch.float64
    return {"float32": torch.float32, "float64": torch.float64}[name]


def build_problem(
    *, beta: float, s: float, sigma: float, dim: int,
    device: torch.device, dtype: torch.dtype,
) -> Problem:
    if beta <= 1 or s <= 0 or dim <= 0:
        raise ValueError("Require beta > 1, s > 0, and dim > 0.")
    j = torch.arange(1, dim + 1, device=device, dtype=dtype)
    lam = j.pow(-beta)
    # lambda_j |theta*_j|^2 = j^(-(1+s beta)).
    theta_star = j.pow(-0.5 * (1.0 + (s - 1.0) * beta))
    return Problem(beta, s, sigma, dim, lam, theta_star)


def excess_loss(delta: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Population excess risk; supports leading batch/seed dimensions."""
    return 0.5 * torch.sum(delta.square() * lam, dim=-1)


def lookahead(delta: torch.Tensor, previous: torch.Tensor, algorithm: str, rho: float) -> torch.Tensor:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if algorithm == "sgd":
        return delta
    momentum = delta - previous
    return delta + rho * momentum if algorithm == "nesterov" else delta


def update(
    delta: torch.Tensor,
    previous: torch.Tensor,
    gradient: torch.Tensor,
    *, algorithm: str, eta: float, rho: float,
) -> torch.Tensor:
    if algorithm == "sgd":
        return delta - eta * gradient
    momentum = delta - previous
    # For Nesterov this equals lookahead(delta) - eta * grad(lookahead).
    return delta + rho * momentum - eta * gradient


def exact_minibatch_gradient(
    delta: torch.Tensor,
    problem: Problem,
    *, batch_size: int, chunk: int, generator: torch.Generator,
) -> torch.Tensor:
    """Fresh Gaussian mini-batch gradient without storing the full batch."""
    if delta.ndim != 1:
        raise ValueError("Exact mini-batch gradient expects one parameter vector.")
    grad = torch.zeros_like(delta)
    sqrt_lam = problem.lam.sqrt()
    remaining = int(batch_size)
    while remaining:
        n = min(chunk, remaining)
        x = torch.randn(
            (n, problem.dim), device=delta.device, dtype=delta.dtype, generator=generator
        )
        x.mul_(sqrt_lam)
        residual = x.mv(delta)
        if problem.sigma:
            eps = torch.randn(n, device=delta.device, dtype=delta.dtype, generator=generator)
            residual.sub_(eps.mul_(problem.sigma))
        grad.add_(x.t().mv(residual))
        remaining -= n
    return grad.div_(float(batch_size))


def aligned_noise_gradient(
    delta: torch.Tensor,
    problem: Problem,
    *, batch_size: int, generator: torch.Generator,
) -> torch.Tensor:
    """Population gradient plus curvature-aligned Gaussian gradient noise.

    This is the covariance model used by the paper's data-scaling experiments:
    Var[xi_j | delta] = lambda_j (E(delta) + sigma^2) / B.
    Leading dimensions of ``delta`` are independent simulation replicas.
    """
    loss = excess_loss(delta, problem.lam)
    scale = ((loss + problem.sigma**2) / float(batch_size)).sqrt().unsqueeze(-1)
    noise = torch.randn(
        delta.shape, device=delta.device, dtype=delta.dtype, generator=generator
    )
    noise.mul_(problem.lam.sqrt()).mul_(scale)
    return delta * problem.lam + noise


def run_aligned_replicas(
    problem: Problem,
    *, algorithm: str, batch_size: int, steps: int, eta: float, rho: float,
    seeds: int, seed: int, tail_fraction: float = 0.0,
) -> np.ndarray:
    """Run independent replicas in parallel and return final/tail-mean losses."""
    if not 0 <= rho < 1:
        raise ValueError(f"rho must lie in [0,1), got {rho}")
    generator = torch.Generator(device=problem.lam.device)
    generator.manual_seed(int(seed))
    delta = -problem.theta_star.expand(seeds, -1).clone()
    previous = delta.clone()
    tail_steps = max(1, int(round(steps * tail_fraction))) if tail_fraction else 1
    tail = torch.zeros(seeds, device=delta.device, dtype=delta.dtype)

    with torch.no_grad():
        for k in range(steps):
            evaluation = lookahead(delta, previous, algorithm, rho)
            gradient = aligned_noise_gradient(
                evaluation, problem, batch_size=batch_size, generator=generator
            )
            new_delta = update(
                delta, previous, gradient, algorithm=algorithm, eta=eta, rho=rho
            )
            previous, delta = delta, new_delta
            if k >= steps - tail_steps:
                tail.add_(excess_loss(delta, problem.lam))
    return tail.div_(tail_steps).cpu().numpy()
