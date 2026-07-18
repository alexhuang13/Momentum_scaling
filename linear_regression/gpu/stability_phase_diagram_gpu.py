#!/usr/bin/env python3
"""GPU stability phase diagram for HB and Nesterov on diagonal linear regression.

Default setting matches linear_regression/cpu/D_scaling.ipynb and is intended for
an NVIDIA H20 / CUDA machine.  Because the default damping is extremely small
(1-rho = 2e-10), the default dtype is float64; float32 would round rho to 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ALGOS = {"HB": "hb", "Nesterov": "nag"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU stability phase diagram for HB/Nesterov")

    # Problem setting copied from cpu/D_scaling.ipynb.
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--s", type=float, default=0.5, dest="s_value")

    # Stability experiment.
    p.add_argument("--one-minus-rho", type=float, default=2e-10)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--threshold", type=float, default=1e-2)
    p.add_argument("--divergence-loss", type=float, default=1e12)

    # Grid.
    p.add_argument("--b-min-exp", type=int, default=0)
    p.add_argument("--b-max-exp", type=int, default=12)
    p.add_argument("--eta-min", type=float, default=2e-15)
    p.add_argument("--eta-max", type=float, default=2e2)
    p.add_argument("--eta-points", type=int, default=60)

    # Runtime.
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None,
                   help="output directory; default is <script_dir>/results")
    p.add_argument("--no-resume", action="store_true", help="ignore existing .npz cache")
    p.add_argument("--plot-only", action="store_true", help="only plot from an existing cache")
    p.add_argument("--save-every-b", action="store_true", default=True)

    return p.parse_args()


def setup_device(device_arg: str, dtype_arg: str) -> tuple[torch.device, torch.dtype]:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    dtype = torch.float64 if dtype_arg == "float64" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = dtype == torch.float32
        print(f"device={device}  gpu={torch.cuda.get_device_name(device)}  dtype={dtype}")
    else:
        print(f"device={device}  dtype={dtype}")
    return device, dtype


def build_problem(dim: int, beta: float, s: float, device: torch.device, dtype: torch.dtype):
    j = torch.arange(1, dim + 1, device=device, dtype=dtype)
    lam = j.pow(-beta)
    var_star = j.pow(-(1.0 + (s - 1.0) * beta))
    theta_star = torch.sqrt(var_star)
    return lam, theta_star


@torch.no_grad()
def excess_loss(delta: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(lam * delta * delta)


@torch.no_grad()
def grad_delta_chunked(
    delta: torch.Tensor,
    B: int,
    lam: torch.Tensor,
    sigma: float,
    chunk: int,
    generator: torch.Generator,
) -> torch.Tensor:
    d = lam.numel()
    sqrt_lam = torch.sqrt(lam)
    grad = torch.zeros_like(delta)
    remaining = int(B)
    while remaining > 0:
        b = min(chunk, remaining)
        X = torch.randn((b, d), device=delta.device, dtype=delta.dtype, generator=generator)
        X.mul_(sqrt_lam)
        eps = torch.randn((b,), device=delta.device, dtype=delta.dtype, generator=generator)
        eps.mul_(sigma)
        resid = X.mv(delta).sub_(eps)
        grad.add_(X.t().mv(resid))
        remaining -= b
    grad.div_(float(B))
    return grad


@torch.no_grad()
def run_stability(
    algo: str,
    B: int,
    eta: float,
    rho: float,
    lam: torch.Tensor,
    theta_star: torch.Tensor,
    sigma: float,
    steps: int,
    window: int,
    threshold: float,
    divergence_loss: float,
    chunk: int,
    seed: int,
) -> dict[str, float | int | bool | str]:
    gen = torch.Generator(device=lam.device)
    gen.manual_seed(int(seed))

    delta = -theta_star.clone()
    delta_prev = delta.clone()
    tail: list[float] = []

    loss = float(excess_loss(delta, lam).item())
    tail.append(loss)

    for t in range(1, steps + 1):
        momentum = delta - delta_prev
        eval_delta = delta + rho * momentum if algo == "nag" else delta
        g = grad_delta_chunked(eval_delta, B, lam, sigma, chunk, gen)
        delta_new = delta - eta * g + rho * momentum
        delta_prev, delta = delta, delta_new

        loss = float(excess_loss(delta, lam).item())
        if (not math.isfinite(loss)) or loss > divergence_loss:
            return {
                "stable": False,
                "final_ma": math.inf,
                "final_loss": loss if math.isfinite(loss) else math.inf,
                "steps": t,
                "reason": "diverged",
            }
        tail.append(loss)
        if len(tail) > window:
            tail.pop(0)

    final_ma = float(sum(tail) / len(tail))
    return {
        "stable": bool(math.isfinite(final_ma) and final_ma < threshold),
        "final_ma": final_ma,
        "final_loss": loss,
        "steps": steps,
        "reason": "finished",
    }


def make_grids(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    b_list = np.array([2**e for e in range(args.b_min_exp, args.b_max_exp + 1)], dtype=np.int64)
    eta_grid = np.logspace(np.log10(args.eta_min), np.log10(args.eta_max), args.eta_points)
    return b_list, eta_grid


def tag_from_args(args: argparse.Namespace) -> str:
    tag = (
        f"stability_gpu_beta{args.beta:g}_s{args.s_value:g}_dim{args.dim}_"
        f"B{args.b_min_exp}-{args.b_max_exp}_eta{args.eta_points}_"
        f"omrho{args.one_minus_rho:.0e}_K{args.steps}_w{args.window}_thr{args.threshold:g}_"
        f"dtype{args.dtype}_seed{args.seed}"
    )
    return tag.replace(".", "_").replace("-", "m")


def init_or_load(cache_path: Path, args: argparse.Namespace, b_list: np.ndarray, eta_grid: np.ndarray):
    shape = (len(ALGOS), len(eta_grid), len(b_list))
    if cache_path.exists() and not args.no_resume:
        data = np.load(cache_path, allow_pickle=False)
        stable = data["stable"].astype(bool)
        final_ma = data["final_ma"]
        final_loss = data["final_loss"]
        steps_done = data["steps_done"]
        if stable.shape == shape:
            print(f"resuming from {cache_path}")
            return stable, final_ma, final_loss, steps_done
        print(f"cache shape {stable.shape} != expected {shape}; starting over")

    stable = np.zeros(shape, dtype=bool)
    final_ma = np.full(shape, np.nan, dtype=np.float64)
    final_loss = np.full(shape, np.nan, dtype=np.float64)
    steps_done = np.zeros(shape, dtype=np.int32)
    return stable, final_ma, final_loss, steps_done


def save_npz(
    cache_path: Path,
    args: argparse.Namespace,
    b_list: np.ndarray,
    eta_grid: np.ndarray,
    stable: np.ndarray,
    final_ma: np.ndarray,
    final_loss: np.ndarray,
    steps_done: np.ndarray,
):
    min_final_ma, min_final_ma_eta, min_final_loss, min_final_loss_eta = min_loss_by_b(
        b_list, eta_grid, final_ma, final_loss
    )
    np.savez_compressed(
        cache_path,
        config=json.dumps(vars(args), sort_keys=True),
        algos=np.array(list(ALGOS.keys())),
        b_list=b_list,
        eta_grid=eta_grid,
        stable=stable,
        final_ma=final_ma,
        final_loss=final_loss,
        steps_done=steps_done,
        min_final_ma=min_final_ma,
        min_final_ma_eta=min_final_ma_eta,
        min_final_loss=min_final_loss,
        min_final_loss_eta=min_final_loss_eta,
    )


def min_loss_by_b(b_list: np.ndarray, eta_grid: np.ndarray, final_ma: np.ndarray, final_loss: np.ndarray):
    n_algos = len(ALGOS)
    n_b = len(b_list)
    min_final_ma = np.full((n_algos, n_b), np.nan)
    min_final_ma_eta = np.full((n_algos, n_b), np.nan)
    min_final_loss = np.full((n_algos, n_b), np.nan)
    min_final_loss_eta = np.full((n_algos, n_b), np.nan)

    for aidx in range(n_algos):
        for j in range(n_b):
            ma_col = final_ma[aidx, :, j]
            ma_finite = np.isfinite(ma_col)
            if np.any(ma_finite):
                i = int(np.argmin(np.where(ma_finite, ma_col, np.inf)))
                min_final_ma[aidx, j] = ma_col[i]
                min_final_ma_eta[aidx, j] = eta_grid[i]

            loss_col = final_loss[aidx, :, j]
            loss_finite = np.isfinite(loss_col)
            if np.any(loss_finite):
                i = int(np.argmin(np.where(loss_finite, loss_col, np.inf)))
                min_final_loss[aidx, j] = loss_col[i]
                min_final_loss_eta[aidx, j] = eta_grid[i]

    return min_final_ma, min_final_ma_eta, min_final_loss, min_final_loss_eta


def max_stable_eta(stable_map: np.ndarray, eta_grid: np.ndarray) -> np.ndarray:
    boundary = np.full(stable_map.shape[1], np.nan)
    for j in range(stable_map.shape[1]):
        idx = np.where(stable_map[:, j])[0]
        if len(idx):
            boundary[j] = eta_grid[idx.max()]
    return boundary


def fit_prefactor(boundary: np.ndarray, b_list: np.ndarray, power: float, one_minus_rho: float) -> float:
    mask = np.isfinite(boundary) & (boundary > 0)
    if not np.any(mask):
        return 1.0
    denom = (b_list[mask].astype(float) ** power) * one_minus_rho
    return float(np.exp(np.mean(np.log(boundary[mask] / denom))))


def save_summary_csv(
    csv_path: Path,
    b_list: np.ndarray,
    eta_grid: np.ndarray,
    stable: np.ndarray,
    final_ma: np.ndarray,
    final_loss: np.ndarray,
):
    min_final_ma, min_final_ma_eta, min_final_loss, min_final_loss_eta = min_loss_by_b(
        b_list, eta_grid, final_ma, final_loss
    )
    algo_names = list(ALGOS.keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "B",
                "eta_max_stable",
                "min_final_MA",
                "eta_at_min_final_MA",
                "min_final_loss",
                "eta_at_min_final_loss",
            ]
        )
        for aidx, name in enumerate(algo_names):
            boundary = max_stable_eta(stable[aidx], eta_grid)
            for j, B in enumerate(b_list):
                writer.writerow(
                    [
                        name,
                        int(B),
                        boundary[j],
                        min_final_ma[aidx, j],
                        min_final_ma_eta[aidx, j],
                        min_final_loss[aidx, j],
                        min_final_loss_eta[aidx, j],
                    ]
                )
    print(f"wrote {csv_path}")


def plot_boundary_pdf(
    pdf_path: Path,
    args: argparse.Namespace,
    b_list: np.ndarray,
    eta_grid: np.ndarray,
    stable: np.ndarray,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    styles = {
        "HB": ("tab:blue", "o", 1.0, r"HB fit: $\eta \propto B(1-\rho)$"),
        "Nesterov": (
            "tab:orange",
            "s",
            args.beta,
            rf"Nesterov fit: $\eta \propto B^{{\beta}}(1-\rho)$, $\beta={args.beta:g}$",
        ),
    }

    for aidx, name in enumerate(ALGOS.keys()):
        color, marker, power, theory_label = styles[name]
        boundary = max_stable_eta(stable[aidx], eta_grid)
        mask = np.isfinite(boundary) & (boundary > 0)
        ax.plot(
            b_list[mask],
            boundary[mask],
            color=color,
            marker=marker,
            linewidth=2.2,
            label=rf"{name}: empirical max stable $\eta$",
        )
        C = fit_prefactor(boundary, b_list, power, args.one_minus_rho)
        ref = C * (b_list.astype(float) ** power) * args.one_minus_rho
        ax.plot(b_list, ref, "--", color=color, alpha=0.85, linewidth=2.0, label=theory_label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"batch size  $B$")
    ax.set_ylabel(r"learning rate stability boundary  $\eta$")
    ax.set_title("HB vs Nesterov stability boundary")
    ax.grid(True, which="both", alpha=0.3)
    legend_title = (
        rf"$1-\rho={args.one_minus_rho:.0e}$, $\beta={args.beta:g}$, $s={args.s_value:g}$"
        + "\n"
        + rf"$K={args.steps}$, MA window={args.window}, threshold={args.threshold:g}"
        + "\n"
        + rf"$B=2^{{{args.b_min_exp}}}\ldots2^{{{args.b_max_exp}}}$, "
        + rf"$\eta\in[{args.eta_min:.0e},{args.eta_max:.0e}]$"
    )
    ax.legend(title=legend_title, fontsize=8, title_fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf_path}")


def main():
    args = parse_args()
    device, dtype = setup_device(args.device, args.dtype)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    b_list, eta_grid = make_grids(args)
    tag = tag_from_args(args)
    cache_path = output_dir / f"{tag}.npz"
    csv_path = output_dir / f"{tag}_summary.csv"
    pdf_path = output_dir / f"{tag}.pdf"

    stable, final_ma, final_loss, steps_done = init_or_load(cache_path, args, b_list, eta_grid)

    if not args.plot_only:
        rho = 1.0 - args.one_minus_rho
        if args.dtype == "float32" and args.one_minus_rho < 1e-7:
            print("WARNING: one-minus-rho is below float32 resolution; use --dtype float64 for faithful rho")
        lam, theta_star = build_problem(args.dim, args.beta, args.s_value, device, dtype)
        e0 = float(excess_loss(-theta_star, lam).item())
        print(f"initial excess loss E0={e0:.6g}")
        print(f"rho={rho:.16g}  one-minus-rho={args.one_minus_rho:.3e}")
        print(f"B grid: 2^{args.b_min_exp}..2^{args.b_max_exp} ({len(b_list)} values)")
        print(f"eta grid: {args.eta_min:.1e}..{args.eta_max:.1e} ({len(eta_grid)} values)")

        algo_items = list(ALGOS.items())
        t0 = time.time()
        for aidx, (name, algo) in enumerate(algo_items):
            print(f"=== {name} ===", flush=True)
            for j, B in enumerate(b_list):
                print(f"B=2^{int(math.log2(int(B))):<2d} ({int(B):6d})", flush=True)
                for i, eta in enumerate(eta_grid):
                    if steps_done[aidx, i, j] > 0 and not args.no_resume:
                        continue
                    seed = args.seed + 1_000_000 * aidx + 10_000 * j + i
                    out = run_stability(
                        algo=algo,
                        B=int(B),
                        eta=float(eta),
                        rho=rho,
                        lam=lam,
                        theta_star=theta_star,
                        sigma=args.sigma,
                        steps=args.steps,
                        window=args.window,
                        threshold=args.threshold,
                        divergence_loss=args.divergence_loss,
                        chunk=args.chunk,
                        seed=seed,
                    )
                    stable[aidx, i, j] = bool(out["stable"])
                    final_ma[aidx, i, j] = float(out["final_ma"])
                    final_loss[aidx, i, j] = float(out["final_loss"])
                    steps_done[aidx, i, j] = int(out["steps"])

                finite = np.isfinite(final_ma[aidx, :, j])
                n_stable = int(stable[aidx, :, j].sum())
                if np.any(finite):
                    best_i = int(np.argmin(np.where(finite, final_ma[aidx, :, j], np.inf)))
                    print(
                        f"  stable eta count: {n_stable}/{len(eta_grid)}; "
                        f"min final-MA loss={final_ma[aidx, best_i, j]:.3e} "
                        f"at eta={eta_grid[best_i]:.3e}",
                        flush=True,
                    )
                else:
                    print(f"  stable eta count: {n_stable}/{len(eta_grid)}; min final-MA loss=nan")

                if args.save_every_b:
                    save_npz(cache_path, args, b_list, eta_grid, stable, final_ma, final_loss, steps_done)

        print(f"scan finished in {(time.time() - t0) / 60:.2f} minutes")
        save_npz(cache_path, args, b_list, eta_grid, stable, final_ma, final_loss, steps_done)

    save_summary_csv(csv_path, b_list, eta_grid, stable, final_ma, final_loss)
    plot_boundary_pdf(pdf_path, args, b_list, eta_grid, stable)


if __name__ == "__main__":
    main()
