import os
import time
import zlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# =============================================================================
# 1. Global configuration
# =============================================================================

SIGMA = 0.1
DIM = 4096
N_RUNS = 10
BASE_SEED = 20260802
TAIL_FRACTION = 0.01

D_FIXED = 2**14

# Keep the original batch-size grid unchanged.
B_GRID = [
    {
        "batch_exponent": half_step / 2.0,
        "B": max(1, int(np.floor(2.0 ** (half_step / 2.0) + 0.5))),
    }
    for half_step in range(1, 20)
]
B_EXPONENT_LIST = [item["batch_exponent"] for item in B_GRID]
B_LIST = [item["B"] for item in B_GRID]

ETA_MULTIPLIERS = [0.3, 1.0, 3.0]
MAX_LEARNING_RATE = 1.0
DIVERGENCE_LOSS = 1e12
GAP_MULTIPLIERS = [0.3, 1.0, 3.0]

ETA_CONSTANTS = {
    "sgd": 0.30,
    "hb": 0.10,
    "nag": 0.10,
}

# Prefactors in (1-rho) = gap_constant * D^{-b}.
GAP_CONSTANTS = {
    "hb": 1.0,
    "nag": 2.0,
}

TRANSIENT_SLACK = 0.01

GROUP_ORDER = ["sgd", "hb", "nag"]
GROUP_LABELS = {
    "sgd": "SGD",
    "hb": "Polyak",
    "nag": "Nesterov",
}
MARKERS = {
    "sgd": "o",
    "hb": "^",
    "nag": "s",
}

# Run all four requested experiments.
EXPERIMENTS = [
    (0.5, 2.0),
    (0.8, 6.0)
]

# Parallelize across the four (s, beta) experiments.
MAX_PARALLEL_EXPERIMENTS = 8

ROOT_OUTPUT_DIR = Path.cwd() / "cbs"

# All four worker processes use the first visible GPU.
CUDA_DEVICE_INDEX = 0
DEVICE = f"cuda:{CUDA_DEVICE_INDEX}"

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (5, 4),
    "legend.fontsize": 10,
    "legend.handlelength": 1,
})


# =============================================================================
# 2. Helpers
# =============================================================================

def format_param(value):
    return f"{value:g}"


def experiment_output_dir(s_value, beta):
    return ROOT_OUTPUT_DIR / f"s_{format_param(s_value)}_beta_{format_param(beta)}"


def stable_seed(*items):
    text = "|".join(str(item) for item in items)
    return (BASE_SEED + zlib.crc32(text.encode("utf-8"))) % (2**32 - 1)


def tail_length(num_steps):
    return max(1, int(np.ceil(num_steps * TAIL_FRACTION)))


# =============================================================================
# 3. Power-law kernel-regression problem
# =============================================================================

def build_problem(dim=DIM, beta=4.0, s=0.3, device=DEVICE):
    if dim < 1:
        raise ValueError("DIM must be positive.")
    if beta <= 1:
        raise ValueError("beta must be greater than 1.")
    if s <= 0:
        raise ValueError("s must be positive.")

    # Construct directly on GPU; simulation tensors stay on CUDA.
    j = torch.arange(
        1, dim + 1, device=device, dtype=torch.float64
    )
    lam = j.pow(-beta).to(torch.float32)
    theta_squared = j.pow(-(1.0 + beta * (s - 1.0)))
    theta_star = theta_squared.sqrt().to(torch.float32)
    return lam, theta_star


def excess_loss_batch(delta, lam):
    # Match the previous NumPy implementation: state is fp32,
    # but the loss reduction accumulates in fp64.
    return 0.5 * torch.sum(
        lam.unsqueeze(0) * delta * delta,
        dim=1,
        dtype=torch.float64,
    )


def excess_loss_single(delta, lam):
    return float(
        (0.5 * torch.sum(lam * delta * delta, dtype=torch.float64)).item()
    )


# =============================================================================
# 4. Vectorized SGD / Heavy-Ball / Nesterov simulator
# =============================================================================

def run_optimizer(
    algo,
    batch_size,
    num_steps,
    eta,
    rho,
    lam,
    theta_star,
    runs=N_RUNS,
    sigma=SIGMA,
    seed=None,
):
    if seed is None:
        seed = stable_seed(algo, batch_size, num_steps, eta, rho)

    if not lam.is_cuda or not theta_star.is_cuda:
        raise RuntimeError("lam and theta_star must live on CUDA.")

    device = lam.device
    runs = int(runs)
    dim = lam.shape[0]

    # CUDA RNG: Gaussian noise is generated directly on cuda:0.
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    delta = (-theta_star).unsqueeze(0).repeat(runs, 1).contiguous()
    delta_previous = delta.clone()

    lam_row = lam.unsqueeze(0)
    sqrt_lam_row = torch.sqrt(lam).unsqueeze(0)

    tail_steps = tail_length(num_steps)
    tail_losses = torch.empty(
        (tail_steps, runs), device=device, dtype=torch.float64
    )

    for step in range(1, num_steps + 1):
        if algo == "nag":
            evaluation_delta = delta + rho * (delta - delta_previous)
        else:
            evaluation_delta = delta

        evaluation_loss = excess_loss_batch(evaluation_delta, lam)

        bad_loss = (
            (~torch.isfinite(evaluation_loss)).any()
            | (evaluation_loss.max() >= DIVERGENCE_LOSS)
        )
        if bool(bad_loss.item()):
            return np.full(runs, np.inf, dtype=np.float64)

        gaussian = torch.randn(
            (runs, dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

        noise_scale = torch.sqrt(
            (evaluation_loss + sigma**2) / batch_size
        ).to(torch.float32)

        noise = noise_scale.unsqueeze(1) * sqrt_lam_row * gaussian
        gradient = lam_row * evaluation_delta + noise

        if algo == "sgd":
            new_delta = delta - eta * gradient
        elif algo == "hb":
            new_delta = (
                delta
                - eta * gradient
                + rho * (delta - delta_previous)
            )
        elif algo == "nag":
            new_delta = evaluation_delta - eta * gradient
        else:
            raise ValueError(f"Unknown optimizer: {algo}")

        if bool((~torch.isfinite(new_delta)).any().item()):
            return np.full(runs, np.inf, dtype=np.float64)

        delta_previous, delta = delta, new_delta

        if step > num_steps - tail_steps:
            current_loss = excess_loss_batch(delta, lam)

            bad_tail = (
                (~torch.isfinite(current_loss)).any()
                | (current_loss.max() >= DIVERGENCE_LOSS)
            )
            if bool(bad_tail.item()):
                return np.full(runs, np.inf, dtype=np.float64)

            tail_losses[
                step - (num_steps - tail_steps) - 1
            ] = current_loss

    # Only the final per-run tail means are copied back to CPU.
    return tail_losses.mean(dim=0).cpu().numpy()


# =============================================================================
# 5. Theory-scaled schedules as functions of batch size
# =============================================================================

def batch_exponent_from_B(B, D=D_FIXED):
    return float(np.log(B) / np.log(D))


def theory_params_for_batch(algo, B, s_value, beta, D=D_FIXED):
    a = batch_exponent_from_B(B, D)

    base_optimal_exponent = s_value / (s_value + 1.0)
    polyak_max_batch_exponent = (
        (2.0 * s_value + 1.0) / (2.0 * s_value + 2.0)
    )
    nag_max_batch_exponent = (
        (2.0 * s_value * beta + 1.0)
        / (2.0 * s_value * beta + beta + 1.0)
    )

    if algo == "sgd":
        c = max(base_optimal_exponent - a, 0.0)
        eta = ETA_CONSTANTS["sgd"] * D ** (-c)
        return {
            "a": a,
            "b": 0.0,
            "c": c,
            "eta": float(eta),
            "one_minus_rho": 1.0,
            "rho": 0.0,
        }

    if algo == "hb":
        if a <= base_optimal_exponent:
            b = 0.0
            c = base_optimal_exponent - a
        elif a <= polyak_max_batch_exponent:
            b = a - base_optimal_exponent
            c = 0.0
        else:
            b = max(1.0 - a - TRANSIENT_SLACK, 0.0)
            c = 0.0

        eta = ETA_CONSTANTS["hb"] * D ** (-c)
        one_minus_rho = min(
            1.0,
            GAP_CONSTANTS["hb"] * D ** (-b),
        )

        return {
            "a": a,
            "b": b,
            "c": c,
            "eta": float(eta),
            "one_minus_rho": float(one_minus_rho),
            "rho": float(1.0 - one_minus_rho),
        }

    if algo == "nag":
        if a <= base_optimal_exponent:
            b = 0.0
            c = base_optimal_exponent - a
        elif a <= nag_max_batch_exponent:
            b = (
                beta
                * ((s_value + 1.0) * a - s_value)
                / (s_value * beta + 1.0)
            )
            c = 0.0
        else:
            b = max(1.0 - a - TRANSIENT_SLACK, 0.0)
            c = 0.0

        eta = ETA_CONSTANTS["nag"] * D ** (-c)
        one_minus_rho = min(
            1.0,
            GAP_CONSTANTS["nag"] * D ** (-b),
        )

        return {
            "a": a,
            "b": b,
            "c": c,
            "eta": float(eta),
            "one_minus_rho": float(one_minus_rho),
            "rho": float(1.0 - one_minus_rho),
        }

    raise ValueError(f"Unknown optimizer: {algo}")


# =============================================================================
# 6. Search around theory-scaled schedules
# =============================================================================

def search_one_batch(
    algo,
    B,
    s_value,
    beta,
    lam,
    theta_star,
    D=D_FIXED,
):
    K = max(1, D // B)
    theory = theory_params_for_batch(
        algo=algo,
        B=B,
        s_value=s_value,
        beta=beta,
        D=D,
    )

    search_rows = []

    if algo == "sgd":
        for eta_mult in ETA_MULTIPLIERS:
            eta = theory["eta"] * eta_mult

            if eta > MAX_LEARNING_RATE:
                continue

            seed = stable_seed("cbs", algo, B, eta_mult)

            tail_mean_losses = run_optimizer(
                algo=algo,
                batch_size=B,
                num_steps=K,
                eta=eta,
                rho=0.0,
                lam=lam,
                theta_star=theta_star,
                seed=seed,
            )

            search_rows.append({
                "eta_mult": eta_mult,
                "gap_mult": np.nan,
                "eta": float(eta),
                "one_minus_rho": 1.0,
                "rho": 0.0,
                "mean_loss": float(tail_mean_losses.mean()),
            })

    else:
        for eta_mult in ETA_MULTIPLIERS:
            eta = theory["eta"] * eta_mult

            if eta > MAX_LEARNING_RATE:
                continue

            for gap_mult in GAP_MULTIPLIERS:
                one_minus_rho = min(
                    1.0,
                    theory["one_minus_rho"] * gap_mult,
                )
                rho = 1.0 - one_minus_rho

                seed = stable_seed(
                    "cbs",
                    algo,
                    B,
                    eta_mult,
                    gap_mult,
                )

                tail_mean_losses = run_optimizer(
                    algo=algo,
                    batch_size=B,
                    num_steps=K,
                    eta=eta,
                    rho=rho,
                    lam=lam,
                    theta_star=theta_star,
                    seed=seed,
                )

                search_rows.append({
                    "eta_mult": eta_mult,
                    "gap_mult": gap_mult,
                    "eta": float(eta),
                    "one_minus_rho": float(one_minus_rho),
                    "rho": float(rho),
                    "mean_loss": float(tail_mean_losses.mean()),
                })

    search_df = pd.DataFrame(search_rows)

    if search_df.empty:
        raise RuntimeError(
            f"No learning-rate candidate satisfies "
            f"eta <= {MAX_LEARNING_RATE} for {algo} at B={B}."
        )

    finite_df = search_df[np.isfinite(search_df["mean_loss"])]

    if finite_df.empty:
        raise RuntimeError(
            f"All admissible candidates diverged for {algo} at B={B}."
        )

    best_index = finite_df["mean_loss"].idxmin()
    best_row = search_df.loc[best_index].copy()

    return {
        "group": algo,
        "B": int(B),
        "K": int(K),
        "mean_loss": float(best_row["mean_loss"]),
        "eta_mult": float(best_row["eta_mult"]),
        "gap_mult": (
            np.nan
            if pd.isna(best_row["gap_mult"])
            else float(best_row["gap_mult"])
        ),
        "eta": float(best_row["eta"]),
        "one_minus_rho": float(best_row["one_minus_rho"]),
        "rho": float(best_row["rho"]),
    }


# =============================================================================
# 7. Plot batch size versus optimal loss
# =============================================================================

def plot_cbs(best_results_df, s_value, beta, output_dir):
    base_optimal_exponent = s_value / (s_value + 1.0)
    b_sgd_critical = D_FIXED ** base_optimal_exponent

    fig, ax = plt.subplots()

    b_values = best_results_df["B"].to_numpy(dtype=np.float64)
    x_min = float(np.min(b_values))
    x_max = 1000.0

    # Keep the original three batch-size regimes unchanged.
    ultra_large_boundary = 128.0

    ax.axvspan(
        x_min,
        b_sgd_critical,
        color="lightskyblue",
        alpha=0.25,
        zorder=0,
    )
    ax.axvspan(
        b_sgd_critical,
        ultra_large_boundary,
        color="lightcoral",
        alpha=0.20,
        zorder=0,
    )
    ax.axvspan(
        ultra_large_boundary,
        x_max,
        color="lightgreen",
        alpha=0.25,
        zorder=0,
    )

    for group in GROUP_ORDER:
        group_df = (
            best_results_df[
                best_results_df["group"] == group
            ]
            .sort_values("B")
        )

        ax.plot(
            group_df["B"].to_numpy(dtype=np.float64),
            group_df["mean_loss"].to_numpy(dtype=np.float64),
            marker=MARKERS[group],
            label=GROUP_LABELS[group],
            zorder=3,
        )

    ax.axvline(
        b_sgd_critical,
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.6,
        zorder=1,
    )
    ax.axvline(
        ultra_large_boundary,
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.6,
        zorder=1,
    )

    ax.set_xscale("log", base=10)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Loss")
    ax.set_xticks(
        [1e1, 1e2, 1e3],
        [r"$10^1$", r"$10^2$", r"$10^3$"],
    )
    ax.legend(loc="best")
    fig.tight_layout()

    pdf_path = output_dir / "cbs.pdf"
    png_path = output_dir / "cbs.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return pdf_path, png_path


# =============================================================================
# 8. Run one (s, beta) experiment
# =============================================================================

def run_experiment(s_value, beta):
    start = time.time()

    # Every process intentionally shares cuda:0.
    torch.cuda.set_device(CUDA_DEVICE_INDEX)
    device = torch.device(DEVICE)

    output_dir = experiment_output_dir(s_value, beta)
    output_dir.mkdir(parents=True, exist_ok=True)

    lam, theta_star = build_problem(
        dim=DIM,
        beta=beta,
        s=s_value,
        device=device,
    )

    print(
        f"[START] pid={os.getpid()} | device={lam.device} | "
        f"s={s_value:g}, beta={beta:g} | "
        f"lambda_1={lam[0]:.6e}, "
        f"lambda_DIM={lam[-1]:.6e}, "
        f"initial_loss={excess_loss_single(-theta_star, lam):.6e}",
        flush=True,
    )

    best_rows = []

    for group in GROUP_ORDER:
        for batch_exponent, B in zip(
            B_EXPONENT_LIST,
            B_LIST,
        ):
            batch_start = time.time()

            best_row = search_one_batch(
                algo=group,
                B=B,
                s_value=s_value,
                beta=beta,
                lam=lam,
                theta_star=theta_star,
            )

            best_row["batch_exponent"] = batch_exponent
            best_rows.append(best_row)

            elapsed = time.time() - batch_start

            message = (
                f"[s={s_value:g}, beta={beta:g}] "
                f"{GROUP_LABELS[group]:8s} | "
                f"x={batch_exponent:4.1f} | "
                f"B={B:5d} | "
                f"K={best_row['K']:5d} | "
                f"best loss={best_row['mean_loss']:.6e} | "
                f"eta_mult={best_row['eta_mult']:.1f}"
            )

            if group != "sgd":
                message += (
                    f" | gap_mult={best_row['gap_mult']:.1f}"
                )

            message += f" | elapsed={elapsed:.1f}s"
            print(message, flush=True)

    best_results_df = (
        pd.DataFrame(best_rows)
        .sort_values(["group", "B"])
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Save ONLY the requested compact optimal-loss CSV.
    # Rows = batch sizes; columns = algorithms.
    # No search table and no hyperparameter table are written to disk.
    # -------------------------------------------------------------------------
    loss_csv = (
        best_results_df[["B", "group", "mean_loss"]]
        .pivot(
            index="B",
            columns="group",
            values="mean_loss",
        )
        .reset_index()
        .rename(columns={
            "sgd": "SGD",
            "hb": "Polyak",
            "nag": "Nesterov",
        })
        [["B", "SGD", "Polyak", "Nesterov"]]
    )

    csv_path = output_dir / "optimal_loss.csv"
    loss_csv.to_csv(csv_path, index=False)

    pdf_path, png_path = plot_cbs(
        best_results_df=best_results_df,
        s_value=s_value,
        beta=beta,
        output_dir=output_dir,
    )

    total_time = time.time() - start

    return {
        "s": s_value,
        "beta": beta,
        "csv": str(csv_path),
        "pdf": str(pdf_path),
        "png": str(png_path),
        "seconds": total_time,
    }


# =============================================================================
# 9. Run all requested experiments in parallel
# =============================================================================

def main():
    ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This version of cbs.py requires a GPU."
        )
    if torch.cuda.device_count() <= CUDA_DEVICE_INDEX:
        raise RuntimeError(
            f"Requested cuda:{CUDA_DEVICE_INDEX}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible."
        )

    print(f"CUDA available = {torch.cuda.is_available()}")
    print(f"CUDA device = {DEVICE}")
    print(f"GPU name = {torch.cuda.get_device_name(CUDA_DEVICE_INDEX)}")
    print(f"D_FIXED = {D_FIXED}")
    print(f"N_RUNS = {N_RUNS}")
    print(f"Parallel experiments = {MAX_PARALLEL_EXPERIMENTS}")
    print(f"Output root = {ROOT_OUTPUT_DIR.resolve()}")
    print("Experiments:")
    for s_value, beta in EXPERIMENTS:
        print(f"  s={s_value:g}, beta={beta:g}")
    print()

    results = []

    # CUDA must use spawn, not the default fork start method.
    # Four processes are created and all four intentionally use cuda:0.
    spawn_context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=MAX_PARALLEL_EXPERIMENTS,
        mp_context=spawn_context,
    ) as executor:
        future_to_exp = {
            executor.submit(
                run_experiment,
                s_value,
                beta,
            ): (s_value, beta)
            for s_value, beta in EXPERIMENTS
        }

        for future in as_completed(future_to_exp):
            s_value, beta = future_to_exp[future]

            try:
                result = future.result()
                results.append(result)
                print(
                    f"[DONE] s={s_value:g}, beta={beta:g} | "
                    f"{result['seconds']:.1f}s | "
                    f"{result['csv']}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[FAILED] s={s_value:g}, beta={beta:g}: {exc}",
                    flush=True,
                )
                raise

    results.sort(key=lambda x: (x["s"], x["beta"]))

    print("\nAll experiments finished.")
    for result in results:
        print(
            f"s={result['s']:g}, beta={result['beta']:g}: "
            f"{result['csv']}, {result['pdf']}, {result['png']}"
        )


if __name__ == "__main__":
    main()
