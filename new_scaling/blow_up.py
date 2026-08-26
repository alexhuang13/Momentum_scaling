#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Experiment config
# =============================================================================

D = 2048
S = 0.75
BETA = 2.0

ETA = 0.05
BATCH_SIZE = 32

STEPS = 500000
NRUNS = 50
RECORD_EVERY = 1

SEED = 42
BLOWUP_RISK = 10

OUTPUT_DIR = Path.cwd() / "blowup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Fig. 3 config: exactly the same
# =============================================================================

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


COLORS = {
    "sgd": "C0",
    "polyak": "C1",
    "nesterov": "C2",
}


# =============================================================================
# Problem
# =============================================================================

def make_problem(d, beta, s, device):
    j = torch.arange(
        1,
        d + 1,
        dtype=torch.float64,
        device=device,
    )

    lam = j ** (-beta)

    theta_star = torch.sqrt(
        j ** (-1.0) * lam ** (s - 1.0)
    )

    theta_star[1::2] *= -1.0

    return theta_star, lam


def population_risk(theta, theta_star, lam):
    return 0.5 * torch.sum(
        lam * (theta - theta_star) ** 2,
        dim=-1,
    )


def stochastic_gradient(
    theta,
    theta_star,
    lam,
    batch_size,
):
    risk = population_risk(
        theta,
        theta_star,
        lam,
    )

    noise_var = (
        lam.unsqueeze(0)
        * risk.unsqueeze(-1)
        / batch_size
    )

    noise = (
        torch.randn_like(theta)
        * torch.sqrt(noise_var)
    )

    grad = (
        lam.unsqueeze(0)
        * (theta - theta_star.unsqueeze(0))
        + noise
    )

    return grad


# =============================================================================
# Optimizers
# =============================================================================

def run_method(
    method,
    theta_star,
    lam,
    eta,
    batch_size,
    steps,
    nruns,
    record_every,
    seed,
    device,
):
    torch.manual_seed(seed)

    d = theta_star.numel()

    theta = torch.zeros(
        nruns,
        d,
        dtype=torch.float64,
        device=device,
    )

    theta_prev = theta.clone()

    step_history = [1]

    risk = population_risk(
        theta,
        theta_star,
        lam,
    )

    risk_history = [
        risk.median().item()
    ]

    for k in range(1, steps + 1):

        rho = 1.0 - 1.0 / (k + 1.0)

        if method == "sgd":

            grad = stochastic_gradient(
                theta,
                theta_star,
                lam,
                batch_size,
            )

            theta = theta - eta * grad

        elif method == "polyak":

            grad = stochastic_gradient(
                theta,
                theta_star,
                lam,
                batch_size,
            )

            theta_next = (
                theta
                + rho * (theta - theta_prev)
                - eta * grad
            )

            theta_prev = theta
            theta = theta_next

        elif method == "nesterov":

            lookahead = (
                theta
                + rho * (theta - theta_prev)
            )

            grad = stochastic_gradient(
                lookahead,
                theta_star,
                lam,
                batch_size,
            )

            theta_next = (
                theta
                + rho * (theta - theta_prev)
                - eta * grad
            )

            theta_prev = theta
            theta = theta_next

        else:
            raise ValueError(method)

        if (
            k % record_every == 0
            or k == steps
        ):
            risk = population_risk(
                theta,
                theta_star,
                lam,
            )

            risk_value = risk.median().item()

            step_history.append(k)
            risk_history.append(risk_value)

            if (
                not np.isfinite(risk_value)
                or risk_value >= BLOWUP_RISK
            ):
                print(
                    f"{method:10s}: blow-up near step {k}"
                )
                break

    return (
        np.asarray(step_history),
        np.asarray(risk_history),
    )


# =============================================================================
# Plot
# =============================================================================

def plot_results(results):
    plt.figure()

    plt.plot(
        results["sgd"][0],
        results["sgd"][1],
        color=COLORS["sgd"],
        label="SGD",
    )

    plt.plot(
        results["polyak"][0],
        results["polyak"][1],
        color=COLORS["polyak"],
        label="Polyak",
    )

    plt.plot(
        results["nesterov"][0],
        results["nesterov"][1],
        color=COLORS["nesterov"],
        label="Nesterov",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(top=2)

    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.legend(
        loc="best",
    )

    plt.tight_layout()

    pdf_path = OUTPUT_DIR / "blowup_time.pdf"
    png_path = OUTPUT_DIR / "blowup_time.png"

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.savefig(
        png_path,
        bbox_inches="tight",
        dpi=200,
    )

    plt.show()

    return pdf_path, png_path


# =============================================================================
# Main
# =============================================================================

def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"device = {device}")
    print(f"B = {BATCH_SIZE}")
    print(f"eta = {ETA}")
    print("rho_k = 1 - 1/(k+1)")

    theta_star, lam = make_problem(
        d=D,
        beta=BETA,
        s=S,
        device=device,
    )

    results = {}

    results["sgd"] = run_method(
        "sgd",
        theta_star,
        lam,
        ETA,
        BATCH_SIZE,
        STEPS,
        NRUNS,
        RECORD_EVERY,
        SEED,
        device,
    )

    results["polyak"] = run_method(
        "polyak",
        theta_star,
        lam,
        ETA,
        BATCH_SIZE,
        STEPS,
        NRUNS,
        RECORD_EVERY,
        SEED + 1,
        device,
    )

    results["nesterov"] = run_method(
        "nesterov",
        theta_star,
        lam,
        ETA,
        BATCH_SIZE,
        STEPS,
        NRUNS,
        RECORD_EVERY,
        SEED + 2,
        device,
    )

    pdf_path, png_path = plot_results(results)

    print(f"Saved figure to: {pdf_path}")
    print(f"Saved figure to: {png_path}")


if __name__ == "__main__":
    main()