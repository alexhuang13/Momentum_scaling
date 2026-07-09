"""Plot results from run_baseline.py (experiment A) and run_scaling.py (B).

    python -m experiments.plot --baseline results/baseline.json
    python -m experiments.plot --scaling  results/scaling.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt

OPT_STYLE = {
    "sgd": ("SGD", "^", "tab:green"),
    "heavy_ball": ("Heavy Ball", "o", "tab:blue"),
    "nesterov": ("Nesterov", "s", "tab:orange"),
}


def _mean_curve(seed_curves):
    """Average a list of equal-length curves elementwise."""
    n = min(len(c) for c in seed_curves)
    return [sum(c[i] for c in seed_curves) / len(seed_curves) for i in range(n)]


def plot_baseline(path, out):
    with open(path) as f:
        data = json.load(f)
    cfg = data["config"]

    # group runs by optimizer
    by_opt = defaultdict(list)
    for run in data["runs"]:
        by_opt[run["meta"]["optimizer"]].append(run)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    for opt, runs in by_opt.items():
        label, marker, color = OPT_STYLE[opt]
        # train loss vs step (average across seeds on the shortest common length)
        tl = _mean_curve([r["train_loss"] for r in runs])
        steps = runs[0]["step"][:len(tl)]
        ax_loss.plot(steps, tl, color=color, label=label)

        acc = _mean_curve([r["test_acc"] for r in runs])
        esteps = runs[0]["eval_step"][:len(acc)]
        ax_acc.plot(esteps, acc, marker=marker, color=color, label=label)

    ax_loss.set_xlabel("optimizer step")
    ax_loss.set_ylabel("train loss (cross-entropy)")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Training loss")
    ax_loss.grid(True, which="both", alpha=0.3)
    ax_loss.legend()

    ax_acc.set_xlabel("optimizer step")
    ax_acc.set_ylabel("test accuracy")
    ax_acc.set_title("Test accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    fig.suptitle(f"Experiment A: {cfg['dataset']} / {cfg['model']} "
                 f"(lr={cfg['lr']}, rho={cfg['rho']})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_scaling(path, out):
    with open(path) as f:
        data = json.load(f)
    cfg = data["config"]
    summary = data["summary"]
    slopes = data["slopes"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for opt, s in summary.items():
        label, marker, color = OPT_STYLE[opt]
        slope = slopes[opt]["slope"]
        ax.plot(s["D"], s["test_loss_mean"], marker=marker, color=color,
                label=f"{label} (slope={slope:.3f})")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("compute budget  D = B*K  (processed samples)")
    ax.set_ylabel("final test loss (cross-entropy)")
    ax.set_title(f"Experiment B: {cfg['dataset']} / {cfg['model']}  "
                 f"(B={cfg['batch_size']}, lr={cfg['lr']}, rho={cfg['rho']})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--scaling", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    if args.baseline:
        plot_baseline(args.baseline, args.out or "results/baseline.png")
    if args.scaling:
        plot_scaling(args.scaling, args.out or "results/scaling.png")
    if not args.baseline and not args.scaling:
        p.error("pass --baseline and/or --scaling")


if __name__ == "__main__":
    main()
