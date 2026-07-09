"""Experiment B: scaling law with compute budget D = B * K.

This ports the notebook's "excess loss vs. data budget D" curve to real networks.
We FIX the batch size B and the full training set, and sweep the number of
optimizer steps K over a geometric grid, so the processed-sample / compute budget
is D = B * K. For each D we train for exactly K steps (looping over the data as
needed) with a SHARED fixed learning rate and rho=0.9, then record the final test
loss (cross-entropy) as the real-network analogue of the excess loss. We fit
log(test_loss) vs log(D) per optimizer and report the slope.

IMPORTANT (honesty note): on real data we do NOT know the source/spectrum
exponents (s, beta), so we do NOT scale B, K, rho with D as the notebook's
theory prescribes -- that scaling requires known exponents. This experiment is
therefore an *empirical* power-law observation of loss vs. compute, not a
reproduction of the exact D^{-s/(s+1)} rate. See README.

Usage:
    python -m experiments.run_scaling --config configs/mnist_fc.yaml
    python -m experiments.run_scaling --dataset cifar10 --model cnn \
        --batch-size 128 --k-min 8 --k-max 15
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os

from .data import get_loaders
from .run_baseline import OPTIMIZERS, load_config
from .train import get_device, train


def parse_args():
    p = argparse.ArgumentParser(description="Experiment B: D = B*K scaling")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--model", type=str, default="fc")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128,
                   help="fixed B; D = B * K")
    p.add_argument("--k-min", type=int, default=7,
                   help="min log2 of step count K")
    p.add_argument("--k-max", type=int, default=14,
                   help="max log2 of step count K")
    p.add_argument("--n-points", type=int, default=8,
                   help="number of K values (log-spaced between k-min/k-max)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out", type=str, default="results/scaling.json")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-bn", action="store_true")
    return p.parse_args()


def k_grid(k_min, k_max, n_points):
    """Geometric grid of step counts K = 2^e for e in [k_min, k_max]."""
    exps = [k_min + (k_max - k_min) * i / (n_points - 1)
            for i in range(n_points)]
    return sorted({int(round(2 ** e)) for e in exps})


def fit_slope(D_list, loss_list):
    """Least-squares slope of log(loss) vs log(D)."""
    xs = [math.log(d) for d in D_list]
    ys = [math.log(max(l, 1e-12)) for l in loss_list]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else float("nan")
    intercept = my - slope * mx
    return slope, intercept


def main():
    args = load_config(parse_args())
    device = get_device(args.device)
    B = args.batch_size
    Ks = k_grid(args.k_min, args.k_max, args.n_points)
    print(f"device={device}  dataset={args.dataset}  model={args.model}  "
          f"B={B}  K grid={Ks}  D=B*K in [{B*Ks[0]}, {B*Ks[-1]}]")

    model_kwargs = {"use_bn": args.use_bn} if args.model == "cnn" else {}
    # A large epoch cap; max_steps=K is the real stopping criterion.
    epoch_cap = max(1, math.ceil(max(Ks) * B / 50000) + 1)

    results = {"config": vars(args), "K_grid": Ks, "batch_size": B, "runs": []}
    summary = {opt: {"D": [], "test_loss_mean": []} for opt in OPTIMIZERS}

    for opt in OPTIMIZERS:
        for K in Ks:
            D = B * K
            losses = []
            for seed in args.seeds:
                train_loader, test_loader, _ = get_loaders(
                    args.dataset, data_dir=args.data_dir, batch_size=B,
                    subset_seed=seed, num_workers=args.num_workers)
                hist = train(args.model, opt, train_loader, test_loader,
                             lr=args.lr, rho=args.rho,
                             weight_decay=args.weight_decay,
                             epochs=epoch_cap, max_steps=K, device=device,
                             seed=seed, model_kwargs=model_kwargs,
                             eval_every_epochs=10 ** 9,  # only final eval
                             verbose=False)
                fin = hist.final()
                fin["D"] = D
                fin["K"] = K
                fin["optimizer"] = opt
                fin["seed"] = seed
                losses.append(fin["test_loss"])
                results["runs"].append(fin)
            mean_loss = sum(losses) / len(losses)
            summary[opt]["D"].append(D)
            summary[opt]["test_loss_mean"].append(mean_loss)
            print(f"  {opt:10s} D={D:>8d} (K={K:>6d})  "
                  f"test_loss={mean_loss:.4e}")

    results["summary"] = summary
    results["slopes"] = {}
    for opt in OPTIMIZERS:
        slope, intercept = fit_slope(summary[opt]["D"],
                                     summary[opt]["test_loss_mean"])
        results["slopes"][opt] = {"slope": slope, "intercept": intercept}
        print(f"fitted slope {opt:10s}: {slope:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
