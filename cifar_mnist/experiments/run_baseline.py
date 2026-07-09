"""Experiment A: baseline optimizer comparison.

For a given (model, dataset) pair, train SGD / Heavy Ball / Nesterov with a SHARED
fixed learning rate (per the chosen setup) and rho=0.9 for the momentum methods,
over the full training set for a fixed number of epochs. Records train-loss /
test-loss / test-accuracy curves and writes them to a JSON file for plotting.

Usage:
    python -m experiments.run_baseline --config configs/mnist_fc.yaml
    python -m experiments.run_baseline --dataset cifar10 --model cnn --lr 0.01 --epochs 40
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from .data import get_loaders
from .train import get_device, train

OPTIMIZERS = ["sgd", "heavy_ball", "nesterov"]


def parse_args():
    p = argparse.ArgumentParser(description="Experiment A: baseline comparison")
    p.add_argument("--config", type=str, default=None,
                   help="optional YAML/JSON config; CLI flags override it")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--model", type=str, default="fc")
    p.add_argument("--lr", type=float, default=0.01,
                   help="SHARED fixed learning rate for all three optimizers")
    p.add_argument("--rho", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out", type=str, default="results/baseline.json")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-bn", action="store_true",
                   help="CNN only: BatchNorm variant, drop FC dropout")
    return p.parse_args()


def load_config(args):
    """Merge an optional config file with CLI args (CLI wins only when set)."""
    if args.config:
        cfg = _read_config(args.config)
        for k, v in cfg.items():
            k2 = k.replace("-", "_")
            if hasattr(args, k2):
                setattr(args, k2, v)
    return args


def _read_config(path):
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
            return yaml.safe_load(text)
        except ImportError:
            raise SystemExit("PyYAML not installed; use a .json config or "
                             "`pip install pyyaml`.")
    return json.loads(text)


def main():
    args = load_config(parse_args())
    device = get_device(args.device)
    print(f"device={device}  dataset={args.dataset}  model={args.model}  "
          f"lr={args.lr}  rho={args.rho}  epochs={args.epochs}")

    model_kwargs = {"use_bn": args.use_bn} if args.model == "cnn" else {}

    results = {"config": vars(args), "runs": []}
    for opt in OPTIMIZERS:
        for seed in args.seeds:
            train_loader, test_loader, meta = get_loaders(
                args.dataset, data_dir=args.data_dir,
                batch_size=args.batch_size, subset_seed=seed,
                num_workers=args.num_workers)
            hist = train(args.model, opt, train_loader, test_loader,
                         lr=args.lr, rho=args.rho,
                         weight_decay=args.weight_decay, epochs=args.epochs,
                         device=device, seed=seed, model_kwargs=model_kwargs)
            results["runs"].append(dataclasses.asdict(hist))
            print(f"  done: {opt} seed={seed}  -> {hist.final()}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"wrote {args.out}  ({len(results['runs'])} runs)")


if __name__ == "__main__":
    main()
