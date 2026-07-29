"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform

import torch

from .core import resolve_device, resolve_dtype, seed_everything
from .experiments import Runtime, data_scaling_experiment, stability_experiment
from .plotting import plot_data_scaling, plot_saved, plot_stability


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reproduce momentum-scaling experiments")
    p.add_argument("experiment", choices=("stability", "data-scaling", "all", "plot"))
    p.add_argument("--preset", choices=("quick", "paper"), default="quick")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    p.add_argument("--dtype", choices=("auto", "float32", "float64"), default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.experiment == "plot":
        plot_saved(args.output_dir)
        return
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)
    print(f"python={platform.python_version()} torch={torch.__version__} device={device} dtype={dtype}")
    runtime = Runtime(device=device, dtype=dtype, seed=args.seed, output_dir=args.output_dir)
    quick = args.preset == "quick"
    if args.experiment in ("stability", "all"):
        payload = stability_experiment(runtime, quick=quick)
        plot_stability(payload, args.output_dir)
    if args.experiment in ("data-scaling", "all"):
        payload = data_scaling_experiment(runtime, quick=quick)
        plot_data_scaling(payload, args.output_dir)


if __name__ == "__main__":
    main()
