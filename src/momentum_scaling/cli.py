"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform

import torch

from .core import resolve_device, resolve_dtype, seed_everything
from .stability import StabilityRuntime, stability_experiment
from .scaling import (
    ScalingRuntime,
    nesterov_scaling_experiment,
    polyak_scaling_experiment,
    scaling_experiment,
)
from .loss_dynamics import LossDynamicsRuntime, loss_dynamics_experiment
from .blowup import BlowupRuntime, blowup_experiment
from .plotting import (
    plot_blowup,
    plot_loss_dynamics,
    plot_saved,
    plot_scaling,
    plot_stability,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reproduce momentum-scaling experiments")
    p.add_argument(
        "experiment",
        choices=(
            "stability",
            "scaling",
            "loss-dynamics",
            "blowup",
            "polyak-scaling",
            "nesterov-scaling",
            "all",
            "plot",
        ),
    )
    p.add_argument("--preset", choices=("quick", "paper"), default="quick")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    p.add_argument("--dtype", choices=("auto", "float32", "float64"), default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.experiment == "plot":
        plot_saved(args.output_dir)
        return
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)
    print(
        f"python={platform.python_version()} torch={torch.__version__} device={device} dtype={dtype}"
    )
    quick = args.preset == "quick"
    if args.experiment in ("stability", "all"):
        runtime = StabilityRuntime(
            device=device, dtype=dtype, seed=args.seed, output_dir=args.output_dir
        )
        payload = stability_experiment(runtime, quick=quick)
        plot_stability(payload, args.output_dir)
    if args.experiment in ("scaling", "all"):
        runtime = ScalingRuntime(
            device=device, dtype=dtype, seed=args.seed, output_dir=args.output_dir
        )
        studies = scaling_experiment(runtime, quick=quick)
        for payload in studies.values():
            plot_scaling(payload, args.output_dir)
    if args.experiment == "polyak-scaling":
        payload = polyak_scaling_experiment(
            device=device,
            dtype=dtype,
            seed=args.seed,
            output_dir=args.output_dir,
            quick=quick,
        )
        plot_scaling(payload, args.output_dir)
    if args.experiment == "nesterov-scaling":
        payload = nesterov_scaling_experiment(
            device=device,
            dtype=dtype,
            seed=args.seed,
            output_dir=args.output_dir,
            quick=quick,
        )
        plot_scaling(payload, args.output_dir)
    if args.experiment in ("loss-dynamics", "all"):
        runtime = LossDynamicsRuntime(
            device=device, dtype=dtype, seed=args.seed, output_dir=args.output_dir
        )
        payload = loss_dynamics_experiment(runtime, quick=quick)
        plot_loss_dynamics(payload, args.output_dir)
    if args.experiment in ("blowup", "all"):
        runtime = BlowupRuntime(
            device=device, dtype=dtype, seed=args.seed, output_dir=args.output_dir
        )
        payload = blowup_experiment(runtime, quick=quick)
        plot_blowup(payload, args.output_dir)


if __name__ == "__main__":
    main()
