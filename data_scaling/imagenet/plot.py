#!/usr/bin/env python3
"""Redraw archived ImageNet results, optionally reaggregating per-run summaries.

All output is written to a separate directory; published results stay unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from source.imagenet_momentum_experiment import (
    aggregate_configurations,
    load_summaries,
    select_best_by_batch,
    write_csv,
)
from source.plot_imagenet_cbs import load_optimal_losses, plot_cbs

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "results/paper_3epochs")
    parser.add_argument("--output", type=Path, default=HERE / "outputs/figures")
    parser.add_argument("--reaggregate", action="store_true", help="Recompute selection from per-run summary.json files")
    parser.add_argument("--required-runs", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--boundaries", type=float, nargs=2, metavar=("SMALL_END", "ULTRA_START"))
    args = parser.parse_args()
    if args.required_runs < 1 or args.dpi < 1:
        parser.error("--required-runs and --dpi must be positive")
    data, output = args.data_dir.expanduser().resolve(), args.output.expanduser().resolve()
    archived = (HERE / "results").resolve()
    if output == data or data in output.parents or output == archived or archived in output.parents:
        parser.error("Use an output directory separate from the input and archived results/")
    if args.reaggregate:
        summaries = load_summaries(data)
        if not summaries:
            parser.error("No per-run summary.json files found in --data-dir")
        aggregated = aggregate_configurations(summaries)
        selected = select_best_by_batch(aggregated, args.required_runs)
        output.mkdir(parents=True, exist_ok=True)
        write_csv(summaries, output / "all_runs.csv")
        write_csv(aggregated, output / "aggregated_configurations.csv")
        write_csv(selected, output / "best_by_batch.csv")
    else:
        # Validate the source table before creating output files.
        load_optimal_losses(data / "best_by_batch.csv")
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data / "best_by_batch.csv", output / "best_by_batch.csv")
    losses = load_optimal_losses(output / "best_by_batch.csv")
    plot_cbs(losses, output, dpi=args.dpi, boundaries=args.boundaries)
    print(f"Saved ImageNet figures to {output}")


if __name__ == "__main__":
    main()
