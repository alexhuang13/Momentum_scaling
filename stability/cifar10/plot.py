#!/usr/bin/env python3
"""Redraw archived CIFAR aggregate results without unavailable raw trial bundles.

Reuses the original boundary extraction, fitting and plotting functions.
This verifies aggregate counts, not raw training histories or their provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from source.plot_cifar_all import boundaries, fit_models, load_cells, plot_all, write_csv

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=HERE / "results/paper_2000steps")
    parser.add_argument("--output", type=Path, default=HERE / "outputs/redrawn_summary")
    parser.add_argument("--stable-fraction", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--individual", action="store_true")
    args = parser.parse_args()
    if not 0 < args.stable_fraction < 1 or args.dpi <= 0:
        parser.error("Require 0 < --stable-fraction < 1 and --dpi > 0")
    source, out = args.summary_dir.expanduser().resolve(), args.output.expanduser().resolve()
    archived = (HERE / "results").resolve()
    if out == source or source in out.parents or out == archived or archived in out.parents:
        parser.error("Use a separate output directory outside the archived results/")
    csv_path = source / "merged_cells.csv"
    cells, excluded = load_cells(csv_path)
    points, audit = boundaries(cells, args.stable_fraction)
    fits = fit_models(points)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "empirical_boundaries.csv", points)
    write_csv(out / "boundary_extraction.csv", audit)
    write_csv(out / "fit_parameters.csv", fits)
    report = {
        "source_csv": str(csv_path),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "mode": "aggregate_summary_only",
        "raw_trials_reaudited": False,
        "retained_cells": len(cells),
        "excluded_cells": len(excluded),
        "stable_fraction": args.stable_fraction,
        "fits": fits,
    }
    (out / "redraw_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_all(cells, points, fits, out, args.dpi, args.individual)
    print(f"Redrew {len(cells)} aggregate cells in {out}; raw trials were not reaudited.")


if __name__ == "__main__":
    main()
