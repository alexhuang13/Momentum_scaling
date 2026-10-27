#!/usr/bin/env python3
"""Prepare the explicit CIFAR stability grid; add --execute to train.

This fresh revised-mode scan is separate from the historical adaptive reruns.
Additional arguments override the defaults below using the original trainer CLI.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PAPER_GRID = (
    "--preset", "full",
    "--batches", "1,2,4,8,16,32,64,128,256",
    "--gaps", "0.015625,0.02209708691207961,0.03125,0.04419417382415922,0.0625,0.08838834764831845,0.125,0.1767766952966369,0.25",
    "--fixed-batch", "4", "--fixed-gap", "0.015625",
    "--lr-min", "0.00390625", "--lr-max", "8", "--lr-count", "23",
    "--seeds", "0,1,2,3,4", "--steps", "2000",
    "--eval-every", "50", "--tail-evals", "10", "--probe-size", "1024",
    "--parameter-safety-mode", "record",
)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--execute", action="store_true")
    options, forwarded = parser.parse_known_args()
    if options.execute and "--dry-run" in forwarded:
        parser.error("Choose either --execute or --dry-run")
    if "--help" in forwarded or "-h" in forwarded:
        print(__doc__, flush=True)
        print("Wrapper option: --execute runs training; the default only writes the grid.\n", flush=True)
    if not options.execute and "--dry-run" not in forwarded:
        forwarded.insert(0, "--dry-run")
    command = [sys.executable, str(HERE / "train.py"), *PAPER_GRID,
               "--devices", os.environ.get("DEVICES", "cuda:0,cuda:1"),
               "--output", os.environ.get("CIFAR_OUTPUT", str(HERE / "outputs/new_scan")),
               *forwarded]
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
