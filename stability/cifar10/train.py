#!/usr/bin/env python3
"""Run the original revised CIFAR trainer with portable data/output defaults.

Use --preset smoke for a synthetic-data training check, or sweep.py for the
documented fresh grid. Additional flags are forwarded to the original CLI.
"""
import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    root = Path(__file__).resolve().parent
    implementation = root / "source/cifar_stability_revised.py"
    sys.argv = [str(implementation),
                "--data-root", os.environ.get("CIFAR10_ROOT", str(root / "data/cifar10")),
                "--output", str(root / "outputs/train"), *sys.argv[1:]]
    runpy.run_path(str(implementation), run_name="__main__")


if __name__ == "__main__":
    main()
