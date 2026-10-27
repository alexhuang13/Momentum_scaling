#!/usr/bin/env python3
"""Train one ImageNet configuration using the original experiment implementation.

Example:
    python train.py --data-root /data/imagenet --algorithm polyak \
        --global-batch 256 --lr 0.08 --momentum 0.9 \
        --sample-budget 3843501 --precision bf16 --output-dir outputs/single
"""
from pathlib import Path
import runpy
import sys


def main() -> None:
    implementation = Path(__file__).resolve().parent / "source/imagenet_momentum_experiment.py"
    sys.argv[0] = str(implementation)
    sys.argv.insert(1, "train")
    runpy.run_path(str(implementation), run_name="__main__")


if __name__ == "__main__":
    main()
