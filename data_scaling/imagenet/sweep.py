#!/usr/bin/env python3
"""Generate or execute the 396 archived ImageNet configurations.

The checked-in grid contains configuration values, not measured outcomes.
Training uses the original experiment script. Default mode only writes commands.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FIELDS = (
    "algorithm", "model", "norm", "seed", "global_batch", "micro_batch",
    "lr", "momentum", "weight_decay", "schedule", "warmup_epochs_equivalent",
    "precision", "sample_budget",
)


def load_grid(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = {"run_name", *FIELDS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing grid columns: {sorted(missing)}")
        rows = list(reader)
    names = [r["run_name"] for r in rows]
    if not rows or len(names) != len(set(names)):
        raise ValueError("Grid must be nonempty and run names must be unique")
    if any(Path(n).name != n or n in (".", "..") for n in names):
        raise ValueError("Run names must be single directory names")
    return rows


def train_command(row: dict[str, str], data: Path, output: Path, workers: int) -> list[str]:
    command = [sys.executable, str(HERE / "source/imagenet_momentum_experiment.py"),
               "train", "--data-root", str(data), "--output-dir", str(output),
               "--workers", str(workers), "--epochs-equivalent", "3",
               "--log-every-updates", "1000", "--final-window-updates", "100"]
    for key in FIELDS:
        command.extend(["--" + key.replace("_", "-"), row[key]])
    return command


def verify_existing(path: Path, row: dict[str, str]) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    aliases = {"micro_batch": "micro_batch_per_device", "sample_budget": "sample_budget_requested"}
    text_fields = {"algorithm", "model", "norm", "schedule", "precision"}
    for key in FIELDS:
        actual = summary.get(aliases.get(key, key))
        expected = row[key]
        if key in text_fields:
            matches = actual == expected
        else:
            matches = actual is not None and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12)
        if not matches:
            raise ValueError(f"Existing summary has different {key}: {path}")
    # A completed divergence outcome is still a result; keep it for aggregation.
    if not summary.get("divergence_reason") and summary.get("updates_completed") != int(row["sample_budget"]) // int(row["global_batch"]):
        raise ValueError(f"Incomplete existing run: {path}; choose a new output directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=HERE / "configs/paper_grid.csv")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("IMAGENET_ROOT", "data/imagenet")))
    parser.add_argument("--runs-root", type=Path, default=HERE / "outputs/paper_3epochs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="Run training; otherwise only generate the command list")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    rows = load_grid(args.grid.expanduser().resolve())
    data, root = args.data_root.expanduser().resolve(), args.runs_root.expanduser().resolve()
    archived = (HERE / "results").resolve()
    if root == archived or archived in root.parents:
        parser.error("Use a new runs directory outside the archived results/")
    if args.execute and not all((data / part).is_dir() for part in ("train", "val")):
        parser.error("--data-root must contain ImageNet train/ and val/ directories")
    root.mkdir(parents=True, exist_ok=True)
    commands = [train_command(r, data, root / r["run_name"], args.workers) for r in rows]
    # One manifest describes the full grid; per-run records remain authoritative.
    with (root / "reproduction_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "commands_reproduce.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(map(shlex.join, commands)) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(rows)} configurations in {root}", flush=True)
    if not args.execute:
        print("Commands generated. Add --execute to train; no training was run.")
        return
    for index, (row, command) in enumerate(zip(rows, commands), 1):
        summary = root / row["run_name"] / "summary.json"
        if summary.exists():
            verify_existing(summary, row)
            print(f"[{index}/{len(rows)}] keep existing outcome: {row['run_name']}", flush=True)
            continue
        print(f"[{index}/{len(rows)}] train: {row['run_name']}", flush=True)
        subprocess.run(command, check=True)
        if not summary.is_file():
            raise RuntimeError(f"Training returned without a summary: {summary}")
    subprocess.run([sys.executable, str(HERE / "source/imagenet_momentum_experiment.py"),
                    "summarize", "--runs-root", str(root), "--required-runs", "1", "--no-plot"], check=True)
    subprocess.run([sys.executable, str(HERE / "source/plot_imagenet_cbs.py"), "--data-dir", str(root)], check=True)


if __name__ == "__main__":
    main()
