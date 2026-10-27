from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


ALGORITHMS = ("sgd", "polyak", "nesterov")


def comma_list(text: str, cast: type = str) -> list:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("the comma-separated list cannot be empty")
    try:
        return [cast(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def float_tag(value: float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def effective_learning_rates(
    batch: int,
    *,
    reference_batch: int,
    reference_effective_lr: float,
    effective_lr_cap: float,
    multipliers: Sequence[float],
) -> list[float]:
    center = min(
        reference_effective_lr * batch / reference_batch,
        effective_lr_cap,
    )
    return sorted({center * multiplier for multiplier in multipliers})


def build_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    algorithm: str,
    batch: int,
    effective_lr: float,
    raw_lr: float,
    momentum: float,
) -> list[str]:
    return [
        sys.executable,
        str(args.script),
        "train",
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(run_dir),
        "--model",
        args.model,
        "--norm",
        args.norm,
        "--algorithm",
        algorithm,
        "--lr",
        format(raw_lr, ".12g"),
        "--momentum",
        format(momentum, ".12g"),
        "--weight-decay",
        format(args.weight_decay, ".12g"),
        "--global-batch",
        str(batch),
        "--max-per-device-batch",
        str(args.max_per_device_batch),
        "--epochs-equivalent",
        format(args.epochs, ".12g"),
        "--schedule",
        args.schedule,
        "--precision",
        args.precision,
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--log-every-updates",
        str(args.log_every_updates),
        "--final-window-updates",
        str(args.final_window_updates),
    ]


def write_manifest(rows: Sequence[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "algorithm",
        "global_batch",
        "epochs_equivalent",
        "effective_lr",
        "raw_lr",
        "momentum",
        "seed",
        "run_dir",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        type=Path,
        default=Path("/root/exp/imagenet_momentum_experiment.py"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/autodl-tmp/imagenet"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/root/exp/results/pilot_momentum"),
    )
    parser.add_argument("--batches", default="32,128,512,2048,4096")
    parser.add_argument("--algorithms", default="sgd,polyak,nesterov")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--reference-batch", type=int, default=256)
    parser.add_argument("--reference-effective-lr", type=float, default=0.1)
    parser.add_argument("--effective-lr-cap", type=float, default=0.4)
    parser.add_argument("--multipliers", default="0.5,1,2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--norm", choices=("group", "batch"), default="group")
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-per-device-batch", type=int, default=128)
    parser.add_argument("--log-every-updates", type=int, default=1000)
    parser.add_argument("--final-window-updates", type=int, default=100)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch jobs; without this flag only commands are generated",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop after a child process returns a nonzero status",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="do not call the main script's summarize command after execution",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.script = args.script.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.runs_root = args.runs_root.expanduser().resolve()

    if not args.script.is_file():
        raise SystemExit(f"main experiment script not found: {args.script}")
    if args.execute and not args.data_root.is_dir():
        raise SystemExit(f"ImageNet root not found: {args.data_root}")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if not 0 <= args.momentum < 1:
        raise SystemExit("--momentum must lie in [0, 1)")

    batches = comma_list(args.batches, int)
    algorithms = comma_list(args.algorithms, str)
    multipliers = comma_list(args.multipliers, float)
    invalid = set(algorithms) - set(ALGORITHMS)
    if invalid:
        raise SystemExit(f"unsupported algorithms: {sorted(invalid)}")
    if any(batch <= 0 for batch in batches):
        raise SystemExit("all batch sizes must be positive")
    if any(multiplier <= 0 for multiplier in multipliers):
        raise SystemExit("all learning-rate multipliers must be positive")

    args.runs_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, list[str], dict[str, object]]] = []
    for batch in batches:
        effective_lrs = effective_learning_rates(
            batch,
            reference_batch=args.reference_batch,
            reference_effective_lr=args.reference_effective_lr,
            effective_lr_cap=args.effective_lr_cap,
            multipliers=multipliers,
        )
        for algorithm in algorithms:
            rho = 0.0 if algorithm == "sgd" else args.momentum
            for effective_lr in effective_lrs:
                raw_lr = effective_lr if algorithm == "sgd" else (1.0 - rho) * effective_lr
                run_name = (
                    f"{algorithm}_B{batch}_geff{float_tag(effective_lr)}_"
                    f"lr{float_tag(raw_lr)}_rho{float_tag(rho)}_seed{args.seed}"
                )
                run_dir = args.runs_root / run_name
                command = build_command(
                    args,
                    run_dir=run_dir,
                    algorithm=algorithm,
                    batch=batch,
                    effective_lr=effective_lr,
                    raw_lr=raw_lr,
                    momentum=rho,
                )
                row: dict[str, object] = {
                    "algorithm": algorithm,
                    "global_batch": batch,
                    "epochs_equivalent": args.epochs,
                    "effective_lr": float(format(effective_lr, ".12g")),
                    "raw_lr": float(format(raw_lr, ".12g")),
                    "momentum": rho,
                    "seed": args.seed,
                    "run_dir": str(run_dir),
                    "status": "pending",
                }
                jobs.append((run_dir, command, row))

    command_file = args.runs_root / "commands_pilot.sh"
    command_file.write_text(
        "#!/usr/bin/env bash\nset -u\n\n"
        + "\n".join(shell_join(command) for _, command, _ in jobs)
        + "\n",
        encoding="utf-8",
    )
    command_file.chmod(0o755)

    manifest_rows = [row for _, _, row in jobs]
    manifest_path = args.runs_root / "pilot_manifest.csv"
    write_manifest(manifest_rows, manifest_path)

    print(f"generated {len(jobs)} configurations")
    print(f"commands: {command_file}")
    print(f"manifest: {manifest_path}")
    for index, (_, command, _) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {shell_join(command)}")

    if not args.execute:
        print("dry run only; add --execute to launch the configurations")
        return

    started = time.time()
    failures = 0
    for index, (run_dir, command, row) in enumerate(jobs, start=1):
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            row["status"] = "skipped_completed"
            print(f"[{index}/{len(jobs)}] skip completed: {run_dir.name}", flush=True)
            write_manifest(manifest_rows, manifest_path)
            continue

        print(f"[{index}/{len(jobs)}] start: {run_dir.name}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0 and summary_path.is_file():
            row["status"] = "completed"
        else:
            failures += 1
            row["status"] = f"failed_returncode_{completed.returncode}"
            print(
                f"configuration failed with return code {completed.returncode}: {run_dir}",
                flush=True,
            )
            if args.stop_on_error:
                write_manifest(manifest_rows, manifest_path)
                raise SystemExit(completed.returncode or 1)
        write_manifest(manifest_rows, manifest_path)

    if not args.no_summary:
        summarize_command = [
            sys.executable,
            str(args.script),
            "summarize",
            "--runs-root",
            str(args.runs_root),
            "--required-runs",
            "1",
        ]
        print(f"summarizing: {shell_join(summarize_command)}", flush=True)
        completed = subprocess.run(summarize_command, check=False)
        if completed.returncode != 0:
            failures += 1
            print(f"summary command failed with return code {completed.returncode}")

    elapsed = time.time() - started
    print(f"pilot finished in {elapsed / 3600:.2f} hours with {failures} failures")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
