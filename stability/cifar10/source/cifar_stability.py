#!/usr/bin/env python3
"""Fixed-horizon CIFAR-10 stability scans for SGD, Polyak, and Nesterov.

Refine a completed scan with the same training protocol:
  python3 cifar_stability.py --refine-from output/fine_grid_seed0_2gpu \
      --seeds 0,1,2,3,4 --devices cuda:0,cuda:1 --output output/refine_5seed_2gpu
Add --dry-run first to inspect counts without training. Repeating the same
command resumes completed-trial caching. Refinement does not resume checkpoints.
The old directory is read-only; imported summaries retain their original hash.
Nonmonotone/censored seeds are reported, never discarded to manufacture a mean.

Merge a full scan and its boundary refinement, without training:
  python3 cifar_stability.py --merge-from output/fine_grid_seed0_2gpu \
      output/refine_5seed_2gpu --output output/merged_heatmaps
Duplicate seeds count once. Colors use observed seeds; coverage is saved too.
"""

from __future__ import annotations

import argparse
import ast
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import sys
import time
import traceback
from typing import Any, Iterable, Sequence


ALGORITHMS = ("sgd", "polyak", "nesterov")
SCHEMA_VERSION = 2
DEFAULT_DATA_ROOT = "/data/public/CIFAR-10/cifar-10-python"
# Previously distributed two-GPU script, before refinement was added. Its
# training functions are unchanged. Unknown old scripts are never silently reused.
LEGACY_TRAINING_SOURCE_HASHES = {
    "3be6106d28077c184cc9791f6ee12d5f89d8f3f810f732a4908bfd85ab24d852",
}
LEGACY_TRAINING_CODE_HASH = "eef1a8b500f88bb66cf05b92ae570ddb0ac5ea59fb5985bfdaf9b0b5d94760cf"
# Exact released refinement version affected by the empty log-axis plot bug.
# Resumption is allowed only with identical protocol/jobs and training code.
PLOT_FIX_RESUME_SOURCE_HASH = "7297aea3f1e9c445a0010a31d85d8a9b3eabbe79f995129013a6502581a82612"
ANALYSIS_ONLY_RESUME_SOURCE_HASHES = {
    PLOT_FIX_RESUME_SOURCE_HASH,
    "0d21bbe5b29cc33e5d49b79046fceaaf46fbbf92be98f921ac87a8864ab2c72a",
}

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "axis": "both",
        "batches": "2,4",
        "gaps": "0.25,0.5",
        "fixed_batch": 2,
        "fixed_gap": 0.25,
        "lrs": "0.0001,100",
        "seeds": "0",
        "steps": 4,
        "eval_every": 1,
        "tail_evals": 2,
        "probe_size": 8,
        "probe_batch": 4,
        "max_micro_batch": 1,
        "workers": 0,
        "threads": 1,
        "device": "cpu",
        "fake_data": True,
    },
    "pilot": {
        "axis": "both",
        "batches": "4,64",
        "gaps": "0.015625,0.125",
        "fixed_batch": 4,
        "fixed_gap": 0.015625,
        "lrs": "0.001,0.01,0.1,1,8",
        "seeds": "0",
        "steps": 2000,
        "eval_every": 50,
        "tail_evals": 5,
        "probe_size": 1024,
        "probe_batch": 256,
        "max_micro_batch": 64,
        "workers": 4,
        "threads": 4,
        "device": "cuda",
        "fake_data": False,
    },
    "full": {
        "axis": "both",
        "batches": "4,8,16,32,64,128,256",
        "gaps": "0.015625,0.03125,0.0625,0.125,0.25",
        "fixed_batch": 4,
        "fixed_gap": 0.015625,
        "lrs": "",
        "seeds": "0,1,2",
        "steps": 2000,
        "eval_every": 50,
        "tail_evals": 5,
        "probe_size": 1024,
        "probe_batch": 256,
        "max_micro_batch": 64,
        "workers": 4,
        "threads": 4,
        "device": "cuda",
        "fake_data": False,
    },
}


def import_training_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
    except ImportError as exc:
        raise SystemExit(
            "PyTorch and torchvision are required. Install a matching build first."
        ) from exc
    return torch, nn, torchvision, datasets, transforms


def comma_list(text: str, cast: Any = str) -> list[Any]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("the comma-separated list cannot be empty")
    try:
        return [cast(item) for item in values]
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def positive_list(text: str, cast: Any = float) -> list[Any]:
    values = comma_list(text, cast)
    if any(not math.isfinite(float(value)) or value <= 0 for value in values):
        raise ValueError("grid entries must be finite and positive")
    if len(values) != len(set(values)):
        raise ValueError("grid entries must be unique")
    return sorted(values)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_hashes() -> dict[str, str]:
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def training_code_hash() -> str:
    """Fingerprint the actual training code separately from planning/plotting."""
    names = {
        "import_training_stack", "seed_everything", "group_norm_factory",
        "build_model", "build_transforms", "PaperMomentum", "optimizer_for",
        "cifar_integrity_metadata", "build_data", "tensor_norm", "evaluate",
        "backward_batch", "classify", "run_trial",
    }
    source = Path(__file__).read_text(encoding="utf-8")
    lines = source.splitlines()
    selected = {
        node.name: "\n".join(lines[node.lineno - 1 : node.end_lineno])
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    }
    if set(selected) != names:
        raise RuntimeError("could not fingerprint all training functions")
    return digest(selected)


def explicit_device(value: str) -> str:
    if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("choose cpu, mps, cuda or cuda:N")
    return value


def device_list(value: str) -> list[str]:
    devices = comma_list(value)
    for device in devices:
        explicit_device(device)
    if len(devices) != len(set(devices)):
        raise ValueError("devices must be distinct")
    return devices


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=PRESETS, default="pilot")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="save the manifest only")
    mode.add_argument(
        "--summarize-only",
        action="store_true",
        help="rebuild tables and plots from saved trials",
    )
    mode.add_argument(
        "--merge-from", type=Path, nargs="+", metavar="RESULT_DIR",
        help="merge saved trial results into a separate heatmap folder; no training",
    )
    parser.add_argument("--max-new-trials", type=positive_int)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--refine-from", type=Path,
        help="completed scan directory; inherit its protocol and refine local LR grids",
    )
    parser.add_argument("--refine-seed", type=int, default=0,
                        help="source seed used to choose the refinement intervals")
    parser.add_argument("--refine-padding", type=float, default=math.sqrt(2),
                        help="multiplicative LR guard point on each side (>1)")
    parser.add_argument("--refine-subdivisions", type=positive_int, default=2,
                        help="split each interior LR interval geometrically")

    parser.add_argument("--axis", choices=("batch", "gap", "both"))
    parser.add_argument("--algorithms", default="sgd,polyak,nesterov")
    parser.add_argument("--batches")
    parser.add_argument("--gaps")
    parser.add_argument("--fixed-batch", type=int)
    parser.add_argument("--fixed-gap", type=float)
    parser.add_argument("--lrs")
    parser.add_argument("--lr-min", type=float, default=2**-10)
    parser.add_argument("--lr-max", type=float, default=8.0)
    parser.add_argument("--lr-count", type=int, default=21)
    parser.add_argument("--seeds")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--tail-evals", type=int)
    parser.add_argument("--probe-size", type=int)
    parser.add_argument("--probe-batch", type=int)
    parser.add_argument("--probe-seed", type=int, default=1729)
    parser.add_argument("--stable-loss-ratio", type=float, default=2.0)
    parser.add_argument("--growth-ratio", type=float, default=1.25)
    parser.add_argument("--learning-ratio", type=float, default=0.95)
    parser.add_argument("--divergence-loss-ratio", type=float, default=100.0)
    parser.add_argument("--max-grad-norm", type=float, default=1e6)
    parser.add_argument("--max-parameter-ratio", type=float, default=100.0)
    parser.add_argument("--required-fraction", type=float, default=1.0)
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--max-micro-batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--device", type=explicit_device)
    parser.add_argument(
        "--devices",
        default="",
        help="comma-separated devices for parallel independent trials, e.g. cuda:0,cuda:1",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    args.provided_options = [token.split("=", 1)[0] for token in raw if token.startswith("--")]
    preset = PRESETS[args.preset]
    for name, value in preset.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    if args.merge_from is not None:
        unsupported = set(args.provided_options) - {"--merge-from", "--output", "--no-plot"}
        if unsupported:
            raise ValueError("merge only reads saved results; remove: " + ", ".join(sorted(unsupported)))
        args.merge_from = [path.expanduser().resolve() for path in args.merge_from]
        if len(args.merge_from) < 2 or len(set(args.merge_from)) != len(args.merge_from):
            raise ValueError("--merge-from needs at least two distinct result directories")
        if args.output is None:
            args.output = args.merge_from[0].with_name("merged_heatmaps")
    if args.refine_from is not None:
        args.refine_from = args.refine_from.expanduser().resolve()
        allowed = {
            "--refine-from", "--refine-seed", "--refine-padding",
            "--refine-subdivisions", "--seeds", "--devices", "--device",
            "--output", "--dry-run", "--summarize-only", "--max-new-trials",
            "--no-plot",
        }
        unsupported = set(args.provided_options) - allowed
        if unsupported:
            raise ValueError(
                "refinement inherits the entire training protocol; remove: "
                + ", ".join(sorted(unsupported))
            )
        if "--seeds" not in args.provided_options:
            args.seeds = "0,1,2,3,4"
        if args.output is None:
            args.output = args.refine_from.with_name(args.refine_from.name + "_refined")
    if args.output is None:
        args.output = Path(__file__).resolve().parent / "output" / args.preset
    args.output = args.output.expanduser().resolve()
    if args.merge_from is not None:
        validate_merge_destination(args.merge_from, args.output)
    if args.refine_from is not None and (
        args.output == args.refine_from
        or args.output in args.refine_from.parents
        or args.refine_from in args.output.parents
    ):
        raise ValueError("refinement output must be separate from the source directory")
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    lr_shape_overridden = any(
        token.split("=", 1)[0] in {"--lr-min", "--lr-max", "--lr-count"}
        for token in raw
    )
    lrs_explicit = any(token.split("=", 1)[0] == "--lrs" for token in raw)
    if lr_shape_overridden and not lrs_explicit:
        args.lrs = ""
    args.device_list = device_list(args.devices) if args.devices else [args.device]
    return args


def validate_protocol(args: argparse.Namespace) -> tuple[list[str], list[int], list[float], list[int]]:
    algorithms = comma_list(args.algorithms)
    if set(algorithms) - set(ALGORITHMS) or len(algorithms) != len(set(algorithms)):
        raise ValueError(f"choose distinct algorithms from {ALGORITHMS}")
    batches = positive_list(args.batches, int)
    gaps = positive_list(args.gaps)
    seeds = comma_list(args.seeds, int)
    if min(seeds) < 0 or max(seeds) >= 2**32 or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    if not 0 < args.fixed_gap <= 1 or any(gap > 1 for gap in gaps):
        raise ValueError("momentum gaps must lie in (0, 1]")
    if args.lrs:
        lrs = positive_list(args.lrs)
    else:
        if not (0 < args.lr_min < args.lr_max < math.inf and args.lr_count >= 2):
            raise ValueError("require 0 < lr-min < lr-max and lr-count >= 2")
        lrs = [
            math.exp(
                math.log(args.lr_min)
                + i / (args.lr_count - 1) * math.log(args.lr_max / args.lr_min)
            )
            for i in range(args.lr_count)
        ]
    if len(lrs) < 2:
        raise ValueError("use at least two learning rates")
    for name in (
        "steps",
        "eval_every",
        "tail_evals",
        "probe_size",
        "probe_batch",
        "max_micro_batch",
        "fixed_batch",
        "threads",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    if 2 * args.tail_evals > math.ceil(args.steps / args.eval_every):
        raise ValueError("need at least 2 * tail-evals probe evaluations")
    for name in (
        "stable_loss_ratio",
        "growth_ratio",
        "divergence_loss_ratio",
        "max_grad_norm",
        "max_parameter_ratio",
        "learning_ratio",
    ):
        if not 0 < getattr(args, name) < math.inf:
            raise ValueError(f"{name} must be finite and positive")
    if not (
        args.learning_ratio < 1
        <= args.stable_loss_ratio
        < args.divergence_loss_ratio
        and args.growth_ratio >= 1
        and args.max_parameter_ratio > 1
    ):
        raise ValueError("inconsistent stability thresholds")
    if not 0 < args.required_fraction <= 1:
        raise ValueError("required-fraction must lie in (0, 1]")
    return algorithms, batches, lrs, seeds


def make_manifest(args: argparse.Namespace) -> dict[str, Any]:
    algorithms, batches, lrs, seeds = validate_protocol(args)
    gaps = positive_list(args.gaps)
    points: list[dict[str, Any]] = []
    if args.axis in ("batch", "both"):
        points.extend(
            {"axis": "batch", "x": batch, "batch": batch, "gap": args.fixed_gap}
            for batch in batches
        )
    if args.axis in ("gap", "both"):
        points.extend(
            {"axis": "gap", "x": gap, "batch": args.fixed_batch, "gap": gap}
            for gap in gaps
        )
    jobs: dict[str, dict[str, Any]] = {}
    for algorithm in algorithms:
        for point in points:
            gap = 1.0 if algorithm == "sgd" else point["gap"]
            if not args.fake_data and args.steps * gap < 30:
                raise ValueError(
                    f"K(1-rho)={args.steps * gap:g} < 30 for {algorithm}; "
                    "increase --steps or use a larger gap"
                )
            for lr in lrs:
                for seed in seeds:
                    trial = {
                        "algorithm": algorithm,
                        "batch": point["batch"],
                        "gap": gap,
                        "lr": lr,
                        "seed": seed,
                    }
                    trial_id = digest(trial)[:20]
                    jobs[trial_id] = {"id": trial_id, **trial}
    ignored = {
        "output",
        "dry_run",
        "summarize_only",
        "max_new_trials",
        "no_plot",
        "device_list",
        "provided_options", "refine_from", "refine_seed", "refine_padding",
        "refine_subdivisions", "merge_from",
    }
    config = {key: value for key, value in vars(args).items() if key not in ignored}
    config["devices"] = list(args.device_list)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "source_hashes": source_hashes(),
        "training_code_hash": training_code_hash(),
        "points": points,
        "algorithms": algorithms,
        "lrs": lrs,
        "seeds": seeds,
        "jobs": list(jobs.values()),
    }
    payload["fingerprint"] = digest(payload)
    return payload


def read_manifest(root: Path, *, verify_source: bool = False) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if digest(body) != manifest.get("fingerprint") or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manifest checksum or schema mismatch")
    if verify_source and manifest["source_hashes"] != source_hashes():
        raise ValueError("source changed; use a new output directory")
    return manifest


def select_resume_manifest(previous: dict[str, Any], requested: dict[str, Any]) -> dict[str, Any]:
    """Retain original result identity across audited analysis-only revisions."""
    if previous["fingerprint"] == requested["fingerprint"]:
        return previous
    ignored = {"fingerprint", "source_hashes"}
    previous_protocol = {key: value for key, value in previous.items() if key not in ignored}
    requested_protocol = {key: value for key, value in requested.items() if key not in ignored}
    if previous_protocol != requested_protocol:
        raise ValueError("output has a different protocol; choose a new --output")
    old_hashes = list(previous["source_hashes"].values())
    if (len(old_hashes) != 1 or old_hashes[0] not in ANALYSIS_ONLY_RESUME_SOURCE_HASHES
            or previous.get("training_code_hash") != LEGACY_TRAINING_CODE_HASH
            or training_code_hash() != LEGACY_TRAINING_CODE_HASH):
        raise ValueError("unrecognized source change; use a new output directory")
    print("Resuming a known analysis-compatible version; training protocol and jobs match.")
    # Do not rewrite the manifest or any saved result to manufacture a match.
    return previous


def result_path(root: Path, trial: dict[str, Any]) -> Path:
    return root / "trials" / trial["id"] / "summary.json"


def read_result(
    root: Path, manifest: dict[str, Any], trial: dict[str, Any]
) -> dict[str, Any] | None:
    path = result_path(root, trial)
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    origin = manifest.get("reused_results", {}).get(trial["id"])
    expected = origin["fingerprint"] if origin else manifest["fingerprint"]
    if row.get("fingerprint") != expected or row.get("trial") != trial:
        raise ValueError(f"mismatched cached result: {path}")
    if origin and digest(row) != origin["summary_digest"]:
        raise ValueError(f"imported result changed: {path}")
    return row


def configuration_key(algorithm: str, point: dict[str, Any]) -> str:
    return digest({
        "algorithm": algorithm, "batch": point["batch"],
        "gap": 1.0 if algorithm == "sgd" else point["gap"],
    })[:20]


def point_lrs(manifest: dict[str, Any], algorithm: str, point: dict[str, Any]) -> list[float]:
    if "refinement" in manifest:
        return manifest["refinement"]["grids"][configuration_key(algorithm, point)]["lrs"]
    return manifest["lrs"]


def refinement_grid(
    lrs: list[float], labels: list[bool], padding: float, subdivisions: int,
) -> tuple[list[float], dict[str, Any]]:
    """Cover all observed stable/unstable reversals, not just the last transition."""
    boundary = boundary_from_grid(lrs, labels)
    stable = [index for index, passed in enumerate(labels) if passed]
    failed = [index for index, passed in enumerate(labels) if not passed]
    if not stable:
        lower, upper = lrs[0] / padding, lrs[0]
    elif not failed:
        lower, upper = lrs[-1], lrs[-1] * padding
    else:
        first_failure, last_stable = min(failed), max(stable)
        lower = lrs[first_failure - 1] if first_failure else lrs[0] / padding
        upper = lrs[last_stable + 1] if last_stable + 1 < len(lrs) else lrs[-1] * padding
    anchors = sorted({lower, upper, *(lr for lr in lrs if lower <= lr <= upper)})
    refined = set(anchors)
    for left, right in zip(anchors, anchors[1:]):
        for index in range(1, subdivisions):
            refined.add(math.exp(math.log(left) + index / subdivisions * math.log(right / left)))
    # Snap rounding-equivalent guard points to old values so cached runs are reused.
    for value in (lower / padding, upper * padding):
        nearest = min(lrs, key=lambda lr: abs(math.log(lr / value)))
        refined.add(nearest if math.isclose(nearest, value, rel_tol=1e-5) else value)
    result = sorted(refined)
    if any(not math.isfinite(lr) or lr <= 0 for lr in result):
        raise ValueError("refinement produced a nonfinite or nonpositive learning rate")
    return result, {"source_status": boundary["status"], "focus_lower": lower, "focus_upper": upper}


def make_refinement_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    source = read_manifest(args.refine_from)
    if source.get("training_code_hash") != training_code_hash():
        old_hashes = set(source["source_hashes"].values())
        if (len(old_hashes) != 1 or not old_hashes <= LEGACY_TRAINING_SOURCE_HASHES
                or training_code_hash() != LEGACY_TRAINING_CODE_HASH):
            raise ValueError("source training code is unknown/different; do not mix these results")
    if args.refine_seed not in source["seeds"]:
        raise ValueError("--refine-seed is absent from the source scan")
    if not math.isfinite(args.refine_padding) or args.refine_padding <= 1:
        raise ValueError("--refine-padding must be finite and >1")
    if args.refine_subdivisions < 2:
        raise ValueError("--refine-subdivisions must be >=2")
    config = copy.deepcopy(source["config"])
    config["seeds"] = args.seeds
    if "--devices" in args.provided_options or "--device" in args.provided_options:
        config["devices"] = list(args.device_list)
    config["device"] = config["devices"][0]
    _, _, _, seeds = validate_protocol(argparse.Namespace(**config))
    source_jobs = {trial["id"]: trial for trial in source["jobs"]}
    grids, jobs, reused = {}, {}, {}
    source_results = {}
    for trial in source["jobs"]:
        row = read_result(args.refine_from, source, trial)
        if row is not None:
            source_results[trial["id"]] = row
    for algorithm in source["algorithms"]:
        for point in source["points"]:
            key = configuration_key(algorithm, point)
            if key in grids:
                continue
            gap = 1.0 if algorithm == "sgd" else point["gap"]
            old_lrs = point_lrs(source, algorithm, point)
            labels = []
            for lr in old_lrs:
                trial = {"algorithm": algorithm, "batch": point["batch"], "gap": gap,
                         "lr": lr, "seed": args.refine_seed}
                row = source_results.get(digest(trial)[:20])
                if row is None or not isinstance(row.get("bounded"), bool):
                    raise ValueError(f"source reference-seed grid is incomplete: {trial}")
                labels.append(row["bounded"])
            new_lrs, info = refinement_grid(old_lrs, labels, args.refine_padding, args.refine_subdivisions)
            grids[key] = {"algorithm": algorithm, "batch": point["batch"], "gap": gap,
                          "lrs": new_lrs, **info}
            for lr in new_lrs:
                for seed in seeds:
                    trial = {"algorithm": algorithm, "batch": point["batch"], "gap": gap,
                             "lr": lr, "seed": seed}
                    trial_id = digest(trial)[:20]
                    jobs[trial_id] = {"id": trial_id, **trial}
                    if trial_id in source_results:
                        row = source_results[trial_id]
                        if source_jobs[trial_id] != jobs[trial_id]:
                            raise ValueError("source trial identity mismatch")
                        reused[trial_id] = {"fingerprint": row["fingerprint"],
                                            "summary_digest": digest(row)}
    payload = {
        "schema_version": SCHEMA_VERSION, "config": config,
        "source_hashes": source_hashes(), "training_code_hash": training_code_hash(),
        "points": source["points"], "algorithms": source["algorithms"], "seeds": seeds,
        "lrs": sorted({lr for grid in grids.values() for lr in grid["lrs"]}),
        "jobs": list(jobs.values()), "reused_results": reused,
        "refinement": {"source_root": str(args.refine_from),
                       "source_fingerprint": source["fingerprint"], "seed": args.refine_seed,
                       "padding": args.refine_padding, "subdivisions": args.refine_subdivisions,
                       "grids": grids},
    }
    payload["fingerprint"] = digest(payload)
    return payload, source


def import_refinement_results(root: Path, manifest: dict[str, Any], source: dict[str, Any]) -> None:
    """Copy completed selected trials without rewriting their provenance or source."""
    source_root = Path(manifest["refinement"]["source_root"])
    atomic_json_dump(source, root / "source_manifest.json")
    for trial in manifest["jobs"]:
        if trial["id"] not in manifest["reused_results"]:
            continue
        if read_result(root, manifest, trial) is not None:
            continue
        row = read_result(source_root, source, trial)
        if row is None or digest(row) != manifest["reused_results"][trial["id"]]["summary_digest"]:
            raise ValueError("source result changed during import")
        directory = result_path(root, trial).parent
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "history.jsonl"):
            original = result_path(source_root, trial).parent / name
            if original.exists():
                shutil.copy2(original, directory / name)
        atomic_json_dump(row, result_path(root, trial))
    write_csv([
        {**{key: value for key, value in grid.items() if key != "lrs"},
         "lr_count": len(grid["lrs"]), "lrs": ",".join(f"{lr:.12g}" for lr in grid["lrs"])}
        for grid in manifest["refinement"]["grids"].values()
    ], root / "refinement_plan.csv")


def validate_reuse_environment(root: Path, manifest: dict[str, Any]) -> None:
    if not manifest.get("reused_results"):
        return
    torch, _, torchvision, datasets, _ = import_training_stack()
    config = manifest["config"]
    if not config["fake_data"]:
        dataset = datasets.CIFAR10(root=config["data_root"], train=True, download=False)
        data_hash = digest(cifar_integrity_metadata(dataset, Path(config["data_root"])))
    else:
        data_hash = None
    for trial in manifest["jobs"]:
        if trial["id"] not in manifest["reused_results"]:
            continue
        row = read_result(root, manifest, trial)
        for name, current in (("torch", torch.__version__), ("torchvision", torchvision.__version__)):
            if row["environment"][name] != current:
                raise ValueError(f"{name} version changed; cannot reuse the old run")
        if data_hash is not None and row["data"]["file_listing_hash"] != data_hash:
            raise ValueError("dataset identity changed; cannot reuse the old run")


def seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def group_norm_factory(nn: Any) -> Any:
    def make_group_norm(channels: int) -> Any:
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)

    return make_group_norm


def build_model(model_name: str, nn: Any, torchvision: Any, num_classes: int) -> Any:
    builder = getattr(torchvision.models, model_name)
    model = builder(
        weights=None,
        num_classes=num_classes,
        norm_layer=group_norm_factory(nn),
    )
    # Standard CIFAR stem: preserve 32x32 resolution before the residual stages.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_transforms(transforms: Any) -> tuple[Any, Any]:
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    probe_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, probe_transform


class PaperMomentum:
    """Factory namespace for the position-velocity momentum optimizer."""

    @staticmethod
    def create(torch: Any, model: Any, algorithm: str, lr: float, momentum: float) -> Any:
        class Optimizer(torch.optim.Optimizer):
            def __init__(self, params: Iterable[Any], lookahead: bool) -> None:
                super().__init__(params, {"lr": lr, "momentum": momentum})
                self.lookahead = lookahead
                self.at_lookahead = False

            @torch.no_grad()
            def move_to_lookahead(self) -> None:
                if not self.lookahead:
                    return
                if self.at_lookahead:
                    raise RuntimeError("parameters are already at the look-ahead point")
                for group in self.param_groups:
                    rho = float(group["momentum"])
                    for parameter in group["params"]:
                        velocity = self.state[parameter].setdefault(
                            "velocity",
                            torch.zeros_like(parameter, memory_format=torch.preserve_format),
                        )
                        parameter.add_(velocity, alpha=rho)
                self.at_lookahead = True

            @torch.no_grad()
            def restore_base(self) -> None:
                if not self.lookahead or not self.at_lookahead:
                    return
                for group in self.param_groups:
                    rho = float(group["momentum"])
                    for parameter in group["params"]:
                        parameter.sub_(self.state[parameter]["velocity"], alpha=rho)
                self.at_lookahead = False

            @torch.no_grad()
            def step(self, closure: Any | None = None) -> Any:
                if self.at_lookahead:
                    raise RuntimeError("restore base parameters before optimizer.step()")
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()
                for group in self.param_groups:
                    learning_rate = float(group["lr"])
                    rho = float(group["momentum"])
                    for parameter in group["params"]:
                        if parameter.grad is None:
                            continue
                        velocity = self.state[parameter].setdefault(
                            "velocity",
                            torch.zeros_like(parameter, memory_format=torch.preserve_format),
                        )
                        velocity.mul_(rho).add_(parameter.grad, alpha=-learning_rate)
                        parameter.add_(velocity)
                return loss

        return Optimizer(model.parameters(), lookahead=algorithm == "nesterov")


def optimizer_for(torch: Any, model: Any, algorithm: str, lr: float, gap: float) -> Any:
    if algorithm == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    momentum = 1.0 - gap
    if not 0 <= momentum < 1:
        raise ValueError("momentum must lie in [0, 1)")
    return PaperMomentum.create(torch, model, algorithm, lr, momentum)


def cifar_integrity_metadata(dataset: Any, root: Path) -> dict[str, Any]:
    folder = root / dataset.base_folder
    declared_files = [
        *dataset.train_list,
        *dataset.test_list,
        (dataset.meta["filename"], dataset.meta["md5"]),
    ]
    entries: list[tuple[str, int, str]] = []
    for filename, expected_md5 in declared_files:
        path = folder / filename
        entries.append((filename, path.stat().st_size, expected_md5))
    return {
        "dataset": "CIFAR-10",
        "root": str(root),
        "base_folder": dataset.base_folder,
        "classes": list(dataset.classes),
        "files": entries,
        "torchvision_integrity_check": True,
    }


def build_data(
    config: dict[str, Any], trial: dict[str, Any], torch: Any, datasets: Any, transforms: Any
) -> tuple[Any, Any, int, dict[str, Any]]:
    train_transform, probe_transform = build_transforms(transforms)
    if config["fake_data"]:
        size = max(128, config["probe_size"])
        kwargs = {
            "size": size,
            "image_size": (3, 32, 32),
            "num_classes": 10,
            "random_offset": 0,
        }
        train_dataset = datasets.FakeData(**kwargs, transform=train_transform)
        probe_dataset = datasets.FakeData(**kwargs, transform=probe_transform)
        classes = 10
        data_hash = digest(kwargs)
    else:
        root = Path(config["data_root"])
        train_dataset = datasets.CIFAR10(
            root=str(root), train=True, transform=train_transform, download=False
        )
        probe_dataset = copy.copy(train_dataset)
        probe_dataset.transform = probe_transform
        classes = len(train_dataset.classes)
        metadata = cifar_integrity_metadata(train_dataset, root)
        data_hash = digest(metadata)
    if len(probe_dataset) < config["probe_size"]:
        raise ValueError("probe-size exceeds training dataset size")
    probe_generator = torch.Generator().manual_seed(config["probe_seed"])
    indices = torch.randperm(len(probe_dataset), generator=probe_generator)[
        : config["probe_size"]
    ]
    probe = torch.utils.data.Subset(probe_dataset, indices.tolist())
    micro_batch = min(config["max_micro_batch"], trial["batch"])
    sampler = torch.utils.data.RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=config["steps"] * trial["batch"],
        generator=torch.Generator().manual_seed(trial["seed"] + 10_000),
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": config["workers"],
        "pin_memory": config["device"].startswith("cuda"),
    }
    if config["workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=trial["batch"],
        generator=torch.Generator().manual_seed(trial["seed"] + 20_000),
        **loader_kwargs,
    )
    probe_loader = torch.utils.data.DataLoader(
        probe,
        batch_size=config["probe_batch"],
        shuffle=False,
        num_workers=0,
        pin_memory=config["device"].startswith("cuda"),
    )
    metadata = {
        "train_size": len(train_dataset),
        "classes": classes,
        "file_listing_hash": data_hash,
        "probe_indices": indices.tolist(),
        "micro_batch": micro_batch,
        "sampling": "with_replacement",
    }
    return train_loader, probe_loader, classes, metadata


def tensor_norm(tensors: Iterable[Any], torch: Any) -> float:
    norms = [torch.linalg.vector_norm(t.detach()) for t in tensors if t is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def evaluate(model: Any, loader: Any, criterion: Any, torch: Any, device: Any) -> tuple[float, float]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum().item())
            count += len(labels)
    model.train()
    return total_loss / count, correct / count


def backward_batch(
    model: Any,
    optimizer: Any,
    criterion: Any,
    images: Any,
    labels: Any,
    device: Any,
    algorithm: str,
    micro_batch: int,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    if algorithm == "nesterov":
        optimizer.move_to_lookahead()
    try:
        for start in range(0, len(labels), micro_batch):
            stop = start + micro_batch
            image_chunk = images[start:stop].to(device, non_blocking=True)
            target = labels[start:stop].to(device, non_blocking=True)
            raw_loss = criterion(model(image_chunk), target)
            weight = len(target) / len(labels)
            (raw_loss * weight).backward()
            loss_sum += float(raw_loss.detach().item()) * weight
    finally:
        if algorithm == "nesterov":
            optimizer.restore_base()
    return loss_sum


def classify(
    initial: float,
    tail: list[float],
    previous: list[float],
    completed: bool,
    reason: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    final_ratio = statistics.fmean(tail) / initial if tail else None
    growth = (
        statistics.fmean(tail) / max(statistics.fmean(previous), 1e-30)
        if tail and previous
        else None
    )
    bounded = bool(
        completed
        and not reason
        and final_ratio is not None
        and growth is not None
        and math.isfinite(final_ratio)
        and math.isfinite(growth)
        and final_ratio <= config["stable_loss_ratio"]
        and growth <= config["growth_ratio"]
    )
    return {
        "survived": bool(completed and not reason),
        "bounded": bounded,
        "learned": bool(bounded and final_ratio <= config["learning_ratio"]),
        "tail_loss_ratio": final_ratio,
        "tail_growth_ratio": growth,
    }


def run_trial(
    root: Path,
    manifest: dict[str, Any],
    trial: dict[str, Any],
    *,
    device_override: str | None = None,
) -> dict[str, Any]:
    cached = read_result(root, manifest, trial)
    if cached is not None:
        print(f"cached {trial['id']}", flush=True)
        return cached
    config = dict(manifest["config"])
    if device_override is not None:
        config["device"] = device_override
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("use one device per process; do not launch this script with torchrun")
    torch, nn, torchvision, datasets, transforms = import_training_stack()
    torch.set_num_threads(config["threads"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    seed_everything(torch, trial["seed"])
    torch.use_deterministic_algorithms(True)
    train_loader, probe_loader, classes, data_info = build_data(
        config, trial, torch, datasets, transforms
    )
    model = build_model(config["model"], nn, torchvision, classes).to(device)
    optimizer = optimizer_for(
        torch, model, trial["algorithm"], trial["lr"], trial["gap"]
    )
    criterion = nn.CrossEntropyLoss()
    initial_loss, initial_accuracy = evaluate(model, probe_loader, criterion, torch, device)
    initial_norm = tensor_norm(model.parameters(), torch)
    if not math.isfinite(initial_loss) or initial_loss <= 0 or not math.isfinite(initial_norm):
        raise RuntimeError("invalid initialization; no stability label can be assigned")

    directory = result_path(root, trial).parent
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        {
            "fingerprint": manifest["fingerprint"],
            "trial": trial,
            "data": data_info,
            "initial_probe_loss": initial_loss,
        },
        directory / "config.json",
    )
    first_row = {
        "step": 0,
        "probe_loss": initial_loss,
        "probe_loss_ratio": 1.0,
        "probe_accuracy": initial_accuracy,
        "parameter_norm": initial_norm,
    }
    recent: deque[float] = deque(maxlen=2 * config["tail_evals"])
    reason, updates = "", 0
    started = time.monotonic()
    print(f"start {trial} K={config['steps']} device={device}", flush=True)
    with (directory / "history.jsonl").open("w", encoding="utf-8") as log:
        log.write(json.dumps(first_row) + "\n")
        for step, (images, labels) in enumerate(train_loader, start=1):
            train_loss = backward_batch(
                model,
                optimizer,
                criterion,
                images,
                labels,
                device,
                trial["algorithm"],
                data_info["micro_batch"],
            )
            grad_norm = tensor_norm((parameter.grad for parameter in model.parameters()), torch)
            parameter_norm = tensor_norm(model.parameters(), torch)
            if not math.isfinite(train_loss) or not math.isfinite(grad_norm):
                reason = "nonfinite_training_loss_or_gradient"
            elif grad_norm > config["max_grad_norm"]:
                reason = "gradient_safety_limit"
            else:
                optimizer.step()
                updates = step
                parameter_norm = tensor_norm(model.parameters(), torch)
                if not math.isfinite(parameter_norm):
                    reason = "nonfinite_parameters"
                elif parameter_norm > initial_norm * config["max_parameter_ratio"]:
                    reason = "parameter_safety_limit"
            row: dict[str, Any] = {
                "step": step,
                "updates_completed": updates,
                "train_loss": train_loss,
                "gradient_norm": grad_norm,
                "parameter_norm": parameter_norm,
                "gradient_location": (
                    "lookahead" if trial["algorithm"] == "nesterov" else "base"
                ),
            }
            if not reason and (step % config["eval_every"] == 0 or step == config["steps"]):
                probe_loss, accuracy = evaluate(model, probe_loader, criterion, torch, device)
                row.update(
                    probe_loss=probe_loss,
                    probe_loss_ratio=probe_loss / initial_loss,
                    probe_accuracy=accuracy,
                )
                recent.append(probe_loss)
                if not math.isfinite(probe_loss):
                    reason = "nonfinite_probe_loss"
                elif probe_loss / initial_loss > config["divergence_loss_ratio"]:
                    reason = "probe_loss_safety_limit"
                print(f"  step={step} probe/L0={probe_loss / initial_loss:.5g}", flush=True)
            row["stop_reason"] = reason
            log.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
            if reason or "probe_loss" in row:
                log.flush()
            if reason:
                break

    tail = list(recent)[-config["tail_evals"] :]
    previous = list(recent)[: -config["tail_evals"]]
    result = {
        "fingerprint": manifest["fingerprint"],
        "trial": trial,
        "smoke_test": config["fake_data"],
        "fake_data": config["fake_data"],
        "updates_completed": updates,
        "samples_processed": updates * trial["batch"],
        "damping_times": config["steps"] * trial["gap"],
        "stop_reason": reason,
        "initial_probe_loss": initial_loss,
        "initial_parameter_norm": initial_norm,
        "tail_probe_losses": tail,
        "previous_probe_losses": previous,
        "seconds": time.monotonic() - started,
        "data": data_info,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()
            ),
            "dtype": "float32",
            "deterministic_algorithms": True,
        },
        **classify(
            initial_loss,
            tail,
            previous,
            updates == config["steps"],
            reason,
            config,
        ),
    }
    atomic_json_dump(result, result_path(root, trial))
    print(
        f"done bounded={result['bounded']} learned={result['learned']} "
        f"reason={reason or 'none'}",
        flush=True,
    )
    return result


def boundary_from_grid(lrs: list[float], accepted: list[bool | None]) -> dict[str, Any]:
    observed = [lr for lr, passed in zip(lrs, accepted) if passed is True]
    output: dict[str, Any] = {
        "largest_observed_stable_lr": max(observed) if observed else None,
        "lower": None,
        "upper": None,
    }
    if any(value is None for value in accepted):
        return {**output, "status": "incomplete"}
    if not any(accepted):
        return {**output, "upper": lrs[0], "status": "left_censored"}
    seen_failure = False
    for passed in accepted:
        if not passed:
            seen_failure = True
        elif seen_failure:
            return {**output, "status": "nonmonotone"}
    if all(accepted):
        return {**output, "lower": lrs[-1], "status": "right_censored"}
    index = accepted.index(False)
    return {
        **output,
        "lower": lrs[index - 1],
        "upper": lrs[index],
        "status": "bracketed",
    }


def summarize(root: Path, *, make_plot: bool = True) -> None:
    manifest = read_manifest(root)
    results: dict[str, dict[str, Any]] = {}
    for trial in manifest["jobs"]:
        row = read_result(root, manifest, trial)
        if row is not None:
            results[trial["id"]] = row
    if len({row["data"]["file_listing_hash"] for row in results.values()}) > 1:
        raise ValueError("dataset identity changed across trials")

    cells: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    lookup = {
        (
            row["trial"]["algorithm"],
            row["trial"]["batch"],
            row["trial"]["gap"],
            row["trial"]["lr"],
            row["trial"]["seed"],
        ): row
        for row in results.values()
    }
    for algorithm in manifest["algorithms"]:
        for point in manifest["points"]:
            base = {
                "axis": point["axis"],
                "x": point["x"],
                "algorithm": algorithm,
                "batch": point["batch"],
                "gap": point["gap"],
            }
            lrs = point_lrs(manifest, algorithm, point)
            accepted: list[bool | None] = []
            for lr in lrs:
                gap = 1.0 if algorithm == "sgd" else point["gap"]
                rows = [
                    lookup.get((algorithm, point["batch"], gap, lr, seed))
                    for seed in manifest["seeds"]
                ]
                observed = [row for row in rows if row is not None]
                rates = {
                    metric
                    + "_fraction": (
                        statistics.fmean(row[metric] for row in observed)
                        if observed
                        else None
                    )
                    for metric in ("survived", "bounded", "learned")
                }
                complete = len(observed) == len(manifest["seeds"])
                accepted.append(
                    rates["bounded_fraction"]
                    >= manifest["config"]["required_fraction"]
                    if complete
                    else None
                )
                cells.append(
                    {
                        **base,
                        "lr": lr,
                        "runs": len(observed),
                        "complete": complete,
                        **rates,
                    }
                )
            boundaries.append(
                {**base, **boundary_from_grid(lrs, accepted)}
            )
            for seed in manifest["seeds"]:
                seed_labels: list[bool | None] = []
                for lr in lrs:
                    gap = 1.0 if algorithm == "sgd" else point["gap"]
                    row = lookup.get((algorithm, point["batch"], gap, lr, seed))
                    seed_labels.append(row["bounded"] if row else None)
                seed_boundary = boundary_from_grid(lrs, seed_labels)
                center = (math.sqrt(seed_boundary["lower"] * seed_boundary["upper"])
                          if seed_boundary["status"] == "bracketed" else None)
                per_seed.append({**base, "seed": seed, **seed_boundary,
                                 "geometric_center": center,
                                 "scan_min": lrs[0], "scan_max": lrs[-1]})

    write_csv(cells, root / "stability_grid.csv")
    write_csv(boundaries, root / "boundaries.csv")
    write_csv(per_seed, root / "per_seed_boundaries.csv")
    averages = seed_boundary_statistics(manifest, per_seed)
    followup = [row for row in per_seed if row["status"] != "bracketed"]
    write_csv(averages, root / "boundary_seed_stats.csv")
    # Keep a header even when there are no follow-up cases.
    with (root / "needs_followup.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(followup)
    write_csv([
        {**row["trial"], "bounded": row["bounded"], "learned": row["learned"],
         "stop_reason": row["stop_reason"], "updates_completed": row["updates_completed"],
         "tail_loss_ratio": row["tail_loss_ratio"], "tail_growth_ratio": row["tail_growth_ratio"],
         "reused": row["trial"]["id"] in manifest.get("reused_results", {})}
        for row in results.values()
    ], root / "trial_diagnostics.csv")
    atomic_json_dump(
        {
            "fingerprint": manifest["fingerprint"],
            "completed": len(results),
            "planned": len(manifest["jobs"]),
            "smoke_test": manifest["config"]["fake_data"],
            "boundaries": boundaries,
            "boundary_seed_stats": averages,
            "reused": len(manifest.get("reused_results", {})),
            "followup_rows": len(followup),
        },
        root / "summary.json",
    )
    if make_plot:
        plot_results(root, manifest, cells, boundaries)
        plot_seed_boundaries(root, manifest, per_seed, averages)
    print(f"summarized {len(results)}/{len(manifest['jobs'])} trials in {root}")


def seed_boundary_statistics(manifest: dict[str, Any], per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Never average away censored, missing, or nonmonotone seeds."""
    output = []
    for algorithm in manifest["algorithms"]:
        for point in manifest["points"]:
            rows = [row for row in per_seed if row["algorithm"] == algorithm
                    and row["axis"] == point["axis"] and row["x"] == point["x"]]
            counts = {status: sum(row["status"] == status for row in rows) for status in (
                "bracketed", "incomplete", "nonmonotone", "left_censored", "right_censored")}
            complete = counts["bracketed"] == len(manifest["seeds"])
            stats = {"axis": point["axis"], "x": point["x"], "algorithm": algorithm,
                     "batch": point["batch"], "gap": point["gap"],
                     "seeds": len(manifest["seeds"]), **counts,
                     "status": "ok" if complete else ("incomplete" if counts["incomplete"] else "needs_followup"),
                     "geometric_mean_lr": None, "arithmetic_mean_lr": None,
                     "log_std": None, "seed_center_min": None, "seed_center_max": None,
                     "bracket_envelope_lower": None, "bracket_envelope_upper": None}
            if complete:
                centers = [row["geometric_center"] for row in rows]
                logs = [math.log(value) for value in centers]
                stats.update(geometric_mean_lr=math.exp(statistics.fmean(logs)),
                             arithmetic_mean_lr=statistics.fmean(centers),
                             log_std=statistics.stdev(logs) if len(logs) > 1 else None,
                             seed_center_min=min(centers), seed_center_max=max(centers),
                             bracket_envelope_lower=min(row["lower"] for row in rows),
                             bracket_envelope_upper=max(row["upper"] for row in rows))
            output.append(stats)
    return output


def plot_seed_boundaries(
    root: Path, manifest: dict[str, Any], per_seed: list[dict[str, Any]], averages: list[dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {"sgd": "#767676", "polyak": "#3775BA", "nesterov": "#B64342"}
    names = {"sgd": "SGD", "polyak": "Polyak", "nesterov": "Nesterov"}
    markers = {"sgd": "^", "polyak": "o", "nesterov": "s"}
    for axis in dict.fromkeys(point["axis"] for point in manifest["points"]):
        xs = [point["x"] for point in manifest["points"] if point["axis"] == axis]
        fig, ax = plt.subplots(figsize=(7.5, 6))
        has_boundary = False
        for index, algorithm in enumerate(manifest["algorithms"]):
            offset = math.exp(0.06 * (index - (len(manifest["algorithms"]) - 1) / 2))
            rows = [row for row in averages if row["axis"] == axis and row["algorithm"] == algorithm]
            # NaNs intentionally break lines at unresolved configurations.
            ys = [row["geometric_mean_lr"] if row["status"] == "ok" else np.nan for row in rows]
            ax.plot([row["x"] * offset for row in rows], ys, marker=markers[algorithm],
                    color=colors[algorithm], linewidth=1.5, label=names[algorithm], zorder=4)
            for row in rows:
                if row["status"] != "ok":
                    continue
                has_boundary = True
                center = row["geometric_mean_lr"]
                ax.errorbar(row["x"] * offset, center,
                            yerr=[[center - row["bracket_envelope_lower"]],
                                  [row["bracket_envelope_upper"] - center]],
                            color=colors[algorithm], capsize=3, fmt="none", zorder=3)
            for row in per_seed:
                if row["axis"] != axis or row["algorithm"] != algorithm:
                    continue
                seed_index = manifest["seeds"].index(row["seed"])
                x = row["x"] * offset * math.exp(0.012 * (seed_index - (len(manifest["seeds"]) - 1) / 2))
                status = row["status"]
                if status == "bracketed":
                    has_boundary = True
                    ax.scatter(x, row["geometric_center"], color=colors[algorithm], s=12, alpha=0.55)
                elif status in ("left_censored", "right_censored"):
                    has_boundary = True
                    y = row["upper"] if status == "left_censored" else row["lower"]
                    ax.scatter(x, y, facecolors="none", edgecolors=colors[algorithm],
                               marker="v" if status == "left_censored" else "^", s=45)
                elif status == "nonmonotone":
                    has_boundary = True
                    ax.scatter(x, row["largest_observed_stable_lr"], color=colors[algorithm], marker="x", s=40)
        # A partially finished grid may have only NaN lines and no finite y data.
        # Fix positive limits from the actual planned LRs before enabling log scale.
        ax.set_ylim(manifest["lrs"][0] / 1.2, manifest["lrs"][-1] * 1.2)
        ax.set(xscale="log", yscale="log",
               xlabel="Batch size B" if axis == "batch" else r"Momentum gap $1-\rho$",
               ylabel="Empirical learning-rate boundary",
               title=("SMOKE TEST — no scientific inference" if manifest["config"]["fake_data"]
                      else f"CIFAR-10: K={manifest['config']['steps']}, {len(manifest['seeds'])} seeds"))
        ax.set_xticks(xs, [f"{value:.4g}" for value in xs], rotation=30 if axis == "gap" else 0)
        ax.set_xlim(min(xs) / 1.2, max(xs) * 1.2)
        ax.minorticks_off()
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)
        if not has_boundary:
            ax.text(0.5, 0.5, "Boundary estimates pending\nComplete the remaining trials for each seed.",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#555555")
        fig.text(0.5, 0.02,
                 "Large markers: geometric mean of per-seed bracket centers; dots: individual seeds.\n"
                 "Bars: min(lower) to max(upper), combining seed spread and grid width (not a CI).\n"
                 "Open triangles: censored; x: nonmonotone. Means require every seed bracketed.\n"
                 "Horizontal offsets separate methods/seeds; no smoothing or forced monotonicity.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        for extension in ("png", "pdf"):
            fig.savefig(root / f"boundary_mean_vs_{axis}.{extension}", dpi=300)
        plt.close(fig)


def plot_results(
    root: Path,
    manifest: dict[str, Any],
    cells: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"sgd": "#767676", "polyak": "#3775BA", "nesterov": "#B64342"}
    names = {"sgd": "SGD", "polyak": "Polyak", "nesterov": "Nesterov"}
    markers = {"sgd": "^", "polyak": "o", "nesterov": "s"}
    settings = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    label = (
        "SMOKE TEST — no scientific inference"
        if manifest["config"]["fake_data"]
        else (
            f"CIFAR-10 empirical stability, K={manifest['config']['steps']}, "
            f"required seed fraction={manifest['config']['required_fraction']:g}"
        )
    )
    with plt.rc_context(settings):
        for axis in dict.fromkeys(point["axis"] for point in manifest["points"]):
            points = [point for point in manifest["points"] if point["axis"] == axis]
            xs = [point["x"] for point in points]
            xlabel = "Batch size B" if axis == "batch" else r"Momentum gap $1-\rho$"
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            for method_index, algorithm in enumerate(manifest["algorithms"]):
                offset = np.exp(
                    0.045
                    * (method_index - (len(manifest["algorithms"]) - 1) / 2)
                )
                rows = [
                    row
                    for row in boundaries
                    if row["axis"] == axis and row["algorithm"] == algorithm
                ]
                bracketed = [row for row in rows if row["status"] == "bracketed"]
                if bracketed:
                    x_values = np.array([row["x"] for row in bracketed]) * offset
                    low = np.array([row["lower"] for row in bracketed])
                    high = np.array([row["upper"] for row in bracketed])
                    centers = np.sqrt(low * high)
                    ax.errorbar(
                        x_values,
                        centers,
                        yerr=np.stack((centers - low, high - centers)),
                        linestyle="none",
                        marker=markers[algorithm],
                        color=colors[algorithm],
                        capsize=4,
                    )
                for row in rows:
                    status = row["status"]
                    if status in ("left_censored", "right_censored"):
                        value = row["upper"] if status == "left_censored" else row["lower"]
                        ax.scatter(
                            row["x"] * offset,
                            value,
                            marker="v" if status == "left_censored" else "^",
                            facecolors="none",
                            edgecolors=colors[algorithm],
                            s=70,
                        )
                    elif status == "nonmonotone":
                        ax.scatter(
                            row["x"] * offset,
                            row["largest_observed_stable_lr"],
                            marker="x",
                            color=colors[algorithm],
                            s=65,
                        )
                ax.plot(
                    [],
                    [],
                    marker=markers[algorithm],
                    linestyle="none",
                    color=colors[algorithm],
                    label=names[algorithm],
                )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel=xlabel,
                ylabel="Empirical learning-rate boundary",
                title=label,
            )
            ax.set_xlim(min(xs) / 1.2, max(xs) * 1.2)
            ax.set_ylim(manifest["lrs"][0] / 1.2, manifest["lrs"][-1] * 1.2)
            ax.set_xticks(xs, [f"{value:.4g}" for value in xs])
            ax.minorticks_off()
            ax.legend()
            ax.grid(alpha=0.15)
            fig.text(
                0.5,
                0.02,
                "Bars: LR grid brackets; open triangles: scan limits; ×: nonmonotone\n"
                "Markers: geometric bracket centers; methods offset horizontally.\n"
                "Incomplete grids omitted; see stability-fraction maps and CSVs.",
                ha="center",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.12, 1, 1))
            for extension in ("png", "pdf"):
                fig.savefig(root / f"boundary_vs_{axis}.{extension}", dpi=300)
            plt.close(fig)

            fig, axes = plt.subplots(
                1,
                len(manifest["algorithms"]),
                figsize=(5 * len(manifest["algorithms"]), 5),
                squeeze=False,
                constrained_layout=True,
            )
            image = None
            for ax, algorithm in zip(axes[0], manifest["algorithms"]):
                colormap = plt.get_cmap("cividis").copy()
                colormap.set_bad("#dddddd")
                ax.set_facecolor("#dddddd")
                observed = {(cell["x"], cell["lr"]): cell["bounded_fraction"]
                            for cell in cells if cell["axis"] == axis
                            and cell["algorithm"] == algorithm and cell["complete"]}
                # Each column has its own LR grid in refinement mode. Real log-LR
                # coordinates avoid stretching the union's uneven grid indices.
                for column, point in enumerate(points):
                    local_lrs = point_lrs(manifest, algorithm, point)
                    log_lrs = np.log(np.array(local_lrs))
                    middle = (log_lrs[:-1] + log_lrs[1:]) / 2
                    edges = np.exp(np.r_[2 * log_lrs[0] - middle[0], middle,
                                          2 * log_lrs[-1] - middle[-1]])
                    values = np.array([observed.get((point["x"], lr), np.nan)
                                       for lr in local_lrs])[:, None]
                    image = ax.pcolormesh([column - 0.5, column + 0.5], edges,
                                          np.ma.masked_invalid(values), cmap=colormap,
                                          vmin=0, vmax=1, shading="flat")
                ax.set_xticks(
                    range(len(xs)),
                    [f"{value:g}" for value in xs],
                    rotation=45,
                    ha="right",
                )
                ax.set(
                    title=names[algorithm],
                    xlabel=xlabel,
                    ylabel="Learning rate (log scale)",
                    yscale="log",
                )
                ax.set_xlim(-0.5, len(xs) - 0.5)
                ax.set_ylim(manifest["lrs"][0] / 1.2, manifest["lrs"][-1] * 1.2)
            assert image is not None
            fig.colorbar(
                image,
                ax=axes[0].tolist(),
                label="Bounded-run fraction (gray: unscanned/incomplete)",
                shrink=0.8,
            )
            fig.suptitle(label)
            for extension in ("png", "pdf"):
                fig.savefig(root / f"stable_fraction_vs_{axis}.{extension}", dpi=300)
            plt.close(fig)


def validate_merge_destination(sources: list[Path], destination: Path) -> None:
    """Keep analysis outputs away from all original experiment files."""
    for source in sources:
        if not (source / "manifest.json").is_file():
            raise ValueError(f"missing source manifest: {source / 'manifest.json'}")
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError("merge output must be separate from every source directory")
    if (destination / "manifest.json").exists() or (destination / "trials").exists():
        raise ValueError("merge output cannot be an experiment directory; choose a new --output")
    marker = destination / "merge_summary.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")).get("kind") != "merged_heatmaps":
            raise ValueError("unrecognized merge output directory; choose a new --output")
    elif destination.exists():
        unrelated = {path.name for path in destination.iterdir()} - {"logs", ".runner.lock"}
        if unrelated:
            raise ValueError("merge output is not empty; choose a new --output")


def collect_merged_results(sources: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pool observed trials, not screenshots or already averaged fractions."""
    scan_fields = {
        "preset", "axis", "algorithms", "batches", "gaps", "fixed_batch", "fixed_gap",
        "lrs", "lr_min", "lr_max", "lr_count", "seeds", "devices", "device", "required_fraction",
    }
    protocol = None
    entries: dict[tuple[Any, ...], dict[str, Any]] = {}
    views: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    slices: dict[tuple[str, str], set[Any]] = {}
    source_reports = []
    dataset_hashes: set[str] = set()
    environments: dict[str, set[Any]] = {}
    duplicates = 0
    current_training_hash = training_code_hash()
    for source in sources:
        manifest = read_manifest(source)
        code_hash = manifest.get("training_code_hash")
        old_hashes = set(manifest["source_hashes"].values())
        if code_hash != current_training_hash and not (
            code_hash is None and len(old_hashes) == 1
            and old_hashes <= LEGACY_TRAINING_SOURCE_HASHES
            and current_training_hash == LEGACY_TRAINING_CODE_HASH
        ):
            raise ValueError(f"source training code is unknown/different: {source}")
        config = manifest["config"]
        comparable = {key: value for key, value in config.items() if key not in scan_fields}
        if protocol is None:
            protocol = comparable
        elif protocol != comparable:
            differences = sorted(key for key in set(protocol) | set(comparable)
                                 if protocol.get(key) != comparable.get(key))
            raise ValueError(f"incompatible training protocol in {source}: {', '.join(differences)}")
        for algorithm in manifest["algorithms"]:
            for point in manifest["points"]:
                gap = 1.0 if algorithm == "sgd" else point["gap"]
                view = (point["axis"], algorithm, point["x"])
                configuration = (algorithm, point["batch"], gap)
                if view in views and views[view] != configuration:
                    raise ValueError(f"incompatible axis slices at {view}")
                views[view] = configuration
                fixed = gap if point["axis"] == "batch" else point["batch"]
                slices.setdefault((point["axis"], algorithm), set()).add(fixed)
        completed = 0
        for trial in manifest["jobs"]:
            gap = 1.0 if trial["algorithm"] == "sgd" else trial["gap"]
            key = (trial["algorithm"], trial["batch"], gap, trial["lr"], trial["seed"])
            if not math.isfinite(trial["lr"]) or trial["lr"] <= 0:
                raise ValueError(f"invalid learning rate in {source}")
            entry = entries.setdefault(key, {"trial": trial, "row": None, "sources": [], "planned_sources": []})
            entry["planned_sources"].append(str(source))
            row = read_result(source, manifest, trial)
            if row is None:
                continue
            if not isinstance(row.get("bounded"), bool):
                raise ValueError(f"invalid bounded result: {result_path(source, trial)}")
            data_hash = row.get("data", {}).get("file_listing_hash")
            if data_hash is not None:
                dataset_hashes.add(data_hash)
            environment = row.get("environment", {})
            for name in ("torch", "torchvision", "dtype", "deterministic_algorithms", "device"):
                value = environment.get(name)
                if value is not None:
                    if name == "device":
                        value = str(value).split(":", 1)[0]
                    environments.setdefault(name, set()).add(value)
            # Imported seed-0 summaries retain the original fingerprint. If a
            # separate run with the same seed disagrees, do not count it twice
            # or silently choose whichever result looks more stable.
            scientific = {name: value for name, value in row.items()
                          if name not in {"fingerprint", "seconds", "environment", "trial"}}
            if entry["row"] is not None:
                if entry["scientific"] != scientific:
                    raise ValueError(f"conflicting duplicate trial {key} in {source}")
                duplicates += 1
            else:
                entry["row"], entry["scientific"] = row, scientific
            entry["sources"].append(str(source))
            completed += 1
        source_reports.append({"root": str(source), "fingerprint": manifest["fingerprint"],
                               "planned": len(manifest["jobs"]), "completed": completed})
    if any(len(values) > 1 for values in slices.values()):
        raise ValueError("cannot combine different fixed batch/gap slices in one heatmap")
    if len(dataset_hashes) > 1:
        raise ValueError("dataset identities differ between saved trials")
    for name, values in environments.items():
        if len(values) > 1:
            raise ValueError(f"saved runtime settings differ: {name}={sorted(values)}")

    configurations: dict[tuple[Any, ...], dict[float, list[dict[str, Any]]]] = {}
    trial_rows = []
    for key, entry in sorted(entries.items()):
        configurations.setdefault(key[:3], {}).setdefault(key[3], []).append(entry)
        row = entry["row"]
        trial_rows.append({
            **entry["trial"], "completed": row is not None,
            "bounded": row["bounded"] if row is not None else None,
            "learned": row.get("learned") if row is not None else None,
            "stop_reason": row.get("stop_reason") if row is not None else None,
            "summary_digest": digest(row) if row is not None else None,
            "sources": ";".join(entry["sources"]),
            "planned_sources": ";".join(entry["planned_sources"]),
        })
    cells = []
    for (axis, algorithm, x), configuration in sorted(views.items()):
        for lr, candidates in sorted(configurations[configuration].items()):
            observed = [entry for entry in candidates if entry["row"] is not None]
            passed = sum(entry["row"]["bounded"] for entry in observed)
            n, planned = len(observed), len(candidates)
            cells.append({
                "axis": axis, "x": x, "algorithm": algorithm, "batch": configuration[1],
                "gap": configuration[2], "lr": lr, "observed_seeds": n, "planned_seeds": planned,
                "bounded_count": passed, "bounded_fraction": passed / n if n else None,
                "all_planned_observed": n == planned,
                "seed_ids": ",".join(str(entry["trial"]["seed"]) for entry in observed),
                "missing_seed_ids": ",".join(str(entry["trial"]["seed"]) for entry in candidates
                                             if entry["row"] is None),
            })
    completed = sum(entry["row"] is not None for entry in entries.values())
    report = {
        "kind": "merged_heatmaps", "sources": source_reports, "protocol": protocol,
        "training_code_hash": current_training_hash, "analysis_source_hashes": source_hashes(),
        "unique_planned_trials": len(entries), "unique_completed_trials": completed,
        "unique_missing_trials": len(entries) - completed, "duplicate_results_removed": duplicates,
        "dataset_hashes": sorted(dataset_hashes),
        "saved_environment": {key: sorted(values) for key, values in environments.items()},
        "fraction_definition": "bounded_count / unique observed seeds at the exact configuration and LR",
        "missing_policy": "missing seeds excluded from denominator; no observations displayed gray",
        "coverage_note": "seed counts vary; boundary points were selected using the original seed",
    }
    return report, cells, trial_rows


def plot_merged_heatmaps(root: Path, report: dict[str, Any], cells: list[dict[str, Any]]) -> None:
    """Plot the full measured LR range with local refinement and explicit coverage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, Normalize
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
    import numpy as np

    names = {"sgd": "SGD", "polyak": "Polyak", "nesterov": "Nesterov"}
    axes_present = [axis for axis in ("batch", "gap") if any(row["axis"] == axis for row in cells)]
    algorithms = [algorithm for algorithm in ALGORITHMS if any(row["algorithm"] == algorithm for row in cells)]
    if not cells or not algorithms:
        raise ValueError("no planned cells available for plotting")
    columns: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in cells:
        columns.setdefault((row["axis"], row["algorithm"], row["x"]), []).append(row)
    edges = {}
    for key, rows in columns.items():
        rows.sort(key=lambda row: row["lr"])
        lrs = np.array([row["lr"] for row in rows], dtype=float)
        logs = np.log(lrs)
        if len(logs) == 1:
            borders = np.array([logs[0] - math.log(2) / 2, logs[0] + math.log(2) / 2])
        else:
            borders = np.r_[logs[0] - (logs[1] - logs[0]) / 2,
                            (logs[:-1] + logs[1:]) / 2,
                            logs[-1] + (logs[-1] - logs[-2]) / 2]
        edges[key] = np.exp(borders)
    lower = min(values[0] for values in edges.values())
    upper = max(values[-1] for values in edges.values())
    max_seeds = max(1, max(row["observed_seeds"] for row in cells))
    settings = {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                "axes.linewidth": 1.1, "font.family": "DejaVu Sans"}

    def render(selected_axes: list[str], filename: str, coverage: bool = False) -> None:
        cmap = plt.get_cmap("viridis", max_seeds) if coverage else plt.get_cmap("cividis")
        norm = BoundaryNorm(np.arange(0.5, max_seeds + 1.5), cmap.N) if coverage else Normalize(0, 1)
        fig, panels = plt.subplots(len(selected_axes), len(algorithms), squeeze=False,
                                   figsize=(5.0 * len(algorithms) + 0.8, 4.1 * len(selected_axes) + 1.2))
        for i, axis_name in enumerate(selected_axes):
            xs = sorted({row["x"] for row in cells if row["axis"] == axis_name})
            for j, algorithm in enumerate(algorithms):
                ax = panels[i, j]
                ax.set_facecolor("#dddddd")
                for x_index, x in enumerate(xs):
                    key = (axis_name, algorithm, x)
                    rows = columns.get(key, [])
                    if not rows:
                        continue
                    values = np.array([
                        (row["observed_seeds"] if coverage else row["bounded_fraction"])
                        if row["observed_seeds"] else np.nan for row in rows
                    ], dtype=float)
                    ax.pcolormesh([x_index - 0.5, x_index + 0.5], edges[key],
                                  np.ma.masked_invalid(values[:, None]), cmap=cmap, norm=norm,
                                  shading="flat", edgecolors="none", rasterized=True)
                    if not coverage:
                        multiple = [row["lr"] for row in rows if row["observed_seeds"] >= 2]
                        ax.scatter([x_index] * len(multiple), multiple, s=8, facecolors="white",
                                   edgecolors="#222222", linewidths=0.4, zorder=3)
                ax.set_ylim(lower, upper)
                ax.set_yscale("log", base=2)
                ax.yaxis.set_major_locator(LogLocator(base=2, numticks=8))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.3g}"))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_xlim(-0.5, len(xs) - 0.5)
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels([f"{x:g}" for x in xs], rotation=45, ha="right")
                if j == 0:
                    ax.set_ylabel("Learning rate (log scale)")
                fixed_values = sorted({row["gap"] if axis_name == "batch" else row["batch"]
                                       for row in cells if row["axis"] == axis_name and row["algorithm"] == algorithm})
                fixed = fixed_values[0] if fixed_values else None
                if axis_name == "batch":
                    label = "Batch size B"
                    if algorithm != "sgd" and fixed is not None:
                        label += f"; fixed gap = {fixed:g}"
                else:
                    label = "Momentum gap 1 − ρ"
                    if fixed is not None:
                        label += f"; fixed B = {fixed:g}"
                ax.set_xlabel(label)
                ax.set_title(names[algorithm] + (" (no momentum)" if axis_name == "gap" and algorithm == "sgd" else ""))
        fake = " [SMOKE / NOT SCIENTIFIC DATA]" if report["protocol"].get("fake_data") else ""
        subject = "Observed seed counts" if coverage else "CIFAR-10 empirical stability"
        fig.suptitle(f"{subject}, K={report['protocol']['steps']}{fake}", y=0.975, fontsize=16)
        footer = ("Counts are per (algorithm, B, gap, LR); duplicated seeds count once. Gray: no observed results."
                  if coverage else
                  "Color = bounded / observed seeds. Dots: n ≥ 2; unmarked colored cells: n = 1. Gray: no observed results.")
        footer += "\nOnly saved observations are combined; see merged_stability_grid.csv for exact LR values and counts."
        fig.text(0.055, 0.025, footer, fontsize=10, va="bottom")
        fig.subplots_adjust(left=0.06, right=0.88, top=0.85 if len(selected_axes) == 1 else 0.90,
                            bottom=0.32 if len(selected_axes) == 1 else 0.21,
                            wspace=0.28, hspace=0.48)
        color_ax = fig.add_axes([0.915, 0.30 if len(selected_axes) == 1 else 0.23,
                                0.012, 0.48 if len(selected_axes) == 1 else 0.58])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax)
        bar.set_label("Observed seeds n" if coverage else "Bounded-run fraction")
        bar.set_ticks(range(1, max_seeds + 1) if coverage else np.linspace(0, 1, 6))
        for suffix in ("png", "pdf"):
            fig.savefig(root / f"{filename}.{suffix}", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    with plt.rc_context(settings):
        render(axes_present, "heatmap_combined")
        for axis in axes_present:
            render([axis], f"heatmap_vs_{axis}")
        render(axes_present, "seed_counts_combined", coverage=True)


def merge_results(sources: list[Path], destination: Path, *, make_plot: bool = True) -> None:
    validate_merge_destination(sources, destination)
    report, cells, trials = collect_merged_results(sources)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(report, destination / "merge_summary.json")
    write_csv(cells, destination / "merged_stability_grid.csv")
    write_csv(trials, destination / "merged_trials.csv")
    (destination / "README.txt").write_text(
        "Full scan + boundary refinement: read-only analysis, no training.\n"
        "heatmap_combined: batch and momentum-gap slices, all saved LR samples.\n"
        "heatmap_vs_batch / heatmap_vs_gap: separate panels.\n"
        "seed_counts_combined: exact number of observed seeds behind the colors.\n"
        "Colors = bounded trials / unique observed seeds at that configuration and LR.\n"
        "Dots mark n >= 2; unmarked colored cells have n = 1. Gray means no observation.\n"
        "A seed reused from the old directory counts once. Missing seeds are not failures.\n"
        "Grid cells use local geometric LR midpoints; no smoothing or monotonic filling.\n"
        "The counts can vary across the figure; this is not a full multi-seed scan.\n"
        "Boundary points were chosen adaptively from the first seed; fractions are descriptive.\n"
        "Bounded means the original finite-horizon criterion, not necessarily successful learning.\n"
        "merged_stability_grid.csv includes LR, bounded_count, observed/planned_seeds and missing seeds.\n"
        "merged_trials.csv and merge_summary.json record deduplication and source provenance.\n",
        encoding="utf-8",
    )
    for source in report["sources"]:
        print(f"Source: {source['root']}; {source['completed']}/{source['planned']} saved trials")
    print(f"Merged: {report['unique_completed_trials']}/{report['unique_planned_trials']} unique saved trials; "
          f"{report['duplicate_results_removed']} duplicate results counted once; "
          f"{report['unique_missing_trials']} missing")
    print("Colors use observed seeds only; single-seed cells are retained and coverage is reported.")
    if make_plot:
        plot_merged_heatmaps(destination, report, cells)
        print(f"Combined heatmap: {destination / 'heatmap_combined.png'}")
    print(f"Merged results: {destination}; no training performed, source directories unchanged.")


def validate_environment(config: dict[str, Any]) -> None:
    torch, _, _, datasets, _ = import_training_stack()
    if not config["fake_data"]:
        root = Path(config["data_root"])
        dataset = datasets.CIFAR10(root=str(root), train=True, download=False)
        if len(dataset) != 50_000 or len(dataset.classes) != 10:
            raise ValueError("expected the standard 50,000-image CIFAR-10 training set")
    for device_name in config["devices"]:
        device = torch.device(device_name)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"requested {device} but only {torch.cuda.device_count()} CUDA devices are visible"
                )
        elif device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")


class Tee:
    def __init__(self, terminal: Any, log: Any) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.terminal, name)


@contextmanager
def output_lock(root: Path) -> Iterable[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runner.lock").open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another runner is using {root}; wait or choose another output"
            ) from exc
        try:
            lock.seek(0)
            lock.truncate()
            lock.write(f"pid={os.getpid()}\n")
            lock.flush()
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def print_boundaries(root: Path) -> None:
    print("\nStability boundaries (empty bounds mean incomplete or censored):")
    with (root / "boundaries.csv").open(newline="", encoding="utf-8") as stream:
        fields = ("algorithm", "axis", "x", "status", "lower", "upper")
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(fields)
        for row in csv.DictReader(stream):
            writer.writerow([row.get(field, "") for field in fields])
    print(f"\nResults: {root}")
    print("Tables: stability_grid.csv, boundaries.csv, per_seed_boundaries.csv")
    print("Seed statistics: boundary_seed_stats.csv; diagnostics: needs_followup.csv, trial_diagnostics.csv")


def trial_worker(
    root_text: str,
    manifest: dict[str, Any],
    device: str,
    job_queue: Any,
) -> None:
    """Run queued trials sequentially on one fixed device."""
    root = Path(root_text)
    log_directory = root / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    device_tag = device.replace(":", "_")
    log_path = log_directory / f"worker_{device_tag}_{os.getpid()}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                print(f"Worker {os.getpid()} started on {device}; log: {log_path}", flush=True)
                while True:
                    item = job_queue.get()
                    if item is None:
                        break
                    index, new_index, total_jobs, trial = item
                    print(
                        f"\n[{index}/{total_jobs}; new trial {new_index}; {device}] "
                        f"{trial['algorithm']} B={trial['batch']} gap={trial['gap']:g} "
                        f"lr={trial['lr']:g} seed={trial['seed']}",
                        flush=True,
                    )
                    run_trial(
                        root,
                        manifest,
                        trial,
                        device_override=device,
                    )
                print(f"Worker {os.getpid()} finished on {device}", flush=True)
            except BaseException:
                traceback.print_exc()
                raise


def run_parallel_trials(
    root: Path,
    manifest: dict[str, Any],
    selected: list[tuple[int, int, dict[str, Any]]],
    devices: list[str],
) -> None:
    """Use one persistent process per device and a shared dynamic job queue."""
    context = multiprocessing.get_context("spawn")
    job_queue = context.Queue()
    workers = [
        context.Process(
            target=trial_worker,
            args=(str(root), manifest, device, job_queue),
            name=f"trial-worker-{device.replace(':', '-')}",
        )
        for device in devices
    ]
    for worker in workers:
        worker.start()
    for index, new_index, trial in selected:
        job_queue.put((index, new_index, len(manifest["jobs"]), trial))
    for _ in workers:
        job_queue.put(None)
    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join()
        raise
    finally:
        job_queue.close()
        job_queue.join_thread()
    failed = [f"{worker.name} (exit {worker.exitcode})" for worker in workers if worker.exitcode]
    if failed:
        raise RuntimeError("worker failure: " + ", ".join(failed))


def execute(args: argparse.Namespace) -> int:
    root = args.output
    if args.merge_from is not None:
        merge_results(args.merge_from, root, make_plot=not args.no_plot)
        return 0
    if args.summarize_only:
        summarize(root, make_plot=not args.no_plot)
        print_boundaries(root)
        return 0
    source = None
    if args.refine_from is not None:
        manifest, source = make_refinement_manifest(args)
        for name in ("steps", "workers", "max_micro_batch", "probe_batch", "preset"):
            setattr(args, name, manifest["config"][name])
        args.device_list = manifest["config"]["devices"]
    else:
        manifest = make_manifest(args)
    if (root / "manifest.json").exists():
        previous = read_manifest(root)
        manifest = select_resume_manifest(previous, manifest)
        if manifest["source_hashes"] != source_hashes():
            # Record the running revision separately; preserve the original manifest.
            running_hashes = source_hashes()
            running_digest = digest(running_hashes)[:20]
            atomic_json_dump({
                "manifest_fingerprint": manifest["fingerprint"],
                "manifest_source_hashes": manifest["source_hashes"],
                "running_source_hashes": running_hashes,
                "training_code_hash": training_code_hash(),
                "compatibility": "audited_analysis_only_revision",
            }, root / "logs" / f"source_revision_{running_digest}.json")
    else:
        atomic_json_dump(manifest, root / "manifest.json")
    if source is not None:
        import_refinement_results(root, manifest, source)
        print(f"Refinement: {len(manifest['refinement']['grids'])} unique configurations; "
              f"seeds={manifest['seeds']}; {len(manifest['reused_results'])} old trials reused")
        print("Training protocol inherited unchanged. Original results are read-only.")
        print(f"Per-configuration LR grids: {root / 'refinement_plan.csv'}")

    jobs = manifest["jobs"]
    cached = {
        trial["id"]
        for trial in jobs
        if read_result(root, manifest, trial) is not None
    }
    print(
        f"Preset: {args.preset}; {len(jobs)} unique trials; "
        f"{len(cached)} completed; {len(jobs) - len(cached)} remaining"
    )
    remaining = [trial for trial in jobs if trial["id"] not in cached]
    print(f"Remaining work: up to {len(remaining) * args.steps:,} updates, "
          f"{sum(trial['batch'] for trial in remaining) * args.steps:,} training samples "
          "(+ probe evaluations; early stopping may reduce this)")
    print(
        f"Full plan: up to {len(jobs) * args.steps:,} updates, "
        f"{sum(trial['batch'] for trial in jobs) * args.steps:,} training samples "
        "(+ probe evaluations)"
    )
    print(
        f"Protocol: CIFAR-10, K={args.steps}, devices={','.join(args.device_list)}, "
        f"workers={args.workers}, max_micro_batch={args.max_micro_batch}, "
        f"probe_batch={args.probe_batch}"
    )
    if manifest["config"]["fake_data"]:
        print("SMOKE TEST: FakeData checks the pipeline; not a scientific result.")
    if args.dry_run:
        print(f"Saved {root / 'manifest.json'}; no training performed.")
        return 0

    pending = [
        (index, trial)
        for index, trial in enumerate(jobs, start=1)
        if trial["id"] not in cached
    ]
    limit = args.max_new_trials if args.max_new_trials is not None else len(pending)
    selected = [
        (index, new_index, trial)
        for new_index, (index, trial) in enumerate(pending[:limit], start=1)
    ]
    status = 0
    try:
        if selected:
            validate_environment(manifest["config"])
            validate_reuse_environment(root, manifest)
            active_devices = args.device_list[: min(len(args.device_list), len(selected))]
            if len(active_devices) == 1:
                device = active_devices[0]
                for index, new_index, trial in selected:
                    print(
                        f"\n[{index}/{len(jobs)}; new trial {new_index}; {device}] "
                        f"{trial['algorithm']} B={trial['batch']} gap={trial['gap']:g} "
                        f"lr={trial['lr']:g} seed={trial['seed']}",
                        flush=True,
                    )
                    run_trial(root, manifest, trial, device_override=device)
            else:
                print(
                    f"Dispatching {len(selected)} trials dynamically across "
                    f"{', '.join(active_devices)}",
                    flush=True,
                )
                run_parallel_trials(root, manifest, selected, active_devices)
        if len(pending) > len(selected):
            print(
                f"Reached --max-new-trials={args.max_new_trials}; "
                "rerun the same command to continue."
            )
    except KeyboardInterrupt:
        status = 130
        print("\nInterrupted. Completed trials are saved; active trials will restart.")
    except (Exception, SystemExit):
        status = 1
        traceback.print_exc()
        print("Training failed. Completed trials remain saved.")
    try:
        summarize(root, make_plot=not args.no_plot)
        print_boundaries(root)
    except (Exception, SystemExit):
        traceback.print_exc()
        print(f"Summary failed; trial files remain in {root / 'trials'}.")
        status = status or 1
    return status


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        with output_lock(args.output):
            log_directory = args.output / "logs"
            log_directory.mkdir(exist_ok=True)
            log_path = log_directory / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log"
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                    print(f"Log: {log_path}", flush=True)
                    try:
                        return execute(args)
                    except (Exception, SystemExit):
                        traceback.print_exc()
                        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
