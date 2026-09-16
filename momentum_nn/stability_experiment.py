#!/usr/bin/env python3
"""Fixed-horizon ImageNet stability scans using the repository's NN optimizers.

Run `sweep --help` for the protocol, `sweep --dry-run` to prepare a manifest,
`run --job-index N` to execute one manifest trial, or `summarize` to rebuild
tables and figures. Training uses one device and exact gradient accumulation.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any

try:
    from . import imagenet_momentum_experiment as nn_experiment
except ImportError:  # Also support `python momentum_nn/stability_experiment.py`.
    import imagenet_momentum_experiment as nn_experiment


ALGORITHMS = ("sgd", "polyak", "nesterov")
SCHEMA_VERSION = 1


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def positive_list(value: str, cast: Any = float) -> list:
    values = nn_experiment.comma_list(value, cast)
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("grid entries must be finite and positive")
    if len(values) != len(set(values)):
        raise ValueError("grid entries must be unique")
    return sorted(values)


def source_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(nn_experiment.__file__))
    }


def make_manifest(args: argparse.Namespace) -> dict:
    algorithms = nn_experiment.comma_list(args.algorithms)
    if set(algorithms) - set(ALGORITHMS) or len(algorithms) != len(set(algorithms)):
        raise ValueError(f"choose distinct algorithms from {ALGORITHMS}")
    batches = positive_list(args.batches, int)
    gaps = positive_list(args.gaps)
    seeds = nn_experiment.comma_list(args.seeds, int)
    if min(seeds) < 0 or max(seeds) >= 2**32 or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    if not 0 < args.fixed_gap <= 1 or any(g > 1 for g in gaps):
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
        raise ValueError("use at least two learning rates to locate a boundary")
    for name in (
        "steps",
        "eval_every",
        "tail_evals",
        "probe_size",
        "probe_batch",
        "max_micro_batch",
        "fixed_batch",
        "fake_train_size",
        "num_classes",
        "crop_size",
        "threads",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.crop_size < 32 or args.num_classes < 2 or args.workers < 0:
        raise ValueError("require crop-size >= 32, num-classes >= 2 and workers >= 0")
    n_evals = math.ceil(args.steps / args.eval_every)
    if 2 * args.tail_evals > n_evals:
        raise ValueError(
            "need at least 2 * tail-evals probe evaluations for tail growth"
        )
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
        args.learning_ratio < 1 <= args.stable_loss_ratio < args.divergence_loss_ratio
        and args.growth_ratio >= 1
        and args.max_parameter_ratio > 1
    ):
        raise ValueError("inconsistent stability / learning / divergence thresholds")
    if not 0 < args.required_fraction <= 1:
        raise ValueError("required-fraction must lie in (0, 1]")
    if args.fake_data and not args.smoke_test:
        raise ValueError(
            "fake data requires --smoke-test; it cannot yield scientific results"
        )
    if not args.fake_data and not args.data_root:
        raise ValueError("supply --data-root with train/<class>/ images")

    # Points shared by both scans have identical trial IDs, including their seeds.
    points: list[dict] = []
    if args.axis in ("batch", "both"):
        points.extend(
            {"axis": "batch", "x": b, "batch": b, "gap": args.fixed_gap}
            for b in batches
        )
    if args.axis in ("gap", "both"):
        points.extend(
            {"axis": "gap", "x": g, "batch": args.fixed_batch, "gap": g} for g in gaps
        )
    jobs: dict[str, dict] = {}
    for algorithm in algorithms:
        for point in points:
            gap = 1.0 if algorithm == "sgd" else point["gap"]
            if not args.smoke_test and args.steps * gap < 30:
                raise ValueError(
                    f"K(1-rho)={args.steps * gap:g} < 30 for {algorithm}; "
                    "increase --steps. Only --smoke-test may bypass this guard."
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
        "command",
        "runs_root",
        "dry_run",
        "shard_index",
        "shard_count",
        "no_plot",
    }
    config = {k: v for k, v in vars(args).items() if k not in ignored}
    config["data_root"] = (
        str(Path(args.data_root).expanduser().resolve()) if args.data_root else ""
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "source_hashes": source_hashes(),
        "points": points,
        "algorithms": algorithms,
        "lrs": lrs,
        "seeds": seeds,
        "jobs": list(jobs.values()),
    }
    payload["fingerprint"] = digest(payload)
    return payload


def read_manifest(root: Path, *, verify_source: bool = False) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if (
        digest(body) != manifest["fingerprint"]
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("manifest checksum or schema mismatch")
    if verify_source and manifest["source_hashes"] != source_hashes():
        raise ValueError(
            "experiment source changed; use a new runs-root to avoid mixing code"
        )
    return manifest


def result_path(root: Path, trial: dict) -> Path:
    return root / "trials" / trial["id"] / "summary.json"


def read_result(root: Path, manifest: dict, trial: dict) -> dict | None:
    path = result_path(root, trial)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if row.get("fingerprint") != manifest["fingerprint"] or row.get("trial") != trial:
        raise ValueError(f"mismatched cached result: {path}")
    return row


def classify(
    initial: float,
    tail: list[float],
    previous: list[float],
    completed: bool,
    reason: str,
    config: dict,
) -> dict:
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


def build_data(
    config: dict, trial: dict, torch: Any, datasets: Any, transforms: Any
) -> tuple:
    train_transform, probe_transform = nn_experiment.build_transforms(
        transforms, config["crop_size"]
    )
    if config["fake_data"]:
        kwargs = {
            "size": config["fake_train_size"],
            "image_size": (3, config["crop_size"], config["crop_size"]),
            "num_classes": config["num_classes"],
            "random_offset": 0,
        }
        train_dataset = datasets.FakeData(**kwargs, transform=train_transform)
        probe_dataset = datasets.FakeData(**kwargs, transform=probe_transform)
        classes = config["num_classes"]
        data_hash = digest(kwargs)
    else:
        train_root = Path(config["data_root"]) / "train"
        train_dataset = datasets.ImageFolder(train_root, train_transform)
        probe_dataset = copy.copy(train_dataset)
        probe_dataset.transform = probe_transform
        classes = len(train_dataset.classes)
        if classes < 2:
            raise ValueError("training requires at least two class folders")
        # File identities/sizes detect common changes without hashing all image bytes.
        listing = [
            (str(Path(p).relative_to(train_root)), y, Path(p).stat().st_size)
            for p, y in train_dataset.samples
        ]
        data_hash = digest({"classes": train_dataset.class_to_idx, "files": listing})
    if len(probe_dataset) < config["probe_size"]:
        raise ValueError("probe-size exceeds training dataset size")
    probe_generator = torch.Generator().manual_seed(config["probe_seed"])
    indices = torch.randperm(len(probe_dataset), generator=probe_generator)[
        : config["probe_size"]
    ]
    probe = torch.utils.data.Subset(probe_dataset, indices.tolist())
    micro_batch = min(config["max_micro_batch"], trial["batch"])
    # Sampling with replacement keeps the batch-noise protocol fixed across B.
    sampler = torch.utils.data.RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=config["steps"] * trial["batch"],
        generator=torch.Generator().manual_seed(trial["seed"] + 10_000),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=trial["batch"],
        num_workers=config["workers"],
        generator=torch.Generator().manual_seed(trial["seed"] + 20_000),
    )
    # Batch size here is the global batch on the host; only micro-batches go to GPU.
    probe_loader = torch.utils.data.DataLoader(
        probe,
        batch_size=config["probe_batch"],
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(config["probe_seed"] + 1),
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


def tensor_norm(tensors: Any, torch: Any) -> float:
    norms = [torch.linalg.vector_norm(t.detach()) for t in tensors if t is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def evaluate(
    model: Any, loader: Any, criterion: Any, torch: Any, device: Any
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device)
            logits = model(images.to(device))
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
    """Accumulate the exact batch-mean gradient, including a short last chunk.

    For Nesterov all chunks use the same look-ahead position; the base position
    is restored even if forward/backward raises. No optimizer update occurs here.
    """
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    if algorithm == "nesterov":
        optimizer.move_to_lookahead()
    try:
        for start in range(0, len(labels), micro_batch):
            stop = start + micro_batch
            target = labels[start:stop].to(device)
            raw_loss = criterion(model(images[start:stop].to(device)), target)
            weight = len(target) / len(labels)
            (raw_loss * weight).backward()
            loss_sum += float(raw_loss.detach().item()) * weight
    finally:
        if algorithm == "nesterov":
            optimizer.restore_base()
    return loss_sum


def run_trial(root: Path, manifest: dict, trial: dict) -> dict:
    cached = read_result(root, manifest, trial)
    if cached is not None:
        print(f"cached {trial['id']}", flush=True)
        return cached
    config = manifest["config"]
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(
            "use one device per trial; shard independent jobs instead of torchrun"
        )
    torch, _, nn, torchvision, datasets, transforms = (
        nn_experiment.import_training_stack()
    )
    torch.set_num_threads(config["threads"])
    requested = config["device"]
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(requested)
    nn_experiment.seed_everything(torch, trial["seed"], deterministic=True)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    train_loader, probe_loader, classes, data_info = build_data(
        config, trial, torch, datasets, transforms
    )
    model_args = argparse.Namespace(model=config["model"], norm="group")
    model = nn_experiment.build_model(model_args, nn, torchvision, classes).to(device)
    optimizer_args = argparse.Namespace(
        algorithm=trial["algorithm"],
        lr=trial["lr"],
        momentum=1 - trial["gap"],
        weight_decay=0.0,
    )
    optimizer = nn_experiment.optimizer_for(optimizer_args, torch, model)
    criterion = nn.CrossEntropyLoss()
    initial_loss, initial_accuracy = evaluate(
        model, probe_loader, criterion, torch, device
    )
    initial_norm = tensor_norm(model.parameters(), torch)
    if (
        not math.isfinite(initial_loss)
        or initial_loss <= 0
        or not math.isfinite(initial_norm)
    ):
        raise RuntimeError("invalid initialization; no stability label can be assigned")
    directory = result_path(root, trial).parent
    directory.mkdir(parents=True, exist_ok=True)
    nn_experiment.atomic_json_dump(
        {
            "fingerprint": manifest["fingerprint"],
            "trial": trial,
            "data": data_info,
            "initial_probe_loss": initial_loss,
        },
        directory / "config.json",
    )
    history = [
        {
            "step": 0,
            "probe_loss": initial_loss,
            "probe_loss_ratio": 1.0,
            "probe_accuracy": initial_accuracy,
            "parameter_norm": initial_norm,
        }
    ]
    recent: deque = deque(maxlen=2 * config["tail_evals"])
    reason, updates = "", 0
    started = time.monotonic()
    print(f"start {trial} K={config['steps']} device={device}", flush=True)
    # Stream diagnostics to disk so an interrupted trial still has inspectable logs.
    with (directory / "history.jsonl").open("w", encoding="utf-8") as log:
        log.write(json.dumps(history[0]) + "\n")
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
            grad_norm = tensor_norm((p.grad for p in model.parameters()), torch)
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
            row = {
                "step": step,
                "updates_completed": updates,
                "train_loss": train_loss,
                "gradient_norm": grad_norm,
                "parameter_norm": parameter_norm,
                "gradient_location": (
                    "lookahead" if trial["algorithm"] == "nesterov" else "base"
                ),
            }
            if not reason and (
                step % config["eval_every"] == 0 or step == config["steps"]
            ):
                probe_loss, accuracy = evaluate(
                    model, probe_loader, criterion, torch, device
                )
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
                print(
                    f"  step={step} probe/L0={probe_loss / initial_loss:.5g}",
                    flush=True,
                )
            row["stop_reason"] = reason
            log.write(json.dumps(nn_experiment.json_safe(row), allow_nan=False) + "\n")
            if reason or "probe_loss" in row:
                log.flush()
            if reason:
                break
    tail = list(recent)[-config["tail_evals"] :]
    previous = list(recent)[: -config["tail_evals"]]
    result = {
        "fingerprint": manifest["fingerprint"],
        "trial": trial,
        "smoke_test": config["smoke_test"],
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
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.machine()
            ),
            "dtype": "float32",
            "deterministic_algorithms": True,
        },
        **classify(
            initial_loss, tail, previous, updates == config["steps"], reason, config
        ),
    }
    nn_experiment.atomic_json_dump(result, result_path(root, trial))
    print(
        f"done bounded={result['bounded']} learned={result['learned']} reason={reason or 'none'}",
        flush=True,
    )
    return result


def boundary_from_grid(lrs: list[float], accepted: list[bool | None]) -> dict:
    """Only bracket a complete, monotone grid; preserve censoring and re-entry."""
    observed = [lr for lr, passed in zip(lrs, accepted) if passed is True]
    out = {
        "largest_observed_stable_lr": max(observed) if observed else None,
        "lower": None,
        "upper": None,
    }
    if any(v is None for v in accepted):
        return {**out, "status": "incomplete"}
    if not any(accepted):
        return {**out, "upper": lrs[0], "status": "left_censored"}
    seen_failure = False
    for passed in accepted:
        if not passed:
            seen_failure = True
        elif seen_failure:
            return {**out, "status": "nonmonotone"}
    if all(accepted):
        return {**out, "lower": lrs[-1], "status": "right_censored"}
    index = accepted.index(False)
    return {**out, "lower": lrs[index - 1], "upper": lrs[index], "status": "bracketed"}


def summarize(root: Path, *, make_plot: bool = True) -> None:
    manifest = read_manifest(root)
    results = {}
    for trial in manifest["jobs"]:
        row = read_result(root, manifest, trial)
        if row is not None:
            results[trial["id"]] = row
    # Never combine trials that read different dataset listings.
    if len({r["data"]["file_listing_hash"] for r in results.values()}) > 1:
        raise ValueError(
            "dataset listing changed across trials; summaries would be confounded"
        )
    cells, boundaries, per_seed = [], [], []
    lookup = {
        (
            r["trial"]["algorithm"],
            r["trial"]["batch"],
            r["trial"]["gap"],
            r["trial"]["lr"],
            r["trial"]["seed"],
        ): r
        for r in results.values()
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
            accepted = []
            for lr in manifest["lrs"]:
                gap = 1.0 if algorithm == "sgd" else point["gap"]
                rows = [
                    lookup.get((algorithm, point["batch"], gap, lr, seed))
                    for seed in manifest["seeds"]
                ]
                observed = [r for r in rows if r is not None]
                rates = {
                    metric
                    + "_fraction": (
                        statistics.fmean(r[metric] for r in observed)
                        if observed
                        else None
                    )
                    for metric in ("survived", "bounded", "learned")
                }
                complete = len(observed) == len(manifest["seeds"])
                accepted.append(
                    rates["bounded_fraction"] >= manifest["config"]["required_fraction"]
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
            boundaries.append({**base, **boundary_from_grid(manifest["lrs"], accepted)})
            for seed in manifest["seeds"]:
                seed_labels = []
                for lr in manifest["lrs"]:
                    gap = 1.0 if algorithm == "sgd" else point["gap"]
                    row = lookup.get((algorithm, point["batch"], gap, lr, seed))
                    seed_labels.append(row["bounded"] if row else None)
                per_seed.append(
                    {
                        **base,
                        "seed": seed,
                        **boundary_from_grid(manifest["lrs"], seed_labels),
                    }
                )
    nn_experiment.write_csv(cells, root / "stability_grid.csv")
    nn_experiment.write_csv(boundaries, root / "boundaries.csv")
    nn_experiment.write_csv(per_seed, root / "per_seed_boundaries.csv")
    nn_experiment.atomic_json_dump(
        {
            "fingerprint": manifest["fingerprint"],
            "completed": len(results),
            "planned": len(manifest["jobs"]),
            "smoke_test": manifest["config"]["smoke_test"],
            "boundaries": boundaries,
        },
        root / "summary.json",
    )
    if make_plot:
        try:
            from .plot_stability import plot
        except ImportError:
            from plot_stability import plot
        plot(root, manifest, cells, boundaries)
    print(f"summarized {len(results)}/{len(manifest['jobs'])} trials in {root}")


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    commands = out.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser(
        "sweep", help="prepare or execute fixed-K stability scans"
    )
    sweep.add_argument("--runs-root", required=True)
    sweep.add_argument("--data-root", default="")
    sweep.add_argument("--axis", choices=("batch", "gap", "both"), default="both")
    sweep.add_argument("--algorithms", default="sgd,polyak,nesterov")
    sweep.add_argument("--batches", default="1,2,4,8,16,32,64,128,256")
    sweep.add_argument(
        "--gaps", default=",".join(str(2.0 ** (-6 + i / 2)) for i in range(10))
    )
    sweep.add_argument("--fixed-gap", type=float, default=2**-6)
    sweep.add_argument("--fixed-batch", type=int, default=4)
    sweep.add_argument(
        "--lrs", default="", help="absolute learning rates shared by all B and methods"
    )
    sweep.add_argument("--lr-min", type=float, default=2**-10)
    sweep.add_argument("--lr-max", type=float, default=8.0)
    sweep.add_argument("--lr-count", type=int, default=50)
    sweep.add_argument("--seeds", default="0,1,2,3,4")
    sweep.add_argument("--steps", type=int, default=2000)
    sweep.add_argument("--eval-every", type=int, default=50)
    sweep.add_argument("--tail-evals", type=int, default=5)
    sweep.add_argument("--probe-size", type=int, default=1024)
    sweep.add_argument("--probe-batch", type=int, default=64)
    sweep.add_argument("--probe-seed", type=int, default=1729)
    sweep.add_argument("--stable-loss-ratio", type=float, default=2.0)
    sweep.add_argument("--growth-ratio", type=float, default=1.25)
    sweep.add_argument("--learning-ratio", type=float, default=0.95)
    sweep.add_argument("--divergence-loss-ratio", type=float, default=100.0)
    sweep.add_argument("--max-grad-norm", type=float, default=1e6)
    sweep.add_argument("--max-parameter-ratio", type=float, default=100.0)
    sweep.add_argument("--required-fraction", type=float, default=1.0)
    sweep.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet18")
    sweep.add_argument("--crop-size", type=int, default=224)
    sweep.add_argument("--max-micro-batch", type=int, default=64)
    sweep.add_argument("--workers", type=int, default=4)
    sweep.add_argument("--threads", type=int, default=4)
    sweep.add_argument(
        "--device", default="auto", help="auto, cpu, mps, cuda or cuda:N"
    )
    sweep.add_argument("--fake-data", action="store_true")
    sweep.add_argument("--fake-train-size", type=int, default=128)
    sweep.add_argument("--num-classes", type=int, default=10)
    sweep.add_argument(
        "--smoke-test",
        action="store_true",
        help="label as pipeline check; bypass K*gap guard",
    )
    sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="save manifest without loading data or torch",
    )
    sweep.add_argument("--shard-count", type=int, default=1)
    sweep.add_argument("--shard-index", type=int, default=0)
    sweep.add_argument("--no-plot", action="store_true")
    run = commands.add_parser("run", help="execute one job from an existing manifest")
    run.add_argument("--runs-root", required=True)
    run.add_argument(
        "--job-index", type=int, required=True, help="zero-based manifest job index"
    )
    summary = commands.add_parser("summarize")
    summary.add_argument("--runs-root", required=True)
    summary.add_argument("--no-plot", action="store_true")
    return out


def main() -> None:
    args = parser().parse_args()
    root = Path(args.runs_root).expanduser().resolve()
    if args.command == "summarize":
        summarize(root, make_plot=not args.no_plot)
    elif args.command == "run":
        manifest = read_manifest(root, verify_source=True)
        if not 0 <= args.job_index < len(manifest["jobs"]):
            raise ValueError("job-index is outside the manifest")
        run_trial(root, manifest, manifest["jobs"][args.job_index])
    else:
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("require 0 <= shard-index < shard-count")
        manifest = make_manifest(args)
        root.mkdir(parents=True, exist_ok=True)
        if (root / "manifest.json").exists():
            if read_manifest(root)["fingerprint"] != manifest["fingerprint"]:
                raise ValueError(
                    "runs-root has a different protocol; choose a new directory"
                )
        else:
            nn_experiment.atomic_json_dump(manifest, root / "manifest.json")
        planned_steps = len(manifest["jobs"]) * args.steps
        planned_samples = sum(j["batch"] for j in manifest["jobs"]) * args.steps
        print(
            f"{len(manifest['jobs'])} unique trials; up to {planned_steps:,} updates, "
            f"{planned_samples:,} training samples (+ probe evaluations)"
        )
        if args.dry_run:
            print(f"saved {root / 'manifest.json'}; no training performed")
            return
        for index, trial in enumerate(manifest["jobs"]):
            if index % args.shard_count == args.shard_index:
                run_trial(root, manifest, trial)
        # Concurrent shards write only their trial files. Summarize after they finish.
        if args.shard_count == 1:
            summarize(root, make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
