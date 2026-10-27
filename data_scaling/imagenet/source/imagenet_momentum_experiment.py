from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


IMAGENET_TRAIN_SIZE = 1_281_167
IMAGENET_VAL_SIZE = 50_000


def comma_list(text: str, cast: Any = str) -> list[Any]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("the comma-separated list cannot be empty")
    try:
        return [cast(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def float_tag(value: float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def sample_std(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return 0.0 if finite else None
    return statistics.stdev(finite)


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


@dataclass
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: Any


def import_training_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.distributed as dist
        import torch.nn as nn
        import torchvision
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
    except ImportError as exc:
        raise SystemExit(
            "PyTorch and torchvision are required. Install a CUDA-compatible "
            "PyTorch build before running this command."
        ) from exc
    return torch, dist, nn, torchvision, datasets, transforms


def setup_distributed(torch: Any, dist: Any, requested_local_rank: int) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(requested_local_rank)))
    distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        if distributed:
            raise RuntimeError("distributed ImageNet training requires CUDA/NCCL in this script")
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return DistributedContext(distributed, rank, local_rank, world_size, device)


def cleanup_distributed(context: DistributedContext, dist: Any) -> None:
    if context.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(torch: Any, seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def group_norm_factory(nn: Any) -> Any:
    def make_group_norm(channels: int) -> Any:
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)

    return make_group_norm


def build_model(args: argparse.Namespace, nn: Any, torchvision: Any, num_classes: int) -> Any:
    if args.model not in {"resnet18", "resnet50"}:
        raise ValueError("this experiment currently supports resnet18 and resnet50")
    builder = getattr(torchvision.models, args.model)
    kwargs: dict[str, Any] = {"weights": None, "num_classes": num_classes}
    if args.norm == "group":
        kwargs["norm_layer"] = group_norm_factory(nn)
    return builder(**kwargs)


def build_transforms(transforms: Any, crop_size: int) -> tuple[Any, Any]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    resize_size = int(round(crop_size / 0.875))
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def build_datasets(
    args: argparse.Namespace,
    datasets: Any,
    transforms: Any,
) -> tuple[Any, Any, int]:
    train_transform, val_transform = build_transforms(transforms, args.crop_size)
    if args.fake_data:
        train_dataset = datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, args.crop_size, args.crop_size),
            num_classes=args.num_classes,
            transform=train_transform,
            random_offset=0,
        )
        val_dataset = datasets.FakeData(
            size=args.fake_val_size,
            image_size=(3, args.crop_size, args.crop_size),
            num_classes=args.num_classes,
            transform=val_transform,
            random_offset=args.fake_train_size,
        )
        return train_dataset, val_dataset, args.num_classes

    root = Path(args.data_root).expanduser().resolve()
    train_root = root / "train"
    val_root = root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(
            f"expected ImageFolder directories {train_root} and {val_root}"
        )
    train_dataset = datasets.ImageFolder(train_root, train_transform)
    val_dataset = datasets.ImageFolder(val_root, val_transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train and validation class folders do not match")
    return train_dataset, val_dataset, len(train_dataset.classes)


def choose_micro_batch(global_batch: int, world_size: int, maximum: int) -> tuple[int, int]:
    if global_batch <= 0:
        raise ValueError("global batch size must be positive")
    if global_batch % world_size != 0:
        raise ValueError(
            f"global batch {global_batch} must be divisible by world size {world_size}"
        )
    per_rank = global_batch // world_size
    micro_batch = min(maximum, per_rank)
    while micro_batch > 0 and per_rank % micro_batch != 0:
        micro_batch -= 1
    if micro_batch <= 0:
        raise ValueError("could not find a valid per-device micro-batch")
    accumulation_steps = per_rank // micro_batch
    return micro_batch, accumulation_steps


class BatchStream:
    def __init__(self, loader: Any, sampler: Any | None) -> None:
        self.loader = loader
        self.sampler = sampler
        self.epoch = 0
        self.iterator: Any | None = None
        self._reset()

    def _reset(self) -> None:
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        self.iterator = iter(self.loader)

    def next(self) -> Any:
        assert self.iterator is not None
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self._reset()
            assert self.iterator is not None
            return next(self.iterator)


def build_loaders(
    args: argparse.Namespace,
    torch: Any,
    train_dataset: Any,
    val_dataset: Any,
    context: DistributedContext,
    micro_batch: int,
) -> tuple[Any, Any, Any | None]:
    sampler = None
    if context.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )

    loader_kwargs: dict[str, Any] = {
        "num_workers": args.workers,
        "pin_memory": context.device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    generator = torch.Generator()
    generator.manual_seed(args.seed + context.rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=micro_batch,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "the training loader is empty; reduce the micro-batch or enlarge the fake dataset"
        )

    val_loader = None
    if context.rank == 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
    return train_loader, val_loader, sampler


def make_grad_scaler(torch: Any, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(torch: Any, device: Any, precision: str) -> Any:
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def set_learning_rate(
    optimizer: Any,
    peak_lr: float,
    update: int,
    total_updates: int,
    warmup_updates: int,
    schedule: str,
) -> float:
    if warmup_updates > 0 and update < warmup_updates:
        learning_rate = peak_lr * float(update + 1) / float(warmup_updates)
    elif schedule == "constant":
        learning_rate = peak_lr
    else:
        denominator = max(1, total_updates - warmup_updates - 1)
        progress = min(1.0, max(0.0, (update - warmup_updates) / denominator))
        learning_rate = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    return learning_rate


def reduce_mean(value: float, torch: Any, dist: Any, context: DistributedContext) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=context.device)
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= context.world_size
    return float(tensor.item())


def reduce_any(flag: bool, torch: Any, dist: Any, context: DistributedContext) -> bool:
    tensor = torch.tensor(int(flag), dtype=torch.int32, device=context.device)
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def topk_correct(output: Any, target: Any, topk: Iterable[int]) -> list[float]:
    requested = list(topk)
    maximum = min(max(requested), output.shape[1])
    _, prediction = output.topk(maximum, dim=1, largest=True, sorted=True)
    prediction = prediction.t()
    correct = prediction.eq(target.reshape(1, -1).expand_as(prediction))
    results = []
    for k in requested:
        effective = min(k, output.shape[1])
        results.append(float(correct[:effective].reshape(-1).float().sum().item()))
    return results


def validate(
    model: Any,
    loader: Any,
    criterion: Any,
    torch: Any,
    context: DistributedContext,
    precision: str,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct1 = 0.0
    correct5 = 0.0
    count = 0
    with torch.no_grad():
        for images, target in loader:
            images = images.to(context.device, non_blocking=True)
            target = target.to(context.device, non_blocking=True)
            with autocast_context(torch, context.device, precision):
                output = model(images)
                loss = criterion(output, target)
            batch_size = int(target.numel())
            top1, top5 = topk_correct(output, target, (1, 5))
            loss_sum += float(loss.item()) * batch_size
            correct1 += top1
            correct5 += top5
            count += batch_size
    return {
        "val_loss": loss_sum / max(1, count),
        "val_top1": 100.0 * correct1 / max(1, count),
        "val_top5": 100.0 * correct5 / max(1, count),
    }


def optimizer_for(args: argparse.Namespace, torch: Any, model: Any) -> Any:
    if args.algorithm == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.0,
            weight_decay=args.weight_decay,
        )
    if not 0.0 <= args.momentum < 1.0:
        raise ValueError("momentum must lie in [0, 1)")

    class PaperMomentum(torch.optim.Optimizer):
        """Position-velocity implementation of equations (3) and (4).

        The state ``velocity`` is exactly theta_k - theta_{k-1}.  For
        Nesterov, training code temporarily moves parameters to
        theta_k + rho * velocity before forward/backward, then restores the
        base point before ``step``.  Validation therefore evaluates theta_k,
        matching the paper rather than a reparameterized look-ahead sequence.
        """

        def __init__(
            self,
            params: Any,
            lr: float,
            momentum: float,
            weight_decay: float,
            lookahead: bool,
        ) -> None:
            super().__init__(
                params,
                {
                    "lr": lr,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                },
            )
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
                    state = self.state[parameter]
                    velocity = state.setdefault(
                        "velocity", torch.zeros_like(parameter, memory_format=torch.preserve_format)
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
                    velocity = self.state[parameter]["velocity"]
                    parameter.sub_(velocity, alpha=rho)
            self.at_lookahead = False

        @torch.no_grad()
        def step(self, closure: Any | None = None) -> Any:
            if self.at_lookahead:
                raise RuntimeError("restore the base parameters before optimizer.step()")
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                learning_rate = float(group["lr"])
                rho = float(group["momentum"])
                weight_decay = float(group["weight_decay"])
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    gradient = parameter.grad
                    state = self.state[parameter]
                    velocity = state.setdefault(
                        "velocity", torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    )
                    if weight_decay != 0.0:
                        if self.lookahead:
                            gradient = gradient.add(parameter, alpha=weight_decay)
                            gradient = gradient.add(
                                velocity,
                                alpha=weight_decay * rho,
                            )
                        else:
                            gradient = gradient.add(parameter, alpha=weight_decay)
                    velocity.mul_(rho).add_(gradient, alpha=-learning_rate)
                    parameter.add_(velocity)
            return loss

    return PaperMomentum(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lookahead=args.algorithm == "nesterov",
    )


def train_one(args: argparse.Namespace) -> dict[str, Any] | None:
    torch, dist, nn, torchvision, datasets, transforms = import_training_stack()
    context = setup_distributed(torch, dist, args.local_rank)
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        seed_everything(torch, args.seed, args.deterministic)
        train_dataset, val_dataset, num_classes = build_datasets(args, datasets, transforms)
        if args.micro_batch > 0:
            expected = args.micro_batch * context.world_size
            if args.global_batch % expected != 0:
                raise ValueError(
                    "global batch must be divisible by micro_batch * world_size"
                )
            micro_batch = args.micro_batch
            accumulation_steps = args.global_batch // expected
        else:
            micro_batch, accumulation_steps = choose_micro_batch(
                args.global_batch,
                context.world_size,
                args.max_per_device_batch,
            )

        train_loader, val_loader, sampler = build_loaders(
            args,
            torch,
            train_dataset,
            val_dataset,
            context,
            micro_batch,
        )

        model = build_model(args, nn, torchvision, num_classes).to(context.device)
        if args.norm == "batch" and args.sync_batchnorm and context.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if context.distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=True,
            )

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optimizer_for(args, torch, model)
        scaler = make_grad_scaler(
            torch,
            enabled=args.precision == "fp16" and context.device.type == "cuda",
        )

        sample_budget = args.sample_budget
        if sample_budget <= 0:
            sample_budget = int(round(args.epochs_equivalent * len(train_dataset)))
        total_updates = max(1, sample_budget // args.global_batch)
        warmup_updates = int(
            round(args.warmup_epochs_equivalent * len(train_dataset) / args.global_batch)
        )
        warmup_updates = min(warmup_updates, total_updates)

        if context.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(
                json.dumps(
                    {
                        "algorithm": args.algorithm,
                        "model": args.model,
                        "norm": args.norm,
                        "world_size": context.world_size,
                        "micro_batch_per_device": micro_batch,
                        "accumulation_steps": accumulation_steps,
                        "global_batch": args.global_batch,
                        "sample_budget_requested": sample_budget,
                        "optimizer_updates": total_updates,
                        "samples_actually_processed": total_updates * args.global_batch,
                        "learning_rate": args.lr,
                        "momentum": 0.0 if args.algorithm == "sgd" else args.momentum,
                    },
                    indent=2,
                ),
                flush=True,
            )

        stream = BatchStream(train_loader, sampler)
        recent_losses: deque[float] = deque(maxlen=max(1, args.final_window_updates))
        log_rows: list[dict[str, Any]] = []
        divergence_count = 0
        diverged = False
        divergence_reason = ""
        updates_completed = 0
        started = time.time()

        model.train()
        for update in range(total_updates):
            learning_rate = set_learning_rate(
                optimizer,
                args.lr,
                update,
                total_updates,
                warmup_updates,
                args.schedule,
            )
            optimizer.zero_grad(set_to_none=True)
            local_loss = 0.0
            if args.algorithm == "nesterov":
                optimizer.move_to_lookahead()

            try:
                for micro_step in range(accumulation_steps):
                    images, target = stream.next()
                    images = images.to(context.device, non_blocking=True)
                    target = target.to(context.device, non_blocking=True)
                    synchronization = nullcontext()
                    if context.distributed and micro_step + 1 < accumulation_steps:
                        synchronization = model.no_sync()
                    with synchronization:
                        with autocast_context(torch, context.device, args.precision):
                            output = model(images)
                            raw_loss = criterion(output, target)
                            loss = raw_loss / accumulation_steps
                        scaler.scale(loss).backward()
                    local_loss += float(raw_loss.detach().item()) / accumulation_steps
            finally:
                if args.algorithm == "nesterov":
                    optimizer.restore_base()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            gradient_norm_tensor = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            )
            gradient_norm = float(gradient_norm_tensor.detach().item())
            local_bad = (
                not math.isfinite(local_loss)
                or not math.isfinite(gradient_norm)
                or gradient_norm > args.divergence_grad_norm
            )
            globally_bad = reduce_any(local_bad, torch, dist, context)
            global_loss = reduce_mean(local_loss, torch, dist, context)

            if global_loss > args.divergence_loss:
                divergence_count += 1
            else:
                divergence_count = 0
            if divergence_count >= args.divergence_patience:
                globally_bad = True
                divergence_reason = (
                    f"loss exceeded {args.divergence_loss} for "
                    f"{args.divergence_patience} consecutive updates"
                )

            if globally_bad:
                diverged = True
                if not divergence_reason:
                    divergence_reason = "non-finite loss/gradient or gradient-norm threshold"
                optimizer.zero_grad(set_to_none=True)
                break

            scaler.step(optimizer)
            scaler.update()
            updates_completed = update + 1
            recent_losses.append(global_loss)

            if context.rank == 0 and (
                update == 0
                or (update + 1) % args.log_every_updates == 0
                or update + 1 == total_updates
            ):
                row = {
                    "update": update + 1,
                    "samples_seen": (update + 1) * args.global_batch,
                    "train_loss": global_loss,
                    "gradient_norm": gradient_norm,
                    "learning_rate": learning_rate,
                    "elapsed_seconds": time.time() - started,
                }
                log_rows.append(row)
                print(json.dumps(row), flush=True)

        if context.distributed:
            dist.barrier()

        validation = {"val_loss": None, "val_top1": None, "val_top5": None}
        if context.rank == 0 and not diverged:
            evaluation_model = model.module if context.distributed else model
            assert val_loader is not None
            validation = validate(
                evaluation_model,
                val_loader,
                criterion,
                torch,
                context,
                args.precision,
            )

        if context.distributed:
            dist.barrier()

        summary: dict[str, Any] | None = None
        if context.rank == 0:
            stable = not diverged and updates_completed == total_updates
            summary = {
                "algorithm": args.algorithm,
                "model": args.model,
                "norm": args.norm,
                "seed": args.seed,
                "global_batch": args.global_batch,
                "micro_batch_per_device": micro_batch,
                "accumulation_steps": accumulation_steps,
                "world_size": context.world_size,
                "lr": args.lr,
                "momentum": 0.0 if args.algorithm == "sgd" else args.momentum,
                "weight_decay": args.weight_decay,
                "schedule": args.schedule,
                "warmup_epochs_equivalent": args.warmup_epochs_equivalent,
                "precision": args.precision,
                "sample_budget_requested": sample_budget,
                "samples_processed": updates_completed * args.global_batch,
                "total_updates_requested": total_updates,
                "updates_completed": updates_completed,
                "stable": stable,
                "divergence_reason": divergence_reason,
                "final_train_loss": mean(list(recent_losses)),
                "elapsed_seconds": time.time() - started,
                **validation,
            }
            write_csv(log_rows, output_dir / "train_metrics.csv")
            atomic_json_dump(summary, output_dir / "summary.json")
            if args.save_checkpoint and stable:
                evaluation_model = model.module if context.distributed else model
                torch.save(
                    {
                        "model": evaluation_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "summary": summary,
                        "args": vars(args),
                    },
                    output_dir / "last.pt",
                )
            print(json.dumps(json_safe(summary), indent=2), flush=True)
        return summary
    finally:
        cleanup_distributed(context, dist)


def check_data(args: argparse.Namespace) -> None:
    _, _, _, _, datasets, transforms = import_training_stack()
    train_transform, val_transform = build_transforms(transforms, args.crop_size)
    root = Path(args.data_root).expanduser().resolve()
    train_dataset = datasets.ImageFolder(root / "train", train_transform)
    val_dataset = datasets.ImageFolder(root / "val", val_transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise SystemExit("train and validation class mappings differ")
    report = {
        "root": str(root),
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "classes": len(train_dataset.classes),
        "expected_train_images": IMAGENET_TRAIN_SIZE,
        "expected_validation_images": IMAGENET_VAL_SIZE,
        "looks_like_full_imagenet1k": (
            len(train_dataset) == IMAGENET_TRAIN_SIZE
            and len(val_dataset) == IMAGENET_VAL_SIZE
            and len(train_dataset.classes) == 1000
        ),
    }
    print(json.dumps(report, indent=2))


def common_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="", help="root containing train/ and val/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--norm", choices=("group", "batch"), default="group")
    parser.add_argument("--sync-batchnorm", action="store_true")
    parser.add_argument("--algorithm", choices=("sgd", "polyak", "nesterov"), default="sgd")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, default=0)
    parser.add_argument("--max-per-device-batch", type=int, default=128)
    parser.add_argument("--sample-budget", type=int, default=0)
    parser.add_argument("--epochs-equivalent", type=float, default=30.0)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-epochs-equivalent", type=float, default=0.0)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-every-updates", type=int, default=100)
    parser.add_argument("--final-window-updates", type=int, default=100)
    parser.add_argument("--divergence-loss", type=float, default=20.0)
    parser.add_argument("--divergence-grad-norm", type=float, default=1e6)
    parser.add_argument("--divergence-patience", type=int, default=5)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--fake-train-size", type=int, default=4096)
    parser.add_argument("--fake-val-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)


def build_train_command(
    script: Path,
    args: argparse.Namespace,
    run_dir: Path,
    algorithm: str,
    global_batch: int,
    learning_rate: float,
    momentum: float,
    seed: int,
) -> list[str]:
    if args.nproc_per_node > 1:
        command = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            str(script),
            "train",
        ]
    else:
        command = [sys.executable, str(script), "train"]

    command.extend(
        [
            "--data-root",
            args.data_root,
            "--output-dir",
            str(run_dir),
            "--model",
            args.model,
            "--norm",
            args.norm,
            "--algorithm",
            algorithm,
            "--lr",
            str(learning_rate),
            "--momentum",
            str(momentum),
            "--weight-decay",
            str(args.weight_decay),
            "--global-batch",
            str(global_batch),
            "--max-per-device-batch",
            str(args.max_per_device_batch),
            "--sample-budget",
            str(args.sample_budget),
            "--epochs-equivalent",
            str(args.epochs_equivalent),
            "--schedule",
            args.schedule,
            "--warmup-epochs-equivalent",
            str(args.warmup_epochs_equivalent),
            "--precision",
            args.precision,
            "--label-smoothing",
            str(args.label_smoothing),
            "--workers",
            str(args.workers),
            "--seed",
            str(seed),
            "--log-every-updates",
            str(args.log_every_updates),
            "--final-window-updates",
            str(args.final_window_updates),
            "--divergence-loss",
            str(args.divergence_loss),
            "--divergence-grad-norm",
            str(args.divergence_grad_norm),
            "--divergence-patience",
            str(args.divergence_patience),
        ]
    )
    if args.sync_batchnorm:
        command.append("--sync-batchnorm")
    if args.deterministic:
        command.append("--deterministic")
    if args.fake_data:
        command.extend(
            [
                "--fake-data",
                "--fake-train-size",
                str(args.fake_train_size),
                "--fake-val-size",
                str(args.fake_val_size),
                "--num-classes",
                str(args.num_classes),
            ]
        )
    return command


def shell_join(command: Sequence[str]) -> str:
    import shlex

    return " ".join(shlex.quote(item) for item in command)


def all_hyperparameter_configurations(
    args: argparse.Namespace,
) -> list[tuple[str, int, float, float]]:
    algorithms = comma_list(args.algorithms, str)
    batches = comma_list(args.batches, int)
    base_lrs = comma_list(args.lrs, float)
    momentums = comma_list(args.momentums, float)
    invalid_algorithms = set(algorithms) - {"sgd", "polyak", "nesterov"}
    if invalid_algorithms:
        raise ValueError(f"unsupported algorithms: {sorted(invalid_algorithms)}")

    configurations: list[tuple[str, int, float, float]] = []
    seen: set[tuple[str, int, float, float]] = set()
    for algorithm in algorithms:
        algorithm_momenta = [0.0] if algorithm == "sgd" else momentums
        for global_batch in batches:
            learning_rates: set[float] = set()
            for base_lr in base_lrs:
                if args.lr_grid_mode in {"absolute", "hybrid"}:
                    learning_rates.add(base_lr)
                if args.lr_grid_mode in {"linear", "hybrid"}:
                    learning_rates.add(base_lr * global_batch / args.reference_batch)
            for learning_rate in sorted(learning_rates):
                for momentum in algorithm_momenta:
                    if not 0.0 <= momentum < 1.0:
                        raise ValueError("momentum values must lie in [0, 1)")
                    configuration = (algorithm, global_batch, learning_rate, momentum)
                    if configuration not in seen:
                        seen.add(configuration)
                        configurations.append(configuration)
    return configurations


def make_sweep_jobs(
    script: Path,
    args: argparse.Namespace,
    root: Path,
    configurations: Sequence[tuple[str, int, float, float]],
    seeds: Sequence[int],
) -> list[tuple[Path, list[str]]]:
    jobs: list[tuple[Path, list[str]]] = []
    for algorithm, global_batch, learning_rate, momentum in configurations:
        for seed in seeds:
            run_name = (
                f"{algorithm}_B{global_batch}_lr{float_tag(learning_rate)}_"
                f"rho{float_tag(momentum)}_seed{seed}"
            )
            run_dir = root / run_name
            command = build_train_command(
                script,
                args,
                run_dir,
                algorithm,
                global_batch,
                learning_rate,
                momentum,
                seed,
            )
            jobs.append((run_dir, command))
    return jobs


def write_job_script(jobs: Sequence[tuple[Path, list[str]]], path: Path) -> None:
    command_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    command_lines.extend(shell_join(command) for _, command in jobs)
    command_lines.append("")
    path.write_text("\n".join(command_lines), encoding="utf-8")
    path.chmod(0o755)
    print(f"wrote {len(jobs)} jobs to {path}")


def execute_jobs(args: argparse.Namespace, jobs: Sequence[tuple[Path, list[str]]]) -> int:
    failures = 0
    for index, (run_dir, command) in enumerate(jobs, start=1):
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            print(f"[{index}/{len(jobs)}] skip completed {run_dir.name}")
            continue
        print(f"[{index}/{len(jobs)}] {shell_join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures += 1
            print(f"job failed with return code {completed.returncode}: {run_dir}")
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
    return failures


def top_configurations_from_coarse_search(
    root: Path,
    search_seed: int,
    top_k: int,
) -> list[tuple[str, int, float, float]]:
    coarse_rows = [
        row for row in load_summaries(root) if int(row.get("seed", -1)) == search_seed
    ]
    if not coarse_rows:
        raise SystemExit(
            "no coarse-search results were found; run `sweep --phase coarse` first"
        )
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in coarse_rows:
        key = (str(row["algorithm"]), int(row["global_batch"]))
        groups.setdefault(key, []).append(row)

    selected: list[tuple[str, int, float, float]] = []
    report_rows: list[dict[str, Any]] = []
    for (algorithm, batch), candidates in sorted(groups.items()):
        stable = [
            row
            for row in candidates
            if bool(row.get("stable")) and row.get("val_loss") is not None
        ]
        stable.sort(key=lambda row: float(row["val_loss"]))
        if not stable:
            raise SystemExit(
                f"no stable configuration for {algorithm}, B={batch}; broaden the LR/rho grid"
            )
        for rank, row in enumerate(stable[:top_k], start=1):
            configuration = (
                algorithm,
                batch,
                float(row["lr"]),
                float(row["momentum"]),
            )
            selected.append(configuration)
            report_rows.append(
                {
                    "algorithm": algorithm,
                    "global_batch": batch,
                    "coarse_rank": rank,
                    "lr": configuration[2],
                    "momentum": configuration[3],
                    "coarse_val_loss": float(row["val_loss"]),
                    "coarse_val_top1": row.get("val_top1"),
                }
            )
    write_csv(report_rows, root / "coarse_top_configurations.csv")
    return selected


def run_sweep(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    root = Path(args.runs_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    all_configurations = all_hyperparameter_configurations(args)
    final_seeds = comma_list(args.seeds, int)
    if args.search_seed not in final_seeds:
        final_seeds = [args.search_seed, *final_seeds]
    failures = 0
    if args.phase in {"coarse", "all"}:
        coarse_jobs = make_sweep_jobs(
            script,
            args,
            root,
            all_configurations,
            [args.search_seed],
        )
        write_job_script(coarse_jobs, root / "commands_coarse.sh")
        if args.dry_run:
            for _, command in coarse_jobs:
                print(shell_join(command))
            if args.phase == "all":
                print("refinement commands require coarse results; rerun with --phase refine afterwards")
            return
        failures += execute_jobs(args, coarse_jobs)
        summarize_runs(root, make_plot=not args.no_plot, required_runs=1)

    if args.phase in {"refine", "all"}:
        selected = top_configurations_from_coarse_search(
            root,
            args.search_seed,
            args.top_k,
        )
        refine_jobs = make_sweep_jobs(
            script,
            args,
            root,
            selected,
            final_seeds,
        )
        write_job_script(refine_jobs, root / "commands_refine.sh")
        if args.dry_run:
            for _, command in refine_jobs:
                print(shell_join(command))
            return
        failures += execute_jobs(args, refine_jobs)
        summarize_runs(
            root,
            make_plot=not args.no_plot,
            required_runs=len(set(final_seeds)),
        )
    if failures:
        print(f"completed with {failures} failed jobs")


def load_summaries(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/summary.json")):
        with path.open("r", encoding="utf-8") as stream:
            row = json.load(stream)
        row["run_dir"] = str(path.parent)
        rows.append(row)
    return rows


def aggregate_configurations(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("algorithm"),
            row.get("global_batch"),
            row.get("lr"),
            row.get("momentum"),
        )
        groups.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (algorithm, batch, learning_rate, momentum), group in groups.items():
        stable_values = [1.0 if bool(row.get("stable")) else 0.0 for row in group]
        stable_rows = [row for row in group if bool(row.get("stable"))]
        val_losses = [
            float(row["val_loss"])
            for row in stable_rows
            if row.get("val_loss") is not None
        ]
        top1_values = [
            float(row["val_top1"])
            for row in stable_rows
            if row.get("val_top1") is not None
        ]
        train_losses = [
            float(row["final_train_loss"])
            for row in stable_rows
            if row.get("final_train_loss") is not None
        ]
        aggregated.append(
            {
                "algorithm": algorithm,
                "global_batch": batch,
                "lr": learning_rate,
                "momentum": momentum,
                "runs": len(group),
                "stable_rate": statistics.fmean(stable_values),
                "val_loss_mean": mean(val_losses),
                "val_loss_std": sample_std(val_losses),
                "val_top1_mean": mean(top1_values),
                "val_top1_std": sample_std(top1_values),
                "train_loss_mean": mean(train_losses),
                "train_loss_std": sample_std(train_losses),
            }
        )
    aggregated.sort(
        key=lambda row: (
            str(row["algorithm"]),
            int(row["global_batch"]),
            float(row["momentum"]),
            float(row["lr"]),
        )
    )
    return aggregated


def select_best_by_batch(
    aggregated: Sequence[dict[str, Any]],
    required_runs: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in aggregated:
        key = (str(row["algorithm"]), int(row["global_batch"]))
        groups.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for key, candidates in groups.items():
        valid = [
            row
            for row in candidates
            if int(row["runs"]) >= required_runs
            and float(row["stable_rate"]) == 1.0
            and row["val_loss_mean"] is not None
        ]
        if not valid:
            selected.append(
                {
                    "algorithm": key[0],
                    "global_batch": key[1],
                    "status": "no_fully_stable_configuration",
                }
            )
            continue
        best = min(valid, key=lambda row: float(row["val_loss_mean"]))
        selected.append({**best, "status": "ok"})
    selected.sort(key=lambda row: (str(row["algorithm"]), int(row["global_batch"])))
    return selected


def plot_results(root: Path, rows: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV summaries were still written")
        return

    colors = {"sgd": "C0", "polyak": "C1", "nesterov": "C2"}
    markers = {"sgd": "o", "polyak": "^", "nesterov": "s"}
    figure, axis = plt.subplots(figsize=(7.0, 5.0))

    for algorithm in ("sgd", "polyak", "nesterov"):
        chosen = [
            row
            for row in rows
            if row.get("algorithm") == algorithm
            and row.get("status") == "ok"
            and row.get("val_loss_mean") is not None
        ]
        chosen.sort(key=lambda row: int(row["global_batch"]))
        if not chosen:
            continue
        x = [int(row["global_batch"]) for row in chosen]
        y = [float(row["val_loss_mean"]) for row in chosen]
        yerr = [float(row.get("val_loss_std") or 0.0) for row in chosen]
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            label={"sgd": "SGD", "polyak": "Polyak", "nesterov": "Nesterov"}[algorithm],
            color=colors[algorithm],
            marker=markers[algorithm],
            linewidth=2.0,
            capsize=3,
        )
    axis.set_ylabel("Optimally tuned validation cross-entropy")
    output = root / "fixed_D_optimal_loss_vs_batch.png"

    axis.set_xscale("log", base=2)
    axis.set_xlabel("Global batch size")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output}")


def summarize_runs(root: Path, make_plot: bool, required_runs: int) -> None:
    summaries = load_summaries(root)
    if not summaries:
        raise SystemExit(f"no summary.json files found under {root}")
    write_csv(summaries, root / "all_runs.csv")
    aggregated = aggregate_configurations(summaries)
    write_csv(aggregated, root / "aggregated_configurations.csv")
    result_rows = select_best_by_batch(aggregated, required_runs=required_runs)
    write_csv(result_rows, root / "best_by_batch.csv")
    if make_plot:
        plot_results(root, result_rows)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("check-data", help="validate the ImageNet directory")
    data_parser.add_argument("--data-root", required=True)
    data_parser.add_argument("--crop-size", type=int, default=224)

    train_parser = subparsers.add_parser("train", help="run one training configuration")
    common_train_arguments(train_parser)

    sweep_parser = subparsers.add_parser("sweep", help="run the fixed-D hyperparameter search")
    sweep_parser.add_argument("--phase", choices=("coarse", "refine", "all"), default="all")
    sweep_parser.add_argument("--data-root", default="")
    sweep_parser.add_argument("--runs-root", required=True)
    sweep_parser.add_argument("--algorithms", default="sgd,polyak,nesterov")
    sweep_parser.add_argument("--batches", default="256,512,1024,2048,4096,8192,16384")
    sweep_parser.add_argument("--lrs", default="0.025,0.05,0.1,0.2,0.4")
    sweep_parser.add_argument(
        "--lr-grid-mode",
        choices=("absolute", "linear", "hybrid"),
        default="hybrid",
    )
    sweep_parser.add_argument("--reference-batch", type=int, default=256)
    sweep_parser.add_argument("--momentums", default="0.5,0.75,0.9,0.95,0.98,0.99")
    sweep_parser.add_argument("--search-seed", type=int, default=0)
    sweep_parser.add_argument("--seeds", default="0,1,2")
    sweep_parser.add_argument("--top-k", type=int, default=3)
    sweep_parser.add_argument("--nproc-per-node", type=int, default=1)
    sweep_parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet18")
    sweep_parser.add_argument("--norm", choices=("group", "batch"), default="group")
    sweep_parser.add_argument("--sync-batchnorm", action="store_true")
    sweep_parser.add_argument("--weight-decay", type=float, default=0.0)
    sweep_parser.add_argument("--max-per-device-batch", type=int, default=128)
    sweep_parser.add_argument("--sample-budget", type=int, default=0)
    sweep_parser.add_argument("--epochs-equivalent", type=float, default=30.0)
    sweep_parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    sweep_parser.add_argument("--warmup-epochs-equivalent", type=float, default=0.0)
    sweep_parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    sweep_parser.add_argument("--label-smoothing", type=float, default=0.0)
    sweep_parser.add_argument("--workers", type=int, default=8)
    sweep_parser.add_argument("--log-every-updates", type=int, default=100)
    sweep_parser.add_argument("--final-window-updates", type=int, default=100)
    sweep_parser.add_argument("--divergence-loss", type=float, default=20.0)
    sweep_parser.add_argument("--divergence-grad-norm", type=float, default=1e6)
    sweep_parser.add_argument("--divergence-patience", type=int, default=5)
    sweep_parser.add_argument("--deterministic", action="store_true")
    sweep_parser.add_argument("--fake-data", action="store_true")
    sweep_parser.add_argument("--fake-train-size", type=int, default=4096)
    sweep_parser.add_argument("--fake-val-size", type=int, default=512)
    sweep_parser.add_argument("--num-classes", type=int, default=1000)
    sweep_parser.add_argument("--dry-run", action="store_true")
    sweep_parser.add_argument("--continue-on-error", action="store_true")
    sweep_parser.add_argument("--no-plot", action="store_true")

    summary_parser = subparsers.add_parser("summarize", help="summarize completed sweep runs")
    summary_parser.add_argument("--runs-root", required=True)
    summary_parser.add_argument("--required-runs", type=int, default=3)
    summary_parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "check-data":
        check_data(args)
    elif args.command == "train":
        train_one(args)
    elif args.command == "sweep":
        run_sweep(args)
    elif args.command == "summarize":
        summarize_runs(
            Path(args.runs_root).expanduser().resolve(),
            make_plot=not args.no_plot,
            required_runs=args.required_runs,
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
