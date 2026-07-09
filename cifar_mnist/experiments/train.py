"""Single training run: given a model + optimizer + loaders, train and record
per-step / per-epoch curves (train loss, test loss, test accuracy, and the
cumulative number of processed samples B*K used as the "compute budget D" axis
in experiment B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from .models import build_model, count_params
from .optimizers import build_optimizer


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class History:
    """Recorded curves for one run."""
    step: List[int] = field(default_factory=list)          # optimizer steps K
    samples: List[int] = field(default_factory=list)       # cumulative B*K (= D)
    train_loss: List[float] = field(default_factory=list)  # smoothed train loss
    eval_step: List[int] = field(default_factory=list)
    eval_samples: List[int] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def final(self) -> dict:
        return {
            "test_loss": self.test_loss[-1] if self.test_loss else float("nan"),
            "test_acc": self.test_acc[-1] if self.test_acc else float("nan"),
            "train_loss": self.train_loss[-1] if self.train_loss else float("nan"),
            "samples": self.samples[-1] if self.samples else 0,
            **self.meta,
        }


@torch.no_grad()
def evaluate(model, loader, device, criterion, optimizer=None):
    """Compute test loss + accuracy. For Nesterov, temporarily convert the
    lookahead params to the true iterate theta before evaluating."""
    synced = False
    if optimizer is not None and hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead(subtract=True)
        synced = True

    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    model.train()

    if synced:
        optimizer.sync_lookahead(subtract=False)  # restore lookahead to resume
    return loss_sum / total, correct / total


def train(model_name: str, optimizer_mode: str, train_loader, test_loader,
          *, lr: float, rho: float = 0.9, weight_decay: float = 0.0,
          epochs: int = 20, max_steps: Optional[int] = None,
          device=None, seed: int = 0, eval_every_epochs: int = 1,
          log_every_steps: int = 50, model_kwargs: Optional[dict] = None,
          verbose: bool = True) -> History:
    """Train one model with one optimizer and return its History.

    ``max_steps`` caps the number of optimizer steps (used by experiment B to fix
    the compute budget D = B * max_steps regardless of epoch boundaries).
    """
    torch.manual_seed(seed)
    device = device or get_device()
    model = build_model(model_name, **(model_kwargs or {})).to(device)
    # Materialise LazyLinear (SmallCNN) with one dummy forward before the optimizer
    # captures parameters.
    xb, _ = next(iter(train_loader))
    with torch.no_grad():
        model(xb.to(device))

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model.parameters(), mode=optimizer_mode,
                                lr=lr, rho=rho, weight_decay=weight_decay)

    hist = History()
    hist.meta = {
        "model": model_name, "optimizer": optimizer_mode, "lr": lr, "rho": rho,
        "epochs": epochs, "seed": seed, "params": count_params(model),
        "batch_size": train_loader.batch_size,
    }

    batch_size = train_loader.batch_size
    step = 0
    running = None
    stop = False
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            step += 1
            lv = loss.item()
            running = lv if running is None else 0.98 * running + 0.02 * lv
            if step % log_every_steps == 0:
                hist.step.append(step)
                hist.samples.append(step * batch_size)
                hist.train_loss.append(running)

            if max_steps is not None and step >= max_steps:
                stop = True
                break
        if stop:
            break

        if (epoch + 1) % eval_every_epochs == 0 or epoch == epochs - 1:
            tl, ta = evaluate(model, test_loader, device, criterion, optimizer)
            hist.eval_step.append(step)
            hist.eval_samples.append(step * batch_size)
            hist.test_loss.append(tl)
            hist.test_acc.append(ta)
            if verbose:
                print(f"[{model_name}/{optimizer_mode}] epoch {epoch+1:>3d} "
                      f"step {step:>6d}  train {running:.4f}  "
                      f"test_loss {tl:.4f}  test_acc {ta:.4f}")

    # Always record a final evaluation (covers the max_steps early-stop path).
    tl, ta = evaluate(model, test_loader, device, criterion, optimizer)
    if not hist.eval_step or hist.eval_step[-1] != step:
        hist.eval_step.append(step)
        hist.eval_samples.append(step * batch_size)
        hist.test_loss.append(tl)
        hist.test_acc.append(ta)
    return hist
