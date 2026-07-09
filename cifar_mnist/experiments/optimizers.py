"""SGD, Heavy Ball, and Nesterov, hand-written to match the update rules used in
`linear_regression_scaling.ipynb` (and the theory in `draft.md`) *exactly*.

The notebook, working in delta_t = theta_t - theta*, defines (with m_t := theta_t - theta_{t-1}):

    SGD        theta_{t+1} = theta_t - eta * g(theta_t)
    HeavyBall  theta_{t+1} = theta_t - eta * g(theta_t)            + rho * m_t
    Nesterov   theta_{t+1} = theta_t - eta * g(theta_t + rho*m_t)  + rho * m_t

We deliberately do NOT use torch.optim.SGD(momentum=..., nesterov=...): PyTorch's
built-in Nesterov uses the Sutskever "gradient at current point" reformulation,
which is algebraically related but has a different learning-rate convention. To
keep the empirical study aligned with the theory, we reimplement the three rules
directly.

Implementation notes
--------------------
* HeavyBall / SGD keep a velocity buffer ``m`` with the recursion
  ``m <- rho*m - eta*g`` and update ``p <- p + m`` (m_{t+1} = theta_{t+1}-theta_t).
* Nesterov needs the gradient at the *extrapolated* point y_t = theta_t + rho*m_t.
  We keep the model parameters AT that lookahead point y_t -- so the gradient the
  training loop computes (at the current params) is already g(y_t), exactly what
  the rule needs. The velocity recursion is identical, and the parameter update
  is ``p(=y) <- y - eta*g + rho*m_{t+1}``.
  Consequence: after the last Nesterov step the parameters hold y_T (the
  lookahead), which differs from theta_T by rho*m_T. Near convergence m_T -> 0 so
  this is negligible; call ``sync_lookahead(subtract=True)`` if you need theta_T
  exactly for evaluation.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


_MODES = ("sgd", "heavy_ball", "nesterov")


class Momentum(Optimizer):
    """Unified SGD / Heavy Ball / Nesterov optimizer.

    Args:
        params: iterable of parameters (or param groups).
        lr:     step size eta.
        rho:    momentum coefficient (ignored / forced to 0 for mode='sgd').
        mode:   one of 'sgd', 'heavy_ball', 'nesterov'.
        weight_decay: optional L2 penalty added to the gradient (default 0).
    """

    def __init__(self, params, lr: float, rho: float = 0.9,
                 mode: str = "heavy_ball", weight_decay: float = 0.0):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0,1), got {rho}")
        if mode == "sgd":
            rho = 0.0
        defaults = dict(lr=lr, rho=rho, mode=mode, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            rho = group["rho"]
            mode = group["mode"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]

                # Velocity recursion is identical for all three modes:
                #   m_{t+1} = rho * m_t - eta * g
                m.mul_(rho).add_(g, alpha=-lr)

                if mode == "nesterov":
                    # params hold the lookahead y_t; g == grad(y_t) already.
                    # y_{t+1} = (y_t - eta*g) + rho * m_{t+1}
                    p.add_(g, alpha=-lr).add_(m, alpha=rho)
                else:
                    # sgd (rho=0) and heavy_ball: theta_{t+1} = theta_t + m_{t+1}
                    p.add_(m)

        return loss

    @torch.no_grad()
    def sync_lookahead(self, subtract: bool = True):
        """For mode='nesterov', convert params between the lookahead point y and
        the true iterate theta (theta = y - rho*m). No-op for other modes.

        subtract=True : y -> theta   (call before evaluating theta_T)
        subtract=False: theta -> y   (undo, to resume training)
        """
        for group in self.param_groups:
            if group["mode"] != "nesterov":
                continue
            rho = group["rho"]
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "m" not in state:
                    continue
                p.add_(state["m"], alpha=(-rho if subtract else rho))


def build_optimizer(params, mode: str, lr: float, rho: float = 0.9,
                    weight_decay: float = 0.0) -> Momentum:
    return Momentum(params, lr=lr, rho=rho, mode=mode, weight_decay=weight_decay)
