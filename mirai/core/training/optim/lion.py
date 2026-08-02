"""Compact in-repo Lion optimizer (~30 lines of math).

Used only when pytorch_optimizer is unavailable.
Reference: https://arxiv.org/abs/2302.06675
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch.optim import Optimizer


class LionFallback(Optimizer):
    """Lion optimizer — sign-based update rule."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | list[dict],
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults: dict[str, Any] = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            wd = float(group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                # update: p ← p - lr * (sign(β1*m + (1-β1)*grad) + wd*p)
                update = (beta1 * m + (1.0 - beta1) * grad).sign_()
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)
                # momentum decay
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss
