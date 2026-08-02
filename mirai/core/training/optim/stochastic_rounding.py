"""BF16 AdamW updates with unbiased stochastic rounding."""

from __future__ import annotations

import math
from typing import Any, Iterable

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Stochastic-rounding AdamW requires torch: {exc}")


_BF16_DISCARDED_BITS = 16
_BF16_ROUNDING_LEVELS = 1 << _BF16_DISCARDED_BITS
_BF16_MANTISSA_MASK = -_BF16_ROUNDING_LEVELS
DEFAULT_UPDATE_CHUNK_SIZE = 1 << 20


def stochastic_round_bfloat16(values: Any) -> Any:
    """Unbiasedly round FP32 values to adjacent BF16 representable values.

    Uniform dither is added to the 16 FP32 mantissa bits discarded by a BF16
    cast, then those bits are truncated. Non-finite inputs are preserved.
    """

    if not isinstance(values, torch.Tensor):
        raise TypeError("stochastic_round_bfloat16 expects a torch.Tensor.")
    source = values.to(dtype=torch.float32).contiguous()
    random_bits = torch.randint(
        0,
        _BF16_ROUNDING_LEVELS,
        source.shape,
        device=source.device,
        dtype=torch.int32,
    )
    source_bits = source.view(torch.int32)
    rounded_bits = torch.bitwise_and(
        source_bits + random_bits,
        _BF16_MANTISSA_MASK,
    )
    rounded = rounded_bits.view(torch.float32).to(dtype=torch.bfloat16)
    return torch.where(
        torch.isfinite(source),
        rounded,
        source.to(dtype=torch.bfloat16),
    )


class StochasticRoundingAdamW(torch.optim.Optimizer):
    """AdamW with BF16 moments and stochastic model-update rounding.

    BF16 parameters and optimizer moments stay in BF16. Only one bounded
    parameter chunk is promoted to FP32 while its update is evaluated.
    Parameters in other floating-point dtypes retain ordinary AdamW math.
    """

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_chunk_size: int = DEFAULT_UPDATE_CHUNK_SIZE,
    ) -> None:
        if float(lr) < 0.0:
            raise ValueError("StochasticRoundingAdamW lr must be >= 0.")
        if not 0.0 <= float(betas[0]) < 1.0:
            raise ValueError("StochasticRoundingAdamW beta1 must be in [0, 1).")
        if not 0.0 <= float(betas[1]) < 1.0:
            raise ValueError("StochasticRoundingAdamW beta2 must be in [0, 1).")
        if float(eps) < 0.0:
            raise ValueError("StochasticRoundingAdamW eps must be >= 0.")
        if float(weight_decay) < 0.0:
            raise ValueError(
                "StochasticRoundingAdamW weight_decay must be >= 0."
            )
        if int(update_chunk_size) <= 0:
            raise ValueError(
                "StochasticRoundingAdamW update_chunk_size must be > 0."
            )
        defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
        }
        super().__init__(params, defaults)
        self.update_chunk_size = int(update_chunk_size)

    @staticmethod
    def _initialize_state(param: Any, state: dict[str, Any]) -> None:
        state["step"] = torch.tensor(0, dtype=torch.int64)
        state["exp_avg"] = torch.zeros_like(
            param,
            memory_format=torch.preserve_format,
        )
        state["exp_avg_sq"] = torch.zeros_like(
            param,
            memory_format=torch.preserve_format,
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "StochasticRoundingAdamW does not support sparse gradients."
                    )
                state = self.state[param]
                if not state:
                    self._initialize_state(param, state)
                state["step"].add_(1)
                step = int(state["step"].item())
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1.0 - beta2,
                )
                self._apply_parameter_update(
                    param=param,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    step=step,
                    lr=float(group["lr"]),
                    weight_decay=float(group["weight_decay"]),
                    beta1=float(beta1),
                    beta2=float(beta2),
                    eps=float(group["eps"]),
                )
        return loss

    def _apply_parameter_update(
        self,
        *,
        param: Any,
        exp_avg: Any,
        exp_avg_sq: Any,
        step: int,
        lr: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> None:
        bias_correction1 = 1.0 - beta1**step
        bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
        step_size = lr / bias_correction1
        parameter_flat = param.reshape(-1)
        first_moment_flat = exp_avg.reshape(-1)
        second_moment_flat = exp_avg_sq.reshape(-1)
        for start in range(0, int(parameter_flat.numel()), self.update_chunk_size):
            end = min(start + self.update_chunk_size, int(parameter_flat.numel()))
            parameter_chunk = parameter_flat[start:end]
            first_moment = first_moment_flat[start:end]
            second_moment = second_moment_flat[start:end]
            if param.dtype == torch.bfloat16:
                updated = parameter_chunk.float()
                if weight_decay != 0.0:
                    updated.mul_(1.0 - lr * weight_decay)
                denominator = (
                    second_moment.float().sqrt_().div_(bias_correction2_sqrt)
                )
                denominator.add_(eps)
                updated.addcdiv_(
                    first_moment.float(),
                    denominator,
                    value=-step_size,
                )
                parameter_chunk.copy_(stochastic_round_bfloat16(updated))
            else:
                if weight_decay != 0.0:
                    parameter_chunk.mul_(1.0 - lr * weight_decay)
                denominator = (
                    second_moment.sqrt().div_(bias_correction2_sqrt).add_(eps)
                )
                parameter_chunk.addcdiv_(
                    first_moment,
                    denominator,
                    value=-step_size,
                )


__all__ = [
    "DEFAULT_UPDATE_CHUNK_SIZE",
    "StochasticRoundingAdamW",
    "stochastic_round_bfloat16",
]
