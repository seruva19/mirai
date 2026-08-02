"""LoRA-FA frozen-A policy and gradient projection contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Protocol, runtime_checkable

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@runtime_checkable
class LoRAFAHost(Protocol):
    """Explicit module contract for LoRA-FA configuration."""

    def set_lora_fa_enabled(self, enabled: bool) -> None: ...


@dataclass(frozen=True)
class LoRAFAReport:
    host_modules: tuple[str, ...]
    frozen_a_parameters: int


class LoRAFAGradientProjector:
    """Apply the paper's fixed-A correction to one B-factor gradient."""

    def __init__(self, lora_a: Any, scale: Callable[[], float]) -> None:
        self._lora_a = lora_a
        self._scale = scale
        self._cache_signature: tuple[Any, ...] | None = None
        self._gram_inverse: Any | None = None

    def _inverse(self, grad: Any) -> Any:
        a = self._lora_a.detach()
        signature = (
            a.device,
            a.dtype,
            tuple(int(dim) for dim in a.shape),
            int(a._version),
        )
        if self._gram_inverse is None or signature != self._cache_signature:
            a32 = a.to(device=grad.device, dtype=torch.float32)
            gram = torch.matmul(a32, a32.transpose(-1, -2))
            rank = int(a32.shape[-2])
            identity = torch.eye(rank, device=gram.device, dtype=gram.dtype)
            self._gram_inverse = torch.linalg.pinv(gram + identity * 1e-8)
            self._cache_signature = signature
        return self._gram_inverse.to(device=grad.device, dtype=torch.float32)

    def __call__(self, grad: Any) -> Any:
        scale = float(self._scale())
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("LoRA-FA requires a finite positive effective scale.")
        corrected = torch.matmul(grad.float(), self._inverse(grad)) / (scale * scale)
        return corrected.to(dtype=grad.dtype)


def configure_lora_fa_factors(
    *,
    lora_a: Any,
    lora_b: Any,
    scale: Callable[[], float],
    enabled: bool,
    hook: Any | None,
) -> Any | None:
    """Freeze/unfreeze A and install/remove the B-gradient correction hook."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("LoRA-FA requires torch.")
    if enabled:
        lora_a.requires_grad_(False)
        lora_b.requires_grad_(True)
        if hook is None:
            hook = lora_b.register_hook(LoRAFAGradientProjector(lora_a, scale))
        return hook
    if hook is not None:
        hook.remove()
    lora_a.requires_grad_(True)
    lora_b.requires_grad_(True)
    return None


def apply_lora_fa(root: Any, *, enabled: bool = True) -> LoRAFAReport:
    """Configure every explicit LoRA-FA host below ``root``."""
    if nn is None:  # pragma: no cover
        raise RuntimeError("LoRA-FA requires torch.")
    hosts: list[str] = []
    frozen = 0
    for name, module in root.named_modules():
        if not isinstance(module, LoRAFAHost):
            continue
        module.set_lora_fa_enabled(bool(enabled))
        hosts.append(str(name))
        frozen += sum(
            int(parameter.numel())
            for parameter_name, parameter in module.named_parameters(recurse=False)
            if parameter_name in {"lora_a", "cond_a"} and not parameter.requires_grad
        )
    if enabled and not hosts:
        raise ValueError("LoRA-FA requested but no compatible LoRA hosts were found.")
    return LoRAFAReport(
        host_modules=tuple(hosts),
        frozen_a_parameters=int(frozen),
    )


__all__ = [
    "LoRAFAHost",
    "LoRAFAGradientProjector",
    "LoRAFAReport",
    "apply_lora_fa",
    "configure_lora_fa_factors",
]
