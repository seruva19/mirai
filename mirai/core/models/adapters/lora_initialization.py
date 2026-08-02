"""Registry and exact math for LoRA factor initialization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from mirai.core.registry import Registry

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


LoRAAInitializer = Callable[[Any], None]
LoRAInitializerRegistry: Registry[LoRAAInitializer] = Registry("LoRA initializer")


@dataclass(frozen=True)
class QuantizationErrorInitializationReport:
    """Frobenius-error evidence for one initialized weight matrix."""

    rank: int
    error_before: float
    error_after: float

    @property
    def relative_error_reduction(self) -> float:
        if self.error_before == 0.0:
            return 0.0
        return 1.0 - (self.error_after / self.error_before)


def initialize_from_principal_weight(
    *,
    lora_a: Any,
    lora_b: Any,
    base_weight: Any,
    scale: float,
) -> None:
    """Move the principal rank-r weight subspace into trainable LoRA factors."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PiSSA initialization requires torch.")
    resolved_scale = float(scale)
    if not math.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("PiSSA requires a finite positive LoRA scale.")
    grouped = base_weight.ndim == 3
    weights = base_weight if grouped else base_weight.unsqueeze(0)
    factors_a = lora_a if grouped else lora_a.unsqueeze(0)
    factors_b = lora_b if grouped else lora_b.unsqueeze(0)
    if weights.ndim != 3 or factors_a.ndim != 3 or factors_b.ndim != 3:
        raise ValueError("PiSSA supports matrix or grouped-expert matrix weights.")
    with torch.no_grad():
        for index in range(int(weights.shape[0])):
            weight = weights[index].detach().float()
            rank = min(
                int(factors_a.shape[1]),
                int(weight.shape[0]),
                int(weight.shape[1]),
            )
            u, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
            roots = (singular_values[:rank] / resolved_scale).sqrt()
            a = roots.unsqueeze(1) * vh[:rank]
            b = u[:, :rank] * roots.unsqueeze(0)
            factors_a[index].zero_()
            factors_b[index].zero_()
            factors_a[index, :rank].copy_(
                a.to(device=factors_a.device, dtype=factors_a.dtype)
            )
            factors_b[index, :, :rank].copy_(
                b.to(device=factors_b.device, dtype=factors_b.dtype)
            )
            residual_delta = (b @ a) * resolved_scale
            weights[index].sub_(
                residual_delta.to(device=weights.device, dtype=weights.dtype)
            )


def initialize_from_gradient_approximation(
    *,
    lora_a: Any,
    lora_b: Any,
    base_weight: Any,
    gradient: Any,
    scale: float,
    stable_gamma: float,
) -> None:
    """LoRA-GA A2rBr initialization from an averaged full-weight gradient."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("LoRA-GA initialization requires torch.")
    if base_weight.ndim != 2 or gradient.ndim != 2:
        raise ValueError("LoRA-GA requires one dense weight matrix.")
    rank = int(lora_a.shape[0])
    if min(gradient.shape) < 2 * rank:
        raise ValueError("LoRA-GA requires min(weight shape) >= 2 * adapter rank.")
    resolved_scale = float(scale)
    gamma = float(stable_gamma)
    if resolved_scale <= 0.0 or gamma <= 0.0:
        raise ValueError("LoRA-GA scale and stable_gamma must be positive.")
    grad = gradient.detach().to(device=base_weight.device, dtype=torch.float32)
    u, singular_values, vh = torch.linalg.svd(grad, full_matrices=False)
    a = vh[rank : 2 * rank]
    b = u[:, :rank] * singular_values[:rank].unsqueeze(0)
    b = b * ((int(base_weight.shape[0]) ** 0.25) / math.sqrt(gamma))
    with torch.no_grad():
        lora_a.copy_(a.to(device=lora_a.device, dtype=lora_a.dtype))
        lora_b.copy_(b.to(device=lora_b.device, dtype=lora_b.dtype))
        delta = lora_b.detach().float() @ lora_a.detach().float()
        base_weight.sub_(
            (delta * resolved_scale).to(
                device=base_weight.device, dtype=base_weight.dtype
            )
        )


def register_lora_initializer(
    name: str,
) -> Callable[[LoRAAInitializer], LoRAAInitializer]:
    return LoRAInitializerRegistry.decorator(name)


def validate_lora_initializer(name: str) -> str:
    value = str(name).strip().lower()
    LoRAInitializerRegistry.get(value)
    return value


def initialize_lora_a(lora_a: Any, name: str) -> None:
    LoRAInitializerRegistry.get(name)(lora_a)


def initializer_requires_quantization_error(name: str) -> bool:
    return validate_lora_initializer(name) == "loftq"


def initializer_requires_activation_calibration(name: str) -> bool:
    return validate_lora_initializer(name) == "eva"


def initialize_from_quantization_error(
    *,
    lora_a: Any,
    lora_b: Any,
    reference_weight: Any,
    quantized_weight: Any,
    scale: float,
) -> QuantizationErrorInitializationReport:
    """Initialize factors from the best rank-r approximation of ``W - Q``.

    This is the one-iteration LoftQ variant: the quantized base is fixed and
    only LoRA factors change. Inputs are one 2D matrix, allowing grouped expert
    callers to process and release one physical expert at a time.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("LoftQ initialization requires torch.")
    if reference_weight.ndim != 2 or quantized_weight.ndim != 2:
        raise ValueError("LoftQ initialization requires two 2D weight matrices.")
    if tuple(reference_weight.shape) != tuple(quantized_weight.shape):
        raise ValueError("LoftQ reference and quantized weights must have equal shapes.")
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("LoftQ factor targets must be 2D matrices.")
    rank = int(lora_a.shape[0])
    expected_a = (rank, int(reference_weight.shape[1]))
    expected_b = (int(reference_weight.shape[0]), rank)
    if tuple(lora_a.shape) != expected_a or tuple(lora_b.shape) != expected_b:
        raise ValueError(
            f"LoftQ factor shapes must be {expected_a} and {expected_b}."
        )
    resolved_scale = float(scale)
    if not math.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("LoftQ initialization requires a finite positive LoRA scale.")

    device = reference_weight.device
    error = reference_weight.detach().to(device=device, dtype=torch.float32) - (
        quantized_weight.detach().to(device=device, dtype=torch.float32)
    )
    error_before = float(torch.linalg.vector_norm(error).item())
    effective_rank = min(rank, int(error.shape[0]), int(error.shape[1]))
    u, singular_values, vh = torch.linalg.svd(error, full_matrices=False)
    roots = (singular_values[:effective_rank] / resolved_scale).clamp_min(0).sqrt()
    resolved_b = u[:, :effective_rank] * roots.unsqueeze(0)
    resolved_a = roots.unsqueeze(1) * vh[:effective_rank, :]

    with torch.no_grad():
        lora_a.zero_()
        lora_b.zero_()
        lora_a[:effective_rank].copy_(
            resolved_a.to(device=lora_a.device, dtype=lora_a.dtype)
        )
        lora_b[:, :effective_rank].copy_(
            resolved_b.to(device=lora_b.device, dtype=lora_b.dtype)
        )
    reconstructed = resolved_b @ resolved_a * resolved_scale
    error_after = float(torch.linalg.vector_norm(error - reconstructed).item())
    if error_after > error_before + max(1e-6, error_before * 1e-6):
        raise RuntimeError("LoftQ initialization increased quantization error.")
    return QuantizationErrorInitializationReport(
        rank=effective_rank,
        error_before=error_before,
        error_after=error_after,
    )


@register_lora_initializer("kaiming")
def _initialize_kaiming(lora_a: Any) -> None:
    if nn is None:  # pragma: no cover
        raise RuntimeError("LoRA initialization requires torch.")
    nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))


@register_lora_initializer("orthogonal")
def _initialize_orthogonal(lora_a: Any) -> None:
    if nn is None:  # pragma: no cover
        raise RuntimeError("LoRA initialization requires torch.")
    if lora_a.dim() == 3:
        for expert_idx in range(int(lora_a.shape[0])):
            nn.init.orthogonal_(lora_a[expert_idx])
    else:
        nn.init.orthogonal_(lora_a)


@register_lora_initializer("loftq")
def _initialize_loftq_staging(lora_a: Any) -> None:
    # The graph is assembled before frozen-weight quantization. Keep the
    # established no-op factors until W-Q becomes available at that boundary.
    _initialize_kaiming(lora_a)


@register_lora_initializer("eva")
def _initialize_eva_staging(lora_a: Any) -> None:
    # EVA is data-dependent and runs immediately before the first optimizer
    # step. Keep a valid zero-delta adapter while the graph is assembled.
    _initialize_kaiming(lora_a)


@register_lora_initializer("pissa")
def _initialize_pissa_staging(lora_a: Any) -> None:
    # Weight-dependent initialization runs after both factors and the residual
    # base are available in the adapter host.
    _initialize_kaiming(lora_a)


@register_lora_initializer("lora_ga")
def _initialize_lora_ga_staging(lora_a: Any) -> None:
    # Gradient calibration replaces both factors before the first optimizer step.
    _initialize_kaiming(lora_a)


@register_lora_initializer("gora")
def _initialize_gora_staging(lora_a: Any) -> None:
    # GoRA reallocates ranks and initializes both factors before optimizer
    # construction. The staging graph remains a valid zero-delta LoRA graph.
    _initialize_kaiming(lora_a)


__all__ = [
    "LoRAInitializerRegistry",
    "QuantizationErrorInitializationReport",
    "initialize_from_quantization_error",
    "initialize_from_principal_weight",
    "initialize_from_gradient_approximation",
    "initialize_lora_a",
    "initializer_requires_activation_calibration",
    "initializer_requires_quantization_error",
    "register_lora_initializer",
    "validate_lora_initializer",
]
