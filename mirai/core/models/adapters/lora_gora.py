"""Exact single-worker GoRA rank allocation and factor initialization.

The implementation follows Equations 5--10 and Algorithm 1 of GoRA
(arXiv:2502.12171). Mirai stores LoRA factors as ``B @ A`` while the paper
names those same geometric factors ``A @ B``; this module maps by shape:
``lora_b`` is the paper's output-side random factor and ``lora_a`` is the
input-side gradient-compressed factor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GoRAAllocationPlan:
    """Deterministic ranks and budget diagnostics for one calibration pass."""

    ranks: Mapping[str, int]
    importance: Mapping[str, float]
    smoothed_reference_budget: float
    smoothed_allocated_budget: float
    actual_reference_parameters: int
    actual_allocated_parameters: int
    fingerprint: str


@dataclass(frozen=True)
class GoRAInitializationDiagnostic:
    """Initial adapter alignment with the negative calibration gradient."""

    cosine_similarity: float
    relative_norm: float


def gora_sensitivity_importance(weight: Any, gradient: Any) -> float:
    """Return ``sum_e mean(abs(W_e * G_e))`` from GoRA Equation 5.

    A regular matrix has one group member. A grouped expert tensor shares one
    executable rank in Mirai, so expert importances are summed and the allocator
    charges the corresponding number of matrix budgets.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("GoRA importance requires torch.")
    if tuple(weight.shape) != tuple(gradient.shape):
        raise ValueError("GoRA weight and gradient shapes must match.")
    if weight.ndim not in {2, 3}:
        raise ValueError("GoRA targets must be matrices or grouped matrices.")
    values = (
        weight.detach().float().cpu() * gradient.detach().float().cpu()
    ).abs()
    if values.ndim == 2:
        result = float(values.mean().item())
    else:
        result = float(values.flatten(1).mean(dim=1).sum().item())
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("GoRA importance must be finite and non-negative.")
    return result


def allocate_gora_ranks(
    shapes: Mapping[str, tuple[int, int, int]],
    importance: Mapping[str, float],
    *,
    reference_rank: int,
    minimum_rank: int,
    maximum_rank: int,
) -> GoRAAllocationPlan:
    """Allocate ranks using GoRA's smoothed parameter budget.

    Each geometry is ``(group_size, out_features, in_features)``. Grouped
    experts retain one shared rank so fused expert execution remains possible;
    their importance and budget are weighted by ``group_size``.
    """

    names = tuple(sorted(str(name) for name in shapes))
    if not names or set(names) != {str(name) for name in importance}:
        raise ValueError("GoRA shapes and importance must name the same targets.")
    ref = int(reference_rank)
    floor = int(minimum_rank)
    ceiling = int(maximum_rank)
    if ref <= 0 or floor <= 0 or ceiling < floor:
        raise ValueError("GoRA ranks require reference > 0 and 0 < min <= max.")

    normalized_shapes: dict[str, tuple[int, int, int]] = {}
    normalized_importance: dict[str, float] = {}
    smoothed_reference_budget = 0.0
    actual_reference_parameters = 0
    for name in names:
        group_size, out_features, in_features = (
            int(value) for value in shapes[name]
        )
        if min(group_size, out_features, in_features) <= 0:
            raise ValueError(f"GoRA target {name!r} has an invalid geometry.")
        if floor > min(out_features, in_features):
            raise ValueError(
                f"GoRA minimum rank {floor} exceeds target {name!r} dimensions."
            )
        score = float(importance[name])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(
                f"GoRA importance for target {name!r} must be finite and non-negative."
            )
        normalized_shapes[name] = (group_size, out_features, in_features)
        normalized_importance[name] = score
        smoothed_reference_budget += (
            group_size * math.sqrt(out_features + in_features) * ref
        )
        actual_reference_parameters += (
            group_size * (out_features + in_features) * ref
        )

    total_importance = math.fsum(normalized_importance.values())
    if total_importance <= 0.0:
        raise ValueError("GoRA calibration produced zero total target importance.")

    ranks: dict[str, int] = {}
    smoothed_allocated_budget = 0.0
    actual_allocated_parameters = 0
    for name in names:
        group_size, out_features, in_features = normalized_shapes[name]
        unit_cost = group_size * math.sqrt(out_features + in_features)
        advantage = normalized_importance[name] / total_importance
        raw_rank = smoothed_reference_budget * advantage / unit_cost
        # The paper specifies nearest-integer rounding. Python's deterministic
        # ties-to-even rule is used for the otherwise unspecified half case.
        resolved = int(round(raw_rank))
        resolved = max(
            floor,
            min(ceiling, out_features, in_features, resolved),
        )
        ranks[name] = resolved
        smoothed_allocated_budget += unit_cost * resolved
        actual_allocated_parameters += (
            group_size * (out_features + in_features) * resolved
        )

    canonical = {
        "ranks": dict(sorted(ranks.items())),
        "importance": {
            name: format(normalized_importance[name], ".17g") for name in names
        },
        "reference_rank": ref,
        "minimum_rank": floor,
        "maximum_rank": ceiling,
    }
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return GoRAAllocationPlan(
        ranks=ranks,
        importance=normalized_importance,
        smoothed_reference_budget=smoothed_reference_budget,
        smoothed_allocated_budget=smoothed_allocated_budget,
        actual_reference_parameters=actual_reference_parameters,
        actual_allocated_parameters=actual_allocated_parameters,
        fingerprint=fingerprint,
    )


def resize_lora_module(module: Any, rank: int) -> None:
    """Replace LoRA factors before optimizer construction or artifact loading."""

    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("LoRA factor resizing requires torch.")
    resolved_rank = int(rank)
    if resolved_rank <= 0:
        raise ValueError("LoRA rank must be positive.")
    lora_a = getattr(module, "lora_a", None)
    lora_b = getattr(module, "lora_b", None)
    if not isinstance(lora_a, nn.Parameter) or not isinstance(lora_b, nn.Parameter):
        raise ValueError("LoRA target does not expose trainable factor parameters.")
    if lora_a.ndim not in {2, 3} or lora_b.ndim != lora_a.ndim:
        raise ValueError("LoRA resizing supports matrix or grouped-matrix factors.")
    if getattr(module, "_lora_fa_hook", None) is not None:
        raise ValueError("LoRA resizing cannot replace an active LoRA-FA projection.")

    requires_grad_a = bool(lora_a.requires_grad)
    requires_grad_b = bool(lora_b.requires_grad)
    if lora_a.ndim == 2:
        new_a_shape = (resolved_rank, int(lora_a.shape[-1]))
        new_b_shape = (int(lora_b.shape[-2]), resolved_rank)
    else:
        groups = int(lora_a.shape[0])
        if groups != int(lora_b.shape[0]):
            raise ValueError("Grouped LoRA factors must have equal group counts.")
        new_a_shape = (groups, resolved_rank, int(lora_a.shape[-1]))
        new_b_shape = (groups, int(lora_b.shape[-2]), resolved_rank)
    module.lora_a = nn.Parameter(
        torch.zeros(new_a_shape, device=lora_a.device, dtype=lora_a.dtype),
        requires_grad=requires_grad_a,
    )
    module.lora_b = nn.Parameter(
        torch.zeros(new_b_shape, device=lora_b.device, dtype=lora_b.dtype),
        requires_grad=requires_grad_b,
    )
    module.rank = resolved_rank
    if hasattr(module, "_timestep_mask_per_sample"):
        module._timestep_mask_per_sample = None
    if hasattr(module, "_timestep_mask_uniform"):
        module._timestep_mask_uniform = None


def initialize_gora_module(
    module: Any,
    gradient: Any,
    *,
    rank: int,
    stable_gamma: float,
    seed: int,
) -> GoRAInitializationDiagnostic:
    """Resize and initialize one Mirai LoRA host from an averaged gradient."""

    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("GoRA initialization requires torch.")
    if str(getattr(module, "_lora_init", "")).strip().lower() != "gora":
        raise ValueError("GoRA initialization requires a GoRA staging adapter.")
    if not bool(getattr(module, "use_rslora", False)):
        raise ValueError("GoRA initialization requires rsLoRA scaling.")
    gamma = float(stable_gamma)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("GoRA stable_gamma must be finite and positive.")
    alpha_tensor = getattr(module, "lora_alpha", None)
    if not isinstance(alpha_tensor, torch.Tensor) or alpha_tensor.numel() != 1:
        raise ValueError("GoRA target must expose a scalar lora_alpha.")
    alpha = float(alpha_tensor.detach().float().item())
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("GoRA alpha must be finite and positive.")

    grad = gradient.detach().float()
    if grad.ndim not in {2, 3}:
        raise ValueError("GoRA gradients must be matrices or grouped matrices.")
    resize_lora_module(module, int(rank))
    groups = 1 if grad.ndim == 2 else int(grad.shape[0])
    out_features = int(grad.shape[-2])
    in_features = int(grad.shape[-1])
    if tuple(module.lora_a.shape) != (
        ((int(rank), in_features) if groups == 1 else (groups, int(rank), in_features))
    ):
        raise ValueError("GoRA input-factor shape does not match the gradient.")
    if tuple(module.lora_b.shape) != (
        ((out_features, int(rank)) if groups == 1 else (groups, out_features, int(rank)))
    ):
        raise ValueError("GoRA output-factor shape does not match the gradient.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    random_shape = (
        (out_features, int(rank))
        if groups == 1
        else (groups, out_features, int(rank))
    )
    # Kaiming uniform with a=sqrt(5) has bound 1/sqrt(fan_in).
    random_output = torch.empty(random_shape, dtype=torch.float32, device="cpu")
    random_output.uniform_(
        -1.0 / math.sqrt(int(rank)),
        1.0 / math.sqrt(int(rank)),
        generator=generator,
    )
    random_output = random_output.to(device=grad.device)
    grad_grouped = grad.unsqueeze(0) if groups == 1 else grad
    output_grouped = random_output.unsqueeze(0) if groups == 1 else random_output
    input_grouped = torch.empty(
        (groups, int(rank), in_features),
        dtype=torch.float32,
        device=grad.device,
    )
    scale = gamma * math.sqrt(out_features) / alpha
    for index in range(groups):
        input_grouped[index].copy_(
            -torch.linalg.pinv(output_grouped[index]) @ grad_grouped[index]
        )
    input_grouped.mul_(scale)

    resolved_a = input_grouped[0] if groups == 1 else input_grouped
    resolved_b = output_grouped[0] if groups == 1 else output_grouped
    with torch.no_grad():
        module.lora_a.copy_(
            resolved_a.to(device=module.lora_a.device, dtype=module.lora_a.dtype)
        )
        module.lora_b.copy_(
            resolved_b.to(device=module.lora_b.device, dtype=module.lora_b.dtype)
        )

    effective = (
        torch.matmul(output_grouped, input_grouped)
        * (alpha / math.sqrt(int(rank)))
    ).flatten()
    target = (-grad_grouped * gamma).flatten()
    effective_norm = torch.linalg.vector_norm(effective)
    target_norm = torch.linalg.vector_norm(target)
    denominator = effective_norm * target_norm
    cosine = (
        0.0
        if float(denominator.item()) == 0.0
        else float(torch.dot(effective, target).div(denominator).item())
    )
    relative_norm = (
        0.0
        if float(target_norm.item()) == 0.0
        else float(effective_norm.div(target_norm).item())
    )
    return GoRAInitializationDiagnostic(
        cosine_similarity=cosine,
        relative_norm=relative_norm,
    )


def gora_target_seed(base_seed: int, target_name: str) -> int:
    """Derive order-independent initialization randomness for one target."""

    digest = hashlib.sha256(str(target_name).encode()).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "big")) & 0x7FFFFFFFFFFFFFFF


__all__ = [
    "GoRAAllocationPlan",
    "GoRAInitializationDiagnostic",
    "allocate_gora_ranks",
    "gora_sensitivity_importance",
    "gora_target_seed",
    "initialize_gora_module",
    "resize_lora_module",
]
