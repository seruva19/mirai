"""Fisher-Rao specialization telemetry for expert-routing distributions.

The Fisher Specialization Index is Equation 4 of
https://arxiv.org/abs/2604.14500.  It is the Fisher-Rao distance between the
empirical marginal expert distribution and the uniform distribution.  This
module deliberately implements only that well-defined distributional metric;
it does not infer expert-parameter Fisher matrices from ordinary accumulated
training gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FisherSpecializationSummary:
    """Layer-aggregated Fisher-Rao distance diagnostics."""

    mean_index: float
    mean_fraction: float
    minimum_index: float
    maximum_index: float
    layer_count: int

    def to_metrics(self) -> dict[str, float]:
        return {
            "moe_fisher_specialization_index": float(self.mean_index),
            "moe_fisher_specialization_fraction": float(self.mean_fraction),
            "moe_fisher_specialization_min_layer": float(self.minimum_index),
            "moe_fisher_specialization_max_layer": float(self.maximum_index),
            "moe_fisher_specialization_layer_count": float(self.layer_count),
        }


def fisher_rao_distance_to_uniform(probabilities: Any) -> tuple[Any, Any]:
    """Return FSI and ``FSI / FSI_max`` for one empirical token population.

    ``probabilities`` has an expert axis last and one or more population axes.
    Rows are normalized before their marginal is formed, which tolerates normal
    floating-point softmax drift while rejecting negative, non-finite, or
    zero-mass rows.  The calculation remains on the input device and returns
    scalar FP32 tensors.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("Fisher specialization telemetry requires torch.")
    if not torch.is_tensor(probabilities):
        raise TypeError("probabilities must be a torch.Tensor.")
    if int(probabilities.ndim) < 2:
        raise ValueError("probabilities require population and expert axes.")
    expert_count = int(probabilities.shape[-1])
    if expert_count < 2:
        raise ValueError("Fisher specialization requires at least two experts.")
    rows = probabilities.detach().float().reshape(-1, expert_count)
    if int(rows.shape[0]) == 0:
        raise ValueError("Fisher specialization requires at least one row.")
    if not bool(torch.isfinite(rows).all().item()):
        raise ValueError("probabilities must be finite.")
    if bool(torch.any(rows < 0).item()):
        raise ValueError("probabilities must be non-negative.")
    row_mass = rows.sum(dim=-1, keepdim=True)
    if bool(torch.any(row_mass <= 0).item()):
        raise ValueError("every probability row must have positive mass.")
    marginal = (rows / row_mass).mean(dim=0)
    coefficient = (
        marginal.clamp_min(0.0).sqrt().sum() / math.sqrt(expert_count)
    ).clamp(-1.0, 1.0)
    index = 2.0 * torch.acos(coefficient)
    maximum = 2.0 * math.acos(1.0 / math.sqrt(expert_count))
    fraction = index / maximum
    return index, fraction


def summarize_fisher_specialization(
    probability_layers: Iterable[Any],
) -> FisherSpecializationSummary | None:
    """Summarize exact per-layer FSI over the currently observed tokens."""

    indices: list[float] = []
    fractions: list[float] = []
    for probabilities in probability_layers:
        if probabilities is None or torch is None or not torch.is_tensor(probabilities):
            continue
        index, fraction = fisher_rao_distance_to_uniform(probabilities)
        indices.append(float(index.cpu().item()))
        fractions.append(float(fraction.cpu().item()))
    if not indices:
        return None
    return FisherSpecializationSummary(
        mean_index=float(sum(indices) / len(indices)),
        mean_fraction=float(sum(fractions) / len(fractions)),
        minimum_index=float(min(indices)),
        maximum_index=float(max(indices)),
        layer_count=len(indices),
    )


__all__ = [
    "FisherSpecializationSummary",
    "fisher_rao_distance_to_uniform",
    "summarize_fisher_specialization",
]
