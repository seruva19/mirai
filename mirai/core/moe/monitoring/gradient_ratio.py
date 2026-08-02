"""Load-balance-to-task gradient diagnostics on router probabilities.

LongCat-Flash Equation 9 measures the weighted load-balancing gradient against
the task gradient on the batch-averaged expert-probability vector:
https://arxiv.org/abs/2509.01322

Providers expose graph-bearing probability tensors, not router parameters.  A
token-axis sum of their gradients is the derivative with respect to the
corresponding batch mean.  The monitor uses ``autograd.grad`` without writing
``.grad`` so the subsequent training backward remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_OBJECTIVE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class RouterProbabilityTarget:
    """One layer's graph-bearing router probabilities."""

    layer: str
    probabilities: Any


@dataclass(frozen=True)
class BalanceGradientProbe:
    """Provider-owned graph references consumed once after loss assembly."""

    targets: tuple[RouterProbabilityTarget, ...]
    objectives: Mapping[str, Any]


def _metric_prefix(objective: str) -> str:
    if objective == "load_balance":
        return "moe_balance_grad_ratio"
    return f"moe_{objective}_grad_ratio"


def _collapsed_gradient_vectors(
    loss: Any,
    tensors: tuple[Any, ...],
) -> tuple[Any | None, ...]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Router gradient-ratio telemetry requires torch.")
    if not tensors:
        return ()
    if not torch.is_tensor(loss) or not bool(loss.requires_grad):
        return tuple(None for _ in tensors)
    gradients = torch.autograd.grad(
        loss,
        tensors,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    vectors: list[Any | None] = []
    for tensor, gradient in zip(tensors, gradients, strict=True):
        if gradient is None:
            vectors.append(None)
            continue
        if int(tensor.ndim) < 1:
            raise ValueError("Router probability targets must have an expert axis.")
        expert_count = int(tensor.shape[-1])
        if expert_count <= 0:
            raise ValueError("Router probability targets require at least one expert.")
        # If p_bar = mean_t p_t, then dL/dp_bar is the token-axis sum of
        # dL/dp_t.  This is the batch-collapsed vector used by Equation 9.
        vectors.append(
            gradient.detach().float().reshape(-1, expert_count).sum(dim=0)
        )
    return tuple(vectors)


def _squared_norm(vectors: tuple[Any | None, ...]) -> Any | None:
    total = None
    for vector in vectors:
        if vector is None:
            continue
        value = vector.square().sum()
        total = value if total is None else total + value
    return total


def measure_balance_task_gradient_ratios(
    *,
    task_loss: Any,
    probe: BalanceGradientProbe,
    alarm_threshold: float = 0.1,
) -> dict[str, float]:
    """Return paper-aligned per-objective ratios without mutating gradients.

    The numerator includes each objective's configured coefficient.  Metrics
    are absent when the task loss has no gradient path to any exposed router
    probability tensor; a zero-weight objective reports an exact zero ratio.
    """

    threshold = float(alarm_threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("alarm_threshold must be finite and > 0.")
    active_targets = tuple(
        target
        for target in probe.targets
        if torch is not None
        and torch.is_tensor(target.probabilities)
        and bool(target.probabilities.requires_grad)
    )
    if not active_targets:
        return {}
    layer_names = tuple(str(target.layer) for target in active_targets)
    if len(set(layer_names)) != len(layer_names):
        raise ValueError("Router probability target layer names must be unique.")
    tensors = tuple(target.probabilities for target in active_targets)
    task_vectors = _collapsed_gradient_vectors(task_loss, tensors)
    task_squared = _squared_norm(task_vectors)
    if task_squared is None:
        return {}
    task_norm = float(torch.sqrt(task_squared).detach().cpu().item())
    if not math.isfinite(task_norm) or task_norm <= 0.0:
        return {}

    metrics: dict[str, float] = {}
    for objective_name, objective_loss in probe.objectives.items():
        name = str(objective_name).strip().lower()
        if not _OBJECTIVE_NAME.fullmatch(name):
            raise ValueError(
                "Balance-gradient objective names must match "
                "'[a-z][a-z0-9_]*'."
            )
        objective_vectors = _collapsed_gradient_vectors(objective_loss, tensors)
        objective_squared = _squared_norm(objective_vectors)
        if objective_squared is None:
            objective_norm = 0.0
        else:
            objective_norm = float(
                torch.sqrt(objective_squared).detach().cpu().item()
            )
        ratio = objective_norm / task_norm
        layer_ratios: list[float] = []
        for task_vector, objective_vector in zip(
            task_vectors, objective_vectors, strict=True
        ):
            if task_vector is None:
                continue
            denominator = float(torch.linalg.vector_norm(task_vector).cpu().item())
            if denominator <= 0.0 or not math.isfinite(denominator):
                continue
            numerator = (
                0.0
                if objective_vector is None
                else float(torch.linalg.vector_norm(objective_vector).cpu().item())
            )
            layer_ratios.append(numerator / denominator)
        prefix = _metric_prefix(name)
        metrics[prefix] = float(ratio)
        metrics[f"{prefix}_max_layer"] = float(max(layer_ratios, default=ratio))
        metrics[f"{prefix}_task_norm"] = float(task_norm)
        metrics[f"{prefix}_objective_norm"] = float(objective_norm)
        metrics[f"{prefix}_alarm"] = float(ratio >= threshold)
    return metrics


__all__ = [
    "BalanceGradientProbe",
    "RouterProbabilityTarget",
    "measure_balance_task_gradient_ratios",
]
