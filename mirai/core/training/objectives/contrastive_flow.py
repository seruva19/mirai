"""Contrastive Flow Matching loss terms.

Implements Eq. 6 and Algorithm 1 from Contrastive Flow Matching
(Stoica et al., ICCV 2025, https://arxiv.org/abs/2506.05350). A negative flow
target is sampled uniformly from the other examples in the current microbatch;
the model is evaluated only once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mirai.core.tensors import is_torch_tensor

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ContrastiveFlowLoss:
    per_sample_loss: Any
    positive_loss: Any
    negative_distance: Any | None
    negative_indices: Any | None


def sample_negative_indices(batch_size: int, *, device: Any = None) -> Any:
    """Sample one uniformly distributed non-self index for every batch item."""

    size = int(batch_size)
    if size < 2:
        raise ValueError(
            "Contrastive Flow Matching requires at least two examples in each "
            "microbatch."
        )
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch is required for Contrastive Flow Matching.")
    anchors = torch.arange(size, device=device, dtype=torch.long)
    if size == 2:
        return 1 - anchors
    draws = torch.randint(0, size - 1, (size,), device=device)
    return draws + (draws >= anchors).to(dtype=torch.long)


def _validate_negative_indices(indices: Any, *, batch_size: int, device: Any) -> Any:
    if torch is None or not is_torch_tensor(indices):
        raise TypeError("negative_indices must be a torch tensor.")
    resolved = indices.to(device=device, dtype=torch.long)
    if tuple(resolved.shape) != (int(batch_size),):
        raise ValueError(
            "negative_indices must contain exactly one index per batch item."
        )
    anchors = torch.arange(int(batch_size), device=device, dtype=torch.long)
    if bool(((resolved < 0) | (resolved >= int(batch_size))).any()):
        raise ValueError("negative_indices contains an out-of-range batch index.")
    if bool((resolved == anchors).any()):
        raise ValueError("Contrastive Flow Matching negatives cannot select self.")
    return resolved


def compute_contrastive_flow_loss(
    *,
    prediction: Any,
    target: Any,
    loss_evaluator: Callable[[Any, Any], Any],
    weight: float,
    negative_indices: Any | None = None,
) -> ContrastiveFlowLoss:
    """Return ``L_positive - weight * L_negative`` for every batch item."""

    value = float(weight)
    positive = loss_evaluator(prediction, target)
    if value == 0.0:
        return ContrastiveFlowLoss(
            per_sample_loss=positive,
            positive_loss=positive,
            negative_distance=None,
            negative_indices=None,
        )
    if not 0.0 < value < 1.0:
        raise ValueError("contrastive flow weight must be in (0, 1).")
    if not is_torch_tensor(prediction) or not is_torch_tensor(target):
        raise TypeError(
            "Contrastive Flow Matching requires torch prediction and target tensors."
        )
    if prediction.ndim < 1 or target.ndim < 1:
        raise ValueError(
            "Contrastive Flow Matching prediction and target need a batch dimension."
        )
    if int(prediction.shape[0]) != int(target.shape[0]):
        raise ValueError(
            "Contrastive Flow Matching prediction and target batch sizes differ."
        )
    size = int(target.shape[0])
    indices = (
        sample_negative_indices(size, device=target.device)
        if negative_indices is None
        else _validate_negative_indices(
            negative_indices,
            batch_size=size,
            device=target.device,
        )
    )
    negative_target = target.index_select(0, indices)
    negative = loss_evaluator(prediction, negative_target)
    if not is_torch_tensor(positive) or not is_torch_tensor(negative):
        raise TypeError(
            "Contrastive Flow Matching requires tensor-valued per-sample losses."
        )
    if tuple(positive.shape) != tuple(negative.shape):
        raise ValueError(
            "Positive and negative Contrastive Flow Matching loss shapes differ."
        )
    return ContrastiveFlowLoss(
        per_sample_loss=positive - value * negative,
        positive_loss=positive,
        negative_distance=negative,
        negative_indices=indices,
    )


__all__ = [
    "ContrastiveFlowLoss",
    "compute_contrastive_flow_loss",
    "sample_negative_indices",
]
