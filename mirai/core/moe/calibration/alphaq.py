"""Calibration-free spectral importance for expert precision planning.

The PL_Alpha_Hill signal follows AlphaQ (arXiv:2606.04980). Mirai combines
that data-free signal with measured errors from its own packed formats rather
than reproducing AlphaQ's GPTQ noise model or external MILP implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SpectralPrecisionEvidence:
    """AlphaQ-style importance for one physical expert projection."""

    module_name: str
    expert_id: int
    projection: str
    alpha_hill: float
    weight_variance: float


def _fixed_aspect_eigenvalues(
    weight: Any,
    *,
    block_size: int = 256,
    max_blocks: int = 256,
) -> Any:
    """Return sorted W^T W eigenvalues from deterministic square blocks."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Spectral precision planning requires torch.")
    matrix = torch.as_tensor(weight).detach()
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("Spectral precision weights must be two-dimensional.")
    rows, columns = (int(value) for value in matrix.shape)
    side = min(int(block_size), rows, columns)
    if side < 2 or int(max_blocks) <= 0:
        raise ValueError("Spectral block size and maximum blocks must be positive.")
    row_starts = range(0, rows - side + 1, side)
    column_starts = range(0, columns - side + 1, side)
    starts = [
        (row_start, column_start) for row_start in row_starts for column_start in column_starts
    ][: int(max_blocks)]
    matrix = matrix.to(device="cpu", dtype=torch.float32)
    values = []
    for row_start, column_start in starts:
        block = matrix[
            row_start : row_start + side,
            column_start : column_start + side,
        ]
        values.append(torch.linalg.svdvals(block).square())
    return torch.cat(values).sort().values


def pl_alpha_hill(
    weight: Any,
    *,
    block_size: int = 256,
    max_blocks: int = 256,
    histogram_bins: int = 100,
) -> float:
    """Estimate the Fix-finger PL exponent from a weight spectrum."""

    eigenvalues = _fixed_aspect_eigenvalues(
        weight,
        block_size=block_size,
        max_blocks=max_blocks,
    )
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if int(eigenvalues.numel()) < 2:
        raise ValueError("Spectral precision requires at least two positive eigenvalues.")
    bins = min(max(2, int(histogram_bins)), int(eigenvalues.numel()))
    log_values = eigenvalues.log10()
    minimum = float(log_values[0].item())
    maximum = float(log_values[-1].item())
    if not maximum > minimum:
        raise ValueError("Spectral precision requires a non-degenerate spectrum.")
    counts = torch.histc(log_values, bins=bins, min=minimum, max=maximum)
    boundaries = torch.linspace(minimum, maximum, bins + 1)
    peak_upper = boundaries[int(counts.argmax().item()) + 1]
    threshold = peak_upper.to(dtype=log_values.dtype)
    tail = eigenvalues[log_values >= threshold]
    if int(tail.numel()) < 2:
        tail = eigenvalues[-2:]
    reference = tail[0].clamp_min(1e-12)
    log_sum = torch.log(tail[1:] / reference).sum()
    if not float(log_sum.item()) > 0.0:
        raise ValueError("Spectral precision produced a degenerate Hill tail.")
    alpha = 1.0 + float(tail.numel() - 1) / float(log_sum.item())
    if not math.isfinite(alpha) or alpha <= 1.0:
        raise ValueError("Spectral precision produced an invalid Hill exponent.")
    return alpha


def alpha_importance_weights(
    alphas: Sequence[float],
    *,
    gamma: float = 0.0,
) -> tuple[tuple[float, ...], float]:
    """Return (median/alpha)^gamma weights and the resolved data-free gamma."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Spectral precision planning requires torch.")
    values = torch.tensor(tuple(float(value) for value in alphas), dtype=torch.float64)
    if (
        not int(values.numel())
        or not bool(torch.isfinite(values).all())
        or bool((values <= 1.0).any())
    ):
        raise ValueError("Alpha values must be finite and greater than one.")
    resolved_gamma = float(gamma)
    if resolved_gamma < 0.0 or not math.isfinite(resolved_gamma):
        raise ValueError("Spectral precision gamma must be finite and non-negative.")
    if resolved_gamma == 0.0:
        variance = float(values.var(correction=0).item())
        spread = float((values.max() - values.min()).item())
        resolved_gamma = (
            float(values.min().item()) * spread / variance
            if variance > 0.0 and spread > 0.0
            else 1.0
        )
    median = float(values.median().item())
    weights = tuple((median / float(value)) ** resolved_gamma for value in values)
    return weights, resolved_gamma


def spectral_projection_evidence(
    targets: Mapping[str, Any],
) -> tuple[SpectralPrecisionEvidence, ...]:
    """Measure bounded, deterministic spectral evidence for provider targets."""

    rows = []
    for name, target in sorted(targets.items()):
        for projection in ("w1", "w2", "w3"):
            source = torch.as_tensor(target.weights[projection]).detach()
            for expert_id in range(int(target.num_experts)):
                weight = source[expert_id]
                rows.append(
                    SpectralPrecisionEvidence(
                        module_name=str(name),
                        expert_id=expert_id,
                        projection=projection,
                        alpha_hill=pl_alpha_hill(weight),
                        weight_variance=float(weight.float().var(correction=0).item()),
                    )
                )
    return tuple(rows)


__all__ = [
    "SpectralPrecisionEvidence",
    "alpha_importance_weights",
    "pl_alpha_hill",
    "spectral_projection_evidence",
]
