"""Per-target activation conditioning diagnostics for adapter allocation."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Mapping, Sequence


def target_conditioning_diagnostics(
    spectra: Mapping[str, Sequence[float]],
    *,
    epsilon: float = 1e-12,
) -> dict[str, dict[str, float | int]]:
    diagnostics: dict[str, dict[str, float | int]] = {}
    for name, raw in spectra.items():
        values = [float(value) for value in raw if float(value) > float(epsilon)]
        if not values:
            continue
        largest = max(values)
        smallest = min(values)
        total = sum(values)
        diagnostics[str(name)] = {
            "condition_number": largest / smallest,
            "effective_rank": int(len(values)),
            "stable_rank": total / largest,
            "spectral_mean": fmean(values),
        }
    return dict(sorted(diagnostics.items()))


def allocation_conditioning_correlation(
    diagnostics: Mapping[str, Mapping[str, float | int]],
    ranks: Mapping[str, int],
) -> float | None:
    """Pearson correlation between log-condition and allocated rank."""
    pairs = [
        (
            math.log(max(float(diag["condition_number"]), 1.0)),
            float(ranks[name]),
        )
        for name, diag in diagnostics.items()
        if name in ranks
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mean_x, mean_y = fmean(xs), fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(denom_x * denom_y)
    return None if denominator == 0.0 else numerator / denominator


__all__ = [
    "allocation_conditioning_correlation",
    "target_conditioning_diagnostics",
]
