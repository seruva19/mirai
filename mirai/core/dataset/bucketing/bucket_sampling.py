"""Carry-over bucket sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class CarryOverBucketState:
    credit: dict[str, float] = field(default_factory=dict)


def choose_bucket_with_carryover(
    bucket_weights: Mapping[str, float],
    *,
    state: CarryOverBucketState | None = None,
) -> tuple[str, CarryOverBucketState]:
    """Choose one bucket using weighted carry-over credit accounting."""

    weights = {str(k): float(v) for k, v in bucket_weights.items() if float(v) > 0.0}
    if not weights:
        raise ValueError("bucket_weights must contain at least one positive weight.")

    total = sum(weights.values())
    st = state or CarryOverBucketState()

    for bucket in sorted(weights):
        st.credit[bucket] = st.credit.get(bucket, 0.0) + (weights[bucket] / total)

    chosen = max(sorted(weights), key=lambda bucket: st.credit[bucket])
    st.credit[chosen] -= 1.0
    return chosen, st


def sample_bucket_sequence(
    bucket_weights: Mapping[str, float],
    *,
    draws: int,
    state: CarryOverBucketState | None = None,
) -> tuple[list[str], CarryOverBucketState]:
    if int(draws) < 0:
        raise ValueError("draws must be >= 0.")
    st = state or CarryOverBucketState()
    out: list[str] = []
    for _ in range(int(draws)):
        bucket, st = choose_bucket_with_carryover(bucket_weights, state=st)
        out.append(bucket)
    return out, st
