"""Durable router-health report construction from step metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean
from typing import Iterable, Mapping


ROUTER_HEALTH_METRICS = (
    "moe_routing_entropy",
    "moe_expert_utilization_cv",
    "moe_top1_monopoly",
    "moe_unique_expert_fraction",
    "moe_max_deadlock_duration",
    "moe_deadlocked_layer_count",
    "moe_deadlocked_layer_count_depth_q1",
    "moe_deadlocked_layer_count_depth_q2",
    "moe_deadlocked_layer_count_depth_q3",
    "moe_deadlocked_layer_count_depth_q4",
    "moe_max_deadlock_duration_depth_q1",
    "moe_max_deadlock_duration_depth_q2",
    "moe_max_deadlock_duration_depth_q3",
    "moe_max_deadlock_duration_depth_q4",
    "moe_router_underflow_fraction",
)


@dataclass(frozen=True)
class MetricSummary:
    observed_steps: int
    coverage: float
    minimum: float
    mean: float
    maximum: float


def build_router_health_report(
    records: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    rows = tuple(records)
    summaries: dict[str, dict[str, float | int]] = {}
    for metric in ROUTER_HEALTH_METRICS:
        values = [
            float(row[metric])
            for row in rows
            if metric in row and math.isfinite(float(row[metric]))
        ]
        if not values:
            continue
        summary = MetricSummary(
            observed_steps=len(values),
            coverage=(len(values) / len(rows) if rows else 0.0),
            minimum=min(values),
            mean=fmean(values),
            maximum=max(values),
        )
        summaries[metric] = {
            "observed_steps": summary.observed_steps,
            "coverage": summary.coverage,
            "min": summary.minimum,
            "mean": summary.mean,
            "max": summary.maximum,
        }
    return {
        "schema_version": 1,
        "record_count": len(rows),
        "status": "complete" if summaries else "incomplete",
        "metrics": summaries,
        "missing_metrics": [
            metric for metric in ROUTER_HEALTH_METRICS if metric not in summaries
        ],
    }


__all__ = ["ROUTER_HEALTH_METRICS", "build_router_health_report"]
