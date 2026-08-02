"""Deterministic, lineage-bound rank redistribution before LoRA injection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ADAPTIVE_RANK_PLAN_SCHEMA_VERSION = 1
_LINEAGE_KEYS = (
    "dataset_snapshot_id",
    "model_snapshot_id",
    "config_snapshot_id",
)


class AdaptiveRankPlanLineageHost:
    """Pipeline mixin delegating the generic lineage hook to a rank plan."""

    _adaptive_rank_plan: AdaptiveRankPlan | None

    def validate_adapter_artifact_lineage(
        self,
        *,
        dataset_snapshot_id: str,
        model_snapshot_id: str,
        config_snapshot_id: str,
    ) -> None:
        validate_adaptive_rank_plan_lineage(
            self._adaptive_rank_plan,
            dataset_snapshot_id=dataset_snapshot_id,
            model_snapshot_id=model_snapshot_id,
            config_snapshot_id=config_snapshot_id,
        )


@dataclass(frozen=True)
class AdaptiveRankPlan:
    """Immutable target ranks selected from calibration spectra."""

    ranks: Mapping[str, int]
    rank_budget: int
    lineage: Mapping[str, str]
    fingerprint: str

    def validate(self) -> AdaptiveRankPlan:
        ranks = {str(name): int(rank) for name, rank in self.ranks.items()}
        if not ranks or any(not name or rank <= 0 for name, rank in ranks.items()):
            raise ValueError("Adaptive rank plans require named targets with positive ranks.")
        if sum(ranks.values()) != int(self.rank_budget):
            raise ValueError("Adaptive rank plan ranks must exactly consume rank_budget.")
        lineage = _validated_lineage(self.lineage)
        expected = _fingerprint(ranks, int(self.rank_budget), lineage)
        if str(self.fingerprint) != expected:
            raise ValueError("Adaptive rank plan fingerprint mismatch.")
        return self

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": ADAPTIVE_RANK_PLAN_SCHEMA_VERSION,
            "rank_budget": int(self.rank_budget),
            "ranks": dict(sorted((str(k), int(v)) for k, v in self.ranks.items())),
            **_validated_lineage(self.lineage),
            "fingerprint": str(self.fingerprint),
        }


def allocate_adaptive_ranks(
    spectra: Mapping[str, Sequence[float]],
    *,
    rank_budget: int,
    minimum_rank: int = 1,
    maximum_rank: int | None = None,
    lineage: Mapping[str, str],
) -> AdaptiveRankPlan:
    """Allocate every rank unit to the largest remaining explained-variance gain.

    Each spectrum is ordered by descending principal-component importance. A
    deterministic global greedy selection is the discrete optimum for this
    separable fixed-budget objective. Target name breaks equal-gain ties.
    """

    names = tuple(sorted(str(name) for name in spectra))
    if not names or len(names) != len(spectra) or any(not name for name in names):
        raise ValueError("Adaptive rank allocation requires unique named targets.")
    floor = int(minimum_rank)
    if floor <= 0:
        raise ValueError("minimum_rank must be positive.")
    ceiling = None if maximum_rank is None else int(maximum_rank)
    if ceiling is not None and ceiling < floor:
        raise ValueError("maximum_rank must be >= minimum_rank.")

    normalized: dict[str, tuple[float, ...]] = {}
    for name in names:
        values = tuple(float(value) for value in spectra[name])
        if not values or any(value < 0.0 for value in values):
            raise ValueError(f"Adaptive rank spectrum for {name!r} must be non-negative.")
        if any(values[index] < values[index + 1] for index in range(len(values) - 1)):
            raise ValueError(f"Adaptive rank spectrum for {name!r} must be descending.")
        limit = len(values) if ceiling is None else min(len(values), ceiling)
        if limit < floor:
            raise ValueError(f"Adaptive rank spectrum for {name!r} is shorter than minimum_rank.")
        normalized[name] = values[:limit]

    budget = int(rank_budget)
    minimum_budget = floor * len(names)
    maximum_budget = sum(len(values) for values in normalized.values())
    if budget < minimum_budget or budget > maximum_budget:
        raise ValueError(
            f"rank_budget must be in [{minimum_budget}, {maximum_budget}] for the supplied spectra."
        )

    ranks = {name: floor for name in names}
    for _ in range(budget - minimum_budget):
        candidates = [
            (-normalized[name][ranks[name]], name)
            for name in names
            if ranks[name] < len(normalized[name])
        ]
        if not candidates:  # defensive; budget bounds already prove feasibility
            raise RuntimeError("Adaptive rank allocator exhausted eligible components.")
        _, selected = min(candidates)
        ranks[selected] += 1

    resolved_lineage = _validated_lineage(lineage)
    fingerprint = _fingerprint(ranks, budget, resolved_lineage)
    return AdaptiveRankPlan(
        ranks=ranks,
        rank_budget=budget,
        lineage=resolved_lineage,
        fingerprint=fingerprint,
    ).validate()


def save_adaptive_rank_plan(path: str | Path, plan: AdaptiveRankPlan) -> None:
    Path(path).write_text(
        json.dumps(plan.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_adaptive_rank_plan(
    path: str | Path,
    *,
    expected_lineage: Mapping[str, str] | None = None,
) -> AdaptiveRankPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Adaptive rank plan must be a JSON object.")
    if int(payload.get("schema_version", 0)) != ADAPTIVE_RANK_PLAN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported adaptive rank plan schema {payload.get('schema_version')!r}.")
    lineage = {key: str(payload.get(key, "")) for key in _LINEAGE_KEYS}
    plan = AdaptiveRankPlan(
        ranks=dict(payload.get("ranks", {})),
        rank_budget=int(payload.get("rank_budget", 0)),
        lineage=lineage,
        fingerprint=str(payload.get("fingerprint", "")),
    ).validate()
    if expected_lineage is not None:
        expected = _validated_lineage(expected_lineage)
        for key in _LINEAGE_KEYS:
            if plan.lineage[key] != expected[key]:
                raise ValueError(
                    f"Adaptive rank plan {key} mismatch: expected {expected[key]!r}, "
                    f"found {plan.lineage[key]!r}."
                )
    return plan


def resolve_adaptive_rank_plan(
    path: str | Path,
    *,
    configured_budget: int,
) -> AdaptiveRankPlan | None:
    text = str(path).strip()
    if not text:
        return None
    plan = load_adaptive_rank_plan(text)
    if int(configured_budget) not in {0, plan.rank_budget}:
        raise ValueError(
            "adapter.rank_budget must be zero or exactly match the adaptive "
            f"rank plan budget {plan.rank_budget}."
        )
    return plan


def validate_adaptive_rank_plan_lineage(
    plan: AdaptiveRankPlan | None,
    *,
    dataset_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str,
) -> None:
    if plan is None:
        return
    expected = {
        "dataset_snapshot_id": str(dataset_snapshot_id),
        "model_snapshot_id": str(model_snapshot_id),
        "config_snapshot_id": str(config_snapshot_id),
    }
    for key, value in expected.items():
        if plan.lineage[key] != value:
            raise ValueError(
                f"Adaptive rank plan {key} mismatch: expected {value!r}, "
                f"found {plan.lineage[key]!r}."
            )


def _validated_lineage(lineage: Mapping[str, str]) -> dict[str, str]:
    resolved = {key: str(lineage.get(key, "")).strip() for key in _LINEAGE_KEYS}
    if not all(resolved.values()):
        raise ValueError("Adaptive rank plans require complete snapshot lineage.")
    return resolved


def _fingerprint(
    ranks: Mapping[str, int], rank_budget: int, lineage: Mapping[str, str]
) -> str:
    canonical = {
        "rank_budget": int(rank_budget),
        "ranks": dict(sorted((str(k), int(v)) for k, v in ranks.items())),
        **dict(sorted((str(k), str(v)) for k, v in lineage.items())),
    }
    encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ADAPTIVE_RANK_PLAN_SCHEMA_VERSION",
    "AdaptiveRankPlanLineageHost",
    "AdaptiveRankPlan",
    "allocate_adaptive_ranks",
    "load_adaptive_rank_plan",
    "resolve_adaptive_rank_plan",
    "save_adaptive_rank_plan",
    "validate_adaptive_rank_plan_lineage",
]
