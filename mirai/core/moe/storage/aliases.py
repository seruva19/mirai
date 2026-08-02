"""Stable logical-to-physical expert alias contract."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Protocol, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class PrototypeConsolidationPlan:
    """Validated mapping from logical experts to a dense physical pool."""

    logical_to_physical: tuple[int, ...]
    prototype_logical_ids: tuple[int, ...]

    @property
    def logical_experts(self) -> int:
        return len(self.logical_to_physical)

    @property
    def physical_experts(self) -> int:
        return len(self.prototype_logical_ids)

    @property
    def logical_to_prototype(self) -> tuple[int, ...]:
        return tuple(
            self.prototype_logical_ids[physical_id]
            for physical_id in self.logical_to_physical
        )


class PrototypeCalibrationObserver(Protocol):
    """Neutral runtime sink for per-route prototype calibration evidence."""

    def record(
        self,
        expert_id: int,
        routing_weights: Any,
        expert_outputs: Any,
    ) -> None: ...


def build_prototype_plan(
    logical_to_prototype: Sequence[int],
    *,
    num_logical_experts: int,
) -> PrototypeConsolidationPlan:
    """Validate one-hop prototype ids and densify physical ids."""
    logical_count = int(num_logical_experts)
    if logical_count < 2:
        raise ValueError("Prototype consolidation requires at least two logical experts.")
    raw = tuple(int(value) for value in logical_to_prototype)
    if len(raw) != logical_count:
        raise ValueError(
            "Prototype mapping length must equal num_logical_experts "
            f"({len(raw)} != {logical_count})."
        )
    invalid = sorted({value for value in raw if value < 0 or value >= logical_count})
    if invalid:
        raise ValueError(f"Prototype mapping contains out-of-range expert ids: {invalid}.")
    prototypes = tuple(sorted(set(raw)))
    if len(prototypes) >= logical_count:
        raise ValueError(
            "Prototype consolidation must reduce the physical expert count; "
            f"got {len(prototypes)} prototypes for {logical_count} logical experts."
        )
    non_idempotent = [prototype for prototype in prototypes if raw[prototype] != prototype]
    if non_idempotent:
        raise ValueError(
            "Retained prototype experts must map to themselves; invalid prototypes: "
            f"{non_idempotent}."
        )
    dense_ids = {prototype: index for index, prototype in enumerate(prototypes)}
    return PrototypeConsolidationPlan(
        logical_to_physical=tuple(dense_ids[value] for value in raw),
        prototype_logical_ids=prototypes,
    )


def validate_logical_to_physical(
    logical_to_physical: Sequence[int],
    *,
    physical_experts: int,
) -> tuple[int, ...]:
    """Validate a manifest/runtime logical-to-physical mapping."""
    physical_count = int(physical_experts)
    values = tuple(int(value) for value in logical_to_physical)
    if physical_count < 1:
        raise ValueError("physical_experts must be positive.")
    if len(values) <= physical_count:
        raise ValueError(
            "A consolidated mapping must have more logical than physical experts."
        )
    invalid = sorted({value for value in values if value < 0 or value >= physical_count})
    if invalid:
        raise ValueError(f"Logical-to-physical mapping contains invalid ids: {invalid}.")
    missing = sorted(set(range(physical_count)) - set(values))
    if missing:
        raise ValueError(
            f"Logical-to-physical mapping does not reference physical ids: {missing}."
        )
    return values


def logical_to_physical_from_manifest_spec(
    spec: Mapping[str, Any],
    *,
    schema_version: int,
) -> tuple[int, ...] | None:
    """Read and validate the optional schema-v3 grouped-expert alias fields."""
    raw = spec.get("logical_to_physical")
    if raw is None:
        if schema_version >= 3 and spec.get("logical_num_experts") is not None:
            raise ValueError(
                "Consolidated grouped-expert manifest is missing logical_to_physical."
            )
        return None
    if schema_version < 3:
        raise ValueError("logical_to_physical requires packed-state schema_version 3.")
    if not isinstance(raw, (list, tuple)):
        raise ValueError("logical_to_physical must be an integer array.")
    values = validate_logical_to_physical(
        raw,
        physical_experts=int(spec.get("num_experts", 0)),
    )
    if int(spec.get("logical_num_experts", 0)) != len(values):
        raise ValueError("logical_num_experts does not match logical_to_physical length.")
    prototypes = spec.get("prototype_logical_ids")
    if not isinstance(prototypes, (list, tuple)):
        raise ValueError("Consolidated grouped-expert manifest has no prototype ids.")
    prototype_ids = tuple(int(value) for value in prototypes)
    physical_count = int(spec.get("num_experts", 0))
    if len(prototype_ids) != physical_count or len(set(prototype_ids)) != len(
        prototype_ids
    ):
        raise ValueError(
            "prototype_logical_ids must contain one unique id per physical expert."
        )
    if any(value < 0 or value >= len(values) for value in prototype_ids):
        raise ValueError("prototype_logical_ids contains an out-of-range logical id.")
    if any(
        values[logical_id] != physical_id
        for physical_id, logical_id in enumerate(prototype_ids)
    ):
        raise ValueError(
            "Each prototype logical id must map to its corresponding physical id."
        )
    return values


def coalesce_alias_routes(
    top_scores: Any,
    top_indices: Any,
    logical_to_physical: Any,
) -> tuple[Any, Any]:
    """Remap logical routes and sum duplicate physical contributions."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("Prototype route coalescing requires torch.")
    if top_scores.shape != top_indices.shape or top_indices.ndim != 2:
        raise ValueError(
            "Prototype route coalescing requires matching [tokens, top_k] tensors."
        )
    mapping = torch.as_tensor(
        logical_to_physical,
        dtype=torch.long,
        device=top_indices.device,
    )
    if mapping.ndim != 1 or int(mapping.numel()) == 0:
        raise ValueError("logical_to_physical must be a non-empty rank-1 mapping.")
    if top_indices.numel() and bool(
        torch.any((top_indices < 0) | (top_indices >= int(mapping.numel()))).item()
    ):
        raise ValueError("Logical route index is outside the consolidation mapping.")
    physical = mapping.index_select(0, top_indices.reshape(-1).long()).reshape_as(
        top_indices
    )
    top_k = int(top_indices.shape[1])
    if top_k == 0:
        return top_scores, physical
    current = torch.arange(top_k, device=top_indices.device).view(1, top_k, 1)
    candidate = torch.arange(top_k, device=top_indices.device).view(1, 1, top_k)
    same_physical = physical.unsqueeze(2) == physical.unsqueeze(1)
    eligible = same_physical & (candidate <= current)
    first_slot = torch.where(
        eligible,
        candidate.expand_as(eligible),
        torch.full_like(candidate.expand_as(eligible), top_k),
    ).amin(dim=2)
    coalesced_scores = torch.zeros_like(top_scores).scatter_add(
        1, first_slot, top_scores
    )
    return coalesced_scores, physical


__all__ = [
    "PrototypeCalibrationObserver",
    "PrototypeConsolidationPlan",
    "build_prototype_plan",
    "coalesce_alias_routes",
    "logical_to_physical_from_manifest_spec",
    "validate_logical_to_physical",
]
