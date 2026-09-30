"""Original-ID preserving expert deletion contract.

DIET removes physical expert tensors while retaining the router's logical expert
axis.  Deleted logical ids map to ``-1`` and are invalid at dispatch; retained
ids map densely in ascending physical order.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

DIET_EXPERT_DELETION_FORMAT = "mirai.moe.expert_deletion"
DIET_EXPERT_DELETION_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class ExpertDeletionPlan:
    logical_num_experts: int
    kept_logical_ids: tuple[int, ...]
    top_k: int
    n_group: int
    topk_group: int
    group_score_topk: int
    norm_topk_prob: bool
    route_scale: float

    @property
    def physical_num_experts(self) -> int:
        return len(self.kept_logical_ids)

    @property
    def logical_to_physical(self) -> tuple[int, ...]:
        result = [-1] * self.logical_num_experts
        for physical_id, logical_id in enumerate(self.kept_logical_ids):
            result[logical_id] = physical_id
        return tuple(result)

    def validate(
        self,
        *,
        top_k: int | None = None,
        n_group: int | None = None,
        group_score_topk: int | None = None,
    ) -> "ExpertDeletionPlan":
        logical = int(self.logical_num_experts)
        kept = tuple(int(value) for value in self.kept_logical_ids)
        if logical < 2:
            raise ValueError("Expert deletion requires at least two logical experts.")
        if not kept or len(kept) >= logical:
            raise ValueError("Expert deletion must retain a non-empty strict subset.")
        if kept != tuple(sorted(set(kept))):
            raise ValueError("kept_logical_ids must be unique and sorted.")
        if kept[0] < 0 or kept[-1] >= logical:
            raise ValueError("kept_logical_ids contains an out-of-range logical id.")
        resolved_top_k = int(self.top_k if top_k is None else top_k)
        resolved_groups = int(self.n_group if n_group is None else n_group)
        resolved_group_score = int(
            self.group_score_topk if group_score_topk is None else group_score_topk
        )
        if resolved_top_k < 1 or len(kept) < resolved_top_k:
            raise ValueError("Expert deletion retains fewer experts than router top_k.")
        if resolved_groups < 1 or int(self.topk_group) < 1:
            raise ValueError("Expert deletion router group counts must be positive.")
        if int(self.topk_group) > resolved_groups:
            raise ValueError("topk_group cannot exceed n_group.")
        if not math.isfinite(float(self.route_scale)) or float(self.route_scale) <= 0:
            raise ValueError("Expert deletion route_scale must be finite and positive.")
        if resolved_groups > 1:
            groups = resolved_groups
            if logical % groups:
                raise ValueError("Logical expert count must be divisible by n_group.")
            per_group = logical // groups
            counts = [0] * groups
            for expert_id in kept:
                counts[expert_id // per_group] += 1
            minimum = max(1, resolved_group_score)
            if any(count < minimum for count in counts):
                raise ValueError(
                    "Expert deletion must retain enough experts in every original "
                    f"router group for group scoring ({minimum} required)."
                )
            if sum(sorted(counts)[: int(self.topk_group)]) < resolved_top_k:
                raise ValueError(
                    "Every possible selected router-group set must contain at "
                    "least top_k retained experts."
                )
        return self

    def manifest_spec(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": DIET_EXPERT_DELETION_FORMAT,
            "schema_version": DIET_EXPERT_DELETION_SCHEMA_VERSION,
            "logical_num_experts": int(self.logical_num_experts),
            "kept_logical_ids": list(self.kept_logical_ids),
            "top_k": int(self.top_k),
            "n_group": int(self.n_group),
            "topk_group": int(self.topk_group),
            "group_score_topk": int(self.group_score_topk),
            "norm_topk_prob": bool(self.norm_topk_prob),
            "route_scale": float(self.route_scale),
        }


def build_expert_deletion_plan(
    logical_num_experts: int,
    kept_logical_ids: Sequence[int],
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    group_score_topk: int = 2,
    norm_topk_prob: bool,
    route_scale: float,
) -> ExpertDeletionPlan:
    return ExpertDeletionPlan(
        logical_num_experts=int(logical_num_experts),
        kept_logical_ids=tuple(int(value) for value in kept_logical_ids),
        top_k=int(top_k),
        n_group=int(n_group),
        topk_group=int(topk_group),
        group_score_topk=int(group_score_topk),
        norm_topk_prob=bool(norm_topk_prob),
        route_scale=float(route_scale),
    ).validate(top_k=top_k, n_group=n_group, group_score_topk=group_score_topk)


def expert_deletion_from_manifest_spec(
    spec: Mapping[str, Any],
    *,
    physical_num_experts: int,
) -> ExpertDeletionPlan | None:
    raw = spec.get("expert_deletion")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("expert_deletion must be an object.")
    if raw.get("format") != DIET_EXPERT_DELETION_FORMAT:
        raise ValueError(f"Unsupported expert deletion format {raw.get('format')!r}.")
    if int(raw.get("schema_version", 0)) != DIET_EXPERT_DELETION_SCHEMA_VERSION:
        raise ValueError("Unsupported expert deletion schema version.")
    if not isinstance(raw.get("norm_topk_prob"), bool):
        raise ValueError("expert_deletion.norm_topk_prob must be a boolean.")
    def exact_int(name: str) -> int:
        value = raw.get(name)
        parsed = int(value)
        if isinstance(value, bool) or float(value) != float(parsed):
            raise ValueError(f"expert_deletion.{name} must be an exact integer.")
        return parsed
    kept = raw.get("kept_logical_ids")
    if not isinstance(kept, (list, tuple)):
        raise ValueError("expert_deletion.kept_logical_ids must be an integer array.")
    if any(
        isinstance(value, bool) or float(value) != float(int(value)) for value in kept
    ):
        raise ValueError("expert_deletion.kept_logical_ids must contain exact integers.")
    plan = build_expert_deletion_plan(
        exact_int("logical_num_experts"),
        kept,
        top_k=exact_int("top_k"),
        n_group=exact_int("n_group"),
        topk_group=exact_int("topk_group"),
        group_score_topk=exact_int("group_score_topk"),
        norm_topk_prob=bool(raw.get("norm_topk_prob")),
        route_scale=float(raw.get("route_scale", 0.0)),
    )
    if plan.physical_num_experts != int(physical_num_experts):
        raise ValueError("Expert deletion physical count does not match num_experts.")
    return plan


def remap_deleted_expert_routes(top_indices: Any, logical_to_physical: Any) -> Any:
    """Map selected live logical ids and reject any deleted route."""
    import torch

    indices = torch.as_tensor(top_indices)
    mapping = torch.as_tensor(
        logical_to_physical, dtype=torch.long, device=indices.device
    )
    if mapping.ndim != 1 or not int(mapping.numel()):
        raise ValueError("logical_to_physical must be a non-empty rank-1 mapping.")
    if indices.numel() and bool(
        torch.any((indices < 0) | (indices >= int(mapping.numel()))).item()
    ):
        raise ValueError("Logical route index is outside the expert deletion mapping.")
    physical = mapping.index_select(0, indices.reshape(-1).long()).reshape_as(indices)
    if physical.numel() and bool(torch.any(physical < 0).item()):
        raise RuntimeError("Router selected an expert deleted by the DIET artifact.")
    return physical


__all__ = [
    "DIET_EXPERT_DELETION_FORMAT",
    "DIET_EXPERT_DELETION_SCHEMA_VERSION",
    "ExpertDeletionPlan",
    "build_expert_deletion_plan",
    "expert_deletion_from_manifest_spec",
    "remap_deleted_expert_routes",
]
