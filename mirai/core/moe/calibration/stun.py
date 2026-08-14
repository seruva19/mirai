"""Structured expert clustering and centroid replacement from STUN.

STUN clusters experts from router-row similarity, retains the expert nearest
the parameter centroid of each cluster, and selectively replaces that expert
with the centroid when very few clusters remain.  The paper follows this stage
with unstructured Wanda or OWL pruning.  Mirai exposes the structured stage
independently so an artifact backend can choose an executable second stage.

Source: Lee et al., "STUN: Structured-Then-Unstructured Pruning for Scalable
MoE Pruning", ACL 2025, Eq. 7/14-17 and Algorithms 1-2.
https://aclanthology.org/2025.acl-long.671/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class StunClusterSelection:
    """One retained physical expert and the logical experts it represents."""

    members: tuple[int, ...]
    representative: int
    reconstruct: bool


@dataclass(frozen=True)
class StunExpertPlan:
    """Deterministic per-layer output of STUN's structured first stage."""

    source_experts: int
    clusters: tuple[StunClusterSelection, ...]

    @property
    def output_experts(self) -> int:
        return len(self.clusters)


def cluster_router_experts(
    router_weight: Any,
    *,
    target_experts: int,
    coactivation: Any | None = None,
    router_distance_weight: float = 1.0,
    coactivation_weight: float = 0.0,
) -> tuple[tuple[int, ...], ...]:
    """Agglomerate router rows until ``target_experts`` clusters remain.

    STUN defines pairwise behavioral similarity as negative router-row
    distance, optionally plus normalized coactivation.  The paper tunes a
    threshold to reach the desired sparsity; this equivalent deterministic
    form directly stops at the requested cluster count.  Complete linkage
    prevents a merge when any cross-cluster pair is dissimilar.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("STUN expert clustering requires torch.")
    weight = torch.as_tensor(router_weight).detach().to(dtype=torch.float32)
    if weight.ndim != 2 or int(weight.shape[0]) < 1:
        raise ValueError("router_weight must have shape [experts, hidden].")
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("router_weight contains non-finite values.")
    experts = int(weight.shape[0])
    target = int(target_experts)
    if target < 1 or target > experts:
        raise ValueError(f"target_experts must be in [1, {experts}], got {target}.")
    lambda_router = float(router_distance_weight)
    lambda_coactivation = float(coactivation_weight)
    if lambda_router < 0 or lambda_coactivation < 0:
        raise ValueError("STUN similarity weights must be non-negative.")
    if lambda_router == 0 and lambda_coactivation == 0:
        raise ValueError("At least one STUN similarity weight must be positive.")

    similarity = -lambda_router * torch.cdist(weight, weight, p=2)
    if coactivation is not None:
        coact = torch.as_tensor(
            coactivation,
            device=similarity.device,
            dtype=torch.float32,
        )
        if tuple(coact.shape) != (experts, experts):
            raise ValueError(
                f"coactivation must have shape [{experts}, {experts}], got "
                f"{tuple(coact.shape)}."
            )
        if not bool(torch.isfinite(coact).all().item()):
            raise ValueError("coactivation contains non-finite values.")
        if bool((coact < 0).any().item()):
            raise ValueError("coactivation must be non-negative.")
        similarity = similarity + lambda_coactivation * coact
    elif lambda_coactivation:
        raise ValueError("coactivation_weight requires coactivation statistics.")

    clusters: list[tuple[int, ...]] = [(index,) for index in range(experts)]
    while len(clusters) > target:
        best_pair: tuple[int, int] | None = None
        best_score = float("-inf")
        for left in range(len(clusters) - 1):
            left_index = torch.as_tensor(
                clusters[left],
                device=similarity.device,
                dtype=torch.long,
            )
            for right in range(left + 1, len(clusters)):
                right_index = torch.as_tensor(
                    clusters[right],
                    device=similarity.device,
                    dtype=torch.long,
                )
                score = float(
                    similarity.index_select(0, left_index)
                    .index_select(1, right_index)
                    .amin()
                    .item()
                )
                if score > best_score:
                    best_score = score
                    best_pair = (left, right)
        if best_pair is None:  # pragma: no cover - impossible for len > target
            raise RuntimeError("STUN clustering could not select a merge.")
        left, right = best_pair
        merged = tuple(sorted((*clusters[left], *clusters[right])))
        clusters[left] = merged
        del clusters[right]
        clusters.sort(key=lambda item: item[0])
    return tuple(clusters)


def select_stun_representatives(
    clusters: tuple[tuple[int, ...], ...],
    expert_weights: Mapping[str, Any],
    *,
    reconstruct_below: int = 3,
) -> StunExpertPlan:
    """Select the expert nearest each multi-projection parameter centroid."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("STUN expert selection requires torch.")
    if not clusters:
        raise ValueError("STUN selection requires at least one cluster.")
    if set(expert_weights) != {"w1", "w2", "w3"}:
        raise ValueError("STUN selection requires w1, w2, and w3 expert weights.")
    dense = {
        str(key): torch.as_tensor(value).detach().to(dtype=torch.float32)
        for key, value in expert_weights.items()
    }
    source_experts = int(dense["w1"].shape[0])
    if source_experts < 1:
        raise ValueError("STUN selection requires at least one source expert.")
    for key, value in dense.items():
        if value.ndim != 3 or int(value.shape[0]) != source_experts:
            raise ValueError(
                f"STUN {key} weights must have shape [experts, out, in]."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"STUN {key} weights contain non-finite values.")
    flattened = sorted(member for cluster in clusters for member in cluster)
    if flattened != list(range(source_experts)):
        raise ValueError("STUN clusters must partition every source expert exactly once.")

    reconstruct = len(clusters) < int(reconstruct_below)
    selections: list[StunClusterSelection] = []
    for cluster in clusters:
        members = tuple(sorted(int(value) for value in cluster))
        index = torch.as_tensor(members, dtype=torch.long, device=dense["w1"].device)
        distances = torch.zeros(len(members), dtype=torch.float64)
        for value in dense.values():
            selected = value.index_select(0, index)
            centroid = selected.mean(dim=0, keepdim=True)
            distances += (
                (selected - centroid)
                .square()
                .flatten(1)
                .sum(dim=1)
                .to(device="cpu", dtype=torch.float64)
            )
        representative = members[int(torch.argmin(distances).item())]
        selections.append(
            StunClusterSelection(
                members=members,
                representative=representative,
                reconstruct=reconstruct,
            )
        )
    return StunExpertPlan(
        source_experts=source_experts,
        clusters=tuple(selections),
    )


def select_stun_representatives_streaming(
    clusters: tuple[tuple[int, ...], ...],
    *,
    source_experts: int,
    load_weight: Callable[[str, int], Any],
    reconstruct_below: int = 3,
) -> StunExpertPlan:
    """Bounded-memory form of centroid-nearest representative selection."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("STUN expert selection requires torch.")
    flattened = sorted(member for cluster in clusters for member in cluster)
    if flattened != list(range(int(source_experts))):
        raise ValueError("STUN clusters must partition every source expert exactly once.")
    reconstruct = len(clusters) < int(reconstruct_below)
    selections: list[StunClusterSelection] = []
    for cluster in clusters:
        members = tuple(sorted(int(value) for value in cluster))
        distances = torch.zeros(len(members), dtype=torch.float64)
        for key in ("w1", "w2", "w3"):
            centroid = None
            for expert_index in members:
                weight = torch.as_tensor(load_weight(key, expert_index)).detach()
                if weight.ndim != 2 or not bool(torch.isfinite(weight).all().item()):
                    raise ValueError(f"STUN {key} expert weight is invalid.")
                compute = weight.to(dtype=torch.float32)
                centroid = compute.clone() if centroid is None else centroid.add(compute)
            if centroid is None:  # pragma: no cover
                raise RuntimeError("STUN cluster unexpectedly contains no experts.")
            centroid.div_(len(members))
            for local_index, expert_index in enumerate(members):
                compute = torch.as_tensor(
                    load_weight(key, expert_index),
                    device=centroid.device,
                ).detach().to(dtype=torch.float32)
                distances[local_index] += float(
                    (compute - centroid).square().sum().item()
                )
        representative = members[int(torch.argmin(distances).item())]
        selections.append(
            StunClusterSelection(
                members=members,
                representative=representative,
                reconstruct=reconstruct,
            )
        )
    return StunExpertPlan(
        source_experts=int(source_experts),
        clusters=tuple(selections),
    )


def apply_stun_plan(
    values: Any,
    plan: StunExpertPlan,
) -> Any:
    """Apply representative selection or centroid reconstruction on axis zero."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("STUN plan application requires torch.")
    source = torch.as_tensor(values).detach()
    if source.ndim < 1 or int(source.shape[0]) != plan.source_experts:
        raise ValueError(
            f"STUN source axis must contain {plan.source_experts} experts."
        )
    outputs = []
    for cluster in plan.clusters:
        if cluster.reconstruct:
            index = torch.as_tensor(
                cluster.members,
                device=source.device,
                dtype=torch.long,
            )
            outputs.append(source.index_select(0, index).mean(dim=0))
        else:
            outputs.append(source[int(cluster.representative)])
    return torch.stack(outputs, dim=0).contiguous()


__all__ = [
    "StunClusterSelection",
    "StunExpertPlan",
    "apply_stun_plan",
    "cluster_router_experts",
    "select_stun_representatives",
    "select_stun_representatives_streaming",
]
