"""Cross-timestep expert-branch feature reuse for sparse-MoE diffusion sampling.

Mechanism provenance: MoECa, "Expert-Aligned Feature Caching in Diffusion
Transformers" ([arXiv:2606.15615](https://arxiv.org/abs/2606.15615)). The
observation the implementation takes from that work is that cross-timestep
redundancy in a routed-expert transformer is carried by the individual expert
branches rather than by the combined token update, so the reusable unit is the
per-routed-slot expert feature and the invalidation signal is a change of that
slot's routed expert. No source from the paper's authors is used; the algorithm
below is written against Mirai's own MoE execution seam.

What is cached, per MoE layer and per cache slot:

* ``features`` -- the pre-combine expert output of every routed slot, laid out
  in the family's canonical slot order, shape ``[routed_slots, d]``.
* ``expert_ids`` -- the routed expert of every slot, the branch-level
  invalidation key.
* ``anchor`` -- the layer input that produced ``features``, used to measure
  drift against the next visit.

Reuse decision for one visit of one layer:

1. Entries whose tensor signature does not match the current call cannot be
   compared and force a full recompute.
2. Relative L2 drift of the layer input against the anchor gates reuse as a
   whole. Above ``drift_threshold`` the entry is invalidated and every branch is
   recomputed.
3. Within the drift gate, only the slots whose routed expert changed are
   recomputed; the remaining branch features are reused.
4. ``max_reuse_span`` bounds how many consecutive visits may reuse before a full
   recompute is forced, so error cannot accumulate without bound.
5. The combine always applies the *current* routing probabilities to the merged
   branch features, so a reused branch still tracks this step's gate weights.

The cache is lossy. The uncached backend execution is the reference path: with
``max_reuse_span=0`` the cache stays armed but never reuses, and the result is
bit-identical to the uncached backend because the merged feature buffer is the
same permutation round-trip the uncached path performs. A partial recompute
reproduces a full recompute of the same slots up to the row-count-dependent
blocking of the underlying GEMM, which is the only numerical difference reuse
does not account for.

Drift is a host-visible scalar, so each cached layer visit synchronizes the
compute device once. Cache residency is
``layers x slots x routed_slots x d`` elements for the features plus one layer
input per entry, in the layer's compute dtype and on the layer's device.

The cache never engages while autograd is enabled: training keeps the uncached
path and allocates no cache state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

EXPERT_FEATURE_CACHE_MODES = ("off", "branch")

_DRIFT_EPS = 1e-12


class ExpertFeatureCacheError(ValueError):
    """Raised when an expert-feature-cache policy cannot be honored."""


def normalize_expert_feature_cache_mode(value: Any) -> str:
    """Canonicalize a configured cache mode, rejecting unknown spellings."""
    text = str(value).strip().lower() or "off"
    if text not in EXPERT_FEATURE_CACHE_MODES:
        raise ExpertFeatureCacheError(
            "inference.expert_feature_cache must be one of "
            + ", ".join(f"'{mode}'" for mode in EXPERT_FEATURE_CACHE_MODES)
            + f"; got '{text}'."
        )
    return text


@dataclass(frozen=True)
class ExpertFeatureCachePolicy:
    """Typed control surface for cross-timestep expert-branch reuse."""

    mode: str = "off"
    drift_threshold: float = 0.05
    max_reuse_span: int = 2
    slots: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", normalize_expert_feature_cache_mode(self.mode)
        )
        drift = float(self.drift_threshold)
        if not 0.0 <= drift <= 1.0:
            raise ExpertFeatureCacheError(
                "inference.expert_feature_cache_drift_threshold must be between "
                f"0 and 1; got {drift}."
            )
        object.__setattr__(self, "drift_threshold", drift)
        span = int(self.max_reuse_span)
        if span < 0:
            raise ExpertFeatureCacheError(
                "inference.expert_feature_cache_max_reuse_span must be >= 0; "
                f"got {span}."
            )
        object.__setattr__(self, "max_reuse_span", span)
        slots = int(self.slots)
        if slots < 1:
            raise ExpertFeatureCacheError(
                f"inference.expert_feature_cache_slots must be >= 1; got {slots}."
            )
        object.__setattr__(self, "slots", slots)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


@runtime_checkable
class ExpertBranchPlan(Protocol):
    """One layer visit, decomposed into branch computation and combine.

    A family owns the routed layout behind this contract. ``expert_ids`` and the
    rows of the tensors ``compute_branch_features`` returns share one canonical
    slot order that is stable across visits of the same layer, which is what
    makes a per-slot expert comparison meaningful across timesteps.
    """

    @property
    def expert_ids(self) -> torch.Tensor:
        """Routed expert of every slot, ``[routed_slots]`` integer."""

    def compute_branch_features(
        self, slot_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Pre-combine expert outputs, ``[routed_slots, d]``.

        ``slot_mask`` selects the slots to compute; unselected rows are zero and
        must be overwritten by the caller. ``None`` computes every slot.
        """

    def combine_branch_features(self, features: torch.Tensor) -> torch.Tensor:
        """Routing-weighted combine of ``features`` into the layer output."""


@runtime_checkable
class ExpertBranchExecutor(Protocol):
    """MoE execution backend that can also expose its branch decomposition."""

    def execute(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Uncached reference execution of one layer visit."""

    def plan_branches(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> ExpertBranchPlan:
        """Decompose one layer visit without computing any expert branch."""


@dataclass
class ExpertFeatureCacheLayerStats:
    """Per-layer reuse accounting for one sampling run."""

    visits: int = 0
    full_recomputes: int = 0
    reuse_visits: int = 0
    reused_branches: int = 0
    recomputed_branches: int = 0
    signature_invalidations: int = 0
    drift_invalidations: int = 0
    span_invalidations: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "visits": int(self.visits),
            "full_recomputes": int(self.full_recomputes),
            "reuse_visits": int(self.reuse_visits),
            "reused_branches": int(self.reused_branches),
            "recomputed_branches": int(self.recomputed_branches),
            "signature_invalidations": int(self.signature_invalidations),
            "drift_invalidations": int(self.drift_invalidations),
            "span_invalidations": int(self.span_invalidations),
        }


@dataclass
class ExpertFeatureCacheTelemetry:
    """Run-report view of what the cache reused and what it recomputed."""

    layers: dict[str, ExpertFeatureCacheLayerStats] = field(default_factory=dict)

    def layer(self, key: str) -> ExpertFeatureCacheLayerStats:
        stats = self.layers.get(key)
        if stats is None:
            stats = ExpertFeatureCacheLayerStats()
            self.layers[key] = stats
        return stats

    def reset(self) -> None:
        self.layers.clear()

    def snapshot(self) -> dict[str, Any]:
        """Aggregate counters plus the per-layer breakdown, JSON-ready."""
        totals = ExpertFeatureCacheLayerStats()
        for stats in self.layers.values():
            totals.visits += stats.visits
            totals.full_recomputes += stats.full_recomputes
            totals.reuse_visits += stats.reuse_visits
            totals.reused_branches += stats.reused_branches
            totals.recomputed_branches += stats.recomputed_branches
            totals.signature_invalidations += stats.signature_invalidations
            totals.drift_invalidations += stats.drift_invalidations
            totals.span_invalidations += stats.span_invalidations
        payload = totals.as_dict()
        branch_total = totals.reused_branches + totals.recomputed_branches
        payload["branch_reuse_ratio"] = (
            float(totals.reused_branches) / float(branch_total)
            if branch_total
            else 0.0
        )
        payload["layers"] = {
            key: stats.as_dict() for key, stats in sorted(self.layers.items())
        }
        return payload


@dataclass
class _CacheEntry:
    signature: tuple[Any, ...]
    features: torch.Tensor
    expert_ids: torch.Tensor
    anchor: torch.Tensor
    reuse_run: int = 0
    age: int = 0


def _entry_signature(
    x_heads: torch.Tensor, topk_indices: torch.Tensor
) -> tuple[Any, ...]:
    return (
        tuple(int(dim) for dim in x_heads.shape),
        str(x_heads.dtype),
        str(x_heads.device),
        tuple(int(dim) for dim in topk_indices.shape),
    )


def _relative_drift(current: torch.Tensor, anchor: torch.Tensor) -> float:
    # One stacked transfer, so a cached layer visit costs a single device sync.
    norms = torch.stack(
        (
            torch.linalg.vector_norm((current - anchor).float()),
            torch.linalg.vector_norm(anchor.float()),
        )
    ).tolist()
    return float(norms[0]) / (float(norms[1]) + _DRIFT_EPS)


class ExpertFeatureCache:
    """Owner of cross-timestep branch feature state and its reuse decision.

    One instance spans a sampling run: it is keyed by layer identity, so every
    MoE layer of the model shares it and the run report aggregates over layers.
    ``reset`` drops all state, which a caller performs between generations.
    """

    def __init__(
        self,
        policy: ExpertFeatureCachePolicy,
        telemetry: ExpertFeatureCacheTelemetry | None = None,
    ) -> None:
        self.policy = policy
        self.telemetry = telemetry if telemetry is not None else ExpertFeatureCacheTelemetry()
        self._entries: dict[int, list[_CacheEntry]] = {}
        self._labels: dict[int, str] = {}

    @property
    def enabled(self) -> bool:
        return self.policy.enabled

    def reset(self) -> None:
        """Drop every cached branch buffer and every counter."""
        self._entries.clear()
        self._labels.clear()
        self.telemetry.reset()

    @property
    def resident_entries(self) -> int:
        """Number of live cache entries; zero whenever the cache never engaged."""
        return sum(len(pool) for pool in self._entries.values())

    def _layer_key(self, module: Any) -> tuple[int, str]:
        key = id(module)
        label = self._labels.get(key)
        if label is None:
            label = f"{type(module).__name__}#{len(self._labels)}"
            self._labels[key] = label
        return key, label

    def execute(
        self,
        executor: ExpertBranchExecutor,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run one layer visit, reusing stable expert branches where allowed."""
        if not self.policy.enabled or torch.is_grad_enabled():
            return executor.execute(module, x_heads, topk_probs, topk_indices)

        key, label = self._layer_key(module)
        stats = self.telemetry.layer(label)
        stats.visits += 1

        plan = executor.plan_branches(module, x_heads, topk_probs, topk_indices)
        expert_ids = plan.expert_ids
        signature = _entry_signature(x_heads, topk_indices)
        pool = self._entries.setdefault(key, [])
        for entry in pool:
            entry.age += 1

        candidate = self._select_candidate(pool, signature, x_heads)
        if candidate is None:
            if pool and not any(entry.signature == signature for entry in pool):
                stats.signature_invalidations += 1
            features = self._full_recompute(plan, stats, expert_ids)
            self._store(pool, signature, features, expert_ids, x_heads, reuse_run=0)
            return plan.combine_branch_features(features)

        entry, drift = candidate
        if drift > self.policy.drift_threshold:
            stats.drift_invalidations += 1
            features = self._full_recompute(plan, stats, expert_ids)
            if len(pool) < self.policy.slots:
                # A visit too far from every held entry is a different
                # trajectory through this layer (sequential CFG produces two),
                # so it takes its own slot instead of evicting the closest one.
                self._store(
                    pool, signature, features, expert_ids, x_heads, reuse_run=0
                )
            else:
                self._refresh(
                    entry, signature, features, expert_ids, x_heads, reuse_run=0
                )
            return plan.combine_branch_features(features)
        if entry.reuse_run >= self.policy.max_reuse_span:
            stats.span_invalidations += 1
            features = self._full_recompute(plan, stats, expert_ids)
            self._refresh(entry, signature, features, expert_ids, x_heads, reuse_run=0)
            return plan.combine_branch_features(features)

        changed = expert_ids != entry.expert_ids
        changed_count = int(changed.sum().item())
        total = int(expert_ids.numel())
        stats.reuse_visits += 1
        stats.recomputed_branches += changed_count
        stats.reused_branches += total - changed_count
        if changed_count == 0:
            features = entry.features
        else:
            recomputed = plan.compute_branch_features(changed)
            features = torch.where(
                changed.reshape(-1, *((1,) * (recomputed.ndim - 1))),
                recomputed,
                entry.features,
            )
        self._refresh(
            entry,
            signature,
            features,
            expert_ids,
            x_heads,
            reuse_run=entry.reuse_run + 1,
        )
        return plan.combine_branch_features(features)

    def _full_recompute(
        self,
        plan: ExpertBranchPlan,
        stats: ExpertFeatureCacheLayerStats,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        stats.full_recomputes += 1
        stats.recomputed_branches += int(expert_ids.numel())
        return plan.compute_branch_features(None)

    def _select_candidate(
        self,
        pool: list[_CacheEntry],
        signature: tuple[Any, ...],
        x_heads: torch.Tensor,
    ) -> tuple[_CacheEntry, float] | None:
        """Best-matching comparable entry and its drift, or ``None``.

        Sequential classifier-free guidance visits one layer twice per timestep
        with unrelated inputs, so the comparison is against the closest entry
        rather than the most recent one; ``slots`` sizes the pool that keeps
        those interleaved trajectories apart.
        """
        best: tuple[_CacheEntry, float] | None = None
        for entry in pool:
            if entry.signature != signature:
                continue
            drift = _relative_drift(x_heads, entry.anchor)
            if best is None or drift < best[1]:
                best = (entry, drift)
        return best

    def _store(
        self,
        pool: list[_CacheEntry],
        signature: tuple[Any, ...],
        features: torch.Tensor,
        expert_ids: torch.Tensor,
        x_heads: torch.Tensor,
        *,
        reuse_run: int,
    ) -> None:
        while len(pool) >= self.policy.slots:
            oldest = max(range(len(pool)), key=lambda index: pool[index].age)
            pool.pop(oldest)
        pool.append(
            _CacheEntry(
                signature=signature,
                features=features.detach().clone(),
                expert_ids=expert_ids.detach().clone(),
                anchor=x_heads.detach().clone(),
                reuse_run=reuse_run,
                age=0,
            )
        )

    @staticmethod
    def _refresh(
        entry: _CacheEntry,
        signature: tuple[Any, ...],
        features: torch.Tensor,
        expert_ids: torch.Tensor,
        x_heads: torch.Tensor,
        *,
        reuse_run: int,
    ) -> None:
        entry.signature = signature
        entry.features = features.detach().clone()
        entry.expert_ids = expert_ids.detach().clone()
        entry.anchor = x_heads.detach().clone()
        entry.reuse_run = reuse_run
        entry.age = 0


class CachedExpertExecution:
    """MoE kernel-backend decorator that adds cross-timestep branch reuse.

    The decorated backend keeps every obligation of the seam it replaces: layout
    inspection and explicit-selection validation are forwarded, and the cached
    path calls back into the same backend for every expert matmul it does not
    reuse.
    """

    name = "expert_feature_cache"

    def __init__(
        self, inner: ExpertBranchExecutor, cache: ExpertFeatureCache
    ) -> None:
        if not hasattr(inner, "plan_branches"):
            raise ExpertFeatureCacheError(
                "inference.expert_feature_cache requires an expert-execution "
                f"backend that exposes its branch decomposition; "
                f"{type(inner).__name__} does not."
            )
        self.inner = inner
        self.cache = cache

    def inspect_expert_layout(self, module: Any) -> tuple[str, ...]:
        return self.inner.inspect_expert_layout(module)  # type: ignore[attr-defined]

    def validate_explicit_alignment(self) -> None:
        self.inner.validate_explicit_alignment()  # type: ignore[attr-defined]

    def execute(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.cache.execute(
            self.inner, module, x_heads, topk_probs, topk_indices
        )


__all__ = [
    "EXPERT_FEATURE_CACHE_MODES",
    "CachedExpertExecution",
    "ExpertBranchExecutor",
    "ExpertBranchPlan",
    "ExpertFeatureCache",
    "ExpertFeatureCacheError",
    "ExpertFeatureCacheLayerStats",
    "ExpertFeatureCachePolicy",
    "ExpertFeatureCacheTelemetry",
    "normalize_expert_feature_cache_mode",
]
