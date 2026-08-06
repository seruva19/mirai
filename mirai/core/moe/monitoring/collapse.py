"""Minority-side routing-collapse diagnostics (single owner).

Companion to :mod:`mirai.core.moe.monitoring.health`, which already implements
the dominant-side and numerical signatures of the visual-DiT sparse-MoE routing
diagnosis (arXiv:2605.19378): expert-response homogenization
(``moe_expert_output_cossim``), single-expert deadlock duration by depth band
(``moe_deadlocked_layer_count*``), and the bf16 sub-ULP update trap
(``moe_router_underflow_fraction``). This module owns the part of that study's
instrumentation those estimators do not express -- the *minority* side of the
load distribution, which is the quantity the source study actually reports per
layer and per step:

    moe_minority_expert_share        least-used expert's share of routed slots
    moe_normalized_minority_share    the same share divided by the uniform share
    moe_dead_expert_fraction         experts that received no routed slot
    moe_underused_expert_fraction    experts below the health baseline
    moe_collapsed_router_fraction    routers under the baseline this step
    moe_max_collapse_duration        longest consecutive collapsed run, in steps
    moe_collapse_rebound_count       routers that crossed back above the baseline
    moe_collapse_plateau_fraction    collapsed routers that stopped moving

Why the minority side is a separate estimator, not a restatement of top-1
monopoly: ``DEADLOCK_MONOPOLY_THRESHOLD`` fires when one expert claims >=90% of a
layer's routed slots. That is reachable for the two-routed-expert configuration
the source study runs, and unreachable for a router with hundreds of experts and
top-k > 1, where the dominant share is bounded near ``top_k / num_experts`` while
almost every expert can still be dead. Both regimes are the same pathology; only
the minority-side statistic detects it in the wide-router regime.

Threshold provenance. The source study calls a layer healthy while the minority
expert holds at least 10% of the tokens against a uniform share of 50%, i.e. one
fifth of uniform, and reports the deepest observed layer at 7.33% with recovery
observed as a layer climbing back across the same 10% line. Expressing that
baseline as a *ratio to the uniform share* (:data:`COLLAPSE_HEALTH_RATIO`) is
what carries it to routers with a different expert count; the absolute 10% does
not. The plateau tolerance is the study's "<0.05% change" observation of a
deadlocked layer that no auxiliary-loss coefficient moved.

Everything here is detached telemetry: inputs are routing selections that have
already been made, no tensor returned by this module carries a graph, and no
estimator is on the loss path. State is per router and per step -- never per
token -- so memory is independent of sequence length and batch size.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# A router is "collapsed" when its least-used expert holds less than this
# fraction of the uniform per-expert share (arXiv:2605.19378 reports the 10%
# minority-expert health baseline against a 50% uniform share).
COLLAPSE_HEALTH_RATIO = 0.2
# Steps a router must stay collapsed before crossing back up counts as recovery
# rather than as step-to-step jitter around the baseline.
COLLAPSE_REBOUND_MIN_STEPS = 2
# Window (in observed steps) over which a collapsed router is checked for the
# "cannot be moved" plateau, and the normalized-share span that counts as flat.
COLLAPSE_PLATEAU_WINDOW = 8
COLLAPSE_PLATEAU_TOLERANCE = 1e-3


@dataclass(frozen=True)
class RouterLoadSignature:
    """Per-router load summary reduced from one step's routed slots.

    One instance describes one routing decision surface -- a layer for a
    single-router family, or a (layer, head) pair for a multi-head router. Its
    size is fixed: the token axis is fully reduced away before the values reach
    Python.

    ``shares`` are fractions of the router's routed slots (tokens x top-k), so
    they sum to one across experts and the uniform share is ``1/num_experts``.
    """

    num_experts: int
    routed_slots: int
    minority_share: float
    top1_share: float
    dead_expert_fraction: float
    underused_expert_fraction: float

    @property
    def normalized_minority_share(self) -> float:
        """Minority share relative to the uniform share (1.0 == balanced)."""
        return float(self.minority_share) * float(self.num_experts)

    def is_collapsed(self, health_ratio: float = COLLAPSE_HEALTH_RATIO) -> bool:
        """Whether the least-used expert sits below the health baseline."""
        return self.normalized_minority_share < float(health_ratio)


def router_expert_counts(indices: Any, *, num_experts: int) -> Any:
    """Routed-slot counts per router as ``[routers, num_experts]``.

    ``indices`` is a detached integer selection tensor whose FIRST axis
    enumerates routers and whose remaining axes are the token and top-k slots
    (``[routers, tokens, top_k]`` for a multi-head router, ``[tokens, top_k]``
    for a single one, which is treated as one router). The count matrix is built
    with a single ``bincount`` over a flattened ``router * num_experts + expert``
    axis, so the whole token population is reduced in one kernel.
    """
    if torch is None:  # pragma: no cover - torch-less environments
        raise RuntimeError("router load signatures require torch.")
    experts = int(num_experts)
    if experts < 2:
        raise ValueError("router load signatures require at least two experts.")
    selection = indices.detach().long()
    if selection.ndim < 2:
        raise ValueError(
            "routed selections must have a router axis and a slot axis."
        )
    if selection.ndim == 2:
        selection = selection.unsqueeze(0)
    routers = int(selection.shape[0])
    flat = selection.reshape(routers, -1)
    if int(flat.numel()) == 0:
        return flat.new_zeros((routers, experts))
    if bool((flat < 0).any()) or bool((flat >= experts).any()):
        raise ValueError(
            "routed expert ids must lie in [0, num_experts); the selection does "
            "not belong to a router with this expert count."
        )
    offsets = torch.arange(routers, device=flat.device).unsqueeze(1) * experts
    return torch.bincount(
        (flat + offsets).reshape(-1), minlength=routers * experts
    ).reshape(routers, experts)


def router_load_signatures(
    counts: Any,
    *,
    health_ratio: float = COLLAPSE_HEALTH_RATIO,
) -> tuple[RouterLoadSignature, ...]:
    """Reduce a ``[routers, num_experts]`` count matrix to per-router summaries.

    Every reduction runs on the count matrix's own device and only a
    ``[routers, 5]`` block of scalars is read back, so the host-visible cost per
    step is a function of the router count alone -- not of the token count, the
    batch, or the expert count.
    """
    if torch is None:  # pragma: no cover - torch-less environments
        raise RuntimeError("router load signatures require torch.")
    matrix = counts.detach()
    if matrix.ndim != 2:
        raise ValueError("expert counts must be a [routers, num_experts] matrix.")
    routers, experts = int(matrix.shape[0]), int(matrix.shape[1])
    if experts < 2:
        raise ValueError("router load signatures require at least two experts.")
    values = matrix.float()
    totals = values.sum(dim=1)
    safe_totals = totals.clamp_min(1.0)
    shares = values / safe_totals.unsqueeze(1)
    # Below this share an expert is "underused"; at ratio 1.0 it would be every
    # expert under uniform, at the paper's 0.2 it is a fifth of uniform.
    floor = float(health_ratio) / float(experts)
    reduced = torch.stack(
        (
            totals,
            shares.min(dim=1).values,
            shares.max(dim=1).values,
            (values <= 0).float().mean(dim=1),
            (shares < floor).float().mean(dim=1),
        ),
        dim=1,
    )
    rows = reduced.to("cpu", torch.float64).tolist()
    signatures = []
    for total, minority, top1, dead, underused in rows:
        observed = int(total)
        signatures.append(
            RouterLoadSignature(
                num_experts=experts,
                routed_slots=observed,
                minority_share=float(minority) if observed else 0.0,
                top1_share=float(top1) if observed else 0.0,
                dead_expert_fraction=float(dead),
                underused_expert_fraction=float(underused),
            )
        )
    if len(signatures) != routers:  # pragma: no cover - shape invariant
        raise RuntimeError("router reduction lost rows.")
    return tuple(signatures)


def selection_load_signatures(
    indices: Any,
    *,
    num_experts: int,
    health_ratio: float = COLLAPSE_HEALTH_RATIO,
) -> tuple[RouterLoadSignature, ...]:
    """``router_expert_counts`` followed by ``router_load_signatures``."""
    return router_load_signatures(
        router_expert_counts(indices, num_experts=num_experts),
        health_ratio=health_ratio,
    )


class RoutingCollapseTracker:
    """Cross-step minority-side collapse state, one small record per router.

    Per router the tracker keeps a collapsed-run counter, the length of the run
    that preceded the most recent recovery, and a bounded ring of recent
    normalized minority shares. Nothing else survives a step, so the resident
    state is ``O(routers x COLLAPSE_PLATEAU_WINDOW)`` regardless of how many
    tokens produced it.
    """

    def __init__(
        self,
        *,
        health_ratio: float = COLLAPSE_HEALTH_RATIO,
        rebound_min_steps: int = COLLAPSE_REBOUND_MIN_STEPS,
        plateau_window: int = COLLAPSE_PLATEAU_WINDOW,
        plateau_tolerance: float = COLLAPSE_PLATEAU_TOLERANCE,
    ) -> None:
        if not 0.0 < float(health_ratio) <= 1.0:
            raise ValueError("health_ratio must lie in (0, 1].")
        if int(plateau_window) < 2:
            raise ValueError("plateau_window must observe at least two steps.")
        self.health_ratio = float(health_ratio)
        self.rebound_min_steps = int(rebound_min_steps)
        self.plateau_window = int(plateau_window)
        self.plateau_tolerance = float(plateau_tolerance)
        self._streaks: dict[str, int] = {}
        self._history: dict[str, deque[float]] = {}

    @property
    def tracked_routers(self) -> int:
        return len(self._streaks)

    def state_size(self) -> int:
        """Number of scalars retained across steps (bounded-memory probe)."""
        return len(self._streaks) + sum(
            len(window) for window in self._history.values()
        )

    def update(
        self, signatures: Mapping[str, RouterLoadSignature]
    ) -> dict[str, float]:
        """Advance per-router state and return this step's collapse metrics.

        ``signatures`` maps a stable router key to its summary for this step.
        Routers absent from the mapping keep their state untouched: a family
        that observes a subset of layers per step does not falsely recover the
        rest. An empty mapping is "nothing measurable this step" and returns no
        metrics.
        """
        if not signatures:
            return {}
        minority: list[float] = []
        normalized: list[float] = []
        dead: list[float] = []
        underused: list[float] = []
        collapsed_keys: list[str] = []
        rebounds = 0
        plateaus = 0
        for key, signature in signatures.items():
            if not isinstance(signature, RouterLoadSignature):
                raise TypeError(
                    f"router '{key}' must report a RouterLoadSignature."
                )
            value = signature.normalized_minority_share
            minority.append(float(signature.minority_share))
            normalized.append(value)
            dead.append(float(signature.dead_expert_fraction))
            underused.append(float(signature.underused_expert_fraction))
            window = self._history.setdefault(
                key, deque(maxlen=self.plateau_window)
            )
            window.append(value)
            if signature.is_collapsed(self.health_ratio):
                collapsed_keys.append(key)
                self._streaks[key] = self._streaks.get(key, 0) + 1
                if len(window) == self.plateau_window and (
                    max(window) - min(window)
                ) <= self.plateau_tolerance:
                    plateaus += 1
            else:
                if self._streaks.get(key, 0) >= self.rebound_min_steps:
                    rebounds += 1
                self._streaks[key] = 0
        observed = float(len(signatures))
        collapsed = float(len(collapsed_keys))
        return {
            "moe_minority_expert_share": sum(minority) / observed,
            "moe_normalized_minority_share": sum(normalized) / observed,
            "moe_dead_expert_fraction": sum(dead) / observed,
            "moe_underused_expert_fraction": sum(underused) / observed,
            "moe_collapsed_router_fraction": collapsed / observed,
            "moe_collapsed_router_count": collapsed,
            "moe_worst_normalized_minority_share": min(normalized),
            "moe_max_collapse_duration": float(
                max(self._streaks.values()) if self._streaks else 0
            ),
            "moe_collapse_rebound_count": float(rebounds),
            "moe_collapse_plateau_fraction": (
                float(plateaus) / collapsed if collapsed else 0.0
            ),
        }


def collapse_metric_keys() -> tuple[str, ...]:
    """Every metric key :meth:`RoutingCollapseTracker.update` can emit."""
    return (
        "moe_minority_expert_share",
        "moe_normalized_minority_share",
        "moe_dead_expert_fraction",
        "moe_underused_expert_fraction",
        "moe_collapsed_router_fraction",
        "moe_collapsed_router_count",
        "moe_worst_normalized_minority_share",
        "moe_max_collapse_duration",
        "moe_collapse_rebound_count",
        "moe_collapse_plateau_fraction",
    )


def router_keys(layer: str, heads: Sequence[int] | int) -> tuple[str, ...]:
    """Stable per-head router keys under one layer name.

    Multi-head routers decide independently per head, so a head is the routing
    decision surface the metrics describe. The key keeps the layer prefix so
    depth ordering stays derivable from the model's own module order.
    """
    span = range(int(heads)) if isinstance(heads, int) else heads
    return tuple(f"{layer}#head{int(head)}" for head in span)


__all__ = [
    "COLLAPSE_HEALTH_RATIO",
    "COLLAPSE_PLATEAU_TOLERANCE",
    "COLLAPSE_PLATEAU_WINDOW",
    "COLLAPSE_REBOUND_MIN_STEPS",
    "RouterLoadSignature",
    "RoutingCollapseTracker",
    "collapse_metric_keys",
    "router_expert_counts",
    "router_keys",
    "router_load_signatures",
    "selection_load_signatures",
]
