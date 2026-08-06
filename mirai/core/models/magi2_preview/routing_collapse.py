"""MAGI-2 binding for model-agnostic routing-collapse telemetry.

MAGI-2's router is aux-loss free: the vendored ``CoreMultiHeadMoE`` computes
sigmoid scores, shifts them by a learned per-expert bias for selection only, and
emits no balance loss, no z-loss, and no routing statistic. Nothing in the
family reports whether its 12 per-layer heads keep their 256 experts alive, so
the ``attn_router`` adapter preset trains the router projection with no
observable routing signal at all.

This module supplies that signal without editing the vendored runtime. The
family already exposes one seam that sees a completed routing decision -- the
optional expert-execution backend consulted by ``CoreMultiHeadMoE._forward_impl``
as ``self._mirai_moe_kernel_backend.execute(module, x_heads, topk_probs,
topk_indices)``. :class:`Magi2RoutingCollapseTap` decorates that seam: it reads
``topk_indices`` under ``no_grad``, reduces it to one
:class:`~mirai.core.moe.monitoring.collapse.RouterLoadSignature` per head, and
then delegates execution unchanged. With no expert-execution backend configured
the tap reproduces the vendored selection between the reference per-expert loop
and the fused kernel, which
``mirai/core/models/magi2_preview/contracts`` pins against an untapped forward.

Aggregation is per (layer, head): 36 MoE layers x 12 heads is 432 routing
decision surfaces, each reduced to five scalars on device before anything is
read back. The token axis and the 256-wide expert axis are both reduced away
inside the reduction, so neither the sequence length nor the expert count
reaches host memory.

Gated by ``model.params.moe_routing_health`` (default ``False``). When the gate
is off no tap is constructed, nothing is attached to the vendored layers, and
this module is never imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mirai.core.models.magi2_preview.quantized_experts import iter_magi2_moe_layers
from mirai.core.moe.monitoring.collapse import (
    COLLAPSE_HEALTH_RATIO,
    RouterLoadSignature,
    RoutingCollapseTracker,
    router_keys,
    selection_load_signatures,
)
from mirai.core.moe.monitoring.health import DeadlockTracker


# Attribute holding the tap on a vendored MoE layer, so attach is idempotent and
# detach can restore exactly the backend the tap wrapped.
MAGI2_COLLAPSE_TAP_ATTR = "_mirai_routing_collapse_tap"


class Magi2RoutingCollapseObserver:
    """Detached sink for per-(layer, head) routing signatures of one forward.

    The observer holds at most one signature per router key: a second forward
    before :meth:`take` overwrites the first rather than accumulating, which is
    what keeps the sink's size a function of the model and not of the number of
    forwards between metric collections.
    """

    def __init__(self, *, health_ratio: float = COLLAPSE_HEALTH_RATIO) -> None:
        self.health_ratio = float(health_ratio)
        self._signatures: dict[str, RouterLoadSignature] = {}
        self._layer_order: list[str] = []
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Silence the tap without detaching it (eval forwards do not observe)."""
        self._enabled = bool(enabled)
        if not self._enabled:
            self._signatures.clear()

    def register_layer(self, layer_key: str, *, num_heads: int) -> None:
        """Declare a layer's router keys in shallow-to-deep attach order."""
        for key in router_keys(layer_key, int(num_heads)):
            if key not in self._layer_order:
                self._layer_order.append(key)

    @property
    def router_order(self) -> tuple[str, ...]:
        """Every registered router key, shallow to deep, heads in head order."""
        return tuple(self._layer_order)

    def observe(self, layer_key: str, topk_indices: Any, *, num_experts: int) -> None:
        """Reduce one layer's routed selection into per-head signatures."""
        if not self._enabled:
            return
        with torch.no_grad():
            signatures = selection_load_signatures(
                topk_indices,
                num_experts=int(num_experts),
                health_ratio=self.health_ratio,
            )
        keys = router_keys(layer_key, len(signatures))
        for key, signature in zip(keys, signatures):
            self._signatures[key] = signature

    def take(self) -> dict[str, RouterLoadSignature]:
        """Consume the signatures observed since the previous call."""
        signatures, self._signatures = self._signatures, {}
        return signatures

    def clear(self) -> None:
        self._signatures.clear()


class Magi2RoutingCollapseTap:
    """Monitoring decorator over the MAGI-2 expert-execution seam.

    ``inner`` is the expert-execution backend this tap wrapped, or ``None`` when
    the family runs its own execution. Execution is never altered: the tap
    observes and then hands the identical operands to whichever path the
    untapped layer would have taken.
    """

    name = "routing_collapse_tap"

    def __init__(
        self,
        observer: Magi2RoutingCollapseObserver,
        *,
        layer_key: str,
        inner: Any | None = None,
    ) -> None:
        self.observer = observer
        self.layer_key = str(layer_key)
        self.inner = inner

    def execute(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        self.observer.observe(
            self.layer_key, topk_indices, num_experts=int(module.num_experts)
        )
        if self.inner is not None:
            return self.inner.execute(module, x_heads, topk_probs, topk_indices)
        # Mirror of the backend-less branch of ``CoreMultiHeadMoE._forward_impl``:
        # the fused kernel needs CUDA and no autograd graph, otherwise the
        # vendored per-expert loop is the reference path.
        if torch.is_grad_enabled() or x_heads.device.type != "cuda":
            return module._torch_forward(x_heads, topk_probs, topk_indices)
        return module._flash_forward(x_heads, topk_probs, topk_indices)


def attach_routing_collapse_tap(
    transformer: Any, observer: Magi2RoutingCollapseObserver
) -> int:
    """Wrap every vendored MoE layer's execution seam with a monitoring tap.

    Re-attaching over an existing tap rebinds it to the backend currently in
    place instead of nesting taps, so a policy change that swaps the execution
    backend keeps exactly one observation per layer.
    """
    attached = 0
    for name, module in iter_magi2_moe_layers(transformer):
        existing = getattr(module, MAGI2_COLLAPSE_TAP_ATTR, None)
        current = module._mirai_moe_kernel_backend
        inner = current.inner if isinstance(current, Magi2RoutingCollapseTap) else current
        if isinstance(existing, Magi2RoutingCollapseTap):
            existing.observer = observer
            existing.inner = inner
            tap = existing
        else:
            tap = Magi2RoutingCollapseTap(observer, layer_key=name, inner=inner)
            setattr(module, MAGI2_COLLAPSE_TAP_ATTR, tap)
        module._mirai_moe_kernel_backend = tap
        observer.register_layer(name, num_heads=int(module.num_heads))
        attached += 1
    return attached


def detach_routing_collapse_tap(transformer: Any) -> int:
    """Restore the pre-tap execution seam on every vendored MoE layer."""
    detached = 0
    for _name, module in iter_magi2_moe_layers(transformer):
        tap = getattr(module, MAGI2_COLLAPSE_TAP_ATTR, None)
        if not isinstance(tap, Magi2RoutingCollapseTap):
            continue
        if module._mirai_moe_kernel_backend is tap:
            module._mirai_moe_kernel_backend = tap.inner
        delattr(module, MAGI2_COLLAPSE_TAP_ATTR)
        detached += 1
    return detached


@dataclass
class Magi2RoutingCollapseState:
    """The whole opt-in telemetry surface of one MAGI-2 pipeline.

    Holding the three pieces together is what lets the pipeline treat the
    feature as present-or-absent: when the gate is off the attribute is ``None``
    and no observer, tracker, or tap exists anywhere.
    """

    observer: Magi2RoutingCollapseObserver
    collapse_tracker: RoutingCollapseTracker
    deadlock_tracker: DeadlockTracker

    @classmethod
    def create(cls, *, health_ratio: float = COLLAPSE_HEALTH_RATIO) -> "Magi2RoutingCollapseState":
        return cls(
            observer=Magi2RoutingCollapseObserver(health_ratio=health_ratio),
            collapse_tracker=RoutingCollapseTracker(health_ratio=health_ratio),
            deadlock_tracker=DeadlockTracker(),
        )

    def collect(self) -> dict[str, float]:
        return collect_magi2_routing_collapse(
            self.observer,
            collapse_tracker=self.collapse_tracker,
            deadlock_tracker=self.deadlock_tracker,
        )


def collect_magi2_routing_collapse(
    observer: Magi2RoutingCollapseObserver,
    *,
    collapse_tracker: RoutingCollapseTracker,
    deadlock_tracker: DeadlockTracker | None = None,
) -> dict[str, float]:
    """Turn one forward's observations into per-step routing-collapse metrics.

    The dominant-side deadlock counters from
    :mod:`mirai.core.moe.monitoring.health` are produced from the same
    observation when ``deadlock_tracker`` is supplied, so the depth-banded
    U-shape view of the source study is available for this family without a
    second traversal. Returns an empty mapping when the forward observed no
    router (an eval forward, or a silenced observer).
    """
    signatures = observer.take()
    if not signatures:
        return {}
    metrics = dict(collapse_tracker.update(signatures))
    if deadlock_tracker is not None:
        order = observer.router_order or tuple(signatures)
        metrics.update(
            deadlock_tracker.update(
                {key: signature.top1_share for key, signature in signatures.items()},
                layer_order=order,
            )
        )
    return metrics


__all__ = [
    "MAGI2_COLLAPSE_TAP_ATTR",
    "Magi2RoutingCollapseObserver",
    "Magi2RoutingCollapseState",
    "Magi2RoutingCollapseTap",
    "attach_routing_collapse_tap",
    "collect_magi2_routing_collapse",
    "detach_routing_collapse_tap",
]
