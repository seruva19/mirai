"""Routing-guided expert selection for LoRA (MoE-Sieve).

Profiles which routed experts are "hot" during a calibration pass and restricts
the per-expert LoRA adapters to only those experts. The frozen base experts are
untouched; only the trainable adapter slices shrink. The mechanism is a 0/1
``active_expert_mask`` on each :class:`ActiveExpertLoRA` (see compressed_weights.py) that
zeroes the adapter output for unselected experts -- which also makes their
gradients exactly zero, so a paged optimizer applies no update to them.

Router (``ffn.router.weight``) and attention LoRA are never affected here; only
the routed-expert adapters (``ffn.experts.w1/w2/w3``) are masked.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterator

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_EXPERT_LORA_KEYS = ("w1", "w2", "w3")


@dataclass(frozen=True)
class LayerExpertSelection:
    """Deterministic selection outcome for one MoE layer."""

    layer: str
    num_experts: int
    num_selected: int
    selected_expert_ids: tuple[int, ...]
    routed_tokens: int
    covered_token_fraction: float

    @property
    def selection_hash(self) -> str:
        payload = ",".join(str(i) for i in self.selected_expert_ids)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class ExpertSelectionReport:
    """Aggregate report over all calibrated MoE layers."""

    fraction: float
    calibration_steps: int
    layers: list[LayerExpertSelection] = field(default_factory=list)
    expert_adapter_params_total: int = 0
    expert_adapter_params_active: int = 0
    trainable_params_before: int = 0
    trainable_params_after_effective: int = 0

    def mean_covered_token_fraction(self) -> float:
        if not self.layers:
            return 0.0
        return sum(l.covered_token_fraction for l in self.layers) / len(self.layers)

    def selection_overlap(self) -> float:
        """Mean pairwise Jaccard overlap of selected sets across layers."""
        sets = [set(l.selected_expert_ids) for l in self.layers]
        if len(sets) < 2:
            return 1.0 if sets else 0.0
        total = 0.0
        pairs = 0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                if not union:
                    continue
                total += len(sets[i] & sets[j]) / len(union)
                pairs += 1
        return total / pairs if pairs else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "calibration_steps": self.calibration_steps,
            "layers": len(self.layers),
            "expert_adapter_params_total": self.expert_adapter_params_total,
            "expert_adapter_params_active": self.expert_adapter_params_active,
            "trainable_params_before": self.trainable_params_before,
            "trainable_params_after_effective": self.trainable_params_after_effective,
            "mean_covered_token_fraction": self.mean_covered_token_fraction(),
            "mean_selection_overlap": self.selection_overlap(),
            "layers_detail": [
                {
                    "layer": l.layer,
                    "num_experts": l.num_experts,
                    "num_selected": l.num_selected,
                    "routed_tokens": l.routed_tokens,
                    "covered_token_fraction": l.covered_token_fraction,
                    "selection_hash": l.selection_hash,
                    "selected_expert_ids": list(l.selected_expert_ids),
                }
                for l in self.layers
            ],
        }


def iter_expert_lora_blocks(model: Any) -> Iterator[tuple[str, Any, Any]]:
    """Yield ``(layer_name, router, experts)`` for MoE blocks with expert LoRA.

    A block qualifies when it exposes a ``router`` (with ``num_experts``) and an
    ``experts`` submodule carrying a non-empty ``expert_lora`` module dict.
    """
    for name, module in model.named_modules():
        router = getattr(module, "router", None)
        experts = getattr(module, "experts", None)
        if router is None or experts is None:
            continue
        if not hasattr(router, "num_experts"):
            continue
        expert_lora = getattr(experts, "expert_lora", None)
        if expert_lora is None or len(expert_lora) == 0:
            continue
        yield name, router, experts


def accumulate_routing_counts(
    model: Any,
    accumulator: dict[str, Any],
) -> dict[str, Any]:
    """Add the current per-layer routed-expert counts into ``accumulator``.

    Reads ``router.last_top_indices`` (populated by the router on every forward),
    reusing the existing routing-stats plumbing rather than adding new hooks. Call
    this while the routing state is still live (before any clear).
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert selection requires torch.")
    for name, router, _experts in iter_expert_lora_blocks(model):
        top = getattr(router, "last_top_indices", None)
        if top is None:
            continue
        _add_counts(accumulator, name, top, int(router.num_experts))
    return accumulator


def _add_counts(
    accumulator: dict[str, Any], name: str, indices: Any, num_experts: int
) -> None:
    counts = torch.bincount(
        indices.detach().reshape(-1).to(torch.long),
        minlength=num_experts,
    ).detach().to(device="cpu", dtype=torch.float64)
    prior = accumulator.get(name)
    accumulator[name] = counts if prior is None else prior + counts


def register_calibration_hooks(
    model: Any, accumulator: dict[str, Any]
) -> list[Any]:
    """Register temporary router forward hooks that accumulate routed counts.

    Needed because the pipeline forward clears ``last_top_indices`` at the end of
    every step; the hook fires right after each router's own forward, while the
    routing tensor is still live. Returns removable handles -- the caller MUST
    remove them once calibration is done. This reads the exact same
    ``last_top_indices`` tensor the routing-stats plumbing uses; it adds no
    persistent hooks and mutates no module state.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert selection requires torch.")
    handles: list[Any] = []
    for name, router, _experts in iter_expert_lora_blocks(model):
        num_experts = int(router.num_experts)

        def _hook(module, _inp, _out, _name=name, _num=num_experts):
            top = getattr(module, "last_top_indices", None)
            if top is not None:
                _add_counts(accumulator, _name, top, _num)

        handles.append(router.register_forward_hook(_hook))
    return handles


def select_hot_experts(counts: Any, fraction: float) -> list[int]:
    """Return the ``ceil(fraction*E)`` hottest expert ids, deterministically.

    Ties are broken by ascending expert id so identical calibration data yields
    an identical selection regardless of platform sort stability.
    """
    num_experts = int(counts.numel())
    if num_experts == 0:
        return []
    k = max(1, math.ceil(float(fraction) * num_experts))
    k = min(k, num_experts)
    values = [float(v) for v in counts.tolist()]
    order = sorted(range(num_experts), key=lambda i: (-values[i], i))
    return sorted(order[:k])


def _expert_adapter_param_numel(experts: Any) -> int:
    total = 0
    expert_lora = getattr(experts, "expert_lora", {})
    for key in _EXPERT_LORA_KEYS:
        adapter = expert_lora[key] if key in expert_lora else None
        if adapter is None:
            continue
        total += int(adapter.lora_a.numel()) + int(adapter.lora_b.numel())
    return total


def _count_trainable_params(model: Any) -> int:
    return int(
        sum(int(p.numel()) for p in model.parameters() if bool(p.requires_grad))
    )


def apply_routing_topk_selection(
    model: Any,
    counts_by_layer: dict[str, Any],
    *,
    fraction: float,
    calibration_steps: int = 0,
) -> ExpertSelectionReport:
    """Set the ``active_expert_mask`` on every expert adapter from calibration.

    ``model`` must be the transformer (or a parent) that owns the MoE blocks.
    Returns a deterministic :class:`ExpertSelectionReport`.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert selection requires torch.")
    report = ExpertSelectionReport(
        fraction=float(fraction), calibration_steps=int(calibration_steps)
    )
    report.trainable_params_before = _count_trainable_params(model)
    active_params = 0
    total_params = 0
    for name, router, experts in iter_expert_lora_blocks(model):
        num_experts = int(router.num_experts)
        counts = counts_by_layer.get(name)
        if counts is None:
            counts = torch.zeros(num_experts, dtype=torch.float64)
        selected = select_hot_experts(counts, fraction)
        for key in _EXPERT_LORA_KEYS:
            if key in experts.expert_lora:
                experts.expert_lora[key].set_active_experts(selected)
        per_expert_numel = _expert_adapter_param_numel(experts)
        per_one = per_expert_numel / num_experts if num_experts else 0.0
        total_params += per_expert_numel
        active_params += int(round(per_one * len(selected)))
        routed = float(counts.sum().item())
        covered = (
            float(counts[selected].sum().item()) / routed if routed > 0 else 0.0
        )
        report.layers.append(
            LayerExpertSelection(
                layer=name,
                num_experts=num_experts,
                num_selected=len(selected),
                selected_expert_ids=tuple(selected),
                routed_tokens=int(routed),
                covered_token_fraction=covered,
            )
        )
    report.expert_adapter_params_total = int(total_params)
    report.expert_adapter_params_active = int(active_params)
    # Raw requires_grad count is unchanged (adapters keep whole-tensor params for
    # checkpoint compat); the *effective* updated count drops the masked slices.
    report.trainable_params_after_effective = int(
        report.trainable_params_before - (total_params - active_params)
    )
    return report
