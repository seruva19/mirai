"""MoE routing-health diagnostics (single owner).

Streaming, detached, default-off diagnostics that extend the shipped
routing-stability surface (entropy / utilization-CV / top1-monopoly /
KL-vs-step0 / unique-experts) with three collapse/homogenization alarms drawn
from the visual-DiT sparse-MoE routing-diagnosis study (arXiv:2605.19378):

    moe_expert_output_cossim       expert-response homogenization ("global soft
                                   saturation": experts collapse to cossim ~1)
    moe_max_deadlock_duration      longest consecutive single-expert deadlock run
    moe_deadlocked_layer_count     layers currently in a single-expert deadlock
                                   ("selective deadlock", U-shaped in depth)
    moe_*_depth_q{1..4}            the same deadlock counters by depth quartile
    moe_router_underflow_fraction  bf16 sub-ULP router-update truncation alarm

Everything here is gated behind ``model.params.moe_routing_health`` (default
``False``). When disabled, the diagnostics do not run or change outputs and
gradients. Inputs are detached router state or gradients; the diagnostics do not
extend the loss graph.

Estimator definitions (see CONFIG_REFERENCE.md for the documented fields):

* ``moe_expert_output_cossim`` is a *proxy*. Exact per-token pairwise cosine
  between expert OUTPUTS is not available: the sparse top-k dispatch only
  materializes each token's own top-k expert outputs, so the all-pairs
  "same tokens through every expert" quantity would require re-running experts.
  Instead we sample ``K`` token slots per step and, per MoE layer, take each
  expert's response column (its detached router affinity across those tokens),
  mean-center it, and average the pairwise cosine similarity between expert
  columns. Homogenized experts respond identically to the same tokens, driving
  the mean-centered cosine toward 1 — the same collapse signature the paper
  reports in output space, observed one step upstream in router-response space.

* ``moe_router_underflow_fraction`` is telemetry only. It reports router weights
  whose intended bf16 update is truncated to zero.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# A layer is "deadlocked" for a step when its top-1 expert claims at least this
# fraction of the routed tokens (routing collapsed onto one dominant expert).
DEADLOCK_MONOPOLY_THRESHOLD = 0.90
# bf16 keeps 7 explicit mantissa bits; ULP at a normal value 2**e is 2**(e-7).
_BF16_MANTISSA_BITS = 7
# Default token-slot sample budget for the cossim estimator.
DEFAULT_COSSIM_SAMPLE_SLOTS = 256
# Quartiles keep the diagnostic model-agnostic while making the shallow/deep
# concentration reported by the source study directly observable.
DEADLOCK_DEPTH_BANDS = ("q1", "q2", "q3", "q4")


def expert_output_cossim(
    score_columns: list[Any],
    *,
    sample_slots: int = DEFAULT_COSSIM_SAMPLE_SLOTS,
    generator: Any = None,
) -> float | None:
    """Mean-centered cosine homogenization proxy over per-layer response columns.

    ``score_columns`` is a list of detached ``[tokens, num_experts]`` router
    score tensors (one per MoE layer). Returns the layer-mean of the average
    pairwise mean-centered cosine similarity between expert columns, estimated
    over at most ``sample_slots`` sampled token rows. ``None`` when nothing is
    measurable (no layer with >= 2 experts and >= 2 tokens).
    """
    if torch is None:
        return None
    per_layer: list[float] = []
    for scores in score_columns:
        if scores is None or not torch.is_tensor(scores):
            continue
        matrix = scores.detach().float()
        if matrix.ndim != 2:
            matrix = matrix.reshape(-1, matrix.shape[-1])
        tokens, num_experts = matrix.shape
        if tokens < 2 or num_experts < 2:
            continue
        if tokens > int(sample_slots) > 0:
            idx = torch.randperm(tokens, generator=generator, device=matrix.device)[
                : int(sample_slots)
            ]
            matrix = matrix.index_select(0, idx)
        # Columns = per-expert response over the sampled tokens. Mean-center so
        # the metric measures shared response *shape*, not the shared positive
        # offset every softmax column carries (which would inflate cosine).
        columns = matrix.t()  # [num_experts, sampled_tokens]
        columns = columns - columns.mean(dim=1, keepdim=True)
        norms = columns.norm(dim=1)
        keep = norms > 1e-12
        if int(keep.sum().item()) < 2:
            continue
        columns = columns[keep]
        norms = norms[keep]
        unit = columns / norms.unsqueeze(1)
        sim = unit @ unit.t()
        kept = int(unit.shape[0])
        # Mean over the strict upper triangle (unique unordered expert pairs).
        pair_sum = float((sim.sum() - sim.diagonal().sum()).item()) / 2.0
        pair_count = kept * (kept - 1) / 2.0
        if pair_count > 0.0:
            per_layer.append(pair_sum / pair_count)
    if not per_layer:
        return None
    return float(sum(per_layer) / len(per_layer))


class DeadlockTracker:
    """Cross-step per-layer single-expert deadlock-duration state.

    Mirrors how ``moe_routing_kl_vs_step0`` keeps a small piece of cross-step
    state on the pipeline: one persistent counter per MoE layer. Each call to
    :meth:`update` advances the counter for every layer whose top-1 monopoly is
    at/above :data:`DEADLOCK_MONOPOLY_THRESHOLD` and resets the rest.
    """

    def __init__(self, threshold: float = DEADLOCK_MONOPOLY_THRESHOLD) -> None:
        self._threshold = float(threshold)
        self._streaks: dict[str, int] = {}

    def update(
        self,
        monopoly_by_layer: dict[str, float],
        *,
        layer_order: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, float]:
        """Advance streaks and return global and depth-quartile counters.

        ``monopoly_by_layer`` maps a stable layer key to its top-1 monopoly
        (max per-expert assignment fraction) for this step. ``layer_order`` is
        the complete shallow-to-deep router order; it may include layers without
        an observation this step. When omitted, mapping insertion order defines
        depth. Empty input returns an empty dict (nothing measurable this step;
        state is untouched).
        """
        if not monopoly_by_layer:
            return {}
        ordered_layers = tuple(
            str(layer)
            for layer in (
                layer_order
                if layer_order is not None
                else monopoly_by_layer.keys()
            )
        )
        if not ordered_layers:
            raise ValueError("layer_order must contain at least one layer")
        if len(set(ordered_layers)) != len(ordered_layers):
            raise ValueError("layer_order must contain unique layer keys")
        layer_positions = {
            layer: position for position, layer in enumerate(ordered_layers)
        }
        missing = set(monopoly_by_layer).difference(layer_positions)
        if missing:
            raise ValueError(
                "layer_order is missing observed layers: "
                + ", ".join(sorted(str(layer) for layer in missing))
            )

        deadlocked = 0
        deadlocked_by_band = [0] * len(DEADLOCK_DEPTH_BANDS)
        for layer, monopoly in monopoly_by_layer.items():
            if float(monopoly) >= self._threshold:
                self._streaks[layer] = self._streaks.get(layer, 0) + 1
                deadlocked += 1
                band = min(
                    len(DEADLOCK_DEPTH_BANDS) - 1,
                    layer_positions[layer]
                    * len(DEADLOCK_DEPTH_BANDS)
                    // len(ordered_layers),
                )
                deadlocked_by_band[band] += 1
            else:
                self._streaks[layer] = 0
        max_duration = max(self._streaks.values()) if self._streaks else 0
        max_duration_by_band = [0] * len(DEADLOCK_DEPTH_BANDS)
        for layer in ordered_layers:
            band = min(
                len(DEADLOCK_DEPTH_BANDS) - 1,
                layer_positions[layer]
                * len(DEADLOCK_DEPTH_BANDS)
                // len(ordered_layers),
            )
            max_duration_by_band[band] = max(
                max_duration_by_band[band],
                self._streaks.get(layer, 0),
            )

        metrics = {
            "moe_max_deadlock_duration": float(max_duration),
            "moe_deadlocked_layer_count": float(deadlocked),
        }
        for index, band in enumerate(DEADLOCK_DEPTH_BANDS):
            metrics[f"moe_deadlocked_layer_count_depth_{band}"] = float(
                deadlocked_by_band[index]
            )
            metrics[f"moe_max_deadlock_duration_depth_{band}"] = float(
                max_duration_by_band[index]
            )
        return metrics


def _bf16_ulp(magnitude: Any) -> Any:
    """bf16 unit-in-the-last-place at each (already non-negative) magnitude."""
    safe = magnitude.clamp_min(torch.finfo(torch.bfloat16).tiny)
    exponent = safe.log2().floor()
    return torch.exp2(exponent - _BF16_MANTISSA_BITS)


def _fp32_ulp(magnitude: Any) -> Any:
    """fp32 unit-in-the-last-place at each (already non-negative) magnitude."""
    safe = magnitude.clamp_min(torch.finfo(torch.float32).tiny)
    exponent = safe.log2().floor()
    # fp32 keeps 23 explicit mantissa bits.
    return torch.exp2(exponent - 23)


def router_underflow_fraction(
    named_params: list[tuple[str, Any]],
    *,
    lr: float,
    fp32_master_names: Any = frozenset(),
) -> float | None:
    """Fraction of router weights whose intended update truncates to zero.

    For every router parameter with a gradient, the intended update is
    ``u = lr * grad`` (elementwise). Round-to-nearest storage drops ``u`` when
    ``|u| < 0.5 * ULP(|w|)``, i.e. below half a unit-in-the-last-place at the
    weight's magnitude. Returns that truncated fraction over all router gradient
    elements.

    The metric evaluates the update against the *effective* storage: bf16 by
    default, but fp32 for any parameter whose qualified name is in
    ``fp32_master_names`` (the opt-in ``router_fp32_master`` promotes exactly
    those to an fp32 master, so the intended update no longer truncates and the
    reported fraction drops to ~0).

    Graceful contract:
      * no router parameters at all           -> ``None`` (metric absent)
      * router present but all grads zero/None -> ``0.0`` (the recommended
        ``train_router=False`` default: nothing intended, nothing truncated)
    """
    if torch is None:
        return None
    master_names = set(fp32_master_names or ())
    total = 0
    underflow = 0
    saw_router = False
    lr_abs = abs(float(lr))
    for name, param in named_params:
        if "router" not in str(name).lower():
            continue
        saw_router = True
        grad = getattr(param, "_mirai_cpu_grad", None)
        if grad is None:
            grad = getattr(param, "grad", None)
        if grad is None:
            continue
        weight = param.detach().float()
        update = grad.detach().float().abs() * lr_abs
        ulp = (
            _fp32_ulp(weight.abs())
            if str(name) in master_names
            else _bf16_ulp(weight.abs())
        )
        # Only updates that are actually attempted (non-zero) can be truncated.
        attempted = update > 0.0
        truncated = attempted & (update < 0.5 * ulp)
        total += int(update.numel())
        underflow += int(truncated.sum().item())
    if not saw_router:
        return None
    if total == 0:
        return 0.0
    return float(underflow) / float(total)


def routing_health_available() -> bool:
    """True when torch is importable (the metrics need tensor ops)."""
    return torch is not None


__all__ = [
    "DEADLOCK_MONOPOLY_THRESHOLD",
    "DEADLOCK_DEPTH_BANDS",
    "DEFAULT_COSSIM_SAMPLE_SLOTS",
    "DeadlockTracker",
    "expert_output_cossim",
    "router_underflow_fraction",
    "routing_health_available",
]
