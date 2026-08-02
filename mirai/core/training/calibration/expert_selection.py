"""Calibration driver for routing-guided expert selection (MoE-Sieve).

Runs a handful of forward-only passes over the training dataloader to profile
which experts are routed most often, then restricts the per-expert LoRA adapters
to those hot experts. Forward-only (no optimizer step, no LoRA grad), so it stays
well within the training memory envelope.
"""

from __future__ import annotations

import sys
from typing import Any

from mirai.core.moe.calibration.selection import (
    ExpertSelectionReport,
    apply_routing_topk_selection,
    iter_expert_lora_blocks,
    register_calibration_hooks,
)
from mirai.core.training.lifecycle.training_step_pre import (
    _build_training_batch_factory,
    resolve_step_sampling_context,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _log(session: Any, message: str) -> None:
    if bool(getattr(session, "log_on_this_rank", True)):
        print(message, file=sys.stderr)


def maybe_calibrate_expert_selection(session: Any) -> ExpertSelectionReport | None:
    """Profile routing and mask expert LoRA adapters if routing_topk is enabled.

    Returns the selection report, or ``None`` when selection is not requested,
    the run is resumed (mask already restored from checkpoint), or there are no
    expert LoRA adapters to mask.
    """
    config = session.config
    adapter = config.adapter
    if str(getattr(adapter, "expert_selection", "all")).strip().lower() != "routing_topk":
        return None
    if torch is None:  # pragma: no cover
        return None
    if bool(getattr(session, "resumed", False)):
        _log(session, "[expert-selection] resumed run: reusing checkpoint mask, skipping calibration.")
        return None

    pipeline = session.trainer.pipeline
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return None
    blocks = list(iter_expert_lora_blocks(transformer))
    if not blocks:
        _log(
            session,
            "[expert-selection] routing_topk requested but no routed-expert LoRA "
            "adapters found (target_preset has no experts?); leaving adapters unmasked.",
        )
        return None

    steps = int(adapter.expert_calibration_steps)
    fraction = float(adapter.expert_topk_fraction)
    sampling_context = resolve_step_sampling_context(session)
    build_batch = _build_training_batch_factory(
        session=session, sampling_context=sampling_context
    )

    # Snapshot the sampling RNG so calibration never perturbs training's stream
    # (the online-resampling batch path draws from session.rng).
    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None

    trainer = session.trainer
    was_training = bool(getattr(transformer, "training", True))
    accumulator: dict[str, Any] = {}
    _log(
        session,
        f"[expert-selection] calibrating routing_topk: fraction={fraction} "
        f"steps={steps} layers={len(blocks)}",
    )
    # Router forward hooks capture routed counts before the pipeline clears the
    # per-forward routing state at the end of each compute_loss.
    handles = register_calibration_hooks(transformer, accumulator)
    try:
        with torch.no_grad():
            for i in range(steps):
                batch = build_batch(i)
                trainer.compute_loss(batch, training=True)
    finally:
        for handle in handles:
            handle.remove()
        if rng is not None and rng_state is not None:
            rng.setstate(rng_state)
        if hasattr(transformer, "train"):
            transformer.train(was_training)

    report = apply_routing_topk_selection(
        transformer,
        accumulator,
        fraction=fraction,
        calibration_steps=steps,
    )
    _emit_report(session, report)
    setattr(session, "expert_selection_report", report)
    return report


def _emit_report(session: Any, report: ExpertSelectionReport) -> None:
    delta = report.trainable_params_before - report.trainable_params_after_effective
    _log(
        session,
        "[expert-selection] selected "
        f"{report.expert_adapter_params_active}/{report.expert_adapter_params_total} "
        f"expert-adapter params across {len(report.layers)} layers; "
        f"trainable(effective) {report.trainable_params_before} -> "
        f"{report.trainable_params_after_effective} (delta -{delta}); "
        f"mean_token_coverage={report.mean_covered_token_fraction():.4f} "
        f"mean_layer_overlap={report.selection_overlap():.4f}",
    )
    for layer in report.layers:
        _log(
            session,
            f"[expert-selection]   {layer.layer}: "
            f"{layer.num_selected}/{layer.num_experts} experts "
            f"hash={layer.selection_hash} "
            f"coverage={layer.covered_token_fraction:.4f} "
            f"routed_tokens={layer.routed_tokens}",
        )
