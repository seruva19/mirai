"""Post-optimizer step processing helpers for training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.training.data.batches import evaluate_val_loss, named_grad_norms
from mirai.core.training.optim.ema import update_ema_state
from mirai.core.training.lifecycle.posthoc_ema import (
    update_and_maybe_save_posthoc_ema,
)
from mirai.core.training.observability.dispatch import dispatch_step_callbacks, emit_event
from mirai.core.training.control.live_control_actions import (
    apply_live_training_control,
    resolve_effective_sample_every_n_steps,
    resolve_effective_val_every_n_steps,
)
from mirai.core.training.residency.memory import current_vram_used_mb
from mirai.core.moe.monitoring.gradients import moe_gradient_health
from mirai.core.moe.monitoring.health import router_underflow_fraction
from mirai.core.moe.monitoring.drift import router_gradient_update_ratios
from mirai.core.training.preview.preview_samples import build_preview_samples
from mirai.core.training.lifecycle.run_reporting import emit_preview_generated
from mirai.core.training.observability.metrics import (
    apply_validation_policy,
    build_step_metrics,
)


@dataclass(frozen=True)
class CompletedTrainingStepResult:
    step_metrics: dict[str, Any]
    validation_ran: bool
    preview_generated: bool
    stop_requested: bool


def process_completed_training_step(
    session: Any,
    *,
    curriculum_profile: Any,
    grad_norm_value: float,
    applied_control: Any | None = None,
) -> CompletedTrainingStepResult:
    config = session.config
    trainer = session.trainer
    optimizer = session.optimizer
    run_state = session.run_state

    if config.training.ema_enabled and run_state.ema_state is not None:
        run_state.ema_state = update_ema_state(
            run_state.ema_state,
            trainer.pipeline.state_dict(),
            decay=config.training.ema_decay,
        )
    update_and_maybe_save_posthoc_ema(session)

    breakdown_every = int(config.training.log_grad_breakdown_every_n_steps)
    grad_breakdown = None
    if breakdown_every > 0 and (run_state.global_step % breakdown_every == 0):
        grad_breakdown = named_grad_norms(session.named_params)
    if bool(trainer.pipeline.get_sparse_moe_capabilities().is_sparse_moe):
        moe_health = moe_gradient_health(session.named_params)
        grad_breakdown = {**(grad_breakdown or {}), **moe_health}
        # Routing-health pack (opt-in): bf16 sub-ULP router-update underflow
        # alarm. Telemetry only; with train_router=False (recommended default)
        # router grads are zero so it reports 0.0. None (no router params) is
        # dropped so the metric stays absent rather than emitting a null.
        if bool(config.model.params.moe_routing_health):
            # When router_fp32_master promotes trainable router params to an fp32
            # master, the effective update accumulates in fp32 -> evaluate those
            # names against fp32 storage so the alarm reflects the real update.
            master_names = getattr(
                trainer.pipeline, "get_router_fp32_master_names", lambda: frozenset()
            )()
            underflow = router_underflow_fraction(
                session.named_params,
                lr=float(optimizer.param_groups[0]["lr"]),
                fp32_master_names=master_names,
            )
            if underflow is not None:
                grad_breakdown = {
                    **(grad_breakdown or {}),
                    "moe_router_underflow_fraction": float(underflow),
                }
            grad_breakdown = {
                **(grad_breakdown or {}),
                **router_gradient_update_ratios(
                    session.named_params,
                    lr=float(optimizer.param_groups[0]["lr"]),
                ),
            }
    step_metrics = build_step_metrics(
        config=config,
        last_metrics=run_state.last_metrics,
        lr=float(optimizer.param_groups[0]["lr"]),
        grad_norm=float(grad_norm_value),
        skipped_steps=int(run_state.skipped_steps),
        vram_used_mb=float(current_vram_used_mb()),
        curriculum_resolution=curriculum_profile.resolution,
        curriculum_frame_count=curriculum_profile.frame_count,
        gradient_offload_ops=int(run_state.gradient_offload_ops),
        optimizer_offload_ops=int(run_state.optimizer_offload_ops),
        grad_breakdown=grad_breakdown,
    )

    control_result = apply_live_training_control(
        session,
        applied_control=applied_control,
        step=int(run_state.global_step),
    )

    val_every = resolve_effective_val_every_n_steps(session)
    validation_ran = False
    if control_result.force_validation or (
        val_every > 0 and run_state.global_step % val_every == 0
    ):
        validation_ran = True
        val_loss = evaluate_val_loss(
            trainer=trainer,
            records=session.val_records,
            batch_size=config.training.batch_size,
            rng=session.rng,
            masked_loss=config.training.masked_loss,
        )
        if val_loss is not None:
            validation_result = apply_validation_policy(
                step_metrics=step_metrics,
                val_loss=float(val_loss),
                early_stop_state=run_state.early_stop_state,
                global_step=run_state.global_step,
                early_stop_patience=int(config.training.early_stop_patience),
                log_on_this_rank=session.log_on_this_rank,
                ckpt_dir=session.ckpt_dir,
                build_ckpt_payload=session.build_ckpt_payload,
            )
            step_metrics = validation_result.step_metrics
            run_state.early_stop_state = validation_result.early_stop_state
            run_state.stopped_early = bool(
                run_state.stopped_early or validation_result.stopped_early
            )
            if validation_result.best_checkpoint_saved:
                emit_event(
                    session.event_bus,
                    session.callbacks,
                    event_type="checkpoint.saved",
                    step=run_state.global_step,
                    payload={
                        "path": str(validation_result.best_checkpoint_path),
                        "reason": "best_val_loss",
                    },
                )

    emit_event(
        session.event_bus,
        session.callbacks,
        event_type="step.completed",
        step=run_state.global_step,
        payload=dict(step_metrics),
    )
    dispatch_step_callbacks(
        session.callbacks,
        step=run_state.global_step,
        metrics=step_metrics,
    )

    sample_every = resolve_effective_sample_every_n_steps(session)
    preview_generated = False
    if (
        session.log_on_this_rank
        and (
            control_result.force_preview
            or (sample_every > 0 and run_state.global_step % sample_every == 0)
        )
    ):
        # Preview/sampling is best-effort: a failure here (e.g. missing native
        # inference assets) must never abort a long training run.
        try:
            preview_results = build_preview_samples(
                config=config,
                trainer=trainer,
                step=run_state.global_step,
                output_dir=session.output_dir,
                params=session.params,
                optimizer=optimizer,
                base_metrics=step_metrics,
                curriculum_resolution=curriculum_profile.resolution,
                curriculum_frame_count=curriculum_profile.frame_count,
                force=bool(control_result.force_preview),
                sample_name_override=str(control_result.preview_name),
                prompt_override=str(control_result.preview_prompt),
            )
            preview_generated = True
        except Exception as exc:  # noqa: BLE001
            preview_results = []
            emit_event(
                session.event_bus,
                session.callbacks,
                event_type="preview.failed",
                step=run_state.global_step,
                payload={"error": f"{type(exc).__name__}: {exc}"},
            )
        for preview_result in preview_results:
            dispatch_step_callbacks(
                session.callbacks,
                step=run_state.global_step,
                metrics=preview_result.metrics,
            )
            emit_preview_generated(
                session,
                step=run_state.global_step,
                sample_name=(
                    str(control_result.preview_name)
                    if control_result.force_preview
                    else str(preview_result.sample_name)
                ),
                sample_path=str(preview_result.preview.get("sample_path", "")),
            )

    if control_result.stop_requested:
        run_state.stopped_early = True

    return CompletedTrainingStepResult(
        step_metrics=dict(step_metrics),
        validation_ran=bool(validation_ran),
        preview_generated=bool(preview_generated),
        stop_requested=bool(control_result.stop_requested),
    )
