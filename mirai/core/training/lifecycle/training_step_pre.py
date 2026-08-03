"""Pre-postprocessing training-step helpers for sampling and optimizer transition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mirai.core.training.data.batches import (
    build_batch_from_records,
    is_oom_runtime_error,
    sample_batch,
    select_deterministic_batch_records,
)
from mirai.core.training.residency.device_placement import move_batch_to_device
from mirai.core.training.observability.dispatch import emit_event
from mirai.core.training.optim.gradients import (
    has_non_finite_gradients,
    restore_gradients_from_cpu,
)
from mirai.core.training.control.live_control_actions import build_live_runtime_state_payload
from mirai.core.training.data.online import build_temporal_groups
from mirai.core.training.runtime.execution import (
    accumulate_training_gradients,
    apply_optimizer_update,
)
from mirai.core.training.training_policy import RecordSelectionContext


@dataclass(frozen=True)
class StepSamplingContext:
    curriculum_profile: Any
    eligible_records: list[dict[str, Any]]
    temporal_base_ids_step: list[str]
    temporal_groups_step: dict[str, list[dict[str, Any]]]
    epoch_index: int


@dataclass(frozen=True)
class StepAdvanceResult:
    advanced: bool
    curriculum_profile: Any
    grad_norm_value: float
    applied_control: Any | None = None


def resolve_step_sampling_context(session: Any) -> StepSamplingContext:
    run_state = session.run_state
    curriculum_profile = session.curriculum.profile_for_step(run_state.global_step)
    eligible_records = session.curriculum.filter_records(
        session.train_records,
        step=run_state.global_step,
    )
    temporal_base_ids_step = session.temporal_base_ids
    temporal_groups_step = session.temporal_groups
    epoch_index = 0
    if bool(session.config.dataset.online_temporal_resampling):
        temporal_base_ids_step, temporal_groups_step = build_temporal_groups(
            eligible_records
        )
        steps_per_epoch = max(
            1,
            math.ceil(
                max(1, len(temporal_base_ids_step))
                / max(1, int(session.config.training.batch_size))
            ),
        )
        epoch_index = int(run_state.global_step // steps_per_epoch)
    return StepSamplingContext(
        curriculum_profile=curriculum_profile,
        eligible_records=eligible_records,
        temporal_base_ids_step=temporal_base_ids_step,
        temporal_groups_step=temporal_groups_step,
        epoch_index=epoch_index,
    )


def run_pre_postprocessed_training_step(
    session: Any,
    *,
    sampling_context: StepSamplingContext,
) -> StepAdvanceResult:
    config = session.config
    trainer = session.trainer
    optimizer = session.optimizer
    scheduler = session.scheduler
    run_state = session.run_state

    if bool(trainer.pipeline.supports_rank_schedule_progress()):
        trainer.pipeline.set_rank_schedule_progress(
            step=int(run_state.global_step),
            start_step=int(config.adapter.rank_schedule_start),
            end_step=int(config.adapter.rank_schedule_end),
            min_scale=float(config.adapter.rank_schedule_min_scale),
        )

    if bool(trainer.pipeline.supports_balance_loss_schedule_progress()):
        trainer.pipeline.set_balance_loss_schedule_progress(
            step=int(run_state.global_step)
        )

    if bool(
        getattr(
            trainer.pipeline,
            "supports_progressive_sparsification_progress",
            lambda: False,
        )()
    ):
        trainer.pipeline.set_progressive_sparsification_progress(
            step=int(run_state.global_step)
        )

    # Stochastic per-step expert-subset routing: install the annealed subset +
    # deterministic per-(seed, step, layer) sampling seed before the forward.
    if bool(
        getattr(trainer.pipeline, "supports_router_subset_progress", lambda: False)()
    ):
        trainer.pipeline.set_router_subset_progress(
            step=int(run_state.global_step),
            seed=int(config.training.seed),
        )
    if bool(
        getattr(trainer.pipeline, "supports_expert_choice_progress", lambda: False)()
    ):
        trainer.pipeline.set_expert_choice_progress(step=int(run_state.global_step))

    emit_event(
        session.event_bus,
        session.callbacks,
        event_type="step.started",
        step=run_state.global_step,
        payload={"global_step": int(run_state.global_step)},
    )
    optimizer.zero_grad(set_to_none=True)
    live_control = getattr(session, "live_control", None)
    applied_control_holder: list[Any] = []
    if live_control is not None:
        live_control.publish_capabilities(
            global_step=int(run_state.global_step),
            active=True,
            runtime_state=build_live_runtime_state_payload(session),
        )
        live_control.poll_pending_request(global_step=int(run_state.global_step))

    accumulation_result = accumulate_training_gradients(
        trainer=trainer,
        grad_accum=session.grad_accum,
        build_batch=_build_training_batch_factory(
            session=session,
            sampling_context=sampling_context,
        ),
        params=session.params,
        gradient_cpu_offload=bool(config.training.gradient_cpu_offload),
        expert_touch_guard_mode=str(config.training.moe_expert_touch_guard),
        expert_touch_max_fraction=float(
            config.training.moe_expert_touch_max_fraction
        ),
        oom_error_predicate=is_oom_runtime_error,
        on_forward_started=lambda micro_batch_index, micro_batch_total: emit_event(
            session.event_bus,
            session.callbacks,
            event_type="forward.started",
            step=run_state.global_step,
            payload={
                "micro_batch_index": int(micro_batch_index),
                "micro_batch_total": int(micro_batch_total),
            },
        ),
        on_forward_completed=lambda loss_pre_accum: emit_event(
            session.event_bus,
            session.callbacks,
            event_type="forward.completed",
            step=run_state.global_step,
            payload={"loss_pre_accum": loss_pre_accum},
        ),
        on_backward_started=lambda loss_value: emit_event(
            session.event_bus,
            session.callbacks,
            event_type="backward.started",
            step=run_state.global_step,
            payload={"loss": loss_value},
        ),
        on_backward_completed=lambda loss_value: emit_event(
            session.event_bus,
            session.callbacks,
            event_type="backward.completed",
            step=run_state.global_step,
            payload={"loss": loss_value},
        ),
        on_microstep_completed=lambda micro_batch_index: _sync_live_control(
            session=session,
            micro_batch_index=int(micro_batch_index),
            applied_control_holder=applied_control_holder,
        ),
        finish_backward_offloads=trainer.pipeline.finish_backward_offloads,
    )
    run_state.gradient_offload_ops += accumulation_result.gradient_offload_ops
    run_state.last_metrics = accumulation_result.last_metrics

    consumes_offloaded_gradients = bool(
        getattr(optimizer, "supports_offloaded_gradients", lambda: False)()
    )
    if bool(config.training.gradient_cpu_offload) and not consumes_offloaded_gradients:
        restore_gradients_from_cpu(session.params)
    non_finite_source = (
        "loss"
        if accumulation_result.non_finite_loss
        else "gradients"
    )
    non_finite_gradients = bool(
        accumulation_result.non_finite_loss
        or has_non_finite_gradients(session.params)
    )
    if not non_finite_gradients:
        emit_event(
            session.event_bus,
            session.callbacks,
            event_type="optimizer.started",
            step=run_state.global_step,
            payload={"lr": float(optimizer.param_groups[0]["lr"])},
        )
    optimizer_update_result = apply_optimizer_update(
        params=session.params,
        optimizer=optimizer,
        scheduler=scheduler,
        pipeline=trainer.pipeline,
        max_grad_norm=float(config.training.max_grad_norm),
        optimizer_cpu_offload=bool(config.training.optimizer_cpu_offload),
        non_finite_gradients=bool(non_finite_gradients),
        non_finite_grad_policy=str(config.training.non_finite_grad_policy),
        consecutive_skipped_steps=int(run_state.consecutive_skipped_steps),
        max_consecutive_skipped_steps=int(config.training.max_consecutive_skipped_steps),
        skipped_steps=int(run_state.skipped_steps),
        global_step=int(run_state.global_step),
        training_policies=getattr(trainer, "training_policies", None),
        non_finite_source=non_finite_source,
    )
    if not optimizer_update_result.advanced:
        emit_event(
            session.event_bus,
            session.callbacks,
            event_type="step.skipped_non_finite",
            step=run_state.global_step,
            payload={
                "policy": str(config.training.non_finite_grad_policy),
                "action": str(optimizer_update_result.non_finite_action),
                "consecutive_skips": int(
                    optimizer_update_result.consecutive_skipped_steps
                ),
                "source": non_finite_source,
            },
        )
        run_state.consecutive_skipped_steps = (
            optimizer_update_result.consecutive_skipped_steps
        )
        run_state.skipped_steps = optimizer_update_result.skipped_steps
        return StepAdvanceResult(
            advanced=False,
            curriculum_profile=sampling_context.curriculum_profile,
            grad_norm_value=0.0,
            applied_control=(
                None if not applied_control_holder else applied_control_holder[-1]
            ),
        )

    run_state.optimizer_offload_ops += optimizer_update_result.optimizer_offload_ops
    run_state.consecutive_skipped_steps = optimizer_update_result.consecutive_skipped_steps
    run_state.skipped_steps = optimizer_update_result.skipped_steps
    run_state.global_step = optimizer_update_result.global_step
    grad_norm_value = float(optimizer_update_result.grad_norm or 0.0)
    emit_event(
        session.event_bus,
        session.callbacks,
        event_type="optimizer.completed",
        step=run_state.global_step,
        payload={
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": grad_norm_value,
            "skipped_steps": int(run_state.skipped_steps),
        },
    )
    return StepAdvanceResult(
        advanced=True,
        curriculum_profile=sampling_context.curriculum_profile,
        grad_norm_value=grad_norm_value,
        applied_control=(
            None if not applied_control_holder else applied_control_holder[-1]
        ),
    )


def _sync_live_control(
    *,
    session: Any,
    micro_batch_index: int,
    applied_control_holder: list[Any],
) -> None:
    live_control = getattr(session, "live_control", None)
    if live_control is None:
        return
    live_control.poll_pending_request(global_step=int(session.run_state.global_step))
    applied = live_control.on_microstep(
        microstep_index=int(micro_batch_index),
        optimizer_steps_committed=int(session.run_state.global_step),
        global_step=int(session.run_state.global_step),
    )
    if applied is not None:
        applied_control_holder.append(applied)


def _build_training_batch_factory(
    *,
    session: Any,
    sampling_context: StepSamplingContext,
):
    config = session.config
    run_state = session.run_state

    def _build_training_batch(micro_batch_index: int) -> dict[str, Any]:
        global_batch_index = (
            (int(run_state.global_step) * int(session.grad_accum))
            + int(micro_batch_index)
        )
        batch_records, _selected_task = session.curriculum.filter_records_for_batch(
            sampling_context.eligible_records,
            step=int(run_state.global_step),
            seed=int(config.training.seed),
            global_batch_index=global_batch_index,
        )
        if bool(config.dataset.online_temporal_resampling):
            temporal_base_ids = sampling_context.temporal_base_ids_step
            temporal_groups = sampling_context.temporal_groups_step
            if batch_records is not sampling_context.eligible_records:
                temporal_base_ids, temporal_groups = build_temporal_groups(batch_records)
            batch = sample_batch(
                batch_records,
                config.training.batch_size,
                session.rng,
                masked_loss=config.training.masked_loss,
                step=run_state.global_step,
                epoch_index=sampling_context.epoch_index,
                online_tag_shuffle=bool(config.dataset.online_tag_shuffle),
                online_tag_shuffle_dropout=float(
                    config.dataset.online_tag_shuffle_dropout
                ),
                online_tag_shuffle_keep_first_n_tags=int(
                    config.dataset.online_tag_shuffle_keep_first_n_tags
                ),
                base_seed=int(config.training.seed),
                online_temporal_resampling=True,
                temporal_base_ids=temporal_base_ids,
                temporal_groups=temporal_groups,
                training_policies=session.trainer.training_policies,
                global_batch_index=global_batch_index,
            )
            return move_batch_to_device(
                batch,
                device=session.compute_device,
                dtype=session.compute_dtype,
            )
        chosen_records = session.trainer.training_policies.select_records(
            RecordSelectionContext(
                records=batch_records,
                batch_size=int(config.training.batch_size),
                base_seed=int(config.training.seed),
                global_batch_index=global_batch_index,
                step=int(run_state.global_step),
            )
        )
        if chosen_records is None:
            chosen_records, _ = select_deterministic_batch_records(
                batch_records,
                batch_size=int(config.training.batch_size),
                base_seed=int(config.training.seed),
                global_batch_index=global_batch_index,
            )
        batch = build_batch_from_records(
            chosen_records,
            masked_loss=bool(config.training.masked_loss),
            step=int(run_state.global_step),
            online_tag_shuffle=bool(config.dataset.online_tag_shuffle),
            online_tag_shuffle_dropout=float(config.dataset.online_tag_shuffle_dropout),
            online_tag_shuffle_keep_first_n_tags=int(
                config.dataset.online_tag_shuffle_keep_first_n_tags
            ),
            base_seed=int(config.training.seed),
            training_policies=session.trainer.training_policies,
            training=True,
            global_batch_index=global_batch_index,
        )
        return move_batch_to_device(
            batch,
            device=session.compute_device,
            dtype=session.compute_dtype,
        )

    return _build_training_batch
