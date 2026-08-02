"""Training-step execution helpers for accumulation and optimizer policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mirai.core.moe.runtime.touch_guard import enforce_expert_touch_guard
from mirai.core.tensors import to_jsonable, to_list, to_python_scalar
from mirai.core.training.optim.gradients import offload_gradients_to_cpu
from mirai.core.training.optim.gradients import resolve_non_finite_policy
from mirai.core.training.optim.gradients import clip_grad_norm_with_offload
from mirai.core.training.objectives.engine import TrainingLossResult
from mirai.core.training.residency.memory import oom_guidance_message
from mirai.core.training.optim.optimizer import offload_optimizer_state_to_cpu

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GradientAccumulationResult:
    last_metrics: dict[str, Any]
    gradient_offload_ops: int


@dataclass(frozen=True)
class OptimizerUpdateResult:
    advanced: bool
    grad_norm: float | None
    skipped_steps: int
    consecutive_skipped_steps: int
    global_step: int
    optimizer_offload_ops: int
    non_finite_action: str


def _compute_loss_result(trainer: Any, batch: dict[str, Any]) -> TrainingLossResult:
    compute_loss_result = getattr(trainer, "compute_loss_result", None)
    if callable(compute_loss_result):
        return compute_loss_result(batch)
    loss, raw = trainer.compute_loss(batch)
    return TrainingLossResult(
        loss=loss,
        loss_pre_accum=raw["loss_pre_accum"],
        per_sample_loss=raw["per_sample_loss"],
        per_sample_loss_normalized=raw.get("per_sample_loss_normalized", raw["per_sample_loss"]),
        loss_weights=raw["loss_weights"],
        weighted_loss=raw["weighted_loss"],
        timesteps=raw["timesteps"],
        prior_loss=raw.get("prior_loss"),
        auxiliary_losses=raw.get("auxiliary_losses"),
        diagnostics=raw.get("diagnostics"),
    )


def accumulate_training_gradients(
    *,
    trainer: Any,
    grad_accum: int,
    build_batch: Callable[[int], dict[str, Any]],
    params: list[Any],
    gradient_cpu_offload: bool,
    oom_error_predicate: Callable[[RuntimeError], bool] | None = None,
    on_forward_started: Callable[[int, int], None] | None = None,
    on_forward_completed: Callable[[Any], None] | None = None,
    on_backward_started: Callable[[Any], None] | None = None,
    on_backward_completed: Callable[[Any], None] | None = None,
    on_microstep_completed: Callable[[int], None] | None = None,
    finish_backward_offloads: Callable[[], None] | None = None,
    expert_touch_guard_mode: str = "off",
    expert_touch_max_fraction: float = 1.0,
) -> GradientAccumulationResult:
    gradient_offload_ops = 0
    last_metrics: dict[str, Any] = {}
    try:
        for micro_batch_index in range(int(grad_accum)):
            batch = build_batch(micro_batch_index)
            if callable(on_forward_started):
                on_forward_started(int(micro_batch_index) + 1, int(grad_accum))
            loss_result = _compute_loss_result(trainer, batch)
            touch_guard = enforce_expert_touch_guard(
                loss_result.diagnostics,
                mode=expert_touch_guard_mode,
                max_fraction=expert_touch_max_fraction,
            )
            loss = loss_result.loss
            loss_pre_accum = to_python_scalar(loss_result.loss_pre_accum)
            if callable(on_forward_completed):
                on_forward_completed(loss_pre_accum)
            if not torch.is_tensor(loss):
                loss = torch.tensor(float(loss), dtype=torch.float32)
            if not torch.isfinite(loss).all():
                raise SystemExit("NaN/Inf loss detected.")
            loss_value = to_python_scalar(loss)
            if callable(on_backward_started):
                on_backward_started(loss_value)
            loss.backward()
            if callable(finish_backward_offloads):
                finish_backward_offloads()
            if callable(on_backward_completed):
                on_backward_completed(loss_value)
            if gradient_cpu_offload:
                gradient_offload_ops += offload_gradients_to_cpu(params)
            if callable(on_microstep_completed):
                on_microstep_completed(int(micro_batch_index))
            last_metrics = {
                "loss": loss_value,
                "loss_pre_accum": loss_pre_accum,
                "per_sample_loss": to_list(loss_result.per_sample_loss),
                "loss_weights": to_list(loss_result.loss_weights),
                "weighted_loss": to_list(loss_result.weighted_loss),
                "timesteps": to_list(loss_result.timesteps),
                "auxiliary_losses": to_jsonable(loss_result.auxiliary_losses or {}),
                "diagnostics": to_jsonable(loss_result.diagnostics or {}),
            }
            if touch_guard.observed_fraction is not None:
                last_metrics["diagnostics"]["moe_expert_touch_fraction"] = (
                    touch_guard.observed_fraction
                )
                last_metrics["diagnostics"]["moe_expert_touch_exceeded"] = (
                    touch_guard.exceeded
                )
    except RuntimeError as exc:
        if callable(oom_error_predicate) and oom_error_predicate(exc):
            raise SystemExit(oom_guidance_message(original_error=str(exc))) from exc
        raise
    return GradientAccumulationResult(
        last_metrics=last_metrics,
        gradient_offload_ops=int(gradient_offload_ops),
    )


def apply_optimizer_update(
    *,
    params: list[Any],
    optimizer: Any,
    scheduler: Any,
    pipeline: Any,
    max_grad_norm: float,
    optimizer_cpu_offload: bool,
    non_finite_gradients: bool,
    non_finite_grad_policy: str,
    consecutive_skipped_steps: int,
    max_consecutive_skipped_steps: int,
    skipped_steps: int,
    global_step: int,
    training_policies: Any | None = None,
) -> OptimizerUpdateResult:
    if bool(non_finite_gradients):
        action, next_skips = resolve_non_finite_policy(
            policy=non_finite_grad_policy,
            consecutive_skips=consecutive_skipped_steps,
            max_consecutive_skipped_steps=max_consecutive_skipped_steps,
        )
        if action == "abort":
            raise SystemExit(
                "Non-finite gradients detected before optimizer.step(). "
                "Aborting due non_finite_grad_policy='abort'."
            )
        optimizer.zero_grad(set_to_none=True)
        if training_policies is not None:
            training_policies.after_optimizer_step(optimizer, applied=False)
        if bool(pipeline.supports_runtime_offload_flush()):
            pipeline.flush_runtime_offloads()
        return OptimizerUpdateResult(
            advanced=False,
            grad_norm=None,
            skipped_steps=int(skipped_steps) + 1,
            consecutive_skipped_steps=int(next_skips),
            global_step=int(global_step),
            optimizer_offload_ops=0,
            non_finite_action=str(action),
        )

    if training_policies is not None:
        training_policies.before_optimizer_step(optimizer)
    optimizer_applied = False
    try:
        grad_norm = clip_grad_norm_with_offload(params, max_norm=float(max_grad_norm))
        pipeline.prepare_optimizer_step()
        try:
            optimizer.step()
            optimizer_applied = True
        finally:
            pipeline.finish_optimizer_step()
    finally:
        if training_policies is not None:
            training_policies.after_optimizer_step(
                optimizer, applied=optimizer_applied
            )
    optimizer_offload_ops = 0
    if optimizer_cpu_offload:
        optimizer_offload_ops = offload_optimizer_state_to_cpu(optimizer)
    scheduler.step()
    if bool(pipeline.supports_runtime_offload_flush()):
        pipeline.flush_runtime_offloads()
    return OptimizerUpdateResult(
        advanced=True,
        grad_norm=float(grad_norm),
        skipped_steps=int(skipped_steps),
        consecutive_skipped_steps=0,
        global_step=int(global_step) + 1,
        optimizer_offload_ops=int(optimizer_offload_ops),
        non_finite_action="",
    )
