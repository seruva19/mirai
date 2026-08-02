"""Typed trainer loss assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mirai.config.schema import TrainingConfig
from mirai.core.models.base import BasePipeline
from mirai.core.moe.monitoring.gradient_ratio import (
    measure_balance_task_gradient_ratios,
)
from mirai.core.tensors import is_torch_tensor, to_list
from mirai.core.training.objectives.flow_loss import (
    apply_noise_offset,
)
from mirai.core.training.objectives.prior import prior_preservation_loss
from mirai.core.training.objectives.sampling import (
    NoiseGenerator,
    TimestepSampler,
)
from mirai.core.training.strategies.base import TrainingInputs, TrainingStrategy


def apply_source_loss_multipliers(per_sample_loss, multipliers):
    if multipliers is None:
        return per_sample_loss
    if is_torch_tensor(per_sample_loss):
        if is_torch_tensor(multipliers):
            return per_sample_loss * multipliers
        return per_sample_loss * per_sample_loss.new_tensor(multipliers)
    if isinstance(multipliers, (int, float)):
        return [value * float(multipliers) for value in per_sample_loss]
    return [
        value * float(mult)
        for value, mult in zip(per_sample_loss, multipliers, strict=True)
    ]


@dataclass(frozen=True)
class TrainingLossResult:
    loss: Any
    loss_pre_accum: Any
    per_sample_loss: Any
    per_sample_loss_normalized: Any
    loss_weights: Any
    weighted_loss: Any
    timesteps: Any
    prior_loss: Any = None
    auxiliary_losses: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None

    def raw_metrics(self) -> dict[str, Any]:
        return {
            "loss_pre_accum": self.loss_pre_accum,
            "per_sample_loss": self.per_sample_loss,
            "per_sample_loss_normalized": self.per_sample_loss_normalized,
            "loss_weights": self.loss_weights,
            "weighted_loss": self.weighted_loss,
            "timesteps": self.timesteps,
            "prior_loss": self.prior_loss,
            "auxiliary_losses": dict(self.auxiliary_losses or {}),
            "diagnostics": dict(self.diagnostics or {}),
        }


def bucket_ids_from_batch(batch: dict[str, Any]) -> list[str] | None:
    raw_bucket_ids = batch.get("bucket_ids")
    if raw_bucket_ids is None:
        raw_bucket_ids = batch.get("bucket_id")
    if raw_bucket_ids is None:
        return None
    if is_torch_tensor(raw_bucket_ids):
        return [str(v) for v in to_list(raw_bucket_ids)]
    if isinstance(raw_bucket_ids, list):
        return [str(v) for v in raw_bucket_ids]
    return [str(raw_bucket_ids)]


def evaluate_prepared_task_loss(
    *,
    config: TrainingConfig,
    batch: dict[str, Any],
    pipeline: BasePipeline,
    strategy: TrainingStrategy,
    objective: Any,
    loss_fn: Callable[[Any, Any], Any],
    inputs: TrainingInputs,
    prediction: Any,
) -> TrainingLossResult:
    """Evaluate only the configured task objective for replayed model inputs.

    Calibration procedures use this seam when teacher and candidate forwards must
    share the exact same corruption, timestep, and conditioning. Pipeline
    auxiliary losses are deliberately excluded because they are not part of the
    source task-preservation objective.
    """

    objective_timesteps = (
        inputs.timestep
        if inputs.objective_timestep is None
        else inputs.objective_timestep
    )
    loss_timesteps = objective.resolve_loss_timesteps(
        prediction=prediction,
        default_timesteps=objective_timesteps,
    )
    if is_torch_tensor(prediction) and is_torch_tensor(loss_timesteps):
        loss_timesteps = loss_timesteps.to(device=prediction.device)
    target = objective.compute_target(
        pipeline=pipeline,
        prediction=prediction,
        clean_latents=inputs.clean_latents,
        noise=inputs.noise,
        timesteps=loss_timesteps,
    )
    objective_loss = objective.compute_per_sample_loss(
        prediction=prediction,
        target=target,
        inputs=inputs,
        loss_fn=loss_fn,
        strategy=strategy,
        config=config,
    )
    per_sample_loss = apply_source_loss_multipliers(
        objective_loss.per_sample_loss,
        batch.get("source_loss_multiplier"),
    )
    reduced = objective.reduce(
        per_sample_loss=per_sample_loss,
        timesteps=loss_timesteps,
        gradient_accumulation=1,
        config=config,
        bucket_ids=bucket_ids_from_batch(batch),
    )
    diagnostics = dict(objective_loss.diagnostics)
    diagnostics.update(dict(reduced.diagnostics or {}))
    return TrainingLossResult(
        loss=reduced.loss,
        loss_pre_accum=reduced.loss_pre_accum,
        per_sample_loss=reduced.per_sample_loss,
        per_sample_loss_normalized=reduced.per_sample_loss_normalized,
        loss_weights=reduced.loss_weights,
        weighted_loss=reduced.weighted_loss,
        timesteps=loss_timesteps,
        diagnostics=diagnostics,
    )


def compute_training_loss(
    *,
    config: TrainingConfig,
    batch: dict[str, Any],
    pipeline: BasePipeline,
    strategy: TrainingStrategy,
    objective: Any,
    loss_fn: Callable[[Any, Any], Any],
    timestep_sampler: TimestepSampler,
    noise_generator: NoiseGenerator,
    predict: Callable[[TrainingInputs], Any],
    strategy_prepare_accepts_training: bool,
    strategy_prepare_accepts_objective: bool = False,
    training: bool = True,
    gradient_accumulation: int | None = None,
) -> TrainingLossResult:
    prepare_kwargs: dict[str, Any] = {
        "batch": batch,
        "pipeline": pipeline,
        "timestep_sampler": timestep_sampler,
        "noise_generator": noise_generator,
    }
    if strategy_prepare_accepts_training:
        prepare_kwargs["training"] = bool(training)
    if strategy_prepare_accepts_objective:
        prepare_kwargs["objective"] = objective
    inputs = strategy.prepare_inputs(**prepare_kwargs)
    objective_timesteps = (
        inputs.timestep
        if inputs.objective_timestep is None
        else inputs.objective_timestep
    )
    accumulation = (
        config.training.gradient_accumulation
        if gradient_accumulation is None
        else int(gradient_accumulation)
    )
    if float(config.training.noise_offset) != 0.0:
        inputs.noise = apply_noise_offset(
            inputs.noise,
            float(config.training.noise_offset),
        )
        inputs.noisy_latents = objective.corrupt(
            pipeline=pipeline,
            clean_latents=inputs.clean_latents,
            noise=inputs.noise,
            timesteps=objective_timesteps,
        )
    prediction = objective.predict(
        inputs=inputs,
        predict=predict,
        pipeline=pipeline,
        config=config,
        training=bool(training),
    )
    loss_timesteps = objective.resolve_loss_timesteps(
        prediction=prediction,
        default_timesteps=objective_timesteps,
    )
    if is_torch_tensor(prediction) and is_torch_tensor(loss_timesteps):
        loss_timesteps = loss_timesteps.to(device=prediction.device)
    target = objective.compute_target(
        pipeline=pipeline,
        prediction=prediction,
        clean_latents=inputs.clean_latents,
        noise=inputs.noise,
        timesteps=loss_timesteps,
    )
    objective_loss = objective.compute_per_sample_loss(
        prediction=prediction,
        target=target,
        inputs=inputs,
        loss_fn=loss_fn,
        strategy=strategy,
        config=config,
    )
    per_sample_loss = objective_loss.per_sample_loss
    per_sample_loss = apply_source_loss_multipliers(
        per_sample_loss,
        batch.get("source_loss_multiplier"),
    )
    loss_result = objective.reduce(
        per_sample_loss=per_sample_loss,
        timesteps=loss_timesteps,
        gradient_accumulation=accumulation,
        config=config,
        bucket_ids=bucket_ids_from_batch(batch),
    )
    pre_accum_loss = loss_result.loss_pre_accum
    prior_loss = None
    if str(batch.get("source_type", "")).strip().lower() == "regularization":
        prior_loss = prior_preservation_loss(
            pipeline=pipeline,
            noisy_latents=inputs.noisy_latents,
            timesteps=inputs.timestep,
            text_embeds=inputs.text_embeds,
        ) * float(config.training.prior_loss_weight)
        pre_accum_loss = prior_loss
        accum = max(int(accumulation), 1)
        loss = pre_accum_loss / accum
    else:
        loss = loss_result.loss
    task_pre_accum_loss = pre_accum_loss
    auxiliary_losses = dict(pipeline.get_training_auxiliary_losses())
    auxiliary_total = None
    for value in auxiliary_losses.values():
        if value is None:
            continue
        auxiliary_total = value if auxiliary_total is None else auxiliary_total + value
    if auxiliary_total is not None:
        pre_accum_loss = pre_accum_loss + auxiliary_total
        loss = pre_accum_loss / max(int(accumulation), 1)
    diagnostics = dict(pipeline.get_training_diagnostics())
    diagnostics.update(objective_loss.diagnostics)
    diagnostics.update(dict(loss_result.diagnostics or {}))
    take_gradient_probe = getattr(pipeline, "take_balance_gradient_probe", None)
    gradient_probe = (
        take_gradient_probe() if callable(take_gradient_probe) else None
    )
    if gradient_probe is not None:
        diagnostics.update(
            measure_balance_task_gradient_ratios(
                task_loss=task_pre_accum_loss,
                probe=gradient_probe,
                alarm_threshold=float(
                    config.model.params.moe_balance_grad_ratio_threshold
                ),
            )
        )
    return TrainingLossResult(
        loss=loss,
        loss_pre_accum=pre_accum_loss,
        per_sample_loss=loss_result.per_sample_loss,
        per_sample_loss_normalized=loss_result.per_sample_loss_normalized,
        loss_weights=loss_result.loss_weights,
        weighted_loss=loss_result.weighted_loss,
        timesteps=loss_timesteps,
        prior_loss=prior_loss,
        auxiliary_losses=auxiliary_losses,
        diagnostics=diagnostics,
    )
