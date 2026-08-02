"""Trainer parameter and optimizer/scheduler assembly for sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.training.optim.optimizer import (
    OptimizerBuildResult,
    SELECTED_EXPERT_OPTIMIZER_TYPES,
    build_cpu_offload_optimizer,
    build_optimizer,
    warmup_optimizer_warning,
)
from mirai.core.training.optim.scheduler import build_scheduler


@dataclass(frozen=True)
class TrainingRuntimeComponents:
    params: list[Any]
    named_params: list[tuple[str, Any]]
    optimizer_result: Any
    scheduler: Any
    warmup_warning: str


def build_training_runtime_components(
    *,
    trainer: Any,
    config: Any,
) -> TrainingRuntimeComponents:
    params = list(trainer.get_trainable_parameters())
    if not params:
        raise ValueError("No trainable parameters exposed by pipeline.")
    named_params = list(trainer.get_named_trainable_parameters())
    selected_expert_optimizer = (
        str(config.optimizer.type).strip().lower()
        in SELECTED_EXPERT_OPTIMIZER_TYPES
    )
    plan_getter = getattr(
        trainer.pipeline,
        "get_selected_expert_parameter_plan",
        None,
    )
    selected_expert_plan = (
        dict(plan_getter())
        if selected_expert_optimizer and callable(plan_getter)
        else {}
    )
    if selected_expert_optimizer and not selected_expert_plan:
        raise ValueError("Selected-expert optimizer has no bound per-parameter plan.")
    lora_pairs: tuple[Any, ...] = ()
    if str(config.optimizer.type).strip().lower() in {
        "lora_pro_adamw",
        "lora_muon",
    }:
        from mirai.core.training.optim.lora_pairs import (
            collect_lora_factor_pairs,
        )

        training_model = trainer.pipeline.get_training_model()
        if training_model is None:
            raise ValueError("Paired LoRA optimizer requires a training model.")
        optimizer_name = (
            "LoRA-Pro"
            if str(config.optimizer.type).strip().lower() == "lora_pro_adamw"
            else "LoRA-Muon"
        )
        lora_pairs = collect_lora_factor_pairs(
            training_model,
            consumer=optimizer_name,
        )
    optimizer_result = build_optimizer(
        params=params,
        named_params=named_params,
        optimizer_type=config.optimizer.type,
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        weight_decay_filter=config.optimizer.weight_decay_filter,
        loraplus_lr_ratio=config.optimizer.loraplus_lr_ratio,
        module_lr_multipliers=config.adapter.lr_multipliers,
        allow_fallback=config.optimizer.allow_fallback,
        selected_expert_ids=(),
        selected_expert_plan=selected_expert_plan,
        stochastic_rounding=config.optimizer.stochastic_rounding,
        prodigy_beta3=config.optimizer.prodigy_beta3,
        prodigy_decouple=config.optimizer.prodigy_decouple,
        prodigy_use_bias_correction=config.optimizer.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=config.optimizer.prodigy_safeguard_warmup,
        prodigy_d0=config.optimizer.prodigy_d0,
        prodigy_d_coef=config.optimizer.prodigy_d_coef,
        prodigy_growth_rate=config.optimizer.prodigy_growth_rate,
        prodigy_slice_p=config.optimizer.prodigy_slice_p,
        lora_pairs=lora_pairs,
        lora_pro_damping=config.optimizer.lora_pro_damping,
        muon_momentum=config.optimizer.muon_momentum,
        muon_nesterov=config.optimizer.muon_nesterov,
        muon_ns_steps=config.optimizer.muon_ns_steps,
        muon_eps=config.optimizer.muon_eps,
        muon_rms_target=config.optimizer.muon_rms_target,
        lora_muon_gauge_rebalance_interval=(
            config.optimizer.lora_muon_gauge_rebalance_interval
        ),
        lora_muon_gauge_rebalance_alpha=(
            config.optimizer.lora_muon_gauge_rebalance_alpha
        ),
    )
    if bool(config.training.optimizer_cpu_offload):
        optimizer_result = OptimizerBuildResult(
            optimizer=build_cpu_offload_optimizer(optimizer_result.optimizer),
            resolved_type=optimizer_result.resolved_type,
            used_fallback=optimizer_result.used_fallback,
        )
    warning = warmup_optimizer_warning(
        optimizer_type=config.optimizer.type,
        warmup_steps=config.training.warmup_steps,
    )
    scheduler = build_scheduler(
        optimizer_result.optimizer,
        warmup_steps=config.training.warmup_steps,
        total_steps=config.training.max_steps,
        scheduler=config.optimizer.scheduler,
        num_cycles=config.optimizer.scheduler_num_cycles,
        power=config.optimizer.scheduler_power,
        min_lr_ratio=config.optimizer.min_lr_ratio,
    )
    return TrainingRuntimeComponents(
        params=params,
        named_params=named_params,
        optimizer_result=optimizer_result,
        scheduler=scheduler,
        warmup_warning=str(warning or ""),
    )
