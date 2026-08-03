"""Step metrics and validation policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mirai.config.schema import TrainingConfig
from mirai.core.persistence.checkpoints import save_checkpoint
from mirai.core.training.evaluation.early_stop import EarlyStopState


@dataclass(frozen=True)
class ValidationPolicyResult:
    step_metrics: dict[str, Any]
    early_stop_state: EarlyStopState
    stopped_early: bool
    best_checkpoint_saved: bool
    best_checkpoint_path: str


def build_step_metrics(
    *,
    config: TrainingConfig,
    last_metrics: dict[str, Any],
    lr: float,
    grad_norm: float,
    skipped_steps: int,
    vram_used_mb: float,
    curriculum_resolution: Any = None,
    curriculum_frame_count: Any = None,
    gradient_offload_ops: int = 0,
    optimizer_offload_ops: int = 0,
    grad_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step_metrics = {
        "loss": float(last_metrics.get("loss", 0.0)),
        "lr": float(lr),
        "grad_norm": float(grad_norm),
        "skipped_steps": int(skipped_steps),
        "vram_used_mb": float(vram_used_mb),
    }
    # Per-step MoE auxiliary VALUES (raw, unweighted): surfaced into
    # metrics.jsonl / step.completed events even when the corresponding loss
    # weights are 0 so router health stays observable (monitoring-gap fix).
    diagnostics = last_metrics.get("diagnostics") or {}
    if isinstance(diagnostics, dict):
        for aux_key in (
            "moe_balance_loss",
            "moe_z_loss",
            # Detached routing-stability scalars from already-collected loads.
            # KL-to-step-zero is absent until the reference snapshot exists.
            "moe_routing_entropy",
            "moe_utilization_cv",
            "moe_top1_monopoly",
            "moe_routing_kl_vs_step0",
            "moe_step_unique_experts",
            # Opt-in Expert-Choice coverage guard.
            "moe_expert_choice_coverage_fraction",
            "moe_expert_choice_min_coverage_fraction",
            "moe_expert_choice_coverage_alarm",
            # Opt-in homogenization and per-depth deadlock alarms.
            "moe_expert_output_cossim",
            "moe_fisher_specialization_index",
            "moe_fisher_specialization_fraction",
            "moe_fisher_specialization_min_layer",
            "moe_fisher_specialization_max_layer",
            "moe_fisher_specialization_layer_count",
            "moe_router_weight_similarity",
            "moe_router_conditioning_ratio",
            "moe_router_per_token_entropy",
            "moe_router_per_token_entropy_fraction",
            "moe_router_mechanism_layer_count",
            "moe_router_conditioning_layer_count",
            "moe_attention_qk_delta2_effective_rank",
            "moe_attention_qk_delta2_effective_rank_min",
            "moe_attention_qk_delta2_effective_rank_max",
            "moe_attention_qk_delta2_spectral_entropy",
            "moe_attention_qk_delta2_head_count",
            "moe_attention_qk_delta2_layer_count",
            "moe_max_deadlock_duration",
            "moe_deadlocked_layer_count",
            "moe_deadlocked_layer_count_depth_q1",
            "moe_deadlocked_layer_count_depth_q2",
            "moe_deadlocked_layer_count_depth_q3",
            "moe_deadlocked_layer_count_depth_q4",
            "moe_max_deadlock_duration_depth_q1",
            "moe_max_deadlock_duration_depth_q2",
            "moe_max_deadlock_duration_depth_q3",
            "moe_max_deadlock_duration_depth_q4",
            "moe_router_logit_drift",
            "moe_expert_touch_fraction",
            # Opt-in LongCat Eq. 9 gradients on batch-collapsed router
            # probabilities. These require an extra graph traversal and are
            # absent when the diagnostic is disabled or the task has no route
            # probability gradient.
            "moe_balance_grad_ratio",
            "moe_balance_grad_ratio_max_layer",
            "moe_balance_grad_ratio_task_norm",
            "moe_balance_grad_ratio_objective_norm",
            "moe_balance_grad_ratio_alarm",
            "moe_phi_balance_grad_ratio",
            "moe_phi_balance_grad_ratio_max_layer",
            "moe_phi_balance_grad_ratio_task_norm",
            "moe_phi_balance_grad_ratio_objective_norm",
            "moe_phi_balance_grad_ratio_alarm",
        ):
            value = diagnostics.get(aux_key)
            if value is not None:
                try:
                    step_metrics[aux_key] = float(value)
                except (TypeError, ValueError):
                    pass
        touch_exceeded = diagnostics.get("moe_expert_touch_exceeded")
        if touch_exceeded is not None:
            step_metrics["moe_expert_touch_exceeded"] = bool(touch_exceeded)
        # Timestep-axis adapter state (T-LoRA rank schedule / bands): compact
        # per-step scalars so the rank mask is observable in metrics.jsonl.
        # Absent unless the opt-in policy is active (defaults unchanged).
        timestep_adapter = diagnostics.get("timestep_adapter")
        if isinstance(timestep_adapter, dict):
            for src_key, dst_key in (
                ("sigma", "timestep_sigma_mean"),
                ("active_rank_fraction", "timestep_active_rank_fraction"),
                ("band_gate", "timestep_band_gate_fraction"),
            ):
                values = timestep_adapter.get(src_key)
                if isinstance(values, (list, tuple)) and values:
                    try:
                        step_metrics[dst_key] = float(
                            sum(float(v) for v in values) / len(values)
                        )
                    except (TypeError, ValueError):
                        pass
            # TC-LoRA gate mean/std (collapse-to-zero / saturation watch).
            # Already-scalar per-step values; only present under the tc_gate
            # schedule so defaults stay unchanged.
            for scalar_key in ("tc_gate_mean", "tc_gate_std"):
                scalar = timestep_adapter.get(scalar_key)
                if scalar is not None:
                    try:
                        step_metrics[scalar_key] = float(scalar)
                    except (TypeError, ValueError):
                        pass
    if curriculum_resolution is not None:
        step_metrics["curriculum_resolution"] = str(curriculum_resolution)
    if curriculum_frame_count is not None:
        step_metrics["curriculum_frame_count"] = int(curriculum_frame_count)
    if config.training.ema_enabled:
        step_metrics["ema_decay"] = float(config.training.ema_decay)
    if config.training.gradient_cpu_offload:
        step_metrics["gradient_cpu_offload_ops"] = int(gradient_offload_ops)
    if config.training.optimizer_cpu_offload:
        step_metrics["optimizer_cpu_offload_ops"] = int(optimizer_offload_ops)
    if grad_breakdown:
        step_metrics.update(dict(grad_breakdown))
    return step_metrics


def apply_validation_policy(
    *,
    step_metrics: dict[str, Any],
    val_loss: float | None,
    early_stop_state: EarlyStopState,
    global_step: int,
    early_stop_patience: int,
    log_on_this_rank: bool,
    ckpt_dir: str | Path,
    build_ckpt_payload: Callable[[int], dict[str, Any]],
) -> ValidationPolicyResult:
    next_metrics = dict(step_metrics)
    next_state = EarlyStopState(
        best_val_loss=float(early_stop_state.best_val_loss),
        best_step=int(early_stop_state.best_step),
        patience_counter=int(early_stop_state.patience_counter),
        best_checkpoint_path=str(early_stop_state.best_checkpoint_path),
    )
    stopped_early = False
    best_checkpoint_saved = False
    best_checkpoint_path = str(next_state.best_checkpoint_path)

    if val_loss is None:
        return ValidationPolicyResult(
            step_metrics=next_metrics,
            early_stop_state=next_state,
            stopped_early=stopped_early,
            best_checkpoint_saved=best_checkpoint_saved,
            best_checkpoint_path=best_checkpoint_path,
        )

    next_metrics["val_loss"] = float(val_loss)
    if float(val_loss) < float(next_state.best_val_loss):
        next_state.best_val_loss = float(val_loss)
        next_state.best_step = int(global_step)
        next_state.patience_counter = 0
        if log_on_this_rank:
            best_path = Path(ckpt_dir) / "best.pt"
            next_state.best_checkpoint_path = str(best_path)
            payload = build_ckpt_payload(global_step)
            payload["early_stop_state"] = next_state.to_dict()
            save_checkpoint(best_path, payload)
            best_checkpoint_saved = True
            best_checkpoint_path = str(best_path)
    else:
        next_state.patience_counter += 1
        patience = int(early_stop_patience)
        if patience > 0 and next_state.patience_counter >= patience:
            stopped_early = True

    return ValidationPolicyResult(
        step_metrics=next_metrics,
        early_stop_state=next_state,
        stopped_early=stopped_early,
        best_checkpoint_saved=best_checkpoint_saved,
        best_checkpoint_path=best_checkpoint_path,
    )
