"""Final artifact export and run-summary helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from mirai.config.schema import TrainingConfig
from mirai.core.training.adapters import (
    adapter_metadata_from_state,
    save_kohya_adapter_safetensors,
    save_lycoris_adapter_safetensors,
)
from mirai.core.lineage import snapshot_component_id
from mirai.core.persistence.checkpoints import save_checkpoint
from mirai.core.training.residency.memory import current_resource_telemetry
from mirai.core.training.lifecycle.run_contract import build_final_run_summary


@dataclass(frozen=True)
class FinalRunArtifacts:
    summary: dict[str, Any]
    final_checkpoint_path: Path
    adapter_checkpoint_path: Path
    adapter_safetensors_path: Path
    adapter_ema_safetensors_path: Path | None
    adapter_lycoris_path: str


def finalize_training_artifacts(
    *,
    config: TrainingConfig,
    run_id: str,
    output_dir: str | Path,
    ckpt_dir: str | Path,
    trainer: Any,
    manifest_path: str | Path,
    manifest_sha256: str,
    build_ckpt_payload: Callable[[int], dict[str, Any]],
    global_step: int,
    stopped_early: bool,
    skipped_steps: int,
    optimizer_type: str,
    optimizer_fallback_used: bool,
    lr: float,
    last_metrics: dict[str, Any],
    ema_state: dict[str, Any] | None,
    gradient_offload_ops: int,
    optimizer_offload_ops: int,
    compile_enabled: bool,
    compile_warning: str,
    runtime_policy: dict[str, Any],
    runtime_policy_notes: list[str],
    log_on_this_rank: bool,
    dataset_snapshot_id: str = "",
    cache_snapshot_id: str = "",
    model_snapshot_id: str = "",
    config_snapshot_id: str = "",
    model_component_id: str = "",
    cache_data_access_mode: str = "",
    indexed_cache_enabled: bool = False,
    indexed_cache_metadata_path: str = "",
    indexed_cache_tensor_path: str = "",
    training_sampling_mode: str = "",
    resumed: bool = False,
    resume_checkpoint_path: str = "",
    resume_checkpoint_step: int = 0,
    resume_lineage_verified: bool = False,
    resume_validation_warnings: list[str] | None = None,
    resource_telemetry: dict[str, Any] | None = None,
) -> FinalRunArtifacts:
    output_root = Path(output_dir)
    checkpoint_root = Path(ckpt_dir)
    final_path = checkpoint_root / "last.pt"
    adapter_path = output_root / "adapter.pt"
    adapter_safetensors_path = output_root / "adapter.safetensors"
    adapter_ema_safetensors_path = (
        output_root / "adapter_ema.safetensors" if config.training.ema_enabled else None
    )
    adapter_type = str(config.adapter.type).strip().lower()
    adapter_lycoris_path = ""
    model_component = str(
        getattr(config.model.params, "denoiser_subfolder", "transformer") or "transformer"
    )
    resolved_model_component_id = str(model_component_id).strip() or snapshot_component_id(
        model_component,
        component_label="denoiser_subfolder",
    )
    resolved_resource_telemetry = (
        dict(resource_telemetry)
        if isinstance(resource_telemetry, dict)
        else current_resource_telemetry()
    )
    training_policies = getattr(trainer, "training_policies", None)
    policy_metadata = (
        training_policies.checkpoint_metadata()
        if training_policies is not None
        else {"active": [], "policies": {}}
    )
    active_policies = list(getattr(training_policies, "active_names", ()))
    summary = build_final_run_summary(
        status="completed" if not stopped_early else "early_stopped",
        run_id=str(run_id),
        manifest_path=str(manifest_path),
        manifest_sha256=str(manifest_sha256),
        dataset_snapshot_id=str(dataset_snapshot_id),
        cache_snapshot_id=str(cache_snapshot_id),
        model_snapshot_id=str(model_snapshot_id),
        config_snapshot_id=str(config_snapshot_id),
        model_component_id=str(resolved_model_component_id),
        optimizer_type=str(optimizer_type),
        optimizer_fallback_used=bool(optimizer_fallback_used),
        lr=float(lr),
        global_step=int(global_step),
        skipped_steps=int(skipped_steps),
        last_checkpoint=str(final_path),
        adapter_path=str(adapter_path),
        adapter_safetensors_path=str(adapter_safetensors_path),
        adapter_type=adapter_type,
        adapter_lycoris_path=adapter_lycoris_path,
        last_metrics=dict(last_metrics),
        ema_enabled=bool(config.training.ema_enabled),
        gradient_cpu_offload=bool(config.training.gradient_cpu_offload),
        optimizer_cpu_offload=bool(config.training.optimizer_cpu_offload),
        gradient_offload_ops=int(gradient_offload_ops),
        optimizer_offload_ops=int(optimizer_offload_ops),
        compile_enabled=bool(compile_enabled),
        compile_warning=str(compile_warning),
        cache_data_access_mode=str(cache_data_access_mode),
        indexed_cache_enabled=bool(indexed_cache_enabled),
        indexed_cache_metadata_path=str(indexed_cache_metadata_path),
        indexed_cache_tensor_path=str(indexed_cache_tensor_path),
        training_sampling_mode=str(training_sampling_mode),
        resumed=bool(resumed),
        resume_checkpoint_path=str(resume_checkpoint_path),
        resume_checkpoint_step=int(resume_checkpoint_step),
        resume_lineage_verified=bool(resume_lineage_verified),
        resume_validation_warnings=list(resume_validation_warnings or []),
        resource_telemetry=resolved_resource_telemetry,
        runtime_policy=dict(runtime_policy),
        runtime_policy_notes=list(runtime_policy_notes),
    )
    if not log_on_this_rank:
        return FinalRunArtifacts(
            summary=summary,
            final_checkpoint_path=final_path,
            adapter_checkpoint_path=adapter_path,
            adapter_safetensors_path=adapter_safetensors_path,
            adapter_ema_safetensors_path=adapter_ema_safetensors_path,
            adapter_lycoris_path=adapter_lycoris_path,
        )

    save_checkpoint(final_path, build_ckpt_payload(global_step))
    live_adapter_state = trainer.pipeline.state_dict()
    allocation_metadata: dict[str, str] = {}
    if (
        config.adapter.rank_pattern
        or config.adapter.alpha_pattern
        or config.adapter.rank_budget > 0
        or config.adapter.use_rslora
    ):
        allocation_metadata = {
            "mirai_adapter_rank_pattern": json.dumps(
                config.adapter.rank_pattern, sort_keys=True, separators=(",", ":")
            ),
            "mirai_adapter_alpha_pattern": json.dumps(
                config.adapter.alpha_pattern, sort_keys=True, separators=(",", ":")
            ),
            "mirai_adapter_rank_budget": str(config.adapter.rank_budget),
            "mirai_adapter_scaling_rule": (
                "alpha_over_sqrt_rank"
                if config.adapter.use_rslora
                else "alpha_over_rank"
            ),
        }
    gora_metadata = trainer.pipeline.get_gora_allocation_metadata()
    if gora_metadata:
        allocation_metadata.update(
            {
                "mirai_adapter_gora_ranks": json.dumps(
                    gora_metadata["ranks"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "mirai_adapter_gora_fingerprint": str(
                    gora_metadata["fingerprint"]
                ),
                "mirai_adapter_scaling_rule": "alpha_over_sqrt_rank",
            }
        )
    initializer_name = str(config.adapter.lora_init).strip().lower()
    initializer_metadata = (
        {"mirai_adapter_initializer": initializer_name}
        if initializer_name != "kaiming"
        else {}
    )
    if initializer_name == "eva":
        initializer_metadata.update(
            {
                "mirai_adapter_eva_calibration_steps": str(
                    config.adapter.eva_calibration_steps
                ),
                "mirai_adapter_eva_samples_per_target": str(
                    config.adapter.eva_samples_per_target
                ),
                "mirai_adapter_eva_convergence_threshold": str(
                    config.adapter.eva_convergence_threshold
                ),
                "mirai_adapter_eva_rank_mode": "fixed_rho_1",
            }
        )
    if initializer_name == "gora":
        initializer_metadata.update(
            {
                "mirai_adapter_gora_calibration_steps": str(
                    config.adapter.gora_calibration_steps
                ),
                "mirai_adapter_gora_min_rank": str(
                    config.adapter.gora_min_rank
                ),
                "mirai_adapter_gora_max_rank": str(
                    config.adapter.gora_max_rank
                ),
                "mirai_adapter_gora_stable_gamma": str(
                    config.adapter.gora_stable_gamma
                ),
            }
        )
    training_policy_metadata = (
        {"mirai_adapter_training_policy": "lora_fa"}
        if bool(config.adapter.use_lora_fa)
        else {}
    )
    adapter_variant_metadata = (
        {"mirai_adapter_variant": "dora"}
        if bool(config.adapter.use_dora)
        else {}
    )
    adapter_contract_metadata = {
        **allocation_metadata,
        **initializer_metadata,
        **training_policy_metadata,
        **adapter_variant_metadata,
    }
    adapter_payload = {
        "adapter_state": live_adapter_state,
        "adapter_state_ema": ema_state if config.training.ema_enabled else None,
        "strategy_metadata": trainer.strategy.get_checkpoint_metadata(),
        "global_step": int(global_step),
        "manifest_sha256": str(manifest_sha256),
        "dataset_snapshot_id": str(dataset_snapshot_id),
        "cache_snapshot_id": str(cache_snapshot_id),
        "model_snapshot_id": str(model_snapshot_id),
        "config_snapshot_id": str(config_snapshot_id),
        "model_component_id": str(resolved_model_component_id),
        "metadata": {
            "schema_version": 1,
            "manifest_sha256": str(manifest_sha256),
            "dataset_snapshot_id": str(dataset_snapshot_id),
            "cache_snapshot_id": str(cache_snapshot_id),
            "model_snapshot_id": str(model_snapshot_id),
            "config_snapshot_id": str(config_snapshot_id),
            "model_component_id": str(resolved_model_component_id),
        },
    }
    if active_policies:
        adapter_payload["training_policy_metadata"] = policy_metadata
        adapter_payload["metadata"]["training_policies"] = active_policies
    if allocation_metadata:
        adapter_payload["metadata"]["adapter_allocation"] = allocation_metadata
    if initializer_metadata:
        adapter_payload["metadata"]["adapter_initializer"] = initializer_metadata
    if training_policy_metadata:
        adapter_payload["metadata"]["adapter_training_policy"] = (
            training_policy_metadata
        )
    if adapter_variant_metadata:
        adapter_payload["metadata"]["adapter_variant"] = (
            adapter_variant_metadata
        )
    save_checkpoint(adapter_path, adapter_payload)
    target_presets = trainer.pipeline.get_adapter_target_presets()
    target_modules = target_presets.get(config.adapter.target_preset, [])
    save_kohya_adapter_safetensors(
        adapter_safetensors_path,
        adapter_state=live_adapter_state,
        rank=config.adapter.rank,
        alpha=config.adapter.alpha,
        target_modules=target_modules,
        model_path=config.model.path,
        model_component=model_component,
        metadata_extra={
            **adapter_metadata_from_state(
                adapter_state=live_adapter_state,
                model_path=config.model.path,
                target_modules=target_modules,
                model_component=model_component,
            ),
            **adapter_contract_metadata,
        },
    )
    if adapter_type in {"loha", "lokr"}:
        lycoris_file = output_root / "adapter_lycoris.safetensors"
        save_lycoris_adapter_safetensors(
            lycoris_file,
            adapter_state=live_adapter_state,
            adapter_type=config.adapter.type,
            rank=config.adapter.rank,
            alpha=config.adapter.alpha,
            target_modules=target_modules,
            model_path=config.model.path,
            model_component=model_component,
            metadata_extra={"adapter_type": adapter_type},
        )
        adapter_lycoris_path = str(lycoris_file)
        summary["adapter_lycoris_path"] = adapter_lycoris_path
    if config.training.ema_enabled and ema_state is not None and adapter_ema_safetensors_path is not None:
        save_kohya_adapter_safetensors(
            adapter_ema_safetensors_path,
            adapter_state=ema_state,
            rank=config.adapter.rank,
            alpha=config.adapter.alpha,
            target_modules=target_modules,
            model_path=config.model.path,
            model_component=model_component,
            metadata_extra=adapter_contract_metadata,
        )
    if resource_telemetry is None:
        summary["resource_telemetry"] = current_resource_telemetry()
    return FinalRunArtifacts(
        summary=summary,
        final_checkpoint_path=final_path,
        adapter_checkpoint_path=adapter_path,
        adapter_safetensors_path=adapter_safetensors_path,
        adapter_ema_safetensors_path=adapter_ema_safetensors_path,
        adapter_lycoris_path=adapter_lycoris_path,
    )
