"""Resume checkpoint compatibility validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mirai.config.schema import (
    AdapterConfig,
    DatasetConfig,
    MemoryConfig,
    ModelParams,
    OptimizerConfig,
    TrainingSection,
)


@dataclass(frozen=True)
class ResumeValidationResult:
    checkpoint_path: str
    checkpoint_global_step: int
    lineage_verified: bool
    warnings: list[str]


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    try:
        return asdict(config)
    except Exception:
        return {}


def _with_defaults(payload: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    result = dict(defaults)
    if isinstance(payload, dict):
        result.update(payload)
    return result


def _resume_contract_from_config(config: Any) -> dict[str, Any]:
    payload = _config_dict(config)
    model_payload = payload.get("model", {})
    model_params = _with_defaults(model_payload.get("params", {}), asdict(ModelParams()))
    training_payload = _with_defaults(
        payload.get("training", {}), asdict(TrainingSection())
    )
    memory_payload = _with_defaults(payload.get("memory", {}), asdict(MemoryConfig()))
    dataset_payload = _with_defaults(payload.get("dataset", {}), asdict(DatasetConfig()))
    optimizer_payload = _with_defaults(
        payload.get("optimizer", {}), asdict(OptimizerConfig())
    )
    # Normalize the adapter snapshot against the current AdapterConfig defaults so
    # that a checkpoint written before a new adapter field existed still compares
    # equal when the current config leaves that field at its default. Without this,
    # every added adapter field silently breaks resume for pre-existing checkpoints.
    adapter_payload = _with_defaults(payload.get("adapter", {}), asdict(AdapterConfig()))
    return {
        "model": {
            "type": model_payload.get("type"),
            "provider_module": model_payload.get("provider_module", ""),
            "dtype": model_payload.get("dtype"),
            "attention_backend": model_payload.get("attention_backend"),
            "params": model_params,
        },
        "strategy": payload.get("strategy", {}),
        "optimizer": optimizer_payload,
        "adapter": {
            key: value
            for key, value in adapter_payload.items()
            if key not in {"init_from"}
        },
        "training": {
            key: value
            for key, value in training_payload.items()
            if key
            in {
                "batch_size",
                "gradient_accumulation",
                "loss_function",
                "objective",
                "contrastive_flow_weight",
                "loss_weighting",
                "min_snr_gamma",
                "timestep_eps",
                "timestep_sampling",
                "timestep_sampling_mean",
                "timestep_sampling_std",
                "timestep_sampling_mode_scale",
                "prior_loss_weight",
                "prior_ratio",
                "gradient_checkpointing",
                "noise_offset",
                "loss_bucket_normalization",
                "blocks_to_swap",
                "block_swap_mode",
                "optimizer_cpu_offload",
                "gradient_cpu_offload",
                "activation_cpu_offload",
                "activation_cpu_offload_defer_layers",
                "activation_cpu_offload_prefetch_layers",
                "activation_cpu_offload_view_replay",
                "masked_loss",
                "compile",
                "block_swap_backward",
                "policy_modules",
                "policy_options",
            }
        },
        "dataset": {
            key: value
            for key, value in dataset_payload.items()
            if key
            in {
                "online_tag_shuffle",
                "online_tag_shuffle_dropout",
                "online_tag_shuffle_keep_first_n_tags",
                "online_temporal_resampling",
                "clips_per_video",
                "tag_shuffle_variants",
                "mask_extension",
                "moe_routing",
            }
        },
        "memory": {
            key: value
            for key, value in memory_payload.items()
            if key
            in {
                "frozen_weight_quantization",
                "frozen_weight_quantization_strategy",
                "frozen_weight_packed_state_path",
                "expert_precision_plan_path",
                "quantization_block_size",
                "weight_residency_strategy",
                "expert_weight_access",
                "expert_dequant_chunk_size",
                "quantize_experts_on_load",
                "router_quantization",
                "router_quantization_calibration_path",
                "moe_kernel_backend",
                "cuda_memory_fraction",
                "minimum_system_memory_gib",
                "trainable_parameter_offload",
            }
        },
    }


def _payload_or_metadata(payload: dict[str, Any], key: str) -> str:
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    return str(payload.get(key) or metadata_dict.get(key) or "").strip()


def validate_resume_checkpoint_compatibility(
    *,
    payload: dict[str, Any],
    config: Any,
    resume_path: str,
    dataset_snapshot_id: str,
    cache_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str = "",
    model_component_id: str = "",
) -> ResumeValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    lineage_verified = True

    for key in (
        "manifest_sha256",
        "dataset_snapshot_id",
        "cache_snapshot_id",
        "model_snapshot_id",
        "config_snapshot_id",
        "model_component_id",
    ):
        top_level = str(payload.get(key) or "").strip()
        nested = str(metadata_dict.get(key) or "").strip()
        if top_level and nested and top_level != nested:
            errors.append(
                f"checkpoint metadata is internally inconsistent for '{key}': "
                f"top-level has '{top_level}' but metadata has '{nested}'."
            )

    lineage_expectations = {
        "dataset_snapshot_id": str(dataset_snapshot_id),
        "cache_snapshot_id": str(cache_snapshot_id),
        "model_snapshot_id": str(model_snapshot_id),
    }
    for key, expected in lineage_expectations.items():
        observed = _payload_or_metadata(payload, key)
        if not observed:
            warnings.append(
                f"Resume checkpoint is missing {key}; lineage verification for this field was skipped."
            )
            lineage_verified = False
            continue
        if observed != expected:
            errors.append(
                f"{key} mismatch: checkpoint has '{observed}' but current run expects '{expected}'."
            )
    expected_config_snapshot_id = str(config_snapshot_id).strip()
    if expected_config_snapshot_id:
        observed_config_snapshot_id = _payload_or_metadata(payload, "config_snapshot_id")
        if not observed_config_snapshot_id:
            warnings.append(
                "Resume checkpoint is missing config_snapshot_id; config-file lineage verification was skipped."
            )
            lineage_verified = False
        elif observed_config_snapshot_id != expected_config_snapshot_id:
            warnings.append(
                "Resume checkpoint was created from a different config snapshot id "
                f"('{observed_config_snapshot_id}' != '{expected_config_snapshot_id}'). "
                "The resolved training contract still matched, so resume remains allowed."
            )
            lineage_verified = False

    expected_model_component_id = str(model_component_id).strip()
    if expected_model_component_id:
        observed_model_component_id = _payload_or_metadata(payload, "model_component_id")
        if observed_model_component_id and observed_model_component_id != expected_model_component_id:
            errors.append(
                "model_component_id mismatch: checkpoint has "
                f"'{observed_model_component_id}' but current run expects "
                f"'{expected_model_component_id}'."
            )

    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        warnings.append(
            "Resume checkpoint is missing embedded config; training contract compatibility checks were skipped."
        )
    else:
        current_contract = _resume_contract_from_config(config)
        checkpoint_contract = _resume_contract_from_config(checkpoint_config)
        for section, expected in current_contract.items():
            observed = checkpoint_contract.get(section)
            if observed != expected:
                errors.append(
                    f"resume contract mismatch in section '{section}'."
                )

    if errors:
        raise ValueError(
            "Resume checkpoint is incompatible with the current run:\n- "
            + "\n- ".join(errors)
        )

    return ResumeValidationResult(
        checkpoint_path=str(resume_path),
        checkpoint_global_step=int(payload.get("global_step", 0)),
        lineage_verified=bool(lineage_verified),
        warnings=list(warnings),
    )
