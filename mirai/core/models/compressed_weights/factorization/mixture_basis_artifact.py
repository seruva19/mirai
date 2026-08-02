"""Offline schema-v4 mixture-of-basis packed-artifact transformation."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from mirai.core.models.compressed_weights.artifact_source import (
    load_grouped_expert_source,
)
from mirai.core.models.compressed_weights.factorization.mixture_basis import (
    MIXTURE_BASIS_PROJECTIONS,
)
from mirai.core.models.compressed_weights.factorization.mixture_basis import (
    MIXTURE_BASIS_PROVIDER_NAME,
)
from mirai.core.models.compressed_weights.factorization.mixture_basis import (
    MIXTURE_BASIS_PROVIDER_SCHEMA_VERSION,
)
from mirai.core.models.compressed_weights.factorization.mixture_basis import (
    factorize_mixture_basis_experts,
)
from mirai.core.models.compressed_weights.quantization.blockwise_fp8 import (
    BLOCKWISE_FP8_FORMATS,
)
from mirai.core.models.compressed_weights.quantization.gguf_quant import GGUF_FORMATS
from mirai.core.models.compressed_weights.quantization.microscaling_quant import (
    MICROSCALING_FORMATS,
)
from mirai.core.models.compressed_weights.quantization.quant import (
    normalize_quant_format,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _down_projection_spec(
    source_spec: Mapping[str, Any],
    source_tensor_map: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    quant_format = normalize_quant_format(
        str(source_spec.get("quant_format", "int8"))
    )
    if quant_format == "nf4":
        role_map = {
            "packed": "w2_nf4",
            "absmax": "w2_nf4_absmax",
            "nested_absmax": "w2_nf4_nabsmax",
            "offset": "w2_nf4_offset",
            "code": "w2_nf4_code",
            "nested_code": "w2_nf4_ncode",
        }
    elif quant_format in GGUF_FORMATS:
        role_map = {"blocks": "w2_gguf"}
    elif quant_format in MICROSCALING_FORMATS:
        role_map = {
            "packed": "w2_mx",
            "scales": "w2_mx_scale",
            "global_scale": "w2_mx_global",
        }
    elif quant_format in BLOCKWISE_FP8_FORMATS:
        role_map = {
            "codes": "w2_fp8",
            "scales": "w2_fp8_scale",
        }
    else:
        role_map = {
            "quantized": "w2_int8",
            "scale": "w2_scale",
        }
    tensor_names: dict[str, str] = {}
    for role, source_role in role_map.items():
        name = str(source_tensor_map.get(source_role, ""))
        if not name:
            raise KeyError(
                f"Grouped expert manifest has no tensor name for {source_role!r}."
            )
        tensor_names[role] = name
    down_spec: dict[str, Any] = {
        "quant_format": quant_format,
        "tensors": tensor_names,
    }
    if quant_format == "nf4":
        metadata = source_spec.get("nf4_meta")
        if not isinstance(metadata, Mapping):
            raise ValueError("NF4 source has no nf4_meta object.")
        down_spec["nf4_meta"] = copy.deepcopy(dict(metadata))
    elif quant_format in MICROSCALING_FORMATS:
        metadata = source_spec.get("microscaling_meta")
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("w2"), Mapping
        ):
            raise ValueError(
                "Microscaling source has no w2 metadata object."
            )
        down_spec["microscaling_meta"] = copy.deepcopy(
            dict(metadata["w2"])
        )
    elif quant_format in BLOCKWISE_FP8_FORMATS:
        metadata = source_spec.get("blockwise_fp8_meta")
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("w2"), Mapping
        ):
            raise ValueError("Blockwise FP8 source has no w2 metadata object.")
        down_spec["blockwise_fp8_meta"] = copy.deepcopy(dict(metadata["w2"]))
    elif quant_format not in GGUF_FORMATS:
        group_sizes = source_spec.get("group_sizes")
        if not isinstance(group_sizes, Mapping):
            raise ValueError("INT8 source has no group_sizes object.")
        down_spec["group_size"] = int(group_sizes.get("w2", 0))
    return down_spec, tensor_names


def factorize_packed_state_mixture_basis(
    tensors: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    rank: int,
    basis_count: int,
    activation: str,
    optimization_steps: int,
    learning_rate: float,
    expert_batch_size: int,
    row_chunk_size: int,
    checkpoint_interval: int,
    factor_dtype: str,
    device: Any,
    source_artifact_fingerprint: str,
    max_covariance_gib: float = 2.0,
    max_optimizer_gib: float = 24.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replace packed w1/w3 tensors with optimized mixture-basis factors."""
    if torch is None:  # pragma: no cover
        raise RuntimeError(
            "Mixture-basis artifact transformation requires torch."
        )
    if manifest.get("format") != "mirai.compressed_weights.packed_state":
        raise ValueError(
            f"Unsupported packed-state format {manifest.get('format')!r}."
        )
    schema_version = int(manifest.get("schema_version", 0))
    if schema_version not in {1, 2, 3}:
        raise ValueError(
            "Mixture-basis transformation requires a schema-v1/v2/v3 "
            "source artifact."
        )
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("Packed-state manifest must contain modules.")
    source_fingerprint = str(source_artifact_fingerprint).strip()
    if not source_fingerprint:
        raise ValueError(
            "Mixture-basis transformation requires a source artifact fingerprint."
        )

    output_tensors = {str(key): value for key, value in tensors.items()}
    output_manifest = copy.deepcopy(dict(manifest))
    output_manifest["schema_version"] = 4
    report_modules: dict[str, Any] = {}
    grouped_count = 0
    for raw_name, raw_spec in output_manifest["modules"].items():
        module_name = str(raw_name)
        if (
            not isinstance(raw_spec, dict)
            or str(raw_spec.get("kind")) != "grouped_experts"
        ):
            continue
        if "physical_weight_provider" in raw_spec:
            raise ValueError(
                f"Module {module_name!r} already has a physical provider."
            )
        source_tensor_map = raw_spec.get("tensors")
        if not isinstance(source_tensor_map, Mapping):
            raise ValueError(
                f"Grouped module {module_name!r} has no tensor map."
            )
        source_keys = {str(value) for value in source_tensor_map.values()}
        missing = sorted(source_keys - set(output_tensors))
        if missing:
            raise KeyError(
                f"Grouped module {module_name!r} is missing tensors: {missing}."
            )
        source_bytes = sum(
            _tensor_bytes(output_tensors[key]) for key in source_keys
        )
        source_module = load_grouped_expert_source(
            raw_spec,
            output_tensors,
        )
        down_spec, down_tensor_map = _down_projection_spec(
            raw_spec,
            source_tensor_map,
        )
        down_keys = set(down_tensor_map.values())
        projection_specs: dict[str, Any] = {}
        provider_tensor_map: dict[str, str] = {
            f"w2_{role}": name for role, name in down_tensor_map.items()
        }
        errors: dict[str, float] = {}
        initial_errors: dict[str, float] = {}
        optimized_errors: dict[str, float] = {}
        mean_to_std: dict[str, float] = {}
        for key in sorted(MIXTURE_BASIS_PROJECTIONS):
            dense = torch.stack(
                [
                    source_module._dequantize_expert(
                        key,
                        expert_index,
                        dtype=torch.float32,
                        device=torch.device(device),
                    )
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    for expert_index in range(source_module.num_experts)
                ],
                dim=0,
            )
            factors = factorize_mixture_basis_experts(
                dense,
                rank=int(rank),
                basis_count=int(basis_count),
                activation=activation,
                optimization_steps=int(optimization_steps),
                learning_rate=float(learning_rate),
                expert_batch_size=int(expert_batch_size),
                row_chunk_size=int(row_chunk_size),
                checkpoint_interval=int(checkpoint_interval),
                factor_dtype=factor_dtype,
                device=torch.device(device),
                max_covariance_gib=float(max_covariance_gib),
                max_optimizer_gib=float(max_optimizer_gib),
            )
            transform_name = f"{module_name}.{key}_mixture_transforms"
            basis_name = f"{module_name}.{key}_mixture_bases"
            coefficient_name = f"{module_name}.{key}_mixture_coefficients"
            output_tensors[transform_name] = factors.transforms
            output_tensors[basis_name] = factors.bases
            output_tensors[coefficient_name] = factors.coefficients
            provider_tensor_map[f"{key}_transforms"] = transform_name
            provider_tensor_map[f"{key}_bases"] = basis_name
            provider_tensor_map[f"{key}_coefficients"] = coefficient_name
            projection_specs[key] = {
                "rank": factors.rank,
                "basis_count": factors.basis_count,
                "activation": factors.activation,
                "transforms": transform_name,
                "bases": basis_name,
                "coefficients": coefficient_name,
            }
            initial_errors[key] = (
                factors.initial_relative_frobenius_error
            )
            optimized_errors[key] = (
                factors.optimized_relative_frobenius_error
            )
            errors[key] = factors.stored_relative_frobenius_error
            mean_to_std[key] = factors.mean_to_std_ratio
            del dense

        for tensor_key in source_keys - down_keys:
            del output_tensors[tensor_key]
        factor_keys = set(provider_tensor_map.values())
        factor_bytes = sum(
            _tensor_bytes(output_tensors[key]) for key in factor_keys
        )
        if factor_bytes >= source_bytes:
            raise ValueError(
                f"Mixture-basis module {module_name!r} would not reduce "
                f"storage ({factor_bytes} >= {source_bytes} bytes)."
            )
        raw_spec["tensors"] = provider_tensor_map
        raw_spec["physical_weight_provider"] = {
            "name": MIXTURE_BASIS_PROVIDER_NAME,
            "schema_version": MIXTURE_BASIS_PROVIDER_SCHEMA_VERSION,
            "factor_dtype": str(factor_dtype),
            "projections": projection_specs,
            "down_projection": down_spec,
        }
        report_modules[module_name] = {
            "source_bytes": source_bytes,
            "factor_bytes": factor_bytes,
            "byte_ratio": factor_bytes / max(1, source_bytes),
            "initial_relative_frobenius_error": initial_errors,
            "optimized_relative_frobenius_error": optimized_errors,
            "relative_frobenius_error": errors,
            "mean_to_std_ratio": mean_to_std,
            "unchanged_projection": "w2",
        }
        grouped_count += 1
    if grouped_count == 0:
        raise ValueError(
            "Packed artifact contains no grouped experts to factorize."
        )
    output_manifest["physical_weight_providers"] = [
        MIXTURE_BASIS_PROVIDER_NAME
    ]
    output_manifest["mixture_basis_transform"] = {
        "format": "mirai.moe.mixture_basis_transform",
        "schema_version": 1,
        "source_artifact_fingerprint": source_fingerprint,
        "rank": int(rank),
        "basis_count": int(basis_count),
        "activation": str(activation).strip().lower(),
        "optimization_steps": int(optimization_steps),
        "learning_rate": float(learning_rate),
        "expert_batch_size": int(expert_batch_size),
        "row_chunk_size": int(row_chunk_size),
        "checkpoint_interval": int(checkpoint_interval),
    }
    source_total = sum(
        item["source_bytes"] for item in report_modules.values()
    )
    factor_total = sum(
        item["factor_bytes"] for item in report_modules.values()
    )
    report = {
        "provider": MIXTURE_BASIS_PROVIDER_NAME,
        "rank": int(rank),
        "basis_count": int(basis_count),
        "activation": str(activation).strip().lower(),
        "optimization_steps": int(optimization_steps),
        "learning_rate": float(learning_rate),
        "factor_dtype": str(factor_dtype),
        "source_artifact_fingerprint": source_fingerprint,
        "grouped_modules": grouped_count,
        "source_expert_bytes": source_total,
        "factor_bytes": factor_total,
        "byte_ratio": factor_total / max(1, source_total),
        "modules": report_modules,
    }
    if not math.isfinite(float(report["byte_ratio"])):
        raise RuntimeError(
            "Mixture-basis artifact byte ratio is non-finite."
        )
    return output_tensors, output_manifest, report


__all__ = ["factorize_packed_state_mixture_basis"]
