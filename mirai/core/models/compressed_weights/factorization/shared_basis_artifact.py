"""Offline schema-v4 shared-basis packed-artifact transformation."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from mirai.core.models.compressed_weights.artifact_source import (
    load_grouped_expert_source,
)
from mirai.core.models.compressed_weights.factorization.shared_basis import SHARED_BASIS_PROVIDER_NAME
from mirai.core.models.compressed_weights.factorization.shared_basis import (
    SHARED_BASIS_PROVIDER_SCHEMA_VERSION,
)
from mirai.core.models.compressed_weights.factorization.shared_basis import factorize_dense_experts
from mirai.core.moe.calibration.whitening import EXPERT_WHITENING_FORMAT
from mirai.core.moe.calibration.whitening import ExpertWhiteningEvidence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def factorize_packed_state_shared_basis(
    tensors: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    rank: int,
    device: Any,
    axis: str = "auto",
    factor_dtype: str = "bfloat16",
    reference_experts: Mapping[str, int] | None = None,
    expert_weights: Mapping[str, Any] | None = None,
    calibration_lineage: Mapping[str, str] | None = None,
    whitening_evidence: Mapping[str, ExpertWhiteningEvidence] | None = None,
    whitening_lineage: Mapping[str, str] | None = None,
    whitening_regularization: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replace packed grouped-expert tensors with registered shared factors."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("Shared-basis artifact transformation requires torch.")
    if manifest.get("format") != "mirai.compressed_weights.packed_state":
        raise ValueError(f"Unsupported packed-state format {manifest.get('format')!r}.")
    schema_version = int(manifest.get("schema_version", 0))
    if schema_version not in {1, 2, 3}:
        raise ValueError(
            "Shared-basis transformation requires a schema-v1/v2/v3 source artifact."
        )
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("Packed-state manifest must contain modules.")
    references = {str(key): int(value) for key, value in (reference_experts or {}).items()}
    unknown_references = sorted(set(references) - set(str(key) for key in modules))
    if unknown_references:
        raise ValueError(f"Reference experts name unknown modules: {unknown_references}.")
    weights = {str(key): value for key, value in (expert_weights or {}).items()}
    grouped_names = {
        str(name)
        for name, spec in modules.items()
        if isinstance(spec, Mapping) and str(spec.get("kind")) == "grouped_experts"
    }
    if weights and set(weights) != grouped_names:
        missing = sorted(grouped_names - set(weights))
        unknown = sorted(set(weights) - grouped_names)
        raise ValueError(
            "Affinity weights must exactly cover grouped modules; "
            f"missing={missing}, unknown={unknown}."
        )
    lineage = {str(key): str(value).strip() for key, value in (calibration_lineage or {}).items()}
    lineage_keys = {"dataset_snapshot_id", "model_snapshot_id", "config_snapshot_id"}
    if weights and (set(lineage) != lineage_keys or not all(lineage.values())):
        raise ValueError("Affinity-weighted factorization requires complete calibration lineage.")
    if lineage and not weights:
        raise ValueError("Calibration lineage requires affinity expert weights.")
    whitening = {
        str(key): value.validate()
        for key, value in (whitening_evidence or {}).items()
    }
    if whitening and set(whitening) != grouped_names:
        missing = sorted(grouped_names - set(whitening))
        unknown = sorted(set(whitening) - grouped_names)
        raise ValueError(
            "Whitening evidence must exactly cover grouped modules; "
            f"missing={missing}, unknown={unknown}."
        )
    whitening_metadata = {
        str(key): str(value).strip()
        for key, value in (whitening_lineage or {}).items()
    }
    whitening_lineage_keys = {
        "dataset_snapshot_id",
        "model_snapshot_id",
        "config_snapshot_id",
        "packed_artifact_fingerprint",
    }
    if whitening and (
        set(whitening_metadata) != whitening_lineage_keys
        or not all(whitening_metadata.values())
    ):
        raise ValueError("Whitened factorization requires complete evidence lineage.")
    if whitening_metadata and not whitening:
        raise ValueError("Whitening lineage requires whitening evidence.")

    output_tensors = {str(key): value for key, value in tensors.items()}
    output_manifest = copy.deepcopy(dict(manifest))
    output_manifest["schema_version"] = 4
    report_modules: dict[str, Any] = {}
    grouped_count = 0
    for raw_name, raw_spec in output_manifest["modules"].items():
        module_name = str(raw_name)
        if not isinstance(raw_spec, dict) or str(raw_spec.get("kind")) != "grouped_experts":
            continue
        if "physical_weight_provider" in raw_spec:
            raise ValueError(f"Module {module_name!r} already has a physical provider.")
        source_tensor_map = raw_spec.get("tensors")
        if not isinstance(source_tensor_map, Mapping):
            raise ValueError(f"Grouped module {module_name!r} has no tensor map.")
        source_keys = {str(value) for value in source_tensor_map.values()}
        missing = sorted(source_keys - set(output_tensors))
        if missing:
            raise KeyError(f"Grouped module {module_name!r} is missing tensors: {missing}.")
        source_bytes = sum(_tensor_bytes(output_tensors[key]) for key in source_keys)
        source_module = load_grouped_expert_source(raw_spec, output_tensors)
        reference = references.get(module_name, 0)
        projection_specs: dict[str, Any] = {}
        provider_tensor_map: dict[str, str] = {}
        errors: dict[str, float] = {}
        weighted_errors: dict[str, float] = {}
        whitened_errors: dict[str, float] = {}
        for key in ("w1", "w2", "w3"):
            dense = torch.stack(
                [
                    source_module._dequantize_expert(
                        key,
                        expert_index,
                        dtype=torch.float32,
                        device=torch.device(device),
                    )
                    for expert_index in range(source_module.num_experts)
                ],
                dim=0,
            )
            factors = factorize_dense_experts(
                dense,
                rank=int(rank),
                axis=axis,
                reference_expert=reference,
                factor_dtype=factor_dtype,
                expert_weights=weights.get(module_name),
                input_covariance=(
                    whitening[module_name].projections[key].normalized(
                        device=torch.device(device),
                        dtype=torch.float64,
                    )
                    if module_name in whitening
                    else None
                ),
                whitening_regularization=float(whitening_regularization),
            )
            basis_name = f"{module_name}.{key}_shared_basis"
            coefficient_name = f"{module_name}.{key}_shared_coefficients"
            output_tensors[basis_name] = factors.basis
            output_tensors[coefficient_name] = factors.coefficients
            provider_tensor_map[f"{key}_basis"] = basis_name
            provider_tensor_map[f"{key}_coefficients"] = coefficient_name
            projection_spec = {
                "axis": factors.axis,
                "rank": factors.rank,
                "basis": basis_name,
                "coefficients": coefficient_name,
            }
            if module_name in whitening:
                projection_spec["whitened"] = True
            projection_specs[key] = projection_spec
            errors[key] = factors.relative_frobenius_error
            if factors.weighted_relative_frobenius_error is not None:
                weighted_errors[key] = factors.weighted_relative_frobenius_error
            if factors.whitened_relative_error is not None:
                whitened_errors[key] = factors.whitened_relative_error
            del dense
        for tensor_key in source_keys:
            del output_tensors[tensor_key]
        factor_keys = set(provider_tensor_map.values())
        factor_bytes = sum(_tensor_bytes(output_tensors[key]) for key in factor_keys)
        raw_spec["tensors"] = provider_tensor_map
        provider_spec = {
            "name": SHARED_BASIS_PROVIDER_NAME,
            "schema_version": SHARED_BASIS_PROVIDER_SCHEMA_VERSION,
            "factor_dtype": str(factor_dtype),
            "reference_expert": reference,
            "basis_estimator": (
                (
                    "whitened_affinity_weighted"
                    if module_name in weights
                    else "whitened_population"
                )
                if module_name in whitening
                else (
                    "affinity_weighted"
                    if module_name in weights
                    else "reference_expert"
                )
            ),
            "projections": projection_specs,
        }
        if module_name in whitening:
            provider_spec["whitening_regularization"] = float(
                whitening_regularization
            )
        raw_spec["physical_weight_provider"] = provider_spec
        module_report = {
            "reference_expert": reference,
            "source_bytes": source_bytes,
            "factor_bytes": factor_bytes,
            "byte_ratio": factor_bytes / max(1, source_bytes),
            "relative_frobenius_error": errors,
            "weighted_relative_frobenius_error": weighted_errors,
        }
        if whitened_errors:
            module_report["whitened_relative_error"] = whitened_errors
        report_modules[module_name] = module_report
        grouped_count += 1
    if grouped_count == 0:
        raise ValueError("Packed artifact contains no grouped experts to factorize.")
    output_manifest["physical_weight_providers"] = [SHARED_BASIS_PROVIDER_NAME]
    if weights:
        output_manifest["shared_basis_calibration"] = {
            "format": "mirai.moe.quantization_calibration",
            **lineage,
        }
    if whitening:
        output_manifest["shared_basis_whitening"] = {
            "format": EXPERT_WHITENING_FORMAT,
            "regularization": float(whitening_regularization),
            **whitening_metadata,
        }
    source_total = sum(item["source_bytes"] for item in report_modules.values())
    factor_total = sum(item["factor_bytes"] for item in report_modules.values())
    report = {
        "provider": SHARED_BASIS_PROVIDER_NAME,
        "rank": int(rank),
        "axis": str(axis),
        "factor_dtype": str(factor_dtype),
        "basis_estimator": (
            "whitened_affinity_weighted"
            if whitening and weights
            else "whitened_population"
            if whitening
            else "affinity_weighted"
            if weights
            else "reference_expert"
        ),
        "grouped_modules": grouped_count,
        "source_expert_bytes": source_total,
        "factor_bytes": factor_total,
        "byte_ratio": factor_total / max(1, source_total),
        "modules": report_modules,
    }
    if whitening:
        report["whitening_regularization"] = float(whitening_regularization)
    if not math.isfinite(float(report["byte_ratio"])):
        raise RuntimeError("Shared-basis artifact byte ratio is non-finite.")
    return output_tensors, output_manifest, report


__all__ = ["factorize_packed_state_shared_basis"]
