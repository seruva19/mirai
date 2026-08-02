"""Expert prototype consolidation contracts and packed-artifact transforms.

Consolidation preserves the logical router expert space while storing a smaller
physical expert pool. It is distinct from structural pruning, which removes
router rows and renumbers the logical model. This module owns only pure planning,
route coalescing, and offline packed-state transformation.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

from mirai.core.models.compressed_weights.artifact_source import (
    merge_grouped_expert_source,
)
from mirai.core.moe.storage.aliases import PrototypeConsolidationPlan
from mirai.core.moe.storage.aliases import build_prototype_plan
from mirai.core.moe.storage.aliases import coalesce_alias_routes
from mirai.core.moe.storage.aliases import validate_logical_to_physical

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _packed_module_keys(modules: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    for spec in modules.values():
        if isinstance(spec, Mapping):
            tensor_names = spec.get("tensors") or {}
            if isinstance(tensor_names, Mapping):
                keys.update(str(value) for value in tensor_names.values())
    return keys


def consolidate_packed_state(
    tensors: Mapping[str, Any],
    manifest: Mapping[str, Any],
    logical_to_prototype_by_module: Mapping[
        str, Sequence[int] | Mapping[str, Any]
    ],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a schema-v3 packed artifact with fewer physical expert tensors.

    Router and other residual tensors are deliberately left untouched. The input
    tensors and manifest are never mutated.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Prototype consolidation requires torch.")
    if manifest.get("format") != "mirai.compressed_weights.packed_state":
        raise ValueError(f"Unsupported packed-state format {manifest.get('format')!r}.")
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("packed-state manifest must have a non-empty modules object.")
    if not logical_to_prototype_by_module:
        raise ValueError("At least one grouped-expert module must be consolidated.")

    new_manifest = copy.deepcopy(dict(manifest))
    new_manifest["schema_version"] = 3
    new_tensors = {str(key): value for key, value in tensors.items()}
    packed_keys = _packed_module_keys(modules)
    summary_delta = 0

    for raw_name, raw_entry in logical_to_prototype_by_module.items():
        module_name = str(raw_name)
        spec = new_manifest["modules"].get(module_name)
        if not isinstance(spec, dict):
            raise ValueError(f"packed-state has no module {module_name!r} to consolidate.")
        if str(spec.get("kind")) != "grouped_experts":
            raise ValueError(
                f"Prototype consolidation requires grouped_experts; {module_name!r} "
                f"is kind={spec.get('kind')!r}."
            )
        logical_count = int(spec.get("num_experts", 0))
        if isinstance(raw_entry, Mapping):
            raw_mapping = raw_entry.get("logical_to_prototype")
            raw_merge_weights = raw_entry.get("merge_weights")
        else:
            raw_mapping = raw_entry
            raw_merge_weights = None
        if not isinstance(raw_mapping, Sequence) or isinstance(
            raw_mapping, (str, bytes)
        ):
            raise ValueError(
                f"Consolidation plan entry {module_name!r} has no expert mapping."
            )
        plan = build_prototype_plan(
            raw_mapping,
            num_logical_experts=logical_count,
        )
        merge_weights: tuple[float, ...] | None = None
        if raw_merge_weights is not None:
            if not isinstance(raw_merge_weights, Sequence) or isinstance(
                raw_merge_weights, (str, bytes)
            ):
                raise ValueError(
                    f"Consolidation merge weights for {module_name!r} must be an array."
                )
            merge_weights = tuple(float(value) for value in raw_merge_weights)
            if len(merge_weights) != logical_count or any(
                not math.isfinite(value) or value < 0.0
                for value in merge_weights
            ):
                raise ValueError(
                    f"Consolidation merge weights for {module_name!r} are invalid."
                )
            for physical_id in range(plan.physical_experts):
                total = sum(
                    merge_weights[logical_id]
                    for logical_id, mapped_id in enumerate(plan.logical_to_physical)
                    if mapped_id == physical_id
                )
                if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        f"Merge weights for {module_name!r} must sum to one "
                        "inside every physical cluster."
                    )
        prototype_index = torch.tensor(plan.prototype_logical_ids, dtype=torch.long)
        tensor_names = spec.get("tensors")
        if not isinstance(tensor_names, Mapping) or not tensor_names:
            raise ValueError(f"Grouped-expert module {module_name!r} has no tensor map.")
        original_shapes = spec.get("shapes")
        if not isinstance(original_shapes, dict):
            raise ValueError(f"Grouped-expert module {module_name!r} has no shapes map.")
        if merge_weights is None:
            for local_name, raw_tensor_key in tensor_names.items():
                tensor_key = str(raw_tensor_key)
                if tensor_key not in new_tensors or tensor_key not in packed_keys:
                    raise KeyError(f"Packed expert tensor {tensor_key!r} is missing.")
                tensor = new_tensors[tensor_key]
                if str(local_name).endswith(("_code", "_ncode")):
                    continue
                if int(tensor.shape[0]) != logical_count:
                    raise ValueError(
                        f"Packed expert tensor {tensor_key!r} axis 0 has size "
                        f"{int(tensor.shape[0])}, expected {logical_count}."
                    )
                new_tensors[tensor_key] = tensor.index_select(
                    0, prototype_index.to(tensor.device)
                ).contiguous()
            spec["consolidation_method"] = "prototype"
        else:
            merged_tensors, merged_metadata = merge_grouped_expert_source(
                spec,
                new_tensors,
                logical_to_physical=plan.logical_to_physical,
                merge_weights=merge_weights,
                physical_experts=plan.physical_experts,
            )
            new_tensors.update(merged_tensors)
            for key in (
                "quant_format",
                "num_experts",
                "group_sizes",
                "shapes",
                "nf4_meta",
                "gguf_meta",
                "microscaling_meta",
            ):
                if key in merged_metadata:
                    spec[key] = merged_metadata[key]
                elif key in {"nf4_meta", "gguf_meta", "microscaling_meta"}:
                    spec.pop(key, None)
            spec["consolidation_method"] = "hierarchical_output"

        for key, raw_shape in list(original_shapes.items()):
            shape = [int(value) for value in raw_shape]
            if not shape or shape[0] != logical_count:
                raise ValueError(
                    "Invalid grouped-expert shape for "
                    f"{module_name!r}.{key}: {shape}."
                )
            old_numel = math.prod(shape)
            shape[0] = plan.physical_experts
            summary_delta += math.prod(shape) - old_numel
            if merge_weights is None:
                original_shapes[key] = shape
        spec["num_experts"] = plan.physical_experts
        spec["logical_num_experts"] = plan.logical_experts
        spec["logical_to_physical"] = list(plan.logical_to_physical)
        spec["prototype_logical_ids"] = list(plan.prototype_logical_ids)

    summary = new_manifest.get("summary")
    if isinstance(summary, dict) and "quantized_numel" in summary:
        summary["quantized_numel"] = int(summary["quantized_numel"]) + summary_delta
    return new_tensors, new_manifest


__all__ = [
    "PrototypeConsolidationPlan",
    "build_prototype_plan",
    "coalesce_alias_routes",
    "consolidate_packed_state",
    "validate_logical_to_physical",
]
