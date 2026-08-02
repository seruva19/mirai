"""Offline STUN-inspired expert pruning followed by compact 2:4 storage."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from mirai.core.models.compressed_weights.artifact_source import (
    load_grouped_expert_source,
)
from mirai.core.models.compressed_weights.quantization.structured_sparse_provider import (
    SPARSE24_PROVIDER_NAME,
    SPARSE24_PROVIDER_SCHEMA_VERSION,
    PackedSparse24,
    pack_sparse24,
)
from mirai.core.moe.calibration.stun import (
    StunExpertPlan,
    cluster_router_experts,
    select_stun_representatives_streaming,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _parent(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else ""


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _resolve_router_tensor(
    module_name: str,
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> str:
    residual = manifest.get("residual_tensors")
    shapes = spec.get("shapes")
    if not isinstance(residual, Mapping) or not isinstance(shapes, Mapping):
        raise ValueError("STUN source requires residual_tensors and shapes maps.")
    experts = int(spec.get("num_experts", 0))
    hidden = int(tuple(shapes["w1"])[-1])
    prefix = _parent(module_name)
    candidates = []
    for tensor_name in residual.values():
        name = str(tensor_name)
        tensor = tensors.get(name)
        if tensor is None:
            continue
        if prefix and not name.startswith(prefix + "."):
            continue
        if (
            tensor.ndim == 2
            and tuple(int(value) for value in tensor.shape) == (experts, hidden)
        ):
            candidates.append(name)
    if len(candidates) != 1:
        raise ValueError(
            f"STUN expected exactly one [{experts}, {hidden}] router tensor beside "
            f"{module_name!r}, found {sorted(candidates)!r}."
        )
    return candidates[0]


def _materialize_cluster_weight(
    source: Any,
    key: str,
    plan: StunExpertPlan,
    cluster_index: int,
    *,
    device: Any,
) -> Any:
    cluster = plan.clusters[int(cluster_index)]
    if not cluster.reconstruct:
        return source._dequantize_expert(
            key,
            cluster.representative,
            dtype=torch.float32,
            device=torch.device(device),
        )
    total = None
    for expert_index in cluster.members:
        value = source._dequantize_expert(
            key,
            expert_index,
            dtype=torch.float32,
            device=torch.device(device),
        )
        total = value.clone() if total is None else total.add(value)
    if total is None:  # pragma: no cover
        raise RuntimeError("STUN cluster unexpectedly contains no experts.")
    return total.div_(len(cluster.members))


def _apply_plan_to_residual(value: Any, plan: StunExpertPlan) -> Any:
    outputs = []
    for cluster in plan.clusters:
        if cluster.reconstruct and (value.is_floating_point() or value.is_complex()):
            index = torch.as_tensor(
                cluster.members,
                device=value.device,
                dtype=torch.long,
            )
            outputs.append(value.index_select(0, index).mean(dim=0))
        else:
            outputs.append(value[int(cluster.representative)])
    return torch.stack(outputs, dim=0).contiguous()


def _stack_packed(states: list[PackedSparse24]) -> PackedSparse24:
    if not states:
        raise ValueError("Cannot stack an empty compact 2:4 expert set.")
    shape = states[0].original_shape
    group_size = states[0].quant_group_size
    if any(
        state.original_shape != shape or state.quant_group_size != group_size
        for state in states
    ):
        raise ValueError("Compact 2:4 expert projections have inconsistent layouts.")
    return PackedSparse24(
        values=torch.stack([state.values for state in states], dim=0).contiguous(),
        scales=torch.stack([state.scales for state in states], dim=0).contiguous(),
        positions=torch.stack([state.positions for state in states], dim=0).contiguous(),
        original_shape=(len(states), *shape),
        quant_group_size=group_size,
    )


def transform_packed_state_stun_sparse24(
    tensors: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    target_experts: int | Mapping[str, int],
    device: Any,
    reconstruct_below: int = 3,
    quant_group_size: int = 32,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create a schema-v4 structured-then-semi-structured expert artifact.

    The first stage follows STUN's router-similarity clustering and
    centroid-nearest representative rule.  STUN's published second stage is
    unstructured Wanda/OWL.  Mirai deliberately substitutes executable 2:4
    sparsity, so the manifest and public claim identify this as an adaptation.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("STUN artifact transformation requires torch.")
    if manifest.get("format") != "mirai.compressed_weights.packed_state":
        raise ValueError(f"Unsupported packed-state format {manifest.get('format')!r}.")
    if int(manifest.get("schema_version", 0)) not in {1, 2, 3}:
        raise ValueError("STUN transformation requires a schema-v1/v2/v3 source artifact.")
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("Packed-state manifest must contain modules.")
    targets = (
        {str(name): int(value) for name, value in target_experts.items()}
        if isinstance(target_experts, Mapping)
        else None
    )
    grouped_names = {
        str(name)
        for name, spec in modules.items()
        if isinstance(spec, Mapping) and str(spec.get("kind")) == "grouped_experts"
    }
    if targets is not None and set(targets) != grouped_names:
        raise ValueError("Per-module STUN targets must exactly cover grouped modules.")

    output_tensors = {str(key): value for key, value in tensors.items()}
    output_manifest = copy.deepcopy(dict(manifest))
    output_manifest["schema_version"] = 4
    report_modules: dict[str, Any] = {}
    transformed = 0
    residual_claims: dict[str, StunExpertPlan] = {}
    for raw_name, raw_spec in output_manifest["modules"].items():
        module_name = str(raw_name)
        if not isinstance(raw_spec, dict) or str(raw_spec.get("kind")) != "grouped_experts":
            continue
        if "physical_weight_provider" in raw_spec:
            raise ValueError(f"Module {module_name!r} already has a physical provider.")
        source_experts = int(raw_spec.get("num_experts", 0))
        target = (
            int(targets[module_name])
            if targets is not None
            else int(target_experts)
        )
        if target < 1 or target >= source_experts:
            raise ValueError(
                f"STUN target for {module_name!r} must be in [1, "
                f"{source_experts - 1}], got {target}."
            )
        source_tensor_map = raw_spec.get("tensors")
        if not isinstance(source_tensor_map, Mapping):
            raise ValueError(f"Grouped module {module_name!r} has no tensor map.")
        source_keys = {str(value) for value in source_tensor_map.values()}
        missing = sorted(source_keys - set(output_tensors))
        if missing:
            raise KeyError(f"Grouped module {module_name!r} is missing tensors: {missing}.")
        router_name = _resolve_router_tensor(
            module_name,
            raw_spec,
            output_manifest,
            output_tensors,
        )
        router_weight = output_tensors[router_name]
        clusters = cluster_router_experts(
            router_weight,
            target_experts=target,
        )
        source = load_grouped_expert_source(raw_spec, output_tensors)

        def load_weight(
            key: str,
            expert_index: int,
            source_module: Any = source,
        ) -> Any:
            return source_module._dequantize_expert(
                key,
                expert_index,
                dtype=torch.float32,
                device=torch.device(device),
            )

        plan = select_stun_representatives_streaming(
            clusters,
            source_experts=source_experts,
            load_weight=load_weight,
            reconstruct_below=int(reconstruct_below),
        )
        provider_tensor_map: dict[str, str] = {}
        projection_specs: dict[str, Any] = {}
        source_bytes = sum(_tensor_bytes(output_tensors[key]) for key in source_keys)
        sparse_error_energy = 0.0
        source_energy = 0.0
        for key in ("w1", "w2", "w3"):
            packed_experts: list[PackedSparse24] = []
            for cluster_index in range(plan.output_experts):
                dense = _materialize_cluster_weight(
                    source,
                    key,
                    plan,
                    cluster_index,
                    device=device,
                )
                packed = pack_sparse24(
                    dense,
                    quant_group_size=int(quant_group_size),
                )
                reconstructed = packed.dense(
                    dtype=torch.float32,
                    device=dense.device,
                )
                sparse_error_energy += float(
                    (dense.float() - reconstructed).square().sum().item()
                )
                source_energy += float(dense.float().square().sum().item())
                packed_experts.append(packed)
            stacked = _stack_packed(packed_experts)
            values_name = f"{module_name}.{key}_sparse24_values"
            scales_name = f"{module_name}.{key}_sparse24_scales"
            positions_name = f"{module_name}.{key}_sparse24_positions"
            output_tensors[values_name] = stacked.values
            output_tensors[scales_name] = stacked.scales
            output_tensors[positions_name] = stacked.positions
            provider_tensor_map[f"{key}_values"] = values_name
            provider_tensor_map[f"{key}_scales"] = scales_name
            provider_tensor_map[f"{key}_positions"] = positions_name
            projection_specs[key] = {
                "values": values_name,
                "scales": scales_name,
                "positions": positions_name,
                "quant_group_size": int(quant_group_size),
            }
        for tensor_key in source_keys:
            del output_tensors[tensor_key]
        packed_keys = set(provider_tensor_map.values())
        packed_bytes = sum(_tensor_bytes(output_tensors[key]) for key in packed_keys)
        if packed_bytes >= source_bytes:
            raise ValueError(
                f"Compact 2:4 payload for {module_name!r} uses {packed_bytes} bytes, "
                f"not less than the {source_bytes}-byte source. Choose a larger "
                "quantization group or keep the source artifact."
            )
        raw_spec["num_experts"] = plan.output_experts
        shapes = raw_spec.get("shapes")
        if not isinstance(shapes, dict):
            raise ValueError(f"Grouped module {module_name!r} has no shapes map.")
        for key, shape in list(shapes.items()):
            patched = list(shape)
            patched[0] = plan.output_experts
            shapes[key] = patched
        raw_spec["tensors"] = provider_tensor_map
        raw_spec["physical_weight_provider"] = {
            "name": SPARSE24_PROVIDER_NAME,
            "schema_version": SPARSE24_PROVIDER_SCHEMA_VERSION,
            "structured_stage": "stun_router_similarity",
            "second_stage": "semi_structured_2_4_adaptation",
            "execution": "on_demand_dense_decode",
            "projections": projection_specs,
        }
        prefix = _parent(module_name)
        for tensor_name in (output_manifest.get("residual_tensors") or {}).values():
            name = str(tensor_name)
            value = output_tensors.get(name)
            if value is None or value.ndim < 1 or int(value.shape[0]) != source_experts:
                continue
            if prefix and not name.startswith(prefix + "."):
                continue
            previous = residual_claims.get(name)
            if previous is not None and previous != plan:
                raise ValueError(f"Residual tensor {name!r} has conflicting STUN plans.")
            output_tensors[name] = _apply_plan_to_residual(value, plan)
            residual_claims[name] = plan
        report_modules[module_name] = {
            "source_experts": source_experts,
            "output_experts": plan.output_experts,
            "clusters": [list(item.members) for item in plan.clusters],
            "representatives": [item.representative for item in plan.clusters],
            "reconstructed": bool(plan.clusters[0].reconstruct),
            "router_tensor": router_name,
            "source_expert_bytes": source_bytes,
            "packed_expert_bytes": packed_bytes,
            "byte_ratio": packed_bytes / max(1, source_bytes),
            "relative_projection_error": math.sqrt(
                sparse_error_energy / max(source_energy, torch.finfo(torch.float32).tiny)
            ),
        }
        transformed += 1
    if transformed == 0:
        raise ValueError("Packed artifact contains no grouped experts to transform.")
    output_manifest["physical_weight_providers"] = [SPARSE24_PROVIDER_NAME]
    output_manifest["structured_expert_compression"] = {
        "structured_stage": "stun_router_similarity",
        "second_stage": "semi_structured_2_4_adaptation",
        "source": "https://aclanthology.org/2025.acl-long.671/",
    }
    source_total = sum(
        int(item["source_expert_bytes"]) for item in report_modules.values()
    )
    packed_total = sum(
        int(item["packed_expert_bytes"]) for item in report_modules.values()
    )
    return output_tensors, output_manifest, {
        "provider": SPARSE24_PROVIDER_NAME,
        "structured_stage": "stun_router_similarity",
        "second_stage": "semi_structured_2_4_adaptation",
        "grouped_modules": transformed,
        "source_expert_bytes": source_total,
        "packed_expert_bytes": packed_total,
        "byte_ratio": packed_total / max(1, source_total),
        "modules": report_modules,
    }


__all__ = ["transform_packed_state_stun_sparse24"]
