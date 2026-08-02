"""Ragged INT8 packed artifacts for FlexMoE prefix-pruned experts.

The offline transform consumes independently lineage-bound channel ranking and
action-plan evidence.  It applies the same expert-local permutation to the gate,
up, and down projections, physically removes every inactive prefix tail, and
stores variable expert widths in a rowwise symmetric-INT8 payload.

Source: Mo et al., "FlexMoE: One-for-All Nested Intra-Expert Pruning for MoE
Language Models", Equations 3-4 and 11-12, arXiv:2606.27866.
https://arxiv.org/abs/2606.27866
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from mirai.core.models.compressed_weights.artifact_source import (
    load_grouped_expert_source,
)
from mirai.core.moe.calibration.flexmoe import (
    FlexMoEActionPlan,
    FlexMoERankingEvidence,
    global_prune_budget,
    retained_width,
)
from mirai.core.moe.storage.physical_weights import (
    PhysicalWeightProviderContext,
    register_physical_weight_provider,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


FLEXMOE_NESTED_PROVIDER_NAME = "flexmoe_nested"
FLEXMOE_NESTED_PROVIDER_SCHEMA_VERSION = 1


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("FlexMoE nested artifacts require torch.")


def _tensor_bytes(value: Any) -> int:
    tensor = torch.as_tensor(value)
    return int(tensor.numel()) * int(tensor.element_size())


def _rowwise_int8(value: Any) -> tuple[Any, Any]:
    dense = torch.as_tensor(value).detach().float()
    if dense.ndim != 2 or min(int(item) for item in dense.shape) < 1:
        raise ValueError("FlexMoE projection slices must be non-empty matrices.")
    if not bool(torch.isfinite(dense).all().item()):
        raise ValueError("FlexMoE projection slice contains non-finite values.")
    scale = dense.abs().amax(dim=1).div(127.0)
    scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    codes = torch.round(dense / scale.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
    return codes.contiguous(), scale.contiguous()


class FlexMoENestedPhysicalWeightProvider:
    """Decode physically ragged expert matrices one expert at a time."""

    name = FLEXMOE_NESTED_PROVIDER_NAME
    ragged_intermediate_widths = True

    def __init__(self, context: PhysicalWeightProviderContext) -> None:
        _require_torch()
        spec = context.spec
        if str(spec.get("name", "")) != self.name:
            raise ValueError("FlexMoE nested provider name mismatch.")
        if int(spec.get("schema_version", 0)) != FLEXMOE_NESTED_PROVIDER_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported FlexMoE nested provider schema {spec.get('schema_version')!r}."
            )
        self.num_experts = int(context.num_experts)
        self._spec = copy.deepcopy(dict(spec))
        self._tensors = context.tensors
        self._shapes = {
            str(key): tuple(int(value) for value in shape) for key, shape in context.shapes.items()
        }
        if set(self._shapes) != {"w1", "w2", "w3"}:
            raise ValueError("FlexMoE nested provider requires w1, w2, and w3 shapes.")
        widths_name = str(spec.get("retained_widths", ""))
        if widths_name not in context.tensors:
            raise ValueError("FlexMoE nested provider is missing retained widths.")
        self._widths = torch.as_tensor(context.tensors[widths_name]).detach().cpu()
        if self._widths.dtype != torch.int64 or tuple(self._widths.shape) != (self.num_experts,):
            raise ValueError("FlexMoE retained widths must be an int64 vector over experts.")
        intermediate = int(self._shapes["w1"][1])
        if bool((self._widths < 1).any().item()) or bool(
            (self._widths > intermediate).any().item()
        ):
            raise ValueError("FlexMoE retained widths exceed the source topology.")
        projections = spec.get("projections")
        if not isinstance(projections, Mapping) or set(projections) != {
            "w1",
            "w2",
            "w3",
        }:
            raise ValueError("FlexMoE nested provider has invalid projection specs.")
        self._projections: dict[str, dict[str, str]] = {}
        for key, raw_projection in projections.items():
            if not isinstance(raw_projection, Mapping):
                raise ValueError(f"FlexMoE projection {key!r} must be an object.")
            names = {
                field: str(raw_projection.get(field, ""))
                for field in ("codes", "scales", "code_offsets", "scale_offsets")
            }
            if any(name not in context.tensors for name in names.values()):
                raise ValueError(f"FlexMoE projection {key!r} tensors are incomplete.")
            code_offsets = torch.as_tensor(context.tensors[names["code_offsets"]])
            scale_offsets = torch.as_tensor(context.tensors[names["scale_offsets"]])
            if code_offsets.dtype != torch.int64 or scale_offsets.dtype != torch.int64:
                raise ValueError("FlexMoE ragged offsets must use int64.")
            if tuple(code_offsets.shape) != (self.num_experts + 1,) or tuple(
                scale_offsets.shape
            ) != (self.num_experts + 1,):
                raise ValueError("FlexMoE ragged offsets must cover every expert.")
            if int(code_offsets[0].item()) != 0 or int(scale_offsets[0].item()) != 0:
                raise ValueError("FlexMoE ragged offsets must begin at zero.")
            if bool((code_offsets[1:] < code_offsets[:-1]).any().item()) or bool(
                (scale_offsets[1:] < scale_offsets[:-1]).any().item()
            ):
                raise ValueError("FlexMoE ragged offsets must be monotonic.")
            code_shape, code_dtype = self._tensor_shape_dtype(names["codes"])
            scale_shape, scale_dtype = self._tensor_shape_dtype(names["scales"])
            if code_dtype not in {"torch.int8", "I8"} or scale_dtype not in {
                "torch.float16",
                "torch.bfloat16",
                "torch.float32",
                "torch.float64",
                "F16",
                "BF16",
                "F32",
                "F64",
            }:
                raise ValueError("FlexMoE ragged codes/scales have invalid dtypes.")
            if int(code_offsets[-1].item()) != math.prod(code_shape) or int(
                scale_offsets[-1].item()
            ) != math.prod(scale_shape):
                raise ValueError("FlexMoE ragged offsets do not span their tensors.")
            self._projections[str(key)] = names
        for expert in range(self.num_experts):
            width = self.retained_intermediate_width(expert)
            hidden = int(self._shapes["w1"][2])
            for key in ("w1", "w3"):
                self._validate_expert_span(key, expert, rows=width, columns=hidden)
            self._validate_expert_span("w2", expert, rows=hidden, columns=width)

    def _tensor_shape_dtype(self, name: str) -> tuple[tuple[int, ...], str]:
        metadata = getattr(self._tensors, "tensor_shape_dtype", None)
        if callable(metadata):
            return metadata(str(name))
        value = torch.as_tensor(self._tensors[str(name)])
        return tuple(int(item) for item in value.shape), str(value.dtype)

    def _range(self, name: str, start: int, end: int) -> Any:
        get_range = getattr(self._tensors, "get_range", None)
        if callable(get_range):
            return get_range(str(name), int(start), int(end))
        return self._tensors[str(name)][int(start) : int(end)]

    def _validate_expert_span(
        self,
        key: str,
        expert: int,
        *,
        rows: int,
        columns: int,
    ) -> None:
        names = self._projections[key]
        code_offsets = torch.as_tensor(self._tensors[names["code_offsets"]])
        scale_offsets = torch.as_tensor(self._tensors[names["scale_offsets"]])
        code_count = int(code_offsets[expert + 1] - code_offsets[expert])
        scale_count = int(scale_offsets[expert + 1] - scale_offsets[expert])
        if code_count != int(rows) * int(columns) or scale_count != int(rows):
            raise ValueError(f"FlexMoE {key} payload shape is invalid for expert {expert}.")

    def retained_intermediate_width(self, expert_index: int) -> int:
        expert = int(expert_index)
        if expert < 0 or expert >= self.num_experts:
            raise IndexError(f"FlexMoE expert index {expert} is out of range.")
        return int(self._widths[expert].item())

    def expert_weight_shape(self, key: str) -> tuple[int, ...]:
        try:
            return self._shapes[str(key)]
        except KeyError as exc:
            raise KeyError(f"Unknown FlexMoE projection {key!r}.") from exc

    def materialize_expert(
        self,
        key: str,
        expert_index: int,
        *,
        dtype: Any,
        device: Any,
    ) -> Any:
        projection = str(key)
        if projection not in self._projections:
            raise KeyError(f"Unknown FlexMoE projection {key!r}.")
        expert = int(expert_index)
        width = self.retained_intermediate_width(expert)
        hidden = int(self._shapes["w1"][2])
        rows, columns = (hidden, width) if projection == "w2" else (width, hidden)
        names = self._projections[projection]
        code_offsets = torch.as_tensor(self._tensors[names["code_offsets"]])
        scale_offsets = torch.as_tensor(self._tensors[names["scale_offsets"]])
        code_start = int(code_offsets[expert].item())
        code_end = int(code_offsets[expert + 1].item())
        scale_start = int(scale_offsets[expert].item())
        scale_end = int(scale_offsets[expert + 1].item())
        codes = torch.as_tensor(self._range(names["codes"], code_start, code_end))
        scales = torch.as_tensor(self._range(names["scales"], scale_start, scale_end))
        return (
            codes.to(device=device, dtype=torch.float32).reshape(rows, columns)
            * scales.to(device=device, dtype=torch.float32).reshape(rows, 1)
        ).to(dtype=dtype)

    def packed_tensor_names(self) -> frozenset[str]:
        names = {str(self._spec["retained_widths"])}
        for projection in self._projections.values():
            names.update(projection.values())
        return frozenset(names)

    def manifest_spec(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._spec)

    def packed_tensors(self) -> Mapping[str, Any]:
        return {name: self._tensors[name] for name in self.packed_tensor_names()}


@register_physical_weight_provider(FLEXMOE_NESTED_PROVIDER_NAME)
def _build_flexmoe_nested_provider(
    context: PhysicalWeightProviderContext,
) -> FlexMoENestedPhysicalWeightProvider:
    return FlexMoENestedPhysicalWeightProvider(context)


def transform_packed_state_flexmoe_nested(
    tensors: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    ranking_by_module: Mapping[str, FlexMoERankingEvidence],
    actions_by_module: Mapping[str, FlexMoEActionPlan],
    ranking_lineage: Mapping[str, str],
    action_lineage: Mapping[str, str],
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Physically rank and prefix-prune every grouped expert module."""

    _require_torch()
    if manifest.get("format") != "mirai.compressed_weights.packed_state":
        raise ValueError(f"Unsupported packed-state format {manifest.get('format')!r}.")
    if int(manifest.get("schema_version", 0)) not in {1, 2, 3}:
        raise ValueError("FlexMoE nested transform requires a schema-v1/v2/v3 source artifact.")
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("Packed-state manifest must contain modules.")
    grouped_names = {
        str(name)
        for name, spec in modules.items()
        if isinstance(spec, Mapping) and str(spec.get("kind")) == "grouped_experts"
    }
    if not grouped_names:
        raise ValueError("FlexMoE nested transform requires grouped expert modules.")
    if set(ranking_by_module) != grouped_names or set(actions_by_module) != grouped_names:
        raise ValueError("FlexMoE ranking and action artifacts must exactly cover grouped modules.")
    required_ranking_lineage = {
        "dataset_snapshot_id",
        "model_snapshot_id",
        "config_snapshot_id",
    }
    required_action_lineage = {*required_ranking_lineage, "ranking_snapshot_id"}
    if any(not str(ranking_lineage.get(key, "")).strip() for key in required_ranking_lineage):
        raise ValueError("FlexMoE ranking lineage is incomplete.")
    if any(not str(action_lineage.get(key, "")).strip() for key in required_action_lineage):
        raise ValueError("FlexMoE action lineage is incomplete.")
    if any(
        str(ranking_lineage[key]) != str(action_lineage[key])
        for key in required_ranking_lineage
    ):
        raise ValueError("FlexMoE ranking and action lineage disagree.")

    output_tensors = {str(name): value for name, value in tensors.items()}
    output_manifest = copy.deepcopy(dict(manifest))
    output_manifest["schema_version"] = 4
    report_modules: dict[str, Any] = {}
    source_tensor_names: set[str] = set()
    packed_total = 0
    for raw_name, raw_spec in output_manifest["modules"].items():
        module_name = str(raw_name)
        if not isinstance(raw_spec, dict) or str(raw_spec.get("kind")) != "grouped_experts":
            continue
        if "physical_weight_provider" in raw_spec:
            raise ValueError(f"Module {module_name!r} already has a physical provider.")
        ranking = ranking_by_module[module_name].validate()
        plan = actions_by_module[module_name].validate()
        shapes = raw_spec.get("shapes")
        tensor_map = raw_spec.get("tensors")
        if not isinstance(shapes, Mapping) or not isinstance(tensor_map, Mapping):
            raise ValueError(f"FlexMoE source module {module_name!r} is incomplete.")
        experts = int(raw_spec.get("num_experts", 0))
        source_shape = tuple(int(value) for value in shapes["w1"])
        if (
            source_shape[0] != experts
            or ranking.num_experts != experts
            or plan.num_experts != experts
        ):
            raise ValueError(f"FlexMoE expert topology mismatch for {module_name!r}.")
        intermediate = int(source_shape[1])
        hidden = int(source_shape[2])
        if (
            ranking.intermediate_size != intermediate
            or tuple(int(value) for value in shapes["w3"]) != source_shape
            or tuple(int(value) for value in shapes["w2"])
            != (
                experts,
                hidden,
                intermediate,
            )
        ):
            raise ValueError(f"FlexMoE projection topology mismatch for {module_name!r}.")
        source_names = {str(value) for value in tensor_map.values()}
        if not source_names.issubset(output_tensors):
            raise ValueError(f"FlexMoE source tensors are missing for {module_name!r}.")
        source_bytes = sum(_tensor_bytes(output_tensors[name]) for name in source_names)
        source_tensor_names.update(source_names)
        source = load_grouped_expert_source(raw_spec, output_tensors)
        permutation = ranking.permutation().to(device=device)
        ratios = plan.retention_ratios().to(device="cpu")
        widths = torch.tensor(
            [retained_width(intermediate, float(value)) for value in ratios],
            dtype=torch.int64,
        )
        if bool((widths == intermediate).all().item()):
            raise ValueError(f"FlexMoE action plan for {module_name!r} prunes no channels.")
        prefix = f"{module_name}.flexmoe_nested"
        widths_name = f"{prefix}.retained_widths"
        output_tensors[widths_name] = widths
        projection_specs: dict[str, Any] = {}
        module_packed_names = {widths_name}
        for key in ("w1", "w2", "w3"):
            codes_parts: list[Any] = []
            scales_parts: list[Any] = []
            code_offsets = [0]
            scale_offsets = [0]
            for expert in range(experts):
                dense = source._dequantize_expert(
                    key,
                    expert,
                    dtype=torch.float32,
                    device=torch.device(device),
                ).unsqueeze(0)
                order = permutation[expert]
                ranked = (
                    dense[0].index_select(1, order)
                    if key == "w2"
                    else dense[0].index_select(0, order)
                )
                width = int(widths[expert].item())
                clipped = ranked[:, :width] if key == "w2" else ranked[:width, :]
                codes, scales = _rowwise_int8(clipped)
                codes_parts.append(codes.reshape(-1).cpu())
                scales_parts.append(scales.cpu())
                code_offsets.append(code_offsets[-1] + int(codes.numel()))
                scale_offsets.append(scale_offsets[-1] + int(scales.numel()))
            names = {
                "codes": f"{prefix}.{key}.codes",
                "scales": f"{prefix}.{key}.scales",
                "code_offsets": f"{prefix}.{key}.code_offsets",
                "scale_offsets": f"{prefix}.{key}.scale_offsets",
            }
            output_tensors[names["codes"]] = torch.cat(codes_parts).contiguous()
            output_tensors[names["scales"]] = torch.cat(scales_parts).contiguous()
            output_tensors[names["code_offsets"]] = torch.tensor(
                code_offsets,
                dtype=torch.int64,
            )
            output_tensors[names["scale_offsets"]] = torch.tensor(
                scale_offsets,
                dtype=torch.int64,
            )
            projection_specs[key] = names
            module_packed_names.update(names.values())
        packed_bytes = sum(_tensor_bytes(output_tensors[name]) for name in module_packed_names)
        packed_total += packed_bytes
        if packed_bytes >= source_bytes:
            raise ValueError(f"FlexMoE output for {module_name!r} is not smaller than its source.")
        raw_spec["physical_weight_provider"] = {
            "name": FLEXMOE_NESTED_PROVIDER_NAME,
            "schema_version": FLEXMOE_NESTED_PROVIDER_SCHEMA_VERSION,
            "retained_widths": widths_name,
            "projections": projection_specs,
        }
        raw_spec["tensors"] = {}
        raw_spec["quant_format"] = "int8"
        report_modules[module_name] = {
            "source_bytes": source_bytes,
            "packed_bytes": packed_bytes,
            "byte_ratio": packed_bytes / source_bytes,
            "prune_budget": plan.prune_budget(),
            "minimum_width": int(widths.min().item()),
            "maximum_width": int(widths.max().item()),
        }

    referenced_tensor_names: set[str] = set()

    def collect_references(value: Any) -> None:
        if isinstance(value, Mapping):
            for child in value.values():
                collect_references(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect_references(child)
        elif isinstance(value, str) and value in output_tensors:
            referenced_tensor_names.add(value)

    collect_references(output_manifest)
    for name in source_tensor_names - referenced_tensor_names:
        output_tensors.pop(name, None)
    source_total = sum(_tensor_bytes(tensors[name]) for name in source_tensor_names)
    prune_budget = float(
        global_prune_budget(
            torch.cat(
                [
                    actions_by_module[name].retention_ratios()
                    for name in sorted(grouped_names)
                ],
                dim=0,
            )
        ).item()
    )
    output_manifest["flexmoe_nested_transform"] = {
        "format": "mirai.moe.flexmoe_nested_transform",
        "schema_version": 1,
        "ranking_lineage": dict(ranking_lineage),
        "action_lineage": dict(action_lineage),
        "global_prune_budget": prune_budget,
        "modules": report_modules,
    }
    return (
        output_tensors,
        output_manifest,
        {
            "modules": report_modules,
            "source_expert_bytes": source_total,
            "packed_expert_bytes": packed_total,
            "byte_ratio": packed_total / source_total,
            "global_prune_budget": prune_budget,
        },
    )


__all__ = [
    "FLEXMOE_NESTED_PROVIDER_NAME",
    "FLEXMOE_NESTED_PROVIDER_SCHEMA_VERSION",
    "FlexMoENestedPhysicalWeightProvider",
    "transform_packed_state_flexmoe_nested",
]
