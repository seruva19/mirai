"""Materialize grouped expert weights for offline packed-artifact transforms."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.compressed_weights.factorization.prototype_projection import (
    CompressedExpertProjectionSource,
)
from mirai.core.models.compressed_weights.execution.experts import (
    CompressedGroupedExperts,
)
from mirai.core.models.compressed_weights.packed.packed_state import (
    _gguf_meta_from_spec,
    _gguf_meta_to_spec,
    _blockwise_fp8_meta_from_spec,
    _blockwise_fp8_meta_to_spec,
    _microscaling_meta_from_spec,
    _microscaling_meta_to_spec,
    _nf4_meta_from_spec,
    _nf4_meta_to_spec,
)
from mirai.core.models.compressed_weights.quantization.gguf_quant import GGUF_FORMATS
from mirai.core.models.compressed_weights.quantization.microscaling_quant import (
    MICROSCALING_FORMATS,
)
from mirai.core.models.compressed_weights.quantization.blockwise_fp8 import (
    BLOCKWISE_FP8_FORMATS,
)
from mirai.core.models.compressed_weights.quantization.quant import (
    NF4_BLOCKSIZE,
    normalize_quant_format,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def load_grouped_expert_source(
    spec: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> CompressedGroupedExperts:
    """Build an offline decoder for one schema-v1/v2/v3 grouped-expert module."""

    quant_format = normalize_quant_format(str(spec.get("quant_format", "int8")))
    num_experts = int(spec.get("num_experts", 0))
    tensor_names = spec.get("tensors")
    shapes = spec.get("shapes")
    group_sizes = spec.get("group_sizes")
    if not isinstance(tensor_names, Mapping) or not isinstance(shapes, Mapping):
        raise ValueError("Grouped expert source requires tensors and shapes maps.")
    if not isinstance(group_sizes, Mapping):
        raise ValueError("Grouped expert source requires a group_sizes map.")
    module = CompressedGroupedExperts.from_empty(
        num_experts=num_experts,
        quant_format=quant_format,
        expert_weight_access="active_dequant",
    )
    if quant_format == "nf4":
        meta = _nf4_meta_from_spec(spec)
        for key in ("w1", "w2", "w3"):
            module.load_nf4_packed_weight(
                key,
                packed=tensors[str(tensor_names[f"{key}_nf4"])],
                absmax=tensors[str(tensor_names[f"{key}_nf4_absmax"])],
                nested_absmax=tensors[str(tensor_names[f"{key}_nf4_nabsmax"])],
                offset=tensors[str(tensor_names[f"{key}_nf4_offset"])],
                code=tensors[str(tensor_names[f"{key}_nf4_code"])],
                nested_code=tensors[str(tensor_names[f"{key}_nf4_ncode"])],
                shape=shapes[key],
                meta=meta,
            )
        return module
    if quant_format in GGUF_FORMATS:
        meta = _gguf_meta_from_spec(spec)
        for key in ("w1", "w2", "w3"):
            module.load_gguf_packed_weight(
                key,
                blocks=tensors[str(tensor_names[f"{key}_gguf"])],
                shape=shapes[key],
                meta=meta,
            )
        return module
    if quant_format in MICROSCALING_FORMATS:
        raw_meta = spec.get("microscaling_meta")
        if not isinstance(raw_meta, Mapping):
            raise ValueError("Grouped expert source requires microscaling metadata.")
        for key in ("w1", "w2", "w3"):
            module.load_microscaling_packed_weight(
                key,
                packed=tensors[str(tensor_names[f"{key}_mx"])],
                scales=tensors[str(tensor_names[f"{key}_mx_scale"])],
                global_scales=tensors[str(tensor_names[f"{key}_mx_global"])],
                shape=shapes[key],
                meta=_microscaling_meta_from_spec(
                    {"microscaling_meta": raw_meta.get(key)}
                ),
            )
        return module
    if quant_format in BLOCKWISE_FP8_FORMATS:
        raw_meta = spec.get("blockwise_fp8_meta")
        if not isinstance(raw_meta, Mapping):
            raise ValueError("Grouped expert source requires blockwise FP8 metadata.")
        for key in ("w1", "w2", "w3"):
            module.load_blockwise_fp8_packed_weight(
                key,
                codes=tensors[str(tensor_names[f"{key}_fp8"])],
                scales=tensors[str(tensor_names[f"{key}_fp8_scale"])],
                shape=shapes[key],
                meta=_blockwise_fp8_meta_from_spec(
                    {"blockwise_fp8_meta": raw_meta.get(key)}
                ),
            )
        return module
    rotations = spec.get("rotations")
    if rotations is not None and not isinstance(rotations, Mapping):
        raise ValueError("Grouped expert source rotations must be a map.")
    for key in ("w1", "w2", "w3"):
        rotation = None
        if isinstance(rotations, Mapping):
            rotation_name = str(rotations.get(key, ""))
            if not rotation_name:
                raise ValueError(
                    "Grouped expert source requires complete w1/w2/w3 rotations."
                )
            rotation = tensors[rotation_name]
        module.load_quantized_weight(
            key,
            weight_int8=tensors[str(tensor_names[f"{key}_int8"])],
            weight_scale=tensors[str(tensor_names[f"{key}_scale"])],
            group_size=int(group_sizes[key]),
            rotation=rotation,
        )
    return module


def merge_grouped_expert_source(
    spec: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    logical_to_physical: tuple[int, ...],
    merge_weights: tuple[float, ...],
    physical_experts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge dense experts one at a time, then re-encode at the source boundary."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Grouped expert merging requires torch.")
    if "logical_to_physical" in spec or int(
        spec.get("logical_num_experts", spec.get("num_experts", 0))
    ) != int(spec.get("num_experts", 0)):
        raise ValueError("Hierarchical merging requires an unconsolidated source.")
    source = load_grouped_expert_source(spec, tensors)
    if source.has_logical_expert_aliases():
        raise ValueError("Hierarchical merging requires an unconsolidated source.")
    if source._physical_weight_provider is not None:
        raise ValueError(
            "Hierarchical merging does not accept provider-backed expert weights."
        )
    logical_count = int(source.num_experts)
    if len(logical_to_physical) != logical_count:
        raise ValueError("Logical-to-physical merge mapping has the wrong length.")
    if len(merge_weights) != logical_count:
        raise ValueError("Expert merge weights have the wrong length.")
    physical_count = int(physical_experts)
    if physical_count < 1:
        raise ValueError("physical_experts must be positive.")
    tensor_names = spec.get("tensors")
    group_sizes = spec.get("group_sizes")
    if not isinstance(tensor_names, Mapping) or not isinstance(group_sizes, Mapping):
        raise ValueError("Grouped expert merge requires tensors and group_sizes maps.")
    quant_format = normalize_quant_format(str(spec.get("quant_format", "int8")))
    nf4_blocksize = NF4_BLOCKSIZE
    if quant_format == "nf4":
        nf4_blocksize = int(_nf4_meta_from_spec(spec).blocksize)
    projection_source = CompressedExpertProjectionSource(source)
    output_tensors: dict[str, Any] = {}
    output_shapes: dict[str, list[int]] = {}
    output_group_sizes: dict[str, int] = {}
    nf4_meta = None
    gguf_meta = None
    microscaling_meta: dict[str, Any] = {}
    blockwise_fp8_meta: dict[str, Any] = {}

    for key in ("w1", "w2", "w3"):
        spec_by_name = {
            projection.name: projection
            for projection in projection_source.prototype_projection_specs()
        }
        projection_spec = spec_by_name[key]
        expert_indexed: dict[str, list[Any]] = {}
        shared: dict[str, Any] = {}
        for physical_id in range(physical_count):
            members = [
                logical_id
                for logical_id, mapped in enumerate(logical_to_physical)
                if mapped == physical_id
            ]
            if not members:
                raise ValueError(
                    f"Physical expert {physical_id} has no logical merge members."
                )
            dense = torch.zeros(projection_spec.shape, dtype=torch.float32)
            for logical_id in members:
                block = projection_source.load_prototype_projection_block(
                    key,
                    logical_id,
                    logical_id + 1,
                    device="cpu",
                    dtype=torch.float32,
                )
                dense.add_(block[0], alpha=float(merge_weights[logical_id]))
                del block
            encoded = CompressedGroupedExperts.from_empty(
                num_experts=1,
                group_sizes=(
                    int(group_sizes[key]) if quant_format == "int8" else None
                ),
                expert_weight_access="active_dequant",
                quant_format=quant_format,
                nf4_blocksize=nf4_blocksize,
            )
            encoded.load_dense_weight(key, dense.unsqueeze(0))
            del dense
            if quant_format == "nf4":
                indexed_names = (
                    f"{key}_nf4",
                    f"{key}_nf4_absmax",
                    f"{key}_nf4_nabsmax",
                    f"{key}_nf4_offset",
                )
                shared_names = (f"{key}_nf4_code", f"{key}_nf4_ncode")
                nf4_meta = encoded._nf4_meta
            elif quant_format in GGUF_FORMATS:
                indexed_names = (f"{key}_gguf",)
                shared_names = ()
                gguf_meta = encoded._gguf_meta
            elif quant_format in MICROSCALING_FORMATS:
                indexed_names = (
                    f"{key}_mx",
                    f"{key}_mx_scale",
                    f"{key}_mx_global",
                )
                shared_names = ()
                microscaling_meta[key] = encoded._microscaling_meta[key]
            elif quant_format in BLOCKWISE_FP8_FORMATS:
                indexed_names = (f"{key}_fp8", f"{key}_fp8_scale")
                shared_names = ()
                blockwise_fp8_meta[key] = encoded._blockwise_fp8_meta[key]
            else:
                indexed_names = (f"{key}_int8", f"{key}_scale")
                shared_names = ()
                output_group_sizes[key] = int(getattr(encoded, f"{key}_group_size"))
            state = encoded.state_dict()
            for local_name in indexed_names:
                expert_indexed.setdefault(local_name, []).append(
                    state[local_name].detach().cpu().contiguous()
                )
            for local_name in shared_names:
                value = state[local_name].detach().cpu().contiguous()
                previous = shared.setdefault(local_name, value)
                if not torch.equal(previous, value):
                    raise ValueError(
                        f"Quantization metadata {local_name!r} changed between experts."
                    )
            del encoded

        output_shapes[key] = [physical_count, *projection_spec.shape]
        for local_name, parts in expert_indexed.items():
            tensor_key = str(tensor_names.get(local_name, ""))
            if not tensor_key:
                raise KeyError(
                    f"Grouped expert manifest has no tensor name for {local_name!r}."
                )
            output_tensors[tensor_key] = torch.cat(parts, dim=0).contiguous()
        for local_name, value in shared.items():
            tensor_key = str(tensor_names.get(local_name, ""))
            if not tensor_key:
                raise KeyError(
                    f"Grouped expert manifest has no tensor name for {local_name!r}."
                )
            output_tensors[tensor_key] = value

    metadata: dict[str, Any] = {
        "quant_format": quant_format,
        "num_experts": physical_count,
        "group_sizes": output_group_sizes,
        "shapes": output_shapes,
    }
    if quant_format == "nf4":
        metadata["nf4_meta"] = _nf4_meta_to_spec(nf4_meta)
    elif quant_format in GGUF_FORMATS:
        metadata["gguf_meta"] = _gguf_meta_to_spec(gguf_meta)
    elif quant_format in MICROSCALING_FORMATS:
        metadata["microscaling_meta"] = {
            key: _microscaling_meta_to_spec(microscaling_meta[key])
            for key in ("w1", "w2", "w3")
        }
    elif quant_format in BLOCKWISE_FP8_FORMATS:
        metadata["blockwise_fp8_meta"] = {
            key: _blockwise_fp8_meta_to_spec(blockwise_fp8_meta[key])
            for key in ("w1", "w2", "w3")
        }
    return output_tensors, metadata


__all__ = [
    "load_grouped_expert_source",
    "merge_grouped_expert_source",
]
