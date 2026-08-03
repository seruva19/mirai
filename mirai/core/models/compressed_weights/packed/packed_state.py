"""Packed-state I/O: inventory/export/load/save + lazy/preloaded mappings (pure move)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mirai.core.lineage import sha256_file
from mirai.core.moe.storage.aliases import logical_to_physical_from_manifest_spec
from mirai.core.moe.runtime.specs import (
    CANONICAL_PACKED_EXPERT_MLP_SPEC,
    normalize_expert_weight_access_policy,
    resolve_packed_shard_size_bytes,
)
from mirai.core.moe.storage.physical_weights import PhysicalWeightProviderContext
from mirai.core.moe.storage.physical_weights import build_physical_weight_provider
from mirai.core.moe.storage.physical_weights import physical_weight_provider_names as get_compressed_weights_physical_weight_providers

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


from ..quantization.quant import CompressedWeightReport, normalize_quant_format
from ..quantization.gguf_quant import GGUF_FORMATS
from ..quantization.microscaling_quant import MICROSCALING_FORMATS
from ..quantization.blockwise_fp8 import BLOCKWISE_FP8_FORMATS
from .packed_contract import (
    COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY,
    COMPRESSED_WEIGHT_PACKED_STATE_SCHEMA_VERSION,
    COMPRESSED_WEIGHT_PACKED_STATE_SUPPORTED_SCHEMA_VERSIONS,
    DEFAULT_PACKED_SHARD_BYTES,
    _blockwise_fp8_meta_from_spec,
    _blockwise_fp8_meta_to_spec,
    _gguf_meta_from_spec,
    _gguf_meta_to_spec,
    _microscaling_meta_from_spec,
    _microscaling_meta_to_spec,
    _module_quant_format,
    _nf4_meta_from_spec,
    _nf4_meta_to_spec,
    _packed_tensor_key,
    _validate_manifest_header,
    get_compressed_weights_packed_state_quant_formats,
)
from ..execution.linear import CompressedLinear
from ..execution.experts import CompressedGroupedExperts
from ..prepare import is_dense_grouped_expert_module
from .packed_residency import LazyPackedTensorMapping
from .packed_residency import _safetensors_dtype_nbytes
from .packed_residency import PackedStateResidencyPolicy
from .packed_graph import assign_packed_state_tensor, replace_packed_child_module

def prepare_compressed_weights_modules_from_manifest(
    root: nn.Module,
    manifest: Mapping[str, Any],
) -> CompressedWeightReport:
    """Replace target modules with empty compressed_weights wrappers from a packed manifest."""
    schema_version = _validate_manifest_header(manifest)
    modules = manifest.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("compressed_weights packed state manifest must include a non-empty modules object.")

    linear_modules = 0
    grouped_modules = 0
    quantized_tensors = 0
    quantized_numel = 0
    last_expert_access = "full_dequant"
    last_expert_chunk_size = 0
    for module_name, raw_spec in modules.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"compressed_weights packed module {module_name!r} must be an object.")
        kind = str(raw_spec.get("kind") or "")
        quant_format = _module_quant_format(raw_spec, schema_version=schema_version)
        module = root.get_submodule(str(module_name)) if str(module_name) else root
        if kind == "linear":
            if not isinstance(module, (nn.Linear, CompressedLinear)):
                raise ValueError(f"Target module {module_name!r} is not a linear module.")
            replacement = CompressedLinear.from_empty(
                in_features=int(raw_spec.get("in_features")),
                out_features=int(raw_spec.get("out_features")),
                group_size=int(raw_spec.get("group_size", 0)),
                has_bias=bool(raw_spec.get("has_bias")),
                quant_format=quant_format,
                nf4_blocksize=int(
                    (raw_spec.get("nf4_meta") or {}).get("blocksize", 64)
                ),
            )
            replace_packed_child_module(root, str(module_name), replacement)
            linear_modules += 1
            quantized_tensors += 1
            quantized_numel += int(replacement.in_features * replacement.out_features)
            continue
        if kind == "grouped_experts":
            if not (
                isinstance(module, CompressedGroupedExperts)
                or is_dense_grouped_expert_module(
                    module,
                    execution_spec=CANONICAL_PACKED_EXPERT_MLP_SPEC,
                )
            ):
                raise ValueError(f"Target module {module_name!r} is not a grouped expert module.")
            access = str(raw_spec.get("expert_weight_access") or "full_dequant")
            chunk = int(raw_spec.get("expert_dequant_chunk_size") or 0)
            replacement = CompressedGroupedExperts.from_empty(
                num_experts=int(raw_spec.get("num_experts")),
                group_sizes="auto",
                expert_weight_access=access,
                expert_dequant_chunk_size=chunk,
                quant_format=quant_format,
                nf4_blocksize=int(
                    (raw_spec.get("nf4_meta") or {}).get("blocksize", 64)
                ),
            )
            aliases = logical_to_physical_from_manifest_spec(
                raw_spec,
                schema_version=schema_version,
            )
            if aliases is not None:
                replacement.configure_logical_expert_aliases(
                    aliases,
                    prototype_logical_ids=raw_spec.get("prototype_logical_ids"),
                )
            replace_packed_child_module(root, str(module_name), replacement)
            grouped_modules += 1
            group_shapes = raw_spec.get("shapes") or {}
            if isinstance(group_shapes, dict):
                quantized_tensors += len(group_shapes)
                quantized_numel += sum(
                    math.prod(int(v) for v in shape)
                    for shape in group_shapes.values()
                    if isinstance(shape, (list, tuple))
                )
            last_expert_access = normalize_expert_weight_access_policy(access)
            last_expert_chunk_size = chunk
            continue
        raise ValueError(f"Unsupported compressed_weights packed module kind {kind!r}.")
    return CompressedWeightReport(
        linear_modules=linear_modules,
        grouped_expert_modules=grouped_modules,
        quantized_tensors=quantized_tensors,
        quantized_numel=quantized_numel,
        expert_weight_access=last_expert_access,
        expert_dequant_chunk_size=last_expert_chunk_size,
    )


def _is_persistent_buffer(root: nn.Module, key: str) -> bool:
    if "." in key:
        module_name, buffer_name = key.rsplit(".", 1)
        owner = root.get_submodule(module_name)
    else:
        owner = root
        buffer_name = key
    return buffer_name not in owner._non_persistent_buffers_set


def _packed_state_inventory(
    root: nn.Module,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, Any]]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("compressed_weights packed state export requires torch.")

    inventory: list[tuple[str, torch.Tensor]] = []
    modules: dict[str, dict[str, Any]] = {}
    packed_tensor_keys: set[str] = set()
    linear_modules = 0
    grouped_modules = 0
    quantized_tensors = 0
    quantized_numel = 0

    for module_name, module in root.named_modules():
        if isinstance(module, CompressedLinear):
            quant_format = normalize_quant_format(getattr(module, "_quant_format", "int8"))
            if quant_format in BLOCKWISE_FP8_FORMATS:
                tensor_names = {
                    "weight_fp8": _packed_tensor_key(module_name, "weight_fp8"),
                    "weight_fp8_scale": _packed_tensor_key(
                        module_name, "weight_fp8_scale"
                    ),
                }
                inventory.append((tensor_names["weight_fp8"], module.weight_fp8))
                inventory.append(
                    (tensor_names["weight_fp8_scale"], module.weight_fp8_scale)
                )
                packed_tensor_keys.update(tensor_names.values())
                has_bias = module.bias is not None
                if has_bias:
                    tensor_names["bias"] = _packed_tensor_key(module_name, "bias")
                    inventory.append((tensor_names["bias"], module.bias))
                    packed_tensor_keys.add(tensor_names["bias"])
                modules[module_name] = {
                    "kind": "linear",
                    "quant_format": quant_format,
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "group_size": 0,
                    "has_bias": bool(has_bias),
                    "blockwise_fp8_meta": _blockwise_fp8_meta_to_spec(
                        module._blockwise_fp8_meta
                    ),
                    "tensors": tensor_names,
                }
                linear_modules += 1
                quantized_tensors += 1
                quantized_numel += int(module.in_features * module.out_features)
                continue
            if quant_format in MICROSCALING_FORMATS:
                tensor_names = {}
                for local_name in (
                    "weight_mx",
                    "weight_mx_scale",
                    "weight_mx_global",
                ):
                    tensor_names[local_name] = _packed_tensor_key(module_name, local_name)
                    inventory.append((tensor_names[local_name], getattr(module, local_name)))
                    packed_tensor_keys.add(tensor_names[local_name])
                has_bias = module.bias is not None
                if has_bias:
                    tensor_names["bias"] = _packed_tensor_key(module_name, "bias")
                    inventory.append((tensor_names["bias"], module.bias))
                    packed_tensor_keys.add(tensor_names["bias"])
                modules[module_name] = {
                    "kind": "linear",
                    "quant_format": quant_format,
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "group_size": 0,
                    "has_bias": bool(has_bias),
                    "microscaling_meta": _microscaling_meta_to_spec(
                        module._microscaling_meta
                    ),
                    "tensors": tensor_names,
                }
                linear_modules += 1
                quantized_tensors += 1
                quantized_numel += int(module.in_features * module.out_features)
                continue
            if quant_format == "nf4":
                tensor_names = {}
                for local_name in (
                    "weight_nf4",
                    "weight_nf4_absmax",
                    "weight_nf4_nabsmax",
                    "weight_nf4_offset",
                    "weight_nf4_code",
                    "weight_nf4_ncode",
                ):
                    tensor_names[local_name] = _packed_tensor_key(module_name, local_name)
                    inventory.append((tensor_names[local_name], getattr(module, local_name)))
                    packed_tensor_keys.add(tensor_names[local_name])
                has_bias = module.bias is not None
                if has_bias:
                    tensor_names["bias"] = _packed_tensor_key(module_name, "bias")
                    inventory.append((tensor_names["bias"], module.bias))
                    packed_tensor_keys.add(tensor_names["bias"])
                modules[module_name] = {
                    "kind": "linear",
                    "quant_format": "nf4",
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "group_size": 0,
                    "has_bias": bool(has_bias),
                    "nf4_meta": _nf4_meta_to_spec(module._nf4_meta),
                    "tensors": tensor_names,
                }
                linear_modules += 1
                quantized_tensors += 1
                quantized_numel += int(module.in_features * module.out_features)
                continue
            tensor_names = {
                "weight_int8": _packed_tensor_key(module_name, "weight_int8"),
                "weight_scale": _packed_tensor_key(module_name, "weight_scale"),
            }
            inventory.append((tensor_names["weight_int8"], module.weight_int8))
            inventory.append((tensor_names["weight_scale"], module.weight_scale))
            packed_tensor_keys.update({tensor_names["weight_int8"], tensor_names["weight_scale"]})
            has_bias = module.bias is not None
            if has_bias:
                tensor_names["bias"] = _packed_tensor_key(module_name, "bias")
                inventory.append((tensor_names["bias"], module.bias))
                packed_tensor_keys.add(tensor_names["bias"])
            modules[module_name] = {
                "kind": "linear",
                "quant_format": "int8",
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "group_size": int(module.quantization_group_size),
                "has_bias": bool(has_bias),
                "tensors": tensor_names,
            }
            linear_modules += 1
            quantized_tensors += 1
            quantized_numel += int(module.weight_int8.numel())
            continue
        if isinstance(module, CompressedGroupedExperts):
            if module.expert_mlp_spec != CANONICAL_PACKED_EXPERT_MLP_SPEC:
                raise RuntimeError(
                    "Packed grouped-expert artifacts currently encode only the "
                    "canonical w1/w3/w2 gated-product layout; export of a "
                    "provider-specific execution layout is not supported."
                )
            if not module.is_fully_loaded():
                missing = sorted(
                    set(module.expert_mlp_spec.tensor_names) - module._loaded_dense_keys
                )
                raise RuntimeError(f"Cannot export incomplete compressed_weights grouped experts {module_name!r}: {missing}.")
            quant_format = normalize_quant_format(getattr(module, "_quant_format", "int8"))
            physical_provider = getattr(module, "_physical_weight_provider", None)
            if physical_provider is not None:
                provider_spec = dict(physical_provider.manifest_spec())
                provider_tensors = physical_provider.packed_tensors()
                tensor_names = {
                    str(name): str(name)
                    for name in physical_provider.packed_tensor_names()
                }
                for tensor_name in sorted(tensor_names):
                    inventory.append((tensor_name, provider_tensors[tensor_name]))
                    packed_tensor_keys.add(tensor_name)
                shapes = {
                    key: tuple(int(value) for value in module.expert_weight_shape(key))
                    for key in ("w1", "w2", "w3")
                }
                modules[module_name] = {
                    "kind": "grouped_experts",
                    "quant_format": quant_format,
                    "num_experts": int(module.num_experts),
                    "expert_weight_access": str(module.expert_weight_access),
                    "expert_dequant_chunk_size": int(module.expert_dequant_chunk_size),
                    "group_sizes": {},
                    "shapes": shapes,
                    "tensors": tensor_names,
                    "physical_weight_provider": provider_spec,
                }
                if module.has_logical_expert_aliases():
                    mapping = module.logical_to_physical_map()
                    modules[module_name]["logical_num_experts"] = len(mapping)
                    modules[module_name]["logical_to_physical"] = list(mapping)
                    modules[module_name]["prototype_logical_ids"] = list(
                        module.prototype_logical_ids
                    )
                grouped_modules += 1
                quantized_tensors += 3
                quantized_numel += sum(math.prod(shape) for shape in shapes.values())
                continue
            tensor_names: dict[str, str] = {}
            group_sizes: dict[str, int] = {}
            shapes: dict[str, tuple[int, ...]] = {}
            rotations: dict[str, str] = {}
            for key in ("w1", "w2", "w3"):
                if quant_format in BLOCKWISE_FP8_FORMATS:
                    for suffix in ("fp8", "fp8_scale"):
                        local_name = f"{key}_{suffix}"
                        tensor_names[local_name] = _packed_tensor_key(
                            module_name, local_name
                        )
                        inventory.append(
                            (tensor_names[local_name], getattr(module, local_name))
                        )
                        packed_tensor_keys.add(tensor_names[local_name])
                    shape = tuple(int(v) for v in module.expert_weight_shape(key))
                    shapes[key] = shape
                    quantized_tensors += 1
                    quantized_numel += math.prod(shape)
                    continue
                if quant_format in MICROSCALING_FORMATS:
                    for suffix in ("mx", "mx_scale", "mx_global"):
                        local_name = f"{key}_{suffix}"
                        tensor_names[local_name] = _packed_tensor_key(
                            module_name, local_name
                        )
                        inventory.append(
                            (tensor_names[local_name], getattr(module, local_name))
                        )
                        packed_tensor_keys.add(tensor_names[local_name])
                    shape = tuple(int(v) for v in module.expert_weight_shape(key))
                    shapes[key] = shape
                    quantized_tensors += 1
                    quantized_numel += math.prod(shape)
                    continue
                if quant_format in GGUF_FORMATS:
                    local_name = f"{key}_gguf"
                    tensor_names[local_name] = _packed_tensor_key(module_name, local_name)
                    inventory.append((tensor_names[local_name], getattr(module, local_name)))
                    packed_tensor_keys.add(tensor_names[local_name])
                    shape = tuple(int(v) for v in module.expert_weight_shape(key))
                    shapes[key] = shape
                    quantized_tensors += 1
                    quantized_numel += math.prod(shape)
                    continue
                if quant_format == "nf4":
                    for suffix in (
                        "nf4",
                        "nf4_absmax",
                        "nf4_nabsmax",
                        "nf4_offset",
                        "nf4_code",
                        "nf4_ncode",
                    ):
                        local_name = f"{key}_{suffix}"
                        tensor_names[local_name] = _packed_tensor_key(module_name, local_name)
                        inventory.append((tensor_names[local_name], getattr(module, local_name)))
                        packed_tensor_keys.add(tensor_names[local_name])
                    shape = tuple(int(v) for v in module.expert_weight_shape(key))
                    shapes[key] = shape
                    quantized_tensors += 1
                    quantized_numel += math.prod(shape)
                    continue
                int8_name = f"{key}_int8"
                scale_name = f"{key}_scale"
                int8_tensor = getattr(module, int8_name)
                scale_tensor = getattr(module, scale_name)
                tensor_names[int8_name] = _packed_tensor_key(module_name, int8_name)
                tensor_names[scale_name] = _packed_tensor_key(module_name, scale_name)
                inventory.append((tensor_names[int8_name], int8_tensor))
                inventory.append((tensor_names[scale_name], scale_tensor))
                packed_tensor_keys.update({tensor_names[int8_name], tensor_names[scale_name]})
                group_sizes[key] = int(getattr(module, f"{key}_group_size"))
                shapes[key] = tuple(int(v) for v in int8_tensor.shape)
                rotation = module.expert_rotation(key)
                if rotation is not None:
                    rotation_name = f"{key}_rotation"
                    tensor_names[rotation_name] = _packed_tensor_key(
                        module_name,
                        rotation_name,
                    )
                    rotations[key] = tensor_names[rotation_name]
                    inventory.append((tensor_names[rotation_name], rotation))
                    packed_tensor_keys.add(tensor_names[rotation_name])
                quantized_tensors += 1
                quantized_numel += int(int8_tensor.numel())
            modules[module_name] = {
                "kind": "grouped_experts",
                "quant_format": quant_format,
                "num_experts": int(module.num_experts),
                "expert_weight_access": str(module.expert_weight_access),
                "expert_dequant_chunk_size": int(module.expert_dequant_chunk_size),
                "group_sizes": group_sizes,
                "shapes": shapes,
                "tensors": tensor_names,
            }
            if rotations:
                if set(rotations) != {"w1", "w2", "w3"}:
                    raise RuntimeError(
                        "Learned expert rotations must cover w1, w2, and w3."
                    )
                if not torch.equal(
                    module.expert_rotation("w1"),
                    module.expert_rotation("w3"),
                ):
                    raise RuntimeError(
                        "Learned expert rotations require w1/w3 to share a matrix."
                    )
                modules[module_name]["rotations"] = rotations
                report = getattr(module, "_learned_rotation_report", None)
                if isinstance(report, Mapping):
                    modules[module_name]["learned_rotation_report"] = dict(report)
            if module.has_logical_expert_aliases():
                mapping = module.logical_to_physical_map()
                modules[module_name]["logical_num_experts"] = len(mapping)
                modules[module_name]["logical_to_physical"] = list(mapping)
                modules[module_name]["prototype_logical_ids"] = list(
                    module.prototype_logical_ids
                )
            if quant_format == "nf4":
                modules[module_name]["nf4_meta"] = _nf4_meta_to_spec(module._nf4_meta)
            elif quant_format in GGUF_FORMATS:
                modules[module_name]["gguf_meta"] = _gguf_meta_to_spec(module._gguf_meta)
            elif quant_format in MICROSCALING_FORMATS:
                modules[module_name]["microscaling_meta"] = {
                    key: _microscaling_meta_to_spec(module._microscaling_meta[key])
                    for key in ("w1", "w2", "w3")
                }
            elif quant_format in BLOCKWISE_FP8_FORMATS:
                modules[module_name]["blockwise_fp8_meta"] = {
                    key: _blockwise_fp8_meta_to_spec(
                        module._blockwise_fp8_meta[key]
                    )
                    for key in ("w1", "w2", "w3")
                }
            grouped_modules += 1

    residual_tensors: dict[str, str] = {}
    state_entries = list(root.named_parameters()) + [
        (key, tensor)
        for key, tensor in root.named_buffers()
        if _is_persistent_buffer(root, str(key))
    ]
    for key, tensor in state_entries:
        tensor_key = str(key)
        if tensor_key in packed_tensor_keys:
            continue
        residual_tensors[tensor_key] = tensor_key
        inventory.append((tensor_key, tensor))

    learned_rotation_reports = {
        str(name): dict(spec["learned_rotation_report"])
        for name, spec in modules.items()
        if isinstance(spec, Mapping)
        and isinstance(spec.get("learned_rotation_report"), Mapping)
    }
    manifest = {
        "schema_version": (
            5
            if any(
                isinstance(spec, Mapping) and "rotations" in spec
                for spec in modules.values()
            )
            else 4
            if any(
                isinstance(spec, Mapping) and "physical_weight_provider" in spec
                for spec in modules.values()
            )
            else 3
            if any(
                isinstance(spec, Mapping) and "logical_to_physical" in spec
                for spec in modules.values()
            )
            else COMPRESSED_WEIGHT_PACKED_STATE_SCHEMA_VERSION
        ),
        "format": "mirai.compressed_weights.packed_state",
        "quant_formats": sorted(
            {str(spec["quant_format"]) for spec in modules.values()}
        ),
        "modules": modules,
        "residual_tensors": residual_tensors,
        "summary": {
            "linear_modules": linear_modules,
            "grouped_expert_modules": grouped_modules,
            "quantized_tensors": quantized_tensors,
            "quantized_numel": quantized_numel,
            "residual_tensors": len(residual_tensors),
        },
    }
    if learned_rotation_reports:
        manifest["learned_expert_rotation"] = {
            "format": "mirai.moe.learned_expert_rotation",
            "schema_version": 1,
            "modules": learned_rotation_reports,
        }
    inventory.sort(key=lambda item: item[0])
    return inventory, manifest

def export_compressed_weights_packed_state(root: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Materialize a packed state in memory for small artifacts and tests."""
    inventory, manifest = _packed_state_inventory(root)
    tensors = {
        key: tensor.detach().to(device="cpu").contiguous()
        for key, tensor in inventory
    }
    return tensors, manifest

def load_compressed_weights_packed_state(
    root: nn.Module,
    tensors: Mapping[str, torch.Tensor],
    manifest: Mapping[str, Any],
    *,
    strict: bool = True,
    expert_weight_access_override: str | None = None,
    expert_dequant_chunk_size_override: int | None = None,
) -> CompressedWeightReport:
    """Restore a packed compressed_weights state into an existing compressed_weights graph."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("compressed_weights packed state restore requires torch.")
    schema_version = _validate_manifest_header(manifest)
    modules = manifest.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("compressed_weights packed state manifest must include a non-empty modules object.")

    used_tensor_keys: set[str] = set()
    linear_modules = 0
    grouped_modules = 0
    quantized_tensors = 0
    quantized_numel = 0
    last_expert_access = "full_dequant"
    last_expert_chunk_size = 0

    for module_name, raw_spec in modules.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"compressed_weights packed module {module_name!r} must be an object.")
        module = root if not module_name else root.get_submodule(str(module_name))
        tensor_names = raw_spec.get("tensors")
        if not isinstance(tensor_names, dict):
            raise ValueError(f"compressed_weights packed module {module_name!r} has no tensor map.")
        kind = str(raw_spec.get("kind") or "")
        quant_format = _module_quant_format(raw_spec, schema_version=schema_version)
        if kind == "linear":
            if not isinstance(module, CompressedLinear):
                raise ValueError(f"Target module {module_name!r} is not CompressedLinear.")
            if int(raw_spec.get("in_features")) != int(module.in_features) or int(raw_spec.get("out_features")) != int(module.out_features):
                raise ValueError(f"Target module {module_name!r} linear feature metadata mismatch.")
            required = (
                ["weight_fp8", "weight_fp8_scale"]
                if quant_format in BLOCKWISE_FP8_FORMATS
                else ["weight_mx", "weight_mx_scale", "weight_mx_global"]
                if quant_format in MICROSCALING_FORMATS
                else
                [
                    "weight_nf4",
                    "weight_nf4_absmax",
                    "weight_nf4_nabsmax",
                    "weight_nf4_offset",
                    "weight_nf4_code",
                    "weight_nf4_ncode",
                ]
                if quant_format == "nf4"
                else ["weight_int8", "weight_scale"]
            )
            if bool(raw_spec.get("has_bias")):
                required.append("bias")
            missing_keys = [key for key in required if str(tensor_names.get(key) or "") not in tensors]
            if missing_keys:
                raise KeyError(f"compressed_weights packed module {module_name!r} missing tensors for {missing_keys}.")
            for key in required:
                used_tensor_keys.add(str(tensor_names[key]))
            if quant_format in BLOCKWISE_FP8_FORMATS:
                module.load_blockwise_fp8_packed_state(
                    codes=tensors[str(tensor_names["weight_fp8"])],
                    scales=tensors[str(tensor_names["weight_fp8_scale"])],
                    meta=_blockwise_fp8_meta_from_spec(raw_spec),
                    bias=tensors[str(tensor_names["bias"])]
                    if bool(raw_spec.get("has_bias"))
                    else None,
                )
            elif quant_format in MICROSCALING_FORMATS:
                module.load_microscaling_packed_state(
                    packed=tensors[str(tensor_names["weight_mx"])],
                    scales=tensors[str(tensor_names["weight_mx_scale"])],
                    global_scale=tensors[str(tensor_names["weight_mx_global"])],
                    meta=_microscaling_meta_from_spec(raw_spec),
                    bias=tensors[str(tensor_names["bias"])]
                    if bool(raw_spec.get("has_bias"))
                    else None,
                )
            elif quant_format == "nf4":
                module.load_nf4_packed_state(
                    packed=tensors[str(tensor_names["weight_nf4"])],
                    absmax=tensors[str(tensor_names["weight_nf4_absmax"])],
                    nested_absmax=tensors[str(tensor_names["weight_nf4_nabsmax"])],
                    offset=tensors[str(tensor_names["weight_nf4_offset"])],
                    code=tensors[str(tensor_names["weight_nf4_code"])],
                    nested_code=tensors[str(tensor_names["weight_nf4_ncode"])],
                    meta=_nf4_meta_from_spec(raw_spec),
                    bias=tensors[str(tensor_names["bias"])] if bool(raw_spec.get("has_bias")) else None,
                )
            else:
                module.load_packed_state(
                    weight_int8=tensors[str(tensor_names["weight_int8"])],
                    weight_scale=tensors[str(tensor_names["weight_scale"])],
                    group_size=int(raw_spec.get("group_size", 0)),
                    bias=tensors[str(tensor_names["bias"])] if bool(raw_spec.get("has_bias")) else None,
                )
            linear_modules += 1
            quantized_tensors += 1
            quantized_numel += int(module.frozen_quantized_numel())
            continue
        if kind == "grouped_experts":
            if not isinstance(module, CompressedGroupedExperts):
                raise ValueError(f"Target module {module_name!r} is not CompressedGroupedExperts.")
            if int(raw_spec.get("num_experts")) != int(module.num_experts):
                raise ValueError(f"Target module {module_name!r} expert count metadata mismatch.")
            aliases = logical_to_physical_from_manifest_spec(
                raw_spec,
                schema_version=schema_version,
            )
            if aliases is not None:
                existing = module.logical_to_physical_map()
                if existing and existing != aliases:
                    raise ValueError(
                        f"Target module {module_name!r} logical expert mapping mismatch."
                    )
                if not existing:
                    module.configure_logical_expert_aliases(
                        aliases,
                        prototype_logical_ids=raw_spec.get(
                            "prototype_logical_ids"
                        ),
                    )
            group_sizes = raw_spec.get("group_sizes")
            if not isinstance(group_sizes, dict):
                raise ValueError(f"compressed_weights packed module {module_name!r} has no group_sizes map.")
            raw_rotations = raw_spec.get("rotations", {})
            if not isinstance(raw_rotations, Mapping):
                raise ValueError(
                    f"compressed_weights packed module {module_name!r} has an "
                    "invalid rotations map."
                )
            rotations: dict[str, torch.Tensor] = {}
            if raw_rotations:
                if schema_version < 5 or quant_format != "int8":
                    raise ValueError(
                        "Learned expert rotations require schema v5 INT8 storage."
                    )
                if set(str(key) for key in raw_rotations) != {"w1", "w2", "w3"}:
                    raise ValueError(
                        "Learned expert rotations must cover w1, w2, and w3."
                    )
                for key in ("w1", "w2", "w3"):
                    tensor_key = str(raw_rotations[key])
                    if tensor_key not in tensors:
                        raise KeyError(
                            f"compressed_weights packed module {module_name!r} "
                            f"is missing learned rotation {key!r}."
                        )
                    rotations[key] = tensors[tensor_key]
                    used_tensor_keys.add(tensor_key)
                if not torch.equal(rotations["w1"], rotations["w3"]):
                    raise ValueError(
                        "Learned expert rotations require identical w1/w3 matrices."
                    )
            access = str(
                expert_weight_access_override
                if expert_weight_access_override is not None
                else raw_spec.get("expert_weight_access") or "full_dequant"
            )
            chunk = int(
                expert_dequant_chunk_size_override
                if expert_dequant_chunk_size_override is not None
                else raw_spec.get("expert_dequant_chunk_size") or 0
            )
            module.set_expert_weight_access_policy(
                expert_weight_access=access,
                expert_dequant_chunk_size=chunk,
            )
            shapes = raw_spec.get("shapes")
            if not isinstance(shapes, dict):
                raise ValueError(f"compressed_weights packed module {module_name!r} has no shapes map.")
            provider_spec = raw_spec.get("physical_weight_provider")
            if provider_spec is not None:
                if schema_version < 4 or not isinstance(provider_spec, Mapping):
                    raise ValueError(
                        f"compressed_weights packed module {module_name!r} has an invalid "
                        "physical_weight_provider declaration."
                    )
                provider = build_physical_weight_provider(
                    str(provider_spec.get("name", "")),
                    PhysicalWeightProviderContext(
                        module_name=str(module_name),
                        num_experts=int(module.num_experts),
                        shapes={
                            str(key): tuple(int(value) for value in shape)
                            for key, shape in shapes.items()
                        },
                        spec=provider_spec,
                        tensors=tensors,
                    ),
                )
                used_tensor_keys.update(provider.packed_tensor_names())
                module.bind_physical_weight_provider(provider)
                quantized_tensors += len(shapes)
                quantized_numel += sum(
                    math.prod(int(value) for value in shape)
                    for shape in shapes.values()
                )
                grouped_modules += 1
                last_expert_access = normalize_expert_weight_access_policy(access)
                last_expert_chunk_size = chunk
                continue
            if quant_format in GGUF_FORMATS:
                gguf_meta = _gguf_meta_from_spec(raw_spec)
                for key in ("w1", "w2", "w3"):
                    local_name = f"{key}_gguf"
                    tensor_key = str(tensor_names.get(local_name) or "")
                    if tensor_key not in tensors:
                        raise KeyError(
                            f"compressed_weights packed module {module_name!r} missing "
                            f"GGUF tensor for {key}."
                        )
                    used_tensor_keys.add(tensor_key)
                    module.load_gguf_packed_weight(
                        key,
                        blocks=tensors[tensor_key],
                        shape=shapes[key],
                        meta=gguf_meta,
                    )
                    quantized_tensors += 1
                    quantized_numel += math.prod(int(v) for v in shapes[key])
                grouped_modules += 1
                last_expert_access = normalize_expert_weight_access_policy(access)
                last_expert_chunk_size = chunk
                continue
            if quant_format in BLOCKWISE_FP8_FORMATS:
                raw_meta = raw_spec.get("blockwise_fp8_meta")
                if not isinstance(raw_meta, Mapping):
                    raise ValueError(
                        f"compressed_weights packed module {module_name!r} has no "
                        "blockwise FP8 metadata."
                    )
                metadata = {
                    key: _blockwise_fp8_meta_from_spec(
                        {"blockwise_fp8_meta": raw_meta.get(key)}
                    )
                    for key in ("w1", "w2", "w3")
                }
                required_by_key = {
                    key: [f"{key}_fp8", f"{key}_fp8_scale"]
                    for key in ("w1", "w2", "w3")
                }
                missing_keys = [
                    name
                    for required in required_by_key.values()
                    for name in required
                    if str(tensor_names.get(name) or "") not in tensors
                ]
                if missing_keys:
                    raise KeyError(
                        f"compressed_weights packed module {module_name!r} "
                        f"missing FP8 tensors for {missing_keys}."
                    )
                used_tensor_keys.update(
                    str(tensor_names[name])
                    for required in required_by_key.values()
                    for name in required
                )
                stream_fp8 = callable(getattr(tensors, "get_slice", None)) and access in {
                    "active_dequant",
                    "chunked_dequant",
                    "fused_kernel",
                }
                if stream_fp8:
                    module.bind_blockwise_fp8_packed_source(
                        source=tensors,
                        tensor_names=tensor_names,
                        shapes=shapes,
                        metadata=metadata,
                    )
                    quantized_tensors += 3
                    quantized_numel += sum(
                        math.prod(int(v) for v in shapes[key])
                        for key in ("w1", "w2", "w3")
                    )
                    grouped_modules += 1
                    last_expert_access = normalize_expert_weight_access_policy(access)
                    last_expert_chunk_size = chunk
                    continue
                for key in ("w1", "w2", "w3"):
                    module.load_blockwise_fp8_packed_weight(
                        key,
                        codes=tensors[str(tensor_names[f"{key}_fp8"])],
                        scales=tensors[str(tensor_names[f"{key}_fp8_scale"])],
                        shape=shapes[key],
                        meta=metadata[key],
                    )
                    quantized_tensors += 1
                    quantized_numel += math.prod(int(v) for v in shapes[key])
                grouped_modules += 1
                last_expert_access = normalize_expert_weight_access_policy(access)
                last_expert_chunk_size = chunk
                continue
            if quant_format in MICROSCALING_FORMATS:
                raw_meta = raw_spec.get("microscaling_meta")
                if not isinstance(raw_meta, Mapping):
                    raise ValueError(
                        f"compressed_weights packed module {module_name!r} has no "
                        "microscaling metadata."
                    )
                for key in ("w1", "w2", "w3"):
                    required = [
                        f"{key}_mx",
                        f"{key}_mx_scale",
                        f"{key}_mx_global",
                    ]
                    missing_keys = [
                        name
                        for name in required
                        if str(tensor_names.get(name) or "") not in tensors
                    ]
                    if missing_keys:
                        raise KeyError(
                            f"compressed_weights packed module {module_name!r} missing "
                            f"microscaling tensors for {missing_keys}."
                        )
                    used_tensor_keys.update(str(tensor_names[name]) for name in required)
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
                    quantized_tensors += 1
                    quantized_numel += math.prod(int(v) for v in shapes[key])
                grouped_modules += 1
                last_expert_access = normalize_expert_weight_access_policy(access)
                last_expert_chunk_size = chunk
                continue
            if quant_format == "nf4":
                meta = _nf4_meta_from_spec(raw_spec)
                get_slice = getattr(tensors, "get_slice", None)
                stream_nf4 = callable(get_slice) and access in {
                    "active_dequant",
                    "chunked_dequant",
                }
                codebooks: dict[str, torch.Tensor] = {}
                for key in ("w1", "w2", "w3"):
                    required = [
                        f"{key}_nf4",
                        f"{key}_nf4_absmax",
                        f"{key}_nf4_nabsmax",
                        f"{key}_nf4_offset",
                        f"{key}_nf4_code",
                        f"{key}_nf4_ncode",
                    ]
                    missing_keys = [
                        name
                        for name in required
                        if str(tensor_names.get(name) or "") not in tensors
                    ]
                    if missing_keys:
                        raise KeyError(
                            f"compressed_weights packed module {module_name!r} missing "
                            f"NF4 tensors for {missing_keys}."
                        )
                    used_tensor_keys.update(str(tensor_names[name]) for name in required)
                    if stream_nf4:
                        codebooks[f"{key}_nf4_code"] = tensors[
                            str(tensor_names[f"{key}_nf4_code"])
                        ]
                        codebooks[f"{key}_nf4_ncode"] = tensors[
                            str(tensor_names[f"{key}_nf4_ncode"])
                        ]
                    else:
                        module.load_nf4_packed_weight(
                            key,
                            packed=tensors[str(tensor_names[f"{key}_nf4"])],
                            absmax=tensors[str(tensor_names[f"{key}_nf4_absmax"])],
                            nested_absmax=tensors[
                                str(tensor_names[f"{key}_nf4_nabsmax"])
                            ],
                            offset=tensors[str(tensor_names[f"{key}_nf4_offset"])],
                            code=tensors[str(tensor_names[f"{key}_nf4_code"])],
                            nested_code=tensors[
                                str(tensor_names[f"{key}_nf4_ncode"])
                            ],
                            shape=shapes[key],
                            meta=meta,
                        )
                    quantized_tensors += 1
                    quantized_numel += math.prod(int(v) for v in shapes[key])
                if stream_nf4:
                    module.bind_nf4_packed_source(
                        source=tensors,
                        tensor_names=tensor_names,
                        shapes=shapes,
                        meta=meta,
                        codebooks=codebooks,
                    )
                grouped_modules += 1
                last_expert_access = normalize_expert_weight_access_policy(access)
                last_expert_chunk_size = chunk
                continue
            get_slice = getattr(tensors, "get_slice", None)
            if callable(get_slice) and access in {"active_dequant", "chunked_dequant"}:
                for key in ("w1", "w2", "w3"):
                    int8_name = f"{key}_int8"
                    scale_name = f"{key}_scale"
                    int8_tensor_key = str(tensor_names.get(int8_name) or "")
                    scale_tensor_key = str(tensor_names.get(scale_name) or "")
                    if int8_tensor_key not in tensors or scale_tensor_key not in tensors:
                        raise KeyError(
                            f"compressed_weights packed module {module_name!r} missing tensors for {key}."
                        )
                    used_tensor_keys.update({int8_tensor_key, scale_tensor_key})
                    quantized_tensors += 1
                    quantized_numel += math.prod(int(v) for v in shapes[key])
                module.bind_packed_source(
                    source=tensors,
                    tensor_names=tensor_names,
                    group_sizes=group_sizes,
                    shapes=shapes,
                    rotations=rotations,
                )
                report = raw_spec.get("learned_rotation_report")
                if isinstance(report, Mapping):
                    module._learned_rotation_report = dict(report)
                grouped_modules += 1
                last_expert_access = normalize_expert_weight_access_policy(access)
                last_expert_chunk_size = chunk
                continue
            for key in ("w1", "w2", "w3"):
                int8_name = f"{key}_int8"
                scale_name = f"{key}_scale"
                if str(tensor_names.get(int8_name) or "") not in tensors or str(tensor_names.get(scale_name) or "") not in tensors:
                    raise KeyError(f"compressed_weights packed module {module_name!r} missing tensors for {key}.")
                used_tensor_keys.add(str(tensor_names[int8_name]))
                used_tensor_keys.add(str(tensor_names[scale_name]))
                module.load_quantized_weight(
                    key,
                    weight_int8=tensors[str(tensor_names[int8_name])],
                    weight_scale=tensors[str(tensor_names[scale_name])],
                    group_size=int(group_sizes[key]),
                    rotation=rotations.get(key),
                )
                quantized_tensors += 1
                quantized_numel += int(getattr(module, int8_name).numel())
            report = raw_spec.get("learned_rotation_report")
            if isinstance(report, Mapping):
                module._learned_rotation_report = dict(report)
            grouped_modules += 1
            last_expert_access = normalize_expert_weight_access_policy(access)
            last_expert_chunk_size = chunk
            continue
        raise ValueError(f"Unsupported compressed_weights packed module kind {kind!r}.")

    residual_tensors = manifest.get("residual_tensors", {})
    if residual_tensors is None:
        residual_tensors = {}
    if not isinstance(residual_tensors, dict):
        raise ValueError("compressed_weights packed state residual_tensors must be an object.")
    grouped_expert_axes: list[tuple[str, int]] = []
    for raw_module_name, raw_spec in modules.items():
        if not isinstance(raw_spec, Mapping) or str(raw_spec.get("kind")) != "grouped_experts":
            continue
        module_name = str(raw_module_name)
        parent = module_name.rsplit(".", 1)[0] if "." in module_name else ""
        grouped_expert_axes.append((parent, int(raw_spec.get("num_experts", 0))))
    grouped_expert_axes.sort(key=lambda item: len(item[0]), reverse=True)

    for raw_key, raw_tensor_key in residual_tensors.items():
        key = str(raw_key)
        tensor_key = str(raw_tensor_key)
        if tensor_key not in tensors:
            raise KeyError(f"compressed_weights packed residual tensor {key!r} missing tensor {tensor_key!r}.")
        prefix_matches = [
            (prefix, expert_count)
            for prefix, expert_count in grouped_expert_axes
            if (key.startswith(prefix + ".") if prefix else True)
        ]
        longest_prefix = max(
            (len(prefix) for prefix, _expert_count in prefix_matches),
            default=-1,
        )
        matching_axes = {
            expert_count
            for prefix, expert_count in prefix_matches
            if len(prefix) == longest_prefix
        }
        if len(matching_axes) > 1:
            raise ValueError(
                f"compressed_weights residual tensor {key!r} has ambiguous "
                "sibling grouped-expert counts."
            )
        assign_packed_state_tensor(
            root,
            key,
            tensors[tensor_key],
            expert_axis_size=(
                next(iter(matching_axes)) if matching_axes else None
            ),
        )
        used_tensor_keys.add(tensor_key)

    if strict:
        unexpected = sorted(set(str(key) for key in tensors) - used_tensor_keys)
        if unexpected:
            raise ValueError(f"Unexpected compressed_weights packed tensors: {unexpected}.")

    return CompressedWeightReport(
        linear_modules=linear_modules,
        grouped_expert_modules=grouped_modules,
        quantized_tensors=quantized_tensors,
        quantized_numel=quantized_numel,
        expert_weight_access=last_expert_access,
        expert_dequant_chunk_size=last_expert_chunk_size,
    )
def save_compressed_weights_packed_state(
    path: str | Path,
    root: nn.Module,
    *,
    metadata: Mapping[str, str] | None = None,
    storage_alignment_bytes: int = 0,
) -> Path:
    inventory, manifest = _packed_state_inventory(root)
    return save_compressed_weights_packed_tensors(
        path, dict(inventory), manifest, metadata=metadata,
        storage_alignment_bytes=storage_alignment_bytes,
    )

def save_compressed_weights_packed_tensors(
    path: str | Path,
    tensors: Mapping[str, torch.Tensor],
    manifest: Mapping[str, Any],
    *,
    metadata: Mapping[str, str] | None = None,
    storage_alignment_bytes: int = 0,
) -> Path:
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required to save compressed_weights packed states.") from exc
    from mirai.core.models.compressed_weights.packed.packed_storage_alignment import (
        STORAGE_ALIGNMENT_METADATA_KEY,
        save_safetensors_with_storage_alignment,
    )

    if torch is None:  # pragma: no cover
        raise RuntimeError("compressed_weights packed-state save requires torch.")
    _validate_manifest_header(manifest)
    inventory = sorted(
        ((str(key), tensor) for key, tensor in tensors.items()),
        key=lambda item: item[0],
    )
    if not inventory:
        raise ValueError("Cannot save an empty compressed_weights packed tensor mapping.")
    invalid = [key for key, tensor in inventory if not isinstance(tensor, torch.Tensor)]
    if invalid:
        raise TypeError(f"Packed-state values must be tensors; invalid keys: {invalid}.")
    payload_metadata = {str(k): str(v) for k, v in dict(metadata or {}).items()}
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    payload_metadata[COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY] = manifest_json
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_limit = resolve_packed_shard_size_bytes()
    partitions: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key, tensor in inventory:
        tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
        if current and current_bytes + tensor_bytes > shard_limit:
            partitions.append(current)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += tensor_bytes
    if current:
        partitions.append(current)
    if len(partitions) <= 1:
        tensors = {key: tensor.detach().to(device="cpu").contiguous()
                   for key, tensor in inventory}
        save_safetensors_with_storage_alignment(
            save_file, tensors, output, metadata=payload_metadata,
            alignment_bytes=storage_alignment_bytes)
        return output
    shard_count = len(partitions)
    weight_map: dict[str, str] = {}
    stem = output.name.removesuffix(".safetensors")
    tensor_by_key = dict(inventory)
    for shard_index, keys in enumerate(partitions, start=1):
        shard_name = f"{stem}-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        shard_path = output.with_name(shard_name)
        shard_tensors = {
            key: tensor_by_key[key].detach().to(device="cpu").contiguous()
            for key in keys
        }
        save_safetensors_with_storage_alignment(
            save_file,
            shard_tensors,
            shard_path,
            metadata=(payload_metadata if shard_index == 1 else None),
            alignment_bytes=storage_alignment_bytes,
        )
        for key in keys:
            weight_map[key] = shard_name
        del shard_tensors
    index_path = output.with_name(output.name + ".index.json")
    index_payload = {
        "format": "mirai.compressed_weights.sharded_safetensors",
        "metadata": {
            **{str(k): str(v) for k, v in dict(metadata or {}).items()},
            COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY: manifest_json,
            **(
                {STORAGE_ALIGNMENT_METADATA_KEY: str(storage_alignment_bytes)}
                if storage_alignment_bytes else {}
            ),
            "total_size": sum(
                int(tensor.numel()) * int(tensor.element_size())
                for _key, tensor in inventory
            ),
        },
        "weight_map": weight_map,
    }
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index_path


def load_compressed_weights_packed_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Materialize all tensors from a one-file or sharded packed artifact.

    Runtime loading remains lazy by default. This explicit materialization API
    is for offline artifact transforms that must rewrite the complete state.
    """
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"compressed_weights packed-state artifact '{input_path}' was not found.")
    read_compressed_weights_packed_state_manifest(input_path)
    source = LazyPackedTensorMapping(input_path)
    return {str(key): source[str(key)] for key in source}


def packed_artifact_fingerprint(path: str | Path) -> str:
    """Fingerprint one-file and indexed sharded packed artifacts."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Packed artifact was not found: {artifact}.")
    if not artifact.name.endswith(".index.json"):
        return "sha256:" + sha256_file(artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("Packed artifact index has no weight_map.")
    shard_names = sorted({str(value) for value in weight_map.values()})
    digest = hashlib.sha256()
    digest.update(artifact.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(sha256_file(artifact).encode("ascii"))
    digest.update(b"\0")
    for shard_name in shard_names:
        if Path(shard_name).name != shard_name:
            raise ValueError("Packed artifact shard names must be local filenames.")
        shard = artifact.with_name(shard_name)
        if not shard.is_file():
            raise FileNotFoundError(f"Packed artifact shard was not found: {shard}.")
        digest.update(shard_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(shard).encode("ascii"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def read_compressed_weights_packed_state_manifest(path: str | Path) -> dict[str, Any]:
    """Read only the packed-state manifest metadata from a safetensors artifact."""
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required to read compressed_weights packed states.") from exc

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"compressed_weights packed-state artifact '{input_path}' was not found. Generate "
            "it with scripts/tools/export_compressed_weights_packed_state.py (which writes the packed "
            ".safetensors and its .index.json sidecar), or unset "
            "memory.frozen_weight_packed_state_path."
        )
    if input_path.name.endswith(".index.json"):
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Packed compressed_weights index metadata must be an object.")
        raw_manifest = metadata.get(COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY)
    else:
        with safe_open(str(input_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            raw_manifest = metadata.get(COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY)
    if not raw_manifest:
        raise ValueError(
            f"compressed_weights packed state metadata key "
            f"{COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY!r} is missing."
        )
    manifest = json.loads(raw_manifest)
    if not isinstance(manifest, dict):
        raise ValueError("compressed_weights packed state manifest must be a JSON object.")
    return manifest


def load_compressed_weights_packed_state_file(
    path: str | Path,
    root: nn.Module,
    *,
    strict: bool = True,
    expert_weight_access_override: str | None = None,
    expert_dequant_chunk_size_override: int | None = None,
    packed_state_preload: str = "off",
    packed_stream_cache_gib: float = 0.0,
    packed_stream_backend: str = "staged", packed_stream_prefetch_depth: int = 0,
) -> CompressedWeightReport:
    """Load one safetensors packed artifact with the selected residency policy."""
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required to load compressed_weights packed states.") from exc

    input_path = Path(path)
    manifest = read_compressed_weights_packed_state_manifest(input_path)
    tensors, _preload_info = PackedStateResidencyPolicy(
        packed_state_preload,
        stream_cache_gib=packed_stream_cache_gib,
        stream_backend=packed_stream_backend, stream_prefetch_depth=packed_stream_prefetch_depth,
    ).open(input_path)
    return load_compressed_weights_packed_state(
        root,
        tensors,
        manifest,
        strict=strict,
        expert_weight_access_override=expert_weight_access_override,
        expert_dequant_chunk_size_override=expert_dequant_chunk_size_override,
    )
