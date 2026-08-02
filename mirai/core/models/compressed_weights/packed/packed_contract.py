"""Versioned manifest and format metadata for compressed packed states."""

from __future__ import annotations

from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..quantization.gguf_quant import _GgufMeta
from ..quantization.microscaling_quant import MicroscalingMeta
from ..quantization.blockwise_fp8 import BlockwiseFP8Meta
from ..quantization.quant import _Nf4Meta, normalize_quant_format


COMPRESSED_WEIGHT_PACKED_STATE_SCHEMA_VERSION = 2
COMPRESSED_WEIGHT_PACKED_STATE_SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {1, 2, 3, 4, 5}
)
COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY = "mirai_compressed_weights_manifest"
DEFAULT_PACKED_SHARD_BYTES = 2 * 1024 * 1024 * 1024


def _validate_manifest_header(manifest: Mapping[str, Any]) -> int:
    version = int(manifest.get("schema_version", 0))
    if version not in COMPRESSED_WEIGHT_PACKED_STATE_SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            "Unsupported compressed_weights packed state schema_version "
            f"{manifest.get('schema_version')!r}."
        )
    if manifest.get("format") != "mirai.compressed_weights.packed_state":
        raise ValueError(
            f"Unsupported compressed_weights packed state format {manifest.get('format')!r}."
        )
    return version


def _module_quant_format(spec: Mapping[str, Any], *, schema_version: int) -> str:
    if schema_version == 1:
        return "int8"
    return normalize_quant_format(str(spec.get("quant_format") or "int8"))


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _torch_dtype(name: Any) -> torch.dtype:
    value = str(name or "").removeprefix("torch.")
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported NF4 packed-state dtype {name!r}.")
    return dtype


def _nf4_meta_to_spec(meta: _Nf4Meta | None) -> dict[str, Any]:
    if meta is None:
        raise RuntimeError("Cannot export NF4 packed state without quantization metadata.")
    return {
        "blocksize": int(meta.blocksize),
        "nested_blocksize": int(meta.nested_blocksize),
        "nested_dtype": _dtype_name(meta.nested_dtype),
        "weight_dtype": _dtype_name(meta.weight_dtype),
    }


def _nf4_meta_from_spec(spec: Mapping[str, Any]) -> _Nf4Meta:
    raw = spec.get("nf4_meta")
    if not isinstance(raw, Mapping):
        raise ValueError("NF4 packed module has no nf4_meta object.")
    return _Nf4Meta(
        blocksize=int(raw.get("blocksize", 0)),
        nested_blocksize=int(raw.get("nested_blocksize", 0)),
        nested_dtype=_torch_dtype(raw.get("nested_dtype")),
        weight_dtype=_torch_dtype(raw.get("weight_dtype")),
    )


def _gguf_meta_to_spec(meta: _GgufMeta | None) -> dict[str, Any]:
    if meta is None:
        raise RuntimeError("Cannot export GGUF packed state without quantization metadata.")
    return {
        "block_format": str(meta.block_format),
        "blocksize": int(meta.blocksize),
        "type_size": int(meta.type_size),
        "weight_dtype": str(meta.weight_dtype),
    }


def _gguf_meta_from_spec(spec: Mapping[str, Any]) -> _GgufMeta:
    raw = spec.get("gguf_meta")
    if not isinstance(raw, Mapping):
        raise ValueError("GGUF packed module has no gguf_meta object.")
    return _GgufMeta(
        block_format=str(raw.get("block_format", "")),
        blocksize=int(raw.get("blocksize", 0)),
        type_size=int(raw.get("type_size", 0)),
        weight_dtype=str(raw.get("weight_dtype", "bfloat16")),
    )


def _microscaling_meta_to_spec(meta: MicroscalingMeta | None) -> dict[str, Any]:
    if meta is None:
        raise RuntimeError(
            "Cannot export microscaling packed state without quantization metadata."
        )
    return {
        "format": str(meta.format),
        "block_size": int(meta.block_size),
        "shape": [int(dim) for dim in meta.shape],
        "padding": int(meta.padding),
    }


def _microscaling_meta_from_spec(spec: Mapping[str, Any]) -> MicroscalingMeta:
    raw = spec.get("microscaling_meta")
    if not isinstance(raw, Mapping):
        raise ValueError("Microscaling packed module has no microscaling_meta object.")
    shape = raw.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape:
        raise ValueError("Microscaling packed module has invalid shape metadata.")
    return MicroscalingMeta(
        format=str(raw.get("format", "")),
        block_size=int(raw.get("block_size", 0)),
        shape=tuple(int(dim) for dim in shape),
        padding=int(raw.get("padding", 0)),
    )


def _blockwise_fp8_meta_to_spec(meta: BlockwiseFP8Meta | None) -> dict[str, Any]:
    if meta is None:
        raise RuntimeError("Cannot export blockwise FP8 state without metadata.")
    return {
        "shape": [int(dim) for dim in meta.shape],
        "weight_block": [int(dim) for dim in meta.weight_block],
        "activation_block": int(meta.activation_block),
    }


def _blockwise_fp8_meta_from_spec(spec: Mapping[str, Any]) -> BlockwiseFP8Meta:
    raw = spec.get("blockwise_fp8_meta")
    if not isinstance(raw, Mapping):
        raise ValueError("Blockwise FP8 packed module has no metadata object.")
    shape = raw.get("shape")
    weight_block = raw.get("weight_block")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError("Blockwise FP8 packed module has invalid shape metadata.")
    if not isinstance(weight_block, (list, tuple)) or len(weight_block) != 2:
        raise ValueError("Blockwise FP8 packed module has invalid block metadata.")
    return BlockwiseFP8Meta(
        shape=tuple(int(dim) for dim in shape),
        weight_block=tuple(int(dim) for dim in weight_block),
        activation_block=int(raw.get("activation_block", 0)),
    )


def get_compressed_weights_packed_state_quant_formats(
    manifest: Mapping[str, Any],
) -> frozenset[str]:
    """Return storage formats after validating every module declaration."""
    schema_version = _validate_manifest_header(manifest)
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError(
            "compressed_weights packed state manifest must include a non-empty modules object."
        )
    return frozenset(
        _module_quant_format(spec, schema_version=schema_version)
        for spec in modules.values()
        if isinstance(spec, Mapping)
    )


def _packed_tensor_key(module_name: str, tensor_name: str) -> str:
    return f"{module_name}.{tensor_name}" if module_name else tensor_name
