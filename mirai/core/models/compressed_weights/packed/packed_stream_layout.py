"""Validated byte-region inventory for safetensors packed-state streaming."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_MAX_HEADER_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class PackedTensorRegion:
    shard: str
    dtype_name: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def load_packed_regions(
    path: Path,
) -> tuple[dict[str, PackedTensorRegion], dict[str, Path]]:
    if not path.is_file():
        raise FileNotFoundError(f"Packed-state artifact {str(path)!r} was not found.")
    if path.name.endswith(".index.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_map = payload.get("weight_map", {})
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError("Packed compressed_weights index has no weight_map.")
        weight_map = {str(key): str(value) for key, value in raw_map.items()}
    else:
        weight_map = {}

    shard_names = sorted(set(weight_map.values())) if weight_map else [path.name]
    shard_paths = {name: path.parent / name for name in shard_names}
    shard_regions = {
        name: _parse_safetensors_header(shard_path, name)
        for name, shard_path in shard_paths.items()
    }
    if not weight_map:
        weight_map = {key: path.name for key in shard_regions[path.name]}

    regions: dict[str, PackedTensorRegion] = {}
    for key, shard_name in weight_map.items():
        try:
            regions[key] = shard_regions[shard_name][key]
        except KeyError as exc:
            raise ValueError(
                f"Packed-state index maps {key!r} to {shard_name!r}, "
                "but that tensor is absent from the shard."
            ) from exc
    return regions, shard_paths


def torch_dtype(name: str) -> Any:
    attributes = {
        "BOOL": "bool",
        "U8": "uint8",
        "I8": "int8",
        "F8_E4M3": "float8_e4m3fn",
        "F8_E4M3FN": "float8_e4m3fn",
        "F8_E5M2": "float8_e5m2",
        "I16": "int16",
        "U16": "uint16",
        "F16": "float16",
        "BF16": "bfloat16",
        "I32": "int32",
        "U32": "uint32",
        "F32": "float32",
        "I64": "int64",
        "U64": "uint64",
        "F64": "float64",
    }
    attribute = attributes.get(str(name).upper())
    dtype = getattr(torch, attribute, None) if attribute is not None else None
    if dtype is None:
        raise ValueError(
            f"Current torch build cannot materialize safetensors dtype {name!r}."
        )
    return dtype


def _parse_safetensors_header(
    path: Path, shard_name: str
) -> dict[str, PackedTensorRegion]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Packed safetensors shard {path.name!r} has no header.")
        header_size = int(struct.unpack("<Q", prefix)[0])
        if header_size <= 0 or header_size > min(_MAX_HEADER_BYTES, size - 8):
            raise ValueError(
                f"Packed safetensors shard {path.name!r} has an invalid header size."
            )
        header = json.loads(handle.read(header_size).decode("utf-8"))
    data_start = 8 + header_size
    regions: dict[str, PackedTensorRegion] = {}
    for key, raw in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"Packed tensor metadata for {key!r} is invalid.")
        dtype_name = str(raw.get("dtype", ""))
        shape = tuple(int(dim) for dim in raw.get("shape", ()))
        offsets = raw.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or int(offsets[0]) < 0
            or int(offsets[1]) < int(offsets[0])
        ):
            raise ValueError(f"Packed tensor offsets for {key!r} are invalid.")
        start, end = (int(offsets[0]), int(offsets[1]))
        nbytes = end - start
        expected = _shape_elements(shape) * _dtype_nbytes(dtype_name)
        if nbytes != expected or data_start + end > size:
            raise ValueError(
                f"Packed tensor region for {key!r} does not match its shape/dtype."
            )
        regions[str(key)] = PackedTensorRegion(
            shard=str(shard_name),
            dtype_name=dtype_name,
            shape=shape,
            offset=data_start + start,
            nbytes=nbytes,
        )
    return regions


def _shape_elements(shape: Sequence[int]) -> int:
    count = 1
    for dim in shape:
        if int(dim) < 0:
            raise ValueError("Packed tensor shapes cannot contain negative dimensions.")
        count *= int(dim)
    return count


def _dtype_nbytes(name: str) -> int:
    sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E4M3FN": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    try:
        return sizes[str(name).upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported packed safetensors dtype {name!r}.") from exc
