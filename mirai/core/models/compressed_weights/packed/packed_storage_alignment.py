"""Storage-aligned safetensors export and direct-read range planning."""

from __future__ import annotations

import json
import os
import shutil
import stat
import struct
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GDS_STORAGE_ALIGNMENT_BYTES = 4096
STORAGE_ALIGNMENT_METADATA_KEY = "mirai_storage_alignment_bytes"
_COPY_BUFFER_BYTES = 16 * 1024 * 1024
_MAX_HEADER_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class PackedStorageAlignment:
    alignment_bytes: int
    file_size: int


@dataclass(frozen=True)
class AlignedReadWindow:
    file_offset: int
    read_bytes: int
    payload_offset: int
    payload_bytes: int

    @property
    def overread_bytes(self) -> int:
        return self.read_bytes - self.payload_bytes


def save_safetensors_with_storage_alignment(
    save_file: Callable[..., None],
    tensors: Mapping[str, Any],
    path: Path,
    *,
    metadata: Mapping[str, str] | None,
    alignment_bytes: int,
) -> None:
    """Save once, then expand only the header to align the complete shard.

    The second pass is one bounded-buffer sequential read and write. Runtime
    streaming never writes to the packed artifact or creates an SSD cache.
    """

    alignment = _normalize_alignment(alignment_bytes)
    payload_metadata = {str(key): str(value) for key, value in dict(metadata or {}).items()}
    if alignment:
        payload_metadata[STORAGE_ALIGNMENT_METADATA_KEY] = str(alignment)
    save_file(dict(tensors), str(path), metadata=(payload_metadata or None))
    if alignment:
        _expand_header_to_file_alignment(path, alignment)
        read_safetensors_storage_alignment(path, required_alignment=alignment)


def read_safetensors_storage_alignment(
    path: Path,
    *,
    required_alignment: int = GDS_STORAGE_ALIGNMENT_BYTES,
) -> PackedStorageAlignment:
    """Validate the opt-in alignment marker and physical shard boundary."""

    required = _normalize_alignment(required_alignment)
    if required == 0:
        raise ValueError("A positive required storage alignment is required.")
    size, _header_size, header = _read_header(path)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Packed safetensors shard {path.name!r} has invalid metadata.")
    try:
        recorded = int(metadata.get(STORAGE_ALIGNMENT_METADATA_KEY, 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Packed safetensors shard {path.name!r} has an invalid storage alignment marker."
        ) from exc
    if recorded != required or size % required:
        raise ValueError(
            f"Packed shard {path.name!r} is not a validated {required}-byte-aligned "
            "artifact. Re-export it with memory.packed_stream_backend='gds'."
        )
    return PackedStorageAlignment(alignment_bytes=recorded, file_size=size)


def plan_aligned_read(
    *,
    offset: int,
    nbytes: int,
    file_size: int,
    alignment_bytes: int = GDS_STORAGE_ALIGNMENT_BYTES,
) -> AlignedReadWindow:
    """Expand an exact payload range to aligned file offset and byte count."""

    alignment = _normalize_alignment(alignment_bytes)
    start = int(offset)
    size = int(nbytes)
    total = int(file_size)
    if start < 0 or size <= 0 or total < 0 or start + size > total:
        raise ValueError("Direct-read payload range is outside its packed shard.")
    aligned_start = (start // alignment) * alignment
    aligned_end = ((start + size + alignment - 1) // alignment) * alignment
    if aligned_end > total:
        raise ValueError(
            "Aligned direct read would cross the packed shard boundary; re-export "
            "the artifact with GDS storage alignment."
        )
    return AlignedReadWindow(
        file_offset=aligned_start,
        read_bytes=aligned_end - aligned_start,
        payload_offset=start - aligned_start,
        payload_bytes=size,
    )


def _normalize_alignment(value: int) -> int:
    alignment = int(value)
    if alignment == 0:
        return 0
    if alignment != GDS_STORAGE_ALIGNMENT_BYTES:
        raise ValueError(
            "Packed storage alignment must be 0 or "
            f"{GDS_STORAGE_ALIGNMENT_BYTES} bytes."
        )
    return alignment


def _read_header(path: Path) -> tuple[int, int, dict[str, Any]]:
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
        raw_header = handle.read(header_size)
    header = json.loads(raw_header.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"Packed safetensors shard {path.name!r} has an invalid header.")
    return size, header_size, header


def _expand_header_to_file_alignment(path: Path, alignment: int) -> None:
    size, header_size, _header = _read_header(path)
    padding = (-size) % alignment
    if padding == 0:
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".aligning", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with path.open("rb") as source, temporary.open("wb") as destination:
            source.seek(8)
            raw_header = source.read(header_size)
            destination.write(struct.pack("<Q", header_size + padding))
            destination.write(raw_header)
            destination.write(b" " * padding)
            shutil.copyfileobj(source, destination, length=_COPY_BUFFER_BYTES)
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
