"""Checkpoint conversion for the released Agnes-Video-2.0-MoE weights.

The release stores one PyTorch archive as byte-split files.  This module joins
those files without buffering the archive, loads it with PyTorch's restricted
unpickler, and converts the tensors to native sharded safetensors.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterator, Mapping, Sequence
import uuid

import torch


AGNES_SOURCE_REVISION = "fc3971e378b53c766baa06898bfe84d7a56653ea"
AGNES_CHECKPOINT_NAME = "ckpt_best.pt"
CHECKPOINT_FORMAT = "mirai.agnes.safetensors"
CHECKPOINT_FORMAT_VERSION = 1

# Hugging Face's tree metadata for the pinned release.  The parts are literal
# byte ranges of one torch.save archive, not independently loadable shards.
RELEASED_PART_SIZES = (8 * 1024**3,) * 6 + (5_933_844_437,)
RELEASED_PART_SHA256 = (
    "a2c1693a4959f23a00f9d60adbb3f42c592f849c7972e79986e63e842d1820b9",
    "0c0df20f5f2d8af0aceb4f73282ea7f873568bd9a087048e7a75d0e7a14a2e82",
    "5ecc03171fb58fab5c51f32c45146687144864864bafd6ab3db54ea8f697afc4",
    "b4f272ce1adb7b58704a6665952654c469eb85e801067cabe008a59441accd44",
    "8379d724f9dff859ad7c5141cbf263567e6c78cfb84da4092e929ea0c1e086e7",
    "333ec43dd565064e327e53241bc2c4edcc78a7fed044086305b6e74a2e8fa315",
    "097820ca62b4652cdcd52866da340ab455ff843297e5542f0936fd2453122f05",
)

_PART_RE = re.compile(r"^(?P<base>.+)\.part-(?P<number>\d{3})$")


def checkpoint_parts(source: str | Path) -> tuple[Path, ...]:
    """Resolve and validate consecutively numbered byte-split checkpoint parts."""

    source_path = Path(source)
    if source_path.is_dir():
        candidates = tuple(source_path.glob(f"{AGNES_CHECKPOINT_NAME}.part-*"))
        expected_base = AGNES_CHECKPOINT_NAME
    else:
        candidates = tuple(source_path.parent.glob(f"{source_path.name}.part-*"))
        expected_base = source_path.name
    numbered: dict[int, Path] = {}
    for path in candidates:
        match = _PART_RE.match(path.name)
        if match is None or match.group("base") != expected_base:
            continue
        number = int(match.group("number"))
        if number in numbered:
            raise ValueError(f"Duplicate checkpoint part number {number:03d}.")
        numbered[number] = path
    if not numbered:
        raise FileNotFoundError(f"No byte-split checkpoint parts found for {source_path}.")
    expected_numbers = list(range(len(numbered)))
    if sorted(numbered) != expected_numbers:
        missing = sorted(set(range(max(numbered) + 1)) - set(numbered))
        raise ValueError(
            "Checkpoint parts must be numbered consecutively from 000; "
            f"missing: {', '.join(f'{number:03d}' for number in missing) or '000'}.")
    return tuple(numbered[number] for number in expected_numbers)


def validate_checkpoint_parts(
    parts: Sequence[str | Path],
    *,
    expected_sizes: Sequence[int] | None = None,
    expected_sha256: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Validate ordering and, when supplied, release size/hash metadata."""

    paths = tuple(Path(part) for part in parts)
    if not paths:
        raise ValueError("At least one checkpoint part is required.")
    for number, path in enumerate(paths):
        match = _PART_RE.match(path.name)
        if match is None or int(match.group("number")) != number:
            raise ValueError(
                f"Checkpoint part {path.name!r} is out of order; expected part-{number:03d}.")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint part does not exist: {path}")
    if expected_sizes is not None:
        sizes = tuple(int(size) for size in expected_sizes)
        if len(paths) != len(sizes):
            raise ValueError(f"Expected {len(sizes)} checkpoint parts, found {len(paths)}.")
        for path, expected in zip(paths, sizes):
            actual = path.stat().st_size
            if actual != expected:
                raise ValueError(
                    f"Checkpoint part {path.name} has size {actual}, expected {expected}.")
    if expected_sha256 is not None:
        hashes = tuple(str(value).lower() for value in expected_sha256)
        if len(paths) != len(hashes):
            raise ValueError(f"Expected {len(hashes)} checkpoint hashes, found {len(paths)} parts.")
        for path, expected in zip(paths, hashes):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise ValueError(
                    f"Checkpoint part {path.name} has SHA-256 {actual}, expected {expected}.")
    return paths


def assemble_checkpoint_parts(parts: Sequence[str | Path], destination: str | Path) -> Path:
    """Stream ordered parts into a new file and publish it atomically."""

    paths = validate_checkpoint_parts(parts)
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with pending.open("xb") as writer:
            for path in paths:
                with path.open("rb") as reader:
                    shutil.copyfileobj(reader, writer, length=8 * 1024**2)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(pending, output)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    return output


@contextmanager
def _assembled_source(parts: Sequence[Path]) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="mirai-agnes-checkpoint-") as temporary:
        assembled = Path(temporary) / AGNES_CHECKPOINT_NAME
        assemble_checkpoint_parts(parts, assembled)
        yield assembled


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Agnes checkpoint must contain a mapping, got {type(payload).__name__}.")
    if isinstance(payload.get("model"), Mapping):
        raw_state = payload["model"]
    elif isinstance(payload.get("state_dict"), Mapping):
        raw_state = payload["state_dict"]
    else:
        raise KeyError("Agnes checkpoint contains neither 'model' nor 'state_dict' tensor weights.")
    prefixed = [str(key).startswith("module.") for key in raw_state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Agnes state dict mixes 'module.'-prefixed and unprefixed keys.")
    strip_module = bool(prefixed and prefixed[0])
    state: dict[str, torch.Tensor] = {}
    for raw_key, value in raw_state.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Agnes state dict must map string keys to tensors only.")
        key = raw_key.removeprefix("module.") if strip_module else raw_key
        if key in state:
            raise ValueError(f"Duplicate Agnes tensor key after normalization: {key}")
        state[key] = value
    if not state:
        raise ValueError("Agnes checkpoint state dict is empty.")
    return state


def load_agnes_pt_checkpoint(
    source: str | Path,
    *,
    verify_released_parts: bool = True,
) -> dict[str, torch.Tensor]:
    """Load a direct or byte-split Agnes PyTorch checkpoint on CPU safely."""

    source_path = Path(source)
    if source_path.is_file():
        return _extract_state_dict(
            torch.load(source_path, map_location="cpu", weights_only=True)
        )
    parts = checkpoint_parts(source_path)
    validate_checkpoint_parts(
        parts,
        expected_sizes=RELEASED_PART_SIZES if verify_released_parts else None,
        expected_sha256=RELEASED_PART_SHA256 if verify_released_parts else None,
    )
    with _assembled_source(parts) as assembled:
        return _extract_state_dict(torch.load(assembled, map_location="cpu", weights_only=True))


def _tensor_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _partition_state_dict(
    state: Mapping[str, torch.Tensor], max_shard_size: int
) -> tuple[dict[str, torch.Tensor], ...]:
    if max_shard_size <= 0:
        raise ValueError("max_shard_size must be positive.")
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_size = 0
    for key in sorted(state):
        tensor = state[key]
        size = _tensor_size(tensor)
        if current and current_size + size > max_shard_size:
            shards.append(current)
            current = {}
            current_size = 0
        current[key] = tensor.detach().cpu().contiguous()
        current_size += size
    if current:
        shards.append(current)
    return tuple(shards)


def convert_agnes_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    max_shard_size: int = 4 * 1024**3,
    verify_released_parts: bool = True,
) -> Path:
    """Convert an Agnes torch archive into an atomic indexed safetensors directory."""

    from safetensors.torch import save_file

    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    try:
        state = load_agnes_pt_checkpoint(
            source, verify_released_parts=verify_released_parts
        )
        shards = _partition_state_dict(state, max_shard_size)
        weight_map: dict[str, str] = {}
        total = len(shards)
        for number, shard in enumerate(shards, start=1):
            name = f"model-{number:05d}-of-{total:05d}.safetensors"
            save_file(shard, str(staging / name))
            weight_map.update((key, name) for key in shard)
        source_path = Path(source)
        source_parts = (
            [part.name for part in checkpoint_parts(source_path)]
            if not source_path.is_file()
            else []
        )
        index = {
            "metadata": {
                "format": CHECKPOINT_FORMAT,
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "source_revision": AGNES_SOURCE_REVISION,
                "source_checkpoint": AGNES_CHECKPOINT_NAME,
                "source_parts": source_parts,
                "total_size": sum(_tensor_size(tensor) for tensor in state.values()),
            },
            "weight_map": weight_map,
        }
        index_path = staging / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "model.safetensors.index.json"


def _safe_shard_path(root: Path, raw_name: object) -> Path:
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("Agnes safetensors index contains an empty shard path.")
    relative = Path(raw_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ValueError(f"Agnes safetensors index contains unsafe shard path: {raw_name!r}.")
    return root / relative


def iter_agnes_safetensors(source: str | Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield converted Agnes tensors one shard at a time with strict index checks."""

    from safetensors.torch import load_file

    path = Path(source)
    if path.is_file() and path.suffix == ".safetensors":
        for key, tensor in load_file(str(path), device="cpu").items():
            yield key, tensor
        return
    index_path = path if path.name.endswith(".index.json") else path / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if (
        metadata.get("format") != CHECKPOINT_FORMAT
        or metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION
        or metadata.get("source_revision") != AGNES_SOURCE_REVISION
    ):
        raise ValueError("Agnes safetensors index lineage does not match this runtime.")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Agnes safetensors index has no weight_map: {index_path}")
    expected_by_shard: dict[str, set[str]] = {}
    for raw_key, raw_shard_name in weight_map.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError("Agnes safetensors index contains an invalid tensor key.")
        _safe_shard_path(index_path.parent, raw_shard_name)
        expected_by_shard.setdefault(str(raw_shard_name), set()).add(raw_key)
    for shard_name in sorted(expected_by_shard):
        shard_path = _safe_shard_path(index_path.parent, shard_name)
        if not shard_path.is_file():
            raise FileNotFoundError(f"Agnes safetensors shard is missing: {shard_path}")
        shard = load_file(str(shard_path), device="cpu")
        expected_keys = expected_by_shard[shard_name]
        actual_keys = set(shard)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Agnes safetensors shard {shard_name!r} does not match its weight_map "
                f"({len(missing)} missing keys, {len(unexpected)} unexpected keys)."
            )
        yield from shard.items()


def load_agnes_safetensors(source: str | Path) -> dict[str, torch.Tensor]:
    """Load a converted Agnes checkpoint; prefer the iterator for bounded memory."""

    state = dict(iter_agnes_safetensors(source))
    if not state:
        raise ValueError("Agnes safetensors checkpoint contains no tensors.")
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Agnes checkpoint parts to safetensors.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-shard-size", type=int, default=4 * 1024**3)
    parser.add_argument(
        "--skip-release-verification",
        action="store_false",
        dest="verify_released_parts",
        help="accept non-release split parts (intended only for local test artifacts)",
    )
    args = parser.parse_args(argv)
    index = convert_agnes_checkpoint(
        args.source,
        args.destination,
        max_shard_size=args.max_shard_size,
        verify_released_parts=args.verify_released_parts,
    )
    print(index)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
