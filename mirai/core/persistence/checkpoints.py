"""Checkpoint save/load helpers for the thin vertical slice."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.persistence.migrations import migrate_checkpoint_payload

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for checkpointing: {exc}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _checksum_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


SHARDED_CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_SHARD_THRESHOLD_BYTES = 512 * 1024 * 1024
DEFAULT_SHARD_SIZE_BYTES = 512 * 1024 * 1024
_TENSOR_REF_KEY = "__mirai_tensor_ref__"
_SHARDED_STUB_KEY = "__mirai_sharded_checkpoint__"


@dataclass(frozen=True)
class CheckpointPolicy:
    """When/where/how a periodic training checkpoint is written.

    Owns the save interval, ``step_{N}.pt`` naming,
    and the sharded-write byte thresholds. Built from existing config keys
    (``logging.save_every_n_steps`` plus the module shard-size defaults); it
    introduces no new config surface.
    """

    save_every_n_steps: int = 1
    shard_threshold_bytes: int = DEFAULT_SHARD_THRESHOLD_BYTES
    shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES

    @classmethod
    def from_save_interval(cls, save_every_n_steps: int) -> "CheckpointPolicy":
        return cls(save_every_n_steps=max(1, int(save_every_n_steps)))

    def should_save(self, step: int) -> bool:
        return int(step) % self.save_every_n_steps == 0

    def checkpoint_path(self, output_dir: str | Path, step: int) -> Path:
        return Path(output_dir) / f"step_{step}.pt"

    def save(self, path: str | Path, payload: dict[str, Any]) -> Path:
        return save_checkpoint(
            path,
            payload,
            shard_threshold_bytes=self.shard_threshold_bytes,
            shard_size_bytes=self.shard_size_bytes,
        )


def _immutable_cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True).contiguous()
    if isinstance(value, dict):
        return {key: _immutable_cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_immutable_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_immutable_cpu_snapshot(item) for item in value)
    return deepcopy(value)


class AsyncCheckpointWriter:
    """Single-flight checkpoint serialization from an immutable CPU snapshot."""

    def __init__(self, *, max_snapshot_bytes: int = 0) -> None:
        if int(max_snapshot_bytes) < 0:
            raise ValueError("max_snapshot_bytes must be >= 0.")
        self.max_snapshot_bytes = int(max_snapshot_bytes)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mirai-ckpt")
        self._pending: Future[Path] | None = None

    def wait(self) -> Path | None:
        pending = self._pending
        if pending is None:
            return None
        self._pending = None
        return pending.result()

    def submit(
        self,
        policy: CheckpointPolicy,
        path: str | Path,
        payload: dict[str, Any],
    ) -> None:
        self.wait()
        tensor_bytes = _payload_tensor_bytes(payload)
        if self.max_snapshot_bytes and tensor_bytes > self.max_snapshot_bytes:
            raise MemoryError(
                "Checkpoint snapshot requires "
                f"{tensor_bytes} bytes, exceeding the configured "
                f"{self.max_snapshot_bytes}-byte bound."
            )
        snapshot = _immutable_cpu_snapshot(payload)
        self._pending = self._executor.submit(policy.save, path, snapshot)

    def close(self) -> Path | None:
        try:
            return self.wait()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)


def save_checkpoint(
    path: str | Path,
    payload: dict[str, Any],
    *,
    shard_threshold_bytes: int = DEFAULT_SHARD_THRESHOLD_BYTES,
    shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES,
) -> Path:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if _payload_tensor_bytes(payload) >= int(shard_threshold_bytes):
        return _save_sharded_checkpoint(
            ckpt_path,
            payload,
            shard_size_bytes=max(1, int(shard_size_bytes)),
        )
    tmp_path = ckpt_path.with_name(ckpt_path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)
    digest = _sha256_file(ckpt_path)
    _checksum_path(ckpt_path).write_text(digest + "\n", encoding="utf-8")
    return ckpt_path


def load_checkpoint(path: str | Path, verify_checksum: bool = True) -> dict[str, Any]:
    ckpt_path = Path(path)
    checksum = _checksum_path(ckpt_path)
    if verify_checksum and checksum.exists():
        expected = checksum.read_text(encoding="utf-8").strip()
        observed = _sha256_file(ckpt_path)
        if expected != observed:
            raise ValueError(
                "Checkpoint checksum mismatch for "
                f"{ckpt_path} (expected {expected}, got {observed})."
            )
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and _SHARDED_STUB_KEY in payload:
        payload = _load_sharded_checkpoint(
            ckpt_path,
            payload,
            verify_checksum=verify_checksum,
        )
    return migrate_checkpoint_payload(payload)


def alias_checkpoint(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_name(destination_path.name + ".tmp-link")
    if temp_path.exists():
        temp_path.unlink()
    try:
        os.link(source_path, temp_path)
    except OSError:
        shutil.copyfile(source_path, temp_path)
    os.replace(temp_path, destination_path)
    source_checksum = _checksum_path(source_path)
    digest = (
        source_checksum.read_text(encoding="utf-8").strip()
        if source_checksum.exists()
        else _sha256_file(source_path)
    )
    _checksum_path(destination_path).write_text(digest + "\n", encoding="utf-8")
    return destination_path


def _payload_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, dict):
        return sum(_payload_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_tensor_bytes(item) for item in value)
    return 0


def _extract_tensors(value: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        key = f"tensor_{len(tensors):08d}"
        tensors[key] = value
        return {_TENSOR_REF_KEY: key}
    if isinstance(value, dict):
        return {key: _extract_tensors(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_extract_tensors(item, tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_extract_tensors(item, tensors) for item in value)
    return value


def _restore_tensors(value: Any, tensors: dict[str, torch.Tensor]) -> Any:
    if isinstance(value, dict):
        if set(value) == {_TENSOR_REF_KEY}:
            key = str(value[_TENSOR_REF_KEY])
            if key not in tensors:
                raise ValueError(f"Sharded checkpoint is missing tensor {key!r}.")
            return tensors[key]
        return {key: _restore_tensors(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_tensors(item, tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_tensors(item, tensors) for item in value)
    return value


def _save_sharded_checkpoint(
    ckpt_path: Path,
    payload: dict[str, Any],
    *,
    shard_size_bytes: int,
) -> Path:
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for sharded checkpoints.") from exc

    old_bundle = _read_existing_bundle_path(ckpt_path)
    bundle_name = f"{ckpt_path.name}.bundle-{uuid.uuid4().hex}"
    temp_dir = Path(
        tempfile.mkdtemp(prefix=bundle_name + ".tmp-", dir=str(ckpt_path.parent))
    )
    final_dir = ckpt_path.parent / bundle_name
    published = False
    tmp_path = ckpt_path.with_name(ckpt_path.name + ".tmp")
    try:
        tensors: dict[str, torch.Tensor] = {}
        skeleton = _extract_tensors(payload, tensors)
        metadata_path = temp_dir / "metadata.pt"
        torch.save(skeleton, metadata_path)
        shard_records: list[dict[str, Any]] = []
        pending: dict[str, torch.Tensor] = {}
        pending_bytes = 0

        def flush() -> None:
            nonlocal pending, pending_bytes
            if not pending:
                return
            shard_name = f"tensors-{len(shard_records) + 1:05d}.safetensors"
            shard_path = temp_dir / shard_name
            save_file(pending, str(shard_path))
            shard_records.append(
                {
                    "file": shard_name,
                    "sha256": _sha256_file(shard_path),
                    "tensor_keys": list(pending),
                }
            )
            pending = {}
            pending_bytes = 0

        for key, tensor in tensors.items():
            # Optimizers commonly expose distinct tensor views over shared storage.
            # Safetensors rejects such aliases, so each persisted entry owns storage.
            cpu_tensor = tensor.detach().to(device="cpu").contiguous().clone()
            tensor_bytes = int(cpu_tensor.numel()) * int(cpu_tensor.element_size())
            if pending and pending_bytes + tensor_bytes > shard_size_bytes:
                flush()
            pending[key] = cpu_tensor
            pending_bytes += tensor_bytes
            if pending_bytes >= shard_size_bytes:
                flush()
        flush()
        manifest = {
            "schema_version": SHARDED_CHECKPOINT_SCHEMA_VERSION,
            "format": "mirai.sharded_checkpoint",
            "metadata_file": metadata_path.name,
            "metadata_sha256": _sha256_file(metadata_path),
            "tensor_count": len(tensors),
            "total_tensor_bytes": _payload_tensor_bytes(payload),
            "shards": shard_records,
        }
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_dir, final_dir)
        stub = {
            _SHARDED_STUB_KEY: SHARDED_CHECKPOINT_SCHEMA_VERSION,
            "bundle": bundle_name,
            "manifest_sha256": _sha256_file(final_dir / "manifest.json"),
        }
        torch.save(stub, tmp_path)
        os.replace(tmp_path, ckpt_path)
        published = True
        _checksum_path(ckpt_path).write_text(
            _sha256_file(ckpt_path) + "\n", encoding="utf-8"
        )
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if final_dir.exists() and not published:
            _remove_owned_bundle(final_dir, checkpoint_parent=ckpt_path.parent)
        raise
    if old_bundle is not None and old_bundle != final_dir:
        _remove_owned_bundle(old_bundle, checkpoint_parent=ckpt_path.parent)
    return ckpt_path


def _load_sharded_checkpoint(
    ckpt_path: Path,
    stub: dict[str, Any],
    *,
    verify_checksum: bool,
) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for sharded checkpoints.") from exc
    bundle = _resolve_bundle_path(ckpt_path, str(stub.get("bundle", "")))
    manifest_path = bundle / "manifest.json"
    if verify_checksum:
        expected_manifest = str(stub.get("manifest_sha256", ""))
        observed_manifest = _sha256_file(manifest_path)
        if expected_manifest != observed_manifest:
            raise ValueError("Sharded checkpoint manifest checksum mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "mirai.sharded_checkpoint"
        or int(manifest.get("schema_version", 0)) != SHARDED_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported sharded checkpoint manifest.")
    metadata_path = bundle / str(manifest["metadata_file"])
    if verify_checksum and _sha256_file(metadata_path) != str(manifest["metadata_sha256"]):
        raise ValueError("Sharded checkpoint metadata checksum mismatch.")
    skeleton = torch.load(metadata_path, map_location="cpu")
    tensors: dict[str, torch.Tensor] = {}
    for record in manifest.get("shards", []):
        shard_path = bundle / str(record["file"])
        if verify_checksum and _sha256_file(shard_path) != str(record["sha256"]):
            raise ValueError(f"Sharded checkpoint tensor checksum mismatch: {shard_path.name}.")
        tensors.update(load_file(str(shard_path), device="cpu"))
    restored = _restore_tensors(skeleton, tensors)
    if not isinstance(restored, dict):
        raise ValueError("Sharded checkpoint root payload must be a dictionary.")
    return restored


def _read_existing_bundle_path(ckpt_path: Path) -> Path | None:
    if not ckpt_path.exists():
        return None
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict) or _SHARDED_STUB_KEY not in payload:
        return None
    try:
        return _resolve_bundle_path(ckpt_path, str(payload.get("bundle", "")))
    except ValueError:
        return None


def _resolve_bundle_path(ckpt_path: Path, bundle_name: str) -> Path:
    if not bundle_name or Path(bundle_name).name != bundle_name:
        raise ValueError("Invalid sharded checkpoint bundle path.")
    bundle = (ckpt_path.parent / bundle_name).resolve()
    if bundle.parent != ckpt_path.parent.resolve():
        raise ValueError("Sharded checkpoint bundle escapes checkpoint directory.")
    if not bundle.is_dir():
        raise FileNotFoundError(bundle)
    return bundle


def _remove_owned_bundle(path: Path, *, checkpoint_parent: Path) -> None:
    resolved = path.resolve()
    parent = checkpoint_parent.resolve()
    if resolved.parent != parent or ".bundle-" not in resolved.name:
        raise ValueError(f"Refusing to remove unowned checkpoint bundle: {resolved}")
    shutil.rmtree(resolved)
