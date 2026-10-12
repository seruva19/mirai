"""Shared mechanics for Agnes release probes."""

from __future__ import annotations

import hashlib
import json
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mirai.core.models.agnes_video.checkpoints import (
    AGNES_SOURCE_REVISION,
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    _safe_shard_path,
    checkpoint_parts,
)
from mirai.vendors.agnes_video.dit import AgnesDiTConfig


class ContractUnavailable(RuntimeError):
    """The configured release probe cannot run in the current environment."""


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise ContractUnavailable("Agnes full-release evidence requires CUDA.")
    return torch.device("cuda")


def checkpoint_digest(source: str | Path) -> str:
    digest = hashlib.sha256()
    source_path = Path(source)
    if source_path.is_file() and source_path.suffix in {".pt", ".safetensors"}:
        parts = (source_path,)
    else:
        index_path = source_path if source_path.name.endswith(".index.json") else source_path / "model.safetensors.index.json"
        if index_path.is_file():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
                metadata = payload["metadata"]
                if (
                    metadata.get("format") != CHECKPOINT_FORMAT
                    or metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION
                    or metadata.get("source_revision") != AGNES_SOURCE_REVISION
                ):
                    raise ValueError("checkpoint lineage mismatch")
                shard_names = sorted(set(payload["weight_map"].values()))
                parts = (index_path, *(_safe_shard_path(index_path.parent, name) for name in shard_names))
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ContractUnavailable(f"Agnes checkpoint index is invalid: {index_path}") from exc
        else:
            try:
                parts = checkpoint_parts(source)
            except (FileNotFoundError, ValueError) as exc:
                raise ContractUnavailable(f"Agnes checkpoint is unavailable: {source}") from exc
    if not parts:
        raise ContractUnavailable(f"Agnes checkpoint is missing: {source}")
    for path in parts:
        if not path.is_file():
            raise ContractUnavailable(f"Agnes checkpoint part is missing: {path}")
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def runtime_identity(pipeline: Any, device: torch.device) -> dict[str, Any]:
    architecture = asdict(pipeline.runtime_config)
    released_architecture = asdict(AgnesDiTConfig())
    compute_dtype = architecture.pop("params_dtype", None)
    released_architecture.pop("params_dtype", None)
    parameters = list(pipeline.transformer.parameters())
    materialized = bool(parameters) and all(not parameter.is_meta for parameter in parameters)
    configured_source = str(getattr(pipeline.model_config, "path", "")).strip()
    checkpoint_constructed = bool(
        materialized
        and configured_source
        and architecture == released_architecture
        and getattr(pipeline, "_adapter_base_fingerprint", None) is not None
    )
    return {
        "family": "agnes-video",
        "pipeline_class": type(pipeline).__name__,
        "transformer_class": type(pipeline.transformer).__name__,
        "source_revision": AGNES_SOURCE_REVISION,
        "device_type": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "architecture": json.loads(json.dumps(architecture, default=str)),
        "compute_dtype": str(compute_dtype),
        "checkpoint_constructed": checkpoint_constructed,
        "configured_checkpoint_source": configured_source,
        "parameter_tensors": len(parameters),
        "all_parameters_materialized": materialized,
    }


def release_training_session(session: Any) -> None:
    """Break heavyweight session references before another full model is built."""
    session.close_callbacks()
    callbacks = getattr(session, "callbacks", None)
    if isinstance(callbacks, list):
        callbacks.clear()
    for field in (
        "trainer", "optimizer", "optimizer_result", "scheduler", "params",
        "named_params", "train_records", "val_records", "temporal_groups",
        "build_ckpt_payload", "live_control", "event_bus",
    ):
        if hasattr(session, field):
            setattr(session, field, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unavailable_reason(exc: BaseException) -> str | None:
    """Classify absent local runtime resources without masking computation faults."""
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, (FileNotFoundError, ModuleNotFoundError)):
            return f"{type(current).__name__}: {current}"
        current = current.__cause__ or current.__context__
    message = str(exc).lower()
    markers = ("assets are missing", "checkpoint is missing", "no byte-split checkpoint", "requires cuda")
    if any(marker in message for marker in markers):
        return str(exc)
    return None


def write_report(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(tensor_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(tensor_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value
