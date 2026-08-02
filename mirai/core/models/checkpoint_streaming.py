"""Bounded-memory loading for native sharded safetensors modules."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


TensorHandler = Callable[[Any], None]
KeyNormalizer = Callable[[str], str]


@dataclass(frozen=True)
class NativeCheckpointLoadReport:
    path: str
    matched_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def identity_checkpoint_key(key: str) -> str:
    return str(key)


def resolve_safetensor_files(
    component_dir: str | Path,
    *,
    index_names: Sequence[str],
    direct_names: Sequence[str],
) -> tuple[Path, ...]:
    root = Path(component_dir)
    for index_name in index_names:
        index_path = root / str(index_name)
        if not index_path.is_file():
            continue
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors index has no weight_map: {index_path}")
        names = sorted({str(name) for name in weight_map.values()})
        files = tuple(root / name for name in names)
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Safetensors index references missing shard files: " + ", ".join(missing)
            )
        return files
    for direct_name in direct_names:
        direct_path = root / str(direct_name)
        if direct_path.is_file():
            return (direct_path,)
    files = tuple(sorted(root.glob("*.safetensors")))
    return files


def iter_safetensor_keys(files: Iterable[str | Path]) -> Iterator[str]:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for native checkpoint loading.") from exc
    for file in files:
        with safe_open(str(Path(file)), framework="pt", device="cpu") as handle:
            yield from (str(key) for key in handle.keys())


def iter_safetensor_tensors(files: Iterable[str | Path]) -> Iterator[tuple[str, Any]]:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for native checkpoint loading.") from exc
    for file in files:
        with safe_open(str(Path(file)), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield str(key), handle.get_tensor(key)


def assign_module_tensor(module: Any, key: str, value: Any) -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for native checkpoint loading.")
    if "." in key:
        module_name, tensor_name = key.rsplit(".", 1)
        owner = module.get_submodule(module_name)
    else:
        owner = module
        tensor_name = key
    if tensor_name in owner._parameters:
        current = owner._parameters[tensor_name]
        if current is not None and tuple(current.shape) != tuple(value.shape):
            raise ValueError(
                f"Checkpoint tensor '{key}' has shape {tuple(value.shape)}, "
                f"expected {tuple(current.shape)}."
            )
        requires_grad = bool(getattr(current, "requires_grad", True))
        owner._parameters[tensor_name] = nn.Parameter(
            value.detach().clone(memory_format=torch.contiguous_format),
            requires_grad=requires_grad,
        )
        return
    if tensor_name in owner._buffers:
        current = owner._buffers[tensor_name]
        if current is not None and tuple(current.shape) != tuple(value.shape):
            raise ValueError(
                f"Checkpoint buffer '{key}' has shape {tuple(value.shape)}, "
                f"expected {tuple(current.shape)}."
            )
        owner._buffers[tensor_name] = value.detach().clone(
            memory_format=torch.contiguous_format
        )
        return
    raise KeyError(f"Native module has no tensor '{key}'.")


def load_module_safetensors(
    module: Any,
    files: Sequence[str | Path],
    *,
    component_path: str | Path,
    strict: bool,
    normalize_key: KeyNormalizer = identity_checkpoint_key,
    expected_keys: Iterable[str] | None = None,
    tensor_handlers: dict[str, TensorHandler] | None = None,
) -> NativeCheckpointLoadReport:
    target = set(str(key) for key in (expected_keys or module.state_dict().keys()))
    handlers = dict(tensor_handlers or {})
    matched: set[str] = set()
    unexpected: list[str] = []
    for raw_key, value in iter_safetensor_tensors(files):
        key = normalize_key(raw_key)
        if key not in target:
            unexpected.append(raw_key)
            continue
        handler = handlers.get(key)
        if handler is None:
            assign_module_tensor(module, key, value)
        else:
            handler(value)
        matched.add(key)
        del value
    missing = tuple(sorted(target - matched))
    unexpected_tuple = tuple(unexpected)
    if strict and (missing or unexpected_tuple):
        raise ValueError(
            "Native checkpoint did not exactly match the module "
            f"({len(missing)} missing keys, {len(unexpected_tuple)} unexpected keys)."
        )
    return NativeCheckpointLoadReport(
        path=str(component_path),
        matched_keys=len(matched),
        missing_keys=missing,
        unexpected_keys=unexpected_tuple,
    )
