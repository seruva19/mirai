"""Tensor/list compatibility helpers."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def to_list(value: Any) -> list[float]:
    if is_torch_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def to_python_scalar(value: Any) -> float:
    if is_torch_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def to_jsonable(value: Any) -> Any:
    if is_torch_tensor(value):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return float(detached.item())
        return detached.reshape(-1).tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
