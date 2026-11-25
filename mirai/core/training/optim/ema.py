"""EMA utilities for adapter weights."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for EMA: {exc}")


_LINEAGE_KEYS = ("__mirai_adapter_lineage__", "__mirai_magi2_pruning_digest__")


def init_ema_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            out[key] = value.detach().cpu().float().clone()
        elif key in _LINEAGE_KEYS and isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().clone()
    return out


def update_ema_state(
    ema_state: dict[str, torch.Tensor],
    live_state: dict[str, Any],
    *,
    decay: float,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = dict(ema_state)
    for key in _LINEAGE_KEYS:
        previous = ema_state.get(key)
        current = live_state.get(key)
        if previous is None and current is None:
            continue
        if (not isinstance(previous, torch.Tensor)
                or not isinstance(current, torch.Tensor)
                or previous.dtype != current.dtype
                or previous.shape != current.shape
                or not torch.equal(previous.cpu(), current.detach().cpu())):
            raise ValueError("Adapter lineage changed during EMA tracking.")
    decay = float(decay)
    one_minus = 1.0 - decay
    for key, value in live_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if not torch.is_floating_point(value):
            continue
        live = value.detach().cpu().float()
        prev = out.get(key)
        if prev is None:
            out[key] = live.clone()
        else:
            out[key] = (prev * decay) + (live * one_minus)
    return out
