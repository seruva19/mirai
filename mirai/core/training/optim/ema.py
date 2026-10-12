"""EMA utilities for adapter weights."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for EMA: {exc}")


def init_ema_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            out[key] = value.detach().cpu().float().clone()
        elif key == "__mirai_adapter_lineage__" and isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().clone()
    return out


def update_ema_state(
    ema_state: dict[str, torch.Tensor],
    live_state: dict[str, Any],
    *,
    decay: float,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = dict(ema_state)
    lineage_key = "__mirai_adapter_lineage__"
    previous_lineage = ema_state.get(lineage_key)
    current_lineage = live_state.get(lineage_key)
    if previous_lineage is not None or current_lineage is not None:
        if (not isinstance(previous_lineage, torch.Tensor)
                or not isinstance(current_lineage, torch.Tensor)
                or previous_lineage.dtype != current_lineage.dtype
                or previous_lineage.shape != current_lineage.shape
                or not torch.equal(previous_lineage.cpu(), current_lineage.detach().cpu())):
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
