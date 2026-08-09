"""Activation-space execution for MAGI-2 packed-weight LoRA adapters."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize


def _adapter(module: Any, tensor_name: str) -> tuple[torch.Tensor, Any]:
    if not parametrize.is_parametrized(module, tensor_name):
        raise RuntimeError(
            f"MAGI-2 activation-space LoRA requires parametrized '{tensor_name}'."
        )
    parametrizations = getattr(module.parametrizations, tensor_name)
    if len(parametrizations) != 1:
        raise RuntimeError(
            "MAGI-2 activation-space LoRA requires exactly one parametrization."
        )
    adapter = parametrizations[0]
    required = ("lora_a", "lora_b", "scale", "runtime_scale")
    if not all(hasattr(adapter, name) for name in required):
        raise TypeError(
            "MAGI-2 activation-space execution received an incompatible adapter."
        )
    return parametrizations.original, adapter


def _lora_linear(
    values: torch.Tensor,
    *,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    low_rank = F.linear(values.to(dtype=lora_a.dtype), lora_a)
    delta = F.linear(low_rank, lora_b)
    return delta.to(dtype=output_dtype) * float(scale)


def execute_grouped_linear_lora(
    module: Any,
    input: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    m_splits: list[int] | None = None,
    gather_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a frozen grouped linear plus LoRA without constructing ``B @ A``."""

    from mirai.vendors.magi2_preview.model.magi2_preview import (
        _m_splits_from,
        _maybe_gather,
        _torch_grouped_linear,
    )

    original, adapter = _adapter(module, "weight")
    groups = int(module.num_experts)
    rows = int(module.out_features)
    cols = int(module.in_features)
    weight = original.view(groups, rows, cols)
    bias = (
        module.bias.view(groups, rows) if module.bias is not None else None
    )
    factor = float(adapter.scale) * float(adapter.runtime_scale)
    if groups == 1:
        base = F.linear(input, weight[0], None if bias is None else bias[0])
        return base + _lora_linear(
            input,
            lora_a=adapter.lora_a,
            lora_b=adapter.lora_b,
            scale=factor,
            output_dtype=base.dtype,
        )

    splits = _m_splits_from(cu_seqlens, m_splits)
    ordered = _maybe_gather(input, gather_ids)
    base = _torch_grouped_linear(input, weight, bias, splits, gather_ids)
    outputs: list[torch.Tensor] = []
    start = 0
    lora_b = adapter.lora_b.view(groups, rows, adapter.rank)
    for group, split in enumerate(splits):
        stop = start + int(split)
        if stop > start:
            outputs.append(
                _lora_linear(
                    ordered[start:stop],
                    lora_a=adapter.lora_a,
                    lora_b=lora_b[group],
                    scale=factor,
                    output_dtype=base.dtype,
                )
            )
        start = stop
    delta = (
        torch.cat(outputs, dim=0)
        if outputs
        else base.new_empty((0, rows))
    )
    return base + delta


def execute_router_lora(module: Any, x_heads: torch.Tensor) -> torch.Tensor:
    """Compute MAGI-2 router logits from base and low-rank branches directly."""

    original, adapter = _adapter(module, "gate")
    heads = int(x_heads.shape[1])
    experts = int(module.num_experts)
    d_head = int(module.d_head)
    base_gate = original.view(heads, experts, d_head).float()
    values = x_heads.float()
    base = torch.einsum("shd,hed->hse", values, base_gate)
    low_rank = torch.einsum("shd,rd->shr", values, adapter.lora_a.float())
    lora_b = adapter.lora_b.view(heads, experts, adapter.rank).float()
    delta = torch.einsum("shr,her->hse", low_rank, lora_b)
    factor = float(adapter.scale) * float(adapter.runtime_scale)
    return base + delta * factor


def attach_magi2_lora_executor(module: Any, tensor_name: str) -> None:
    """Bind the family execution seam for one supported packed tensor."""

    if tensor_name == "weight" and all(
        hasattr(module, name)
        for name in ("num_experts", "out_features", "in_features")
    ):
        module._mirai_lora_executor = execute_grouped_linear_lora
        return
    if tensor_name == "gate" and all(
        hasattr(module, name) for name in ("num_experts", "d_head")
    ):
        module._mirai_router_lora_executor = execute_router_lora
        return
    raise TypeError(
        f"MAGI-2 has no activation-space LoRA executor for '{tensor_name}' on "
        f"{type(module).__name__}."
    )


__all__ = [
    "attach_magi2_lora_executor",
    "execute_grouped_linear_lora",
    "execute_router_lora",
]
