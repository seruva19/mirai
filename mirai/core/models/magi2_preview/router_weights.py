"""Symmetric INT8 storage for MAGI-2's head-local router projection."""

from __future__ import annotations

import torch
from torch.nn.utils import parametrize

from mirai.core.models.compressed_weights.quantization.quant import _quantize_weight
from mirai.core.models.magi2_preview.quantized_experts import iter_magi2_moe_layers


def _router_weight(layer):
    return (layer._mirai_router_weight_int8.float() * layer._mirai_router_weight_scale).view(
        layer.num_heads, layer.num_experts, layer.d_head,
    )


class _Int8RouterLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, layer):
        ctx.layer = layer
        ctx.input_dtype = values.dtype
        with torch.autocast(device_type=values.device.type, enabled=False):
            return torch.einsum("shd,hed->hse", values.float(), _router_weight(layer))

    @staticmethod
    def backward(ctx, grad_output):
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            dx = torch.einsum("hse,hed->shd", grad_output.float(), _router_weight(ctx.layer))
        return dx.to(ctx.input_dtype), None


def execute_int8_router(layer, values):
    return _Int8RouterLogits.apply(values, layer)


@torch.no_grad()
def pack_magi2_router_weights(transformer) -> None:
    layers = tuple(iter_magi2_moe_layers(transformer))
    if not layers:
        raise ValueError("MAGI-2 router packing matched no routed expert layers.")
    pending = []
    for name, layer in layers:
        if layer.ep_size != 1:
            raise ValueError("MAGI-2 INT8 router storage requires single-device routing.")
        if hasattr(layer, "_mirai_router_weight_int8"):
            continue
        if parametrize.is_parametrized(layer, "gate"):
            gate = layer.parametrizations.gate.original
        else:
            gate = layer._parameters["gate"]
        if not torch.isfinite(gate).all():
            raise ValueError(f"MAGI-2 router {name!r} contains non-finite weights.")
        # The original parameter can still be trainable before inference setup
        # freezes the model. A configured LoRA leaves only its factors trainable.
        codes, scale, _ = _quantize_weight(gate, group_size=0)
        pending.append((layer, gate, codes, scale))
    for layer, gate, codes, scale in pending:
        layer.register_buffer("_mirai_router_weight_int8", codes, persistent=False)
        layer.register_buffer("_mirai_router_weight_scale", scale, persistent=False)
        layer._mirai_int8_router_executor = execute_int8_router
        gate.requires_grad_(False)
        gate.data = gate.new_empty(0)
