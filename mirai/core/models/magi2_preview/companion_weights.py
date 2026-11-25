"""Packed frozen base projections for MAGI-2 Preview attention and MLPs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parametrize

from mirai.core.models.compressed_weights.execution.linear import CompressedLinear


MAGI2_COMPANION_FORMATS = ("none", "fp8", "nf4")


def execute_companion_linear(module, input, cu_seqlens=None, m_splits=None, gather_ids=None):
    from mirai.vendors.magi2_preview.model.magi2_preview import _m_splits_from, _maybe_gather

    projections = module._mirai_companion_projections
    bias = None if module.bias is None else module.bias.view(module.num_experts, module.out_features)

    def run(index, values):
        output = projections[index].forward_weight_only(values)
        return output if bias is None else output + bias[index].to(output.dtype)

    if module.num_experts == 1:
        return run(0, input)
    ordered = _maybe_gather(input, gather_ids)
    outputs = []
    start = 0
    for index, count in enumerate(_m_splits_from(cu_seqlens, m_splits)):
        if count > 0:
            outputs.append(run(index, ordered[start:start + count]))
        start += count
    return torch.cat(outputs, dim=0) if outputs else ordered.new_empty((0, module.out_features))


@torch.no_grad()
def pack_magi2_companion_weights(transformer, *, quant_format: str, blocksize: int) -> tuple[nn.Module, ...]:
    from mirai.vendors.magi2_preview.model.magi2_preview import CustomGroupedLinear

    if quant_format == "none":
        return ()
    if quant_format not in MAGI2_COMPANION_FORMATS:
        raise ValueError(f"Unsupported MAGI-2 companion format {quant_format!r}.")
    packed = []
    for layer in transformer.block.layers:
        for root in (layer.attention, layer.mlp):
            targets = [module for module in root.modules() if isinstance(module, CustomGroupedLinear)]
            for module in targets:
                if hasattr(module, "_mirai_companion_projections"):
                    raise ValueError("MAGI-2 companion weights are already packed.")
                if parametrize.is_parametrized(module, "weight"):
                    raise ValueError("Pack MAGI-2 companion weights before attaching adapters.")
                source = module.weight.detach()
                projections = nn.ModuleList()
                for weight in source.split(module.out_features, dim=0):
                    projection = CompressedLinear.from_empty(
                        in_features=module.in_features, out_features=module.out_features,
                        group_size=0, has_bias=False, quant_format=quant_format,
                        nf4_blocksize=blocksize,
                    )
                    projection.load_dense_weight(source=weight)
                    projections.append(projection)
                module.add_module("_mirai_companion_projections", projections)
                # The empty original keeps the existing adapter state keys.
                module.weight = nn.Parameter(source.new_empty(0), requires_grad=False)
                module._mirai_lora_executor = execute_companion_linear
                packed.append(module)
    if not packed:
        raise ValueError("MAGI-2 companion packing matched no grouped linear projections.")
    return tuple(packed)
