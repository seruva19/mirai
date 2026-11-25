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


def _companion_linears(transformer):
    from mirai.vendors.magi2_preview.model.magi2_preview import CustomGroupedLinear

    for layer in transformer.block.layers:
        for root in (layer.attention, layer.mlp):
            yield from (module for module in root.modules() if isinstance(module, CustomGroupedLinear))


@torch.no_grad()
def _pack_companion_linear(module, source, *, quant_format, blocksize):
    if hasattr(module, "_mirai_companion_projections"):
        raise ValueError("MAGI-2 companion weights are already packed.")
    if parametrize.is_parametrized(module, "weight"):
        raise ValueError("Pack MAGI-2 companion weights before attaching adapters.")
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


def _companion_targets(transformer, quant_format):
    if quant_format not in MAGI2_COMPANION_FORMATS:
        raise ValueError(f"Unsupported MAGI-2 companion format {quant_format!r}.")
    targets = tuple(_companion_linears(transformer)) if quant_format != "none" else ()
    if quant_format != "none" and not targets:
        raise ValueError("MAGI-2 companion packing matched no grouped linear projections.")
    return targets


def pack_magi2_companion_weights(transformer, *, quant_format: str, blocksize: int) -> tuple[nn.Module, ...]:
    targets = _companion_targets(transformer, quant_format)
    for module in targets:
        _pack_companion_linear(module, module.weight.detach(), quant_format=quant_format, blocksize=blocksize)
    return targets


def stream_pack_magi2_companion_weights(
    transformer, *, checkpoint_dir: str, quant_format: str, blocksize: int,
) -> tuple[tuple[nn.Module, ...], set[str]]:
    from safetensors import safe_open

    from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import read_safetensors_weight_map

    targets = _companion_targets(transformer, quant_format)
    if not targets:
        return (), set()
    names = {module: name for name, module in transformer.named_modules()}
    weight_map = read_safetensors_weight_map(checkpoint_dir)
    weight_names = {module: f"{names[module]}.weight" for module in targets}
    for weight_name in weight_names.values():
        if weight_name not in weight_map:
            raise ValueError(f"MAGI-2 checkpoint is missing companion weight {weight_name!r}.")
    preloaded = set()
    for module, weight_name in weight_names.items():
        prefix = f"{names[module]}."
        with safe_open(weight_map[weight_name], framework="pt", device="cpu") as handle:
            source = handle.get_tensor(weight_name)
        if source.shape != module.weight.shape or not source.is_floating_point():
            raise ValueError(f"MAGI-2 checkpoint companion weight {weight_name!r} has incompatible shape or dtype.")
        _pack_companion_linear(
            module, source.to(dtype=module.weight.dtype), quant_format=quant_format, blocksize=blocksize,
        )
        del source
        preloaded.add(weight_name)
        preloaded.update(
            prefix + "_mirai_companion_projections." + key
            for key in module._mirai_companion_projections.state_dict()
        )
    return targets, preloaded
