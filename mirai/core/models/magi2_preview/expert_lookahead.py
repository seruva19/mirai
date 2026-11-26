"""Speculative host reads for future MAGI-2 routed expert layers."""

from __future__ import annotations

import torch

from .quantized_experts import (
    MAGI2_NF4_GROUP_FIELDS,
    Magi2PackedExpertStore,
    magi2_expert_store,
)


def _warm_residual_experts(hidden, streams, future_layers, max_experts):
    if hidden.shape[0] == 0:
        return
    stride = max(1, (hidden.shape[0] + 31) // 32)
    sampled = hidden[::stride].reshape(-1, streams, hidden.shape[-1] // streams)
    residual = sampled.mean(dim=1)
    requests = {}
    reader = None
    for layer in future_layers:
        experts = getattr(layer.mlp, "moe_mlp", None)
        if experts is None or experts.router.expert_bias.device != hidden.device:
            continue
        store = magi2_expert_store(experts)
        view = getattr(store, "_packed_range_reader", None)
        if view is None:
            continue
        if reader is not None and reader is not view.reader:
            raise RuntimeError("MAGI-2 expert lookahead requires a shared range reader.")
        reader = view.reader
        _, indices = experts._route(residual.reshape(-1, experts.num_heads, experts.d_head), observe=False)
        heads = torch.arange(indices.shape[0], device=indices.device)[:, None, None]
        groups = (indices + heads * experts.num_experts).flatten()
        aliases = getattr(experts, "_mirai_expert_alias_indices", None)
        if aliases is not None:
            groups = aliases.to(groups.device).index_select(0, groups)
        ids, counts = groups.unique(sorted=True, return_counts=True)
        ids = ids[counts.argsort(descending=True, stable=True)[:max_experts]].tolist()
        keys = tuple(view)
        if not isinstance(store, Magi2PackedExpertStore):
            keys = tuple(key for key in keys if key.endswith(tuple(f"_{suffix}" for suffix in MAGI2_NF4_GROUP_FIELDS)))
        requests.update({view.prefix + key: ids for key in keys})
    if reader is not None:
        reader.warm_expert_cache(requests)


def attach_magi2_expert_lookahead(transformer, *, depth: int, max_experts: int):
    layers = tuple(transformer.block.layers)

    @torch.no_grad()
    def prefetch(index, hidden):
        future = layers[index + 1:index + 1 + depth]
        if future:
            _warm_residual_experts(hidden, layers[index].config_mhc.num_stream, future, max_experts)

    transformer.block._mirai_expert_lookahead = prefetch
