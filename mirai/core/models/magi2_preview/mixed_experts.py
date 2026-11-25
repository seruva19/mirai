"""Per-projection mixed precision for MAGI-2's flattened expert groups."""

from __future__ import annotations

import torch
from torch import nn

from .quantized_experts import (
    MAGI2_PACKED_EXPERT_FORMATS,
    MAGI2_NF4_STORE_ATTR,
    MAGI2_ROUTED_EXPERT_TENSOR_NAMES,
    Magi2Nf4ExpertStore,
    _new_expert_store,
    iter_magi2_moe_layers,
)


_PLAN_PROJECTIONS = {"w1": "W_gate", "w2": "W_down", "w3": "W_up"}
_SUPPORTED_FORMATS = {*MAGI2_PACKED_EXPERT_FORMATS, "bf16"}


class _Bf16Groups(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight.to(torch.bfloat16).contiguous(), persistent=False)

    def materialize_segment(self, key, start, stop, *, dtype, device, group_ids):
        selected = group_ids[start:stop].to(self.weight.device)
        return self.weight.index_select(0, selected).to(device=device, dtype=dtype)


class Magi2MixedExpertStore(Magi2Nf4ExpertStore):
    """Compact format buckets with a stable mapping to the logical group IDs."""

    def __init__(self, *, num_groups: int, blocksize: int, base_format: str, formats) -> None:
        super().__init__(num_groups=num_groups, blocksize=blocksize)
        self.quant_format = base_format
        self.buckets = nn.ModuleDict()
        self._formats = {key: tuple(value) for key, value in formats.items()}
        self._locations: dict[str, tuple[tuple[str, int], ...]] = {}

    @torch.no_grad()
    def quantize_dense(self, key: str, source: torch.Tensor) -> None:
        assignments = self._formats[key]
        locations = [None] * len(assignments)
        for fmt in sorted(set(assignments)):
            ids = [index for index, value in enumerate(assignments) if value == fmt]
            name = f"{key}_{fmt}"
            for position, index in enumerate(ids):
                locations[index] = (name, position)
            selected = source.index_select(0, torch.tensor(ids, device=source.device))
            if fmt == "bf16":
                bucket = _Bf16Groups(selected)
            else:
                bucket = _new_expert_store(
                    num_groups=len(ids), blocksize=self.blocksize, quant_format=fmt
                )
                bucket.quantize_dense(key, selected)
            self.buckets[name] = bucket
        self._locations[key] = tuple(locations)
        self._shapes[key] = tuple(source.shape)
        self._dtypes[key] = source.dtype

    @torch.no_grad()
    def materialize_segment(self, key, group_start, group_stop, *, dtype, device, group_ids=None):
        self._require_loaded(key)
        count = self.num_groups if group_ids is None else group_ids.numel()
        if not 0 <= group_start < group_stop <= count:
            raise ValueError("MAGI-2 mixed expert segment is out of bounds.")
        ids = range(group_start, group_stop) if group_ids is None else group_ids[group_start:group_stop].tolist()
        by_bucket = {}
        for output_index, group_index in enumerate(ids):
            name, local_index = self._locations[key][group_index]
            destinations, sources = by_bucket.setdefault(name, ([], []))
            destinations.append(output_index)
            sources.append(local_index)
        output = torch.empty((group_stop - group_start, *self._shapes[key][1:]), dtype=dtype, device=device)
        for name, (destinations, sources) in by_bucket.items():
            selected = self.buckets[name].materialize_segment(
                key, 0, len(sources), dtype=dtype, device=device,
                group_ids=torch.tensor(sources),
            )
            output.index_copy_(0, torch.tensor(destinations, device=device), selected)
        return output

    def packed_state_descriptor(self):
        raise ValueError("Mixed-precision MAGI-2 stores currently require the original dense checkpoint and precision plan.")


def quantize_magi2_mixed_experts(transformer, *, plan_path, base_format, blocksize):
    from mirai.core.moe.calibration.precision import TensorPrecisionPlan, load_precision_plan
    from mirai.core.moe.calibration.router_repair import router_tensor_fingerprint

    plan = load_precision_plan(plan_path)
    if not isinstance(plan, TensorPrecisionPlan):
        raise ValueError("Mixed-precision MAGI-2 requires a lineage-bound v2 tensor precision plan.")
    layers = dict(iter_magi2_moe_layers(transformer))
    if set(plan.module_names()) != set(layers) or not layers:
        raise ValueError("MAGI-2 precision plan does not match model layer topology.")
    assignments = {name: {key: [] for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES} for name in layers}
    for entry in sorted(plan.assignments, key=lambda entry: entry.expert_id):
        assignments[entry.module_name][_PLAN_PROJECTIONS[entry.projection]].append(entry.quant_format)
    source_tensors = {}
    for name, layer in layers.items():
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            source = getattr(layer, key)
            formats = assignments[name][key]
            if len(formats) != source.shape[0] or not set(formats) <= _SUPPORTED_FORMATS:
                raise ValueError(f"Invalid group coverage or format in MAGI-2 precision plan for {name}.{key}.")
            if source.requires_grad:
                raise ValueError(f"MAGI-2 precision plan requires frozen weights: {name}.{key}.")
            source_tensors[f"{name}.{key}"] = source
    if router_tensor_fingerprint(source_tensors) != plan.source_weight_fingerprint:
        raise ValueError("MAGI-2 precision plan does not match the source-weight fingerprint.")
    del source_tensors

    stores = {}
    for name, layer in layers.items():
        store = Magi2MixedExpertStore(
            num_groups=int(layer.local_flatten_num_experts), blocksize=blocksize,
            base_format=base_format, formats=assignments[name],
        )
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            store.quantize_dense(key, layer._parameters[key])
            del layer._parameters[key]
        layer.add_module(MAGI2_NF4_STORE_ATTR, store)
        stores[name] = store
    return stores
