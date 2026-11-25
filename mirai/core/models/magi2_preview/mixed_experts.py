"""Per-projection mixed precision for MAGI-2's flattened expert groups."""

from __future__ import annotations

from hashlib import sha256
import json
from math import prod

import torch
from torch import nn

from .quantized_experts import (
    MAGI2_PACKED_EXPERT_FORMATS,
    MAGI2_NF4_STORE_ATTR,
    MAGI2_NF4_CODE_FIELDS,
    MAGI2_NF4_GROUP_FIELDS,
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
    is_mixed_precision = True

    def __init__(self, *, num_groups: int, blocksize: int, base_format: str, formats, lineage) -> None:
        super().__init__(num_groups=num_groups, blocksize=blocksize)
        self.quant_format = base_format
        self.buckets = nn.ModuleDict()
        self._formats = {key: tuple(value) for key, value in formats.items()}
        self.lineage = dict(lineage)
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
            if fmt == "bf16":
                selected = torch.empty((len(ids), *source.shape[1:]), dtype=torch.bfloat16, device=source.device)
                for row, index in enumerate(ids):
                    selected[row].copy_(source[index])
                bucket = _Bf16Groups(selected)
            else:
                bucket = _new_expert_store(
                    num_groups=len(ids), blocksize=self.blocksize, quant_format=fmt
                )
                bucket.quantize_dense(key, source, group_ids=ids)
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
        specs = {}
        for key, formats in self._formats.items():
            for fmt in sorted(set(formats)):
                name = f"{key}_{fmt}"
                bucket = self.buckets[name]
                specs[name] = (
                    {"buffers": ["weight"]}
                    if fmt == "bf16"
                    else bucket.packed_state_descriptor(keys=(key,))
                )
        return {
            "num_groups": self.num_groups,
            "blocksize": self.blocksize,
            "expert_shapes": self._shapes,
            "expert_dtypes": {key: str(dtype).removeprefix("torch.") for key, dtype in self._dtypes.items()},
            "formats": self._formats,
            "bucket_specs": specs,
            "lineage": self.lineage,
        }

    @classmethod
    def from_packed_state(cls, *, descriptor, buffers, blocksize, base_format, group_shapes):
        formats, specs, lineage = (
            descriptor.get("formats"), descriptor.get("bucket_specs"), descriptor.get("lineage"),
        )
        keys = set(MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
        groups = group_shapes["W_gate"][0]
        if (
            not isinstance(formats, dict) or set(formats) != keys or not isinstance(specs, dict)
            or descriptor.get("num_groups") != groups or descriptor.get("blocksize") != blocksize
            or not isinstance(lineage, dict) or not all(
                isinstance(lineage.get(key), str) and lineage[key].strip()
                for key in ("plan_sha256", "source_weight_fingerprint", "model_snapshot_id", "dataset_snapshot_id", "config_snapshot_id")
            )
        ):
            raise ValueError("Invalid mixed MAGI-2 artifact topology or lineage.")
        if any(
            not isinstance(values, list) or len(values) != groups
            or any(not isinstance(fmt, str) or fmt not in _SUPPORTED_FORMATS for fmt in values)
            for values in formats.values()
        ):
            raise ValueError("Invalid mixed MAGI-2 format assignments.")
        if set(specs) != {f"{key}_{fmt}" for key, values in formats.items() for fmt in set(values)}:
            raise ValueError("Invalid mixed MAGI-2 bucket inventory.")
        store = cls(num_groups=groups, blocksize=blocksize, base_format=base_format, formats=formats, lineage=lineage)
        consumed = set()
        for key, values in formats.items():
            shape = group_shapes[key]
            dtype = getattr(torch, str(descriptor["expert_dtypes"][key]), None)
            if tuple(descriptor["expert_shapes"][key]) != shape or dtype not in {torch.float32, torch.float16, torch.bfloat16}:
                raise ValueError("Mixed MAGI-2 projection shape or dtype is invalid.")
            locations = [None] * groups
            for fmt in sorted(set(values)):
                ids = [index for index, value in enumerate(values) if value == fmt]
                name = f"{key}_{fmt}"
                spec = specs[name]
                names = spec["buffers"]
                prefix = f"buckets.{name}."
                fields = {buffer: buffers[prefix + buffer] for buffer in names}
                consumed.update(prefix + buffer for buffer in names)
                bucket_shape = (len(ids), *shape[1:])
                if fmt == "bf16":
                    if set(fields) != {"weight"} or fields["weight"].shape != bucket_shape or fields["weight"].dtype != torch.bfloat16:
                        raise ValueError("Invalid BF16 bucket in a mixed MAGI-2 artifact.")
                    bucket = _Bf16Groups(fields["weight"])
                else:
                    bucket = _new_expert_store(num_groups=len(ids), blocksize=blocksize, quant_format=fmt)
                    if tuple(spec["expert_shapes"][key]) != bucket_shape or spec["expert_dtypes"][key] != descriptor["expert_dtypes"][key]:
                        raise ValueError("Mixed MAGI-2 bucket shape or dtype mismatch.")
                    if fmt == "nf4":
                        _validate_nf4_bucket(key, fields, spec, bucket_shape, blocksize)
                    bucket.restore_packed_state(descriptor=spec, buffers=fields, keys=(key,))
                store.buckets[name] = bucket
                for position, index in enumerate(ids):
                    locations[index] = (name, position)
            store._locations[key] = tuple(locations)
            store._shapes[key] = shape
            store._dtypes[key] = dtype
        if consumed != set(buffers):
            raise ValueError("Mixed MAGI-2 artifact has unexpected packed buffers.")
        return store


def _validate_nf4_bucket(key, fields, spec, shape, blocksize):
    expected = {f"{key}_{suffix}" for suffix in (*MAGI2_NF4_GROUP_FIELDS, *MAGI2_NF4_CODE_FIELDS)}
    if set(fields) != expected:
        raise ValueError("Invalid NF4 mixed MAGI-2 bucket inventory.")
    meta = spec["nf4_meta"]
    nested = meta["nested_blocksize"]
    if meta["blocksize"] != blocksize or type(nested) is not int or nested < 1:
        raise ValueError("Invalid mixed MAGI-2 NF4 block metadata.")
    groups, *matrix = shape
    elements = prod(matrix)
    count = (elements + blocksize - 1) // blocksize
    sizes = ((elements + 1) // 2, count, (count + nested - 1) // nested, 1)
    for suffix, size in zip(MAGI2_NF4_GROUP_FIELDS, sizes, strict=True):
        value = fields[f"{key}_{suffix}"]
        if value.ndim < 1 or value.shape[0] != groups or value.numel() != groups * size:
            raise ValueError("Invalid mixed MAGI-2 NF4 field shape.")
    if fields[f"{key}_nf4_code"].numel() != 16 or fields[f"{key}_nf4_ncode"].numel() != 256:
        raise ValueError("Invalid mixed MAGI-2 NF4 codebook size.")


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
    plan_payload = plan.to_dict()
    lineage = {key: value for key, value in plan_payload.items() if key.endswith("_snapshot_id") or key == "source_weight_fingerprint"}
    lineage["plan_sha256"] = sha256(json.dumps(plan_payload, sort_keys=True).encode()).hexdigest()

    stores = {}
    for name, layer in layers.items():
        store = Magi2MixedExpertStore(
            num_groups=int(layer.local_flatten_num_experts), blocksize=blocksize,
            base_format=base_format, formats=assignments[name], lineage=lineage,
        )
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            store.quantize_dense(key, layer._parameters[key])
            del layer._parameters[key]
        layer.add_module(MAGI2_NF4_STORE_ATTR, store)
        stores[name] = store
    return stores
