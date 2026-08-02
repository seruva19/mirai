"""Compact 2:4 physical expert-weight provider for packed artifacts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping

from mirai.core.moe.storage.physical_weights import PhysicalWeightProviderContext
from mirai.core.moe.storage.physical_weights import register_physical_weight_provider

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SPARSE24_PROVIDER_NAME = "stun_sparse24"
SPARSE24_PROVIDER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PackedSparse24:
    """Two int8 values and their positions for every group of four."""

    values: Any
    scales: Any
    positions: Any
    original_shape: tuple[int, ...]
    quant_group_size: int

    def dense(self, *, dtype: Any, device: Any) -> Any:
        return unpack_sparse24(self, dtype=dtype, device=device)


def pack_sparse24(
    weight: Any,
    *,
    quant_group_size: int = 32,
) -> PackedSparse24:
    """Magnitude-prune to 2:4 and block-quantize the retained values."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Compact 2:4 packing requires torch.")
    source = torch.as_tensor(weight).detach()
    if source.ndim < 2 or int(source.shape[-1]) % 4:
        raise ValueError("Compact 2:4 packing requires input dimension divisible by four.")
    if not bool(torch.isfinite(source).all().item()):
        raise ValueError("Compact 2:4 source contains non-finite values.")
    group_size = int(quant_group_size)
    if group_size <= 0:
        raise ValueError("quant_group_size must be positive.")

    grouped = source.to(dtype=torch.float32).reshape(*source.shape[:-1], -1, 4)
    positions = grouped.abs().topk(k=2, dim=-1, sorted=False).indices
    positions = positions.sort(dim=-1).values.to(dtype=torch.uint8)
    selected = grouped.gather(-1, positions.to(dtype=torch.long))
    groups = int(selected.shape[-2])
    blocks = math.ceil(groups / group_size)
    padded_groups = blocks * group_size
    if padded_groups != groups:
        selected = torch.nn.functional.pad(
            selected,
            (0, 0, 0, padded_groups - groups),
        )
    blocked = selected.reshape(*selected.shape[:-2], blocks, group_size, 2)
    max_abs = blocked.abs().amax(dim=(-1, -2))
    scales = max_abs / 127.0
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = torch.round(
        blocked / safe_scales.unsqueeze(-1).unsqueeze(-1)
    ).clamp_(-127, 127)
    quantized = quantized.to(dtype=torch.int8)
    quantized = quantized.reshape(*selected.shape[:-2], padded_groups, 2)[
        ..., :groups, :
    ]
    return PackedSparse24(
        values=quantized.to(device="cpu").contiguous(),
        scales=scales.to(device="cpu", dtype=torch.bfloat16).contiguous(),
        positions=positions.to(device="cpu").contiguous(),
        original_shape=tuple(int(value) for value in source.shape),
        quant_group_size=group_size,
    )


def validate_packed_sparse24(state: PackedSparse24) -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Compact 2:4 validation requires torch.")
    if len(state.original_shape) < 2 or int(state.original_shape[-1]) % 4:
        raise ValueError("Compact 2:4 original shape is invalid.")
    groups = int(state.original_shape[-1]) // 4
    prefix = tuple(int(value) for value in state.original_shape[:-1])
    if tuple(state.values.shape) != (*prefix, groups, 2):
        raise ValueError("Compact 2:4 values shape does not match original shape.")
    if tuple(state.positions.shape) != tuple(state.values.shape):
        raise ValueError("Compact 2:4 positions must match values.")
    blocks = math.ceil(groups / int(state.quant_group_size))
    if tuple(state.scales.shape) != (*prefix, blocks):
        raise ValueError("Compact 2:4 scales shape does not match quantization blocks.")
    positions = state.positions.to(dtype=torch.int64)
    if bool(((positions < 0) | (positions > 3)).any().item()):
        raise ValueError("Compact 2:4 positions must be in [0, 3].")
    if bool((positions[..., 0] >= positions[..., 1]).any().item()):
        raise ValueError("Compact 2:4 positions must be strictly increasing.")
    if not bool(torch.isfinite(state.scales).all().item()):
        raise ValueError("Compact 2:4 scales contain non-finite values.")
    if bool((state.scales < 0).any().item()):
        raise ValueError("Compact 2:4 scales must be non-negative.")


def unpack_sparse24(
    state: PackedSparse24,
    *,
    dtype: Any,
    device: Any,
) -> Any:
    """Decode a compact 2:4 tensor without retaining a dense base copy."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Compact 2:4 decoding requires torch.")
    validate_packed_sparse24(state)
    groups = int(state.original_shape[-1]) // 4
    group_size = int(state.quant_group_size)
    scales = state.scales.to(device=device, dtype=torch.float32)
    expanded_scales = scales.repeat_interleave(group_size, dim=-1)[..., :groups]
    values = state.values.to(device=device, dtype=torch.float32)
    values = values * expanded_scales.unsqueeze(-1)
    dense = torch.zeros(
        (*state.original_shape[:-1], groups, 4),
        device=device,
        dtype=torch.float32,
    )
    dense.scatter_(
        -1,
        state.positions.to(device=device, dtype=torch.long),
        values,
    )
    return dense.reshape(state.original_shape).to(dtype=dtype)


class Sparse24PhysicalWeightProvider:
    """Decode schema-v1 compact 2:4 expert projections on demand."""

    name = SPARSE24_PROVIDER_NAME

    def __init__(self, context: PhysicalWeightProviderContext) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Compact 2:4 provider requires torch.")
        version = int(context.spec.get("schema_version", 0))
        if version != SPARSE24_PROVIDER_SCHEMA_VERSION:
            raise ValueError(f"Unsupported compact 2:4 provider schema {version}.")
        projections = context.spec.get("projections")
        if not isinstance(projections, Mapping) or set(projections) != {
            "w1",
            "w2",
            "w3",
        }:
            raise ValueError("Compact 2:4 provider requires w1, w2, and w3.")
        self.num_experts = int(context.num_experts)
        self._manifest_spec = copy.deepcopy(dict(context.spec))
        self._shapes = {
            str(key): tuple(int(value) for value in shape)
            for key, shape in context.shapes.items()
        }
        self._specs: dict[str, dict[str, Any]] = {}
        self._tensors = context.tensors
        names: set[str] = set()
        for key, raw in projections.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"Compact 2:4 projection {key!r} must be an object.")
            projection = dict(raw)
            required = {
                str(projection.get("values", "")),
                str(projection.get("scales", "")),
                str(projection.get("positions", "")),
            }
            if "" in required or not required.issubset(context.tensors):
                raise KeyError(f"Compact 2:4 tensors for {key!r} are missing.")
            shape = self._shapes.get(str(key))
            if shape is None or len(shape) != 3 or shape[0] != self.num_experts:
                raise ValueError(f"Compact 2:4 projection {key!r} has invalid shape.")
            if shape[-1] % 4:
                raise ValueError(
                    f"Compact 2:4 projection {key!r} input dimension is not divisible by four."
                )
            self._specs[str(key)] = projection
            names.update(required)
        self._names = frozenset(names)

    def expert_weight_shape(self, key: str) -> tuple[int, ...]:
        try:
            return self._shapes[str(key)]
        except KeyError as exc:
            raise AttributeError(f"Unknown compact 2:4 projection {key!r}.") from exc

    def packed_tensor_names(self) -> frozenset[str]:
        return self._names

    def manifest_spec(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._manifest_spec)

    def packed_tensors(self) -> Mapping[str, Any]:
        return {name: self._tensors[name] for name in self._names}

    def _slice(self, name: str, expert_index: int) -> Any:
        get_slice = getattr(self._tensors, "get_slice", None)
        if callable(get_slice):
            return get_slice(name, int(expert_index))
        return self._tensors[name][int(expert_index)]

    def materialize_expert(
        self,
        key: str,
        expert_index: int,
        *,
        dtype: Any,
        device: Any,
    ) -> Any:
        index = int(expert_index)
        if index < 0 or index >= self.num_experts:
            raise IndexError(f"Expert index {index} is outside [0, {self.num_experts}).")
        projection = self._specs[str(key)]
        shape = self._shapes[str(key)][1:]
        state = PackedSparse24(
            values=self._slice(str(projection["values"]), index),
            scales=self._slice(str(projection["scales"]), index),
            positions=self._slice(str(projection["positions"]), index),
            original_shape=shape,
            quant_group_size=int(projection["quant_group_size"]),
        )
        return state.dense(dtype=dtype, device=device).detach()


@register_physical_weight_provider(SPARSE24_PROVIDER_NAME)
def _build_sparse24_provider(
    context: PhysicalWeightProviderContext,
) -> Sparse24PhysicalWeightProvider:
    return Sparse24PhysicalWeightProvider(context)


__all__ = [
    "PackedSparse24",
    "SPARSE24_PROVIDER_NAME",
    "SPARSE24_PROVIDER_SCHEMA_VERSION",
    "Sparse24PhysicalWeightProvider",
    "pack_sparse24",
    "unpack_sparse24",
    "validate_packed_sparse24",
]
