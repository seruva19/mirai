"""Quantization primitives for compressed frozen-weight storage."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from mirai.core.moe.runtime.specs import resolve_quantization_workspace_bytes

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


DEFAULT_GROUP_SIZES = (256, 64, 16)
NF4_BLOCKSIZE = 64
# gguf_iq4/gguf_iq3 are the default-off sub-4-bit GGUF k-quant formats (owner seam
# in gguf_quant.py). Listed here so normalize_quant_format accepts them; storage +
# dequant dispatch live behind explicit `_quant_format in GGUF_FORMATS` branches.
QUANT_FORMATS = (
    "fp8",
    "int8",
    "nf4",
    "gguf_iq4",
    "gguf_iq3",
    "mxfp8_e4m3",
    "mxfp4",
    "nvfp4",
)
DEFAULT_QUANTIZATION_WORKSPACE_BYTES = 256 * 1024 * 1024
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompressedWeightReport:
    linear_modules: int
    grouped_expert_modules: int
    quantized_tensors: int
    quantized_numel: int
    expert_weight_access: str = "full_dequant"
    expert_dequant_chunk_size: int = 0
    skipped_modules: tuple[str, ...] = ()

    @property
    def replaced_modules(self) -> int:
        return int(self.linear_modules + self.grouped_expert_modules)


def normalize_compressed_weights_strategy(strategy: str | None) -> str:
    value = str(strategy or "auto").strip().lower()
    return "auto" if value == "" else value


def parse_group_sizes(value: str | int | Iterable[int] | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_GROUP_SIZES
    if isinstance(value, int):
        groups = (int(value),)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "auto"}:
            return DEFAULT_GROUP_SIZES
        groups = tuple(int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip())
    else:
        groups = tuple(int(v) for v in value)
    if not groups:
        raise ValueError("compressed_weights group size list is empty.")
    invalid = [g for g in groups if not _is_power_of_four(g)]
    if invalid:
        raise ValueError(
            "compressed_weights group sizes must be powers of 4 greater than or equal to 4; "
            f"got {invalid}."
        )
    return tuple(dict.fromkeys(groups))


def best_group_size(in_features: int, group_sizes: str | int | Iterable[int] | None = None) -> int:
    for group_size in sorted(parse_group_sizes(group_sizes), reverse=True):
        if int(in_features) % int(group_size) == 0:
            return int(group_size)
    return 0


def _is_power_of_four(value: int) -> bool:
    if int(value) < 4:
        return False
    current = int(value)
    while current > 1 and current % 4 == 0:
        current //= 4
    return current == 1


def _hadamard(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not _is_power_of_four(size):
        raise ValueError(f"Hadamard size must be a power of 4 >= 4, got {size}.")
    key = (int(size), str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / math.sqrt(size)
    _HADAMARD_CACHE[key] = h
    return h


def _validate_rotation(
    rotation: torch.Tensor,
    group_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    resolved_group = int(group_size)
    if tuple(rotation.shape) != (resolved_group, resolved_group):
        raise ValueError(
            "compressed_weights rotation shape must match its group size; "
            f"expected {(resolved_group, resolved_group)}, got {tuple(rotation.shape)}."
        )
    value = rotation.detach().to(device=device, dtype=dtype)
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("compressed_weights rotation must be finite.")
    identity = torch.eye(resolved_group, device=device, dtype=dtype)
    error = (value.transpose(0, 1) @ value - identity).abs().amax()
    tolerance = 2e-3 if dtype in {torch.float16, torch.bfloat16} else 2e-4
    if float(error.item()) > tolerance:
        raise ValueError(
            "compressed_weights rotation must be orthogonal; "
            f"max Gram error is {float(error.item()):.6g}."
        )
    return value


def _rotate_last_dim(
    weight_2d: torch.Tensor,
    group_size: int,
    *,
    inverse: bool,
    rotation: torch.Tensor | None = None,
) -> torch.Tensor:
    if int(group_size) <= 0:
        return weight_2d
    out_features, in_features = weight_2d.shape
    if in_features % int(group_size) != 0:
        raise ValueError(
            f"compressed_weights group size {group_size} does not divide in_features={in_features}."
        )
    h = (
        _hadamard(
            int(group_size),
            device=weight_2d.device,
            dtype=weight_2d.dtype,
        )
        if rotation is None
        else _validate_rotation(
            rotation,
            int(group_size),
            device=weight_2d.device,
            dtype=weight_2d.dtype,
        )
    )
    grouped = weight_2d.reshape(out_features, in_features // int(group_size), int(group_size))
    rot = h.transpose(0, 1) if inverse else h
    return torch.matmul(grouped, rot).reshape(out_features, in_features)


def _quantization_workspace_bytes() -> int:
    return resolve_quantization_workspace_bytes(DEFAULT_QUANTIZATION_WORKSPACE_BYTES)


def _quantize_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
    rotation: torch.Tensor | None = None,
    workspace_bytes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    with torch.no_grad():
        original_shape = tuple(weight.shape)
        if len(original_shape) < 2:
            raise ValueError(f"compressed_weights expects at least 2D weights, got {original_shape}.")
        in_features = int(original_shape[-1])
        rows = int(weight.numel() // max(in_features, 1))
        resolved_group = int(group_size) if int(group_size) > 0 and in_features % int(group_size) == 0 else 0
        source = weight.detach().reshape(rows, in_features)
        budget = int(workspace_bytes or _quantization_workspace_bytes())
        if budget <= 0:
            raise ValueError("compressed_weights quantization workspace must be > 0 bytes.")
        # Rotation, scaling, division, and rounding can coexist. This conservative
        # estimate keeps conversion memory bounded independently of tensor size.
        bytes_per_row = max(1, in_features * 16)
        rows_per_chunk = max(1, min(rows, budget // bytes_per_row))
        quantized_2d = torch.empty(
            (rows, in_features),
            dtype=torch.int8,
            device=weight.device,
        )
        scale_2d = torch.empty((rows, 1), dtype=torch.float32, device=weight.device)
        for start in range(0, rows, rows_per_chunk):
            end = min(rows, start + rows_per_chunk)
            work = source[start:end].float()
            if resolved_group > 0:
                work = _rotate_last_dim(
                    work,
                    resolved_group,
                    inverse=False,
                    rotation=rotation,
                )
            chunk_scale = (work.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
            quantized_2d[start:end].copy_(
                (work / chunk_scale).round().clamp(-127, 127).to(torch.int8)
            )
            scale_2d[start:end].copy_(chunk_scale)
            del work, chunk_scale
        quantized = quantized_2d.reshape(original_shape).contiguous()
        scale_shape = (*original_shape[:-1], 1)
        return quantized, scale_2d.reshape(scale_shape).contiguous(), resolved_group


def _dequantize_weight(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
    rotation: torch.Tensor | None = None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    q = quantized.to(device=device)
    s = scale.to(device=device, dtype=torch.float32)
    original_shape = tuple(q.shape)
    in_features = int(original_shape[-1])
    rows = int(q.numel() // max(in_features, 1))
    work = (q.float() * s.float()).reshape(rows, in_features)
    if int(group_size) > 0:
        work = _rotate_last_dim(
            work,
            int(group_size),
            inverse=True,
            rotation=rotation,
        )
    return work.reshape(original_shape).to(dtype=dtype)


def _is_contiguous_run(expert_indices: "list[int] | tuple[int, ...]") -> bool:
    """True when indices are a strictly ascending run start, start+1, ..., start+n-1.

    Such a run can be served as one contiguous slice/view of the expert axis.
    """
    n = len(expert_indices)
    if n == 0:
        return False
    return int(expert_indices[-1]) - int(expert_indices[0]) == n - 1


def normalize_quant_format(value: str | None) -> str:
    text = str(value or "int8").strip().lower()
    aliases = {
        "": "int8",
        "auto": "int8",
        "gguf_iq4_xs": "gguf_iq4",
        "iq4_xs": "gguf_iq4",
        "gguf_iq3_xxs": "gguf_iq3",
        "iq3_xxs": "gguf_iq3",
    }
    normalized = aliases.get(text, text)
    if normalized not in QUANT_FORMATS:
        raise ValueError(
            "compressed_weights quant_format must be one of: " + ", ".join(QUANT_FORMATS) + "."
        )
    return normalized


def _import_bnb_functional() -> Any:
    try:
        import bitsandbytes.functional as bnb_functional
    except (ImportError, OSError) as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            "memory.frozen_weight_quantization='nf4' requires bitsandbytes with CUDA "
            "4-bit operators (bitsandbytes.functional.quantize_4bit/dequantize_4bit)."
        ) from exc
    return bnb_functional


@dataclass(frozen=True)
class _Nf4Meta:
    """Static (per-tensor invariant) NF4 double-quantization metadata."""

    blocksize: int
    nested_blocksize: int
    nested_dtype: torch.dtype
    weight_dtype: torch.dtype


def _nf4_quantize_2d(
    weight: torch.Tensor,
    *,
    blocksize: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], _Nf4Meta]:
    bnb = _import_bnb_functional()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "nf4 frozen-weight quantization requires CUDA; "
            "bitsandbytes.quantize_4bit is a CUDA operator."
        )
    with torch.no_grad():
        source = weight.detach().to(device="cuda", dtype=torch.bfloat16).contiguous()
        packed, quant_state = bnb.quantize_4bit(
            source,
            blocksize=int(blocksize),
            compress_statistics=True,
            quant_type="nf4",
        )
        # Free the bf16 CUDA upload transient the moment quantization is done: the
        # packed NF4 payload and every double-quant statistic in ``quant_state`` own
        # their own storage, so nothing below reads ``source`` again. On the on-load
        # path this runs once per expert/linear (thousands of times), so dropping the
        # reference eagerly keeps the live CUDA working set at ~one tensor instead of
        # trailing the last upload until the enclosing frame returns.
        del source
        nested = quant_state.state2
        fields = {
            "packed": packed.contiguous(),
            "absmax": quant_state.absmax.contiguous(),
            "nested_absmax": nested.absmax.contiguous(),
            "offset": quant_state.offset.reshape(()).contiguous(),
        }
        codes = {
            "code": quant_state.code.contiguous(),
            "nested_code": nested.code.contiguous(),
        }
        meta = _Nf4Meta(
            blocksize=int(quant_state.blocksize),
            nested_blocksize=int(nested.blocksize),
            nested_dtype=nested.dtype,
            weight_dtype=quant_state.dtype,
        )
        return fields, codes, meta


def _nf4_dequantize(
    fields: Mapping[str, torch.Tensor],
    codes: Mapping[str, torch.Tensor],
    meta: _Nf4Meta,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    bnb = _import_bnb_functional()
    from bitsandbytes.functional import QuantState

    target = torch.device(device)
    nested_state = QuantState(
        absmax=fields["nested_absmax"].to(device=target),
        code=codes["nested_code"].to(device=target),
        blocksize=int(meta.nested_blocksize),
        dtype=meta.nested_dtype,
    )
    quant_state = QuantState(
        absmax=fields["absmax"].to(device=target),
        shape=torch.Size(tuple(int(dim) for dim in shape)),
        code=codes["code"].to(device=target),
        blocksize=int(meta.blocksize),
        quant_type="nf4",
        dtype=meta.weight_dtype,
        offset=fields["offset"].to(device=target),
        state2=nested_state,
    )
    weight = bnb.dequantize_4bit(
        fields["packed"].to(device=target),
        quant_state,
        blocksize=int(meta.blocksize),
        quant_type="nf4",
    )
    return weight.reshape(tuple(int(dim) for dim in shape)).to(dtype=dtype)
