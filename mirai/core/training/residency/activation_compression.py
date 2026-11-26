"""Opt-in low-rank compression of autograd-saved activation matrices.

The randomized range projection follows the online activation-compression
family introduced by LoRAct (arXiv:2509.23472). Mirai keeps an uncompressed
reference path and only accepts a factorization when it stores fewer elements.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache, partial
from math import prod
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _LowRankSavedTensor:
    left: Any
    right: Any
    shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class _Int8SavedRows:
    codes: Any
    scales: Any
    dtype: Any
    device: Any = None
    host_ready: Any = None
    code_shape: tuple[int, ...] | None = None
    mask: Any = None


@dataclass(frozen=True)
class _Int8SavedBlocks:
    codes: Any
    scales: Any
    shape: tuple[int, ...]
    dtype: Any


class Int8ActivationCompression:
    """Compress non-leaf activations with per-block INT8 codes and FP32 scales."""

    def __init__(self, *, min_bytes: int) -> None:
        if min_bytes < 0:
            raise ValueError("Activation compression minimum bytes must be non-negative.")
        self.min_bytes = min_bytes
        self.compressed_tensors = 0
        self.original_bytes = 0
        self.stored_bytes = 0

    def _pack(self, tensor):
        if (
            not tensor.requires_grad or tensor.is_leaf or tensor.ndim < 2
            or tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}
            or tensor.layout != torch.strided or tensor.nbytes < self.min_bytes
        ):
            return tensor
        base = tensor._base
        if base is not None and base.is_leaf:
            return tensor
        block = 256
        count = tensor.numel()
        padded = count + (-count) % block
        stored_bytes = padded + 4 * (padded // block)
        if stored_bytes >= tensor.nbytes:
            return tensor
        values = torch.nn.functional.pad(tensor.detach().reshape(-1).float(), (0, padded - count)).view(-1, block)
        finite = torch.where(torch.isfinite(values), values, 0.0)
        scales = (finite.abs().amax(dim=1, keepdim=True) / 126).clamp_min_(torch.finfo(torch.float32).tiny)
        codes = (finite / scales).round_().clamp_(-126, 126).to(torch.int8)
        # Reserved codes preserve exceptional values without a device-to-host flag.
        codes.masked_fill_(torch.isnan(values), -128)
        codes.masked_fill_(torch.isneginf(values), -127)
        codes.masked_fill_(torch.isposinf(values), 127)
        self.compressed_tensors += 1
        self.original_bytes += tensor.nbytes
        self.stored_bytes += stored_bytes
        return _Int8SavedBlocks(codes, scales, tuple(tensor.shape), tensor.dtype)

    @staticmethod
    def _unpack(value):
        if not isinstance(value, _Int8SavedBlocks):
            return value
        codes = value.codes
        decoded = codes.float() * value.scales
        decoded.masked_fill_(codes == -128, float("nan"))
        decoded.masked_fill_(codes == -127, -float("inf"))
        decoded.masked_fill_(codes == 127, float("inf"))
        return decoded.flatten()[:prod(value.shape)].view(value.shape).to(value.dtype)

    def context(self):
        return torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)


@lru_cache(maxsize=None)
def training_cuda_copy_streams(device):
    return torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)


def _pack_int8_saved_rows(tensor, *, offload=False, lz4=False, bitpack=False):
    if tensor.ndim != 2 or tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return tensor.detach()
    rows, width = tensor.shape
    if rows == 0 or width + 4 >= width * tensor.element_size():
        return tensor.detach()
    values = tensor.detach().float()
    if not bool(torch.isfinite(values).all()):
        return tensor.detach()
    scales = (values.abs().amax(dim=1, keepdim=True) / 127).clamp_min_(torch.finfo(torch.float32).tiny)
    codes = (values / scales).round_().clamp_(-127, 127).to(torch.int8)
    if offload and tensor.is_cuda:
        code_shape = None
        mask = None
        if bitpack and codes.numel() >= 65536:
            from .zero_bitpack import pack_zero_bitmap

            packed = pack_zero_bitmap(codes)
            if packed is not None:
                code_shape = tuple(codes.shape)
                codes, mask = packed
        if lz4 and codes.numel() >= 65536:
            from .nvcomp_lz4 import encode_cuda_lz4

            encoded = encode_cuda_lz4(codes)
            if encoded is not None:
                code_shape = tuple(codes.shape)
                codes = encoded
        device = tensor.device
        d2h, _ = training_cuda_copy_streams(device)
        d2h.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(d2h):
            host_codes = torch.empty_like(codes, device="cpu", pin_memory=True)
            host_scales = torch.empty_like(scales, device="cpu", pin_memory=True)
            host_codes.copy_(codes, non_blocking=True)
            host_scales.copy_(scales, non_blocking=True)
            host_mask = None
            if mask is not None:
                host_mask = torch.empty_like(mask, device="cpu", pin_memory=True)
                host_mask.copy_(mask, non_blocking=True)
                mask.record_stream(d2h)
            codes.record_stream(d2h)
            scales.record_stream(d2h)
            ready = d2h.record_event()
        return _Int8SavedRows(host_codes, host_scales, tensor.dtype, device, ready, code_shape, host_mask)
    return _Int8SavedRows(codes, scales, tensor.dtype)


def _unpack_int8_saved_rows(value):
    if not isinstance(value, _Int8SavedRows):
        return value
    codes, scales = value.codes, value.scales
    mask = value.mask
    if value.host_ready is not None:
        _, h2d = training_cuda_copy_streams(value.device)
        compute = torch.cuda.current_stream(value.device)
        with torch.cuda.stream(h2d):
            h2d.wait_event(value.host_ready)
            codes = codes.to(value.device, non_blocking=True)
            scales = scales.to(value.device, non_blocking=True)
            if mask is not None:
                mask = mask.to(value.device, non_blocking=True)
            ready = h2d.record_event()
        compute.wait_event(ready)
        codes.record_stream(compute)
        scales.record_stream(compute)
        if mask is not None:
            mask.record_stream(compute)
    if mask is not None:
        from .zero_bitpack import unpack_zero_bitmap

        codes = unpack_zero_bitmap(codes, mask, value.code_shape, dtype=torch.int8)
    elif value.code_shape is not None:
        from .nvcomp_lz4 import decode_cuda_lz4

        decoded = torch.empty(value.code_shape, device=value.device, dtype=torch.int8)
        decode_cuda_lz4(codes, decoded)
        codes = decoded
    return (codes.float() * scales).to(value.dtype)


def int8_saved_rows_context(*, offload=False, lz4=False, bitpack=False):
    """Compress saved activation rows in a scope with no saved float weights."""
    if (lz4 or bitpack) and not offload or (lz4 and bitpack):
        raise ValueError("Select one activation compression codec with checkpoint offload enabled.")
    pack = partial(_pack_int8_saved_rows, offload=True, lz4=lz4, bitpack=bitpack) if offload else _pack_int8_saved_rows
    return torch.autograd.graph.saved_tensors_hooks(pack, _unpack_int8_saved_rows)


class LowRankActivationCompression:
    """Saved-tensor hook using a deterministic randomized range projection."""

    def __init__(self, *, rank: int, min_bytes: int, seed: int = 0) -> None:
        if int(rank) <= 0:
            raise ValueError("Activation compression rank must be positive.")
        if int(min_bytes) < 0:
            raise ValueError("Activation compression minimum bytes must be non-negative.")
        self.rank = int(rank)
        self.min_bytes = int(min_bytes)
        self.seed = int(seed)
        self.compressed_tensors = 0
        self.original_bytes = 0
        self.stored_bytes = 0

    def _pack(self, tensor: Any) -> Any:
        if (
            not isinstance(tensor, torch.Tensor)
            or not tensor.is_floating_point()
            or tensor.ndim < 2
            or tensor.numel() * tensor.element_size() < self.min_bytes
        ):
            return tensor
        rows = int(tensor.numel() // tensor.shape[-1])
        columns = int(tensor.shape[-1])
        rank = min(self.rank, rows, columns)
        if rank <= 0 or rank * (rows + columns) >= rows * columns:
            return tensor
        matrix = tensor.detach().reshape(rows, columns)
        generator = torch.Generator(device=matrix.device)
        generator.manual_seed(self.seed + self.compressed_tensors)
        omega = torch.randn(
            columns,
            rank,
            device=matrix.device,
            dtype=torch.float32,
            generator=generator,
        )
        left, _ = torch.linalg.qr(matrix.float() @ omega, mode="reduced")
        right = left.transpose(0, 1) @ matrix.float()
        left = left.to(dtype=tensor.dtype)
        right = right.to(dtype=tensor.dtype)
        original_bytes = tensor.numel() * tensor.element_size()
        stored_bytes = (left.numel() + right.numel()) * left.element_size()
        if stored_bytes >= original_bytes:
            return tensor
        self.compressed_tensors += 1
        self.original_bytes += int(original_bytes)
        self.stored_bytes += int(stored_bytes)
        return _LowRankSavedTensor(
            left=left,
            right=right,
            shape=tuple(int(value) for value in tensor.shape),
            dtype=tensor.dtype,
        )

    @staticmethod
    def _unpack(value: Any) -> Any:
        if not isinstance(value, _LowRankSavedTensor):
            return value
        return (value.left.float() @ value.right.float()).reshape(value.shape).to(value.dtype)

    def context(self):
        if torch is None:  # pragma: no cover
            return nullcontext()
        return torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)


def activation_compression_context(
    *,
    enabled: bool,
    rank: int,
    min_bytes: int,
    seed: int = 0,
    codec: str = "low_rank",
):
    if not enabled:
        return nullcontext()
    if codec == "int8":
        return Int8ActivationCompression(min_bytes=min_bytes).context()
    if codec != "low_rank":
        raise ValueError("Activation compression codec must be low_rank or int8.")
    return LowRankActivationCompression(
        rank=rank,
        min_bytes=min_bytes,
        seed=seed,
    ).context()
