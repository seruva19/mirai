"""Opt-in low-rank compression of autograd-saved activation matrices.

The randomized range projection follows the online activation-compression
family introduced by LoRAct (arXiv:2509.23472). Mirai keeps an uncompressed
reference path and only accepts a factorization when it stores fewer elements.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
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


def _pack_int8_saved_rows(tensor):
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
    return _Int8SavedRows(codes, scales, tensor.dtype)


def _unpack_int8_saved_rows(value):
    if not isinstance(value, _Int8SavedRows):
        return value
    return (value.codes.float() * value.scales).to(value.dtype)


def int8_saved_rows_context():
    """Compress saved activation rows in a scope with no saved float weights."""
    return torch.autograd.graph.saved_tensors_hooks(_pack_int8_saved_rows, _unpack_int8_saved_rows)


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
):
    if not enabled:
        return nullcontext()
    return LowRankActivationCompression(
        rank=rank,
        min_bytes=min_bytes,
        seed=seed,
    ).context()
