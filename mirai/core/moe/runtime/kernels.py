"""Model-agnostic dropless MoE kernel backends."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Callable


MOE_KERNEL_BACKENDS = {
    "auto",
    "torch",
    "rotated_int8",
    "compiled_packed",
    "megablocks",
    # Selects a model-family-owned grouped-GEMM expert execution seam; it has no
    # generic direct-routed implementation in this registry.
    "grouped",
}


def normalize_moe_kernel_backend(value: str | None) -> str:
    text = str(value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "default": "auto",
        "pytorch": "torch",
        "batched": "torch",
        "megablocks_grouped": "megablocks",
        "dmoe": "megablocks",
        "grouped_gemm": "grouped",
    }
    normalized = aliases.get(text, text)
    if normalized not in MOE_KERNEL_BACKENDS:
        raise ValueError(
            "memory.moe_kernel_backend must be one of: "
            + ", ".join(sorted(MOE_KERNEL_BACKENDS))
            + "."
        )
    return normalized


@dataclass(frozen=True)
class RoutedExpertBatch:
    tokens: Any
    counts: Any
    restore_state: Any


class MegaBlocksKernelBackend:
    """Adapter for MegaBlocks dropless routing and grouped GEMM execution."""

    name = "megablocks"

    def __init__(self, *, ops: Any | None = None, grouped_gemm_ops: Any | None = None) -> None:
        self.ops = ops if ops is not None else _import_megablocks_ops()
        self.grouped_gemm_ops = (
            grouped_gemm_ops if grouped_gemm_ops is not None else _import_grouped_gemm_ops()
        )

    def route(
        self,
        tokens: Any,
        top_scores: Any,
        top_indices: Any,
        *,
        num_experts: int,
    ) -> RoutedExpertBatch:
        if tokens.device.type != "cuda":
            raise RuntimeError("MegaBlocks routing requires CUDA tensors.")
        if top_scores.shape != top_indices.shape or top_indices.ndim != 2:
            raise ValueError("MegaBlocks routing requires matching [tokens, top_k] tensors.")
        if bool((top_scores == 0).any()):
            raise ValueError(
                "MegaBlocks dropless routing does not accept zero-weight expert assignments."
            )
        top_k = int(top_indices.shape[1])
        flat_experts = top_indices.reshape(-1).to(dtype=_torch().int32)
        sort_end_bit = max(1, int(math.ceil(math.log2(max(2, int(num_experts))))))
        with _torch().no_grad():
            bin_ids, indices = self.ops.sort(flat_experts, sort_end_bit)
            counts = self.ops.histogram(flat_experts, int(num_experts))
            bins = self.ops.inclusive_cumsum(counts, 0)
            if bins.ndim == 0:
                bins = bins.view(1)
        routed = self.ops.gather(
            tokens.reshape(-1, tokens.shape[-1]),
            indices,
            bin_ids,
            bins,
            top_k,
        )
        return RoutedExpertBatch(
            tokens=routed,
            counts=counts,
            restore_state=(indices, bin_ids, top_scores.reshape(-1), bins, top_k),
        )

    def restore(self, output: Any, routed: RoutedExpertBatch) -> Any:
        indices, bin_ids, scores, bins, top_k = routed.restore_state
        return self.ops.scatter(output, indices, bin_ids, scores, bins, top_k)

    def compute(
        self,
        experts: Any,
        routed: RoutedExpertBatch,
        *,
        fallback: Callable[[Any, Any], Any],
    ) -> Any:
        runner = getattr(experts, "run_megablocks_grouped", None)
        if callable(runner):
            return runner(
                routed.tokens,
                routed.counts,
                grouped_gemm_ops=self.grouped_gemm_ops,
            )
        return fallback(routed.tokens, routed.counts)

    def execute_direct(
        self,
        experts: Any,
        tokens: Any,
        top_scores: Any,
        top_indices: Any,
    ) -> Any | None:
        _ = experts, tokens, top_scores, top_indices
        return None


class TorchChunkedKernelBackend:
    """Direct routed execution without a tokens-times-top-k materialization."""

    name = "torch_chunked"

    def execute_direct(
        self,
        experts: Any,
        tokens: Any,
        top_scores: Any,
        top_indices: Any,
    ) -> Any:
        runner = getattr(experts, "run_direct_routed", None)
        if not callable(runner):
            raise RuntimeError(
                "torch_chunked requires an expert provider implementing "
                "run_direct_routed(tokens, top_scores, top_indices)."
            )
        return runner(tokens, top_scores, top_indices)


class RotatedInt8KernelBackend:
    """Expert GEMM that rotates activations instead of un-rotating weights.

    Deletes the dense fp32 Hadamard rotation the dequant path pays on every
    forward/recompute/backward by moving it onto the (much smaller) activations,
    while keeping cuBLAS bmm. Accuracy matches the dequant path; it is not
    bit-identical to it, so it must be selected explicitly, never via 'auto'.
    """

    name = "rotated_int8"

    def execute_direct(
        self,
        experts: Any,
        tokens: Any,
        top_scores: Any,
        top_indices: Any,
    ) -> Any:
        runner = getattr(experts, "run_direct_routed_rotated_int8", None)
        if not callable(runner):
            raise RuntimeError(
                "rotated_int8 requires an expert provider implementing "
                "run_direct_routed_rotated_int8(tokens, top_scores, top_indices)."
            )
        return runner(tokens, top_scores, top_indices)


class CompiledPackedKernelBackend(TorchChunkedKernelBackend):
    """Direct routing whose packed decoder is fused by TorchInductor."""

    name = "compiled_packed"


def build_moe_kernel_backend(
    value: str | None,
    *,
    direct_routed: bool = False,
) -> (
    MegaBlocksKernelBackend
    | TorchChunkedKernelBackend
    | RotatedInt8KernelBackend
    | CompiledPackedKernelBackend
    | None
):
    backend = normalize_moe_kernel_backend(value)
    if backend == "grouped":
        raise ValueError(
            "memory.moe_kernel_backend='grouped' selects a model-family-owned "
            "grouped-GEMM expert execution seam and has no generic direct-routed "
            "backend; the family pipeline must build it."
        )
    if backend in {"auto", "torch"}:
        return TorchChunkedKernelBackend() if direct_routed else None
    if backend == "rotated_int8":
        if not direct_routed:
            raise ValueError("rotated_int8 requires a direct-routed expert provider.")
        return RotatedInt8KernelBackend()
    if backend == "compiled_packed":
        if not direct_routed:
            raise ValueError("compiled_packed requires direct-routed experts.")
        return CompiledPackedKernelBackend()
    return MegaBlocksKernelBackend()


def megablocks_runtime_available() -> bool:
    try:
        _import_megablocks_ops()
        _import_grouped_gemm_ops()
    except (ImportError, RuntimeError):
        return False
    return True


def _import_megablocks_ops() -> Any:
    try:
        return importlib.import_module("megablocks.ops")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "memory.moe_kernel_backend='megablocks' requires the Apache-2.0 "
            "MegaBlocks CUDA ops package."
        ) from exc


def _import_grouped_gemm_ops() -> Any:
    try:
        module = importlib.import_module("grouped_gemm")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "memory.moe_kernel_backend='megablocks' requires grouped_gemm CUDA ops."
        ) from exc
    ops = getattr(module, "ops", None)
    if ops is None or not callable(getattr(ops, "gmm", None)):
        raise RuntimeError("grouped_gemm is installed without the gmm CUDA operator.")
    return ops


def _torch() -> Any:
    return importlib.import_module("torch")
