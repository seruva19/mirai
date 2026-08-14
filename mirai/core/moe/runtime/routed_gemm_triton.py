# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for resident-BF16 routed grouped projections.

The blocked matrix multiplication follows Triton's public matrix-multiplication
tutorial. Routing adds optional indexed row loads and stores; group ownership is
resolved from cumulative row boundaries by a fixed-depth binary search.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import torch

from .routed_gemm_autotune import (
    RoutedGemmAutotuner,
    RoutedGemmBenchmarkResult,
    RoutedGemmKernelConfig,
    build_routed_gemm_environment_fingerprint,
    build_routed_gemm_shape_key,
)


_AUTOTUNERS: dict[tuple[str, str], RoutedGemmAutotuner] = {}
_TMA_ALLOCATOR_READY = False


def _ensure_tma_allocator() -> None:
    global _TMA_ALLOCATOR_READY
    if _TMA_ALLOCATOR_READY:
        return
    import triton
    from triton.runtime import _allocation

    if type(_allocation._allocator.get()).__name__ == "NullAllocator":
        triton.set_allocator(
            lambda size, alignment, stream: torch.empty(
                int(size), device=torch.cuda.current_device(), dtype=torch.uint8
            )
        )
    _TMA_ALLOCATOR_READY = True


def _conservative_config(n_size: int) -> RoutedGemmKernelConfig:
    return RoutedGemmKernelConfig(32 if n_size <= 32 else 64, 32, 4, 2, False)


def _active_autotuner(activation: torch.Tensor, n_size: int) -> RoutedGemmAutotuner:
    from .specs import get_active_moe_optimization_policy

    policy = get_active_moe_optimization_policy()
    mode = policy.moe_routed_gemm_tuning
    path = policy.moe_routed_gemm_cache_path
    identity = (mode, path)
    tuner = _AUTOTUNERS.get(identity)
    if tuner is None:
        environment = None
        if mode != "off":
            environment = build_routed_gemm_environment_fingerprint(
                activation, kernel_abi_fingerprint="routed-grouped-mm-v2"
            )
        tuner = RoutedGemmAutotuner(
            mode=mode, environment_fingerprint=environment,
            cache_path=path or None, conservative_config=_conservative_config(n_size),
        )
        _AUTOTUNERS[identity] = tuner
    return tuner


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def routed_grouped_mm(
        x, w, boundaries, row_map, out_map, y,
        stride_wg: tl.constexpr, stride_wk: tl.constexpr, stride_wn: tl.constexpr,
        rows: tl.constexpr, groups: tl.constexpr, k_size: tl.constexpr,
        n_size: tl.constexpr, gather: tl.constexpr, scatter: tl.constexpr,
        SEARCH_STEPS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        block_n = tl.program_id(1)
        lo = tl.zeros((), tl.int32)
        hi = tl.full((), groups, tl.int32)
        for _ in range(SEARCH_STEPS):
            mid = (lo + hi) // 2
            active = lo < hi
            boundary = tl.load(boundaries + tl.minimum(mid, groups - 1), mask=active, other=rows)
            take_left = row < boundary
            hi = tl.where(active & take_left, mid, hi)
            lo = tl.where(active & ~take_left, mid + 1, lo)
        group = lo
        source = tl.load(row_map + row) if gather else row
        destination = tl.load(out_map + row) if scatter else row
        ns = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for k0 in range(0, k_size, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            xv = tl.load(x + source * k_size + ks, mask=ks < k_size, other=0.0)
            wv = tl.load(
                w + group * stride_wg + ks[:, None] * stride_wk + ns[None, :] * stride_wn,
                mask=(ks[:, None] < k_size) & (ns[None, :] < n_size), other=0.0,
            )
            acc += tl.reshape(tl.dot(xv[None, :], wv), (BLOCK_N,))
        tl.store(y + destination * n_size + ns, acc, mask=ns < n_size)

    return routed_grouped_mm, triton


@cache
def _tiled_indexed_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tiled_routed_grouped_mm(
        x, w, boundaries, row_map, out_map, y,
        stride_wg: tl.constexpr, stride_wk: tl.constexpr, stride_wn: tl.constexpr,
        k_size: tl.constexpr, n_size: tl.constexpr,
        gather: tl.constexpr, scatter: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        group = tl.program_id(0)
        block_m = tl.program_id(1)
        block_n = tl.program_id(2)
        start = tl.load(boundaries + group - 1, mask=group > 0, other=0)
        stop = tl.load(boundaries + group)
        row0 = start + block_m * BLOCK_M
        if row0 < stop:
            grouped_rows = row0 + tl.arange(0, BLOCK_M)
            valid_rows = grouped_rows < stop
            source_rows = (
                tl.load(row_map + grouped_rows, mask=valid_rows, other=0)
                if gather else grouped_rows
            )
            destination_rows = (
                tl.load(out_map + grouped_rows, mask=valid_rows, other=0)
                if scatter else grouped_rows
            )
            ns = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for k0 in range(0, k_size, BLOCK_K):
                ks = k0 + tl.arange(0, BLOCK_K)
                xv = tl.load(
                    x + source_rows[:, None] * k_size + ks[None, :],
                    mask=valid_rows[:, None] & (ks[None, :] < k_size),
                    other=0.0,
                )
                wv = tl.load(
                    w
                    + group * stride_wg
                    + ks[:, None] * stride_wk
                    + ns[None, :] * stride_wn,
                    mask=(ks[:, None] < k_size) & (ns[None, :] < n_size),
                    other=0.0,
                )
                acc += tl.dot(xv, wv)
            tl.store(
                y + destination_rows[:, None] * n_size + ns[None, :],
                acc,
                mask=valid_rows[:, None] & (ns[None, :] < n_size),
            )

    return tiled_routed_grouped_mm, triton


@cache
def _tma_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def regular_grouped_mm_tma(
        x, w, boundaries, y,
        rows: tl.constexpr, k_size: tl.constexpr, n_size: tl.constexpr,
        groups: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        group = tl.program_id(0)
        block_m = tl.program_id(1)
        block_n = tl.program_id(2)
        start = tl.load(boundaries + group - 1, mask=group > 0, other=0)
        stop = tl.load(boundaries + group)
        row0 = start + block_m * BLOCK_M
        n0 = block_n * BLOCK_N
        x_desc = tl.make_tensor_descriptor(
            x, shape=[rows, k_size], strides=[k_size, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )
        w_desc = tl.make_tensor_descriptor(
            w, shape=[groups, k_size, n_size],
            strides=[k_size * n_size, n_size, 1],
            block_shape=[1, BLOCK_K, BLOCK_N],
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, k_size, BLOCK_K):
            k_offset = tl.full((), k0, tl.int32)
            xv = x_desc.load([row0.to(tl.int32), k_offset])
            wv = w_desc.load([
                group.to(tl.int32), k_offset, n0.to(tl.int32)
            ]).reshape((BLOCK_K, BLOCK_N))
            acc += tl.dot(xv, wv)
        ms = row0 + tl.arange(0, BLOCK_M)
        ns = n0 + tl.arange(0, BLOCK_N)
        tl.store(
            y + ms[:, None] * n_size + ns[None, :],
            acc,
            mask=(ms[:, None] < stop) & (ms[:, None] < rows) & (ns[None, :] < n_size),
        )

    return regular_grouped_mm_tma, triton


def _regular_tma_support(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    gather: bool,
    scatter: bool,
    role: str,
) -> tuple[bool, str]:
    """Return whether host tensor descriptors can represent this projection."""

    if gather or scatter:
        return False, "indexed gather/scatter requires the pointer architecture"
    if role != "forward":
        return False, "the TMA specialization currently supports the forward role"
    if not activation.is_contiguous() or not weight.is_contiguous():
        return False, "TMA operands must be contiguous"
    element_bytes = activation.element_size()
    if (activation.stride(0) * element_bytes) % 16:
        return False, "TMA activation leading stride must be 16-byte aligned"
    if any((weight.stride(axis) * weight.element_size()) % 16 for axis in (0, 1)):
        return False, "TMA weight leading strides must be 16-byte aligned"
    capability = torch.cuda.get_device_capability(activation.device)
    if capability[0] != 9:
        return False, "TMA requires Hopper"
    return True, "contiguous non-indexed Hopper projection"


def _launch_regular_grouped_mm_tma(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    rows: int,
    config: RoutedGemmKernelConfig,
) -> torch.Tensor:
    _ensure_tma_allocator()
    groups, k_size, n_size = map(int, weight.shape)
    block_m = 32
    block_k = min(64, max(32, config.block_k))
    block_n = min(128, max(32, config.block_n))
    output = activation.new_empty((rows, n_size))
    kernel, triton = _tma_kernel()
    # Every group gets enough row tiles for the worst possible distribution.
    # The boundary mask prevents tiles from committing rows owned by a neighbor.
    kernel[(groups, triton.cdiv(rows, block_m), triton.cdiv(n_size, block_n))](
        activation, weight, boundaries, output,
        rows=rows, k_size=k_size, n_size=n_size, groups=groups,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=config.num_warps, num_stages=config.num_stages,
    )
    return output


@cache
def _reduction_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def weighted_reduce(grouped, grouped_to_assignment, assignment_to_token,
                        coefficients, output, rows: tl.constexpr,
                        width: tl.constexpr, BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        ns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        assignment = tl.load(grouped_to_assignment + row)
        token = tl.load(assignment_to_token + assignment)
        coefficient = tl.load(coefficients + assignment).to(tl.float32)
        value = tl.load(grouped + row * width + ns, mask=ns < width, other=0.0).to(tl.float32)
        tl.atomic_add(output + token * width + ns, value * coefficient, mask=ns < width)

    return weighted_reduce, triton


@cache
def _dw_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def grouped_dw(x, dy, boundaries, gather_rows, dw,
                   rows_total: tl.constexpr, k_size: tl.constexpr, n_size: tl.constexpr,
                   gather: tl.constexpr, BLOCK_M: tl.constexpr,
                   BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
        group = tl.program_id(0)
        ks = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        ns = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
        start = tl.load(boundaries + group - 1, mask=group > 0, other=0)
        stop = tl.load(boundaries + group)
        acc = tl.zeros((BLOCK_K, BLOCK_N), tl.float32)
        for row0 in range(0, rows_total, BLOCK_M):
            rows = row0 + tl.arange(0, BLOCK_M)
            mask_rows = (rows >= start) & (rows < stop)
            source = tl.load(gather_rows + rows, mask=mask_rows, other=0) if gather else rows
            xv = tl.load(
                x + source[:, None] * k_size + ks[None, :],
                mask=mask_rows[:, None] & (ks[None, :] < k_size), other=0.0,
            )
            dyv = tl.load(
                dy + rows[:, None] * n_size + ns[None, :],
                mask=mask_rows[:, None] & (ns[None, :] < n_size), other=0.0,
            )
            acc += tl.dot(tl.trans(xv), dyv)
        tl.store(
            dw + group * k_size * n_size + ks[:, None] * n_size + ns[None, :],
            acc, mask=(ks[:, None] < k_size) & (ns[None, :] < n_size),
        )

    return grouped_dw, triton


def triton_grouped_dw(
    activation: torch.Tensor,
    grad_grouped: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    groups: int,
    gather_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped dW with one device launch and FP32 tensor-core accumulation."""
    k_size = int(activation.shape[1])
    n_size = int(grad_grouped.shape[1])
    output = activation.new_empty((groups, k_size, n_size))
    empty = torch.empty(0, dtype=torch.int64, device=activation.device)
    gather = gather_rows is not None
    tuner = _active_autotuner(activation, n_size)
    config = tuner.conservative_config
    if tuner.mode != "off":
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("routed GEMM dW tuning must finish before CUDA Graph capture")
        counts = torch.diff(torch.cat((boundaries.new_zeros(1), boundaries)))
        key = build_routed_gemm_shape_key(
            activation, output, counts, implementation="indexed_sm80", role="dw",
            fusion="gather" if gather else "grouped", top_k=1,
        )

        def benchmark(candidate: RoutedGemmKernelConfig) -> RoutedGemmBenchmarkResult:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            trial = _launch_grouped_dw(
                activation, grad_grouped, boundaries,
                gather_rows if gather else empty, groups, candidate, gather=gather,
            )
            end.record(); end.synchronize()
            del trial
            return RoutedGemmBenchmarkResult(candidate, start.elapsed_time(end) * 1000.0, 1)

        reference = _launch_grouped_dw(
            activation, grad_grouped, boundaries,
            gather_rows if gather else empty, groups, _conservative_config(n_size),
            gather=gather,
        )

        def verify(winner: RoutedGemmBenchmarkResult) -> None:
            candidate_output = _launch_grouped_dw(
                activation, grad_grouped, boundaries,
                gather_rows if gather else empty, groups, winner.config, gather=gather,
            )
            torch.testing.assert_close(candidate_output, reference, rtol=2e-2, atol=2e-2)

        config = tuner.resolve_config(key, benchmark=benchmark, verify=verify)
    return _launch_grouped_dw(
        activation, grad_grouped, boundaries,
        gather_rows if gather else empty, groups, config, gather=gather,
    )


def _launch_grouped_dw(
    activation: torch.Tensor, grad_grouped: torch.Tensor,
    boundaries: torch.Tensor, gather_rows: torch.Tensor, groups: int,
    config: RoutedGemmKernelConfig, *, gather: bool,
) -> torch.Tensor:
    k_size = int(activation.shape[1])
    n_size = int(grad_grouped.shape[1])
    output = activation.new_empty((groups, k_size, n_size))
    kernel, triton = _dw_kernel()
    block_k = min(32, config.block_k)
    block_n = min(64, config.block_n)
    kernel[(groups, triton.cdiv(k_size, block_k), triton.cdiv(n_size, block_n))](
        activation, grad_grouped, boundaries,
        gather_rows, output,
        rows_total=int(grad_grouped.shape[0]), k_size=k_size, n_size=n_size, gather=gather,
        BLOCK_M=32, BLOCK_K=block_k, BLOCK_N=block_n,
        num_warps=config.num_warps, num_stages=config.num_stages,
    )
    return output


def _weighted_reduce(
    grouped: torch.Tensor,
    grouped_to_assignment: torch.Tensor,
    assignment_to_token: torch.Tensor,
    coefficients: torch.Tensor,
    token_rows: int,
) -> torch.Tensor:
    # FP32 atomics make accumulation order explicit and avoid BF16 loss during
    # top-k combination. Conversion follows after every routed row is reduced.
    output_fp32 = torch.zeros(
        token_rows, grouped.shape[1], device=grouped.device, dtype=torch.float32
    )
    kernel, triton = _reduction_kernel()
    block_n = 64
    kernel[(grouped.shape[0], triton.cdiv(grouped.shape[1], block_n))](
        grouped, grouped_to_assignment, assignment_to_token, coefficients.reshape(-1),
        output_fp32, rows=grouped.shape[0], width=grouped.shape[1],
        BLOCK_N=block_n, num_warps=4,
    )
    return output_fp32.to(grouped.dtype)


def triton_routed_grouped_mm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    gather_rows: torch.Tensor | None = None,
    scatter_rows: torch.Tensor | None = None,
    output_rows: int | None = None,
    role: str = "forward",
    top_k: int = 1,
    segmented: bool = False,
    max_group_rows: int | None = None,
) -> torch.Tensor:
    """Launch the resident-BF16 routed projection on the current CUDA stream."""

    if gather_rows is not None and scatter_rows is not None:
        raise ValueError(
            "one routed projection cannot combine input gather and output scatter"
        )

    if activation.ndim != 2 or weight.ndim != 3:
        raise ValueError("routed Triton projection requires rank-2 activation and rank-3 weight")
    if activation.device.type != "cuda" or weight.device != activation.device:
        raise ValueError("routed Triton projection requires CUDA operands on one device")
    if activation.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("routed Triton projection requires BF16 activation and weight")
    if not activation.is_contiguous():
        raise ValueError("routed Triton projection requires contiguous activations")
    if int(activation.shape[1]) != int(weight.shape[1]):
        raise ValueError("routed Triton projection reduction widths must match")
    if boundaries.ndim != 1 or int(boundaries.numel()) != int(weight.shape[0]):
        raise ValueError("routed boundaries must be rank-1 with one entry per group")
    if boundaries.dtype not in (torch.int32, torch.int64):
        raise TypeError("routed boundaries must have int32 or int64 dtype")
    if boundaries.device != activation.device or not boundaries.is_contiguous():
        raise ValueError("routed boundaries must be contiguous on the operand device")
    # Routed rows are part of the operator's shape contract.  Reading the final
    # boundary on the host would synchronize CUDA and makes this launch unsafe
    # inside torch.compile or CUDA Graph capture.
    rows = int(
        output_rows
        if output_rows is not None
        else (
            gather_rows.shape[0]
            if gather_rows is not None
            else scatter_rows.shape[0]
            if scatter_rows is not None
            else activation.shape[0]
        )
    )
    groups, k_size, n_size = map(int, weight.shape)
    if max_group_rows is not None:
        max_group_rows = int(max_group_rows)
        if max_group_rows <= 0 or max_group_rows > rows:
            raise ValueError("max_group_rows must be in [1, routed_rows]")
    if groups == 0 and rows:
        raise ValueError("non-empty routed projection requires at least one group")
    for mapping in (gather_rows, scatter_rows):
        if mapping is None:
            continue
        if mapping.ndim != 1 or int(mapping.numel()) != rows:
            raise ValueError("routed row mapping must be rank-1 with one entry per routed row")
        if mapping.dtype not in (torch.int32, torch.int64):
            raise TypeError("routed row mapping must have int32 or int64 dtype")
        if mapping.device != activation.device:
            raise ValueError("routed row mapping must share the operand device")
    if rows == 0:
        return (
            activation.new_empty((0, n_size))
            + activation.sum() * 0
            + weight.sum() * 0
        )
    torch._assert_async(boundaries[0] >= 0, "routed boundaries must be non-negative")
    if groups > 1:
        torch._assert_async(
            torch.all(boundaries[1:] >= boundaries[:-1]),
            "routed boundaries must be non-decreasing",
        )
    torch._assert_async(
        boundaries[-1] == rows,
        "terminal routed boundary must equal the routed-row count",
    )
    if max_group_rows is not None:
        starts = torch.cat((boundaries.new_zeros(1), boundaries[:-1]))
        torch._assert_async(
            torch.all(boundaries - starts <= max_group_rows),
            "max_group_rows must bound every routed group",
        )
    if gather_rows is not None:
        torch._assert_async(
            torch.all((gather_rows >= 0) & (gather_rows < int(activation.shape[0]))),
            "routed row mapping contains an out-of-range index",
        )
    if scatter_rows is not None:
        torch._assert_async(
            torch.all(
                torch.sort(scatter_rows.to(torch.int64)).values
                == torch.arange(rows, device=scatter_rows.device, dtype=torch.int64)
            ),
            "routed scatter mapping must be a permutation",
        )
    empty = torch.empty(0, dtype=torch.int64, device=activation.device)
    gather = gather_rows is not None
    scatter = scatter_rows is not None
    row_map = gather_rows.contiguous() if gather else empty
    out_map = scatter_rows.contiguous() if scatter else empty
    from .specs import get_active_moe_optimization_policy
    policy = get_active_moe_optimization_policy()
    tma_supported, tma_reason = _regular_tma_support(
        activation, weight, gather=gather, scatter=scatter, role=role
    )
    # Indexed routing requires pointer addressing; regular TMA accepts only
    # contiguous, non-indexed forward projections.
    use_tma = policy.moe_routed_gemm_architecture == "tma_regular"
    if policy.moe_routed_gemm_architecture == "tma_regular" and not tma_supported:
        raise RuntimeError(
            "memory.moe_routed_gemm_architecture='tma_regular' is unavailable: "
            + tma_reason
        )
    tuner = _active_autotuner(activation, n_size)
    config = tuner.conservative_config
    if tuner.mode != "off":
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "routed GEMM tuning/cache resolution must finish before CUDA Graph capture"
            )
        counts = torch.diff(
            torch.cat((boundaries.new_zeros(1), boundaries))
        )
        implementation = "regular_tma_sm90" if use_tma else "indexed_sm80"
        fusion = "gather" if gather else "scatter" if scatter else "grouped"
        key = build_routed_gemm_shape_key(
            activation, weight, counts, implementation=implementation,
            role=role, fusion=fusion, top_k=top_k, segmented=segmented,
        )

        def benchmark(candidate: RoutedGemmKernelConfig) -> RoutedGemmBenchmarkResult:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            trial = (
                _launch_regular_grouped_mm_tma(
                    activation, weight, boundaries, rows, candidate
                )
                if use_tma else
                _launch_routed_grouped_mm(
                    activation, weight, boundaries, row_map, out_map, rows, candidate,
                    gather=gather, scatter=scatter, max_group_rows=max_group_rows,
                )
            )
            end.record()
            end.synchronize()
            del trial
            return RoutedGemmBenchmarkResult(candidate, start.elapsed_time(end) * 1000.0, 1)

        reference = (
            _launch_regular_grouped_mm_tma(
                activation, weight, boundaries, rows, _conservative_config(n_size)
            )
            if use_tma else
            _launch_routed_grouped_mm(
                activation, weight, boundaries, row_map, out_map, rows,
                _conservative_config(n_size), gather=gather, scatter=scatter,
                max_group_rows=max_group_rows,
            )
        )

        def verify(winner: RoutedGemmBenchmarkResult) -> None:
            candidate_output = (
                _launch_regular_grouped_mm_tma(
                    activation, weight, boundaries, rows, winner.config
                )
                if use_tma else
                _launch_routed_grouped_mm(
                    activation, weight, boundaries, row_map, out_map, rows,
                    winner.config, gather=gather, scatter=scatter,
                    max_group_rows=max_group_rows,
                )
            )
            torch.testing.assert_close(candidate_output, reference, rtol=2e-2, atol=2e-2)

        config = tuner.resolve_config(key, benchmark=benchmark, verify=verify)
    if use_tma:
        return _launch_regular_grouped_mm_tma(
            activation, weight, boundaries, rows, config
        )
    return _launch_routed_grouped_mm(
        activation, weight, boundaries, row_map, out_map, rows, config,
        gather=gather, scatter=scatter, max_group_rows=max_group_rows,
    )


def _launch_routed_grouped_mm(
    activation: torch.Tensor, weight: torch.Tensor, boundaries: torch.Tensor,
    row_map: torch.Tensor, out_map: torch.Tensor, rows: int,
    config: RoutedGemmKernelConfig, *, gather: bool, scatter: bool,
    max_group_rows: int | None = None,
) -> torch.Tensor:
    groups, k_size, n_size = map(int, weight.shape)
    output = activation.new_empty((rows, n_size))
    if _use_tiled_indexed_kernel(
        rows=rows, groups=groups, k_size=k_size, n_size=n_size,
        max_group_rows=max_group_rows,
    ):
        kernel, triton = _tiled_indexed_kernel()
        block_m = 64
        block_n = min(128, max(32, triton.next_power_of_2(n_size)))
        block_k = min(32, max(16, triton.next_power_of_2(k_size)))
        grid_rows = rows if max_group_rows is None else max_group_rows
        kernel[(groups, triton.cdiv(grid_rows, block_m), triton.cdiv(n_size, block_n))](
            activation, weight, boundaries, row_map, out_map, output,
            stride_wg=int(weight.stride(0)), stride_wk=int(weight.stride(1)),
            stride_wn=int(weight.stride(2)),
            k_size=k_size, n_size=n_size,
            gather=gather, scatter=scatter,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=8, num_stages=3,
        )
        return output
    kernel, triton = _kernel()
    block_n = min(config.block_n, max(32, triton.next_power_of_2(n_size)))
    kernel[(rows, triton.cdiv(n_size, block_n))](
        activation, weight, boundaries, row_map, out_map, output,
        stride_wg=int(weight.stride(0)), stride_wk=int(weight.stride(1)),
        stride_wn=int(weight.stride(2)),
        rows=rows, groups=groups, k_size=k_size, n_size=n_size,
        gather=gather, scatter=scatter, SEARCH_STEPS=max(1, groups.bit_length()),
        BLOCK_N=block_n, BLOCK_K=config.block_k,
        num_warps=config.num_warps, num_stages=config.num_stages,
    )
    return output


def _use_tiled_indexed_kernel(
    *, rows: int, groups: int, k_size: int, n_size: int,
    max_group_rows: int | None = None,
) -> bool:
    """Select 2-D tiles when inactive group tiles remain tightly bounded."""

    return (
        1 <= groups
        and (groups <= 32 or max_group_rows is not None)
        and rows >= 64
        and k_size >= 64
        and n_size >= 64
    )


class _RoutedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, activation, weight, boundaries, gather_rows, scatter_rows):
        ctx.save_for_backward(activation, weight, boundaries, gather_rows, scatter_rows)
        ctx.input_rows = int(activation.shape[0])
        return triton_routed_grouped_mm(
            activation, weight, boundaries,
            gather_rows=gather_rows if gather_rows.numel() else None,
            scatter_rows=scatter_rows if scatter_rows.numel() else None,
        )

    @staticmethod
    def backward(ctx: Any, grad_output):
        activation, weight, boundaries, gather_rows, scatter_rows = ctx.saved_tensors
        grouped_grad = grad_output
        if scatter_rows.numel():
            grouped_grad = grad_output.index_select(0, scatter_rows.to(torch.int64))
        grad_input = None
        if ctx.needs_input_grad[0]:
            grouped_dx = triton_routed_grouped_mm(
                grouped_grad.contiguous(), weight.transpose(-2, -1), boundaries,
                role="dx",
            )
            if gather_rows.numel():
                grad_input = grouped_dx.new_zeros((ctx.input_rows, grouped_dx.shape[1]))
                grad_input.index_add_(0, gather_rows.to(torch.int64), grouped_dx)
            else:
                grad_input = grouped_dx
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = triton_grouped_dw(
                activation, grouped_grad.contiguous(), boundaries,
                groups=int(weight.shape[0]),
                gather_rows=gather_rows if gather_rows.numel() else None,
            )
        return grad_input, grad_weight, None, None, None


def routed_projection(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    gather_rows: torch.Tensor | None = None,
    scatter_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    from .routed_gemm_ops import routed_projection_op

    empty = torch.empty(0, dtype=torch.int64, device=activation.device)
    gather = gather_rows if gather_rows is not None else empty
    scatter = scatter_rows if scatter_rows is not None else empty
    rows = (
        gather.shape[0] if gather.numel() else
        scatter.shape[0] if scatter.numel() else activation.shape[0]
    )
    return routed_projection_op(
        activation, weight, boundaries,
        gather, scatter, rows,
    )


class _RoutedWeightedProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, activation, weight, boundaries, grouped_to_assignment,
        assignment_to_token, coefficients, token_rows, max_group_rows,
    ):
        grouped = triton_routed_grouped_mm(
            activation,
            weight,
            boundaries,
            max_group_rows=(None if int(max_group_rows) <= 0 else int(max_group_rows)),
        )
        output = _weighted_reduce(
            grouped, grouped_to_assignment, assignment_to_token, coefficients,
            int(token_rows),
        )
        ctx.save_for_backward(
            activation, weight, boundaries, grouped_to_assignment,
            assignment_to_token, coefficients, grouped,
        )
        ctx.max_group_rows = int(max_group_rows)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output):
        (activation, weight, boundaries, grouped_to_assignment,
         assignment_to_token, coefficients, grouped) = ctx.saved_tensors
        assignment = grouped_to_assignment.to(torch.int64)
        token_for_grouped = assignment_to_token.index_select(0, assignment).to(torch.int64)
        coefficient_for_grouped = coefficients.reshape(-1).index_select(0, assignment)
        grouped_grad = grad_output.index_select(0, token_for_grouped)
        grouped_grad = grouped_grad * coefficient_for_grouped[:, None].to(grouped_grad.dtype)
        grad_input = triton_routed_grouped_mm(
            grouped_grad.contiguous(), weight.transpose(-2, -1), boundaries,
            role="dx",
            max_group_rows=(
                None if ctx.max_group_rows <= 0 else ctx.max_group_rows
            ),
        ) if ctx.needs_input_grad[0] else None
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = triton_grouped_dw(
                activation, grouped_grad.contiguous(), boundaries,
                groups=int(weight.shape[0]),
            )
        grad_coefficients = None
        if ctx.needs_input_grad[5]:
            contribution = (grouped.float() * grad_output.index_select(0, token_for_grouped).float()).sum(1)
            grad_coefficients = torch.zeros_like(coefficients).reshape(-1)
            grad_coefficients.index_copy_(0, assignment, contribution.to(coefficients.dtype))
            grad_coefficients = grad_coefficients.view_as(coefficients)
        return (
            grad_input, grad_weight, None, None, None, grad_coefficients, None, None
        )


def routed_weighted_projection(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    grouped_to_assignment: torch.Tensor,
    assignment_to_token: torch.Tensor,
    coefficients: torch.Tensor,
    token_rows: int,
    max_group_rows: int | None = None,
) -> torch.Tensor:
    """Final projection plus routing-weighted token reduction with autograd."""
    rows = int(activation.shape[0]) if activation.ndim == 2 else -1
    if token_rows < 0:
        raise ValueError("token_rows must be non-negative")
    for name, mapping in (
        ("grouped_to_assignment", grouped_to_assignment),
        ("assignment_to_token", assignment_to_token),
    ):
        if mapping.ndim != 1:
            raise ValueError(f"{name} must be rank-1")
        if mapping.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must have int32 or int64 dtype")
        if mapping.device != activation.device or not mapping.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the operand device")
    if int(grouped_to_assignment.numel()) != rows:
        raise ValueError("grouped_to_assignment must contain one entry per routed row")
    if coefficients.device != activation.device:
        raise ValueError("routing coefficients must share the operand device")
    if not coefficients.is_contiguous():
        raise ValueError("routing coefficients must be contiguous")
    if int(coefficients.numel()) != int(assignment_to_token.numel()):
        raise ValueError("routing coefficients and assignment_to_token must have equal length")
    if rows:
        torch._assert_async(
            torch.all(
                (grouped_to_assignment >= 0)
                & (grouped_to_assignment < int(assignment_to_token.numel()))
            ),
            "grouped_to_assignment contains an out-of-range assignment index",
        )
    if assignment_to_token.numel():
        torch._assert_async(
            torch.all((assignment_to_token >= 0) & (assignment_to_token < token_rows)),
            "assignment_to_token contains an out-of-range token index",
        )
    return _RoutedWeightedProjection.apply(
        activation, weight, boundaries, grouped_to_assignment,
        assignment_to_token, coefficients, int(token_rows),
        0 if max_group_rows is None else int(max_group_rows),
    )
