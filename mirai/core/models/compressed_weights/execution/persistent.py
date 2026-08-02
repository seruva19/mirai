# SPDX-License-Identifier: Apache-2.0
"""Expert-sliced persistent grouped GEMM for chunk-streamed frozen weights."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from mirai.vendors.qwen3_moe_fused.autotuning import (
    GRID_FACTOR,
    get_autotune_configs,
    get_autotune_keys,
    get_num_sms,
    prune_configs,
)
from mirai.vendors.qwen3_moe_fused.forward import exceeds_smem_capacity, is_int_tensor

from .dispatch_preprocess import build_dispatch_plan
from .dispatch_ops import combine_routed_tokens, permute_routed_tokens
from mirai.core.moe.runtime.gemm import resolve_moe_gemm_backend, run_grouped_dx


def _prune_expert_slice_configs(configs, args, **kwargs):
    adjusted = dict(args)
    adjusted["NUM_EXPERTS"] = adjusted["TOTAL_EXPERTS"]
    return prune_configs(exceeds_smem_capacity, configs, adjusted, **kwargs)


@triton.autotune(
    configs=get_autotune_configs(),
    key=get_autotune_keys(),
    prune_configs_by={"early_config_prune": _prune_expert_slice_configs},
    cache_results=True,
)
@triton.jit
def _grouped_gemm_expert_slice_kernel(
    x_ptr,
    w_ptr,
    m_offsets_ptr,
    y_ptr,
    M: int,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOTAL_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    expert_start: int,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
    LOOP_ORDER: tl.constexpr = 0,
) -> None:
    """Process one fixed expert slice using global, device-resident offsets.

    Scheduling follows the Apache-2.0 persistent grouped-GEMM design from
    woct0rdho/transformers-qwen3-moe-fused. Unlike the upstream primitive, row
    offsets are absolute in the full sorted token layout while ``w_ptr`` holds
    only the current fixed expert slice.
    """
    tidx = tl.program_id(0)
    processed_tiles = 0
    for local_expert_idx in range(NUM_EXPERTS):
        global_expert_idx = expert_start + local_expert_idx
        has_previous = global_expert_idx > 0
        m_start = tl.load(
            m_offsets_ptr + global_expert_idx - 1,
            mask=has_previous,
            other=0,
        ).to(tl.int32)
        m_end = tl.load(m_offsets_ptr + global_expert_idx).to(tl.int32)
        m_size = m_end - m_start

        if m_size > 0:
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            while tidx >= processed_tiles and tidx < processed_tiles + num_tiles:
                tile_idx = tidx - processed_tiles
                if LOOP_ORDER == 0:
                    tile_m_idx = tile_idx % num_m_tiles
                    tile_n_idx = tile_idx // num_m_tiles
                else:
                    tile_m_idx = tile_idx // num_n_tiles
                    tile_n_idx = tile_idx % num_n_tiles

                offs_k = tl.arange(0, BLOCK_SIZE_K)
                offs_m = (
                    m_start
                    + tile_m_idx * BLOCK_SIZE_M
                    + tl.arange(0, BLOCK_SIZE_M)
                )
                x_ptrs = (
                    x_ptr
                    + stride_xm * offs_m[:, None]
                    + stride_xk * offs_k[None, :]
                )
                mask_m = offs_m < m_end

                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                w_ptrs = (
                    w_ptr
                    + stride_we * local_expert_idx
                    + stride_wn * offs_n[:, None]
                    + stride_wk * offs_k[None, :]
                )
                mask_n = offs_n < N
                accumulator = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                )

                for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
                    mask_k = offs_k < K
                    x = tl.load(
                        x_ptrs, mask=mask_m[:, None] & mask_k[None, :]
                    )
                    w = tl.load(
                        w_ptrs, mask=mask_n[:, None] & mask_k[None, :]
                    )
                    accumulator += tl.dot(x, w.T)
                    offs_k += BLOCK_SIZE_K
                    x_ptrs += stride_xk * BLOCK_SIZE_K
                    w_ptrs += stride_wk * BLOCK_SIZE_K

                y = accumulator.to(y_ptr.dtype.element_ty)
                y_ptrs = (
                    y_ptr
                    + stride_ym * offs_m[:, None]
                    + stride_yn * offs_n[None, :]
                )
                tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])
                tidx += NUM_SMS

            processed_tiles += num_tiles


def grouped_gemm_expert_slice(
    x: torch.Tensor,
    w: torch.Tensor,
    m_offsets: torch.Tensor,
    output: torch.Tensor,
    *,
    expert_start: int,
    total_experts: int,
    transpose_w: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Write one contiguous expert slice into a full sorted-layout output."""
    if not x.is_cuda or w.device != x.device or output.device != x.device:
        raise ValueError("Expert-sliced persistent GEMM requires one CUDA device.")
    if m_offsets.device != x.device or not is_int_tensor(m_offsets):
        raise ValueError("m_offsets must be an integer tensor on the input device.")
    if x.ndim != 2 or w.ndim != 3 or output.ndim != 2 or m_offsets.ndim != 1:
        raise ValueError("Expected x/output 2D, weights 3D, and offsets 1D.")
    if not x.is_contiguous() or not w.is_contiguous() or not output.is_contiguous():
        raise ValueError("Expert-sliced persistent GEMM tensors must be contiguous.")
    if not m_offsets.is_contiguous():
        raise ValueError("Expert offsets must be contiguous.")

    m, _ = x.shape
    local_experts, n, k = w.shape
    stride_we, stride_wn, stride_wk = w.stride()
    if transpose_w:
        n, k = k, n
        stride_wn, stride_wk = stride_wk, stride_wn
    if x.shape[1] != k or output.shape != (m, n):
        raise ValueError("Input, weight, and output dimensions are inconsistent.")
    if int(total_experts) != int(m_offsets.numel()):
        raise ValueError("total_experts must match the offsets length.")
    if int(expert_start) < 0 or int(expert_start) + local_experts > int(total_experts):
        raise ValueError("Expert slice is outside the global expert range.")
    if dtype is not None and output.dtype != dtype:
        raise ValueError("The provided output does not match the requested dtype.")
    if m == 0 or local_experts == 0:
        return output

    with torch.cuda.device(x.device):
        total_blocks = get_num_sms() * GRID_FACTOR
        grid = lambda meta: (total_blocks,)
        _grouped_gemm_expert_slice_kernel[grid](
            x,
            w,
            m_offsets,
            output,
            m,
            n,
            k,
            local_experts,
            int(total_experts),
            total_blocks,
            int(expert_start),
            x.stride(0),
            x.stride(1),
            stride_we,
            stride_wn,
            stride_wk,
            output.stride(0),
            output.stride(1),
        )
    return output


class FrozenPersistentStreamedLinear(torch.autograd.Function):
    """Persistent frozen linear over global offsets and fixed weight chunks."""

    @staticmethod
    def forward(ctx, x, owner, key, m_offsets, chunk_size):
        expert_count = int(owner.num_experts)
        chunk = max(1, min(int(chunk_size), expert_count))
        output = None
        for start in range(0, expert_count, chunk):
            stop = min(expert_count, start + chunk)
            expert_indices = tuple(range(start, stop))
            weight = owner._dequant_expert_stack(
                key, expert_indices, dtype=x.dtype, device=x.device
            ).contiguous()
            if output is None:
                output = torch.empty(
                    (x.shape[0], weight.shape[1]),
                    device=x.device,
                    dtype=x.dtype,
                )
            grouped_gemm_expert_slice(
                x.contiguous(),
                weight,
                m_offsets,
                output,
                expert_start=start,
                total_experts=expert_count,
            )
        if output is None:
            raise RuntimeError("Persistent expert execution requires experts.")
        ctx.owner = owner
        ctx.key = key
        ctx.chunk_size = chunk
        ctx.weight_dtype = x.dtype
        ctx.save_for_backward(m_offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (m_offsets,) = ctx.saved_tensors
        expert_count = int(ctx.owner.num_experts)
        dx_backend = resolve_moe_gemm_backend("dx", device=grad_output.device)
        if dx_backend != "auto" and dx_backend != "persistent":
            weight = ctx.owner._dequant_expert_stack(
                ctx.key,
                tuple(range(expert_count)),
                dtype=ctx.weight_dtype,
                device=grad_output.device,
            ).contiguous()
            grad_input = run_grouped_dx(
                grad_output.to(dtype=ctx.weight_dtype).contiguous(),
                weight,
                m_offsets,
                backend=dx_backend,
            )
            return grad_input, None, None, None, None
        grad_input = None
        grad_output = grad_output.to(dtype=ctx.weight_dtype).contiguous()
        for start in range(0, expert_count, ctx.chunk_size):
            stop = min(expert_count, start + ctx.chunk_size)
            expert_indices = tuple(range(start, stop))
            weight = ctx.owner._dequant_expert_stack(
                ctx.key,
                expert_indices,
                dtype=ctx.weight_dtype,
                device=grad_output.device,
            ).contiguous()
            if grad_input is None:
                grad_input = torch.empty(
                    (grad_output.shape[0], weight.shape[2]),
                    device=grad_output.device,
                    dtype=ctx.weight_dtype,
                )
            grouped_gemm_expert_slice(
                grad_output,
                weight,
                m_offsets,
                grad_input,
                expert_start=start,
                total_experts=expert_count,
                transpose_w=True,
            )
        if grad_input is None:
            raise RuntimeError("Persistent expert execution requires experts.")
        return grad_input, None, None, None, None


class FrozenPersistentStreamedLinearPair(torch.autograd.Function):
    """Two persistent frozen linears sharing fixed-chunk paired dequant."""

    @staticmethod
    def forward(ctx, x, owner, key_a, key_b, m_offsets, chunk_size):
        expert_count = int(owner.num_experts)
        chunk = max(1, min(int(chunk_size), expert_count))
        out_a = None
        out_b = None
        x_contiguous = x.contiguous()
        for start in range(0, expert_count, chunk):
            stop = min(expert_count, start + chunk)
            expert_indices = tuple(range(start, stop))
            weight_a, weight_b = owner._dequant_expert_pair(
                key_a,
                key_b,
                expert_indices,
                dtype=x.dtype,
                device=x.device,
            )
            weight_a = weight_a.contiguous()
            weight_b = weight_b.contiguous()
            if out_a is None:
                shape = (x.shape[0], weight_a.shape[1])
                out_a = torch.empty(shape, device=x.device, dtype=x.dtype)
                out_b = torch.empty(shape, device=x.device, dtype=x.dtype)
            grouped_gemm_expert_slice(
                x_contiguous,
                weight_a,
                m_offsets,
                out_a,
                expert_start=start,
                total_experts=expert_count,
            )
            grouped_gemm_expert_slice(
                x_contiguous,
                weight_b,
                m_offsets,
                out_b,
                expert_start=start,
                total_experts=expert_count,
            )
        if out_a is None or out_b is None:
            raise RuntimeError("Persistent expert execution requires experts.")
        ctx.owner = owner
        ctx.key_a = key_a
        ctx.key_b = key_b
        ctx.chunk_size = chunk
        ctx.weight_dtype = x.dtype
        ctx.save_for_backward(m_offsets)
        return out_a, out_b

    @staticmethod
    def backward(ctx, grad_out_a, grad_out_b):
        (m_offsets,) = ctx.saved_tensors
        expert_count = int(ctx.owner.num_experts)
        dx_backend = resolve_moe_gemm_backend("dx", device=grad_out_a.device)
        if dx_backend != "auto" and dx_backend != "persistent":
            experts = tuple(range(expert_count))
            weight_a, weight_b = ctx.owner._dequant_expert_pair(
                ctx.key_a,
                ctx.key_b,
                experts,
                dtype=ctx.weight_dtype,
                device=grad_out_a.device,
            )
            grad_a = run_grouped_dx(
                grad_out_a.to(dtype=ctx.weight_dtype).contiguous(),
                weight_a.contiguous(),
                m_offsets,
                backend=dx_backend,
            )
            grad_b = run_grouped_dx(
                grad_out_b.to(dtype=ctx.weight_dtype).contiguous(),
                weight_b.contiguous(),
                m_offsets,
                backend=dx_backend,
            )
            return grad_a + grad_b, None, None, None, None, None
        grad_a = None
        grad_b = None
        grad_out_a = grad_out_a.to(dtype=ctx.weight_dtype).contiguous()
        grad_out_b = grad_out_b.to(dtype=ctx.weight_dtype).contiguous()
        for start in range(0, expert_count, ctx.chunk_size):
            stop = min(expert_count, start + ctx.chunk_size)
            expert_indices = tuple(range(start, stop))
            weight_a, weight_b = ctx.owner._dequant_expert_pair(
                ctx.key_a,
                ctx.key_b,
                expert_indices,
                dtype=ctx.weight_dtype,
                device=grad_out_a.device,
            )
            weight_a = weight_a.contiguous()
            weight_b = weight_b.contiguous()
            if grad_a is None:
                shape = (grad_out_a.shape[0], weight_a.shape[2])
                grad_a = torch.empty(
                    shape, device=grad_out_a.device, dtype=ctx.weight_dtype
                )
                grad_b = torch.empty(
                    shape, device=grad_out_a.device, dtype=ctx.weight_dtype
                )
            grouped_gemm_expert_slice(
                grad_out_a,
                weight_a,
                m_offsets,
                grad_a,
                expert_start=start,
                total_experts=expert_count,
                transpose_w=True,
            )
            grouped_gemm_expert_slice(
                grad_out_b,
                weight_b,
                m_offsets,
                grad_b,
                expert_start=start,
                total_experts=expert_count,
                transpose_w=True,
            )
        if grad_a is None or grad_b is None:
            raise RuntimeError("Persistent expert execution requires experts.")
        return grad_a + grad_b, None, None, None, None, None


def _adapter_output(
    owner, key, x, expert_indices, counts, m_offsets, route_gate=None
):
    adapter = owner.expert_lora[key] if key in owner.expert_lora else None
    if adapter is None:
        return None
    return adapter.sorted_subset_forward(
        x,
        expert_indices=expert_indices,
        m_offsets=m_offsets,
        counts=counts,
        route_gate=route_gate,
    )


def streamed_linear(
    owner,
    key: str,
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    counts: torch.Tensor,
    m_offsets: torch.Tensor,
    *,
    chunk_size: int,
    route_gate=None,
) -> torch.Tensor:
    output = FrozenPersistentStreamedLinear.apply(
        x, owner, key, m_offsets, int(chunk_size)
    )
    adapter = _adapter_output(
        owner, key, x, expert_indices, counts, m_offsets, route_gate
    )
    return output if adapter is None else output + adapter


def streamed_linear_pair(
    owner,
    key_a: str,
    key_b: str,
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    counts: torch.Tensor,
    m_offsets: torch.Tensor,
    *,
    chunk_size: int,
    route_gate=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_a, out_b = FrozenPersistentStreamedLinearPair.apply(
        x, owner, key_a, key_b, m_offsets, int(chunk_size)
    )
    adapter_a = _adapter_output(
        owner, key_a, x, expert_indices, counts, m_offsets, route_gate
    )
    adapter_b = _adapter_output(
        owner, key_b, x, expert_indices, counts, m_offsets, route_gate
    )
    if adapter_a is not None:
        out_a = out_a + adapter_a
    if adapter_b is not None:
        out_b = out_b + adapter_b
    return out_a, out_b


def run_persistent_dispatch(
    owner,
    tokens: torch.Tensor,
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    chunk_size: int,
    output: torch.Tensor,
    routed_adapter_gate=None,
) -> torch.Tensor:
    plan = build_dispatch_plan(top_indices, top_scores, owner.num_experts)
    return run_persistent_dispatch_plan(
        owner,
        tokens,
        plan,
        chunk_size=chunk_size,
        output=output,
        routed_adapter_gate=routed_adapter_gate,
    )


def run_persistent_dispatch_plan(
    owner,
    tokens: torch.Tensor,
    plan,
    *,
    chunk_size: int,
    output: torch.Tensor,
    routed_adapter_gate=None,
) -> torch.Tensor:
    """Run a prebuilt sorted dispatch plan through streamed grouped GEMM."""
    sorted_token_indices = plan.sorted_token_indices
    sorted_scores = plan.sorted_scores
    counts = plan.counts
    # Kernel contract: contiguous int32 inclusive-cumsum m_offsets with
    # numel() == num_experts and m_offsets[-1] == total_slots.
    m_offsets = plan.offsets
    expert_ids = torch.arange(
        owner.num_experts, device=tokens.device, dtype=torch.long
    )
    sorted_tokens = permute_routed_tokens(tokens, sorted_token_indices)
    route_gate = (
        None
        if routed_adapter_gate is None
        else routed_adapter_gate.resolve_sorted(sorted_token_indices, counts)
    )
    gate, up = streamed_linear_pair(
        owner,
        "w1",
        "w3",
        sorted_tokens,
        expert_ids,
        counts,
        m_offsets,
        chunk_size=chunk_size,
        route_gate=route_gate,
    )
    hidden = F.silu(gate) * up
    owner._capture_routed_intermediates(hidden, plan.sort_positions)
    expert_output = streamed_linear(
        owner,
        "w2",
        hidden,
        expert_ids,
        counts,
        m_offsets,
        chunk_size=chunk_size,
        route_gate=route_gate,
    )
    owner._capture_routed_outputs(expert_output, plan.sort_positions)
    combined = combine_routed_tokens(
        expert_output,
        sorted_scores,
        sorted_token_indices,
        output_rows=int(tokens.shape[0]),
        output_dtype=tokens.dtype,
    )
    return output.to(dtype=tokens.dtype) + combined
