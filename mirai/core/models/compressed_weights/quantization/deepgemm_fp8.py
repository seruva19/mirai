# SPDX-License-Identifier: Apache-2.0
"""Optional DeepGEMM execution for Mirai's DeepSeek-style FP8 experts.

DeepGEMM owns the native grouped forward GEMM. Mirai retains the portable
blockwise implementation as the reference and deliberately computes frozen
weight Dgrad from a high-precision dequantized weight. The latter follows the
DeepSeek-V3 training recipe: block-scaled FP8 is used for forward GEMMs, while
activation gradients avoid the numerically fragile 128x128 Dgrad scaling.

DeepGEMM source: https://github.com/deepseek-ai/DeepGEMM
DeepSeek-V3 report: https://arxiv.org/abs/2412.19437
"""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - torch-less static analysis
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from .blockwise_fp8 import (
    BLOCKWISE_FP8_BLOCK_SIZE,
    BlockwiseFP8Meta,
    _quantize_activation_tiles,
    _validate_blockwise_fp8_layout,
    dequantize_blockwise_fp8_weight,
)


def _routed_group_ids(boundaries: "torch.Tensor") -> "torch.Tensor":
    """Return one expert id per unpadded grouped row."""
    ends = boundaries.to(dtype=torch.int64)
    starts = torch.cat((ends.new_zeros(1), ends[:-1]))
    counts = ends - starts
    return torch.arange(
        int(boundaries.numel()), device=boundaries.device, dtype=torch.int32
    ).repeat_interleave(counts)


def _aligned_routed_input(
    grouped_inputs: "torch.Tensor",
    boundaries: "torch.Tensor",
    alignment: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Pad each group separately for DeepGEMM's contiguous-M layout.

    Returns the aligned activation, its row-to-group ids, and the indices that
    recover the unpadded grouped rows from the aligned result.
    """
    ends = boundaries.detach().to("cpu", torch.int64).tolist()
    pieces = []
    group_pieces = []
    valid_pieces = []
    source_start = 0
    aligned_start = 0
    for group, source_stop_value in enumerate(ends):
        source_stop = int(source_stop_value)
        count = source_stop - source_start
        if count:
            aligned_count = math.ceil(count / alignment) * alignment
            piece = grouped_inputs[source_start:source_stop]
            if aligned_count != count:
                piece = F.pad(piece, (0, 0, 0, aligned_count - count))
            pieces.append(piece)
            group_pieces.append(
                torch.full(
                    (aligned_count,), group, device=grouped_inputs.device,
                    dtype=torch.int32,
                )
            )
            valid_pieces.append(
                torch.arange(
                    aligned_start, aligned_start + count,
                    device=grouped_inputs.device, dtype=torch.int64,
                )
            )
            aligned_start += aligned_count
        source_start = source_stop
    if not pieces:
        empty_rows = grouped_inputs.new_empty((0, grouped_inputs.shape[1]))
        return (
            empty_rows,
            torch.empty(0, device=grouped_inputs.device, dtype=torch.int32),
            torch.empty(0, device=grouped_inputs.device, dtype=torch.int64),
        )
    return torch.cat(pieces), torch.cat(group_pieces), torch.cat(valid_pieces)


def _require_deepgemm():
    try:
        import deep_gemm
        from deep_gemm.utils import get_mk_alignment_for_contiguous_layout
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "DeepGEMM FP8 execution requires an importable deep_gemm build."
        ) from exc
    kernel = getattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous", None)
    if not callable(kernel):
        raise RuntimeError(
            "The installed deep_gemm build does not provide "
            "m_grouped_fp8_gemm_nt_contiguous."
        )
    return deep_gemm, get_mk_alignment_for_contiguous_layout


def _as_e4m3(codes: "torch.Tensor") -> "torch.Tensor":
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("DeepGEMM FP8 execution requires torch.float8_e4m3fn.")
    return codes.contiguous().view(torch.float8_e4m3fn)


def _native_grouped_forward(
    inputs: "torch.Tensor",
    codes: "torch.Tensor",
    scales: "torch.Tensor",
    meta: BlockwiseFP8Meta,
) -> "torch.Tensor":
    _validate_blockwise_fp8_layout(codes, scales, meta)
    if inputs.ndim != 3 or codes.ndim != 3 or scales.ndim != 3:
        raise ValueError("DeepGEMM FP8 expects [E,M,K] inputs and expert weights.")
    if not (inputs.shape[0] == codes.shape[0] == scales.shape[0]):
        raise ValueError("DeepGEMM FP8 expert axes must match.")
    if inputs.device.type != "cuda":
        raise RuntimeError("DeepGEMM FP8 execution requires CUDA tensors.")
    capability = torch.cuda.get_device_capability(inputs.device)
    if capability[0] != 9:
        raise RuntimeError(
            "Mirai's FP32-scale DeepGEMM path requires SM90; Blackwell uses a "
            "different UE8M0 scale contract."
        )
    if int(inputs.shape[-1]) != int(meta.shape[1]):
        raise ValueError("DeepGEMM FP8 activation width does not match the weight.")
    block = BLOCKWISE_FP8_BLOCK_SIZE
    if meta.shape[0] % block or meta.shape[1] % block:
        raise RuntimeError(
            "DeepGEMM FP8 requires expert input/output widths divisible by 128."
        )

    deep_gemm, get_alignment = _require_deepgemm()
    experts, tokens_per_expert, in_features = map(int, inputs.shape)
    alignment = int(get_alignment())
    if alignment <= 0:
        raise RuntimeError("deep_gemm returned an invalid grouped-M alignment.")
    aligned_tokens = math.ceil(tokens_per_expert / alignment) * alignment
    padded = F.pad(inputs, (0, 0, 0, aligned_tokens - tokens_per_expert))
    flat = padded.reshape(experts * aligned_tokens, in_features)
    activation_codes, activation_scales = _quantize_activation_tiles(
        flat,
        padded_in=in_features,
    )
    grouped_layout = torch.arange(
        experts,
        device=inputs.device,
        dtype=torch.int32,
    ).repeat_interleave(aligned_tokens)
    output = torch.empty(
        (experts * aligned_tokens, meta.shape[0]),
        device=inputs.device,
        dtype=torch.bfloat16,
    )
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (_as_e4m3(activation_codes.reshape_as(flat)), activation_scales),
        (_as_e4m3(codes), scales),
        output,
        grouped_layout,
    )
    return output.reshape(experts, aligned_tokens, meta.shape[0])[
        :, :tokens_per_expert
    ].to(dtype=inputs.dtype)


def _native_routed_forward(
    grouped_inputs: "torch.Tensor",
    codes: "torch.Tensor",
    scales: "torch.Tensor",
    meta: BlockwiseFP8Meta,
    boundaries: "torch.Tensor",
) -> "torch.Tensor":
    """Run unequal contiguous expert groups and return unpadded grouped rows."""
    _validate_blockwise_fp8_layout(codes, scales, meta)
    if grouped_inputs.ndim != 2 or codes.ndim != 3 or scales.ndim != 3:
        raise ValueError(
            "DeepGEMM routed FP8 expects [R,K] inputs and [E,N,K] expert weights."
        )
    if int(codes.shape[0]) != int(boundaries.numel()):
        raise ValueError("DeepGEMM routed FP8 expert axes must match routing groups.")
    if grouped_inputs.device.type != "cuda":
        raise RuntimeError("DeepGEMM FP8 execution requires CUDA tensors.")
    if boundaries.device != grouped_inputs.device:
        raise ValueError("DeepGEMM routed metadata and activations must share a device.")
    capability = torch.cuda.get_device_capability(grouped_inputs.device)
    if capability[0] != 9:
        raise RuntimeError(
            "Mirai's FP32-scale DeepGEMM path requires SM90; Blackwell uses a "
            "different UE8M0 scale contract."
        )
    if int(grouped_inputs.shape[1]) != int(meta.shape[1]):
        raise ValueError("DeepGEMM FP8 activation width does not match the weight.")
    block = BLOCKWISE_FP8_BLOCK_SIZE
    if meta.shape[0] % block or meta.shape[1] % block:
        raise RuntimeError(
            "DeepGEMM FP8 requires expert input/output widths divisible by 128."
        )
    terminal = int(boundaries[-1].item()) if boundaries.numel() else 0
    if terminal != int(grouped_inputs.shape[0]):
        raise ValueError("DeepGEMM routed boundary must end at the routed-row count.")
    if not grouped_inputs.numel():
        return grouped_inputs.new_empty((0, meta.shape[0]))

    deep_gemm, get_alignment = _require_deepgemm()
    alignment = int(get_alignment())
    if alignment <= 0:
        raise RuntimeError("deep_gemm returned an invalid grouped-M alignment.")
    aligned, group_ids, valid_rows = _aligned_routed_input(
        grouped_inputs, boundaries, alignment
    )
    activation_codes, activation_scales = _quantize_activation_tiles(
        aligned, padded_in=int(aligned.shape[1])
    )
    output = torch.empty(
        (aligned.shape[0], meta.shape[0]), device=aligned.device, dtype=torch.bfloat16
    )
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (_as_e4m3(activation_codes.reshape_as(aligned)), activation_scales),
        (_as_e4m3(codes), scales),
        output,
        group_ids,
    )
    return output.index_select(0, valid_rows).to(dtype=grouped_inputs.dtype)


if torch is not None:

    class _DeepGEMMBlockwiseFP8(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, codes, scales, meta):
            output = _native_grouped_forward(inputs, codes, scales, meta)
            ctx.save_for_backward(codes, scales)
            ctx.meta = meta
            ctx.input_dtype = inputs.dtype
            return output

        @staticmethod
        def backward(ctx, grad_output):
            codes, scales = ctx.saved_tensors
            weight = dequantize_blockwise_fp8_weight(
                codes,
                scales,
                ctx.meta,
                dtype=torch.float32,
                device=grad_output.device,
            )
            grad_input = torch.bmm(grad_output.float(), weight)
            return grad_input.to(dtype=ctx.input_dtype), None, None, None


    class _DeepGEMMRoutedBlockwiseFP8(torch.autograd.Function):
        @staticmethod
        def forward(ctx, grouped_inputs, codes, scales, meta, boundaries):
            output = _native_routed_forward(
                grouped_inputs, codes, scales, meta, boundaries
            )
            ctx.save_for_backward(codes, scales, boundaries)
            ctx.meta = meta
            ctx.input_dtype = grouped_inputs.dtype
            return output

        @staticmethod
        def backward(ctx, grad_output):
            codes, scales, boundaries = ctx.saved_tensors
            weight = dequantize_blockwise_fp8_weight(
                codes, scales, ctx.meta, dtype=torch.float32,
                device=grad_output.device,
            )
            pieces = []
            start = 0
            for group, stop_value in enumerate(
                boundaries.detach().to("cpu", torch.int64).tolist()
            ):
                stop = int(stop_value)
                if stop > start:
                    pieces.append(grad_output[start:stop].float() @ weight[group])
                start = stop
            grad_input = (
                torch.cat(pieces)
                if pieces
                else grad_output.new_empty((0, weight.shape[-1]), dtype=torch.float32)
            )
            return grad_input.to(dtype=ctx.input_dtype), None, None, None, None


def deepgemm_blockwise_fp8_batched_linear(
    inputs: "torch.Tensor",
    codes: "torch.Tensor",
    scales: "torch.Tensor",
    meta: BlockwiseFP8Meta,
) -> "torch.Tensor":
    """Run one native M-grouped FP8 forward with high-precision Dgrad."""
    if torch is None:
        raise RuntimeError("DeepGEMM FP8 execution requires torch.")
    single = inputs.ndim == 2
    if single:
        inputs = inputs.unsqueeze(0)
        if codes.ndim == 2:
            codes, scales = codes.unsqueeze(0), scales.unsqueeze(0)
    output = _DeepGEMMBlockwiseFP8.apply(inputs, codes, scales, meta)
    return output[0] if single else output


def deepgemm_blockwise_fp8_routed_linear(
    inputs: "torch.Tensor",
    codes: "torch.Tensor",
    scales: "torch.Tensor",
    meta: BlockwiseFP8Meta,
    layout,
    fusion=None,
    *,
    routing_weights: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Run routed DeepGEMM forward with generic gather/combine semantics.

    DeepGEMM produces unpadded expert-grouped rows. Assignment scatter and
    routing-weighted token reduction are adjacent differentiable operations;
    weighted reduction accumulates directly into token rows and does not create
    an assignment-order expert-output tensor.
    """
    if torch is None:
        raise RuntimeError("DeepGEMM FP8 execution requires torch.")
    from mirai.core.moe.runtime.routed_gemm import (
        RoutedFusionSpec,
        RoutedOutputMode,
    )

    fusion = fusion or RoutedFusionSpec()
    layout.validate(device=inputs.device)
    if inputs.ndim != 2:
        raise ValueError("DeepGEMM routed FP8 activation must be rank-2.")
    if int(codes.shape[0]) != layout.group_count:
        raise ValueError("DeepGEMM routed FP8 weight group axis must match layout.")
    routed_rows = layout.token_count * layout.top_k
    if fusion.gather_tokens:
        if int(inputs.shape[0]) != layout.token_count:
            raise ValueError("gathered activation must contain token_count rows")
        source_tokens = torch.div(
            layout.assignment_rows, layout.top_k, rounding_mode="floor"
        )
        grouped_inputs = inputs.index_select(0, source_tokens.to(torch.int64))
    else:
        if int(inputs.shape[0]) != routed_rows:
            raise ValueError("grouped activation must contain the routed-row count")
        grouped_inputs = inputs
    grouped = _DeepGEMMRoutedBlockwiseFP8.apply(
        grouped_inputs, codes, scales, meta, layout.boundaries
    )
    if fusion.output is RoutedOutputMode.GROUPED:
        return grouped
    if fusion.output is RoutedOutputMode.ASSIGNMENT:
        return torch.empty_like(grouped).index_copy(
            0, layout.assignment_rows.to(torch.int64), grouped
        )
    if routing_weights is None or tuple(routing_weights.shape) != (
        layout.token_count, layout.top_k
    ):
        raise ValueError("routing_weights must have shape (token_count, top_k)")
    return routed_weighted_combine(grouped, layout, routing_weights)


def routed_weighted_combine(
    grouped: "torch.Tensor",
    layout,
    routing_weights: "torch.Tensor",
) -> "torch.Tensor":
    """Combine grouped rows directly into tokens with coefficient autograd."""
    if tuple(routing_weights.shape) != (layout.token_count, layout.top_k):
        raise ValueError("routing_weights must have shape (token_count, top_k)")
    grouped_assignment = layout.assignment_rows.to(torch.int64)
    token_rows = torch.div(grouped_assignment, layout.top_k, rounding_mode="floor")
    coefficients = routing_weights.reshape(-1).index_select(0, grouped_assignment)
    reduced = torch.zeros(
        (layout.token_count, grouped.shape[1]), device=grouped.device,
        dtype=torch.float32,
    )
    reduced.index_add_(
        0, token_rows, grouped.float() * coefficients.float().unsqueeze(1)
    )
    return reduced.to(dtype=grouped.dtype)
