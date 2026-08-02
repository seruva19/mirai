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
