"""DeepSeek-style block-scaled E4M3 reference execution.

Weights use one FP32 dequantization scale per 128x128 tile and activations use
one online scale per token and 128 input channels.  Every K tile contributes to
an FP32 accumulator.  Backward deliberately dequantizes the frozen weight and
computes input gradients in FP32; DeepSeek-V3 Appendix B.2 reports that applying
the 128x128 scheme to Dgrad is unstable.

Sources:
https://arxiv.org/abs/2412.19437
https://github.com/deepseek-ai/DeepSeek-V3
"""

from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from .microscaling_quant import _decode_e4m3, _encode_e4m3


BLOCKWISE_FP8_FORMATS = ("fp8",)
BLOCKWISE_FP8_BLOCK_SIZE = 128
_E4M3_MAX = 448.0


@dataclass(frozen=True)
class BlockwiseFP8Meta:
    shape: tuple[int, int]
    weight_block: tuple[int, int] = (
        BLOCKWISE_FP8_BLOCK_SIZE,
        BLOCKWISE_FP8_BLOCK_SIZE,
    )
    activation_block: int = BLOCKWISE_FP8_BLOCK_SIZE


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("blockwise FP8 quantization requires torch.")


def _padded_shape(meta: BlockwiseFP8Meta) -> tuple[int, int]:
    out_features, in_features = meta.shape
    block_out, block_in = meta.weight_block
    return (
        math.ceil(out_features / block_out) * block_out,
        math.ceil(in_features / block_in) * block_in,
    )


def validate_blockwise_fp8_payload(
    codes: "torch.Tensor",
    scales: "torch.Tensor",
    meta: BlockwiseFP8Meta,
) -> None:
    """Validate one matrix or a leading batch of equally shaped matrices."""
    _require_torch()
    _validate_blockwise_fp8_layout(codes, scales, meta)
    if bool(((codes & 0x7F) == 0x7F).any().item()):
        raise ValueError("blockwise FP8 payload cannot contain E4M3 NaN codes.")
    if not bool((torch.isfinite(scales) & (scales > 0.0)).all().item()):
        raise ValueError("blockwise FP8 dequantization scales must be finite and positive.")


def _validate_blockwise_fp8_layout(
    codes: "torch.Tensor",
    scales: "torch.Tensor",
    meta: BlockwiseFP8Meta,
) -> None:
    if tuple(meta.weight_block) != (
        BLOCKWISE_FP8_BLOCK_SIZE,
        BLOCKWISE_FP8_BLOCK_SIZE,
    ) or int(meta.activation_block) != BLOCKWISE_FP8_BLOCK_SIZE:
        raise ValueError("blockwise FP8 metadata must use 128x128/1x128 scaling.")
    if len(meta.shape) != 2 or min(meta.shape) <= 0:
        raise ValueError("blockwise FP8 metadata must declare a positive 2D shape.")
    if codes.dtype != torch.uint8 or scales.dtype != torch.float32:
        raise ValueError("blockwise FP8 codes/scales must use uint8/float32 storage.")
    if tuple(codes.shape[-2:]) != tuple(meta.shape):
        raise ValueError("blockwise FP8 code shape does not match metadata.")
    block_out, block_in = meta.weight_block
    expected_scale_shape = (
        math.ceil(meta.shape[0] / block_out),
        math.ceil(meta.shape[1] / block_in),
    )
    if tuple(scales.shape[:-2]) != tuple(codes.shape[:-2]) or tuple(
        scales.shape[-2:]
    ) != expected_scale_shape:
        raise ValueError("blockwise FP8 scale shape does not match the code payload.")


def quantize_blockwise_fp8_weight(
    weight: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", BlockwiseFP8Meta]:
    """Encode a 2D frozen weight with FP32 128x128 dequantization scales."""
    _require_torch()
    if weight.ndim != 2:
        raise ValueError(f"blockwise FP8 expects a 2D weight, got {tuple(weight.shape)}.")
    source = weight.detach().float()
    if not bool(torch.isfinite(source).all().item()):
        raise ValueError("blockwise FP8 quantization requires finite weights.")
    meta = BlockwiseFP8Meta(tuple(int(dim) for dim in source.shape))
    padded_out, padded_in = _padded_shape(meta)
    padded = F.pad(
        source,
        (0, padded_in - meta.shape[1], 0, padded_out - meta.shape[0]),
    )
    block = BLOCKWISE_FP8_BLOCK_SIZE
    tiles = padded.reshape(padded_out // block, block, padded_in // block, block)
    amax = tiles.abs().amax(dim=(1, 3))
    scales = (amax / _E4M3_MAX).clamp(min=1e-30).to(torch.float32)
    normalized = tiles / scales[:, None, :, None]
    codes = _encode_e4m3(normalized).reshape(padded_out, padded_in)
    codes = codes[: meta.shape[0], : meta.shape[1]].contiguous()
    _validate_blockwise_fp8_layout(codes, scales, meta)
    return codes, scales.contiguous(), meta


def dequantize_blockwise_fp8_weight(
    codes: "torch.Tensor",
    scales: "torch.Tensor",
    meta: BlockwiseFP8Meta,
    *,
    dtype: "torch.dtype",
    device: "torch.device",
) -> "torch.Tensor":
    """Decode one matrix or a leading batch of matrices."""
    _require_torch()
    _validate_blockwise_fp8_layout(codes, scales, meta)
    target = torch.device(device)
    padded_out, padded_in = _padded_shape(meta)
    block = BLOCKWISE_FP8_BLOCK_SIZE
    stored = codes.to(device=target)
    padded = F.pad(
        stored,
        (0, padded_in - meta.shape[1], 0, padded_out - meta.shape[0]),
    )
    leading = tuple(int(dim) for dim in padded.shape[:-2])
    tiles = padded.reshape(
        *leading,
        padded_out // block,
        block,
        padded_in // block,
        block,
    )
    decoded = _decode_e4m3(tiles)
    decoded = decoded * scales.to(device=target)[..., :, None, :, None]
    weight = decoded.reshape(*leading, padded_out, padded_in)
    return weight[..., : meta.shape[0], : meta.shape[1]].to(dtype=dtype)


def _quantize_activation_tiles(
    x_2d: "torch.Tensor",
    *,
    padded_in: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    block = BLOCKWISE_FP8_BLOCK_SIZE
    padded = F.pad(x_2d.float(), (0, padded_in - int(x_2d.shape[-1])))
    tiles = padded.reshape(-1, padded_in // block, block)
    # The official reference clamps activation amax to 1e-4 before scaling.
    scales = tiles.abs().amax(dim=-1).clamp(min=1e-4) / _E4M3_MAX
    codes = _encode_e4m3(tiles / scales.unsqueeze(-1))
    return codes, scales


def _blockwise_fp8_matmul_2d(
    x_2d: "torch.Tensor",
    codes: "torch.Tensor",
    scales: "torch.Tensor",
    meta: BlockwiseFP8Meta,
) -> "torch.Tensor":
    _validate_blockwise_fp8_layout(codes, scales, meta)
    if int(x_2d.shape[-1]) != meta.shape[1]:
        raise ValueError("blockwise FP8 activation width does not match the weight.")
    padded_out, padded_in = _padded_shape(meta)
    block = BLOCKWISE_FP8_BLOCK_SIZE
    activation_codes, activation_scales = _quantize_activation_tiles(
        x_2d,
        padded_in=padded_in,
    )
    weight_codes = F.pad(
        codes.to(device=x_2d.device),
        (0, padded_in - meta.shape[1], 0, padded_out - meta.shape[0]),
    )
    accumulator = torch.zeros(
        (int(x_2d.shape[0]), padded_out),
        dtype=torch.float32,
        device=x_2d.device,
    )
    output_scales = scales.to(device=x_2d.device).repeat_interleave(block, dim=-2)
    for k_block in range(padded_in // block):
        activation = (
            _decode_e4m3(activation_codes[:, k_block])
            * activation_scales[:, k_block].unsqueeze(-1)
        )
        weight = _decode_e4m3(
            weight_codes[:, k_block * block : (k_block + 1) * block]
        ) * output_scales[:, k_block].unsqueeze(-1)
        accumulator.addmm_(activation, weight.transpose(0, 1))
    return accumulator[:, : meta.shape[0]]


if torch is not None:

    class _BlockwiseFP8Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, codes, scales, meta):
            leading = tuple(int(dim) for dim in inputs.shape[:-1])
            output = _blockwise_fp8_matmul_2d(
                inputs.reshape(-1, inputs.shape[-1]),
                codes,
                scales,
                meta,
            )
            ctx.save_for_backward(codes, scales)
            ctx.meta = meta
            ctx.input_dtype = inputs.dtype
            return output.reshape(*leading, meta.shape[0]).to(dtype=inputs.dtype)

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
            grad_input = grad_output.float() @ weight
            return grad_input.to(dtype=ctx.input_dtype), None, None, None


def blockwise_fp8_linear(
    inputs: "torch.Tensor",
    codes: "torch.Tensor",
    scales: "torch.Tensor",
    meta: BlockwiseFP8Meta,
) -> "torch.Tensor":
    """W8A8 reference linear with a high-precision frozen-weight Dgrad."""
    _require_torch()
    return _BlockwiseFP8Linear.apply(inputs, codes, scales, meta)


def blockwise_fp8_batched_linear(
    inputs: "torch.Tensor",
    codes: "torch.Tensor",
    scales: "torch.Tensor",
    meta: BlockwiseFP8Meta,
) -> "torch.Tensor":
    """Apply the reference linear to matching leading expert batches."""
    _require_torch()
    if inputs.ndim != 3 or codes.ndim != 3 or scales.ndim != 3:
        raise ValueError("batched blockwise FP8 expects [E,M,K] inputs and weights.")
    if not (inputs.shape[0] == codes.shape[0] == scales.shape[0]):
        raise ValueError("batched blockwise FP8 expert axes must match.")
    return torch.stack(
        [
            blockwise_fp8_linear(inputs[index], codes[index], scales[index], meta)
            for index in range(int(inputs.shape[0]))
        ],
        dim=0,
    )
