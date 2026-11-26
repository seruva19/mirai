"""Native block-scaled E2M1 matmul over Mirai microscaling payloads."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .microscaling_quant import MicroscalingMeta, dequantize_microscaling, quantize_microscaling


FP4_FORMATS = frozenset({"nvfp4", "mxfp4"})


def _swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    # cuBLAS 32x4x4 block-scale layout, as exposed by torchao's to_blocked:
    # https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/utils.py
    rows, columns = scales.shape
    padded = F.pad(scales, (0, (-columns) % 4, 0, (-rows) % 128))
    return padded.reshape(-1, 4, 32, padded.shape[1] // 4, 4).permute(
        0, 3, 2, 1, 4
    ).contiguous().flatten()


def _native_payload(
    codes: torch.Tensor, scales: torch.Tensor, rows: int, columns: int,
    *, block_size: int, scale_dtype: torch.dtype, padded_rows: int, padded_columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = columns // block_size
    if codes.shape != (rows * groups, block_size // 2) or scales.shape != (rows * groups,):
        raise ValueError("Native FP4 payload does not match the row-aligned matrix.")
    matrix = F.pad(codes.reshape(rows, columns // 2), (0, (padded_columns - columns) // 2, 0, padded_rows - rows))
    scale_matrix = F.pad(
        scales.reshape(rows, groups),
        (0, (padded_columns - columns) // block_size, 0, padded_rows - rows),
    )
    return matrix.view(torch.float4_e2m1fn_x2), _swizzle_scales(scale_matrix).view(scale_dtype)


class _NativeFP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codes, scales, global_scale, meta):
        features, columns = meta.shape
        rows = x.numel() // columns
        ctx.save_for_backward(codes, scales, global_scale)
        ctx.meta = meta
        ctx.dtype = x.dtype
        if rows == 0:
            return x.new_empty((*x.shape[:-1], features))
        block = meta.block_size
        padded_rows = rows + (-rows) % 32
        padded_features = features + (-features) % 32
        padded_columns = columns + (-columns) % 128
        a_codes, a_scales, a_global, _ = quantize_microscaling(
            meta.format, x.reshape(rows, columns)
        )
        if meta.format == "mxfp4" and torch.cuda.get_device_capability(x.device)[0] == 12:
            from .native_mxfp4_triton import native_mxfp4_matmul

            out = native_mxfp4_matmul(
                a_codes, codes.to(x.device).contiguous(), a_scales, scales.to(x.device).contiguous(),
                rows=rows, features=features, columns=columns,
            )
            return out.reshape(*x.shape[:-1], features).to(x.dtype)
        scale_dtype = torch.float8_e4m3fn if meta.format == "nvfp4" else torch.float8_e8m0fnu
        a, sa = _native_payload(
            a_codes, a_scales, rows, columns, block_size=block,
            scale_dtype=scale_dtype, padded_rows=padded_rows, padded_columns=padded_columns,
        )
        b, sb = _native_payload(
            codes.to(x.device), scales.to(x.device), features, columns, block_size=block,
            scale_dtype=scale_dtype, padded_rows=padded_features, padded_columns=padded_columns,
        )
        if meta.format == "nvfp4":
            out = torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.float32, use_fast_accum=False)
            out.mul_(a_global * global_scale.to(x.device))
        else:
            recipe = F.ScalingType.BlockWise1x32
            swizzle = F.SwizzleType.SWIZZLE_32_4_4
            out = F.scaled_mm(
                a, b.t(), sa, recipe, sb, recipe,
                swizzle_a=swizzle, swizzle_b=swizzle, output_dtype=torch.float32,
                use_fast_accum=False,
            )
        return out[:rows, :features].reshape(*x.shape[:-1], features).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        weight = dequantize_microscaling(
            *ctx.saved_tensors, ctx.meta, dtype=torch.float32, device=grad_output.device
        )
        return (grad_output.float() @ weight).to(ctx.dtype), None, None, None, None


def native_fp4_linear(x, codes, scales, global_scale, meta: MicroscalingMeta):
    from mirai.core.moe.runtime.gemm import probe_backend

    if meta.format not in FP4_FORMATS or len(meta.shape) != 2:
        raise ValueError("Native FP4 requires an NVFP4 or MXFP4 matrix.")
    if x.shape[-1] != meta.shape[1] or meta.shape[1] % meta.block_size:
        raise ValueError("Native FP4 requires matching input width aligned to the stored scale block.")
    if meta.format == "mxfp4" and (not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] != 12) and not (
        callable(getattr(F, "scaled_mm", None)) and hasattr(torch, "float8_e8m0fnu")
    ):
        raise RuntimeError("Native MXFP4 requires PyTorch scaled_mm with E8M0 scales.")
    status = probe_backend("scaled_mm_fp4", device=x.device)
    if not status.available:
        raise RuntimeError(status.reason)
    return _NativeFP4Linear.apply(x, codes, scales, global_scale, meta)
