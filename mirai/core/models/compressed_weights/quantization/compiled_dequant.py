"""Capability-gated TorchInductor kernels for packed frozen-weight decode."""

from __future__ import annotations

from functools import lru_cache

import torch

from .gguf_quant import dequantize_gguf
from .microscaling_quant import dequantize_microscaling


def _require_compile() -> None:
    if not callable(getattr(torch, "compile", None)):
        raise RuntimeError("compiled_packed requires torch.compile.")


@lru_cache(maxsize=32)
def _microscaling_decoder(fmt, block_size, shape, padding, dtype, device_text):
    _require_compile()
    from .microscaling_quant import MicroscalingMeta

    meta = MicroscalingMeta(fmt, block_size, shape, padding)
    device = torch.device(device_text)

    def decode(packed, scales, global_scale):
        return dequantize_microscaling(
            packed,
            scales,
            global_scale,
            meta,
            dtype=dtype,
            device=device,
        )

    return torch.compile(decode, fullgraph=True, dynamic=False)


def compiled_dequantize_microscaling(
    packed, scales, global_scale, meta, *, dtype, device
):
    decoder = _microscaling_decoder(
        meta.format,
        int(meta.block_size),
        tuple(meta.shape),
        int(meta.padding),
        dtype,
        str(torch.device(device)),
    )
    return decoder(packed, scales, global_scale)


@lru_cache(maxsize=32)
def _gguf_decoder(fmt, shape, dtype, device_text):
    _require_compile()
    device = torch.device(device_text)

    def decode(blocks):
        return dequantize_gguf(
            fmt,
            blocks,
            shape=shape,
            dtype=dtype,
            device=device,
        )

    return torch.compile(decode, fullgraph=True, dynamic=False)


def compiled_dequantize_gguf(fmt, blocks, *, shape, dtype, device):
    return _gguf_decoder(
        str(fmt), tuple(shape), dtype, str(torch.device(device))
    )(blocks)


__all__ = [
    "compiled_dequantize_gguf",
    "compiled_dequantize_microscaling",
]
