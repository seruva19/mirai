"""Lossless CUDA byte compression through the NVIDIA nvCOMP Python API."""

from __future__ import annotations

from functools import lru_cache
from threading import local

import torch


_thread_state = local()


def _codec(device):
    # nvCOMP codecs hold mutable scratch and must stay on their creating thread.
    try:
        from nvidia import nvcomp
    except (ImportError, OSError) as exc:
        raise RuntimeError("LZ4 activation offload requires NVIDIA nvCOMP for the active CUDA major.") from exc

    stream = torch.cuda.current_stream(device)
    key = (stream.device.index, int(stream.cuda_stream))
    cache = getattr(_thread_state, "codecs", None)
    if cache is None:
        cache = _thread_state.codecs = {}
    if key not in cache:
        codec = nvcomp.Codec(algorithm="LZ4", device_id=key[0], cuda_stream=key[1])
        cache[key] = (codec, lru_cache(maxsize=8)(codec.compression_config))
    codec, plan = cache[key]
    return nvcomp, codec, plan, key[1]


def encode_cuda_lz4(source: torch.Tensor) -> torch.Tensor | None:
    """Return a compressed CUDA byte view when it reduces the transfer size."""
    if not source.is_cuda:
        raise ValueError("nvCOMP LZ4 requires a CUDA tensor.")
    data = source.contiguous().view(torch.uint8).flatten()
    if not data.numel():
        return None
    nvcomp, codec, plan, stream = _codec(data.device)
    input_array = nvcomp.as_array(data, cuda_stream=stream)
    capacity = codec.get_max_comp_buffer_size(input_array)
    output = torch.empty(capacity, device=data.device, dtype=torch.uint8)
    encoded = codec.encode(
        input_array, out=nvcomp.as_array(output, cuda_stream=stream), compression_config=plan(data.numel()),
    )
    # Query the logical size before slicing the PyTorch-owned allocation.
    length = int(encoded.shape[0])
    return output[:length] if length < data.numel() else None


def decode_cuda_lz4(source: torch.Tensor, out: torch.Tensor) -> None:
    """Restore a CUDA payload into a contiguous destination of the original size."""
    if not source.is_cuda or source.device != out.device or not out.is_contiguous():
        raise ValueError("nvCOMP LZ4 requires contiguous CUDA output on the source device.")
    nvcomp, codec, _, stream = _codec(source.device)
    encoded = nvcomp.as_array(source.view(torch.uint8).flatten(), cuda_stream=stream)
    destination = nvcomp.as_array(out.view(torch.uint8).flatten(), cuda_stream=stream)
    codec.decode(encoded, out=destination)
