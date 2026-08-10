"""Measured activation-weighted error for frozen expert formats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


EXPERT_TENSOR_PRECISION_FORMATS = (
    "bf16",
    "fp8",
    "int8",
    "nf4",
    "gguf_iq4",
    "gguf_iq3",
    "gguf_iq2",
    "mxfp4",
    "nvfp4",
    "mxfp8_e4m3",
)


@dataclass(frozen=True)
class ProjectionFormatMeasurement:
    quant_format: str
    weighted_mse: float
    stored_bytes: int


def _buffer_bytes(module: Any) -> int:
    return sum(
        int(buffer.numel()) * int(buffer.element_size())
        for buffer in module.buffers()
    )


def measure_projection_format(
    weight: Any,
    importance: Any,
    *,
    quant_format: str,
    projection: str,
    group_sizes: str | int | tuple[int, ...] | None = None,
) -> ProjectionFormatMeasurement:
    """Quantize once and measure diagonal-Hessian output error.

    The objective is ``mean_rows(sum_j E[x_j²] * (q_ij - w_ij)²)``. It is the
    diagonal input-covariance approximation used by importance-matrix
    quantization, evaluated against Mirai's actual packed encoder and decoder.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert precision measurement requires torch.")
    source = torch.as_tensor(weight).detach()
    if source.ndim != 2 or int(source.numel()) == 0:
        raise ValueError("Projection precision measurement requires a 2D weight.")
    weights = torch.as_tensor(importance).detach().to(
        device=source.device, dtype=torch.float64
    )
    if weights.ndim != 1 or int(weights.numel()) != int(source.shape[1]):
        raise ValueError("Importance vector must match the projection input axis.")
    if not bool(torch.isfinite(weights).all().item()) or bool(
        (weights < 0).any().item()
    ):
        raise ValueError("Importance values must be finite and non-negative.")
    normalizer = float(weights.sum().item()) * int(source.shape[0])
    if not normalizer > 0.0:
        raise ValueError("Importance measurement requires positive total mass.")
    fmt = str(quant_format).strip().lower()
    if fmt not in EXPERT_TENSOR_PRECISION_FORMATS:
        raise ValueError(f"Unsupported expert tensor precision format {fmt!r}.")
    if str(projection) not in {"w1", "w2", "w3"}:
        raise ValueError("Projection precision measurement requires w1, w2, or w3.")
    if fmt == "bf16":
        reconstructed = source.to(torch.bfloat16).to(torch.float32)
        stored_bytes = int(source.numel()) * 2
    else:
        from mirai.core.models.compressed_weights.execution.experts import (
            CompressedGroupedExperts,
        )

        host = CompressedGroupedExperts.from_empty(
            num_experts=1,
            group_sizes=group_sizes,
            expert_weight_access="active_dequant",
            quant_format=fmt,
        )
        host.load_dense_weight(str(projection), source.unsqueeze(0))
        reconstructed = host._dequantize_expert(  # noqa: SLF001
            str(projection),
            0,
            dtype=torch.float32,
            device=source.device,
        )
        stored_bytes = _buffer_bytes(host)
        del host
    difference = reconstructed.to(torch.float64) - source.to(torch.float64)
    weighted_mse = float(
        (difference.square() * weights.unsqueeze(0)).sum().item() / normalizer
    )
    if not torch.isfinite(torch.tensor(weighted_mse)) or stored_bytes <= 0:
        raise ValueError("Projection format measurement produced invalid evidence.")
    return ProjectionFormatMeasurement(
        quant_format=fmt,
        weighted_mse=weighted_mse,
        stored_bytes=int(stored_bytes),
    )


__all__ = [
    "EXPERT_TENSOR_PRECISION_FORMATS",
    "ProjectionFormatMeasurement",
    "measure_projection_format",
]
