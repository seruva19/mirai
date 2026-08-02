"""Frozen expert GEMM that rotates activations instead of un-rotating weights.

The dequant path reconstructs each frozen expert weight as bf16 before its GEMM
by unpacking int8, scaling, and undoing the Hadamard rotation with a dense fp32
matmul (`quant.py:_dequantize_weight` -> `_rotate_last_dim`).

The rotation is unnecessary on the weights. The stored int8 `q` satisfies
`q * s == W_orig @ H` (quant rotates before quantizing), and H is symmetric and
orthogonal (`H @ H == I`, verified), so:

    y = x @ W_orig^T = x @ (q*s @ H)^T = (x @ H) @ (q*s)^T

Rotate the ACTIVATIONS once per chunk (a single [M, K] tensor shared by w1/w3),
and because the per-row scale `s` indexes the OUTPUT dimension it factors out of
the contraction:

    y = ((x @ H) @ q^T) * s      with q cast to bf16 (|q| <= 127 is exact)

The weight GEMM consumes `q` directly without a weight-side rotation matmul.
Applying `s` after accumulation also avoids rounding `q*s` before contraction.
Backward follows the same differentiable torch operations.

NOT bit-identical to the dequant default (reduction order and rounding placement
differ), so it is opt-in and never reached through "auto".
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import guard mirrors the package
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .quant import _rotate_last_dim


def rotate_activations(
    x: "torch.Tensor",
    group_size: int,
    *,
    rotation: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Apply the stored orthogonal rotation along the last (K) dimension.

    ``rotation=None`` preserves the fixed-Hadamard baseline.  A learned matrix
    uses the same groupwise convention as weight quantization: stored weights
    approximate ``W @ R``, so runtime activations use ``x @ R``.
    """

    if int(group_size) <= 0:
        return x
    shape = tuple(x.shape)
    k = int(shape[-1])
    rows = int(x.numel() // max(k, 1))
    rotated = _rotate_last_dim(
        x.reshape(rows, k).float(),
        int(group_size),
        inverse=False,
        rotation=rotation,
    )
    return rotated.reshape(shape).to(dtype=x.dtype)


def rotated_int8_linear(
    x: "torch.Tensor",
    quantized: "torch.Tensor",
    scale: "torch.Tensor",
    group_size: int,
    *,
    rotation: "torch.Tensor | None" = None,
    x_already_rotated: bool = False,
) -> "torch.Tensor":
    """y = ((x @ H) @ q^T) * s for a batched expert chunk, via cuBLAS bmm.

    Shapes: x [E, M, K], quantized [E, N, K] int8, scale [E, N, 1] (or [E, N]).
    Returns [E, M, N]. Pass ``x_already_rotated=True`` when the caller shares one
    rotation across several weights (w1/w3), so the [M, K] rotation is paid once.

    All ops are autograd-aware, so this doubles as its own backward: dX flows
    back through the epilogue scale, the bmm against ``q``, and the activation
    rotation -- matching ``dX = (grad * s) @ q @ H`` without a hand-written pass.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("rotated_int8_linear requires torch.")
    x_rot = (
        x
        if x_already_rotated
        else rotate_activations(
            x,
            int(group_size),
            rotation=rotation,
        )
    )
    # FP32/TF32 preserves the unscaled accumulation until the scale epilogue.
    # BF16 would round the K-term sum before scaling and violate parity.
    raw = torch.bmm(x_rot.float(), quantized.float().transpose(-2, -1))
    e, _, n = raw.shape
    scaled = raw * scale.reshape(e, 1, n).float()
    return scaled.to(dtype=x_rot.dtype)
