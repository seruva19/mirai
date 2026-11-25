"""Spatial tiling for the MAGI-2 TurboVAE decoder."""

from __future__ import annotations

from collections.abc import Callable

import torch


def decode_spatial_tiles(
    latent: torch.Tensor,
    decode: Callable[[torch.Tensor], torch.Tensor | list[torch.Tensor]],
    *,
    tile_size: int,
    overlap: int,
) -> torch.Tensor:
    """Decode overlapping latent tiles and assemble the pixel output on CPU."""
    if not 0 < overlap < tile_size:
        raise ValueError("TurboVAE tile overlap must be positive and smaller than the tile size.")
    height, width = latent.shape[-2:]
    stride = tile_size - overlap
    rows = range(0, height - overlap, stride) if height > tile_size else (0,)
    columns = range(0, width - overlap, stride) if width > tile_size else (0,)
    output = weights = None
    scale = None
    for top in rows:
        bottom = min(top + tile_size, height)
        for left in columns:
            right = min(left + tile_size, width)
            tile = decode(latent[..., top:bottom, left:right].contiguous())
            if isinstance(tile, list):
                tile = torch.cat(tile, dim=0)
            tile = tile.detach().to(device="cpu", dtype=torch.float32)
            scale_h, rem_h = divmod(tile.shape[-2], bottom - top)
            scale_w, rem_w = divmod(tile.shape[-1], right - left)
            if rem_h or rem_w or min(scale_h, scale_w) < 1:
                raise ValueError("TurboVAE tile output must have integer spatial scaling.")
            if output is None:
                scale = (scale_h, scale_w)
                output = tile.new_zeros((*tile.shape[:-2], height * scale_h, width * scale_w))
                weights = tile.new_zeros(output.shape[-2:])
            if scale != (scale_h, scale_w) or output.shape[:-2] != tile.shape[:-2]:
                raise ValueError("TurboVAE tile outputs disagree on the decoded shape.")
            wy = _axis_weights(tile.shape[-2], overlap * scale_h, top > 0, bottom < height)
            wx = _axis_weights(tile.shape[-1], overlap * scale_w, left > 0, right < width)
            blend = wy[:, None] * wx[None, :]
            y = slice(top * scale_h, bottom * scale_h)
            x = slice(left * scale_w, right * scale_w)
            output[..., y, x].addcmul_(tile, blend)
            weights[y, x].add_(blend)
    if output is None or weights is None:
        raise ValueError("TurboVAE requires a nonempty spatial grid.")
    return output.div_(weights)


def _axis_weights(length: int, overlap: int, leading: bool, trailing: bool) -> torch.Tensor:
    weights = torch.ones(length, dtype=torch.float32, device="cpu")
    if leading or trailing:
        ramp = torch.arange(1, overlap + 1, dtype=torch.float32, device="cpu") / (overlap + 1)
        if leading:
            weights[:overlap].mul_(ramp)
        if trailing:
            weights[-overlap:].mul_(ramp.flip(0))
    return weights
