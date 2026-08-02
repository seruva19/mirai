"""Shared helpers for native DiT-style sparse-MoE denoisers."""

from __future__ import annotations

from typing import Any

from mirai.core.tensors import is_torch_tensor

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def as_latent_tensor(
    value: Any,
    *,
    dtype: Any | None = None,
    device: Any | None = None,
) -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Native sparse-MoE DiT helpers require torch.")
    if is_torch_tensor(value):
        tensor = value
        if dtype is not None or device is not None:
            tensor = tensor.to(
                dtype=dtype if dtype is not None else tensor.dtype,
                device=device if device is not None else tensor.device,
            )
    else:
        tensor = torch.tensor(value, dtype=dtype or torch.float32, device=device)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1, 1, 1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(tensor.shape[0], 1, 1, 1, 1)
    if tensor.ndim == 2:
        return tensor.reshape(tensor.shape[0], 1, 1, 1, tensor.shape[1])
    return tensor


def text_scalar(text_embeds: dict[str, Any], *, batch: int, like: Any) -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Native sparse-MoE DiT helpers require torch.")
    value = text_embeds.get("t5", None)
    device = like.device
    dtype = like.dtype
    if is_torch_tensor(value):
        tensor = value.to(device=device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor.reshape(1, 1).expand(batch, 1)
        if tensor.shape[0] == batch:
            return tensor.reshape(batch, -1).mean(dim=1, keepdim=True)
        return tensor.reshape(1, -1).mean(dim=1, keepdim=True).expand(batch, 1)
    if isinstance(value, (list, tuple)):
        tensor = torch.tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 1 and tensor.numel() == batch:
            return tensor.reshape(batch, 1)
        return tensor.reshape(1, -1).mean(dim=1, keepdim=True).expand(batch, 1)
    return torch.zeros((batch, 1), device=device, dtype=dtype)


class LatentPatchCodec:
    def __init__(self, *, latent_channels: int, patch_size: int, model_label: str) -> None:
        self.latent_channels = int(latent_channels)
        self.patch_size = int(patch_size)
        self.model_label = str(model_label)

    def normalize_latents(self, latents: Any) -> tuple[Any, bool]:
        tensor = as_latent_tensor(latents)
        if tensor.ndim == 4:
            return tensor.unsqueeze(2), False
        if tensor.ndim != 5:
            raise ValueError(
                f"{self.model_label} expects latents shaped [B,C,H,W] or [B,C,T,H,W]."
            )
        return tensor, True

    def patchify(self, latents: Any) -> tuple[Any, tuple[int, int, int, int, int], bool]:
        x, had_video_dim = self.normalize_latents(latents)
        batch, channels, frames, height, width = [int(v) for v in x.shape]
        patch = self.patch_size
        if channels != self.latent_channels:
            raise ValueError(
                f"{self.model_label} configured for latent_channels={self.latent_channels}, "
                f"but input has {channels} channels."
            )
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"{self.model_label} patch_size={patch} requires H and W divisible by patch size."
            )
        h_patches = height // patch
        w_patches = width // patch
        tokens = (
            x.reshape(batch, channels, frames, h_patches, patch, w_patches, patch)
            .permute(0, 2, 3, 5, 1, 4, 6)
            .reshape(batch, frames * h_patches * w_patches, channels * patch * patch)
        )
        return tokens, (batch, channels, frames, height, width), had_video_dim

    def unpatchify(
        self,
        tokens: Any,
        shape: tuple[int, int, int, int, int],
        *,
        had_video_dim: bool,
    ) -> Any:
        batch, channels, frames, height, width = shape
        patch = self.patch_size
        h_patches = height // patch
        w_patches = width // patch
        x = (
            tokens.reshape(batch, frames, h_patches, w_patches, channels, patch, patch)
            .permute(0, 4, 1, 2, 5, 3, 6)
            .reshape(batch, channels, frames, height, width)
        )
        return x if had_video_dim else x.squeeze(2)

    def position_coordinates(
        self,
        *,
        shape: tuple[int, int, int, int, int],
        like: Any,
    ) -> Any:
        _batch, _channels, frames, height, width = shape
        patch = self.patch_size
        h_patches = height // patch
        w_patches = width // patch
        t = torch.linspace(0.0, 1.0, frames, device=like.device, dtype=like.dtype)
        y = torch.linspace(0.0, 1.0, h_patches, device=like.device, dtype=like.dtype)
        x = torch.linspace(0.0, 1.0, w_patches, device=like.device, dtype=like.dtype)
        tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
        return torch.stack([tt, yy, xx], dim=-1).reshape(1, -1, 3)
