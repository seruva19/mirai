"""Diffusers-free helpers adapted for the native LingBot-Video transformer.

Source: https://github.com/Robbyant/lingbot-video (Apache-2.0). See the adjacent
LICENSE.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from mirai.core.models.attention_backends import dispatch_attention


@dataclass(frozen=True)
class ContextParallelInput:
    split_dim: int
    expected_dims: int


@dataclass(frozen=True)
class ContextParallelOutput:
    gather_dim: int
    expected_dims: int


@dataclass
class Transformer3DModelOutput:
    sample: torch.Tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        if int(index) != 0:
            raise IndexError(index)
        return self.sample


def capture_init_config(init: Callable[..., None]) -> Callable[..., None]:
    signature = inspect.signature(init)

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        config = {
            key: value
            for key, value in bound.arguments.items()
            if key != "self"
        }
        self.config = SimpleNamespace(**config)
        init(self, *args, **kwargs)

    return wrapped


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        *,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.flip_sin_to_cos = bool(flip_sin_to_cos)
        self.downscale_freq_shift = float(downscale_freq_shift)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.num_channels // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        )
        exponent = exponent / max(half_dim - self.downscale_freq_shift, 1.0)
        emb = timesteps.float().reshape(-1, 1) * torch.exp(exponent).reshape(1, -1)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        *,
        act_fn: str = "silu",
        sample_proj_bias: bool = True,
    ) -> None:
        super().__init__()
        if act_fn != "silu":
            raise ValueError(f"Unsupported LingBot timestep activation '{act_fn}'.")
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=sample_proj_bias)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


def dispatch_attention_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    parallel_config: Any | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    _ = parallel_config
    return dispatch_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        backend=backend,
    )
