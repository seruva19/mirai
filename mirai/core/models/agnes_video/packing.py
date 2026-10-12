"""Agnes release token and coordinate layouts.

Derived from Agnes-AI/Agnes-Video-2.0-MoE at
fc3971e378b53c766baa06898bfe84d7a56653ea, inference/pipeline/data_proxy.py.
Copyright (c) 2026 SandAI. Licensed under Apache-2.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def coordinates(shape: tuple[int, int, int], reference: tuple[int, int, int],
                *, like: torch.Tensor, offset: int = 0) -> torch.Tensor:
    axes = [torch.arange(size, device=like.device, dtype=torch.float32) for size in shape]
    axes[0] = axes[0] + offset
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    metadata = grid.new_tensor((*shape, *reference)).expand(grid.shape[0], -1)
    return torch.cat((grid, metadata), dim=-1)


@dataclass(frozen=True)
class PackedAgnesInputs:
    tokens: torch.Tensor
    coordinates: torch.Tensor
    modalities: torch.Tensor
    cu_seqlens: torch.Tensor
    video_shape: tuple[int, ...]
    audio_lengths: tuple[int, ...]
    text_lengths: tuple[int, ...]
    audio_channels: int
    patch_size: int = 2

    def unpack(self, prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, frames, height, width = self.video_shape
        patch = self.patch_size
        count = frames * (height // patch) * (width // patch)
        videos, audios = [], []
        for index in range(batch):
            start = int(self.cu_seqlens[index])
            # The release output head is patch-major even though input unfolding
            # is channel-major. This asymmetry is part of checkpoint semantics.
            video = prediction[start:start + count, :channels * patch * patch]
            video = video.reshape(frames, height // patch, width // patch,
                                  patch, patch, channels)
            videos.append(video.permute(5, 0, 1, 3, 2, 4).reshape(channels, frames, height, width))
            length = self.audio_lengths[index]
            audios.append(prediction[start + count:start + count + length, :self.audio_channels])
        return torch.stack(videos), torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)


def pack_inputs(video: torch.Tensor, audio: torch.Tensor, text: torch.Tensor, *,
                audio_lengths: torch.Tensor | None = None,
                text_lengths: torch.Tensor | None = None,
                patch_size: int = 2) -> PackedAgnesInputs:
    if video.ndim != 5 or audio.ndim != 3 or text.ndim != 3:
        raise ValueError("Agnes requires video [B,C,T,H,W], audio [B,L,D], text [B,S,D].")
    batch, channels, frames, height, width = video.shape
    if audio.shape[0] != batch or text.shape[0] != batch:
        raise ValueError("Agnes modalities must cover the same batch.")
    if height % patch_size or width % patch_size:
        raise ValueError("Agnes latent spatial dimensions must be divisible by patch_size.")
    alens = tuple(int(v) for v in (audio_lengths if audio_lengths is not None else
                                  [audio.shape[1]] * batch))
    tlens = tuple(int(v) for v in (text_lengths if text_lengths is not None else
                                  [text.shape[1]] * batch))
    if len(alens) != batch or len(tlens) != batch:
        raise ValueError("Agnes modality lengths must cover the same batch.")
    if any(v < 1 or v > audio.shape[1] for v in alens) or any(v < 1 or v > text.shape[1] for v in tlens):
        raise ValueError("Agnes modality lengths are outside their tensor bounds.")
    patches = video.unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6).reshape(batch, -1, channels * patch_size**2)
    token_parts, coord_parts, modality_parts, boundaries = [], [], [], [0]
    shape = (frames, height // patch_size, width // patch_size)
    reference = (frames, 32, 32) if max(shape[1:]) > 32 else shape
    for index in range(batch):
        values = (patches[index], audio[index, :alens[index]], text[index, :tlens[index]])
        width_max = max(value.shape[-1] for value in values)
        token_parts.append(torch.cat([F.pad(v, (0, width_max-v.shape[-1])) for v in values]))
        coord_parts.append(torch.cat((
            coordinates(shape, reference, like=video),
            coordinates((alens[index], 1, 1), ((alens[index]-1)//4+1, 1, 1), like=video),
            coordinates((tlens[index], 1, 1), (1, 1, 1), like=video, offset=-tlens[index]),
        )))
        modality_parts.append(torch.cat([torch.full((v.shape[0],), modality, device=video.device,
                                                  dtype=torch.long) for modality, v in enumerate(values)]))
        boundaries.append(boundaries[-1] + sum(v.shape[0] for v in values))
    return PackedAgnesInputs(torch.cat(token_parts), torch.cat(coord_parts),
                             torch.cat(modality_parts),
                             torch.tensor(boundaries, device=video.device, dtype=torch.int32),
                             tuple(video.shape), alens, tlens, int(audio.shape[-1]), patch_size)
