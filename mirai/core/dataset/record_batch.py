"""Helpers for converting cached record payloads into batched tensors."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for record batching: {exc}")


def latent_value_to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    return torch.as_tensor(value, dtype=torch.float32)


def text_embed_value_to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float()
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    if tensor.ndim == 2:
        return tensor
    # Flatten trailing dims while preserving sequence axis.
    return tensor.reshape(tensor.shape[0], -1)


def stack_latent_tensors(values: list[Any]) -> torch.Tensor:
    tensors = [latent_value_to_tensor(v) for v in values]
    if not tensors:
        raise ValueError("Cannot stack an empty latent list.")
    max_rank = max(int(t.ndim) for t in tensors)
    aligned: list[torch.Tensor] = []
    for t in tensors:
        out = t
        while out.ndim < max_rank:
            out = out.unsqueeze(0)
        aligned.append(out)
    max_shape = [max(int(t.shape[d]) for t in aligned) for d in range(max_rank)]
    batch = torch.zeros((len(aligned), *max_shape), dtype=torch.float32)
    for idx, t in enumerate(aligned):
        slices = tuple(slice(0, int(dim)) for dim in t.shape)
        batch[(idx, *slices)] = t
    return batch


def pad_text_embeddings(values: list[Any]) -> torch.Tensor:
    embeds = [text_embed_value_to_tensor(v) for v in values]
    if not embeds:
        raise ValueError("Cannot pad an empty embedding list.")
    max_seq = max(int(e.shape[0]) for e in embeds)
    max_dim = max(int(e.shape[1]) for e in embeds)
    padded = torch.zeros((len(embeds), max_seq, max_dim), dtype=torch.float32)
    for idx, embed in enumerate(embeds):
        seq, dim = int(embed.shape[0]), int(embed.shape[1])
        padded[idx, :seq, :dim] = embed
    return padded
