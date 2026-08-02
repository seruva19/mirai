"""Batch collation helpers."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for collation: {exc}")


def _to_embed_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        out = value
    else:
        out = torch.tensor(value, dtype=torch.float32)
    if out.ndim == 1:
        return out.unsqueeze(-1).to(dtype=torch.float32)
    if out.ndim == 2:
        return out.to(dtype=torch.float32)
    raise ValueError(
        "Each sample text embedding must be rank-1 or rank-2 tensor "
        f"(got rank={out.ndim})."
    )


def collate_t5_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad variable-length T5 embeddings and produce an attention mask."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    latents = torch.stack([torch.as_tensor(sample["latents"]) for sample in samples], dim=0)
    embeds = [_to_embed_tensor(sample["text_embeds"]) for sample in samples]

    max_seq = max(int(embed.shape[0]) for embed in embeds)
    embed_dim = int(embeds[0].shape[1])
    for embed in embeds:
        if int(embed.shape[1]) != embed_dim:
            raise ValueError("All text embeddings in a batch must share embedding width.")

    padded = torch.zeros((len(samples), max_seq, embed_dim), dtype=embeds[0].dtype)
    mask = torch.zeros((len(samples), max_seq), dtype=torch.bool)
    for idx, embed in enumerate(embeds):
        seq = int(embed.shape[0])
        padded[idx, :seq, :] = embed
        mask[idx, :seq] = True

    out = {
        "latents": latents,
        "text_embeds": padded,
        "text_mask": mask,
    }
    if all("clip_embed" in sample for sample in samples):
        clip_embeds = [_to_embed_tensor(sample["clip_embed"]) for sample in samples]
        clip_max_seq = max(int(embed.shape[0]) for embed in clip_embeds)
        clip_dim = int(clip_embeds[0].shape[1])
        for embed in clip_embeds:
            if int(embed.shape[1]) != clip_dim:
                raise ValueError("All clip embeddings in a batch must share embedding width.")
        clip_padded = torch.zeros((len(samples), clip_max_seq, clip_dim), dtype=clip_embeds[0].dtype)
        for idx, embed in enumerate(clip_embeds):
            seq = int(embed.shape[0])
            clip_padded[idx, :seq, :] = embed
        out["clip_embed"] = clip_padded
    return out
