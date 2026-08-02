"""Bounded routing consistency over a video token grid.

Conceptual source: Graph of Tokens, https://arxiv.org/abs/2505.00792.
"""

from __future__ import annotations

from typing import Any, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def sampled_spatiotemporal_edges(
    grid: tuple[int, int, int],
    *,
    max_edges: int,
    device: Any,
) -> tuple[Any, Any]:
    """Return deterministic adjacent-token edges without materializing the graph."""
    frames, height, width = (int(value) for value in grid)
    if min(frames, height, width) <= 0:
        raise ValueError("Video routing grid dimensions must be positive.")
    if int(max_edges) <= 0:
        raise ValueError("Video routing max_edges must be positive.")
    temporal = max(0, frames - 1) * height * width
    vertical = frames * max(0, height - 1) * width
    horizontal = frames * height * max(0, width - 1)
    total = temporal + vertical + horizontal
    count = min(int(max_edges), total)
    if count == 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty
    edge_ids = torch.div(
        torch.arange(count, device=device, dtype=torch.long) * total,
        count,
        rounding_mode="floor",
    )
    source = torch.empty_like(edge_ids)
    target = torch.empty_like(edge_ids)

    is_temporal = edge_ids < temporal
    local = edge_ids[is_temporal]
    source[is_temporal] = local
    target[is_temporal] = local + height * width

    is_vertical = (edge_ids >= temporal) & (edge_ids < temporal + vertical)
    local = edge_ids[is_vertical] - temporal
    frame = torch.div(local, (height - 1) * width, rounding_mode="floor")
    within = local.remainder((height - 1) * width)
    source[is_vertical] = frame * height * width + within
    target[is_vertical] = source[is_vertical] + width

    is_horizontal = edge_ids >= temporal + vertical
    local = edge_ids[is_horizontal] - temporal - vertical
    frame_row = torch.div(local, width - 1, rounding_mode="floor")
    column = local.remainder(width - 1)
    source[is_horizontal] = frame_row * width + column
    target[is_horizontal] = source[is_horizontal] + 1
    return source, target


def spatiotemporal_routing_consistency_loss(
    probabilities: Any,
    *,
    video_offsets: Sequence[int],
    grid: tuple[int, int, int],
    max_edges: int,
) -> Any:
    """Penalize routing-distribution differences between adjacent video tokens."""
    if not torch.is_tensor(probabilities) or probabilities.ndim != 2:
        raise ValueError("Router probabilities must have shape [tokens, experts].")
    video_tokens = int(grid[0]) * int(grid[1]) * int(grid[2])
    offsets = tuple(int(value) for value in video_offsets)
    if not offsets:
        raise ValueError("At least one video-token offset is required.")
    if any(value < 0 or value + video_tokens > int(probabilities.shape[0]) for value in offsets):
        raise ValueError("Video-token span exceeds router probability rows.")
    source, target = sampled_spatiotemporal_edges(
        grid, max_edges=int(max_edges), device=probabilities.device
    )
    if source.numel() == 0:
        return probabilities.float().sum() * 0.0
    normalized = probabilities.float()
    normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    losses = []
    for offset in offsets:
        left = normalized.index_select(0, source + offset)
        right = normalized.index_select(0, target + offset)
        losses.append((left - right).square().sum(dim=-1).mean())
    return torch.stack(losses).mean()


__all__ = [
    "sampled_spatiotemporal_edges",
    "spatiotemporal_routing_consistency_loss",
]
