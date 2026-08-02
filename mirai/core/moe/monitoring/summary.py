"""Router diagnostics helpers."""

from __future__ import annotations

from typing import Any

from mirai.core.moe.routing.contracts import RoutingStats

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def summarize_routing_stats(
    stats: list[RoutingStats],
    *,
    include_expert_vectors: bool = True,
) -> dict[str, Any]:
    if not stats:
        return {
            "layers": 0,
            "dead_experts": 0,
            "mean_routing_entropy": 0.0,
            "mean_min_expert_fraction": 0.0,
            "mean_active_expert_fraction": 0.0,
            "mean_load_cv": 0.0,
            "max_to_mean_load": 0.0,
        }
    entropy = sum(item.routing_entropy for item in stats) / len(stats)
    min_fraction = sum(min(item.expert_fraction or (0.0,)) for item in stats) / len(stats)
    if include_expert_vectors:
        layers_detail = [item.as_dict() for item in stats]
    else:
        layers_detail = [
            {
                "layer": item.layer,
                "tokens": item.tokens,
                "selected_tokens": item.selected_tokens,
                "dead_experts": item.dead_experts,
                "routing_entropy": item.routing_entropy,
                "min_expert_fraction": min(item.expert_fraction or (0.0,)),
                "max_expert_fraction": max(item.expert_fraction or (0.0,)),
                "active_expert_fraction": item.active_expert_fraction,
                "load_cv": item.load_cv,
                "max_to_mean_load": item.max_to_mean_load,
            }
            for item in stats
        ]
    return {
        "layers": len(stats),
        "dead_experts": sum(item.dead_experts for item in stats),
        "mean_routing_entropy": float(entropy),
        "mean_min_expert_fraction": float(min_fraction),
        "mean_active_expert_fraction": float(
            sum(item.active_expert_fraction for item in stats) / len(stats)
        ),
        "mean_load_cv": float(sum(item.load_cv for item in stats) / len(stats)),
        "max_to_mean_load": float(max(item.max_to_mean_load for item in stats)),
        "layers_detail": layers_detail,
    }


def summarize_routing_by_diffusion_timestep(
    router_states: list[tuple[Any, int, int, int]],
    timesteps: Any,
) -> dict[str, Any]:
    """Report routed assignment fractions in fixed denoising-time buckets."""
    if torch is None or not torch.is_tensor(timesteps):
        return {}
    timestep_values = timesteps.detach().float().reshape(-1).cpu()
    boundaries = (0.0, 0.25, 0.5, 0.75, 1.000001)
    bucket_counts: dict[str, Any] = {}
    for bucket_idx in range(4):
        low, high = boundaries[bucket_idx], boundaries[bucket_idx + 1]
        label = f"{low:.2f}-{min(high, 1.0):.2f}"
        total_counts = None
        samples = 0
        for indices, batch_size, tokens_per_sample, num_experts in router_states:
            if int(batch_size) != int(timestep_values.numel()):
                continue
            shaped = indices.detach().reshape(
                int(batch_size), int(tokens_per_sample), -1
            ).cpu()
            mask = (timestep_values >= low) & (timestep_values < high)
            if not bool(mask.any()):
                continue
            selected = shaped[mask].reshape(-1)
            counts = torch.bincount(selected, minlength=int(num_experts)).float()
            total_counts = counts if total_counts is None else total_counts + counts
            samples += int(mask.sum().item())
        if total_counts is not None and float(total_counts.sum().item()) > 0.0:
            fractions = total_counts / total_counts.sum()
            bucket_counts[label] = {
                "samples": int(samples),
                "dead_experts": int((total_counts == 0).sum().item()),
                "min_expert_fraction": float(fractions.min().item()),
                "max_expert_fraction": float(fractions.max().item()),
                "active_expert_fraction": float((total_counts > 0).float().mean().item()),
                "load_cv": float(
                    (fractions.std(unbiased=False) / fractions.mean().clamp_min(1e-20)).item()
                ),
                "max_to_mean_load": float(
                    (fractions.max() / fractions.mean().clamp_min(1e-20)).item()
                ),
            }
    return bucket_counts
