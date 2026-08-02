"""LingBot binding for model-agnostic Mixture-of-Depths policies."""

from __future__ import annotations

from typing import Any

from mirai.core.moe.routing.depth import MixtureOfDepthsSpec


def configure_lingbot_depth_policy(
    pipeline: Any,
    *,
    policy_name: str,
    policy: Any,
) -> bool:
    if policy_name != "mixture_of_depths":
        return False
    if not isinstance(policy, MixtureOfDepthsSpec):
        raise TypeError("mixture_of_depths requires MixtureOfDepthsSpec.")
    spec = policy.validate()
    blocks = tuple(pipeline.transformer.blocks)
    routed = frozenset(spec.routed_layers(len(blocks)))
    if not routed:
        raise ValueError(
            "Mixture-of-Depths schedule does not select any transformer blocks."
        )
    pipeline._mixture_of_depths_spec = spec
    pipeline.transformer._mirai_mixture_of_depths_spec = spec
    for index, block in enumerate(blocks):
        block._mirai_mixture_of_depths_capacity_fraction = (
            float(spec.capacity_fraction) if index in routed else 0.0
        )
        block.attn._mirai_capture_received_attention = index + 1 in routed
        block.attn._mirai_attention_query_chunk_size = int(
            spec.attention_query_chunk_size
        )
    return True


def mixture_of_depths_diagnostics(transformer: Any) -> dict[str, float]:
    spec = getattr(transformer, "_mirai_mixture_of_depths_spec", None)
    if not isinstance(spec, MixtureOfDepthsSpec):
        return {}
    selections = [
        getattr(block, "_mirai_last_depth_selection", None)
        for block in transformer.blocks
    ]
    selections = [selection for selection in selections if selection is not None]
    if not selections:
        return {}
    selected = sum(sum(item.selected_visual_tokens) for item in selections)
    processed = sum(sum(item.processed_tokens) for item in selections)
    return {
        "mixture_of_depths_capacity_fraction": float(spec.capacity_fraction),
        "mixture_of_depths_routed_layer_count": float(len(selections)),
        "mixture_of_depths_selected_visual_tokens": float(selected),
        "mixture_of_depths_processed_tokens": float(processed),
    }


__all__ = [
    "configure_lingbot_depth_policy",
    "mixture_of_depths_diagnostics",
]
