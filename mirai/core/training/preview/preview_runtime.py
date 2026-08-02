"""Preview runtime lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class PreviewRuntimeContext:
    pipeline: Any
    previous_training: bool
    original_block_swap_state: dict[str, Any]
    block_swap_overridden: bool = False
    adapter_merged: bool = False
    adapter_unmerged: bool = False
    vae_loaded: bool = False
    vae_unloaded: bool = False
    scaling_factor: float = 0.18215


def begin_preview_runtime(
    *,
    pipeline: Any,
    sample_blocks_to_swap: int,
    training_blocks_to_swap: int,
    block_swap_mode: str,
    block_swap_backward: bool,
) -> PreviewRuntimeContext:
    previous_training = True
    if hasattr(pipeline, "_is_training"):
        previous_training = bool(getattr(pipeline, "_is_training"))
    pipeline.eval()
    context = PreviewRuntimeContext(
        pipeline=pipeline,
        previous_training=previous_training,
        original_block_swap_state=_snapshot_block_swap_state(pipeline),
        scaling_factor=float(pipeline.get_vae_scaling_factor()),
    )
    context.block_swap_overridden = _override_preview_block_swap(
        pipeline=pipeline,
        original_state=context.original_block_swap_state,
        sample_blocks_to_swap=sample_blocks_to_swap,
        training_blocks_to_swap=training_blocks_to_swap,
        block_swap_mode=block_swap_mode,
        block_swap_backward=block_swap_backward,
    )
    return context


def activate_preview_runtime_assets(context: PreviewRuntimeContext) -> None:
    pipeline = context.pipeline
    if bool(pipeline.supports_adapter_merge_unmerge()):
        context.adapter_merged = bool(pipeline.merge_adapter())
    if bool(pipeline.supports_preview_latent_decode()):
        context.scaling_factor = float(pipeline.load_vae_to_gpu())
        context.vae_loaded = True


def restore_preview_runtime(context: PreviewRuntimeContext) -> None:
    pipeline = context.pipeline
    if bool(pipeline.supports_preview_latent_decode()):
        pipeline.unload_vae_from_gpu()
        context.vae_unloaded = True
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if bool(pipeline.supports_adapter_merge_unmerge()):
        context.adapter_unmerged = bool(pipeline.unmerge_adapter()) or bool(context.adapter_merged)
    _restore_block_swap_state(pipeline, context.original_block_swap_state)
    if context.previous_training:
        pipeline.train()


def _snapshot_block_swap_state(pipeline: Any) -> dict[str, Any]:
    state = pipeline.get_block_swap_state()
    if not isinstance(state, dict):
        return {"enabled": False}
    return dict(state)


def _restore_block_swap_state(pipeline: Any, state: dict[str, Any]) -> None:
    if bool(pipeline.supports_weight_residency_strategy()):
        pipeline.set_weight_residency_strategy(
            strategy=str(state.get("weight_residency_strategy", "disabled")),
            blocks_to_swap=int(state.get("blocks_to_swap", 0)),
            mode=str(state.get("mode", "sync")),
            block_swap_backward=bool(state.get("block_swap_backward", True)),
        )
        return
    if not bool(pipeline.supports_block_swap()):
        return
    if not state.get("enabled"):
        pipeline.set_block_swap(blocks_to_swap=0, mode="sync", block_swap_backward=True)
        return
    pipeline.set_block_swap(
        blocks_to_swap=int(state.get("blocks_to_swap", 0)),
        mode=str(state.get("mode", "sync")),
        block_swap_backward=bool(state.get("block_swap_backward", True)),
    )


def _override_preview_block_swap(
    *,
    pipeline: Any,
    original_state: dict[str, Any],
    sample_blocks_to_swap: int,
    training_blocks_to_swap: int,
    block_swap_mode: str,
    block_swap_backward: bool,
) -> bool:
    supports_weight_residency = bool(pipeline.supports_weight_residency_strategy())
    supports_block_swap = bool(pipeline.supports_block_swap())
    if (
        int(sample_blocks_to_swap) >= 0
        and int(training_blocks_to_swap) > int(sample_blocks_to_swap)
        and supports_weight_residency
    ):
        pipeline.set_weight_residency_strategy(
            strategy=str(original_state.get("weight_residency_strategy", "block_swap")),
            blocks_to_swap=int(sample_blocks_to_swap),
            mode=str(block_swap_mode),
            block_swap_backward=bool(block_swap_backward),
        )
        return True
    if (
        int(sample_blocks_to_swap) >= 0
        and int(training_blocks_to_swap) > int(sample_blocks_to_swap)
        and supports_block_swap
    ):
        pipeline.set_block_swap(
            blocks_to_swap=int(sample_blocks_to_swap),
            mode=str(block_swap_mode),
            block_swap_backward=bool(block_swap_backward),
        )
        return True
    return False
