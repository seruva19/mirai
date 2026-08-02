"""Preview sample assembly for training-loop callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.config.schema import TrainingConfig
from mirai.core.training.optim.gradients import release_gradient_memory
from mirai.core.training.optim.optimizer import offload_optimizer_state_to_cpu
from mirai.core.training.preview.preview import generate_preview
from mirai.core.training.training_policy import PreviewRequestContext

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PreviewSampleResult:
    sample_name: str
    preview: dict[str, Any]
    metrics: dict[str, Any]


def _resolve_sample_pack(config: TrainingConfig) -> list[tuple[str, int, str]]:
    prompts = [str(v) for v in getattr(config.logging, "sample_prompts", []) if str(v).strip()]
    seeds = [int(v) for v in getattr(config.logging, "sample_seeds", [])]
    if not prompts:
        prompts = [str(config.logging.sample_prompt)]
    if not seeds:
        seeds = [int(config.logging.sample_seed)]
    out: list[tuple[str, int, str]] = []
    for idx, prompt in enumerate(prompts):
        seed = seeds[idx] if idx < len(seeds) else seeds[-1]
        name = "preview" if len(prompts) == 1 else f"preview_{idx}"
        out.append((prompt, int(seed), name))
    return out


def _prepare_preview_pause(
    *,
    params: list[Any],
    optimizer: Any,
    optimizer_cpu_offload: bool,
) -> dict[str, int]:
    # Preview runs after the optimizer step; gradients are stale and only pin
    # VRAM, so release them (CUDA-safe) rather than relocating to CPU.
    grad_offloads = release_gradient_memory(params)
    opt_offloads = 0
    if bool(optimizer_cpu_offload):
        opt_offloads = offload_optimizer_state_to_cpu(optimizer)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "preview_pause_gradient_offload_ops": int(grad_offloads),
        "preview_pause_optimizer_offload_ops": int(opt_offloads),
    }


def generate_pack_preview(
    *,
    config: TrainingConfig,
    trainer: Any,
    step: int,
    output_dir: str | Path,
    prompt: str,
    seed: int,
    sample_name: str,
    curriculum_resolution: Any = None,
    curriculum_frame_count: Any = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Run one preview through ``generate_preview`` with the standard wiring.

    The default preview loop and any preview-owning training policy share this,
    so a plugin reuses the exact config-sourced arguments and overrides only what
    it needs (e.g. ``denoise_steps`` for a custom timestep grid, ``sample_name``
    for A/B variants) instead of re-deriving every argument by hand.
    """
    kwargs: dict[str, Any] = dict(
        trainer=trainer,
        step=step,
        output_dir=output_dir,
        prompt=prompt,
        seed=seed,
        sample_name=sample_name,
        sample_blocks_to_swap=config.logging.sample_blocks_to_swap,
        training_blocks_to_swap=config.training.blocks_to_swap,
        block_swap_mode=config.training.block_swap_mode,
        block_swap_backward=config.training.block_swap_backward,
        sample_cfg_scale=config.logging.sample_cfg_scale,
        sample_negative_prompt=config.logging.sample_negative_prompt,
        sample_solver=config.logging.sample_solver,
        sample_resolution=(
            str(curriculum_resolution)
            if curriculum_resolution is not None
            else config.logging.sample_resolution
        ),
        sample_frame_count=(
            int(curriculum_frame_count)
            if curriculum_frame_count is not None
            else config.logging.sample_frame_count
        ),
    )
    kwargs.update(overrides)
    return generate_preview(**kwargs)


def build_preview_samples(
    *,
    config: TrainingConfig,
    trainer: Any,
    step: int,
    output_dir: str | Path,
    params: list[Any],
    optimizer: Any,
    base_metrics: dict[str, Any],
    curriculum_resolution: Any = None,
    curriculum_frame_count: Any = None,
    force: bool = False,
    sample_name_override: str = "",
    prompt_override: str = "",
) -> list[PreviewSampleResult]:
    sample_every = int(config.logging.sample_every_n_steps)
    if not bool(force) and (sample_every <= 0 or int(step) % sample_every != 0):
        return []
    metrics_with_pause = dict(base_metrics)
    metrics_with_pause.update(
        _prepare_preview_pause(
            params=params,
            optimizer=optimizer,
            optimizer_cpu_offload=config.training.optimizer_cpu_offload,
        )
    )
    results: list[PreviewSampleResult] = []
    sample_pack = _resolve_sample_pack(config)
    if bool(force):
        prompt_value = str(prompt_override).strip()
        sample_name_value = str(sample_name_override).strip() or "manual_preview"
        first_prompt, first_seed, _ = sample_pack[0]
        sample_pack = [
            (
                prompt_value or str(first_prompt),
                int(first_seed),
                sample_name_value,
            )
        ]

    # Preview generation is a pluggable step: an active training policy may own
    # it exclusively (custom timestep grids, A/B adapter variants, skip/extend
    # packs). With no preview-owning policy (the default) this returns None and
    # the standard loop below runs byte-identically.
    training_policies = getattr(trainer, "training_policies", None)
    if training_policies is not None:
        claimed = training_policies.run_previews(
            PreviewRequestContext(
                config=config,
                trainer=trainer,
                step=int(step),
                output_dir=output_dir,
                sample_pack=tuple(sample_pack),
                base_metrics=dict(metrics_with_pause),
                curriculum_resolution=curriculum_resolution,
                curriculum_frame_count=curriculum_frame_count,
                force=bool(force),
            )
        )
        if claimed is not None:
            return list(claimed)

    for prompt, seed, sample_name in sample_pack:
        preview = generate_pack_preview(
            config=config,
            trainer=trainer,
            step=step,
            output_dir=output_dir,
            prompt=prompt,
            seed=seed,
            sample_name=sample_name,
            curriculum_resolution=curriculum_resolution,
            curriculum_frame_count=curriculum_frame_count,
        )
        preview["sample_name"] = sample_name
        preview_metrics = dict(metrics_with_pause)
        preview_metrics.update(preview)
        results.append(
            PreviewSampleResult(
                sample_name=str(sample_name),
                preview=dict(preview),
                metrics=preview_metrics,
            )
        )
    return results
