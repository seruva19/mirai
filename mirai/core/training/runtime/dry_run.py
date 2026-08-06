"""Dry-run diagnostics for the training CLI."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mirai.config.runtime_policy import runtime_policy_summary
from mirai.core.models.base import SyntheticBatchSpec
from mirai.core.tensors import to_jsonable, to_python_scalar
from mirai.core.training.data.batches import grad_norm
from mirai.core.training.residency.device_placement import resolve_compute_dtype

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def build_dry_run_report(
    *,
    trainer: Any,
    config: Any,
    runtime_policy_notes: list[str],
) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch is required for dry-run diagnostics.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    batch = _build_synthetic_moe_batch(
        config,
        pipeline=getattr(trainer, "pipeline", None),
        dtype=_resolve_pipeline_input_dtype(trainer, config),
    )
    params = list(trainer.get_trainable_parameters())
    for param in params:
        param.grad = None
    compute_loss_result = getattr(trainer, "compute_loss_result", None)
    if callable(compute_loss_result):
        loss_result = compute_loss_result(batch)
        loss = loss_result.loss
        loss_pre_accum = loss_result.loss_pre_accum
        auxiliary_losses = loss_result.auxiliary_losses or {}
        diagnostics = loss_result.diagnostics or {}
    else:
        loss, raw = trainer.compute_loss(batch)
        loss_pre_accum = raw["loss_pre_accum"]
        auxiliary_losses = raw.get("auxiliary_losses", {})
        diagnostics = raw.get("diagnostics", {})
    if torch.is_tensor(loss):
        loss.backward()
    grad_norm_value = grad_norm(params)
    vram_peak_mb = 0.0
    if torch.cuda.is_available():
        vram_peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    compilation_diagnostics_getter = getattr(
        trainer, "get_compilation_diagnostics", None
    )
    compilation_diagnostics = (
        compilation_diagnostics_getter()
        if callable(compilation_diagnostics_getter)
        else {}
    )
    return {
        "status": "dry_run_ok",
        "loss": to_python_scalar(loss),
        "loss_pre_accum": to_python_scalar(loss_pre_accum),
        "grad_norm": grad_norm_value,
        "trainable_params": sum(int(p.numel()) for p in params),
        "vram_peak_mb": vram_peak_mb,
        "compile_enabled": bool(getattr(trainer, "compile_enabled", False)),
        "compile_warning": str(getattr(trainer, "compile_warning", "")),
        "compilation": compilation_diagnostics,
        "runtime_policy": runtime_policy_summary(config),
        "runtime_policy_notes": list(runtime_policy_notes),
        "auxiliary_losses": to_jsonable(auxiliary_losses),
        "diagnostics": to_jsonable(diagnostics),
        "dry_run_batch": {
            "latent_shape": list(batch["latents"].shape),
            "text_embed_shape": list(batch["text_embeds"].shape),
        },
        "resolved_config": asdict(config),
    }


def _resolve_synthetic_batch_spec(pipeline: Any) -> SyntheticBatchSpec:
    """Read the model-owned synthetic-batch contract, if the model declares one.

    Keeps the dry-run harness family-agnostic: a model whose forward validates
    conditioning widths states them here, everything else keeps the generic
    config-derived shapes.
    """
    getter = getattr(pipeline, "get_synthetic_batch_spec", None)
    return getter() if callable(getter) else SyntheticBatchSpec()


def _build_synthetic_moe_batch(
    config: Any,
    *,
    pipeline: Any | None = None,
    dtype: Any | None = None,
) -> dict[str, Any]:
    batch_size = int(config.training.batch_size)
    input_dtype = resolve_compute_dtype(config) if dtype is None else dtype
    params = config.model.params
    spec = _resolve_synthetic_batch_spec(pipeline)
    frame_buckets = list(getattr(config.dataset, "frame_buckets", []) or [1])
    frame_count = max(1, int(frame_buckets[0]))
    channels = max(
        1,
        int(
            params.latent_channels
            if spec.latent_channels is None
            else spec.latent_channels
        ),
    )
    patch = max(1, int(params.patch_size))
    height = max(patch * 2, 2)
    width = max(patch * 2, 2)
    numel = batch_size * channels * frame_count * height * width
    latents = torch.linspace(0.1, 0.9, steps=max(1, numel), dtype=input_dtype)
    latents = latents.reshape(batch_size, channels, frame_count, height, width)
    text_embed_width = spec.text_embed_width
    text_embeds = (
        torch.ones((batch_size,), dtype=input_dtype)
        if text_embed_width is None
        else torch.ones(
            (batch_size, max(1, int(spec.text_embed_tokens)), int(text_embed_width)),
            dtype=input_dtype,
        )
    )
    return {
        "latents": latents,
        "text_embeds": text_embeds,
    }


def _resolve_pipeline_input_dtype(trainer: Any, config: Any) -> Any:
    pipeline = getattr(trainer, "pipeline", None)
    get_training_model = getattr(pipeline, "get_training_model", None)
    model = get_training_model() if callable(get_training_model) else None
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if bool(parameter.is_floating_point()):
                return parameter.dtype
    return resolve_compute_dtype(config)
