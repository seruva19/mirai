"""Executable contracts for model-agnostic conditioned inference."""

from __future__ import annotations

import pytest
import torch

from mirai.core.inference.conditioning import (
    IMAGE_TO_VIDEO,
    TEXT_TO_IMAGE,
    VIDEO_TO_VIDEO,
    InferenceConditioningRequest,
    PreparedInferenceConditioning,
    denoising_schedule_start,
    normalize_inference_task,
)
from mirai.core.training.preview.preview import run_native_denoise_loop


def test_task_aliases_and_invalid_media_combinations_fail_before_execution() -> None:
    assert normalize_inference_task("ti2v") == IMAGE_TO_VIDEO
    assert normalize_inference_task("v2v") == VIDEO_TO_VIDEO
    assert (
        InferenceConditioningRequest(task="t2i", frame_count=1).validate().task
        == TEXT_TO_IMAGE
    )
    with pytest.raises(ValueError, match="requires input_image"):
        InferenceConditioningRequest(task="ti2v").validate()
    with pytest.raises(ValueError, match="requires frame_count=1"):
        InferenceConditioningRequest(task="t2i", frame_count=17).validate()
    with pytest.raises(ValueError, match="only meaningful"):
        InferenceConditioningRequest(
            task="t2v", denoising_strength=0.5
        ).validate()


def test_condition_latent_pins_only_the_temporal_prefix() -> None:
    latent = torch.zeros(2, 4, 3, 3)
    condition = torch.full((1, 2, 1, 3, 3), 7.0)
    prepared = PreparedInferenceConditioning(
        task=IMAGE_TO_VIDEO,
        condition_latent=condition,
    )
    returned = prepared.pin_condition(latent)
    assert returned.data_ptr() == latent.data_ptr()
    assert torch.equal(latent[:, :1], torch.full((2, 1, 3, 3), 7.0))
    assert torch.count_nonzero(latent[:, 1:]) == 0


def test_v2v_schedule_strength_uses_standard_floor_truncation() -> None:
    assert denoising_schedule_start(20, 1.0) == 0
    assert denoising_schedule_start(20, 0.55) == 9
    assert denoising_schedule_start(20, 0.0) == 20


class _ConditionedPipeline:
    def __init__(self, prepared: PreparedInferenceConditioning) -> None:
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.prepared = prepared
        self.forward_inputs: list[torch.Tensor] = []

    def get_training_model(self):
        return self.model

    def preview_latent_geometry(self, *, frame_count: int, height: int, width: int):
        _ = frame_count, height, width
        return (1, 3, 2, 2)

    def prepare_inference_conditioning(self, request, *, device: str, generator):
        _ = request, device, generator
        return self.prepared

    def load_text_encoder(self, *, device: str) -> None:
        _ = device

    def offload_text_encoder(self) -> None:
        pass

    def encode_conditioned_prompt(self, prompt: str, *, prepared, device: str):
        _ = prompt, prepared
        return torch.zeros(1, 1, device=device)

    def resolve_flow_shift_for_latent_shape(self, latent_shape) -> float:
        _ = latent_shape
        return 1.0

    def forward(self, sample, timestep, text_embeds):
        _ = timestep, text_embeds
        self.forward_inputs.append(sample.detach().clone())
        return torch.zeros_like(sample)


def test_ti2v_condition_is_reapplied_before_every_forward_and_after_step() -> None:
    condition = torch.full((1, 1, 1, 2, 2), 3.0)
    prepared = PreparedInferenceConditioning(
        task=IMAGE_TO_VIDEO,
        condition_latent=condition,
    )
    pipeline = _ConditionedPipeline(prepared)
    result, _ = run_native_denoise_loop(
        pipeline=pipeline,
        prompt="p",
        negative_prompt="",
        cfg_scale=1.0,
        seed=1,
        step=0,
        denoise_steps=3,
        scheduler="euler",
        frame_count=9,
        height=16,
        width=16,
        conditioning=InferenceConditioningRequest(
            task=IMAGE_TO_VIDEO,
            input_image=object(),
            frame_count=9,
            height=16,
            width=16,
        ),
    )
    assert len(pipeline.forward_inputs) == 3
    for sample in pipeline.forward_inputs:
        assert torch.equal(sample[:, :, :1], torch.full((1, 1, 1, 2, 2), 3.0))
    assert torch.equal(result[:, :1], torch.full((1, 1, 2, 2), 3.0))


def test_zero_strength_v2v_returns_source_without_model_forward() -> None:
    source = torch.randn(1, 1, 3, 2, 2)
    prepared = PreparedInferenceConditioning(
        task=VIDEO_TO_VIDEO,
        source_latent=source.clone(),
        denoising_strength=0.0,
    )
    pipeline = _ConditionedPipeline(prepared)
    result, stats = run_native_denoise_loop(
        pipeline=pipeline,
        prompt="p",
        negative_prompt="",
        cfg_scale=1.0,
        seed=1,
        step=0,
        denoise_steps=4,
        scheduler="euler",
        frame_count=9,
        height=16,
        width=16,
        conditioning=InferenceConditioningRequest(
            task=VIDEO_TO_VIDEO,
            input_video=object(),
            denoising_strength=0.0,
            frame_count=9,
            height=16,
            width=16,
        ),
    )
    assert torch.equal(result, source[0])
    assert stats == []
    assert pipeline.forward_inputs == []
