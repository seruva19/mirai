"""Contracts for progressive resolution/frame/task curriculum."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
import torch

from mirai.config.schema import StrategyConfig, TrainingConfig
from mirai.core.dataset.cache import build_cache
from mirai.core.dataset.registration import register_dataset
from mirai.core.training.data.batches import build_batch_from_records
from mirai.core.training.data.curriculum import CurriculumSchedule
from mirai.core.training.lifecycle.training_step_pre import (
    StepSamplingContext,
    _build_training_batch_factory,
)
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.core.training.strategies.multi_task_video import MultiTaskVideoStrategy


def _schedule() -> CurriculumSchedule:
    return CurriculumSchedule.from_config(
        {
            "enabled": True,
            "resolution_schedule": {"0": "256x256", "10": "512x512"},
            "frame_schedule": {"0": 1, "10": 17},
            "task_mix_schedule": {
                "0": {"text_to_image": 1},
                "10": {"text_to_video": 6, "image_to_video": 3},
            },
        }
    )


def _record(sample_id: str, task: str, resolution: int, frames: int) -> dict:
    return {
        "sample_id": sample_id,
        "latent": torch.zeros(1, frames, 2, 2),
        "text_embed": torch.zeros(2, 4),
        "bucket_h": resolution,
        "bucket_w": resolution,
        "bucket_frames": frames,
        "metadata": {"training_task": task},
    }


def test_profile_uses_real_cache_bucket_fields_and_stage_task_mix() -> None:
    schedule = _schedule()
    records = [
        _record("image", "text_to_image", 256, 1),
        _record("video", "text_to_video", 512, 17),
        _record("i2v", "image_to_video", 512, 17),
    ]
    assert [record["sample_id"] for record in schedule.filter_records(records, step=0)] == [
        "image"
    ]
    assert [record["sample_id"] for record in schedule.filter_records(records, step=10)] == [
        "video",
        "i2v",
    ]
    assert schedule.profile_for_step(10).task_weights == (
        ("text_to_video", 6.0),
        ("image_to_video", 3.0),
    )
    schedule.validate_records(records)


def test_task_choice_is_stateless_and_resume_exact() -> None:
    schedule = _schedule()
    first = [
        schedule.select_task(step=10, seed=41, global_batch_index=index)
        for index in range(64)
    ]
    replay = [
        schedule.select_task(step=10, seed=41, global_batch_index=index)
        for index in range(64)
    ]
    assert first == replay
    assert {"text_to_video", "image_to_video"} <= set(first)


def test_positive_weight_without_eligible_records_fails_before_training() -> None:
    records = [
        _record("image", "text_to_image", 256, 1),
        _record("video", "text_to_video", 512, 17),
    ]
    with pytest.raises(ValueError, match="image_to_video"):
        _schedule().validate_records(records)


def test_batch_materializes_one_canonical_task() -> None:
    records = [
        _record("a", "text_to_video", 512, 17),
        _record("b", "text_to_video", 512, 17),
    ]
    batch = build_batch_from_records(records)
    assert batch["training_task"] == "text_to_video"

    mixed = [records[0], _record("c", "image_to_video", 512, 17)]
    with pytest.raises(ValueError, match="cannot mix"):
        build_batch_from_records(mixed)


class _Sampler:
    def sample(self, batch_size, *, like):
        return torch.zeros(batch_size, dtype=like.dtype, device=like.device)


class _Noise:
    def sample(self, like):
        return torch.ones_like(like)


class _Pipeline:
    def __init__(self) -> None:
        self.seen_shapes: list[tuple[int, ...]] = []

    def apply_noise(self, latents, noise, timesteps):
        _ = timesteps
        return latents + noise

    def prepare_model_timesteps(self, timesteps, *, latents):
        self.seen_shapes.append(tuple(int(value) for value in latents.shape))
        return timesteps

    def i2v_conditioning_frame_dim(self, *, latents):
        _ = latents
        return 2

    def i2v_conditioning_forward_kwargs(self, *, condition_frame_indexes):
        return {"condition_frame_indexes": condition_frame_indexes}


def test_multi_task_strategy_delegates_and_recomputes_shape_dependent_timestep() -> None:
    strategy = MultiTaskVideoStrategy(
        StrategyConfig(
            type="multi_task_video",
            params={"first_frame_conditioning_p": 1.0},
        )
    )
    pipeline = _Pipeline()
    sampler = _Sampler()
    noise = _Noise()

    image_latents = torch.zeros(2, 4, 1, 2, 2)
    image_inputs = strategy.prepare_inputs(
        {
            "training_task": "text_to_image",
            "latents": image_latents,
            "text_embeds": torch.zeros(2, 2, 4),
        },
        pipeline,
        sampler,
        noise,
    )
    assert image_inputs.loss_mask is None

    video_latents = torch.zeros(2, 4, 3, 2, 2)
    i2v_inputs = strategy.prepare_inputs(
        {
            "training_task": "image_to_video",
            "latents": video_latents,
            "text_embeds": torch.zeros(2, 2, 4),
        },
        pipeline,
        sampler,
        noise,
    )
    assert torch.count_nonzero(i2v_inputs.loss_mask[:, :, 0]) == 0
    assert pipeline.seen_shapes == [
        tuple(image_latents.shape),
        tuple(video_latents.shape),
    ]


def test_unknown_curriculum_keys_and_tasks_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unknown training.curriculum"):
        CurriculumSchedule.from_config({"enabled": True, "stgae": {}})
    with pytest.raises(ValueError, match="unsupported tasks"):
        CurriculumSchedule.from_config(
            {
                "enabled": True,
                "task_mix_schedule": {"0": {"unknown_task": 1}},
            }
        )


def test_cache_copies_required_training_task_from_registration(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save(torch.tensor([0.1]), data_dir / "sample.pt")
    (data_dir / "sample.txt").write_text("caption", encoding="utf-8")
    registration_path = data_dir / "registration.json"
    registration = register_dataset(
        dataset_path=data_dir,
        output_path=registration_path,
        split_seed=1,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        compliance_enabled=False,
        usage_mode="internal",
    )
    registration["samples"][0]["training_task"] = "text_to_video"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")

    payload = build_cache(
        data_dir,
        tmp_path / "cache.pt",
        required_registration_metadata_keys=("training_task",),
    )
    assert payload["records"][0]["metadata"]["training_task"] == "text_to_video"


class _RecordSelectingPolicy:
    def __init__(self) -> None:
        self.seen_records: list[list[dict]] = []

    def select_records(self, context):
        self.seen_records.append(list(context.records))
        return None

    def augment_batch(self, batch, **_kwargs):
        return batch


def test_task_filter_composes_before_other_record_selection_policies() -> None:
    curriculum = CurriculumSchedule.from_config(
        {
            "enabled": True,
            "task_mix_schedule": {"0": {"text_to_video": 1}},
        }
    )
    records = [
        _record("t2v", "text_to_video", 512, 17),
        _record("i2v", "image_to_video", 512, 17),
    ]
    policies = _RecordSelectingPolicy()
    session = SimpleNamespace(
        config=SimpleNamespace(
            dataset=SimpleNamespace(
                online_temporal_resampling=False,
                online_tag_shuffle=False,
                online_tag_shuffle_dropout=0.0,
                online_tag_shuffle_keep_first_n_tags=1,
            ),
            training=SimpleNamespace(batch_size=1, masked_loss=False, seed=7),
        ),
        run_state=SimpleNamespace(global_step=0),
        grad_accum=1,
        curriculum=curriculum,
        trainer=SimpleNamespace(training_policies=policies),
        compute_device=torch.device("cpu"),
        compute_dtype=torch.float32,
        rng=random.Random(7),
    )
    context = StepSamplingContext(
        curriculum_profile=curriculum.profile_for_step(0),
        eligible_records=records,
        temporal_base_ids_step=[],
        temporal_groups_step={},
        epoch_index=0,
    )
    batch = _build_training_batch_factory(
        session=session,
        sampling_context=context,
    )(0)
    assert batch["training_task"] == "text_to_video"
    assert [
        record["metadata"]["training_task"]
        for record in policies.seen_records[0]
    ] == ["text_to_video"]


def test_runtime_contract_binds_task_mix_to_multi_task_strategy() -> None:
    config = TrainingConfig()
    config.training.curriculum = {
        "enabled": True,
        "task_mix_schedule": {"0": {"text_to_video": 1}},
    }
    with pytest.raises(ValueError, match="multi_task_video"):
        validate_training_runtime_config(config)

    config.strategy = StrategyConfig(type="multi_task_video")
    validate_training_runtime_config(config)
