"""Behavioral contracts for frozen-base representation preservation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mirai.config.schema import TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.training.policies.representation_preservation import (
    RepresentationPreservationTrainingPolicy,
)
from mirai.core.training.policies.representation_preservation import (
    _validate_config,
)
from mirai.core.training.policies.representation_preservation import (
    representation_cosine_loss,
)
from mirai.core.training.training_policy import PredictionContext
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import TrainingPolicySet
from mirai.core.training.trainer import Trainer


class _Pipeline:
    def __init__(self) -> None:
        self.scale = 1.0

    def get_lora_scale(self) -> float:
        return self.scale

    def set_lora_scale(self, scale: float) -> None:
        self.scale = float(scale)


def _policy() -> RepresentationPreservationTrainingPolicy:
    return RepresentationPreservationTrainingPolicy(
        weight=0.25,
        teacher_fingerprint="base:v1",
        epsilon=1e-8,
    )


def test_cosine_formula_and_teacher_detachment() -> None:
    student = torch.tensor([[1.0, 0.0], [1.0, 1.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)

    loss = representation_cosine_loss(student, teacher)
    expected = torch.tensor((1.0 + (1.0 - 2.0**-0.5)) / 2.0)
    torch.testing.assert_close(loss, expected)
    loss.backward()

    assert student.grad is not None
    assert teacher.grad is None


def test_policy_runs_same_input_frozen_base_and_restores_rng_and_scale() -> None:
    policy = _policy()
    pipeline = _Pipeline()
    weight = torch.tensor(2.0, requires_grad=True)
    observed: list[tuple[float, torch.Tensor]] = []

    def predict(_inputs):
        noise = torch.rand(2, 3)
        observed.append((pipeline.scale, noise.detach().clone()))
        return noise + pipeline.scale * weight

    student = policy.predict(
        pipeline=pipeline,
        inputs=SimpleNamespace(),
        predict=predict,
        training=True,
    )
    losses = policy.prediction_auxiliary_losses(
        PredictionContext(
            batch={},
            inputs=SimpleNamespace(),
            prediction=student,
            training=True,
        )
    )

    assert [scale for scale, _noise in observed] == [0.0, 1.0]
    torch.testing.assert_close(observed[0][1], observed[1][1])
    assert pipeline.scale == 1.0
    assert losses["representation_preservation"].grad_fn is not None
    losses["representation_preservation"].backward()
    assert weight.grad is not None


def test_policy_restores_scale_when_teacher_forward_fails() -> None:
    policy = _policy()
    pipeline = _Pipeline()

    with pytest.raises(RuntimeError, match="teacher failed"):
        policy.predict(
            pipeline=pipeline,
            inputs=SimpleNamespace(),
            predict=lambda _inputs: (_ for _ in ()).throw(RuntimeError("teacher failed")),
            training=True,
        )
    assert pipeline.scale == 1.0


def test_validation_rejects_mutable_or_unsupported_teacher_paths() -> None:
    config = TrainingConfig.from_dict(
        {
            "adapter": {"type": "lora", "train_router": False},
            "training": {
                "policy_options": {
                    "representation_preservation": {
                        "enabled": True,
                        "weight": 0.1,
                    }
                }
            },
        }
    )
    assert _validate_config(config) == []
    config.adapter.train_router = True
    assert any("train_router=false" in error for error in _validate_config(config))


def test_policy_set_default_path_runs_one_unmodified_prediction() -> None:
    calls = 0

    def predict(_inputs):
        nonlocal calls
        calls += 1
        return torch.ones(1, 2)

    result = TrainingPolicySet().predict(
        pipeline=_Pipeline(),
        inputs=SimpleNamespace(),
        predict=predict,
        training=True,
    )
    torch.testing.assert_close(result, torch.ones(1, 2))
    assert calls == 1


def test_policy_set_rejects_prediction_owner_collision_before_execution() -> None:
    calls = 0

    class _Owner(TrainingPolicy):
        def __init__(self, name: str) -> None:
            self.name = name

        def claims_prediction(self) -> bool:
            return True

        def predict(self, **_kwargs):
            nonlocal calls
            calls += 1
            return torch.ones(1, 2)

    with pytest.raises(ValueError, match="Multiple training policies"):
        TrainingPolicySet([_Owner("left"), _Owner("right")]).predict(
            pipeline=_Pipeline(),
            inputs=SimpleNamespace(),
            predict=lambda _inputs: torch.ones(1, 2),
            training=True,
        )
    assert calls == 0


def test_policy_skips_prior_regularization_batches() -> None:
    policy = _policy()
    pipeline = _Pipeline()
    policy.before_forward(
        pipeline,
        {"source_type": "regularization"},
        training=True,
    )
    calls = 0

    def predict(_inputs):
        nonlocal calls
        calls += 1
        return torch.ones(1, 2)

    result = policy.predict(
        pipeline=pipeline,
        inputs=SimpleNamespace(),
        predict=predict,
        training=True,
    )
    torch.testing.assert_close(result, torch.ones(1, 2))
    assert calls == 1
    assert policy.prediction_auxiliary_losses(
        PredictionContext(
            batch={"source_type": "regularization"},
            inputs=SimpleNamespace(),
            prediction=torch.ones(1, 2),
            training=True,
        )
    ) == {}


def test_trainer_adds_loss_and_adapter_gradients_on_native_testbed() -> None:
    register_builtin_components()
    config = TrainingConfig.from_dict(
        {
            "model": {
                "type": "sparse_moe_test",
                "path": "./models/sparse_moe_test",
                "params": {
                    "variant": "tiny-video",
                    "num_experts": 4,
                    "experts_per_token": 2,
                    "shared_experts": 1,
                    "hidden_size": 16,
                    "num_layers": 2,
                },
            },
            "adapter": {"type": "lora", "train_router": False},
            "training": {
                "batch_size": 2,
                "gradient_checkpointing": "off",
                "policy_options": {
                    "representation_preservation": {
                        "enabled": True,
                        "weight": 0.1,
                    }
                },
            },
        }
    )
    trainer = Trainer(config)
    loss, raw = trainer.compute_loss(
        {
            "latents": torch.randn(2, 1, 2, 2, 2),
            "text_embeds": torch.ones(2, 4),
        }
    )
    assert torch.isfinite(loss)
    assert "representation_preservation" in raw["auxiliary_losses"]
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainer.get_trainable_parameters()
    )
