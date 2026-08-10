"""Contracts for persistent grouped-GEMM autotune warm-up."""

from __future__ import annotations

import pytest

from mirai.core.moe.runtime.autotune_warmup import (
    GroupedGemmWarmupProblem,
    grouped_gemm_warmup_problems,
    warmup_persistent_grouped_gemm,
)
from mirai.core.moe.runtime.specs import ExpertTensorSpec
from mirai.core.training.runtime import trainer
from mirai.config.schema import MemoryConfig, TrainingConfig


def _spec(name: str, role: str, shape: tuple[int, int, int]) -> ExpertTensorSpec:
    return ExpertTensorSpec(
        name=name,
        owner_module="blocks.0.experts",
        tensor_name=name,
        role=role,
        layout=("expert", "out", "in"),
        shape=shape,
    )


def test_provider_specs_produce_unique_autotune_keys() -> None:
    specs = [
        _spec("w1", "gate", (8, 32, 16)),
        _spec("w3", "up", (8, 32, 16)),
        _spec("w2", "down", (8, 16, 32)),
    ]
    assert grouped_gemm_warmup_problems(specs, routed_rows=64) == (
        GroupedGemmWarmupProblem(8, 16, 32, 64),
        GroupedGemmWarmupProblem(8, 32, 16, 64),
    )


def test_empty_problem_list_is_a_dependency_free_noop() -> None:
    assert warmup_persistent_grouped_gemm(()) == ()


def test_invalid_warmup_dimensions_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="positive"):
        GroupedGemmWarmupProblem(8, 16, 32, 0)
    with pytest.raises(ValueError, match="routed_rows"):
        grouped_gemm_warmup_problems([], routed_rows=0)


def test_trainer_default_does_not_inspect_provider(monkeypatch) -> None:
    class Pipeline:
        def get_expert_tensor_specs(self):
            raise AssertionError("disabled warm-up must not inspect the provider")

    monkeypatch.setattr(
        trainer,
        "warmup_persistent_grouped_gemm",
        lambda problems: (_ for _ in ()).throw(AssertionError("unexpected warm-up")),
    )
    trainer._warmup_moe_autotune(config=TrainingConfig(), pipeline=Pipeline())


def test_trainer_routes_provider_shapes_to_warmup(monkeypatch) -> None:
    captured = []
    config = TrainingConfig(
        memory=MemoryConfig(
            moe_dispatch="triton_persistent", moe_autotune_warmup_rows=32
        )
    )

    class Pipeline:
        def get_expert_tensor_specs(self):
            return [_spec("w1", "gate", (4, 16, 8))]

    monkeypatch.setattr(
        trainer,
        "warmup_persistent_grouped_gemm",
        lambda problems: captured.extend(problems),
    )
    trainer._warmup_moe_autotune(config=config, pipeline=Pipeline())
    assert captured == [GroupedGemmWarmupProblem(4, 8, 16, 32)]


def test_trainer_rejects_warmup_for_nonpersistent_dispatch() -> None:
    config = TrainingConfig(
        memory=MemoryConfig(moe_autotune_warmup_rows=32)
    )
    with pytest.raises(ValueError, match="triton_persistent"):
        trainer._warmup_moe_autotune(config=config, pipeline=object())
