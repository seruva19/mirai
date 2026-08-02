"""Behavioral contracts for step-conditioned progressive MoE sparsification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mirai.config.schema import ConfigError, TrainingConfig  # noqa: E402
from mirai.core.models.lingbot_video.pipeline import (  # noqa: E402
    LingBotVideoPipeline,
)
from mirai.core.moe.routing.progressive_sparsification import (  # noqa: E402
    ProgressiveSparsificationBand,
    ProgressiveSparsificationPolicy,
)


def _params(**overrides):
    values = {
        "moe_progressive_sparsification_transition_step": 0,
        "moe_progressive_sparsification_policy": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _config(**overrides) -> TrainingConfig:
    params = {
        "variant": "tiny-video",
        "hidden_size": 12,
        "attention_heads": 2,
        "num_layers": 3,
        "num_experts": 4,
        "experts_per_token": 1,
        "shared_experts": 0,
        "latent_channels": 1,
    }
    params.update(overrides)
    return TrainingConfig.from_dict({"model": {"params": params}})


def test_empty_schedule_constructs_no_runtime_owner() -> None:
    assert (
        ProgressiveSparsificationPolicy.from_model_params(
            _params(),
            target_top_k=(1, 1),
            num_experts=4,
        )
        is None
    )


def test_policy_switches_exactly_at_transition_boundary() -> None:
    policy = ProgressiveSparsificationPolicy(
        (
            ProgressiveSparsificationBand(
                first_layer=0,
                end_layer=1,
                top_k=4,
            ),
            ProgressiveSparsificationBand(
                first_layer=1,
                end_layer=3,
                top_k=2,
            ),
        ),
        transition_step=90,
        target_top_k=(1, 1, 1, 1),
        num_experts=4,
    )
    assert [policy.top_k(layer_index=i, step=0) for i in range(4)] == [
        4,
        2,
        2,
        1,
    ]
    assert [policy.top_k(layer_index=i, step=89) for i in range(4)] == [
        4,
        2,
        2,
        1,
    ]
    assert [policy.top_k(layer_index=i, step=90) for i in range(4)] == [
        1,
        1,
        1,
        1,
    ]
    assert policy.is_early(step=89)
    assert not policy.is_early(step=90)


def test_policy_rejects_non_sparsifying_or_overlapping_bands() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        ProgressiveSparsificationPolicy(
            (
                ProgressiveSparsificationBand(
                    first_layer=0,
                    end_layer=1,
                    top_k=1,
                ),
            ),
            transition_step=10,
            target_top_k=(1, 1),
            num_experts=4,
        )
    with pytest.raises(ValueError, match="must not overlap"):
        ProgressiveSparsificationPolicy(
            (
                ProgressiveSparsificationBand(
                    first_layer=0,
                    end_layer=2,
                    top_k=3,
                ),
                ProgressiveSparsificationBand(
                    first_layer=1,
                    end_layer=3,
                    top_k=2,
                ),
            ),
            transition_step=10,
            target_top_k=(1, 1, 1),
            num_experts=4,
        )


def test_config_requires_complete_compatible_schedule() -> None:
    with pytest.raises(ConfigError, match="requires both"):
        _config(moe_progressive_sparsification_transition_step=10)
    with pytest.raises(ConfigError, match="must exceed"):
        _config(
            moe_progressive_sparsification_transition_step=10,
            moe_progressive_sparsification_policy=[
                {"first_layer": 0, "end_layer": 1, "top_k": 1}
            ],
        )
    with pytest.raises(ConfigError, match="cannot compose with moe_dynamic_topk"):
        _config(
            moe_progressive_sparsification_transition_step=10,
            moe_progressive_sparsification_policy=[
                {"first_layer": 0, "end_layer": 1, "top_k": 2}
            ],
            moe_dynamic_topk_min=1,
            moe_dynamic_topk_average=1.5,
        )


def test_lingbot_provider_applies_early_width_then_restores_target() -> None:
    pipeline = LingBotVideoPipeline.from_training_config(
        _config(
            moe_progressive_sparsification_transition_step=10,
            moe_progressive_sparsification_policy=[
                {"first_layer": 0, "end_layer": 1, "top_k": 4},
                {"first_layer": 1, "end_layer": 3, "top_k": 2},
            ],
        )
    )
    routers = pipeline._moe_router_modules
    assert pipeline.supports_progressive_sparsification_progress()
    assert [int(router.top_k) for router in routers] == [1, 1, 1]

    pipeline.set_progressive_sparsification_progress(step=0)
    assert [int(router.top_k) for router in routers] == [4, 2, 2]
    pipeline.set_progressive_sparsification_progress(step=9)
    assert [int(router.top_k) for router in routers] == [4, 2, 2]
    pipeline.set_progressive_sparsification_progress(step=10)
    assert [int(router.top_k) for router in routers] == [1, 1, 1]


def test_early_and_target_forwards_are_finite_and_differentiable() -> None:
    pipeline = LingBotVideoPipeline.from_training_config(
        _config(
            moe_progressive_sparsification_transition_step=2,
            moe_progressive_sparsification_policy=[
                {"first_layer": 0, "end_layer": 3, "top_k": 2},
            ],
        )
    )
    pipeline.train()
    block = next(
        block.ffn
        for block in pipeline.transformer.blocks
        if hasattr(getattr(block, "ffn", None), "router")
    )
    hidden = torch.randn(1, 5, 12, requires_grad=True)

    pipeline.set_progressive_sparsification_progress(step=0)
    early = block(hidden)
    assert early.shape == hidden.shape
    assert bool(torch.isfinite(early).all().item())
    early.square().mean().backward()
    assert hidden.grad is not None
    assert bool(torch.isfinite(hidden.grad).all().item())

    hidden_target = hidden.detach().clone().requires_grad_(True)
    pipeline.set_progressive_sparsification_progress(step=2)
    target = block(hidden_target)
    assert target.shape == hidden_target.shape
    assert bool(torch.isfinite(target).all().item())
    target.square().mean().backward()
    assert hidden_target.grad is not None
    assert bool(torch.isfinite(hidden_target.grad).all().item())
