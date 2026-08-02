"""Behavioral contracts for depth-aware router settings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mirai.config.schema import ConfigError, TrainingConfig  # noqa: E402
from mirai.core.models.lingbot_video.pipeline import (  # noqa: E402
    LingBotVideoPipeline,
)
from mirai.core.models.lingbot_video.router_runtime import (  # noqa: E402
    _weighted_router_auxiliary_losses,
)
from mirai.core.moe.routing.layer_policy import (  # noqa: E402
    LayerRouterBand,
    LayerRouterPolicy,
)


def _params(**overrides):
    values = {
        "moe_layer_router_policy": [],
        "num_layers": 4,
        "num_experts": 8,
        "experts_per_token": 2,
        "expert_subset_fraction": 1.0,
        "moe_router_z_loss_weight": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_empty_policy_constructs_no_runtime_owner() -> None:
    assert LayerRouterPolicy.from_model_params(_params()) is None


def test_policy_resolves_independent_depth_controls_and_fallbacks() -> None:
    policy = LayerRouterPolicy(
        (
            LayerRouterBand(
                first_layer=0,
                end_layer=1,
                top_k=1,
                subset_fraction=0.25,
                z_loss_weight=0.0,
            ),
            LayerRouterBand(
                first_layer=3,
                end_layer=4,
                top_k=4,
                z_loss_weight=0.2,
            ),
        ),
        num_layers=4,
        num_experts=8,
        fallback_top_k=2,
        fallback_subset_fraction=0.75,
        fallback_z_loss_weight=0.01,
    )
    assert policy.resolve(0).top_k == 1
    assert policy.resolve(0).subset_fraction == 0.25
    assert policy.resolve(0).z_loss_weight == 0.0
    assert policy.resolve(1).top_k == 2
    assert policy.resolve(1).subset_fraction == 0.75
    assert policy.resolve(3).top_k == 4
    assert policy.resolve(3).subset_fraction == 0.75
    assert policy.resolve(3).z_loss_weight == 0.2


def test_policy_rejects_overlap_and_subsets_narrower_than_routing_width() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        LayerRouterPolicy(
            (
                LayerRouterBand(first_layer=0, end_layer=2, top_k=1),
                LayerRouterBand(first_layer=1, end_layer=3, top_k=2),
            ),
            num_layers=3,
            num_experts=4,
            fallback_top_k=2,
            fallback_subset_fraction=1.0,
            fallback_z_loss_weight=0.0,
        )
    with pytest.raises(ValueError, match="at least top_k"):
        LayerRouterPolicy(
            (
                LayerRouterBand(
                    first_layer=0,
                    end_layer=1,
                    top_k=3,
                    subset_fraction=0.5,
                ),
            ),
            num_layers=2,
            num_experts=4,
            fallback_top_k=2,
            fallback_subset_fraction=1.0,
            fallback_z_loss_weight=0.0,
        )


def test_per_layer_z_loss_is_mean_of_individually_weighted_terms() -> None:
    terms = [torch.tensor(2.0), torch.tensor(3.0), torch.tensor(5.0)]
    actual = _weighted_router_auxiliary_losses(
        [],
        terms,
        load_balance_weight=0.0,
        z_loss_weight=0.1,
        z_loss_weights=[0.0, None, 0.4],
    )
    torch.testing.assert_close(
        actual["moe_router_z"],
        torch.tensor((0.0 * 2.0 + 0.1 * 3.0 + 0.4 * 5.0) / 3.0),
    )
    legacy = _weighted_router_auxiliary_losses(
        [],
        terms,
        load_balance_weight=0.0,
        z_loss_weight=0.1,
        z_loss_weights=[None, None, None],
    )
    expected_legacy = torch.stack(terms).mean() * 0.1
    assert torch.equal(legacy["moe_router_z"], expected_legacy)


def test_config_rejects_ambiguous_or_incompatible_layer_policies() -> None:
    base = {
        "model": {
            "params": {
                "variant": "tiny-video",
                "num_layers": 3,
                "num_experts": 4,
                "experts_per_token": 2,
            }
        }
    }
    base["model"]["params"]["moe_layer_router_policy"] = [
        {"first_layer": 0, "end_layer": 2, "top_k": 1},
        {"first_layer": 1, "end_layer": 3, "z_loss_weight": 0.1},
    ]
    with pytest.raises(ConfigError, match="must not overlap"):
        TrainingConfig.from_dict(base)

    base["model"]["params"]["moe_layer_router_policy"] = [
        {
            "first_layer": 0,
            "end_layer": 1,
            "top_k": 3,
            "subset_fraction": 0.5,
        }
    ]
    with pytest.raises(ConfigError, match="fewer subset experts"):
        TrainingConfig.from_dict(base)


def test_lingbot_provider_binds_top_k_subset_and_z_loss_by_layer() -> None:
    config = TrainingConfig.from_dict(
        {
            "model": {
                "params": {
                    "variant": "tiny-video",
                    "hidden_size": 12,
                    "attention_heads": 2,
                    "num_layers": 3,
                    "num_experts": 4,
                    "experts_per_token": 2,
                    "shared_experts": 0,
                    "latent_channels": 1,
                    "moe_router_z_loss_weight": 0.01,
                    "moe_layer_router_policy": [
                        {
                            "first_layer": 0,
                            "end_layer": 1,
                            "top_k": 1,
                            "subset_fraction": 0.5,
                            "z_loss_weight": 0.0,
                        },
                        {
                            "first_layer": 2,
                            "end_layer": 3,
                            "top_k": 3,
                            "z_loss_weight": 0.2,
                        },
                    ],
                }
            }
        }
    )
    pipeline = LingBotVideoPipeline.from_training_config(config)
    routers = pipeline._moe_router_modules
    assert [int(router.top_k) for router in routers] == [1, 2, 3]
    assert [
        float(getattr(router, "_mirai_z_loss_weight")) for router in routers
    ] == [0.0, 0.01, 0.2]
    pipeline.set_router_subset_progress(step=0, seed=7)
    assert [bool(router._subset_active) for router in routers] == [
        True,
        False,
        False,
    ]
    assert [int(router._subset_size) for router in routers] == [2, 4, 4]
    sparse_blocks = [
        block.ffn
        for block in pipeline.transformer.blocks
        if hasattr(getattr(block, "ffn", None), "router")
    ]
    hidden = torch.randn(1, 5, 12, requires_grad=True)
    output = sparse_blocks[0](hidden)
    output.square().mean().backward()
    assert hidden.grad is not None
    assert bool(torch.isfinite(hidden.grad).all().item())
    assert sparse_blocks[0].router.weight.grad is not None
    assert bool(torch.isfinite(sparse_blocks[0].router.weight.grad).all().item())
