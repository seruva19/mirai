"""Behavioral contracts for the model-agnostic lightweight expert pool."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from mirai.core.moe.routing.lightweight_experts import (  # noqa: E402
    LightweightExpertPool,
    _null_aware_balance,
    export_lightweight_expert_state,
    load_lightweight_expert_state,
)
from mirai.config.schema import TrainingConfig  # noqa: E402
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline  # noqa: E402
from mirai.core.training.adapters import (  # noqa: E402
    load_adapter_payload,
    normalize_adapter_state,
    save_kohya_adapter_safetensors,
)
from mirai.vendors.lingbot_video.transformer_lingbot_video import (  # noqa: E402
    LingBotVideoRouter,
    LingBotVideoSparseMoeBlock,
)


def _pool(
    *,
    zero_experts: int = 2,
    copy_experts: int = 0,
    constant_experts: int = 0,
    top_k: int = 2,
) -> LightweightExpertPool:
    lightweight_experts = zero_experts + copy_experts + constant_experts
    return LightweightExpertPool(
        physical_experts=2,
        hidden_size=3,
        zero_experts=zero_experts,
        copy_experts=copy_experts,
        constant_experts=constant_experts,
        top_k=top_k,
        balance_mode="global",
        initial_router_rows=torch.zeros(lightweight_experts, 3),
    )


def test_null_aware_balance_uses_one_average_frequency_for_null_slots() -> None:
    probabilities = torch.tensor(
        [[0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]],
        requires_grad=True,
    )
    selected = torch.tensor([[0, 2], [2, 3]])
    loss = _null_aware_balance(
        probabilities,
        selected,
        physical_experts=2,
        zero_experts=2,
        batch_size=1,
        tokens_per_sample=2,
        mode="global",
    )
    torch.testing.assert_close(loss, torch.tensor(2.2))
    loss.backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()


def test_selected_zero_slots_do_not_dilute_physical_gate_mass() -> None:
    pool = _pool(zero_experts=2, top_k=3)
    decision = pool.route(
        torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
        physical_correction_bias=torch.zeros(2),
        score_func="softmax",
        route_scale=1.0,
        physical_choice_transform=None,
        batch_size=1,
        tokens_per_sample=1,
        training=True,
    )
    assert int(decision.physical_active_mask.sum()) == 2
    torch.testing.assert_close(
        decision.physical_scores.sum(dim=-1), torch.ones(1)
    )
    active_scores = decision.physical_scores[
        decision.physical_active_mask
    ].sort(descending=True).values
    torch.testing.assert_close(
        active_scores,
        torch.softmax(torch.tensor([4.0, 3.0]), dim=0),
    )


def test_all_zero_selection_bypasses_physical_dispatch() -> None:
    pool = _pool(zero_experts=2, top_k=2)
    decision = pool.route(
        torch.tensor([[-4.0, -3.0, 4.0, 3.0]]),
        physical_correction_bias=torch.zeros(2),
        score_func="softmax",
        route_scale=1.0,
        physical_choice_transform=None,
        batch_size=1,
        tokens_per_sample=1,
        training=True,
    )
    assert not bool(decision.physical_active_mask.any())
    assert torch.equal(
        decision.physical_scores, torch.zeros_like(decision.physical_scores)
    )


def test_copy_expert_returns_identity_without_physical_dispatch() -> None:
    pool = _pool(
        zero_experts=0,
        copy_experts=1,
        top_k=1,
    )
    tokens = torch.tensor([[1.0, -2.0, 3.0]])
    decision = pool.route(
        torch.tensor([[-4.0, -3.0, 4.0]]),
        physical_correction_bias=torch.zeros(2),
        score_func="softmax",
        route_scale=1.0,
        physical_choice_transform=None,
        batch_size=1,
        tokens_per_sample=1,
        training=True,
    )
    assert not bool(decision.physical_active_mask.any())
    torch.testing.assert_close(
        pool.output_contribution(
            tokens,
            decision.logical_indices,
            decision.logical_output_scores,
        ),
        tokens,
    )


def test_constant_expert_matches_two_way_mixture_and_trains() -> None:
    pool = _pool(
        zero_experts=0,
        constant_experts=1,
        top_k=1,
    )
    with torch.no_grad():
        pool.constant_vectors.copy_(torch.tensor([[3.0, 1.0, -1.0]]))
    tokens = torch.tensor([[1.0, -1.0, 3.0]])
    decision = pool.route(
        torch.tensor([[-4.0, -3.0, 4.0]]),
        physical_correction_bias=torch.zeros(2),
        score_func="softmax",
        route_scale=1.0,
        physical_choice_transform=None,
        batch_size=1,
        tokens_per_sample=1,
        training=True,
    )
    contribution = pool.output_contribution(
        tokens,
        decision.logical_indices,
        decision.logical_output_scores,
    )
    torch.testing.assert_close(
        contribution,
        0.5 * tokens + 0.5 * pool.constant_vectors,
    )
    contribution.square().sum().backward()
    assert pool.constant_vectors.grad is not None
    assert pool.constant_gates.grad is not None
    assert torch.isfinite(pool.constant_vectors.grad).all()
    assert torch.isfinite(pool.constant_gates.grad).all()


def test_nonzero_routes_share_one_output_normalization() -> None:
    pool = _pool(
        zero_experts=1,
        copy_experts=1,
        top_k=2,
    )
    tokens = torch.tensor([[2.0, 1.0, -1.0]])
    logits = torch.tensor([[3.0, -4.0, -5.0, 2.0]])
    decision = pool.route(
        logits,
        physical_correction_bias=torch.zeros(2),
        score_func="softmax",
        route_scale=1.0,
        physical_choice_transform=None,
        batch_size=1,
        tokens_per_sample=1,
        training=False,
    )
    copy_weight = decision.logical_output_scores[
        decision.logical_indices == pool.copy_start
    ]
    physical_weight = decision.physical_scores[
        decision.physical_active_mask
    ]
    torch.testing.assert_close(
        copy_weight + physical_weight,
        torch.ones(1),
    )


def test_pretrained_router_row_derivation_is_deterministic_and_cyclic() -> None:
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    pool = LightweightExpertPool.from_physical_router(
        weight,
        zero_experts=3,
        top_k=2,
        balance_mode="sequence",
    )
    torch.testing.assert_close(
        pool.router_weight,
        torch.stack((weight[0], weight[1], weight[0])),
    )


def test_pool_router_rows_receive_finite_gradients() -> None:
    pool = _pool(zero_experts=2, top_k=2)
    tokens = torch.tensor([[1.0, -1.0, 0.5], [0.5, 1.0, -1.0]])
    logical = pool.append_logits(tokens, torch.zeros(2, 2))
    decision = pool.route(
        logical,
        physical_correction_bias=torch.zeros(2),
        score_func="softmax",
        route_scale=1.0,
        physical_choice_transform=None,
        batch_size=1,
        tokens_per_sample=2,
        training=True,
    )
    (decision.load_balance_loss + decision.z_loss).backward()
    assert pool.router_weight.grad is not None
    assert torch.isfinite(pool.router_weight.grad).all()


def test_lingbot_provider_dispatches_only_physical_indices() -> None:
    block = LingBotVideoSparseMoeBlock(
        hidden_size=3,
        intermediate_size=6,
        num_experts=2,
        top_k=1,
        moe_intermediate_size=4,
        score_func="softmax",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=0,
    )
    with torch.no_grad():
        block.router.weight.fill_(-1.0)
        for name in ("w1", "w2", "w3"):
            getattr(block.experts, name).normal_()
    pool = LightweightExpertPool(
        physical_experts=2,
        hidden_size=3,
        zero_experts=1,
        top_k=1,
        balance_mode="sequence",
        initial_router_rows=torch.ones(1, 3),
    )
    block.router.lightweight_experts = pool
    block.router.set_lightweight_expert_extension(pool)
    block.train()
    output = block(torch.ones(1, 2, 3))
    assert torch.equal(output, torch.zeros_like(output))
    assert not bool(block.router.last_route_active_mask.any())
    assert int(block.router.last_top_indices.max()) < 2


def test_disabled_router_path_is_object_identical() -> None:
    router = LingBotVideoRouter(
        hidden_size=3,
        num_experts=2,
        top_k=1,
        score_func="softmax",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        route_scale=1.0,
    )
    router.eval()
    tokens = torch.randn(4, 3)
    before = router(tokens)
    router.set_lightweight_expert_extension(None)
    after = router(tokens)
    for left, right in zip(before, after, strict=True):
        assert torch.equal(left, right)


def test_versioned_state_round_trip_and_topology_tamper_rejection() -> None:
    source = torch.nn.Module()
    source.pool = _pool()
    with torch.no_grad():
        source.pool.router_weight.add_(1.25)
        source.pool.correction_bias.copy_(torch.tensor([0.1, -0.2]))
    state = export_lightweight_expert_state(source)

    target = torch.nn.Module()
    target.pool = _pool()
    load_lightweight_expert_state(target, state)
    torch.testing.assert_close(
        target.pool.router_weight, source.pool.router_weight
    )
    torch.testing.assert_close(
        target.pool.correction_bias, source.pool.correction_bias
    )

    tampered = copy.deepcopy(state)
    tampered["lightweight_experts.topology"]["pool"]["top_k"] = 1
    with pytest.raises(ValueError, match="topology"):
        load_lightweight_expert_state(target, tampered)


def test_enabled_pool_requires_state_and_disabled_model_rejects_it() -> None:
    enabled = torch.nn.Module()
    enabled.pool = _pool()
    with pytest.raises(ValueError, match="requires adapter state"):
        load_lightweight_expert_state(enabled, {})

    state = export_lightweight_expert_state(enabled)
    with pytest.raises(ValueError, match="has no lightweight expert pool"):
        load_lightweight_expert_state(torch.nn.Module(), state)


def _pipeline_config() -> TrainingConfig:
    return TrainingConfig.from_dict(
        {
            "model": {
                "params": {
                    "variant": "tiny-video",
                    "hidden_size": 12,
                    "attention_heads": 2,
                    "num_layers": 1,
                    "num_experts": 2,
                    "experts_per_token": 1,
                    "shared_experts": 0,
                    "moe_zero_experts": 1,
                    "moe_copy_experts": 1,
                    "moe_constant_experts": 1,
                    "moe_lightweight_top_k": 2,
                }
            },
            "adapter": {
                "type": "lora",
                "target_preset": "attn_only",
                "rank": 2,
                "alpha": 2,
            },
        }
    )


def test_provider_adapter_lifecycle_trains_and_round_trips_pool_state() -> None:
    config = _pipeline_config()
    torch.manual_seed(91)
    source = LingBotVideoPipeline.from_training_config(config)
    assert source.validate_config(config) == []
    source.set_adapter_config(config.adapter)
    source_pools = [
        module
        for module in source.transformer.modules()
        if isinstance(module, LightweightExpertPool)
    ]
    assert len(source_pools) == 1
    assert source_pools[0].router_weight.requires_grad
    with torch.no_grad():
        source_pools[0].router_weight.add_(0.75)
    state = source.state_dict()

    torch.manual_seed(91)
    target = LingBotVideoPipeline.from_training_config(config)
    target.set_adapter_config(config.adapter)
    target.load_adapter_state(state)
    target_pool = next(
        module
        for module in target.transformer.modules()
        if isinstance(module, LightweightExpertPool)
    )
    torch.testing.assert_close(
        target_pool.router_weight, source_pools[0].router_weight
    )
    assert any(
        "lightweight_experts.router_weight" in name
        for name, _parameter in target.get_named_trainable_parameters()
    )


def test_portable_safetensors_preserves_mirai_pool_extension(tmp_path) -> None:
    root = torch.nn.Module()
    root.pool = _pool()
    state = {
        "layer.lora_a": torch.randn(2, 3),
        "layer.lora_b": torch.randn(4, 2),
        "layer.lora_alpha": torch.tensor(2.0),
        **export_lightweight_expert_state(root),
    }
    path = tmp_path / "adapter.safetensors"
    save_kohya_adapter_safetensors(
        path,
        adapter_state=state,
        rank=2,
        alpha=2.0,
        target_modules=["layer"],
        model_path="model",
    )
    restored_state = normalize_adapter_state(
        load_adapter_payload(path), lora_format="auto"
    )
    target = torch.nn.Module()
    target.pool = _pool()
    load_lightweight_expert_state(target, restored_state)
    torch.testing.assert_close(
        target.pool.router_weight, root.pool.router_weight
    )
