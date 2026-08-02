"""Behavioral contracts for grouped adjugate experts."""

from __future__ import annotations

import copy

import pytest
import torch

from mirai.config.schema import ConfigError, TrainingConfig
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.routing.adjugate_experts import (
    ADJUGATE_EXPERT_STATE_PREFIX,
    AdjugateExpertPool,
    AdjugateExpertTopology,
    aggregate_adjugate_group_routes,
    export_adjugate_expert_state,
    load_adjugate_expert_state,
)
from mirai.vendors.lingbot_video.transformer_lingbot_video import (
    LingBotVideoRuntimeOptions,
    LingBotVideoMLP,
    LingBotVideoSparseMoeBlock,
)


class _ScaleExpert(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.calls = 0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return tokens * self.scale


def _lingbot_pool(
    *,
    num_experts: int = 4,
    num_groups: int = 2,
    hidden_size: int = 4,
    intermediate_size: int = 3,
    scale: float = 0.25,
) -> AdjugateExpertPool:
    def factory(_group_index: int) -> LingBotVideoMLP:
        return LingBotVideoMLP(hidden_size, intermediate_size)

    def zero_output(expert: LingBotVideoMLP) -> None:
        with torch.no_grad():
            expert.down_proj.weight.zero_()

    return AdjugateExpertPool(
        topology=AdjugateExpertTopology(
            num_experts=num_experts,
            num_groups=num_groups,
            scale=scale,
        ),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        expert_kind="swiglu",
        expert_factory=factory,
        zero_output_initializer=zero_output,
    )


def _tiny_config(**params: object) -> TrainingConfig:
    model_params: dict[str, object] = {
        "variant": "tiny-video",
        "hidden_size": 12,
        "attention_heads": 2,
        "num_layers": 1,
        "num_experts": 4,
        "experts_per_token": 2,
        "shared_experts": 0,
        **params,
    }
    return TrainingConfig.from_dict(
        {
            "model": {"params": model_params},
            "adapter": {
                "type": "lora",
                "target_preset": "attn_only",
                "rank": 2,
                "alpha": 2,
            },
        }
    )


def test_topology_enforces_disjoint_groups_and_scale_bound() -> None:
    topology = AdjugateExpertTopology(
        num_experts=8,
        num_groups=4,
        scale=0.5,
    ).validate()
    assert topology.experts_per_group == 2
    assert topology.maximum_scale == 0.5
    with pytest.raises(ValueError, match="divide"):
        AdjugateExpertTopology(
            num_experts=8,
            num_groups=3,
            scale=0.1,
        ).validate()
    with pytest.raises(ValueError, match="num_groups / num_experts"):
        AdjugateExpertTopology(
            num_experts=8,
            num_groups=2,
            scale=0.3,
        ).validate()


def test_routes_sum_repeated_group_mass_exactly() -> None:
    decision = aggregate_adjugate_group_routes(
        torch.tensor([[0, 1, 4, 7], [2, 3, 2, 3]]),
        torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]),
        topology=AdjugateExpertTopology(
            num_experts=8,
            num_groups=4,
            scale=0.5,
        ),
    )
    torch.testing.assert_close(
        decision.group_weights,
        torch.tensor(
            [
                [0.3, 0.0, 0.3, 0.4],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(
        decision.active_group_mask,
        decision.group_weights != 0,
    )


def test_pool_evaluates_each_active_token_group_once() -> None:
    topology = AdjugateExpertTopology(
        num_experts=4,
        num_groups=2,
        scale=0.25,
    )
    pool = AdjugateExpertPool(
        topology=topology,
        hidden_size=2,
        intermediate_size=1,
        expert_kind="scale",
        expert_factory=lambda group: _ScaleExpert(group + 1),
        zero_output_initializer=lambda _expert: None,
    )
    tokens = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    output = pool.output_contribution(
        tokens,
        torch.tensor([[0, 1, 2], [2, 3, 2]]),
        torch.tensor([[0.3, 0.2, 0.5], [0.2, 0.3, 0.5]]),
    )
    torch.testing.assert_close(
        output,
        torch.stack((tokens[0] * 0.375, tokens[1] * 0.5)),
    )
    assert [expert.calls for expert in pool.experts] == [1, 1]


def test_zero_output_upcycling_and_gradient_unlock() -> None:
    torch.manual_seed(41)
    pool = _lingbot_pool()
    tokens = torch.randn(5, 4)
    indices = torch.tensor(
        [[0, 1], [2, 3], [0, 2], [1, 3], [0, 1]]
    )
    scores = torch.tensor(
        [[0.6, 0.4], [0.2, 0.8], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]
    )
    cold = pool(tokens, indices, scores)
    assert torch.equal(cold, torch.zeros_like(cold))
    cold.square().sum().backward()
    down_gradients = [
        expert.down_proj.weight.grad for expert in pool.experts
    ]
    assert all(gradient is not None for gradient in down_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in down_gradients)

    for parameter in pool.parameters():
        parameter.grad = None
    pool(tokens, indices, scores).sum().backward()
    assert any(
        expert.down_proj.weight.grad is not None
        and bool((expert.down_proj.weight.grad != 0).any())
        for expert in pool.experts
    )
    optimizer = torch.optim.SGD(pool.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    pool(tokens, indices, scores).square().sum().backward()
    assert any(
        expert.gate_proj.weight.grad is not None
        and bool((expert.gate_proj.weight.grad != 0).any())
        for expert in pool.experts
    )


def test_state_round_trip_is_topology_strict() -> None:
    source = torch.nn.Module()
    source.pool = _lingbot_pool()
    with torch.no_grad():
        source.pool.experts[0].down_proj.weight.add_(0.75)
    state = export_adjugate_expert_state(source)
    assert int(state[f"{ADJUGATE_EXPERT_STATE_PREFIX}schema_version"]) == 1

    target = torch.nn.Module()
    target.pool = _lingbot_pool()
    load_adjugate_expert_state(target, state)
    for key, value in source.pool.state_dict().items():
        torch.testing.assert_close(target.pool.state_dict()[key], value)

    mismatch = torch.nn.Module()
    mismatch.pool = _lingbot_pool(num_groups=4, scale=0.25)
    with pytest.raises(ValueError, match="topology"):
        load_adjugate_expert_state(mismatch, state)
    with pytest.raises(ValueError, match="no adjugate expert pool"):
        load_adjugate_expert_state(torch.nn.Module(), state)


def test_disabled_lingbot_path_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(73)
    baseline = LingBotVideoSparseMoeBlock(
        hidden_size=4,
        intermediate_size=8,
        num_experts=4,
        top_k=2,
        moe_intermediate_size=6,
        score_func="softmax",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=0,
    )
    with torch.no_grad():
        baseline.router.weight.normal_()
        for name in ("w1", "w2", "w3"):
            getattr(baseline.experts, name).normal_()
    candidate = copy.deepcopy(baseline)
    runtime_options = LingBotVideoRuntimeOptions(moe_expert_backend="loop")
    baseline._mirai_runtime_options = runtime_options
    candidate._mirai_runtime_options = runtime_options
    candidate.set_adjugate_expert_extension(None)
    left = torch.randn(2, 3, 4, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    left_output = baseline(left)
    right_output = candidate(right)
    assert torch.equal(left_output, right_output)
    left_output.square().sum().backward()
    right_output.square().sum().backward()
    assert torch.equal(left.grad, right.grad)
    for (_, left_parameter), (_, right_parameter) in zip(
        baseline.named_parameters(),
        candidate.named_parameters(),
        strict=True,
    ):
        assert torch.equal(left_parameter.grad, right_parameter.grad)


def test_pipeline_installs_trains_and_round_trips_adjugate_state() -> None:
    config = _tiny_config(
        moe_adjugate_experts=True,
        moe_adjugate_expert_groups=2,
        moe_adjugate_expert_intermediate_size=4,
        moe_adjugate_expert_scale=0.25,
    )
    torch.manual_seed(97)
    source = LingBotVideoPipeline.from_training_config(config)
    assert source.validate_config(config) == []
    source.set_adapter_config(config.adapter)
    pools = [
        module
        for module in source.transformer.modules()
        if isinstance(module, AdjugateExpertPool)
    ]
    assert len(pools) == 1
    assert all(parameter.requires_grad for parameter in pools[0].parameters())
    with torch.no_grad():
        pools[0].experts[0].down_proj.weight.add_(0.125)
    state = source.state_dict()
    assert any(
        str(key).startswith(ADJUGATE_EXPERT_STATE_PREFIX)
        for key in state
    )

    torch.manual_seed(97)
    target = LingBotVideoPipeline.from_training_config(config)
    target.set_adapter_config(config.adapter)
    target.load_adapter_state(state)
    target_pool = next(
        module
        for module in target.transformer.modules()
        if isinstance(module, AdjugateExpertPool)
    )
    for key, value in pools[0].state_dict().items():
        torch.testing.assert_close(target_pool.state_dict()[key], value)


def test_config_is_explicit_and_default_off() -> None:
    default = TrainingConfig()
    assert not default.model.params.moe_adjugate_experts
    assert default.model.params.moe_adjugate_expert_groups == 0
    pipeline = LingBotVideoPipeline.from_training_config(_tiny_config())
    assert not any(
        isinstance(module, AdjugateExpertPool)
        for module in pipeline.transformer.modules()
    )

    with pytest.raises(ConfigError, match="must be > 0"):
        _tiny_config(moe_adjugate_experts=True)
    with pytest.raises(ConfigError, match="must divide"):
        _tiny_config(
            moe_adjugate_experts=True,
            moe_adjugate_expert_groups=3,
        )
    with pytest.raises(ConfigError, match="must be <="):
        _tiny_config(
            moe_adjugate_experts=True,
            moe_adjugate_expert_groups=2,
            moe_adjugate_expert_scale=0.75,
        )
