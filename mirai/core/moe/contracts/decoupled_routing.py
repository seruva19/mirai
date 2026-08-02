"""Behavioral contracts for diffusion-aware decoupled routing."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from mirai.config.schema import TrainingConfig  # noqa: E402
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline  # noqa: E402
from mirai.core.moe.routing.decoupled import (  # noqa: E402
    DecoupledRouterConditioner,
    export_decoupled_routing_state,
    load_decoupled_routing_state,
)
from mirai.core.training.adapters import (  # noqa: E402
    load_adapter_payload,
    normalize_adapter_state,
    save_diffusers_adapter_safetensors,
)
from mirai.vendors.lingbot_video.transformer_lingbot_video import (  # noqa: E402
    LingBotVideoRouter,
)


def _conditioner() -> DecoupledRouterConditioner:
    return DecoupledRouterConditioner(
        hidden_size=3,
        num_experts=2,
        timestep_weight=0.5,
    )


def test_conditioner_is_additive_content_and_independent_timestep_projection() -> None:
    conditioner = _conditioner()
    with torch.no_grad():
        conditioner.timestep_projection.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        )
    content = torch.tensor([[[1.0, 2.0, 3.0]]])
    timestep = torch.tensor([[[4.0, 5.0, 6.0]]])
    content_weight = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    logits = conditioner(
        content_tokens=content,
        timestep_hidden=timestep,
        content_weight=content_weight,
    )
    torch.testing.assert_close(
        logits,
        torch.tensor([[[3.0, 7.0]]]),
    )


def test_router_uses_unmodulated_content_and_raw_timestep_only_when_enabled() -> None:
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
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
    router.set_expert_choice_extension(
        lambda logits, _dtype: logits.detach().clone()
    )
    content = torch.tensor([[[1.0, 2.0, 0.0]]])
    first_expert_input = torch.tensor([[[9.0, -4.0, 0.0]]])
    second_expert_input = torch.tensor([[[-8.0, 7.0, 0.0]]])
    timestep = torch.tensor([[[0.5, 0.25, 0.0]]])

    disabled_first = router.forward_expert_choice(
        first_expert_input, content, timestep
    )
    disabled_second = router.forward_expert_choice(
        second_expert_input, content, timestep
    )
    assert not torch.equal(disabled_first, disabled_second)

    conditioner = _conditioner()
    with torch.no_grad():
        conditioner.timestep_projection.fill_(0.25)
    router.set_decoupled_routing(conditioner)
    enabled_first = router.forward_expert_choice(
        first_expert_input, content, timestep
    )
    enabled_second = router.forward_expert_choice(
        second_expert_input, content, timestep
    )
    torch.testing.assert_close(enabled_first, enabled_second)


def test_conditioner_gradients_do_not_flow_through_modulated_expert_input() -> None:
    conditioner = _conditioner()
    content = torch.randn(1, 2, 3, requires_grad=True)
    timestep = torch.randn(1, 1, 3, requires_grad=True)
    expert_input = torch.randn(1, 2, 3, requires_grad=True)
    logits = conditioner(
        content_tokens=content,
        timestep_hidden=timestep,
        content_weight=torch.randn(2, 3),
    )
    logits.square().sum().backward()
    assert content.grad is not None
    assert timestep.grad is not None
    assert conditioner.timestep_projection.grad is not None
    assert expert_input.grad is None


def test_versioned_state_round_trip_and_topology_guard() -> None:
    source = torch.nn.Module()
    source.conditioner = _conditioner()
    with torch.no_grad():
        source.conditioner.timestep_projection.add_(0.75)
    state = export_decoupled_routing_state(source)

    target = torch.nn.Module()
    target.conditioner = _conditioner()
    load_decoupled_routing_state(target, state)
    torch.testing.assert_close(
        target.conditioner.timestep_projection,
        source.conditioner.timestep_projection,
    )

    tampered = copy.deepcopy(state)
    tampered["decoupled_routing.topology"]["conditioner"][
        "timestep_weight"
    ] = 0.25
    with pytest.raises(ValueError, match="topology"):
        load_decoupled_routing_state(target, tampered)


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
                    "moe_routing_mode": "expert_choice",
                    "moe_router_timestep_weight": 0.5,
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


def test_provider_adapter_lifecycle_trains_and_round_trips_conditioner() -> None:
    config = _pipeline_config()
    torch.manual_seed(37)
    source = LingBotVideoPipeline.from_training_config(config)
    assert source.validate_config(config) == []
    source.set_adapter_config(config.adapter)
    source_conditioners = [
        module
        for module in source.transformer.modules()
        if isinstance(module, DecoupledRouterConditioner)
    ]
    assert len(source_conditioners) == 1
    assert source_conditioners[0].timestep_projection.requires_grad
    with torch.no_grad():
        source_conditioners[0].timestep_projection.add_(0.125)
    state = source.state_dict()

    torch.manual_seed(37)
    target = LingBotVideoPipeline.from_training_config(config)
    target.set_adapter_config(config.adapter)
    target.load_adapter_state(state)
    target_conditioner = next(
        module
        for module in target.transformer.modules()
        if isinstance(module, DecoupledRouterConditioner)
    )
    torch.testing.assert_close(
        target_conditioner.timestep_projection,
        source_conditioners[0].timestep_projection,
    )


def test_portable_safetensors_preserves_decoupled_routing_state(
    tmp_path,
) -> None:
    root = torch.nn.Module()
    root.conditioner = _conditioner()
    state = {
        "layer.lora_a": torch.randn(2, 3),
        "layer.lora_b": torch.randn(4, 2),
        "layer.lora_alpha": torch.tensor(2.0),
        **export_decoupled_routing_state(root),
    }
    path = tmp_path / "adapter.safetensors"
    save_diffusers_adapter_safetensors(
        path,
        adapter_state=state,
        rank=2,
        alpha=2.0,
        target_modules=["layer"],
        model_path="model",
    )
    restored = normalize_adapter_state(
        load_adapter_payload(path),
        lora_format="auto",
    )
    target = torch.nn.Module()
    target.conditioner = _conditioner()
    load_decoupled_routing_state(target, restored)
    torch.testing.assert_close(
        target.conditioner.timestep_projection,
        root.conditioner.timestep_projection,
    )
