"""Behavioral contracts for the checkpoint-preserving CoE adaptation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from mirai.config.schema import ConfigError, TrainingConfig
from mirai.core.training.adapters import (
    load_adapter_payload,
    normalize_adapter_state,
    save_diffusers_adapter_safetensors,
)
from mirai.core.training.runtime.gpu_lease import (
    acquire_gpu_lease,
    resolve_lease_lock_path,
)
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.routing.chain_of_experts import (
    CHAIN_OF_EXPERTS_STATE_PREFIX,
    ChainOfExpertsExtension,
    ChainOfExpertsSpec,
    chain_of_experts_metrics,
    export_chain_of_experts_state,
    load_chain_of_experts_state,
)
from mirai.vendors.lingbot_video.transformer_lingbot_video import (
    LingBotVideoSparseMoeBlock,
)


def _extension(*, hidden_size: int = 4, num_experts: int = 3, rank: int = 2):
    return ChainOfExpertsExtension(
        ChainOfExpertsSpec(
            hidden_size=hidden_size,
            num_experts=num_experts,
            router_rank=rank,
        )
    )


def _block() -> LingBotVideoSparseMoeBlock:
    block = LingBotVideoSparseMoeBlock(
        hidden_size=4,
        intermediate_size=8,
        num_experts=3,
        top_k=2,
        moe_intermediate_size=5,
        score_func="softmax",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=1,
    )
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    return block


def _config(**model_params) -> TrainingConfig:
    params = {
        "variant": "tiny-video",
        "hidden_size": 12,
        "attention_heads": 2,
        "num_layers": 1,
        "num_experts": 2,
        "experts_per_token": 1,
        "shared_experts": 0,
        "moe_chain_of_experts": True,
        "moe_chain_router_rank": 2,
        **model_params,
    }
    return TrainingConfig.from_dict(
        {
            "model": {"params": params},
            "adapter": {
                "type": "lora",
                "target_preset": "attn_only",
                "rank": 2,
                "alpha": 2,
            },
        }
    )


def test_spec_is_two_step_and_validates_rank() -> None:
    assert _extension().topology()["communication_steps"] == 2
    with pytest.raises(ValueError, match="router_rank"):
        ChainOfExpertsSpec(hidden_size=4, num_experts=3, router_rank=0).validate()
    with pytest.raises(ValueError, match="exactly two"):
        ChainOfExpertsSpec(
            hidden_size=4,
            num_experts=3,
            router_rank=2,
            communication_steps=3,
        ).validate()


def test_low_rank_router_delta_and_zero_scale_formula() -> None:
    extension = _extension()
    with torch.no_grad():
        extension.router_down.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
        )
        extension.router_up.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
        )
    tokens = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    torch.testing.assert_close(
        extension.router_logit_delta(tokens),
        torch.tensor([[2.0, 6.0, -4.0]]),
    )
    first = torch.randn(2, 3, 4)
    continuation = torch.randn_like(first)
    assert torch.equal(extension.combine(first, continuation), first)
    with torch.no_grad():
        extension.continuation_scale.fill_(0.25)
    torch.testing.assert_close(
        extension.combine(first, continuation),
        first + continuation * 0.25,
    )


def test_enabled_zero_scale_preserves_native_block_output_exactly() -> None:
    torch.manual_seed(11)
    block = _block().eval()
    hidden = torch.randn(2, 5, 4)
    native = block(hidden)
    block.set_chain_of_experts_extension(_extension())
    adapted = block(hidden)
    assert torch.equal(adapted, native)


def test_two_step_recurrence_matches_explicit_reference() -> None:
    torch.manual_seed(17)
    block = _block().eval()
    extension = _extension()
    with torch.no_grad():
        extension.router_up.normal_(mean=0.0, std=0.1)
        extension.continuation_scale.fill_(1.0)
    block.set_chain_of_experts_extension(extension)
    hidden = torch.randn(1, 6, 4)

    first = block._forward_once(hidden)
    recurrent = hidden + first
    second = block._forward_once(
        recurrent,
        router_logit_delta=extension.router_logit_delta(
            recurrent.reshape(-1, recurrent.shape[-1])
        ),
    )
    expected = first + second
    actual = block(hidden)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_scale_and_second_router_receive_gradients() -> None:
    torch.manual_seed(23)
    block = _block().train()
    extension = _extension()
    with torch.no_grad():
        extension.router_up.normal_(mean=0.0, std=0.1)
        extension.continuation_scale.fill_(0.5)
    block.set_chain_of_experts_extension(extension)
    hidden = torch.randn(2, 4, 4, requires_grad=True)
    block(hidden).square().mean().backward()
    assert hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())
    for parameter in (
        extension.continuation_scale,
        extension.router_down,
        extension.router_up,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(torch.any(parameter.grad != 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_lifecycle_and_state_round_trip() -> None:
    """Verify grouped-expert BF16 gradients and persistence on CUDA."""

    with acquire_gpu_lease(
        lock_path=resolve_lease_lock_path(Path.cwd()),
        timeout_seconds=0.0,
    ):
        _run_cuda_bfloat16_lifecycle_and_state_round_trip()


def _run_cuda_bfloat16_lifecycle_and_state_round_trip() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(29)
    block = LingBotVideoSparseMoeBlock(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
        moe_intermediate_size=16,
        score_func="softmax",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=0,
    )
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    block = block.to(device=device, dtype=torch.bfloat16).train()
    extension = _extension(hidden_size=16, num_experts=4, rank=4).to(device)
    with torch.no_grad():
        extension.router_up.normal_(mean=0.0, std=0.1)
        extension.continuation_scale.fill_(0.5)
    block.set_chain_of_experts_extension(extension)
    optimizer = torch.optim.AdamW(extension.parameters(), lr=1e-3)
    hidden = torch.randn(
        2,
        8,
        16,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    loss = block(hidden).float().square().mean()
    loss.backward()
    assert hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())
    for parameter in extension.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
    optimizer.step()

    source = torch.nn.Module()
    source.chain = extension
    state = export_chain_of_experts_state(source)
    target = torch.nn.Module()
    target.chain = _extension(hidden_size=16, num_experts=4, rank=4).to(device)
    load_chain_of_experts_state(target, state)
    probe_tokens = torch.randn(7, 16, device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(
        target.chain.router_logit_delta(probe_tokens),
        source.chain.router_logit_delta(probe_tokens),
        rtol=0.0,
        atol=0.0,
    )


def test_route_transition_metrics_are_unordered() -> None:
    root = torch.nn.Module()
    root.chain = _extension()
    root.chain.record_routes(
        torch.tensor([[0, 1], [0, 2]]),
        torch.tensor([[1, 0], [1, 2]]),
    )
    metrics = chain_of_experts_metrics(root)
    assert metrics["moe_chain_route_retention"] == pytest.approx(0.75)
    assert metrics["moe_chain_route_switch_fraction"] == pytest.approx(0.5)


def test_versioned_state_round_trip_and_topology_guard() -> None:
    source = torch.nn.Module()
    source.chain = _extension()
    with torch.no_grad():
        source.chain.router_up.add_(0.25)
        source.chain.continuation_scale.fill_(0.75)
    state = export_chain_of_experts_state(source)

    target = torch.nn.Module()
    target.chain = _extension()
    load_chain_of_experts_state(target, state)
    for key, value in source.chain.state_dict().items():
        torch.testing.assert_close(target.chain.state_dict()[key], value)

    tampered = copy.deepcopy(state)
    tampered[f"{CHAIN_OF_EXPERTS_STATE_PREFIX}topology"]["chain"][
        "router_rank"
    ] = 1
    with pytest.raises(ValueError, match="topology"):
        load_chain_of_experts_state(target, tampered)


def test_provider_lifecycle_trains_and_round_trips_chain_state() -> None:
    config = _config()
    torch.manual_seed(31)
    source = LingBotVideoPipeline.from_training_config(config)
    assert source.validate_config(config) == []
    source.set_adapter_config(config.adapter)
    source_chain = next(
        module
        for module in source.transformer.modules()
        if isinstance(module, ChainOfExpertsExtension)
    )
    assert all(parameter.requires_grad for parameter in source_chain.parameters())
    with torch.no_grad():
        source_chain.router_up.add_(0.125)
        source_chain.continuation_scale.fill_(0.5)
    state = source.state_dict()

    torch.manual_seed(31)
    target = LingBotVideoPipeline.from_training_config(config)
    target.set_adapter_config(config.adapter)
    target.load_adapter_state(state)
    target_chain = next(
        module
        for module in target.transformer.modules()
        if isinstance(module, ChainOfExpertsExtension)
    )
    for key, value in source_chain.state_dict().items():
        torch.testing.assert_close(target_chain.state_dict()[key], value)


def test_default_pipeline_constructs_no_chain_modules() -> None:
    config = TrainingConfig.from_dict(
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
                }
            }
        }
    )
    pipeline = LingBotVideoPipeline.from_training_config(config)
    assert not any(
        isinstance(module, ChainOfExpertsExtension)
        for module in pipeline.transformer.modules()
    )


def test_config_rejects_incoherent_or_unsupported_combinations() -> None:
    with pytest.raises(ConfigError, match="must be 0"):
        TrainingConfig.from_dict(
            {"model": {"params": {"moe_chain_router_rank": 2}}}
        )
    with pytest.raises(ConfigError, match="token_choice"):
        _config(moe_routing_mode="expert_choice")
    with pytest.raises(ConfigError, match="lightweight"):
        _config(moe_zero_experts=1, moe_lightweight_top_k=1)
    with pytest.raises(ConfigError, match="spatiotemporal"):
        _config(moe_spatiotemporal_routing_weight=0.1)


def test_portable_safetensors_preserves_chain_state(tmp_path) -> None:
    root = torch.nn.Module()
    root.chain = _extension()
    with torch.no_grad():
        root.chain.router_up.add_(0.375)
    state = {
        "layer.lora_a": torch.randn(2, 3),
        "layer.lora_b": torch.randn(4, 2),
        "layer.lora_alpha": torch.tensor(2.0),
        **export_chain_of_experts_state(root),
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
    target.chain = _extension()
    load_chain_of_experts_state(target, restored)
    torch.testing.assert_close(target.chain.router_up, root.chain.router_up)
