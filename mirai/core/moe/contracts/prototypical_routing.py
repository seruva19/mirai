"""Behavioral contracts for residual ProMoE routing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams, TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.models.lingbot_video.route_extensions import (
    bind_lingbot_route_extensions,
)
from mirai.core.moe.routing.prototypical import (
    PrototypicalRouterExtension,
    PrototypicalRoutingSpec,
    export_prototypical_routing_state,
    load_prototypical_routing_state,
    routing_contrastive_loss,
)
from mirai.core.training.policies.prototypical_routing import (
    validate_prototypical_routing_config,
)
from mirai.core.training.runtime.gpu_lease import (
    acquire_gpu_lease,
    resolve_lease_lock_path,
)
from mirai.core.training.training_policy import TrainingPolicySet
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoRouter


def _cpu_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    outputs = []
    q_offsets = cu_seqlens_q.detach().cpu().tolist()
    k_offsets = cu_seqlens_k.detach().cpu().tolist()
    for q_start, q_end, k_start, k_end in zip(
        q_offsets[:-1],
        q_offsets[1:],
        k_offsets[:-1],
        k_offsets[1:],
        strict=True,
    ):
        q = query[q_start:q_end].transpose(0, 1)
        k = key[k_start:k_end].transpose(0, 1)
        v = value[k_start:k_end].transpose(0, 1)
        outputs.append(F.scaled_dot_product_attention(q, k, v).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _spec(seed: int = 17) -> PrototypicalRoutingSpec:
    return PrototypicalRoutingSpec(seed=seed)


def _extension(*, seed: int = 17, device: str = "cpu") -> PrototypicalRouterExtension:
    return PrototypicalRouterExtension(
        hidden_size=4,
        num_experts=4,
        spec=_spec(seed),
        initialization_seed=seed,
        device=device,
    )


def _enabled_config(**options) -> TrainingConfig:
    config = TrainingConfig()
    config.model.type = "lingbot-video"
    config.model.params.moe_balance_mode = "off"
    config.model.params.moe_router_z_loss_weight = 0.0
    config.training.policy_options = {
        "prototypical_routing": {"enabled": True, **options}
    }
    return config


def _pipeline(*, checkpointing: str = "off") -> LingBotVideoPipeline:
    pipeline = LingBotVideoPipeline(
        ModelConfig(
            type="lingbot-video",
            path="./models/lingbot_video",
            params=ModelParams(
                variant="tiny-video",
                latent_channels=2,
                num_experts=4,
                experts_per_token=2,
                shared_experts=1,
                hidden_size=16,
                num_layers=2,
                attention_heads=2,
                patch_size=1,
                moe_balance_mode="off",
            ),
        )
    )
    pipeline.set_adapter_config(
        AdapterConfig(
            type="lora",
            target_preset="attn_routed_experts",
            rank=2,
            alpha=2.0,
        )
    )
    pipeline.configure_training_policy("prototypical_routing", _spec())
    pipeline.set_gradient_checkpointing(checkpointing)
    pipeline.train()
    return pipeline


def test_routing_contrastive_loss_matches_equation_six_with_topk_membership() -> None:
    tokens = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
    )
    assignments = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 1]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    actual = routing_contrastive_loss(
        tokens, assignments, prototypes, temperature=0.07
    )
    means = torch.stack(
        [tokens[(assignments == expert).any(dim=-1)].mean(dim=0) for expert in range(3)]
    )
    logits = F.normalize(prototypes, dim=-1) @ F.normalize(means, dim=-1).T
    expected = F.cross_entropy(logits / 0.07, torch.arange(3))
    torch.testing.assert_close(actual, expected)


def test_zero_scale_preserves_group_limited_native_routes_and_global_rng() -> None:
    torch.manual_seed(29)
    before = torch.random.get_rng_state()
    extension = _extension(seed=31)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)

    router = LingBotVideoRouter(4, 4, 2, "sigmoid", True, 2, 1, 1.0)
    with torch.no_grad():
        router.weight.copy_(torch.randn(4, 4, generator=torch.Generator().manual_seed(7)))
    tokens = torch.randn(9, 4, generator=torch.Generator().manual_seed(11))
    reference = router(tokens)[:2]
    bind_lingbot_route_extensions(
        router,
        layer_name="router",
        diversity=None,
        expert_dropout=None,
        dynamic_topk=None,
        router_temperature=None,
        selective_sinkhorn=None,
        prototypical=extension,
    )
    router.eval()
    actual = router(tokens, route_scope_mask=torch.ones(9, dtype=torch.bool))[:2]
    assert torch.equal(actual[0], reference[0])
    torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)


def test_visual_scope_and_gradients_are_isolated() -> None:
    extension = _extension()
    with torch.no_grad():
        extension.prototypes.copy_(torch.eye(4))
        extension.residual_scale.fill_(4.0)
    tokens = torch.eye(4, requires_grad=True)
    native_choice = torch.zeros(4, 4)
    native_gate = torch.full((4, 4), 0.25)
    native_indices = torch.tensor([[1], [0], [0], [0]])
    native_weights = torch.full((4, 1), 0.25)
    routes = extension.select(
        tokens,
        native_choice,
        native_gate,
        native_indices,
        native_weights,
        route_scope_mask=torch.tensor([True, True, False, False]),
        valid_token_mask=None,
        norm_topk_prob=False,
        route_scale=1.0,
        training=True,
    )
    assert torch.equal(routes.top_indices[:2, 0], torch.tensor([0, 1]))
    assert torch.equal(routes.top_indices[2:], native_indices[2:])
    torch.testing.assert_close(routes.top_weights[2:], native_weights[2:])
    assert extension.last_contrastive_loss is not None
    (routes.top_weights.sum() + extension.last_contrastive_loss).backward()
    assert extension.prototypes.grad is not None
    assert extension.residual_scale.grad is not None
    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad[:2]) > 0
    assert torch.count_nonzero(tokens.grad[2:]) == 0


def test_versioned_state_round_trip_and_topology_fail_closed() -> None:
    source = nn.Module()
    source.router = _extension(seed=5)
    with torch.no_grad():
        source.router.prototypes.add_(0.5)
        source.router.residual_scale.fill_(0.25)
    state = export_prototypical_routing_state(source)
    target = nn.Module()
    target.router = _extension(seed=5)
    load_prototypical_routing_state(target, state)
    torch.testing.assert_close(target.router.prototypes, source.router.prototypes)
    torch.testing.assert_close(target.router.residual_scale, source.router.residual_scale)

    mismatched = nn.Module()
    mismatched.router = _extension(seed=6)
    with pytest.raises(ValueError, match="topology"):
        load_prototypical_routing_state(mismatched, state)
    with pytest.raises(ValueError, match="requires adapter state"):
        load_prototypical_routing_state(target, {})


def test_policy_is_default_off_and_rejects_nonpaper_combinations() -> None:
    register_builtin_components()
    assert "prototypical_routing" not in TrainingPolicySet.from_config(
        TrainingConfig()
    ).active_names
    policies = TrainingPolicySet.from_config(_enabled_config(seed=43))
    assert "prototypical_routing" in policies.active_names
    metadata = policies.checkpoint_metadata()["policies"]["prototypical_routing"]
    assert metadata["residual_scale_init"] == 0.0
    assert metadata["scope"] == "provider_selected_visual_tokens"

    unknown = _enabled_config(mystery=1)
    assert any("unknown option 'mystery'" in error for error in validate_prototypical_routing_config(unknown))
    conflict = _enabled_config()
    conflict.model.params.moe_balance_mode = "aux_loss"
    assert any("moe_balance_mode='off'" in error for error in validate_prototypical_routing_config(conflict))


@pytest.mark.parametrize("checkpointing", ["off", "standard", "aggressive"])
def test_pipeline_forward_backward_and_adapter_round_trip(
    checkpointing: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mirai.vendors.lingbot_video.transformer_lingbot_video."
        "dispatch_varlen_attention",
        _cpu_varlen_attention,
    )
    pipeline = _pipeline(checkpointing=checkpointing)
    prediction = pipeline.forward(
        torch.randn(1, 2, 2, 4, 4),
        torch.tensor([0.5]),
        {"lingbot": torch.randn(1, 3, 16)},
    )
    auxiliary = pipeline.get_training_auxiliary_losses()
    assert "moe_routing_contrastive" in auxiliary
    loss = prediction.float().square().mean() + auxiliary["moe_routing_contrastive"]
    loss.backward()
    prototype_parameters = [
        parameter
        for name, parameter in pipeline.named_parameters()
        if "prototypical_routing" in name
    ]
    assert prototype_parameters
    assert all(parameter.grad is not None for parameter in prototype_parameters)
    assert bool(torch.isfinite(loss))

    state = pipeline.state_dict()
    restored = _pipeline(checkpointing=checkpointing)
    restored.load_state_dict(state)
    source_state = export_prototypical_routing_state(pipeline.transformer)
    restored_state = export_prototypical_routing_state(restored.transformer)
    assert source_state.keys() == restored_state.keys()
    for key in source_state:
        if torch.is_tensor(source_state[key]):
            torch.testing.assert_close(source_state[key], restored_state[key])
        else:
            assert source_state[key] == restored_state[key]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_update_and_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mirai.vendors.lingbot_video.transformer_lingbot_video."
        "dispatch_varlen_attention",
        _cpu_varlen_attention,
    )
    with acquire_gpu_lease(
        lock_path=resolve_lease_lock_path(Path.cwd()), timeout_seconds=0.0
    ):
        pipeline = _pipeline(checkpointing="aggressive").to(
            device="cuda:0", dtype=torch.bfloat16
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = pipeline.forward(
                torch.randn(1, 2, 2, 4, 4, device="cuda", dtype=torch.bfloat16),
                torch.tensor([0.5], device="cuda", dtype=torch.bfloat16),
                {"lingbot": torch.randn(1, 3, 16, device="cuda", dtype=torch.bfloat16)},
            )
        auxiliary = pipeline.get_training_auxiliary_losses()
        loss = prediction.float().square().mean() + auxiliary["moe_routing_contrastive"]
        loss.backward()
        assert bool(torch.isfinite(loss))
        state = pipeline.state_dict()
        restored = _pipeline(checkpointing="aggressive").to(
            device="cuda:0", dtype=torch.bfloat16
        )
        restored.load_state_dict(state)
        assert export_prototypical_routing_state(restored.transformer)
