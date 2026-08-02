"""Behavioral contracts for timestep-dependent Expert-Choice capacity."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mirai.config.schema import ConfigError, TrainingConfig  # noqa: E402
from mirai.core.models.lingbot_video.pipeline import (  # noqa: E402
    LingBotVideoPipeline,
)
from mirai.core.models.flow import shifted_sigma  # noqa: E402
from mirai.core.moe.routing.routers import route_expert_choice_logits  # noqa: E402
from mirai.core.moe.routing.timestep_capacity import (  # noqa: E402
    TimestepExpertChoiceCapacityPolicy,
)
from mirai.vendors.lingbot_video.transformer_lingbot_video import (  # noqa: E402
    LingBotVideoRuntimeOptions,
    LingBotVideoRouter,
    LingBotVideoSparseMoeBlock,
)


def _policy(
    *,
    sampling: str = "uniform",
    mean: float = 0.0,
    std: float = 1.0,
    mode_scale: float = 1.29,
    flow_shift: float = 1.0,
) -> TimestepExpertChoiceCapacityPolicy:
    return TimestepExpertChoiceCapacityPolicy(
        schedule="linear_reverse",
        capacity_factor_span=0.5,
        timestep_sampling=sampling,
        timestep_sampling_mean=mean,
        timestep_sampling_std=std,
        timestep_sampling_mode_scale=mode_scale,
        flow_shift=flow_shift,
    )


@pytest.mark.parametrize("sampling", ["uniform", "logit_normal", "mode"])
def test_sampler_cdf_recovers_known_quantiles(sampling: str) -> None:
    quantiles = torch.tensor([0.1, 0.3, 0.7, 0.9])
    if sampling == "uniform":
        timesteps = quantiles
        policy = _policy(sampling=sampling)
    elif sampling == "logit_normal":
        mean, std = 0.4, 1.3
        normal = torch.distributions.Normal(0.0, 1.0)
        timesteps = torch.sigmoid(
            mean + std * normal.icdf(quantiles)
        )
        policy = _policy(sampling=sampling, mean=mean, std=std)
    else:
        scale = 1.1
        uniform = 1.0 - quantiles
        cosine = torch.cos((math.pi / 2.0) * uniform)
        timesteps = (
            1.0
            - uniform
            - scale * (cosine.square() - 1.0 + uniform)
        )
        policy = _policy(sampling=sampling, mode_scale=scale)

    torch.testing.assert_close(
        policy.timestep_cdf(timesteps),
        quantiles,
        atol=2e-6,
        rtol=0.0,
    )


def test_sampler_cdf_inverts_rectified_flow_shift() -> None:
    raw_timesteps = torch.tensor([0.1, 0.3, 0.7, 0.9])
    policy = _policy(flow_shift=3.0)
    noise_levels = shifted_sigma(raw_timesteps, 3.0)
    torch.testing.assert_close(
        policy.timestep_cdf(noise_levels),
        raw_timesteps,
        atol=2e-6,
        rtol=0.0,
    )


def test_sampler_cdf_inverts_distinct_per_sample_flow_shifts() -> None:
    raw_timesteps = torch.tensor([0.2, 0.8])
    flow_shifts = torch.tensor([1.0, 3.0])
    noise_levels = shifted_sigma(raw_timesteps, flow_shifts)
    policy = _policy()
    torch.testing.assert_close(
        policy.timestep_cdf(
            noise_levels,
            flow_shifts=flow_shifts,
        ),
        raw_timesteps,
        atol=2e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize("sampling", ["uniform", "logit_normal", "mode"])
def test_capacity_is_reverse_monotone_and_compute_matched(sampling: str) -> None:
    policy = _policy(sampling=sampling, mean=0.35, std=1.2, mode_scale=1.0)
    quantiles = (torch.arange(1000, dtype=torch.float32) + 0.5) / 1000.0
    if sampling == "uniform":
        timesteps = quantiles
    elif sampling == "logit_normal":
        normal = torch.distributions.Normal(0.0, 1.0)
        timesteps = torch.sigmoid(
            0.35 + 1.2 * normal.icdf(quantiles)
        )
    else:
        uniform = 1.0 - quantiles
        cosine = torch.cos((math.pi / 2.0) * uniform)
        timesteps = (
            1.0
            - uniform
            - (cosine.square() - 1.0 + uniform)
        )

    capacities = policy.capacities(
        timesteps,
        tokens_per_sample=12,
        num_experts=3,
        fallback_capacity_factor=1.0,
    )
    assert capacities[0] > capacities[-1]
    assert float(capacities.float().mean()) == pytest.approx(4.0, abs=1e-7)


def test_per_sample_capacity_masks_padding_from_dispatch_and_coverage() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [0.0, 4.0]],
            [[4.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 4.0]],
        ]
    )
    decision = route_expert_choice_logits(
        logits,
        capacity_factor=1.0,
        capacity_per_sample=torch.tensor([1, 3]),
        route_scale=1.0,
        layer_name="capacity.contract",
        output_dtype=torch.float32,
    )
    assert tuple(decision.expert_token_indices.shape) == (2, 2, 3)
    assert decision.coverage.per_sample_capacity == (1, 3)
    assert decision.stats.selected_tokens == 8
    assert int((decision.expert_token_weights[0] != 0).sum()) == 2
    assert int((decision.expert_token_weights[1] != 0).sum()) == 6

    token_experts, token_weights = (
        LingBotVideoSparseMoeBlock._expert_choice_token_routes(
            decision,
            tokens_per_sample=4,
            num_experts=2,
        )
    )
    assert int((token_weights != 0).sum()) == 8
    assert tuple(token_experts.shape)[0] == 8


def test_disabled_policy_preserves_static_capacity() -> None:
    policy = TimestepExpertChoiceCapacityPolicy()
    capacities = policy.capacities(
        torch.tensor([0.1, 0.9]),
        tokens_per_sample=12,
        num_experts=3,
        fallback_capacity_factor=1.0,
    )
    assert capacities.tolist() == [4, 4]


def test_provider_binds_current_timesteps_and_configured_sampler() -> None:
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
                    "moe_routing_mode": "expert_choice",
                    "moe_expert_choice_timestep_capacity_schedule": (
                        "linear_reverse"
                    ),
                    "moe_expert_choice_timestep_capacity_span": 0.5,
                }
            },
            "training": {
                "timestep_sampling": "logit_normal",
                "timestep_sampling_mean": 0.35,
                "timestep_sampling_std": 1.2,
            },
            "adapter": {
                "type": "lora",
                "target_preset": "attn_only",
                "rank": 2,
                "alpha": 2,
            },
        }
    )
    pipeline = LingBotVideoPipeline.from_training_config(config)
    pipeline.set_adapter_config(config.adapter)
    pipeline._expert_choice_runtime_sigmas = shifted_sigma(
        torch.tensor([0.05, 0.95]),
        float(config.model.params.flow_shift),
    )
    router = next(
        module
        for module in pipeline.transformer.modules()
        if isinstance(module, LingBotVideoRouter)
    )
    decision = router.forward_expert_choice(
        torch.randn(2, 8, 12),
        None,
        None,
    )
    assert decision.coverage.per_sample_capacity[0] > 4
    assert decision.coverage.per_sample_capacity[1] < 4
    assert (
        sum(decision.coverage.per_sample_capacity)
        == 2 * 4
    )


def test_config_requires_explicit_expert_choice_enablement() -> None:
    payload = {
        "model": {
            "params": {
                "moe_expert_choice_timestep_capacity_schedule": "linear_reverse",
                "moe_expert_choice_timestep_capacity_span": 0.5,
            }
        }
    }
    with pytest.raises(ConfigError, match="requires moe_routing_mode"):
        TrainingConfig.from_dict(payload)

    payload["model"]["params"]["moe_routing_mode"] = "expert_choice"
    config = TrainingConfig.from_dict(payload)
    assert (
        config.model.params.moe_expert_choice_timestep_capacity_schedule
        == "linear_reverse"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_per_sample_capacity_dispatch_preserves_gradients() -> None:
    device = torch.device("cuda")
    policy = _policy()
    timesteps = torch.tensor([0.1, 0.9], device=device)
    block = LingBotVideoSparseMoeBlock(
        hidden_size=4,
        intermediate_size=8,
        num_experts=2,
        top_k=1,
        moe_intermediate_size=8,
        score_func="softmax",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=0,
    ).to(device)
    block._mirai_runtime_options = LingBotVideoRuntimeOptions(
        moe_expert_backend="loop"
    )
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(mean=0.0, std=0.02)

    def route(logits, output_dtype):
        capacities = policy.capacities(
            timesteps,
            tokens_per_sample=int(logits.shape[1]),
            num_experts=int(logits.shape[2]),
            fallback_capacity_factor=1.0,
        )
        return route_expert_choice_logits(
            logits,
            capacity_factor=1.0,
            capacity_per_sample=capacities,
            route_scale=1.0,
            layer_name="capacity.cuda",
            output_dtype=output_dtype,
            z_loss_weight=1e-4,
        )

    block.router.set_expert_choice_extension(route)
    hidden = torch.randn(2, 8, 4, device=device, requires_grad=True)
    output = block(hidden)
    decision = block.router.training_expert_choice_decision
    (output.square().mean() + decision.z_loss).backward()

    assert torch.isfinite(output).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert block.router.weight.grad is not None
    assert torch.isfinite(block.router.weight.grad).all()
