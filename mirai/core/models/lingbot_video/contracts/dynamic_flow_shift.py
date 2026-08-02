"""Behavioral contracts for token-count-aware rectified-flow shifting."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mirai.config.schema import ConfigError, TrainingConfig  # noqa: E402
from mirai.core.models.flow import shifted_sigma  # noqa: E402
from mirai.core.models.flow_shift import DynamicFlowShiftPolicy  # noqa: E402
from mirai.core.models.lingbot_video.pipeline import (  # noqa: E402
    LingBotVideoPipeline,
)
from mirai.core.training.strategies.text_to_video import (  # noqa: E402
    TextToVideoStrategy,
)


class _FixedTimestepSampler:
    def __init__(self, timesteps):
        self.timesteps = timesteps

    def sample(self, batch_size: int, like=None):
        _ = like
        assert int(batch_size) == int(self.timesteps.shape[0])
        return self.timesteps


class _FixedNoiseGenerator:
    def sample(self, like):
        return torch.ones_like(like)


def _dynamic_config() -> TrainingConfig:
    return TrainingConfig.from_dict(
        {
            "model": {
                "params": {
                    "variant": "tiny-video",
                    "flow_shift": 1.0,
                    "flow_shift_mode": "dynamic",
                    "flow_shift_base_seq_len": 8,
                    "flow_shift_max_seq_len": 32,
                    "flow_shift_max": 2.0,
                    "hidden_size": 12,
                    "attention_heads": 2,
                    "num_layers": 1,
                    "num_experts": 2,
                    "experts_per_token": 1,
                    "shared_experts": 0,
                    "latent_channels": 1,
                    "patch_size": 1,
                }
            }
        }
    )


def test_policy_implements_square_root_rule_and_upper_clamp() -> None:
    policy = DynamicFlowShiftPolicy(
        mode="dynamic",
        base_shift=1.0,
        base_seq_len=8,
        max_seq_len=32,
        max_shift=2.0,
    )
    shifts = policy.shifts_for_token_counts(
        torch.tensor([2, 8, 18, 32, 128])
    )
    torch.testing.assert_close(
        shifts,
        torch.tensor([0.5, 1.0, 1.5, 2.0, 2.0]),
    )


def test_constant_provider_path_returns_original_timestep_object() -> None:
    config = TrainingConfig.from_dict(
        {
            "model": {
                "params": {
                    "variant": "tiny-video",
                    "hidden_size": 12,
                    "attention_heads": 2,
                    "num_layers": 1,
                    "latent_channels": 1,
                }
            }
        }
    )
    pipeline = LingBotVideoPipeline.from_training_config(config)
    latents = torch.zeros(1, 1, 2, 2, 2)
    timesteps = torch.tensor([0.25])
    assert pipeline.prepare_model_timesteps(timesteps, latents=latents) is timesteps


def test_provider_uses_one_shift_for_corruption_conditioning_and_inference() -> None:
    config = _dynamic_config()
    pipeline = LingBotVideoPipeline.from_training_config(config)
    latents = torch.zeros(1, 1, 2, 2, 2)
    noise = torch.ones_like(latents)
    raw_timesteps = torch.tensor([0.25])
    expected_shift = 1.0
    expected_sigma = shifted_sigma(raw_timesteps, expected_shift)

    noisy = pipeline.apply_noise(latents, noise, raw_timesteps)
    model_timesteps = pipeline.prepare_model_timesteps(
        raw_timesteps,
        latents=latents,
    )

    torch.testing.assert_close(
        noisy,
        expected_sigma.reshape(1, 1, 1, 1, 1).expand_as(noisy),
    )
    torch.testing.assert_close(model_timesteps, expected_sigma)
    assert pipeline.resolve_flow_shift_for_latent_shape(
        tuple(latents.shape)
    ) == pytest.approx(expected_shift)


def test_strategy_keeps_raw_objective_timestep_separate_from_model_sigma() -> None:
    config = _dynamic_config()
    pipeline = LingBotVideoPipeline.from_training_config(config)
    strategy = TextToVideoStrategy(config.strategy)
    latents = torch.zeros(1, 1, 3, 2, 2)
    raw_timesteps = torch.tensor([0.25])
    inputs = strategy.prepare_inputs(
        {"latents": latents, "text_embeds": torch.zeros(1, 12)},
        pipeline,
        _FixedTimestepSampler(raw_timesteps),
        _FixedNoiseGenerator(),
    )
    expected_shift = (12.0 / 8.0) ** 0.5
    expected_sigma = shifted_sigma(raw_timesteps, expected_shift)

    assert inputs.objective_timestep is raw_timesteps
    torch.testing.assert_close(inputs.timestep, expected_sigma)
    torch.testing.assert_close(
        inputs.noisy_latents,
        expected_sigma.reshape(1, 1, 1, 1, 1).expand_as(latents),
    )


def test_dynamic_provider_forward_and_adapter_gradients_are_finite() -> None:
    config = _dynamic_config()
    pipeline = LingBotVideoPipeline.from_training_config(config)
    pipeline.set_adapter_config(config.adapter)
    latents = torch.randn(1, 1, 2, 2, 2)
    noise = torch.randn_like(latents)
    raw_timesteps = torch.tensor([0.4])
    noisy = pipeline.apply_noise(latents, noise, raw_timesteps)
    model_timesteps = pipeline.prepare_model_timesteps(
        raw_timesteps,
        latents=latents,
    )
    output = pipeline(
        noisy,
        model_timesteps,
        {"t5": torch.randn(1, 1, 12)},
    )
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    trainable = pipeline.get_trainable_parameters()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trainable
    )


def test_dynamic_config_rejects_non_square_root_anchor() -> None:
    payload = {
        "model": {
            "params": {
                "flow_shift": 1.0,
                "flow_shift_mode": "dynamic",
                "flow_shift_base_seq_len": 8,
                "flow_shift_max_seq_len": 32,
                "flow_shift_max": 3.0,
            }
        }
    }
    with pytest.raises(ConfigError, match="must equal flow_shift"):
        TrainingConfig.from_dict(payload)
