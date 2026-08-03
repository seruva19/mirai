"""Behavioral contracts for saliency-harnessing SharpMoE routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams, TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.models.flow import clamp_timesteps, shifted_sigma
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.routing.saliency import (
    SaliencyHarnessingRouter,
    SharpMoESpec,
    export_sharp_moe_state,
    load_sharp_moe_state,
)
from mirai.core.training.objectives.sharp_moe import SharpMoETrajectoryObjective
from mirai.core.training.objectives.engine import compute_training_loss
from mirai.core.training.preview.preview import run_native_denoise_loop
from mirai.core.training.policies.sharp_moe import validate_sharp_moe_config
from mirai.core.training.strategies.base import TrainingInputs
from mirai.core.training.runtime.gpu_lease import (
    acquire_gpu_lease,
    resolve_lease_lock_path,
)
from mirai.core.training.training_policy import TrainingPolicySet


def _router(seed: int = 7) -> SaliencyHarnessingRouter:
    return SaliencyHarnessingRouter(
        hidden_size=4,
        num_experts=3,
        bottleneck_size=2,
        initialization_seed=seed,
    )


def _enabled_config() -> TrainingConfig:
    config = TrainingConfig()
    config.model.type = "lingbot-video"
    config.model.params.moe_balance_mode = "off"
    config.model.params.moe_aux_loss_weight = 0.0
    config.model.params.moe_router_z_loss_weight = 0.0
    config.training.objective = "sharp_moe_trajectory"
    config.training.loss_weighting = "uniform"
    config.training.policy_options = {
        "sharp_moe": {
            "enabled": True,
            "trajectory_steps": 3,
            "router_hidden_dim": 8,
            "seed": 19,
        }
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
                moe_aux_loss_weight=0.0,
                moe_router_z_loss_weight=0.0,
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
    pipeline.configure_sharp_moe(
        SharpMoESpec(trajectory_steps=3, router_hidden_dim=8, seed=19),
    )
    pipeline.set_gradient_checkpointing(checkpointing)
    pipeline.train()
    return pipeline


def test_zero_output_initialization_preserves_rng_and_is_trainable() -> None:
    torch.manual_seed(31)
    before = torch.random.get_rng_state()
    router = _router()
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)

    tokens = torch.randn(5, 4, generator=torch.Generator().manual_seed(11))
    output = router(tokens)
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert torch.count_nonzero(router.output_weight.grad) > 0
    assert torch.count_nonzero(router.input_weight.grad) == 0

    router.zero_grad(set_to_none=True)
    with torch.no_grad():
        router.output_weight.fill_(0.25)
    router(tokens).square().mean().backward()
    assert torch.count_nonzero(router.input_weight.grad) > 0


def test_visual_scope_masks_text_guidance_exactly() -> None:
    router = _router()
    with torch.no_grad():
        router.output_weight.fill_(0.5)
    tokens = torch.randn(4, 4, generator=torch.Generator().manual_seed(13))
    output = router(
        tokens,
        route_scope_mask=torch.tensor([True, False, True, False]),
    )
    assert torch.count_nonzero(output[[1, 3]]) == 0
    assert torch.count_nonzero(output[[0, 2]]) > 0


def test_versioned_state_round_trip_and_topology_rejection() -> None:
    source = nn.Module()
    source.router = _router(seed=5)
    with torch.no_grad():
        source.router.output_weight.add_(0.2)
    state = export_sharp_moe_state(source)
    target = nn.Module()
    target.router = _router(seed=5)
    load_sharp_moe_state(target, state)
    torch.testing.assert_close(
        target.router.output_weight, source.router.output_weight
    )

    mismatch = nn.Module()
    mismatch.router = _router(seed=6)
    with pytest.raises(ValueError, match="topology"):
        load_sharp_moe_state(mismatch, state)
    with pytest.raises(ValueError, match="requires adapter state"):
        load_sharp_moe_state(target, {})


@dataclass
class _FlowPipeline:
    guidance: list[torch.Tensor]

    def apply_noise(self, clean, noise, timesteps):
        t = timesteps.reshape(-1, 1)
        return (1.0 - t) * clean + t * noise

    def prepare_model_timesteps(self, timesteps, *, latents):
        _ = latents
        return timesteps

    def compute_target(self, *, noise, clean_latents, timesteps):
        _ = timesteps
        return noise - clean_latents

    def get_training_auxiliary_losses(self):
        return {}

    def get_training_diagnostics(self):
        return {}

    def take_balance_gradient_probe(self):
        return None


class _MSEStrategy:
    def compute_per_sample_loss(self, *, prediction, target, inputs, loss_fn):
        _ = inputs, loss_fn
        return (prediction - target).square().mean(dim=1)


class _PreparedStrategy(_MSEStrategy):
    def __init__(self, inputs: TrainingInputs) -> None:
        self.inputs = inputs

    def prepare_inputs(self, **_kwargs):
        return self.inputs


class _SharpInferencePipeline:
    def __init__(self) -> None:
        self.model = nn.Linear(1, 1, bias=False)
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def get_training_model(self):
        return self.model

    def preview_latent_geometry(self, *, frame_count: int, height: int, width: int):
        _ = frame_count, height, width
        return (1, 2, 2, 2)

    def load_text_encoder(self, *, device: str) -> None:
        _ = device

    def offload_text_encoder(self) -> None:
        pass

    def encode_prompt(self, prompt: str, *, device: str):
        _ = prompt
        return torch.zeros(1, 1, device=device)

    def resolve_flow_shift_for_latent_shape(self, latent_shape) -> float:
        _ = latent_shape
        return 1.0

    def uses_previous_clean_routing_guidance(self) -> bool:
        return True

    def forward(self, sample, timestep, text_embeds, **kwargs):
        _ = timestep, text_embeds
        guidance = kwargs["routing_guidance_latents"]
        self.calls.append((sample.detach().clone(), guidance.detach().clone()))
        return torch.full_like(sample, 0.25)


def test_recursive_trajectory_uses_previous_detached_clean_prediction() -> None:
    config = _enabled_config()
    objective = SharpMoETrajectoryObjective()
    objective.configure(config)
    pipeline = _FlowPipeline(guidance=[])
    clean = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    noise = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    inputs = TrainingInputs(
        noisy_latents=clean,
        timestep=torch.zeros(2),
        noise=noise,
        clean_latents=clean,
        text_embeds={},
    )

    def predict(step_inputs):
        pipeline.guidance.append(
            step_inputs.extra_forward_kwargs["routing_guidance_latents"].clone()
        )
        return torch.full_like(step_inputs.noisy_latents, 0.25, requires_grad=True)

    trajectory = objective.predict(
        inputs=inputs,
        predict=predict,
        pipeline=pipeline,
        config=config,
        training=True,
    )
    assert tuple(trajectory.timesteps.shape) == (3, 2)
    torch.testing.assert_close(
        trajectory.timesteps[0], torch.full((2,), 0.999)
    )
    assert bool(torch.all(trajectory.timesteps[:-1] >= trajectory.timesteps[1:]))
    torch.testing.assert_close(
        pipeline.guidance[0],
        pipeline.apply_noise(clean, noise, trajectory.timesteps[0]),
    )
    expected_second = (
        trajectory.inputs[0].noisy_latents
        - trajectory.timesteps[0].reshape(-1, 1)
        * trajectory.predictions[0]
    ).detach()
    torch.testing.assert_close(pipeline.guidance[1], expected_second)
    assert not pipeline.guidance[1].requires_grad

    targets = objective.compute_target(
        pipeline=pipeline,
        prediction=trajectory,
        clean_latents=clean,
        noise=noise,
        timesteps=trajectory.timesteps.mean(0),
    )
    terms = objective.compute_per_sample_loss(
        prediction=trajectory,
        target=targets,
        inputs=inputs,
        loss_fn=None,
        strategy=_MSEStrategy(),
        config=config,
    )
    expected = torch.stack(
        [(value - (noise - clean)).square().mean(dim=1) for value in trajectory.predictions]
    ).mean(dim=0)
    torch.testing.assert_close(terms.per_sample_loss, expected)


def test_recursive_trajectory_uses_post_shift_sigma_for_model_and_clean_estimate(
) -> None:
    config = _enabled_config()
    config.training.policy_options["sharp_moe"]["trajectory_steps"] = 2
    objective = SharpMoETrajectoryObjective()
    objective.configure(config)

    class _ShiftedPipeline(_FlowPipeline):
        def prepare_model_timesteps(self, timesteps, *, latents):
            _ = latents
            return shifted_sigma(clamp_timesteps(timesteps, 1e-5), 3.0)

        def apply_noise(self, clean, noise, timesteps):
            sigma = self.prepare_model_timesteps(timesteps, latents=clean).reshape(
                -1, 1
            )
            return (1.0 - sigma) * clean + sigma * noise

    pipeline = _ShiftedPipeline(guidance=[])
    clean = torch.tensor([[1.0, 2.0]])
    noise = torch.tensor([[5.0, 6.0]])
    inputs = TrainingInputs(
        noisy_latents=clean,
        timestep=torch.zeros(1),
        noise=noise,
        clean_latents=clean,
        text_embeds={},
    )

    def predict(step_inputs):
        pipeline.guidance.append(
            step_inputs.extra_forward_kwargs["routing_guidance_latents"].clone()
        )
        return torch.full_like(
            step_inputs.noisy_latents, 0.25, requires_grad=True
        )

    trajectory = objective.predict(
        inputs=inputs,
        predict=predict,
        pipeline=pipeline,
        config=config,
        training=True,
    )
    raw = trajectory.timesteps[0]
    sigma = shifted_sigma(clamp_timesteps(raw, 1e-5), 3.0)
    torch.testing.assert_close(trajectory.inputs[0].timestep, sigma)
    torch.testing.assert_close(
        pipeline.guidance[0],
        (1.0 - sigma.reshape(-1, 1)) * clean + sigma.reshape(-1, 1) * noise,
    )
    torch.testing.assert_close(
        pipeline.guidance[1],
        (
            trajectory.inputs[0].noisy_latents
            - sigma.reshape(-1, 1) * trajectory.predictions[0]
        ).detach(),
    )


def test_objective_rng_resume_is_exact() -> None:
    config = _enabled_config()
    source = SharpMoETrajectoryObjective()
    source.configure(config)
    state = source.state_dict()
    restored = SharpMoETrajectoryObjective()
    restored.configure(config)
    restored.load_state_dict(state)
    pipeline = _FlowPipeline(guidance=[])
    inputs = TrainingInputs(
        noisy_latents=torch.zeros(1, 2),
        timestep=torch.zeros(1),
        noise=torch.ones(1, 2),
        clean_latents=torch.zeros(1, 2),
        text_embeds={},
    )
    predict = lambda step_inputs: torch.zeros_like(step_inputs.noisy_latents)
    left = source.predict(
        inputs=inputs, predict=predict, pipeline=pipeline, config=config, training=True
    )
    right = restored.predict(
        inputs=inputs, predict=predict, pipeline=pipeline, config=config, training=True
    )
    torch.testing.assert_close(left.timesteps, right.timesteps)


def test_generic_loss_engine_executes_the_complete_trajectory() -> None:
    config = _enabled_config()
    objective = SharpMoETrajectoryObjective()
    objective.configure(config)
    inputs = TrainingInputs(
        noisy_latents=torch.zeros(2, 3),
        timestep=torch.zeros(2),
        noise=torch.ones(2, 3),
        clean_latents=torch.zeros(2, 3),
        text_embeds={},
    )
    result = compute_training_loss(
        config=config,
        batch={},
        pipeline=_FlowPipeline(guidance=[]),
        strategy=_PreparedStrategy(inputs),
        objective=objective,
        loss_fn=lambda prediction, target: (prediction - target).square(),
        timestep_sampler=object(),
        noise_generator=object(),
        predict=lambda step_inputs: torch.zeros_like(step_inputs.noisy_latents),
        strategy_prepare_accepts_training=True,
        strategy_prepare_accepts_objective=True,
    )
    assert bool(torch.isfinite(result.loss))
    assert tuple(result.timesteps.shape) == (2,)
    assert int(result.diagnostics["sharp_moe_trajectory_steps"]) == 3


def test_policy_is_default_off_and_rejects_wrong_objective() -> None:
    register_builtin_components()
    assert "sharp_moe" not in TrainingPolicySet.from_config(
        TrainingConfig()
    ).active_names
    assert "sharp_moe" in TrainingPolicySet.from_config(
        _enabled_config()
    ).active_names
    wrong = _enabled_config()
    wrong.training.objective = "flow_matching"
    assert any(
        "sharp_moe_trajectory" in error
        for error in validate_sharp_moe_config(wrong)
    )


@pytest.mark.parametrize("cfg_mode", ["sequential", "batched"])
def test_inference_carries_clean_guidance_through_cfg(cfg_mode: str) -> None:
    pipeline = _SharpInferencePipeline()
    run_native_denoise_loop(
        pipeline=pipeline,
        prompt="condition",
        negative_prompt="negative",
        cfg_scale=2.0,
        seed=23,
        step=0,
        denoise_steps=3,
        scheduler="euler",
        frame_count=5,
        height=16,
        width=16,
        cfg_mode=cfg_mode,
    )
    calls_per_step = 1 if cfg_mode == "batched" else 2
    assert len(pipeline.calls) == 3 * calls_per_step
    first_sample, first_guidance = pipeline.calls[0]
    torch.testing.assert_close(first_guidance, first_sample)
    for offset in range(0, len(pipeline.calls), calls_per_step):
        group = pipeline.calls[offset : offset + calls_per_step]
        for _sample, guidance in group[1:]:
            torch.testing.assert_close(guidance, group[0][1])
        if cfg_mode == "batched":
            half = int(group[0][1].shape[0]) // 2
            torch.testing.assert_close(group[0][1][:half], group[0][1][half:])
    assert not torch.equal(pipeline.calls[0][1], pipeline.calls[-1][1])


@pytest.mark.parametrize("checkpointing", ["off", "standard", "aggressive"])
def test_pipeline_forward_backward_and_adapter_round_trip(checkpointing: str) -> None:
    pipeline = _pipeline(checkpointing=checkpointing)
    noisy = torch.randn(1, 2, 2, 4, 4)
    guidance = torch.randn_like(noisy)
    prediction = pipeline.forward(
        noisy,
        torch.tensor([0.5]),
        {"lingbot": torch.randn(1, 3, 16)},
        routing_guidance_latents=guidance,
    )
    prediction.float().square().mean().backward()
    saliency_parameters = [
        parameter
        for name, parameter in pipeline.named_parameters()
        if "saliency_harnessing" in name
    ]
    assert saliency_parameters
    assert all(parameter.grad is not None for parameter in saliency_parameters)

    state = pipeline.state_dict()
    restored = _pipeline(checkpointing=checkpointing)
    restored.load_state_dict(state)
    source_state = export_sharp_moe_state(pipeline.transformer)
    restored_state = export_sharp_moe_state(restored.transformer)
    assert source_state.keys() == restored_state.keys()
    for key in source_state:
        if torch.is_tensor(source_state[key]):
            torch.testing.assert_close(source_state[key], restored_state[key])
        else:
            assert source_state[key] == restored_state[key]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_update_and_reload() -> None:
    with acquire_gpu_lease(
        lock_path=resolve_lease_lock_path(Path.cwd()), timeout_seconds=0.0
    ):
        pipeline = _pipeline(checkpointing="aggressive").to(
            device="cuda:0", dtype=torch.bfloat16
        )
        parameters = [
            parameter
            for name, parameter in pipeline.named_parameters()
            if "saliency_harnessing" in name
        ]
        optimizer = torch.optim.SGD(parameters, lr=0.1)
        before = [parameter.detach().clone() for parameter in parameters]
        noisy = torch.randn(
            1, 2, 2, 4, 4, device="cuda", dtype=torch.bfloat16
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = pipeline.forward(
                noisy,
                torch.tensor([0.5], device="cuda", dtype=torch.bfloat16),
                {
                    "lingbot": torch.randn(
                        1, 3, 16, device="cuda", dtype=torch.bfloat16
                    )
                },
                routing_guidance_latents=torch.randn_like(noisy),
            )
            loss = prediction.float().square().mean()
        loss.backward()
        optimizer.step()
        assert bool(torch.isfinite(loss))
        assert any(
            not torch.equal(previous, parameter.detach())
            for previous, parameter in zip(before, parameters, strict=True)
        )
        state = pipeline.state_dict()
        restored = _pipeline(checkpointing="aggressive").to(
            device="cuda:0", dtype=torch.bfloat16
        )
        restored.load_state_dict(state)
        assert export_sharp_moe_state(restored.transformer)
