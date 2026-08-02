"""Recursive full-trajectory flow matching for SharpMoE routing.

The rollout follows Algorithm 1 of SharpMoE: one clean latent/noise pair is
evaluated at descending uniformly sampled timesteps, and each predicted clean
latent guides routing at the following step.  The trajectory-allocation KL loss
from the paper is intentionally absent: the paper excludes fixed-width
token-choice DiTs from that loss, which is the topology supported here.

Source: https://arxiv.org/abs/2606.26938
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from mirai.core.registry import register_objective
from mirai.core.training.objectives.base import ObjectiveLossTerms
from mirai.core.training.objectives.flow_matching import FlowMatchingObjective

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SHARP_MOE_OBJECTIVE_STATE_VERSION = 1


@dataclass(frozen=True)
class SharpMoETrajectoryPrediction:
    """Predictions and prepared inputs for every recursive rollout point."""

    predictions: tuple[Any, ...]
    inputs: tuple[Any, ...]
    timesteps: Any


def _broadcast_timesteps(timesteps: Any, like: Any) -> Any:
    return timesteps.reshape(
        int(timesteps.shape[0]),
        *((1,) * (int(like.ndim) - 1)),
    ).to(device=like.device, dtype=like.dtype)


@register_objective("sharp_moe_trajectory")
class SharpMoETrajectoryObjective(FlowMatchingObjective):
    """Paper-defined recursive trajectory objective for saliency routing."""

    def __init__(self) -> None:
        super().__init__()
        self._trajectory_steps = 0
        self._seed = 0
        self._generator: Any | None = None

    def configure(self, config: Any) -> None:
        super().configure(config)
        options = getattr(config.training, "policy_options", {}).get(
            "sharp_moe", {}
        )
        self._trajectory_steps = int(options.get("trajectory_steps", 10))
        self._seed = int(options.get("seed", config.training.seed))
        if torch is None:  # pragma: no cover
            return
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self._seed)

    def state_dict(self) -> dict[str, Any]:
        if self._generator is None:
            return {}
        return {
            "schema_version": SHARP_MOE_OBJECTIVE_STATE_VERSION,
            "trajectory_steps": int(self._trajectory_steps),
            "seed": int(self._seed),
            "generator_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._generator is None:
            if state:
                raise RuntimeError("SharpMoE objective requires torch state.")
            return
        if int(state.get("schema_version", -1)) != SHARP_MOE_OBJECTIVE_STATE_VERSION:
            raise ValueError("Unsupported SharpMoE objective state version.")
        expected = (int(self._trajectory_steps), int(self._seed))
        observed = (
            int(state.get("trajectory_steps", -1)),
            int(state.get("seed", -1)),
        )
        if observed != expected:
            raise ValueError(
                "SharpMoE objective topology does not match the checkpoint."
            )
        generator_state = state.get("generator_state")
        if not torch.is_tensor(generator_state):
            raise ValueError("SharpMoE checkpoint is missing generator state.")
        self._generator.set_state(generator_state.detach().cpu())

    def predict(
        self,
        *,
        inputs: Any,
        predict: Any,
        pipeline: Any,
        config: Any,
        training: bool,
    ) -> SharpMoETrajectoryPrediction:
        _ = config, training
        if torch is None or self._generator is None:  # pragma: no cover
            raise RuntimeError("SharpMoE trajectory prediction requires torch.")
        batch = int(inputs.clean_latents.shape[0])
        sampled = torch.rand(
            (self._trajectory_steps, batch),
            generator=self._generator,
            dtype=torch.float32,
            device="cpu",
        )
        sampled = torch.sort(sampled, dim=0, descending=True).values
        sampled[0].fill_(0.999)
        sampled = sampled.to(device=inputs.clean_latents.device)

        predictions: list[Any] = []
        prepared_inputs: list[Any] = []
        previous_clean = None
        for step_timesteps in sampled.unbind(dim=0):
            noisy = pipeline.apply_noise(
                inputs.clean_latents,
                inputs.noise,
                step_timesteps,
            )
            guidance = noisy.detach() if previous_clean is None else previous_clean
            extra = dict(inputs.extra_forward_kwargs)
            extra["routing_guidance_latents"] = guidance
            step_inputs = replace(
                inputs,
                noisy_latents=noisy,
                timestep=step_timesteps,
                objective_timestep=step_timesteps,
                extra_forward_kwargs=extra,
            )
            prediction = predict(step_inputs)
            previous_clean = (
                noisy
                - _broadcast_timesteps(step_timesteps, prediction) * prediction
            ).detach()
            predictions.append(prediction)
            prepared_inputs.append(step_inputs)
        return SharpMoETrajectoryPrediction(
            predictions=tuple(predictions),
            inputs=tuple(prepared_inputs),
            timesteps=sampled,
        )

    def resolve_loss_timesteps(
        self,
        *,
        prediction: Any,
        default_timesteps: Any,
    ) -> Any:
        _ = default_timesteps
        if not isinstance(prediction, SharpMoETrajectoryPrediction):
            raise TypeError("SharpMoE objective received a non-trajectory prediction.")
        return prediction.timesteps.mean(dim=0)

    def compute_target(
        self,
        *,
        pipeline: Any,
        prediction: Any,
        clean_latents: Any,
        noise: Any,
        timesteps: Any,
    ) -> tuple[Any, ...]:
        _ = timesteps
        if not isinstance(prediction, SharpMoETrajectoryPrediction):
            raise TypeError("SharpMoE objective received a non-trajectory prediction.")
        return tuple(
            pipeline.compute_target(
                noise=noise,
                clean_latents=clean_latents,
                timesteps=step_inputs.objective_timestep,
            )
            for step_inputs in prediction.inputs
        )

    def compute_per_sample_loss(
        self,
        *,
        prediction: Any,
        target: Any,
        inputs: Any,
        loss_fn: Any,
        strategy: Any,
        config: Any,
    ) -> ObjectiveLossTerms:
        _ = inputs, config
        if not isinstance(prediction, SharpMoETrajectoryPrediction):
            raise TypeError("SharpMoE objective received a non-trajectory prediction.")
        if len(target) != len(prediction.predictions):
            raise RuntimeError("SharpMoE target and prediction trajectories differ.")
        losses = [
            strategy.compute_per_sample_loss(
                prediction=step_prediction,
                target=step_target,
                inputs=step_inputs,
                loss_fn=loss_fn,
            )
            for step_prediction, step_target, step_inputs in zip(
                prediction.predictions,
                target,
                prediction.inputs,
                strict=True,
            )
        ]
        per_sample = torch.stack(losses, dim=0).mean(dim=0)
        times = prediction.timesteps.detach().float()
        return ObjectiveLossTerms(
            per_sample_loss=per_sample,
            diagnostics={
                "sharp_moe_trajectory_steps": int(times.shape[0]),
                "sharp_moe_timestep_min": times.min(),
                "sharp_moe_timestep_max": times.max(),
                "sharp_moe_timestep_mean": times.mean(),
            },
        )


__all__ = [
    "SHARP_MOE_OBJECTIVE_STATE_VERSION",
    "SharpMoETrajectoryObjective",
    "SharpMoETrajectoryPrediction",
]
