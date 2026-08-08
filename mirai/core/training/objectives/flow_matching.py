"""Flow-matching / diffusion training objective (the default).

Delegates the forward process and target to the pipeline's flow definition and
applies the configured timestep-dependent loss weighting.
"""

from __future__ import annotations

from mirai.core.registry import register_objective
from mirai.core.training.objectives.base import (
    ObjectiveLossTerms,
    TrainingObjective,
)
from mirai.core.training.objectives.adaptive_weighting import (
    NoiseLevelUncertaintyWeighting,
)
from mirai.core.training.objectives.contrastive_flow import (
    compute_contrastive_flow_loss,
)
from mirai.core.training.objectives.flow_loss import compute_flow_matching_loss
from mirai.core.training.objectives.latent_wavelet import (
    compute_latent_wavelet_loss,
    reconstruct_clean_rectified_flow,
)


@register_objective("flow_matching")
class FlowMatchingObjective(TrainingObjective):
    samples_noise = True
    samples_timesteps = True

    def __init__(self) -> None:
        self._adaptive_weighting = None

    def configure(self, config) -> None:
        mode = str(config.training.loss_weighting).strip().lower()
        self._adaptive_weighting = (
            NoiseLevelUncertaintyWeighting(channels=128)
            if mode == "adaptive_uncertainty"
            else None
        )

    def get_named_trainable_parameters(self):
        if self._adaptive_weighting is None:
            return ()
        return tuple(
            (f"adaptive_weighting.{name}", parameter)
            for name, parameter in self._adaptive_weighting.named_parameters()
            if parameter.requires_grad
        )

    def state_dict(self):
        if self._adaptive_weighting is None:
            return {}
        return {
            "adaptive_weighting": self._adaptive_weighting.state_dict(),
        }

    def load_state_dict(self, state):
        if self._adaptive_weighting is None:
            if state:
                raise ValueError(
                    "Checkpoint contains adaptive uncertainty state but "
                    "training.loss_weighting is not 'adaptive_uncertainty'."
                )
            return
        adaptive_state = state.get("adaptive_weighting")
        if not isinstance(adaptive_state, dict):
            raise ValueError(
                "Checkpoint is missing adaptive uncertainty weighting state."
            )
        self._adaptive_weighting.load_state_dict(
            adaptive_state,
            strict=True,
        )

    def to(self, *, device) -> None:
        if self._adaptive_weighting is not None:
            self._adaptive_weighting.to(device=device)

    def train(self, mode: bool = True) -> None:
        if self._adaptive_weighting is not None:
            self._adaptive_weighting.train(mode)

    def corrupt(self, *, pipeline, clean_latents, noise, timesteps):
        return pipeline.apply_noise(clean_latents, noise, timesteps)

    def compute_target(self, *, pipeline, prediction, clean_latents, noise, timesteps):
        return pipeline.compute_target(
            noise=noise,
            clean_latents=clean_latents,
            timesteps=timesteps,
        )

    def compute_per_sample_loss(
        self,
        *,
        prediction,
        target,
        inputs,
        loss_fn,
        strategy,
        config,
    ):
        weight = float(config.training.contrastive_flow_weight)
        result = compute_contrastive_flow_loss(
            prediction=prediction,
            target=target,
            loss_evaluator=lambda pred, expected: strategy.compute_per_sample_loss(
                prediction=pred,
                target=expected,
                inputs=inputs,
                loss_fn=loss_fn,
            ),
            weight=weight,
        )
        per_sample_loss = result.per_sample_loss
        diagnostics = {}
        if result.negative_distance is not None:
            diagnostics = {
                "contrastive_flow_positive_loss": result.positive_loss.detach().mean(),
                "contrastive_flow_negative_distance": (
                    result.negative_distance.detach().mean()
                ),
                "contrastive_flow_weight": weight,
            }
        wavelet_weight = float(
            getattr(config.training, "latent_wavelet_loss_weight", 0.0)
        )
        if wavelet_weight > 0.0:
            predicted_clean = reconstruct_clean_rectified_flow(
                prediction=prediction,
                noisy_latents=inputs.noisy_latents,
                sigmas=inputs.timestep,
            )
            wavelet = compute_latent_wavelet_loss(
                predicted_clean=predicted_clean,
                clean_latents=inputs.clean_latents,
            )
            per_sample_loss = (
                per_sample_loss + wavelet_weight * wavelet.per_sample_loss
            )
            diagnostics.update(
                {
                    "latent_wavelet_loss": wavelet.per_sample_loss.detach().mean(),
                    "latent_wavelet_low_frequency_loss": (
                        wavelet.low_frequency_loss.detach().mean()
                    ),
                    "latent_wavelet_high_frequency_loss": (
                        wavelet.high_frequency_loss.detach().mean()
                    ),
                    "latent_wavelet_loss_weight": wavelet_weight,
                }
            )
        return ObjectiveLossTerms(
            per_sample_loss=per_sample_loss,
            diagnostics=diagnostics,
        )

    def reduce(self, *, per_sample_loss, timesteps, gradient_accumulation, config, bucket_ids):
        return compute_flow_matching_loss(
            per_sample_loss=per_sample_loss,
            timesteps=timesteps,
            gradient_accumulation=gradient_accumulation,
            loss_weighting=config.training.loss_weighting,
            min_snr_gamma=config.training.min_snr_gamma,
            timestep_eps=config.training.timestep_eps,
            adaptive_weighting=self._adaptive_weighting,
            bucket_ids=bucket_ids,
            loss_bucket_normalization=config.training.loss_bucket_normalization,
        )
