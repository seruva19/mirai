"""Plain regression objective (a non-diffusion example).

The model input is the clean signal without noise corruption, and the target is
the clean signal itself, reduced with uniform timestep-independent weighting.
The objective uses the shared trainer core, loop, and loss engine.
"""

from __future__ import annotations


from mirai.core.registry import register_objective
from mirai.core.training.objectives.flow_loss import compute_flow_matching_loss
from mirai.core.training.objectives.base import TrainingObjective


@register_objective("regression")
class RegressionObjective(TrainingObjective):
    samples_noise = False
    samples_timesteps = False

    def corrupt(self, *, pipeline, clean_latents, noise, timesteps):
        # No forward-process corruption: the model sees the clean signal directly.
        return clean_latents

    def compute_target(self, *, pipeline, prediction, clean_latents, noise, timesteps):
        return clean_latents

    def reduce(self, *, per_sample_loss, timesteps, gradient_accumulation, config, bucket_ids):
        # Uniform reduction — no timestep-dependent weighting.
        return compute_flow_matching_loss(
            per_sample_loss=per_sample_loss,
            timesteps=timesteps,
            gradient_accumulation=gradient_accumulation,
            loss_weighting="uniform",
            min_snr_gamma=config.training.min_snr_gamma,
            timestep_eps=config.training.timestep_eps,
            bucket_ids=bucket_ids,
            loss_bucket_normalization=config.training.loss_bucket_normalization,
        )
