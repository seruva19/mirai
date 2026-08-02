"""Training-objective abstraction.

The objective is the part of training that defines *what is learned*, decoupled
from the architecture (the pipeline) and the conditioning policy (the strategy):

- how clean data is turned into the model input (the forward/corruption process),
- what the regression/prediction target is,
- how per-sample losses are weighted and reduced.

The default ``flow_matching`` objective implements diffusion/flow training.
Plain regression and next-frame prediction register their own objectives without
changes to the trainer core, loop, or loss engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObjectiveLossTerms:
    """Per-sample objective value plus detached objective diagnostics."""

    per_sample_loss: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)


class TrainingObjective(ABC):
    """Pluggable training objective (forward process + target + loss reduction)."""

    #: Whether this objective consumes sampled Gaussian noise. Diffusion/flow
    #: objectives do; a plain regression/AR objective does not. Informational for
    #: strategies/tooling — the default loop always samples (cheap, ignored if
    #: unused).
    samples_noise: bool = True

    #: Whether this objective conditions on a diffusion timestep.
    samples_timesteps: bool = True

    def configure(self, config: Any) -> None:
        """Bind optional objective-owned trainable state from configuration."""

        _ = config

    def get_named_trainable_parameters(self) -> tuple[tuple[str, Any], ...]:
        """Return objective-owned optimizer parameters, if any."""

        return ()

    def state_dict(self) -> dict[str, Any]:
        """Return objective-owned checkpoint state."""

        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore objective-owned checkpoint state."""

        if state:
            raise ValueError(
                f"{type(self).__name__} does not own checkpoint state."
            )

    def to(self, *, device: Any) -> None:
        """Move objective-owned trainable state to its execution device."""

        _ = device

    def train(self, mode: bool = True) -> None:
        """Set objective-owned modules to training or evaluation mode."""

        _ = mode

    def predict(
        self,
        *,
        inputs: Any,
        predict: Any,
        pipeline: Any,
        config: Any,
        training: bool,
    ) -> Any:
        """Run the model calls required by this objective.

        Most objectives use one prepared input. Objectives whose mathematical
        definition owns a rollout may override this hook without teaching the
        generic loss engine about their internal prediction structure.
        """

        _ = pipeline, config, training
        return predict(inputs)

    def resolve_loss_timesteps(
        self,
        *,
        prediction: Any,
        default_timesteps: Any,
    ) -> Any:
        """Return the timestep coordinate used by weighting and reporting."""

        _ = prediction
        return default_timesteps

    @abstractmethod
    def corrupt(
        self,
        *,
        pipeline: Any,
        clean_latents: Any,
        noise: Any,
        timesteps: Any,
    ) -> Any:
        """Produce the model input from clean data (e.g. add flow noise)."""

    @abstractmethod
    def compute_target(
        self,
        *,
        pipeline: Any,
        prediction: Any,
        clean_latents: Any,
        noise: Any,
        timesteps: Any,
    ) -> Any:
        """Return the regression target the prediction is compared against."""

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
        """Evaluate the objective before global weighting and reduction."""

        _ = config
        return ObjectiveLossTerms(
            per_sample_loss=strategy.compute_per_sample_loss(
                prediction=prediction,
                target=target,
                inputs=inputs,
                loss_fn=loss_fn,
            )
        )

    @abstractmethod
    def reduce(
        self,
        *,
        per_sample_loss: Any,
        timesteps: Any,
        gradient_accumulation: int,
        config: Any,
        bucket_ids: list[str] | None,
    ) -> Any:
        """Weight and reduce per-sample losses into a ``FlowLossResult``."""
