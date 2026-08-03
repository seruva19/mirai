"""Parameter-free Dispersive Loss for diffusion/flow representations.

The objective is the InfoNCE-L2 dispersive term from Wang and He,
"Diffuse and Disperse" (arXiv:2506.09027).  It uses every ordered pair,
including the constant diagonal, exactly as the paper and official MIT-licensed
reference implementation do.  Mirai evaluates the same objective in bounded
feature chunks so video-token activations never require a persistent FP32 copy.

Paper: https://arxiv.org/abs/2506.09027
Reference: https://github.com/raywang4/DispLoss
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from mirai.core.models.providers import get_model_family_provider
from mirai.core.training.training_policy import (
    TrainingPolicy,
    register_training_policy,
)


DEFAULT_DISPERSIVE_CHUNK_FEATURES = 1_048_576
_OPTION_KEYS = frozenset(
    {
        "enabled",
        "weight",
        "temperature",
        "layer_fraction",
        "chunk_features",
    }
)


@dataclass(frozen=True)
class DispersiveLossSpec:
    weight: float = 0.5
    temperature: float = 0.5
    layer_fraction: float = 0.25
    chunk_features: int = DEFAULT_DISPERSIVE_CHUNK_FEATURES

    def validate(self) -> "DispersiveLossSpec":
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("Dispersive Loss weight must be finite and > 0.")
        if (
            not math.isfinite(float(self.temperature))
            or float(self.temperature) <= 0.0
        ):
            raise ValueError("Dispersive Loss temperature must be finite and > 0.")
        if (
            not math.isfinite(float(self.layer_fraction))
            or not 0.0 < float(self.layer_fraction) <= 1.0
        ):
            raise ValueError("Dispersive Loss layer_fraction must be in (0, 1].")
        if int(self.chunk_features) <= 0:
            raise ValueError("Dispersive Loss chunk_features must be > 0.")
        return self

    def resolve_layer_index(self, depth: int) -> int:
        if int(depth) <= 0:
            raise ValueError("Dispersive Loss requires at least one model block.")
        self.validate()
        return min(
            int(depth) - 1,
            max(0, math.ceil(float(self.layer_fraction) * int(depth)) - 1),
        )


class _ChunkedDispersiveL2(torch.autograd.Function):
    """Compute the full-batch objective without retaining an FP32 activation clone."""

    @staticmethod
    def forward(
        ctx: Any,
        representations: torch.Tensor,
        temperature: float,
        chunk_features: int,
    ) -> torch.Tensor:
        if representations.ndim < 2:
            raise ValueError("Dispersive representations must include a batch axis.")
        batch = int(representations.shape[0])
        if batch < 2:
            raise ValueError("Dispersive Loss requires a physical batch of at least 2.")
        flat = representations.reshape(batch, -1)
        features = int(flat.shape[1])
        if features <= 0:
            raise ValueError("Dispersive representations cannot be empty.")
        step = int(chunk_features)
        distances = torch.zeros(
            (batch, batch),
            device=flat.device,
            dtype=torch.float32,
        )
        for start in range(0, features, step):
            chunk = flat[:, start : start + step].float()
            squared_norm = chunk.square().sum(dim=1)
            distances.add_(
                squared_norm[:, None]
                + squared_norm[None, :]
                - 2.0 * (chunk @ chunk.transpose(0, 1))
            )
        distances.div_(float(features)).clamp_min_(0.0)
        distances.fill_diagonal_(0.0)
        logits = distances.mul(-1.0 / float(temperature))
        probabilities = torch.softmax(logits.reshape(-1), dim=0).reshape(
            batch, batch
        )
        ctx.save_for_backward(representations, probabilities)
        ctx.temperature = float(temperature)
        ctx.chunk_features = step
        ctx.features = features
        return torch.logsumexp(logits.reshape(-1), dim=0) - 2.0 * math.log(batch)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        representations, probabilities = ctx.saved_tensors
        batch = int(representations.shape[0])
        flat = representations.reshape(batch, -1)
        symmetric_probability = probabilities + probabilities.transpose(0, 1)
        symmetric_probability = symmetric_probability.clone()
        symmetric_probability.fill_diagonal_(0.0)
        row_mass = symmetric_probability.sum(dim=1)
        scale = -2.0 / (float(ctx.temperature) * float(ctx.features))
        grad_flat = torch.empty_like(flat)
        for start in range(0, int(ctx.features), int(ctx.chunk_features)):
            chunk = flat[:, start : start + int(ctx.chunk_features)].float()
            chunk_grad = scale * (
                row_mass[:, None] * chunk - symmetric_probability @ chunk
            )
            chunk_grad.mul_(grad_output.float())
            grad_flat[:, start : start + int(ctx.chunk_features)] = chunk_grad.to(
                dtype=flat.dtype
            )
        return grad_flat.reshape_as(representations), None, None


def dispersive_l2_loss(
    representations: torch.Tensor,
    *,
    temperature: float,
    chunk_features: int = DEFAULT_DISPERSIVE_CHUNK_FEATURES,
) -> torch.Tensor:
    """Return log-mean-exp of negative normalized squared L2 distances."""

    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("Dispersive Loss temperature must be finite and > 0.")
    if int(chunk_features) <= 0:
        raise ValueError("Dispersive Loss chunk_features must be > 0.")
    return _ChunkedDispersiveL2.apply(
        representations,
        float(temperature),
        int(chunk_features),
    )


def dispersive_l2_loss_reference(
    representations: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Small-tensor autograd reference for output and gradient contracts."""

    if representations.ndim < 2 or int(representations.shape[0]) < 2:
        raise ValueError("Dispersive Loss requires a physical batch of at least 2.")
    flat = representations.reshape(int(representations.shape[0]), -1).float()
    distances = torch.cdist(flat, flat, p=2).square() / float(flat.shape[1])
    logits = distances.mul(-1.0 / float(temperature))
    return torch.logsumexp(logits.reshape(-1), dim=0) - 2.0 * math.log(
        int(flat.shape[0])
    )


class DispersiveLossController:
    """Own the model-independent loss and resolved block location."""

    def __init__(self, spec: DispersiveLossSpec) -> None:
        self.spec = spec.validate()
        self.layer_index: int | None = None
        self.last_unweighted_loss: torch.Tensor | None = None

    def bind_depth(self, depth: int) -> int:
        resolved = self.spec.resolve_layer_index(depth)
        if self.layer_index is not None and self.layer_index != resolved:
            raise ValueError("Dispersive Loss was already bound to another depth.")
        self.layer_index = resolved
        return resolved

    def is_layer_enabled(self, layer_index: int) -> bool:
        if self.layer_index is None:
            raise RuntimeError("Dispersive Loss has not been bound to a model depth.")
        return int(layer_index) == int(self.layer_index)

    def loss(self, representations: torch.Tensor) -> torch.Tensor:
        value = dispersive_l2_loss(
            representations,
            temperature=float(self.spec.temperature),
            chunk_features=int(self.spec.chunk_features),
        )
        self.last_unweighted_loss = value.detach()
        return value * float(self.spec.weight)

    def diagnostics(self) -> dict[str, float | int]:
        output: dict[str, float | int] = {}
        if self.layer_index is not None:
            output["dispersive_loss_layer_index"] = int(self.layer_index)
        if self.last_unweighted_loss is not None:
            output["dispersive_loss_raw"] = float(
                self.last_unweighted_loss.float().cpu().item()
            )
        return output


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get(
        "dispersive_loss", {}
    )


def validate_dispersive_loss_config(config: Any) -> list[str]:
    options = _options(config)
    unknown = sorted(set(options) - _OPTION_KEYS)
    errors = [f"unknown option '{name}'" for name in unknown]
    if not bool(options.get("enabled", False)):
        return errors
    try:
        DispersiveLossSpec(
            weight=float(options.get("weight", 0.5)),
            temperature=float(options.get("temperature", 0.5)),
            layer_fraction=float(options.get("layer_fraction", 0.25)),
            chunk_features=int(
                options.get("chunk_features", DEFAULT_DISPERSIVE_CHUNK_FEATURES)
            ),
        ).validate()
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    if int(config.training.batch_size) < 2:
        errors.append("physical training.batch_size must be >= 2")
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_dispersive_loss_policy(config):
        errors.append(
            f"model.type '{config.model.type}' does not expose video-representation capture"
        )
    return errors


class DispersiveLossTrainingPolicy(TrainingPolicy):
    name = "dispersive_loss"
    priority = 40

    def __init__(self, controller: DispersiveLossController) -> None:
        self.controller = controller

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_dispersive_loss(self.controller)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "weight": float(self.controller.spec.weight),
            "temperature": float(self.controller.spec.temperature),
            "layer_fraction": float(self.controller.spec.layer_fraction),
            "chunk_features": int(self.controller.spec.chunk_features),
            "layer_index": self.controller.layer_index,
        }


@register_training_policy(
    "dispersive_loss",
    validate_config=validate_dispersive_loss_config,
)
def build_dispersive_loss_training_policy(
    config: Any,
) -> DispersiveLossTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    spec = DispersiveLossSpec(
        weight=float(options.get("weight", 0.5)),
        temperature=float(options.get("temperature", 0.5)),
        layer_fraction=float(options.get("layer_fraction", 0.25)),
        chunk_features=int(
            options.get("chunk_features", DEFAULT_DISPERSIVE_CHUNK_FEATURES)
        ),
    ).validate()
    return DispersiveLossTrainingPolicy(DispersiveLossController(spec))


__all__ = [
    "DEFAULT_DISPERSIVE_CHUNK_FEATURES",
    "DispersiveLossController",
    "DispersiveLossSpec",
    "DispersiveLossTrainingPolicy",
    "build_dispersive_loss_training_policy",
    "dispersive_l2_loss",
    "dispersive_l2_loss_reference",
    "validate_dispersive_loss_config",
]
