"""Inference-session component-residency option resolution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceComponentResidency:
    text_encoder: bool = False
    vae: bool = False


def resolve_inference_component_residency(
    *,
    config_text_encoder: bool,
    config_vae: bool,
    text_encoder_override: bool | None,
    vae_override: bool | None,
) -> InferenceComponentResidency:
    """Resolve CLI/API overrides without weakening default-off config behavior."""
    return InferenceComponentResidency(
        text_encoder=(
            bool(config_text_encoder)
            if text_encoder_override is None
            else bool(text_encoder_override)
        ),
        vae=bool(config_vae) if vae_override is None else bool(vae_override),
    )
