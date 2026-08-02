"""Runtime policy guards for sparse-MoE diffusion training."""

from __future__ import annotations

from typing import Any

from mirai.config.schema import TrainingConfig
from mirai.core.models.providers import (
    get_model_family_provider,
    is_registered_native_model_type,
    is_registered_sparse_moe_model_type,
    load_configured_model_provider_module,
)


def is_supported_native_model_type(model_type: str) -> bool:
    return is_registered_native_model_type(model_type)


def is_supported_sparse_moe_model_type(model_type: str) -> bool:
    return is_registered_sparse_moe_model_type(model_type)


def has_available_sparse_moe_backend(model_type: str) -> bool:
    provider = get_model_family_provider(str(model_type))
    if provider is None or not provider.is_sparse_moe_model():
        return False
    return bool(provider.has_available_runtime_backend(None))


def _provider_for_config(cfg: TrainingConfig):
    load_configured_model_provider_module(cfg)
    return get_model_family_provider(str(cfg.model.type))


def validate_cli_model_contract(
    cfg: TrainingConfig,
    *,
    entrypoint: str,
) -> None:
    provider = _provider_for_config(cfg)
    if provider is not None and provider.is_sparse_moe_model():
        return
    if provider is not None and provider.is_native_model():
        raise ValueError(
            f"[{entrypoint}] model.type='{cfg.model.type}' is not a sparse-MoE "
            "diffusion model: its provider is registered as native but not sparse-MoE. "
            "Select a sparse-MoE native family; see agent/AGENTS.md."
        )
    raise ValueError(
        f"[{entrypoint}] model.type='{cfg.model.type}' is not a registered sparse-MoE "
        "native path: no ModelFamilyProvider is registered for it (unknown or removed "
        "family). Register a native provider + pipeline per "
        "agent/AGENTS.md, or set model.type to a registered sparse-MoE "
        "family."
    )


def apply_runtime_policy(
    cfg: TrainingConfig,
    *,
    entrypoint: str,
) -> list[str]:
    """Mutate config in-place to enforce provider-declared native policy."""
    provider = _provider_for_config(cfg)
    if provider is None:
        return []
    return list(provider.apply_runtime_policy(cfg, entrypoint=entrypoint))


def validate_runtime_compatibility(
    cfg: TrainingConfig,
    *,
    entrypoint: str,
) -> None:
    provider = _provider_for_config(cfg)
    if provider is None:
        raise ValueError(
            f"[{entrypoint}] model.type='{cfg.model.type}' is not a registered "
            "sparse-MoE native path."
        )
    errors = list(provider.validate_runtime_compatibility(cfg))
    if errors:
        joined = "\n".join(f"- {message}" for message in errors)
        raise ValueError(
            f"[{entrypoint}] Runtime compatibility validation failed:\n{joined}"
        )


def validate_native_backend_availability(
    cfg: TrainingConfig,
    *,
    entrypoint: str,
) -> None:
    provider = _provider_for_config(cfg)
    if provider is None:
        raise ValueError(
            f"[{entrypoint}] model.type='{cfg.model.type}' is not a registered "
            "sparse-MoE native path."
        )
    errors = list(provider.validate_native_backend_availability(cfg))
    if errors:
        joined = "\n".join(f"- {message}" for message in errors)
        raise ValueError(
            f"[{entrypoint}] Native backend availability validation failed:\n{joined}"
        )


def runtime_policy_summary(cfg: TrainingConfig) -> dict[str, Any]:
    provider = _provider_for_config(cfg)
    return {
        "model_type": str(cfg.model.type).strip().lower(),
        "provider_module": str(getattr(cfg.model, "provider_module", "") or ""),
        "registered_provider": bool(provider is not None),
        "native_model": bool(provider is not None and provider.is_native_model()),
        "sparse_moe_model": bool(
            provider is not None and provider.is_sparse_moe_model()
        ),
        "runtime_backend_available": bool(
            provider is not None and provider.has_available_runtime_backend(cfg)
        ),
        "strict_native_assets": bool(cfg.model.params.strict_native_assets),
        "model_variant": str(getattr(cfg.model.params, "variant", "") or ""),
        "denoiser_subfolder": str(
            getattr(cfg.model.params, "denoiser_subfolder", "transformer") or "transformer"
        ),
    }
