"""Training startup preprocessing for cache generation."""

from __future__ import annotations

from pathlib import Path

from mirai.core.dataset.cache import build_cache_from_config
from mirai.core.dataset.media.media_preprocessor import preprocess_video_media_to_pt
from mirai.core.models.providers import NativeCacheEncoderConfig, build_native_cache_encoder
from mirai.core.models.providers import get_model_family_provider


def _resolve_native_latent_channels(config: object, *, provider: object | None = None) -> int:
    if provider is None:
        provider = get_model_family_provider(str(getattr(config.model, "type", "")))
    if provider is None:
        return int(getattr(config.model.params, "latent_channels", 1))
    if not bool(provider.is_native_model()):
        return int(getattr(config.model.params, "latent_channels", 1))
    encoder = build_native_cache_encoder(
        NativeCacheEncoderConfig(
            # Build the provider encoder in a feature-complete mode so any
            # family-specific latent layout requirements can be inferred.
            enabled=True,
            model_type=str(config.model.type),
            variant=str(config.model.params.variant),
            model_path=str(config.model.path),
            dtype_name=str(getattr(config.model, "dtype", "bf16")),
            max_frames=33,
            denoiser_subfolder=str(
                getattr(config.model.params, "denoiser_subfolder", "transformer")
            ),
            strict_assets=False,
            text_encoder_path=str(getattr(config.model.params, "text_encoder_path", "")),
            enable_bucketing=False,
            resolution_buckets=None,
            frame_buckets=None,
            bucket_resize_mode=str(getattr(config.dataset, "bucket_resize_mode", "resize_crop")),
        )
    )
    if encoder is None:
        return int(getattr(config.model.params, "latent_channels", 1))
    if hasattr(encoder, "latent_channels"):
        return int(getattr(encoder, "latent_channels"))
    return int(getattr(config.model.params, "latent_channels", 1))


def _resolve_max_frames(config: object) -> int:
    frame_buckets = list(getattr(config.dataset, "frame_buckets", [33])) or [33]
    if bool(getattr(config.dataset, "enable_bucketing", False)):
        return max(int(v) for v in frame_buckets)
    return 33


def ensure_cache_artifacts_for_training(config: object) -> None:
    """Build cache lazily when enabled and missing.

    This is intentionally thin and orchestration-level so strict runtime behavior
    still applies unless `dataset.auto_preprocess_cache` is true.
    """

    cache_path = Path(config.dataset.cache_path)
    if cache_path.exists():
        return
    if not bool(getattr(config.dataset, "auto_preprocess_cache", False)):
        return

    dataset_path = Path(config.dataset.path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if bool(getattr(config.dataset, "preprocess_raw_media_to_pt", False)):
        provider = get_model_family_provider(str(config.model.type))
        supports_native_cache = getattr(provider, "supports_native_cache_encoding", None)
        allows_asset_free_cache = getattr(provider, "allows_asset_free_cache_encoding", None)
        has_cache_capability_contract = bool(
            callable(supports_native_cache) and callable(allows_asset_free_cache)
        )
        uses_native_media_encoder = bool(
            provider is not None
            and provider.is_native_model()
            and has_cache_capability_contract
            and supports_native_cache(config)
            and not allows_asset_free_cache(config)
        )
        if not uses_native_media_encoder:
            preprocess_video_media_to_pt(
                dataset_path,
                latent_channels=(
                    _resolve_native_latent_channels(config)
                    if has_cache_capability_contract
                    else _resolve_native_latent_channels(config, provider=provider)
                ),
                max_frames=_resolve_max_frames(config),
            )

    build_cache_from_config(config)
