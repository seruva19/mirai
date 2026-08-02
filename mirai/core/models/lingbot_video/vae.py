"""Native Wan VAE ownership for Lingbot cache and preview paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mirai.core.models.checkpoint_streaming import load_module_safetensors
from mirai.core.models.checkpoint_streaming import resolve_safetensor_files
from mirai.vendors.lingbot_video.autoencoder_kl_wan import AutoencoderKLWan

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LINGBOT_VAE_ARTIFACT_FORMAT = "mirai.lingbot_video.vae_latent"
LINGBOT_VAE_ARTIFACT_SCHEMA_VERSION = 1

_HF_VAE_HINT = (
    "Fetch the upstream LingBot-Video repository's vae/ subfolder, e.g. "
    "`hf download <lingbot-video-repo> --include 'vae/*'`, or point model.path at "
    "a snapshot that contains vae/config.json + vae/diffusion_pytorch_model.safetensors."
)


def resolve_lingbot_vae_dir(model_path: str | Path) -> Path:
    root = Path(model_path)
    direct = root if (root / "config.json").is_file() else root / "vae"
    if not direct.is_dir() or not (direct / "config.json").is_file():
        raise FileNotFoundError(
            f"Lingbot native VAE assets (vae/config.json) were not found under "
            f"'{model_path}'. {_HF_VAE_HINT}"
        )
    return direct


def load_lingbot_vae_config(model_path: str | Path) -> dict[str, Any]:
    vae_dir = resolve_lingbot_vae_dir(model_path)
    payload = json.loads((vae_dir / "config.json").read_text(encoding="utf-8"))
    return {str(key): value for key, value in payload.items() if not str(key).startswith("_")}


def load_lingbot_native_vae(model_path: str | Path) -> AutoencoderKLWan:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for Lingbot VAE loading.")
    vae_dir = resolve_lingbot_vae_dir(model_path)
    config = load_lingbot_vae_config(vae_dir)
    with torch.device("meta"):
        vae = AutoencoderKLWan(**config)
    files = resolve_safetensor_files(
        vae_dir,
        index_names=("diffusion_pytorch_model.safetensors.index.json",),
        direct_names=("diffusion_pytorch_model.safetensors",),
    )
    if not files:
        raise FileNotFoundError(
            f"Lingbot native VAE weights (safetensors) were not found under "
            f"'{vae_dir}'. {_HF_VAE_HINT}"
        )
    load_module_safetensors(
        vae,
        files,
        component_path=vae_dir,
        strict=True,
    )
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def normalize_video_pixels(video: torch.Tensor) -> torch.Tensor:
    value = video.detach().float()
    if value.ndim != 5 or int(value.shape[1]) != 3:
        raise ValueError(
            f"Lingbot VAE input must be [B,3,T,H,W], got {tuple(value.shape)}."
        )
    minimum = float(value.min().item())
    maximum = float(value.max().item())
    if minimum >= -1.0 and maximum <= 1.0 and minimum < 0.0:
        return value
    if minimum >= 0.0 and maximum <= 1.0:
        return value.mul(2.0).sub(1.0)
    if minimum >= 0.0 and maximum <= 255.0:
        return value.div(127.5).sub(1.0)
    raise ValueError(
        "Lingbot VAE pixels must be in [-1,1], [0,1], or [0,255]."
    )


def vae_latent_to_dit_latent(latents: torch.Tensor, config: Any) -> torch.Tensor:
    channels = int(latents.shape[1])
    mean_values = list(getattr(config, "latents_mean"))
    std_values = list(getattr(config, "latents_std"))
    if len(mean_values) != channels or len(std_values) != channels:
        raise ValueError(
            "Lingbot VAE latent normalization metadata does not match latent channels."
        )
    mean = torch.tensor(mean_values, device=latents.device, dtype=torch.float32).view(
        1, channels, 1, 1, 1
    )
    std = torch.tensor(std_values, device=latents.device, dtype=torch.float32).view(
        1, channels, 1, 1, 1
    )
    return (latents.float() - mean) / std


def dit_latent_to_vae_latent(latents: torch.Tensor, config: Any) -> torch.Tensor:
    channels = int(latents.shape[1])
    mean = torch.tensor(
        list(getattr(config, "latents_mean")),
        device=latents.device,
        dtype=torch.float32,
    ).view(1, channels, 1, 1, 1)
    std = torch.tensor(
        list(getattr(config, "latents_std")),
        device=latents.device,
        dtype=torch.float32,
    ).view(1, channels, 1, 1, 1)
    return latents.float() * std + mean


def encode_video_to_lingbot_latent(
    vae: AutoencoderKLWan,
    video: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    sample_posterior: bool = True,
) -> torch.Tensor:
    pixels = normalize_video_pixels(video)
    with torch.no_grad():
        posterior = vae.encode(pixels).latent_dist
        latents = posterior.sample(generator) if sample_posterior else posterior.mode()
        return vae_latent_to_dit_latent(latents, vae.config)


def build_lingbot_vae_artifact(
    latent: torch.Tensor,
    *,
    model_variant: str,
    model_component_id: str,
) -> dict[str, Any]:
    value = latent.detach().cpu().float().contiguous()
    if value.ndim == 5 and int(value.shape[0]) == 1:
        value = value[0]
    if value.ndim != 4:
        raise ValueError(f"Lingbot VAE artifact latent must be CTHW, got {tuple(value.shape)}.")
    return {
        "format": LINGBOT_VAE_ARTIFACT_FORMAT,
        "schema_version": LINGBOT_VAE_ARTIFACT_SCHEMA_VERSION,
        "model_type": "lingbot-video",
        "model_variant": str(model_variant),
        "model_component_id": str(model_component_id),
        "latent_layout": "CTHW",
        "normalization": "wan_vae_latents_mean_std",
        "dit_latent": value,
    }


def validate_lingbot_vae_artifact(
    payload: dict[str, Any],
    *,
    expected_variant: str,
    expected_component_id: str,
) -> torch.Tensor:
    if str(payload.get("format", "")) != LINGBOT_VAE_ARTIFACT_FORMAT:
        raise ValueError("Lingbot .pt artifact is missing the native VAE latent format marker.")
    if int(payload.get("schema_version", 0)) != LINGBOT_VAE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported Lingbot VAE latent artifact schema version.")
    if str(payload.get("model_type", "")).strip().lower() != "lingbot-video":
        raise ValueError("Lingbot VAE latent artifact has incompatible model_type.")
    if str(payload.get("model_variant", "")).strip().lower() != str(expected_variant).strip().lower():
        raise ValueError("Lingbot VAE latent artifact has incompatible model_variant.")
    if str(payload.get("model_component_id", "")) != str(expected_component_id):
        raise ValueError("Lingbot VAE latent artifact has incompatible model_component_id.")
    if str(payload.get("latent_layout", "")) != "CTHW":
        raise ValueError("Lingbot VAE latent artifact must declare CTHW layout.")
    if str(payload.get("normalization", "")) != "wan_vae_latents_mean_std":
        raise ValueError("Lingbot VAE latent artifact has incompatible normalization.")
    return torch.as_tensor(payload["dit_latent"]).detach().cpu().float().contiguous()
