"""Provider-owned native cache encoder for LingBot-Video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mirai.core.dataset.native_encode import (
    _IMAGE_SUFFIXES,
    VIDEO_MEDIA_SUFFIXES,
    _load_video_media,
    _looks_like_latent,
    _to_cthw,
    BucketInfo,
    NativeCacheStatus,
)
from mirai.core.models.lingbot_video.checkpoints import load_lingbot_transformer_config
from mirai.core.models.lingbot_video.text_encoder import LingBotVideoTextEncoder
from mirai.core.models.lingbot_video.text_encoder import PROMPT_TEMPLATE
from mirai.core.models.lingbot_video.text_encoder import TOKEN_LENGTH
from mirai.core.models.lingbot_video.text_encoder import resolve_text_encoder_dtype
from mirai.core.models.lingbot_video.vae import encode_video_to_lingbot_latent
from mirai.core.models.lingbot_video.vae import load_lingbot_native_vae
from mirai.core.models.lingbot_video.vae import validate_lingbot_vae_artifact
from mirai.core.models.providers import NativeCacheEncoderConfig
from mirai.core.training.residency.component_residency import (
    ComponentResidencySpec,
    SequentialComponentResidency,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_ALT_CHANNELS = {1, 2, 3, 4, 6, 8, 16, 24, 32, 48}

# Prompt template and token budget are owned by the shared native text encoder.
__all__ = ["LingBotVideoNativeCacheEncoder", "PROMPT_TEMPLATE", "TOKEN_LENGTH"]


def _dtype_from_name(name: str) -> Any:
    return resolve_text_encoder_dtype(name)


def _load_pt_dict(path: Path) -> dict[str, Any] | None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for LingBot-Video cache encoding.")
    payload = torch.load(path, map_location="cpu")
    return payload if isinstance(payload, dict) else None


def _load_pt_payload(path: Path) -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for LingBot-Video cache encoding.")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("latent", "latents", "dit_latent", "video_latent"):
            if key in payload:
                return payload[key]
    return payload


class LingBotVideoNativeCacheEncoder:
    """Native LingBot cache encoder.

    This encoder intentionally supports only the pieces Mirai can own natively:
    precomputed DiT latent tensors and Qwen3-VL prompt embeddings. Pixel/video
    VAE encoding still requires a native VAE wrapper and must fail explicitly.
    """

    def __init__(self, config: NativeCacheEncoderConfig) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for LingBot-Video cache encoding.")
        self.config = config
        self.model_root = Path(config.model_path)
        self.text_encoder_dir = self._resolve_text_encoder_dir(config)
        self.processor_dir = self.model_root / "processor"
        self.max_frames = max(1, int(config.max_frames))
        self.latent_channels = self._resolve_latent_channels(config)
        self._text_backend = LingBotVideoTextEncoder(
            text_encoder_dir=self.text_encoder_dir,
            processor_dir=self.processor_dir,
            dtype=_dtype_from_name(config.dtype_name),
        )
        self._text_loaded = False
        self._vae: Any | None = None
        self._vae_loaded = False
        # Reflects the most recent latent-encoding path. Precomputed DiT-latent
        # passthrough never touches the native VAE assets; native VAE encoding of
        # pixel/video media does. The initial value matches the native VAE mode so
        # the pre-encode cache fingerprint (built from status()) stays stable.
        self._vae_mode = "native_wan_vae"
        self._component_residency = SequentialComponentResidency(
            compute_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            idle_device="cpu",
        )

    def _resolve_latent_channels(self, config: NativeCacheEncoderConfig) -> int:
        try:
            transformer_config = load_lingbot_transformer_config(config.model_path)
            return int(transformer_config.get("in_channels", 16))
        except Exception:
            if bool(config.strict_assets):
                raise
            return 16

    def _resolve_text_encoder_dir(self, config: NativeCacheEncoderConfig) -> Path:
        override = str(config.text_encoder_path or "").strip()
        if override:
            return Path(override)
        return self.model_root / "text_encoder"

    def _normalize_channel_layout(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(
                "LingBot-Video latent input must be convertible to [C,T,H,W] layout."
            )
        channels = int(tensor.shape[0])
        frames = int(tensor.shape[1])
        expected_frames = int(self.max_frames)
        if channels == self.latent_channels:
            return tensor
        if channels in _ALT_CHANNELS and frames == expected_frames:
            return tensor
        if channels == expected_frames and frames in _ALT_CHANNELS:
            return tensor.permute(1, 0, 2, 3).contiguous()
        raise ValueError(
            f"Unsupported LingBot-Video latent layout {tuple(tensor.shape)}; "
            f"expected [C,T,H,W] with C={self.latent_channels}, "
            f"or alternate [C,T,H,W] where T={expected_frames}."
        )

    def _adapt_channel_count(self, tensor: torch.Tensor) -> torch.Tensor:
        channels = int(tensor.shape[0])
        if channels == self.latent_channels:
            return tensor
        if channels <= 0:
            raise ValueError(
                f"Cannot adapt LingBot-Video latent channels from non-positive input channels {channels}."
            )
        repeats = (self.latent_channels + channels - 1) // channels
        out = tensor.repeat_interleave(repeats, dim=0)
        if out.shape[0] > self.latent_channels:
            out = out[: self.latent_channels]
        return out

    def status(self) -> NativeCacheStatus:
        return NativeCacheStatus(
            enabled=bool(self.config.enabled),
            vae_loaded=bool(self._vae_loaded),
            text_loaded=bool(self._text_loaded),
            clip_loaded=False,
            vae_mode=str(self._vae_mode),
            text_mode="qwen3vl_hidden_state",
            clip_mode="none",
        )

    def _model_component_id(self) -> str:
        subfolder = str(self.config.denoiser_subfolder or "transformer").strip()
        return f"denoiser_subfolder:{subfolder}"

    def _load_vae_assets(self) -> None:
        if self._vae_loaded:
            return
        self._vae = load_lingbot_native_vae(self.model_root)
        self._vae.enable_tiling()
        self._component_residency.register(
            ComponentResidencySpec(
                name="vae",
                module=self._vae,
                compute_dtype=torch.float32,
            )
        )
        self._vae_loaded = True

    def _prepare_video_for_vae(self, media_path: Path) -> tuple[torch.Tensor, BucketInfo]:
        video = _load_video_media(media_path, max_frames=self.max_frames)
        frame_count = int(video.shape[1])
        valid_frames = max(1, 1 + 4 * max(0, (frame_count - 1) // 4))
        video = video[:, :valid_frames]
        height = int(video.shape[-2])
        width = int(video.shape[-1])
        target_height = max(16, (height // 16) * 16)
        target_width = max(16, (width // 16) * 16)
        if bool(self.config.enable_bucketing) and self.config.resolution_buckets:
            from mirai.core.dataset.bucketing.bucket_resolve import choose_resolution_bucket
            from mirai.core.dataset.media.media_resize import resize_crop_tensor

            target_height, target_width = choose_resolution_bucket(
                height,
                width,
                list(self.config.resolution_buckets),
            )
            video = resize_crop_tensor(
                video,
                target_height,
                target_width,
                mode=str(self.config.bucket_resize_mode),
            )
        elif target_height != height or target_width != width:
            top = max(0, (height - target_height) // 2)
            left = max(0, (width - target_width) // 2)
            video = video[:, :, top : top + target_height, left : left + target_width]
        return video, BucketInfo(
            bucket_id=f"{target_height}x{target_width}x{valid_frames}",
            bucket_h=target_height,
            bucket_w=target_width,
            bucket_frames=valid_frames,
        )

    def _load_text_assets(self) -> None:
        if self._text_loaded:
            return
        self._text_backend.load()
        self._component_residency.register(
            ComponentResidencySpec(
                name="text_encoder",
                module=self._text_backend.model,
                compute_dtype=_dtype_from_name(self.config.dtype_name),
            )
        )
        self._text_loaded = True

    def _compute_crop_start(self) -> int:
        self._load_text_assets()
        return self._text_backend.crop_start()

    def encode_text(self, caption: str) -> torch.Tensor:
        self._load_text_assets()
        # crop-start uses CPU tokenization only; resolve it before activating so
        # the residency window wraps exactly the on-device forward pass.
        self._text_backend.crop_start()
        with self._component_residency.activate("text_encoder"):
            return self._text_backend.encode(str(caption))

    def encode_text_for_media(
        self,
        caption: str,
        media_path: Path,
    ) -> tuple[torch.Tensor, list[int] | None] | torch.Tensor:
        if media_path.suffix.lower() == ".pt":
            payload = _load_pt_dict(media_path)
            if payload is not None:
                for key in ("text_embed", "text_embeds", "lingbot", "t5"):
                    if key in payload:
                        embed = torch.as_tensor(payload[key]).detach().cpu().float()
                        mask_value = payload.get("text_mask", payload.get("attention_mask"))
                        mask = (
                            [int(v) for v in torch.as_tensor(mask_value).reshape(-1).tolist()]
                            if mask_value is not None
                            else None
                        )
                        return embed, mask
        return self.encode_text(caption)

    def encode_latent(self, media_path: Path) -> tuple[torch.Tensor, BucketInfo | None]:
        suffix = media_path.suffix.lower()
        if suffix == ".pt":
            payload = _load_pt_dict(media_path)
            if payload is not None and str(payload.get("format", "")):
                tensor = validate_lingbot_vae_artifact(
                    payload,
                    expected_variant=str(self.config.variant),
                    expected_component_id=self._model_component_id(),
                )
            elif bool(self.config.strict_assets):
                raise ValueError(
                    "Strict Lingbot training requires a versioned native VAE latent "
                    "artifact; unversioned .pt tensors are not accepted."
                )
            else:
                tensor = torch.as_tensor(_load_pt_payload(media_path)).detach().cpu().float()
            tensor = _to_cthw(tensor)
            try:
                tensor = self._normalize_channel_layout(tensor)
                if not bool(self.config.strict_assets):
                    tensor = self._adapt_channel_count(tensor)
            except Exception:
                raise ValueError(
                    "LingBot-Video .pt cache input must contain precomputed DiT "
                    f"latents shaped [C,T,H,W] with C={self.latent_channels}; got {tuple(tensor.shape)}. "
                    f"Alternate [C,T,H,W] layouts are normalized only when the frame count "
                    f"matches {self.max_frames}."
                ) from None
            if _looks_like_latent(tensor, latent_channels=self.latent_channels):
                # Precomputed DiT-latent passthrough must not require model assets.
                # The patch-size divisibility check is best-effort: when the
                # transformer config is unavailable (asset-free passthrough) and we
                # are not in strict-assets mode, fall back to the canonical LingBot
                # patch grid instead of hard-failing on a missing config.json.
                try:
                    patch = load_lingbot_transformer_config(self.config.model_path).get(
                        "patch_size", [1, 2, 2]
                    )
                except Exception:
                    if bool(self.config.strict_assets):
                        raise
                    patch = [1, 2, 2]
                patch_t, patch_h, patch_w = [int(v) for v in patch]
                if (
                    int(tensor.shape[1]) % patch_t
                    or int(tensor.shape[2]) % patch_h
                    or int(tensor.shape[3]) % patch_w
                ):
                    raise ValueError(
                        f"Lingbot latent shape {tuple(tensor.shape)} is incompatible "
                        f"with transformer patch_size={(patch_t, patch_h, patch_w)}."
                    )
                self._vae_mode = "precomputed_lingbot_dit_latent_passthrough"
                return tensor.contiguous(), None
            raise ValueError(
                "LingBot-Video .pt cache input must contain precomputed DiT "
                f"latents shaped [C,T,H,W] with C={self.latent_channels}; got {tuple(tensor.shape)}. "
                "Alternate [C,T,H,W] layouts are normalized only when the frame count "
                f"matches {self.max_frames}."
            )
        if suffix in VIDEO_MEDIA_SUFFIXES:
            self._load_vae_assets()
            self._vae_mode = "native_wan_vae"
            video, bucket = self._prepare_video_for_vae(media_path)
            with self._component_residency.activate("vae") as vae:
                device = next(vae.parameters()).device
                latent = encode_video_to_lingbot_latent(
                    vae,
                    video.unsqueeze(0).to(device=device, dtype=torch.float32),
                    sample_posterior=False,
                )
            return latent[0].detach().cpu().float().contiguous(), bucket
        if suffix in _IMAGE_SUFFIXES:
            raise ValueError(
                "LingBot image cache encoding is unsupported; provide video media "
                "or a versioned native VAE latent artifact."
            )
        raise ValueError(f"Unsupported LingBot-Video cache media extension: {media_path.suffix}")

    def encode_clip(self, media_path: Path) -> None:
        _ = media_path
        return None
