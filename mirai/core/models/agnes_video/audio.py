"""Native audio-only OmniVAE loading and deterministic posterior semantics.

Adapted from OpenMOSS/OmniVAE AudioVideoVAE.encode_audio/decode_audio:
https://github.com/OpenMOSS/OmniVAE/tree/d30e99d2d21f9461e394237ee98fc55ae0c011c2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


DAC_DEFAULTS = {
    "encoder_dim": 64, "encoder_rates": [2, 4, 8, 8], "latent_dim": None,
    "decoder_dim": 1536, "decoder_rates": [8, 8, 4, 2], "n_codebooks": 9,
    "codebook_size": 1024, "codebook_dim": 8, "quantizer_dropout": False,
    "sample_rate": 24000, "continuous": True,
}
IGNORED_STATE_PREFIXES = (
    "video_vae.", "contrastive_head.", "llm_caption_head.",
    "image_distill_proj.", "video_distill_proj.", "audio_distill_proj.",
)


def dac_constructor_options(config: dict[str, Any]) -> dict[str, Any]:
    audio = (config.get("model") or {}).get("audio") or {}
    if not isinstance(audio, dict):
        raise ValueError("OmniVAE model.audio must be a configuration mapping.")
    unknown = set(audio) - set(DAC_DEFAULTS) - {"audio_sample_rate", "pretrained_model_name_or_path"}
    if unknown:
        raise ValueError(f"Unsupported OmniVAE audio configuration keys: {sorted(unknown)}.")
    options = {key: audio.get(key, default) for key, default in DAC_DEFAULTS.items()}
    options["sample_rate"] = audio.get("sample_rate", audio.get("audio_sample_rate", 24000))
    if not options["continuous"]:
        raise ValueError("Agnes requires a continuous OmniVAE audio posterior.")
    return options


def load_dac_state(model: nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for name in ("model_state_dict", "state_dict", "module", "model"):
            if isinstance(payload.get(name), dict):
                payload = payload[name]
                break
    if not isinstance(payload, dict) or not payload or not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError("OmniVAE checkpoint must contain a tensor state dictionary.")
    prefixed = any(name.removeprefix("module.").startswith("audio_vae.") for name in payload)
    state = {}
    for original, tensor in payload.items():
        name = original.removeprefix("module.")
        if prefixed:
            if name.startswith(IGNORED_STATE_PREFIXES):
                continue
            if not name.startswith("audio_vae."):
                raise ValueError(f"Unexpected OmniVAE checkpoint key '{original}'.")
            name = name.removeprefix("audio_vae.")
        if name in state:
            raise ValueError(f"Duplicate OmniVAE checkpoint key '{name}'.")
        state[name] = tensor
    model.load_state_dict(state, strict=True, assign=True)


class NativeOmniAudio(nn.Module):
    def __init__(self, audio_vae: nn.Module):
        super().__init__()
        self.audio_vae = audio_vae
        self._decode_warm_key = None

    def encode_audio(self, audio: torch.Tensor, *, sample_posterior: bool = False):
        posterior, _, _, _, _ = self.audio_vae.encode(self.audio_vae.preprocess(audio))
        return (posterior.sample() if sample_posterior else posterior.mode()), posterior

    def decode_audio(self, latent: torch.Tensor):
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            deterministic=True,
            benchmark=False,
            allow_tf32=torch.backends.cudnn.allow_tf32,
        ):
            if not self.training and not torch.is_grad_enabled():
                key = (tuple(latent.shape), tuple(latent.stride()), str(latent.device),
                       latent.dtype, torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32)
                if key != self._decode_warm_key:
                    # Prime shape-specific GPU kernels before returning PCM;
                    # cold and warmed DAC calls can choose different rounding.
                    self.audio_vae.decode(latent)
                    self._decode_warm_key = key
            return self.audio_vae.decode(latent)
