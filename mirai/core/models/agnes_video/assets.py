"""Explicit native conditioning assets for Agnes Video.

Audio loading follows OpenMOSS/OmniVAE's reconstruction API and the Agnes
OmniAudioFeatureExtractor wrapper. Video uses the attributed Wan2.2 implementation.
Sources: https://huggingface.co/Agnes-AI/Agnes-Video-2.0-MoE/tree/fc3971e378b53c766baa06898bfe84d7a56653ea/inference_code
and https://github.com/OpenMOSS/OmniVAE.
"""

from __future__ import annotations

import math
import inspect
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


class AgnesAssets:
    def __init__(self, options: Mapping[str, Any], *, weight_dtype: torch.dtype | None = None):
        self.options = dict(options)
        self.weight_dtype = torch.float32 if weight_dtype is None else weight_dtype
        self.text_weight_dtype = torch.bfloat16 if weight_dtype is None else weight_dtype
        self.root = Path(str(options.get("model_path", options.get("path", ""))))
        self.text_encoder: Any = None
        self.tokenizer: Any = None
        self.video_vae: Any = None
        self.audio_vae: Any = None
        self._audio_geometry: tuple[int, int, int] | None = None

    def _path(self, key: str, default: str) -> Path:
        path = Path(str(self.options.get(key) or self.root / default)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Agnes {key} assets are missing at {path}.")
        return path

    def load_text_encoder(self, device: str = "cpu") -> Any:
        if self.text_encoder is None:
            from transformers import AutoTokenizer
            from transformers.models.t5gemma import T5GemmaEncoderModel

            path = self._path("text_encoder_path", "text_encoder")
            self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            self.text_encoder = T5GemmaEncoderModel.from_pretrained(
                str(path), is_encoder_decoder=False, torch_dtype=self.text_weight_dtype,
                local_files_only=True,
            ).eval().requires_grad_(False)
        self.text_encoder.to(device)
        return self.text_encoder

    @torch.no_grad()
    def encode_prompt(self, prompt: str, device: str = "cpu") -> torch.Tensor:
        model = self.load_text_encoder(device)
        tokens = self.tokenizer([str(prompt)], return_tensors="pt").to(device)
        if tokens["input_ids"].numel() == 0:
            raise ValueError(
                "Agnes T5Gemma tokenizer produced zero tokens; prompt conditioning "
                "must contain at least one token. Omit the negative prompt to use "
                "the released Agnes default."
            )
        hidden = model(**tokens).last_hidden_state.half()[:, :640]
        if hidden.shape[-1] != 3584:
            raise ValueError(f"Agnes requires T5Gemma hidden width 3584, got {hidden.shape[-1]}.")
        return hidden

    def load_video_vae(self, device: str = "cpu") -> Any:
        if self.video_vae is None:
            from mirai.vendors.agnes_video.vae import Wan2_2_VAE

            self.video_vae = Wan2_2_VAE(
                vae_pth=str(self._path("video_vae_path", "vae/Wan2.2_VAE.pth")), device=device,
            ).to(device).to(self.weight_dtype)
            self.video_vae.vae.requires_grad_(False).eval()
        self.video_vae.to(device)
        return self.video_vae

    def load_audio_vae(self, device: str = "cpu") -> Any:
        if self.audio_vae is None:
            code_root = self._path("audio_vae_code_root", "OmniVAE")
            package_parent = code_root / "vae" if (code_root / "vae" / "omnivae").is_dir() else code_root
            if not (package_parent / "omnivae").is_dir():
                raise FileNotFoundError(f"OmniVAE package is missing under {code_root}.")
            package_path = str(package_parent.resolve())
            if package_path not in sys.path:
                sys.path.insert(0, package_path)
            import yaml
            from omnivae.models.audio_vae_dac import DAC
            from .audio import NativeOmniAudio, dac_constructor_options, load_dac_state
            if not Path(inspect.getfile(DAC)).resolve().is_relative_to(package_parent.resolve()):
                raise ValueError("OmniVAE is already imported from a different audio_vae_code_root; use a fresh runtime.")

            checkpoint = self._path("audio_vae_path", "audio_vae")
            with (checkpoint / "config.yaml").open(encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            if not isinstance(config, dict):
                raise ValueError("OmniVAE config.yaml must contain a configuration mapping.")
            options = dac_constructor_options(config)
            with torch.device("meta"):
                dac = DAC(**options)
            load_dac_state(dac, checkpoint / "state_dict.pt")
            hop = int(dac.hop_length)
            rate = int(dac.sample_rate)
            dim = int(dac.latent_dim)
            if min(hop, rate, dim) <= 0:
                raise ValueError("OmniVAE audio geometry must be positive.")
            if dim != 128:
                raise ValueError("Agnes requires an OmniVAE audio latent dimension of 128.")
            self.audio_vae = NativeOmniAudio(dac).eval().requires_grad_(False)
            self._audio_geometry = (rate, hop, dim)
        self.audio_vae.to(device=device, dtype=torch.float32)
        return self.audio_vae

    def audio_geometry(self) -> tuple[int, int, int]:
        if self._audio_geometry is None:
            self.load_audio_vae()
        assert self._audio_geometry is not None
        return self._audio_geometry

    @torch.no_grad()
    def encode_audio(self, waveform: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        model = self.load_audio_vae(device)
        _, hop, _ = self.audio_geometry()
        waveform = waveform.to(device=device, dtype=torch.float32).reshape(1, 1, -1)
        latent, _ = model.encode_audio(waveform, sample_posterior=False)
        length = max(1, math.ceil(waveform.shape[-1] / hop))
        return latent[0, :, :length].transpose(0, 1).contiguous()

    @torch.no_grad()
    def decode_audio(self, latent: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        return self.load_audio_vae(device).decode_audio(latent.transpose(-1, -2).unsqueeze(0).to(device=device, dtype=torch.float32)).contiguous()

    def offload_text_encoder(self) -> None:
        if self.text_encoder is not None:
            self.text_encoder.to("cpu")

    def offload_video_vae(self) -> None:
        if self.video_vae is not None:
            self.video_vae.to("cpu")

    def offload_audio_vae(self) -> None:
        if self.audio_vae is not None:
            self.audio_vae.to("cpu")
