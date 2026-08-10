"""Provider-owned native inference hooks for LingBot-Video.

Single owner of the `NativeVideoPipeline` text-encoder / VAE inference contract
for the LingBot-Video family. The pipeline (``lingbot_video.py``) delegates its
`load_text_encoder` / `encode_prompt` / `offload_text_encoder` / `load_vae` /
`decode_latents_native` / `offload_vae` hooks here so the model file stays thin.

Native-only: the text encoder reuses the exact shared Qwen3-VL loader/encoder
(``lingbot_video_text_encoder``) that the cache path uses — guaranteeing
embedding parity with cached-latent training — and the VAE is the vendored plain
``nn.Module`` ``AutoencoderKLWan`` loaded from safetensors. No diffusers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mirai.core.inference.conditioning import (
    IMAGE_TO_VIDEO,
    VIDEO_TO_VIDEO,
    InferenceConditioningRequest,
    PreparedInferenceConditioning,
)
from mirai.core.models.lingbot_video.text_encoder import LingBotVideoTextEncoder
from mirai.core.models.lingbot_video.text_encoder import resolve_text_encoder_dtype
from mirai.core.models.lingbot_video.vae import dit_latent_to_vae_latent
from mirai.core.models.lingbot_video.vae import encode_video_to_lingbot_latent
from mirai.core.models.lingbot_video.vae import load_lingbot_native_vae
from mirai.vendors.lingbot_video.autoencoder_kl_wan import unpatchify

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _load_rgb_image(source: Any) -> Any:
    from PIL import Image

    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")
    tensor = torch.as_tensor(source).detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError("Input image tensor must be CHW or HWC.")
    if int(tensor.shape[0]) in {1, 3, 4}:
        tensor = tensor.permute(1, 2, 0)
    if int(tensor.shape[-1]) not in {1, 3, 4}:
        raise ValueError("Input image tensor must have 1, 3, or 4 channels.")
    if float(tensor.max()) <= 1.0:
        tensor = tensor.mul(255.0)
    array = tensor.clamp(0, 255).to(torch.uint8).numpy()
    return Image.fromarray(array).convert("RGB")


def _cover_resize_crop(image: Any, *, height: int, width: int) -> Any:
    from PIL import Image

    source_w, source_h = image.size
    scale = max(float(width) / source_w, float(height) / source_h)
    resized_w = max(int(width), int(round(source_w * scale)))
    resized_h = max(int(height), int(round(source_h * scale)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    left = max(0, (resized_w - int(width)) // 2)
    top = max(0, (resized_h - int(height)) // 2)
    return resized.crop((left, top, left + int(width), top + int(height)))


def _pil_video_tensor(images: list[Any]) -> "torch.Tensor":
    import numpy as np

    frames = [
        torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        for image in images
    ]
    return torch.stack(frames, dim=1).unsqueeze(0)


def _load_video_frames(
    source: Any,
    *,
    frame_count: int,
    height: int,
    width: int,
) -> list[Any]:
    if not isinstance(source, (str, Path)):
        tensor = torch.as_tensor(source).detach().cpu().float()
        if tensor.ndim == 5 and int(tensor.shape[0]) == 1:
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise ValueError("Input video tensor must be CTHW, TCHW, or THWC.")
        if int(tensor.shape[0]) in {1, 3, 4}:
            tensor = tensor.permute(1, 2, 3, 0)
        elif int(tensor.shape[1]) in {1, 3, 4}:
            tensor = tensor.permute(0, 2, 3, 1)
        if int(tensor.shape[0]) < int(frame_count):
            raise ValueError(
                f"Input video has {int(tensor.shape[0])} frames; "
                f"{int(frame_count)} are required."
            )
        indexes = torch.linspace(
            0, int(tensor.shape[0]) - 1, int(frame_count)
        ).round().long()
        return [
            _cover_resize_crop(
                _load_rgb_image(tensor[index]), height=height, width=width
            )
            for index in indexes
        ]

    import cv2
    from PIL import Image

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < int(frame_count):
            raise ValueError(
                f"Input video has {total} frames; {int(frame_count)} are required."
            )
        indexes = (
            torch.linspace(0, total - 1, int(frame_count)).round().long().tolist()
        )
        frames: list[Any] = []
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"Could not decode frame {index} from {path}.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(
                _cover_resize_crop(
                    Image.fromarray(rgb, mode="RGB"),
                    height=height,
                    width=width,
                )
            )
        return frames
    finally:
        capture.release()


def _to_single_bcthw(latent: Any) -> "torch.Tensor":
    tensor = torch.as_tensor(latent)
    if tensor.ndim == 4:
        return tensor.unsqueeze(0)
    if tensor.ndim == 5:
        if int(tensor.shape[0]) != 1:
            raise ValueError(
                "LingBot-Video native decode expects one latent video per list "
                f"entry ([C,T,H,W] or [1,C,T,H,W]); got batch {tuple(tensor.shape)}."
            )
        return tensor
    raise ValueError(
        "LingBot-Video native decode latent must be [C,T,H,W] or [1,C,T,H,W]; "
        f"got shape {tuple(tensor.shape)}."
    )


def causal_chunked_decode(
    vae: Any,
    vae_latent: "torch.Tensor",
    *,
    chunk_size: int,
) -> "torch.Tensor":
    """Decode a single VAE-space latent [1,C,T_lat,H,W] to pixels [1,3,T,H,W].

    Reproduces ``AutoencoderKLWan._decode`` frame-by-frame (causal feat-cache
    continuity), so the result is bit-identical to a whole-tensor ``vae.decode``
    regardless of ``chunk_size``. ``chunk_size`` only bounds how many decoded
    latent-frames are held on the compute device before being moved to host,
    which never changes the values (``torch.cat`` is exact/associative).
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for LingBot-Video native decode.")
    chunk = max(1, int(chunk_size))
    vae.clear_cache()
    x = vae.post_quant_conv(vae_latent)
    num_frame = int(x.shape[2])
    collected: list[torch.Tensor] = []
    pending: list[torch.Tensor] = []
    for i in range(num_frame):
        vae._conv_idx = [0]
        out_i = vae.decoder(
            x[:, :, i : i + 1, :, :],
            feat_cache=vae._feat_map,
            feat_idx=vae._conv_idx,
            first_chunk=(i == 0),
        )
        pending.append(out_i)
        if len(pending) >= chunk:
            collected.append(torch.cat(pending, dim=2).detach().to("cpu"))
            pending = []
    if pending:
        collected.append(torch.cat(pending, dim=2).detach().to("cpu"))
    out = torch.cat(collected, dim=2)
    if vae.config.patch_size is not None:
        out = unpatchify(out, patch_size=vae.config.patch_size)
    out = torch.clamp(out, min=-1.0, max=1.0)
    vae.clear_cache()
    return out


class LingBotVideoNativeInference:
    """Native text/VAE inference assets for a LingBot-Video model snapshot."""

    def __init__(self, model_config: Any) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("LingBot-Video native inference requires torch.")
        self.model_config = model_config
        self.model_root = Path(str(model_config.path))
        params = model_config.params
        text_encoder_override = str(getattr(params, "text_encoder_path", "") or "").strip()
        text_encoder_dir = (
            Path(text_encoder_override)
            if text_encoder_override
            else self.model_root / "text_encoder"
        )
        self.text_backend = LingBotVideoTextEncoder(
            text_encoder_dir=text_encoder_dir,
            processor_dir=self.model_root / "processor",
            dtype=resolve_text_encoder_dtype(str(getattr(model_config, "dtype", "bf16"))),
        )
        self.vae_chunk_size = max(1, int(getattr(params, "vae_chunk_size", 16)))
        self.vae_tiling = False
        self.vae_tile_size = 256
        self.vae_tile_stride = 192
        self._vae: Any | None = None

    # -- text encoder ------------------------------------------------------
    def load_text_encoder(self, *, device: str) -> None:
        self.text_backend.load()
        self.text_backend.model.to(device=torch.device(device))

    def encode_prompt(
        self, prompt: str, *, device: str, image: Any | None = None
    ) -> "torch.Tensor":
        self.text_backend.load()
        self.text_backend.model.to(device=torch.device(device))
        return self.text_backend.encode(str(prompt), image=image)

    def offload_text_encoder(self) -> None:
        if self.text_backend.loaded:
            self.text_backend.model.to(device=torch.device("cpu"))

    # -- vae ---------------------------------------------------------------
    def load_vae(self, *, device: str) -> None:
        if self._vae is None:
            self._vae = load_lingbot_native_vae(self.model_root)
        if self.vae_tiling:
            self._vae.enable_tiling(
                tile_sample_min_height=self.vae_tile_size,
                tile_sample_min_width=self.vae_tile_size,
                tile_sample_stride_height=self.vae_tile_stride,
                tile_sample_stride_width=self.vae_tile_stride,
            )
        self._vae.to(device=torch.device(device))

    def configure_vae_tiling(
        self,
        *,
        enabled: bool,
        tile_size: int,
        tile_stride: int,
    ) -> None:
        self.vae_tiling = bool(enabled)
        self.vae_tile_size = int(tile_size)
        self.vae_tile_stride = int(tile_stride)
        if self._vae is not None and self.vae_tiling:
            self._vae.enable_tiling(
                tile_sample_min_height=self.vae_tile_size,
                tile_sample_min_width=self.vae_tile_size,
                tile_sample_stride_height=self.vae_tile_stride,
                tile_sample_stride_width=self.vae_tile_stride,
            )

    def offload_vae(self) -> None:
        if self._vae is not None:
            self._vae.to(device=torch.device("cpu"))

    def encode_video_native(
        self,
        video: Any,
        *,
        generator: Any | None = None,
        sample_posterior: bool = True,
    ) -> "torch.Tensor":
        """VAE-encode a pixel video ``[B,3,T,H,W]`` into a DiT-space latent.

        The inverse of :meth:`decode_latents_native`, used by the refiner upscale
        (decode base latent -> bicubic resize -> re-encode). Moves the video onto
        the resident VAE's device; the VAE stays fp32 (``load_vae`` never casts
        dtype), so a posterior sample with a device-matched generator is exact.
        """
        if self._vae is None:
            raise RuntimeError(
                "LingBot-Video native VAE is not loaded; call load_vae() first."
            )
        vae = self._vae
        param = next(vae.parameters())
        pixels = torch.as_tensor(video).to(device=param.device, dtype=torch.float32)
        if pixels.ndim != 5 or int(pixels.shape[1]) != 3:
            raise ValueError(
                "LingBot-Video native encode expects one pixel video shaped "
                f"[B,3,T,H,W]; got shape {tuple(pixels.shape)}."
            )
        with torch.no_grad():
            return encode_video_to_lingbot_latent(
                vae,
                pixels,
                generator=generator,
                sample_posterior=bool(sample_posterior),
            )

    def prepare_conditioning(
        self,
        request: InferenceConditioningRequest,
        *,
        device: str,
        generator: Any,
    ) -> PreparedInferenceConditioning:
        request.validate()
        if request.task not in {IMAGE_TO_VIDEO, VIDEO_TO_VIDEO}:
            return PreparedInferenceConditioning.unconditioned(request)

        self.load_vae(device=device)
        try:
            if request.task == IMAGE_TO_VIDEO:
                image = _cover_resize_crop(
                    _load_rgb_image(request.input_image),
                    height=int(request.height),
                    width=int(request.width),
                )
                pixels = _pil_video_tensor([image])
                latent = self.encode_video_native(
                    pixels,
                    generator=generator,
                    sample_posterior=True,
                )
                return PreparedInferenceConditioning(
                    task=request.task,
                    prompt_media=image,
                    condition_latent=latent,
                    denoising_strength=1.0,
                )

            frames = _load_video_frames(
                request.input_video,
                frame_count=int(request.frame_count),
                height=int(request.height),
                width=int(request.width),
            )
            latent = self.encode_video_native(
                _pil_video_tensor(frames),
                generator=None,
                sample_posterior=False,
            )
            return PreparedInferenceConditioning(
                task=request.task,
                source_latent=latent,
                denoising_strength=float(request.denoising_strength),
            )
        finally:
            self.offload_vae()

    def decode_latents_native(self, latents: list[Any]) -> "torch.Tensor":
        if self._vae is None:
            raise RuntimeError(
                "LingBot-Video native VAE is not loaded; call load_vae() first."
            )
        vae = self._vae
        param = next(vae.parameters())
        device = param.device
        dtype = param.dtype
        videos: list[torch.Tensor] = []
        for latent in latents:
            dit_latent = _to_single_bcthw(latent).float()
            vae_latent = dit_latent_to_vae_latent(dit_latent, vae.config).to(
                device=device, dtype=dtype
            )
            with torch.no_grad():
                if self.vae_tiling:
                    decoded = vae.decode(vae_latent).sample.detach().to("cpu")
                else:
                    decoded = causal_chunked_decode(
                        vae, vae_latent, chunk_size=self.vae_chunk_size
                    )
            # [1,3,T,H,W] pixels in [-1,1] -> [T,3,H,W] in [0,1]
            frames = decoded[0].permute(1, 0, 2, 3).contiguous().float()
            frames = frames.add(1.0).mul_(0.5).clamp_(0.0, 1.0)
            videos.append(frames)
        if len(videos) == 1:
            return videos[0]
        return torch.stack(videos, dim=0)
