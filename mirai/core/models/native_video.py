"""Native video pipeline contracts for sparse-MoE diffusion models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mirai.core.models.base import BasePipeline

if TYPE_CHECKING:
    from mirai.core.inference.conditioning import (
        InferenceConditioningRequest,
        PreparedInferenceConditioning,
    )

def _ceil_div(value: int, divisor: int) -> int:
    return max(1, (max(1, int(value)) + max(1, int(divisor)) - 1) // max(1, int(divisor)))


@dataclass(frozen=True)
class VideoLatentLayout:
    """Model-owned mapping from requested pixels/frames to latent geometry."""

    latent_channels: int
    temporal_downsample: int = 1
    spatial_downsample: int = 8
    layout: str = "BCTHW"
    frame_count_modulus: int | None = None
    frame_count_remainder: int = 0
    frame_count_rule: str = ""
    request_spatial_multiple: int | None = None

    def validate_request(self, *, frame_count: int, height: int, width: int) -> None:
        frame_count = int(frame_count)
        height = int(height)
        width = int(width)
        if frame_count <= 0:
            raise ValueError("frame_count must be > 0.")
        if height <= 0:
            raise ValueError("height must be > 0.")
        if width <= 0:
            raise ValueError("width must be > 0.")
        if str(self.layout).upper() != "BCTHW":
            raise ValueError(f"Unsupported video latent layout '{self.layout}'.")

        modulus_value = self.frame_count_modulus
        if modulus_value is not None:
            modulus = int(modulus_value)
            if modulus <= 0:
                raise ValueError("frame_count_modulus must be > 0 when set.")
            expected = int(self.frame_count_remainder) % modulus
            if frame_count % modulus != expected:
                rule = str(self.frame_count_rule).strip() or (
                    f"{expected} modulo {modulus}"
                )
                raise ValueError(
                    f"frame_count={frame_count} is unsupported for this video "
                    f"latent layout; expected {rule}."
                )

        multiple_value = self.request_spatial_multiple
        if multiple_value is not None:
            multiple = int(multiple_value)
            if multiple <= 0:
                raise ValueError("request_spatial_multiple must be > 0 when set.")
            if height % multiple != 0 or width % multiple != 0:
                raise ValueError(
                    f"height={height} and width={width} are unsupported for this "
                    f"video latent layout; both must be multiples of {multiple}."
                )

    def preview_geometry(
        self,
        *,
        frame_count: int,
        height: int,
        width: int,
    ) -> tuple[int, int, int, int]:
        self.validate_request(frame_count=frame_count, height=height, width=width)
        return (
            max(1, int(self.latent_channels)),
            _ceil_div(frame_count, self.temporal_downsample),
            _ceil_div(height, self.spatial_downsample),
            _ceil_div(width, self.spatial_downsample),
        )


class NativeVideoPipeline(BasePipeline):
    """Base contract for native sparse-MoE video diffusion backends."""

    def get_video_latent_layout(self) -> VideoLatentLayout:
        raise NotImplementedError(
            f"{type(self).__name__} must declare its video latent layout."
        )

    def preview_latent_geometry(
        self, *, frame_count: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        return self.get_video_latent_layout().preview_geometry(
            frame_count=frame_count,
            height=height,
            width=width,
        )

    def i2v_conditioning_frame_dim(self, *, latents: Any) -> int:
        if int(getattr(latents, "ndim", 0)) == 5:
            layout = str(self.get_video_latent_layout().layout).upper()
            if layout == "BCTHW":
                return 2
            raise ValueError(f"Unsupported video latent layout '{layout}'.")
        return super().i2v_conditioning_frame_dim(latents=latents)

    def load_text_encoder(self, *, device: str) -> None:
        _ = device
        raise NotImplementedError(
            f"{type(self).__name__} does not implement native text encoder loading."
        )

    def encode_prompt(self, prompt: str, *, device: str) -> Any:
        _ = prompt, device
        raise NotImplementedError(
            f"{type(self).__name__} does not implement native prompt encoding."
        )

    def prepare_inference_conditioning(
        self,
        request: "InferenceConditioningRequest",
        *,
        device: str,
        generator: Any,
    ) -> "PreparedInferenceConditioning":
        """Prepare optional media conditioning through a family-owned hook."""
        _ = device, generator
        from mirai.core.inference.conditioning import (
            IMAGE_TO_VIDEO,
            VIDEO_TO_VIDEO,
            PreparedInferenceConditioning,
        )

        request.validate()
        if request.task in {IMAGE_TO_VIDEO, VIDEO_TO_VIDEO}:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement {request.task} conditioning."
            )
        return PreparedInferenceConditioning.unconditioned(request)

    def encode_conditioned_prompt(
        self,
        prompt: str,
        *,
        prepared: "PreparedInferenceConditioning",
        device: str,
    ) -> Any:
        """Encode text plus provider-prepared prompt media when supported."""
        if prepared.prompt_media is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement multimodal prompt encoding."
            )
        return self.encode_prompt(prompt, device=device)

    def offload_text_encoder(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement native text encoder offload."
        )

    def load_vae(self, *, device: str) -> None:
        _ = device
        raise NotImplementedError(
            f"{type(self).__name__} does not implement native VAE loading."
        )

    def decode_latents_native(self, latents: list[Any]) -> Any:
        _ = latents
        raise NotImplementedError(
            f"{type(self).__name__} does not implement native VAE decoding."
        )

    def offload_vae(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement native VAE offload."
        )
