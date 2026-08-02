"""Model-agnostic conditioning contract for native image/video inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TEXT_TO_VIDEO = "text_to_video"
TEXT_TO_IMAGE = "text_to_image"
IMAGE_TO_VIDEO = "image_to_video"
VIDEO_TO_VIDEO = "video_to_video"

INFERENCE_TASK_ALIASES = {
    "t2v": TEXT_TO_VIDEO,
    TEXT_TO_VIDEO: TEXT_TO_VIDEO,
    "t2i": TEXT_TO_IMAGE,
    TEXT_TO_IMAGE: TEXT_TO_IMAGE,
    "ti2v": IMAGE_TO_VIDEO,
    "i2v": IMAGE_TO_VIDEO,
    "text_image_to_video": IMAGE_TO_VIDEO,
    IMAGE_TO_VIDEO: IMAGE_TO_VIDEO,
    "v2v": VIDEO_TO_VIDEO,
    VIDEO_TO_VIDEO: VIDEO_TO_VIDEO,
}
INFERENCE_TASKS = (
    TEXT_TO_VIDEO,
    TEXT_TO_IMAGE,
    IMAGE_TO_VIDEO,
    VIDEO_TO_VIDEO,
)


def normalize_inference_task(value: str) -> str:
    key = str(value or TEXT_TO_VIDEO).strip().lower()
    try:
        return INFERENCE_TASK_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(INFERENCE_TASKS)
        raise ValueError(
            f"Unknown inference task '{value}'; expected one of: {supported}."
        ) from exc


@dataclass(frozen=True)
class InferenceConditioningRequest:
    """Per-generation media conditioning, independent of any model family."""

    task: str = TEXT_TO_VIDEO
    input_image: Any | None = None
    input_video: Any | None = None
    denoising_strength: float = 1.0
    frame_count: int = 17
    height: int = 480
    width: int = 832

    def validate(self) -> "InferenceConditioningRequest":
        task = normalize_inference_task(self.task)
        object.__setattr__(self, "task", task)
        strength = float(self.denoising_strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("denoising_strength must be between 0 and 1.")
        if int(self.frame_count) <= 0 or int(self.height) <= 0 or int(self.width) <= 0:
            raise ValueError("frame_count, height, and width must be > 0.")

        has_image = self.input_image is not None and (
            not isinstance(self.input_image, str) or bool(self.input_image.strip())
        )
        has_video = self.input_video is not None and (
            not isinstance(self.input_video, str) or bool(self.input_video.strip())
        )
        if task == TEXT_TO_IMAGE:
            if int(self.frame_count) != 1:
                raise ValueError("text_to_image requires frame_count=1.")
            if has_image or has_video:
                raise ValueError("text_to_image does not accept input media.")
        elif task == IMAGE_TO_VIDEO:
            if not has_image or has_video:
                raise ValueError(
                    "image_to_video requires input_image and does not accept input_video."
                )
        elif task == VIDEO_TO_VIDEO:
            if not has_video or has_image:
                raise ValueError(
                    "video_to_video requires input_video and does not accept input_image."
                )
        elif has_image or has_video:
            raise ValueError("text_to_video does not accept input media.")

        if task != VIDEO_TO_VIDEO and strength != 1.0:
            raise ValueError(
                "denoising_strength is only meaningful for video_to_video."
            )
        return self


@dataclass
class PreparedInferenceConditioning:
    """Provider-prepared tensors consumed by the generic denoise loop."""

    task: str
    prompt_media: Any | None = None
    condition_latent: Any | None = None
    source_latent: Any | None = None
    denoising_strength: float = 1.0

    @classmethod
    def unconditioned(
        cls, request: InferenceConditioningRequest
    ) -> "PreparedInferenceConditioning":
        request.validate()
        return cls(
            task=request.task,
            denoising_strength=float(request.denoising_strength),
        )

    def pin_condition(self, latent: Any) -> Any:
        """Overwrite the conditioned temporal prefix without changing its tail."""
        if self.condition_latent is None:
            return latent
        condition = self.condition_latent
        if int(getattr(latent, "ndim", 0)) == 4:
            if int(getattr(condition, "ndim", 0)) == 5:
                if int(condition.shape[0]) != 1:
                    raise ValueError("Condition latent batch must be one.")
                condition = condition[0]
            if int(getattr(condition, "ndim", 0)) != 4:
                raise ValueError("Condition latent must be CTHW or 1CTHW.")
            if tuple(latent.shape[:1] + latent.shape[2:]) != tuple(
                condition.shape[:1] + condition.shape[2:]
            ):
                raise ValueError(
                    "Condition latent channels/spatial shape does not match target "
                    f"latent: {tuple(condition.shape)} vs {tuple(latent.shape)}."
                )
            if int(condition.shape[1]) > int(latent.shape[1]):
                raise ValueError("Condition latent is longer than target latent.")
            latent[:, : int(condition.shape[1])].copy_(
                condition.to(device=latent.device, dtype=latent.dtype)
            )
            return latent
        if int(getattr(latent, "ndim", 0)) == 5:
            if int(getattr(condition, "ndim", 0)) == 4:
                condition = condition.unsqueeze(0)
            if int(getattr(condition, "ndim", 0)) != 5:
                raise ValueError("Condition latent must be CTHW or BCTHW.")
            if int(condition.shape[0]) not in {1, int(latent.shape[0])}:
                raise ValueError("Condition latent batch is incompatible with target.")
            if tuple(latent.shape[:2] + latent.shape[3:]) != tuple(
                condition.shape[:2] + condition.shape[3:]
            ):
                raise ValueError(
                    "Condition latent channels/spatial shape does not match target "
                    f"latent: {tuple(condition.shape)} vs {tuple(latent.shape)}."
                )
            if int(condition.shape[2]) > int(latent.shape[2]):
                raise ValueError("Condition latent is longer than target latent.")
            latent[:, :, : int(condition.shape[2])].copy_(
                condition.to(device=latent.device, dtype=latent.dtype)
            )
            return latent
        raise ValueError("Target latent must be CTHW or BCTHW.")


def denoising_schedule_start(num_steps: int, strength: float) -> int:
    """Return the first active step using the standard img2img truncation rule."""
    steps = max(1, int(num_steps))
    value = float(strength)
    if not 0.0 <= value <= 1.0:
        raise ValueError("denoising_strength must be between 0 and 1.")
    active_steps = min(steps, int(steps * value))
    return steps - active_steps


def blend_source_with_noise(source: Any, noise: Any, sigma: Any) -> Any:
    """Construct a rectified-flow state at the selected inference sigma."""
    return (1.0 - sigma) * source + sigma * noise


__all__ = [
    "IMAGE_TO_VIDEO",
    "INFERENCE_TASKS",
    "InferenceConditioningRequest",
    "PreparedInferenceConditioning",
    "TEXT_TO_IMAGE",
    "TEXT_TO_VIDEO",
    "VIDEO_TO_VIDEO",
    "blend_source_with_noise",
    "denoising_schedule_start",
    "normalize_inference_task",
]
