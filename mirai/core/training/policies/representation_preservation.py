"""Frozen-base representation preservation for video adapter training.

The cosine objective follows MoE-ViE, arXiv:2608.17402, Eq. 10. Mirai obtains
the teacher from the same native pipeline with LoRA scaled to zero, avoiding a
second copy of frozen model weights while preserving exact model inputs.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from mirai.core.lineage import snapshot_id_for_path
from mirai.core.training.training_policy import PredictionContext
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


POLICY_NAME = "representation_preservation"
_ALLOWED_OPTIONS = frozenset({"enabled", "weight", "epsilon"})


def representation_cosine_loss(
    student: Any,
    teacher: Any,
    *,
    epsilon: float = 1e-8,
) -> Any:
    if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
        raise TypeError("Representation preservation requires torch tensors.")
    if student.shape != teacher.shape:
        raise ValueError(
            "Student and teacher representations must have identical shapes; "
            f"got {tuple(student.shape)} and {tuple(teacher.shape)}."
        )
    if student.ndim < 2:
        raise ValueError("Representations require a batch dimension and features.")
    if int(student.shape[0]) == 0:
        raise ValueError("Representation batches cannot be empty.")
    student_flat = student.float().flatten(1)
    teacher_flat = teacher.detach().to(device=student.device).float().flatten(1)
    return (1.0 - F.cosine_similarity(
        student_flat,
        teacher_flat,
        dim=1,
        eps=float(epsilon),
    )).mean()


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get(POLICY_NAME, {})


def _validate_config(config: Any) -> list[str]:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return []
    errors: list[str] = []
    unknown = sorted(set(options) - _ALLOWED_OPTIONS)
    if unknown:
        errors.append("unknown option(s): " + ", ".join(str(key) for key in unknown))
    weight = float(options.get("weight", 0.0))
    if not math.isfinite(weight) or weight <= 0.0:
        errors.append("weight must be finite and > 0")
    epsilon = float(options.get("epsilon", 1e-8))
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        errors.append("epsilon must be finite and > 0")
    if str(config.adapter.type).strip().lower() != "lora":
        errors.append("requires adapter.type='lora'")
    if getattr(config.adapter, "train_router", None) is not False:
        errors.append("requires adapter.train_router=false")
    if bool(getattr(config.training, "compile", False)):
        errors.append("requires training.compile=false")
    if str(getattr(config.training, "objective", "")).strip().lower() == "sharp_moe":
        errors.append("does not support training.objective='sharp_moe'")
    return errors


class RepresentationPreservationTrainingPolicy(TrainingPolicy):
    name = POLICY_NAME
    priority = 140

    def __init__(self, *, weight: float, teacher_fingerprint: str, epsilon: float) -> None:
        self.weight = float(weight)
        self.teacher_fingerprint = str(teacher_fingerprint).strip()
        self.epsilon = float(epsilon)
        self._teacher_prediction: Any | None = None
        self._active_forward = True

    def claims_prediction(self) -> bool:
        return True

    def before_forward(
        self, pipeline: Any, batch: Mapping[str, Any], *, training: bool
    ) -> None:
        _ = pipeline
        self._active_forward = bool(training) and str(
            batch.get("source_type", "")
        ).strip().lower() != "regularization"

    def predict(
        self,
        *,
        pipeline: Any,
        inputs: Any,
        predict: Callable[[Any], Any],
        training: bool,
    ) -> Any | None:
        if not training or not self._active_forward:
            self._teacher_prediction = None
            return predict(inputs)
        if self._teacher_prediction is not None:
            raise RuntimeError("Unconsumed representation teacher prediction.")
        original_scale = float(pipeline.get_lora_scale())
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            pipeline.set_lora_scale(0.0)
            with torch.no_grad():
                teacher = predict(inputs)
        finally:
            pipeline.set_lora_scale(original_scale)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        student = predict(inputs)
        self._teacher_prediction = teacher
        return student

    def prediction_auxiliary_losses(
        self, context: PredictionContext
    ) -> Mapping[str, Any]:
        if not context.training or not self._active_forward:
            self._teacher_prediction = None
            return {}
        teacher = self._teacher_prediction
        self._teacher_prediction = None
        if teacher is None:
            raise RuntimeError("Representation teacher prediction was not captured.")
        loss = representation_cosine_loss(
            context.prediction,
            teacher,
            epsilon=self.epsilon,
        )
        return {"representation_preservation": self.weight * loss}

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "weight": self.weight,
            "teacher_fingerprint": self.teacher_fingerprint,
            "epsilon": self.epsilon,
            "teacher": "same_pipeline_lora_scale_zero",
        }


@register_training_policy(POLICY_NAME, validate_config=_validate_config)
def build_representation_preservation_training_policy(
    config: Any,
) -> RepresentationPreservationTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return RepresentationPreservationTrainingPolicy(
        weight=float(options.get("weight", 0.0)),
        teacher_fingerprint=snapshot_id_for_path(config.model.path),
        epsilon=float(options.get("epsilon", 1e-8)),
    )


__all__ = [
    "POLICY_NAME",
    "RepresentationPreservationTrainingPolicy",
    "build_representation_preservation_training_policy",
    "representation_cosine_loss",
]
