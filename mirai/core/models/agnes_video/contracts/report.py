"""Validation of independently produced Agnes release evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mirai.core.models.agnes_video.checkpoints import AGNES_SOURCE_REVISION

SCHEMA_VERSION = 1
RELEASE_ARCHITECTURE = {
    "num_layers": 40,
    "hidden_size": 5120,
    "text_in_channels": 3584,
    "video_in_channels": 192,
    "audio_in_channels": 128,
    "num_routed_experts": 3,
    "moe_top_k": 1,
}


def _mapping(value: Any, label: str, failures: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        failures.append(f"{label} must be an object.")
        return {}
    return value


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def _nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0


def _sha256(value: Any) -> bool:
    digest = str(value or "")
    return len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest.lower())


def _validate_common(payload: Any, *, contract: str) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    report = _mapping(payload, "report", failures)
    if report.get("schema_version") != SCHEMA_VERSION:
        failures.append(f"schema_version must equal {SCHEMA_VERSION}.")
    if report.get("contract") != contract:
        failures.append(f"contract must equal '{contract}'.")
    if report.get("status") != "passed":
        failures.append("status is not passed.")
    runtime = _mapping(report.get("runtime"), "runtime", failures)
    if runtime.get("family") != "agnes-video":
        failures.append("runtime.family must equal 'agnes-video'.")
    if runtime.get("pipeline_class") != "AgnesVideoPipeline":
        failures.append("runtime.pipeline_class must identify AgnesVideoPipeline.")
    if runtime.get("source_revision") != AGNES_SOURCE_REVISION:
        failures.append("runtime.source_revision does not match the pinned Agnes source.")
    if runtime.get("device_type") != "cuda" or not runtime.get("cuda_available"):
        failures.append("the release contract requires an actual CUDA runtime.")
    if not str(runtime.get("cuda_device_name", "")).strip():
        failures.append("runtime.cuda_device_name is required.")
    architecture = _mapping(runtime.get("architecture"), "runtime.architecture", failures)
    for field, expected in RELEASE_ARCHITECTURE.items():
        if architecture.get(field) != expected:
            failures.append(f"runtime.architecture.{field} must equal released value {expected}.")
    if runtime.get("checkpoint_constructed") is not True:
        failures.append("runtime must be constructed from the configured checkpoint, not an injected transformer.")
    if runtime.get("all_parameters_materialized") is not True:
        failures.append("runtime must contain materialized native checkpoint tensors.")
    if not str(runtime.get("configured_checkpoint_source", "")).strip():
        failures.append("runtime.configured_checkpoint_source is required.")
    if not isinstance(runtime.get("parameter_tensors"), int) or int(runtime.get("parameter_tensors", 0)) <= 0:
        failures.append("runtime.parameter_tensors must be positive.")
    snapshot = _mapping(report.get("snapshot"), "snapshot", failures)
    if not str(snapshot.get("source", "")).strip():
        failures.append("snapshot.source is required.")
    sha = str(snapshot.get("sha256", ""))
    if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha.lower()):
        failures.append("snapshot.sha256 must be a SHA-256 digest.")
    return failures, report


def validate_train_report(payload: Any) -> list[str]:
    failures, report = _validate_common(payload, contract="agnes_native_train_step")
    proof = _mapping(report.get("proof"), "proof", failures)
    for field in ("joint_loss", "video_loss", "audio_loss"):
        if not _positive_number(proof.get(field)):
            failures.append(f"proof.{field} must be finite and positive.")
    for field in ("adapter_gradient_tensors", "adapter_changed_tensors"):
        if not isinstance(proof.get(field), int) or int(proof.get(field, 0)) <= 0:
            failures.append(f"proof.{field} must be positive.")
    for field in ("video_target_gradient", "audio_target_gradient"):
        if not proof.get(field):
            failures.append(f"proof.{field} must be true.")
    for field in ("only_lora_trainable", "optimizer_step", "adapter_roundtrip_same_base", "source_mask_verified", "noise_mask_verified"):
        if proof.get(field) is not True:
            failures.append(f"proof.{field} must be true.")
    resume = _mapping(report.get("resume_equivalence"), "resume_equivalence", failures)
    if resume.get("steps") != 2 or resume.get("split_step") != 1:
        failures.append("resume equivalence must compare two steps split after step one.")
    for field in ("adapter_state", "optimizer_state", "scheduler_state", "python_rng", "torch_rng"):
        if resume.get(field) is not True:
            failures.append(f"resume_equivalence.{field} must be true.")
    artifacts = _mapping(report.get("artifacts"), "artifacts", failures)
    for field in ("adapter", "checkpoint"):
        path = str(artifacts.get(field, ""))
        if not path or not Path(path).is_file():
            failures.append(f"artifacts.{field} must name an existing file.")
    return failures


def validate_inference_report(payload: Any) -> list[str]:
    failures, report = _validate_common(payload, contract="agnes_native_inference")
    proof = _mapping(report.get("proof"), "proof", failures)
    if proof.get("native_inference") is not True:
        failures.append("proof.native_inference must be true.")
    if proof.get("text_encoder") != "T5GemmaEncoderModel":
        failures.append("proof.text_encoder must identify T5GemmaEncoderModel.")
    for field in ("joint_denoising", "video_decoded", "audio_decoded", "audio_muxed", "adapter_scale_effect"):
        if proof.get(field) is not True:
            failures.append(f"proof.{field} must be true.")
    media = _mapping(report.get("media"), "media", failures)
    path = str(media.get("path", ""))
    if not path or not Path(path).is_file() or Path(path).stat().st_size <= 0:
        failures.append("media.path must name a non-empty artifact.")
    for field in ("width", "height", "frames", "audio_samples", "sample_rate"):
        if not isinstance(media.get(field), int) or int(media.get(field, 0)) <= 0:
            failures.append(f"media.{field} must be positive.")
    for field in ("video_duration_seconds", "audio_duration_seconds"):
        if not _positive_number(media.get(field)):
            failures.append(f"media.{field} must be finite and positive.")
    fps = media.get("fps")
    if not _positive_number(fps):
        failures.append("media.fps must be finite and positive.")
    elif _positive_number(media.get("video_duration_seconds")) and _positive_number(media.get("audio_duration_seconds")):
        sample_rate = int(media.get("sample_rate", 0) or 0)
        tolerance = (1.0 / float(fps)) + (2048.0 / max(1, sample_rate))
        difference = abs(float(media["video_duration_seconds"]) - float(media["audio_duration_seconds"]))
        if difference > tolerance:
            failures.append("media audio/video durations exceed frame and AAC-padding tolerance.")
    if not isinstance(report.get("seed"), int):
        failures.append("seed must be an integer.")
    scale = _mapping(report.get("adapter_scale"), "adapter_scale", failures)
    if scale.get("baseline") != 0.0 or scale.get("candidate") != 1.0:
        failures.append("adapter scale proof must compare 0.0 with 1.0.")
    if not _positive_number(scale.get("max_abs_latent_difference")):
        failures.append("adapter_scale.max_abs_latent_difference must be finite and positive.")
    modality_differences = []
    for field in ("video_max_abs_latent_difference", "audio_max_abs_latent_difference"):
        value = scale.get(field)
        if not _nonnegative_number(value):
            failures.append(f"adapter_scale.{field} must be finite and nonnegative.")
        else:
            modality_differences.append(float(value))
    if len(modality_differences) == 2 and _positive_number(scale.get("max_abs_latent_difference")):
        if not math.isclose(
            float(scale["max_abs_latent_difference"]), max(modality_differences), rel_tol=0.0, abs_tol=0.0
        ):
            failures.append("adapter_scale.max_abs_latent_difference must equal the per-modality maximum.")
    paired = _mapping(report.get("paired_inputs"), "paired_inputs", failures)
    baseline_inputs = _mapping(paired.get("baseline"), "paired_inputs.baseline", failures)
    candidate_inputs = _mapping(paired.get("candidate"), "paired_inputs.candidate", failures)
    if baseline_inputs.get("seed") != report.get("seed") or candidate_inputs.get("seed") != report.get("seed"):
        failures.append("paired_inputs seeds must equal the report seed.")
    digest_fields = (
        "initial_video_sha256",
        "initial_audio_sha256",
        "conditioned_context_sha256",
        "unconditional_context_sha256",
        "sigma_schedule_sha256",
    )
    for field in digest_fields:
        if not _sha256(baseline_inputs.get(field)):
            failures.append(f"paired_inputs.baseline.{field} must be a SHA-256 digest.")
        if not _sha256(candidate_inputs.get(field)):
            failures.append(f"paired_inputs.candidate.{field} must be a SHA-256 digest.")
        if baseline_inputs.get(field) != candidate_inputs.get(field):
            failures.append(f"paired_inputs {field} must match across adapter scales.")
    return failures
