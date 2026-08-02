"""Versioned release-evidence contracts and verification helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.training.release.operational_certification_contract import (
    validate_resume_equivalence_certification_report,
)
from mirai.core.training.release.export_certification_contract import (
    validate_export_certification_report,
)
from mirai.core.training.release.operational_matrix_contract import (
    validate_operational_certification_matrix_report,
)
from mirai.core.training.release.promotion_gate_contract import validate_promotion_gate_report
from mirai.core.models.providers import (
    get_model_family_provider,
    release_supported_model_types,
)


RELEASE_EVIDENCE_SCHEMA_VERSION = 1
ALLOWED_RESUME_RELEASE_WARNING_MARKERS = (
    "different config snapshot id",
    "resume remains allowed",
)
REQUIRED_RESOURCE_TELEMETRY_FIELDS = frozenset(
    {
        "process_rss_mb",
        "cuda_available",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
    }
)
RELEASE_RUNTIME_IDENTITY_FIELDS = (
    "model_type",
    "strict_native_assets",
    "model_variant",
    "denoiser_subfolder",
)


def checkpoint_from_train_summary(
    summary: dict[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> Path:
    """Resolve and validate the checkpoint declared by a training summary."""

    raw_path = str(summary.get("last_checkpoint", "")).strip()
    if not raw_path:
        raise RuntimeError(f"{label} did not report last_checkpoint")
    checkpoint = Path(raw_path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = repo_root / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise RuntimeError(
            f"{label} reported a checkpoint that does not exist: {checkpoint}"
        )
    return checkpoint


@dataclass(frozen=True)
class ReleaseVerificationReport:
    status: str
    train_summary: str
    eval_report: str
    resume_summary: str
    native_smoke_report: str
    failures: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RELEASE_EVIDENCE_SCHEMA_VERSION,
            "status": str(self.status),
            "train_summary": str(self.train_summary),
            "eval_report": str(self.eval_report),
            "resume_summary": str(self.resume_summary),
            "native_smoke_report": str(self.native_smoke_report),
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class ReleaseBundleVerificationReport:
    status: str
    bundle_path: str
    failures: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RELEASE_EVIDENCE_SCHEMA_VERSION,
            "status": str(self.status),
            "bundle_path": str(self.bundle_path),
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class ReleaseEvidenceBundle:
    status: str
    config: str
    evidence_root: str
    cache_summary: str
    train_summary: str
    eval_report: str
    partial_summary: str
    resume_summary: str
    resume_certification_report: str
    export_certification_report: str
    operational_matrix_report: str
    promotion_gate_report: str
    native_smoke_report: str
    verify_release_report: str
    split_step: int
    commands: list[dict[str, Any]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RELEASE_EVIDENCE_SCHEMA_VERSION,
            "status": str(self.status),
            "config": str(self.config),
            "evidence_root": str(self.evidence_root),
            "cache_summary": str(self.cache_summary),
            "train_summary": str(self.train_summary),
            "eval_report": str(self.eval_report),
            "partial_summary": str(self.partial_summary),
            "resume_summary": str(self.resume_summary),
            "resume_certification_report": str(self.resume_certification_report),
            "export_certification_report": str(self.export_certification_report),
            "operational_matrix_report": str(self.operational_matrix_report),
            "promotion_gate_report": str(self.promotion_gate_report),
            "native_smoke_report": str(self.native_smoke_report),
            "verify_release_report": str(self.verify_release_report),
            "split_step": int(self.split_step),
            "commands": list(self.commands),
        }


def build_release_verification_report(**kwargs: Any) -> dict[str, object]:
    return ReleaseVerificationReport(**kwargs).to_dict()


def build_release_bundle_verification_report(**kwargs: Any) -> dict[str, object]:
    return ReleaseBundleVerificationReport(**kwargs).to_dict()


def build_release_evidence_bundle(**kwargs: Any) -> dict[str, object]:
    return ReleaseEvidenceBundle(**kwargs).to_dict()


def load_release_json(path: str | Path) -> dict[str, Any]:
    in_path = Path(path)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {in_path}.")
    return payload


def append_if_missing_file(
    failures: list[str],
    *,
    payload: dict[str, Any],
    key: str,
    label: str,
) -> None:
    raw = str(payload.get(key, "")).strip()
    if not raw:
        failures.append(f"{label}: missing '{key}' path.")
        return
    if not Path(raw).exists():
        failures.append(f"{label}: path '{raw}' does not exist.")


def load_embedded_or_path_json(
    payload: dict[str, Any],
    *,
    key: str,
    label: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    embedded = payload.get(key)
    if isinstance(embedded, dict):
        return embedded, []
    raw_path = str(payload.get(f"{key}_path", "")).strip()
    if not raw_path:
        return None, [f"{label}: provide either '{key}' object or '{key}_path'."]
    path = Path(raw_path)
    if not path.exists():
        return None, [f"{label}: path '{raw_path}' does not exist."]
    try:
        return load_release_json(path), []
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"{label}: failed to load '{raw_path}': {exc}"]


def validate_train_summary(
    payload: dict[str, Any],
    *,
    label: str,
    require_production_runtime: bool = False,
) -> list[str]:
    failures: list[str] = []
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"completed", "early_stopped"}:
        failures.append(f"{label}: unexpected status '{status}'.")
    if str(payload.get("cache_data_access_mode", "")).strip() != "indexed_safetensors":
        failures.append(f"{label}: cache_data_access_mode must be 'indexed_safetensors'.")
    if not bool(payload.get("indexed_cache_enabled", False)):
        failures.append(f"{label}: indexed_cache_enabled must be true.")
    sampling_mode = str(payload.get("training_sampling_mode", "")).strip()
    if sampling_mode not in {"deterministic_epoch_shuffle", "temporal_resampling"}:
        failures.append(f"{label}: unsupported training_sampling_mode '{sampling_mode}'.")
    if require_production_runtime:
        failures.extend(
            validate_production_runtime_policy(
                payload.get("runtime_policy", {}),
                label=f"{label}.runtime_policy",
            )
        )
        failures.extend(
            validate_resource_telemetry(
                payload.get("resource_telemetry"),
                label=f"{label}.resource_telemetry",
            )
        )
    append_if_missing_file(failures, payload=payload, key="last_checkpoint", label=label)
    append_if_missing_file(failures, payload=payload, key="adapter_path", label=label)
    return failures


def validate_resource_telemetry(payload: Any, *, label: str) -> list[str]:
    failures: list[str] = []
    if not isinstance(payload, dict):
        return [f"{label}: must be an object."]
    missing = sorted(REQUIRED_RESOURCE_TELEMETRY_FIELDS.difference(payload))
    if missing:
        failures.append(f"{label}: missing fields {', '.join(missing)}.")
    if "cuda_available" in payload and not isinstance(payload.get("cuda_available"), bool):
        failures.append(f"{label}: cuda_available must be a boolean.")
    for key in sorted(REQUIRED_RESOURCE_TELEMETRY_FIELDS - {"cuda_available"}):
        if key not in payload:
            continue
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            failures.append(f"{label}: {key} must be numeric.")
            continue
        if value < 0.0:
            failures.append(f"{label}: {key} must be >= 0.")
    return failures


def validate_resume_summary(payload: dict[str, Any]) -> list[str]:
    failures = validate_train_summary(
        payload,
        label="resume_summary",
        require_production_runtime=True,
    )
    if not bool(payload.get("resumed", False)):
        failures.append("resume_summary: resumed must be true.")
    if not str(payload.get("resume_checkpoint_path", "")).strip():
        failures.append("resume_summary: resume_checkpoint_path is required.")
    if int(payload.get("resume_checkpoint_step", 0)) <= 0:
        failures.append("resume_summary: resume_checkpoint_step must be > 0.")
    warnings = payload.get("resume_validation_warnings", [])
    warning_list = [str(item) for item in list(warnings or [])]
    allowed_config_snapshot_warning = bool(warning_list) and all(
        all(marker in warning for marker in ALLOWED_RESUME_RELEASE_WARNING_MARKERS)
        for warning in warning_list
    )
    if not bool(payload.get("resume_lineage_verified", False)) and not allowed_config_snapshot_warning:
        failures.append("resume_summary: resume_lineage_verified must be true.")
    if warning_list and not allowed_config_snapshot_warning:
        failures.append("resume_summary: resume_validation_warnings must be empty.")
    return failures


def validate_eval_report(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if str(payload.get("status", "")).strip().lower() != "ok":
        failures.append("eval_report: status must be 'ok'.")
    if not bool(payload.get("absolute_thresholds_passed", False)):
        failures.append("eval_report: absolute_thresholds_passed must be true.")
    if not bool(payload.get("frame_metrics_complete", False)):
        failures.append("eval_report: frame_metrics_complete must be true.")
    if list(payload.get("frame_metric_decode_failures", []) or []):
        failures.append("eval_report: frame_metric_decode_failures must be empty.")
    failures.extend(
        validate_production_runtime_policy(
            payload.get("runtime_policy", {}),
            label="eval_report.runtime_policy",
        )
    )
    return failures


def validate_native_smoke_report(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if str(payload.get("status", "")).strip().lower() != "ok":
        failures.append("native_smoke_report: status must be 'ok'.")
    steps = payload.get("steps", [])
    if not isinstance(steps, list) or not steps:
        failures.append("native_smoke_report: steps must be a non-empty list.")
    else:
        bad_steps = [
            idx
            for idx, step in enumerate(steps)
            if not isinstance(step, dict) or int(step.get("returncode", 1)) != 0
        ]
        if bad_steps:
            failures.append(
                "native_smoke_report: failing steps at indices "
                + ", ".join(str(idx) for idx in bad_steps)
                + "."
            )
    for key in ("checkpoint_path", "eval_path", "infer_path"):
        append_if_missing_file(
            failures,
            payload=payload,
            key=key,
            label="native_smoke_report",
        )
    failures.extend(
        validate_production_runtime_policy(
            payload.get("runtime_policy", {}),
            label="native_smoke_report.runtime_policy",
        )
    )
    runtime_policy = payload.get("runtime_policy", {})
    runtime_policy_dict = runtime_policy if isinstance(runtime_policy, dict) else {}
    model_type = str(runtime_policy_dict.get("model_type", "")).strip().lower()
    provider = get_model_family_provider(model_type) if model_type else None
    if provider is not None:
        failures.extend(
            provider.validate_release_native_smoke_evidence(
                payload,
                runtime_policy=runtime_policy_dict,
                label="native_smoke_report",
            )
        )
    return failures


def validate_production_runtime_policy(policy: Any, *, label: str) -> list[str]:
    failures: list[str] = []
    if not isinstance(policy, dict):
        return [f"{label}: runtime_policy must be an object."]
    model_type = str(policy.get("model_type", "")).strip().lower()
    provider = get_model_family_provider(model_type) if model_type else None
    if provider is None or not provider.is_release_supported():
        supported = ", ".join(release_supported_model_types()) or "(none registered)"
        failures.append(f"{label}: model_type must be one of {supported}.")
    elif not provider.is_release_eligible():
        failures.append(
            f"{label}: model_type '{model_type}' is a synthetic validation fixture "
            "and is not production release eligible."
        )
    if not bool(policy.get("strict_native_assets", False)):
        failures.append(f"{label}: strict_native_assets must be true.")
    return failures


def validate_cache_summary(payload: dict[str, Any], *, label: str) -> list[str]:
    failures: list[str] = []
    if str(payload.get("status", "")).strip().lower() != "cached":
        failures.append(f"{label}: status must be 'cached'.")
    if int(payload.get("num_records", 0)) <= 0:
        failures.append(f"{label}: num_records must be > 0.")
    indexed_cache = payload.get("indexed_cache", {})
    if not isinstance(indexed_cache, dict) or not bool(indexed_cache.get("enabled", False)):
        failures.append(f"{label}: indexed_cache.enabled must be true.")
    failures.extend(
        validate_production_runtime_policy(
            payload.get("runtime_policy", {}),
            label=f"{label}.runtime_policy",
        )
    )
    return failures


def validate_release_runtime_policy_consistency(
    artifacts: dict[str, dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    observed: dict[str, tuple[str, str]] = {}
    for label, artifact in artifacts.items():
        policy = artifact.get("runtime_policy", {})
        if not isinstance(policy, dict):
            continue
        for field in RELEASE_RUNTIME_IDENTITY_FIELDS:
            value = _normalized_runtime_identity_value(policy, field)
            if field not in observed:
                observed[field] = (label, value)
                continue
            first_label, first_value = observed[field]
            if value != first_value:
                failures.append(
                    "release_evidence_bundle.runtime_policy: "
                    f"{label}.{field}='{value}' does not match "
                    f"{first_label}.{field}='{first_value}'."
                )
    return failures


def _normalized_runtime_identity_value(policy: dict[str, Any], field: str) -> str:
    if field == "model_type":
        return str(policy.get(field, "")).strip().lower()
    if field == "strict_native_assets":
        return str(bool(policy.get(field, False))).lower()
    if field == "denoiser_subfolder":
        return str(policy.get(field, "transformer") or "transformer").strip()
    return str(policy.get(field, "") or "").strip()


def validate_release_evidence_bundle(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    evidence_root = str(payload.get("evidence_root", "")).strip()
    if not evidence_root:
        failures.append("release_evidence_bundle: evidence_root is required.")
    elif not Path(evidence_root).exists():
        failures.append(f"release_evidence_bundle: evidence_root '{evidence_root}' does not exist.")
    if int(payload.get("split_step", 0)) <= 0:
        failures.append("release_evidence_bundle: split_step must be > 0.")
    commands = payload.get("commands", [])
    if not isinstance(commands, list) or not commands:
        failures.append("release_evidence_bundle: commands must be a non-empty list.")

    path_keys = (
        "cache_summary",
        "train_summary",
        "eval_report",
        "partial_summary",
        "resume_summary",
        "resume_certification_report",
        "export_certification_report",
        "operational_matrix_report",
        "promotion_gate_report",
        "native_smoke_report",
        "verify_release_report",
    )
    for key in path_keys:
        append_if_missing_file(failures, payload=payload, key=key, label="release_evidence_bundle")

    if failures:
        return failures

    cache_summary = load_release_json(str(payload["cache_summary"]))
    train_summary = load_release_json(str(payload["train_summary"]))
    eval_report = load_release_json(str(payload["eval_report"]))
    partial_summary = load_release_json(str(payload["partial_summary"]))
    resume_summary = load_release_json(str(payload["resume_summary"]))
    resume_certification_report = load_release_json(str(payload["resume_certification_report"]))
    export_certification_report = load_release_json(str(payload["export_certification_report"]))
    operational_matrix_report = load_release_json(str(payload["operational_matrix_report"]))
    promotion_gate_report = load_release_json(str(payload["promotion_gate_report"]))
    native_smoke_report = load_release_json(str(payload["native_smoke_report"]))
    verify_release_report = load_release_json(str(payload["verify_release_report"]))

    failures.extend(
        validate_cache_summary(
            cache_summary,
            label="release_evidence_bundle.cache_summary",
        )
    )
    failures.extend(
        validate_train_summary(
            train_summary,
            label="release_evidence_bundle.train_summary",
            require_production_runtime=True,
        )
    )
    failures.extend(validate_eval_report(eval_report))
    failures.extend(
        validate_train_summary(
            partial_summary,
            label="release_evidence_bundle.partial_summary",
            require_production_runtime=True,
        )
    )
    failures.extend(validate_resume_summary(resume_summary))
    failures.extend(validate_resume_equivalence_certification_report(resume_certification_report))
    failures.extend(validate_export_certification_report(export_certification_report))
    if str(export_certification_report.get("status", "")).strip().lower() != "ok":
        failures.append("release_evidence_bundle.export_certification_report: status must be 'ok'.")
    failures.extend(validate_operational_certification_matrix_report(operational_matrix_report))
    failures.extend(validate_promotion_gate_report(promotion_gate_report))
    failures.extend(validate_native_smoke_report(native_smoke_report))
    failures.extend(
        validate_release_runtime_policy_consistency(
            {
                "cache_summary": cache_summary,
                "train_summary": train_summary,
                "eval_report": eval_report,
                "partial_summary": partial_summary,
                "resume_summary": resume_summary,
                "native_smoke_report": native_smoke_report,
            }
        )
    )

    verify_status = str(verify_release_report.get("status", "")).strip().lower()
    if verify_status != "passed":
        failures.append("release_evidence_bundle.verify_release_report: status must be 'passed'.")
    if str(payload.get("status", "")).strip().lower() == "passed" and failures:
        failures.append("release_evidence_bundle: bundle status is 'passed' but referenced evidence is not fully green.")
    if str(payload.get("status", "")).strip().lower() == "passed":
        if str(operational_matrix_report.get("status", "")).strip().lower() != "ok":
            failures.append("release_evidence_bundle: passed bundle requires operational matrix status 'ok'.")
        if str(promotion_gate_report.get("status", "")).strip().lower() != "ok":
            failures.append("release_evidence_bundle: passed bundle requires promotion gate status 'ok'.")
    return failures
