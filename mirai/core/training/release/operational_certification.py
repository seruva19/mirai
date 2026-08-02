"""Operational certification builders for release evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mirai.core.persistence.checkpoints import load_checkpoint
from mirai.core.training.release.operational_certification_contract import (
    build_resume_equivalence_certification_report,
)
from mirai.core.training.release.operational_matrix_contract import (
    build_operational_certification_matrix_report,
    build_operational_matrix_entry,
)
from mirai.core.training.release.release_evidence_contract import load_release_json

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for operational certification: {exc}")


def _flatten_tensor_tree(payload: Any, *, prefix: str = "") -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        out: dict[str, torch.Tensor] = {}
        for key, value in payload.items():
            key_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_tensor_tree(value, prefix=key_prefix))
        return out
    if isinstance(payload, list):
        out: dict[str, torch.Tensor] = {}
        for idx, value in enumerate(payload):
            key_prefix = f"{prefix}[{idx}]"
            out.update(_flatten_tensor_tree(value, prefix=key_prefix))
        return out
    if torch.is_tensor(payload):
        return {prefix: payload.detach().cpu().float()}
    return {}


def build_resume_equivalence_certification(
    *,
    train_summary_path: str | Path,
    resume_summary_path: str | Path,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> dict[str, object]:
    train_summary = load_release_json(train_summary_path)
    resume_summary = load_release_json(resume_summary_path)
    full_checkpoint = Path(str(train_summary.get("last_checkpoint", ""))).resolve()
    resumed_checkpoint = Path(str(resume_summary.get("last_checkpoint", ""))).resolve()
    full_payload = load_checkpoint(full_checkpoint)
    resumed_payload = load_checkpoint(resumed_checkpoint)

    full_tensors = _flatten_tensor_tree(full_payload.get("trainer_state", {}).get("pipeline", {}))
    resumed_tensors = _flatten_tensor_tree(resumed_payload.get("trainer_state", {}).get("pipeline", {}))
    full_keys = set(full_tensors)
    resumed_keys = set(resumed_tensors)
    shared_keys = sorted(full_keys & resumed_keys)
    missing_keys = sorted(full_keys - resumed_keys)
    extra_keys = sorted(resumed_keys - full_keys)
    mismatched_keys: list[str] = []
    max_abs_delta = 0.0
    for key in shared_keys:
        left = full_tensors[key]
        right = resumed_tensors[key]
        max_abs_delta = max(max_abs_delta, float(torch.max(torch.abs(left - right)).item()))
        if not torch.allclose(left, right, atol=float(atol), rtol=float(rtol)):
            mismatched_keys.append(str(key))
    global_step_match = int(full_payload.get("global_step", 0)) == int(resumed_payload.get("global_step", 0))
    status = (
        "ok"
        if global_step_match and not missing_keys and not extra_keys and not mismatched_keys and shared_keys
        else "failed"
    )
    return build_resume_equivalence_certification_report(
        status=status,
        train_summary=str(Path(train_summary_path).resolve()),
        resume_summary=str(Path(resume_summary_path).resolve()),
        full_checkpoint=str(full_checkpoint),
        resumed_checkpoint=str(resumed_checkpoint),
        global_step_match=global_step_match,
        compared_tensor_count=len(shared_keys),
        missing_tensor_keys=missing_keys,
        extra_tensor_keys=extra_keys,
        mismatched_tensor_keys=mismatched_keys,
        max_abs_delta=float(max_abs_delta),
        tolerated_atol=float(atol),
        tolerated_rtol=float(rtol),
    )


def build_operational_certification_matrix(
    *,
    cache_summary_path: str | Path,
    cache_summary_payload: dict[str, Any],
    train_summary_path: str | Path,
    train_summary_payload: dict[str, Any],
    eval_report_path: str | Path,
    eval_report_payload: dict[str, Any],
    partial_summary_path: str | Path,
    partial_summary_payload: dict[str, Any],
    resume_summary_path: str | Path,
    resume_certification_report_path: str | Path,
    resume_certification_payload: dict[str, Any],
    export_certification_report_path: str | Path,
    export_certification_payload: dict[str, Any],
    strict_train_summary_path: str | Path,
    strict_train_summary_payload: dict[str, Any],
    strict_partial_summary_path: str | Path,
    strict_partial_summary_payload: dict[str, Any],
    strict_resume_summary_path: str | Path,
    strict_resume_certification_report_path: str | Path,
    strict_resume_certification_payload: dict[str, Any],
    native_smoke_report_path: str | Path,
    native_smoke_payload: dict[str, Any],
    verify_release_report_path: str | Path,
    verification_payload: dict[str, Any],
    split_step: int,
) -> dict[str, object]:
    resume_summary = load_release_json(resume_summary_path)
    strict_resume_summary = load_release_json(strict_resume_summary_path)
    resume_warnings = list(resume_summary.get("resume_validation_warnings", []) or [])
    strict_resume_warnings = list(strict_resume_summary.get("resume_validation_warnings", []) or [])
    allowed_resume_warning = bool(resume_warnings) and all(
        "different config snapshot id" in str(warning)
        and "resume remains allowed" in str(warning)
        for warning in resume_warnings
    )
    allowed_strict_resume_warning = bool(strict_resume_warnings) and all(
        "different config snapshot id" in str(warning)
        and "resume remains allowed" in str(warning)
        for warning in strict_resume_warnings
    )
    entries = [
        build_operational_matrix_entry(
            category="cache_build",
            scenario="release_cache_build",
            status=(
                "ok"
                if str(cache_summary_payload.get("status", "")).strip().lower() == "cached"
                and int(cache_summary_payload.get("num_records", 0)) > 0
                and bool(
                    dict(cache_summary_payload.get("indexed_cache", {})).get("enabled", False)
                )
                else "failed"
            ),
            report_path=str(Path(cache_summary_path).resolve()),
            summary={
                "num_records": int(cache_summary_payload.get("num_records", 0)),
                "dataset_snapshot_id": str(
                    cache_summary_payload.get("dataset_snapshot_id", "")
                ),
                "indexed_cache_enabled": bool(
                    dict(cache_summary_payload.get("indexed_cache", {})).get("enabled", False)
                ),
            },
        ),
        build_operational_matrix_entry(
            category="full_run",
            scenario=f"release_full_run_max_steps_{int(train_summary_payload.get('global_step', 0))}",
            status=(
                "ok"
                if str(train_summary_payload.get("status", "")).strip().lower()
                in {"completed", "early_stopped"}
                else "failed"
            ),
            report_path=str(Path(train_summary_path).resolve()),
            summary={
                "global_step": int(train_summary_payload.get("global_step", 0)),
                "last_checkpoint": str(train_summary_payload.get("last_checkpoint", "")),
                "resumed": bool(train_summary_payload.get("resumed", False)),
            },
        ),
        build_operational_matrix_entry(
            category="interrupted_partial_run",
            scenario=f"release_partial_run_split_step_{int(split_step)}",
            status=(
                "ok"
                if str(partial_summary_payload.get("status", "")).strip().lower()
                in {"completed", "early_stopped"}
                and int(partial_summary_payload.get("global_step", 0)) == int(split_step)
                else "failed"
            ),
            report_path=str(Path(partial_summary_path).resolve()),
            summary={
                "global_step": int(partial_summary_payload.get("global_step", 0)),
                "expected_split_step": int(split_step),
                "last_checkpoint": str(partial_summary_payload.get("last_checkpoint", "")),
            },
        ),
        build_operational_matrix_entry(
            category="eval_gate",
            scenario="release_eval_gate",
            status=(
                "ok"
                if str(eval_report_payload.get("status", "")).strip().lower() == "ok"
                and bool(eval_report_payload.get("absolute_thresholds_passed", False))
                and bool(eval_report_payload.get("frame_metrics_complete", False))
                else "failed"
            ),
            report_path=str(Path(eval_report_path).resolve()),
            summary={
                "absolute_thresholds_passed": bool(
                    eval_report_payload.get("absolute_thresholds_passed", False)
                ),
                "frame_metrics_complete": bool(
                    eval_report_payload.get("frame_metrics_complete", False)
                ),
                "num_samples": int(eval_report_payload.get("num_samples", 0)),
            },
        ),
        build_operational_matrix_entry(
            category="resume_equivalence",
            scenario=f"release_resume_split_step_{int(split_step)}",
            status=str(resume_certification_payload.get("status", "failed")),
            report_path=str(Path(resume_certification_report_path).resolve()),
            summary={
                "compared_tensor_count": int(
                    resume_certification_payload.get("compared_tensor_count", 0)
                ),
                "max_abs_delta": float(resume_certification_payload.get("max_abs_delta", 0.0)),
                "global_step_match": bool(
                    resume_certification_payload.get("global_step_match", False)
                ),
                "persistent_workers": True,
            },
        ),
        build_operational_matrix_entry(
            category="resume_reentry",
            scenario=f"release_resume_reentry_split_step_{int(split_step)}",
            status=(
                "ok"
                if bool(resume_summary.get("resumed", False))
                and int(resume_summary.get("resume_checkpoint_step", 0)) > 0
                and (
                    bool(resume_summary.get("resume_lineage_verified", False))
                    or allowed_resume_warning
                )
                else "failed"
            ),
            report_path=str(Path(resume_summary_path).resolve()),
            summary={
                "resume_checkpoint_step": int(
                    resume_summary.get("resume_checkpoint_step", 0)
                ),
                "resume_lineage_verified": bool(
                    resume_summary.get("resume_lineage_verified", False)
                ),
                "resume_warning_count": len(resume_warnings),
                "persistent_workers": True,
            },
        ),
        build_operational_matrix_entry(
            category="export_import",
            scenario="release_adapter_export_contract",
            status=str(export_certification_payload.get("status", "failed")),
            report_path=str(Path(export_certification_report_path).resolve()),
            summary={
                "entry_count": int(export_certification_payload.get("entry_count", 0)),
                "failing_entries": list(export_certification_payload.get("failing_entries", []) or []),
            },
        ),
        build_operational_matrix_entry(
            category="strict_full_run",
            scenario=f"strict_full_run_max_steps_{int(strict_train_summary_payload.get('global_step', 0))}",
            status=(
                "ok"
                if str(strict_train_summary_payload.get("status", "")).strip().lower()
                in {"completed", "early_stopped"}
                else "failed"
            ),
            report_path=str(Path(strict_train_summary_path).resolve()),
            summary={
                "global_step": int(strict_train_summary_payload.get("global_step", 0)),
                "last_checkpoint": str(strict_train_summary_payload.get("last_checkpoint", "")),
                "persistent_workers": False,
            },
        ),
        build_operational_matrix_entry(
            category="strict_interrupted_partial_run",
            scenario=f"strict_partial_run_split_step_{int(split_step)}",
            status=(
                "ok"
                if str(strict_partial_summary_payload.get("status", "")).strip().lower()
                in {"completed", "early_stopped"}
                and int(strict_partial_summary_payload.get("global_step", 0)) == int(split_step)
                else "failed"
            ),
            report_path=str(Path(strict_partial_summary_path).resolve()),
            summary={
                "global_step": int(strict_partial_summary_payload.get("global_step", 0)),
                "expected_split_step": int(split_step),
                "last_checkpoint": str(strict_partial_summary_payload.get("last_checkpoint", "")),
                "persistent_workers": False,
            },
        ),
        build_operational_matrix_entry(
            category="strict_resume_equivalence",
            scenario=f"strict_resume_split_step_{int(split_step)}",
            status=str(strict_resume_certification_payload.get("status", "failed")),
            report_path=str(Path(strict_resume_certification_report_path).resolve()),
            summary={
                "compared_tensor_count": int(
                    strict_resume_certification_payload.get("compared_tensor_count", 0)
                ),
                "max_abs_delta": float(
                    strict_resume_certification_payload.get("max_abs_delta", 0.0)
                ),
                "global_step_match": bool(
                    strict_resume_certification_payload.get("global_step_match", False)
                ),
                "persistent_workers": False,
            },
        ),
        build_operational_matrix_entry(
            category="strict_resume_reentry",
            scenario=f"strict_resume_reentry_split_step_{int(split_step)}",
            status=(
                "ok"
                if bool(strict_resume_summary.get("resumed", False))
                and int(strict_resume_summary.get("resume_checkpoint_step", 0)) > 0
                and (
                    bool(strict_resume_summary.get("resume_lineage_verified", False))
                    or allowed_strict_resume_warning
                )
                else "failed"
            ),
            report_path=str(Path(strict_resume_summary_path).resolve()),
            summary={
                "resume_checkpoint_step": int(
                    strict_resume_summary.get("resume_checkpoint_step", 0)
                ),
                "resume_lineage_verified": bool(
                    strict_resume_summary.get("resume_lineage_verified", False)
                ),
                "resume_warning_count": len(strict_resume_warnings),
                "persistent_workers": False,
            },
        ),
        build_operational_matrix_entry(
            category="native_smoke",
            scenario="native_supported_contract",
            status=(
                "ok"
                if str(native_smoke_payload.get("status", "")).strip().lower() == "ok"
                else "failed"
            ),
            report_path=str(Path(native_smoke_report_path).resolve()),
            summary={
                "step_count": len(list(native_smoke_payload.get("steps", []) or [])),
                "checkpoint_path": str(native_smoke_payload.get("checkpoint_path", "")),
                "infer_path": str(native_smoke_payload.get("infer_path", "")),
            },
        ),
        build_operational_matrix_entry(
            category="release_verifier",
            scenario="strict_release_bundle",
            status=(
                "ok"
                if str(verification_payload.get("status", "")).strip().lower() == "passed"
                else "failed"
            ),
            report_path=str(Path(verify_release_report_path).resolve()),
            summary={
                "failure_count": len(list(verification_payload.get("failures", []) or [])),
                "train_summary": str(verification_payload.get("train_summary", "")),
            },
        ),
    ]
    failing_entries = [
        f"{entry['category']}:{entry['scenario']}"
        for entry in entries
        if str(entry.get("status", "")).strip().lower() != "ok"
    ]
    return build_operational_certification_matrix_report(
        status="ok" if not failing_entries else "failed",
        entry_count=len(entries),
        failing_entries=failing_entries,
        entries=entries,
    )
