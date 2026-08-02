"""Output-directory ownership and manifest-retention helpers for training sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_MANAGED_OUTPUT_NAMES = (
    "adapter.pt",
    "adapter.safetensors",
    "adapter_ema.safetensors",
    "adapter_lycoris.safetensors",
    "events.jsonl",
    "metrics.jsonl",
    "run_manifest.json",
)
_MANAGED_OUTPUT_DIRS = (
    "checkpoints",
    "run_manifests",
    "samples",
    "tb_logs",
)


@dataclass(frozen=True)
class SessionOutputPlan:
    output_root: Path
    run_id: str
    manifest_path: Path
    attempt_manifest_path: Path
    ckpt_dir: Path


def build_training_run_id() -> str:
    return f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


def prepare_session_output_plan(
    *,
    config: Any,
    output_dir: str | Path,
    resume_path: str = "",
) -> SessionOutputPlan:
    output_root = Path(output_dir)
    _validate_output_location(
        output_root=output_root,
        dataset_path=getattr(config.dataset, "path", ""),
        model_path=getattr(config.model, "path", ""),
        cache_path=getattr(config.dataset, "cache_path", ""),
    )
    _validate_output_reuse(
        output_root=output_root,
        resume_path=resume_path,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = build_training_run_id()
    ckpt_dir = output_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    attempt_manifest_dir = output_root / "run_manifests"
    attempt_manifest_dir.mkdir(parents=True, exist_ok=True)
    return SessionOutputPlan(
        output_root=output_root,
        run_id=run_id,
        manifest_path=output_root / "run_manifest.json",
        attempt_manifest_path=attempt_manifest_dir / f"{run_id}.json",
        ckpt_dir=ckpt_dir,
    )


def _validate_output_location(
    *,
    output_root: Path,
    dataset_path: str | Path,
    model_path: str | Path,
    cache_path: str | Path,
) -> None:
    resolved_output = output_root.resolve()
    if resolved_output.exists() and not resolved_output.is_dir():
        raise ValueError(
            f"logging.output_dir must resolve to a directory, got file '{resolved_output}'."
        )

    lineage_paths = (
        ("dataset.path", Path(dataset_path)),
        ("model.path", Path(model_path)),
    )
    for label, root in lineage_paths:
        resolved_root = root.resolve()
        if _is_same_or_nested(resolved_output, resolved_root):
            raise ValueError(
                f"logging.output_dir='{resolved_output}' must not be inside {label}='{resolved_root}'. "
                "Training outputs would mutate lineage-bearing source assets."
            )
    resolved_cache = Path(cache_path).resolve()
    if resolved_output == resolved_cache:
        raise ValueError(
            f"logging.output_dir='{resolved_output}' must not equal dataset.cache_path='{resolved_cache}'."
        )


def _validate_output_reuse(*, output_root: Path, resume_path: str) -> None:
    managed_paths = _existing_managed_paths(output_root)
    if not managed_paths or str(resume_path).strip():
        return
    joined = ", ".join(managed_paths)
    raise ValueError(
        f"logging.output_dir='{output_root}' already contains trainer artifacts ({joined}). "
        "Use a fresh output directory for a new run or pass --resume to continue explicitly."
    )


def _existing_managed_paths(output_root: Path) -> list[str]:
    if not output_root.exists():
        return []
    found: list[str] = []
    for name in _MANAGED_OUTPUT_NAMES:
        if (output_root / name).exists():
            found.append(name)
    for name in _MANAGED_OUTPUT_DIRS:
        if (output_root / name).exists():
            found.append(f"{name}/")
    return found


def _is_same_or_nested(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
