"""Verification for downloaded MoE model snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.moe.artifacts.manifest import DEFAULT_DOWNLOAD_MANIFEST
from mirai.core.moe.artifacts.manifest import read_validated_download_manifest


@dataclass(frozen=True)
class SnapshotVerificationReport:
    model_dir: str
    manifest_path: str
    repo_id: str
    variant: str
    file_count: int
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "verified",
            "model_dir": self.model_dir,
            "manifest_path": self.manifest_path,
            "repo_id": self.repo_id,
            "variant": self.variant,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }


def _safe_manifest_relative_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Snapshot manifest contains a file record without a path.")
    normalized = raw_path.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Snapshot manifest contains unsafe file path: {raw_path!r}.")
    return path


def verify_downloaded_snapshot(
    model_dir: str | Path,
    *,
    expected_variant: str | None = None,
    manifest_name: str = DEFAULT_DOWNLOAD_MANIFEST,
) -> SnapshotVerificationReport:
    root = Path(model_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Model snapshot directory does not exist: {root}.")

    partials = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.part") if p.is_file())
    if partials:
        preview = ", ".join(partials[:5])
        extra = "" if len(partials) <= 5 else f", ... ({len(partials)} total)"
        raise ValueError(f"Model snapshot has incomplete partial files: {preview}{extra}.")

    manifest_path = root / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Model snapshot manifest not found: {manifest_path}. "
            "Complete the download before using this checkpoint for training."
        )
    manifest = read_validated_download_manifest(manifest_path)
    if manifest.get("status") != "downloaded":
        raise ValueError(f"Model snapshot manifest status must be 'downloaded', got {manifest.get('status')!r}.")

    variant = str(manifest.get("variant") or "")
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"Model snapshot variant mismatch: expected {expected_variant!r}, got {variant!r}."
        )

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Model snapshot manifest must include a non-empty files list.")

    total_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Model snapshot manifest files must be objects.")
        rel_path = _safe_manifest_relative_path(item.get("path"))
        expected_size = item.get("size")
        if expected_size is not None and (
            not isinstance(expected_size, int) or expected_size < 0
        ):
            raise ValueError(f"Invalid size for model snapshot file {rel_path.as_posix()!r}.")
        path = root / rel_path
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Model snapshot file missing: {path}.")
        observed_size = path.stat().st_size
        if expected_size is not None and observed_size != expected_size:
            raise ValueError(
                f"Model snapshot file size mismatch for {rel_path.as_posix()!r}: "
                f"expected {expected_size}, observed {observed_size}."
            )
        total_bytes += observed_size

    return SnapshotVerificationReport(
        model_dir=root.as_posix(),
        manifest_path=manifest_path.resolve().as_posix(),
        repo_id=str(manifest.get("repo_id") or ""),
        variant=variant,
        file_count=len(files),
        total_bytes=total_bytes,
    )
