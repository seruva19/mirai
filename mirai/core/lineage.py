"""Snapshot and lineage helpers shared across training artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from urllib.parse import quote


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    h = hashlib.sha256()
    with resolved.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class SnapshotDescriptor:
    snapshot_id: str
    resolved_path: str
    source_kind: str
    source_manifest_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "snapshot_id": str(self.snapshot_id),
            "resolved_path": str(self.resolved_path),
            "source_kind": str(self.source_kind),
            "source_manifest_path": str(self.source_manifest_path),
        }


def snapshot_descriptor_for_path(
    path: str | Path,
    *,
    manifest_candidates: tuple[str, ...] = ("registration.json", "download_manifest.json"),
) -> SnapshotDescriptor:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return SnapshotDescriptor(
            snapshot_id=f"missing:{resolved.as_posix()}",
            resolved_path=resolved.as_posix(),
            source_kind="missing",
            source_manifest_path="",
        )
    if resolved.is_file():
        return SnapshotDescriptor(
            snapshot_id=f"file-sha256:{sha256_file(resolved)}",
            resolved_path=resolved.as_posix(),
            source_kind="file",
            source_manifest_path="",
        )

    for manifest_name in manifest_candidates:
        manifest_path = resolved / manifest_name
        if manifest_path.exists() and manifest_path.is_file():
            return SnapshotDescriptor(
                snapshot_id=f"{manifest_name}:sha256:{sha256_file(manifest_path)}",
                resolved_path=resolved.as_posix(),
                source_kind="manifest",
                source_manifest_path=manifest_path.resolve().as_posix(),
            )

    h = hashlib.sha256()
    for file_path in sorted([p for p in resolved.rglob("*") if p.is_file()]):
        rel = file_path.relative_to(resolved).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(file_path).encode("ascii"))
        h.update(b"\0")
    return SnapshotDescriptor(
        snapshot_id=f"tree-sha256:{h.hexdigest()}",
        resolved_path=resolved.as_posix(),
        source_kind="tree",
        source_manifest_path="",
    )


def normalize_snapshot_component(component: str) -> str:
    text = str(component).replace("\\", "/").strip().strip("/")
    if any(ch in text for ch in ("\0", "\n", "\r", "\t")):
        raise ValueError("Snapshot component identity contains control characters.")
    return text


def snapshot_component_id(
    component: str,
    *,
    component_label: str = "component",
) -> str:
    normalized = normalize_snapshot_component(component)
    if not normalized:
        return ""
    label = normalize_snapshot_component(component_label)
    if not label:
        raise ValueError("Snapshot component label cannot be empty.")
    encoded_label = quote(label, safe="._-")
    encoded_component = quote(normalized, safe="._-/")
    return f"{encoded_label}:{encoded_component}"


def bind_snapshot_component(
    descriptor: SnapshotDescriptor,
    *,
    component: str,
    component_label: str = "component",
) -> SnapshotDescriptor:
    component_id = snapshot_component_id(
        component,
        component_label=component_label,
    )
    if not component_id:
        return descriptor
    label = normalize_snapshot_component(component_label)
    return SnapshotDescriptor(
        snapshot_id=f"{descriptor.snapshot_id}|{component_id}",
        resolved_path=descriptor.resolved_path,
        source_kind=f"{descriptor.source_kind}+{label}",
        source_manifest_path=descriptor.source_manifest_path,
    )


def snapshot_id_for_path(
    path: str | Path,
    *,
    manifest_candidates: tuple[str, ...] = ("registration.json", "download_manifest.json"),
) -> str:
    return snapshot_descriptor_for_path(
        path,
        manifest_candidates=manifest_candidates,
    ).snapshot_id
