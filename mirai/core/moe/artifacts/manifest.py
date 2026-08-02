"""Download manifest checksum helpers for model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_DOWNLOAD_MANIFEST = "download_manifest.json"


def manifest_payload_sha256(payload: Mapping[str, Any]) -> str:
    payload_no_checksum = dict(payload)
    payload_no_checksum.pop("manifest_sha256", None)
    canonical = json.dumps(payload_no_checksum, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_download_manifest(
    output_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    manifest_name: str = DEFAULT_DOWNLOAD_MANIFEST,
) -> Path:
    output_path = Path(output_dir)
    payload_no_checksum = dict(payload)
    payload_no_checksum.pop("manifest_sha256", None)
    payload_with_checksum = dict(payload_no_checksum)
    payload_with_checksum["manifest_sha256"] = manifest_payload_sha256(payload_no_checksum)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / manifest_name
    manifest_path.write_text(
        json.dumps(payload_with_checksum, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def read_validated_download_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest payload must be a JSON object.")
    checksum = payload.get("manifest_sha256")
    if not isinstance(checksum, str) or not checksum:
        raise ValueError("Manifest checksum missing.")
    observed = manifest_payload_sha256(payload)
    if observed != checksum:
        raise ValueError(
            "Manifest checksum mismatch: "
            f"expected {checksum}, observed {observed}."
        )
    return payload
