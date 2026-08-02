"""Dataset registration helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mirai.core.dataset.split import build_split_assignments

_MEDIA_SUFFIXES = {
    ".pt",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
}
DEFAULT_REGISTRATION_FILENAME = "registration.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def resolve_registration_path(
    dataset_path: str | Path,
    registration_path: str | Path | None = None,
) -> Path:
    if registration_path is not None and str(registration_path).strip():
        return Path(registration_path)
    return Path(dataset_path) / DEFAULT_REGISTRATION_FILENAME


def _require_compliance_metadata(
    *,
    dataset_dir: Path,
    require_provenance: bool,
    require_rights_attestation: bool,
) -> dict[str, Any]:
    meta_path = dataset_dir / "dataset_metadata.json"
    if not meta_path.exists():
        raise ValueError(
            "Dataset compliance gate is enabled but dataset_metadata.json is missing."
        )
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset_metadata.json must be a JSON object.")
    if require_provenance and not str(payload.get("provenance", "")).strip():
        raise ValueError("Dataset compliance gate requires non-empty 'provenance'.")
    if not str(payload.get("license", "")).strip():
        raise ValueError("Dataset compliance gate requires non-empty 'license'.")
    if require_rights_attestation and bool(payload.get("rights_attested")) is not True:
        raise ValueError(
            "Dataset compliance gate requires 'rights_attested=true' in dataset_metadata.json."
        )
    return payload


def validate_dataset_compliance(
    *,
    dataset_path: str | Path,
    compliance_enabled: bool,
    usage_mode: str,
    require_provenance: bool = True,
    require_rights_attestation: bool = True,
) -> dict[str, Any] | None:
    dataset_dir = Path(dataset_path)
    forced_compliance = compliance_enabled or usage_mode.strip().lower() == "redistributable"
    if not forced_compliance:
        return None
    return _require_compliance_metadata(
        dataset_dir=dataset_dir,
        require_provenance=require_provenance,
        require_rights_attestation=require_rights_attestation,
    )


def enforce_dataset_compliance(
    *,
    dataset_path: str | Path,
    compliance_enabled: bool,
    usage_mode: str,
    require_provenance: bool = True,
    require_rights_attestation: bool = True,
) -> None:
    validate_dataset_compliance(
        dataset_path=dataset_path,
        compliance_enabled=compliance_enabled,
        usage_mode=usage_mode,
        require_provenance=require_provenance,
        require_rights_attestation=require_rights_attestation,
    )


def register_dataset(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    compliance_enabled: bool,
    usage_mode: str,
    require_provenance: bool = True,
    require_rights_attestation: bool = True,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_dir}")

    forced_compliance = compliance_enabled or usage_mode.strip().lower() == "redistributable"
    compliance_meta: dict[str, Any] | None = None
    if forced_compliance:
        compliance_meta = validate_dataset_compliance(
            dataset_path=dataset_dir,
            compliance_enabled=True,
            usage_mode=usage_mode,
            require_provenance=require_provenance,
            require_rights_attestation=require_rights_attestation,
        )

    media_paths = sorted(
        [path for path in dataset_dir.iterdir() if path.is_file() and path.suffix.lower() in _MEDIA_SUFFIXES]
    )
    samples: list[dict[str, Any]] = []
    for media in media_paths:
        sample_id = media.stem
        caption_path = media.with_suffix(".txt")
        samples.append(
            {
                "sample_id": sample_id,
                "group_id": sample_id,
                "media_path": str(media),
                "caption_path": str(caption_path) if caption_path.exists() else "",
            }
        )

    assignments = build_split_assignments(
        samples=samples,
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    for sample in samples:
        sample["split"] = assignments[sample["sample_id"]]

    payload: dict[str, Any] = {
        "version": 1,
        "dataset_path": str(dataset_dir),
        "split_seed": int(split_seed),
        "ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "usage_mode": usage_mode,
        "compliance_enforced": bool(forced_compliance),
        "samples": samples,
    }
    if compliance_meta is not None:
        payload["compliance"] = compliance_meta

    _atomic_write_json(Path(output_path), payload)
    return payload


def load_registration(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))
