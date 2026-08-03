"""Explicit model download step (thin-slice)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.moe.artifacts.downloads import get_download_repo_by_variant
from mirai.core.moe.artifacts.downloads import get_moe_artifact_manifest_metadata
from mirai.core.moe.artifacts.manifest import DEFAULT_DOWNLOAD_MANIFEST
from mirai.core.moe.artifacts.manifest import read_validated_download_manifest
from mirai.core.moe.artifacts.manifest import write_download_manifest


REPO_BY_VARIANT = get_download_repo_by_variant()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(REPO_BY_VARIANT), required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional glob(s) forwarded to snapshot_download allow_patterns.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel Hugging Face download workers. Lower this to reduce bandwidth pressure.",
    )
    return p.parse_args()


def _write_manifest(output_dir: Path, payload: dict) -> None:
    write_download_manifest(output_dir, payload)


def _snapshot_file_inventory(output_dir: Path) -> list[dict[str, object]]:
    """Record downloaded snapshot payload files, excluding local cache metadata."""

    root = output_dir.resolve()
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if (
            relative == DEFAULT_DOWNLOAD_MANIFEST
            or relative.startswith(".cache/")
            or relative.startswith(".git/")
        ):
            continue
        records.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "status": "downloaded",
            }
        )
    if not records:
        raise RuntimeError("Downloaded model snapshot contains no payload files.")
    return records


def read_validated_manifest(path: str | Path) -> dict:
    return read_validated_download_manifest(path)


def _resolve_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder

        cached = HfFolder.get_token()
        if cached:
            return str(cached)
    except Exception:
        pass
    return None


def main() -> int:
    args = parse_args()
    repo_id = REPO_BY_VARIANT[args.variant]
    out_dir = Path(args.output_dir)

    token = _resolve_hf_token()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required for model downloads. Install it.\n"
            f"Import error: {exc}"
        )

    allow_patterns = [str(v) for v in args.allow_pattern if str(v).strip()]
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        resume_download=True,
        token=token,
        allow_patterns=allow_patterns if allow_patterns else None,
        max_workers=max(1, int(args.max_workers)),
    )
    payload = {
        "status": "downloaded",
        "variant": args.variant,
        "repo_id": repo_id,
        "moe_artifact": get_moe_artifact_manifest_metadata(args.variant),
        "output_dir": str(Path(local_dir).resolve()),
        "allow_patterns": allow_patterns,
        "files": _snapshot_file_inventory(Path(local_dir)),
    }
    _write_manifest(out_dir, payload)
    read_validated_manifest(out_dir / "download_manifest.json")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
