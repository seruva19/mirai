"""Explicit model download step (thin-slice)."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.moe.artifacts.downloads import get_download_repo_by_variant
from mirai.core.moe.artifacts.downloads import artifact_allow_patterns
from mirai.core.moe.artifacts.downloads import get_moe_artifact_download_spec
from mirai.core.moe.artifacts.downloads import get_moe_artifact_manifest_metadata
from mirai.core.moe.artifacts.downloads import verify_pinned_artifact_files
from mirai.core.moe.artifacts.downloads import verify_artifact_manifest_source
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
        "--include-source-code",
        action="store_true",
        help="Also download source-code patterns declared by the artifact spec.",
    )
    p.add_argument(
        "--mirror",
        action="store_true",
        help="Use the public mirror declared by the artifact spec.",
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


def _snapshot_file_inventory(
    output_dir: Path, *, expected_hashes: dict[str, str] | None = None
) -> list[dict[str, object]]:
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
        record: dict[str, object] = {
            "path": relative,
            "size": int(path.stat().st_size),
            "status": "downloaded",
        }
        expected = (expected_hashes or {}).get(relative)
        if expected is not None:
            record["sha256"] = expected
        records.append(record)
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


_LFS_POINTER = re.compile(
    r"\Aversion https://git-lfs.github.com/spec/v1\s+"
    r"oid sha256:(?P<oid>[0-9a-f]{64})\s+size (?P<size>\d+)\s*\Z"
)
_MIRROR_STAGING_MARKER = ".mirai-mirror-download.json"
_TRANSIENT_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


def _gitee_repo_parts(url: str) -> tuple[str, str]:
    match = re.fullmatch(r"https://gitee\.com/([^/]+)/([^/]+?)(?:\.git)?", url.strip())
    if match is None:
        raise ValueError(f"Unsupported Git-LFS mirror URL: {url!r}.")
    return match.group(1), match.group(2)


def _sanitized_url(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _request_with_retries(session, method: str, url: str, *, attempts: int = 5, **kwargs):
    last_status = None
    for attempt in range(attempts):
        try:
            response = getattr(session, method)(url, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            response = getattr(exc, "response", None)
            last_status = getattr(response, "status_code", None)
            transient = last_status in _TRANSIENT_HTTP_STATUS or last_status is None
            if not transient or attempt + 1 >= attempts:
                status = f" HTTP {last_status}" if last_status is not None else ""
                raise RuntimeError(
                    f"Mirror request failed{status}: {_sanitized_url(url)}"
                ) from None
            time.sleep(min(8.0, float(2**attempt)))
    raise AssertionError("unreachable")


def _download_lfs_payload(
    session,
    *,
    batch_url: str,
    oid: str,
    expected_size: int,
    partial: Path,
    attempts: int = 5,
) -> None:
    """Resume an LFS object, reacquiring its ephemeral URL after stream failures."""

    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > expected_size:
            partial.unlink()
            offset = 0
        batch_response = _request_with_retries(
            session,
            "post",
            batch_url,
            headers={
                "Accept": "application/vnd.git-lfs+json",
                "Content-Type": "application/vnd.git-lfs+json",
            },
            json={
                "operation": "download",
                "transfers": ["basic"],
                "objects": [{"oid": oid, "size": expected_size}],
            },
            timeout=60,
        )
        href = batch_response.json()["objects"][0]["actions"]["download"]["href"]
        try:
            payload = _request_with_retries(
                session,
                "get",
                href,
                stream=True,
                headers={"Range": f"bytes={offset}-"} if offset else {},
                timeout=(60, 600),
            )
            status_code = int(getattr(payload, "status_code", 200))
            if offset and status_code != 206:
                partial.unlink(missing_ok=True)
                offset = 0
            with payload:
                with partial.open("ab" if offset else "xb") as handle:
                    for chunk in payload.iter_content(chunk_size=8 * 1024**2):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size != expected_size:
                raise OSError("incomplete mirror payload stream")
            if _sha256_file(partial) != oid:
                raise ValueError("mirror payload digest mismatch")
            return
        except ValueError:
            partial.unlink(missing_ok=True)
            raise
        except Exception:
            if attempt + 1 >= attempts:
                raise RuntimeError(
                    f"Mirror payload download failed: {_sanitized_url(href)}"
                ) from None
            time.sleep(min(8.0, float(2**attempt)))
    raise AssertionError("unreachable")


def _git_blob_sha1(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def _prepare_mirror_staging(output_dir: Path, artifact) -> Path:
    staging = output_dir.with_name(f".{output_dir.name}.mirror-partial")
    marker = staging / _MIRROR_STAGING_MARKER
    identity = {
        "variant": artifact.variant,
        "mirror_url": artifact.mirror_url,
        "mirror_revision": artifact.mirror_revision,
    }
    if staging.exists():
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise FileExistsError(
                f"Mirror staging path exists but is not owned by this download: {staging}"
            )
    else:
        staging.mkdir(parents=True)
        marker.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")
    return staging


def _download_gitee_mirror(artifact, output_dir: Path) -> Path:
    """Download a declared Gitee Git-LFS mirror without exposing signed URLs."""
    import requests

    if not artifact.mirror_url or not artifact.mirror_revision:
        raise ValueError(f"Artifact {artifact.variant!r} does not declare a mirror.")
    if output_dir.exists():
        raise FileExistsError(f"Mirror destination already exists: {output_dir}")
    owner, repo = _gitee_repo_parts(artifact.mirror_url)
    api_root = f"https://gitee.com/api/v5/repos/{owner}/{repo}"
    staging = _prepare_mirror_staging(output_dir, artifact)
    session = requests.Session()
    try:
        tree_response = _request_with_retries(
            session,
            "get",
            f"{api_root}/git/trees/{artifact.mirror_revision}",
            params={"recursive": "1"},
            timeout=60,
        )
        entries = {str(item["path"]): item for item in tree_response.json()["tree"]}
        for relative_name in artifact.required_patterns:
            entry = entries.get(relative_name)
            if not isinstance(entry, dict) or entry.get("type") != "blob":
                raise FileNotFoundError(f"Mirror revision has no file {relative_name!r}.")
            destination = staging / Path(relative_name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and int(entry.get("size", -1)) != 135:
                existing = destination.read_bytes()
                if _git_blob_sha1(existing) == str(entry["sha"]):
                    continue
            blob_response = _request_with_retries(
                session, "get", f"{api_root}/git/blobs/{entry['sha']}", timeout=60
            )
            content = base64.b64decode(blob_response.json()["content"])
            pointer = _LFS_POINTER.fullmatch(content.decode("utf-8", errors="replace"))
            if pointer is None:
                if _git_blob_sha1(content) != str(entry["sha"]):
                    raise ValueError(f"Mirror Git blob integrity mismatch for {relative_name}.")
                destination.write_bytes(content)
                continue
            oid = pointer.group("oid")
            expected_size = int(pointer.group("size"))
            if destination.is_file() and destination.stat().st_size == expected_size:
                if _sha256_file(destination) == oid:
                    continue
            batch_url = f"https://gitee.com/{owner}/{repo}.git/info/lfs/objects/batch"
            partial = destination.with_name(destination.name + ".download")
            _download_lfs_payload(
                session,
                batch_url=batch_url,
                oid=oid,
                expected_size=expected_size,
                partial=partial,
            )
            os.replace(partial, destination)
        _derive_filtered_transformers_snapshot(staging, artifact)
        (staging / _MIRROR_STAGING_MARKER).unlink()
        os.replace(staging, output_dir)
    except BaseException:
        raise
    return output_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derive_filtered_transformers_snapshot(root: Path, artifact) -> None:
    if not artifact.index_key_prefixes:
        return
    index_path = root / "model.safetensors.index.json"
    source_path = root / "model.safetensors.index.source.json"
    if source_path.is_file():
        source_bytes = source_path.read_bytes()
    else:
        source_bytes = index_path.read_bytes()
        source_path.write_bytes(source_bytes)
    payload = json.loads(source_bytes)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Mirror safetensors index has no weight_map.")
    selected = {
        key: shard
        for key, shard in weight_map.items()
        if any(str(key).startswith(prefix) for prefix in artifact.index_key_prefixes)
    }
    if not selected:
        raise ValueError("Mirror index filter selected no tensors.")
    missing_shards = sorted(
        {str(shard) for shard in selected.values() if not (root / str(shard)).is_file()}
    )
    if missing_shards:
        raise FileNotFoundError("Filtered mirror index references missing shards.")
    payload["weight_map"] = selected
    payload["metadata"] = {
        "mirai_derivation": "prefix_filtered_index_v1",
        "source_repo_id": artifact.repo_id,
        "source_revision": artifact.revision,
        "source_index_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_total_size": payload.get("metadata", {}).get("total_size"),
    }
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact.encoder_only_config:
        config_path = root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["architectures"] = ["T5GemmaEncoderModel"]
        config["is_encoder_decoder"] = False
        config["tie_word_embeddings"] = True
        encoder = config.get("encoder")
        if not isinstance(encoder, dict):
            raise ValueError("Encoder-only derivation requires an encoder config.")
        encoder["is_decoder"] = False
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_id = REPO_BY_VARIANT[args.variant]
    artifact = get_moe_artifact_download_spec(args.variant)
    out_dir = Path(args.output_dir)

    token = _resolve_hf_token()

    requested_patterns = [str(v) for v in args.allow_pattern if str(v).strip()]
    allow_patterns = artifact_allow_patterns(
        args.variant,
        requested_patterns,
        include_source_code=bool(args.include_source_code),
    )
    if args.mirror:
        local_dir = str(_download_gitee_mirror(artifact, out_dir))
    else:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "huggingface_hub is required for model downloads. Install it.\n"
                f"Import error: {exc}"
            )
        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=artifact.revision,
            local_dir=str(out_dir),
            resume_download=True,
            token=token,
            allow_patterns=allow_patterns,
            max_workers=max(1, int(args.max_workers)),
        )
    verify_pinned_artifact_files(local_dir, args.variant)
    expected_hashes = {item.path: item.sha256 for item in artifact.expected_files}
    payload = {
        "schema_version": 1,
        "status": "downloaded",
        "variant": args.variant,
        "repo_id": repo_id,
        "transport": "git_lfs_mirror" if args.mirror else "huggingface_model",
        "moe_artifact": get_moe_artifact_manifest_metadata(args.variant),
        "output_dir": str(Path(local_dir).resolve()),
        "revision": artifact.revision,
        "allow_patterns": list(allow_patterns or ()),
        "files": _snapshot_file_inventory(
            Path(local_dir), expected_hashes=expected_hashes
        ),
    }
    _write_manifest(out_dir, payload)
    read_validated_manifest(out_dir / "download_manifest.json")
    verify_artifact_manifest_source(out_dir, args.variant)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
