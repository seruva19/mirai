"""Cache fingerprint helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def caption_sha256(caption: str) -> str:
    return _sha256_bytes(caption.encode("utf-8"))


def text_cache_fingerprint(
    *,
    caption: str,
    encoder_id: str,
    policy_version: str,
    tokenizer_revision: str = "",
    encoder_revision: str = "",
    encoder_dtype: str = "",
) -> str:
    payload = (
        f"caption={caption_sha256(caption)};"
        f"encoder={encoder_id};"
        f"dtype={encoder_dtype};"
        f"policy={policy_version};"
        f"tokenizer_rev={tokenizer_revision};"
        f"encoder_rev={encoder_revision}"
    )
    return _sha256_bytes(payload.encode("utf-8"))


def media_fingerprint(path: str | Path, *, strict: bool) -> str:
    p = Path(path)
    if strict:
        return f"strict:{file_sha256(p)}"
    stat = p.stat()
    return f"fast:{stat.st_size}:{stat.st_mtime_ns}"


def latent_cache_fingerprint(
    *,
    media_fp: str,
    vae_id: str,
    vae_dtype: str,
    preprocess_policy_hash: str,
) -> str:
    payload = (
        f"media={media_fp};"
        f"vae={vae_id};"
        f"dtype={vae_dtype};"
        f"preprocess={preprocess_policy_hash}"
    )
    return _sha256_bytes(payload.encode("utf-8"))


def selective_rebuild_plan(
    previous: dict[str, str],
    current: dict[str, str],
) -> dict[str, bool]:
    return {
        "rebuild_latent": previous.get("latent") != current.get("latent"),
        "rebuild_text": previous.get("text") != current.get("text"),
    }
