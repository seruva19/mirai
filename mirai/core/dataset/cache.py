"""Cache pipeline for lineage-bound dataset latents and conditioning tensors."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import hashlib
import json
import os
import random
import sys
from contextlib import contextmanager
from typing import Any

from mirai.core.dataset.registration import load_registration, resolve_registration_path
from mirai.core.dataset.record_batch import (
    latent_value_to_tensor,
    text_embed_value_to_tensor,
)
from mirai.core.dataset.fingerprint import (
    latent_cache_fingerprint,
    media_fingerprint,
    selective_rebuild_plan,
    text_cache_fingerprint,
)
from mirai.core.dataset.native_encode import (
    ValidationCacheEncoder,
    native_cache_encoder_contract_error,
    validate_native_cache_encoder,
)
from mirai.core.models.providers import (
    NativeCacheEncoderConfig,
    build_native_cache_encoder,
    get_model_family_provider,
)
from mirai.core.models.providers import model_supports_native_cache_encoding
from mirai.core.lineage import snapshot_component_id, snapshot_descriptor_for_path
from mirai.core.persistence.migrations import migrate_cache_payload

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for cache pipeline: {exc}")


@dataclass
class CachedRecord:
    sample_id: str
    latent: Any
    text_embed: Any
    clip_embed: Any | None = None
    caption_variant_index: int = 0
    caption: str = ""
    base_sample_id: str = ""
    clip_index: int = 0
    text_mask: list[int] | None = None
    loss_mask: float | None = None
    split: str = "train"
    fingerprints: dict[str, str] | None = None
    bucket_id: str = ""
    bucket_h: int = 0
    bucket_w: int = 0
    bucket_frames: int = 0
    metadata: dict[str, Any] | None = None


def _caption_embed(caption: str) -> float:
    digest = hashlib.blake2b(caption.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return (value % 10000) / 10000.0


def _shuffle_caption_tags(caption: str, *, seed_text: str) -> str:
    text = caption.strip()
    if not text:
        return text
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        joiner = ", "
    else:
        parts = [part for part in text.split(" ") if part]
        joiner = " "
    if len(parts) <= 1:
        return text
    seed_bytes = hashlib.blake2b(seed_text.encode("utf-8"), digest_size=8).digest()
    rng = random.Random(int.from_bytes(seed_bytes, "little"))
    rng.shuffle(parts)
    return joiner.join(parts)


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


def _collect_media_files(dataset_dir: Path, *, mask_extension: str) -> list[Path]:
    out: list[Path] = []
    preprocessed_latent_stems: set[str] = {
        path.stem
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pt"
    }
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _MEDIA_SUFFIXES:
            continue
        if (
            path.suffix.lower() != ".pt"
            and path.stem in preprocessed_latent_stems
        ):
            continue
        if mask_extension and path.name.endswith(str(mask_extension)):
            continue
        out.append(path)
    return out


def _is_finite_payload_value(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    try:
        return bool(torch.isfinite(torch.tensor(float(value), dtype=torch.float32)).item())
    except Exception:
        return False


def _record_skipped_sample(
    skipped: list[dict[str, str]],
    skip_reasons: dict[str, str],
    *,
    sample_id: str,
    status: str,
    reason: str,
) -> None:
    """Skips are per-sample data failures and must never be silent."""

    print(
        f"[cache] warning: skipping sample '{sample_id}': {status} ({reason})",
        file=sys.stderr,
    )
    skipped.append({"sample_id": sample_id, "status": status})
    skip_reasons.setdefault(status, str(reason))


def _format_skip_breakdown(
    skipped: list[dict[str, str]],
    skip_reasons: dict[str, str],
) -> str:
    counts: dict[str, int] = {}
    for entry in skipped:
        status = str(entry.get("status", "skipped_unknown"))
        counts[status] = counts.get(status, 0) + 1
    parts: list[str] = []
    for status in sorted(counts):
        reason = skip_reasons.get(status, "")
        parts.append(
            f"{counts[status]} {status}({reason})" if reason else f"{counts[status]} {status}"
        )
    return ", ".join(parts)


def _normalize_split(value: Any, *, source_label: str) -> str:
    split = str(value or "train").strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Invalid split '{value}' in {source_label}. Expected one of: train, val, test."
        )
    return split


def _validate_fingerprints(value: Any, *, source_label: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Invalid fingerprints in {source_label}: expected object.")
    out: dict[str, str] = {}
    for key in ("latent", "text"):
        raw = value.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            raise ValueError(
                f"Invalid fingerprints in {source_label}: field '{key}' must be non-empty."
            )
        out[key] = text
    return out


def _load_registration_map(
    *,
    dataset_dir: Path,
    registration_path: str | Path | None,
) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    resolved_path = resolve_registration_path(dataset_dir, registration_path=registration_path)
    if not resolved_path.exists():
        return None, {}
    payload = load_registration(resolved_path)
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(
            f"Invalid dataset registration at {resolved_path}: 'samples' must be a list."
        )
    out: dict[str, dict[str, Any]] = {}
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(
                f"Invalid dataset registration at {resolved_path}: sample[{idx}] is not an object."
            )
        sample_id = str(sample.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(
                f"Invalid dataset registration at {resolved_path}: sample[{idx}] is missing sample_id."
            )
        if sample_id in out:
            raise ValueError(
                f"Invalid dataset registration at {resolved_path}: duplicate sample_id '{sample_id}'."
            )
        sample_copy = dict(sample)
        sample_copy["split"] = _normalize_split(
            sample_copy.get("split"),
            source_label=f"dataset registration '{resolved_path}' sample '{sample_id}'",
        )
        out[sample_id] = sample_copy
    return resolved_path, out


def estimate_cache_disk_bytes(*, num_records: int) -> int:
    # Conservative serialized-record estimate used for capacity warnings.
    return int(4096 + (max(0, num_records) * 96))


def build_cache_from_config(cfg: Any) -> dict[str, Any]:
    """Build cache using normalized fields from a loaded runtime config.

    This helper centralizes the cache build contract used by both `scripts/cache.py`
    and training startup preflight paths.
    """

    frame_buckets = list(cfg.dataset.frame_buckets) if cfg.dataset.frame_buckets else [33]
    enable_bucketing = bool(cfg.dataset.enable_bucketing)
    if enable_bucketing:
        from mirai.core.dataset.bucketing.bucket_resolve import (
            parse_resolution_buckets,
            validate_frame_buckets,
        )

        validate_frame_buckets(frame_buckets)
        resolution_buckets = parse_resolution_buckets(cfg.dataset)
    else:
        resolution_buckets = None
    native_max_frames = max(frame_buckets) if enable_bucketing else 33
    curriculum_payload = (
        cfg.training.curriculum
        if isinstance(cfg.training.curriculum, dict)
        else {}
    )
    curriculum_task_mix_enabled = bool(
        curriculum_payload.get("enabled", False)
        and curriculum_payload.get("task_mix_schedule")
    )
    return build_cache(
        cfg.dataset.path,
        cfg.dataset.cache_path,
        max_cache_skip_ratio=cfg.dataset.max_cache_skip_ratio,
        cache_mode=cfg.dataset.cache_mode,
        cache_compression=cfg.dataset.cache_compression,
        fp8_text_encoder=cfg.dataset.fp8_text_encoder,
        clips_per_video=cfg.dataset.clips_per_video,
        tag_shuffle_variants=cfg.dataset.tag_shuffle_variants,
        partial_recovery=cfg.dataset.partial_recovery,
        mask_extension=cfg.dataset.mask_extension,
        caption_format=cfg.dataset.caption_format,
        native_encode=model_supports_native_cache_encoding(str(cfg.model.type), cfg),
        model_type=cfg.model.type,
        model_variant=cfg.model.params.variant,
        model_path=cfg.model.path,
        model_dtype=cfg.model.dtype,
        model_denoiser_subfolder=str(
            getattr(cfg.model.params, "denoiser_subfolder", "transformer")
        ),
        native_max_frames=native_max_frames,
        strict_native_assets=bool(cfg.model.params.strict_native_assets),
        text_encoder_path=str(getattr(cfg.model.params, "text_encoder_path", "")),
        enable_bucketing=enable_bucketing,
        resolution_buckets=resolution_buckets,
        frame_buckets=frame_buckets,
        bucket_resize_mode=str(cfg.dataset.bucket_resize_mode),
        routing_domain_metadata_key=str(
            cfg.dataset.moe_routing.domain_metadata_key
        ).strip(),
        required_registration_metadata_keys=(
            ("training_task",)
            if curriculum_task_mix_enabled
            or str(cfg.strategy.type).strip().lower() == "multi_task_video"
            else ()
        ),
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _checksum_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _verify_checksum(path: Path, verify_checksum: bool) -> None:
    if not verify_checksum:
        return
    checksum = _checksum_path(path)
    if not checksum.exists():
        return
    expected = checksum.read_text(encoding="utf-8").strip()
    observed = _sha256_file(path)
    if expected != observed:
        raise ValueError(
            f"Cache checksum mismatch for {path} "
            f"(expected {expected}, got {observed})."
        )


def _write_checksum(path: Path) -> None:
    _checksum_path(path).write_text(_sha256_file(path) + "\n", encoding="utf-8")


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def indexed_cache_metadata_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return path.with_name(path.name + ".index.json")


def indexed_cache_tensor_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return path.with_name(path.name + ".tensors.safetensors")


def _read_checksum_text(path: Path) -> str:
    checksum_path = _checksum_path(path)
    if not checksum_path.exists():
        return ""
    return checksum_path.read_text(encoding="utf-8").strip()


class IndexedCacheRecord:
    def __init__(self, metadata: dict[str, Any], tensor_cache: "MmapSafetensorsCache"):
        self._metadata = dict(metadata)
        self._tensor_cache = tensor_cache

    def _tensor_key_for(self, key: str) -> str | None:
        if key == "latent":
            return str(self._metadata["latent_key"])
        if key == "text_embed":
            return str(self._metadata["text_embed_key"])
        if key == "clip_embed":
            value = str(self._metadata.get("clip_embed_key", "")).strip()
            return value or None
        return None

    def __getitem__(self, key: str) -> Any:
        tensor_key = self._tensor_key_for(key)
        if tensor_key is not None:
            return self._tensor_cache[tensor_key]
        if key in {"latent", "text_embed"}:
            raise KeyError(key)
        return self._metadata[key]

    def get(self, key: str, default: Any = None) -> Any:
        tensor_key = self._tensor_key_for(key)
        if tensor_key is not None:
            return self._tensor_cache[tensor_key]
        if key == "clip_embed":
            return default
        return self._metadata.get(key, default)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in {"latent", "text_embed"}:
            return True
        if key == "clip_embed":
            return self._tensor_key_for(key) is not None
        return key in self._metadata

    @staticmethod
    def batch_get(records: list[Any], key: str) -> list[Any]:
        if not records:
            return []
        if (
            key in {"latent", "text_embed", "clip_embed"}
            and all(isinstance(record, IndexedCacheRecord) for record in records)
        ):
            typed_records = [record for record in records if isinstance(record, IndexedCacheRecord)]
            first = typed_records[0]
            if all(record._tensor_cache is first._tensor_cache for record in typed_records):
                out: list[Any] = [None] * len(typed_records)
                tensor_keys: list[str] = []
                positions: list[int] = []
                for idx, record in enumerate(typed_records):
                    tensor_key = record._tensor_key_for(key)
                    if tensor_key is None:
                        continue
                    positions.append(idx)
                    tensor_keys.append(tensor_key)
                fetched = first._tensor_cache.get_many(tensor_keys)
                for pos, value in zip(positions, fetched, strict=True):
                    out[pos] = value
                return out
        return [record.get(key) if key == "clip_embed" else record[key] for record in records]


def save_indexed_cache_sidecar(
    cache_path: str | Path,
    records: list[dict[str, Any]],
    *,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_path = indexed_cache_metadata_path(cache_path)
    tensor_path = indexed_cache_tensor_path(cache_path)
    tensors: dict[str, torch.Tensor] = {}
    metadata_records: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        metadata = dict(record)
        latent_key = f"latent__{idx}"
        text_embed_key = f"text_embed__{idx}"
        tensors[latent_key] = latent_value_to_tensor(metadata.pop("latent")).clone()
        tensors[text_embed_key] = text_embed_value_to_tensor(metadata.pop("text_embed")).clone()
        metadata["latent_key"] = latent_key
        metadata["text_embed_key"] = text_embed_key
        clip_value = metadata.pop("clip_embed", None)
        if clip_value is not None:
            clip_embed_key = f"clip_embed__{idx}"
            tensors[clip_embed_key] = text_embed_value_to_tensor(clip_value).clone()
            metadata["clip_embed_key"] = clip_embed_key
        metadata_records.append(metadata)

    save_tensor_cache_safetensors(tensor_path, tensors)
    payload = {
        "version": 1,
        "num_records": len(metadata_records),
        "source_cache_checksum": _read_checksum_text(Path(cache_path)),
        "tensor_path": str(tensor_path),
        **dict(metadata_extra or {}),
        "records": metadata_records,
    }
    _atomic_write_text(
        metadata_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _write_checksum(metadata_path)
    return {
        "metadata_path": str(metadata_path),
        "tensor_path": str(tensor_path),
        "num_records": len(metadata_records),
    }


def load_indexed_cache_metadata(
    cache_path: str | Path,
    *,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    metadata_path = indexed_cache_metadata_path(cache_path)
    _verify_checksum(metadata_path, verify_checksum=verify_checksum)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid indexed cache metadata at {metadata_path}: expected object.")
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid indexed cache metadata at {metadata_path}: 'records' must be a list.")
    expected_checksum = str(payload.get("source_cache_checksum", "")).strip()
    observed_checksum = _read_checksum_text(Path(cache_path))
    if expected_checksum and observed_checksum and expected_checksum != observed_checksum:
        raise ValueError(
            f"Indexed cache metadata at {metadata_path} does not match cache checksum for {cache_path}."
        )
    return payload


def ensure_indexed_cache_sidecar(
    cache_path: str | Path,
    *,
    verify_checksum: bool = True,
) -> dict[str, Any] | None:
    metadata_path = indexed_cache_metadata_path(cache_path)
    tensor_path = indexed_cache_tensor_path(cache_path)
    if metadata_path.exists() and tensor_path.exists():
        return {
            "metadata_path": str(metadata_path),
            "tensor_path": str(tensor_path),
        }
    try:
        cache = load_cache(cache_path, verify_checksum=verify_checksum)
    except Exception:
        return None
    try:
        return save_indexed_cache_sidecar(
            cache_path,
            list(cache.get("records", [])),
        )
    except ModuleNotFoundError:
        return None


def load_indexed_cache_records(
    cache_path: str | Path,
    *,
    verify_checksum: bool = True,
) -> list[IndexedCacheRecord] | None:
    ensured = ensure_indexed_cache_sidecar(cache_path, verify_checksum=verify_checksum)
    if ensured is None:
        return None
    try:
        payload = load_indexed_cache_metadata(cache_path, verify_checksum=verify_checksum)
    except ValueError:
        cache = load_cache(cache_path, verify_checksum=verify_checksum)
        try:
            save_indexed_cache_sidecar(
                cache_path,
                list(cache.get("records", [])),
            )
        except ModuleNotFoundError:
            return None
        payload = load_indexed_cache_metadata(cache_path, verify_checksum=verify_checksum)
    tensor_cache = MmapSafetensorsCache(
        indexed_cache_tensor_path(cache_path),
        verify_checksum=verify_checksum,
    )
    return [
        IndexedCacheRecord(record, tensor_cache)
        for record in payload.get("records", [])
        if isinstance(record, dict)
    ]


def _atomic_torch_save(path: Path, payload: dict, *, cache_compression: str = "none") -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    compression = cache_compression.strip().lower()
    if compression == "none":
        torch.save(payload, tmp_path)
    elif compression == "zstd":
        try:
            import zstandard as zstd
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "cache_compression='zstd' requires the 'zstandard' package."
            ) from exc
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        compressor = zstd.ZstdCompressor(level=3)
        compressed = compressor.compress(buffer.getvalue())
        tmp_path.write_bytes(compressed)
    else:
        raise ValueError(f"Unsupported cache_compression '{cache_compression}'.")
    os.replace(tmp_path, path)


@contextmanager
def cache_write_lock(cache_file: Path):
    lock_path = cache_file.with_name(cache_file.name + ".lock")
    fd = None
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Cache write lock is already held for {cache_file}."
        ) from exc
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        if lock_path.exists():
            lock_path.unlink()


def validate_cache_payload(payload: dict, *, source_label: str) -> None:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid cache payload in {source_label}: 'records' must be a list.")
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid cache payload in {source_label}: record[{idx}] is not a table."
            )
        sample_id = str(record.get("sample_id", f"idx_{idx}"))
        for field_name in ("latent", "text_embed"):
            value = record.get(field_name)
            if value is None:
                continue
            if not _is_finite_payload_value(value):
                raise ValueError(
                    "Non-finite cached value detected "
                    f"in shard '{source_label}', sample '{sample_id}', field '{field_name}'."
                )
        clip_embed = record.get("clip_embed")
        if clip_embed is not None and not _is_finite_payload_value(clip_embed):
            raise ValueError(
                "Non-finite cached value detected "
                f"in shard '{source_label}', sample '{sample_id}', field 'clip_embed'."
            )
        mask = record.get("text_mask")
        if mask is not None:
            if not isinstance(mask, list) or not all(int(v) in {0, 1} for v in mask):
                raise ValueError(
                    f"Invalid text_mask in shard '{source_label}', sample '{sample_id}'."
                )
        loss_mask = record.get("loss_mask")
        if loss_mask is not None:
            if not torch.isfinite(torch.tensor(float(loss_mask), dtype=torch.float32)):
                raise ValueError(
                    f"Invalid loss_mask in shard '{source_label}', sample '{sample_id}'."
                )
        split = record.get("split", "train")
        _normalize_split(split, source_label=f"cache '{source_label}', sample '{sample_id}'")
        _validate_fingerprints(
            record.get("fingerprints"),
            source_label=f"cache '{source_label}', sample '{sample_id}'",
        )


def build_cache(
    dataset_path: str | Path,
    cache_path: str | Path,
    *,
    max_cache_skip_ratio: float = 0.2,
    cache_mode: str = "disk",
    cache_compression: str = "none",
    fp8_text_encoder: bool = False,
    clips_per_video: int = 1,
    tag_shuffle_variants: int = 1,
    partial_recovery: bool = True,
    mask_extension: str = ".mask.pt",
    caption_format: str = "raw",
    native_encode: bool = False,
    model_type: str = "",
    model_variant: str = "",
    model_path: str = "",
    model_dtype: str = "bf16",
    model_denoiser_subfolder: str = "transformer",
    native_max_frames: int = 33,
    strict_native_assets: bool = False,
    text_encoder_path: str = "",
    registration_path: str | Path | None = None,
    routing_domain_metadata_key: str = "",
    required_registration_metadata_keys: tuple[str, ...] = (),
    # bucketing config
    enable_bucketing: bool = False,
    resolution_buckets: list[tuple[int, int]] | None = None,
    frame_buckets: list[int] | None = None,
    bucket_resize_mode: str = "resize_crop",
) -> dict:
    dataset_dir = Path(dataset_path)
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    caption_format_normalized = str(caption_format or "raw").strip().lower()
    provider = get_model_family_provider(str(model_type)) if str(model_type).strip() else None
    if provider is None and caption_format_normalized != "raw":
        raise ValueError(
            f"dataset.caption_format='{caption_format_normalized}' requires a "
            "registered model-family provider."
        )
    if provider is not None:
        provider.validate_dataset_caption_format(caption_format_normalized)

    with cache_write_lock(cache_file):
        records: list[CachedRecord] = []
        skipped: list[dict[str, str]] = []
        skip_reasons: dict[str, str] = {}
        clips_per_video = max(1, int(clips_per_video))
        tag_shuffle_variants = max(1, int(tag_shuffle_variants))
        recovered_count = 0
        recovered_map: dict[str, dict] = {}
        if (
            partial_recovery
            and cache_mode.strip().lower() == "disk"
            and cache_file.exists()
        ):
            try:
                previous = load_cache(cache_file, verify_checksum=False)
                for record in previous.get("records", []):
                    if isinstance(record, dict) and "sample_id" in record:
                        recovered_map[str(record["sample_id"])] = record
            except Exception:
                recovered_map = {}
        # When bucketing is enabled, max_frames is driven by the largest frame bucket.
        _effective_max_frames = (
            max(int(f) for f in frame_buckets)
            if enable_bucketing and frame_buckets
            else int(native_max_frames)
        )
        native_config = NativeCacheEncoderConfig(
            enabled=bool(native_encode),
            model_type=str(model_type),
            variant=str(model_variant),
            model_path=str(model_path),
            dtype_name=str(model_dtype),
            max_frames=_effective_max_frames,
            denoiser_subfolder=str(model_denoiser_subfolder or "transformer"),
            strict_assets=bool(strict_native_assets),
            text_encoder_path=str(text_encoder_path),
            enable_bucketing=bool(enable_bucketing),
            resolution_buckets=list(resolution_buckets) if resolution_buckets else None,
            frame_buckets=list(frame_buckets) if frame_buckets else None,
            bucket_resize_mode=str(bucket_resize_mode),
        )
        media_files = _collect_media_files(dataset_dir, mask_extension=str(mask_extension))
        if not media_files:
            raise ValueError(
                "No media files found in dataset.path. Populate dataset.path with "
                "media and captions before caching."
            )
        native = build_native_cache_encoder(native_config)
        if (
            native is None
            and provider is None
            and not native_encode
            and all(path.suffix.lower() == ".pt" for path in media_files)
        ):
            native = ValidationCacheEncoder(
                enabled=False,
                model_type="precomputed",
                variant="precomputed",
                model_path="",
                dtype_name=str(model_dtype),
                max_frames=_effective_max_frames,
                enable_bucketing=False,
            )
        if native is None:
            raise ValueError(
                f"Model family '{str(model_type).strip().lower()}' must provide "
                "build_native_cache_encoder() for raw-media cache construction."
            )
        # Conformance is checked before any sample is touched: a missing method
        # discovered inside the loop is indistinguishable from per-sample data
        # failure and would silently empty the cache.
        validate_native_cache_encoder(
            native,
            source=f"model family '{str(model_type).strip().lower()}'",
        )
        native_status = native.status()
        component_id = str(model_denoiser_subfolder or "transformer").strip() or "transformer"
        model_component_id = snapshot_component_id(
            component_id,
            component_label="denoiser_subfolder",
        )
        text_encoder_id = (
            f"{str(model_type).strip().lower()}:{str(model_variant).strip().lower()}:"
            f"{component_id}:{native_status.text_mode or 'hash'}"
        )
        latent_vae_id = (
            f"{str(model_type).strip().lower()}:{str(model_variant).strip().lower()}:"
            f"{component_id}:{native_status.vae_mode or 'fallback'}:"
            f"{int(native.latent_channels)}"
        )
        model_is_native = bool(provider is not None and provider.is_native_model())
        allows_asset_free_cache = bool(
            provider is not None and provider.allows_asset_free_cache_encoding()
        )
        supports_native_cache = bool(
            provider is not None and provider.supports_native_cache_encoding()
        )
        if media_files and model_is_native and not supports_native_cache and not allows_asset_free_cache:
            raise ValueError(
                f"Model family '{str(model_type).strip().lower()}' does not declare "
                "native cache encoding or asset-free cache projection. Build a "
                "provider-owned native cache encoder or train from a precomputed "
                "cache artifact instead of using scripts/cache.py for raw media."
            )
        if (
            media_files
            and model_is_native
            and supports_native_cache
            and not bool(native_status.enabled)
            and not allows_asset_free_cache
        ):
            raise ValueError(
                f"Model family '{str(model_type).strip().lower()}' requires native "
                "cache encoding for raw media, but native cache encoding is disabled."
            )
        resolved_registration_path, registration_map = _load_registration_map(
            dataset_dir=dataset_dir,
            registration_path=registration_path,
        )
        if registration_map:
            media_sample_ids = {path.stem for path in media_files}
            registration_sample_ids = set(registration_map)
            missing_ids = sorted(media_sample_ids - registration_sample_ids)
            extra_ids = sorted(registration_sample_ids - media_sample_ids)
            if missing_ids or extra_ids:
                details: list[str] = []
                if missing_ids:
                    details.append(
                        "dataset media missing from registration: " + ", ".join(missing_ids)
                    )
                if extra_ids:
                    details.append(
                        "registration entries missing from dataset: " + ", ".join(extra_ids)
                    )
                joined = "; ".join(details)
                raise ValueError(
                    "Dataset registration does not match current dataset contents. "
                    f"{joined}. Re-run scripts/tools/register_dataset.py before caching."
                )
        for media_file in media_files:
            registration_sample = registration_map.get(media_file.stem)
            split_value = (
                str(registration_sample["split"])
                if registration_sample is not None
                else "train"
            )
            routing_metadata: dict[str, Any] = {}
            routing_key = str(routing_domain_metadata_key).strip()
            required_metadata_keys = {
                str(key).strip()
                for key in required_registration_metadata_keys
                if str(key).strip()
            }
            if routing_key:
                required_metadata_keys.add(routing_key)
            for metadata_key in sorted(required_metadata_keys):
                routing_value = (
                    registration_sample.get(metadata_key)
                    if registration_sample is not None
                    else None
                )
                if routing_value is None or not str(routing_value).strip():
                    raise ValueError(
                        f"Dataset registration sample '{media_file.stem}' is missing "
                        f"non-empty required metadata '{metadata_key}'."
                    )
                routing_metadata[metadata_key] = routing_value
            media_fp = media_fingerprint(media_file, strict=True)
            mask_file = media_file.with_suffix(str(mask_extension)) if mask_extension else Path()
            mask_fp = (
                media_fingerprint(mask_file, strict=True)
                if mask_extension and mask_file.exists()
                else "none"
            )
            latent_fp = latent_cache_fingerprint(
                media_fp=media_fp,
                vae_id=latent_vae_id,
                vae_dtype=str(model_dtype),
                preprocess_policy_hash=(
                    f"native_max_frames={int(native_max_frames)};"
                    f"enable_bucketing={bool(enable_bucketing)};"
                    f"bucket_resize_mode={str(bucket_resize_mode)};"
                    f"mask_fp={mask_fp};"
                    f"mask_extension={str(mask_extension)}"
                ),
            )
            _bucket_info = None
            try:
                latent_value, _bucket_info = native.encode_latent(media_file)
            except Exception as exc:
                if native_cache_encoder_contract_error(exc, native):
                    raise ValueError(
                        f"Native cache encoder {type(native).__name__} violates the "
                        f"NativeCacheEncoder contract during encode_latent(): {exc}"
                    ) from exc
                if supports_native_cache:
                    raise ValueError(
                        f"Native cache encoder failed for '{media_file}': {exc}"
                    ) from exc
                if media_file.suffix.lower() == ".pt":
                    tensor = torch.load(media_file, map_location="cpu").float()
                    latent_value = float(tensor.reshape(-1).mean().item())
                    _bucket_info = None
                else:
                    _record_skipped_sample(
                        skipped,
                        skip_reasons,
                        sample_id=media_file.stem,
                        status="skipped_decode_error",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                    continue
            if not _is_finite_payload_value(latent_value):
                _record_skipped_sample(
                    skipped,
                    skip_reasons,
                    sample_id=media_file.stem,
                    status="skipped_nan",
                    reason="non-finite latent",
                )
                continue
            clip_embed_value = None
            try:
                clip_embed_value = native.encode_clip(media_file)
            except Exception as exc:
                if native_cache_encoder_contract_error(exc, native):
                    raise ValueError(
                        f"Native cache encoder {type(native).__name__} violates the "
                        f"NativeCacheEncoder contract during encode_clip(): {exc}"
                    ) from exc
                _record_skipped_sample(
                    skipped,
                    skip_reasons,
                    sample_id=media_file.stem,
                    status="skipped_clip_error",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                continue
            if clip_embed_value is not None and not _is_finite_payload_value(clip_embed_value):
                _record_skipped_sample(
                    skipped,
                    skip_reasons,
                    sample_id=media_file.stem,
                    status="skipped_clip_nan",
                    reason="non-finite clip embedding",
                )
                continue
            loss_mask_value: float | None = None
            if mask_extension:
                if mask_file.exists():
                    mask_tensor = torch.load(mask_file, map_location="cpu").float()
                    mask_mean = float(mask_tensor.reshape(-1).mean().item())
                    if not torch.isfinite(torch.tensor(mask_mean, dtype=torch.float32)):
                        _record_skipped_sample(
                            skipped,
                            skip_reasons,
                            sample_id=media_file.stem,
                            status="skipped_mask_nan",
                            reason="non-finite loss mask",
                        )
                        continue
                    loss_mask_value = mask_mean
            captions: list[str] = []
            caption_json = media_file.with_suffix(".json")
            if caption_json.exists():
                try:
                    payload = json.loads(caption_json.read_text(encoding="utf-8"))
                    maybe = payload.get("captions")
                    if isinstance(maybe, list):
                        captions = [str(v) for v in maybe if str(v).strip() != ""]
                except Exception:
                    captions = []
            if not captions:
                caption_file = media_file.with_suffix(".txt")
                caption = (
                    caption_file.read_text(encoding="utf-8").strip()
                    if caption_file.exists()
                    else ""
                )
                captions = [caption]
            if tag_shuffle_variants > 1:
                expanded: list[str] = []
                for idx, caption in enumerate(captions):
                    expanded.append(caption)
                    for variant_idx in range(1, tag_shuffle_variants):
                        expanded.append(
                            _shuffle_caption_tags(
                                caption,
                                seed_text=f"{media_file.stem}:{idx}:{variant_idx}",
                            )
                        )
                captions = expanded

            for clip_idx in range(clips_per_video):
                for idx, caption in enumerate(captions):
                    sample_id = media_file.stem if len(captions) == 1 else f"{media_file.stem}::cap{idx}"
                    base_sample_id = sample_id
                    if clips_per_video > 1:
                        sample_id = f"{sample_id}::clip{clip_idx}"
                    # The family owns caption semantics. Resolution happens at
                    # this single seam immediately before fingerprinting and
                    # encoding, so cache lineage records the transformed text.
                    if provider is not None:
                        caption = provider.resolve_dataset_caption(
                            caption,
                            caption_format=caption_format_normalized,
                        )
                    media_text_encoder = getattr(native, "encode_text_for_media", None)
                    text_policy_version = "cache-text-v1"
                    if callable(media_text_encoder):
                        text_policy_version = f"{text_policy_version};media_fp={media_fp}"
                    # caption_format is folded into the text fingerprint's policy
                    # so flipping the key invalidates text embeds (regen, no silent
                    # stale reuse). Only appended when non-default, keeping "raw"
                    # caches byte-identical to pre-feature fingerprints.
                    if caption_format_normalized != "raw":
                        text_policy_version = (
                            f"{text_policy_version};caption_format={caption_format_normalized}"
                        )
                    text_fp = text_cache_fingerprint(
                        caption=caption,
                        encoder_id=text_encoder_id,
                        policy_version=text_policy_version,
                        encoder_dtype=str(model_dtype),
                    )
                    current_fingerprints = {
                        "latent": latent_fp,
                        "text": text_fp,
                    }
                    text_mask_override = None
                    if callable(media_text_encoder):
                        encoded_text = media_text_encoder(caption, media_file)
                    else:
                        encoded_text = native.encode_text(caption)
                    if isinstance(encoded_text, tuple) and len(encoded_text) == 2:
                        text_embed, text_mask_override = encoded_text
                    else:
                        text_embed = encoded_text
                    if isinstance(text_embed, torch.Tensor):
                        mask_len = int(text_embed.shape[0]) if text_embed.ndim >= 2 else 1
                        default_text_mask = [1] * max(1, mask_len)
                    else:
                        default_text_mask = [1]
                    if text_mask_override is not None:
                        default_text_mask = [int(v) for v in list(text_mask_override)]
                    # Unpack bucket info for tagging records.
                    _bid = _bucket_info.bucket_id if _bucket_info is not None else ""
                    _bh = _bucket_info.bucket_h if _bucket_info is not None else 0
                    _bw = _bucket_info.bucket_w if _bucket_info is not None else 0
                    _bf = _bucket_info.bucket_frames if _bucket_info is not None else 0

                    recovered = recovered_map.get(sample_id)
                    if recovered is not None:
                        recovered_fingerprints = _validate_fingerprints(
                            recovered.get("fingerprints"),
                            source_label=f"recovered cache sample '{sample_id}'",
                        )
                        plan = selective_rebuild_plan(
                            recovered_fingerprints or {},
                            current_fingerprints,
                        )
                        recovered_loss_mask = recovered.get("loss_mask", loss_mask_value)
                        recovered_count += 1
                        records.append(
                            CachedRecord(
                                sample_id=sample_id,
                                latent=(
                                    latent_value
                                    if plan.get("rebuild_latent", True)
                                    else recovered.get("latent", latent_value)
                                ),
                                text_embed=(
                                    text_embed
                                    if plan.get("rebuild_text", True)
                                    else recovered.get("text_embed", text_embed)
                                ),
                                clip_embed=(
                                    clip_embed_value
                                    if plan.get("rebuild_latent", True)
                                    else recovered.get("clip_embed", clip_embed_value)
                                ),
                                caption_variant_index=int(recovered.get("caption_variant_index", idx)),
                                caption=str(recovered.get("caption", caption)),
                                base_sample_id=str(recovered.get("base_sample_id", base_sample_id)),
                                clip_index=int(recovered.get("clip_index", clip_idx)),
                                text_mask=list(recovered.get("text_mask", default_text_mask)),
                                loss_mask=(
                                    float(loss_mask_value)
                                    if plan.get("rebuild_latent", True) and loss_mask_value is not None
                                    else float(recovered_loss_mask)
                                    if recovered_loss_mask is not None
                                    else None
                                ),
                                split=split_value,
                                fingerprints=current_fingerprints,
                                bucket_id=str(recovered.get("bucket_id", _bid)),
                                bucket_h=int(recovered.get("bucket_h", _bh)),
                                bucket_w=int(recovered.get("bucket_w", _bw)),
                                bucket_frames=int(recovered.get("bucket_frames", _bf)),
                                metadata=dict(routing_metadata),
                            )
                        )
                        continue
                    records.append(
                        CachedRecord(
                            sample_id=sample_id,
                            latent=latent_value,
                            text_embed=text_embed,
                            clip_embed=clip_embed_value,
                            caption_variant_index=idx,
                            caption=caption,
                            base_sample_id=base_sample_id,
                            clip_index=clip_idx,
                            text_mask=default_text_mask,
                            loss_mask=loss_mask_value,
                            split=split_value,
                            fingerprints=current_fingerprints,
                            bucket_id=_bid,
                            bucket_h=_bh,
                            bucket_w=_bw,
                            bucket_frames=_bf,
                            metadata=dict(routing_metadata),
                        )
                    )

        if not records:
            if skipped:
                raise ValueError(
                    "Cache encoding produced no valid records. "
                    f"{len(skipped)}/{len(media_files)} samples skipped: "
                    f"{_format_skip_breakdown(skipped, skip_reasons)}"
                )
            raise ValueError("Cache encoding produced no valid records.")

        total_seen = len(media_files)
        if total_seen > 0:
            skip_ratio = len(skipped) / total_seen
            if skip_ratio > max_cache_skip_ratio:
                raise ValueError(
                    "Cache skip ratio exceeded max_cache_skip_ratio: "
                    f"{skip_ratio:.3f} > {max_cache_skip_ratio:.3f}; skipped: "
                    f"{_format_skip_breakdown(skipped, skip_reasons)}"
                )

        native_status = native.status()
        dataset_snapshot = snapshot_descriptor_for_path(
            dataset_dir,
            manifest_candidates=("registration.json",),
        )
        payload = {
            "version": 1,
            "fp8_text_encoder": bool(fp8_text_encoder),
            "cache_mode": str(cache_mode),
            "cache_compression": str(cache_compression),
            "tag_shuffle_variants": int(tag_shuffle_variants),
            "clips_per_video": int(clips_per_video),
            "partial_recovery": bool(partial_recovery),
            "recovered_records": int(recovered_count),
            "dataset_registration_present": bool(resolved_registration_path is not None),
            "dataset_registration_path": (
                str(resolved_registration_path) if resolved_registration_path is not None else ""
            ),
            "dataset_snapshot_id": str(dataset_snapshot.snapshot_id),
            "dataset_snapshot_source_path": str(dataset_snapshot.resolved_path),
            "dataset_snapshot_source_kind": str(dataset_snapshot.source_kind),
            "dataset_snapshot_source_manifest": str(dataset_snapshot.source_manifest_path),
            "model_type": str(model_type).strip().lower(),
            "model_variant": str(model_variant).strip().lower(),
            "model_component_id": str(model_component_id),
            "model_component_label": "denoiser_subfolder",
            "denoiser_subfolder": str(component_id),
            "num_records": len(records),
            "num_skipped": len(skipped),
            "estimated_disk_bytes": estimate_cache_disk_bytes(num_records=len(records)),
            "skipped": skipped,
            "native_cache_status": {
                "enabled": bool(native_status.enabled),
                "vae_loaded": bool(native_status.vae_loaded),
                "text_loaded": bool(native_status.text_loaded),
                "clip_loaded": bool(native_status.clip_loaded),
                "vae_mode": native_status.vae_mode,
                "text_mode": native_status.text_mode,
                "clip_mode": native_status.clip_mode,
            },
            "records": [
                {
                    "sample_id": r.sample_id,
                    "latent": r.latent,
                    "text_embed": r.text_embed,
                    "clip_embed": r.clip_embed,
                    "caption_variant_index": r.caption_variant_index,
                    "caption": r.caption,
                    "base_sample_id": r.base_sample_id,
                    "clip_index": int(r.clip_index),
                    "text_mask": r.text_mask if r.text_mask is not None else [1],
                    "loss_mask": r.loss_mask,
                    "split": r.split,
                    "fingerprints": dict(r.fingerprints or {}),
                    "bucket_id": r.bucket_id,
                    "bucket_h": r.bucket_h,
                    "bucket_w": r.bucket_w,
                    "bucket_frames": r.bucket_frames,
                    "metadata": dict(r.metadata or {}),
                }
                for r in records
            ],
        }
        if cache_mode.strip().lower() == "disk":
            if cache_compression.strip().lower() == "zstd" and cache_file.suffix.lower() != ".zst":
                raise ValueError(
                    "cache_compression='zstd' requires cache_path with '.zst' extension."
                )
            _atomic_torch_save(
                cache_file,
                payload,
                cache_compression=cache_compression,
            )
            _write_checksum(cache_file)
            try:
                indexed_sidecar = save_indexed_cache_sidecar(
                    cache_file,
                    list(payload.get("records", [])),
                    metadata_extra={
                        "dataset_snapshot_id": str(payload.get("dataset_snapshot_id", "")),
                        "dataset_snapshot_source_path": str(
                            payload.get("dataset_snapshot_source_path", "")
                        ),
                        "dataset_snapshot_source_kind": str(
                            payload.get("dataset_snapshot_source_kind", "")
                        ),
                        "dataset_snapshot_source_manifest": str(
                            payload.get("dataset_snapshot_source_manifest", "")
                        ),
                        "model_type": str(payload.get("model_type", "")),
                        "model_variant": str(payload.get("model_variant", "")),
                        "model_component_id": str(payload.get("model_component_id", "")),
                        "model_component_label": str(
                            payload.get("model_component_label", "")
                        ),
                        "denoiser_subfolder": str(payload.get("denoiser_subfolder", "")),
                    },
                )
            except ModuleNotFoundError:
                indexed_sidecar = None
            payload["indexed_cache"] = {
                "enabled": bool(indexed_sidecar is not None),
                "metadata_path": (
                    str(indexed_sidecar.get("metadata_path", ""))
                    if indexed_sidecar is not None
                    else ""
                ),
                "tensor_path": (
                    str(indexed_sidecar.get("tensor_path", ""))
                    if indexed_sidecar is not None
                    else ""
                ),
            }
        elif cache_mode.strip().lower() != "ram":
            raise ValueError(f"Unsupported cache_mode '{cache_mode}'.")
        validate_cache_payload(payload, source_label=str(cache_file))
        return payload


def load_cache(cache_path: str | Path, verify_checksum: bool = True) -> dict:
    path = Path(cache_path)
    _verify_checksum(path, verify_checksum=verify_checksum)
    if path.suffix.lower() == ".zst":
        try:
            import zstandard as zstd
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Loading zstd-compressed cache requires the 'zstandard' package."
            ) from exc
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(path.read_bytes())
        payload = torch.load(io.BytesIO(raw), map_location="cpu")
    else:
        payload = torch.load(path, map_location="cpu")
    payload = migrate_cache_payload(payload)
    validate_cache_payload(payload, source_label=str(path))
    return payload


def save_tensor_cache_safetensors(path: str | Path, tensors: dict[str, torch.Tensor]) -> Path:
    from safetensors.torch import save_file

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, out)
    _write_checksum(out)
    return out


def load_tensor_cache_safetensors(
    path: str | Path, verify_checksum: bool = True
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    in_path = Path(path)
    _verify_checksum(in_path, verify_checksum=verify_checksum)
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(in_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out


class MmapSafetensorsCache:
    """Lazy, key-based safetensors accessor."""

    def __init__(self, path: str | Path, verify_checksum: bool = True):
        from safetensors import safe_open

        self._path = Path(path)
        _verify_checksum(self._path, verify_checksum=verify_checksum)
        with safe_open(str(self._path), framework="pt", device="cpu") as f:
            self._keys = tuple(f.keys())

    @property
    def path(self) -> Path:
        return self._path

    def keys(self) -> tuple[str, ...]:
        return self._keys

    def __getitem__(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(str(self._path), framework="pt", device="cpu") as f:
            if key not in f.keys():
                raise KeyError(key)
            return f.get_tensor(key)

    def get_many(self, keys: list[str]) -> list[torch.Tensor]:
        from safetensors import safe_open

        if not keys:
            return []
        with safe_open(str(self._path), framework="pt", device="cpu") as f:
            available = set(f.keys())
            missing = [key for key in keys if key not in available]
            if missing:
                raise KeyError(", ".join(missing))
            return [f.get_tensor(key) for key in keys]
