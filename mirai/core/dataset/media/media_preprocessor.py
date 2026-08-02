"""Media-side preprocessing helpers for cache-ready training assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mirai.core.dataset.native_encode import VIDEO_MEDIA_SUFFIXES

@dataclass(frozen=True)
class MediaPreprocessResult:
    generated: int
    skipped_existing: int
    skipped_unsupported: int


def _iter_raw_media_files(
    dataset_dir: Path,
) -> tuple[list[Path], set[str]]:
    raw_media: list[Path] = []
    latent_stems: set[str] = set()
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pt":
            latent_stems.add(path.stem)
            continue
        if suffix in VIDEO_MEDIA_SUFFIXES:
            raw_media.append(path)
    return raw_media, latent_stems


def preprocess_video_media_to_pt(
    dataset_path: str | Path,
    *,
    latent_channels: int,
    max_frames: int,
) -> MediaPreprocessResult:
    _ = latent_channels, max_frames
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise ValueError(f"Preprocess dataset path does not exist: {dataset_dir}")
    raw_media, latent_stems = _iter_raw_media_files(dataset_dir)
    if not raw_media:
        return MediaPreprocessResult(
            generated=0,
            skipped_existing=0,
            skipped_unsupported=0,
        )
    unresolved = [path for path in raw_media if path.stem not in latent_stems]
    if unresolved:
        raise RuntimeError(
            "Raw video preprocessing requires a model-family native cache encoder; "
            "generic pixel projection is not a valid diffusion latent."
        )
    return MediaPreprocessResult(
        generated=0,
        skipped_existing=len(raw_media),
        skipped_unsupported=0,
    )
