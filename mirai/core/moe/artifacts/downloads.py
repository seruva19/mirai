"""Downloadable sparse-MoE artifact provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

from mirai.core.moe.artifacts.catalog import get_open_sparse_moe_model_spec


@dataclass(frozen=True)
class MoEArtifactDownloadSpec:
    variant: str
    repo_id: str
    spec_id: str
    artifact_license: str = "unknown"
    repo_kind: str = "huggingface_model"
    revision: str | None = None
    required_patterns: tuple[str, ...] = ()
    default_patterns: tuple[str, ...] = ()
    optional_source_patterns: tuple[str, ...] = ()
    expected_files: tuple["PinnedArtifactFile", ...] = ()
    auxiliary_sources: tuple["AuxiliaryArtifactSource", ...] = ()
    mirror_url: str | None = None
    mirror_revision: str | None = None
    index_key_prefixes: tuple[str, ...] = ()
    encoder_only_config: bool = False


@dataclass(frozen=True)
class PinnedArtifactFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class AuxiliaryArtifactSource:
    role: str
    repo_id: str
    revision: str
    artifact_license: str
    access: str = "public"
    download_url: str | None = None
    download_revision: str | None = None
    required_patterns: tuple[str, ...] = ()
    license_url: str | None = None


_AGNES_PART_HASHES = (
    "a2c1693a4959f23a00f9d60adbb3f42c592f849c7972e79986e63e842d1820b9",
    "0c0df20f5f2d8af0aceb4f73282ea7f873568bd9a087048e7a75d0e7a14a2e82",
    "5ecc03171fb58fab5c51f32c45146687144864864bafd6ab3db54ea8f697afc4",
    "b4f272ce1adb7b58704a6665952654c469eb85e801067cabe008a59441accd44",
    "8379d724f9dff859ad7c5141cbf263567e6c78cfb84da4092e929ea0c1e086e7",
    "333ec43dd565064e327e53241bc2c4edcc78a7fed044086305b6e74a2e8fa315",
    "097820ca62b4652cdcd52866da340ab455ff843297e5542f0936fd2453122f05",
)


_AGNES_FILES = tuple(
    PinnedArtifactFile(
        path=f"checkpoint_parts/ckpt_best.pt.part-{number:03d}",
        size=(8 * 1024**3 if number < 6 else 5_933_844_437),
        sha256=digest,
    )
    for number, digest in enumerate(_AGNES_PART_HASHES)
)


MOE_ARTIFACT_DOWNLOAD_SPECS: tuple[MoEArtifactDownloadSpec, ...] = (
    MoEArtifactDownloadSpec(
        variant="lingbot-video-moe-30b-a3b",
        repo_id="robbyant/lingbot-video-moe-30b-a3b",
        spec_id="lingbot_video",
        artifact_license="apache-2.0",
    ),
    MoEArtifactDownloadSpec(
        variant="lingbot-video-moe-dmd-30b-a3b",
        repo_id="robbyant/lingbot-video-moe-dmd-30b-a3b",
        spec_id="lingbot_video",
        artifact_license="apache-2.0",
    ),
    MoEArtifactDownloadSpec(
        variant="agnes-video-2.0-moe",
        repo_id="Agnes-AI/Agnes-Video-2.0-MoE",
        spec_id="agnes_video",
        artifact_license="apache-2.0",
        revision="fc3971e378b53c766baa06898bfe84d7a56653ea",
        required_patterns=tuple(file.path for file in _AGNES_FILES),
        optional_source_patterns=(
            "inference_code/**",
            "LICENSE",
            "NOTICE",
            "README.md",
        ),
        expected_files=_AGNES_FILES,
        auxiliary_sources=(
            AuxiliaryArtifactSource(
                role="text_encoder",
                repo_id="google/t5gemma-9b-9b-ul2",
                revision="7071a2ace9c8353e12e723e5382cc7256ee41d9f",
                artifact_license="gemma",
                access="public_mirror",
                download_url="https://gitee.com/hf-models/t5gemma-9b-9b-ul2.git",
                download_revision="7071a2ace9c8353e12e723e5382cc7256ee41d9f",
                required_patterns=(
                    "README.md",
                    "config.json",
                    "generation_config.json",
                    "model.safetensors.index.json",
                    "model-00001-of-00009.safetensors",
                    "model-00002-of-00009.safetensors",
                    "model-00003-of-00009.safetensors",
                    "model-00004-of-00009.safetensors",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer.model",
                    "tokenizer_config.json",
                ),
                license_url="https://ai.google.dev/gemma/terms",
            ),
            AuxiliaryArtifactSource(
                role="video_vae",
                repo_id="Wan-AI/Wan2.2-TI2V-5B",
                revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
                artifact_license="apache-2.0",
            ),
            AuxiliaryArtifactSource(
                role="audio_vae",
                repo_id="OpenMOSS-Team/OmniVAE",
                revision="e87bfcc1747942972680ba5cbabfea6822d79bc7",
                artifact_license="apache-2.0",
            ),
        ),
    ),
    MoEArtifactDownloadSpec(
        variant="agnes-t5gemma-9b-9b-ul2",
        repo_id="google/t5gemma-9b-9b-ul2",
        spec_id="agnes_video",
        artifact_license="gemma",
        revision="7071a2ace9c8353e12e723e5382cc7256ee41d9f",
        required_patterns=(
            "README.md",
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00009.safetensors",
            "model-00002-of-00009.safetensors",
            "model-00003-of-00009.safetensors",
            "model-00004-of-00009.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ),
        mirror_url="https://gitee.com/hf-models/t5gemma-9b-9b-ul2.git",
        mirror_revision="7071a2ace9c8353e12e723e5382cc7256ee41d9f",
        index_key_prefixes=("model.encoder.",),
        encoder_only_config=True,
    ),
    MoEArtifactDownloadSpec(
        variant="agnes-wan2.2-ti2v-5b",
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        spec_id="agnes_video",
        artifact_license="apache-2.0",
        revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
        default_patterns=("Wan2.2_VAE.pth", "LICENSE*"),
    ),
    MoEArtifactDownloadSpec(
        variant="agnes-omnivae",
        repo_id="OpenMOSS-Team/OmniVAE",
        spec_id="agnes_video",
        artifact_license="apache-2.0",
        revision="e87bfcc1747942972680ba5cbabfea6822d79bc7",
        default_patterns=("models/vae/audio_only/recon/*", "LICENSE*"),
    ),
)


def get_moe_artifact_download_specs() -> tuple[MoEArtifactDownloadSpec, ...]:
    return MOE_ARTIFACT_DOWNLOAD_SPECS


def get_moe_artifact_download_spec(variant: str) -> MoEArtifactDownloadSpec:
    key = str(variant).strip().lower()
    for spec in MOE_ARTIFACT_DOWNLOAD_SPECS:
        if key == spec.variant:
            return spec
    available = ", ".join(spec.variant for spec in MOE_ARTIFACT_DOWNLOAD_SPECS)
    raise KeyError(f"Unknown sparse-MoE artifact variant '{variant}'. Available: {available}.")


def get_download_repo_by_variant() -> dict[str, str]:
    return {spec.variant: spec.repo_id for spec in MOE_ARTIFACT_DOWNLOAD_SPECS}


def get_moe_artifact_manifest_metadata(variant: str) -> dict[str, object]:
    artifact = get_moe_artifact_download_spec(variant)
    try:
        source = get_open_sparse_moe_model_spec(artifact.spec_id)
    except KeyError:
        source = None
    metadata: dict[str, object] = {
        "schema_version": 1,
        "variant": artifact.variant,
        "repo_kind": artifact.repo_kind,
        "repo_id": artifact.repo_id,
        "source_spec_id": artifact.spec_id,
        "artifact_license": artifact.artifact_license,
        "revision": artifact.revision,
        "default_patterns": list(artifact.default_patterns),
        "mirror_url": artifact.mirror_url,
        "mirror_revision": artifact.mirror_revision,
        "index_key_prefixes": list(artifact.index_key_prefixes),
        "encoder_only_config": artifact.encoder_only_config,
        "auxiliary_sources": [
            {
                "role": item.role,
                "repo_id": item.repo_id,
                "revision": item.revision,
                "artifact_license": item.artifact_license,
                "access": item.access,
                "download_url": item.download_url,
                "download_revision": item.download_revision,
                "required_patterns": list(item.required_patterns),
                "license_url": item.license_url,
            }
            for item in artifact.auxiliary_sources
        ],
    }
    if source is not None:
        metadata.update(
            display_name=source.display_name,
            native_model_type=source.native_model_type,
            modality=source.modality,
            routing=source.routing,
            mirai_integration_level=source.integration_level,
            mirai_integration_blockers=list(source.integration_blockers),
            reference_source_license=source.source_code_license,
            catalog_license=source.license,
            source_urls=list(source.source_urls),
        )
    return metadata


def artifact_allow_patterns(
    variant: str,
    requested: Sequence[str] = (),
    *,
    include_source_code: bool = False,
) -> tuple[str, ...] | None:
    artifact = get_moe_artifact_download_spec(variant)
    patterns = list(artifact.required_patterns)
    requested_patterns = [str(item) for item in requested if str(item).strip()]
    patterns.extend(requested_patterns or artifact.default_patterns)
    if include_source_code:
        patterns.extend(artifact.optional_source_patterns)
    return tuple(dict.fromkeys(patterns)) or None


def verify_pinned_artifact_files(root: str | Path, variant: str) -> None:
    artifact = get_moe_artifact_download_spec(variant)
    base = Path(root).resolve()
    for expected in artifact.expected_files:
        relative = Path(expected.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Artifact registry contains unsafe path: {expected.path!r}.")
        path = base / relative
        if not path.is_file():
            raise FileNotFoundError(f"Pinned artifact file missing: {path}.")
        size = path.stat().st_size
        if size != expected.size:
            raise ValueError(
                f"Pinned artifact size mismatch for {expected.path}: "
                f"expected {expected.size}, observed {size}."
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(chunk)
        observed = digest.hexdigest()
        if observed != expected.sha256:
            raise ValueError(
                f"Pinned artifact SHA-256 mismatch for {expected.path}: "
                f"expected {expected.sha256}, observed {observed}."
            )


def verify_artifact_manifest_source(
    root: str | Path,
    variant: str,
    *,
    manifest_name: str = "download_manifest.json",
) -> None:
    from mirai.core.moe.artifacts.manifest import read_validated_download_manifest

    artifact = get_moe_artifact_download_spec(variant)
    manifest = read_validated_download_manifest(Path(root) / manifest_name)
    if manifest.get("repo_id") != artifact.repo_id:
        raise ValueError(
            f"Artifact repository mismatch: expected {artifact.repo_id!r}, "
            f"got {manifest.get('repo_id')!r}."
        )
    if manifest.get("revision") != artifact.revision:
        raise ValueError(
            f"Artifact revision mismatch: expected {artifact.revision!r}, "
            f"got {manifest.get('revision')!r}."
        )
