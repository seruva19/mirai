"""Downloadable sparse-MoE artifact provenance."""

from __future__ import annotations

from dataclasses import dataclass

from mirai.core.moe.artifacts.catalog import get_open_sparse_moe_model_spec


@dataclass(frozen=True)
class MoEArtifactDownloadSpec:
    variant: str
    repo_id: str
    spec_id: str
    artifact_license: str = "unknown"
    repo_kind: str = "huggingface_model"


MOE_ARTIFACT_DOWNLOAD_SPECS: tuple[MoEArtifactDownloadSpec, ...] = (
    MoEArtifactDownloadSpec(
        variant="lingbot-video-moe-30b-a3b",
        repo_id="robbyant/lingbot-video-moe-30b-a3b",
        spec_id="lingbot_video",
        artifact_license="apache-2.0",
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
    source = get_open_sparse_moe_model_spec(artifact.spec_id)
    return {
        "schema_version": 1,
        "variant": artifact.variant,
        "repo_kind": artifact.repo_kind,
        "repo_id": artifact.repo_id,
        "source_spec_id": source.model_id,
        "display_name": source.display_name,
        "native_model_type": source.native_model_type,
        "modality": source.modality,
        "routing": source.routing,
        "mirai_integration_level": source.integration_level,
        "mirai_integration_blockers": list(source.integration_blockers),
        "artifact_license": artifact.artifact_license,
        "reference_source_license": source.source_code_license,
        "catalog_license": source.license,
        "source_urls": list(source.source_urls),
    }
