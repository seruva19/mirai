"""Known sparse/MoE diffusion model sources and integration status."""

from __future__ import annotations

from mirai.core.moe.routing.contracts import OpenSparseMoEModelSpec


OPEN_SPARSE_MOE_MODELS: tuple[OpenSparseMoEModelSpec, ...] = (
    OpenSparseMoEModelSpec(
        model_id="lingbot_video",
        display_name="LingBot-Video MoE",
        modality="video",
        status="code_and_weights_available",
        routing="token_choice_top_k_group_limited",
        source_urls=(
            "https://github.com/Robbyant/lingbot-video",
            "https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b",
        ),
        license="apache-2.0",
        source_code_license="apache-2.0",
        artifact_license="apache-2.0",
        native_model_type="lingbot-video",
        integration_level="native_training_and_inference",
        integration_blockers=(),
        notes=(
            "Apache-2.0 sparse-MoE video family integrated through native Mirai modules.",
        ),
    ),
)


def get_open_sparse_moe_model_specs() -> tuple[OpenSparseMoEModelSpec, ...]:
    return OPEN_SPARSE_MOE_MODELS


def get_open_sparse_moe_model_spec(model_id: str) -> OpenSparseMoEModelSpec:
    key = str(model_id).strip().lower().replace("-", "_")
    for spec in OPEN_SPARSE_MOE_MODELS:
        aliases = {
            spec.model_id,
            spec.native_model_type.replace("-", "_"),
            spec.native_model_type.replace("-", ""),
        }
        if key in aliases:
            return spec
    available = ", ".join(spec.model_id for spec in OPEN_SPARSE_MOE_MODELS)
    raise KeyError(f"Unknown sparse MoE model spec '{model_id}'. Available: {available}.")
