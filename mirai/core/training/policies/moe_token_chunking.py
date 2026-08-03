"""Config-driven single-GPU MoE token-chunk checkpointing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.moe.runtime.token_chunking import MoETokenChunkPolicy
from mirai.core.training.training_policy import (
    TrainingPolicy,
    register_training_policy,
)


@dataclass
class MoETokenChunkTrainingPolicy(TrainingPolicy):
    """Install token-axis chunk recomputation through a provider capability."""

    controller: MoETokenChunkPolicy
    name = "moe_token_chunking"
    priority = 45

    def configure_pipeline(self, pipeline: Any) -> None:
        capabilities = pipeline.get_sparse_moe_capabilities()
        if not bool(capabilities.is_sparse_moe):
            raise ValueError(
                "training.moe_token_chunk_size requires a native sparse-MoE model."
            )
        pipeline.configure_moe_token_chunking(self.controller)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {"token_chunk_size": int(self.controller.token_chunk_size)}


def validate_moe_token_chunking_config(config: Any) -> list[str]:
    size = int(getattr(config.training, "moe_token_chunk_size", 0))
    errors: list[str] = []
    if size < 0:
        errors.append("training.moe_token_chunk_size must be >= 0.")
    if size > 0 and float(
        getattr(config.adapter, "lora_parameter_dropout", 0.0)
    ) > 0.0:
        errors.append(
            "training.moe_token_chunk_size is incompatible with "
            "adapter.lora_parameter_dropout because parameter masks must be "
            "shared by every token in one forward."
        )
    return errors


@register_training_policy(
    "moe_token_chunking",
    validate_config=validate_moe_token_chunking_config,
)
def build_moe_token_chunking_training_policy(
    config: Any,
) -> MoETokenChunkTrainingPolicy | None:
    size = int(getattr(config.training, "moe_token_chunk_size", 0))
    if size <= 0:
        return None
    return MoETokenChunkTrainingPolicy(
        controller=MoETokenChunkPolicy(token_chunk_size=size)
    )


__all__ = [
    "MoETokenChunkTrainingPolicy",
    "build_moe_token_chunking_training_policy",
    "validate_moe_token_chunking_config",
]
