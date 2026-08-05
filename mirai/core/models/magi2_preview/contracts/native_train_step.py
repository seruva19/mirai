"""Executable reduced-shape MAGI-2 native forward/backward probe."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy
from mirai.vendors.magi2_preview.common.magi2_config import (
    DataProxyConfig,
    MHCConfig,
    ModelConfig,
    MoEConfig,
)
from mirai.vendors.magi2_preview.model.magi2_preview import Transformer
from mirai.vendors.magi2_preview.pipeline.preview_data_proxy import Magi2DataProxy, ModelInput


def _train_step(
    model: torch.nn.Module, pipeline: Magi2PreviewPipeline, packed: tuple
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    for parameter in pipeline.get_trainable_parameters():
        parameter.grad = None
    output = model(*packed)
    loss = output.float().square().mean()
    loss.backward()
    return (
        output.detach(),
        loss.detach(),
        tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in pipeline.get_trainable_parameters()
        ),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MAGI-2 native train-step probe requires CUDA.")
    device = torch.device("cuda")
    config = ModelConfig(
        num_layers=3,
        hidden_size=256,
        head_dim=64,
        num_query_groups=4,
        video_in_channels=48,
        audio_in_channels=64,
        text_in_channels=5120,
        intermediate_factor=2,
        mm_layers=[0, 1],
        mhc_config=MHCConfig(enable=True, num_stream=4),
        moe_config=MoEConfig(
            num_experts=4,
            top_k=2,
            moe_layers=[2],
            expert_intermediate_size=64,
            shared_expert_intermediate_size=64,
            modality_specific_expert_intermediate_size=64,
            num_heads=4,
        ),
    )
    config.num_heads_q = config.hidden_size // config.head_dim
    config.num_heads_kv = config.num_query_groups
    model = Transformer(config, ep_size=1)
    model.initialize_for_skip_load()
    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = model
    pipeline._lora_scale = 1.0
    pipeline.set_adapter_config(
        SimpleNamespace(type="lora", target_preset="attn_router", rank=2, alpha=2.0)
    )
    model.block.gradient_checkpointing = True
    model.to(device=device).train()
    proxy = Magi2DataProxy(DataProxyConfig())
    latent = torch.randn(1, 48, 1, 2, 2, device=device, dtype=torch.float32)
    packed = proxy.process_input(
        ModelInput(
            x_t=latent,
            audio_x_t=torch.zeros(1, 0, 64, device=device, dtype=torch.bfloat16),
            audio_feat_len=torch.zeros(1, device=device, dtype=torch.long),
            txt_feat=torch.randn(1, 2, 5120, device=device),
            txt_feat_len=torch.full((1,), 2, device=device, dtype=torch.long),
            t=torch.full((1,), 0.5, device=device),
            per_token_video_t=torch.full(
                (1, 1, 1, 2, 2), 0.5, device=device, dtype=torch.bfloat16
            ),
            per_token_audio_t=torch.empty(
                (1, 0, 1), device=device, dtype=torch.bfloat16
            ),
        )
    )
    output, loss, gradients = _train_step(model, pipeline, packed)
    if any(value is None or not torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("MAGI-2 native adapter gradients are missing or non-finite.")

    # Grouped expert execution must reproduce the vendored reference loop.
    pipeline.configure_moe_optimization_policy(
        MoEOptimizationPolicy(kernel_backend="grouped")
    )
    grouped_output, grouped_loss, grouped_gradients = _train_step(
        model, pipeline, packed
    )
    if not torch.allclose(
        output.float(), grouped_output.float(), rtol=2e-2, atol=2e-3
    ):
        raise RuntimeError("MAGI-2 grouped MoE output diverges from the reference loop.")
    if abs(float(loss) - float(grouped_loss)) > 2e-3 * max(1.0, abs(float(loss))):
        raise RuntimeError("MAGI-2 grouped MoE loss diverges from the reference loop.")
    for reference, grouped in zip(gradients, grouped_gradients):
        if grouped is None or not torch.allclose(
            reference.float(), grouped.float(), rtol=2e-2, atol=2e-3
        ):
            raise RuntimeError(
                "MAGI-2 grouped MoE adapter gradients diverge from the reference loop."
            )
    pipeline.configure_moe_optimization_policy(MoEOptimizationPolicy())

    adapter_state = pipeline.state_dict()
    pipeline.set_lora_scale(0.25)
    if pipeline.get_lora_scale() != 0.25:
        raise RuntimeError("MAGI-2 inference adapter scale was not applied.")
    pipeline.load_adapter_state(adapter_state)
    print(
        json.dumps(
            {
                "status": "passed",
                "loss": float(loss.detach()),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "output_shape": list(output.shape),
                "adapter_tensors": len(adapter_state),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
