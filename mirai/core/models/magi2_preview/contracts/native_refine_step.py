"""Executable reduced-shape MAGI-2 refiner stage probe.

The refiner is the family's second inference stage and cannot be exercised off
an accelerator: the vendored module imports FlashAttention at import time and
its window attention is device-only. This probe therefore owns the parts of the
stage that only a real forward can answer — that the released zero-audio
conditioning packs and runs, and that the resampled latent geometry survives the
data proxy — while the geometry and re-noise math are checked on CPU by
``tests/test_magi2_preview_contract.py``.

Which attention path serves the run is part of what the probe reports. The
refiner's window attention dispatches through ``torch.ops.magi2``, but a bound
native FlexAttention backend serves that operator itself, so the probe resolves
a backend first and then requires only the operators the reduced architecture
still reaches. Its layers are all local, so a bound backend leaves none.

``MIRAI_MAGI2_REFINER_ATTENTION_BACKEND`` selects the path, defaulting to
``auto``. The environment is the surface here because the probe is invoked from
``agent/checks.json`` as a fixed argv with no config file behind it, and the
selection has to be forceable to A/B the native path against the vendored one on
the same machine. It stays scoped to this script: nothing in the runtime reads
it, and the shipped control surface remains
``model.params.family_params.refiner_attention_backend``.
"""

from __future__ import annotations

import json
import os

import torch

from mirai.core.models.magi2_preview.refiner import (
    magi2_refiner_latent_frames,
    magi2_refiner_renoise,
    magi2_refiner_upsample,
)
from mirai.core.models.magi2_preview.refiner_attention import (
    MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT,
    attach_refiner_attention_backend,
    normalize_refiner_attention_backend,
    refiner_required_magi2_ops,
    resolve_magi2_refiner_attention,
    validate_refiner_flex_support,
)


# Probe-scoped override of the attention path, so the native and vendored paths
# can be forced on one machine. Absent, the released default resolution applies.
MAGI2_REFINER_ATTENTION_BACKEND_ENV = "MIRAI_MAGI2_REFINER_ATTENTION_BACKEND"


def _requested_attention_backend() -> str:
    return normalize_refiner_attention_backend(
        os.environ.get(
            MAGI2_REFINER_ATTENTION_BACKEND_ENV,
            MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT,
        )
    )


def _reduced_refiner_config():
    from mirai.vendors.magi2_preview.common.magi2_config import Magi2RefinerModelConfig

    config = Magi2RefinerModelConfig(
        num_layers=2,
        hidden_size=256,
        head_dim=64,
        num_query_groups=2,
        video_in_channels=48,
        audio_in_channels=64,
        text_in_channels=5120,
        mm_layers=[0],
        local_attn_layers=[0, 1],
        enable_attn_gating=True,
        post_norm_layers=[],
        activation_type="swiglu7",
    )
    config.num_heads_q = config.hidden_size // config.head_dim
    config.num_heads_kv = config.num_query_groups
    return config


def _reduced_proxy_config():
    from mirai.vendors.magi2_preview.common.magi2_config import (
        Magi2RefinerDataProxyConfig,
    )

    return Magi2RefinerDataProxyConfig(
        t_patch_size=1,
        patch_size=1,
        frame_receptive_field=11,
        spatial_rope_interpolation="extra",
        coords_style="v1",
        text_offset=0,
        magi2_refiner_condition_input="none",
        attn_config={
            "mode": "window",
            "block_t_size": 8,
            "block_size": 4,
            "window": {
                "level": "block",
                "block_mode": "grid",
                "block_t_radius": 2,
                "block_h_radius": 2,
                "block_w_radius": 2,
                "win_size": 384,
                "frame_receptive_field": -1,
                "auto_range_merge": True,
                "sparse_load": False,
                "full_attn_layers": [],
            },
        },
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MAGI-2 refine-stage probe requires CUDA.")
    from mirai.vendors.magi2_preview.common.magi_compiler_compat import (
        require_magi2_custom_ops,
    )

    requested_backend = _requested_attention_backend()
    arch_config = _reduced_refiner_config()
    backend = resolve_magi2_refiner_attention(requested_backend)
    if backend is not None:
        validate_refiner_flex_support(torch.device("cuda"))
    required_ops = refiner_required_magi2_ops(arch_config, backend)
    if required_ops:
        require_magi2_custom_ops("The MAGI-2 refine-stage probe", required_ops)

    from mirai.vendors.magi2_preview.model.magi2_refiner import Transformer
    from mirai.vendors.magi2_preview.pipeline.inference_engine import EvalInput
    from mirai.vendors.magi2_preview.pipeline.refiner_data_proxy import (
        Magi2RefinerDataProxy,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = Transformer(arch_config)
    # The refiner is normally strict-loaded from released shards, so its module
    # tree allocates uninitialized storage. A probe that never reads a
    # checkpoint has to seed it before any finiteness claim is meaningful.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.02)
    model.to(device=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    attach_refiner_attention_backend(model, backend)

    proxy = Magi2RefinerDataProxy(_reduced_proxy_config())
    preview_latent_frames = 3
    preview = torch.randn(1, 48, preview_latent_frames, 4, 4, device=device)
    latents = magi2_refiner_upsample(preview, latent_height=4, latent_width=4)
    expected_frames = magi2_refiner_latent_frames(preview_latent_frames)
    if int(latents.shape[2]) != expected_frames:
        raise RuntimeError(
            f"MAGI-2 refiner resample produced {int(latents.shape[2])} latent "
            f"frames, expected {expected_frames}."
        )
    latents = magi2_refiner_renoise(latents, torch.randn_like(latents), 0.8389104)

    # The released profile refines video with no audio track at all.
    empty_audio = torch.zeros(1, 0, 64, device=device, dtype=torch.float32)
    text = torch.randn(1, 2, 5120, device=device)
    packed = proxy.process_input(
        EvalInput(
            x_t=latents,
            audio_x_t=empty_audio,
            audio_feat_len=torch.tensor([0], device=device),
            txt_feat=text,
            txt_feat_len=torch.tensor([int(text.shape[1])], device=device),
            ref_audio_feat=empty_audio,
            ref_audio_feat_len=torch.tensor([0], device=device),
            ref_video_feat=torch.empty_like(latents),
            ref_video_feat_len=torch.tensor([0], device=device),
        )
    )
    with torch.inference_mode():
        prediction = model(*packed)
        video_velocity, audio_velocity = proxy.process_output(prediction)

    if tuple(video_velocity.shape) != tuple(latents.shape):
        raise RuntimeError(
            "MAGI-2 refiner velocity shape "
            f"{tuple(video_velocity.shape)} does not match its latent "
            f"{tuple(latents.shape)}."
        )
    if not torch.isfinite(video_velocity).all():
        raise RuntimeError("MAGI-2 refiner velocity is non-finite.")
    if audio_velocity is not None and audio_velocity.numel() > 0:
        raise RuntimeError(
            "MAGI-2 refiner returned audio velocity for an empty audio track."
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "attention_backend": requested_backend,
                "attention_path": (
                    "vendor_eager" if backend is None else "native_flex"
                ),
                "required_magi2_ops": list(required_ops),
                "preview_latent_frames": preview_latent_frames,
                "refined_latent_frames": expected_frames,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "velocity_shape": list(video_velocity.shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
