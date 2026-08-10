"""Provider-owned MAGI-2 refiner stage.

The refiner is a SEPARATE checkpoint of a DIFFERENT architecture class from the
preview denoiser: a dense 30-layer transformer
(:class:`mirai.vendors.magi2_preview.model.magi2_refiner.Transformer`) with no
routed experts, living in the ``refiner/`` subfolder of the released snapshot.
It consumes the finished preview latent rather than re-running generation.

Stage shape, mirroring SandAI's reference ``evaluate_magi2_refiner_with_latent``
(``mirai/vendors/magi2_preview/pipeline/inference_engine.py``):

1. The preview latent ``[C, T, H, W]`` is resampled by trilinear interpolation
   to ``[C, 2T - 1, H', W']``. The temporal rule is ``2T - 1``, not a factor-two
   resize, and ``align_corners=True`` is load-bearing: it pins the first and
   last preview latent frames to the first and last refiner frames, so the
   inserted frames are true midpoints of neighbouring preview frames.
2. The resampled latent is re-noised once, variance-preserving, at a FIXED
   index into a zero-terminal-SNR ``sqrt(alphas_cumprod)`` table:
   ``x * sigma + n * sqrt(1 - sigma^2)``. ``sigma`` is the SIGNAL coefficient,
   so a larger index means less signal.
3. A short Flow-UniPC denoise runs over the full timestep grid.

The preview transformer is positioned at temporal stride 8 while the shared
Turbo VAE decoder expands ``4 * (T - 1) + 1`` frames, which is why a preview-only
clip plays at half rate. The refiner's own stride is 4
(``magi2_refiner_vae_stride = [4, 16, 16]``), and ``2T - 1`` latent frames decode
to ``4 * (2T - 2) + 1 == 8 * (T - 1) + 1`` physical frames — exactly the frame
count the request denotes on its 25 fps timeline. Refining is therefore what
makes the file play at 25 fps rather than 12.5.

This module owns two things:

1. :class:`Magi2Refiner` — the on-demand lifecycle of the refiner transformer
   (build from the vendored refiner architecture JSON, strict-load the
   ``refiner/`` shards, place it on the compute device through the family's own
   block-residency mechanism, release afterwards).
2. :func:`run_refine` — the family-owned inference *policy* loop. It reaches the
   model only through the pipeline's provider hooks
   (``release_base_transformer`` / ``load_text_encoder`` / ``encode_prompt`` /
   ``offload_text_encoder`` / ``refiner_forward``), never model internals.

The pure geometry and re-noise math (:func:`magi2_refiner_latent_frames`,
:func:`magi2_refiner_upsample`, :func:`magi2_refiner_renoise`) is unit-tested on
CPU without any released weights.

Attribution: SandAI MAGI-2-preview, Apache-2.0
(https://github.com/SandAI-org/MAGI-2-preview).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from mirai.core.models.magi2_preview.refiner_attention import (
    MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT,
    normalize_refiner_attention_backend,
)


# Subfolder of the released snapshot holding the refiner shards. The upstream
# config states it as ``${MAGI2_CKPT_ROOT}/refiner``; the path is resolved from
# ``model.path`` here instead of from the environment.
MAGI2_REFINER_SUBFOLDER = "refiner"

# The refiner's own temporal VAE stride (``magi2_refiner_vae_stride[0]``). The
# preview stride is 8, so the ``2T - 1`` resample is what lands the latent on the
# decoder's native grid.
MAGI2_REFINER_VAE_TEMPORAL_STRIDE = 4

# Playback rate of a refined clip: the decoder emits the full requested frame
# count instead of the preview's half-rate expansion.
MAGI2_REFINER_OUTPUT_FPS = 25.0
MAGI2_RELEASED_REFINER_HEIGHT = 1088
MAGI2_RELEASED_REFINER_WIDTH = 1920
# Bound the refiner MLP's temporary up-projection by evaluating independent
# token rows in deterministic chunks.
MAGI2_REFINER_MLP_CHUNK_TOKENS = 16_384

# The vendored refiner sampler is Flow-UniPC, as the preview sampler is.
MAGI2_REFINER_SOLVER = "unipc"

# Length of the zero-terminal-SNR discretization the re-noise index addresses.
MAGI2_REFINER_SIGMA_TABLE_SIZE = 1000

_MAGI2_REFINER_ASSET_HINT = (
    "The MAGI-2 refiner is a separate checkpoint in the 'refiner/' subfolder of "
    "the released snapshot, holding model.safetensors.index.json and its shards. "
    "Fetch it from the release and point model.path at a snapshot root that "
    "contains it."
)


class _ModalityChunkDispatcher:
    """Minimal dispatcher view for one contiguous modality/token chunk."""

    def __init__(self, *, modality: int, tokens: int, modalities: int) -> None:
        self.group_size_cpu = [0] * int(modalities)
        self.group_size_cpu[int(modality)] = int(tokens)

    def dispatch(self, value: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(value, self.group_size_cpu, dim=0))

    @staticmethod
    def undispatch(*groups: torch.Tensor) -> torch.Tensor:
        return torch.cat(groups, dim=0)


@dataclass
class _PreparedRefinerBranch:
    x: torch.Tensor
    rope: torch.Tensor
    permute_mapping: torch.Tensor
    inv_permute_mapping: torch.Tensor
    varlen_handler: Any
    local_attn_handler: Any
    modality_dispatcher: Any
    cp_split_sizes: list[int]
    video_mask: torch.Tensor
    audio_mask: torch.Tensor


def _prepare_refiner_branch(
    transformer: Any, packed: tuple[Any, Any, Any, Any, Any]
) -> _PreparedRefinerBranch:
    """Run one branch through dispatch and the input adapter."""
    from mirai.vendors.magi2_preview.infra.distributed import psm
    from mirai.vendors.magi2_preview.model.magi2_refiner import (
        CompactRefinerTokens,
        Modality,
        ModalityDispatcher,
        embed_compact_refiner_tokens,
    )

    x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler = packed
    if isinstance(x, CompactRefinerTokens):
        x = embed_compact_refiner_tokens(transformer.pre_adapter, x)
    else:
        raise TypeError("Paired MAGI-2 refiner execution requires compact tokens.")
    if int(psm.get_world_size("cp")) != 1:
        raise RuntimeError(
            "Paired MAGI-2 refiner execution supports the single-GPU release surface only."
        )
    modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
    permute_mapping = modality_dispatcher.permute_mapping
    inv_permute_mapping = modality_dispatcher.inv_permute_mapping
    video_mask = modality_mapping == Modality.VIDEO
    audio_mask = modality_mapping == Modality.AUDIO
    rope = transformer.pre_adapter.rope(coords_mapping)
    x = ModalityDispatcher.permute(
        x.to(transformer.config.params_dtype), permute_mapping
    )
    return _PreparedRefinerBranch(
        x=x,
        rope=rope,
        permute_mapping=permute_mapping,
        inv_permute_mapping=inv_permute_mapping,
        varlen_handler=varlen_handler,
        local_attn_handler=local_attn_handler,
        modality_dispatcher=modality_dispatcher,
        cp_split_sizes=[int(x.shape[0])],
        video_mask=video_mask,
        audio_mask=audio_mask,
    )


def _finish_refiner_branch(transformer: Any, branch: _PreparedRefinerBranch) -> torch.Tensor:
    """Apply inverse modality order and the output adapter for one branch."""
    from mirai.vendors.magi2_preview.model.magi2_refiner import ModalityDispatcher

    value = ModalityDispatcher.inv_permute(branch.x, branch.inv_permute_mapping)
    value = transformer.post_adapter(value, branch.video_mask, branch.audio_mask)
    return value


def _refiner_mlp_forward(
    module: Any, value: torch.Tensor, modality_dispatcher: Any
) -> torch.Tensor:
    """The vendored MLP's exact operations, factored for bounded token chunks."""
    value = module.pre_norm(
        value, modality_dispatcher=modality_dispatcher
    ).to(torch.bfloat16)
    value = module.up_gate_proj(
        value, modality_dispatcher=modality_dispatcher
    ).to(torch.float32)
    value = module.activation_func(value).to(torch.bfloat16)
    return module.down_proj(
        value, modality_dispatcher=modality_dispatcher
    ).to(torch.float32)


@torch.compiler.disable
def _chunked_refiner_mlp_forward(
    module: Any, value: torch.Tensor, modality_dispatcher: Any
) -> torch.Tensor:
    """Evaluate MLP rows in chunks while preserving modality-expert routing."""
    chunk_tokens = int(module._mirai_chunk_tokens)
    if int(value.shape[0]) <= chunk_tokens:
        return _refiner_mlp_forward(module, value, modality_dispatcher)

    modalities = int(module.pre_norm.num_modality)
    if modalities > 1:
        group_sizes = [int(size) for size in modality_dispatcher.group_size_cpu]
    else:
        group_sizes = [int(value.shape[0])]
    if value.is_cuda:
        # Layer rotation produces allocator segments with different shapes.
        # Flush at this full-output boundary so the next contiguous output can
        # reuse them; chunk arithmetic and values are unchanged.
        torch.cuda.empty_cache()
    output = torch.empty(
        (int(value.shape[0]), int(module.down_proj.out_features)),
        device=value.device,
        dtype=torch.float32,
    )
    group_offset = 0
    for modality, group_tokens in enumerate(group_sizes):
        group_end = group_offset + group_tokens
        for start in range(group_offset, group_end, chunk_tokens):
            end = min(start + chunk_tokens, group_end)
            dispatcher = (
                _ModalityChunkDispatcher(
                    modality=modality,
                    tokens=end - start,
                    modalities=modalities,
                )
                if modalities > 1
                else modality_dispatcher
            )
            chunk = _refiner_mlp_forward(module, value[start:end], dispatcher)
            output[start:end].copy_(chunk)
            del chunk
        group_offset = group_end
    return output


def attach_refiner_mlp_chunking(
    transformer: Any, *, chunk_tokens: int = MAGI2_REFINER_MLP_CHUNK_TOKENS
) -> int:
    """Arm bounded-memory MLP execution on every released refiner layer."""
    from mirai.vendors.magi2_preview.model.magi2_refiner import MLP

    resolved = int(chunk_tokens)
    if resolved < 1:
        raise ValueError("MAGI-2 refiner MLP chunk size must be >= 1 token.")
    attached = 0
    for module in transformer.modules():
        if not isinstance(module, MLP):
            continue
        module._mirai_chunk_tokens = resolved
        module.forward = MethodType(_chunked_refiner_mlp_forward, module)
        attached += 1
    return attached


@torch.compiler.disable
def _chunked_refiner_attention_forward(
    module: Any,
    hidden_states: torch.Tensor,
    rope: torch.Tensor,
    permute_mapping: torch.Tensor,
    inv_permute_mapping: torch.Tensor,
    varlen_handler: Any,
    local_attn_handler: Any,
    modality_dispatcher: Any,
    cp_split_sizes: list[int],
) -> torch.Tensor:
    """Bound full-token QKV/rotary temporaries on the single-GPU path."""
    del inv_permute_mapping, varlen_handler
    from mirai.vendors.magi2_preview.model.magi2_refiner import (
        apply_rotary_emb,
        flash_attn_with_cp,
        flex_flash_attn_with_cp,
    )

    tokens = int(hidden_states.shape[0])
    chunk_tokens = int(module._mirai_chunk_tokens)
    modalities = int(module.pre_norm.num_modality)
    group_sizes = (
        [int(size) for size in modality_dispatcher.group_size_cpu]
        if modalities > 1
        else [tokens]
    )
    q = torch.empty(
        (tokens, int(module.config.num_heads_q), int(module.config.head_dim)),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    k = torch.empty(
        (tokens, int(module.config.num_heads_kv), int(module.config.head_dim)),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    v = torch.empty_like(k)
    gate = (
        torch.empty(
            (tokens, int(module.config.num_heads_q), 1),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        if bool(module.config.enable_attn_gating)
        else None
    )
    sin_emb, cos_emb = rope.tensor_split(2, -1)
    group_offset = 0
    for modality, group_tokens in enumerate(group_sizes):
        group_end = group_offset + group_tokens
        for start in range(group_offset, group_end, chunk_tokens):
            end = min(start + chunk_tokens, group_end)
            dispatcher = (
                _ModalityChunkDispatcher(
                    modality=modality,
                    tokens=end - start,
                    modalities=modalities,
                )
                if modalities > 1
                else modality_dispatcher
            )
            normalized = module.pre_norm(
                hidden_states[start:end], modality_dispatcher=dispatcher
            ).to(torch.bfloat16)
            qkv = module.linear_qkv(
                normalized, modality_dispatcher=dispatcher
            ).to(torch.float32)
            q_chunk, k_chunk, v_chunk = torch.split(
                qkv, [module.q_size, module.kv_size, module.kv_size], dim=1
            )
            q_chunk = q_chunk.view(
                -1, module.config.num_heads_q, module.config.head_dim
            )
            k_chunk = k_chunk.view(
                -1, module.config.num_heads_kv, module.config.head_dim
            )
            v_chunk = v_chunk.view(
                -1, module.config.num_heads_kv, module.config.head_dim
            )
            q_chunk = module.q_norm(
                q_chunk, modality_dispatcher=dispatcher
            )
            k_chunk = module.k_norm(
                k_chunk, modality_dispatcher=dispatcher
            )
            original_positions = permute_mapping[start:end]
            q_chunk = apply_rotary_emb(
                q_chunk.unsqueeze(0),
                cos_emb[original_positions],
                sin_emb[original_positions],
            ).squeeze(0)
            k_chunk = apply_rotary_emb(
                k_chunk.unsqueeze(0),
                cos_emb[original_positions],
                sin_emb[original_positions],
            ).squeeze(0)
            q.index_copy_(0, original_positions, q_chunk.to(torch.bfloat16))
            k.index_copy_(0, original_positions, k_chunk.to(torch.bfloat16))
            v.index_copy_(0, original_positions, v_chunk.to(torch.bfloat16))
            if gate is not None:
                gate[start:end].copy_(
                    module.linear_g(
                        normalized, modality_dispatcher=dispatcher
                    ).to(torch.float32).unsqueeze(-1)
                )
        group_offset = group_end
    del normalized, qkv, q_chunk, k_chunk, v_chunk
    if module.config.use_local_attn:
        attended = flex_flash_attn_with_cp(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            local_attn_handler.q_ranges,
            local_attn_handler.k_ranges,
            local_attn_handler.attn_type_map,
            local_attn_handler.max_seqlen_q,
            bool(getattr(local_attn_handler, "auto_range_merge", False)),
            bool(getattr(local_attn_handler, "sparse_load", False)),
            cp_split_sizes,
            module._mirai_refiner_attention_backend,
        )
    else:
        attended = flash_attn_with_cp(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), cp_split_sizes
        )
    del q, k, v

    output = torch.empty(
        (tokens, int(module.linear_proj.out_features)),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    group_offset = 0
    for modality, group_tokens in enumerate(group_sizes):
        group_end = group_offset + group_tokens
        for start in range(group_offset, group_end, chunk_tokens):
            end = min(start + chunk_tokens, group_end)
            dispatcher = (
                _ModalityChunkDispatcher(
                    modality=modality,
                    tokens=end - start,
                    modalities=modalities,
                )
                if modalities > 1
                else modality_dispatcher
            )
            original_positions = permute_mapping[start:end]
            attended_chunk = attended[original_positions]
            if gate is not None:
                attended_chunk = attended_chunk * torch.sigmoid(gate[start:end])
            attended_chunk = attended_chunk.reshape(
                -1, module.config.num_heads_q * module.config.head_dim
            ).to(torch.bfloat16)
            output[start:end].copy_(
                module.linear_proj(
                    attended_chunk, modality_dispatcher=dispatcher
                ).to(torch.bfloat16)
            )
            del attended_chunk
        group_offset = group_end
    return output


def attach_refiner_attention_chunking(
    transformer: Any, *, chunk_tokens: int = MAGI2_REFINER_MLP_CHUNK_TOKENS
) -> int:
    """Arm bounded QKV/rotary projections on every refiner attention layer."""
    from mirai.vendors.magi2_preview.model.magi2_refiner import Attention

    resolved = int(chunk_tokens)
    if resolved < 1:
        raise ValueError("MAGI-2 refiner attention chunk size must be >= 1 token.")
    attached = 0
    for module in transformer.modules():
        if not isinstance(module, Attention):
            continue
        module._mirai_chunk_tokens = resolved
        module.forward = MethodType(_chunked_refiner_attention_forward, module)
        attached += 1
    return attached


# --------------------------------------------------------------------------- #
# Pure geometry and re-noise math
# --------------------------------------------------------------------------- #
def magi2_refiner_latent_frames(preview_latent_frames: int) -> int:
    """Refiner latent length for ``preview_latent_frames`` preview latents.

    ``2T - 1`` keeps both endpoints and inserts one frame between each
    neighbouring pair, so the covered interval is unchanged while the sampling
    rate doubles.
    """
    frames = int(preview_latent_frames)
    if frames < 1:
        raise ValueError(
            f"preview latent frame count must be >= 1, got {frames}."
        )
    return 2 * frames - 1


def magi2_refiner_decoded_frames(preview_latent_frames: int) -> int:
    """Physical frames a refined clip decodes to, at ``MAGI2_REFINER_OUTPUT_FPS``.

    The shared Turbo VAE expands ``4 * (T' - 1) + 1`` frames from ``T'`` latent
    frames, and ``T' = 2T - 1``, so the result is ``8 * (T - 1) + 1`` — the frame
    count the request itself denotes.
    """
    refined = magi2_refiner_latent_frames(preview_latent_frames)
    return MAGI2_REFINER_VAE_TEMPORAL_STRIDE * (refined - 1) + 1


def magi2_refiner_sigma_table(device: Any | None = None) -> torch.Tensor:
    """The zero-terminal-SNR ``sqrt(alphas_cumprod)`` table the index addresses.

    Built by the vendored discretization so the table this stage re-noises
    against is the released one rather than a restatement of it. Descending:
    entry 0 keeps almost all signal and the final entry keeps none.
    """
    from mirai.vendors.magi2_preview.pipeline.inference_engine import (
        ZeroSNRDDPMDiscretization,
    )

    table = ZeroSNRDDPMDiscretization()(
        MAGI2_REFINER_SIGMA_TABLE_SIZE, do_append_zero=False, flip=True
    )
    return table if device is None else table.to(device=device)


def magi2_refiner_renoise_sigma(noise_index: int) -> float:
    """Signal coefficient at ``noise_index`` of the zero-terminal-SNR table."""
    index = int(noise_index)
    if not (0 <= index < MAGI2_REFINER_SIGMA_TABLE_SIZE):
        raise ValueError(
            f"refiner noise index must lie in [0, {MAGI2_REFINER_SIGMA_TABLE_SIZE}), "
            f"got {index}."
        )
    return float(magi2_refiner_sigma_table()[index])


def magi2_refiner_upsample(
    latent: torch.Tensor, *, latent_height: int, latent_width: int
) -> torch.Tensor:
    """Resample a ``[B, C, T, H, W]`` preview latent to ``[B, C, 2T-1, H', W']``.

    ``align_corners=True`` is required rather than incidental: it fixes the
    endpoints of the preview trajectory, so the inserted frames are midpoints of
    real preview frames. With ``align_corners=False`` the whole trajectory is
    shifted by half a preview frame and the endpoints are extrapolated.
    """
    import torch.nn.functional as F

    if latent.ndim != 5:
        raise ValueError(
            f"magi2_refiner_upsample expects [B,C,T,H,W], got {tuple(latent.shape)}."
        )
    height = int(latent_height)
    width = int(latent_width)
    if height < 1 or width < 1:
        raise ValueError(
            f"refiner latent size must be positive, got {height}x{width}."
        )
    frames = magi2_refiner_latent_frames(int(latent.shape[2]))
    return F.interpolate(
        latent,
        size=(frames, height, width),
        mode="trilinear",
        align_corners=True,
    )


def magi2_refiner_renoise(
    latent: torch.Tensor, noise: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Variance-preserving single-shot re-noise at ``sigma``.

    ``sigma`` is the SIGNAL coefficient of the zero-terminal-SNR table, so the
    noise weight is ``sqrt(1 - sigma^2)``. This is not the rectified-flow blend
    the preview training path uses; the refiner is entered at a fixed corruption
    level rather than at a flow timestep.
    """
    value = float(sigma)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"refiner re-noise sigma must lie in [0, 1], got {value}.")
    if tuple(latent.shape) != tuple(noise.shape):
        raise ValueError(
            "refiner re-noise requires matching shapes, got "
            f"{tuple(latent.shape)} and {tuple(noise.shape)}."
        )
    return latent * value + noise * (1.0 - value**2) ** 0.5


# --------------------------------------------------------------------------- #
# Resolved request
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Magi2RefineSettings:
    """The refinement actually applied, after release-profile resolution.

    Every field is reported back to the caller, so a run states the values that
    drove it rather than the values that were typed.
    """

    steps: int
    cfg_scale: float
    shift: float
    height: int
    width: int
    noise_index: int
    scheduler: str

    def as_request(self) -> dict[str, Any]:
        return {
            "steps": int(self.steps),
            "cfg_scale": float(self.cfg_scale),
            "shift": float(self.shift),
            "height": int(self.height),
            "width": int(self.width),
            "noise_index": int(self.noise_index),
            "scheduler": str(self.scheduler),
        }


# --------------------------------------------------------------------------- #
# Refiner transformer lifecycle (on-demand load / release)
# --------------------------------------------------------------------------- #
class Magi2Refiner:
    """On-demand owner of the refiner transformer and its data proxy.

    Built lazily for the refine stage, placed on the compute device through the
    same block-residency mechanism the preview transformer uses, and released
    afterwards so it never co-resides with the preview denoiser.
    """

    def __init__(
        self,
        model_config: Any,
        *,
        config_path: str,
        subfolder: str = MAGI2_REFINER_SUBFOLDER,
        attention_backend: str = MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT,
    ) -> None:
        self.model_config = model_config
        self.config_path = str(config_path)
        self.subfolder = str(subfolder or MAGI2_REFINER_SUBFOLDER)
        self.attention_backend = normalize_refiner_attention_backend(attention_backend)
        self._transformer: Any | None = None
        self._data_proxy: Any | None = None
        self._runtime_config: Any | None = None
        self._block_swap_manager: Any | None = None
        self._block_hook_handles: list[Any] = []
        self._paired_staging = False

    # -- asset location ----------------------------------------------------
    def snapshot_root(self) -> Path:
        root = Path(str(self.model_config.path)).expanduser()
        return root.parent if root.name == "preview" else root

    def checkpoint_dir(self) -> Path:
        return self.snapshot_root() / self.subfolder

    def has_weights(self) -> bool:
        """True when the refiner subfolder carries a sharded checkpoint index."""
        return (self.checkpoint_dir() / "model.safetensors.index.json").is_file()

    # -- release profile ---------------------------------------------------
    def runtime_config(self) -> Any:
        """The vendored refiner profile, with the checkpoint path resolved.

        ``magi2_refiner_model_path`` is stated upstream as an environment
        expansion; it is rewritten here from ``model.path`` so the released
        profile can be read without an environment contract.
        """
        if self._runtime_config is None:
            from mirai.vendors.magi2_preview.common.magi2_config import load_config

            config = load_config(self.config_path)
            if config.magi2_refiner_arch_config is None:
                raise RuntimeError(
                    f"'{self.config_path}' declares no magi2_refiner_arch_config, so "
                    "it is a preview-only profile and cannot drive the refiner "
                    "stage. Point family_params.refiner_config_path at the "
                    "refiner architecture JSON."
                )
            config.evaluation_config.magi2_refiner_model_path = str(
                self.checkpoint_dir()
            )
            self._runtime_config = config
        return self._runtime_config

    def settings(
        self,
        *,
        steps: int | None,
        cfg_scale: float | None,
        shift: float | None,
        height: int | None,
        width: int | None,
        preview_height: int,
        preview_width: int,
        scheduler: str,
    ) -> Magi2RefineSettings:
        """Resolve a request against the release profile.

        An absent value takes the released refiner profile; a supplied value
        overrides it. The target resolution falls back to the released 1080p
        delivery grid.
        """
        evaluation = self.runtime_config().evaluation_config
        resolved_scheduler = str(scheduler or MAGI2_REFINER_SOLVER).strip().lower()
        if resolved_scheduler != MAGI2_REFINER_SOLVER:
            raise RuntimeError(
                f"The MAGI-2 refiner implements the '{MAGI2_REFINER_SOLVER}' solver "
                f"only; got '{resolved_scheduler}'. Its sampler is the vendored "
                "Flow-UniPC multistep scheduler and has no other schedule."
            )
        resolved_shift = (
            float(evaluation.magi2_refiner_shift)
            if evaluation.magi2_refiner_shift is not None
            else float(evaluation.shift)
        )
        settings = Magi2RefineSettings(
            steps=int(
                evaluation.magi2_refiner_num_inference_steps if steps is None else steps
            ),
            cfg_scale=float(
                evaluation.magi2_refiner_video_txt_guidance_scale
                if cfg_scale is None
                else cfg_scale
            ),
            shift=resolved_shift if shift is None else float(shift),
            height=int(MAGI2_RELEASED_REFINER_HEIGHT if height is None else height),
            width=int(MAGI2_RELEASED_REFINER_WIDTH if width is None else width),
            noise_index=int(evaluation.magi2_refiner_noise_value),
            scheduler=resolved_scheduler,
        )
        if settings.steps < 1:
            raise RuntimeError(
                f"refiner steps must be >= 1, got {settings.steps}."
            )
        if settings.shift <= 0.0:
            raise RuntimeError(
                f"refiner flow shift must be > 0, got {settings.shift}."
            )
        if settings.cfg_scale < 0.0:
            raise RuntimeError(
                f"refiner CFG scale must be >= 0, got {settings.cfg_scale}."
            )
        stride = tuple(int(value) for value in evaluation.magi2_refiner_vae_stride)
        for label, value in (("height", settings.height), ("width", settings.width)):
            multiple = stride[1] if label == "height" else stride[2]
            if value < multiple or value % multiple:
                raise RuntimeError(
                    f"refiner {label}={value} must be a positive multiple of "
                    f"{multiple}, the refiner VAE spatial stride."
                )
        if not (0 <= settings.noise_index < MAGI2_REFINER_SIGMA_TABLE_SIZE):
            raise RuntimeError(
                "family_params.refiner_config_path states "
                f"magi2_refiner_noise_value={settings.noise_index}, outside the "
                f"[0, {MAGI2_REFINER_SIGMA_TABLE_SIZE}) zero-terminal-SNR table."
            )
        return settings

    def latent_size(self, settings: Magi2RefineSettings) -> tuple[int, int]:
        """Refiner latent height and width for a resolved request."""
        stride = tuple(
            int(value)
            for value in self.runtime_config().evaluation_config.magi2_refiner_vae_stride
        )
        return settings.height // stride[1], settings.width // stride[2]

    # -- lifecycle ---------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._transformer is not None

    @property
    def transformer(self) -> Any:
        if self._transformer is None:
            raise RuntimeError("MAGI-2 refiner is not loaded; call load() first.")
        return self._transformer

    @property
    def data_proxy(self) -> Any:
        if self._data_proxy is None:
            raise RuntimeError("MAGI-2 refiner is not loaded; call load() first.")
        return self._data_proxy

    @torch.compiler.disable
    def forward_pair(
        self,
        first: tuple[Any, Any, Any, Any, Any],
        second: tuple[Any, Any, Any, Any, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate two CFG branches while staging each transformer layer once."""
        transformer = self.transformer
        branches = [
            _prepare_refiner_branch(transformer, first),
            _prepare_refiner_branch(transformer, second),
        ]
        manager = self._block_swap_manager
        self._paired_staging = True
        try:
            for index, layer in enumerate(transformer.block.layers):
                if manager is not None:
                    manager.before_block(index)
                for branch in branches:
                    branch.x = layer(
                        branch.x,
                        branch.rope,
                        permute_mapping=branch.permute_mapping,
                        inv_permute_mapping=branch.inv_permute_mapping,
                        varlen_handler=branch.varlen_handler,
                        local_attn_handler=branch.local_attn_handler,
                        modality_dispatcher=branch.modality_dispatcher,
                        cp_split_sizes=branch.cp_split_sizes,
                    )
                if manager is not None:
                    manager.after_block(index)
        finally:
            self._paired_staging = False
        return (
            _finish_refiner_branch(transformer, branches[0]),
            _finish_refiner_branch(transformer, branches[1]),
        )

    def load(self, *, device: str, residency: Any | None = None) -> None:
        """Build (once) and place the refiner transformer on ``device``.

        The architecture comes from the vendored refiner JSON rather than from a
        ``config.json`` beside the shards, because the release ships none; the
        shards are then loaded strictly, so a checkpoint whose key set does not
        match the declared architecture fails instead of loading partially.

        The refiner's attention is dispatched through operators MagiCompiler
        registers, falling back to the eager implementations vendored beside
        them when the operator namespace is empty. That at least one of the two
        is reachable is a precondition of the stage rather than something to
        discover mid-forward. The precondition covers only the operators the
        configured architecture actually reaches: a bound native attention
        backend serves the local-attention operator itself, and a profile whose
        layers are all local never reaches the dense one.
        """
        from mirai.core.models.magi2_preview.refiner_attention import (
            attach_refiner_attention_backend,
            refiner_required_magi2_ops,
            resolve_magi2_refiner_attention,
            validate_refiner_flex_support,
        )
        from mirai.vendors.magi2_preview.common.magi_compiler_compat import (
            require_magi2_custom_ops,
        )

        config = self.runtime_config()
        backend = resolve_magi2_refiner_attention(self.attention_backend)
        if backend is not None:
            validate_refiner_flex_support()
        required = refiner_required_magi2_ops(
            config.magi2_refiner_arch_config, backend
        )
        if required:
            require_magi2_custom_ops("The MAGI-2 refiner stage", required)
        if not self.has_weights():
            raise RuntimeError(
                "--refine requested but no MAGI-2 refiner weights were found under "
                f"'{self.checkpoint_dir()}'. {_MAGI2_REFINER_ASSET_HINT}"
            )
        if self._transformer is None:
            from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import (
                load_safetensors_into_model,
            )
            from mirai.vendors.magi2_preview.model.magi2_refiner import (
                Transformer as Magi2RefinerTransformer,
            )
            from mirai.vendors.magi2_preview.pipeline.refiner_data_proxy import (
                Magi2RefinerDataProxy,
            )

            transformer = Magi2RefinerTransformer(config.magi2_refiner_arch_config)
            load_safetensors_into_model(
                transformer,
                str(self.checkpoint_dir()),
                desc="Loading MAGI-2 refiner shards",
            )
            transformer.eval()
            for parameter in transformer.parameters():
                parameter.requires_grad_(False)
            self._transformer = transformer
            self._data_proxy = Magi2RefinerDataProxy(
                config.evaluation_config.magi2_refiner_data_proxy_config
            )
        attach_refiner_mlp_chunking(self._transformer)
        attach_refiner_attention_backend(self._transformer, backend)
        attach_refiner_attention_chunking(self._transformer)
        self._place(device=device, residency=residency)

    def _place(self, *, device: str, residency: Any | None) -> None:
        """Place the refiner using the family's own residency mechanism.

        With no residency request the dense refiner is simply resident. With
        one, its layers stream host-to-device exactly as the preview blocks do:
        the refiner is a different architecture but the same
        ``block.layers`` stack shape, so the family's block-swap manager binds
        to it unchanged.
        """
        target = torch.device(device)
        transformer = self.transformer
        if residency is None or not bool(getattr(residency, "enabled", False)):
            transformer.to(target)
            return

        from mirai.core.training.residency.block_swap import BlockSwapManager
        from mirai.core.training.residency.tensor_residency import (
            move_tensors_outside_modules,
        )

        units = list(enumerate(transformer.block.layers))
        layers = [module for _index, module in units]
        move_tensors_outside_modules(
            transformer, excluded_modules=layers, device=target
        )
        manager = BlockSwapManager(
            total_blocks=len(units),
            blocks_to_swap=min(len(units), int(residency.blocks_to_swap)),
            mode=str(residency.mode),
            # Refinement is inference-only, so no backward window exists to
            # keep a block resident for.
            block_swap_backward=False,
            block_residency_planner=str(residency.block_residency_planner),
            block_swap_prefetch_depth=int(residency.block_swap_prefetch_depth),
            block_residency_priority=str(residency.block_residency_priority),
            block_swap_transfer_strategy=str(residency.block_swap_transfer_strategy),
            disk_offload_dir=residency.offload_dir,
        )
        manager.bind(units, device=target)
        self._release_hooks()

        # Staging rebinds ``Parameter.data`` between host and device from inside
        # the vendored ``@torch.compile(dynamic=True)`` refiner forward. Dynamo
        # cannot trace the resulting ``aten.set_`` across two devices and aborts
        # with "Unhandled FakeTensor Device Propagation". Staging is a residency
        # side effect that must happen eagerly whether or not the forward is
        # compiled, so it is an explicit graph break. The disable covers the two
        # staging calls only; the compiled forward around them is untouched.
        from torch import _dynamo

        stage_in = _dynamo.disable(lambda idx: manager.before_block(idx))
        stage_out = _dynamo.disable(lambda idx: manager.after_block(idx))

        for index, layer in units:
            self._block_hook_handles.append(
                layer.register_forward_pre_hook(
                    lambda _module, _args, idx=index: (
                        None if self._paired_staging else stage_in(idx)
                    )
                )
            )
            self._block_hook_handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, output, idx=index: (
                        None if self._paired_staging else stage_out(idx),
                        output,
                    )[1]
                )
            )
        self._block_swap_manager = manager

    def _release_hooks(self) -> None:
        for handle in self._block_hook_handles:
            handle.remove()
        self._block_hook_handles.clear()

    def release(self) -> None:
        """Drop the refiner off the compute device and free its VRAM."""
        self._release_hooks()
        manager = self._block_swap_manager
        if manager is not None:
            manager.release_device()
        self._block_swap_manager = None
        if self._transformer is not None:
            self._transformer.to(device=torch.device("cpu"))
            self._transformer = None
        self._data_proxy = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Refine policy loop (family-owned; model access via provider hooks only)
# --------------------------------------------------------------------------- #
def _release_cuda_workspace(device: torch.device) -> None:
    """Make completed asynchronous work reclaimable at a memory boundary."""

    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def run_refine(
    *,
    pipeline: Any,
    refiner: Magi2Refiner,
    base_latent: torch.Tensor,
    settings: Magi2RefineSettings,
    prompt: str,
    negative_prompt: str,
    seed: int,
    device: str,
) -> torch.Tensor:
    """Resample, re-noise and short-denoise a preview latent into a refined one.

    Returns ``[C, 2T-1, H', W']`` so the result flows straight into the standard
    native VAE decode.

    The preview transformer is released before any refiner state is allocated.
    This operation does not restore it; the owning inference session restores
    placement before a subsequent preview denoise.

    This loop takes no compute dtype, because the vendored refiner owns its own
    dtype policy and it is not a policy a caller may override. Its ``Adapter``
    and ``PostAdapter`` projections are constructed at ``dtype=torch.float32``
    and write into float32 buffers through masked ``index_put_``, while the
    transformer block casts to ``config.params_dtype`` on entry and the post
    adapter casts back on exit. An outer ``torch.amp.autocast`` demotes those
    projections without demoting the buffers they write into, and the vendored
    ``@torch.compile(dynamic=True)`` forward rejects the resulting dtype
    crossing outright. Latents therefore enter float32 and no autocast wraps the
    denoise.
    """
    if base_latent.ndim != 4:
        raise RuntimeError(
            "MAGI-2 refinement expects a [C,T,H,W] preview latent, got "
            f"{tuple(base_latent.shape)}."
        )
    compute_device = torch.device(device)
    generator_factory = getattr(pipeline, "refiner_noise_generator", None)
    if callable(generator_factory):
        generator = generator_factory(seed=int(seed), device=compute_device)
    else:
        generator = torch.Generator(device=compute_device)
        generator.manual_seed(int(seed))

    # The preview latent is complete, so its sparse-MoE transformer leaves the
    # device before the refiner's weights are allocated.
    pipeline.release_base_transformer()

    # An adjacent preview already encoded these exact prompts. Consume its
    # detached pair when available; a resumed refiner-only call falls back to a
    # fresh encode before the refiner becomes resident.
    take_context = getattr(pipeline, "take_refiner_context", None)
    cached_context = take_context() if callable(take_context) else None
    if cached_context is None:
        pipeline.load_text_encoder(device=str(compute_device))
        try:
            context = torch.as_tensor(
                pipeline.encode_prompt(prompt, device=str(compute_device))
            )
            context_null = torch.as_tensor(
                pipeline.encode_prompt(negative_prompt or "", device=str(compute_device))
            )
        finally:
            pipeline.offload_text_encoder()
    else:
        context, context_null = cached_context
    context = _as_batched_context(context, compute_device)
    context_null = _as_batched_context(context_null, compute_device)

    latent_height, latent_width = refiner.latent_size(settings)
    latents = magi2_refiner_upsample(
        base_latent.unsqueeze(0).to(device=compute_device, dtype=torch.float32),
        latent_height=latent_height,
        latent_width=latent_width,
    )
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=compute_device,
        dtype=latents.dtype,
    )
    latents = magi2_refiner_renoise(
        latents, noise, magi2_refiner_renoise_sigma(settings.noise_index)
    )

    from mirai.vendors.magi2_preview.pipeline.sampler import (
        FlowUniPCMultistepScheduler,
    )

    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(
        int(settings.steps), device=compute_device, shift=float(settings.shift)
    )

    try:
        refiner.load(
            device=str(compute_device),
            residency=pipeline.refiner_residency_request(),
        )
        scale = float(settings.cfg_scale)
        with torch.inference_mode():
            for timestep in scheduler.timesteps:
                # The refiner evaluates the conditional and unconditional
                # branches as two separate forwards. Unlike the preview sampler
                # it is NOT packed into one B=2 forward: its window attention
                # asserts batch size 1.
                if scale <= 0.0:
                    velocity = pipeline.refiner_forward(latents, context_null)
                else:
                    paired_forward = getattr(pipeline, "refiner_cfg_forward", None)
                    if callable(paired_forward):
                        v_cond, v_uncond = paired_forward(
                            latents, context, context_null
                        )
                    else:
                        v_cond = pipeline.refiner_forward(latents, context)
                        # The forward is asynchronous, so synchronize before
                        # returning its cached allocator segments for reuse.
                        _release_cuda_workspace(compute_device)
                        v_uncond = pipeline.refiner_forward(latents, context_null)
                    velocity = v_uncond + scale * (v_cond - v_uncond)
                    del v_cond, v_uncond
                # Upstream expands the guidance scale over the frame axis so a
                # per-frame schedule can lower it on the leading frames. The
                # released refiner profile leaves that schedule off, so the
                # scalar blend above is the released behavior; a per-frame
                # schedule would have to enter as its own explicit option.
                latents = scheduler.step(
                    velocity, timestep, latents, return_dict=False
                )[0]
                del velocity
                _release_cuda_workspace(compute_device)
    finally:
        refiner.release()
    refined = latents.squeeze(0) if latents.ndim == 5 else latents
    return refined.detach().float()


def _as_batched_context(context: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Coerce an encoded prompt to the ``[1, S, W]`` the refiner proxy packs."""
    value = context.to(device=device, dtype=torch.float32)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or int(value.shape[0]) != 1:
        raise RuntimeError(
            "MAGI-2 refinement conditions on one prompt at a time; got context "
            f"of shape {tuple(value.shape)}."
        )
    return value


__all__ = [
    "MAGI2_RELEASED_REFINER_HEIGHT",
    "MAGI2_RELEASED_REFINER_WIDTH",
    "MAGI2_REFINER_MLP_CHUNK_TOKENS",
    "MAGI2_REFINER_OUTPUT_FPS",
    "MAGI2_REFINER_SIGMA_TABLE_SIZE",
    "MAGI2_REFINER_SOLVER",
    "MAGI2_REFINER_SUBFOLDER",
    "MAGI2_REFINER_VAE_TEMPORAL_STRIDE",
    "Magi2RefineSettings",
    "Magi2Refiner",
    "attach_refiner_attention_chunking",
    "attach_refiner_mlp_chunking",
    "magi2_refiner_decoded_frames",
    "magi2_refiner_latent_frames",
    "magi2_refiner_renoise",
    "magi2_refiner_renoise_sigma",
    "magi2_refiner_sigma_table",
    "magi2_refiner_upsample",
    "run_refine",
]
