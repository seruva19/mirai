# Adapted from the native LingBot-Video denoiser (Apache-2.0).
# Source: https://github.com/Robbyant/lingbot-video
# Diffusers base classes are represented by local nn.Module-compatible helpers.
# See the adjacent LICENSE.

import math
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mirai.core.models.attention_backends import dispatch_varlen_attention
from mirai.core.moe.routing.depth import attention_with_received_scores
from mirai.core.moe.routing.depth import select_depth_tokens

from .native_compat import ContextParallelInput, ContextParallelOutput
from .native_compat import TimestepEmbedding, Timesteps
from .native_compat import Transformer3DModelOutput
from .native_compat import capture_init_config, dispatch_attention_fn

@contextmanager
def _suspend_observers(*observers):
    with ExitStack() as stack:
        for observer in observers:
            if observer is not None:
                stack.enter_context(observer.suspended())
        yield


@contextmanager
def _stack_contexts(*contexts):
    with ExitStack() as stack:
        for context in contexts:
            stack.enter_context(context)
        yield


def _extend_checkpoint_context_fn(context_fn, recompute_context_fn):
    def combined():
        forward_context, recompute_context = context_fn()
        return (
            forward_context,
            _stack_contexts(recompute_context, recompute_context_fn()),
        )

    return combined

try:
    from .moe_pack_kernels import reorder_tokens_triton_pack
    from .moe_restore_kernels import restore_tokens_triton
    from .sglang_moe_shim import (
        LightSglangMoeRunnerConfig,
        LightSglangStandardTopKOutput,
        ensure_sglang_moe_ready,
        fp8_scale_from_amax,
        quantize_to_fp8_e4m3fn,
        sglang_fused_experts,
    )
except ImportError:  # pragma: no cover - allows direct file loading in diagnostics.
    from moe_pack_kernels import reorder_tokens_triton_pack
    from moe_restore_kernels import restore_tokens_triton
    from sglang_moe_shim import (
        LightSglangMoeRunnerConfig,
        LightSglangStandardTopKOutput,
        ensure_sglang_moe_ready,
        fp8_scale_from_amax,
        quantize_to_fp8_e4m3fn,
        sglang_fused_experts,
    )


LINGBOT_VIDEO_FP32_MODULES = (
    "time_embedder",
    "time_modulation",
    "scale_shift_table",
    "norm",
    "norm1",
    "norm2",
    "norm_q",
    "norm_k",
    "norm_post_attn",
    "norm_post_ffn",
    "norm_out",
    "norm_out_modulation",
    "router",
)


def _torch_grouped_mm_supported(device: torch.device) -> bool:
    """Return whether this torch build can execute grouped MM on ``device``."""

    return bool(
        device.type == "cuda"
        and hasattr(torch, "_grouped_mm")
        and torch.cuda.get_device_capability(device) == (9, 0)
    )


def should_keep_in_fp32(name: str) -> bool:
    return any(module_name in name.split(".") for module_name in LINGBOT_VIDEO_FP32_MODULES)


@dataclass(frozen=True)
class LingBotVideoRuntimeOptions:
    """Per-forward controls for optional native LingBot execution paths."""

    moe_expert_backend: str = "grouped_mm"
    moe_pad_backend: str = "auto"
    moe_reorder_backend: str = "sort"
    moe_restore_backend: str = "chunked_scatter"
    moe_restore_chunk_size: int = 128
    fused_qkv_linear: bool = False
    inference_bf16_fastmath: bool = False


_DEFAULT_RUNTIME_OPTIONS = LingBotVideoRuntimeOptions()


def set_lingbot_video_runtime_options(
    module: nn.Module,
    options: LingBotVideoRuntimeOptions,
) -> None:
    """Attach immutable execution options to a model and all checkpointed blocks."""

    for child in module.modules():
        child._mirai_runtime_options = options


def _runtime_options(owner: nn.Module) -> LingBotVideoRuntimeOptions:
    return getattr(owner, "_mirai_runtime_options", _DEFAULT_RUNTIME_OPTIONS)


def _moe_expert_backend(owner: nn.Module) -> str:
    return _runtime_options(owner).moe_expert_backend


def _moe_pad_backend(owner: nn.Module, training: bool = True) -> str:
    """Resolve the MoE grouped-token pad backend.

    ``auto`` selects ``loop`` for eager training and ``vectorized`` for
    inference or compiler tracing. Both construct identical indices and values;
    the vectorized implementation avoids scalar host reads and supports graph
    capture.
    """
    configured = _runtime_options(owner).moe_pad_backend
    if configured != "auto":
        return configured
    if torch.compiler.is_compiling():
        return "vectorized"
    return "loop" if training else "vectorized"


def _moe_reorder_backend(owner: nn.Module) -> str:
    return _runtime_options(owner).moe_reorder_backend


def _moe_restore_backend(owner: nn.Module) -> str:
    return _runtime_options(owner).moe_restore_backend


def _infer_bf16_fastmath(owner: nn.Module, training: bool) -> bool:
    """Opt-in bf16 fast math for eval-only dtype-cast hotspots.

    The option keeps selected eval elementwise operations in bf16. It changes
    numerics, remains disabled by default, and never engages during training.
    """
    return (not training) and _runtime_options(owner).inference_bf16_fastmath


def _all_to_all_split_cat(
    local_input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    input_list = [
        tensor.contiguous()
        for tensor in torch.tensor_split(local_input, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class LingBotVideoRMSNorm(nn.Module):
    """RMSNorm with fp32 accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(
        self, hidden_states: torch.Tensor, *, bf16_fastmath: bool = False
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        if bf16_fastmath:
            # Opt-in eval fast-math: keep the whole normalization in the input
            # (bf16) dtype instead of promoting the activation to fp32 and back.
            # Only the q/k norms pass this flag (gated by env + eval); every other
            # RMSNorm uses the configured fp32 reference path.
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight.to(input_dtype) * hidden_states
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, *, bf16_fastmath: bool = False
) -> torch.Tensor:
    """Apply complex RoPE to `(B, S, H, D)` attention tensors.

    The default path casts ``x`` to fp32 (``view_as_complex`` requires float),
    does the complex rotation, and casts back. Under ``bf16_fastmath`` (opt-in,
    eval only) an algebraically-equivalent real-valued rotation runs entirely in
    the input dtype, skipping the bf16<->fp32 round trip. Numerically distinct
    from the fp32 path (that is the point of the flag); default off is unchanged.
    """
    if bf16_fastmath:
        cos = freqs_cis.real.unsqueeze(2).to(x.dtype)  # (B, S, 1, D/2)
        sin = freqs_cis.imag.unsqueeze(2).to(x.dtype)
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        x_r = pairs[..., 0]
        x_i = pairs[..., 1]
        out_r = x_r * cos - x_i * sin
        out_i = x_r * sin + x_i * cos
        return torch.stack((out_r, out_i), dim=-1).flatten(3).to(x.dtype)
    with torch.amp.autocast("cuda", enabled=False):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(x_c * freqs_cis.unsqueeze(2)).flatten(3)
        return out.type_as(x)


class LingBotVideoRotaryEmbedding(nn.Module):
    """Complex64 RoPE table indexed by position ids."""

    def __init__(self, axes_dims: Tuple[int, ...], axes_lens: Tuple[int, ...], theta: float):
        super().__init__()
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = list(axes_lens)
        self.theta = theta
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: Tuple[int, ...], end: Tuple[int, ...], theta: float):
        freqs_cis = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (
                theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
            )
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return freqs_cis

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids: (S, 3) int → (S, head_dim/2) complex64
        device = position_ids.device
        # Same shape => same deterministic grid => same max positions, so the
        # rebuild check (a device->host .tolist() sync per forward) only needs
        # to run when the grid changes. After the first forward of a run this
        # path is sync-free (B5-R2).
        # A set, not a single key: cond/uncond forwards alternate between two
        # text lengths (two shapes) every step — a single cached key would miss
        # on every call. The freqs table only ever grows, so once a shape was
        # validated it stays covered.
        shape_key = tuple(position_ids.shape)
        validated = getattr(self, "_validated_shape_keys", None)
        if validated is None:
            validated = set()
            self._validated_shape_keys = validated
        if self.freqs_cis is not None and shape_key in validated:
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            max_vals = position_ids.max(dim=0).values.tolist()
            needs_rebuild = self.freqs_cis is None or any(
                m >= l for m, l in zip(max_vals, self.axes_lens)
            )
            if needs_rebuild:
                for i in range(len(self.axes_lens)):
                    if max_vals[i] >= self.axes_lens[i]:
                        self.axes_lens[i] = int(max_vals[i] * 1.5) + 1
                self.freqs_cis = self.precompute_freqs_cis(
                    self.axes_dims, tuple(self.axes_lens), theta=self.theta
                )
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
            elif self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
            if len(validated) < 64:
                validated.add(shape_key)

        return torch.cat([self.freqs_cis[i][position_ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


def make_joint_position_ids(
    text_len: int, grid_t: int, grid_h: int, grid_w: int, device: torch.device
) -> torch.Tensor:
    """3D positions in [video; text] order. Text t-axis is 1..text_len; video t-axis starts at text_len+1.

    Matches patchify_and_embed: cap start (1,0,0); vision start (cap_len+1,0,0);
    freqs ordered with x first and cap second (same order as cat_interleave).
    """
    tt = torch.arange(grid_t, device=device, dtype=torch.int32) + (text_len + 1)
    hh = torch.arange(grid_h, device=device, dtype=torch.int32)
    ww = torch.arange(grid_w, device=device, dtype=torch.int32)
    grid = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1).flatten(0, 2)
    text_t = torch.arange(text_len, device=device, dtype=torch.int32) + 1
    text_pos = torch.stack(
        [text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)], dim=-1
    )
    return torch.cat([grid, text_pos], dim=0)  # (Nx + L, 3)


def _cat_interleave(
    a: torch.Tensor,
    len_a: list[int],
    b: torch.Tensor,
    len_b: list[int],
) -> torch.Tensor:
    a_split = torch.split(a, len_a, dim=1)
    b_split = torch.split(b, len_b, dim=1)
    blocks: list[torch.Tensor] = []
    for x_part, text_part in zip(a_split, b_split):
        blocks.extend([x_part, text_part])
    return torch.cat(blocks, dim=1)


class LingBotVideoTextEmbedder(nn.Module):
    """Matches CondProjection: RMSNorm(text_dim, eps=1e-6 fixed) -> Linear-SiLU-Linear."""

    def __init__(self, text_dim: int, hidden_size: int):
        super().__init__()
        self.norm = LingBotVideoRMSNorm(text_dim, eps=1e-6)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.linear_2(F.silu(self.linear_1(x)))


class LingBotVideoAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, norm_eps, qkv_bias, out_bias):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.norm_q = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.norm_k = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=out_bias)
        self.backend = "auto"
        self._mirai_capture_received_attention = False
        self._mirai_attention_query_chunk_size = 128
        self._mirai_last_received_attention = None

    def forward(
        self,
        x,
        rotary_emb,
        attention_mask=None,
        packed_indices: Optional[dict[str, torch.Tensor]] = None,
        parallel_config=None,
    ):
        B, S, _ = x.shape
        if _runtime_options(self).fused_qkv_linear:
            weight = torch.cat(
                (self.to_q.weight, self.to_k.weight, self.to_v.weight),
                dim=0,
            )
            bias = None
            if self.to_q.bias is not None:
                bias = torch.cat(
                    (self.to_q.bias, self.to_k.bias, self.to_v.bias),
                    dim=0,
                )
            qkv = F.linear(x, weight, bias)
            q, k, v = qkv.view(B, S, 3, self.num_heads, self.head_dim).unbind(2)
        else:
            q = self.to_q(x).unflatten(2, (self.num_heads, self.head_dim))
            k = self.to_k(x).unflatten(2, (self.num_heads, self.head_dim))
            v = self.to_v(x).unflatten(2, (self.num_heads, self.head_dim))
        fastmath = _infer_bf16_fastmath(self, self.training)
        q = apply_rotary_emb(
            self.norm_q(q, bf16_fastmath=fastmath), rotary_emb, bf16_fastmath=fastmath
        )
        k = apply_rotary_emb(
            self.norm_k(k, bf16_fastmath=fastmath), rotary_emb, bf16_fastmath=fastmath
        )
        # A-MoD consumes the previous block's mean received attention.  Its
        # exact striped path produces the ordinary output and routing statistic
        # together instead of materializing a second full attention matrix.
        self._mirai_last_received_attention = None
        if bool(self._mirai_capture_received_attention):
            if parallel_config is not None:
                raise ValueError(
                    "Mixture-of-Depths attention routing does not support "
                    "context-parallel execution."
                )
            out, received = attention_with_received_scores(
                q,
                k,
                v,
                attention_mask=attention_mask,
                cu_seqlens=(
                    packed_indices["cu_seqlens_kv"]
                    if packed_indices is not None
                    else None
                ),
                query_chunk_size=int(self._mirai_attention_query_chunk_size),
            )
            self._mirai_last_received_attention = received.detach()
        # dispatch_attention_fn expects (B, S, H, D) in and out (same as the diffusers Wan processor)
        elif packed_indices is None:
            out = dispatch_attention_fn(
                q,
                k,
                v,
                attn_mask=attention_mask,
                parallel_config=parallel_config,
                backend=self.backend,
            )
        else:
            if parallel_config is None:
                out = dispatch_varlen_attention(
                    q.reshape(-1, self.num_heads, self.head_dim),
                    k.reshape(-1, self.num_heads, self.head_dim),
                    v.reshape(-1, self.num_heads, self.head_dim),
                    cu_seqlens_q=packed_indices["cu_seqlens_kv"],
                    cu_seqlens_k=packed_indices["cu_seqlens_kv"],
                    max_seqlen_q=packed_indices["max_seqlen_in_batch_kv"],
                    max_seqlen_k=packed_indices["max_seqlen_in_batch_kv"],
                    backend=self.backend,
                )
                out = out.reshape(B, S, self.num_heads, self.head_dim)
            else:
                group = parallel_config.context_parallel_config._ulysses_mesh.get_group()
                world_size = dist.get_world_size(group)
                local_heads = self.num_heads // world_size
                q_global = _all_to_all_split_cat(
                    q.reshape(B, S, self.num_heads * self.head_dim),
                    scatter_dim=2,
                    gather_dim=1,
                    group=group,
                ).view(B, S * world_size, local_heads, self.head_dim)
                k_global = _all_to_all_split_cat(
                    k.reshape(B, S, self.num_heads * self.head_dim),
                    scatter_dim=2,
                    gather_dim=1,
                    group=group,
                ).view(B, S * world_size, local_heads, self.head_dim)
                v_global = _all_to_all_split_cat(
                    v.reshape(B, S, self.num_heads * self.head_dim),
                    scatter_dim=2,
                    gather_dim=1,
                    group=group,
                ).view(B, S * world_size, local_heads, self.head_dim)
                q_flat = q_global.reshape(-1, local_heads, self.head_dim)
                k_flat = k_global.reshape(-1, local_heads, self.head_dim)
                v_flat = v_global.reshape(-1, local_heads, self.head_dim)
                out_global = dispatch_varlen_attention(
                    q_flat,
                    k_flat,
                    v_flat,
                    cu_seqlens_q=packed_indices["cu_seqlens_kv"],
                    cu_seqlens_k=packed_indices["cu_seqlens_kv"],
                    max_seqlen_q=packed_indices["max_seqlen_in_batch_kv"],
                    max_seqlen_k=packed_indices["max_seqlen_in_batch_kv"],
                    backend=self.backend,
                )
                out_global = out_global.reshape(B, S * world_size, local_heads * self.head_dim)
                out = _all_to_all_split_cat(
                    out_global,
                    scatter_dim=1,
                    gather_dim=2,
                    group=group,
                ).view(B, S, self.num_heads, self.head_dim)
        return self.to_out(out.flatten(2, 3).type_as(x))


class LingBotVideoMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingBotVideoRouter(nn.Module):
    """Matches the TokenChoiceTopKRouter inference path (no capacity/jitter/load stats).

    The asymmetry must be preserved: selection uses the bias-added score, while gating
    weights gather the bias-free score.
    """

    def __init__(self, hidden_size, num_experts, top_k, score_func, norm_topk_prob,
                 n_group, topk_group, route_scale):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts), persistent=True)
        # Stochastic per-step expert-subset routing runtime state (opt-in, off by
        # default leaves routing unchanged). Set once per step by the
        # pipeline via ``set_subset_runtime``; consumed inside ``forward`` only
        # while training. See mirai/core/moe/routing/subset.py.
        self._subset_active = False
        self._subset_size = 0
        self._subset_pool_factor = 2.0
        self._subset_seed = 0
        self._subset_kl_weight = 0.0
        self.training_subset_kl = None
        self.last_subset_ids = None
        self.last_unique_experts = None
        self._dataset_routing_mode = "emergent"
        self._dataset_domain_experts = ()
        self._dataset_prior_scale = 0.0
        self._dataset_hard_active = False
        self._dataset_token_allowed = None
        self.last_dataset_affinity_hit_rate = None
        self.last_route_active_mask = None
        self._route_selection_extension = None
        self._route_selection_layer_name = ""
        self._router_logit_extension = None
        self._route_score_extension = None
        self._router_distillation_extension = None
        self.training_router_distillation = None
        self.training_gradient_probabilities = None
        self._balance_gradient_ratio_capture = False
        self._expert_choice_extension = None
        self.decoupled_routing = None
        self.training_expert_choice_decision = None
        self._lightweight_expert_extension = None
        self.training_lightweight_balance_loss = None
        self.training_lightweight_z_loss = None
        self.last_logical_top_indices = None
        self.last_logical_top_scores = None
        self.runtime_lightweight_decision = None

    def enable_int8_weight(self, calibrated_scale=None) -> None:
        """Replace a frozen FP weight with symmetric per-output-channel INT8."""
        if hasattr(self, "weight_int8"):
            if calibrated_scale is not None:
                requested = torch.as_tensor(
                    calibrated_scale,
                    dtype=self.weight_scale.dtype,
                    device=self.weight_scale.device,
                )
                if not torch.equal(requested, self.weight_scale):
                    raise ValueError(
                        "Router INT8 storage is already initialized with different "
                        "calibration scales."
                    )
            return
        if hasattr(self, "parametrizations") and "weight" in self.parametrizations:
            raise ValueError(
                "INT8 router storage cannot be combined with a router adapter."
            )
        weight = self._parameters.get("weight")
        if weight is None:
            raise RuntimeError("Router has no floating-point weight to quantize.")
        if bool(weight.requires_grad):
            raise ValueError(
                "INT8 router storage requires a frozen router; disable router "
                "training or router-targeted adapters."
            )
        resolved = weight.detach().float()
        if calibrated_scale is None:
            scale = (
                resolved.abs()
                .amax(dim=1)
                .clamp_min(torch.finfo(torch.float32).eps)
                / 127.0
            )
        else:
            scale = torch.as_tensor(
                calibrated_scale,
                dtype=torch.float32,
                device=resolved.device,
            )
            if (
                scale.shape != (int(self.num_experts),)
                or not bool(torch.isfinite(scale).all().item())
                or bool(torch.any(scale <= 0).item())
            ):
                raise ValueError(
                    "Calibrated router scale must be finite, positive, and match "
                    "the expert axis."
                )
        quantized = torch.round(resolved / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        del self._parameters["weight"]
        self.register_buffer("weight_int8", quantized.contiguous(), persistent=True)
        self.register_buffer("weight_scale", scale.contiguous(), persistent=True)

    def _execution_weight(self, *, device, dtype):
        if hasattr(self, "weight_int8"):
            return (
                self.weight_int8.to(device=device, dtype=dtype)
                * self.weight_scale.to(device=device, dtype=dtype).unsqueeze(1)
            )
        return self.weight.to(device=device, dtype=dtype)

    def router_weight_shape(self) -> tuple[int, ...]:
        weight = self.weight_int8 if hasattr(self, "weight_int8") else self.weight
        return tuple(int(dim) for dim in weight.shape)

    def router_weight_dtype(self) -> str:
        weight = self.weight_int8 if hasattr(self, "weight_int8") else self.weight
        return str(weight.dtype)

    def set_expert_choice_extension(self, extension) -> None:
        if extension is not None and not callable(extension):
            raise TypeError("expert-choice extension must be callable or None")
        self._expert_choice_extension = extension

    def set_balance_gradient_ratio_capture(self, enabled: bool) -> None:
        self._balance_gradient_ratio_capture = bool(enabled)
        if not self._balance_gradient_ratio_capture:
            self.training_gradient_probabilities = None

    def set_decoupled_routing(self, conditioner) -> None:
        if conditioner is not None and not isinstance(conditioner, nn.Module):
            raise TypeError("decoupled router conditioner must be a module or None")
        self.decoupled_routing = conditioner

    def forward_expert_choice(
        self,
        expert_hidden,
        router_hidden,
        timestep_hidden,
    ):
        if self._expert_choice_extension is None:
            raise RuntimeError("Expert-Choice routing was not configured.")
        with torch.amp.autocast(expert_hidden.device.type, enabled=False):
            if self.decoupled_routing is None:
                routing_hidden = expert_hidden
                logits = F.linear(
                    routing_hidden.float(),
                    self._execution_weight(
                        device=routing_hidden.device,
                        dtype=torch.float32,
                    ),
                )
            else:
                if router_hidden is None:
                    raise ValueError(
                        "Decoupled Expert-Choice requires unmodulated router input."
                    )
                routing_hidden = router_hidden
                logits = self.decoupled_routing(
                    content_tokens=routing_hidden,
                    timestep_hidden=timestep_hidden,
                    content_weight=self._execution_weight(
                        device=routing_hidden.device,
                        dtype=torch.float32,
                    ),
                )
        score_logits = self._transform_router_logits(logits)
        decision = self._expert_choice_extension(score_logits, expert_hidden.dtype)
        self.training_expert_choice_decision = decision if self.training else None
        self.last_scores = score_logits.float().softmax(dim=-1).reshape(
            -1, int(self.num_experts)
        ).detach()
        self.training_logits = logits if self.training else None
        self.training_scores = self.last_scores if self.training else None
        if self.training and self._balance_gradient_ratio_capture:
            gradient_probabilities = getattr(
                decision, "router_probabilities", None
            )
            if gradient_probabilities is None:
                gradient_probabilities = score_logits.float().softmax(dim=-1)
            self.training_gradient_probabilities = gradient_probabilities
        else:
            self.training_gradient_probabilities = None
        self.training_router_distillation = (
            self._router_distillation_extension(
                routing_hidden.reshape(-1, routing_hidden.shape[-1]),
                logits.reshape(-1, logits.shape[-1]),
            )
            if self.training and self._router_distillation_extension is not None
            else None
        )
        return decision

    def set_router_logit_extension(self, *, layer_name, transform) -> None:
        """Install an optional provider-owned transform before score evaluation."""
        if transform is not None and not callable(transform):
            raise TypeError("router logit extension must be callable or None")
        self._route_selection_layer_name = str(layer_name)
        self._router_logit_extension = transform

    def _transform_router_logits(self, logits):
        if self._router_logit_extension is None:
            return logits
        return self._router_logit_extension(
            self._route_selection_layer_name,
            logits,
            training=bool(self.training),
        )

    def set_route_selection_extension(self, *, layer_name, selector) -> None:
        """Install an optional host-owned top-k selector callable."""
        if selector is not None and not callable(selector):
            raise TypeError("route selection extension must be callable or None")
        self._route_selection_layer_name = str(layer_name)
        self._route_selection_extension = selector

    def set_route_score_extension(self, regularizer) -> None:
        if regularizer is not None and not callable(regularizer):
            raise TypeError("route score extension must be callable or None")
        self._route_score_extension = regularizer

    def set_lightweight_expert_extension(self, extension) -> None:
        """Install an optional logical expert pool without changing base weights."""

        self._lightweight_expert_extension = extension

    def set_router_distillation_extension(self, extension) -> None:
        if extension is not None and not callable(extension):
            raise TypeError("router distillation extension must be callable or None")
        self._router_distillation_extension = extension

    def set_dataset_routing_runtime(
        self,
        *,
        mode: str,
        domain_experts: tuple[tuple[int, ...], ...],
        prior_scale: float,
        hard_active: bool,
    ) -> None:
        self._dataset_routing_mode = str(mode).strip().lower()
        self._dataset_domain_experts = tuple(
            tuple(int(value) for value in expert_ids)
            for expert_ids in domain_experts
        )
        self._dataset_prior_scale = float(prior_scale)
        self._dataset_hard_active = bool(hard_active)

    def _dataset_routing_inputs(self, logits):
        mode = self._dataset_routing_mode
        self._dataset_token_allowed = None
        if not self.training or mode not in {"soft_affinity", "hard_affinity"}:
            return logits, None
        batch_size = int(getattr(self, "training_batch_size", 0))
        tokens_per_sample = int(getattr(self, "training_tokens_per_sample", 0))
        if (
            batch_size <= 0
            or tokens_per_sample <= 0
            or batch_size * tokens_per_sample != int(logits.shape[0])
            or len(self._dataset_domain_experts) != batch_size
        ):
            raise RuntimeError(
                "Dataset routing affinity requires one domain expert set per "
                "sample and valid router token-shape metadata."
            )
        allowed = torch.zeros(
            (batch_size, int(self.num_experts)),
            device=logits.device,
            dtype=torch.bool,
        )
        for sample_index, expert_ids in enumerate(self._dataset_domain_experts):
            index = torch.as_tensor(expert_ids, device=logits.device, dtype=torch.long)
            allowed[sample_index].scatter_(0, index, True)
        token_allowed = allowed.repeat_interleave(tokens_per_sample, dim=0)
        self._dataset_token_allowed = token_allowed
        if mode == "soft_affinity":
            prior = token_allowed.to(dtype=logits.dtype) * float(self._dataset_prior_scale)
            return logits + prior, None
        if self._dataset_hard_active:
            return logits, token_allowed
        return logits, None

    def set_subset_runtime(
        self,
        *,
        active: bool,
        size: int,
        pool_factor: float,
        seed: int,
        kl_weight: float,
    ) -> None:
        """Install the per-step expert-subset routing state (pipeline-driven)."""
        self._subset_active = bool(active)
        self._subset_size = int(size)
        self._subset_pool_factor = float(pool_factor)
        self._subset_seed = int(seed)
        self._subset_kl_weight = float(kl_weight)

    def clear_subset_runtime(self) -> None:
        self._subset_active = False
        self._subset_size = 0
        self.training_subset_kl = None
        self.last_subset_ids = None
        self.last_unique_experts = None

    def _subset_should_apply(self) -> bool:
        return bool(
            self.training
            and self._subset_active
            and 0 < int(self._subset_size) < int(self.num_experts)
            and int(self._subset_size) >= int(self.top_k)
        )

    def _apply_expert_subset(self, scores, scores_for_choice, logits):
        """Mask router scores to a per-step hot-biased expert subset.

        Returns the masked ``scores_for_choice`` (out-of-subset -> -inf) so the
        downstream top-k re-routes tokens within the subset. Records the subset
        ids / unique-count and (when the router is trainable and a KL weight is
        set) the reverse-KL router-consistency term.
        """
        from mirai.core.moe.routing.subset import (
            routing_mass_from_selection,
            select_expert_subset,
            subset_mask_from_ids,
            subset_reverse_kl,
        )

        num_experts = int(self.num_experts)
        # Natural top-k (unmasked) purely to weight the sampler by routing mass.
        natural_top = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        routing_mass = routing_mass_from_selection(
            scores, natural_top, num_experts=num_experts
        )
        generator = torch.Generator(device=routing_mass.device)
        generator.manual_seed(int(self._subset_seed))
        subset_ids = select_expert_subset(
            routing_mass,
            size=int(self._subset_size),
            pool_factor=float(self._subset_pool_factor),
            generator=generator,
        )
        mask = subset_mask_from_ids(
            num_experts, subset_ids, device=scores_for_choice.device
        )
        neg_inf = torch.finfo(scores_for_choice.dtype).min
        masked = scores_for_choice.masked_fill(~mask.unsqueeze(0), neg_inf)
        self.last_subset_ids = subset_ids.detach()
        self.last_unique_experts = int(subset_ids.numel())
        if (
            float(self._subset_kl_weight) > 0.0
            and "weight" in self._parameters
            and self.weight.requires_grad
        ):
            self.training_subset_kl = subset_reverse_kl(logits, subset_ids) * float(
                self._subset_kl_weight
            )
        else:
            self.training_subset_kl = None
        return masked

    def _group_limited_choice_scores(self, scores_for_choice):
        seq_len = scores_for_choice.shape[0]
        experts_per_group = self.num_experts // self.n_group
        grouped = scores_for_choice.view(seq_len, self.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(seq_len, self.n_group, experts_per_group)
            .reshape(seq_len, -1)
        )
        masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        return masked

    def _group_limited_topk(self, scores_for_choice):
        masked = self._group_limited_choice_scores(scores_for_choice)
        return torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

    def _forward_lightweight_experts(self, tokens, logits):
        extension = self._lightweight_expert_extension
        logical_logits = extension.append_logits(tokens, logits)
        score_logits = self._transform_router_logits(logical_logits)
        physical_choice_transform = (
            self._group_limited_choice_scores
            if self.n_group is not None and self.n_group > 1
            else None
        )
        decision = extension.route(
            score_logits,
            physical_correction_bias=self.e_score_correction_bias,
            score_func=self.score_func,
            route_scale=float(self.route_scale),
            physical_choice_transform=physical_choice_transform,
            batch_size=int(getattr(self, "training_batch_size", 0)),
            tokens_per_sample=int(
                getattr(self, "training_tokens_per_sample", 0)
            ),
            training=bool(self.training),
        )
        top_indices = decision.physical_indices
        top_scores = decision.physical_scores
        active_mask = decision.physical_active_mask
        if self._route_score_extension is not None and self.training:
            top_scores = self._route_score_extension(
                self._route_selection_layer_name,
                top_indices,
                top_scores,
                training=True,
            )
            active_mask = active_mask & (top_scores != 0)
        self.last_route_active_mask = active_mask.detach()
        self.last_top_indices = top_indices.detach()
        self.last_top_scores = top_scores.detach()
        self.last_scores = decision.logical_probabilities[
            :, : self.num_experts
        ].detach()
        self.last_logical_top_indices = decision.logical_indices.detach()
        self.last_logical_top_scores = decision.logical_selected_scores.detach()
        self.runtime_lightweight_decision = decision
        if self.training:
            self.training_top_indices = top_indices
            self.training_unbiased_top_indices = top_indices
            self.training_logits = logits
            self.training_scores = decision.logical_probabilities[
                :, : self.num_experts
            ]
            self.training_gradient_probabilities = (
                decision.logical_probabilities
                if self._balance_gradient_ratio_capture
                else None
            )
            self.training_lightweight_balance_loss = (
                decision.load_balance_loss
            )
            self.training_lightweight_z_loss = decision.z_loss
            self.training_router_distillation = (
                self._router_distillation_extension(tokens, logits)
                if self._router_distillation_extension is not None
                else None
            )
        return (
            top_indices,
            top_scores.to(tokens.dtype),
            logits,
            decision.logical_probabilities[:, : self.num_experts],
            decision.logical_probabilities[:, : self.num_experts],
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        logit_delta: Optional[torch.Tensor] = None,
        valid_token_mask: Optional[torch.Tensor] = None,
        route_scope_mask: Optional[torch.Tensor] = None,
    ):
        if not self.training:
            self.training_gradient_probabilities = None
        if _infer_bf16_fastmath(self, self.training):
            # Opt-in eval fast-math: run the router projection (and the softmax /
            # sigmoid it feeds) in bf16 rather than forcing the fp32 promotion.
            # Weights live in fp32 (kept there by should_keep_in_fp32), so cast to
            # the token dtype to avoid a mixed-dtype matmul.
            logits = F.linear(
                tokens,
                self._execution_weight(device=tokens.device, dtype=tokens.dtype),
            )
        else:
            with torch.amp.autocast(tokens.device.type, enabled=False):
                logits = F.linear(
                    tokens.float(),
                    self._execution_weight(device=tokens.device, dtype=torch.float32),
                )
        if logit_delta is not None:
            if tuple(logit_delta.shape) != tuple(logits.shape):
                raise ValueError(
                    "Router logit delta must match the native router logits shape."
                )
            logits = logits + logit_delta.to(
                device=logits.device,
                dtype=logits.dtype,
            )
        if self._lightweight_expert_extension is not None:
            return self._forward_lightweight_experts(tokens, logits)
        score_logits = self._transform_router_logits(logits)
        routed_logits, hard_affinity_mask = self._dataset_routing_inputs(
            score_logits
        )
        if self.score_func == "softmax":
            scores = F.softmax(routed_logits, dim=-1)
        else:
            scores = routed_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        # Stochastic per-step expert-subset routing (opt-in). Mask the router
        # scores to the sampled subset so tokens re-route within it; when active
        # we use plain masked top-k (bypassing group-limiting, which is a
        # pretraining load device -- documented deviation). Reduces to the exact
        # full-expert path once the subset spans all experts.
        if hard_affinity_mask is not None:
            scores_for_choice = scores_for_choice.masked_fill(
                ~hard_affinity_mask, torch.finfo(scores_for_choice.dtype).min
            )
            top_indices = torch.topk(
                scores_for_choice, k=self.top_k, dim=-1, sorted=False
            )[1]
        elif self._subset_should_apply():
            scores_for_choice = self._apply_expert_subset(
                scores, scores_for_choice, logits
            )
            top_indices = torch.topk(
                scores_for_choice, k=self.top_k, dim=-1, sorted=False
            )[1]
        elif self.n_group is not None and self.n_group > 1:
            top_indices = self._group_limited_topk(scores_for_choice)
        else:
            top_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        top_scores = scores.gather(1, top_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale
        apply_route_extension = bool(
            self._route_selection_extension is not None
            and (
                self.training
                or bool(
                    getattr(
                        self._route_selection_extension,
                        "_mirai_apply_in_eval",
                        False,
                    )
                )
            )
        )
        if apply_route_extension:
            selected = self._route_selection_extension(
                self._route_selection_layer_name,
                scores_for_choice,
                top_indices,
                training=bool(self.training),
                tokens=tokens,
                score_logits=score_logits,
                native_gate_scores=scores,
                native_top_weights=top_scores,
                valid_token_mask=valid_token_mask,
                route_scope_mask=route_scope_mask,
                norm_topk_prob=bool(self.norm_topk_prob),
                route_scale=float(self.route_scale),
                choice_score_transform=(
                    self._group_limited_choice_scores
                    if self.n_group is not None and self.n_group > 1
                    else None
                ),
            )
            if selected is not None:
                if torch.is_tensor(selected):
                    top_indices = selected
                    top_scores = scores.gather(1, top_indices)
                    if self.top_k > 1 and self.norm_topk_prob:
                        top_scores = top_scores / (
                            top_scores.sum(dim=-1, keepdim=True) + 1e-20
                        )
                    top_scores = top_scores * self.route_scale
                else:
                    top_indices = selected.top_indices
                    top_scores = selected.top_weights
                if tuple(top_indices.shape) != tuple(top_scores.shape):
                    raise RuntimeError(
                        "Route-selection extension returned mismatched indices and weights."
                    )
        if self._route_score_extension is not None and self.training:
            top_scores = self._route_score_extension(
                self._route_selection_layer_name,
                top_indices,
                top_scores,
                training=True,
            )
            self.last_route_active_mask = (top_scores != 0).detach()
        else:
            self.last_route_active_mask = None
        self.last_top_indices = top_indices.detach()
        self.last_top_scores = top_scores.detach()
        self.last_scores = scores.detach()
        token_allowed = self._dataset_token_allowed
        if token_allowed is None:
            self.last_dataset_affinity_hit_rate = None
        else:
            affinity_hits = token_allowed.gather(1, top_indices)
            self.last_dataset_affinity_hit_rate = float(
                affinity_hits.float().mean().detach().cpu().item()
            )
        if self.training:
            # Training-only bookkeeping for the router auxiliary losses
            # (_block_router_auxiliary_terms). The unbiased (bias-free) scores and
            # their top-k, plus these training_* attributes, are consumed solely by
            # the aux-loss path -- nothing in the eval/denoise loop reads them. In
            # eval they stay unset (absent) so the extra softmax/sigmoid + top-k
            # over [N, num_experts] is skipped entirely. Computed here (after
            # selection) rather than inline so the eval fast-path drops the dead
            # work; the values are bit-identical to the pre-change training path.
            if self.score_func == "softmax":
                unbiased_scores = F.softmax(score_logits, dim=-1)
            else:
                unbiased_scores = score_logits.sigmoid()
            unbiased_top_indices = torch.topk(
                unbiased_scores, k=self.top_k, dim=-1, sorted=False
            )[1]
            self.training_top_indices = top_indices
            self.training_unbiased_top_indices = unbiased_top_indices
            self.training_logits = logits
            self.training_scores = scores
            self.training_gradient_probabilities = (
                scores if self._balance_gradient_ratio_capture else None
            )
            self.training_router_distillation = (
                self._router_distillation_extension(tokens, logits)
                if self._router_distillation_extension is not None
                else None
            )
        return top_indices, top_scores.to(tokens.dtype), logits, scores, scores_for_choice


class LingBotVideoGroupedExperts(nn.Module):
    """Weight layout matches GroupedExperts: w1 [E,I,H], w2 [E,H,I], w3 [E,I,H]. Eager per-expert compute."""

    mirai_expert_tensor_host = True

    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self._mirai_linear_extension = None

    def set_linear_extension(self, extension):
        """Install a non-owning expert-linear executor supplied by the provider."""
        self._mirai_linear_extension = None if extension is None else (extension,)

    def linear_extension(self):
        """Return the installed executor without registering duplicate parameters."""
        return (
            None
            if self._mirai_linear_extension is None
            else self._mirai_linear_extension[0]
        )

    def set_routed_adapter_tokens_per_sample(self, value):
        extension = self.linear_extension()
        setter = getattr(
            extension,
            "set_routed_adapter_tokens_per_sample",
            None,
        )
        if callable(setter):
            setter(int(value))


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class LingBotVideoSparseMoeBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k,
                 moe_intermediate_size, score_func, norm_topk_prob, n_group, topk_group,
                 routed_scaling_factor, n_shared_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router = LingBotVideoRouter(
            hidden_size, num_experts, top_k, score_func, norm_topk_prob,
            n_group, topk_group, routed_scaling_factor,
        )
        self.experts = LingBotVideoGroupedExperts(num_experts, hidden_size, moe_intermediate_size)
        self._mirai_expert_output_observer = None
        self._mirai_expert_intermediate_observer = None
        self._mirai_importance_calibration_observer = None
        self._sglang_w13_cache: Optional[torch.Tensor] = None
        self._sglang_w13_cache_key = None
        self._sglang_fp8_cache = None
        self._sglang_fp8_cache_key = None
        self.shared_experts = None
        self.adjugate_experts = None
        self._mirai_adjugate_expert_extension = None
        self._mirai_moe_kernel_backend = None
        self._mirai_token_chunk_policy = None
        self.chain_of_experts = None
        if n_shared_experts is not None and n_shared_experts > 0:
            self.shared_experts = LingBotVideoMLP(
                hidden_size, moe_intermediate_size * n_shared_experts
            )

    def set_token_chunk_policy(self, policy):
        self._mirai_token_chunk_policy = policy

    def set_chain_of_experts_extension(self, extension):
        """Attach an optional model-agnostic two-step MoE extension."""

        if extension is not None:
            for method_name in (
                "router_logit_delta",
                "continuation_input",
                "combine",
                "record_routes",
            ):
                if not callable(getattr(extension, method_name, None)):
                    raise TypeError(
                        "Chain-of-Experts extension must expose "
                        f"{method_name}()."
                    )
        self.chain_of_experts = extension

    def set_adjugate_expert_extension(self, extension):
        """Install a provider-owned grouped adjugate-expert executor."""

        if extension is not None and not callable(
            getattr(extension, "output_contribution", None)
        ):
            raise TypeError(
                "Adjugate expert extension must expose output_contribution()."
            )
        self._mirai_adjugate_expert_extension = (
            None if extension is None else (extension,)
        )

    def set_importance_calibration_observer(self, observer):
        if observer is None or not callable(getattr(observer, "record", None)):
            raise TypeError("Importance calibration observer must expose record().")
        if self._mirai_importance_calibration_observer is not None:
            raise RuntimeError("An importance calibration observer is already attached.")
        self._mirai_importance_calibration_observer = observer

    def clear_importance_calibration_observer(self):
        self._mirai_importance_calibration_observer = None

    def set_expert_output_observer(self, observer):
        self._mirai_expert_output_observer = observer
        setter = getattr(self.experts, "set_routed_output_observer", None)
        if callable(setter):
            setter(observer)

    def get_expert_output_observer(self):
        return self._mirai_expert_output_observer

    def _bind_expert_output_routes(
        self,
        top_indices: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> None:
        observer = self._mirai_expert_output_observer
        if observer is None:
            return
        bind_routes = getattr(observer, "bind_routes", None)
        if callable(bind_routes):
            bind_routes(top_indices, top_scores)

    def set_expert_intermediate_observer(self, observer):
        self._mirai_expert_intermediate_observer = observer
        setter = getattr(self.experts, "set_routed_intermediate_observer", None)
        if callable(setter):
            setter(observer)

    def _capture_expert_intermediate(self, hidden: torch.Tensor) -> None:
        observer = self._mirai_expert_intermediate_observer
        if self.training and observer is not None and bool(observer.is_enabled):
            observer.capture_sorted_chunk(hidden)

    @staticmethod
    def _expert_choice_token_routes(decision, *, tokens_per_sample, num_experts):
        indices = decision.expert_token_indices
        weights = decision.expert_token_weights
        batch, experts, capacity = indices.shape
        flat_tokens = (
            indices
            + torch.arange(batch, device=indices.device)[:, None, None]
            * int(tokens_per_sample)
        ).reshape(-1)
        flat_experts = (
            torch.arange(experts, device=indices.device)[None, :, None]
            .expand(batch, experts, capacity)
            .reshape(-1)
        )
        flat_weights = weights.reshape(-1)
        active = flat_weights != 0
        flat_tokens = flat_tokens[active]
        flat_experts = flat_experts[active]
        flat_weights = flat_weights[active]
        multiplicity = torch.bincount(
            flat_tokens, minlength=int(batch * tokens_per_sample)
        )
        width = max(1, int(multiplicity.max().item()))
        order = torch.argsort(flat_tokens, stable=True)
        sorted_tokens = flat_tokens[order]
        starts = torch.cumsum(multiplicity, dim=0) - multiplicity
        slots = torch.arange(order.numel(), device=indices.device) - torch.repeat_interleave(
            starts, multiplicity
        )
        token_experts = torch.zeros(
            (batch * tokens_per_sample, width),
            device=indices.device,
            dtype=torch.long,
        )
        token_weights = torch.zeros(
            (batch * tokens_per_sample, width),
            device=weights.device,
            dtype=weights.dtype,
        )
        token_experts[sorted_tokens, slots] = flat_experts[order]
        token_weights[sorted_tokens, slots] = flat_weights[order]
        if bool((token_experts < 0).any()) or bool((token_experts >= num_experts).any()):
            raise RuntimeError("Expert-Choice produced an invalid expert id.")
        return token_experts, token_weights

    def _run_with_intermediate_capture(
        self,
        runner,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        sorted_positions: torch.Tensor,
        *,
        num_tokens: int,
        top_k: int,
        route_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observer = self._mirai_expert_intermediate_observer
        active = self.training and observer is not None and bool(observer.is_enabled)
        if not active:
            return runner(
                tokens,
                counts,
                route_token_indices=route_token_indices,
            )
        observer.begin_sorted(
            sorted_positions,
            num_tokens=int(num_tokens),
            top_k=int(top_k),
            device=tokens.device,
        )
        try:
            output = runner(
                tokens,
                counts,
                route_token_indices=route_token_indices,
            )
            observer.end_sorted()
            return output
        except BaseException:
            observer.abort_capture()
            raise

    def _reorder_tokens(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        num_experts: int,
        *,
        drop_slots: bool = True,
    ):
        backend = _moe_reorder_backend(self)
        if backend in {"triton_pack", "pack", "triton"}:
            return reorder_tokens_triton_pack(tokens, top_scores, top_indices, num_experts)
        if backend not in {"sort", "argsort", "default"}:
            raise ValueError(
                f"Unsupported model.params.moe_reorder_backend={backend!r}; "
                "expected sort or triton_pack"
            )
        num_tokens = tokens.shape[0]
        top_k = top_indices.shape[1]
        flat_scores = top_scores.reshape(-1)
        flat_indices = top_indices.reshape(-1)
        if drop_slots:
            # Drop zero-score slots (training / subset / min-token gating). The
            # single-arg where is a nonzero with a device->host sync per call —
            # kept only where slots can actually be zeroed.
            active_positions = torch.where(flat_scores != 0)[0]
            active_experts = flat_indices[active_positions]

            counts = torch.zeros(num_experts, device=tokens.device, dtype=torch.int64)
            counts.scatter_add_(
                0, active_experts, torch.ones_like(active_experts, dtype=torch.int64)
            )

            sort_order = torch.argsort(active_experts, stable=True)
            sorted_positions = active_positions[sort_order]
        else:
            # Sync-free eval path (expert_selection="all": no slot is ever
            # zeroed, so the filter is the identity). Bit-identical layout:
            # active_positions == arange(N*top_k), hence
            # sorted_positions == argsort(flat_indices, stable) and the
            # scatter_add over flat_indices equals the filtered one. Emits
            # zero device->host copies (B5-R2).
            counts = torch.zeros(num_experts, device=tokens.device, dtype=torch.int64)
            counts.scatter_add_(
                0, flat_indices, torch.ones_like(flat_indices, dtype=torch.int64)
            )
            sorted_positions = torch.argsort(flat_indices, stable=True)
        sorted_scores = flat_scores[sorted_positions]
        original_token_idx = sorted_positions // top_k
        permuted_tokens = tokens[original_token_idx]
        return permuted_tokens, counts, sorted_positions, sorted_scores, num_tokens, top_k

    @staticmethod
    def _pad_grouped_tokens_loop(tokens: torch.Tensor, counts: torch.Tensor, align: int = 8):
        num_tokens = tokens.shape[0]
        num_experts = int(counts.shape[0])
        max_len = _round_up_to_multiple(num_tokens + num_experts * align, align)
        counts_i64 = counts.to(torch.int64)
        total_per_expert = torch.clamp_min(counts_i64, align)
        aligned_counts = (
            (total_per_expert + align - 1) // align * align
        ).to(torch.int32)
        write_offsets = torch.cumsum(aligned_counts, dim=0) - aligned_counts
        start_indices = torch.cumsum(counts_i64, dim=0) - counts_i64

        fill_value = num_tokens
        permuted_indices = torch.full(
            (max_len,), fill_value, dtype=torch.int64, device=tokens.device
        )
        for expert_idx in range(num_experts):
            length = int(counts_i64[expert_idx].item())
            if length == 0:
                continue
            write_start = int(write_offsets[expert_idx].item())
            start = int(start_indices[expert_idx].item())
            permuted_indices[write_start:write_start + length] = torch.arange(
                start, start + length, device=tokens.device, dtype=torch.int64
            )

        tokens_with_pad = torch.vstack((tokens, tokens.new_zeros((tokens.shape[-1],))))
        input_shape = tokens_with_pad.shape
        return input_shape, tokens_with_pad[permuted_indices], permuted_indices, aligned_counts

    @staticmethod
    def _pad_grouped_tokens_vectorized(tokens: torch.Tensor, counts: torch.Tensor, align: int = 8):
        num_tokens = tokens.shape[0]
        num_experts = int(counts.shape[0])
        max_len = _round_up_to_multiple(num_tokens + num_experts * align, align)
        counts_i64 = counts.to(torch.int64)
        total_per_expert = torch.clamp_min(counts_i64, align)
        aligned_counts_i64 = (total_per_expert + align - 1) // align * align
        write_offsets = torch.cumsum(aligned_counts_i64, dim=0) - aligned_counts_i64
        end_offsets = torch.cumsum(aligned_counts_i64, dim=0)
        start_indices = torch.cumsum(counts_i64, dim=0) - counts_i64

        slots = torch.arange(max_len, dtype=torch.int64, device=tokens.device)
        expert_idx = torch.bucketize(slots, end_offsets, right=True)
        valid_expert = expert_idx < num_experts
        safe_expert_idx = expert_idx.clamp(max=num_experts - 1)
        local_idx = slots - write_offsets[safe_expert_idx]
        source_idx = start_indices[safe_expert_idx] + local_idx
        valid = valid_expert & (local_idx < counts_i64[safe_expert_idx])
        fill = torch.full_like(source_idx, num_tokens)
        permuted_indices = torch.where(valid, source_idx, fill)

        tokens_with_pad = torch.vstack((tokens, tokens.new_zeros((tokens.shape[-1],))))
        input_shape = tokens_with_pad.shape
        return (
            input_shape,
            tokens_with_pad[permuted_indices],
            permuted_indices,
            aligned_counts_i64.to(torch.int32),
        )

    def _pad_grouped_tokens(
        self,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        align: int = 8,
        *,
        training: bool = True,
    ):
        backend = _moe_pad_backend(self, training)
        if backend in {"loop", "default"}:
            return LingBotVideoSparseMoeBlock._pad_grouped_tokens_loop(tokens, counts, align)
        if backend in {"vectorized", "torch"}:
            return LingBotVideoSparseMoeBlock._pad_grouped_tokens_vectorized(tokens, counts, align)
        raise ValueError(
            f"Unsupported model.params.moe_pad_backend={backend!r}; expected loop or vectorized"
        )

    @staticmethod
    def _unpad_grouped_tokens(output: torch.Tensor, input_shape: torch.Size, permuted_indices: torch.Tensor):
        unpermuted = output.new_empty(input_shape)
        unpermuted[permuted_indices, :] = output
        return unpermuted[:-1]

    def _run_grouped_experts(
        self,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        *,
        route_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._mirai_importance_calibration_observer is not None:
            return self._run_experts_for_loop(
                tokens,
                counts,
                route_token_indices=route_token_indices,
            )
        batched_runner = getattr(self.experts, "run_batched", None)
        if callable(batched_runner) and tokens.device.type == "cuda":
            capability = torch.cuda.get_device_capability(tokens.device)
            if capability < (9, 0):
                return batched_runner(tokens, counts)
        # PyTorch currently exposes ``_grouped_mm`` on builds where the kernel
        # itself is SM90-only. Attribute presence is therefore not a sufficient
        # capability probe: supported SM80/SM89 Mirai hosts must retain the
        # reference expert loop instead of failing inside the CUDA op.
        if not _torch_grouped_mm_supported(tokens.device):
            return self._run_experts_for_loop(tokens, counts)
        input_shape, padded_tokens, permuted_indices, aligned_counts = self._pad_grouped_tokens(
            tokens, counts, training=self.training
        )
        offsets = torch.cumsum(aligned_counts, dim=0, dtype=torch.int32)
        grouped_linear = getattr(self.experts, "grouped_linear", None)
        extension_getter = getattr(self.experts, "linear_extension", None)
        extension = extension_getter() if callable(extension_getter) else None
        grouped_tokens = padded_tokens.bfloat16()
        if extension is not None:
            route_gate = extension.resolve_route_gate(route_token_indices, counts)
            if route_gate is not None:
                route_gate_with_pad = torch.cat(
                    (
                        route_gate,
                        torch.zeros(
                            1,
                            device=route_gate.device,
                            dtype=torch.bool,
                        ),
                    )
                )
                route_gate = route_gate_with_pad[permuted_indices]
            h = F.silu(extension.grouped_linear(
                self.experts,
                "w1",
                grouped_tokens,
                offsets=offsets,
                route_gate=route_gate,
            ))
            h = h * extension.grouped_linear(
                self.experts,
                "w3",
                grouped_tokens,
                offsets=offsets,
                route_gate=route_gate,
            )
            self._capture_expert_intermediate(
                self._unpad_grouped_tokens(
                    h,
                    torch.Size((*input_shape[:-1], h.shape[-1])),
                    permuted_indices,
                )
            )
            out = extension.grouped_linear(
                self.experts,
                "w2",
                h,
                offsets=offsets,
                route_gate=route_gate,
            ).type_as(padded_tokens)
        elif callable(grouped_linear):
            h = F.silu(grouped_linear("w1", grouped_tokens, offsets=offsets))
            h = h * grouped_linear("w3", grouped_tokens, offsets=offsets)
            self._capture_expert_intermediate(
                self._unpad_grouped_tokens(
                    h,
                    torch.Size((*input_shape[:-1], h.shape[-1])),
                    permuted_indices,
                )
            )
            out = grouped_linear("w2", h, offsets=offsets).type_as(padded_tokens)
        else:
            h = F.silu(
                torch._grouped_mm(
                    grouped_tokens,
                    self.experts.w1.bfloat16().transpose(-2, -1),
                    offs=offsets,
                )
            )
            h = h * torch._grouped_mm(
                grouped_tokens,
                self.experts.w3.bfloat16().transpose(-2, -1),
                offs=offsets,
            )
            self._capture_expert_intermediate(
                self._unpad_grouped_tokens(
                    h,
                    torch.Size((*input_shape[:-1], h.shape[-1])),
                    permuted_indices,
                )
            )
            out = torch._grouped_mm(
                h,
                self.experts.w2.bfloat16().transpose(-2, -1),
                offs=offsets,
            ).type_as(padded_tokens)
        return self._unpad_grouped_tokens(out, input_shape, permuted_indices)

    def _run_experts_for_loop(
        self,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        *,
        route_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        extension_getter = getattr(self.experts, "linear_extension", None)
        extension = extension_getter() if callable(extension_getter) else None
        if extension is not None:
            return extension.run_for_loop(
                self.experts,
                tokens,
                counts,
                route_token_indices=route_token_indices,
            )
        int8_runner = getattr(self.experts, "run_for_loop", None)
        if callable(int8_runner):
            return int8_runner(tokens, counts)
        count_list = counts.tolist()
        splits = torch.split(tokens, count_list, dim=0)
        outputs = []
        for expert_idx, expert_tokens in enumerate(splits):
            if expert_tokens.numel() == 0:
                continue
            observer = self._mirai_importance_calibration_observer
            if observer is not None:
                observer.record(expert_idx, ("w1", "w3"), expert_tokens)
            h = F.silu(expert_tokens @ self.experts.w1[expert_idx].transpose(-2, -1))
            h = h * (expert_tokens @ self.experts.w3[expert_idx].transpose(-2, -1))
            self._capture_expert_intermediate(h)
            if observer is not None:
                observer.record(expert_idx, "w2", h)
            h = h @ self.experts.w2[expert_idx].transpose(-2, -1)
            outputs.append(h)
        if not outputs:
            return tokens.new_zeros(tokens.shape)
        return torch.cat(outputs, dim=0)

    def _get_sglang_w13(self) -> torch.Tensor:
        key = (
            self.experts.w1.data_ptr(),
            self.experts.w3.data_ptr(),
            self.experts.w1.device,
            self.experts.w3.device,
            self.experts.w1.dtype,
            self.experts.w3.dtype,
        )
        if self._sglang_w13_cache is None or self._sglang_w13_cache_key != key:
            self._sglang_w13_cache = torch.cat(
                (self.experts.w1.bfloat16(), self.experts.w3.bfloat16()), dim=1
            ).contiguous()
            self._sglang_w13_cache_key = key
        return self._sglang_w13_cache

    @staticmethod
    def _quantize_fp8_weight_per_expert(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_float = weight.float()
        scale = fp8_scale_from_amax(weight_float.abs().amax(dim=(1, 2)))
        quantized = quantize_to_fp8_e4m3fn(weight_float, scale[:, None, None]).contiguous()
        return quantized, scale.contiguous()

    def _get_sglang_fp8_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            self.experts.w1.data_ptr(),
            self.experts.w2.data_ptr(),
            self.experts.w3.data_ptr(),
            self.experts.w1.device,
            self.experts.w2.device,
            self.experts.w3.device,
            self.experts.w1.dtype,
            self.experts.w2.dtype,
            self.experts.w3.dtype,
        )
        if self._sglang_fp8_cache is None or self._sglang_fp8_cache_key != key:
            w13 = torch.cat((self.experts.w1.float(), self.experts.w3.float()), dim=1).contiguous()
            w13_fp8, w13_scale = self._quantize_fp8_weight_per_expert(w13)
            w2_fp8, w2_scale = self._quantize_fp8_weight_per_expert(self.experts.w2)
            self._sglang_fp8_cache = (w13_fp8, w2_fp8, w13_scale, w2_scale)
            self._sglang_fp8_cache_key = key
        return self._sglang_fp8_cache

    def _run_sglang_triton_experts(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        ensure_sglang_moe_ready()
        topk_output = LightSglangStandardTopKOutput(
            top_scores.float(),
            top_indices.to(torch.int32),
            torch.empty(0, device=tokens.device),
        )
        runner_config = LightSglangMoeRunnerConfig(
            num_experts=self.num_experts,
            num_local_experts=self.num_experts,
            activation="silu",
            is_gated=True,
            inplace=False,
        )
        return sglang_fused_experts(
            tokens.contiguous().bfloat16(),
            self._get_sglang_w13(),
            self.experts.w2.bfloat16().contiguous(),
            topk_output,
            runner_config,
        ).type_as(tokens)

    def _run_sglang_triton_fp8_experts(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        ensure_sglang_moe_ready()
        topk_output = LightSglangStandardTopKOutput(
            top_scores.float(),
            top_indices.to(torch.int32),
            torch.empty(0, device=tokens.device),
        )
        runner_config = LightSglangMoeRunnerConfig(
            num_experts=self.num_experts,
            num_local_experts=self.num_experts,
            activation="silu",
            is_gated=True,
            inplace=False,
        )
        w13_fp8, w2_fp8, w13_scale, w2_scale = self._get_sglang_fp8_weights()
        return sglang_fused_experts(
            tokens.contiguous().bfloat16(),
            w13_fp8,
            w2_fp8,
            topk_output,
            runner_config,
            use_fp8_w8a8=True,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
        ).type_as(tokens)

    def _run_selected_experts(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        *,
        drop_slots: Optional[bool] = None,
        token_offset: int = 0,
    ) -> torch.Tensor:
        should_drop_slots = self.training if drop_slots is None else bool(drop_slots)
        if (
            self.training
            and self._mirai_expert_output_observer is not None
            and bool(self._mirai_expert_output_observer.is_enabled)
        ):
            self._bind_expert_output_routes(top_indices, top_scores)
        kernel_backend = self._mirai_moe_kernel_backend
        if kernel_backend is not None:
            direct_output = kernel_backend.execute_direct(
                self.experts,
                tokens,
                top_scores,
                top_indices,
            )
            if direct_output is not None:
                observer_is_unhandled = (
                    self._mirai_expert_output_observer is not None
                    and not bool(
                        getattr(
                            self.experts,
                            "supports_routed_output_observer",
                            False,
                        )
                    )
                )
                intermediate_is_unhandled = (
                    self._mirai_expert_intermediate_observer is not None
                    and not bool(
                        getattr(
                            self.experts,
                            "supports_routed_intermediate_observer",
                            False,
                        )
                    )
                )
                if (
                    self.training
                    and observer_is_unhandled
                    and bool(self._mirai_expert_output_observer.is_enabled)
                ):
                    raise RuntimeError(
                        "Expert-output orthogonality is unsupported by a direct "
                        "MoE kernel that does not expose routed expert outputs."
                    )
                if (
                    self.training
                    and intermediate_is_unhandled
                    and bool(self._mirai_expert_intermediate_observer.is_enabled)
                ):
                    raise RuntimeError(
                        "SwiGLU specialization is unsupported by a direct MoE "
                        "kernel that does not expose expert intermediates."
                    )
                return direct_output
            routed = kernel_backend.route(
                tokens,
                top_scores,
                top_indices,
                num_experts=self.router.num_experts,
            )
            expert_output = kernel_backend.compute(
                self.experts,
                routed,
                fallback=self._run_grouped_experts,
            )
            if (
                self.training
                and self._mirai_expert_output_observer is not None
                and bool(self._mirai_expert_output_observer.is_enabled)
            ):
                sorted_positions, _, _, _, top_k = routed.restore_state
                self._mirai_expert_output_observer.capture_sorted(
                    expert_output,
                    sorted_positions,
                    num_tokens=int(tokens.shape[0]),
                    top_k=int(top_k),
                )
            return kernel_backend.restore(expert_output, routed)
        backend = _moe_expert_backend(self)
        prefers_for_loop = getattr(self.experts, "prefers_for_loop", None)
        if callable(prefers_for_loop) and bool(prefers_for_loop()):
            backend = "loop"
        if backend in {"loop", "for_loop", "eager"}:
            (
                permuted_tokens,
                counts,
                sorted_positions,
                sorted_scores,
                num_tokens,
                top_k,
            ) = self._reorder_tokens(
                tokens,
                top_scores,
                top_indices,
                self.router.num_experts,
                drop_slots=should_drop_slots,
            )
            expert_output = self._run_with_intermediate_capture(
                self._run_experts_for_loop,
                permuted_tokens,
                counts,
                sorted_positions,
                num_tokens=int(num_tokens),
                top_k=int(top_k),
                route_token_indices=torch.div(
                    sorted_positions, top_k, rounding_mode="floor"
                )
                + int(token_offset),
            )
            if (
                self.training
                and self._mirai_expert_output_observer is not None
                and bool(self._mirai_expert_output_observer.is_enabled)
            ):
                self._mirai_expert_output_observer.capture_sorted(
                    expert_output,
                    sorted_positions,
                    num_tokens=int(num_tokens),
                    top_k=int(top_k),
                )
            return self._restore_tokens(
                expert_output,
                sorted_positions,
                sorted_scores,
                num_tokens,
                top_k,
            )
        if backend in {"grouped_mm", "torch_grouped_mm", "default"}:
            (
                permuted_tokens,
                counts,
                sorted_positions,
                sorted_scores,
                num_tokens,
                top_k,
            ) = self._reorder_tokens(
                tokens,
                top_scores,
                top_indices,
                self.router.num_experts,
                drop_slots=should_drop_slots,
            )
            expert_output = self._run_with_intermediate_capture(
                self._run_grouped_experts,
                permuted_tokens,
                counts,
                sorted_positions,
                num_tokens=int(num_tokens),
                top_k=int(top_k),
                route_token_indices=torch.div(
                    sorted_positions, top_k, rounding_mode="floor"
                )
                + int(token_offset),
            )
            if (
                self.training
                and self._mirai_expert_output_observer is not None
                and bool(self._mirai_expert_output_observer.is_enabled)
            ):
                self._mirai_expert_output_observer.capture_sorted(
                    expert_output,
                    sorted_positions,
                    num_tokens=int(num_tokens),
                    top_k=int(top_k),
                )
            return self._restore_tokens(
                expert_output,
                sorted_positions,
                sorted_scores,
                num_tokens,
                top_k,
            )
        if backend in {"sglang_triton", "triton", "sglang"}:
            if (
                self.training
                and self._mirai_expert_intermediate_observer is not None
                and bool(self._mirai_expert_intermediate_observer.is_enabled)
            ):
                raise RuntimeError(
                    "SwiGLU specialization requires a dispatch backend that "
                    "exposes expert intermediates."
                )
            if (
                self.training
                and self._mirai_expert_output_observer is not None
                and bool(self._mirai_expert_output_observer.is_enabled)
            ):
                raise RuntimeError(
                    "Expert-output orthogonality requires a dispatch backend that "
                    "exposes sorted expert outputs."
                )
            return self._run_sglang_triton_experts(tokens, top_scores, top_indices)
        if backend in {"sglang_triton_fp8", "triton_fp8", "sglang_fp8"}:
            if (
                self.training
                and self._mirai_expert_intermediate_observer is not None
                and bool(self._mirai_expert_intermediate_observer.is_enabled)
            ):
                raise RuntimeError(
                    "SwiGLU specialization requires a dispatch backend that "
                    "exposes expert intermediates."
                )
            if (
                self.training
                and self._mirai_expert_output_observer is not None
                and bool(self._mirai_expert_output_observer.is_enabled)
            ):
                raise RuntimeError(
                    "Expert-output orthogonality requires a dispatch backend that "
                    "exposes sorted expert outputs."
                )
            return self._run_sglang_triton_fp8_experts(tokens, top_scores, top_indices)
        raise ValueError(
            f"Unsupported model.params.moe_expert_backend={backend!r}; "
            "expected grouped_mm, loop, sglang_triton, or sglang_triton_fp8"
        )

    def _restore_tokens(
        self,
        expert_output: torch.Tensor,
        sorted_positions: torch.Tensor,
        sorted_scores: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        backend = _moe_restore_backend(self)
        if backend in {"triton", "triton_fused", "fused"}:
            return LingBotVideoSparseMoeBlock._restore_tokens_triton(
                expert_output,
                sorted_positions,
                sorted_scores,
                num_tokens,
                top_k,
            )
        if backend in {"index_add", "index_add_", "scatter_add"}:
            return LingBotVideoSparseMoeBlock._restore_tokens_index_add(
                expert_output,
                sorted_positions,
                sorted_scores,
                num_tokens,
                top_k,
            )
        if backend in {"weighted_scatter", "weighted", "fast_scatter"}:
            return LingBotVideoSparseMoeBlock._restore_tokens_weighted_scatter(
                expert_output,
                sorted_positions,
                sorted_scores,
                num_tokens,
                top_k,
            )
        if backend in {"chunked", "chunked_scatter", "scatter_chunked"}:
            return self._restore_tokens_chunked_scatter(
                expert_output,
                sorted_positions,
                sorted_scores,
                num_tokens,
                top_k,
            )
        if backend not in {"scatter", "default"}:
            raise ValueError(
                f"Unsupported model.params.moe_restore_backend={backend!r}; "
                "expected scatter, chunked_scatter, weighted_scatter, index_add, or triton"
            )
        dim = expert_output.shape[-1]
        unsorted = torch.zeros(
            (num_tokens * top_k, dim),
            dtype=expert_output.dtype,
            device=expert_output.device,
        )
        unsorted[sorted_positions] = expert_output
        unsorted = unsorted.reshape(num_tokens, top_k, dim)

        scores_unsorted = torch.zeros(
            num_tokens * top_k,
            dtype=sorted_scores.dtype,
            device=sorted_scores.device,
        )
        scores_unsorted[sorted_positions] = sorted_scores
        scores_unsorted = scores_unsorted.reshape(num_tokens, top_k, 1)
        return (unsorted.float() * scores_unsorted).sum(dim=1).to(expert_output.dtype)

    def _restore_tokens_chunked_scatter(
        self,
        expert_output: torch.Tensor,
        sorted_positions: torch.Tensor,
        sorted_scores: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        dim = expert_output.shape[-1]
        chunk_size = int(_runtime_options(self).moe_restore_chunk_size)
        if chunk_size <= 0:
            raise ValueError("model.params.moe_restore_chunk_size must be positive")

        scores_unsorted = torch.zeros(
            num_tokens * top_k,
            dtype=sorted_scores.dtype,
            device=sorted_scores.device,
        )
        scores_unsorted[sorted_positions] = sorted_scores
        scores_unsorted = scores_unsorted.reshape(num_tokens, top_k, 1)
        output = expert_output.new_empty((num_tokens, dim))
        for start in range(0, dim, chunk_size):
            end = min(start + chunk_size, dim)
            unsorted = torch.zeros(
                (num_tokens * top_k, end - start),
                dtype=expert_output.dtype,
                device=expert_output.device,
            )
            unsorted[sorted_positions] = expert_output[:, start:end]
            unsorted = unsorted.reshape(num_tokens, top_k, end - start)
            output[:, start:end] = (unsorted.float() * scores_unsorted).sum(dim=1).to(
                expert_output.dtype
            )
        return output

    @staticmethod
    def _restore_tokens_triton(
        expert_output: torch.Tensor,
        sorted_positions: torch.Tensor,
        sorted_scores: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        return restore_tokens_triton(
            expert_output,
            sorted_positions,
            sorted_scores,
            num_tokens,
            top_k,
        )

    @staticmethod
    def _restore_tokens_weighted_scatter(
        expert_output: torch.Tensor,
        sorted_positions: torch.Tensor,
        sorted_scores: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        weighted = (expert_output * sorted_scores[:, None].to(expert_output.dtype)).to(expert_output.dtype)
        unsorted = torch.zeros(
            (num_tokens * top_k, expert_output.shape[-1]),
            dtype=expert_output.dtype,
            device=expert_output.device,
        )
        unsorted[sorted_positions] = weighted
        return unsorted.reshape(num_tokens, top_k, expert_output.shape[-1]).sum(dim=1)

    @staticmethod
    def _restore_tokens_index_add(
        expert_output: torch.Tensor,
        sorted_positions: torch.Tensor,
        sorted_scores: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        token_indices = torch.div(sorted_positions, top_k, rounding_mode="floor")
        weighted = expert_output.float() * sorted_scores[:, None].float()
        out = torch.zeros(
            (num_tokens, expert_output.shape[-1]),
            dtype=torch.float32,
            device=expert_output.device,
        )
        out.index_add_(0, token_indices, weighted)
        return out.to(expert_output.dtype)

    @staticmethod
    def _chain_training_snapshot(router):
        return {
            name: getattr(router, name, None)
            for name in (
                "training_top_indices",
                "training_unbiased_top_indices",
                "training_logits",
                "training_scores",
                "training_gradient_probabilities",
                "training_subset_kl",
                "training_router_distillation",
                "training_lightweight_balance_loss",
                "training_lightweight_z_loss",
            )
        }

    def _merge_chain_training_snapshots(
        self,
        first,
        second,
        *,
        tokens_per_sample: int,
    ) -> None:
        if not self.training:
            return
        tensor_fields = (
            "training_top_indices",
            "training_unbiased_top_indices",
            "training_logits",
            "training_scores",
            "training_gradient_probabilities",
        )
        for name in tensor_fields:
            left = first[name]
            right = second[name]
            if left is None and right is None:
                setattr(self.router, name, None)
                continue
            if left is None or right is None:
                raise RuntimeError(
                    f"CoE router training state {name!r} is present for only one step."
                )
            setattr(self.router, name, torch.cat((left, right), dim=0))
        scalar_fields = (
            "training_subset_kl",
            "training_router_distillation",
            "training_lightweight_balance_loss",
            "training_lightweight_z_loss",
        )
        for name in scalar_fields:
            left = first[name]
            right = second[name]
            if left is None and right is None:
                setattr(self.router, name, None)
            elif left is None or right is None:
                raise RuntimeError(
                    f"CoE router auxiliary {name!r} is present for only one step."
                )
            else:
                setattr(self.router, name, (left + right) * 0.5)
        self.router.training_aux_tokens_per_sample = int(tokens_per_sample) * 2

    def _forward_once(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        router_input: Optional[torch.Tensor] = None,
        timestep_router_input: Optional[torch.Tensor] = None,
        router_logit_delta: Optional[torch.Tensor] = None,
        route_scope_mask: Optional[torch.Tensor] = None,
        saliency_router_input: Optional[torch.Tensor] = None,
    ):
        # hidden_states: (B, S, H); padding_mask: (B*S,) with 1=valid (only needed when B>1)
        B = hidden_states.shape[0]
        tokens = hidden_states.view(-1, self.hidden_size)
        self.router.training_batch_size = int(B)
        self.router.training_tokens_per_sample = int(hidden_states.shape[1])
        self.router.training_aux_tokens_per_sample = None
        route_shape_setter = getattr(
            self.experts,
            "set_routed_adapter_tokens_per_sample",
            None,
        )
        if callable(route_shape_setter):
            route_shape_setter(int(hidden_states.shape[1]))
        saliency_extension = getattr(self.router, "saliency_harnessing", None)
        if saliency_extension is not None:
            if saliency_router_input is None:
                raise RuntimeError(
                    "SharpMoE routing requires saliency guidance tokens."
                )
            if tuple(saliency_router_input.shape) != tuple(hidden_states.shape):
                raise ValueError(
                    "SharpMoE saliency tokens must match the MoE hidden-state shape."
                )
            saliency_delta = saliency_extension(
                saliency_router_input.reshape(-1, self.hidden_size),
                route_scope_mask=route_scope_mask,
            )
            router_logit_delta = (
                saliency_delta
                if router_logit_delta is None
                else router_logit_delta + saliency_delta
            )
        if self.router._expert_choice_extension is None:
            top_indices, top_scores, logits, scores, scores_for_choice = self.router(
                tokens,
                logit_delta=router_logit_delta,
                valid_token_mask=padding_mask,
                route_scope_mask=route_scope_mask,
            )
            del logits, scores, scores_for_choice
        else:
            decision = self.router.forward_expert_choice(
                hidden_states,
                router_input,
                timestep_router_input,
            )
            top_indices, top_scores = self._expert_choice_token_routes(
                decision,
                tokens_per_sample=int(hidden_states.shape[1]),
                num_experts=int(self.router.num_experts),
            )
            self.router.last_top_indices = top_indices.detach()
            self.router.last_top_scores = top_scores.detach()
            self.router.last_route_active_mask = (top_scores != 0).detach()
            if self.training:
                self.router.training_top_indices = top_indices
                self.router.training_unbiased_top_indices = top_indices
        if padding_mask is not None:
            pm = padding_mask.unsqueeze(-1).to(top_scores.dtype)
            top_scores = top_scores * pm
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-9)
            top_scores = top_scores * self.router.route_scale

        drop_slots = bool(self.router._expert_choice_extension is not None)
        chunk_policy = self._mirai_token_chunk_policy
        if chunk_policy is not None and bool(getattr(chunk_policy, "enabled", False)):
            observer = self._mirai_expert_output_observer
            intermediate_observer = self._mirai_expert_intermediate_observer
            if (
                self.training
                and (
                    (observer is not None and bool(observer.is_enabled))
                    or (
                        intermediate_observer is not None
                        and bool(intermediate_observer.is_enabled)
                    )
                )
            ):
                raise RuntimeError(
                    "MoE token chunk checkpointing is incompatible with active "
                    "expert-output or expert-intermediate observers."
                )
            extension_getter = getattr(self.experts, "linear_extension", None)
            extension = (
                extension_getter() if callable(extension_getter) else None
            )
            routed_gate_getter = getattr(
                self.experts, "routed_adapter_gate", None
            )
            routed_gate = (
                routed_gate_getter()
                if callable(routed_gate_getter)
                else getattr(extension, "_routed_adapter_gate", None)
            )
            if routed_gate is not None:
                raise RuntimeError(
                    "MoE token chunk checkpointing does not support a "
                    "sample-selective routed adapter gate."
                )

            def run_chunk(
                input_tokens,
                input_scores,
                input_indices,
                token_offset,
            ):
                return self._run_selected_experts(
                    input_tokens,
                    input_scores,
                    input_indices,
                    drop_slots=drop_slots,
                    token_offset=int(token_offset),
                )

            out = chunk_policy.execute(
                tokens,
                top_scores,
                top_indices,
                runner=run_chunk,
                training=bool(self.training),
            )
        else:
            out = self._run_selected_experts(
                tokens,
                top_scores,
                top_indices,
                drop_slots=drop_slots,
            )
        adjugate = (
            None
            if self._mirai_adjugate_expert_extension is None
            else self._mirai_adjugate_expert_extension[0]
        )
        if adjugate is not None:
            out = out + adjugate.output_contribution(
                tokens,
                top_indices,
                top_scores,
            ).to(dtype=out.dtype)
        lightweight = self.router._lightweight_expert_extension
        if lightweight is not None:
            decision = self.router.runtime_lightweight_decision
            out = out + lightweight.output_contribution(
                tokens,
                decision.logical_indices,
                decision.logical_output_scores,
            ).to(dtype=out.dtype)

        out = out.view(B, -1, self.hidden_size)
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            out = out + shared_output
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        router_input: Optional[torch.Tensor] = None,
        timestep_router_input: Optional[torch.Tensor] = None,
        route_scope_mask: Optional[torch.Tensor] = None,
        saliency_router_input: Optional[torch.Tensor] = None,
    ):
        extension = self.chain_of_experts
        if extension is None:
            return self._forward_once(
                hidden_states,
                padding_mask,
                router_input,
                timestep_router_input,
                route_scope_mask=route_scope_mask,
                saliency_router_input=saliency_router_input,
            )
        if self.router._expert_choice_extension is not None:
            raise RuntimeError(
                "Chain-of-Experts requires token-choice routing."
            )
        first_output = self._forward_once(
            hidden_states,
            padding_mask,
            router_input,
            timestep_router_input,
            route_scope_mask=route_scope_mask,
            saliency_router_input=saliency_router_input,
        )
        first_training = self._chain_training_snapshot(self.router)
        first_indices = self.router.last_top_indices
        continuation_input = extension.continuation_input(
            hidden_states,
            first_output,
        )
        logit_delta = extension.router_logit_delta(
            continuation_input.reshape(-1, self.hidden_size)
        )
        continuation_output = self._forward_once(
            continuation_input,
            padding_mask,
            router_input,
            timestep_router_input,
            router_logit_delta=logit_delta,
            route_scope_mask=route_scope_mask,
            saliency_router_input=saliency_router_input,
        )
        second_training = self._chain_training_snapshot(self.router)
        second_indices = self.router.last_top_indices
        extension.record_routes(first_indices, second_indices)
        self._merge_chain_training_snapshots(
            first_training,
            second_training,
            tokens_per_sample=int(hidden_states.shape[1]),
        )
        return extension.combine(first_output, continuation_output)


class LingBotVideoBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        norm_eps,
        qkv_bias,
        out_bias,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        decoder_sparse_step,
        mlp_only_layers,
        n_shared_experts,
        score_func,
        norm_topk_prob,
        n_group,
        topk_group,
        routed_scaling_factor,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        h = hidden_size
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6 * h))
        self.norm1 = LingBotVideoRMSNorm(h, norm_eps)
        self.attn = LingBotVideoAttention(
            h, num_attention_heads, norm_eps, qkv_bias, out_bias
        )
        self.norm_post_attn = LingBotVideoRMSNorm(h, norm_eps)
        self.norm2 = LingBotVideoRMSNorm(h, norm_eps)
        # Sparsity decision matches MoEBlock: mlp_only_layers + decoder_sparse_step + num_experts
        if layer_idx not in mlp_only_layers and (
            num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0
        ):
            self.ffn = LingBotVideoSparseMoeBlock(
                h, intermediate_size, num_experts, num_experts_per_tok,
                moe_intermediate_size, score_func, norm_topk_prob,
                n_group, topk_group, routed_scaling_factor,
                n_shared_experts,
            )
        else:
            self.ffn = LingBotVideoMLP(h, intermediate_size)
        self.norm_post_ffn = LingBotVideoRMSNorm(h, norm_eps)

    def forward(
        self,
        x,
        temb6,
        rotary_emb,
        attention_mask=None,
        moe_padding_mask=None,
        packed_indices: Optional[dict[str, torch.Tensor]] = None,
        parallel_config=None,
        router_timestep_input=None,
        moe_route_scope_mask=None,
        moe_saliency_states=None,
        mixture_of_depths_scores=None,
        mixture_of_depths_cu_seqlens=None,
        mixture_of_depths_valid_mask=None,
    ):
        if mixture_of_depths_scores is not None:
            if parallel_config is not None:
                raise ValueError(
                    "Mixture-of-Depths does not support context-parallel execution."
                )
            if moe_route_scope_mask is None or mixture_of_depths_valid_mask is None:
                raise RuntimeError(
                    "Mixture-of-Depths requires provider-owned eligible and valid "
                    "token masks."
                )
            if mixture_of_depths_cu_seqlens is None:
                raise RuntimeError(
                    "Mixture-of-Depths requires per-sample cumulative lengths."
                )
            capacity_fraction = float(
                getattr(self, "_mirai_mixture_of_depths_capacity_fraction", 0.0)
            )
            selection = select_depth_tokens(
                mixture_of_depths_scores,
                eligible_mask=moe_route_scope_mask,
                valid_mask=mixture_of_depths_valid_mask,
                cu_seqlens=mixture_of_depths_cu_seqlens,
                capacity_fraction=capacity_fraction,
            )
            indices = selection.flat_indices
            flat_x = x.reshape(1, -1, x.shape[-1])

            def gather_tokens(value, *, allow_broadcast=False):
                if value is None:
                    return None
                if allow_broadcast and int(value.shape[1]) == 1:
                    return value
                flat = value.reshape(1, -1, *value.shape[2:])
                if int(flat.shape[1]) != int(flat_x.shape[1]):
                    raise ValueError(
                        "Mixture-of-Depths token-aligned inputs must match the "
                        "joint sequence length."
                    )
                return flat.index_select(1, indices)

            selected_x = flat_x.index_select(1, indices)
            selected_temb6 = gather_tokens(temb6, allow_broadcast=True)
            selected_rotary = gather_tokens(rotary_emb)
            selected_router_timestep = gather_tokens(
                router_timestep_input,
                allow_broadcast=True,
            )
            selected_scope = gather_tokens(moe_route_scope_mask)
            selected_saliency = gather_tokens(moe_saliency_states)
            selected_packed_indices = None
            if int(selection.cu_seqlens.numel()) > 2:
                lengths = selection.cu_seqlens[1:] - selection.cu_seqlens[:-1]
                selected_packed_indices = {
                    "cu_seqlens_kv": selection.cu_seqlens,
                    "max_seqlen_in_batch_kv": int(lengths.max().item()),
                }
            updated = self.forward(
                selected_x,
                selected_temb6,
                selected_rotary,
                attention_mask=None,
                moe_padding_mask=None,
                packed_indices=selected_packed_indices,
                parallel_config=None,
                router_timestep_input=selected_router_timestep,
                moe_route_scope_mask=selected_scope,
                moe_saliency_states=selected_saliency,
            )
            self._mirai_last_depth_selection = selection
            return flat_x.index_copy(1, indices, updated).reshape_as(x)

        expected_tokens = x.shape[0] * x.shape[1]
        if temb6.ndim == 2 and temb6.shape[0] == expected_tokens:
            token_modulation = temb6.view(x.shape[0], x.shape[1], -1)
        elif (
            temb6.ndim == 3
            and temb6.shape[0] == x.shape[0]
            and (temb6.shape[1] == 1 or temb6.shape[1] == x.shape[1])
        ):
            token_modulation = temb6
        else:
            raise ValueError(
                "LingBotVideoBlock expects temb6 shaped (B*S,6D), (B,S,6D), "
                f"or broadcastable (B,1,6D); got {tuple(temb6.shape)} for "
                f"hidden states {tuple(x.shape)}."
            )
        # Dense and MoE AdaLN modulation keeps scale_shift_table in FP32.
        mod = token_modulation + self.scale_shift_table.unsqueeze(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        # AdaLN modulation and normalization run in FP32; cast to the bulk
        # compute dtype only at the BF16 linear boundary.
        bulk_dtype = getattr(self, "_mirai_compute_dtype", self.attn.to_q.weight.dtype)
        attn_in = (self.norm1(x) * scale_msa + shift_msa).to(bulk_dtype)
        attn_out = self.attn(
            attn_in,
            rotary_emb,
            attention_mask,
            packed_indices=packed_indices,
            parallel_config=parallel_config,
        )
        x = x + (gate_msa * self.norm_post_attn(attn_out)).to(x.dtype)

        ffn_router_input = self.norm2(x)
        ffn_in = (ffn_router_input * scale_mlp + shift_mlp).to(bulk_dtype)
        if isinstance(self.ffn, LingBotVideoSparseMoeBlock):
            ffn_out = self.ffn(
                ffn_in,
                padding_mask=moe_padding_mask,
                router_input=ffn_router_input.to(bulk_dtype),
                timestep_router_input=router_timestep_input,
                route_scope_mask=moe_route_scope_mask,
                saliency_router_input=moe_saliency_states,
            )
        else:
            ffn_out = self.ffn(ffn_in)
        ffn_normed = self.norm_post_ffn(ffn_out)
        x = x + (gate_mlp * ffn_normed).to(x.dtype)
        return x


def _module_has_trainable_params(module: "nn.Module") -> bool:
    """True when any parameter under ``module`` requires grad.

    A fully-frozen block gains nothing from gradient-checkpoint recompute: the
    recompute reproduces activations solely to backprop into parameters that do
    not exist. Skipping the checkpoint wrapper for such a block is numerically
    transparent (checkpointing is a memory/compute trade, not a value change),
    so it never perturbs the loss. With per-block LoRA every block carries
    adapter params, so this fires on zero blocks in the standard LingBot path;
    it is a guard for partial-adapter configs (attention-only, MoE-Sieve
    subsets) where whole blocks can be genuinely frozen.
    """
    for parameter in module.parameters():
        if parameter.requires_grad:
            return True
    return False


def _block_router_auxiliary_terms(
    block: LingBotVideoBlock,
    *,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ffn = block.ffn
    if not isinstance(ffn, LingBotVideoSparseMoeBlock):
        zero = like.float().sum() * 0.0
        return zero, zero
    router = ffn.router
    expert_choice = getattr(router, "training_expert_choice_decision", None)
    if expert_choice is not None:
        return (
            expert_choice.load_balance_loss.float(),
            expert_choice.z_loss.float(),
        )
    lightweight_balance = getattr(
        router, "training_lightweight_balance_loss", None
    )
    lightweight_z = getattr(router, "training_lightweight_z_loss", None)
    if lightweight_balance is not None and lightweight_z is not None:
        return lightweight_balance.float(), lightweight_z.float()
    indices = router.training_top_indices
    probabilities = router.training_scores
    logits = router.training_logits
    if indices is None or probabilities is None or logits is None:
        zero = like.float().sum() * 0.0
        return zero, zero
    experts = int(router.num_experts)
    normalized = probabilities.float()
    normalized = normalized / normalized.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-20)
    mode = str(
        getattr(block, "_mirai_moe_aux_loss_type", "sequence")
    ).strip().lower()
    injected_fraction = getattr(
        router, "_mirai_global_batch_load_fraction", None
    )
    if mode == "sequence" and injected_fraction is not None:
        expert_fraction = injected_fraction.detach().to(
            device=normalized.device,
            dtype=normalized.dtype,
        )
        balance = experts * torch.sum(
            expert_fraction * normalized.mean(dim=0)
        )
    elif mode == "sequence":
        batch_size = int(getattr(router, "training_batch_size", 0))
        tokens_per_sample = int(
            getattr(router, "training_aux_tokens_per_sample", None)
            or getattr(router, "training_tokens_per_sample", 0)
        )
        unbiased = getattr(router, "training_unbiased_top_indices", None)
        if (
            unbiased is None
            or batch_size <= 0
            or tokens_per_sample <= 0
            or batch_size * tokens_per_sample != int(probabilities.shape[0])
        ):
            raise RuntimeError(
                "Sequence-wise MoE auxiliary loss requires router batch and "
                "per-sequence token metadata."
            )
        top_k = int(unbiased.shape[-1])
        probability_mean = normalized.reshape(
            batch_size, tokens_per_sample, experts
        ).mean(dim=1)
        counts = normalized.new_zeros((batch_size, experts))
        counts.scatter_add_(
            1,
            unbiased.reshape(batch_size, tokens_per_sample * top_k),
            normalized.new_ones((batch_size, tokens_per_sample * top_k)),
        )
        frequency = (
            counts * (float(experts) / float(top_k * tokens_per_sample))
        ).detach()
        balance = (frequency * probability_mean).sum(dim=-1).mean()
    elif mode == "global":
        # Global-batch load balancing (mirai): when the runtime has accumulated
        # the token-fraction load across the gradient-accumulation window it sets
        # `_mirai_global_batch_load_fraction` on the router; use it as the
        # detached load `f_i`. Absent (default microbatch scope) -> per-micro-batch
        # bincount. The differentiable mean-probability factor stays local either
        # way, so at accumulation == 1 the accumulated fraction equals the local
        # bincount and returns the same counts.
        if injected_fraction is not None:
            expert_fraction = injected_fraction.detach().to(
                device=normalized.device, dtype=normalized.dtype
            )
        else:
            selected = indices.reshape(-1)
            expert_fraction = torch.bincount(
                selected,
                minlength=experts,
            ).float() / float(max(1, selected.numel()))
        balance = experts * torch.sum(expert_fraction * normalized.mean(dim=0))
    elif mode == "disabled":
        balance = normalized.sum() * 0.0
    else:
        raise ValueError(f"Unsupported LingBot MoE auxiliary loss type '{mode}'.")
    z_loss = torch.logsumexp(logits.float(), dim=-1).square().mean()
    return balance, z_loss


class LingBotVideoTransformer3DModel(nn.Module):
    _supports_gradient_checkpointing = False
    _no_split_modules = ["LingBotVideoBlock"]
    _keep_in_fp32_modules = list(LINGBOT_VIDEO_FP32_MODULES)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None or dtype == torch.float32:
            return super().to(*args, **kwargs)

        dtype_is_floating = torch.is_floating_point(torch.empty((), dtype=dtype))
        if not dtype_is_floating:
            return super().to(*args, **kwargs)

        if device is not None:
            super().to(device=device, non_blocking=non_blocking)

        for name, param in self.named_parameters():
            if not torch.is_floating_point(param):
                continue
            target_dtype = torch.float32 if should_keep_in_fp32(name) else dtype
            param.data = param.data.to(dtype=target_dtype, non_blocking=non_blocking)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=target_dtype, non_blocking=non_blocking)

        for name, buffer in self.named_buffers():
            if not torch.is_floating_point(buffer):
                continue
            target_dtype = torch.float32 if should_keep_in_fp32(name) else dtype
            buffer.data = buffer.data.to(dtype=target_dtype, non_blocking=non_blocking)

        return self

    @capture_init_config
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        depth: int = 24,
        intermediate_size: int = 6144,
        text_dim: int = 2560,
        freq_dim: int = 256,
        norm_eps: float = 1e-6,
        rope_theta: float = 256.0,
        axes_dims: Tuple[int, int, int] = (32, 48, 48),
        axes_lens: Tuple[int, int, int] = (8192, 1024, 1024),
        qkv_bias: bool = False,
        out_bias: bool = True,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        num_experts: int = 0,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 512,
        decoder_sparse_step: int = 1,
        mlp_only_layers: Tuple[int, ...] = (),
        n_shared_experts: Optional[int] = None,
        score_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        head_dim = hidden_size // num_attention_heads
        assert head_dim == sum(axes_dims), f"head_dim {head_dim} != sum(axes_dims) {sum(axes_dims)}"
        mlp_only_layers = tuple(mlp_only_layers)

        self.patch_embedder = nn.Linear(
            in_channels * math.prod(patch_size), hidden_size, bias=patch_embed_bias
        )
        self.time_proj = Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            freq_dim, hidden_size, act_fn="silu", sample_proj_bias=timestep_mlp_bias
        )
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.text_embedder = LingBotVideoTextEmbedder(text_dim, hidden_size)
        self.rope = LingBotVideoRotaryEmbedding(axes_dims, axes_lens, rope_theta)
        self.blocks = nn.ModuleList(
            [
                LingBotVideoBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    norm_eps=norm_eps,
                    qkv_bias=qkv_bias,
                    out_bias=out_bias,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    moe_intermediate_size=moe_intermediate_size,
                    decoder_sparse_step=decoder_sparse_step,
                    mlp_only_layers=mlp_only_layers,
                    n_shared_experts=n_shared_experts,
                    score_func=score_func,
                    norm_topk_prob=norm_topk_prob,
                    n_group=n_group,
                    topk_group=topk_group,
                    routed_scaling_factor=routed_scaling_factor,
                    layer_idx=i,
                )
                for i in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.norm_out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj_out = nn.Linear(hidden_size, math.prod(patch_size) * out_channels)
        self.cp_joint = nn.Identity()
        self.cp_rotary = nn.Identity()
        self.cp_temb_input = nn.Identity()
        self.cp_temb6 = nn.Identity()
        self.cp_out = nn.Identity()
        self._mirai_gradient_checkpointing = "off"
        self._mirai_dispersive_loss_runtime = None
        self._mirai_dispersive_loss_terms = ()
        self._mirai_prototypical_routing_enabled = False
        self._mirai_sharp_moe_enabled = False
        self._mirai_mixture_of_depths_spec = None
        self._mirai_mixture_of_depths_metrics = ()
        self._mirai_checkpoint_prototypical_routing_terms = ()
        self._cp_plan = {
            "cp_joint": {
                "input": ContextParallelInput(split_dim=1, expected_dims=3),
            },
            "cp_rotary": {
                "input": ContextParallelInput(split_dim=1, expected_dims=3),
            },
            "cp_temb_input": {
                "input": ContextParallelInput(split_dim=1, expected_dims=3),
            },
            "cp_temb6": {
                "input": ContextParallelInput(split_dim=1, expected_dims=3),
            },
            "cp_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        # Host-side text-length cache (B5): the per-forward
        # ``encoder_attention_mask.sum(...).cpu().tolist()`` D2H sync is constant
        # across every denoise forward of a fixed prompt. Cache the resolved
        # lengths keyed on the mask tensor's identity so the sync happens once per
        # unique mask (cond and uncond alternate -> both entries stay warm).
        self._text_lens_cache: dict[int, tuple[torch.Tensor, list[int]]] = {}

    def set_dispersive_loss_runtime(self, runtime):
        """Attach a provider-owned video-representation regularizer."""

        if runtime is not None:
            for method_name in (
                "begin_forward",
                "is_layer_enabled",
                "loss_for_hidden_states",
            ):
                if not callable(getattr(runtime, method_name, None)):
                    raise TypeError(
                        "Dispersive Loss runtime must expose "
                        f"{method_name}()."
                    )
        self._mirai_dispersive_loss_runtime = runtime

    def _resolve_text_lens_list(
        self,
        text_lens: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        seq_len: int,
        batch: int,
    ) -> list[int]:
        """Host-side text lengths without a per-forward device->host sync.

        * No mask -> every sample uses the full text length ``seq_len`` (a Python
          int already), so the list is materialized with zero device reads.
        * With a mask -> the lengths are the mask row-sums, constant for a fixed
          prompt across the whole denoise loop. Cache them keyed on the mask's
          identity (the reference is held in the cache value, so the id cannot be
          reused by a different live tensor). cond/uncond masks differ in length
          and get separate entries -> no cross-contamination. Values are identical
          to ``[int(v) for v in text_lens.detach().cpu().tolist()]``; only the
          repeated D2H is removed.
        """
        if encoder_attention_mask is None:
            return [seq_len] * batch
        key = id(encoder_attention_mask)
        cached = self._text_lens_cache.get(key)
        if cached is not None and cached[0] is encoder_attention_mask:
            return cached[1]
        lens_list = [int(v) for v in text_lens.detach().cpu().tolist()]
        if len(self._text_lens_cache) >= 16:
            self._text_lens_cache.clear()
        self._text_lens_cache[key] = (encoder_attention_mask, lens_list)
        return lens_list

    def forward(
        self,
        hidden_states: torch.Tensor,             # (B, C, T, H, W)
        timestep: torch.Tensor,                  # (B,) ∈ [0, 1000](= sigma*1000)
        encoder_hidden_states: torch.Tensor,     # (B, L, text_dim)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # (B, L) 1=valid
        routing_guidance_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        B, C, T, H, W = hidden_states.shape
        pF, pH, pW = self.config.patch_size
        gt, gh, gw = T // pF, H // pH, W // pW
        n_video = gt * gh * gw
        L = encoder_hidden_states.shape[1]
        device = hidden_states.device
        if encoder_attention_mask is not None:
            text_lens = encoder_attention_mask.sum(dim=-1).long()
        else:
            text_lens = torch.full((B,), L, dtype=torch.long, device=device)
        text_lens_list = self._resolve_text_lens_list(
            text_lens, encoder_attention_mask, L, B
        )
        packed_batch = B > 1
        dispersive_runtime = (
            self._mirai_dispersive_loss_runtime if self.training else None
        )
        if dispersive_runtime is not None:
            dispersive_runtime.begin_forward(
                batch_size=int(B),
                video_tokens=int(n_video),
                text_lengths=text_lens_list,
                packed_batch=bool(packed_batch),
            )

        # patchify: token order (f h w), feature order (pf ph pw c) -- matches patchify_and_embed
        patch_tokens = hidden_states.reshape(B, C, gt, pF, gh, pH, gw, pW)
        patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
            B,
            n_video,
            pF * pH * pW * C,
        )
        if packed_batch:
            packed_patch_tokens = patch_tokens.reshape(1, B * n_video, -1)
            x = torch.cat(
                [self.patch_embedder(patch_tokens[i : i + 1]) for i in range(B)],
                dim=1,
            )
        else:
            x = self.patch_embedder(patch_tokens)

        if packed_batch:
            text_parts = [
                self.text_embedder(encoder_hidden_states[i : i + 1, : text_lens_list[i], :])
                for i in range(B)
            ]
            text = torch.cat(text_parts, dim=1)
            joint = _cat_interleave(
                x,
                [n_video] * B,
                text,
                text_lens_list,
            )
        else:
            text = self.text_embedder(encoder_hidden_states)
            joint = torch.cat([x, text], dim=1)  # [video; text]
        moe_saliency_states = None
        if routing_guidance_states is not None:
            if not bool(self._mirai_sharp_moe_enabled):
                raise RuntimeError(
                    "Routing guidance was supplied without an enabled SharpMoE policy."
                )
            if tuple(routing_guidance_states.shape) != tuple(hidden_states.shape):
                raise ValueError(
                    "SharpMoE routing guidance must match hidden_states shape."
                )
            guidance_tokens = routing_guidance_states.reshape(
                B, C, gt, pF, gh, pH, gw, pW
            )
            guidance_tokens = guidance_tokens.permute(
                0, 2, 4, 6, 3, 5, 7, 1
            ).reshape(B, n_video, pF * pH * pW * C)
            if packed_batch:
                guidance_video = torch.cat(
                    [
                        self.patch_embedder(guidance_tokens[i : i + 1])
                        for i in range(B)
                    ],
                    dim=1,
                )
                guidance_text = torch.zeros_like(text)
                moe_saliency_states = _cat_interleave(
                    guidance_video,
                    [n_video] * B,
                    guidance_text,
                    text_lens_list,
                )
            else:
                guidance_video = self.patch_embedder(guidance_tokens)
                guidance_text = torch.zeros_like(text)
                moe_saliency_states = torch.cat(
                    [guidance_video, guidance_text], dim=1
                )
        joint_seq_len = joint.shape[1]
        moe_route_scope_mask = None
        if bool(
            self._mirai_prototypical_routing_enabled
            or self._mirai_sharp_moe_enabled
            or self._mirai_mixture_of_depths_spec is not None
        ):
            if packed_batch:
                scope_parts = [
                    torch.cat(
                        (
                            torch.ones(n_video, device=device, dtype=torch.bool),
                            torch.zeros(
                                text_lens_list[i], device=device, dtype=torch.bool
                            ),
                        )
                    )
                    for i in range(B)
                ]
                moe_route_scope_mask = torch.cat(scope_parts).reshape(1, -1)
            else:
                moe_route_scope_mask = torch.zeros(
                    (B, joint_seq_len), device=device, dtype=torch.bool
                )
                moe_route_scope_mask[:, :n_video] = True

        # Per-sample RoPE: video t-axis start = real text length of this sample + 1
        rotary_parts = [
            self.rope(make_joint_position_ids(text_lens_list[i], gt, gh, gw, device))
            for i in range(B)
        ]
        if packed_batch:
            rotary = torch.cat(rotary_parts, dim=0).unsqueeze(0)
        else:
            rotary = torch.stack(rotary_parts, dim=0)  # (B, S, head_dim/2) complex64

        parallel_config = getattr(self, "_parallel_config", None)
        use_packed_attention = parallel_config is not None

        attention_mask = None
        moe_padding_mask = None
        packed_indices = None
        has_padding = encoder_attention_mask is not None and bool((text_lens < L).any())
        if packed_batch or use_packed_attention:
            sample_seq_lens = [n_video + text_len for text_len in text_lens_list]
            cu_seqlens = torch.zeros(B + 1, device=device, dtype=torch.int32)
            cu_seqlens[1:] = torch.cumsum(
                torch.tensor(sample_seq_lens, device=device, dtype=torch.int32),
                dim=0,
            )
            packed_indices = {
                "cu_seqlens_kv": cu_seqlens,
                "max_seqlen_in_batch_kv": max(sample_seq_lens),
            }
            has_padding = False
        if has_padding:
            key_mask = torch.cat(
                [torch.ones(B, n_video, dtype=torch.bool, device=device),
                 encoder_attention_mask.bool()],
                dim=1,
            )
            attention_mask = key_mask[:, None, None, :]      # (B,1,1,S) → SDPA broadcast
            moe_padding_mask = key_mask.reshape(-1).float()  # (B*S,)
        packed_cp = packed_indices is not None and parallel_config is not None
        padding_size = 0
        if packed_cp:
            cp_config = parallel_config.context_parallel_config
            cp_world_size = int(getattr(cp_config, "ulysses_degree", getattr(cp_config, "_world_size", 1)))
            padding_size = (cp_world_size - (joint_seq_len % cp_world_size)) % cp_world_size
            if padding_size:
                joint = torch.cat(
                    [
                        joint,
                        torch.zeros(
                            joint.shape[0],
                            padding_size,
                            joint.shape[2],
                            device=joint.device,
                            dtype=joint.dtype,
                        ),
                    ],
                    dim=1,
                )
                if moe_saliency_states is not None:
                    moe_saliency_states = torch.cat(
                        [
                            moe_saliency_states,
                            torch.zeros(
                                moe_saliency_states.shape[0],
                                padding_size,
                                moe_saliency_states.shape[2],
                                device=moe_saliency_states.device,
                                dtype=moe_saliency_states.dtype,
                            ),
                        ],
                        dim=1,
                    )
                if moe_route_scope_mask is not None:
                    moe_route_scope_mask = torch.cat(
                        (
                            moe_route_scope_mask,
                            torch.zeros(
                                (moe_route_scope_mask.shape[0], padding_size),
                                device=device,
                                dtype=torch.bool,
                            ),
                        ),
                        dim=1,
                    )
                rotary = torch.cat(
                    [
                        rotary,
                        torch.zeros(
                            rotary.shape[0],
                            padding_size,
                            rotary.shape[2],
                            device=rotary.device,
                            dtype=rotary.dtype,
                        ),
                    ],
                    dim=1,
                )
                if packed_indices is None:
                    raise RuntimeError("packed_indices must be initialized for packed context parallel.")
                packed_indices["cu_seqlens_kv"] = torch.cat(
                    [
                        packed_indices["cu_seqlens_kv"],
                        packed_indices["cu_seqlens_kv"][-1:] + padding_size,
                    ],
                    dim=0,
                )
                packed_indices["max_seqlen_in_batch_kv"] = max(
                    int(packed_indices["max_seqlen_in_batch_kv"]),
                    int(padding_size),
                )
                joint_seq_len = joint.shape[1]

        timestep_for_embed = timestep.float()
        timestep_proj = self.time_proj(timestep_for_embed)
        t_emb = self.time_embedder(timestep_proj)                            # (B, D)
        if packed_batch:
            temb_input = torch.cat(
                [
                    t_emb[i : i + 1].unsqueeze(1).expand(1, n_video + text_lens_list[i], -1)
                    for i in range(B)
                ],
                dim=1,
            )
            if padding_size:
                temb_input = torch.cat(
                    [
                        temb_input,
                        torch.zeros(
                            temb_input.shape[0],
                            padding_size,
                            temb_input.shape[2],
                            device=temb_input.device,
                            dtype=temb_input.dtype,
                        ),
                    ],
                    dim=1,
                )
            temb6 = self.time_modulation(temb_input.reshape(joint_seq_len, -1))
            temb6 = temb6.reshape(1, joint_seq_len, -1)
        elif packed_cp:
            temb_input = t_emb.unsqueeze(1).expand(B, joint_seq_len, -1)
            temb6 = self.time_modulation(temb_input.reshape(B * joint_seq_len, -1))
            temb6 = temb6.reshape(B, joint_seq_len, -1)
        else:
            # Every token in one sample receives the same timestep modulation.
            # Keep one FP32 value per sample and broadcast inside each block.
            temb_input = t_emb.unsqueeze(1)
            temb6 = self.time_modulation(t_emb).unsqueeze(1)

        joint = self.cp_joint(joint)
        if moe_saliency_states is not None:
            moe_saliency_states = self.cp_joint(moe_saliency_states)
        if (
            moe_route_scope_mask is not None
            and int(moe_route_scope_mask.shape[0] * moe_route_scope_mask.shape[1])
            != int(joint.shape[0] * joint.shape[1])
        ):
            raise RuntimeError(
                "Optional routing policies require the provider's visual-token mask "
                "to match the local token layout."
            )
        rotary = self.cp_rotary(rotary)
        if packed_cp:
            temb_input = self.cp_temb_input(temb_input)
        temb6 = self.cp_temb6(temb6)

        checkpoint_mode = str(getattr(self, "_mirai_gradient_checkpointing", "off")).lower()
        use_checkpoint = bool(self.training) and checkpoint_mode not in {"off", "false", "none", "0", ""}
        aggressive_checkpoint = checkpoint_mode == "aggressive"
        selective_checkpoint = checkpoint_mode == "selective"
        block_swap_manager = getattr(self, "_mirai_block_swap_manager", None)
        checkpoint_router_terms = []
        checkpoint_expert_orthogonality_terms = []
        checkpoint_expert_intermediate_terms = []
        checkpoint_router_specialization_terms = []
        checkpoint_prototypical_routing_terms = []
        dispersive_loss_terms = []
        compile_token_bucket_plan = getattr(
            self, "_mirai_compile_token_bucket_plan", None
        )
        mixture_of_depths_spec = self._mirai_mixture_of_depths_spec
        mixture_of_depths_layers = (
            frozenset(mixture_of_depths_spec.routed_layers(len(self.blocks)))
            if mixture_of_depths_spec is not None
            else frozenset()
        )
        if mixture_of_depths_spec is not None and parallel_config is not None:
            raise ValueError(
                "Mixture-of-Depths is a single-GPU policy and cannot be combined "
                "with context-parallel execution."
            )
        if packed_indices is not None:
            mixture_of_depths_cu_seqlens = packed_indices["cu_seqlens_kv"]
        else:
            mixture_of_depths_cu_seqlens = torch.tensor(
                [0, int(joint.shape[0] * joint.shape[1])],
                device=joint.device,
                dtype=torch.int32,
            )
        if moe_padding_mask is None:
            mixture_of_depths_valid_mask = torch.ones(
                (joint.shape[0], joint.shape[1]),
                device=joint.device,
                dtype=torch.bool,
            )
        else:
            mixture_of_depths_valid_mask = moe_padding_mask.reshape(
                joint.shape[0], joint.shape[1]
            ).bool()
        for block in self.blocks:
            block._mirai_last_depth_selection = None
        for block_idx, block in enumerate(self.blocks):
            mixture_of_depths_scores = None
            if block_idx in mixture_of_depths_layers:
                mixture_of_depths_scores = getattr(
                    self.blocks[block_idx - 1].attn,
                    "_mirai_last_received_attention",
                    None,
                )
                if mixture_of_depths_scores is None:
                    raise RuntimeError(
                        "Mixture-of-Depths routed block did not receive the "
                        "previous dense block's attention scores."
                    )
            if compile_token_bucket_plan is not None:
                token_count = int(joint.shape[1])
                joint = compile_token_bucket_plan.mark(joint, dim=1)
                if (
                    torch.is_tensor(rotary)
                    and int(rotary.ndim) >= 1
                    and int(rotary.shape[0]) == token_count
                ):
                    rotary = compile_token_bucket_plan.mark(
                        rotary,
                        dim=0,
                        record_hit=False,
                    )
                if (
                    torch.is_tensor(temb6)
                    and int(temb6.ndim) >= 2
                    and int(temb6.shape[1]) == token_count
                ):
                    temb6 = compile_token_bucket_plan.mark(
                        temb6,
                        dim=1,
                        record_hit=False,
                    )
                if (
                    torch.is_tensor(moe_padding_mask)
                    and int(moe_padding_mask.ndim) >= 1
                    and int(moe_padding_mask.shape[-1]) == token_count
                ):
                    moe_padding_mask = compile_token_bucket_plan.mark(
                        moe_padding_mask,
                        dim=-1,
                        record_hit=False,
                    )
                if (
                    torch.is_tensor(attention_mask)
                    and int(attention_mask.ndim) >= 1
                    and int(attention_mask.shape[-1]) == token_count
                ):
                    attention_mask = compile_token_bucket_plan.mark(
                        attention_mask,
                        dim=-1,
                        record_hit=False,
                    )
            block._mirai_moe_aux_loss_type = str(
                getattr(self, "_mirai_moe_aux_loss_type", "sequence")
            )
            if use_checkpoint:
                import torch.utils.checkpoint as checkpoint

                def run_block(
                    input_joint,
                    input_temb6,
                    input_router_timestep,
                    *,
                    current_block=block,
                    current_idx=block_idx,
                    current_rotary=rotary,
                    current_attention_mask=attention_mask,
                    current_moe_padding_mask=moe_padding_mask,
                    current_route_scope_mask=moe_route_scope_mask,
                    current_depth_scores=mixture_of_depths_scores,
                ):
                    if block_swap_manager is not None:
                        block_swap_manager.before_block(current_idx)
                    output = current_block(
                        input_joint,
                        input_temb6,
                        current_rotary,
                        current_attention_mask,
                        current_moe_padding_mask,
                        packed_indices=packed_indices,
                        parallel_config=parallel_config,
                        router_timestep_input=input_router_timestep,
                        moe_route_scope_mask=current_route_scope_mask,
                        moe_saliency_states=moe_saliency_states,
                        mixture_of_depths_scores=current_depth_scores,
                        mixture_of_depths_cu_seqlens=mixture_of_depths_cu_seqlens,
                        mixture_of_depths_valid_mask=mixture_of_depths_valid_mask,
                    )
                    observer = getattr(
                        getattr(current_block, "ffn", None),
                        "_mirai_expert_output_observer",
                        None,
                    )
                    intermediate_observer = getattr(
                        getattr(current_block, "ffn", None),
                        "_mirai_expert_intermediate_observer",
                        None,
                    )
                    orthogonality = output.new_zeros(())
                    if observer is not None:
                        terms = observer.take_losses()
                        if terms:
                            orthogonality = torch.stack(terms).mean()
                    intermediate = output.new_zeros(())
                    if intermediate_observer is not None:
                        terms = intermediate_observer.take_losses()
                        if terms:
                            intermediate = torch.stack(terms).mean()
                    if aggressive_checkpoint:
                        balance, z_loss = _block_router_auxiliary_terms(
                            current_block,
                            like=output,
                        )
                        specialization = output.new_empty((0,))
                        prototypical = output.new_empty((0,))
                        extension = getattr(
                            getattr(current_block.ffn, "router", None),
                            "prototypical_routing",
                            None,
                        )
                        if extension is not None:
                            term = getattr(extension, "last_contrastive_loss", None)
                            if term is not None:
                                prototypical = term.reshape(1)
                        runtime = getattr(
                            self, "_mirai_router_specialization_runtime", None
                        )
                        if (
                            runtime is not None
                            and isinstance(
                                current_block.ffn, LingBotVideoSparseMoeBlock
                            )
                            and float(runtime.coupling_weight) > 0.0
                        ):
                            specialization = runtime.checkpoint_topk_mass(
                                current_block.ffn.router
                            )
                        return (
                            output,
                            balance,
                            z_loss,
                            orthogonality,
                            specialization,
                            intermediate,
                            prototypical,
                        )
                    if observer is not None or intermediate_observer is not None:
                        return output, orthogonality, intermediate
                    return output

                # Frozen-module checkpoint skip: a block participates in
                # backward only when it owns trainable params or a gradient
                # flows through its inputs from an earlier trainable block.
                # Otherwise (leading fully-frozen prefix) no autograd graph is
                # needed at all, so the checkpoint save/recompute pass is pure
                # waste — run the block directly. Bit-identical output either
                # way (checkpointing is a memory/compute trade, never a value
                # change). A frozen block *behind* a trainable one still
                # checkpoints: grad w.r.t. its input needs the intermediates,
                # and recompute keeps them off the VRAM budget.
                needs_backward = (
                    bool(joint.requires_grad)
                    or bool(temb6.requires_grad)
                    or bool(temb_input.requires_grad)
                    or _module_has_trainable_params(block)
                )
                if needs_backward:
                    # Reentrant checkpointing requires a grad-requiring input;
                    # applied lazily at the first non-skipped block so a frozen
                    # prefix is not forced onto the checkpoint path.
                    if aggressive_checkpoint and not joint.requires_grad:
                        joint = joint.detach().requires_grad_(True)
                    checkpoint_kwargs = {
                        "use_reentrant": bool(aggressive_checkpoint),
                    }
                    if selective_checkpoint:
                        checkpoint_kwargs["context_fn"] = getattr(
                            self, "_mirai_selective_checkpoint_context_fn"
                        )
                    observer = getattr(
                        getattr(block, "ffn", None),
                        "_mirai_expert_output_observer",
                        None,
                    )
                    intermediate_observer = getattr(
                        getattr(block, "ffn", None),
                        "_mirai_expert_intermediate_observer",
                        None,
                    )
                    if not aggressive_checkpoint and (
                        observer is not None or intermediate_observer is not None
                    ):
                        suspension = (
                            lambda observer=observer,
                            intermediate_observer=intermediate_observer: (
                                _suspend_observers(observer, intermediate_observer)
                            )
                        )
                        if selective_checkpoint:
                            checkpoint_kwargs["context_fn"] = (
                                _extend_checkpoint_context_fn(
                                    checkpoint_kwargs["context_fn"], suspension
                                )
                            )
                        else:
                            checkpoint_kwargs["context_fn"] = (
                                lambda suspension=suspension: (
                                    nullcontext(),
                                    suspension(),
                                )
                            )
                    checkpoint_output = checkpoint.checkpoint(
                        run_block,
                        joint,
                        temb6,
                        temb_input,
                        **checkpoint_kwargs,
                    )
                else:
                    checkpoint_output = run_block(joint, temb6, temb_input)
                if aggressive_checkpoint:
                    (
                        joint,
                        balance,
                        z_loss,
                        orthogonality,
                        specialization,
                        intermediate,
                        prototypical,
                    ) = checkpoint_output
                    if isinstance(block.ffn, LingBotVideoSparseMoeBlock):
                        checkpoint_router_terms.append((balance, z_loss))
                        if block.ffn._mirai_expert_output_observer is not None:
                            checkpoint_expert_orthogonality_terms.append(orthogonality)
                        if specialization.numel() > 0:
                            checkpoint_router_specialization_terms.append(
                                specialization
                            )
                        if (
                            block.ffn._mirai_expert_intermediate_observer
                            is not None
                        ):
                            checkpoint_expert_intermediate_terms.append(
                                intermediate
                            )
                        if prototypical.numel() > 0:
                            checkpoint_prototypical_routing_terms.append(
                                prototypical.reshape(())
                            )
                else:
                    if (
                        isinstance(block.ffn, LingBotVideoSparseMoeBlock)
                        and (
                            block.ffn._mirai_expert_output_observer is not None
                            or block.ffn._mirai_expert_intermediate_observer
                            is not None
                        )
                    ):
                        joint, orthogonality, intermediate = checkpoint_output
                        if block.ffn._mirai_expert_output_observer is not None:
                            checkpoint_expert_orthogonality_terms.append(
                                orthogonality
                            )
                        if (
                            block.ffn._mirai_expert_intermediate_observer
                            is not None
                        ):
                            checkpoint_expert_intermediate_terms.append(
                                intermediate
                            )
                    else:
                        joint = checkpoint_output
                if block_swap_manager is not None:
                    block_swap_manager.after_block(block_idx)
            else:
                if block_swap_manager is not None:
                    block_swap_manager.before_block(block_idx)
                try:
                    joint = block(
                        joint,
                        temb6,
                        rotary,
                        attention_mask,
                        moe_padding_mask,
                        packed_indices=packed_indices,
                        parallel_config=parallel_config,
                        router_timestep_input=temb_input,
                        moe_route_scope_mask=moe_route_scope_mask,
                        moe_saliency_states=moe_saliency_states,
                        mixture_of_depths_scores=mixture_of_depths_scores,
                        mixture_of_depths_cu_seqlens=mixture_of_depths_cu_seqlens,
                        mixture_of_depths_valid_mask=mixture_of_depths_valid_mask,
                    )
                finally:
                    if block_swap_manager is not None:
                        block_swap_manager.after_block(block_idx)
            if (
                dispersive_runtime is not None
                and dispersive_runtime.is_layer_enabled(block_idx)
            ):
                dispersive_loss_terms.append(
                    dispersive_runtime.loss_for_hidden_states(block_idx, joint)
                )
        self._mirai_dispersive_loss_terms = tuple(dispersive_loss_terms)
        self._mirai_checkpoint_router_auxiliary_terms = tuple(checkpoint_router_terms)
        self._mirai_checkpoint_expert_orthogonality_terms = tuple(
            checkpoint_expert_orthogonality_terms
        )
        self._mirai_checkpoint_expert_intermediate_terms = tuple(
            checkpoint_expert_intermediate_terms
        )
        self._mirai_checkpoint_router_specialization_terms = tuple(
            checkpoint_router_specialization_terms
        )
        self._mirai_checkpoint_prototypical_routing_terms = tuple(
            checkpoint_prototypical_routing_terms
        )
        if not packed_cp:
            joint = self.cp_out(joint)

        if temb_input.shape[1] == 1:
            final_mod = self.norm_out_modulation(temb_input[:, 0]).unsqueeze(1)
        else:
            final_mod = self.norm_out_modulation(
                temb_input.reshape(joint.shape[0] * joint.shape[1], -1)
            ).reshape(joint.shape[0], joint.shape[1], -1)
        shift, scale = final_mod.chunk(2, dim=-1)
        final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
        output_dtype = getattr(self, "_mirai_compute_dtype", self.proj_out.weight.dtype)
        projected = self.proj_out(final_hidden.to(output_dtype))
        if packed_cp:
            projected = self.cp_out(projected)
            if padding_size:
                projected = projected[:, :-padding_size, :]
        if packed_batch:
            split_lengths: list[int] = []
            for text_len in text_lens_list:
                split_lengths.extend([n_video, text_len])
            parts = torch.split(projected, split_lengths, dim=1)
            x = torch.cat(parts[::2], dim=1).reshape(B, n_video, -1)
        else:
            x = projected[:, :n_video]

        # unpatchify (matches the rearrange in postprocess)
        Cout = self.config.out_channels
        x = x.reshape(B, gt, gh, gw, pF, pH, pW, Cout)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, H, W)

        if not return_dict:
            return (x,)
        return Transformer3DModelOutput(sample=x)
