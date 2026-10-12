# Copyright (c) 2026 SandAI. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
# Modified native single-device implementation for Mirai.
# Source: https://huggingface.co/Agnes-AI/Agnes-Video-2.0-MoE/blob/fc3971e378b53c766baa06898bfe84d7a56653ea/inference_code/inference/model/dit/dit_module.py
"""Checkpoint-compatible native Agnes DiT with isolated packed attention."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class AgnesDiTConfig:
    num_layers: int = 40
    hidden_size: int = 5120
    head_dim: int = 128
    num_query_groups: int = 8
    video_in_channels: int = 192
    audio_in_channels: int = 128
    text_in_channels: int = 3584
    mm_layers: tuple[int, ...] = (0, 1, 2, 3, 36, 37, 38, 39)
    sparse_moe_layers: tuple[int, ...] = tuple(range(4, 36))
    gelu7_layers: tuple[int, ...] = (0, 1, 2, 3)
    post_norm_layers: tuple[int, ...] = ()
    num_routed_experts: int = 3
    moe_top_k: int = 1
    moe_router_noise: float = 0.0
    moe_aux_loss_coef: float = 0.01
    token_sparse_moe_convert_mm_mlp: bool = False
    enable_attn_gating: bool = True
    params_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.hidden_size % self.head_dim or self.head_dim < 8:
            raise ValueError("hidden_size must be divisible by head_dim >= 8")
        if (self.hidden_size // self.head_dim) % self.num_query_groups:
            raise ValueError("query heads must be divisible by key/value heads")
        if not 1 <= self.moe_top_k <= self.num_routed_experts:
            raise ValueError("invalid expert top-k")


def swiglu7(x, alpha=1.702, limit=7.0):
    gate, linear = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(max=limit)
    return gate * torch.sigmoid(alpha * gate) * (linear.clamp(-limit, limit) + 1)


def gelu7(x, alpha=1.702, limit=7.0):
    x = x.clamp(max=limit)
    return x * torch.sigmoid(alpha * x)


class ModalityDispatcher:
    def __init__(self, modality_mapping):
        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        self.group_size_cpu = torch.bincount(modality_mapping, minlength=3).tolist()


class _FrozenNormAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized, weight, start, end):
        ctx.weight, ctx.rows = weight, (start, end)
        scale = weight[start:end].to(device=normalized.device) + 1
        return normalized * scale

    @staticmethod
    def backward(ctx, gradient):
        start, end = ctx.rows
        scale = ctx.weight[start:end].to(device=gradient.device) + 1
        return gradient * scale, None, None, None


class MultiModalityRMSNorm(nn.Module):
    def __init__(self, dim, num_modality=1, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim * num_modality))
        self.num_modality, self.eps = num_modality, eps

    def forward(self, x, modality_dispatcher=None):
        floating = x.float()
        normalized = floating * torch.rsqrt(floating.square().mean(-1, keepdim=True) + self.eps)
        if self.num_modality == 1:
            if not self.weight.requires_grad:
                return _FrozenNormAffine.apply(normalized, self.weight, 0, self.weight.numel()).to(x.dtype)
            return (normalized * (self.weight + 1)).to(x.dtype)
        chunks = normalized.split(modality_dispatcher.group_size_cpu, 0)
        if not self.weight.requires_grad:
            width = self.weight.numel() // self.num_modality
            return torch.cat([_FrozenNormAffine.apply(chunk, self.weight, index * width, (index + 1) * width)
                for index, chunk in enumerate(chunks)]).to(x.dtype)
        return torch.cat([chunk * (weight + 1) for chunk, weight in zip(chunks, self.weight.chunk(3))]).to(x.dtype)


class _BF16ComputeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, row_start, row_end):
        cast = x.to(torch.bfloat16)
        if ctx.needs_input_grad[1]:
            ctx.save_for_backward(cast)
        ctx.weight = weight
        ctx.input_dtype = x.dtype
        ctx.input_shape = x.shape
        ctx.rows = (row_start, row_end)
        selected = weight[row_start:row_end].to(device=x.device, dtype=torch.bfloat16)
        return F.linear(cast, selected).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        weight = ctx.weight
        grad2 = grad.float().reshape(-1, grad.shape[-1])
        start, end = ctx.rows
        grad_x = None
        if ctx.needs_input_grad[0]:
            selected = weight[start:end].to(device=grad.device, dtype=torch.float32)
            grad_x = (grad2 @ selected).reshape(ctx.input_shape).to(ctx.input_dtype)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            (x,) = ctx.saved_tensors
            rows = grad2.t() @ x.float().reshape(-1, x.shape[-1])
            grad_weight = torch.zeros_like(weight)
            grad_weight[start:end] = rows.to(device=weight.device, dtype=weight.dtype)
        return grad_x, grad_weight, None, None


class BaseLinear(nn.Module):
    def __init__(self, in_features, out_features, num_experts=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features * num_experts, in_features, dtype=torch.bfloat16))
        self.num_experts = num_experts
        self.in_features, self.out_features = in_features, out_features
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x, modality_dispatcher=None):
        return _BF16ComputeLinear.apply(x, self.weight, 0, self.out_features)


class NativeMoELinear(BaseLinear):
    def forward(self, x, modality_dispatcher=None):
        chunks = x.split(modality_dispatcher.group_size_cpu, 0)
        outputs = [_BF16ComputeLinear.apply(chunk, self.weight,
            index * self.out_features, (index + 1) * self.out_features)
            for index, chunk in enumerate(chunks)]
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.cat(outputs).to(x.dtype)


def create_linear(in_features, out_features, num_modality):
    return (NativeMoELinear if num_modality > 1 else BaseLinear)(in_features, out_features, num_modality)


class ElementWiseFourierEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        n = dim // 8
        self.register_buffer("bands", 10000.0 ** (-torch.arange(n).float() / n))

    def forward(self, coords):
        xyz, sizes, refs = coords[:, :3].float(), coords[:, 3:6].float(), coords[:, 6:9].float()
        scales = (refs - 1) / (sizes - 1)
        scales = torch.where((refs == 1) & (sizes == 1), 1.0, scales)
        if not torch.isfinite(scales).all():
            raise ValueError("invalid coordinate scale")
        center = (sizes - 1) / 2
        center = center * center.new_tensor([0, 1, 1])
        phase = (xyz - center).unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), dim=1).flatten(1)


def apply_rotary_emb_torch(x, cos, sin):
    half = cos.shape[-1]
    first, second = x[..., :half], x[..., half:2 * half]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    output = x.clone()
    left, right = output[..., :half], output[..., half:2 * half]
    with torch.autocast(device_type=x.device.type, enabled=False):
        left.copy_(first)
        left.mul_(cos)
        left.addcmul_(second, sin, value=-1.0)
        right.copy_(second)
        right.mul_(cos)
        right.addcmul_(first, sin)
    return output


def _released_inference_attention_function(device):
    """Resolve the release's plain FA3-on-Hopper, FA2, then SDPA order."""
    device = torch.device(device)
    if device.type != "cuda":
        return "auto", None
    if torch.cuda.get_device_capability(device)[0] == 9:
        try:
            from flash_attn_interface import flash_attn_func
            return "flash3", flash_attn_func
        except ModuleNotFoundError:
            pass
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        return "flash2", flash_attn_func
    except ModuleNotFoundError:
        return "auto", None


class Attention(nn.Module):
    def __init__(self, config, num_modality):
        super().__init__()
        self.num_heads_q = config.hidden_size // config.head_dim
        self.num_heads_kv, self.head_dim = config.num_query_groups, config.head_dim
        self.gating_size = self.num_heads_q if config.enable_attn_gating else 0
        self.q_size, self.kv_size = config.hidden_size, self.num_heads_kv * self.head_dim
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality)
        self.q_norm = MultiModalityRMSNorm(self.head_dim, num_modality)
        self.k_norm = MultiModalityRMSNorm(self.head_dim, num_modality)
        self.linear_qkv = create_linear(config.hidden_size, self.q_size + 2 * self.kv_size + self.gating_size, num_modality)
        self.linear_proj = create_linear(config.hidden_size, config.hidden_size, num_modality)
        self.backend = "auto"
        self._backend_resolutions = {}
        self.released_inference_attention = False
        self._released_attention_functions = {}

    def prepare_backend(self, *, device, needs_backward):
        from mirai.core.models.attention_backends import (
            attention_backend_status,
            normalize_attention_backend, varlen_attention_backward_status,
        )
        backend = normalize_attention_backend(self.backend)
        device = torch.device(device)
        if self.released_inference_attention and not self.training and backend == "auto" and not needs_backward:
            key = str(device)
            if key not in self._released_attention_functions:
                self._released_attention_functions[key] = _released_inference_attention_function(device)
            selected = self._released_attention_functions[key][0]
            self._backend_resolutions[("released_auto", key, False)] = selected
            return selected
        cache_key = (backend, str(device), bool(needs_backward))
        if cache_key in self._backend_resolutions:
            return self._backend_resolutions[cache_key]
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        # Capability probes backpropagate through random tensors. Their graphs
        # must be outside checkpoint capture, and cold/warm probes share RNG state.
        with torch.random.fork_rng(devices=devices):
            def status_for(candidate):
                return (varlen_attention_backward_status(candidate, device=device)
                        if needs_backward else attention_backend_status(candidate, device=device, varlen=True))
            selected = backend
            if backend == "auto":
                for candidate in ("flash4", "flash3"):
                    if status_for(candidate).available:
                        selected = candidate
                        break
            elif backend in {"flash3", "flash4", "flex"} and needs_backward:
                status = status_for(backend)
                if not status.available:
                    raise RuntimeError(f"attention backend {backend!r} cannot backpropagate: {status.reason}.")
        self._backend_resolutions[cache_key] = selected
        return selected

    def _packed_attention(self, query, key, value, cu_seqlens):
        from mirai.core.models.attention_backends import (
            dispatch_attention, dispatch_varlen_attention, normalize_attention_backend,
        )
        needs_backward = torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value))
        selected = self.prepare_backend(device=query.device, needs_backward=needs_backward)
        if (self.released_inference_attention and not self.training
                and normalize_attention_backend(self.backend) == "auto" and not needs_backward):
            _, function = self._released_attention_functions[str(query.device)]
            outputs = []
            bounds = cu_seqlens.tolist()
            for start, end in zip(bounds, bounds[1:]):
                q, k, v = [tensor[start:end].unsqueeze(0) for tensor in (query, key, value)]
                if function is not None:
                    output = function(q, k, v)
                    output = output[0] if isinstance(output, tuple) else output
                else:
                    q, k, v = [tensor.transpose(1, 2).contiguous() for tensor in (q, k, v)]
                    factor = q.shape[1] // k.shape[1]
                    k, v = k.repeat_interleave(factor, 1), v.repeat_interleave(factor, 1)
                    output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                                           is_causal=False).transpose(1, 2)
                outputs.append(output.squeeze(0))
            return torch.cat(outputs)
        if selected in {"flash3", "flash4", "flex"}:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            maximum = int(lengths.max())
            return dispatch_varlen_attention(query, key, value,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=maximum, max_seqlen_k=maximum, backend=selected)
        factor = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(factor, 1), value.repeat_interleave(factor, 1)
        bounds = cu_seqlens.tolist()
        return torch.cat([dispatch_attention(query[start:end].unsqueeze(0),
            key[start:end].unsqueeze(0), value[start:end].unsqueeze(0), backend=selected).squeeze(0)
            for start, end in zip(bounds, bounds[1:])])

    def forward(self, x, rope, dispatcher, cu_seqlens):
        x = self.pre_norm(x, dispatcher).to(torch.bfloat16)
        q, k, v, g = self.linear_qkv(x, dispatcher).split((self.q_size, self.kv_size, self.kv_size, self.gating_size), -1)
        q = self.q_norm(q.reshape(-1, self.num_heads_q, self.head_dim), dispatcher)
        k = self.k_norm(k.reshape(-1, self.num_heads_kv, self.head_dim), dispatcher)
        v = v.reshape(-1, self.num_heads_kv, self.head_dim)
        q, k, v = [t[dispatcher.inv_permute_mapping] for t in (q, k, v)]
        sin, cos = rope.chunk(2, -1)
        q, k = [apply_rotary_emb_torch(t, cos, sin).to(torch.bfloat16) for t in (q, k)]
        out = self._packed_attention(q, k, v.to(torch.bfloat16), cu_seqlens)[dispatcher.permute_mapping]
        if self.gating_size:
            out = out * g.reshape(-1, self.num_heads_q, 1).sigmoid()
        return self.linear_proj(out.flatten(1).to(torch.bfloat16), dispatcher)


class _FrozenDenseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base):
        ctx.base = base
        ctx.input_dtype = x.dtype
        ctx.compute_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        weight = base.weight.to(device=x.device, dtype=ctx.compute_dtype)
        bias = None if base.bias is None else base.bias.to(device=x.device, dtype=ctx.compute_dtype)
        return F.linear(x.to(ctx.compute_dtype), weight, bias)

    @staticmethod
    def backward(ctx, grad):
        weight = ctx.base.weight.to(device=grad.device, dtype=ctx.compute_dtype)
        return (grad.to(ctx.compute_dtype) @ weight).to(ctx.input_dtype), None


def frozen_dense_linear(x, base):
    """Frozen dense projection with canonical module-owned weight rematerialization."""
    return _FrozenDenseLinear.apply(x, base)


class FrozenDenseLinear(nn.Linear):
    def forward(self, x):
        if not self.weight.requires_grad and (self.bias is None or not self.bias.requires_grad):
            return frozen_dense_linear(x, self)
        return super().forward(x)


class ExpertLinear(FrozenDenseLinear):
    def forward(self, x):
        x = x.to(torch.bfloat16)
        if not self.weight.requires_grad:
            return frozen_dense_linear(x, self)
        if not torch.is_autocast_enabled(x.device.type):
            # The source grouped path uses BF16 experts with FP32 residuals.
            # The module path keeps that boundary without a dtype-mismatch.
            return F.linear(x, self.weight.to(x.dtype), self.bias)
        return super().forward(x)


class ExpertMLP(nn.Module):
    def __init__(self, config, gelu=False):
        super().__init__()
        mid = config.hidden_size * 4 if gelu else int(config.hidden_size * 8 / 3) // 4 * 4
        self.up_gate_proj = ExpertLinear(config.hidden_size, mid if gelu else 2 * mid, bias=False, dtype=config.params_dtype)
        self.down_proj = ExpertLinear(mid, config.hidden_size, bias=False, dtype=config.params_dtype)
        self.activation_func = gelu7 if gelu else swiglu7

    def forward(self, x):
        projected = self.up_gate_proj(x.to(torch.bfloat16)).to(torch.bfloat16)
        z = self.activation_func(projected).to(torch.bfloat16)
        return self.down_proj(z).to(torch.bfloat16)


class MLP(nn.Module):
    def __init__(self, config, num_modality, gelu=False):
        super().__init__()
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality)
        mid = config.hidden_size * 4 if gelu else int(config.hidden_size * 8 / 3) // 4 * 4
        self.up_gate_proj = create_linear(config.hidden_size, mid if gelu else 2 * mid, num_modality)
        self.down_proj = create_linear(mid, config.hidden_size, num_modality)
        self.activation_func = gelu7 if gelu else swiglu7

    def forward(self, x, dispatcher):
        x = self.pre_norm(x, dispatcher).to(torch.bfloat16)
        return self.down_proj(self.activation_func(self.up_gate_proj(x, dispatcher)).to(torch.bfloat16), dispatcher)


@runtime_checkable
class InferenceDeltaProjection(Protocol):
    def apply_inference_delta(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply an activation-space adapter to a precomputed base projection."""
        ...


class SparseMoEMLP(nn.Module):
    def __init__(self, config, num_modality=1, gelu=False):
        super().__init__()
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality)
        self.router = FrozenDenseLinear(config.hidden_size, config.num_routed_experts, bias=False, dtype=torch.float32)
        nn.init.normal_(self.router.weight, std=0.01)
        self.experts = nn.ModuleList([ExpertMLP(config, gelu) for _ in range(config.num_routed_experts)])
        self.top_k, self.router_noise = config.moe_top_k, config.moe_router_noise
        self._last_aux_loss = None
        self._last_expert_frac = None
        self.released_inference_grouped = False

    @property
    def last_aux_loss(self):
        return self._last_aux_loss

    @property
    def routing_stats(self):
        return self._last_expert_frac

    def _can_use_inference_bmm(self):
        if (not self.released_inference_grouped or self.training or torch.is_grad_enabled()
                or self.top_k != 1 or len(self.experts) < 2):
            return False
        projections = [projection for expert in self.experts
                       for projection in (expert.up_gate_proj, expert.down_proj)]
        if not all(isinstance(projection, (nn.Linear, InferenceDeltaProjection)) for projection in projections):
            return False
        shapes = [(expert.up_gate_proj.weight.shape, expert.down_proj.weight.shape) for expert in self.experts]
        return all(shape == shapes[0] for shape in shapes)

    def _inference_bmm(self, x, indices, gates):
        # Preserve the release's stable top-1 dispatch and expert padding so its
        # batched GEMM uses the same operands, shapes, and rounding boundaries.
        ids = indices[:, 0]
        order = ids.argsort(stable=True)
        sorted_ids = ids[order]
        counts = torch.bincount(sorted_ids, minlength=len(self.experts))
        offsets = counts.cumsum(0) - counts
        columns = torch.arange(len(x), device=x.device) - offsets[sorted_ids]
        maximum = int(counts.max().item())
        if maximum == 0:
            return torch.zeros_like(x)
        padded = x.new_zeros(len(self.experts), maximum, x.shape[1])
        padded_gates = x.new_zeros(len(self.experts), maximum)
        padded[sorted_ids, columns] = x[order]
        padded_gates[sorted_ids, columns] = gates[:, 0].to(x.dtype)[order]
        up_weights = torch.stack([expert.up_gate_proj.weight.to(x.dtype) for expert in self.experts])
        projected = torch.bmm(padded, up_weights.transpose(-2, -1)).to(x.dtype)
        lengths = counts.tolist()
        for index, (expert, length) in enumerate(zip(self.experts, lengths)):
            if length and isinstance(expert.up_gate_proj, InferenceDeltaProjection):
                projected[index, :length] = expert.up_gate_proj.apply_inference_delta(
                    padded[index, :length], projected[index, :length])
        hidden = self.experts[0].activation_func(projected).to(x.dtype)
        down_weights = torch.stack([expert.down_proj.weight.to(x.dtype) for expert in self.experts])
        prediction = torch.bmm(hidden, down_weights.transpose(-2, -1)).to(x.dtype)
        for index, (expert, length) in enumerate(zip(self.experts, lengths)):
            if length and isinstance(expert.down_proj, InferenceDeltaProjection):
                prediction[index, :length] = expert.down_proj.apply_inference_delta(
                    hidden[index, :length], prediction[index, :length])
        prediction = prediction * padded_gates.unsqueeze(-1)
        sorted_output = prediction[sorted_ids, columns]
        output = sorted_output.new_empty(len(x), sorted_output.shape[1])
        output[order] = sorted_output
        return output

    def forward(self, x, dispatcher=None):
        x = self.pre_norm(x, dispatcher).to(torch.bfloat16)
        logits = self.router(x.float())
        if self.training and self.router_noise:
            logits = logits + torch.randn_like(logits) * self.router_noise
        probs = logits.softmax(-1)
        gates, indices = probs.topk(self.top_k, dim=-1)
        gates = gates / gates.sum(-1, keepdim=True).clamp_min(1e-9)
        fractions = F.one_hot(indices[:, 0], len(self.experts)).float().mean(0)
        self._last_expert_frac = fractions.detach()
        self._last_aux_loss = len(self.experts) * (fractions * probs.mean(0)).sum() if self.training else None
        if self._can_use_inference_bmm():
            return self._inference_bmm(x, indices, gates)
        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            token_gate = (gates * (indices == expert_id)).sum(-1)
            rows = torch.nonzero(token_gate > 0, as_tuple=False).flatten()
            if rows.numel():
                y = expert(x.index_select(0, rows))
                output = output.index_add(0, rows, y * token_gate[rows, None].to(y.dtype))
        return output


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.video_embedder = FrozenDenseLinear(config.video_in_channels, config.hidden_size)
        self.audio_embedder = FrozenDenseLinear(config.audio_in_channels, config.hidden_size)
        self.text_embedder = FrozenDenseLinear(config.text_in_channels, config.hidden_size)
        self.rope = ElementWiseFourierEmbed(config.head_dim)
        self.config = config

    def forward(self, tokens, coords, modality):
        output_dtype = (torch.get_autocast_dtype("cuda")
                        if tokens.is_cuda and torch.is_autocast_enabled("cuda") else tokens.dtype)
        output = tokens.new_zeros((len(tokens), self.config.hidden_size), dtype=output_dtype)
        for index, embedder in enumerate((self.video_embedder, self.audio_embedder, self.text_embedder)):
            mask = modality == index
            output[mask] = embedder(tokens[mask, :embedder.in_features].float()).to(output_dtype)
        return output, self.rope(coords)


class TransFormerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        modalities = 3 if layer_idx in config.mm_layers else 1
        self.attention = Attention(config, modalities)
        sparse = layer_idx in config.sparse_moe_layers or (config.token_sparse_moe_convert_mm_mlp and bool(config.sparse_moe_layers) and layer_idx in config.mm_layers)
        self.mlp = (SparseMoEMLP if sparse else MLP)(config, modalities, layer_idx in config.gelu7_layers)
        self.post_norm = layer_idx in config.post_norm_layers
        if self.post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(config.hidden_size, modalities)
            self.mlp_post_norm = MultiModalityRMSNorm(config.hidden_size, modalities)

    def forward(self, x, rope, dispatcher, cu_seqlens):
        attention = self.attention(x, rope, dispatcher, cu_seqlens)
        if self.post_norm:
            attention = self.attn_post_norm(attention, dispatcher)
        x = x + attention
        mlp = self.mlp(x, dispatcher)
        if self.post_norm:
            mlp = self.mlp_post_norm(mlp, dispatcher)
        return x + mlp


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransFormerLayer(config, index) for index in range(config.num_layers)])
        self.gradient_checkpointing = False

    def forward(self, x, rope, dispatcher, cu_seqlens):
        for layer in self.layers:
            layer.attention.prepare_backend(device=x.device, needs_backward=torch.is_grad_enabled())
            if torch.is_grad_enabled():
                # A frozen QKV path may not require gradients even when later
                # trainable projections do; resolve both paths before capture.
                layer.attention.prepare_backend(device=x.device, needs_backward=False)
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(layer, x, rope, dispatcher, cu_seqlens, use_reentrant=False)
            else:
                x = layer(x, rope, dispatcher, cu_seqlens)
        return x


class AgnesDiT(nn.Module):
    def __init__(self, config=AgnesDiTConfig()):
        super().__init__()
        self.config = config
        self.adapter = Adapter(config)
        self.block = TransformerBlock(config)
        self.final_norm_video = MultiModalityRMSNorm(config.hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(config.hidden_size)
        self.final_linear_video = FrozenDenseLinear(config.hidden_size, config.video_in_channels, bias=False)
        self.final_linear_audio = FrozenDenseLinear(config.hidden_size, config.audio_in_channels, bias=False)

    @property
    def gradient_checkpointing(self):
        return self.block.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, enabled):
        self.block.gradient_checkpointing = bool(enabled)

    def forward(self, tokens, coords, modality, cu_seqlens):
        if coords.shape != (len(tokens), 9) or modality.shape != (len(tokens),):
            raise ValueError("invalid packed token metadata")
        bounds = cu_seqlens.tolist()
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != len(tokens) or any(a >= b for a, b in zip(bounds, bounds[1:])):
            raise ValueError("cu_seqlens must delimit nonempty packed samples")
        if ((modality < 0) | (modality > 2)).any():
            raise ValueError("invalid modality")
        dispatcher = ModalityDispatcher(modality)
        x, rope = self.adapter(tokens, coords, modality)
        x = x.to(self.config.params_dtype)[dispatcher.permute_mapping]
        x = self.block(x, rope, dispatcher, cu_seqlens)[dispatcher.inv_permute_mapping]
        output = x.new_zeros((len(tokens), max(self.config.video_in_channels, self.config.audio_in_channels)))
        for index, norm, projection in ((0, self.final_norm_video, self.final_linear_video), (1, self.final_norm_audio, self.final_linear_audio)):
            mask = modality == index
            output[mask, :projection.out_features] = projection(norm(x[mask].float())).to(output.dtype)
        return output

    def collect_moe_aux_loss(self):
        losses = [layer.mlp._last_aux_loss for layer in self.block.layers if isinstance(layer.mlp, SparseMoEMLP) and layer.mlp._last_aux_loss is not None]
        return torch.stack(losses).mean() if losses else None


DiTModel = AgnesDiT
