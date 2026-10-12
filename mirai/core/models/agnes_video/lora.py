"""Activation-space LoRA for Agnes modality-specific native projections."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mirai.core.models.adapters.lora import LoRALinear
from mirai.vendors.agnes_video.dit import frozen_dense_linear


class AgnesLoRALinear(LoRALinear):
    def _base_forward(self, x):
        return frozen_dense_linear(x, self.base)

    def apply_inference_delta(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Keep the activation LoRA delta when the base uses grouped GEMM."""
        if self.training or self.use_dora or self._timestep_mask_per_sample is not None or self._tc_gate_hypernet is not None:
            raise RuntimeError("Grouped Agnes projection requires plain inference LoRA.")
        a = self.lora_a.to(device=x.device, dtype=x.dtype)
        b = self.lora_b.to(device=x.device, dtype=x.dtype)
        delta = F.linear(F.linear(x, a), b)
        scale = (self._alpha() / (self.rank ** 0.5 if self.use_rslora else self.rank)
                 * self._lora_scale * self._rank_schedule_scale)
        return base_output + delta * scale


class ModalityLoRALinear(nn.Module):
    """Independent low-rank deltas on the release's stacked modality weights."""

    def __init__(self, base: nn.Module, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("Agnes LoRA rank must be positive.")
        self.base = base
        self.num_experts = base.num_experts
        self.in_features = int(base.weight.shape[1])
        self.out_features = int(base.weight.shape[0] // self.num_experts)
        self.lora_a = nn.Parameter(torch.empty(self.num_experts, rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.num_experts, self.out_features, rank))
        self.register_buffer("lora_alpha", torch.tensor(float(alpha)))
        self.rank = int(rank)
        self.runtime_scale = 1.0
        self.rank_dropout = 0.0
        self.parameter_dropout = 0.0
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        base.requires_grad_(False)

    def forward(self, x: torch.Tensor, modality_dispatcher=None) -> torch.Tensor:
        output = self.base(x, modality_dispatcher)
        chunks = x.split(modality_dispatcher.group_size_cpu, 0) if self.num_experts > 1 else (x,)
        deltas = []
        for index, chunk in enumerate(chunks):
            a = F.dropout(self.lora_a[index], self.parameter_dropout, self.training)
            b = F.dropout(self.lora_b[index], self.parameter_dropout, self.training)
            hidden = F.linear(chunk.float(), a.float())
            hidden = F.dropout(hidden, self.rank_dropout, self.training)
            deltas.append(F.linear(hidden, b.float()))
        delta = torch.cat(deltas) * (self.lora_alpha / self.rank) * self.runtime_scale
        return output + delta.to(output.dtype)


def adapter_modules(model: nn.Module):
    return (module for module in model.modules() if isinstance(module, (ModalityLoRALinear, LoRALinear)))
