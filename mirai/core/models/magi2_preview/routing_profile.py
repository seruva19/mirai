"""Observe packed expert IDs for a single MAGI-2 calibration forward."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .quantized_experts import iter_magi2_moe_layers


class _RoutingProfileTap:
    def __init__(self, inner, num_heads, num_experts):
        self.inner = inner
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.touched = None

    def execute(self, module, x_heads, topk_probs, topk_indices):
        head_offsets = torch.arange(self.num_heads, device=topk_indices.device) * self.num_experts
        ids = (topk_indices + head_offsets[:, None, None]).reshape(-1)
        if self.touched is None:
            self.touched = torch.zeros(self.num_heads * self.num_experts, device=ids.device, dtype=torch.bool)
        self.touched.scatter_(0, ids.long(), True)
        if self.inner is None:
            return module._native_expert_forward(x_heads, topk_probs, topk_indices)
        return self.inner.execute(module, x_heads, topk_probs, topk_indices)


@contextmanager
def observe_magi2_expert_ids(transformer):
    """Record logical layer/head/expert IDs without changing expert execution."""
    layers = []
    routed = set()
    try:
        for _, module in iter_magi2_moe_layers(transformer):
            original = module._mirai_moe_kernel_backend
            tap = _RoutingProfileTap(original, int(module.num_heads), int(module.num_experts))
            layers.append((module, original, tap))
            module._mirai_moe_kernel_backend = tap
        if not layers:
            raise ValueError("Routing profile collection requires MAGI-2 MoE layers.")
        yield routed
        masks = []
        device = next((tap.touched.device for _, _, tap in layers if tap.touched is not None), None)
        if device is None:
            raise RuntimeError("The calibration forward did not execute any MAGI-2 routed experts.")
        for _, _, tap in layers:
            mask = tap.touched
            if mask is None:
                mask = torch.zeros(tap.num_heads * tap.num_experts, device=device, dtype=torch.bool)
            masks.append(mask)
        routed.update(torch.cat(masks).cpu().nonzero().flatten().tolist())
    finally:
        for module, original, _ in layers:
            module._mirai_moe_kernel_backend = original
