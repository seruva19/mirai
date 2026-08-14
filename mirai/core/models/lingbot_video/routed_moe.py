# SPDX-License-Identifier: Apache-2.0
"""LingBot resident-BF16 adapter for the generic routed Triton projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mirai.core.moe.runtime.routed_gemm import (
    RoutedFusionSpec,
    RoutedGroupLayout,
    RoutedOutputMode,
    routed_gemm_verdict,
)
from mirai.core.moe.runtime.routed_gemm_triton import (
    routed_projection,
    routed_weighted_projection,
)


@dataclass(frozen=True)
class LingBotRoutedTritonBackend:
    mode: str
    direct_only: bool = True
    supports_routed_output_observer: bool = False
    supports_routed_intermediate_observer: bool = False

    def execute_direct(self, experts, tokens, top_scores, top_indices):
        flat_scores = top_scores.reshape(-1)
        flat_experts = top_indices.reshape(-1)
        grouped_to_assignment = flat_experts.argsort(stable=True)
        counts = torch.bincount(flat_experts, minlength=int(experts.w1.shape[0]))
        boundaries = counts.cumsum(0).to(torch.int32)
        top_k = int(top_indices.shape[1])
        layout = RoutedGroupLayout(
            boundaries=boundaries,
            assignment_rows=grouped_to_assignment,
            token_count=int(tokens.shape[0]),
            top_k=top_k,
            group_count=int(experts.w1.shape[0]),
            provider_mapping=("expert",),
        )
        verdicts = [
            routed_gemm_verdict(
                self.mode,
                tokens,
                weight.transpose(-2, -1),
                RoutedFusionSpec(gather_tokens=True),
                training=torch.is_grad_enabled(),
                resident=True,
                quantized=False,
                layout=layout,
            )
            for weight in (experts.w1, experts.w3)
        ]
        verdicts.append(
            routed_gemm_verdict(
                self.mode,
                tokens.new_empty((int(flat_scores.numel()), int(experts.w2.shape[-1]))),
                experts.w2.transpose(-2, -1),
                RoutedFusionSpec(output=RoutedOutputMode.WEIGHTED_TOKEN_REDUCTION),
                training=torch.is_grad_enabled(),
                resident=True,
                quantized=False,
                layout=layout,
            )
        )
        verdict = next(
            (item for item in verdicts if item.selected != "triton" or not item.supported),
            verdicts[-1],
        )
        if verdict.selected != "triton" or not verdict.supported:
            if self.mode == "triton":
                raise RuntimeError(
                    "memory.moe_routed_gemm='triton' " + verdict.reason + "."
                )
            return None
        layout.validate(device=tokens.device, check_values=False)
        gather_rows = torch.div(grouped_to_assignment, top_k, rounding_mode="floor")
        gate = routed_projection(
            tokens, experts.w1.transpose(-2, -1), boundaries,
            gather_rows=gather_rows,
        )
        up = routed_projection(
            tokens, experts.w3.transpose(-2, -1), boundaries,
            gather_rows=gather_rows,
        )
        hidden = F.silu(gate) * up
        assignment_to_token = torch.div(
            torch.arange(flat_scores.numel(), device=tokens.device),
            top_k,
            rounding_mode="floor",
        )
        return routed_weighted_projection(
            hidden, experts.w2.transpose(-2, -1), boundaries,
            grouped_to_assignment=grouped_to_assignment,
            assignment_to_token=assignment_to_token,
            coefficients=flat_scores,
            token_rows=int(tokens.shape[0]),
        ).type_as(tokens)
