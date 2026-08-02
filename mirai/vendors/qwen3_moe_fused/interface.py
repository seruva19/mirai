# Vendored from woct0rdho/transformers-qwen3-moe-fused (Apache-2.0).
# Source: https://github.com/woct0rdho/transformers-qwen3-moe-fused
# Reference: qwen3_moe_fused/grouped_gemm/interface.py.
# The adjacent LICENSE is Apache-2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

import torch

from .backward_dw import grouped_gemm_backward_dw
from .forward import grouped_gemm_forward


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, m_offsets):
        ctx.save_for_backward(x, w, m_offsets)
        return grouped_gemm_forward(x, w, m_offsets)

    @staticmethod
    def backward(ctx, dy):
        x, w, m_offsets = ctx.saved_tensors

        if x.requires_grad:
            dx = grouped_gemm_forward(dy, w, m_offsets, x.dtype, transpose_w=True)
        else:
            dx = None

        if w.requires_grad:
            dw = grouped_gemm_backward_dw(x, dy, m_offsets, w.dtype)
        else:
            dw = None

        return dx, dw, None


grouped_gemm = GroupedGemm.apply
