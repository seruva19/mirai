"""Bounded GPU behavioral probe for optional MoE memory and execution extensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from mirai.core.models.compressed_weights.execution.mixed_precision import (
    MixedPrecisionGroupedExperts,
)
from mirai.core.models.compressed_weights.quantization.structured_sparsity import (
    prune_to_2_4,
    sparse_2_4_linear,
)
from mirai.core.training.residency.activation_compression import (
    LowRankActivationCompression,
)
from mirai.core.training.runtime.gpu_lease import (
    acquire_gpu_lease,
    resolve_lease_lock_path,
)


class _DenseExperts(nn.Module):
    def __init__(self, *, experts: int, width: int, device: torch.device) -> None:
        super().__init__()
        self.num_experts = int(experts)
        shape = (experts, width, width)
        self.w1 = nn.Parameter(torch.randn(shape, device=device), requires_grad=False)
        self.w2 = nn.Parameter(torch.randn(shape, device=device), requires_grad=False)
        self.w3 = nn.Parameter(torch.randn(shape, device=device), requires_grad=False)


def run_probe(device: torch.device) -> dict[str, float | int | bool]:
    torch.manual_seed(19)

    weight = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
    sparse_state = prune_to_2_4(weight)
    inputs = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
    dense_output = sparse_2_4_linear(inputs, sparse_state, backend="reference")
    sparse_output = sparse_2_4_linear(inputs, sparse_state, backend="cuda")
    torch.testing.assert_close(sparse_output, dense_output, rtol=0.02, atol=0.02)

    mixed = MixedPrecisionGroupedExperts(
        _DenseExperts(experts=2, width=64, device=device),
        formats=("int8", "mxfp4"),
    )
    tokens = torch.randn(16, 64, device=device, requires_grad=True)
    scores = torch.softmax(torch.randn(16, 2, device=device), dim=-1).requires_grad_(True)
    indices = torch.tensor([[0, 1]] * 16, device=device)
    mixed_output = mixed.run_direct_routed(tokens, scores, indices)
    mixed_output.float().square().mean().backward()
    if tokens.grad is None or scores.grad is None:
        raise AssertionError("Mixed-precision routed execution lost trainable gradients.")

    compression = LowRankActivationCompression(rank=8, min_bytes=0, seed=23)
    low_rank = (
        torch.randn(1024, 8, device=device)
        @ torch.randn(8, 512, device=device)
    ).requires_grad_(True)
    projection = (
        torch.randn(512, 8, device=device)
        @ torch.randn(8, 128, device=device)
    ).requires_grad_(True)
    reference_low_rank = low_rank.detach().clone().requires_grad_(True)
    reference_projection = projection.detach().clone().requires_grad_(True)
    reference_loss = (reference_low_rank @ reference_projection).square().mean()
    reference_loss.backward()
    with compression.context():
        loss = (low_rank @ projection).square().mean()
    loss.backward()
    if compression.compressed_tensors <= 0:
        raise AssertionError("Activation compression did not claim an eligible tensor.")
    if compression.stored_bytes >= compression.original_bytes:
        raise AssertionError("Activation compression did not reduce saved storage.")
    torch.testing.assert_close(loss, reference_loss, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        low_rank.grad, reference_low_rank.grad, rtol=3e-2, atol=3e-3
    )
    torch.testing.assert_close(
        projection.grad, reference_projection.grad, rtol=3e-2, atol=3e-3
    )

    return {
        "structured_2_4_parity": True,
        "mixed_precision_gradients": True,
        "activation_compressed_tensors": int(compression.compressed_tensors),
        "activation_original_bytes": int(compression.original_bytes),
        "activation_stored_bytes": int(compression.stored_bytes),
        "activation_storage_ratio": float(
            compression.stored_bytes / compression.original_bytes
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This probe requires an explicitly leased CUDA device.")
    with acquire_gpu_lease(
        lock_path=resolve_lease_lock_path(Path.cwd()),
        timeout_seconds=0.0,
    ):
        report = run_probe(device)
        torch.cuda.synchronize(device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
