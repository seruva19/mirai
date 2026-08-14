"""Compiler and static CUDA Graph bucket contracts for routed GEMM."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from mirai.core.moe.runtime.routed_gemm_graph import RoutedGemmGraphBucket


_REMOTE_GPU = bool(torch.cuda.is_available()) and __import__("os").environ.get(
    "MIRAI_REMOTE_GPU_TESTS"
) == "1"


def test_registered_operator_has_fake_shape_contract() -> None:
    from mirai.core.moe.runtime.routed_gemm_ops import routed_projection_op
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode()
    with mode:
        activation = torch.empty(5, 17, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(3, 17, 23, device="cuda", dtype=torch.bfloat16)
        boundaries = torch.empty(3, device="cuda", dtype=torch.int32)
        empty = torch.empty(0, device="cuda", dtype=torch.int64)
        output = routed_projection_op(activation, weight, boundaries, empty, empty, 5)
    assert output.shape == (5, 23)
    assert output.dtype == torch.bfloat16
    assert output.device.type == "cuda"


def test_weighted_operator_has_fake_shape_contract() -> None:
    from mirai.core.moe.runtime.routed_gemm_ops import routed_weighted_projection_op
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        output = routed_weighted_projection_op(
            torch.empty(8, 17, device="cuda", dtype=torch.bfloat16),
            torch.empty(3, 17, 23, device="cuda", dtype=torch.bfloat16),
            torch.empty(3, device="cuda", dtype=torch.int32),
            torch.empty(8, device="cuda", dtype=torch.int64),
            torch.empty(8, device="cuda", dtype=torch.int64),
            torch.empty(8, device="cuda", dtype=torch.bfloat16),
            4,
        )
    assert output.shape == (4, 23)
    assert output.dtype == torch.bfloat16


def test_graph_bucket_rejects_shape_or_fusion_drift() -> None:
    bucket = RoutedGemmGraphBucket(4, 8, 3, 17, 23, True, False)
    activation = torch.empty(4, 17)
    weight = torch.empty(3, 17, 23)
    gather = torch.empty(8, dtype=torch.int64)
    empty = torch.empty(0, dtype=torch.int64)
    bucket.validate_call(activation, weight, gather, empty)
    with pytest.raises(ValueError, match="does not match"):
        bucket.validate_call(activation, weight, gather[:7], empty)
    with pytest.raises(ValueError, match="cannot gather and scatter"):
        RoutedGemmGraphBucket(4, 8, 3, 17, 23, True, True)


def test_prewarm_is_forbidden_during_capture() -> None:
    from mirai.core.moe.runtime.routed_gemm_graph import prewarm_routed_gemm_bucket

    bucket = RoutedGemmGraphBucket(4, 4, 3, 17, 23, False, False)
    with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
        with pytest.raises(RuntimeError, match="before CUDA Graph capture"):
            prewarm_routed_gemm_bucket(
                bucket, torch.empty(4, 17), torch.empty(3, 17, 23),
                torch.empty(3, dtype=torch.int32),
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
            )


@pytest.mark.skipif(not _REMOTE_GPU, reason="requires leased remote CUDA GPU")
def test_routed_projection_compiles_forward_and_backward_without_graph_break() -> None:
    from mirai.core.moe.runtime.routed_gemm_triton import routed_projection

    torch.manual_seed(41)
    activation = torch.randn(12, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(3, 32, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    boundaries = torch.tensor([3, 7, 12], device="cuda", dtype=torch.int32)

    def run(x, w):
        return routed_projection(x, w, boundaries).float().square().mean()

    eager = run(activation, weight)
    eager_grads = torch.autograd.grad(eager, (activation, weight))
    compiled = torch.compile(run, backend="inductor", fullgraph=True)
    candidate = compiled(activation, weight)
    candidate_grads = torch.autograd.grad(candidate, (activation, weight))
    torch.testing.assert_close(candidate, eager, rtol=2e-2, atol=2e-2)
    for actual, expected in zip(candidate_grads, eager_grads):
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not _REMOTE_GPU, reason="requires leased remote CUDA GPU")
def test_routed_projection_cuda_graph_replays_updated_static_inputs() -> None:
    from mirai.core.moe.runtime.routed_gemm_graph import prewarm_routed_gemm_bucket
    from mirai.core.moe.runtime.routed_gemm_triton import routed_projection

    activation = torch.randn(12, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(3, 32, 48, device="cuda", dtype=torch.bfloat16)
    boundaries = torch.tensor([3, 7, 12], device="cuda", dtype=torch.int32)
    empty = torch.empty(0, device="cuda", dtype=torch.int64)
    bucket = RoutedGemmGraphBucket.from_tensors(
        activation, weight, routed_rows=12, gather=False, scatter=False
    )
    prewarm_routed_gemm_bucket(bucket, activation, weight, boundaries, empty, empty)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = routed_projection(activation, weight, boundaries)
    graph.replay()
    first = captured.clone()
    activation.copy_(torch.randn_like(activation))
    expected = routed_projection(activation, weight, boundaries)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)
    assert not torch.equal(first, captured)
