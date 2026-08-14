"""Pure contracts for routed grouped-GEMM layout and reference semantics."""

import os
import subprocess
import sys

import pytest
import torch

from mirai.core.moe.runtime.routed_gemm import (
    RoutedFusionSpec,
    RoutedGroupLayout,
    RoutedOutputMode,
    normalize_routed_gemm_mode,
    routed_gemm_reference,
    routed_gemm_verdict,
)


def _layout() -> RoutedGroupLayout:
    return RoutedGroupLayout(
        boundaries=torch.tensor([1, 1, 4], dtype=torch.int32),
        assignment_rows=torch.tensor([2, 0, 3, 1], dtype=torch.int64),
        token_count=2,
        top_k=2,
        group_count=3,
        provider_mapping=("head", "expert"),
    )


def test_gather_reference_and_repeated_token_gradient() -> None:
    layout = _layout()
    x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    weight = torch.randn(3, 3, 4, dtype=torch.double)
    output = routed_gemm_reference(x, weight, layout, RoutedFusionSpec(gather_tokens=True))
    token_rows = torch.div(layout.assignment_rows, 2, rounding_mode="floor")
    gathered = x.index_select(0, token_rows)
    expected = torch.cat((gathered[:1] @ weight[0], gathered[1:] @ weight[2]))
    assert torch.equal(output, expected)
    output.sum().backward()
    expected_grad = torch.zeros_like(x).index_add(
        0,
        token_rows,
        torch.cat((weight[0].sum(1)[None], weight[2].sum(1).expand(3, -1))),
    )
    assert torch.allclose(x.grad, expected_grad)


def test_scatter_and_inference_reduction() -> None:
    layout = _layout()
    x = torch.randn(4, 3)
    weight = torch.randn(3, 3, 2)
    grouped = routed_gemm_reference(x, weight, layout)
    scattered = routed_gemm_reference(
        x, weight, layout, RoutedFusionSpec(output=RoutedOutputMode.ASSIGNMENT)
    )
    assert torch.equal(scattered.index_select(0, layout.assignment_rows), grouped)
    with torch.no_grad():
        coefficients = torch.tensor([[0.25, 0.75], [0.4, 0.6]])
        reduced = routed_gemm_reference(
            x,
            weight,
            layout,
            RoutedFusionSpec(output=RoutedOutputMode.WEIGHTED_TOKEN_REDUCTION),
            routing_weights=coefficients,
        )
    assert torch.allclose(reduced, (scattered.view(2, 2, -1) * coefficients[..., None]).sum(1))


def test_reference_weighted_reduction_returns_coefficient_and_weight_gradients() -> None:
    layout = _layout()
    x = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(3, 3, 2, requires_grad=True)
    coefficients = torch.randn(2, 2, requires_grad=True)
    reduced = routed_gemm_reference(
        x, weight, layout,
        RoutedFusionSpec(output=RoutedOutputMode.WEIGHTED_TOKEN_REDUCTION),
        routing_weights=coefficients,
    )
    reduced.square().sum().backward()
    assert x.grad is not None and weight.grad is not None and coefficients.grad is not None
    assert torch.equal(weight.grad[1], torch.zeros_like(weight.grad[1]))


@pytest.mark.parametrize(
    "boundaries,rows,match",
    [
        ([1, 0, 4], [0, 1, 2, 3], "non-decreasing"),
        ([0, 0, 3], [0, 1, 2, 3], "terminal"),
        ([1, 1, 4], [0, 1, 2, 4], "out-of-range"),
        ([1, 1, 4], [0, 1, 1, 3], "permutation"),
    ],
)
def test_layout_rejects_invalid_metadata(boundaries, rows, match) -> None:
    layout = RoutedGroupLayout(torch.tensor(boundaries), torch.tensor(rows), 2, 2, 3)
    with pytest.raises(ValueError, match=match):
        layout.validate()


def test_empty_layout_and_incompatible_fusions() -> None:
    layout = RoutedGroupLayout(torch.tensor([0, 0]), torch.empty(0, dtype=torch.int64), 0, 1, 2)
    activation = torch.empty(0, 3, requires_grad=True)
    weight = torch.randn(2, 3, 5, requires_grad=True)
    result = routed_gemm_reference(activation, weight, layout)
    assert result.shape == (0, 5)
    result.sum().backward()
    assert torch.equal(activation.grad, torch.zeros_like(activation))
    assert torch.equal(weight.grad, torch.zeros_like(weight))
    with pytest.raises(ValueError, match="cannot request"):
        RoutedFusionSpec(gather_tokens=True, output=RoutedOutputMode.ASSIGNMENT)


def test_mode_and_capability_fallback_are_explicit() -> None:
    x = torch.randn(2, 3)
    weight = torch.randn(3, 3, 4)
    fusion = RoutedFusionSpec(gather_tokens=True)
    assert normalize_routed_gemm_mode(None) == "disabled"
    assert routed_gemm_verdict("auto", x, weight, fusion, training=True, resident=True, quantized=False).selected == "reference"
    verdict = routed_gemm_verdict("triton", x, weight, fusion, training=True, resident=True, quantized=False)
    assert not verdict.supported
    assert "CUDA" in verdict.reason


def test_capability_verdict_covers_layout_fusion_and_backend_dimensions() -> None:
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(3, 3, 4, dtype=torch.bfloat16)
    fusion = RoutedFusionSpec(gather_tokens=True)
    layout = _layout()
    verdict = routed_gemm_verdict(
        "triton",
        x,
        weight,
        fusion,
        training=True,
        resident=True,
        quantized=False,
        layout=layout,
        architecture="tma_regular",
        triton_available=False,
    )
    assert not verdict.supported
    assert "CUDA" in verdict.reason
    assert "Triton" in verdict.reason
    assert "tma_regular" in verdict.reason
    quantized = routed_gemm_verdict(
        "auto",
        x,
        weight,
        fusion,
        training=False,
        resident=False,
        quantized=True,
        layout=layout,
        triton_available=True,
    )
    assert quantized.supported and quantized.selected == "reference"
    assert "unquantized" in quantized.reason


def test_disabled_mode_imports_no_kernel_or_tuning_owner() -> None:
    probe = """
import sys
from mirai.core.moe.runtime.routed_gemm import normalize_routed_gemm_mode
assert normalize_routed_gemm_mode('disabled') == 'disabled'
for name in (
    'mirai.core.moe.runtime.routed_gemm_triton',
    'mirai.core.moe.runtime.routed_gemm_autotune',
    'mirai.core.moe.runtime.routed_gemm_ops',
    'triton',
):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_capability_verdict_rejects_provider_and_training_fusion_gaps() -> None:
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(2, 3, 4, dtype=torch.bfloat16)
    verdict = routed_gemm_verdict(
        "triton",
        x,
        weight,
        RoutedFusionSpec(output=RoutedOutputMode.WEIGHTED_TOKEN_REDUCTION),
        training=True,
        resident=True,
        quantized=False,
        provider_declared=False,
        fusion_gradients_supported=False,
        triton_available=True,
    )
    assert not verdict.supported
    assert "provider-declared" in verdict.reason
    assert "training gradients" in verdict.reason


def test_disabled_policy_constructs_no_runtime_state() -> None:
    probe = """
import sys
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy
policy = MoEOptimizationPolicy(moe_routed_gemm='disabled')
assert policy.moe_routed_gemm == 'disabled'
assert policy.moe_routed_gemm_tuning == 'off'
assert policy.moe_routed_gemm_cache_path == ''
assert 'mirai.core.moe.runtime.routed_gemm_triton' not in sys.modules
assert 'mirai.core.moe.runtime.routed_gemm_autotune' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", probe], check=True)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("boundaries", torch.tensor([[1]], dtype=torch.int32), "rank-1"),
        ("boundaries", torch.tensor([1.0]), "int32 or int64"),
        ("boundaries", torch.tensor([1, 1], dtype=torch.int32), "one entry per group"),
        ("gather", torch.tensor([[0]], dtype=torch.int64), "rank-1"),
        ("gather", torch.tensor([0.0]), "int32 or int64"),
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_routed_entry_rejects_malformed_static_metadata(field, value, match) -> None:
    from mirai.core.moe.runtime.routed_gemm_triton import triton_routed_grouped_mm

    activation = torch.empty(1, 3, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(1, 3, 4, device="cuda", dtype=torch.bfloat16)
    boundaries = torch.tensor([1], device="cuda", dtype=torch.int32)
    value = value.to("cuda")
    kwargs = {"gather_rows": value} if field == "gather" else {}
    if field == "boundaries":
        boundaries = value
    with pytest.raises((TypeError, ValueError), match=match):
        triton_routed_grouped_mm(activation, weight, boundaries, **kwargs)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_weighted_entry_rejects_malformed_static_metadata() -> None:
    from mirai.core.moe.runtime.routed_gemm_triton import routed_weighted_projection

    activation = torch.empty(2, 3, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(1, 3, 4, device="cuda", dtype=torch.bfloat16)
    boundaries = torch.tensor([2], device="cuda", dtype=torch.int32)
    assignment = torch.arange(2, device="cuda", dtype=torch.int64)
    coefficients = torch.ones(2, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="rank-1"):
        routed_weighted_projection(
            activation, weight, boundaries,
            grouped_to_assignment=assignment.view(1, 2),
            assignment_to_token=assignment,
            coefficients=coefficients,
            token_rows=2,
        )
    with pytest.raises(TypeError, match="int32 or int64"):
        routed_weighted_projection(
            activation, weight, boundaries,
            grouped_to_assignment=assignment.float(),
            assignment_to_token=assignment,
            coefficients=coefficients,
            token_rows=2,
        )
    with pytest.raises(ValueError, match="equal length"):
        routed_weighted_projection(
            activation, weight, boundaries,
            grouped_to_assignment=assignment,
            assignment_to_token=assignment[:1],
            coefficients=coefficients,
            token_rows=2,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_max_group_rows_hint_matches_unhinted_projection() -> None:
    pytest.importorskip("triton")
    from mirai.core.moe.runtime.routed_gemm_triton import (
        _tiled_indexed_kernel,
        triton_routed_grouped_mm,
    )

    kernel, _ = _tiled_indexed_kernel()
    assert "rows" not in kernel.arg_names

    device = torch.device("cuda")
    counts = torch.tensor([0, 17, 3, 0, 31, 1] * 64, device=device)
    boundaries = counts.cumsum(0, dtype=torch.int32)
    rows = int(boundaries[-1].item())
    generator = torch.Generator(device=device).manual_seed(913)
    activation = torch.randn(
        rows, 73, device=device, dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(
        counts.numel(), 73, 79,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    expected = triton_routed_grouped_mm(activation, weight, boundaries)
    actual = triton_routed_grouped_mm(
        activation, weight, boundaries, max_group_rows=31
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    with pytest.raises(ValueError, match="max_group_rows"):
        triton_routed_grouped_mm(
            activation, weight, boundaries, max_group_rows=0
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_triton_gather_and_input_gradient_parity() -> None:
    pytest.importorskip("triton")
    from mirai.core.moe.runtime.routed_gemm_triton import routed_projection

    device = torch.device("cuda")
    layout = RoutedGroupLayout(
        torch.tensor([1, 1, 4], device=device, dtype=torch.int32),
        torch.tensor([2, 0, 3, 1], device=device, dtype=torch.int64),
        2, 2, 3,
    )
    x_ref = torch.randn(2, 17, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    weight = torch.randn(3, 17, 23, device=device, dtype=torch.bfloat16)
    reference = routed_gemm_reference(
        x_ref, weight, layout, RoutedFusionSpec(gather_tokens=True)
    )
    candidate = routed_projection(
        x_tri, weight, layout.boundaries,
        gather_rows=torch.div(layout.assignment_rows, 2, rounding_mode="floor"),
    )
    torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=2e-2)
    grad = torch.randn_like(candidate)
    candidate.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(x_tri.grad, x_ref.grad, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_triton_rejects_combined_gather_and_scatter() -> None:
    pytest.importorskip("triton")
    from mirai.core.moe.runtime.routed_gemm_triton import routed_projection

    activation = torch.randn(2, 3, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2, 3, 4, device="cuda", dtype=torch.bfloat16)
    boundaries = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    mapping = torch.arange(2, device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match="cannot combine"):
        routed_projection(
            activation, weight, boundaries,
            gather_rows=mapping, scatter_rows=mapping,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
@pytest.mark.parametrize(
    "seed,input_rows,k_size,n_size,counts,mode",
    [
        (11, 5, 15, 19, (0, 1, 0, 8), "gather"),
        (12, 4, 33, 17, (7, 0, 1, 0), "gather"),
        (13, 9, 17, 35, (0, 0, 9), "scatter"),
        (14, 6, 31, 23, (1, 7, 0, 2), "scatter"),
    ],
)
def test_triton_randomized_irregular_routing_parity(
    seed, input_rows, k_size, n_size, counts, mode
) -> None:
    pytest.importorskip("triton")
    from mirai.core.moe.runtime.routed_gemm_triton import routed_projection

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    routed_rows = sum(counts)
    boundaries = torch.tensor(counts, device=device, dtype=torch.int32).cumsum(0)
    weight = torch.randn(
        len(counts), k_size, n_size, device=device, dtype=torch.bfloat16,
        generator=generator,
    )
    gather_rows = None
    scatter_rows = None
    if mode == "gather":
        # Deliberately repeat source rows so backward must reduce contributions.
        row_map = torch.arange(routed_rows, device=device, dtype=torch.int64) % input_rows
        gather_rows = row_map.roll(seed % max(routed_rows, 1))
        activation_rows = input_rows
    else:
        scatter_rows = torch.randperm(routed_rows, device=device, generator=generator)
        activation_rows = routed_rows
    x_ref = torch.randn(
        activation_rows, k_size, device=device, dtype=torch.bfloat16,
        generator=generator, requires_grad=True,
    )
    x_tri = x_ref.detach().clone().requires_grad_(True)

    grouped_input = x_ref if gather_rows is None else x_ref.index_select(0, gather_rows)
    pieces = []
    start = 0
    for group, stop in enumerate(boundaries.tolist()):
        if stop > start:
            pieces.append(grouped_input[start:stop] @ weight[group])
        start = stop
    grouped_reference = torch.cat(pieces, 0)
    reference = grouped_reference
    if scatter_rows is not None:
        reference = torch.empty_like(grouped_reference).index_copy(
            0, scatter_rows, grouped_reference
        )
    candidate = routed_projection(
        x_tri, weight, boundaries,
        gather_rows=gather_rows, scatter_rows=scatter_rows,
    )
    torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=2e-2)
    upstream = torch.randn(candidate.shape, device=device, dtype=torch.bfloat16, generator=generator)
    candidate.backward(upstream)
    reference.backward(upstream)
    torch.testing.assert_close(x_tri.grad, x_ref.grad, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_triton_all_empty_groups_return_typed_empty_output() -> None:
    pytest.importorskip("triton")
    from mirai.core.moe.runtime.routed_gemm_triton import routed_projection

    x = torch.empty(0, 17, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        4, 17, 23, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    boundaries = torch.zeros(4, device="cuda", dtype=torch.int32)
    output = routed_projection(x, weight, boundaries)
    assert output.shape == (0, 23)
    assert output.dtype == torch.bfloat16
    output.sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))
    assert torch.equal(weight.grad, torch.zeros_like(weight))


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_triton_weighted_reduction_input_weight_and_coefficient_gradient_parity() -> None:
    pytest.importorskip("triton")
    from mirai.core.moe.runtime.routed_gemm_triton import routed_weighted_projection

    device = torch.device("cuda")
    torch.manual_seed(29)
    boundaries = torch.tensor([2, 2, 6], device=device, dtype=torch.int32)
    grouped_to_assignment = torch.tensor([3, 0, 5, 1, 4, 2], device=device)
    assignment_to_token = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
    x_ref = torch.randn(6, 17, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    w_ref = torch.randn(3, 17, 19, device=device, dtype=torch.bfloat16, requires_grad=True)
    w_tri = w_ref.detach().clone().requires_grad_(True)
    c_ref = torch.randn(6, device=device, dtype=torch.bfloat16, requires_grad=True)
    c_tri = c_ref.detach().clone().requires_grad_(True)
    pieces = torch.cat((x_ref[:2] @ w_ref[0], x_ref[2:] @ w_ref[2]))
    token_for_grouped = assignment_to_token.index_select(0, grouped_to_assignment)
    coeff_grouped = c_ref.index_select(0, grouped_to_assignment)
    reference = torch.zeros(3, 19, device=device, dtype=torch.bfloat16).index_add(
        0, token_for_grouped, pieces * coeff_grouped[:, None]
    )
    candidate = routed_weighted_projection(
        x_tri, w_tri, boundaries,
        grouped_to_assignment=grouped_to_assignment,
        assignment_to_token=assignment_to_token,
        coefficients=c_tri, token_rows=3,
    )
    upstream = torch.randn_like(candidate)
    reference.backward(upstream)
    candidate.backward(upstream)
    for actual, expected in (
        (candidate, reference), (x_tri.grad, x_ref.grad),
        (w_tri.grad, w_ref.grad), (c_tri.grad, c_ref.grad),
    ):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)
    assert torch.equal(w_tri.grad[1], torch.zeros_like(w_tri.grad[1]))


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_triton_repeated_invocations_are_stream_local() -> None:
    pytest.importorskip("triton")
    from mirai.core.moe.runtime.routed_gemm_triton import routed_projection

    device = torch.device("cuda")
    generators = [
        torch.Generator(device=device).manual_seed(seed) for seed in (101, 202)
    ]
    problems = []
    for generator, counts in zip(generators, ((0, 3, 1, 5), (4, 0, 4, 1))):
        rows = sum(counts)
        boundaries = torch.tensor(
            counts, device=device, dtype=torch.int32
        ).cumsum(0)
        activation = torch.randn(
            rows, 31, device=device, dtype=torch.bfloat16, generator=generator
        )
        weight = torch.randn(
            len(counts), 31, 29,
            device=device, dtype=torch.bfloat16, generator=generator,
        )
        reference = routed_gemm_reference(
            activation, weight,
            RoutedGroupLayout(
                boundaries,
                torch.arange(rows, device=device, dtype=torch.int64),
                rows,
                1,
                len(counts),
            ),
        )
        problems.append((activation, weight, boundaries, reference))

    # Inputs and references are produced on the default stream before the two
    # independent streams begin consuming them.
    torch.cuda.synchronize(device)
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    outputs: list[list[torch.Tensor]] = [[], []]
    for stream_index, stream in enumerate(streams):
        activation, weight, boundaries, _ = problems[stream_index]
        with torch.cuda.stream(stream):
            for _ in range(4):
                outputs[stream_index].append(
                    routed_projection(activation, weight, boundaries)
                )

    for stream in streams:
        stream.synchronize()
    for stream_outputs, (_, _, _, reference) in zip(outputs, problems):
        for output in stream_outputs:
            torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_triton_successive_architecture_paths_survive_allocator_churn() -> None:
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("regular TMA contract requires Hopper")

    from mirai.core.moe.runtime.routed_gemm_triton import routed_projection
    from mirai.core.moe.runtime.specs import (
        MoEOptimizationPolicy,
        active_moe_optimization_policy,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(303)

    # Retain a fragmented set of allocations while both paths launch. The
    # bounded footprint exercises allocator reuse without depending on total
    # device capacity.
    pressure = [
        torch.empty(4 * 1024 * 1024, device=device, dtype=torch.uint8)
        for _ in range(32)
    ]
    pressure = pressure[1::2]

    regular_counts = (17, 0, 63, 9)
    regular_boundaries = torch.tensor(
        regular_counts, device=device, dtype=torch.int32
    ).cumsum(0, dtype=torch.int32)
    regular_x = torch.randn(
        sum(regular_counts), 96,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    regular_weight = torch.randn(
        4, 96, 80, device=device, dtype=torch.bfloat16, generator=generator
    )
    regular_reference = routed_gemm_reference(
        regular_x,
        regular_weight,
        RoutedGroupLayout(
            regular_boundaries,
            torch.arange(sum(regular_counts), device=device, dtype=torch.int64),
            sum(regular_counts),
            1,
            4,
        ),
    )
    with active_moe_optimization_policy(
        MoEOptimizationPolicy(moe_routed_gemm_architecture="tma_regular")
    ):
        regular_output = routed_projection(
            regular_x, regular_weight, regular_boundaries
        )

    indexed_counts = (0, 1, 6)
    indexed_boundaries = torch.tensor(
        indexed_counts, device=device, dtype=torch.int32
    ).cumsum(0, dtype=torch.int32)
    gather_rows = torch.tensor(
        [3, 0, 1, 3, 2, 0, 2], device=device, dtype=torch.int64
    )
    indexed_x = torch.randn(
        4, 17, device=device, dtype=torch.bfloat16, generator=generator
    )
    indexed_weight = torch.randn(
        3, 17, 23, device=device, dtype=torch.bfloat16, generator=generator
    )
    indexed_reference = routed_gemm_reference(
        indexed_x.index_select(0, gather_rows),
        indexed_weight,
        RoutedGroupLayout(
            indexed_boundaries,
            torch.arange(7, device=device, dtype=torch.int64),
            7,
            1,
            3,
        ),
    )
    with active_moe_optimization_policy(
        MoEOptimizationPolicy(moe_routed_gemm_architecture="indexed")
    ):
        indexed_output = routed_projection(
            indexed_x,
            indexed_weight,
            indexed_boundaries,
            gather_rows=gather_rows,
        )

    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        regular_output, regular_reference, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        indexed_output, indexed_reference, rtol=2e-2, atol=2e-2
    )
    assert len(pressure) == 16


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
@pytest.mark.parametrize("mapping", ("gather", "scatter"))
def test_tiled_indexed_projection_handles_skew_empty_groups_and_tails(
    mapping: str,
) -> None:
    pytest.importorskip("triton")
    from mirai.core.moe.runtime.routed_gemm_triton import routed_projection

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(909)
    counts = (0, 1, 0, 129, 1)
    rows = sum(counts)
    boundaries = torch.tensor(counts, device=device, dtype=torch.int32).cumsum(0)
    weight = torch.randn(
        len(counts), 73, 79,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    assignment_rows = torch.arange(rows, device=device, dtype=torch.int64)
    layout = RoutedGroupLayout(
        boundaries, assignment_rows, rows, 1, len(counts)
    )
    if mapping == "gather":
        activation = torch.randn(
            67, 73, device=device, dtype=torch.bfloat16, generator=generator
        )
        row_map = torch.arange(rows, device=device, dtype=torch.int64) % 67
        candidate = routed_projection(
            activation, weight, boundaries, gather_rows=row_map
        )
        reference = routed_gemm_reference(
            activation.index_select(0, row_map), weight, layout
        )
    else:
        activation = torch.randn(
            rows, 73, device=device, dtype=torch.bfloat16, generator=generator
        )
        row_map = torch.randperm(rows, device=device, generator=generator)
        candidate = routed_projection(
            activation, weight, boundaries, scatter_rows=row_map
        )
        grouped = routed_gemm_reference(activation, weight, layout)
        reference = torch.empty_like(grouped).index_copy(0, row_map, grouped)

    torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
@pytest.mark.parametrize("group_count", (1022, 1023, 1024))
def test_cuda_group_segmentation_matches_reference_around_backend_limit(
    group_count: int,
) -> None:
    from mirai.core.moe.runtime.gemm import grouped_mm_op, run_grouped_mm

    op = grouped_mm_op()
    if op is None:
        pytest.skip("CUDA grouped matrix multiplication is unavailable")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(400 + group_count)
    counts = torch.ones(group_count, device=device, dtype=torch.int32)
    if group_count >= 1024:
        # Repeated boundaries straddle the split while the final group ensures
        # that both segments participate in execution.
        counts[1021:1024] = 0
    offsets = counts.cumsum(0, dtype=torch.int32)
    rows = int(offsets[-1].item())
    activation = torch.randn(
        rows, 16, device=device, dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(
        group_count, 16, 16,
        device=device, dtype=torch.bfloat16, generator=generator,
    )

    candidate = run_grouped_mm(op, activation, weight, offsets)
    group_rows = torch.repeat_interleave(
        torch.arange(group_count, device=device), counts.to(torch.int64)
    )
    reference = torch.bmm(
        activation.unsqueeze(1), weight.index_select(0, group_rows)
    ).squeeze(1)
    torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=2e-2)
