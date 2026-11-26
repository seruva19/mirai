"""Behavioral contract for NF4-packed MAGI-2 Preview routed experts."""

from __future__ import annotations

import dataclasses
import gc
import weakref

import pytest
import torch
from torch.nn.utils import parametrize

from mirai.core.models.magi2_preview import grouped_moe
from mirai.core.models.magi2_preview import quantized_experts
from mirai.core.models.magi2_preview.grouped_moe import (
    _CONSUMED_POLICY_FIELDS,
    _MemoryBoundedSwiGlu7,
    _QUANTIZED_CONSUMED_POLICY_FIELDS,
    _memory_bounded_swiglu7,
    _memory_bounded_swiglu7_backward,
    Magi2GroupedMoEBackend,
    Magi2GroupedMoEPlan,
    Magi2GroupedMoEPolicyError,
    Magi2QuantizedGroupedMoEBackend,
    attach_grouped_moe_backend,
    magi2_grouped_mm_alignment_violations,
    resolve_magi2_moe_execution,
    run_segmented_grouped_expert_mlp,
    run_segmented_grouped_linear,
)
from mirai.core.models.magi2_preview.pipeline import LowRankWeight, Magi2PreviewPipeline
from mirai.core.models.magi2_preview.packed_experts import (
    load_magi2_nf4_packed_state,
    save_magi2_nf4_packed_state,
)
from mirai.core.models.magi2_preview.quantized_experts import (
    MAGI2_ROUTED_EXPERT_TENSOR_NAMES,
    Magi2Nf4ExpertStore,
    Magi2QuantizedExpertError,
    install_magi2_nf4_expert_stores,
    magi2_expert_store,
    quantize_magi2_experts_in_place,
)
from mirai.core.models.magi2_preview.triton_swiglu import (
    triton_swiglu7,
    triton_swiglu7_backward,
)
from mirai.core.moe.runtime.gemm import grouped_mm_stride_violations, grouped_mm_operand
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy


_CUDA = torch.cuda.is_available()


def _has_bitsandbytes() -> bool:
    try:
        import bitsandbytes.functional  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        return False
    return True


_NF4 = _CUDA and _has_bitsandbytes()
_requires_nf4 = pytest.mark.skipif(not _NF4, reason="NF4 storage requires CUDA and bitsandbytes")


# ---------------------------------------------------------------------------
# Segmented grouped execution (device-independent reference parity)
# ---------------------------------------------------------------------------


def test_memory_bounded_swiglu7_matches_reference_output_and_gradients() -> None:
    torch.manual_seed(2)
    gate = (torch.randn(4101, 32) * 4.0).to(torch.bfloat16).requires_grad_()
    up = (torch.randn(4101, 32) * 4.0).to(torch.bfloat16).requires_grad_()
    reference_gate = gate.detach().clone().requires_grad_()
    reference_up = up.detach().clone().requires_grad_()
    grad_output = torch.randn_like(gate)

    actual = _MemoryBoundedSwiGlu7.apply(gate, up, torch.bfloat16)
    gate_fp32 = reference_gate.float().clamp(max=7.0)
    up_fp32 = reference_up.float().clamp(min=-7.0, max=7.0)
    expected = (gate_fp32 * torch.sigmoid(1.702 * gate_fp32) * (up_fp32 + 1.0)).to(torch.bfloat16)
    actual_grads = torch.autograd.grad(actual, (gate, up), grad_output)
    expected_grads = torch.autograd.grad(expected, (reference_gate, reference_up), grad_output)

    assert torch.equal(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(actual_grad, expected_grad, atol=0.0625, rtol=0.01)


@pytest.mark.skipif(not _CUDA, reason="requires CUDA")
def test_triton_swiglu7_matches_reference_output_and_gradients() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(19)
    gate = (torch.randn(4101, 32, device="cuda") * 4.0).to(torch.bfloat16)
    up = (torch.randn_like(gate) * 4.0).to(torch.bfloat16)
    grad_output = torch.randn_like(gate)

    expected = _memory_bounded_swiglu7(gate, up, output_dtype=torch.bfloat16)
    actual = triton_swiglu7(gate, up, output_dtype=torch.bfloat16)
    expected_grads = _memory_bounded_swiglu7_backward(gate, up, grad_output)
    actual_grads = triton_swiglu7_backward(gate, up, grad_output)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-2, atol=2e-2)


def _sorted_layout(*, groups: int, rows_per_group: list[int], columns: int, dtype: torch.dtype):
    offsets = torch.tensor(list(torch.tensor(rows_per_group).cumsum(0).tolist()), dtype=torch.int32)
    total = int(offsets[-1])
    x_sorted = torch.randn(total, columns, dtype=dtype)
    return x_sorted, offsets, [int(value) for value in offsets.tolist()], groups


@pytest.mark.parametrize("span", [1, 2, 3, 5, 16])
def test_packed_segment_metadata_matches_full_host_layout(span: int) -> None:
    from mirai.core.moe.runtime.gemm import grouped_mm_segments

    counts = torch.tensor([0, 0, 5, 1, 0, 0, 0, 0, 2, 7, 0], dtype=torch.int32)
    offsets = counts.cumsum(0, dtype=torch.int32)
    reference = grouped_mm_segments(offsets.tolist(), max_groups=span)
    metadata = grouped_moe._packed_group_segments(offsets, max_groups=span, routed=True)
    assert tuple(segment for segment, _ in metadata) == reference
    assert [maximum for _, maximum in metadata] == [
        int(counts[segment.group_start : segment.group_stop].max())
        for segment in reference
    ]
    assert grouped_moe._packed_group_segments(offsets[:0], max_groups=span, routed=True) == ()


def test_segmented_backward_releases_weights_after_their_last_use(monkeypatch) -> None:
    weights = {
        "W_gate": torch.randn(1, 8, 16),
        "W_up": torch.randn(1, 8, 16),
        "W_down": torch.randn(1, 16, 8),
    }
    materialized = {}

    class Store:
        has_fused_gate_up = False

        def segment_group_span(self):
            return 1

        def materialize_segment(self, key, start, stop, *, dtype, device):
            weight = weights[key][start:stop].to(dtype=dtype, device=device).clone()
            materialized[key] = weakref.ref(weight)
            return weight

    linear_dx = grouped_moe._grouped_linear_dx

    def check_lifetimes(rows, weight, offsets, backend):
        if weight is materialized["W_gate"]():
            assert materialized["W_down"]() is None
        if weight is materialized["W_up"]():
            assert materialized["W_gate"]() is None
        return linear_dx(rows, weight, offsets, backend)

    monkeypatch.setattr(grouped_moe, "_grouped_linear_dx", check_lifetimes)
    x = torch.randn(3, 8, requires_grad=True)
    output = grouped_moe._SegmentedQuantizedExpertMlp.apply(
        x, torch.tensor([3], dtype=torch.int32), Store(), "bmm", "bmm", "torch"
    )
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(reference() is None for reference in materialized.values())


def test_segmented_grouped_linear_reproduces_the_unsegmented_result() -> None:
    """Segmenting the group axis is a pure execution schedule, not new math."""
    torch.manual_seed(3)
    groups = 7
    rows_per_group = [3, 0, 5, 1, 0, 4, 2]
    x_sorted, offsets, boundaries, _ = _sorted_layout(
        groups=groups, rows_per_group=rows_per_group, columns=6, dtype=torch.float32
    )
    weight = torch.randn(groups, 6, 5)

    def materialize(start: int, stop: int) -> torch.Tensor:
        return weight[start:stop].contiguous()

    whole = run_segmented_grouped_linear(
        x_sorted,
        offsets,
        boundaries=boundaries,
        materialize=materialize,
        backend="bmm",
        max_groups=groups,
        columns=5,
    )
    for span in (1, 2, 3, groups, groups + 4):
        chunked = run_segmented_grouped_linear(
            x_sorted,
            offsets,
            boundaries=boundaries,
            materialize=materialize,
            backend="bmm",
            max_groups=span,
            columns=5,
        )
        assert torch.equal(whole, chunked), span


def test_segmented_grouped_dx_reproduces_the_unsegmented_result() -> None:
    torch.manual_seed(5)
    groups = 5
    rows_per_group = [2, 4, 0, 3, 1]
    grad_output, offsets, boundaries, _ = _sorted_layout(
        groups=groups, rows_per_group=rows_per_group, columns=5, dtype=torch.float32
    )
    weight = torch.randn(groups, 6, 5)

    def materialize(start: int, stop: int) -> torch.Tensor:
        return weight[start:stop].contiguous()

    whole = run_segmented_grouped_linear(
        grad_output,
        offsets,
        boundaries=boundaries,
        materialize=materialize,
        backend="bmm",
        max_groups=groups,
        columns=6,
        transposed=True,
    )
    for span in (1, 2, groups):
        chunked = run_segmented_grouped_linear(
            grad_output,
            offsets,
            boundaries=boundaries,
            materialize=materialize,
            backend="bmm",
            max_groups=span,
            columns=6,
            transposed=True,
        )
        assert torch.equal(whole, chunked), span


def test_segmented_execution_visits_every_group_exactly_once() -> None:
    """Segments partition the group axis; no group is skipped or replayed."""
    torch.manual_seed(11)
    groups = 9
    rows_per_group = [1] * groups
    x_sorted, offsets, boundaries, _ = _sorted_layout(
        groups=groups, rows_per_group=rows_per_group, columns=4, dtype=torch.float32
    )
    weight = torch.randn(groups, 4, 4)
    visited: list[tuple[int, int]] = []

    def materialize(start: int, stop: int) -> torch.Tensor:
        visited.append((start, stop))
        segment = weight[start:stop].contiguous()
        # A materialized segment is a fresh contiguous buffer, so the grouped
        # stride precondition is decided on it exactly as on a stored tensor.
        assert segment.is_contiguous()
        return segment

    run_segmented_grouped_linear(
        x_sorted,
        offsets,
        boundaries=boundaries,
        materialize=materialize,
        backend="bmm",
        max_groups=4,
        columns=4,
    )
    assert visited == [(0, 4), (4, 8), (8, 9)]


def test_segmented_expert_mlp_matches_projection_wide_schedule() -> None:
    """Fusing the three projections by group range retains the released math."""

    torch.manual_seed(13)
    groups = 7
    rows_per_group = [3, 0, 5, 1, 0, 4, 2]
    x_sorted, offsets, boundaries, _ = _sorted_layout(
        groups=groups, rows_per_group=rows_per_group, columns=6, dtype=torch.float32
    )
    weights = {
        "W_gate": torch.randn(groups, 6, 10),
        "W_up": torch.randn(groups, 6, 10),
        "W_down": torch.randn(groups, 10, 6),
    }

    def project(key: str, values: torch.Tensor) -> torch.Tensor:
        return run_segmented_grouped_linear(
            values,
            offsets,
            boundaries=boundaries,
            materialize=lambda start, stop: weights[key][start:stop].contiguous(),
            backend="bmm",
            max_groups=3,
            columns=int(weights[key].shape[-1]),
        )

    gate = project("W_gate", x_sorted).float().clamp(max=7.0)
    up = project("W_up", x_sorted).float().clamp(min=-7.0, max=7.0)
    hidden = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    expected = project("W_down", hidden)
    visits: list[tuple[str, int, int]] = []

    def materialize(key: str, start: int, stop: int) -> torch.Tensor:
        visits.append((key, start, stop))
        return weights[key][start:stop].contiguous()

    actual = run_segmented_grouped_expert_mlp(
        x_sorted,
        offsets,
        boundaries=boundaries,
        materialize=materialize,
        backend="bmm",
        max_groups=3,
        output_columns=6,
    )
    assert torch.equal(expected, actual)
    assert visits == [
        (key, start, stop)
        for start, stop in ((0, 3), (3, 6), (6, 7))
        for key in ("W_gate", "W_up", "W_down")
    ]


def test_dequantized_segment_layout_satisfies_the_grouped_precondition() -> None:
    """A freshly dequantized segment buffer is an aligned grouped_mm operand."""
    segment = torch.empty(4, 256, 1280, dtype=torch.bfloat16, device="meta")
    assert segment.is_contiguous()
    assert (
        grouped_mm_stride_violations((grouped_mm_operand(segment, label="dequantized segment"),))
        == ()
    )


# ---------------------------------------------------------------------------
# Policy surface
# ---------------------------------------------------------------------------


def test_quantized_storage_adds_exactly_four_consumed_policy_fields() -> None:
    assert _QUANTIZED_CONSUMED_POLICY_FIELDS - _CONSUMED_POLICY_FIELDS == {
        "expert_weight_access",
        "expert_dequant_chunk_size",
        "quantize_experts_on_load",
        "moe_expert_autograd",
    }


# One non-default value per policy field that stays unconsumed once the routed
# experts are packed. The sweep is exhaustive by construction against the
# dataclass, so a field added to the shared policy fails closed here.
_QUANTIZED_NON_DEFAULT_POLICY_VALUES: dict[str, dict[str, object]] = {
    "expert_device_cache_gib": {"expert_device_cache_gib": 1.0},
    "expert_device_cache_policy": {"expert_device_cache_policy": "routing_frequency"},
    "expert_transfer_profile_path": {"expert_transfer_profile_path": "transfer_profile.json"},
    "device_residency_budget_gib": {"device_residency_budget_gib": 1.0},
    "router_quantization_calibration_path": {
        "router_quantization": "int8_per_channel",
        "router_quantization_calibration_path": "calibration.pt",
    },
    "packed_state_preload": {"packed_state_preload": "ram"},
    "packed_stream_cache_gib": {
        "packed_state_preload": "off",
        "packed_stream_cache_gib": 1.0,
    },
    "packed_stream_backend": {
        "packed_state_preload": "off",
        "packed_stream_backend": "gds",
    },
    "packed_stream_prefetch_depth": {"packed_stream_prefetch_depth": 2},
    "moe_dispatch": {"moe_dispatch": "legacy"},
    "moe_dispatch_preprocess": {"moe_dispatch_preprocess": "device"},
    "moe_gemm_backend_dw": {"moe_gemm_backend_dw": "bmm"},
    "moe_batched_dequant": {"moe_batched_dequant": False},
    "moe_pair_dequant": {"moe_pair_dequant": False},
    "moe_batched_gather": {"moe_batched_gather": True},
    "packed_shard_size_mb": {"packed_shard_size_mb": 512},
    "int8_workspace_mb": {"int8_workspace_mb": 64},
}


def test_quantized_policy_rejection_covers_every_unconsumed_field() -> None:
    declared = {field.name for field in dataclasses.fields(MoEOptimizationPolicy)}
    assert (
        set(_QUANTIZED_NON_DEFAULT_POLICY_VALUES) | set(_QUANTIZED_CONSUMED_POLICY_FIELDS)
        == declared
    )


@pytest.mark.parametrize("field_name", sorted(_QUANTIZED_NON_DEFAULT_POLICY_VALUES))
def test_quantized_policy_rejects_every_unconsumed_field(field_name: str) -> None:
    policy = MoEOptimizationPolicy(
        kernel_backend="grouped", **_QUANTIZED_NON_DEFAULT_POLICY_VALUES[field_name]
    )
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(policy, quantized_experts=True)


def test_magi2_grouped_plan_accepts_policy_for_a_bound_packed_stream() -> None:
    policy = MoEOptimizationPolicy(
        expert_weight_access="active_group_dequant", packed_state_preload="off",
        packed_stream_backend="io_uring", packed_stream_cache_gib=1.0,
        packed_stream_prefetch_depth=2,
    )
    assert resolve_magi2_moe_execution(policy, quantized_experts=True, packed_stream_bound=True)
    with pytest.raises(Magi2GroupedMoEPolicyError, match="packed_state_preload"):
        resolve_magi2_moe_execution(policy, quantized_experts=True)


def test_expert_access_keys_are_consumed_only_for_packed_experts() -> None:
    policy = MoEOptimizationPolicy(
        kernel_backend="grouped",
        expert_weight_access="chunked_dequant",
        expert_dequant_chunk_size=8,
        quantize_experts_on_load=True,
    )
    with pytest.raises(Magi2GroupedMoEPolicyError, match="expert_weight_access"):
        resolve_magi2_moe_execution(policy)
    plan = resolve_magi2_moe_execution(policy, quantized_experts=True)
    assert plan == Magi2GroupedMoEPlan(forward_backend="auto", dx_backend="auto")


def test_segmented_expert_autograd_is_explicit_and_packed_only() -> None:
    policy = MoEOptimizationPolicy(
        kernel_backend="grouped",
        moe_expert_autograd="segmented_recompute",
    )
    with pytest.raises(Magi2GroupedMoEPolicyError, match="moe_expert_autograd"):
        resolve_magi2_moe_execution(policy)
    assert resolve_magi2_moe_execution(policy, quantized_experts=True) == (
        Magi2GroupedMoEPlan(forward_backend="auto", dx_backend="auto")
    )
    with pytest.raises(ValueError, match="moe_expert_autograd"):
        MoEOptimizationPolicy(moe_expert_autograd="automatic")


def test_moe_activation_backend_is_explicit() -> None:
    policy = MoEOptimizationPolicy(
        kernel_backend="grouped",
        moe_activation_backend="triton",
    )
    assert resolve_magi2_moe_execution(policy) == Magi2GroupedMoEPlan(
        forward_backend="auto",
        dx_backend="auto",
    )
    with pytest.raises(ValueError, match="moe_activation_backend"):
        MoEOptimizationPolicy(moe_activation_backend="automatic")


def test_packed_experts_leave_no_reference_loop_to_fall_back_to() -> None:
    """``auto`` must resolve to grouped execution and ``torch`` must be rejected."""
    plan = resolve_magi2_moe_execution(MoEOptimizationPolicy(), quantized_experts=True)
    assert plan == Magi2GroupedMoEPlan(forward_backend="auto", dx_backend="auto")
    with pytest.raises(Magi2GroupedMoEPolicyError, match="reference loop"):
        resolve_magi2_moe_execution(
            MoEOptimizationPolicy(kernel_backend="torch"), quantized_experts=True
        )
    assert resolve_magi2_moe_execution(MoEOptimizationPolicy()) is None


def test_packed_experts_admit_default_off_routed_fusion_policy() -> None:
    assert resolve_magi2_moe_execution(
        MoEOptimizationPolicy(kernel_backend="grouped", moe_routed_gemm="triton"),
        quantized_experts=True,
    ) == Magi2GroupedMoEPlan(
        forward_backend="auto", dx_backend="auto", routed_gemm="triton"
    )
    assert resolve_magi2_moe_execution(
        MoEOptimizationPolicy(kernel_backend="grouped", moe_routed_gemm="auto"),
        quantized_experts=True,
    ) == Magi2GroupedMoEPlan(
        forward_backend="auto", dx_backend="auto", routed_gemm="auto"
    )


# ---------------------------------------------------------------------------
# Store contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expert_autograd", ["standard", "segmented_recompute"])
@pytest.mark.parametrize("training", [False, True])
def test_active_groups_decode_only_routed_weights_and_preserve_backward(
    monkeypatch, expert_autograd, training, tmp_path
) -> None:
    from mirai.core.models.magi2_preview import packed_experts

    generator = torch.Generator().manual_seed(29)
    store = Magi2Nf4ExpertStore(num_groups=5)
    store._meta = quantized_experts._Nf4Meta(
        blocksize=64, nested_blocksize=256,
        nested_dtype=torch.float32, weight_dtype=torch.float32,
    )
    decoded = []
    dense_weights = []

    def dequantize(fields, *args, **kwargs):
        group = int(fields["offset"])
        decoded.append(group)
        return dense_weights[int(fields["packed"].flatten()[0])][group]

    monkeypatch.setattr(quantized_experts, "_nf4_dequantize", dequantize)
    for index, (key, shape) in enumerate({"W_gate": (8, 16), "W_up": (8, 16), "W_down": (16, 8)}.items()):
        store._shapes[key] = (5, *shape)
        store._dtypes[key] = torch.float32
        dense_weights.append(torch.randn(5, *shape, generator=generator))
        store._set_buffer(f"{key}_nf4", torch.full((5, 64), index, dtype=torch.uint8))
        store._set_buffer(f"{key}_nf4_absmax", torch.zeros(5, 2, dtype=torch.uint8))
        store._set_buffer(f"{key}_nf4_nabsmax", torch.ones(5, 1))
        store._set_buffer(f"{key}_nf4_offset", torch.arange(5, dtype=torch.float32))
        store._set_buffer(f"{key}_nf4_code", torch.zeros(16))
        store._set_buffer(f"{key}_nf4_ncode", torch.zeros(256))

    offsets = torch.tensor([0, 2, 2, 2, 5], dtype=torch.int32)
    inputs = torch.randn(5, 8, generator=generator)
    backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(), expert_autograd=expert_autograd
    )
    monkeypatch.setattr(type(backend), "_store", lambda self, module: store)

    def run(access):
        store.set_expert_weight_access_policy(
            expert_weight_access=access, expert_dequant_chunk_size=3
        )
        with torch.set_grad_enabled(training):
            x = inputs.clone().requires_grad_(training)
            output = backend.branch_features(
                object(), x.unsqueeze(1), torch.arange(5), offsets, "bmm", "bmm"
            )
            if training:
                return output.detach(), torch.autograd.grad(output.square().sum(), x)[0]
            return (output,)

    expected = run("chunked_dequant")
    decoded.clear()
    actual = run("active_group_dequant")
    visits = 6 if training else 3
    assert sorted(decoded) == [1] * visits + [4] * visits
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference)

    source = store
    model = torch.nn.Module()
    model.local_num_heads, model.num_experts, model.local_flatten_num_experts = 1, 5, 5
    model.d_head, model.d_expert = 8, 16
    monkeypatch.setattr(packed_experts, "iter_magi2_moe_layers", lambda _: [("layer", model)])
    monkeypatch.setattr(packed_experts, "magi2_expert_store", lambda _: source)
    legacy_path = save_magi2_nf4_packed_state(tmp_path / "separate", model)
    store = load_magi2_nf4_packed_state(legacy_path, model, blocksize=64)["layer"]
    assert not store.has_fused_gate_up
    for observed, reference in zip(run("active_group_dequant"), expected):
        torch.testing.assert_close(observed, reference)

    path = save_magi2_nf4_packed_state(tmp_path / "packed", model, fuse_gate_up=True)
    store = load_magi2_nf4_packed_state(path, model, blocksize=64, range_read_backend="staged")["layer"]
    assert store.has_fused_gate_up
    assert all(name.endswith(("_nf4_code", "_nf4_ncode")) for name, _ in store.named_buffers())
    assert not hasattr(store, "W_gate_nf4")
    materialize = store.materialize_segment
    requested = []

    def record_materialization(key, *args, **kwargs):
        requested.append(key)
        return materialize(key, *args, **kwargs)

    monkeypatch.setattr(store, "materialize_segment", record_materialization)
    for observed, reference in zip(run("active_group_dequant"), expected):
        torch.testing.assert_close(observed, reference)
    assert requested[:2] == ["W_gate_up", "W_down"]
    passes = 2 if training else 1
    assert sorted(requested) == ["W_down"] * passes + ["W_gate_up"] * passes


def test_store_rejects_access_policies_it_has_no_operand_for() -> None:
    store = Magi2Nf4ExpertStore(num_groups=8)
    assert store.expert_weight_access == "full_dequant"
    assert store.segment_group_span() == 8
    store.set_expert_weight_access_policy(
        expert_weight_access="chunked_dequant", expert_dequant_chunk_size=3
    )
    assert store.segment_group_span() == 3
    for access in ("active_dequant", "fused_kernel"):
        with pytest.raises(Magi2QuantizedExpertError, match="expert_weight_access"):
            store.set_expert_weight_access_policy(expert_weight_access=access)
    with pytest.raises(Magi2QuantizedExpertError, match="chunk_size"):
        store.set_expert_weight_access_policy(
            expert_weight_access="chunked_dequant", expert_dequant_chunk_size=0
        )


def _reduced_moe(
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int = 256,
    expert_intermediate_size: int = 128,
    num_heads: int = 2,
    num_experts: int = 4,
    top_k: int = 2,
    routed_weight_device: str | None = None,
) -> torch.nn.Module:
    from mirai.vendors.magi2_preview.model.magi2_preview import (
        CoreMultiHeadMoE,
        CoreMultiHeadMoEConfig,
    )

    torch.manual_seed(0)
    module = CoreMultiHeadMoE(
        CoreMultiHeadMoEConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k,
            expert_intermediate_size=expert_intermediate_size,
            num_layers=1,
            params_dtype=dtype,
            score_func="sigmoid",
            route_norm=True,
            route_scale=4.9,
        ),
        routed_weight_device=routed_weight_device,
    )
    with torch.no_grad():
        for tensor in (module.gate, module.W_gate, module.W_up, module.W_down):
            tensor.normal_(std=0.1)
        module.router.expert_bias.normal_(std=0.05)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module if routed_weight_device == "meta" else module.to(device=device)


def _container(module: torch.nn.Module) -> torch.nn.Module:
    holder = torch.nn.Module()
    holder.moe_mlp = module
    container = torch.nn.Module()
    container.mlp = holder
    return container


def test_magi2_working_set_replays_optimizer_windows_and_preserves_eval_routes(tmp_path) -> None:
    from mirai.core.models.magi2_preview.working_set import Magi2ExpertWorkingSet

    experts = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32, num_experts=8)
    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.block = torch.nn.Module()
    pipeline.transformer.block.layers = torch.nn.ModuleList([_container(experts)])
    policy = Magi2ExpertWorkingSet(fraction=0.25, hold_steps=4, anneal_steps=8)
    pipeline._expert_working_set = policy
    pipeline._expert_working_set_window = None
    rows = torch.randn(5, 2, experts.d_head, requires_grad=True)
    baseline = experts._route(rows)

    assert pipeline.supports_router_subset_progress()
    pipeline.set_router_subset_progress(step=0, seed=13)
    first = experts._mirai_router_candidates
    assert first.shape == (2, 2)
    pipeline.set_router_subset_progress(step=3, seed=13)
    assert experts._mirai_router_candidates is first
    weights, routes = experts._route(rows)
    assert (routes[..., None] == first[:, None, None, :]).any(dim=-1).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.full((2, 5), experts.config.route_scale))
    assert torch.isfinite(torch.autograd.grad(weights.square().sum(), rows)[0]).all()

    source = _container(_reduced_moe(device=torch.device("cpu"), dtype=torch.float32, num_experts=8))
    quantize_magi2_experts_in_place(source, quant_format="fp8")
    artifact = save_magi2_nf4_packed_state(tmp_path / "packed", source)
    target = _container(_reduced_moe(device=torch.device("cpu"), dtype=torch.float32, num_experts=8))
    store = load_magi2_nf4_packed_state(
        artifact, target, blocksize=64, quant_format="fp8",
        range_read_backend="staged", stream_cache_bytes=1 << 20,
    )["mlp.moe_mlp"]
    reader = store._packed_range_reader.reader
    try:
        head_base = torch.arange(2)[:, None, None] * experts.num_experts
        groups = (routes + head_base).unique()
        options = dict(device="cpu", dtype=torch.float32)
        store.materialize_segment("W_gate", 0, len(groups), group_ids=groups, **options)
        first_read = reader.snapshot()["read_bytes"]
        assert first_read > 0
        _, next_routes = experts._route(-rows)
        groups = (next_routes + head_base).unique()
        store.materialize_segment("W_gate", 0, len(groups), group_ids=groups, **options)
        assert reader.snapshot()["read_bytes"] == first_read
    finally:
        reader.close()

    experts.eval()
    for actual, expected in zip(experts._route(rows), baseline):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    experts.train()
    pipeline.set_router_subset_progress(step=4, seed=13)
    second = experts._mirai_router_candidates
    assert second.shape == (2, 5)
    pipeline._expert_working_set_window = None
    pipeline.set_router_subset_progress(step=7, seed=13)
    torch.testing.assert_close(experts._mirai_router_candidates, second)
    assert not any("candidates" in name for name in experts.state_dict())
    pipeline.set_router_subset_progress(step=8, seed=13)
    assert experts._mirai_router_candidates is None


def test_installing_stores_removes_the_dense_expert_state() -> None:
    """The dense stack must be gone before any checkpoint tensor is read."""
    module = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    container = _container(module)
    before = set(container.state_dict())
    assert {"mlp.moe_mlp.W_gate", "mlp.moe_mlp.W_up", "mlp.moe_mlp.W_down"} <= before

    stores = install_magi2_nf4_expert_stores(container)
    assert set(stores) == {"mlp.moe_mlp"}
    assert magi2_expert_store(module) is stores["mlp.moe_mlp"]
    after = set(container.state_dict())
    for tensor_name in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
        assert f"mlp.moe_mlp.{tensor_name}" not in after
    # The router projection and its bias are untouched.
    assert "mlp.moe_mlp.gate" in after
    assert "mlp.moe_mlp.router.expert_bias" in after
    # NF4 payload buffers are derived at load time, not adapter state.
    assert not any(key.startswith("mlp.moe_mlp.mirai_nf4_experts") for key in after)
    # Installing twice is idempotent rather than a second, empty store.
    assert install_magi2_nf4_expert_stores(container)["mlp.moe_mlp"] is stores["mlp.moe_mlp"]


def test_expert_tensor_specs_describe_the_router_and_three_routed_tensors() -> None:
    module = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = _container(module)

    specs = {spec.name: spec for spec in pipeline.get_expert_tensor_specs()}
    assert set(specs) == {
        "mlp.moe_mlp.gate",
        "mlp.moe_mlp.W_gate",
        "mlp.moe_mlp.W_up",
        "mlp.moe_mlp.W_down",
    }
    router = specs["mlp.moe_mlp.gate"]
    assert router.router and not router.routed and router.quantizable
    assert router.adapter_targetable
    assert router.dtype == str(torch.float32)
    groups = int(module.local_flatten_num_experts)
    for name, role in (
        ("W_gate", "gate"),
        ("W_up", "up"),
        ("W_down", "down"),
    ):
        spec = specs[f"mlp.moe_mlp.{name}"]
        assert spec.role == role
        assert spec.routed and spec.quantizable and not spec.adapter_targetable
        assert spec.layout == ("expert", "out", "in")
        assert spec.shape[0] == groups
    assert specs["mlp.moe_mlp.W_gate"].shape == (groups, 128, 128)
    assert specs["mlp.moe_mlp.W_down"].shape == (groups, 128, 128)


def test_quantized_backend_refuses_a_layer_without_packed_experts() -> None:
    module = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    backend = Magi2QuantizedGroupedMoEBackend(Magi2GroupedMoEPlan())
    with pytest.raises(Magi2GroupedMoEPolicyError, match="packed NF4 experts"):
        backend.inspect_expert_layout(module)


# ---------------------------------------------------------------------------
# NF4 numerics (CUDA + bitsandbytes)
# ---------------------------------------------------------------------------


def _attach_router_lora(module: torch.nn.Module) -> LowRankWeight:
    torch.manual_seed(23)
    low_rank = LowRankWeight(tuple(module.gate.shape), rank=2, alpha=2.0).to(
        device=module.gate.device
    )
    parametrize.register_parametrization(module, "gate", low_rank)
    module.parametrizations["gate"].original.requires_grad_(False)
    adapter = module.parametrizations["gate"][0]
    with torch.no_grad():
        adapter.lora_b.normal_(std=0.1)
    return adapter


def _run(module: torch.nn.Module, adapter: LowRankWeight, hidden_size: int):
    torch.manual_seed(7)
    hidden = torch.randn(16, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = module._forward_impl(hidden)
    loss = output.float().square().mean()
    loss.backward()
    return (
        output.detach().float(),
        hidden.grad.detach().float().clone(),
        adapter.lora_a.grad.detach().float().clone(),
        adapter.lora_b.grad.detach().float().clone(),
    )


def _dense_from_store(store: Magi2Nf4ExpertStore, key: str) -> torch.Tensor:
    groups = store.num_groups
    return store.materialize_segment(
        key, 0, groups, dtype=torch.bfloat16, device=torch.device("cuda")
    )


@pytest.mark.parametrize("criterion", ["reap", "aimer", "diet"])
def test_magi2_pruning_remaps_each_head_and_preserves_top_k(tmp_path, criterion) -> None:
    from mirai.core.models.magi2_preview.expert_pruning import (
        Magi2ExpertPruningPlan, apply_magi2_expert_pruning, build_magi2_diet_pruning_plan,
        build_magi2_expert_pruning_plan,
    )
    from mirai.vendors.magi2_preview.model.magi2_preview import _mh_moe_compute_topk

    layer = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32, hidden_size=32,
                        expert_intermediate_size=16, num_heads=2, num_experts=4, top_k=2)
    model = _container(layer)
    name = "mlp.moe_mlp"
    scores = torch.tensor([[9., 3., 8., 1.], [3., 9., 1., 8.]])
    if criterion == "diet":
        plan = build_magi2_diet_pruning_plan(model, {name: ((0, 1, 2), (0, 1, 3))})
    else:
        plan = build_magi2_expert_pruning_plan(
            model, {name: scores}, criterion=criterion, keep_fraction=0.75,
        )
    expected_ids = (1, 2, 3, 4, 6, 7) if criterion == "aimer" else (0, 1, 2, 4, 5, 7)
    snapshot = {key: getattr(layer, key).clone() for key in ("gate", "W_gate", "W_up", "W_down")}
    bias = layer.router.expert_bias.clone()
    path = tmp_path / "pruning.json"
    plan.save(path)
    parsed = Magi2ExpertPruningPlan.load(path)
    assert parsed == plan
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._expert_pruning_digest = plan.topology_digest()
    metadata = {"__mirai_magi2_pruning_digest__": torch.tensor(
        list(parsed.topology_digest()), dtype=torch.uint8,
    )}
    pipeline.validate_adapter_payload({"adapter_state": metadata})
    with pytest.raises(ValueError, match="pruning topology"):
        pipeline.validate_adapter_payload({})
    pipeline._expert_pruning_digest = dataclasses.replace(
        plan, kept_by_head={name: ((0, 1, 3), (0, 1, 2))},
    ).topology_digest()
    with pytest.raises(ValueError, match="pruning topology"):
        pipeline.validate_adapter_payload(metadata)
    bad_plan = dataclasses.replace(parsed, kept_by_head={name: ((0,), (1,))})
    with pytest.raises(ValueError, match="below top-k"):
        apply_magi2_expert_pruning(model, bad_plan)
    torch.testing.assert_close(layer.gate, snapshot["gate"])
    apply_magi2_expert_pruning(model, parsed)
    ids = torch.tensor(expected_ids)
    for key, weight in snapshot.items():
        torch.testing.assert_close(getattr(layer, key), weight[ids])
    x = torch.randn(3, 2, 16)
    logits = torch.einsum("shd,hed->hse", x, snapshot["gate"][ids].view(2, 3, 16))
    probabilities, indices = _mh_moe_compute_topk(
        router_logits=logits, top_k=2, score_func=layer.config.score_func,
        expert_bias=bias[ids].view(2, 3), route_norm=layer.config.route_norm,
    )
    actual_probs, actual_ids = layer._route(x)
    torch.testing.assert_close(actual_ids, indices)
    torch.testing.assert_close(actual_probs, probabilities * layer.config.route_scale)
    assert layer.local_flatten_num_experts == 6 and layer.topk == 2
    hidden = x.reshape(3, 32).requires_grad_()
    layer._forward_impl(hidden).sum().backward()
    assert torch.isfinite(hidden.grad).all()


def test_aimer_pruning_ranks_dense_projections_within_each_head(tmp_path) -> None:
    from copy import deepcopy
    import json

    from safetensors.torch import save_file

    from mirai.core.models.magi2_preview.expert_pruning import (
        build_magi2_aimer_pruning_plan, build_magi2_aimer_pruning_plan_from_shards,
        verify_magi2_pruning_source,
    )

    layer = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    transformer = _container(layer)
    nonzeros = (1, 2, 4, 8, 8, 4, 2, 1)
    with torch.no_grad():
        for weight in (layer.W_gate, layer.W_up, layer.W_down):
            weight.zero_()
            for group, count in enumerate(nonzeros):
                weight[group].view(-1)[:count] = 1
        layer.router.expert_bias.fill_(-2)
        layer.router.expert_bias_ema.fill_(3)
    tensors = transformer.state_dict()
    save_file({key: value.double() for key, value in tensors.items() if key.endswith("W_up")},
              str(tmp_path / "a.safetensors"))
    save_file({key: value for key, value in tensors.items() if not key.endswith("W_up")},
              str(tmp_path / "b.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "a.safetensors" if key.endswith("W_up") else "b.safetensors" for key in tensors},
    }), encoding="utf-8")
    skeleton = deepcopy(transformer).to("meta")
    with torch.no_grad():
        layer.router.expert_bias.copy_(layer.router.expert_bias_ema)
    plan = build_magi2_aimer_pruning_plan(transformer, keep_fraction=0.5, max_block_elements=1)
    sharded = build_magi2_aimer_pruning_plan_from_shards(
        skeleton, tmp_path, keep_fraction=0.5, max_block_elements=1,
    )
    assert sharded == plan
    assert plan.criterion == "aimer"
    assert plan.kept_by_head == {"mlp.moe_mlp": ((0, 1), (2, 3))}
    assert verify_magi2_pruning_source(transformer, plan) == plan.topology_digest()
    with torch.no_grad():
        layer.W_down[0, 0, 0] = 2
    with pytest.raises(ValueError, match="fingerprint"):
        verify_magi2_pruning_source(transformer, plan)


def test_magi2_mixture_basis_preserves_packed_down_and_selected_group_order(tmp_path) -> None:
    from safetensors.torch import save_file

    from mirai.core.models.magi2_preview.basis_experts import (
        Magi2MixtureBasisExpertStore, stream_factorize_magi2_mixture_basis_experts,
    )
    from mirai.core.models.magi2_preview.quantized_experts import MAGI2_NF4_STORE_ATTR
    from mirai.core.models.compressed_weights.factorization.mixture_basis import (
        MixtureBasisBlockSource, factorize_mixture_basis_experts,
    )

    layer = _reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, hidden_size=16,
        expert_intermediate_size=16, num_heads=1, num_experts=2, top_k=1,
    )
    save_file(_container(layer).state_dict(), str(tmp_path / "model.safetensors"))
    gate = layer.W_gate.transpose(-1, -2)
    reads = []

    def read(groups, rows):
        reads.append((groups.stop - groups.start, rows.stop - rows.start))
        return gate[groups, rows]

    options = dict(rank=2, basis_count=2, optimization_steps=1, expert_batch_size=1, row_chunk_size=5)
    bounded = factorize_mixture_basis_experts(MixtureBasisBlockSource(tuple(gate.shape), read), **options)
    dense = factorize_mixture_basis_experts(gate, **options)
    assert reads and all(groups <= 1 and rows <= 5 for groups, rows in reads)
    torch.testing.assert_close(bounded.transforms, dense.transforms, rtol=0, atol=0)
    torch.testing.assert_close(bounded.bases, dense.bases, rtol=0, atol=0)
    skeleton = _container(_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, hidden_size=16,
        expert_intermediate_size=16, num_heads=1, num_experts=2, top_k=1,
        routed_weight_device="meta",
    ))
    streamed = stream_factorize_magi2_mixture_basis_experts(
        skeleton, checkpoint_dir=str(tmp_path), rank=2, basis_count=2,
        optimization_steps=1, down_format="fp8", blocksize=64,
    )["mlp.moe_mlp"]
    assert streamed.is_fully_loaded()
    assert not hasattr(skeleton.mlp.moe_mlp, "W_gate")
    store = Magi2MixtureBasisExpertStore(num_groups=2, down_format="fp8")
    for key in ("W_gate", "W_up", "W_down"):
        store.factorize_projection(
            key, getattr(layer, key), rank=2, basis_count=2, optimization_steps=1,
            checkpoint_interval=1, factor_dtype="float32",
        )
        del layer._parameters[key]
        kwargs = dict(device="cpu", dtype=torch.float32)
        whole = store.materialize_segment(key, 0, 2, **kwargs)
        selected = store.materialize_segment(key, 0, 3, group_ids=torch.tensor([1, 0, 1]), **kwargs)
        torch.testing.assert_close(selected, whole[[1, 0, 1]])
        if key == "W_down":
            torch.testing.assert_close(whole, store.down.materialize_segment(key, 0, 2, **kwargs), rtol=0, atol=0)
    assert store.is_fully_loaded()
    layer.add_module(MAGI2_NF4_STORE_ATTR, store)
    artifact = save_magi2_nf4_packed_state(tmp_path / "mixture", _container(layer))
    with pytest.raises(ValueError, match="Invalid MAGI-2 mixture-basis factors"):
        load_magi2_nf4_packed_state(
            artifact, skeleton, blocksize=64, quant_format="mixture_basis",
            expected_mixture_basis=(3, 2, "fp8"),
        )
    restored = load_magi2_nf4_packed_state(
        artifact, skeleton, blocksize=64, quant_format="mixture_basis",
        expected_mixture_basis=(2, 2, "fp8"),
    )["mlp.moe_mlp"]
    assert dict(restored.named_buffers()).keys() == dict(store.named_buffers()).keys()
    for name, value in store.named_buffers():
        torch.testing.assert_close(dict(restored.named_buffers())[name], value, rtol=0, atol=0)
    streamed_restore = load_magi2_nf4_packed_state(
        artifact, skeleton, blocksize=64, quant_format="mixture_basis",
        expected_mixture_basis=(2, 2, "fp8"), range_read_backend="staged", stream_prefetch_depth=2,
    )["mlp.moe_mlp"]
    assert set(dict(streamed_restore.named_buffers())) == {"W_gate_bases", "W_up_bases"}
    for key in ("W_gate", "W_up", "W_down"):
        options = dict(device="cpu", dtype=torch.float32, group_ids=torch.tensor([1, 0, 1]))
        streamed_restore.prefetch_segment(key, 0, 3, **options)
        torch.testing.assert_close(
            streamed_restore.materialize_segment(key, 0, 3, **options),
            restored.materialize_segment(key, 0, 3, **options), rtol=0, atol=0,
        )
    assert streamed_restore._packed_range_reader.snapshot()["prefetch_consumed"] == 6
    x = torch.randn(3, 1, 16, requires_grad=True)
    backend = Magi2QuantizedGroupedMoEBackend(Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm"))
    output = backend.execute(layer, x, *layer._route(x))
    torch.testing.assert_close(
        backend.execute(skeleton.mlp.moe_mlp, x, *layer._route(x)), output, rtol=0, atol=0,
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(torch.autograd.grad(output.sum(), x)[0]).all()


def test_magi2_shared_basis_reconstructs_selected_native_groups(tmp_path) -> None:
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from mirai.core.models.magi2_preview.basis_experts import (
        Magi2SharedBasisExpertStore, factorize_magi2_shared_basis_experts,
        fingerprint_magi2_basis_checkpoint,
        stream_factorize_magi2_shared_basis_experts,
    )

    generator = torch.Generator().manual_seed(7)
    store = Magi2SharedBasisExpertStore(num_groups=4)
    for key, shape in {"W_gate": (8, 16), "W_up": (8, 16), "W_down": (16, 8)}.items():
        if shape[0] < shape[1]:
            source = torch.randn(4, 8, 2, generator=generator) @ torch.randn(2, 16, generator=generator)
        else:
            source = torch.randn(16, 2, generator=generator) @ torch.randn(4, 2, 8, generator=generator)
        store.factorize_projection(key, source, rank=2, factor_dtype="float32")
        actual = store.materialize_segment(
            key, 1, 4, dtype=torch.float32, device="cpu", group_ids=torch.tensor([0, 3, 1, 3]),
        )
        torch.testing.assert_close(actual, source[[3, 1, 3]], rtol=2e-5, atol=1e-5)
    assert store.is_fully_loaded()
    assert store.payload_bytes() < 3 * 4 * 8 * 16 * 4
    blocks = []
    long_source = source.repeat(5, 1, 1)

    def read_block(start, stop):
        blocks.append((start, stop))
        return long_source[start:stop]

    bounded = Magi2SharedBasisExpertStore(num_groups=20)
    bounded.factorize_projection_blocks(
        "W_down", read_block, shape=tuple(long_source.shape), source_dtype=torch.float32,
        rank=2, factor_dtype="float32",
    )
    assert blocks == [(0, 8), (8, 16), (16, 20)] * 2
    torch.testing.assert_close(
        bounded.materialize_segment("W_down", 15, 18, dtype=torch.float32, device="cpu"),
        long_source[15:18], rtol=2e-5, atol=1e-5,
    )
    layer = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    transformer = _container(layer)
    save_file(transformer.state_dict(), str(tmp_path / "model.safetensors"))
    skeleton = _container(_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, routed_weight_device="meta",
    ))
    streamed = stream_factorize_magi2_shared_basis_experts(
        skeleton, checkpoint_dir=str(tmp_path), rank=2,
    )["mlp.moe_mlp"]
    installed = factorize_magi2_shared_basis_experts(transformer, rank=2)["mlp.moe_mlp"]
    assert installed.is_fully_loaded()
    assert not hasattr(skeleton.mlp.moe_mlp, "W_gate")
    for key, value in installed.named_buffers():
        torch.testing.assert_close(dict(streamed.named_buffers())[key], value, rtol=0, atol=0)
    artifact = save_magi2_nf4_packed_state(tmp_path / "basis", transformer)
    target = _container(_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, routed_weight_device="meta",
    ))
    with pytest.raises(ValueError, match="Invalid shared-basis factors"):
        load_magi2_nf4_packed_state(
            artifact, target, blocksize=64, quant_format="shared_basis", expected_shared_basis_rank=3,
        )
    assert target.mlp.moe_mlp.W_gate.is_meta
    loaded = load_magi2_nf4_packed_state(
        artifact, target, blocksize=64, quant_format="shared_basis", expected_shared_basis_rank=2,
    )["mlp.moe_mlp"]
    for key, value in installed.named_buffers():
        torch.testing.assert_close(dict(loaded.named_buffers())[key], value, rtol=0, atol=0)
    ranged = load_magi2_nf4_packed_state(
        artifact, target, blocksize=64, quant_format="shared_basis",
        expected_shared_basis_rank=2, range_read_backend="staged", stream_prefetch_depth=2,
    )["mlp.moe_mlp"]
    assert all(name.endswith("_basis") for name, _ in ranged.named_buffers())
    options = dict(device="cpu", dtype=torch.float32, group_ids=torch.tensor([7, 0, 7]))
    ranged.prefetch_segment("W_up", 0, 3, **options)
    torch.testing.assert_close(
        ranged.materialize_segment("W_up", 0, 3, **options),
        loaded.materialize_segment("W_up", 0, 3, **options), rtol=0, atol=0,
    )
    assert ranged._packed_range_reader.snapshot()["prefetch_consumed"] == 1
    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = transformer
    pipeline.model_config = SimpleNamespace(params=SimpleNamespace(variant="test"))
    pipeline._expert_shared_basis_rank = 2
    pipeline._expert_pruning_plan_path = ""
    pipeline._nf4_blocksize = 64
    pipeline._expert_rotation_group_size = 0
    pipeline._expert_basis_source_sha256 = fingerprint_magi2_basis_checkpoint(str(tmp_path))
    packed = pipeline.save_packed_frozen_weight_state(
        tmp_path / "bound-basis", metadata={"model_variant": "test", "denoiser_subfolder": "transformer"},
    )
    pipeline._packed_expert_state_path = str(packed)
    pipeline._restore_packed_experts(target, shared_basis_rank=2)
    from safetensors.torch import load_file

    checkpoint = tmp_path / "model.safetensors"
    weights = load_file(str(checkpoint))
    weights["mlp.moe_mlp.W_gate"] = weights["mlp.moe_mlp.W_gate"] + 1
    changed = tmp_path / "changed"
    changed.mkdir()
    save_file(weights, str(changed / "model.safetensors"))
    pipeline._expert_basis_source_sha256 = fingerprint_magi2_basis_checkpoint(str(changed))
    with pytest.raises(ValueError, match="magi2_basis_source_sha256"):
        pipeline._restore_packed_experts(target, shared_basis_rank=2)
    tokens = torch.rand(2, layer.num_heads, layer.d_head, requires_grad=True)
    backend = Magi2QuantizedGroupedMoEBackend(Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm"))
    output = backend.execute(layer, tokens, *layer._route(tokens))
    assert torch.isfinite(output).all()
    assert torch.isfinite(torch.autograd.grad(output.sum(), tokens)[0]).all()


def test_magi2_hierarchical_merge_keeps_logical_routing_and_compacts_each_head(tmp_path, monkeypatch) -> None:
    from copy import deepcopy
    from dataclasses import replace
    import json
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from mirai.core.models.magi2_preview.expert_pruning import (
        Magi2ExpertConsolidationPlan, apply_magi2_hierarchical_merge,
        build_magi2_hierarchical_merge_plan, observe_magi2_hierarchical_merge_head,
        build_magi2_hierarchical_merge_plan_from_evidence,
        stream_merge_magi2_experts,
    )
    from mirai.core.moe.calibration.prototypes import MeanExpertOutputAccumulator, PrototypeCalibrationEvidence
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewModelFamilyProvider
    from mirai.core.training.calibration import expert_prototypes

    layer = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    layer.router.expert_bias.copy_(layer.router.expert_bias_ema)
    reference = deepcopy(layer)
    transformer = _container(layer)
    evidence = []
    for positions in ((0.0, 0.1, 6.0, 6.1), (0.0, 6.0, 0.1, 6.1)):
        positions = torch.tensor(positions, dtype=torch.float64)
        evidence.append(PrototypeCalibrationEvidence(
            torch.ones(4), torch.tensor([1, 3, 2, 2]), (positions[:, None] - positions).abs(),
            distance_metric="mean_output_euclidean",
        ))
    plan = build_magi2_hierarchical_merge_plan(
        transformer, {"mlp.moe_mlp": tuple(evidence)}, reduction_ratio=0.5,
    )
    tokens = torch.randn(5, 2, layer.d_head)
    before = layer._route(tokens)
    for head in range(2):
        accumulator = MeanExpertOutputAccumulator(layer.num_experts)
        observe_magi2_hierarchical_merge_head(layer, head, tokens[:, head], accumulator)
        observation = accumulator.evidence()
        torch.testing.assert_close(observation.selected_count, before[1][head].flatten().bincount(minlength=4))
        assert observation.distance_metric == "mean_output_euclidean"
    provider = Magi2PreviewModelFamilyProvider()
    monkeypatch.setattr(expert_prototypes, "get_model_family_provider", lambda _: provider)
    monkeypatch.setattr(expert_prototypes, "resolve_step_sampling_context", lambda _: None)
    monkeypatch.setattr(expert_prototypes, "_build_training_batch_factory", lambda **_: lambda step: tokens.flatten(1))
    trainer = SimpleNamespace(
        pipeline=SimpleNamespace(transformer=transformer), begin_validation=lambda: None,
        end_validation=lambda _: None, compute_loss=lambda batch, **_: layer._forward_impl(batch),
    )
    session = SimpleNamespace(
        trainer=trainer, manifest=SimpleNamespace(dataset_snapshot_id="data", model_snapshot_id="base", config_snapshot_id="config"),
        config=SimpleNamespace(
            model=SimpleNamespace(type="magi2-preview", params=SimpleNamespace(expert_consolidation="hierarchical_output")),
            memory=SimpleNamespace(frozen_weight_packed_state_path=""),
        ),
    )
    evidence_path = tmp_path / "merge-evidence.safetensors"
    report = expert_prototypes.run_prototype_calibration_session(
        session, output_path=evidence_path, calibration_steps=1, max_block_elements=1,
        max_output_tokens_per_observation=3,
    )
    assert all(item["output_tokens_per_expert"] == 3 for item in report.modules.values())
    assert not hasattr(layer, "_mirai_prototype_callbacks")
    calibrated = build_magi2_hierarchical_merge_plan_from_evidence(transformer, evidence_path, reduction_ratio=0.5)
    assert calibrated.source_weight_fingerprint == plan.source_weight_fingerprint
    path = tmp_path / "merge.json"
    plan.save(path)
    restored = Magi2ExpertConsolidationPlan.load(path)
    assert restored == plan
    assert restored.topology_digest() == plan.topology_digest()
    save_file(transformer.state_dict(), str(tmp_path / "model.safetensors"))
    skeleton = _container(_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, routed_weight_device="meta",
    ))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        stream_merge_magi2_experts(skeleton, replace(plan, source_weight_fingerprint="sha256:wrong"), tmp_path)
    assert skeleton.mlp.moe_mlp.W_gate.is_meta
    preloaded, stores = stream_merge_magi2_experts(skeleton, restored, tmp_path)
    assert not stores
    assert preloaded == set(skeleton.state_dict())
    apply_magi2_hierarchical_merge(transformer, restored)
    assert layer.num_experts == 4
    assert layer._mirai_expert_aliases == (0, 0, 1, 1, 2, 3, 2, 3)
    for key, value in transformer.state_dict().items():
        torch.testing.assert_close(skeleton.state_dict()[key], value, rtol=0, atol=0)
    for current, prior in zip(layer._route(tokens), before, strict=True):
        torch.testing.assert_close(current, prior, rtol=0, atol=0)
    with torch.no_grad():
        for key in ("W_gate", "W_up", "W_down"):
            source = getattr(reference, key)
            compact = getattr(layer, key)
            torch.testing.assert_close(compact[0], 0.25 * source[0] + 0.75 * source[1])
            assert compact.shape[0] == 4
            source.copy_(compact[list(layer._mirai_expert_aliases)])
    torch.testing.assert_close(
        layer._forward_impl(tokens.flatten(1)), reference._forward_impl(tokens.flatten(1)),
        rtol=0, atol=0,
    )
    backend = Magi2GroupedMoEBackend(Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm"))
    x = tokens.detach().requires_grad_(True)
    probs, ids = layer._route(x)
    grouped = backend.execute(layer, x, probs, ids)
    expected = reference._forward_impl(x.flatten(1)).view_as(x)
    torch.testing.assert_close(grouped, expected, rtol=2e-5, atol=1e-5)
    actual_dx = torch.autograd.grad(grouped.square().sum(), x)[0]
    reference_probs, reference_ids = reference._route(x)
    grouped_reference = backend.execute(reference, x, reference_probs, reference_ids)
    expected_dx = torch.autograd.grad(grouped_reference.square().sum(), x)[0]
    torch.testing.assert_close(actual_dx, expected_dx, rtol=5e-5, atol=5e-5)
    branch_plan = backend.plan_branches(layer, x.detach(), probs.detach(), ids)
    assert branch_plan.groups == 4
    torch.testing.assert_close(
        branch_plan.combine_branch_features(branch_plan.compute_branch_features(None)),
        expected.detach(), rtol=2e-5, atol=1e-5,
    )
    duplicate_ids = torch.tensor([[0, 1], [0, 2]])[:, None, :].expand(2, tokens.shape[0], 2)
    coefficients = torch.rand(duplicate_ids.shape, requires_grad=True)
    visited_rows = []
    branch_features = backend.branch_features

    def record_branches(module, inputs, rows, offsets, forward_backend, dx_backend):
        visited_rows.append(rows.numel())
        return branch_features(module, inputs, rows, offsets, forward_backend, dx_backend)

    backend.branch_features = record_branches
    compact = backend.execute(layer, x, coefficients, duplicate_ids)
    expanded = backend.execute(reference, x, coefficients, duplicate_ids)
    assert visited_rows == [tokens.shape[0] * 2, tokens.shape[0] * 4]
    torch.testing.assert_close(compact, expanded, rtol=2e-5, atol=1e-5)
    for actual, wanted in zip(
        torch.autograd.grad(compact.sum(), (x, coefficients)),
        torch.autograd.grad(expanded.sum(), (x, coefficients)),
        strict=True,
    ):
        torch.testing.assert_close(actual, wanted, rtol=5e-5, atol=5e-5)
    from mirai.core.models.magi2_preview.quantized_experts import _new_expert_store

    packed = _container(_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, routed_weight_device="meta",
    ))
    preloaded, stores = stream_merge_magi2_experts(packed, restored, tmp_path, quant_format="fp8")
    packed_layer = packed.mlp.moe_mlp
    store = stores["mlp.moe_mlp"]
    assert set(packed.state_dict()) == preloaded
    assert store.num_groups == 4
    reference_store = _new_expert_store(num_groups=4, blocksize=64, quant_format="fp8")
    for key in ("W_gate", "W_up", "W_down"):
        reference_store.quantize_dense(key, getattr(layer, key))
        args = dict(device=torch.device("cpu"), dtype=torch.float32)
        weights = store.materialize_segment(key, 0, 4, **args)
        torch.testing.assert_close(weights, reference_store.materialize_segment(key, 0, 4, **args), rtol=0, atol=0)
        getattr(reference, key).data.copy_(weights[list(layer._mirai_expert_aliases)])
    packed_backend = Magi2QuantizedGroupedMoEBackend(backend.plan)
    packed_output = packed_backend.execute(packed_layer, tokens, *packed_layer._route(tokens))
    dense_output = backend.execute(reference, tokens, *reference._route(tokens))
    torch.testing.assert_close(packed_output, dense_output, rtol=2e-5, atol=1e-5)
    artifact = save_magi2_nf4_packed_state(
        tmp_path / "merged", packed, consolidation_digest=restored.topology_digest(),
    )
    target = _container(_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, routed_weight_device="meta",
    ))
    preloaded, unmaterialized = stream_merge_magi2_experts(
        target, restored, tmp_path, prepare_packed_restore=True,
    )
    assert not unmaterialized
    assert target.mlp.moe_mlp.W_down.is_meta
    with pytest.raises(ValueError, match="consolidation lineage"):
        load_magi2_nf4_packed_state(
            artifact, target, blocksize=64, quant_format="fp8", expected_consolidation_digest=b"0" * 32,
        )
    restored_store = load_magi2_nf4_packed_state(
        artifact, target, blocksize=64, quant_format="fp8",
        expected_consolidation_digest=restored.topology_digest(),
    )["mlp.moe_mlp"]
    assert set(target.state_dict()) == preloaded
    for key, value in store.named_buffers():
        torch.testing.assert_close(dict(restored_store.named_buffers())[key], value, rtol=0, atol=0)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layers"]["mlp.moe_mlp"]["logical_to_physical"][0] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible expert aliases"):
        load_magi2_nf4_packed_state(
            artifact, target, blocksize=64, quant_format="fp8",
            expected_consolidation_digest=restored.topology_digest(),
        )


def test_magi2_diet_head_replay_matches_native_expert_deletion() -> None:
    from copy import deepcopy

    from mirai.core.models.magi2_preview.expert_pruning import (
        apply_magi2_expert_pruning, build_magi2_diet_pruning_plan,
        capture_magi2_diet_head, select_magi2_diet_pruning_plan,
    )
    from mirai.core.moe.calibration.diet import (
        DietODLConfig, DietPairedCapture, derive_diet_signatures, replay_deletion_responses,
    )

    model = _container(_reduced_moe(device=torch.device("cpu"), dtype=torch.float32))
    layer = model.mlp.moe_mlp
    tokens = torch.randn(3, 2, layer.d_head)
    captures = tuple(capture_magi2_diet_head(layer, head, tokens[:, head]) for head in range(2))
    original = layer._forward_impl(tokens.flatten(1)).view_as(tokens)
    pruned = deepcopy(model)
    name = "mlp.moe_mlp"
    deletion = build_magi2_diet_pruning_plan(pruned, {name: ((0, 2, 3), (0, 1, 3))})
    apply_magi2_expert_pruning(pruned, deletion)
    observed = pruned.mlp.moe_mlp._forward_impl(tokens.flatten(1)).view_as(tokens) - original
    for head, deleted in enumerate((1, 2)):
        replay = replay_deletion_responses(captures[head])[deleted]
        torch.testing.assert_close(replay.float(), observed[:, head], rtol=1e-4, atol=1e-5)
    evidence = tuple(
        derive_diet_signatures(
            DietPairedCapture(
                capture, capture_magi2_diet_head(layer, head, -tokens[:, head]),
                pair_ids=torch.zeros(3, dtype=torch.long), token_positions=torch.arange(3),
            ),
            guidance_scale=3.0,
        )
        for head, capture in enumerate(captures)
    )
    plan = select_magi2_diet_pruning_plan(
        model, {name: evidence}, keep_fraction=0.5,
        config=DietODLConfig(random_restarts=0, anneal_steps=0),
    )
    assert plan.criterion == "diet"
    assert plan.source_weight_fingerprint == deletion.source_weight_fingerprint
    assert [len(ids) for ids in plan.kept_by_head[name]] == [2, 2]


def test_magi2_diet_observers_align_video_rows_across_cfg_branches(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from mirai.core.models.magi2_preview.expert_pruning import (
        build_magi2_diet_pruning_plan_from_evidence, capture_magi2_diet_head,
    )
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewModelFamilyProvider
    from mirai.core.moe.calibration.diet import DietPairedCapture, derive_diet_signatures, load_diet_evidence
    from mirai.core.training.calibration import diet
    from mirai.vendors.magi2_preview.model.magi2_preview import ModalityDispatcher, MultiHeadMoELayer

    layer = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    transformer = _container(layer)
    parent = transformer.mlp
    parent.pre_norm = lambda values, **_: values
    parent.split_linear = torch.nn.Identity()
    parent.merge_linear = torch.nn.Identity()
    parent._shared_expert_forward = lambda values, _: torch.zeros_like(values)
    provider = Magi2PreviewModelFamilyProvider()
    pipeline = SimpleNamespace(transformer=transformer, get_training_model=lambda: transformer)
    targets = provider.build_diet_calibration_targets(pipeline)
    video = torch.randn(2, layer.num_heads * layer.d_head)

    def generate(**kwargs):
        assert kwargs["cfg_mode"] == "paired"
        for sign, modalities in ((1, [0, 1, 0, 2]), (-1, [2, 0, 1, 1, 0])):
            mapping = torch.tensor(modalities)
            values = torch.randn(len(modalities), video.shape[-1])
            values[mapping == 0] = sign * video
            dispatcher = ModalityDispatcher(mapping, 3)
            MultiHeadMoELayer.forward(parent, dispatcher._permute(values), dispatcher)

    monkeypatch.setattr(diet, "get_model_family_provider", lambda _: provider)
    config = SimpleNamespace(
        memory=SimpleNamespace(),
        model=SimpleNamespace(type="magi2-preview", params=SimpleNamespace(
            expert_pruning="prune", expert_pruning_criterion="diet",
        )),
    )
    output = tmp_path / "diet.safetensors"
    diet.run_diet_calibration_session(
        SimpleNamespace(cfg=config, pipeline=pipeline, generate=generate),
        requests=[diet.DietCalibrationRequest("p", "n", 0, 1, 3.0, 1, 16, 16)],
        output_path=output, samples_per_layer=2, source_packed_state=None,
        calibration_manifest_id="manifest", config_snapshot_id="config",
    )
    evidence, lineage = load_diet_evidence(output)
    assert lineage.dense_source_sha256 == provider.diet_dense_source_fingerprint(pipeline)
    for head, name in enumerate(targets):
        values = video[:, head * layer.d_head:(head + 1) * layer.d_head]
        pair = DietPairedCapture(
            capture_magi2_diet_head(layer, head, values),
            capture_magi2_diet_head(layer, head, -values),
            pair_ids=torch.zeros(2, dtype=torch.long), token_positions=torch.arange(2),
        )
        expected = derive_diet_signatures(pair, guidance_scale=3.0)
        assert evidence[name].topology == expected.topology
        torch.testing.assert_close(evidence[name].signatures, expected.signatures.float())
    assert not hasattr(parent, "_mirai_diet_callbacks")
    plan = build_magi2_diet_pruning_plan_from_evidence(transformer, output, keep_fraction=0.5)
    assert plan.source_weight_fingerprint == f"sha256:{lineage.dense_source_sha256}"
    with torch.no_grad():
        layer.W_gate[0, 0, 0].add_(1)
    with pytest.raises(ValueError, match="different MAGI-2 dense source"):
        build_magi2_diet_pruning_plan_from_evidence(transformer, output, keep_fraction=0.5)


def test_pruning_calibration_keeps_head_local_route_ids(monkeypatch) -> None:
    from mirai.core.models.magi2_preview.expert_pruning import (
        build_magi2_expert_pruning_plan, build_magi2_pruning_calibration_targets,
    )
    from mirai.core.moe.calibration.pruning import (
        ExpertPruningRoutedOutputObserver, ExpertPruningSaliencyAccumulator,
    )

    layer = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    transformer = _container(layer)
    target, = build_magi2_pruning_calibration_targets(transformer).values()
    assert target.num_experts == 8
    tokens = torch.ones(2, layer.num_heads * layer.d_head)
    ids = torch.tensor([[[0, 1], [0, 1]], [[0, 2], [0, 2]]])
    weights = torch.tensor([[[0.25, 0.75], [0.5, 0.5]], [[0.2, 0.8], [0.9, 0.1]]])
    monkeypatch.setattr(layer, "_route", lambda _: (weights, ids))
    expected = layer._forward_impl(tokens)
    accumulator = ExpertPruningSaliencyAccumulator(target.num_experts, criterion="reap")
    target.host.set_expert_output_observer(ExpertPruningRoutedOutputObserver(accumulator))
    torch.testing.assert_close(layer._forward_impl(tokens), expected)
    evidence = accumulator.evidence()
    assert evidence.selected_count.tolist() == [2, 2, 0, 0, 2, 0, 2, 0]
    ones = torch.ones(layer.d_head)
    gate = (ones @ layer.W_gate).clamp(max=7.0)
    up = (ones @ layer.W_up).clamp(min=-7.0, max=7.0)
    hidden = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    norms = torch.bmm(hidden[:, None, :], layer.W_down).flatten(1).double().norm(dim=1)
    expected_weights = torch.tensor([0.75, 1.25, 0, 0, 1.1, 0, 0.9, 0], dtype=torch.float64)
    torch.testing.assert_close(evidence.score_sum, norms * expected_weights, rtol=2e-6, atol=1e-6)
    scores = evidence.scores()
    flat = build_magi2_expert_pruning_plan(transformer, {target.name: scores}, criterion="reap", keep_fraction=0.5)
    assert flat.kept_by_head[target.name] == ((0, 1), (0, 2))


def test_pruned_attention_adapter_restores_only_to_verified_source(tmp_path, monkeypatch) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from safetensors.torch import save_file
    from mirai.core.training.adapters import load_adapter_payload, normalize_adapter_state, save_kohya_adapter_safetensors
    from mirai.core.training.optim.ema import init_ema_state, update_ema_state
    from mirai.core.models.magi2_preview.expert_pruning import (
        apply_magi2_expert_pruning, build_magi2_expert_pruning_plan,
        stream_prune_magi2_experts, verify_magi2_pruning_source,
    )
    from mirai.vendors.magi2_preview.common import magi2_config
    from mirai.vendors.magi2_preview.model import magi2_preview as vendored_model
    from mirai.vendors.magi2_preview.model.magi2_preview import CustomGroupedLinear
    from mirai.vendors.magi2_preview.pipeline import preview_data_proxy

    def fixture():
        root = torch.nn.Module()
        root.layer = _container(_reduced_moe(device=torch.device("cpu"), dtype=torch.float32))
        root.layer.attention = torch.nn.Module()
        root.layer.attention.linear_proj = CustomGroupedLinear(16, 16, 1, 1, 1, dtype=torch.float32)
        with torch.no_grad():
            root.layer.attention.linear_proj.weight.normal_(std=0.1)
            router = root.layer.mlp.moe_mlp.router
            router.expert_bias_ema.copy_(router.expert_bias)
        pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = root
        return pipeline

    config = SimpleNamespace(type="lora", target_preset="attn_only", rank=2, alpha=2.0)
    source, target = fixture(), fixture()
    name = "layer.mlp.moe_mlp"
    plan = build_magi2_expert_pruning_plan(
        source.transformer, {name: torch.arange(8).view(2, 4)}, criterion="reap", keep_fraction=0.5,
    )
    save_file(source.transformer.state_dict(), str(tmp_path / "model.safetensors"))
    path = tmp_path / "pruning.json"
    plan.save(path)
    apply_magi2_expert_pruning(source.transformer, plan)
    source._expert_pruning_digest = plan.topology_digest()
    source.set_adapter_config(config)
    source_adapter = source.transformer.layer.attention.linear_proj.parametrizations.weight[0]
    with torch.no_grad():
        source_adapter.lora_b.fill_(0.25)
    payload = source.state_dict()
    target.set_adapter_config(config)
    with pytest.raises(ValueError, match="pruning topology"):
        target.load_adapter_state(payload)
    routed = target.transformer.layer.mlp.moe_mlp
    with torch.no_grad():
        routed.gate[0, 0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        verify_magi2_pruning_source(target.transformer, plan)
    target = fixture()
    config_stub = SimpleNamespace(
        arch_config=None, engine_config=SimpleNamespace(),
        evaluation_config=SimpleNamespace(data_proxy_config=None),
    )
    monkeypatch.setattr(magi2_config, "load_config", lambda _: config_stub)
    monkeypatch.setattr(vendored_model, "Transformer", lambda *args, **kwargs: target.transformer)
    monkeypatch.setattr(preview_data_proxy, "Magi2DataProxy", lambda _: None)
    target.options = SimpleNamespace(config_path="")
    target.model_config = SimpleNamespace(path=str(tmp_path), params=SimpleNamespace(variant="test"))
    streamed = fixture()
    streamed.options = target.options
    streamed.model_config = target.model_config
    streamed._expert_pruning_plan_path = str(path)
    streamed._quantize_experts_on_load = False
    streamed._packed_expert_state_path = ""
    streamed._configure_attention_backend = lambda _: None

    def build_skeleton(*args, **kwargs):
        assert kwargs["routed_weight_device"] == "meta"
        layer = streamed.transformer.layer.mlp.moe_mlp
        for key in ("W_gate", "W_up", "W_down"):
            setattr(layer, key, torch.nn.Parameter(getattr(layer, key).to("meta")))
        return streamed.transformer

    monkeypatch.setattr(vendored_model, "Transformer", build_skeleton)
    skeleton = build_skeleton(routed_weight_device="meta")
    with pytest.raises(ValueError, match="fingerprint"):
        stream_prune_magi2_experts(
            skeleton, replace(plan, source_weight_fingerprint="sha256:wrong"), tmp_path,
        )
    assert skeleton.layer.mlp.moe_mlp.W_gate.is_meta
    assert skeleton.layer.mlp.moe_mlp.num_experts == 4
    _, streamed.transformer, _ = streamed._build_model()
    assert streamed._expert_pruning_digest == plan.topology_digest()
    expected_routed = source.transformer.layer.mlp.moe_mlp
    actual_routed = streamed.transformer.layer.mlp.moe_mlp
    assert actual_routed.num_experts == expected_routed.num_experts == 2
    for key, value in expected_routed.state_dict().items():
        torch.testing.assert_close(actual_routed.state_dict()[key], value, rtol=0, atol=0)

    from mirai.core.models.magi2_preview.quantized_experts import _new_expert_store

    streamed = fixture()
    streamed.options = target.options
    streamed.model_config = target.model_config
    streamed._expert_pruning_plan_path = str(path)
    streamed._quantize_experts_on_load = True
    streamed._frozen_weight_quantization = "fp8"
    streamed._expert_precision_plan_path = ""
    streamed._packed_expert_state_path = ""
    streamed._configure_attention_backend = lambda _: None
    _, streamed.transformer, _ = streamed._build_model()
    packed_layer = streamed.transformer.layer.mlp.moe_mlp
    store = streamed._expert_stores[name]
    assert packed_layer.num_experts == 2
    assert not hasattr(packed_layer, "W_gate")
    assert streamed._expert_pruning_digest == plan.topology_digest()
    torch.testing.assert_close(packed_layer.gate, expected_routed.gate, rtol=0, atol=0)
    reference = _new_expert_store(num_groups=4, blocksize=64, quant_format="fp8")
    for key in ("W_gate", "W_up", "W_down"):
        reference.quantize_dense(key, getattr(expected_routed, key))
        kwargs = dict(device=torch.device("cpu"), dtype=torch.float32)
        torch.testing.assert_close(
            store.materialize_segment(key, 0, 4, **kwargs),
            reference.materialize_segment(key, 0, 4, **kwargs),
            rtol=0, atol=0,
        )

    from mirai.core.models.magi2_preview.packed_experts import (
        Magi2PackedExpertStateError, load_magi2_nf4_packed_state,
    )

    artifact = streamed.save_packed_frozen_weight_state(
        tmp_path / "packed", metadata={"model_variant": "test", "denoiser_subfolder": "transformer"},
    )
    with pytest.raises(Magi2PackedExpertStateError, match="pruning lineage"):
        load_magi2_nf4_packed_state(artifact, target.transformer, blocksize=64, quant_format="fp8")
    streamed = fixture()
    streamed.options = target.options
    streamed.model_config = target.model_config
    streamed._expert_pruning_plan_path = str(path)
    streamed._quantize_experts_on_load = False
    streamed._frozen_weight_quantization = "fp8"
    streamed._nf4_blocksize = 64
    streamed._expert_rotation_group_size = 0
    streamed._packed_expert_state_path = str(artifact)
    streamed._configure_attention_backend = lambda _: None
    _, streamed.transformer, _ = streamed._build_model()
    restored = streamed._expert_stores[name]
    assert streamed._expert_pruning_digest == plan.topology_digest()
    assert streamed.transformer.layer.mlp.moe_mlp.num_experts == 2
    for key, value in store.named_buffers():
        torch.testing.assert_close(dict(restored.named_buffers())[key], value, rtol=0, atol=0)
    wrong_survivors = replace(plan, kept_by_head={name: ((0, 1), (0, 1))})
    with pytest.raises(Magi2PackedExpertStateError, match="pruning lineage"):
        load_magi2_nf4_packed_state(
            artifact, streamed.transformer, blocksize=64, quant_format="fp8",
            expected_pruning_digest=wrong_survivors.topology_digest(),
        )

    monkeypatch.setattr(vendored_model, "Transformer", lambda *args, **kwargs: target.transformer)
    target._expert_pruning_adapter_source_plan_path = str(path)
    target._expert_pruning_plan_path = ""
    target._quantize_experts_on_load = False
    target._packed_expert_state_path = ""
    target._configure_attention_backend = lambda _: None
    _, target.transformer, _ = target._build_model()
    target.set_adapter_config(config)
    target.load_adapter_state(payload)
    routed = target.transformer.layer.mlp.moe_mlp
    assert routed.num_experts == 4
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, payload[key], rtol=0, atol=0)
    ema = update_ema_state(init_ema_state(payload), payload, decay=0.5)
    exported = save_kohya_adapter_safetensors(
        tmp_path / "adapter.safetensors", adapter_state=ema, rank=2, alpha=2.0,
        target_modules=["attention.linear_proj"], model_path=str(tmp_path),
    )
    target.load_adapter_state(normalize_adapter_state(load_adapter_payload(exported)))
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, payload[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="unexpected"):
        target.load_state_dict({**payload, f"{name}.gate": routed.gate}, strict=False)


@pytest.mark.parametrize("rotation_group_size", [0, 4])
def test_magi2_precision_calibration_uses_native_projection_layout(tmp_path, rotation_group_size, monkeypatch) -> None:
    from types import SimpleNamespace

    from mirai.core.models.magi2_preview.mixed_experts import (
        build_magi2_precision_calibration_targets,
        quantize_magi2_mixed_experts,
        stream_quantize_magi2_mixed_experts,
    )
    from mirai.core.moe.calibration.imatrix import ExpertImportanceAccumulator
    from mirai.core.moe.calibration.precision import allocate_tensor_precision
    from mirai.core.moe.calibration.router_repair import router_tensor_fingerprint
    from mirai.core.training.calibration.expert_precision import _measurement_evidence, _source_tensors

    layer = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32, hidden_size=32,
                        expert_intermediate_size=24, num_heads=2, num_experts=4, top_k=1)
    model = _container(layer)
    targets = build_magi2_precision_calibration_targets(
        model, blocksize=64, rotation_group_size=rotation_group_size,
    )
    name, target = next(iter(targets.items()))
    observer = ExpertImportanceAccumulator(
        num_experts=8, input_dims={"w1": 16, "w2": 24, "w3": 16},
    )
    layer.set_importance_calibration_observer(observer)
    try:
        inputs = torch.ones(2, 2, 16)
        layer._torch_forward(inputs, torch.ones(2, 2, 1), torch.zeros(2, 2, 1, dtype=torch.long))
    finally:
        layer.clear_importance_calibration_observer()
    observation = observer.evidence()
    assert observation.observation_counts["w2"].tolist() == [2, 0, 0, 0, 2, 0, 0, 0]
    assert _source_tensors(targets, include_router=True)[f"{name}.W_gate"] is layer.W_gate
    assert target.weights["w2"].shape == (8, 16, 24)
    rows = _measurement_evidence(targets, {name: observation}, formats=("fp8", "bf16"))
    assert len(rows) == 24
    assert any(row.routing_frequency == 0.0 and row.format_bytes["fp8"] > 0 for row in rows)
    plan = allocate_tensor_precision(
        rows, budget_bytes=sum(row.format_bytes["fp8"] for row in rows),
        allowed_formats=("fp8", "bf16"), dataset_snapshot_id="synthetic",
        model_snapshot_id="synthetic", config_snapshot_id="synthetic",
        source_weight_fingerprint=router_tensor_fingerprint(
            _source_tensors(targets, include_router=bool(rotation_group_size)),
        ),
    )
    path = plan.save(tmp_path / "plan.json")
    from safetensors.torch import save_file
    from mirai.core.models.magi2_preview.expert_pruning import (
        build_magi2_expert_pruning_plan, stream_prune_magi2_experts,
    )

    with torch.no_grad():
        layer.router.expert_bias.copy_(layer.router.expert_bias_ema)
    pruning_plan = build_magi2_expert_pruning_plan(
        model, {name: torch.arange(8)}, criterion="reap", keep_fraction=0.5,
    )
    checkpoint = {key: value.double() for key, value in model.state_dict().items()}
    save_file(checkpoint, str(tmp_path / "model.safetensors"))
    streamed_model = _container(_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, hidden_size=32,
        expert_intermediate_size=24, num_heads=2, num_experts=4, top_k=1,
        routed_weight_device="meta",
    ))
    if rotation_group_size:
        with pytest.raises(ValueError, match="fingerprint"):
            quantize_magi2_mixed_experts(model, plan_path=path, base_format="fp8", blocksize=64)
    store = quantize_magi2_mixed_experts(
        model, plan_path=path, base_format="fp8", blocksize=64,
        rotation_group_size=rotation_group_size,
    )[name]
    assert store.payload_bytes() == plan.estimated_bytes
    options = dict(
        checkpoint_dir=str(tmp_path), plan_path=path, base_format="fp8", blocksize=64,
        rotation_group_size=rotation_group_size,
    )
    tampered = f"{name}.gate" if rotation_group_size else f"{name}.W_gate"
    checkpoint[tampered].view(-1)[0] += 1
    save_file(checkpoint, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="fingerprint"):
        stream_quantize_magi2_mixed_experts(streamed_model, **options)
    assert hasattr(streamed_model.mlp.moe_mlp, "W_gate")
    checkpoint[tampered].view(-1)[0] -= 1
    save_file(checkpoint, str(tmp_path / "model.safetensors"))
    pruned_model = _container(_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, hidden_size=32,
        expert_intermediate_size=24, num_heads=2, num_experts=4, top_k=1,
        routed_weight_device="meta",
    ))
    pruned_stores, preloaded = stream_quantize_magi2_mixed_experts(
        pruned_model, **options, pruning_plan=pruning_plan,
    )
    selected = [2, 3, 6, 7]
    route = pruned_model.mlp.moe_mlp
    assert route.num_experts == 2
    assert preloaded == {f"{name}.gate", f"{name}.router.expert_bias", f"{name}.router.expert_bias_ema"}
    torch.testing.assert_close(route.gate, layer.gate[selected], rtol=0, atol=0)
    artifact = save_magi2_nf4_packed_state(
        tmp_path / "pruned-mixed", pruned_model, pruning_digest=pruning_plan.topology_digest(),
    )
    restore_model = _container(_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, hidden_size=32,
        expert_intermediate_size=24, num_heads=2, num_experts=4, top_k=1,
        routed_weight_device="meta",
    ))
    stream_prune_magi2_experts(restore_model, pruning_plan, tmp_path, prepare_packed_restore=True)
    restored = load_magi2_nf4_packed_state(
        artifact, restore_model, blocksize=64, quant_format="fp8",
        expected_pruning_digest=pruning_plan.topology_digest(),
        range_read_backend="staged",
        stream_prefetch_depth=2,
    )[name]
    assert restored.payload_bytes() == pruned_stores[name].payload_bytes()
    assert not list(restored.buffers())
    for key in ("W_gate", "W_up", "W_down"):
        kwargs = dict(device=torch.device("cpu"), dtype=torch.float32)
        original_rows = pruned_stores[name].materialize_segment(key, 0, 4, **kwargs)
        restored.prefetch_segment(key, 0, 4, **kwargs)
        torch.testing.assert_close(
            original_rows,
            store.materialize_segment(key, 0, 8, **kwargs)[selected], rtol=0, atol=0,
        )
        torch.testing.assert_close(restored.materialize_segment(key, 0, 4, **kwargs), original_rows, rtol=0, atol=0)
    assert restored._packed_range_reader.snapshot()["prefetch_consumed"] > 0
    from mirai.vendors.magi2_preview.common import magi2_config
    from mirai.vendors.magi2_preview.model import magi2_preview as vendored_model
    from mirai.vendors.magi2_preview.pipeline import preview_data_proxy

    config = SimpleNamespace(
        engine_config=SimpleNamespace(), arch_config=None,
        evaluation_config=SimpleNamespace(data_proxy_config=None),
    )
    monkeypatch.setattr(magi2_config, "load_config", lambda _: config)
    def build_transformer(_config, *, ep_size, routed_weight_device):
        assert ep_size == 1
        assert routed_weight_device == "meta"
        assert streamed_model.mlp.moe_mlp.W_gate.device.type == routed_weight_device
        return streamed_model

    monkeypatch.setattr(vendored_model, "Transformer", build_transformer)
    monkeypatch.setattr(preview_data_proxy, "Magi2DataProxy", lambda _: None)
    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.model_config = SimpleNamespace(path=str(tmp_path))
    pipeline.options = SimpleNamespace(config_path="")
    pipeline._quantize_experts_on_load = True
    pipeline._packed_expert_state_path = ""
    pipeline._expert_precision_plan_path = str(path)
    pipeline._expert_pruning_plan_path = ""
    pipeline._expert_rotation_group_size = rotation_group_size
    pipeline._nf4_blocksize = 64
    pipeline._frozen_weight_quantization = "fp8"
    pipeline._configure_attention_backend = lambda _: None
    pipeline._attach_moe_execution = lambda: None
    _, pipeline.transformer, _ = pipeline._build_model()
    streamed = pipeline._expert_stores[name]
    pipeline.enable_quantized_frozen_weights("fp8", precision_plan_path=str(path))
    assert pipeline._expert_stores[name] is streamed
    torch.testing.assert_close(pipeline.transformer.mlp.moe_mlp.gate, checkpoint[f"{name}.gate"].float())
    assert streamed.lineage == store.lineage
    for key in ("W_gate", "W_down", "W_up"):
        torch.testing.assert_close(
            streamed.materialize_segment(key, 0, 8, dtype=torch.float32, device="cpu"),
            store.materialize_segment(key, 0, 8, dtype=torch.float32, device="cpu"),
            rtol=0, atol=0,
        )
    if rotation_group_size:
        artifact = save_magi2_nf4_packed_state(tmp_path / "packed", model)
        restored_model = _container(_reduced_moe(
            device=torch.device("cpu"), dtype=torch.float32, hidden_size=32,
            expert_intermediate_size=24, num_heads=2, num_experts=4, top_k=1,
        ))
        restored = load_magi2_nf4_packed_state(artifact, restored_model, blocksize=64, quant_format="fp8")[name]
        assert restored.rotation_group_size == rotation_group_size
        kwargs = dict(dtype=torch.float32, device="cpu", group_ids=torch.tensor([4, 0, 4]))
        for key in ("W_gate", "W_down", "W_up"):
            torch.testing.assert_close(
                restored.materialize_segment(key, 0, 3, **kwargs),
                store.materialize_segment(key, 0, 3, **kwargs),
                rtol=0, atol=0,
            )
            normal = restored.materialize_segment(key, 0, 3, **kwargs)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                autocast = restored.materialize_segment(key, 0, 3, **kwargs)
            torch.testing.assert_close(autocast, normal, rtol=0, atol=0)
        gate, up, down = (
            store.materialize_segment(key, 0, 1, dtype=torch.float32, device="cpu")[0]
            for key in ("W_gate", "W_up", "W_down")
        )
        x = torch.randn(3, 16, requires_grad=True)
        gate_values = (x @ gate).clamp(max=7.0)
        up_values = (x @ up).clamp(min=-7.0, max=7.0)
        expected = (gate_values * torch.sigmoid(1.702 * gate_values) * (up_values + 1.0)) @ down
        actual = grouped_moe._SegmentedQuantizedExpertMlp.apply(
            x, torch.tensor([3], dtype=torch.int32), restored, "bmm", "bmm", "torch",
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), x)[0], torch.autograd.grad(expected.sum(), x)[0],
        )


@pytest.mark.parametrize("alternate_format", ["nvfp4", "gguf_iq1"])
def test_magi2_mixed_plan_preserves_group_order_and_checks_source_fingerprint(tmp_path, alternate_format) -> None:
    import json

    from mirai.core.models.magi2_preview.mixed_experts import quantize_magi2_mixed_experts
    from mirai.core.moe.calibration.precision import TensorPrecisionAssignment, TensorPrecisionPlan
    from mirai.core.moe.calibration.router_repair import router_tensor_fingerprint

    layer = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32, hidden_size=16,
                        expert_intermediate_size=16, num_heads=1, num_experts=4, top_k=1)
    model = _container(layer)
    module_name = "mlp.moe_mlp"
    weights = {key: getattr(layer, key).detach().clone() for key in ("W_gate", "W_up", "W_down")}
    assignments = tuple(
        TensorPrecisionAssignment(module_name, group, role, fmt, 1, 0.0)
        for group, fmt in enumerate(("fp8", "bf16", alternate_format, "fp8"))
        for role in ("w1", "w2", "w3")
    )
    plan = TensorPrecisionPlan(
        2, assignments, 12, 0.0, 12, "dataset", "model", "config", "wrong",
    )
    path = tmp_path / "precision.json"
    path.write_text(json.dumps(plan.to_dict()))
    with pytest.raises(ValueError, match="fingerprint"):
        quantize_magi2_mixed_experts(model, plan_path=path, base_format="fp8", blocksize=64)
    assert hasattr(layer, "W_gate")
    fingerprint = router_tensor_fingerprint({f"{module_name}.{key}": value for key, value in weights.items()})
    path.write_text(json.dumps(dataclasses.replace(plan, source_weight_fingerprint=fingerprint).to_dict()))
    store = quantize_magi2_mixed_experts(model, plan_path=path, base_format="fp8", blocksize=64)[module_name]
    kwargs = dict(dtype=torch.float32, device=torch.device("cpu"))
    whole = store.materialize_segment("W_gate", 0, 4, **kwargs)
    torch.testing.assert_close(whole[1], weights["W_gate"][1].bfloat16().float(), rtol=0, atol=0)
    from mirai.core.models.magi2_preview.quantized_experts import _new_expert_store

    reference = _new_expert_store(num_groups=2, blocksize=64, quant_format="fp8")
    reference.quantize_dense("W_gate", weights["W_gate"][[0, 3]])
    torch.testing.assert_close(
        whole[[0, 3]], reference.materialize_segment("W_gate", 0, 2, **kwargs), rtol=0, atol=0,
    )
    selection = torch.tensor([3, 0, 3])
    selected = store.materialize_segment("W_gate", 0, 3, group_ids=selection, **kwargs)
    torch.testing.assert_close(selected, whole[selection], rtol=0, atol=0)
    assert not hasattr(layer, "W_gate")
    artifact = save_magi2_nf4_packed_state(tmp_path / "packed", model)
    restored_model = _container(_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32, hidden_size=16,
        expert_intermediate_size=16, num_heads=1, num_experts=4, top_k=1,
    ))
    restored = load_magi2_nf4_packed_state(
        artifact, restored_model, blocksize=64, quant_format="fp8"
    )[module_name]
    assert restored.lineage == store.lineage
    for key in weights:
        expected = store.materialize_segment(key, 0, 3, group_ids=selection, **kwargs)
        actual = restored.materialize_segment(key, 0, 3, group_ids=selection, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["layers"][module_name]["formats"]["W_gate"][2] = "bf16"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bucket inventory"):
        load_magi2_nf4_packed_state(artifact, restored_model, blocksize=64, quant_format="fp8")


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_magi2_fp4_packs_contiguous_contraction_blocks(fmt) -> None:
    from mirai.core.models.compressed_weights.quantization.microscaling_quant import (
        dequantize_microscaling, quantize_microscaling,
    )

    weights = torch.randn(2, 32, 64)
    store = quantized_experts.Magi2PackedExpertStore(
        num_groups=2, quant_format=fmt, native_nvfp4_layout=fmt == "nvfp4",
    )
    store.quantize_dense("W_gate", weights)
    payload = quantize_microscaling(fmt, weights[1].t().contiguous())
    for actual, expected in zip((store.W_gate_codes[1], store.W_gate_scales[1],
                                store.W_gate_global_scale[1]), payload[:3], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected = dequantize_microscaling(*payload, dtype=torch.float32, device=torch.device("cpu")).t()
    torch.testing.assert_close(
        store.materialize_segment("W_gate", 1, 2, dtype=torch.float32, device="cpu")[0],
        expected, rtol=0, atol=0,
    )


@pytest.mark.parametrize("block_checkpoint", [False, True])
def test_magi2_int8_checkpoint_saves_only_expert_activations(monkeypatch, block_checkpoint) -> None:
    from torch.utils.checkpoint import checkpoint

    from mirai.core.training.residency import activation_compression

    module = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    quantize_magi2_experts_in_place(_container(module), quant_format="fp8")
    plan = Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    reference = Magi2QuantizedGroupedMoEBackend(plan, expert_autograd="segmented_recompute")
    compressed = Magi2QuantizedGroupedMoEBackend(
        plan, expert_autograd="segmented_recompute", int8_expert_checkpoint=True,
    )
    pack = activation_compression._pack_int8_saved_rows
    observed = []

    def inspect(tensor):
        result = pack(tensor)
        observed.append((tensor.dtype, tensor.numel() * tensor.element_size(), result))
        return result

    monkeypatch.setattr(activation_compression, "_pack_int8_saved_rows", inspect)
    rows = torch.randn(5, 2, module.d_head, requires_grad=True)
    def execute(backend, inputs):
        return backend.execute(module, inputs, *module._route(inputs))

    if block_checkpoint:
        expected = checkpoint(lambda inputs: execute(reference, inputs), rows, use_reentrant=False)
        actual = checkpoint(lambda inputs: execute(compressed, inputs), rows, use_reentrant=False)
    else:
        expected = execute(reference, rows)
        actual = execute(compressed, rows)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(observed) == 2
    dtype, original_bytes, packed = observed[0]
    assert dtype == rows.dtype
    assert packed.codes.dtype == torch.int8
    assert packed.codes.numel() + 4 * packed.scales.numel() < original_bytes
    assert not observed[1][2].is_floating_point()
    expected_dx = torch.autograd.grad(expected.square().sum(), rows, retain_graph=True)[0]
    actual_dx = torch.autograd.grad(actual.square().sum(), rows)[0]
    assert (actual_dx - expected_dx).norm() < 0.02 * expected_dx.norm()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not _CUDA, reason="requires CUDA"))])
def test_magi2_hybrid_experts_preserve_interleaved_rows_and_gradients(monkeypatch, device) -> None:
    module = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    store = quantize_magi2_experts_in_place(_container(module), quant_format="fp8")["mlp.moe_mlp"]
    store.set_expert_weight_access_policy(expert_weight_access="active_group_dequant", expert_dequant_chunk_size=1)
    rows = torch.randn(12, module.d_head, device=device, requires_grad=True)
    offsets = torch.tensor([1, 4, 4, 6, 10, 11, 11, 12], dtype=torch.int32, device=device)
    active_store, active_offsets = grouped_moe._packed_expert_layout(store, offsets)
    expected = grouped_moe._SegmentedQuantizedExpertMlp.apply(
        rows, active_offsets, active_store, "bmm", "bmm", "torch",
    )
    materialize = store.materialize_segment
    worker_groups = set()

    def trace(*args, group_ids, **kwargs):
        import threading

        if threading.current_thread().name.startswith("mirai-magi2-cold"):
            worker_groups.update(group_ids.tolist())
        return materialize(*args, group_ids=group_ids, **kwargs)

    monkeypatch.setattr(store, "materialize_segment", trace)
    actual = grouped_moe._hybrid_expert_features(
        rows, offsets, store, cpu_max_tokens=2,
        forward_backend="bmm", dx_backend="bmm", activation_backend="torch",
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert worker_groups == {0, 3, 5, 7}
    grad = torch.randn_like(actual)
    torch.testing.assert_close(
        torch.autograd.grad(actual, rows, grad)[0],
        torch.autograd.grad(expected, rows, grad)[0], rtol=1e-5, atol=1e-6,
    )


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_magi2_native_fp4_sm120_matches_decoded_operands(fmt) -> None:
    from mirai.core.models.compressed_weights.quantization.microscaling_quant import (
        dequantize_microscaling, quantize_microscaling,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires an SM120 CUDA device")
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(13)
    weights = torch.randn(2, 128, 64, generator=generator)
    store = quantized_experts.Magi2PackedExpertStore(
        num_groups=2, quant_format=fmt, native_nvfp4_layout=fmt == "nvfp4",
    )
    store.quantize_dense("W_gate", weights)
    x = torch.randn(9, 128, generator=generator).to(device=device, dtype=torch.bfloat16)
    actual = store.native_fp4_projection("W_gate", 1, x)
    decoded_x = dequantize_microscaling(
        *quantize_microscaling(fmt, x), dtype=torch.float32, device=device,
    )
    decoded_w = store.materialize_segment("W_gate", 1, 2, dtype=torch.float32, device=device)[0]
    expected = (decoded_x @ decoded_w).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.025)


@pytest.mark.parametrize("sparse_export,compact_runtime", [(False, False), (True, False), (True, True)])
def test_magi2_packed_expert_lora_composes_with_int8_segment_recompute(sparse_export, compact_runtime) -> None:
    from copy import deepcopy
    from types import SimpleNamespace

    def pipeline_with_adapter():
        module = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
        pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = _container(module)
        pipeline._expert_stores = quantize_magi2_experts_in_place(pipeline.transformer, quant_format="fp8")
        pipeline._compact_expert_lora = compact_runtime
        pipeline.set_adapter_config(SimpleNamespace(
            target_preset="attn_router_routed_experts", rank=2, alpha=2.0, sparse_expert_export=sparse_export,
            expert_selection="routing_topk" if compact_runtime else "all",
        ))
        return pipeline, module

    pipeline, module = pipeline_with_adapter()
    adapters = module._mirai_expert_lora
    with torch.no_grad():
        for adapter in adapters.values():
            adapter.lora_b.normal_(std=0.01)
            adapter.set_active_experts([1, 6])
    if compact_runtime:
        pipeline.prepare_compact_expert_lora()
    plan = Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    reference = Magi2QuantizedGroupedMoEBackend(plan, expert_autograd="segmented_recompute")
    compressed = Magi2QuantizedGroupedMoEBackend(plan, expert_autograd="segmented_recompute", int8_expert_checkpoint=True)
    rows = torch.randn(5, 2, module.d_head, requires_grad=True)
    scores = torch.ones(2, 5, 2) / 2
    routes = torch.tensor([[[1, 0]] * 5, [[2, 0]] * 5])

    expected = reference.execute(module, rows, scores, routes)
    actual = compressed.execute(module, rows, scores, routes)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    inputs = (rows, *(adapter.lora_b for adapter in adapters.values()))
    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
    for value, ref in zip(actual_grads, expected_grads, strict=True):
        assert (value - ref).norm() < 0.03 * ref.norm()
    for grad in actual_grads[1:]:
        if compact_runtime:
            assert grad.shape[0] == 2 and grad.abs().sum() > 0
        else:
            assert grad[[1, 6]].abs().sum() > 0
            assert torch.count_nonzero(grad[[0, 2, 3, 4, 5, 7]]) == 0

    restored, restored_module = pipeline_with_adapter()
    state = pipeline.state_dict()
    prefix = "mlp.moe_mlp._mirai_expert_lora.W_gate"
    if sparse_export:
        assert state[f"{prefix}.lora_a_selected"].shape[0] == 2
        assert f"{prefix}.lora_a" not in state
        invalid = dict(state)
        invalid[f"{prefix}.active_expert_ids"] = torch.tensor([1, 5])
        with pytest.raises(ValueError, match="Invalid selected expert IDs or mask"):
            restored.load_adapter_state(invalid)
    else:
        assert state[f"{prefix}.lora_a"].shape[0] == 8
    if compact_runtime:
        restored.prepare_compact_expert_lora(state)
        assert restored_module._mirai_expert_lora["W_gate"].lora_a.shape[0] == 2
    restored.load_adapter_state(state)
    assert restored_module._mirai_expert_lora["W_gate"].active_expert_ids() == [1, 6]
    torch.testing.assert_close(reference.execute(restored_module, rows, scores, routes), expected)
    if compact_runtime:
        params = [adapter.lora_b for adapter in adapters.values()]
        optimizer = torch.optim.AdamW(params)
        for param, grad in zip(params, actual_grads[1:], strict=True):
            param.grad = grad
        optimizer.step()
        state = pipeline.state_dict()
        restored.load_state_dict(state)
        restored_params = [adapter.lora_b for adapter in restored_module._mirai_expert_lora.values()]
        resumed = torch.optim.AdamW(restored_params)
        resumed.load_state_dict(deepcopy(optimizer.state_dict()))
        for param, restored_param in zip(params, restored_params, strict=True):
            assert resumed.state[restored_param]["exp_avg"].shape == param.shape
            restored_param.grad = param.grad.clone()
        optimizer.step()
        resumed.step()
        for param, restored_param in zip(params, restored_params, strict=True):
            torch.testing.assert_close(param, restored_param, rtol=0, atol=0)


def test_magi2_routing_topk_calibration_reaches_head_local_adapters(monkeypatch) -> None:
    from types import SimpleNamespace

    from mirai.core.moe.calibration.selection import select_hot_experts
    from mirai.core.training.calibration import expert_selection

    module = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = _container(module)
    pipeline._expert_stores = quantize_magi2_experts_in_place(pipeline.transformer, quant_format="fp8")
    config = SimpleNamespace(
        target_preset="attn_router_routed_experts", rank=2, alpha=2.0,
        expert_selection="routing_topk", expert_calibration_steps=1, expert_topk_fraction=0.25,
    )
    pipeline.set_adapter_config(config)
    inputs = torch.randn(11, 2, module.d_head)
    _, indices = module._route(inputs)
    flat = indices + torch.arange(2)[:, None, None] * module.num_experts
    expected = select_hot_experts(torch.bincount(flat.flatten(), minlength=8), 0.25)
    monkeypatch.setattr(expert_selection, "resolve_step_sampling_context", lambda _: None)
    monkeypatch.setattr(expert_selection, "_build_training_batch_factory", lambda **_: lambda _: inputs)
    trainer = SimpleNamespace(pipeline=pipeline, compute_loss=lambda batch, **_: module._route(batch))
    session = SimpleNamespace(config=SimpleNamespace(adapter=config), trainer=trainer, log_on_this_rank=False)

    report = expert_selection.maybe_calibrate_expert_selection(session)
    assert report.layers[0].routed_tokens == indices.numel()
    assert report.layers[0].selected_expert_ids == tuple(expected)
    assert report.expert_adapter_params_active < report.expert_adapter_params_total
    assert all(adapter.active_expert_ids() == expected for adapter in module._mirai_expert_lora.values())
    assert module._mirai_expert_selection_observer is None


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_magi2_native_fp4_dispatches_active_groups_and_preserves_input_gradient(monkeypatch, tmp_path, fmt) -> None:
    from mirai.core.models.compressed_weights.quantization import native_fp4

    def model():
        return _container(_reduced_moe(device=torch.device("cpu"), dtype=torch.float32))

    source = model()
    if fmt == "nvfp4":
        legacy = model()
        quantize_magi2_experts_in_place(legacy, quant_format=fmt)
        legacy_artifact = save_magi2_nf4_packed_state(tmp_path / "legacy", legacy)
        with pytest.raises(ValueError, match="native NVFP4 layout"):
            load_magi2_nf4_packed_state(
                legacy_artifact, model(), blocksize=64, quant_format=fmt, require_native_nvfp4_layout=True,
            )
        _write_expert_shard(tmp_path, source.mlp.moe_mlp)
        quantized_experts.install_magi2_nf4_expert_stores(source, quant_format=fmt, native_nvfp4_layout=True)
        quantized_experts.stream_quantize_magi2_experts(
            source, checkpoint_dir=str(tmp_path), quant_format=fmt, native_nvfp4_layout=True,
        )
    else:
        quantize_magi2_experts_in_place(source, quant_format=fmt)
    artifact = save_magi2_nf4_packed_state(tmp_path / "native", source)
    target = model()
    store = load_magi2_nf4_packed_state(
        artifact, target, blocksize=64, quant_format=fmt, range_read_backend="staged",
        require_native_nvfp4_layout=fmt == "nvfp4",
    )["mlp.moe_mlp"]
    module = target.mlp.moe_mlp
    store.set_expert_weight_access_policy(expert_weight_access="active_group_dequant", expert_dequant_chunk_size=1)
    native = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="scaled_mm_fp4", dx_backend="bmm"),
    )
    reference = Magi2QuantizedGroupedMoEBackend(Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm"))
    monkeypatch.setattr(native, "_resolve", lambda *_: ("scaled_mm_fp4", "bmm"))
    calls = []

    def emulate(x, codes, scales, global_scale, meta):
        calls.append(meta.shape)
        weight = native_fp4.dequantize_microscaling(
            codes, scales, global_scale, meta, dtype=x.dtype, device=x.device,
        )
        return x @ weight.t()

    monkeypatch.setattr(native_fp4, "native_fp4_linear", emulate)
    rows = torch.randn(3, 2, module.d_head, requires_grad=True)
    groups = torch.tensor([[[3, 1]] * 3, [[2, 0]] * 3])
    scores = torch.rand(2, 3, 2, requires_grad=True)
    expected = reference.execute(module, rows, scores, groups)
    with torch.no_grad():
        output = native.execute(module, rows, scores, groups)
    torch.testing.assert_close(output, expected)
    assert len(calls) == 12
    actual = native.execute(module, rows, scores, groups)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), (rows, scores)),
        torch.autograd.grad(expected.sum(), (rows, scores)),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("fmt", ["gguf_iq1", "gguf_iq2", "gguf_iq3", "fp8", "nvfp4", "mxfp4"])
def test_magi2_additional_packed_formats_round_trip_and_execute(tmp_path, fmt) -> None:
    def model():
        return _container(_reduced_moe(
            device=torch.device("cpu"), dtype=torch.float32, hidden_size=16,
            expert_intermediate_size=16, num_heads=1, num_experts=2, top_k=1,
        ))

    source = model()
    store = quantize_magi2_experts_in_place(source, quant_format=fmt)["mlp.moe_mlp"]
    artifact = save_magi2_nf4_packed_state(tmp_path / fmt, source)
    restored = load_magi2_nf4_packed_state(
        artifact, model(), blocksize=64, quant_format=fmt
    )["mlp.moe_mlp"]
    assert restored.quant_format == fmt
    for key in ("W_gate", "W_up", "W_down"):
        kwargs = dict(dtype=torch.float32, device=torch.device("cpu"))
        whole = store.materialize_segment(key, 0, 2, **kwargs)
        actual = restored.materialize_segment(key, 0, 1, group_ids=torch.tensor([1]), **kwargs)
        torch.testing.assert_close(actual, whole[1:], rtol=0, atol=0)
    x = torch.randn(3, 16, requires_grad=True)
    offsets = torch.tensor([0, 3], dtype=torch.int32)
    actual = grouped_moe._SegmentedQuantizedExpertMlp.apply(
        x, offsets, restored, "bmm", "bmm", "torch"
    )
    expected = grouped_moe._SegmentedQuantizedExpertMlp.apply(
        x, offsets, store, "bmm", "bmm", "torch"
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(expected.sum(), x)[0], rtol=0, atol=0)
    with pytest.raises(ValueError, match="does not match requested format"):
        load_magi2_nf4_packed_state(artifact, model(), blocksize=64)


def test_magi2_staged_layers_share_one_packed_reader(tmp_path, monkeypatch) -> None:
    def model():
        root = _container(_reduced_moe(device=torch.device("cpu"), dtype=torch.float32))
        root.second = _container(_reduced_moe(device=torch.device("cpu"), dtype=torch.float32))
        return root

    source = model()
    with torch.no_grad():
        source.second.mlp.moe_mlp.W_gate.mul_(2)
    resident = quantize_magi2_experts_in_place(source, quant_format="fp8")
    artifact = save_magi2_nf4_packed_state(tmp_path / "packed", source)
    ranged = load_magi2_nf4_packed_state(
        artifact, model(), blocksize=64, quant_format="fp8", range_read_backend="staged",
        stream_cache_bytes=1024, stream_prefetch_depth=2,
    )
    assert len({id(store._packed_range_reader.reader) for store in ranged.values()}) == 1
    for name, store in ranged.items():
        options = dict(dtype=torch.float32, device="cpu")
        torch.testing.assert_close(
            store.materialize_segment("W_gate", 0, 1, **options),
            resident[name].materialize_segment("W_gate", 0, 1, **options), rtol=0, atol=0,
        )
    name = "mlp.moe_mlp"
    active = ranged[name]
    reference = resident[name]
    for store in (active, reference):
        store.set_expert_weight_access_policy(expert_weight_access="chunked_dequant", expert_dequant_chunk_size=1)
    x = torch.randn(3, active.expert_weight_shape("W_gate")[1], requires_grad=True)
    offsets = torch.tensor([1, 2, 3] + [3] * (active.num_groups - 3), dtype=torch.int32)
    actual = grouped_moe._QuantizedGroupedExpertLinear.apply(x, offsets, active, "W_gate", "bmm", "bmm")
    expected = grouped_moe._QuantizedGroupedExpertLinear.apply(x, offsets, reference, "W_gate", "bmm", "bmm")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0], torch.autograd.grad(expected.sum(), x)[0], rtol=0, atol=0,
    )
    snapshot = active._packed_range_reader.reader.snapshot()
    assert snapshot["prefetch_consumed"] >= 4
    assert snapshot["prefetch_peak_queued"] <= 2
    before = snapshot["prefetch_consumed"]
    actual = grouped_moe._SegmentedQuantizedExpertMlp.apply(x, offsets, active, "bmm", "bmm", "torch")
    expected = grouped_moe._SegmentedQuantizedExpertMlp.apply(x, offsets, reference, "bmm", "bmm", "torch")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0], torch.autograd.grad(expected.sum(), x)[0], rtol=0, atol=0,
    )
    snapshot = active._packed_range_reader.reader.snapshot()
    assert snapshot["prefetch_consumed"] - before > 8
    assert snapshot["prefetch_queued"] == 0
    before = snapshot["prefetch_consumed"]

    def routed_reference(rows, weights, boundaries, *, gather_rows=None, max_group_rows=None):
        if gather_rows is not None:
            rows = rows.index_select(0, gather_rows)
        return grouped_moe._grouped_linear(rows, weights, boundaries, "bmm")

    monkeypatch.setattr(grouped_moe, "_segment_routed_projection", routed_reference)
    sorted_rows = torch.tensor([2, 0, 2])
    actual = grouped_moe._SegmentedRoutedQuantizedExpertMlp.apply(x, offsets, sorted_rows, active, "torch")
    expected = grouped_moe._SegmentedRoutedQuantizedExpertMlp.apply(x, offsets, sorted_rows, reference, "torch")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0], torch.autograd.grad(expected.sum(), x)[0], rtol=0, atol=0,
    )
    snapshot = active._packed_range_reader.reader.snapshot()
    assert snapshot["prefetch_consumed"] - before > 8
    assert snapshot["prefetch_queued"] == 0
    from mirai.core.training.residency import tensor_residency

    events = []

    class DummyPool:
        def __init__(self, budget):
            events.append("allocate")
            self.pinned_bytes = budget

        def copy_to_device(self, host_tensors, *, device, stream):
            events.append("copy")
            return [tensor.clone().to(device) for tensor in host_tensors]

        def close(self):
            events.append("close")

    monkeypatch.setattr(tensor_residency, "PinnedHostStagingPool", DummyPool)
    reader = active._packed_range_reader.reader
    key = "mlp.moe_mlp:W_gate_scales"
    reader.get_slices_to_device(key, [3, 4], device="cpu")
    before = reader.snapshot()
    overlapping = reader.get_slices_to_device(key, [4, 5, 4], device="cpu")
    after = reader.snapshot()
    assert after["read_bytes"] - before["read_bytes"] == reader.tensor_nbytes(key) // active.num_groups
    assert after["cache_hits"] - before["cache_hits"] == 1
    assert after["cache_peak_bytes"] <= 1024
    torch.testing.assert_close(overlapping[0], overlapping[2])
    reader._pin_cap_bytes = 1024
    for _ in range(2):
        output, = reader._copy_staged_to_device([torch.ones(2)], "cpu", torch.float64, None)
        torch.testing.assert_close(output, torch.ones(2, dtype=torch.float64))
    assert reader.snapshot()["pinned_peak_bytes"] == 1024
    reader.close()
    assert reader.snapshot()["pinned_in_flight_bytes"] == 0
    assert events == ["allocate", "copy", "copy", "close"]


@pytest.mark.parametrize("backend", ["direct_io", "io_uring"])
def test_linux_direct_packed_reads_trim_aligned_windows(tmp_path, backend) -> None:
    import errno
    import os
    import sys

    from safetensors.torch import save_file

    from mirai.core.models.compressed_weights.packed.packed_storage_alignment import (
        read_safetensors_storage_alignment, save_safetensors_with_storage_alignment,
    )
    from mirai.core.models.compressed_weights.packed.packed_stream_io import DirectPackedShardReader, IoUringPackedShardReader

    if sys.platform != "linux":
        pytest.skip("O_DIRECT requires Linux")
    if backend == "io_uring" and not os.environ.get("MIRAI_PACKED_IO_URING_LIBRARY"):
        pytest.skip("io_uring bridge is unavailable")
    payload = torch.arange(6144).to(torch.uint8)
    path = tmp_path / "direct.safetensors"
    save_safetensors_with_storage_alignment(
        save_file, {"weights": payload}, path, metadata=None, alignment_bytes=4096,
    )
    size = read_safetensors_storage_alignment(path).file_size
    payload_offset = size - payload.numel()
    try:
        reader_type = IoUringPackedShardReader if backend == "io_uring" else DirectPackedShardReader
        reader = reader_type(path)
        try:
            target = bytearray(200)
            physical_bytes = reader.readinto(memoryview(target), payload_offset + 1990)
        finally:
            reader.close()
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.EOPNOTSUPP}:
            pytest.skip("Temporary filesystem does not support O_DIRECT")
        raise
    assert physical_bytes == 8192
    assert target == payload[1990:2190].numpy().tobytes()


def test_packed_restore_rejects_head_layout_and_projection_mismatches(tmp_path, monkeypatch) -> None:
    import json
    from contextlib import contextmanager

    import safetensors

    def model(*, heads=2, experts=2):
        return _container(_reduced_moe(
            device=torch.device("cpu"), dtype=torch.float32, hidden_size=16 * heads,
            expert_intermediate_size=16, num_heads=heads, num_experts=experts, top_k=1,
        ))

    source = model()
    quantize_magi2_experts_in_place(source, quant_format="fp8")
    artifact = save_magi2_nf4_packed_state(tmp_path / "packed", source, storage_alignment_bytes=4096)
    shard = artifact / "layer_000.safetensors"
    assert shard.stat().st_size % 4096 == 0
    references = {}
    safe_open = safetensors.safe_open

    @contextmanager
    def record_open(*args, **kwargs):
        with safe_open(*args, **kwargs) as handle:
            class RecordingHandle:
                def keys(self):
                    return handle.keys()

                def get_tensor(self, key):
                    value = handle.get_tensor(key)
                    references[key] = value
                    return value

            yield RecordingHandle()

    monkeypatch.setattr(safetensors, "safe_open", record_open)
    mapped = load_magi2_nf4_packed_state(
        artifact, model(), blocksize=64, quant_format="fp8", retain_mapped_buffers=True,
    )["mlp.moe_mlp"]
    assert all(value.data_ptr() == references[name].data_ptr() for name, value in mapped.named_buffers())
    expected = mapped.materialize_segment("W_gate", 1, 3, dtype=torch.float32, device="cpu")
    resident = load_magi2_nf4_packed_state(artifact, model(), blocksize=64, quant_format="fp8")["mlp.moe_mlp"]
    assert all(value.data_ptr() != references[name].data_ptr() for name, value in resident.named_buffers())
    torch.testing.assert_close(
        resident.materialize_segment("W_gate", 1, 3, dtype=torch.float32, device="cpu"),
        expected, rtol=0, atol=0,
    )
    ranged = load_magi2_nf4_packed_state(
        artifact, model(), blocksize=64, quant_format="fp8", range_read_backend="staged",
    )["mlp.moe_mlp"]
    assert not list(ranged.buffers())
    assert ranged.payload_bytes() == resident.payload_bytes()
    torch.testing.assert_close(
        ranged.materialize_segment(
            "W_gate", 0, 3, dtype=torch.float32, device="cpu", group_ids=torch.tensor([2, 1, 2]),
        ),
        expected[[1, 0, 1]], rtol=0, atol=0,
    )
    from mirai.core.models.compressed_weights.packed import packed_stream_gds

    gds_shards = []

    class FakeGdsReader:
        def __init__(self, shard_paths):
            gds_shards.extend(shard_paths.values())

        def close(self):
            pass

    monkeypatch.setattr(packed_stream_gds, "KvikioGdsReader", FakeGdsReader)
    gds = load_magi2_nf4_packed_state(
        artifact, model(), blocksize=64, quant_format="fp8", range_read_backend="gds",
    )["mlp.moe_mlp"]
    assert gds_shards == [shard]
    assert not list(gds.buffers())
    with pytest.raises(ValueError, match="requires a CUDA target"):
        gds.materialize_segment("W_gate", 0, 1, dtype=torch.float32, device="cpu")
    import ctypes
    from types import SimpleNamespace

    from mirai.core.models.compressed_weights.packed import packed_stream_io

    calls = []
    shard_bytes = shard.read_bytes()

    def preadv(fd, buffers, offset):
        target, = buffers
        assert fd == 7
        assert ctypes.addressof(ctypes.c_char.from_buffer(target)) % 4096 == 0
        assert len(target) % 4096 == offset % 4096 == 0
        calls.append((offset, len(target)))
        target[:] = shard_bytes[offset:offset + len(target)]
        return len(target)

    monkeypatch.setattr(packed_stream_io, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(packed_stream_io, "os", SimpleNamespace(
        O_RDONLY=0, O_DIRECT=16384, open=lambda path, flags: 7 if flags == 16384 else None,
        preadv=preadv, close=lambda fd: None,
    ))
    direct = load_magi2_nf4_packed_state(
        artifact, model(), blocksize=64, quant_format="fp8", range_read_backend="direct_io",
    )["mlp.moe_mlp"]
    torch.testing.assert_close(
        direct.materialize_segment("W_gate", 1, 3, dtype=torch.float32, device="cpu"), expected, rtol=0, atol=0,
    )
    assert calls
    stats = direct._packed_range_reader.reader.snapshot()
    assert stats["direct_io_read_bytes"] == sum(length for _, length in calls)
    assert stats["direct_io_overread_bytes"] == stats["direct_io_read_bytes"] - stats["read_bytes"] > 0
    unaligned = save_magi2_nf4_packed_state(tmp_path / "unaligned", source)
    with pytest.raises(ValueError, match="4096-byte-aligned"):
        load_magi2_nf4_packed_state(
            unaligned, model(), blocksize=64, quant_format="fp8", range_read_backend="gds",
        )
    target = model(heads=1, experts=4)
    with pytest.raises(ValueError, match="head topology"):
        load_magi2_nf4_packed_state(artifact, target, blocksize=64, quant_format="fp8")
    assert hasattr(target.mlp.moe_mlp, "W_gate")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layers"]["mlp.moe_mlp"]["expert_shapes"]["W_down"][1] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = model()
    with pytest.raises(ValueError, match="projection shapes"):
        load_magi2_nf4_packed_state(artifact, target, blocksize=64, quant_format="fp8")
    assert hasattr(target.mlp.moe_mlp, "W_down")


def test_nf4_restore_rejects_incompatible_statistics_without_mutation() -> None:
    from mirai.core.models.magi2_preview.quantized_experts import Magi2Nf4ExpertStore

    store = Magi2Nf4ExpertStore(num_groups=2, blocksize=64)
    buffers = {
        "W_gate_nf4": torch.zeros(2, 128, dtype=torch.uint8),
        "W_gate_nf4_absmax": torch.zeros(2, 4, dtype=torch.uint8),
        "W_gate_nf4_nabsmax": torch.ones(2, 1),
        "W_gate_nf4_offset": torch.zeros(2),
        "W_gate_nf4_code": torch.zeros(16),
        "W_gate_nf4_ncode": torch.zeros(256),
    }
    descriptor = {
        "num_groups": 2, "blocksize": 64, "expert_shapes": {"W_gate": [2, 16, 16]},
        "expert_dtypes": {"W_gate": "bfloat16"}, "buffers": list(buffers),
        "nf4_meta": {"blocksize": 64, "nested_blocksize": 256,
                     "nested_dtype": "float32", "weight_dtype": "bfloat16"},
    }
    with pytest.raises(ValueError, match="nf4_absmax"):
        store.restore_packed_state(
            descriptor=descriptor, buffers={**buffers, "W_gate_nf4_absmax": torch.zeros(2, 3)},
            keys=("W_gate",),
        )
    assert not store.is_loaded("W_gate")
    store.restore_packed_state(descriptor=descriptor, buffers=buffers, keys=("W_gate",))
    assert store.is_loaded("W_gate")


@_requires_nf4
def test_magi2_nf4_packed_state_round_trip(tmp_path) -> None:
    source_module = _reduced_moe(
        device=torch.device("cuda"), dtype=torch.bfloat16
    )
    source_container = _container(source_module)
    source_store = quantize_magi2_experts_in_place(source_container)[
        "mlp.moe_mlp"
    ]
    artifact = tmp_path / "magi2_nf4.safetensors"
    save_magi2_nf4_packed_state(
        artifact, source_container, metadata={"model_variant": "reduced"}
    )

    mismatch_module = _reduced_moe(
        device=torch.device("cpu"), dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="model_variant"):
        load_magi2_nf4_packed_state(
            artifact,
            _container(mismatch_module),
            blocksize=source_store.blocksize,
            expected_metadata={"model_variant": "another-model"},
        )

    restored_module = _reduced_moe(
        device=torch.device("cpu"), dtype=torch.bfloat16
    )
    restored_container = _container(restored_module)
    restored_store = load_magi2_nf4_packed_state(
        artifact, restored_container, blocksize=source_store.blocksize
    )["mlp.moe_mlp"]

    assert restored_store.payload_bytes() == source_store.payload_bytes()
    for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
        assert restored_store.expert_weight_shape(key) == source_store.expert_weight_shape(key)
        expected = source_store.materialize_segment(
            key, 0, source_store.num_groups, dtype=torch.bfloat16, device=torch.device("cuda")
        )
        actual = restored_store.materialize_segment(
            key, 0, restored_store.num_groups, dtype=torch.bfloat16, device=torch.device("cuda")
        )
        assert torch.equal(actual, expected)


@_requires_nf4
@pytest.mark.parametrize("access", ["full_dequant", "chunked_dequant"])
@pytest.mark.parametrize("expert_autograd", ["standard", "segmented_recompute"])
def test_nf4_grouped_execution_matches_the_reference_loop_on_its_own_weights(
    access: str,
    expert_autograd: str,
) -> None:
    """Invariant 5: the packed path reproduces the vendored math exactly.

    The comparison runs the reference per-expert loop on the dequantized
    weights, so any difference is execution, not quantization error.
    """
    hidden_size = 256
    packed = _reduced_moe(
        device=torch.device("cuda"), dtype=torch.bfloat16, hidden_size=hidden_size
    )
    reference = _reduced_moe(
        device=torch.device("cuda"), dtype=torch.bfloat16, hidden_size=hidden_size
    )
    stores = quantize_magi2_experts_in_place(_container(packed))
    store = stores["mlp.moe_mlp"]
    store.set_expert_weight_access_policy(expert_weight_access=access, expert_dequant_chunk_size=3)
    with torch.no_grad():
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            getattr(reference, key).copy_(_dense_from_store(store, key))
    packed_adapter = _attach_router_lora(packed)
    reference_adapter = _attach_router_lora(reference)
    packed._mirai_moe_kernel_backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm"),
        expert_autograd=expert_autograd,
    )

    expected = _run(reference, reference_adapter, hidden_size)
    actual = _run(packed, packed_adapter, hidden_size)
    for want, got in zip(expected, actual):
        assert torch.allclose(want, got, rtol=2e-2, atol=2e-3)
    assert expected[2].abs().max() > 0.0
    assert expected[3].abs().max() > 0.0


@_requires_nf4
@pytest.mark.parametrize("expert_autograd", ["standard", "segmented_recompute"])
def test_nf4_routed_segments_match_existing_nf4_output_and_gradients(
    expert_autograd: str,
) -> None:
    hidden_size = 256
    module = _reduced_moe(
        device=torch.device("cuda"), dtype=torch.bfloat16, hidden_size=hidden_size
    )
    store = quantize_magi2_experts_in_place(_container(module))["mlp.moe_mlp"]
    store.set_expert_weight_access_policy(
        expert_weight_access="chunked_dequant", expert_dequant_chunk_size=3
    )
    adapter = _attach_router_lora(module)
    module._mirai_moe_kernel_backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm"),
        expert_autograd=expert_autograd,
    )
    reference = _run(module, adapter, hidden_size)
    module.zero_grad(set_to_none=True)
    adapter.lora_a.grad = None
    adapter.lora_b.grad = None
    module._mirai_moe_kernel_backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(
            forward_backend="bmm", dx_backend="bmm", routed_gemm="triton"
        ),
        expert_autograd=expert_autograd,
    )
    candidate = _run(module, adapter, hidden_size)
    for expected, actual in zip(reference, candidate, strict=True):
        assert torch.allclose(expected, actual, rtol=2e-2, atol=2e-3)


@_requires_nf4
def test_nf4_routed_inference_reduces_segment_outputs_in_token_order() -> None:
    hidden_size = 256
    module = _reduced_moe(
        device=torch.device("cuda"), dtype=torch.bfloat16, hidden_size=hidden_size
    )
    store = quantize_magi2_experts_in_place(_container(module))["mlp.moe_mlp"]
    store.set_expert_weight_access_policy(
        expert_weight_access="chunked_dequant", expert_dequant_chunk_size=3
    )
    torch.manual_seed(7)
    hidden = torch.randn(16, hidden_size, device="cuda", dtype=torch.bfloat16)
    module._mirai_moe_kernel_backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    )
    with torch.no_grad():
        reference = module._forward_impl(hidden)
    module._mirai_moe_kernel_backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(routed_gemm="triton")
    )
    with torch.no_grad():
        candidate = module._forward_impl(hidden)
    assert torch.allclose(reference, candidate, rtol=2e-2, atol=6.25e-2)


@_requires_nf4
def test_nf4_routed_backward_revisits_only_segment_local_weights() -> None:
    hidden_size = 256
    module = _reduced_moe(
        device=torch.device("cuda"), dtype=torch.bfloat16, hidden_size=hidden_size
    )
    store = quantize_magi2_experts_in_place(_container(module))["mlp.moe_mlp"]
    store.set_expert_weight_access_policy(
        expert_weight_access="chunked_dequant", expert_dequant_chunk_size=3
    )
    visits: list[tuple[str, int, int]] = []
    live: list[weakref.ReferenceType[torch.Tensor]] = []
    peak_live = 0
    materialize = store.materialize_segment

    def recording_materialize(key, start, stop, **kwargs):
        nonlocal peak_live
        visits.append((str(key), int(start), int(stop)))
        live[:] = [reference for reference in live if reference() is not None]
        result = materialize(key, start, stop, **kwargs)
        live.append(weakref.ref(result))
        peak_live = max(peak_live, len(live))
        return result

    store.materialize_segment = recording_materialize  # type: ignore[method-assign]
    adapter = _attach_router_lora(module)
    module._mirai_moe_kernel_backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(routed_gemm="triton"),
        expert_autograd="segmented_recompute",
    )
    _run(module, adapter, hidden_size)
    ranges = [(start, stop) for _key, start, stop in visits]
    assert ranges
    assert all(0 <= start < stop <= store.num_groups for start, stop in ranges)
    assert all(stop - start <= store.segment_group_span() for start, stop in ranges)
    for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
        key_ranges = [(start, stop) for seen, start, stop in visits if seen == key]
        assert len(key_ranges) % 2 == 0
        midpoint = len(key_ranges) // 2
        assert key_ranges[:midpoint] == key_ranges[midpoint:]
    assert peak_live <= len(MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
    torch.cuda.synchronize()
    gc.collect()
    assert not any(reference() is not None for reference in live)


@_requires_nf4
def test_nf4_grouped_execution_stays_within_quantization_error_of_bf16() -> None:
    """The packed experts approximate the released BF16 experts, not replace them."""
    hidden_size = 256
    bf16 = _reduced_moe(device=torch.device("cuda"), dtype=torch.bfloat16, hidden_size=hidden_size)
    packed = _reduced_moe(
        device=torch.device("cuda"), dtype=torch.bfloat16, hidden_size=hidden_size
    )
    quantize_magi2_experts_in_place(_container(packed))
    bf16_adapter = _attach_router_lora(bf16)
    packed_adapter = _attach_router_lora(packed)
    bf16._mirai_moe_kernel_backend = Magi2GroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    )
    packed._mirai_moe_kernel_backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    )
    expected = _run(bf16, bf16_adapter, hidden_size)
    actual = _run(packed, packed_adapter, hidden_size)
    # A four-expert synthetic layer with random weights runs the packed values
    # through the swiglu7 clamp ladder and two matmuls, so the bound below is a
    # coarse "approximates rather than replaces" guard on these shapes, not a
    # fidelity claim about the released checkpoint.
    scale = expected[0].abs().mean().clamp(min=1e-6)
    assert (expected[0] - actual[0]).abs().mean() < 0.25 * scale
    cosine = torch.nn.functional.cosine_similarity(
        expected[0].reshape(1, -1), actual[0].reshape(1, -1)
    )
    assert float(cosine) > 0.95
    for tensor in actual:
        assert torch.isfinite(tensor).all()
    assert actual[3].abs().max() > 0.0


@_requires_nf4
def test_token_chunked_nf4_backward_supports_sparse_production_group_count() -> None:
    hidden_size = 96
    module = _reduced_moe(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        hidden_size=hidden_size,
        expert_intermediate_size=16,
        num_heads=12,
        num_experts=256,
        top_k=6,
    )
    store = quantize_magi2_experts_in_place(_container(module))["mlp.moe_mlp"]
    store.set_expert_weight_access_policy(
        expert_weight_access="chunked_dequant",
        expert_dequant_chunk_size=384,
    )
    adapter = _attach_router_lora(module)
    module._mirai_moe_kernel_backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(
            forward_backend="torch_grouped",
            dx_backend="torch_grouped",
        ),
        expert_autograd="segmented_recompute",
    )

    result = _run(module, adapter, hidden_size)

    for tensor in result:
        assert torch.isfinite(tensor).all()


@_requires_nf4
def test_chunked_and_full_dequant_are_the_same_computation() -> None:
    hidden_size = 256
    module = _reduced_moe(
        device=torch.device("cuda"), dtype=torch.bfloat16, hidden_size=hidden_size
    )
    store = quantize_magi2_experts_in_place(_container(module))["mlp.moe_mlp"]
    adapter = _attach_router_lora(module)
    module._mirai_moe_kernel_backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    )
    store.set_expert_weight_access_policy(expert_weight_access="full_dequant")
    full = _run(module, adapter, hidden_size)
    module.zero_grad(set_to_none=True)
    adapter.lora_a.grad = None
    adapter.lora_b.grad = None
    store.set_expert_weight_access_policy(
        expert_weight_access="chunked_dequant", expert_dequant_chunk_size=2
    )
    chunked = _run(module, adapter, hidden_size)
    for want, got in zip(full, chunked):
        assert torch.equal(want, got)


@_requires_nf4
def test_packed_experts_never_enter_autograd() -> None:
    """No dense expert weight survives the forward, and none carries a gradient."""
    hidden_size = 256
    module = _reduced_moe(
        device=torch.device("cuda"), dtype=torch.bfloat16, hidden_size=hidden_size
    )
    store = quantize_magi2_experts_in_place(_container(module))["mlp.moe_mlp"]
    adapter = _attach_router_lora(module)
    module._mirai_moe_kernel_backend = Magi2QuantizedGroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    )
    _run(module, adapter, hidden_size)
    for buffer in store.buffers():
        assert not buffer.requires_grad
        assert buffer.grad_fn is None
    assert not any(name in module._parameters for name in MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
    # The packed payload is a fraction of the dense stack it replaced.
    dense_bytes = sum(
        int(torch.tensor(shape).prod().item()) * 2
        for shape in (store.expert_weight_shape(key) for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
    )
    assert store.payload_bytes() < dense_bytes // 3


@_requires_nf4
def test_quantized_seam_attaches_through_the_pipeline_policy() -> None:
    module = _reduced_moe(device=torch.device("cuda"), dtype=torch.bfloat16)
    container = _container(module)
    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = container
    pipeline._nf4_blocksize = 64
    pipeline._frozen_weight_quantization = "nf4"
    pipeline._quantize_experts_on_load = False
    pipeline._expert_stores = {}
    pipeline._moe_optimization_policy = MoEOptimizationPolicy(
        expert_weight_access="chunked_dequant", expert_dequant_chunk_size=4
    )

    assert not pipeline.has_quantized_frozen_weights()
    pipeline.enable_quantized_frozen_weights("nf4")
    assert pipeline.has_quantized_frozen_weights()
    assert isinstance(module._mirai_moe_kernel_backend, Magi2QuantizedGroupedMoEBackend)
    store = magi2_expert_store(module)
    assert store is not None
    assert store.expert_weight_access == "chunked_dequant"
    assert store.segment_group_span() == 4
    report = pipeline.get_quantized_frozen_weight_report()
    assert report["quant_format"] == "nf4"
    assert report["grouped_expert_modules"] == 1
    assert report["quantized_tensors"] == 3
    assert report["packed_bytes"] > 0
    # A second call is idempotent rather than a second quantization pass.
    pipeline.enable_quantized_frozen_weights("nf4")
    assert magi2_expert_store(module) is store
    with pytest.raises(ValueError, match="frozen_weight_quantization"):
        pipeline.enable_quantized_frozen_weights("int8")


@_requires_nf4
def test_alignment_verdict_is_read_from_the_packed_layout() -> None:
    module = _reduced_moe(device=torch.device("cuda"), dtype=torch.bfloat16)
    store = quantize_magi2_experts_in_place(_container(module))["mlp.moe_mlp"]
    backend = Magi2QuantizedGroupedMoEBackend(Magi2GroupedMoEPlan())
    assert backend.inspect_expert_layout(module) == ()
    assert (
        magi2_grouped_mm_alignment_violations(
            w_gate=store.layout_probe("W_gate"),
            w_up=store.layout_probe("W_up"),
            w_down=store.layout_probe("W_down"),
        )
        == ()
    )
    attached = attach_grouped_moe_backend(_container(module), backend)
    assert attached == 1


# ---------------------------------------------------------------------------
# Streaming quantized load
# ---------------------------------------------------------------------------


def _write_expert_shard(path, module: torch.nn.Module) -> None:
    from safetensors.torch import save_file

    save_file(
        {
            f"mlp.moe_mlp.{name}": getattr(module, name).detach().cpu().contiguous()
            for name in MAGI2_ROUTED_EXPERT_TENSOR_NAMES
        },
        str(path / "model.safetensors"),
    )


def test_streaming_load_names_a_missing_routed_tensor(tmp_path) -> None:
    """An incomplete expert stack is a lineage mismatch, not a degraded mode."""
    from safetensors.torch import save_file

    from mirai.core.models.magi2_preview.quantized_experts import (
        stream_quantize_magi2_experts,
    )

    module = _reduced_moe(device=torch.device("cpu"), dtype=torch.float32)
    save_file(
        {"mlp.moe_mlp.W_gate": module.W_gate.detach().cpu().contiguous()},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(Magi2QuantizedExpertError, match="mlp.moe_mlp.W_up"):
        stream_quantize_magi2_experts(_container(module), checkpoint_dir=str(tmp_path))


@_requires_nf4
def test_streaming_load_packs_every_routed_tensor_from_its_shard(tmp_path) -> None:
    from mirai.core.models.magi2_preview.quantized_experts import (
        stream_quantize_magi2_experts,
    )

    source = _reduced_moe(device=torch.device("cpu"), dtype=torch.bfloat16)
    _write_expert_shard(tmp_path, source)
    dense = {
        name: getattr(source, name).detach().clone() for name in MAGI2_ROUTED_EXPERT_TENSOR_NAMES
    }

    target = _reduced_moe(device=torch.device("cpu"), dtype=torch.bfloat16)
    container = _container(target)
    stores = stream_quantize_magi2_experts(container, checkpoint_dir=str(tmp_path))
    store = stores["mlp.moe_mlp"]
    assert store.is_fully_loaded()
    # The dense parameters are gone and the payload replaced them in place.
    assert not any(name in target._parameters for name in MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
    for name in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
        restored = store.materialize_segment(
            name, 0, store.num_groups, dtype=torch.bfloat16, device=torch.device("cuda")
        )
        expected = dense[name].to(device="cuda")
        assert restored.shape == expected.shape
        error = (restored.float() - expected.float()).abs().mean()
        assert float(error) < 0.1 * float(expected.float().abs().mean())
    # The packed payload is host-resident, exactly where the shards were read to.
    assert all(buffer.device.type == "cpu" for buffer in store.buffers())
