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
    "device_residency_budget_gib": {"device_residency_budget_gib": 1.0},
    "router_quantization": {"router_quantization": "int8_per_channel"},
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

    def dequantize(fields, *args, **kwargs):
        packed = fields["packed"]
        decoded.append(int(fields["offset"]))
        return packed

    monkeypatch.setattr(quantized_experts, "_nf4_dequantize", dequantize)
    for key, shape in {"W_gate": (8, 16), "W_up": (8, 16), "W_down": (16, 8)}.items():
        store._shapes[key] = (5, *shape)
        store._dtypes[key] = torch.float32
        store._set_buffer(f"{key}_nf4", torch.randn(5, *shape, generator=generator))
        for suffix in ("absmax", "nabsmax", "offset"):
            store._set_buffer(f"{key}_nf4_{suffix}", torch.arange(5))
        for suffix in ("code", "ncode"):
            store._set_buffer(f"{key}_nf4_{suffix}", torch.zeros(1))

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
    monkeypatch.setattr(packed_experts, "iter_magi2_moe_layers", lambda _: [("layer", model)])
    monkeypatch.setattr(packed_experts, "magi2_expert_store", lambda _: source)
    monkeypatch.setattr(
        packed_experts, "install_magi2_nf4_expert_stores",
        lambda *args, **kwargs: {"layer": Magi2Nf4ExpertStore(num_groups=5)},
    )
    legacy_path = save_magi2_nf4_packed_state(tmp_path / "separate", model)
    store = load_magi2_nf4_packed_state(legacy_path, model, blocksize=64)["layer"]
    assert not store.has_fused_gate_up
    for observed, reference in zip(run("active_group_dequant"), expected):
        torch.testing.assert_close(observed, reference)

    path = save_magi2_nf4_packed_state(tmp_path / "packed", model, fuse_gate_up=True)
    store = load_magi2_nf4_packed_state(path, model, blocksize=64)["layer"]
    assert store.has_fused_gate_up
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
    assert router.router and not router.routed and not router.quantizable
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


def test_magi2_mixed_plan_preserves_group_order_and_checks_source_fingerprint(tmp_path) -> None:
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
        for group, fmt in enumerate(("fp8", "bf16", "nvfp4", "fp8"))
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


@pytest.mark.parametrize("fmt", ["gguf_iq2", "gguf_iq3", "fp8", "nvfp4"])
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
