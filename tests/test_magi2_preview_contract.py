from __future__ import annotations

import dataclasses
import importlib.machinery
import importlib.util
import os
import pathlib
import subprocess
import sys
import types

import pytest
import torch
from torch.nn.utils import parametrize

from mirai.config.loader import load_config
from mirai.core.models.magi2_preview.grouped_moe import (
    _CONSUMED_POLICY_FIELDS,
    _MemoryBoundedSwiGlu7,
    _QUANTIZED_CONSUMED_POLICY_FIELDS,
    Magi2GroupedMoEBackend,
    Magi2GroupedMoEPlan,
    Magi2GroupedMoEPolicyError,
    attach_grouped_moe_backend,
    magi2_grouped_mm_alignment_violations,
    resolve_magi2_moe_execution,
    select_grouped_backends,
    validate_grouped_moe_backend_support,
)
from mirai.core.moe.runtime.gemm import BackendProbe, grouped_mm_op
from mirai.core.models.magi2_preview.pipeline import LowRankWeight
from mirai.core.models.magi2_preview.lora_execution import (
    attach_magi2_lora_executor,
    execute_router_lora,
)
from mirai.core.models.native_video import resolve_output_fps
from mirai.core.models.providers import (
    NativeCacheEncoderConfig,
    get_model_family_provider,
)
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy


def test_magi2_fa3_inference_does_not_require_sink_extension(monkeypatch) -> None:
    from mirai.vendors.magi2_preview.model import magi2_preview as model

    query = torch.ones(3, 2, 4, dtype=torch.bfloat16)
    offsets = torch.tensor([0, 3], dtype=torch.int32)

    def fake_fwd(*args):
        return query.clone(), torch.zeros(2, 3), None

    monkeypatch.setattr(model, "flash_attn_3_cuda", types.SimpleNamespace(fwd=fake_fwd))
    monkeypatch.setattr(model, "fa3_varlen_func_with_sink", None)
    with torch.no_grad():
        output = model._fa3_varlen_func_with_sink_inference(
            query,
            query,
            query,
            cu_seqlens_q=offsets,
            cu_seqlens_k=offsets,
            max_seqlen_q=3,
            max_seqlen_k=3,
            softcap=-1.0,
            sink=torch.zeros(1, 2),
            sink_layout="sh",
        )

    assert torch.equal(output, torch.full_like(query, 0.5))


def test_magi2_provider_is_native_sparse_moe() -> None:
    provider = get_model_family_provider("magi2-preview")
    assert provider is not None
    assert provider.is_native_model()
    assert provider.is_sparse_moe_model()
    assert provider.supports_batched_cfg_inference()
    assert provider.require_pipeline_type().__name__ == "Magi2PreviewPipeline"
    assert provider.validate_family_params({"audio_tokens": 1}) == []
    assert provider.validate_family_params({"unknown": True})


def test_magi2_offload_example_loads() -> None:
    config = load_config("configs/magi2_preview/train_offload.toml")
    assert config.model.type == "magi2-preview"
    assert config.training.blocks_to_swap == 40
    assert config.memory.weight_residency_strategy == "block_swap"
    assert config.adapter.target_preset == "attn_router"


def test_magi2_low_rank_weight_preserves_default_and_gradients() -> None:
    base = torch.randn(2, 3, 5)
    adapter = LowRankWeight(tuple(base.shape), rank=2, alpha=2.0)
    reference = adapter(base)
    assert torch.equal(reference, base)
    adapter.lora_b.data.normal_()
    loss = adapter(base).square().mean()
    loss.backward()
    assert adapter.lora_a.grad is not None
    assert adapter.lora_b.grad is not None
    assert torch.isfinite(adapter.lora_a.grad).all()
    assert torch.isfinite(adapter.lora_b.grad).all()


def test_magi2_grouped_linear_lora_executes_without_dense_delta() -> None:
    from mirai.vendors.magi2_preview.model.magi2_preview import GroupedLinearBase

    torch.manual_seed(41)
    module = GroupedLinearBase(
        in_features=7,
        out_features=5,
        num_experts=3,
        bias=False,
        dtype=torch.float32,
    )
    module.weight.data.normal_()
    module.weight.requires_grad_(False)
    parametrize.register_parametrization(
        module,
        "weight",
        LowRankWeight(tuple(module.weight.shape), rank=3, alpha=6.0),
    )
    module.parametrizations.weight.original.requires_grad_(False)
    adapter = module.parametrizations.weight[0]
    adapter.lora_b.data.normal_()
    splits = [3, 0, 4]
    reference_input = torch.randn(7, 7, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_(True)

    reference = module(reference_input, m_splits=splits)
    reference.square().mean().backward()
    reference_grads = (
        reference_input.grad.detach().clone(),
        adapter.lora_a.grad.detach().clone(),
        adapter.lora_b.grad.detach().clone(),
    )
    adapter.lora_a.grad = None
    adapter.lora_b.grad = None
    attach_magi2_lora_executor(module, "weight")

    def reject_materialization(_base: torch.Tensor) -> torch.Tensor:
        raise AssertionError("activation-space LoRA must not materialize B @ A")

    adapter.forward = reject_materialization
    actual = module(actual_input, m_splits=splits)
    actual.square().mean().backward()
    actual_grads = (
        actual_input.grad,
        adapter.lora_a.grad,
        adapter.lora_b.grad,
    )

    torch.testing.assert_close(actual, reference)
    for expected, observed in zip(reference_grads, actual_grads, strict=True):
        torch.testing.assert_close(observed, expected)
    state_keys = set(module.state_dict())
    assert "parametrizations.weight.0.lora_a" in state_keys
    assert "parametrizations.weight.0.lora_b" in state_keys


def test_magi2_router_lora_executes_without_dense_delta() -> None:
    module, adapter = _build_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32
    )
    torch.manual_seed(43)
    reference_input = torch.randn(6, 2, 8, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_(True)
    gate = module.gate.view(2, 4, 8).float()
    reference = torch.einsum("shd,hed->hse", reference_input.float(), gate)
    reference.square().mean().backward()
    reference_grads = (
        reference_input.grad.detach().clone(),
        adapter.lora_a.grad.detach().clone(),
        adapter.lora_b.grad.detach().clone(),
    )
    adapter.lora_a.grad = None
    adapter.lora_b.grad = None
    attach_magi2_lora_executor(module, "gate")

    def reject_materialization(_base: torch.Tensor) -> torch.Tensor:
        raise AssertionError("activation-space router LoRA must not materialize B @ A")

    adapter.forward = reject_materialization
    actual = execute_router_lora(module, actual_input)
    actual.square().mean().backward()
    actual_grads = (
        actual_input.grad,
        adapter.lora_a.grad,
        adapter.lora_b.grad,
    )

    torch.testing.assert_close(actual, reference)
    for expected, observed in zip(reference_grads, actual_grads, strict=True):
        torch.testing.assert_close(observed, expected)


def _build_reduced_moe(
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int = 16,
    expert_intermediate_size: int = 12,
) -> tuple[torch.nn.Module, LowRankWeight]:
    """Reduced-shape vendored MoE layer with a router LoRA, as attn_router uses."""
    from mirai.vendors.magi2_preview.model.magi2_preview import (
        CoreMultiHeadMoE,
        CoreMultiHeadMoEConfig,
    )

    torch.manual_seed(0)
    module = CoreMultiHeadMoE(
        CoreMultiHeadMoEConfig(
            hidden_size=hidden_size,
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=expert_intermediate_size,
            num_layers=1,
            params_dtype=dtype,
            score_func="sigmoid",
            route_norm=True,
            route_scale=4.9,
        )
    )
    with torch.no_grad():
        for tensor in (module.gate, module.W_gate, module.W_up, module.W_down):
            tensor.normal_(std=0.1)
        module.router.expert_bias.normal_(std=0.05)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    parametrize.register_parametrization(
        module, "gate", LowRankWeight(tuple(module.gate.shape), rank=2, alpha=2.0)
    )
    module.parametrizations["gate"].original.requires_grad_(False)
    adapter = module.parametrizations["gate"][0]
    with torch.no_grad():
        adapter.lora_b.normal_(std=0.1)
    module.to(device=device)
    return module, adapter


def _run_reduced_moe(
    *,
    backend: str | None,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int = 16,
    expert_intermediate_size: int = 12,
    token_chunk_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    module, adapter = _build_reduced_moe(
        device=device,
        dtype=dtype,
        hidden_size=hidden_size,
        expert_intermediate_size=expert_intermediate_size,
    )
    if backend is not None:
        module._mirai_moe_kernel_backend = Magi2GroupedMoEBackend(
            Magi2GroupedMoEPlan(forward_backend=backend, dx_backend=backend),
            token_chunk_size=token_chunk_size,
        )
    torch.manual_seed(7)
    hidden = torch.randn(5, hidden_size, device=device, dtype=dtype, requires_grad=True)
    output = module._forward_impl(hidden)
    loss = output.float().square().mean()
    loss.backward()
    return (
        output.detach(),
        loss.detach(),
        hidden.grad.detach().clone(),
        adapter.lora_a.grad.detach().clone(),
        adapter.lora_b.grad.detach().clone(),
    )


def test_magi2_grouped_moe_matches_reference_loop_on_cpu() -> None:
    device = torch.device("cpu")
    reference = _run_reduced_moe(backend=None, device=device, dtype=torch.float32)
    grouped = _run_reduced_moe(backend="bmm", device=device, dtype=torch.float32)
    for expected, actual in zip(reference, grouped):
        assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-6)
    assert reference[3].abs().max() > 0.0
    assert reference[4].abs().max() > 0.0


def test_magi2_fp32_swiglu_reference_keeps_saved_operands_immutable() -> None:
    torch.manual_seed(23)
    gate = torch.randn(5, 7, requires_grad=True)
    up = torch.randn(5, 7, requires_grad=True)
    gate_before = gate.detach().clone()
    up_before = up.detach().clone()

    output = _MemoryBoundedSwiGlu7.apply(gate, up, torch.float32)
    output.square().mean().backward()

    torch.testing.assert_close(gate, gate_before, rtol=0, atol=0)
    torch.testing.assert_close(up, up_before, rtol=0, atol=0)
    assert gate.grad is not None and torch.isfinite(gate.grad).all()
    assert up.grad is not None and torch.isfinite(up.grad).all()


def test_magi2_grouped_moe_matches_reference_loop_in_bf16_on_cpu() -> None:
    """The fp32 clamp/activation ladder must hold for BF16 expert weights."""
    device = torch.device("cpu")
    reference = _run_reduced_moe(backend=None, device=device, dtype=torch.bfloat16)
    grouped = _run_reduced_moe(backend="bmm", device=device, dtype=torch.bfloat16)
    for expected, actual in zip(reference, grouped):
        assert torch.allclose(
            expected.float(), actual.float(), rtol=2e-2, atol=2e-3
        )
    assert reference[3].abs().max() > 0.0
    assert reference[4].abs().max() > 0.0


def test_magi2_grouped_moe_token_chunks_preserve_outputs_and_gradients() -> None:
    """Token scheduling changes workspace only, including under autograd."""

    device = torch.device("cpu")
    whole = _run_reduced_moe(
        backend="bmm", device=device, dtype=torch.float32, token_chunk_size=0
    )
    chunked = _run_reduced_moe(
        backend="bmm", device=device, dtype=torch.float32, token_chunk_size=2
    )
    for expected, actual in zip(whole, chunked):
        assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-6)


def test_magi2_grouped_moe_token_chunks_route_once() -> None:
    module, _adapter = _build_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32
    )
    module._mirai_moe_kernel_backend = Magi2GroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm"),
        token_chunk_size=2,
    )
    route = module._route
    calls = 0

    def counted_route(x_heads):
        nonlocal calls
        calls += 1
        return route(x_heads)

    module._route = counted_route
    module._forward_impl(torch.randn(5, 16))
    assert calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_magi2_grouped_moe_matches_reference_loop_on_cuda() -> None:
    """BF16 parity on shapes that satisfy the torch_grouped 16-byte precondition.

    ``d_head`` 64 and ``expert_intermediate_size`` 64 are both multiples of 8
    BF16 elements, as the real MAGI-2 256/1280 layout is, so an ``auto`` plan can
    reach ``torch_grouped`` here instead of silently staying on ``bmm``.
    """
    device = torch.device("cuda")
    shapes = {"hidden_size": 128, "expert_intermediate_size": 64}
    reference = _run_reduced_moe(
        backend=None, device=device, dtype=torch.bfloat16, **shapes
    )
    grouped = _run_reduced_moe(
        backend="auto", device=device, dtype=torch.bfloat16, **shapes
    )
    for expected, actual in zip(reference, grouped):
        assert torch.allclose(
            expected.float(), actual.float(), rtol=2e-2, atol=2e-3
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_magi2_routed_triton_provider_output_loss_and_gradient_parity() -> None:
    """Exercise policy resolution and the complete provider-owned MoE backend.

    Two heads by four experts proves the flattened effective-group mapping;
    token chunking forces multiple production ``execute`` calls. The retained
    top-k probabilities verify the combine gradient independently of the router
    adapter parameters that produced them.
    """

    pytest.importorskip("triton")
    device = torch.device("cuda")
    hidden_size = 64
    intermediate_size = 48

    def run(routed_mode: str):
        module, adapter = _build_reduced_moe(
            device=device,
            dtype=torch.bfloat16,
            hidden_size=hidden_size,
            expert_intermediate_size=intermediate_size,
        )
        policy = MoEOptimizationPolicy(
            kernel_backend="grouped",
            moe_gemm_backend="bmm",
            moe_routed_gemm=routed_mode,
        )
        plan = resolve_magi2_moe_execution(policy)
        assert plan is not None and plan.routed_gemm == routed_mode
        module._mirai_moe_kernel_backend = Magi2GroupedMoEBackend(
            plan, token_chunk_size=2
        )
        observed_probabilities: list[torch.Tensor] = []
        route = module._route

        def retained_route(x_heads):
            probabilities, indices = route(x_heads)
            probabilities.retain_grad()
            observed_probabilities.append(probabilities)
            return probabilities, indices

        module._route = retained_route
        torch.manual_seed(41)
        hidden = torch.randn(
            5, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        output = module._forward_impl(hidden)
        loss = output.float().square().mean()
        loss.backward()
        assert len(observed_probabilities) == 1
        probabilities = observed_probabilities[0]
        assert probabilities.grad is not None
        return (
            output.detach(),
            loss.detach(),
            hidden.grad.detach(),
            probabilities.detach(),
            probabilities.grad.detach(),
            adapter.lora_a.grad.detach(),
            adapter.lora_b.grad.detach(),
        )

    reference = run("disabled")
    candidate = run("triton")
    for expected, actual in zip(reference, candidate, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=3e-2, atol=3e-2
        )
    assert candidate[6].abs().max() > 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_magi2_routed_triton_selected_expert_step_and_save_load_parity() -> None:
    """Train a resident expert tensor through the production provider seam."""
    import io

    pytest.importorskip("triton")
    device = torch.device("cuda")

    def run(*, routed: bool):
        module, adapter = _build_reduced_moe(
            device=device, dtype=torch.bfloat16,
            hidden_size=64, expert_intermediate_size=48,
        )
        module.W_down.requires_grad_(True)
        if routed:
            policy = MoEOptimizationPolicy(
                kernel_backend="grouped", moe_gemm_backend="bmm",
                moe_routed_gemm="triton",
            )
            plan = resolve_magi2_moe_execution(policy)
            assert plan is not None
            module._mirai_moe_kernel_backend = Magi2GroupedMoEBackend(
                plan, token_chunk_size=2
            )
        optimizer = torch.optim.SGD(
            [module.W_down, adapter.lora_a, adapter.lora_b], lr=1e-2
        )
        torch.manual_seed(53)
        hidden = torch.randn(5, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        output = module._forward_impl(hidden)
        loss = output.float().square().mean()
        loss.backward()
        observed = (
            output.detach(), loss.detach(), hidden.grad.detach(),
            module.W_down.grad.detach(), adapter.lora_a.grad.detach(),
            adapter.lora_b.grad.detach(),
        )
        optimizer.step()
        stepped = module.W_down.detach().clone()
        if routed:
            payload = io.BytesIO()
            torch.save(module.state_dict(), payload)
            payload.seek(0)
            restored, _ = _build_reduced_moe(
                device=device, dtype=torch.bfloat16,
                hidden_size=64, expert_intermediate_size=48,
            )
            restored.load_state_dict(torch.load(payload, map_location=device, weights_only=True))
            torch.testing.assert_close(restored.W_down, module.W_down, rtol=0, atol=0)
        return observed + (stepped,)

    reference = run(routed=False)
    candidate = run(routed=True)
    for expected, actual in zip(reference, candidate, strict=True):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=4e-2, atol=4e-2)
    assert candidate[3].abs().max() > 0
    assert candidate[5].abs().max() > 0


def _reduced_expert_weights(
    *, d_head: int, d_expert: int, dtype: torch.dtype, groups: int = 4
) -> dict[str, torch.Tensor]:
    return {
        "w_gate": torch.zeros(groups, d_head, d_expert, dtype=dtype),
        "w_up": torch.zeros(groups, d_head, d_expert, dtype=dtype),
        "w_down": torch.zeros(groups, d_expert, d_head, dtype=dtype),
    }


def test_magi2_grouped_mm_alignment_predicate_reads_real_layouts() -> None:
    """The precondition is derived from strides, not from a shape heuristic."""
    assert (
        magi2_grouped_mm_alignment_violations(
            **_reduced_expert_weights(
                d_head=256, d_expert=1280, dtype=torch.bfloat16
            )
        )
        == ()
    )
    violations = magi2_grouped_mm_alignment_violations(
        **_reduced_expert_weights(d_head=8, d_expert=12, dtype=torch.bfloat16)
    )
    assert violations
    # d_head 8 is 16 bytes in BF16 and passes; the 12-element expert
    # intermediate is 24 bytes and is the offending dimension, in the stored
    # weight, in the activation rows, and in the transposed dX weight view.
    assert all("24 bytes" in reason for reason in violations)
    assert any("W_gate forward expert weight" in reason for reason in violations)
    assert any("dX transposed weight view" in reason for reason in violations)
    assert all("16 bytes" in reason for reason in violations)
    # The same shapes are aligned for 4-byte elements.
    assert (
        magi2_grouped_mm_alignment_violations(
            **_reduced_expert_weights(d_head=8, d_expert=12, dtype=torch.float32)
        )
        == ()
    )


def _always_available(name: str) -> BackendProbe:
    return BackendProbe(name, True, "test probe")


def test_magi2_unaligned_experts_downgrade_auto_and_reject_explicit() -> None:
    """Auto falls back once; an explicit torch_grouped never downgrades silently."""
    violations = magi2_grouped_mm_alignment_violations(
        **_reduced_expert_weights(d_head=8, d_expert=12, dtype=torch.bfloat16)
    )
    auto = Magi2GroupedMoEPlan(forward_backend="auto", dx_backend="auto")
    assert select_grouped_backends(
        auto,
        probe=_always_available,
        alignment_violations=violations,
        device_label="cuda:0",
    ) == ("bmm", "bmm")
    assert select_grouped_backends(
        auto,
        probe=_always_available,
        alignment_violations=(),
        device_label="cuda:0",
    ) == ("torch_grouped", "torch_grouped")

    explicit = Magi2GroupedMoEPlan(
        forward_backend="torch_grouped", dx_backend="torch_grouped"
    )
    with pytest.raises(Magi2GroupedMoEPolicyError) as excinfo:
        select_grouped_backends(
            explicit,
            probe=_always_available,
            alignment_violations=violations,
            device_label="cuda:0",
        )
    message = str(excinfo.value)
    assert "16 bytes" in message
    assert "24 bytes" in message
    assert "W_gate" in message


def test_magi2_grouped_backend_records_expert_layout_at_attach_time() -> None:
    """The verdict comes from the attached module, before any forward runs."""
    module, _adapter = _build_reduced_moe(
        device=torch.device("cpu"), dtype=torch.bfloat16
    )
    container = torch.nn.Module()
    container.moe_mlp = module

    auto_backend = Magi2GroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="auto", dx_backend="auto")
    )
    assert attach_grouped_moe_backend(container, auto_backend) == 1
    assert auto_backend.alignment_violations
    assert auto_backend._resolve(torch.device("cpu")) == ("bmm", "bmm")

    explicit_backend = Magi2GroupedMoEBackend(
        Magi2GroupedMoEPlan(
            forward_backend="torch_grouped", dx_backend="torch_grouped"
        )
    )
    with pytest.raises(Magi2GroupedMoEPolicyError, match="16 bytes"):
        attach_grouped_moe_backend(container, explicit_backend)


def test_magi2_grouped_backend_backstops_alignment_at_first_forward() -> None:
    """A backend that never saw attach applies the same policy on first use."""
    module, _adapter = _build_reduced_moe(
        device=torch.device("cpu"), dtype=torch.bfloat16
    )
    backend = Magi2GroupedMoEBackend(
        Magi2GroupedMoEPlan(
            forward_backend="torch_grouped", dx_backend="torch_grouped"
        )
    )
    assert backend.alignment_violations == ()
    with pytest.raises(Magi2GroupedMoEPolicyError, match="16 bytes"):
        backend._resolve(torch.device("cpu"), module)


def test_magi2_grouped_moe_rejects_trainable_expert_weights() -> None:
    module, _adapter = _build_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32
    )
    module.W_gate.requires_grad_(True)
    module._mirai_moe_kernel_backend = Magi2GroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    )
    with pytest.raises(RuntimeError, match="frozen expert weights"):
        module._forward_impl(torch.randn(4, 16))


def test_magi2_moe_policy_defaults_keep_the_reference_path() -> None:
    assert resolve_magi2_moe_execution(MoEOptimizationPolicy()) is None
    assert resolve_magi2_moe_execution(MoEOptimizationPolicy(kernel_backend="torch")) is None
    plan = resolve_magi2_moe_execution(
        MoEOptimizationPolicy(kernel_backend="grouped", moe_gemm_backend="bmm")
    )
    assert plan == Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    role_plan = resolve_magi2_moe_execution(
        MoEOptimizationPolicy(
            kernel_backend="grouped",
            moe_gemm_backend="bmm",
            moe_gemm_backend_dx="torch_grouped",
        )
    )
    assert role_plan.dx_backend == "torch_grouped"


def test_magi2_moe_policy_rejects_unsupported_fields() -> None:
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(MoEOptimizationPolicy(kernel_backend="megablocks"))
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(
            MoEOptimizationPolicy(
                kernel_backend="grouped", expert_weight_access="full_dequant"
            )
        )
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(
            MoEOptimizationPolicy(kernel_backend="grouped", moe_dispatch="triton")
        )
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(
            MoEOptimizationPolicy(
                kernel_backend="grouped", moe_gemm_backend_dw="torch_grouped"
            )
        )
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(
            MoEOptimizationPolicy(kernel_backend="grouped", moe_gemm_backend="persistent")
        )


# One non-default value per policy field MAGI-2 does not consume. Fields with a
# cross-field constraint carry the co-required keys so the policy itself builds.
_NON_DEFAULT_POLICY_VALUES: dict[str, dict[str, object]] = {
    "expert_weight_access": {"expert_weight_access": "full_dequant"},
    "expert_dequant_chunk_size": {"expert_dequant_chunk_size": 8},
    "expert_device_cache_gib": {"expert_device_cache_gib": 1.0},
    "device_residency_budget_gib": {"device_residency_budget_gib": 1.0},
    "quantize_experts_on_load": {"quantize_experts_on_load": True},
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


def test_magi2_policy_rejection_covers_every_unconsumed_field() -> None:
    declared = {field.name for field in dataclasses.fields(MoEOptimizationPolicy)}
    assert (
        set(_NON_DEFAULT_POLICY_VALUES) | set(_QUANTIZED_CONSUMED_POLICY_FIELDS)
        == declared
    )


@pytest.mark.parametrize("field_name", sorted(_NON_DEFAULT_POLICY_VALUES))
def test_magi2_moe_policy_rejects_every_unconsumed_field(field_name: str) -> None:
    policy = MoEOptimizationPolicy(
        kernel_backend="grouped", **_NON_DEFAULT_POLICY_VALUES[field_name]
    )
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(policy)


def test_magi2_grouped_backend_support_is_validated_before_execution() -> None:
    cpu = torch.device("cpu")
    validate_grouped_moe_backend_support(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm"), device=cpu
    )
    plan = Magi2GroupedMoEPlan(
        forward_backend="torch_grouped", dx_backend="torch_grouped"
    )
    if grouped_mm_op() is None:
        with pytest.raises(Magi2GroupedMoEPolicyError, match="torch build"):
            validate_grouped_moe_backend_support(plan, device=cpu)
    else:
        # The device architecture gate needs the execution device, which weight
        # residency assigns after the policy is configured.
        validate_grouped_moe_backend_support(plan, device=cpu)
    if torch.cuda.is_available():
        from mirai.core.moe.runtime.gemm import probe_backend

        probe = probe_backend("torch_grouped", device=torch.device("cuda"))
        if probe.available:
            validate_grouped_moe_backend_support(
                plan, device=torch.device("cuda")
            )
        else:
            with pytest.raises(Magi2GroupedMoEPolicyError):
                validate_grouped_moe_backend_support(
                    plan, device=torch.device("cuda")
                )


def test_magi2_provider_declares_moe_kernel_backend_capability() -> None:
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    capabilities = pipeline.get_memory_feature_capabilities()
    assert capabilities.moe_kernel_backend
    # Packed routed experts add the expert-storage and direct-restore controls;
    # the router keeps its released FP32 projection.
    assert capabilities.quantized_frozen_weights
    assert capabilities.expert_tensor_specs
    assert capabilities.expert_weight_access_policy
    assert capabilities.quantize_experts_on_load
    assert capabilities.packed_frozen_weight_state
    assert not capabilities.router_quantization_policy
    assert not capabilities.trainable_parameter_offload


def test_magi2_pipeline_attaches_and_detaches_the_grouped_seam() -> None:
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

    module, _adapter = _build_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32
    )
    container = torch.nn.Module()
    container.moe_mlp = module
    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = container

    pipeline.configure_moe_optimization_policy(MoEOptimizationPolicy())
    assert module._mirai_moe_kernel_backend is None

    state_before = {
        key: value.detach().clone() for key, value in module.state_dict().items()
    }
    hidden = torch.randn(3, int(module.num_heads * module.d_head))
    output_before = module._forward_impl(hidden)
    pipeline.configure_moe_optimization_policy(
        MoEOptimizationPolicy(moe_routed_gemm="disabled")
    )
    assert module._mirai_moe_kernel_backend is None
    output_after = module._forward_impl(hidden)
    torch.testing.assert_close(output_after, output_before, rtol=0, atol=0)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)

    pipeline.configure_moe_optimization_policy(
        MoEOptimizationPolicy(kernel_backend="grouped", moe_gemm_backend="bmm")
    )
    assert isinstance(module._mirai_moe_kernel_backend, Magi2GroupedMoEBackend)
    assert module._mirai_moe_kernel_backend.plan.forward_backend == "bmm"

    pipeline.configure_moe_optimization_policy(MoEOptimizationPolicy())
    assert module._mirai_moe_kernel_backend is None


def test_magi2_explicit_disabled_preserves_output_loss_and_gradients() -> None:
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

    def run(policy: MoEOptimizationPolicy):
        module, adapter = _build_reduced_moe(
            device=torch.device("cpu"), dtype=torch.float32
        )
        container = torch.nn.Module()
        container.moe_mlp = module
        pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = container
        pipeline.configure_moe_optimization_policy(policy)
        assert module._mirai_moe_kernel_backend is None
        torch.manual_seed(812)
        hidden = torch.randn(
            3, int(module.num_heads * module.d_head), requires_grad=True
        )
        output = module._forward_impl(hidden)
        loss = output.square().mean()
        loss.backward()
        return (
            output.detach(),
            loss.detach(),
            hidden.grad.detach(),
            adapter.lora_a.grad.detach(),
            adapter.lora_b.grad.detach(),
            {key: value.detach().clone() for key, value in module.state_dict().items()},
        )

    implicit = run(MoEOptimizationPolicy())
    explicit = run(MoEOptimizationPolicy(moe_routed_gemm="disabled"))
    for expected, actual in zip(implicit[:-1], explicit[:-1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert implicit[-1].keys() == explicit[-1].keys()
    for key, expected in implicit[-1].items():
        torch.testing.assert_close(explicit[-1][key], expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("MIRAI_REMOTE_GPU_TESTS") != "1",
    reason="configured remote CUDA validation required",
)
def test_magi2_explicit_disabled_config_allocates_no_cuda_state() -> None:
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

    module, _ = _build_reduced_moe(
        device=torch.device("cuda"), dtype=torch.bfloat16
    )
    container = torch.nn.Module()
    container.moe_mlp = module
    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = container
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    pipeline.configure_moe_optimization_policy(
        MoEOptimizationPolicy(moe_routed_gemm="disabled")
    )
    torch.cuda.synchronize()
    assert module._mirai_moe_kernel_backend is None
    assert torch.cuda.memory_allocated() == allocated_before

_NATIVE_ONLY_IMPORT_PROBE = """
import importlib
import sys

# Any attempt to import Diffusers, at any depth, fails inside this probe.
sys.modules["diffusers"] = None

for module_name in (
    "mirai.vendors.magi2_preview.common.native_config",
    "mirai.vendors.magi2_preview.model.turbo_vaed",
    "mirai.vendors.magi2_preview.pipeline.sampler",
    "mirai.vendors.magi2_preview.pipeline.inference_engine",
    "mirai.core.models.magi2_preview.pipeline",
):
    importlib.import_module(module_name)

loaded = [name for name, module in sys.modules.items() if name.split(".")[0] == "diffusers" and module is not None]
assert not loaded, loaded
print("native-only")
"""


def test_magi2_load_and_forward_path_imports_without_diffusers() -> None:
    """The vendored load/sampling path must import with Diffusers unavailable."""
    result = subprocess.run(
        [sys.executable, "-c", _NATIVE_ONLY_IMPORT_PROBE],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert "native-only" in result.stdout


# Reference values below were recorded from the vendored Flow-UniPC scheduler
# while it still derived from the Diffusers SchedulerMixin/ConfigMixin pair
# (diffusers 0.38.0.dev0, torch 2.8.0, CPU, float32). They pin the numerical
# behavior of the de-mixined scheduler and stay checkable without Diffusers.
_REFERENCE_TIMESTEPS = [999, 922, 817, 666, 428]
_REFERENCE_SIGMAS = [
    0.9996664524078369,
    0.9227216839790344,
    0.8178097009658813,
    0.666296124458313,
    0.42826521396636963,
    0.0,
]
_REFERENCE_FINAL_STATE = [
    0.00241873, -0.0345197, -0.00316913, -0.96296185, -0.00124159, 0.22802615,
    0.00136874, -0.02314733, 0.16864161, -0.34140122, -1.25907004, -0.07026298,
    1.20232749, -0.04369771, 1.28708112, 0.63499749, -2.51118183, 1.28877664,
    0.00323494, -1.22862971, -1.28806329, -0.00306869, -0.00766827, 0.01063976,
]
_REFERENCE_STEP_SDE = [
    -0.43580237, -0.03616618, 0.74579, -1.38215542, 0.02491212, 0.74293435,
    -0.35706922, -0.82400393, 0.03100988, -1.21992922, -0.85443711, 0.37083352,
    -0.02005237, -0.35716024, 0.2563442, 0.86101246, -2.47145104, 0.73862797,
    -1.3879627, -1.04105484, -0.5656082, 0.37187937, 0.38976991, 1.00858998,
]
_REFERENCE_RANDN = [
    0.61268586, -1.17535365, -0.76464927, -0.66656566, 0.74436599, -0.64531738,
]


def _reference_sample() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    return torch.randn(1, 2, 3, 4, generator=generator, dtype=torch.float32)


def test_magi2_flow_unipc_scheduler_reproduces_recorded_reference() -> None:
    from mirai.vendors.magi2_preview.pipeline.sampler import (
        FlowUniPCMultistepScheduler,
    )

    scheduler = FlowUniPCMultistepScheduler()
    assert scheduler.config.solver_type == "bh2"
    assert scheduler.config.prediction_type == "flow_prediction"
    assert len(scheduler) == 1000

    scheduler.set_timesteps(5, device="cpu", shift=3.0)
    assert scheduler.timesteps.tolist() == _REFERENCE_TIMESTEPS
    assert scheduler.sigmas.tolist() == pytest.approx(_REFERENCE_SIGMAS, abs=1e-7)

    state = _reference_sample()
    for index, timestep in enumerate(scheduler.timesteps):
        model_output = torch.sin(state * (index + 1)) * 0.5
        state = scheduler.step(model_output, timestep, state, return_dict=False)[0]
    assert state.flatten().tolist() == pytest.approx(_REFERENCE_FINAL_STATE, abs=1e-6)


def test_magi2_flow_unipc_scheduler_step_returns_named_output() -> None:
    from mirai.vendors.magi2_preview.pipeline.sampler import (
        FlowUniPCMultistepScheduler,
    )

    sample = _reference_sample()
    named_scheduler = FlowUniPCMultistepScheduler()
    named_scheduler.set_timesteps(5, device="cpu", shift=3.0)
    named = named_scheduler.step(
        torch.zeros_like(sample), named_scheduler.timesteps[0], sample.clone()
    )

    tuple_scheduler = FlowUniPCMultistepScheduler()
    tuple_scheduler.set_timesteps(5, device="cpu", shift=3.0)
    plain = tuple_scheduler.step(
        torch.zeros_like(sample),
        tuple_scheduler.timesteps[0],
        sample.clone(),
        return_dict=False,
    )

    assert isinstance(named.prev_sample, torch.Tensor)
    assert torch.equal(named.prev_sample, plain[0])


def test_magi2_flow_unipc_stochastic_step_is_generator_reproducible() -> None:
    from mirai.vendors.magi2_preview.pipeline.sampler import (
        FlowUniPCMultistepScheduler,
        randn_tensor,
    )

    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(5, device="cpu", shift=3.0)
    sample = _reference_sample()
    velocity = torch.linspace(-1, 1, sample.numel()).reshape(sample.shape)
    generator = torch.Generator().manual_seed(7)
    stochastic = scheduler.step_sde(
        velocity, 1, sample.clone(), noise_theta=0.5, generator=generator
    )
    assert stochastic.flatten().tolist() == pytest.approx(
        _REFERENCE_STEP_SDE, abs=1e-6
    )

    noise_generator = torch.Generator().manual_seed(99)
    noise = randn_tensor(
        (2, 3),
        generator=noise_generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert noise.flatten().tolist() == pytest.approx(_REFERENCE_RANDN, abs=1e-6)
    with pytest.raises(TypeError):
        randn_tensor((2, 3), generator=[torch.Generator()])


def test_magi2_native_config_registration_matches_upstream_filtering() -> None:
    from mirai.vendors.magi2_preview.common.native_config import (
        NativeConfigMixin,
        register_to_config,
    )

    class _Probe(NativeConfigMixin):
        @register_to_config
        def __init__(self, a: int = 1, b: str = "x", **kwargs) -> None:
            self.seen = dict(kwargs)
            if a < 0:
                self.register_to_config(a=0)

    probe = _Probe.from_config({"a": 5, "unknown": 7, "_class_name": "Z"})
    assert probe.config.a == 5
    assert probe.config.b == "x"
    assert probe.seen == {}
    assert sorted(probe.config) == ["a", "b"]

    overridden = _Probe(-3)
    assert overridden.config.a == 0
    with pytest.raises(AttributeError):
        overridden.config.a = 1


def _bare_pipeline():
    """A pipeline object without the released weights behind it.

    ``__new__`` skips ``_build_model``, which is the only part of construction
    that needs the checkpoint; every seam probed below is pure policy.
    """
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    return pipeline


# --- Text conditioning is never fabricated ---------------------------------


def test_magi2_missing_text_embedding_is_rejected() -> None:
    pipeline = _bare_pipeline()
    like = torch.zeros(2, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="carried none"):
        pipeline._text_features({"text_mask": None}, batch=2, like=like)
    with pytest.raises(ValueError, match="carried none"):
        pipeline._text_features(None, batch=2, like=like)


def test_magi2_wrong_width_text_embedding_is_rejected_as_lineage_mismatch() -> None:
    """A foreign encoder width is a cache-lineage error, not something to broadcast."""
    pipeline = _bare_pipeline()
    like = torch.zeros(2, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="lineage"):
        pipeline._text_features({"t5": torch.zeros(2, 3, 4096)}, batch=2, like=like)
    # A scalar-per-sample payload is the shape the removed fabrication accepted.
    with pytest.raises(ValueError, match="lineage"):
        pipeline._text_features({"t5": torch.ones(2)}, batch=2, like=like)
    with pytest.raises(ValueError, match="latent batch"):
        pipeline._text_features({"t5": torch.zeros(1, 3, 5120)}, batch=2, like=like)


def test_magi2_qwen_width_text_embedding_is_accepted() -> None:
    from mirai.core.models.magi2_preview.pipeline import MAGI2_TEXT_EMBED_WIDTH

    assert MAGI2_TEXT_EMBED_WIDTH == 5120
    pipeline = _bare_pipeline()
    like = torch.zeros(2, 1, 1, 1, 1)
    value, lengths = pipeline._text_features(
        {"t5": torch.zeros(2, 3, MAGI2_TEXT_EMBED_WIDTH)}, batch=2, like=like
    )
    assert tuple(value.shape) == (2, 3, MAGI2_TEXT_EMBED_WIDTH)
    assert lengths.tolist() == [3, 3]


# --- The audio placeholder is seeded, not stolen from the global stream ------


class _AudioProbeProxy:
    """Data proxy that hands the assembled model input straight through."""

    def process_input(self, model_input):
        return (model_input,)

    def process_output(self, prediction):
        return prediction, None


def _audio_probe_transformer(model_input):
    """Fold the audio placeholder into the returned video tensor."""
    return model_input.x_t + model_input.audio_x_t.mean()


def _audio_probe_pipeline(seed: int):
    from mirai.core.models.magi2_preview.pipeline import (
        Magi2PreviewPipeline,
        Magi2RuntimeOptions,
    )

    pipeline = _bare_pipeline()
    pipeline.options = Magi2RuntimeOptions(config_path="", audio_tokens=-1)
    pipeline.data_proxy = _AudioProbeProxy()
    pipeline.transformer = _audio_probe_transformer
    pipeline._audio_noise_generator = (
        Magi2PreviewPipeline._build_audio_noise_generator(seed)
    )
    return pipeline


def _audio_probe_loss(pipeline, *, latent_frames: int) -> torch.Tensor:
    latents = torch.zeros(1, 48, latent_frames, 2, 2)
    text = {"magi2": torch.zeros(1, 3, 5120)}
    prediction = pipeline.forward(latents, torch.full((1,), 0.5), text)
    return prediction.square().mean()


def test_magi2_training_audio_placeholder_is_reproducible_under_the_run_seed() -> None:
    """Same seed, same batch, same loss - the audio draw is not global RNG."""
    pytest.importorskip("mirai.vendors.magi2_preview.pipeline.preview_data_proxy")

    first = [
        _audio_probe_loss(_audio_probe_pipeline(7), latent_frames=4).item()
        for _ in range(1)
    ]
    second = [
        _audio_probe_loss(_audio_probe_pipeline(7), latent_frames=4).item()
        for _ in range(1)
    ]
    assert first == second

    other = _audio_probe_loss(_audio_probe_pipeline(11), latent_frames=4).item()
    assert other != first[0]

    # Fresh noise per training forward is deliberate: the second call of one run
    # advances the family stream instead of repeating the first draw.
    pipeline = _audio_probe_pipeline(7)
    step_one = _audio_probe_loss(pipeline, latent_frames=4).item()
    step_two = _audio_probe_loss(pipeline, latent_frames=4).item()
    assert step_one != step_two
    assert step_one == first[0]


def test_magi2_training_audio_placeholder_never_touches_the_global_stream() -> None:
    """Changing the audio track length must not perturb global torch RNG."""
    pytest.importorskip("mirai.vendors.magi2_preview.pipeline.preview_data_proxy")

    torch.manual_seed(1234)
    baseline = torch.random.get_rng_state()
    for latent_frames in (2, 4, 9):
        _audio_probe_loss(_audio_probe_pipeline(7), latent_frames=latent_frames)
        assert torch.equal(torch.random.get_rng_state(), baseline)
    assert torch.randn(1).item() == pytest.approx(
        torch.randn(1, generator=torch.Generator().manual_seed(1234)).item()
    )


# --- Sampling policies are answered, not discarded --------------------------


def test_magi2_native_sampler_rejects_policies_it_does_not_implement() -> None:
    """cfg_mode and the solver name reach the sampler and are answered."""
    pipeline = _bare_pipeline()
    call = dict(
        noise=torch.zeros(48, 1, 1, 1),
        context=torch.zeros(1, 1, 5120),
        context_null=torch.zeros(1, 1, 5120),
        denoise_steps=1,
        guidance_scale=1.0,
        generator=torch.Generator(),
    )
    with pytest.raises(ValueError, match="unipc"):
        pipeline.sample_native_preview(**call, solver_name="euler")
    with pytest.raises(ValueError, match="B=2"):
        pipeline.sample_native_preview(**call, cfg_mode="sequential")


def test_magi2_shipped_examples_ask_for_implemented_sampling_policies() -> None:
    config = load_config("configs/magi2_preview/inference_offload.toml")
    assert config.inference.cfg_mode == "batched"
    training = load_config("configs/magi2_preview/train_offload.toml")
    assert training.logging.sample_solver == "unipc"


def test_native_denoise_loop_leaves_cfg_policy_to_the_family_when_unset() -> None:
    """An unset cfg_mode must not be forced onto a family that owns its CFG."""
    import inspect

    from mirai.core.training.preview.preview import run_native_denoise_loop

    signature = inspect.signature(run_native_denoise_loop)
    assert signature.parameters["cfg_mode"].default is None


# --- Gradient checkpointing modes -------------------------------------------


def test_magi2_gradient_checkpointing_accepts_only_implemented_modes() -> None:
    pipeline = _bare_pipeline()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.block = torch.nn.Module()

    pipeline.set_gradient_checkpointing("standard")
    assert pipeline.transformer.block.gradient_checkpointing is True
    pipeline.set_gradient_checkpointing("off")
    assert pipeline.transformer.block.gradient_checkpointing is False
    pipeline.set_gradient_checkpointing(True)
    assert pipeline.transformer.block.gradient_checkpointing is True

    for mode in ("selective", "aggressive"):
        with pytest.raises(ValueError, match=mode):
            pipeline.set_gradient_checkpointing(mode)


def test_magi2_preset_requests_an_implemented_checkpointing_mode() -> None:
    config = load_config("configs/magi2_preview/train_offload.toml")
    assert config.training.gradient_checkpointing == "standard"


# --- Adapter state loading honours strict -----------------------------------


def test_magi2_load_state_dict_honours_strict() -> None:
    module, _adapter = _build_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32
    )
    container = torch.nn.Module()
    container.moe_mlp = module
    pipeline = _bare_pipeline()
    pipeline.transformer = container

    full = pipeline.state_dict()
    assert full, "the reduced MoE layer must expose a LoRA surface"
    pipeline.load_state_dict(full)

    partial = {key: value for key, value in list(full.items())[1:]}
    with pytest.raises(ValueError, match="missing"):
        pipeline.load_state_dict(partial)
    with pytest.raises(ValueError, match="missing"):
        pipeline.load_state_dict({})
    pipeline.load_state_dict(partial, strict=False)

    stray = dict(full)
    stray["not.a.lora.tensor"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        pipeline.load_state_dict(stray)


# --- Latent geometry --------------------------------------------------------


def test_magi2_frame_count_maps_to_the_transformer_temporal_stride() -> None:
    """8n+1 frames round-trip: a request spans 8*(T-1)+1 of the 25 fps timeline.

    The transformer is positioned at temporal stride 8 (``vae_stride[0] = 8``,
    ``time_pos_fps = 3.125``), so a request denotes 25 fps-equivalent frames.
    Declaring stride 4 here would halve every horizon: the native ten-second
    length would come out as a five-second request.
    """
    pipeline = _bare_pipeline()
    layout = pipeline.get_video_latent_layout()
    assert layout.temporal_downsample == 8
    assert (layout.frame_count_modulus, layout.frame_count_remainder) == (8, 1)
    for frames in (57, 65, 161, 249):
        channels, t_lat, h_lat, w_lat = pipeline.preview_latent_geometry(
            frame_count=frames, height=256, width=448
        )
        assert (channels, h_lat, w_lat) == (48, 16, 28)
        assert 8 * (t_lat - 1) + 1 == frames


def test_magi2_unrepresentable_frame_count_is_rejected() -> None:
    """Counts off the 8n+1 grid fail, including 4n+1 counts that are not 8n+1."""
    pipeline = _bare_pipeline()
    for frames in (16, 18, 11, 125, 250):
        with pytest.raises(ValueError, match="8n"):
            pipeline.preview_latent_geometry(
                frame_count=frames, height=256, width=448
            )
    with pytest.raises(ValueError, match="multiples of 16"):
        pipeline.preview_latent_geometry(frame_count=57, height=250, width=448)


def test_magi2_generation_envelope_bounds_are_enforced() -> None:
    """The native horizon is accepted and refiner-space lengths are rejected.

    Latent ``T`` 32 is the full length the preview model is trained for, so the
    249-frame request that resolves to it must succeed. ``T`` above 32 belongs
    to the refiner's temporal upsample rather than to a longer preview.
    """
    from mirai.core.models.magi2_preview.pipeline import (
        MAGI2_MAX_SAMPLING_LATENT_FRAMES,
        MAGI2_MIN_SAMPLING_LATENT_FRAMES,
    )

    pipeline = _bare_pipeline()
    assert (MAGI2_MIN_SAMPLING_LATENT_FRAMES, MAGI2_MAX_SAMPLING_LATENT_FRAMES) == (
        8,
        32,
    )

    assert pipeline.preview_latent_geometry(
        frame_count=249, height=256, width=448
    ) == (48, 32, 16, 28)
    assert pipeline.preview_latent_geometry(
        frame_count=57, height=256, width=448
    ) == (48, 8, 16, 28)

    with pytest.raises(ValueError, match="native horizon"):
        pipeline.preview_latent_geometry(frame_count=257, height=256, width=448)
    with pytest.raises(ValueError, match="refiner"):
        pipeline.preview_latent_geometry(frame_count=497, height=256, width=448)
    for frames in (49, 17, 9, 1):
        with pytest.raises(ValueError, match="Short-horizon"):
            pipeline.preview_latent_geometry(
                frame_count=frames, height=256, width=448
            )


def test_magi2_preview_only_decode_is_half_rate() -> None:
    """The declared output rate matches the frames the Turbo VAE emits.

    The decoder expands T latents into 4*(T-1)+1 physical frames while the
    request spans 8*(T-1)+1 frames of the 25 fps timeline, so the clip covers
    the requested duration only at 12.5 fps. 25 fps needs the refiner, which
    this path does not run.
    """
    from mirai.core.models.magi2_preview.pipeline import (
        MAGI2_NATIVE_OUTPUT_FPS,
        MAGI2_REQUEST_FPS,
        _magi2_decoded_frames,
        _magi2_request_frames,
    )

    pipeline = _bare_pipeline()
    layout = pipeline.get_video_latent_layout()
    assert layout.native_output_fps == MAGI2_NATIVE_OUTPUT_FPS == 12.5
    assert resolve_output_fps(pipeline=pipeline, requested=None) == 12.5
    # An explicit request still wins over the declared native rate.
    assert resolve_output_fps(pipeline=pipeline, requested=24.0) == 24.0

    for latent_frames in (8, 16, 32):
        requested = _magi2_request_frames(latent_frames)
        decoded = _magi2_decoded_frames(latent_frames)
        assert requested == 2 * decoded - 1
        assert (requested - 1) / MAGI2_REQUEST_FPS == pytest.approx(
            (decoded - 1) / MAGI2_NATIVE_OUTPUT_FPS
        )


def test_magi2_shipped_preview_sample_length_is_inside_the_envelope() -> None:
    config = load_config("configs/magi2_preview/train_offload.toml")
    pipeline = _bare_pipeline()
    channels, t_lat, _h, _w = pipeline.preview_latent_geometry(
        frame_count=config.logging.sample_frame_count, height=256, width=448
    )
    assert channels == 48
    assert t_lat == 16


def test_magi2_cache_frame_trimming_matches_the_layout_rule() -> None:
    """Cache-time trimming lands on the same 8n+1 grid the layout declares.

    A cache trimmed on a different stride writes latents whose length the
    generation path cannot express, so the two rules are checked together.
    """
    from mirai.core.models.magi2_preview.cache import magi2_cache_frame_trim

    layout = _bare_pipeline().get_video_latent_layout()
    modulus = int(layout.frame_count_modulus)
    remainder = int(layout.frame_count_remainder) % modulus
    for raw_frames, expected in ((17, 17), (24, 17), (25, 25), (120, 113), (7, 1)):
        trimmed = magi2_cache_frame_trim(raw_frames)
        assert trimmed == expected
        assert trimmed <= raw_frames
        assert trimmed % modulus == remainder


# --- Refiner stage ----------------------------------------------------------


_REFINER_CONFIG = str(
    pathlib.Path(__file__).resolve().parents[1]
    / "mirai"
    / "vendors"
    / "magi2_preview"
    / "configs"
    / "magi2_refiner.json"
)


def _refiner_pipeline(model_path: str = "./models/MAGI-2-preview"):
    """A pipeline whose only configured surface is the refiner stage."""
    from mirai.core.models.magi2_preview.pipeline import Magi2RuntimeOptions

    pipeline = _bare_pipeline()
    pipeline.model_config = type("_ModelConfig", (), {"path": model_path})()
    pipeline.options = Magi2RuntimeOptions(
        config_path="", refiner_config_path=_REFINER_CONFIG
    )
    return pipeline


def test_magi2_refiner_resamples_the_preview_latent_onto_the_decoder_grid() -> None:
    """T -> 2T-1 is what makes a refined clip decode at the requested length.

    The preview transformer sits at temporal stride 8 while the shared decoder
    expands stride 4, which is the half-rate preview. The refiner's ``2T - 1``
    resample lands the latent on the decoder's own grid, so the decoded count
    equals the frames the request denotes on its 25 fps timeline.
    """
    from mirai.core.models.magi2_preview.pipeline import (
        _magi2_decoded_frames,
        _magi2_request_frames,
    )
    from mirai.core.models.magi2_preview.refiner import (
        magi2_refiner_decoded_frames,
        magi2_refiner_latent_frames,
    )

    assert magi2_refiner_latent_frames(32) == 63
    for latent_frames in (8, 16, 32):
        refined = magi2_refiner_latent_frames(latent_frames)
        decoded = magi2_refiner_decoded_frames(latent_frames)
        # The refined clip carries 2*(preview decode) - 1 frames ...
        assert decoded == 2 * _magi2_decoded_frames(latent_frames) - 1
        # ... which is exactly the request itself.
        assert decoded == _magi2_request_frames(latent_frames)
        assert refined == 2 * latent_frames - 1
    with pytest.raises(ValueError, match=">= 1"):
        magi2_refiner_latent_frames(0)


def test_magi2_refiner_upsample_pins_the_preview_endpoints() -> None:
    """``align_corners=True`` is load-bearing, not incidental.

    With it, preview frame ``k`` reappears at refined frame ``2k`` and the
    inserted frames are true midpoints. Without it the whole trajectory shifts
    by half a preview frame, which is a different clip.
    """
    import torch.nn.functional as F

    from mirai.core.models.magi2_preview.refiner import magi2_refiner_upsample

    torch.manual_seed(0)
    latent = torch.randn(1, 3, 5, 4, 6)
    upsampled = magi2_refiner_upsample(latent, latent_height=4, latent_width=6)
    assert tuple(upsampled.shape) == (1, 3, 9, 4, 6)
    assert torch.equal(
        upsampled,
        F.interpolate(latent, size=(9, 4, 6), mode="trilinear", align_corners=True),
    )
    for index in range(5):
        assert torch.allclose(upsampled[:, :, 2 * index], latent[:, :, index], atol=1e-6)
    for index in range(4):
        midpoint = 0.5 * (latent[:, :, index] + latent[:, :, index + 1])
        assert torch.allclose(upsampled[:, :, 2 * index + 1], midpoint, atol=1e-5)
    misaligned = F.interpolate(
        latent, size=(9, 4, 6), mode="trilinear", align_corners=False
    )
    assert not torch.allclose(upsampled, misaligned, atol=1e-4)

    # Spatial resampling shares the same call, so a wider target is expressible.
    wider = magi2_refiner_upsample(latent, latent_height=8, latent_width=12)
    assert tuple(wider.shape) == (1, 3, 9, 8, 12)
    with pytest.raises(ValueError, match=r"\[B,C,T,H,W\]"):
        magi2_refiner_upsample(latent[0], latent_height=4, latent_width=6)


def test_magi2_refiner_renoise_is_the_variance_preserving_mix() -> None:
    """``sigma`` is the SIGNAL coefficient, so the noise weight is its complement.

    Reading it as a noise level would invert the corruption: index 220 keeps
    most of the preview rather than almost none of it.
    """
    from mirai.core.models.magi2_preview.refiner import magi2_refiner_renoise

    torch.manual_seed(1)
    latent = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    sigma = 0.8389104008674622
    assert torch.allclose(
        magi2_refiner_renoise(latent, noise, sigma),
        latent * sigma + noise * (1.0 - sigma**2) ** 0.5,
    )
    assert torch.equal(magi2_refiner_renoise(latent, noise, 1.0), latent)
    assert torch.allclose(magi2_refiner_renoise(latent, noise, 0.0), noise)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        magi2_refiner_renoise(latent, noise, 1.5)
    with pytest.raises(ValueError, match="matching shapes"):
        magi2_refiner_renoise(latent, noise[:1], sigma)


def test_magi2_refiner_noise_index_addresses_the_released_sigma_table() -> None:
    """The re-noise level comes from the vendored zero-terminal-SNR table."""
    from mirai.core.models.magi2_preview.refiner import (
        MAGI2_REFINER_SIGMA_TABLE_SIZE,
        magi2_refiner_renoise_sigma,
        magi2_refiner_sigma_table,
    )
    from mirai.vendors.magi2_preview.pipeline.inference_engine import (
        ZeroSNRDDPMDiscretization,
    )

    table = magi2_refiner_sigma_table()
    assert torch.equal(
        table, ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)
    )
    assert int(table.numel()) == MAGI2_REFINER_SIGMA_TABLE_SIZE
    # Zero terminal SNR: the last entry keeps no signal at all.
    assert float(table[-1]) == 0.0
    assert float(table[0]) > 0.999
    assert bool((table[1:] - table[:-1] <= 0.0).all())
    assert magi2_refiner_renoise_sigma(220) == pytest.approx(0.8389104, abs=1e-6)
    for index in (-1, MAGI2_REFINER_SIGMA_TABLE_SIZE):
        with pytest.raises(ValueError, match="noise index"):
            magi2_refiner_renoise_sigma(index)


def test_magi2_refiner_resolves_the_released_refinement_profile() -> None:
    """``--refine`` alone must run the shipped profile, not the generic CLI values.

    Steps, guidance and shift are stated by the released refiner config; an
    unset request resolves to them, and a stated value overrides them.
    """
    from mirai.core.models.magi2_preview.refiner import Magi2Refiner

    refiner = Magi2Refiner(
        type("_ModelConfig", (), {"path": "./models/MAGI-2-preview"})(),
        config_path=_REFINER_CONFIG,
    )
    geometry = dict(preview_height=256, preview_width=448, scheduler="unipc")
    resolved = refiner.settings(
        steps=None, cfg_scale=None, shift=None, height=None, width=None, **geometry
    )
    assert (resolved.steps, resolved.cfg_scale, resolved.shift) == (5, 2.0, 5.0)
    assert resolved.noise_index == 220
    # An unset target resolves to the released 1080p delivery grid.
    assert (resolved.height, resolved.width) == (1088, 1920)
    assert refiner.latent_size(resolved) == (68, 120)
    assert resolved.as_request()["scheduler"] == "unipc"

    overridden = refiner.settings(
        steps=7, cfg_scale=1.5, shift=3.0, height=512, width=896, **geometry
    )
    assert (overridden.steps, overridden.cfg_scale, overridden.shift) == (7, 1.5, 3.0)
    assert refiner.latent_size(overridden) == (32, 56)

    with pytest.raises(RuntimeError, match="unipc"):
        refiner.settings(
            steps=None,
            cfg_scale=None,
            shift=None,
            height=None,
            width=None,
            preview_height=256,
            preview_width=448,
            scheduler="euler",
        )
    for bad in ({"steps": 0}, {"cfg_scale": -1.0}, {"shift": 0.0}):
        call = dict(
            steps=None, cfg_scale=None, shift=None, height=None, width=None, **geometry
        )
        call.update(bad)
        with pytest.raises(RuntimeError):
            refiner.settings(**call)
    with pytest.raises(RuntimeError, match="spatial stride"):
        refiner.settings(
            steps=None, cfg_scale=None, shift=None, height=250, width=448, **geometry
        )


def test_magi2_refinement_without_weights_names_the_missing_snapshot_dir() -> None:
    """A refine request with no refiner checkpoint fails before the base denoise."""
    pipeline = _refiner_pipeline()
    assert pipeline.supports_refiner()
    assert not pipeline.has_refiner_weights()
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.validate_refinement_request(
            {"scheduler": "unipc"}, frames=249, height=256, width=448
        )
    message = str(excinfo.value)
    assert "refiner" in message
    assert "model.safetensors.index.json" in message


def test_magi2_refinement_rejects_another_familys_re_noise_mechanism() -> None:
    """``t_thresh`` describes a different refiner; it is refused, not ignored."""
    pipeline = _refiner_pipeline()
    for key in ("t_thresh", "sigma_tail_steps"):
        with pytest.raises(RuntimeError, match="does not implement"):
            pipeline.validate_refinement_request(
                {"scheduler": "unipc", key: 0.85}, frames=249, height=256, width=448
            )
    # Unset values are the normal case and must not trip the same guard.
    with pytest.raises(RuntimeError, match="requires a separate checkpoint"):
        pipeline.validate_refinement_request(
            {"scheduler": "unipc", "t_thresh": None, "sigma_tail_steps": None},
            frames=249,
            height=256,
            width=448,
        )


def test_magi2_refinement_is_validated_against_the_generation_envelope() -> None:
    """A refine request outside the preview envelope fails at pre-flight."""
    pipeline = _refiner_pipeline()
    with pytest.raises(ValueError, match="Short-horizon"):
        pipeline.validate_refinement_request(
            {"scheduler": "unipc"}, frames=17, height=256, width=448
        )
    with pytest.raises(ValueError, match="native horizon"):
        pipeline.validate_refinement_request(
            {"scheduler": "unipc"}, frames=257, height=256, width=448
        )


def test_magi2_refinement_arms_the_full_rate_output(
    tmp_path: pathlib.Path,
) -> None:
    """A refined clip is written at 25 fps; an unrefined one stays at 12.5.

    The rate follows the configured stage rather than the request, so the
    layout answers it once the refinement has been resolved.
    """
    from mirai.core.models.magi2_preview.refiner import MAGI2_REFINER_OUTPUT_FPS

    root = tmp_path / "MAGI-2-preview"
    (root / "refiner").mkdir(parents=True)
    (root / "refiner" / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    pipeline = _refiner_pipeline(str(root))
    assert resolve_output_fps(pipeline=pipeline, requested=None) == 12.5
    resolved = pipeline.validate_refinement_request(
        {"scheduler": "unipc"}, frames=249, height=256, width=448
    )
    assert resolved == {
        "steps": 5,
        "cfg_scale": 2.0,
        "shift": 5.0,
        "height": 1088,
        "width": 1920,
        "noise_index": 220,
        "scheduler": "unipc",
    }
    assert (
        resolve_output_fps(pipeline=pipeline, requested=None)
        == MAGI2_REFINER_OUTPUT_FPS
        == 25.0
    )
    # An explicit request still wins over the armed rate.
    assert resolve_output_fps(pipeline=pipeline, requested=30.0) == 30.0


def test_magi2_refiner_continues_the_preview_rng_stream() -> None:
    """Refiner noise follows the release draw order instead of re-seeding."""
    pipeline = _bare_pipeline()
    source = torch.Generator(device="cpu")
    source.manual_seed(42)
    torch.randn((48, 32, 32, 56), generator=source)
    torch.randn((1, 249, 64), generator=source)
    pipeline._refiner_generator_state = source.get_state().clone()

    expected = torch.randn((17,), generator=source)
    resumed = pipeline.refiner_noise_generator(seed=999, device=torch.device("cpu"))
    actual = torch.randn(expected.shape, generator=resumed)
    assert torch.equal(actual, expected)


def test_magi2_vae_decode_window_is_bounded_by_full_latent_geometry() -> None:
    from mirai.core.models.magi2_preview.pipeline import (
        resolve_magi2_vae_decode_chunk_size,
    )

    resolve = resolve_magi2_vae_decode_chunk_size
    assert resolve(
        latent_frames=32,
        latent_height=32,
        latent_width=56,
        released_chunk_size=7,
    ) == 7
    assert resolve(
        latent_frames=63,
        latent_height=42,
        latent_width=74,
        released_chunk_size=7,
    ) == 7
    assert resolve(
        latent_frames=63,
        latent_height=68,
        latent_width=120,
        released_chunk_size=7,
    ) == 2
    assert resolve(
        latent_frames=63,
        latent_height=68,
        latent_width=120,
        released_chunk_size=7,
        requested_chunk_size=7,
    ) == 7


def test_magi2_vae_decode_chunk_override_is_scoped_even_on_failure() -> None:
    from mirai.core.models.magi2_preview.pipeline import Magi2RuntimeOptions

    class _FailingVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.first_chunk_size = 7
            self.step_size = 7
            self.observed = None

        def decode(self, value, *, output_offload):
            self.observed = (
                self.first_chunk_size,
                self.step_size,
                bool(output_offload),
                tuple(value.shape),
            )
            raise RuntimeError("decode failed")

    pipeline = _bare_pipeline()
    pipeline._vae = _FailingVAE()
    pipeline.options = Magi2RuntimeOptions(
        config_path="", vae_decode_chunk_size=2
    )
    with pytest.raises(RuntimeError, match="decode failed"):
        pipeline.decode_latents_native([torch.zeros(1, 3, 1, 1)])

    assert pipeline._vae.observed == (2, 2, True, (1, 1, 3, 1, 1))
    assert (pipeline._vae.first_chunk_size, pipeline._vae.step_size) == (7, 7)
    assert pipeline._last_vae_decode_chunk_size == 2


def test_magi2_release_base_transformer_ends_block_residency_binding() -> None:
    pipeline = _bare_pipeline()
    pipeline.transformer = torch.nn.Linear(2, 2)

    class _Handle:
        removed = False

        def remove(self) -> None:
            self.removed = True

    class _Manager:
        released = 0

        def release_device(self) -> None:
            self.released += 1

    handle = _Handle()
    manager = _Manager()
    pipeline._block_hook_handles = [handle]
    pipeline._block_swap_manager = manager

    pipeline.release_base_transformer()

    assert handle.removed is True
    assert pipeline._block_hook_handles == []
    assert manager.released == 1
    assert all(parameter.device.type == "cpu" for parameter in pipeline.parameters())


def test_magi2_refiner_release_ends_block_residency_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mirai.core.models.magi2_preview.refiner import Magi2Refiner

    class _Manager:
        released = 0

        def release_device(self) -> None:
            self.released += 1

    manager = _Manager()
    refiner = object.__new__(Magi2Refiner)
    refiner._block_hook_handles = []
    refiner._block_swap_manager = manager
    refiner._transformer = None
    refiner._data_proxy = object()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    refiner.release()

    assert manager.released == 1
    assert refiner._block_swap_manager is None
    assert refiner._data_proxy is None


def test_magi2_refiner_hooks_match_the_seam_the_session_consults() -> None:
    """The session drives refinement through two hooks; both must conform."""
    import inspect

    from mirai.core.inference.session import InferenceSession
    from mirai.core.models.base import BasePipeline
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

    for name in ("validate_refinement_request", "refine_inference_latent"):
        family = inspect.signature(getattr(Magi2PreviewPipeline, name))
        assert family == inspect.signature(getattr(BasePipeline, name)), name
    # The policy loop's own hooks live on the family, not on generic core.
    for name in (
        "supports_refiner",
        "has_refiner_weights",
        "load_refiner",
        "release_refiner",
        "release_base_transformer",
        "refiner_forward",
        "refiner_cfg_forward",
        "refiner_residency_request",
    ):
        assert callable(getattr(Magi2PreviewPipeline, name)), name
        assert not hasattr(BasePipeline, name), name
    session = inspect.signature(InferenceSession._validate_refine_request)
    assert set(session.parameters) == {"self", "refine", "frames", "height", "width"}


def _find_spec_stub(present: set[str]):
    """``find_spec`` that reports ``present`` names as installed, others as absent."""
    real = importlib.util.find_spec

    def find_spec(name: str, *args, **kwargs):
        if name in present:
            return importlib.machinery.ModuleSpec(name, loader=None)
        if name in _MAGI2_SPEC_PROBED:
            return None
        return real(name, *args, **kwargs)

    return find_spec


# Names the probes under test ask about; anything else keeps the real answer.
_MAGI2_SPEC_PROBED = {"magi_attention", "flash_attn"}


def test_magi2_attention_ops_resolve_to_the_vendored_eager_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call site binds to the registered operator, else to the eager function.

    ``magi_register_custom_op`` publishes ``torch.ops.magi2.*`` only under a real
    MagiCompiler; the callable it decorates is a complete implementation either
    way. Resolution must therefore follow the implementation, not the
    registration, while still preferring a registered operator when one exists.
    """
    from mirai.vendors.magi2_preview.common.magi_compiler_compat import (
        MAGI2_OP_NAMESPACE,
        resolve_magi2_op,
    )

    namespace = getattr(torch.ops, MAGI2_OP_NAMESPACE)

    def eager(*args: object, **kwargs: object) -> str:
        return "eager"

    def registered(*args: object, **kwargs: object) -> str:
        return "registered"

    resolve_magi2_op.cache_clear()
    try:
        if not hasattr(namespace, "flex_flash_attn_func"):
            assert resolve_magi2_op("flex_flash_attn_func", eager) is eager
            resolve_magi2_op.cache_clear()

        monkeypatch.setattr(
            namespace, "flex_flash_attn_func", registered, raising=False
        )
        assert resolve_magi2_op("flex_flash_attn_func", eager) is registered
        # The binding is resolved once, so the second call is the same object.
        assert resolve_magi2_op("flex_flash_attn_func", eager) is registered
    finally:
        resolve_magi2_op.cache_clear()


def test_magi2_unimportable_magi_attention_does_not_select_the_magi_attention_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present-but-unimportable ``magi_attention`` falls through to eager math.

    MagiAttention's extensions build for compute capability 9.0 only, so the
    distribution can be on the path while its entry point does not import. A
    presence probe would send a Hopper-reporting device into a branch that then
    fails; the import probe must resolve to nothing so the branch is not taken
    and the vendored eager path runs instead. Import reachability is all this
    establishes: an entry point that imports can still fail in its kernel build
    at call time, which is why a guaranteed path is selected rather than probed.
    """
    from mirai.vendors.magi2_preview.common.magi_compiler_compat import (
        magi_attention_flex_flash_attn_func,
    )

    magi_attention_flex_flash_attn_func.cache_clear()
    try:
        monkeypatch.setattr(
            importlib.util, "find_spec", _find_spec_stub({"magi_attention"})
        )
        # Reported as present on the path, yet no entry point resolves.
        assert importlib.util.find_spec("magi_attention") is not None
        assert magi_attention_flex_flash_attn_func() is None

        magi_attention_flex_flash_attn_func.cache_clear()
        monkeypatch.setattr(importlib.util, "find_spec", _find_spec_stub(set()))
        assert magi_attention_flex_flash_attn_func() is None
    finally:
        magi_attention_flex_flash_attn_func.cache_clear()


def test_magi2_attention_probe_reaches_single_gpu_functional_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The functional kernel remains usable without the distributed API shim."""
    import types

    from mirai.vendors.magi2_preview.common import magi_compiler_compat as compat

    expected = object()
    functional = types.SimpleNamespace(flex_flash_attn_func=lambda: expected)

    def import_module(name: str):
        if name == "magi_attention.api":
            raise ImportError("magi_attn_comm is not installed")
        if name == "magi_attention.functional.flex_flash_attn":
            return functional
        raise AssertionError(name)

    compat.magi_attention_flex_flash_attn_func.cache_clear()
    try:
        monkeypatch.setattr(
            importlib.util, "find_spec", _find_spec_stub({"magi_attention"})
        )
        monkeypatch.setattr(importlib, "import_module", import_module)
        assert compat.magi_attention_flex_flash_attn_func() is functional.flex_flash_attn_func
    finally:
        compat.magi_attention_flex_flash_attn_func.cache_clear()


def test_magi2_refiner_precondition_fails_only_when_no_attention_path_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stage is refused when neither operator nor eager math is reachable.

    An empty ``torch.ops.magi2`` namespace is not by itself a blocker: the eager
    implementations vendored beside the operators still run, provided their
    FlashAttention-2 dependency is installed. Each of the three states must be
    reported for what it is.
    """
    import types

    from mirai.vendors.magi2_preview.common.magi_compiler_compat import (
        MAGI2_EAGER_IMPL_MODULE,
        MAGI2_REFINER_REQUIRED_OPS,
        missing_magi2_custom_ops,
        require_magi2_custom_ops,
    )

    if not missing_magi2_custom_ops(MAGI2_REFINER_REQUIRED_OPS):
        pytest.skip("a real MagiCompiler registered the operators in this environment")

    eager = types.ModuleType(MAGI2_EAGER_IMPL_MODULE)
    for name in MAGI2_REFINER_REQUIRED_OPS:
        setattr(eager, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, MAGI2_EAGER_IMPL_MODULE, eager)

    # Eager implementation present, FlashAttention-2 present: the stage runs.
    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_stub({"flash_attn"}))
    require_magi2_custom_ops("The MAGI-2 refiner stage")

    # Eager implementation present, FlashAttention-2 absent.
    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_stub(set()))
    with pytest.raises(RuntimeError) as absent_dependency:
        require_magi2_custom_ops("The MAGI-2 refiner stage")
    assert "flash_attn" in str(absent_dependency.value)

    # Neither a registered operator nor an eager implementation.
    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_stub({"flash_attn"}))
    monkeypatch.setitem(
        sys.modules, MAGI2_EAGER_IMPL_MODULE, types.ModuleType(MAGI2_EAGER_IMPL_MODULE)
    )
    with pytest.raises(RuntimeError) as no_implementation:
        require_magi2_custom_ops("The MAGI-2 refiner stage")
    message = str(no_implementation.value)
    assert "torch.ops.magi2.flex_flash_attn_func" in message
    assert "torch.ops.magi2.flash_attn_func" in message
    assert "magi_compiler" in message


def test_magi2_refiner_uses_a_stage_specific_safe_transfer_mode() -> None:
    """The refiner reuses residency but not the preview's unsafe prefetch mode.

    A run configured for block swapping must stream the refiner's layers too;
    the refiner is a different architecture but the same ``block.layers`` stack.
    """
    from mirai.core.models.magi2_preview.pipeline import (
        Magi2PreviewPipeline,
        Magi2ResidencyRequest,
        Magi2RuntimeOptions,
    )

    pipeline = _bare_pipeline()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.block = torch.nn.Module()
    pipeline.transformer.block.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2) for _ in range(4)]
    )
    pipeline._adapter_configured = True
    pipeline.options = Magi2RuntimeOptions(config_path="")
    assert pipeline.refiner_residency_request() is None

    Magi2PreviewPipeline.set_weight_residency_strategy(
        pipeline, strategy="block_swap", blocks_to_swap=3, mode="async"
    )
    request = pipeline.refiner_residency_request()
    assert isinstance(request, Magi2ResidencyRequest)
    assert (request.enabled, request.blocks_to_swap, request.mode) == (True, 3, "sync")
    assert request.offload_dir is None

    pipeline.options = Magi2RuntimeOptions(
        config_path="", refiner_block_swap_mode="async"
    )
    assert pipeline.refiner_residency_request().mode == "async"

    Magi2PreviewPipeline.set_weight_residency_strategy(
        pipeline, strategy="disabled", blocks_to_swap=0, mode="async"
    )
    assert pipeline.refiner_residency_request().enabled is False


class _StubRefinerAssets:
    """Stand-in for the loaded refiner: geometry only, no weights, no device."""

    def __init__(self) -> None:
        self.loaded = 0
        self.released = 0

    def latent_size(self, settings) -> tuple[int, int]:
        return int(settings.height) // 16, int(settings.width) // 16

    def load(self, *, device: str, residency) -> None:
        self.loaded += 1

    def release(self) -> None:
        self.released += 1


class _StubRefinePipeline:
    """The provider hooks ``run_refine`` reaches the model through."""

    def __init__(self) -> None:
        # Constructed at float32 exactly as the vendored Adapter and PostAdapter
        # projections are, so its output dtype reports whether an autocast is in
        # force around the forward.
        self.projection = torch.nn.Linear(4, 4, dtype=torch.float32)
        self.observed: list[tuple[torch.dtype, torch.dtype]] = []
        self.cached_context = None
        self.text_loads = 0

    def release_base_transformer(self) -> None:
        return None

    def load_text_encoder(self, device: str) -> None:
        self.text_loads += 1

    def take_refiner_context(self):
        pair = self.cached_context
        self.cached_context = None
        return pair

    def encode_prompt(self, prompt: str, device: str) -> torch.Tensor:
        return torch.zeros(2, 8, dtype=torch.float32)

    def offload_text_encoder(self) -> None:
        return None

    def refiner_residency_request(self):
        return None

    def refiner_forward(self, latents, context):
        self.observed.append(
            (latents.dtype, self.projection(torch.zeros(1, 4)).dtype)
        )
        return torch.zeros_like(latents)


def test_magi2_refine_loop_leaves_the_vendored_dtype_policy_alone() -> None:
    """The refiner denoise runs at float32 with no autocast around it.

    The vendored refiner owns its dtype policy: its ``Adapter`` and
    ``PostAdapter`` linears are float32 and write into float32 buffers through
    masked ``index_put_``. An outer autocast demotes the projections without
    demoting the buffers, and the vendored ``@torch.compile`` forward rejects
    the crossing. The loop therefore declares no compute dtype at all, and both
    the latent it hands the model and a float32 projection evaluated inside the
    forward stay float32.
    """
    import inspect

    from mirai.core.models.magi2_preview.refiner import (
        Magi2RefineSettings,
        run_refine,
    )

    assert "dtype" not in inspect.signature(run_refine).parameters

    settings = Magi2RefineSettings(
        steps=2,
        cfg_scale=2.0,
        shift=5.0,
        height=64,
        width=64,
        noise_index=220,
        scheduler="unipc",
    )
    pipeline = _StubRefinePipeline()
    refined = run_refine(
        pipeline=pipeline,
        refiner=_StubRefinerAssets(),
        base_latent=torch.zeros(4, 3, 4, 4, dtype=torch.float32),
        settings=settings,
        prompt="a",
        negative_prompt="",
        seed=0,
        device="cpu",
    )
    assert tuple(refined.shape) == (4, 5, 4, 4)
    # Two CFG branches per step: the refiner is never packed into one B=2 forward.
    assert len(pipeline.observed) == 2 * settings.steps
    assert pipeline.observed == [
        (torch.float32, torch.float32) for _ in pipeline.observed
    ]
    assert pipeline.text_loads == 1


def test_magi2_refine_loop_uses_paired_cfg_when_provider_exposes_it() -> None:
    from mirai.core.models.magi2_preview.refiner import (
        Magi2RefineSettings,
        run_refine,
    )

    class _PairedPipeline(_StubRefinePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.paired_calls = 0

        def refiner_cfg_forward(self, latents, context, context_null):
            self.paired_calls += 1
            self.observed.extend(
                [
                    (latents.dtype, self.projection(torch.zeros(1, 4)).dtype),
                    (latents.dtype, self.projection(torch.zeros(1, 4)).dtype),
                ]
            )
            return torch.zeros_like(latents), torch.zeros_like(latents)

        def refiner_forward(self, latents, context):
            raise AssertionError("paired CFG must not fall back to separate forwards")

    pipeline = _PairedPipeline()
    settings = Magi2RefineSettings(
        steps=2,
        cfg_scale=2.0,
        shift=5.0,
        height=64,
        width=64,
        noise_index=220,
        scheduler="unipc",
    )
    run_refine(
        pipeline=pipeline,
        refiner=_StubRefinerAssets(),
        base_latent=torch.zeros(4, 3, 4, 4, dtype=torch.float32),
        settings=settings,
        prompt="a",
        negative_prompt="",
        seed=0,
        device="cpu",
    )

    assert pipeline.paired_calls == settings.steps
    assert len(pipeline.observed) == 2 * settings.steps
    assert pipeline.observed == [
        (torch.float32, torch.float32) for _ in pipeline.observed
    ]


def test_magi2_refiner_waits_before_releasing_cuda_workspace(monkeypatch) -> None:
    from mirai.core.models.magi2_preview.refiner import _release_cuda_workspace

    events: list[str] = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append(f"sync:{device.type}")
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))
    device = type("_Device", (), {"type": "cuda"})()

    _release_cuda_workspace(device)

    assert events == ["sync:cuda", "empty"]


def test_magi2_refine_loop_reuses_adjacent_preview_context() -> None:
    from mirai.core.models.magi2_preview.refiner import (
        Magi2RefineSettings,
        run_refine,
    )

    pipeline = _StubRefinePipeline()
    pipeline.cached_context = (
        torch.zeros(1, 2, 8, dtype=torch.float32),
        torch.ones(1, 2, 8, dtype=torch.float32),
    )
    run_refine(
        pipeline=pipeline,
        refiner=_StubRefinerAssets(),
        base_latent=torch.zeros(4, 3, 4, 4, dtype=torch.float32),
        settings=Magi2RefineSettings(
            steps=1,
            cfg_scale=2.0,
            shift=5.0,
            height=64,
            width=64,
            noise_index=220,
            scheduler="unipc",
        ),
        prompt="a",
        negative_prompt="",
        seed=0,
        device="cpu",
    )
    assert pipeline.text_loads == 0
    assert pipeline.cached_context is None


def test_magi2_refiner_residency_staging_runs_outside_the_compiled_forward() -> None:
    """Block-swap staging is a graph break, not traced state.

    Staging rebinds ``Parameter.data`` host-to-device from inside the vendored
    ``@torch.compile(dynamic=True)`` forward, which Dynamo cannot trace across
    two devices. The hooks must therefore route through a dynamo-disabled
    callable while still staging every block eagerly.
    """
    from mirai.core.models.magi2_preview.pipeline import Magi2ResidencyRequest
    from mirai.core.models.magi2_preview.refiner import Magi2Refiner

    refiner = Magi2Refiner(
        type("_ModelConfig", (), {"path": "./models/MAGI-2-preview"})(),
        config_path=_REFINER_CONFIG,
    )
    transformer = torch.nn.Module()
    transformer.block = torch.nn.Module()
    transformer.block.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2) for _ in range(3)]
    )
    refiner._transformer = transformer

    staged: list[tuple[str, int]] = []

    class _RecordingManager:
        def bind(self, units, *, device) -> None:
            return None

        def before_block(self, index: int) -> None:
            staged.append(("in", int(index)))

        def after_block(self, index: int) -> None:
            staged.append(("out", int(index)))

    import mirai.core.training.residency.block_swap as block_swap

    saved = block_swap.BlockSwapManager
    block_swap.BlockSwapManager = lambda **kwargs: _RecordingManager()
    try:
        refiner._place(
            device="cpu",
            residency=Magi2ResidencyRequest(
                enabled=True,
                blocks_to_swap=3,
                mode="async",
                block_residency_planner="static",
                block_swap_prefetch_depth=1,
                block_residency_priority="uniform",
                block_swap_transfer_strategy="default",
                offload_dir=None,
            ),
        )
    finally:
        block_swap.BlockSwapManager = saved

    for index, layer in enumerate(transformer.block.layers):
        layer(torch.zeros(1, 2))
    assert staged == [
        ("in", 0), ("out", 0), ("in", 1), ("out", 1), ("in", 2), ("out", 2)
    ]

    # ``_torchdynamo_disable`` is the marker Dynamo itself reads to decide that a
    # call is a graph break rather than traceable state. Each hook must close
    # over exactly one such callable: the pre-hook stages in, the post-hook out.
    layer = transformer.block.layers[0]
    for hooks in (layer._forward_pre_hooks, layer._forward_hooks):
        hook = next(iter(hooks.values()))
        disabled = [
            cell.cell_contents
            for cell in (hook.__closure__ or ())
            if getattr(cell.cell_contents, "_torchdynamo_disable", False)
        ]
        assert len(disabled) == 1


def test_magi2_provider_accepts_and_validates_refiner_family_params() -> None:
    provider = get_model_family_provider("magi2-preview")
    assert provider is not None
    assert provider.validate_family_params(
        {"refiner_config_path": _REFINER_CONFIG, "refiner_subfolder": "refiner"}
    ) == []
    assert provider.validate_family_params({"refiner_subfolder": "/etc"})
    assert provider.validate_family_params({"refiner_subfolder": "../elsewhere"})
    assert provider.validate_family_params({"refiner_config_path": 3})
    assert provider.validate_family_params({"vae_decode_chunk_size": 0}) == []
    assert provider.validate_family_params({"vae_decode_chunk_size": 2}) == []
    assert provider.validate_family_params({"vae_decode_chunk_size": -1})
    assert provider.validate_family_params({"vae_decode_chunk_size": True})


_REFINER_DEFAULT_OFF_PROBE = """
import importlib
import sys

importlib.import_module("mirai.core.models.magi2_preview.pipeline")
importlib.import_module("mirai.core.models.magi2_preview.refiner")

vendored = [
    name
    for name in sys.modules
    if name in {
        "mirai.vendors.magi2_preview.model.magi2_refiner",
        "mirai.vendors.magi2_preview.pipeline.refiner_data_proxy",
    }
]
assert not vendored, vendored

from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

assert Magi2PreviewPipeline._refiner is None
assert Magi2PreviewPipeline._refine_settings is None
print("refiner-absent")
"""


def test_magi2_refiner_is_absent_until_a_refinement_is_requested() -> None:
    """Default-off means no refiner state and no vendored refiner import.

    The refiner transformer and its data proxy are the two heaviest vendored
    modules of the family; importing them for a run that never refines is a
    cost the default path must not pay.
    """
    result = subprocess.run(
        [sys.executable, "-c", _REFINER_DEFAULT_OFF_PROBE],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert "refiner-absent" in result.stdout


# --- Environment escape hatches ---------------------------------------------


_ENV_MUTATION_PROBE = """
import os

sentinel = dict(os.environ)
import mirai.vendors.magi2_preview.model.magi2_preview  # noqa: F401

changed = {
    key: (sentinel.get(key), value)
    for key, value in os.environ.items()
    if sentinel.get(key) != value
}
removed = sorted(set(sentinel) - set(os.environ))
assert not changed, changed
assert not removed, removed
print("env-unchanged")
"""


def test_importing_the_vendored_model_does_not_mutate_the_environment() -> None:
    """Import-time env writes leak into every child process; there must be none."""
    result = subprocess.run(
        [sys.executable, "-c", _ENV_MUTATION_PROBE],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    )
    if result.returncode != 0 and "ModuleNotFoundError" in result.stderr:
        pytest.skip(
            "vendored MAGI-2 runtime unavailable: "
            + result.stderr.strip().splitlines()[-1]
        )
    assert result.returncode == 0, result.stderr
    assert "env-unchanged" in result.stdout


def test_magi2_load_path_has_no_environment_escape_hatch() -> None:
    """No environment variable may skip the checkpoint read."""
    import mirai.vendors.magi2_preview.infra.checkpoint.load_checkpoint as load_checkpoint
    from mirai.core.models.magi2_preview import pipeline as mirai_pipeline

    for module in (load_checkpoint, mirai_pipeline):
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        for name in ("SKIP_LOAD_MODEL", "MIRAI_MAGI2_SKIP_LOAD"):
            for line in source.splitlines():
                if name in line:
                    assert line.lstrip().startswith("#"), (module.__name__, line)


def test_magi2_evaluation_cfg_defaults_ignore_the_environment() -> None:
    from mirai.vendors.magi2_preview.common.magi2_config import EvaluationConfig

    baseline = EvaluationConfig()
    assert baseline.video_txt_guidance_scale == 5.0
    assert baseline.audio_txt_guidance_scale == 5.0
    assert baseline.use_skimmed_cfg_linear is False
    assert baseline.skimmed_cfg_scale == 5.0
    assert baseline.cfg_rescale == 0.0

    keys = ("VG", "AG", "USE_SKIMMED_CFG_LINEAR", "SKIMMED_CFG_SCALE", "CFG_RESCALE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(
            {
                "VG": "1.5",
                "AG": "2.5",
                "USE_SKIMMED_CFG_LINEAR": "1",
                "SKIMMED_CFG_SCALE": "3.5",
                "CFG_RESCALE": "0.7",
            }
        )
        assert EvaluationConfig() == baseline
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_magi2_router_bias_source_env_is_rejected() -> None:
    """Which bias pairs with the released EMA weights is not a runtime choice."""
    from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import (
        _apply_router_bias_ema,
    )

    key = "block.layers.0.mlp.moe_mlp.router.expert_bias"
    state = {key: torch.zeros(2), key + "_ema": torch.ones(2)}
    assert torch.equal(_apply_router_bias_ema(dict(state))[key], torch.ones(2))

    previous = os.environ.get("MAGI2_ROUTER_BIAS_SOURCE")
    try:
        os.environ["MAGI2_ROUTER_BIAS_SOURCE"] = "main"
        with pytest.raises(RuntimeError, match="expert_bias_ema"):
            _apply_router_bias_ema(dict(state))
    finally:
        if previous is None:
            os.environ.pop("MAGI2_ROUTER_BIAS_SOURCE", None)
        else:
            os.environ["MAGI2_ROUTER_BIAS_SOURCE"] = previous


def test_magi2_streaming_safetensors_load_is_strict(tmp_path) -> None:
    from safetensors.torch import save_file

    from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import (
        load_safetensors_into_model,
    )

    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    expected = {
        key: torch.randn_like(value) if value.is_floating_point() else value.clone()
        for key, value in model.state_dict().items()
    }
    save_file(expected, tmp_path / "model.safetensors")
    load_safetensors_into_model(model, str(tmp_path))
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())

    save_file({"0.weight": expected["0.weight"]}, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="missing keys"):
        load_safetensors_into_model(model, str(tmp_path))


# --- Expert-execution backend precedence ------------------------------------


def test_magi2_configured_backend_wins_over_the_default_triton_path() -> None:
    """An attached backend is the selected path with and without autograd."""
    module, _adapter = _build_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32
    )
    calls: list[str] = []

    class _RecordingBackend:
        def execute(self, owner, x_heads, topk_probs, topk_indices):
            calls.append("backend")
            return owner._torch_forward(x_heads, topk_probs, topk_indices)

    module._mirai_moe_kernel_backend = _RecordingBackend()
    hidden = torch.randn(5, 16)
    with torch.no_grad():
        module._forward_impl(hidden)
    module._forward_impl(hidden)
    assert calls == ["backend", "backend"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_magi2_triton_flash_forward_matches_the_reference_loop() -> None:
    """The fused Triton default must reproduce the vendored per-expert loop.

    ``d_head`` and the expert intermediate follow the kernel's own tiling
    preconditions (multiples of 64 and 32). Both paths run BF16 experts with
    fp32 accumulation but sum the selected experts in a different order, so the
    band is stated against the output scale rather than per element: a few
    near-zero rows carry a large relative error at BF16 while the whole tensor
    stays within one part in fifty of its own magnitude.
    """
    pytest.importorskip("triton", reason="the fused MoE kernel is Triton-only")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 8:
        pytest.skip("the vendored MoE kernel targets SM80+ tensor cores")

    module, _adapter = _build_reduced_moe(
        device=device,
        dtype=torch.bfloat16,
        hidden_size=128,
        expert_intermediate_size=32,
    )
    hidden = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        x_heads = hidden.view(-1, module.num_heads, module.d_head)
        topk_probs, topk_indices = module._route(x_heads)
        reference = module._torch_forward(x_heads, topk_probs, topk_indices).float()
        fused = module._flash_forward(x_heads, topk_probs, topk_indices).float()
    scale = reference.abs().max()
    assert scale > 0.0
    assert (reference - fused).abs().max() <= 2e-2 * scale


def test_magi2_provider_declares_native_cache_encoding() -> None:
    """Raw media must route to the family encoder, not the generic projector.

    The provider implements ``build_native_cache_encoder``; without the matching
    capability the cache builder disables native encoding and the generic media
    preprocessor rejects every raw video.
    """
    provider = get_model_family_provider("magi2-preview")
    assert provider is not None
    assert provider.supports_native_cache_encoding()
    assert not provider.allows_asset_free_cache_encoding()
    encoder = provider.build_native_cache_encoder(
        NativeCacheEncoderConfig(
            enabled=True,
            model_type="magi2-preview",
            variant="preview",
            model_path="./models/MAGI-2-preview",
            dtype_name="bf16",
            max_frames=17,
        )
    )
    assert encoder is not None
    assert int(encoder.latent_channels) == 48


def test_magi2_cache_encoder_satisfies_the_native_cache_encoder_contract(
    tmp_path: pathlib.Path,
) -> None:
    """The cache loop duck-types the encoder; conformance must be provable here.

    A member missing from the encoder empties the cache one silent skip at a
    time, so the contract is validated when the encoder is constructed.
    """
    from mirai.core.dataset.native_encode import validate_native_cache_encoder
    from mirai.core.models.magi2_preview.cache import Magi2PreviewNativeCacheEncoder

    provider = get_model_family_provider("magi2-preview")
    assert provider is not None
    encoder = provider.build_native_cache_encoder(
        NativeCacheEncoderConfig(
            enabled=True,
            model_type="magi2-preview",
            variant="preview",
            model_path="./models/MAGI-2-preview",
            dtype_name="bf16",
            max_frames=17,
        )
    )
    assert isinstance(encoder, Magi2PreviewNativeCacheEncoder)
    assert validate_native_cache_encoder(encoder, source="test") is encoder
    # MAGI-2 has no CLIP conditioning: the contract member exists and is a no-op.
    assert encoder.encode_clip(tmp_path / "clip.mp4") is None


class _RecordingVae:
    """Stands in for the Wan2.2 VAE: records its input and honours its strides."""

    def __init__(self) -> None:
        self.vae = torch.nn.Linear(1, 1)
        self.seen: list[tuple[int, ...]] = []

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        self.seen.append(tuple(video.shape))
        batch, _channels, frames, height, width = video.shape
        latent_frames = 1 + (int(frames) - 1) // 8
        return torch.zeros(batch, 48, latent_frames, height // 16, width // 16)


def _magi2_cache_encoder(
    monkeypatch: pytest.MonkeyPatch,
    source: torch.Tensor,
    **bucket_config: object,
) -> tuple[object, _RecordingVae]:
    """A MAGI-2 cache encoder over a synthetic clip, with the heavy parts stubbed."""
    from mirai.core.models.magi2_preview import cache as cache_module

    encoder = cache_module.Magi2PreviewNativeCacheEncoder(
        NativeCacheEncoderConfig(
            enabled=True,
            model_type="magi2-preview",
            variant="preview",
            model_path="./models/MAGI-2-preview",
            dtype_name="bf16",
            max_frames=17,
            **bucket_config,  # type: ignore[arg-type]
        )
    )
    vae = _RecordingVae()
    monkeypatch.setattr(
        cache_module, "_load_video_media", lambda path, max_frames: source.clone()
    )
    monkeypatch.setattr(
        cache_module.Magi2PreviewNativeCacheEncoder, "_load_vae", lambda self: vae
    )
    return encoder, vae


def _synthetic_clip(frames: int, height: int, width: int) -> torch.Tensor:
    """A [C, T, H, W] byte-range clip whose content varies across every axis."""
    torch.manual_seed(3)
    return torch.randint(0, 256, (3, frames, height, width)).float()


def test_magi2_cache_encoding_lands_on_the_configured_resolution_bucket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A clip wider than its bucket is resized to the bucket, not cached native.

    Caching at source resolution is what silently inflates the token count: a
    2.4:1 source against a 1:2 bucket carries several times the sequence length
    the training geometry declares, and the mismatch only surfaces as an
    attention allocation far downstream.
    """
    from mirai.core.dataset.media.media_resize import resize_crop_tensor

    source = _synthetic_clip(frames=17, height=80, width=192)
    encoder, vae = _magi2_cache_encoder(
        monkeypatch,
        source,
        enable_bucketing=True,
        resolution_buckets=[(32, 64)],
        frame_buckets=[17],
    )

    latent, bucket = encoder.encode_latent(tmp_path / "clip.mp4")

    # The VAE saw the bucket, not the source.
    assert vae.seen == [(1, 3, 17, 32, 64)]
    assert tuple(latent.shape) == (48, 3, 2, 4)
    # Lineage records the bucket the sample was actually encoded at.
    assert (bucket.bucket_h, bucket.bucket_w, bucket.bucket_frames) == (32, 64, 17)
    assert bucket.bucket_id == "32x64x17"

    # The mismatched aspect follows the shared convention exactly: cover-scale on
    # the shorter side, then center-crop. Anything else (a stretch, or a corner
    # crop) would change the framing the dataset layer already assumes.
    expected = resize_crop_tensor(source, 32, 64, mode="resize_crop")
    prepared, _bucket = encoder._prepare_video_for_vae(tmp_path / "clip.mp4")
    torch.testing.assert_close(prepared, expected)


def test_magi2_cache_without_buckets_keeps_the_source_on_the_latent_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """With no configured bucket the clip is only cropped onto the patch grid."""
    source = _synthetic_clip(frames=17, height=70, width=100)
    encoder, vae = _magi2_cache_encoder(monkeypatch, source)

    _latent, bucket = encoder.encode_latent(tmp_path / "clip.mp4")

    assert vae.seen == [(1, 3, 17, 64, 96)]
    assert (bucket.bucket_h, bucket.bucket_w) == (64, 96)


def test_magi2_cache_rejects_a_bucket_off_the_latent_spatial_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A bucket the latent grid cannot express fails instead of being encoded."""
    from mirai.core.models.magi2_preview.cache import MAGI2_SPATIAL_STRIDE

    assert (
        MAGI2_SPATIAL_STRIDE
        == _bare_pipeline().get_video_latent_layout().spatial_downsample
    )

    source = _synthetic_clip(frames=17, height=80, width=192)
    encoder, vae = _magi2_cache_encoder(
        monkeypatch,
        source,
        enable_bucketing=True,
        resolution_buckets=[(40, 72)],
        frame_buckets=[17],
    )
    with pytest.raises(ValueError, match="multiple of 16"):
        encoder.encode_latent(tmp_path / "clip.mp4")
    assert vae.seen == []


def test_magi2_cache_loop_produces_records_without_clip_conditioning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A MAGI-2 raw-media cache build must yield records, not 20 silent skips."""
    from mirai.core.dataset import cache as cache_module
    from mirai.core.dataset.native_encode import BucketInfo
    from mirai.core.models.magi2_preview.cache import Magi2PreviewNativeCacheEncoder

    def _fake_latent(
        self: Magi2PreviewNativeCacheEncoder, media_path: pathlib.Path
    ) -> tuple[torch.Tensor, BucketInfo]:
        _ = media_path
        return (
            torch.zeros(self.latent_channels, 1, 2, 2, dtype=torch.float32),
            BucketInfo(bucket_id="32x32x1", bucket_h=32, bucket_w=32, bucket_frames=1),
        )

    monkeypatch.setattr(Magi2PreviewNativeCacheEncoder, "encode_latent", _fake_latent)
    monkeypatch.setattr(
        Magi2PreviewNativeCacheEncoder,
        "encode_text",
        lambda self, caption: torch.zeros(4, 8, dtype=torch.float32),
    )

    dataset_dir = tmp_path / "videos"
    dataset_dir.mkdir()
    (dataset_dir / "clip0.mp4").write_bytes(b"")
    (dataset_dir / "clip0.txt").write_text("a caption", encoding="utf-8")

    payload = cache_module.build_cache(
        dataset_dir,
        tmp_path / "cache.pt",
        native_encode=True,
        model_type="magi2-preview",
        model_variant="preview",
        model_path=str(tmp_path / "model"),
    )
    assert int(payload["num_records"]) == 1
    assert int(payload["num_skipped"]) == 0
    assert payload["records"][0]["clip_embed"] is None


def test_magi2_auto_preprocess_defers_to_the_native_cache_encoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """`preprocess_raw_media_to_pt` must not reach the generic pixel projector."""
    from mirai.core.training.data import preprocessing

    config = load_config("configs/magi2_preview/train_offload.toml")
    assert config.dataset.auto_preprocess_cache
    assert config.dataset.preprocess_raw_media_to_pt
    dataset_dir = tmp_path / "videos"
    dataset_dir.mkdir()
    (dataset_dir / "clip.mp4").write_bytes(b"")
    config = dataclasses.replace(
        config,
        dataset=dataclasses.replace(
            config.dataset,
            path=str(dataset_dir),
            cache_path=str(tmp_path / "cache" / "dataset_cache.pt"),
        ),
    )

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("generic media preprocessing must not run for MAGI-2")

    built: list[object] = []
    monkeypatch.setattr(preprocessing, "preprocess_video_media_to_pt", _fail)
    monkeypatch.setattr(
        preprocessing, "build_cache_from_config", lambda cfg: built.append(cfg)
    )
    preprocessing.ensure_cache_artifacts_for_training(config)
    assert built == [config]


def test_magi2_dry_run_batch_satisfies_the_pipeline_conditioning_contract() -> None:
    """The synthetic dry-run batch must pass MAGI-2's own batch validation.

    The transformer is never built: the validated surfaces are the latent
    channel check the forward performs and the Qwen3.5 width check in
    ``_text_features``, both of which reject the generic dry-run shapes.
    """
    from mirai.core.models.magi2_preview.pipeline import (
        MAGI2_TEXT_EMBED_WIDTH,
        Magi2PreviewPipeline,
    )
    from mirai.core.training.runtime.dry_run import _build_synthetic_moe_batch

    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    config = load_config("configs/magi2_preview/train_offload.toml")
    batch = _build_synthetic_moe_batch(config, pipeline=pipeline, dtype=torch.float32)

    latents = batch["latents"]
    assert latents.ndim == 5
    assert int(latents.shape[0]) == int(config.training.batch_size)
    assert int(latents.shape[1]) == 48
    text, lengths = pipeline._text_features(
        batch["text_embeds"], batch=int(latents.shape[0]), like=latents
    )
    assert int(text.shape[-1]) == MAGI2_TEXT_EMBED_WIDTH
    assert int(text.shape[0]) == int(latents.shape[0])
    assert int(lengths.sum()) == int(text.shape[0]) * int(text.shape[1])


def test_dry_run_batch_keeps_generic_shapes_without_a_model_spec() -> None:
    """Models that declare no synthetic-batch spec keep the previous shapes."""
    from mirai.core.training.runtime.dry_run import _build_synthetic_moe_batch

    config = load_config("configs/lingbot_video/train_bf16.toml")
    batch = _build_synthetic_moe_batch(config, pipeline=None, dtype=torch.float32)
    frames = max(1, int(list(config.dataset.frame_buckets or [1])[0]))
    patch = max(1, int(config.model.params.patch_size))
    assert tuple(batch["latents"].shape) == (
        int(config.training.batch_size),
        int(config.model.params.latent_channels),
        frames,
        max(patch * 2, 2),
        max(patch * 2, 2),
    )
    assert tuple(batch["text_embeds"].shape) == (int(config.training.batch_size),)

    class _PipelineWithoutSpec:
        pass

    unspecified = _build_synthetic_moe_batch(
        config, pipeline=_PipelineWithoutSpec(), dtype=torch.float32
    )
    assert tuple(unspecified["latents"].shape) == tuple(batch["latents"].shape)
    assert tuple(unspecified["text_embeds"].shape) == tuple(batch["text_embeds"].shape)
