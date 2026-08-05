from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import sys

import pytest
import torch
from torch.nn.utils import parametrize

from mirai.config.loader import load_config
from mirai.core.models.magi2_preview.grouped_moe import (
    _CONSUMED_POLICY_FIELDS,
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
from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    module, adapter = _build_reduced_moe(
        device=device,
        dtype=dtype,
        hidden_size=hidden_size,
        expert_intermediate_size=expert_intermediate_size,
    )
    if backend is not None:
        module._mirai_moe_kernel_backend = Magi2GroupedMoEBackend(
            Magi2GroupedMoEPlan(forward_backend=backend, dx_backend=backend)
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
    assert set(_NON_DEFAULT_POLICY_VALUES) | set(_CONSUMED_POLICY_FIELDS) == declared


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
    assert not capabilities.expert_weight_access_policy


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

    pipeline.configure_moe_optimization_policy(
        MoEOptimizationPolicy(kernel_backend="grouped", moe_gemm_backend="bmm")
    )
    assert isinstance(module._mirai_moe_kernel_backend, Magi2GroupedMoEBackend)
    assert module._mirai_moe_kernel_backend.plan.forward_backend == "bmm"

    pipeline.configure_moe_optimization_policy(MoEOptimizationPolicy())
    assert module._mirai_moe_kernel_backend is None


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
