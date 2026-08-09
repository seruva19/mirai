"""Model-agnostic expert tensor and MoE memory policy contracts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable

from mirai.core.moe.runtime.gemm import (
    normalize_moe_gemm_backend,
    normalize_moe_gemm_role_backend,
)
from mirai.core.moe.runtime.kernels import normalize_moe_kernel_backend


MOE_DISPATCH_MODES = ("vectorized", "legacy", "triton", "triton_persistent")

# Select where count, offset, and stable-order dispatch preprocessing runs.
# ``host`` assembles per-chunk counts on the host; ``device`` and ``sonic`` use
# device-side planning with one allocation-size transfer.
MOE_DISPATCH_PREPROCESS_MODES = ("host", "device", "sonic")

EXPERT_TENSOR_ROLES = {
    "unknown",
    "router",
    "input",
    "gate",
    "up",
    "down",
    "shared_gate",
    "shared_up",
    "shared_down",
}

EXPERT_TENSOR_AXES = {"expert", "out", "in", "hidden", "intermediate"}

EXPERT_MLP_ACTIVATIONS = {"silu", "gelu", "relu", "identity"}
EXPERT_MLP_COMBINERS = {"gated_product", "activated"}

EXPERT_WEIGHT_ACCESS_POLICIES = {
    "auto",
    "disabled",
    "full_dequant",
    "active_dequant",
    "chunked_dequant",
    "fused_kernel",
}

ROUTER_QUANTIZATION_POLICIES = {"disabled", "int8_per_channel"}

PACKED_STATE_PRELOAD_MODES = ("off", "ram", "pinned")
PACKED_STREAM_BACKENDS = ("staged", "gds")
MOE_EXPERT_AUTOGRAD_MODES = ("standard", "segmented_recompute")

@dataclass(frozen=True)
class ExpertTensorSpec:
    """Concrete expert tensor declaration owned by a native model pipeline."""

    name: str
    owner_module: str
    tensor_name: str
    role: str
    layout: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str = ""
    quantizable: bool = True
    adapter_targetable: bool = True
    routed: bool = True
    shared: bool = False
    router: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "owner_module", str(self.owner_module))
        object.__setattr__(self, "tensor_name", str(self.tensor_name))
        object.__setattr__(self, "role", str(self.role).strip().lower())
        object.__setattr__(self, "layout", tuple(str(axis) for axis in self.layout))
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        if not self.name:
            raise ValueError("ExpertTensorSpec.name must be non-empty.")
        if not self.tensor_name:
            raise ValueError(f"ExpertTensorSpec '{self.name}' has empty tensor_name.")
        if self.role not in EXPERT_TENSOR_ROLES:
            supported = ", ".join(sorted(EXPERT_TENSOR_ROLES))
            raise ValueError(
                f"ExpertTensorSpec '{self.name}' has unsupported role "
                f"'{self.role}'. Supported: {supported}."
            )
        if len(self.layout) != len(self.shape):
            raise ValueError(
                f"ExpertTensorSpec '{self.name}' layout rank {len(self.layout)} "
                f"does not match shape rank {len(self.shape)}."
            )
        unknown_axes = [axis for axis in self.layout if axis not in EXPERT_TENSOR_AXES]
        if unknown_axes:
            supported = ", ".join(sorted(EXPERT_TENSOR_AXES))
            raise ValueError(
                f"ExpertTensorSpec '{self.name}' has unsupported axes "
                f"{unknown_axes}. Supported: {supported}."
            )
        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"ExpertTensorSpec '{self.name}' shape must be positive.")
        if self.routed and "expert" not in self.layout and not self.router:
            raise ValueError(
                f"ExpertTensorSpec '{self.name}' is routed but has no expert axis."
            )


@dataclass(frozen=True)
class ExpertProjectionRole:
    """Bind one semantic expert-MLP role to a model-owned tensor name."""

    role: str
    tensor_name: str

    def __post_init__(self) -> None:
        role = str(self.role).strip().lower()
        tensor_name = str(self.tensor_name).strip()
        if role not in {"input", "gate", "up", "down"}:
            raise ValueError(f"Unsupported expert projection role {role!r}.")
        if not tensor_name:
            raise ValueError("Expert projection tensor_name must be non-empty.")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "tensor_name", tensor_name)


@dataclass(frozen=True)
class ExpertMLPExecutionSpec:
    """Model-owned projection layout and nonlinearity for grouped experts."""

    projections: tuple[ExpertProjectionRole, ...]
    activation: str = "silu"
    combiner: str = "gated_product"

    def __post_init__(self) -> None:
        projections = tuple(self.projections)
        roles = [projection.role for projection in projections]
        names = [projection.tensor_name for projection in projections]
        if len(roles) != len(set(roles)):
            raise ValueError("Expert MLP projection roles must be unique.")
        if len(names) != len(set(names)):
            raise ValueError("Expert MLP tensor names must be unique.")
        activation = str(self.activation).strip().lower()
        combiner = str(self.combiner).strip().lower()
        if activation not in EXPERT_MLP_ACTIVATIONS:
            raise ValueError(
                "Expert MLP activation must be one of: "
                + ", ".join(sorted(EXPERT_MLP_ACTIVATIONS))
                + "."
            )
        if combiner not in EXPERT_MLP_COMBINERS:
            raise ValueError(
                "Expert MLP combiner must be one of: "
                + ", ".join(sorted(EXPERT_MLP_COMBINERS))
                + "."
            )
        required = (
            {"gate", "up", "down"}
            if combiner == "gated_product"
            else {"input", "down"}
        )
        if set(roles) != required:
            raise ValueError(
                f"Expert MLP combiner {combiner!r} requires roles "
                f"{sorted(required)}, got {sorted(roles)}."
            )
        object.__setattr__(self, "projections", projections)
        object.__setattr__(self, "activation", activation)
        object.__setattr__(self, "combiner", combiner)

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(projection.tensor_name for projection in self.projections)

    @property
    def uses_gated_product(self) -> bool:
        return self.combiner == "gated_product"

    def tensor_for_role(self, role: str) -> str:
        normalized = str(role).strip().lower()
        for projection in self.projections:
            if projection.role == normalized:
                return projection.tensor_name
        raise KeyError(f"Expert MLP layout has no {normalized!r} projection.")

    @classmethod
    def from_tensor_specs(
        cls,
        specs: Iterable[ExpertTensorSpec],
        *,
        activation: str = "silu",
        combiner: str | None = None,
    ) -> "ExpertMLPExecutionSpec":
        routed = tuple(
            spec
            for spec in specs
            if spec.routed and not spec.router and spec.role in {"input", "gate", "up", "down"}
        )
        roles = {spec.role for spec in routed}
        resolved_combiner = str(combiner or "").strip().lower()
        if not resolved_combiner:
            if roles == {"gate", "up", "down"}:
                resolved_combiner = "gated_product"
            elif roles == {"input", "down"}:
                resolved_combiner = "activated"
            else:
                raise ValueError(
                    "Cannot infer expert MLP combiner from roles "
                    f"{sorted(roles)}."
                )
        return cls(
            projections=tuple(
                ExpertProjectionRole(spec.role, spec.tensor_name) for spec in routed
            ),
            activation=activation,
            combiner=resolved_combiner,
        )


CANONICAL_PACKED_EXPERT_MLP_SPEC = ExpertMLPExecutionSpec(
    projections=(
        ExpertProjectionRole("gate", "w1"),
        ExpertProjectionRole("up", "w3"),
        ExpertProjectionRole("down", "w2"),
    ),
    activation="silu",
    combiner="gated_product",
)


@dataclass(frozen=True)
class MoEOptimizationPolicy:
    """Memory/runtime policy for generic routed-expert training."""

    expert_weight_access: str = "auto"
    expert_dequant_chunk_size: int = 0
    expert_device_cache_gib: float = 0.0
    device_residency_budget_gib: float = 0.0
    quantize_experts_on_load: bool = False
    router_quantization: str = "disabled"
    router_quantization_calibration_path: str = ""
    kernel_backend: str = "auto"
    packed_state_preload: str = "pinned"
    packed_stream_cache_gib: float = 0.0
    packed_stream_backend: str = "staged"
    packed_stream_prefetch_depth: int = 0
    # Kernel and dispatch behavior is controlled only by the typed memory config.
    moe_dispatch: str = "vectorized"
    moe_dispatch_preprocess: str = "host"
    moe_gemm_backend: str = "auto"
    moe_gemm_backend_forward: str = ""
    moe_gemm_backend_dx: str = ""
    moe_gemm_backend_dw: str = ""
    moe_batched_dequant: bool = True
    moe_pair_dequant: bool = True
    moe_batched_gather: bool = False
    moe_expert_autograd: str = "standard"
    packed_shard_size_mb: int = 2048
    int8_workspace_mb: int = 0

    def __post_init__(self) -> None:
        access = normalize_expert_weight_access_policy(self.expert_weight_access)
        router = normalize_router_quantization_policy(self.router_quantization)
        router_calibration_path = str(
            self.router_quantization_calibration_path
        ).strip()
        kernel_backend = normalize_moe_kernel_backend(self.kernel_backend)
        preload = normalize_packed_state_preload(self.packed_state_preload)
        stream_backend = normalize_packed_stream_backend(self.packed_stream_backend)
        dispatch = normalize_moe_dispatch_mode(self.moe_dispatch)
        dispatch_preprocess = normalize_moe_dispatch_preprocess(
            self.moe_dispatch_preprocess
        )
        gemm_backend = normalize_moe_gemm_backend(self.moe_gemm_backend)
        gemm_forward = normalize_moe_gemm_role_backend(self.moe_gemm_backend_forward)
        gemm_dx = normalize_moe_gemm_role_backend(self.moe_gemm_backend_dx)
        gemm_dw = normalize_moe_gemm_role_backend(self.moe_gemm_backend_dw)
        chunk = int(self.expert_dequant_chunk_size)
        device_cache_gib = float(self.expert_device_cache_gib)
        residency_budget_gib = float(self.device_residency_budget_gib)
        shard_mb = int(self.packed_shard_size_mb)
        workspace_mb = int(self.int8_workspace_mb)
        stream_cache_gib = float(self.packed_stream_cache_gib)
        stream_prefetch_depth = int(self.packed_stream_prefetch_depth)
        expert_autograd = str(self.moe_expert_autograd).strip().lower()
        if expert_autograd not in MOE_EXPERT_AUTOGRAD_MODES:
            raise ValueError(
                "memory.moe_expert_autograd must be one of: "
                + ", ".join(MOE_EXPERT_AUTOGRAD_MODES)
                + "."
            )
        if access == "chunked_dequant" and chunk <= 0:
            raise ValueError(
                "memory.expert_dequant_chunk_size must be > 0 when "
                "memory.expert_weight_access='chunked_dequant'."
            )
        if chunk < 0:
            raise ValueError("memory.expert_dequant_chunk_size must be >= 0.")
        if device_cache_gib < 0.0:
            raise ValueError("memory.expert_device_cache_gib must be >= 0.")
        if residency_budget_gib < 0.0:
            raise ValueError("memory.device_residency_budget_gib must be >= 0.")
        if shard_mb <= 0:
            raise ValueError("memory.packed_shard_size_mb must be > 0.")
        if workspace_mb < 0:
            raise ValueError("memory.int8_workspace_mb must be >= 0.")
        if stream_cache_gib < 0.0:
            raise ValueError("memory.packed_stream_cache_gib must be >= 0.")
        if stream_prefetch_depth < 0 or stream_prefetch_depth > 16:
            raise ValueError(
                "memory.packed_stream_prefetch_depth must be between 0 and 16."
            )
        if router_calibration_path and router != "int8_per_channel":
            raise ValueError(
                "memory.router_quantization_calibration_path requires "
                "memory.router_quantization='int8_per_channel'."
            )
        if stream_cache_gib > 0.0 and preload != "off":
            raise ValueError(
                "memory.packed_stream_cache_gib requires "
                "memory.packed_state_preload='off'."
            )
        if stream_backend != "staged" and preload != "off":
            raise ValueError(
                "memory.packed_stream_backend='gds' requires "
                "memory.packed_state_preload='off'."
            )
        if stream_backend == "gds" and stream_cache_gib > 0.0:
            raise ValueError(
                "memory.packed_stream_cache_gib must be 0 when "
                "memory.packed_stream_backend='gds'."
            )
        object.__setattr__(self, "expert_weight_access", access)
        object.__setattr__(self, "expert_dequant_chunk_size", chunk)
        object.__setattr__(self, "expert_device_cache_gib", device_cache_gib)
        object.__setattr__(
            self, "device_residency_budget_gib", residency_budget_gib
        )
        object.__setattr__(self, "router_quantization", router)
        object.__setattr__(
            self,
            "router_quantization_calibration_path",
            router_calibration_path,
        )
        object.__setattr__(self, "kernel_backend", kernel_backend)
        object.__setattr__(self, "packed_state_preload", preload)
        object.__setattr__(self, "packed_stream_cache_gib", stream_cache_gib)
        object.__setattr__(self, "packed_stream_backend", stream_backend)
        object.__setattr__(
            self, "packed_stream_prefetch_depth", stream_prefetch_depth
        )
        object.__setattr__(self, "quantize_experts_on_load", bool(self.quantize_experts_on_load))
        object.__setattr__(self, "moe_dispatch", dispatch)
        object.__setattr__(self, "moe_dispatch_preprocess", dispatch_preprocess)
        object.__setattr__(self, "moe_gemm_backend", gemm_backend)
        object.__setattr__(self, "moe_gemm_backend_forward", gemm_forward)
        object.__setattr__(self, "moe_gemm_backend_dx", gemm_dx)
        object.__setattr__(self, "moe_gemm_backend_dw", gemm_dw)
        object.__setattr__(self, "moe_batched_dequant", bool(self.moe_batched_dequant))
        object.__setattr__(self, "moe_pair_dequant", bool(self.moe_pair_dequant))
        object.__setattr__(self, "moe_batched_gather", bool(self.moe_batched_gather))
        object.__setattr__(self, "moe_expert_autograd", expert_autograd)
        object.__setattr__(self, "packed_shard_size_mb", shard_mb)
        object.__setattr__(self, "int8_workspace_mb", workspace_mb)

    @classmethod
    def from_memory_config(cls, memory: Any) -> MoEOptimizationPolicy:
        return cls(
            expert_weight_access=getattr(memory, "expert_weight_access", "auto"),
            expert_dequant_chunk_size=int(getattr(memory, "expert_dequant_chunk_size", 0)),
            expert_device_cache_gib=float(
                getattr(memory, "expert_device_cache_gib", 0.0)
            ),
            device_residency_budget_gib=float(
                getattr(memory, "device_residency_budget_gib", 0.0)
            ),
            quantize_experts_on_load=bool(getattr(memory, "quantize_experts_on_load", False)),
            router_quantization=getattr(memory, "router_quantization", "disabled"),
            router_quantization_calibration_path=getattr(
                memory,
                "router_quantization_calibration_path",
                "",
            ),
            kernel_backend=getattr(memory, "moe_kernel_backend", "auto"),
            packed_state_preload=getattr(memory, "packed_state_preload", "pinned"),
            packed_stream_cache_gib=float(
                getattr(memory, "packed_stream_cache_gib", 0.0)
            ),
            packed_stream_backend=getattr(
                memory, "packed_stream_backend", "staged"
            ),
            packed_stream_prefetch_depth=int(
                getattr(memory, "packed_stream_prefetch_depth", 0)
            ),
            moe_dispatch=getattr(memory, "moe_dispatch", "vectorized"),
            moe_dispatch_preprocess=getattr(
                memory, "moe_dispatch_preprocess", "host"
            ),
            moe_gemm_backend=getattr(memory, "moe_gemm_backend", "auto"),
            moe_gemm_backend_forward=getattr(
                memory, "moe_gemm_backend_forward", ""
            ),
            moe_gemm_backend_dx=getattr(memory, "moe_gemm_backend_dx", ""),
            moe_gemm_backend_dw=getattr(memory, "moe_gemm_backend_dw", ""),
            moe_batched_dequant=bool(getattr(memory, "moe_batched_dequant", True)),
            moe_pair_dequant=bool(getattr(memory, "moe_pair_dequant", True)),
            moe_batched_gather=bool(getattr(memory, "moe_batched_gather", False)),
            moe_expert_autograd=getattr(memory, "moe_expert_autograd", "standard"),
            packed_shard_size_mb=int(getattr(memory, "packed_shard_size_mb", 2048)),
            int8_workspace_mb=int(getattr(memory, "int8_workspace_mb", 0)),
        )

    def requests_runtime_behavior(self) -> bool:
        return (
            self.expert_weight_access not in {"", "auto", "disabled"}
            or self.expert_device_cache_gib > 0.0
            or self.device_residency_budget_gib > 0.0
            or self.quantize_experts_on_load
            or self.router_quantization != "disabled"
            or bool(self.router_quantization_calibration_path)
            or self.kernel_backend not in {"", "auto", "torch"}
        )


def normalize_expert_weight_access_policy(value: str | None) -> str:
    text = str(value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "none": "disabled",
        "off": "disabled",
        "full": "full_dequant",
        "dequant": "full_dequant",
        "active": "active_dequant",
        "selective": "active_dequant",
        "selective_dequant": "active_dequant",
        "chunked": "chunked_dequant",
        "chunk": "chunked_dequant",
        "fused": "fused_kernel",
        "kernel": "fused_kernel",
    }
    normalized = aliases.get(text, text)
    if normalized not in EXPERT_WEIGHT_ACCESS_POLICIES:
        supported = ", ".join(sorted(EXPERT_WEIGHT_ACCESS_POLICIES))
        raise ValueError(
            "memory.expert_weight_access must be one of: "
            f"{supported}."
        )
    return normalized


def normalize_router_quantization_policy(value: str | None) -> str:
    text = str(value or "disabled").strip().lower()
    aliases = {
        "": "disabled",
        "none": "disabled",
        "off": "disabled",
        "int8": "int8_per_channel",
        "per_channel_int8": "int8_per_channel",
    }
    normalized = aliases.get(text, text)
    if normalized not in ROUTER_QUANTIZATION_POLICIES:
        supported = ", ".join(sorted(ROUTER_QUANTIZATION_POLICIES))
        raise ValueError(
            "memory.router_quantization must be one of: "
            f"{supported}."
        )
    return normalized


def normalize_moe_dispatch_mode(value: str | None) -> str:
    text = str(value if value is not None else "vectorized").strip().lower()
    if text == "":
        return "vectorized"
    if text not in MOE_DISPATCH_MODES:
        raise ValueError(
            "memory.moe_dispatch must be one of: " + ", ".join(MOE_DISPATCH_MODES) + "."
        )
    return text


def normalize_moe_dispatch_preprocess(value: str | None) -> str:
    text = str(value if value is not None else "host").strip().lower()
    if text == "":
        return "host"
    if text not in MOE_DISPATCH_PREPROCESS_MODES:
        raise ValueError(
            "memory.moe_dispatch_preprocess must be one of: "
            + ", ".join(MOE_DISPATCH_PREPROCESS_MODES)
            + "."
        )
    return text


# ---------------------------------------------------------------------------
# Active MoE performance policy
# ---------------------------------------------------------------------------
# The trainer publishes the policy parsed from ``[memory]``. Direct contract
# probes that do not construct a trainer receive the default policy.

_ACTIVE_MOE_OPTIMIZATION_POLICY: "MoEOptimizationPolicy | None" = None


def set_active_moe_optimization_policy(policy: "MoEOptimizationPolicy | None") -> None:
    global _ACTIVE_MOE_OPTIMIZATION_POLICY
    _ACTIVE_MOE_OPTIMIZATION_POLICY = policy


@contextmanager
def active_moe_optimization_policy(policy: "MoEOptimizationPolicy"):
    """Temporarily publish a policy for an isolated runtime contract probe."""

    previous = _ACTIVE_MOE_OPTIMIZATION_POLICY
    set_active_moe_optimization_policy(policy)
    try:
        yield policy
    finally:
        set_active_moe_optimization_policy(previous)


def get_active_moe_optimization_policy() -> "MoEOptimizationPolicy":
    return _ACTIVE_MOE_OPTIMIZATION_POLICY or MoEOptimizationPolicy()


def resolve_moe_dispatch_mode() -> str:
    return get_active_moe_optimization_policy().moe_dispatch


def resolve_moe_dispatch_preprocess() -> str:
    return get_active_moe_optimization_policy().moe_dispatch_preprocess


def resolve_moe_batched_dequant() -> bool:
    return get_active_moe_optimization_policy().moe_batched_dequant


def resolve_moe_pair_dequant() -> bool:
    return get_active_moe_optimization_policy().moe_pair_dequant


def resolve_moe_batched_gather() -> bool:
    return get_active_moe_optimization_policy().moe_batched_gather


def resolve_packed_shard_size_bytes() -> int:
    mb = int(get_active_moe_optimization_policy().packed_shard_size_mb)
    if mb <= 0:
        raise ValueError("memory.packed_shard_size_mb must be > 0.")
    return mb * 1024 * 1024


def resolve_quantization_workspace_bytes(default_bytes: int) -> int:
    mb = int(get_active_moe_optimization_policy().int8_workspace_mb)
    if mb < 0:
        raise ValueError("memory.int8_workspace_mb must be >= 0.")
    if mb == 0:
        return int(default_bytes)
    return mb * 1024 * 1024


def normalize_packed_state_preload(value: str | None) -> str:
    """Normalize memory.packed_state_preload (off|ram|pinned).

    ``off`` serves disk-backed packed experts from safetensors slicing; ``ram``
    preloads them into pageable CPU RAM once; ``pinned`` preloads into page-locked
    CPU RAM for asynchronous host-to-device copies.
    """
    text = str(value or "off").strip().lower()
    aliases = {
        "": "off",
        "none": "off",
        "disabled": "off",
        "false": "off",
        "0": "off",
        "disk": "off",
        "cpu": "ram",
        "pageable": "ram",
        "pin": "pinned",
        "pinned_ram": "pinned",
    }
    normalized = aliases.get(text, text)
    if normalized not in PACKED_STATE_PRELOAD_MODES:
        raise ValueError(
            "memory.packed_state_preload must be one of: "
            + ", ".join(PACKED_STATE_PRELOAD_MODES)
            + "."
        )
    return normalized


def normalize_packed_stream_backend(value: str | None) -> str:
    """Normalize the explicit packed disk reader implementation."""

    text = str(value or "staged").strip().lower()
    aliases = {"direct_range": "staged", "kvikio": "gds"}
    normalized = aliases.get(text, text)
    if normalized not in PACKED_STREAM_BACKENDS:
        raise ValueError(
            "memory.packed_stream_backend must be one of: "
            + ", ".join(PACKED_STREAM_BACKENDS)
            + "."
        )
    return normalized


def validate_expert_tensor_specs(specs: Iterable[ExpertTensorSpec]) -> tuple[ExpertTensorSpec, ...]:
    result = tuple(specs)
    names = [spec.name for spec in result]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate expert tensor specs: {duplicates}.")
    return result


def expert_tensor_specs_by_name(
    specs: Iterable[ExpertTensorSpec],
) -> dict[str, ExpertTensorSpec]:
    return {spec.name: spec for spec in validate_expert_tensor_specs(specs)}
