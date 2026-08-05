# SPDX-License-Identifier: Apache-2.0
"""Grouped-GEMM backend registry: a role-aware selection + probe layer that sits
BELOW ``memory.moe_dispatch``.

Where ``moe_dispatch`` picks the whole dispatch strategy (vectorized / legacy /
triton / triton_persistent), this registry names the underlying GEMM PRIMITIVE
for the three matmul roles of a routed-expert MLP independently:

* ``forward``   -- the expert weight matmul in the forward pass,
* ``dx``        -- the input-gradient matmul in backward,
* ``dw``        -- the adapter (LoRA) weight-gradient matmul.

Backends (``MOE_GEMM_BACKENDS``):

* ``bmm``           -- padded ``torch.bmm`` (device-agnostic).
* ``persistent``    -- the vendored sorted-contiguous persistent Triton kernel
                       (CUDA + Triton + SM80+).
* ``torch_grouped`` -- the public ``torch.nn.functional.grouped_mm`` op (probed;
                       falls back to the private ``torch._grouped_mm`` when the
                       public name is absent). BF16 CUDA SM80+ jagged-offset
                       contract, plus a 16-byte operand stride precondition
                       checked by :func:`grouped_mm_stride_violation`.
* ``deepgemm_fp8``  -- native M-grouped DeepGEMM forward for block-scaled FP8
                       experts (CUDA SM90; optional external dependency).

``auto`` delegates primitive selection to ``moe_dispatch``. An explicit backend
selection validates availability on the target device and raises when its
requirements are not met. A role-specific config selection takes precedence
over the shared backend.

This module is self-contained on purpose: it depends only on ``torch`` and (lazily)
on the active :class:`MoEOptimizationPolicy` published by the trainer runtime, so
it never imports a sibling feature-owner module and never reaches up into
``mirai.config`` / ``mirai.core.models`` / ``mirai.core.training``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch-less static analysis
    torch = None  # type: ignore[assignment]


MOE_GEMM_BACKENDS = ("auto", "bmm", "persistent", "torch_grouped", "deepgemm_fp8")
MOE_GEMM_ROLES = ("forward", "dx", "dw")

# Accepted spellings that normalize onto a canonical backend name. Kept small and
# unambiguous; the canonical names themselves always round-trip.
_ALIASES = {
    "": "auto",
    "default": "auto",
    "padded": "bmm",
    "vectorized": "bmm",
    "sorted": "persistent",
    "sorted_contiguous": "persistent",
    "grouped_mm": "torch_grouped",
    "torch_grouped_mm": "torch_grouped",
    "public": "torch_grouped",
    "deepgemm": "deepgemm_fp8",
}


def normalize_moe_gemm_backend(value: str | None) -> str:
    """Normalize the MAIN backend key (``memory.moe_gemm_backend``).

    Empty / unset resolves to ``auto`` (the byte-identical alias of today's
    dispatch). Raises ``ValueError`` on an unknown name.
    """
    text = str(value if value is not None else "auto").strip().lower()
    text = _ALIASES.get(text, text)
    if text not in MOE_GEMM_BACKENDS:
        raise ValueError(
            "memory.moe_gemm_backend must be one of: "
            + ", ".join(MOE_GEMM_BACKENDS)
            + "."
        )
    return text


def normalize_moe_gemm_role_backend(value: str | None) -> str:
    """Normalize a PER-ROLE override key.

    An empty string means "inherit the main key" and is returned as ``""``. Any
    other value is normalized like the main key (``auto`` is allowed and pins the
    role to auto even when the main key is explicit).
    """
    text = str(value if value is not None else "").strip().lower()
    if text in ("", "inherit"):
        return ""
    text = _ALIASES.get(text, text)
    if text not in MOE_GEMM_BACKENDS:
        raise ValueError(
            "memory.moe_gemm_backend_<role> must be '' (inherit) or one of: "
            + ", ".join(MOE_GEMM_BACKENDS)
            + "."
        )
    return text


# ---------------------------------------------------------------------------
# Probes (torch build + device arch only; no vendored-kernel compile here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendProbe:
    """One backend's availability verdict plus a human-readable reason."""

    name: str
    available: bool
    reason: str


def _triton_importable() -> bool:
    return importlib.util.find_spec("triton") is not None


def grouped_mm_op():
    """Return the grouped-mm callable, preferring the PUBLIC name.

    ``torch.nn.functional.grouped_mm`` is the public probe target; when it is
    absent the private ``torch._grouped_mm`` is used as a fallback. Returns
    ``None`` when neither exists (older torch).
    """
    if torch is None:
        return None
    functional = getattr(getattr(torch, "nn", None), "functional", None)
    public = getattr(functional, "grouped_mm", None) if functional is not None else None
    if callable(public):
        return public
    private = getattr(torch, "_grouped_mm", None)
    return private if callable(private) else None


def torch_grouped_public_available() -> bool:
    """Whether the PUBLIC ``torch.nn.functional.grouped_mm`` name exists."""
    if torch is None:
        return False
    functional = getattr(getattr(torch, "nn", None), "functional", None)
    return callable(getattr(functional, "grouped_mm", None))


def _cuda_capability(device=None) -> Optional[tuple[int, int]]:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        index = (
            device.index
            if device is not None and getattr(device, "type", None) == "cuda"
            else None
        )
        major, minor = torch.cuda.get_device_capability(index)
        return int(major), int(minor)
    except Exception:  # pragma: no cover - defensive
        return None


def _requires_cuda_sm80(name: str, device, *, label: str) -> BackendProbe:
    """Shared availability check for the two CUDA SM80+ grouped backends."""
    if device is not None and getattr(device, "type", None) != "cuda":
        return BackendProbe(name, False, f"{label} requires a CUDA device")
    if not torch.cuda.is_available():
        return BackendProbe(name, False, f"{label} requires CUDA")
    capability = _cuda_capability(device)
    if capability is not None and capability[0] < 8:
        return BackendProbe(
            name,
            False,
            f"{label} requires SM80+, found SM{capability[0]}{capability[1]}",
        )
    return BackendProbe(name, True, label)


def probe_backend(name: str, *, device=None) -> BackendProbe:
    """Resolve whether ``name`` can run on ``device`` (or, if ``device`` is None,
    on this host in principle). Never raises; returns a :class:`BackendProbe`."""
    name = normalize_moe_gemm_backend(name)
    if torch is None:
        return BackendProbe(name, False, "torch is not importable")
    if name == "auto":
        return BackendProbe(
            "auto", True, "auto inherits the active moe_dispatch selection"
        )
    if name == "bmm":
        return BackendProbe("bmm", True, "torch.bmm padded primitive (device-agnostic)")
    if name == "persistent":
        probe = _requires_cuda_sm80(
            name, device, label="vendored sorted-contiguous persistent grouped GEMM"
        )
        if not probe.available:
            return probe
        if not _triton_importable():
            return BackendProbe(
                name, False, "persistent grouped GEMM requires an importable triton"
            )
        return probe
    if name == "deepgemm_fp8":
        if importlib.util.find_spec("deep_gemm") is None:
            return BackendProbe(name, False, "deepgemm_fp8 requires an importable deep_gemm")
        if device is not None and getattr(device, "type", None) != "cuda":
            return BackendProbe(name, False, "deepgemm_fp8 requires a CUDA device")
        if not torch.cuda.is_available():
            return BackendProbe(name, False, "deepgemm_fp8 requires CUDA")
        capability = _cuda_capability(device)
        if capability is not None and capability[0] != 9:
            return BackendProbe(
                name,
                False,
                "deepgemm_fp8 requires SM90, found "
                f"SM{capability[0]}{capability[1]}",
            )
        return BackendProbe(name, True, "DeepGEMM M-grouped block-scaled FP8")
    # torch_grouped
    if grouped_mm_op() is None:
        return BackendProbe(
            name,
            False,
            "torch.nn.functional.grouped_mm / torch._grouped_mm is unavailable in "
            "this torch build",
        )
    if device is not None and getattr(device, "type", None) != "cuda":
        return BackendProbe(name, False, "torch grouped_mm requires a CUDA device")
    if not torch.cuda.is_available():
        return BackendProbe(name, False, "torch grouped_mm requires CUDA")
    # The PUBLIC torch.nn.functional.grouped_mm documents a BF16 SM80+ contract;
    # The private ``torch._grouped_mm`` surface is restricted to SM90 because it
    # has no stable public compatibility contract.
    public = torch_grouped_public_available()
    min_major = 8 if public else 9
    arch_label = "SM80+ (public grouped_mm)" if public else "SM90+ (private torch._grouped_mm)"
    capability = _cuda_capability(device)
    if capability is not None and capability[0] < min_major:
        return BackendProbe(
            name,
            False,
            f"torch grouped_mm requires {arch_label}, found "
            f"SM{capability[0]}{capability[1]}",
        )
    surface = (
        "public torch.nn.functional.grouped_mm"
        if public
        else "torch._grouped_mm (private fallback, SM90+)"
    )
    return BackendProbe(name, True, surface)


# ---------------------------------------------------------------------------
# Operand layout precondition (pure; never executes the op)
# ---------------------------------------------------------------------------


# ``torch._grouped_mm`` requires one of each operand's two innermost dimensions
# to be contiguous and the other one's stride to be a multiple of 16 BYTES, so
# the element count that satisfies it depends on the dtype (8 for BF16, 4 for
# FP32). The op raises ``RuntimeError: strides should be multiple of 16 bytes``
# when the precondition does not hold; this predicate decides it in advance.
GROUPED_MM_ALIGNMENT_BYTES = 16


@dataclass(frozen=True)
class GroupedMMOperand:
    """One operand's layout exactly as ``torch grouped_mm`` will receive it."""

    label: str
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    element_size: int


def grouped_mm_operand(tensor, *, label: str) -> GroupedMMOperand:
    """Capture ``tensor``'s layout, including a non-contiguous transposed view."""
    return GroupedMMOperand(
        label=label,
        sizes=tuple(int(value) for value in tensor.shape),
        strides=tuple(int(value) for value in tensor.stride()),
        element_size=int(tensor.element_size()),
    )


def grouped_mm_row_operand(
    *, label: str, columns: int, element_size: int
) -> GroupedMMOperand:
    """Layout of a contiguous ``[tokens, columns]`` activation operand.

    The token count never reaches the stride check: a contiguous 2-D activation
    has strides ``(columns, 1)`` for any row count, so ``columns`` alone decides
    alignment.
    """
    columns = int(columns)
    return GroupedMMOperand(
        label=label,
        sizes=(1, columns),
        strides=(columns, 1),
        element_size=int(element_size),
    )


def grouped_mm_stride_violation(operand: GroupedMMOperand) -> Optional[str]:
    """Return ``None`` when ``operand`` satisfies the stride precondition.

    Otherwise return a message naming the operand, its strides in elements and
    in bytes, and the 16-byte requirement. This mirrors the ATen check that one
    of the two innermost dimensions is unit-strided while the other carries a
    stride that is a whole number of 16-byte blocks.
    """
    sizes = operand.sizes
    strides = operand.strides
    if len(sizes) < 2 or len(strides) != len(sizes):
        return (
            f"{operand.label} must be at least 2-D for torch grouped_mm; got "
            f"sizes {sizes} and strides {strides}."
        )
    element_size = max(1, int(operand.element_size))
    alignment = max(1, GROUPED_MM_ALIGNMENT_BYTES // element_size)
    if strides[-2] == 1 and strides[-1] >= max(1, sizes[-2]):
        checked, dim = strides[-1], len(sizes) - 1
    elif strides[-1] == 1 and strides[-2] >= max(1, sizes[-1]):
        checked, dim = strides[-2], len(sizes) - 2
    else:
        return (
            f"{operand.label} with sizes {sizes} and strides {strides} has no "
            "unit-strided innermost dimension, which torch grouped_mm rejects."
        )
    if checked % alignment == 0:
        return None
    return (
        f"{operand.label} with sizes {sizes} and strides {strides} carries "
        f"stride {checked} on dim {dim}, i.e. {checked * element_size} bytes for "
        f"{element_size}-byte elements; torch grouped_mm requires every operand "
        f"stride to be a multiple of {GROUPED_MM_ALIGNMENT_BYTES} bytes "
        f"({alignment} elements for this dtype)."
    )


def grouped_mm_stride_violations(
    operands: "tuple[GroupedMMOperand, ...] | list[GroupedMMOperand]",
) -> tuple[str, ...]:
    """Every distinct precondition failure across ``operands``, in order."""
    violations: list[str] = []
    for operand in operands:
        reason = grouped_mm_stride_violation(operand)
        if reason is not None and reason not in violations:
            violations.append(reason)
    return tuple(violations)


# ---------------------------------------------------------------------------
# Resolution (per-role config over main config, fail-fast on explicit)
# ---------------------------------------------------------------------------


def _active_policy():
    # Lazy import keeps this module free of a specs import cycle: specs imports the
    # normalizers above at module load, so this module must not import
    # it at top level.
    from mirai.core.moe.runtime.specs import get_active_moe_optimization_policy

    return get_active_moe_optimization_policy()


def _main_backend() -> str:
    return normalize_moe_gemm_backend(
        getattr(_active_policy(), "moe_gemm_backend", "auto")
    )


def _role_backend(role: str) -> str:
    """The per-role override, or ``""`` when the role inherits the main key."""
    return normalize_moe_gemm_role_backend(
        getattr(_active_policy(), f"moe_gemm_backend_{role}", "")
    )


def _remediation(role: str, backend: str, reason: str) -> str:
    key = (
        "memory.moe_gemm_backend"
        if role == "forward"
        else f"memory.moe_gemm_backend_{role}"
    )
    return (
        f"{key}='{backend}' was requested for the '{role}' role but is not "
        f"available: {reason}. Choose 'bmm' (device-agnostic), set it back to "
        f"'auto' (inherit moe_dispatch), or run on a torch build / device that "
        f"provides '{backend}'."
    )


def resolve_moe_gemm_backend(role: str, *, device=None) -> str:
    """Resolve the backend for ``role`` (one of ``MOE_GEMM_ROLES``).

    Precedence: per-role config > main config, with empty per-role values
    inheriting the main key. When ``device`` is provided and the
    resolved backend is an explicit (non-``auto``) choice, an unavailable backend
    raises ``RuntimeError`` with a remediation message. ``auto`` never raises.
    """
    if role not in MOE_GEMM_ROLES:
        raise ValueError("role must be one of: " + ", ".join(MOE_GEMM_ROLES) + ".")
    backend = _role_backend(role)
    if backend == "":
        backend = _main_backend()
    if backend == "deepgemm_fp8" and role != "forward":
        raise RuntimeError(
            "deepgemm_fp8 is a forward-only backend; frozen-weight Dgrad remains "
            "high precision and adapter gradients remain in the adapter path."
        )
    if backend != "auto" and device is not None:
        probe = probe_backend(backend, device=device)
        if not probe.available:
            raise RuntimeError(_remediation(role, backend, probe.reason))
    return backend


def describe(*, device=None) -> dict:
    """Diagnostic snapshot for dry-run / release-evidence surfaces.

    Reports each concrete backend's availability + reason, the grouped_mm surface
    (public vs private), and the currently-resolved role selections (without the
    fail-fast, so it is safe to call for reporting even when an explicit choice is
    unavailable on this device)."""
    return {
        "config_key": "memory.moe_gemm_backend",
        "roles": {role: resolve_moe_gemm_backend(role) for role in MOE_GEMM_ROLES},
        "backends": {
            name: {"available": probe.available, "reason": probe.reason}
            for name in ("bmm", "persistent", "torch_grouped", "deepgemm_fp8")
            for probe in (probe_backend(name, device=device),)
        },
        "grouped_mm": {
            "public": torch_grouped_public_available(),
            "any": grouped_mm_op() is not None,
        },
    }


def run_grouped_dx(
    grad_output,
    weight,
    offsets,
    *,
    backend: str,
):
    """Execute sorted-layout input-gradient GEMM with an independent backend."""
    selected = normalize_moe_gemm_backend(backend)
    if selected == "torch_grouped":
        op = grouped_mm_op()
        if op is None:
            raise RuntimeError("torch grouped_mm is unavailable for the dX role.")
        return op(grad_output.contiguous(), weight.contiguous(), offs=offsets)
    if selected == "persistent":
        from mirai.core.models.compressed_weights.execution.persistent import (
            grouped_gemm_expert_slice,
        )

        result = grad_output.new_empty((grad_output.shape[0], weight.shape[-1]))
        return grouped_gemm_expert_slice(
            grad_output.contiguous(),
            weight.contiguous(),
            offsets,
            result,
            expert_start=0,
            total_experts=int(weight.shape[0]),
            transpose_w=True,
        )
    if selected != "bmm":
        raise ValueError("run_grouped_dx requires an explicit grouped-GEMM backend.")
    boundaries = offsets.detach().to(device="cpu", dtype=torch.int64).tolist()
    result = grad_output.new_empty((grad_output.shape[0], weight.shape[-1]))
    start = 0
    for expert_idx, stop in enumerate(boundaries):
        stop = int(stop)
        if stop > start:
            result[start:stop] = grad_output[start:stop] @ weight[expert_idx]
        start = stop
    return result
