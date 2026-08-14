# SPDX-License-Identifier: Apache-2.0
"""Typed routing-layout contract for fused grouped expert projections.

This owner keeps layout semantics model-agnostic and free of optional kernel
dependencies. Providers flatten their effective group axes before
constructing :class:`RoutedGroupLayout`; execution backends consume only this
validated form.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util
from typing import Any

import torch


class RoutedGemmMode(str, Enum):
    DISABLED = "disabled"
    AUTO = "auto"
    TRITON = "triton"


class RoutedOutputMode(str, Enum):
    GROUPED = "grouped"
    ASSIGNMENT = "assignment"
    WEIGHTED_TOKEN_REDUCTION = "weighted_token_reduction"


@dataclass(frozen=True)
class RoutedFusionSpec:
    """Requested indexed-input and indexed-output transformations."""

    gather_tokens: bool = False
    output: RoutedOutputMode = RoutedOutputMode.GROUPED

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", RoutedOutputMode(self.output))
        if self.gather_tokens and self.output is not RoutedOutputMode.GROUPED:
            raise ValueError(
                "one grouped projection cannot request token gather and output "
                "scatter/reduction simultaneously"
            )


@dataclass(frozen=True)
class RoutedGroupLayout:
    """Stable expert-grouped rows mapped to token-assignment order.

    ``assignment_rows[grouped_row]`` is the flattened token-assignment row in
    ``[0, token_count * top_k)``. Repeated boundaries represent empty groups.
    ``provider_mapping`` is a descriptive provider-owned axis declaration, such
    as ``("head", "expert")``; core never interprets its values.
    """

    boundaries: torch.Tensor
    assignment_rows: torch.Tensor
    token_count: int
    top_k: int
    group_count: int
    provider_mapping: tuple[str, ...] = ("expert",)

    def validate(
        self,
        *,
        device: torch.device | None = None,
        check_values: bool = True,
    ) -> None:
        if self.boundaries.ndim != 1 or self.assignment_rows.ndim != 1:
            raise ValueError("routing boundaries and assignment_rows must be rank-1")
        if not self.boundaries.is_contiguous() or not self.assignment_rows.is_contiguous():
            raise ValueError("routing metadata must be contiguous")
        if self.boundaries.dtype not in (torch.int32, torch.int64):
            raise TypeError("routing boundaries must have int32 or int64 dtype")
        if self.assignment_rows.dtype not in (torch.int32, torch.int64):
            raise TypeError("routing assignment_rows must have int32 or int64 dtype")
        if self.boundaries.device != self.assignment_rows.device:
            raise ValueError("routing metadata tensors must share a device")
        if device is not None and self.boundaries.device != device:
            raise ValueError("routing metadata and operands must share a device")
        if self.token_count < 0 or self.top_k <= 0 or self.group_count < 0:
            raise ValueError("token_count/group_count must be non-negative and top_k positive")
        if int(self.boundaries.numel()) != self.group_count:
            raise ValueError("routing boundaries must contain one entry per group")
        routed_rows = self.token_count * self.top_k
        if int(self.assignment_rows.numel()) != routed_rows:
            raise ValueError("assignment_rows must contain token_count * top_k entries")
        if not check_values:
            return
        values = self.boundaries.detach().to("cpu", torch.int64).tolist()
        if any(value < 0 for value in values) or any(
            right < left for left, right in zip(values, values[1:])
        ):
            raise ValueError("routing boundaries must be non-negative and non-decreasing")
        terminal = values[-1] if values else 0
        if terminal != routed_rows:
            raise ValueError("terminal routing boundary must equal the routed-row count")
        if routed_rows:
            indices = self.assignment_rows.detach().to("cpu", torch.int64)
            if int(indices.min()) < 0 or int(indices.max()) >= routed_rows:
                raise ValueError("assignment_rows contains an out-of-range index")
            if torch.unique(indices).numel() != routed_rows:
                raise ValueError("assignment_rows must be a permutation of assignment rows")
        if not self.provider_mapping or any(not str(axis) for axis in self.provider_mapping):
            raise ValueError("provider_mapping must name at least one effective-group axis")


@dataclass(frozen=True)
class RoutedGemmVerdict:
    selected: str
    supported: bool
    reason: str


def normalize_routed_gemm_mode(value: str | None) -> str:
    text = str(value or "disabled").strip().lower()
    aliases = {"off": "disabled", "none": "disabled", "automatic": "auto"}
    text = aliases.get(text, text)
    try:
        return RoutedGemmMode(text).value
    except ValueError as exc:
        raise ValueError(
            "memory.moe_routed_gemm must be one of: disabled, auto, triton."
        ) from exc


def routed_gemm_reference(
    activation: torch.Tensor,
    weight: torch.Tensor,
    layout: RoutedGroupLayout,
    fusion: RoutedFusionSpec | None = None,
    *,
    routing_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Portable mathematical reference with first-order autograd."""

    fusion = RoutedFusionSpec() if fusion is None else fusion
    layout.validate(device=activation.device)
    if activation.ndim != 2 or weight.ndim != 3:
        raise ValueError("activation must be rank-2 and weight rank-3")
    if weight.device != activation.device:
        raise ValueError("activation and weight must share a device")
    if int(weight.shape[0]) != layout.group_count:
        raise ValueError("weight group axis must equal layout.group_count")
    if int(activation.shape[1]) != int(weight.shape[1]):
        raise ValueError("activation and weight reduction widths must match")
    routed_rows = layout.token_count * layout.top_k
    if fusion.gather_tokens:
        if int(activation.shape[0]) != layout.token_count:
            raise ValueError("gathered activation must contain token_count rows")
        source_tokens = torch.div(layout.assignment_rows, layout.top_k, rounding_mode="floor")
        grouped_input = activation.index_select(0, source_tokens.to(torch.int64))
    else:
        if int(activation.shape[0]) != routed_rows:
            raise ValueError("grouped activation must contain the routed-row count")
        grouped_input = activation
    pieces: list[torch.Tensor] = []
    start = 0
    for group, stop_value in enumerate(layout.boundaries.detach().to("cpu").tolist()):
        stop = int(stop_value)
        if stop > start:
            pieces.append(grouped_input[start:stop] @ weight[group])
        start = stop
    grouped = (
        torch.cat(pieces, dim=0)
        if pieces
        else activation.new_empty((0, weight.shape[2]))
        + activation.sum() * 0
        + weight.sum() * 0
    )
    if fusion.output is RoutedOutputMode.GROUPED:
        return grouped
    assignment = torch.empty_like(grouped).index_copy(
        0, layout.assignment_rows.to(torch.int64), grouped
    )
    if fusion.output is RoutedOutputMode.ASSIGNMENT:
        return assignment
    if routing_weights is None or tuple(routing_weights.shape) != (
        layout.token_count,
        layout.top_k,
    ):
        raise ValueError("routing_weights must have shape (token_count, top_k)")
    return (assignment.view(layout.token_count, layout.top_k, -1) * routing_weights[..., None]).sum(1)


def routed_gemm_verdict(
    mode: str,
    activation: Any,
    weight: Any,
    fusion: RoutedFusionSpec,
    *,
    training: bool,
    resident: bool,
    quantized: bool,
    layout: RoutedGroupLayout | None = None,
    architecture: str = "auto",
    triton_available: bool | None = None,
    provider_declared: bool = True,
    fusion_gradients_supported: bool = True,
    observer_compatible: bool = True,
) -> RoutedGemmVerdict:
    """Pre-execution capability verdict; ``auto`` may choose the reference path."""

    requested = normalize_routed_gemm_mode(mode)
    if requested == "disabled":
        return RoutedGemmVerdict("reference", True, "fused routed GEMM is disabled")
    failures: list[str] = []
    activation_device = getattr(activation, "device", None)
    weight_device = getattr(weight, "device", None)
    if getattr(activation, "device", None) is None or activation.device.type != "cuda":
        failures.append("CUDA activation")
    if weight_device != activation_device:
        failures.append("operands on one device")
    if getattr(activation, "ndim", None) != 2 or getattr(weight, "ndim", None) != 3:
        failures.append("rank-2 activation and rank-3 weight")
    elif int(activation.shape[1]) != int(weight.shape[1]):
        failures.append("matching reduction widths")
    if (
        getattr(activation, "dtype", None) != torch.bfloat16
        or getattr(weight, "dtype", None) != torch.bfloat16
    ):
        failures.append("resident BF16 operands")
    if not bool(getattr(activation, "is_contiguous", lambda: False)()):
        failures.append("contiguous activation")
    if not resident or quantized:
        failures.append("resident unquantized expert weights")
    if not provider_declared:
        failures.append("provider-declared routed execution")
    if not observer_compatible:
        failures.append("no incompatible routed observers")
    if (
        training
        and fusion.output is RoutedOutputMode.WEIGHTED_TOKEN_REDUCTION
        and not fusion_gradients_supported
    ):
        failures.append("training gradients for weighted token reduction")
    available = (
        importlib.util.find_spec("triton") is not None
        if triton_available is None
        else bool(triton_available)
    )
    if not available:
        failures.append("importable Triton")
    arch = str(architecture or "auto").strip().lower()
    if arch not in {"auto", "indexed", "tma_regular"}:
        failures.append("known routed architecture")
    if arch == "tma_regular" and fusion.gather_tokens:
        failures.append("non-indexed input for tma_regular")
    if arch == "tma_regular" and fusion.output is not RoutedOutputMode.GROUPED:
        failures.append("grouped output for tma_regular")
    if activation_device is not None and activation_device.type == "cuda":
        try:
            major, minor = torch.cuda.get_device_capability(activation_device)
            if int(major) < 8:
                failures.append(f"SM80+ device (found SM{major}{minor})")
            if arch == "tma_regular" and int(major) != 9:
                failures.append(f"SM90 device for tma_regular (found SM{major}{minor})")
        except (RuntimeError, AssertionError):
            failures.append("queryable CUDA capability")
    if layout is not None:
        try:
            layout.validate(device=activation_device, check_values=False)
            if getattr(weight, "ndim", None) == 3 and int(weight.shape[0]) != layout.group_count:
                failures.append("weight group axis matching routing layout")
            expected_rows = layout.token_count if fusion.gather_tokens else layout.token_count * layout.top_k
            if getattr(activation, "ndim", None) == 2 and int(activation.shape[0]) != expected_rows:
                failures.append("activation rows matching routing layout")
        except (TypeError, ValueError) as exc:
            failures.append(f"valid routing metadata ({exc})")
    if failures:
        reason = "requires " + ", ".join(failures)
        if requested == "auto":
            return RoutedGemmVerdict("reference", True, reason)
        return RoutedGemmVerdict("triton", False, reason)
    return RoutedGemmVerdict("triton", True, "resident BF16 CUDA routed fusion")
