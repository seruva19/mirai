"""Bounded expert-projection contracts for prototype calibration."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Protocol, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class ExpertProjectionSpec:
    """One named expert projection with a fixed per-expert dense shape."""

    name: str
    shape: tuple[int, ...]

    @property
    def elements_per_expert(self) -> int:
        return math.prod(self.shape)

    def validate(self) -> ExpertProjectionSpec:
        if not str(self.name).strip():
            raise ValueError("Expert projection name must be non-empty.")
        if not self.shape or any(int(value) <= 0 for value in self.shape):
            raise ValueError(
                f"Expert projection {self.name!r} must have a positive dense shape."
            )
        return self


class ExpertProjectionSource(Protocol):
    """Block-readable expert weights; implementations own storage/dequantization."""

    num_experts: int

    def prototype_projection_specs(self) -> Sequence[ExpertProjectionSpec]: ...

    def load_prototype_projection_block(
        self,
        projection_name: str,
        start_expert: int,
        stop_expert: int,
        *,
        device: Any,
        dtype: Any,
    ) -> Any: ...


class PrototypeCalibrationHost(Protocol):
    """Runtime attachment seam consumed by the generic calibration runner."""

    def set_prototype_calibration_observer(self, observer: Any) -> None: ...

    def clear_prototype_calibration_observer(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class PrototypeCalibrationTarget:
    """Provider-owned binding of one logical module to runtime and weight seams."""

    name: str
    host: PrototypeCalibrationHost
    projection_source: ExpertProjectionSource

    def validate(self) -> PrototypeCalibrationTarget:
        if not str(self.name).strip():
            raise ValueError("Prototype calibration target name must be non-empty.")
        if int(self.projection_source.num_experts) < 2:
            raise ValueError(
                f"Prototype calibration target {self.name!r} requires two experts."
            )
        return self


def projection_block_experts(
    spec: ExpertProjectionSpec,
    *,
    max_block_elements: int,
) -> int:
    """Resolve the maximum expert count in one dense projection block."""
    spec.validate()
    budget = int(max_block_elements)
    if budget < 1:
        raise ValueError("max_block_elements must be positive.")
    return max(1, budget // spec.elements_per_expert)


def normalized_parameter_distance_streaming(
    source: ExpertProjectionSource,
    *,
    max_block_elements: int,
    device: Any = "cpu",
    dtype: Any = None,
    epsilon: float = 1e-12,
) -> Any:
    """Compute exact mean normalized Frobenius distance with block residency.

    The distance engine retains at most two requested dense blocks. A storage
    adapter may additionally need one expert-sized dequantization scratch; an
    expert larger than the requested budget is therefore irreducible.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Prototype projection distance requires torch.")
    expert_count = int(source.num_experts)
    if expert_count < 2:
        raise ValueError("Prototype projection distance requires at least two experts.")
    budget = int(max_block_elements)
    if budget < 1:
        raise ValueError("max_block_elements must be positive.")
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    compute_dtype = torch.float64 if dtype is None else dtype
    if compute_dtype not in {torch.float32, torch.float64}:
        raise ValueError("Prototype projection distance dtype must be float32 or float64.")
    specs = tuple(spec.validate() for spec in source.prototype_projection_specs())
    if not specs:
        raise ValueError("Prototype projection source did not declare any projections.")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("Prototype projection names must be unique.")

    total = torch.zeros((expert_count, expert_count), dtype=compute_dtype)
    for spec in specs:
        block_experts = projection_block_experts(
            spec,
            max_block_elements=budget,
        )
        projection_distance = torch.zeros_like(total)
        for left_start in range(0, expert_count, block_experts):
            left_stop = min(expert_count, left_start + block_experts)
            left = _load_projection_block(
                source,
                spec,
                left_start,
                left_stop,
                device=device,
                dtype=compute_dtype,
            )
            left_norm = torch.linalg.vector_norm(left, ord=2, dim=1)
            for right_start in range(left_start, expert_count, block_experts):
                right_stop = min(expert_count, right_start + block_experts)
                if right_start == left_start:
                    right = left
                    right_norm = left_norm
                else:
                    right = _load_projection_block(
                        source,
                        spec,
                        right_start,
                        right_stop,
                        device=device,
                        dtype=compute_dtype,
                    )
                    right_norm = torch.linalg.vector_norm(right, ord=2, dim=1)
                pairwise = torch.cdist(left, right, p=2)
                normalized = (2.0 * pairwise) / (
                    left_norm.unsqueeze(1) + right_norm.unsqueeze(0) + (2.0 * eps)
                )
                block = normalized.detach().to(device="cpu", dtype=compute_dtype)
                projection_distance[
                    left_start:left_stop,
                    right_start:right_stop,
                ] = block
                if right_start != left_start:
                    projection_distance[
                        right_start:right_stop,
                        left_start:left_stop,
                    ] = block.T
                del right, right_norm, pairwise, normalized, block
            del left, left_norm
        projection_distance.fill_diagonal_(0.0)
        total += projection_distance
    total /= float(len(specs))
    total.fill_diagonal_(0.0)
    return total


def _load_projection_block(
    source: ExpertProjectionSource,
    spec: ExpertProjectionSpec,
    start: int,
    stop: int,
    *,
    device: Any,
    dtype: Any,
) -> Any:
    block = source.load_prototype_projection_block(
        spec.name,
        int(start),
        int(stop),
        device=device,
        dtype=dtype,
    )
    tensor = torch.as_tensor(block)
    expected = (int(stop) - int(start), *spec.shape)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"Projection source returned {tuple(tensor.shape)} for {spec.name!r}; "
            f"expected {expected}."
        )
    requested_device = torch.device(device)
    wrong_device = tensor.device.type != requested_device.type or (
        requested_device.index is not None
        and tensor.device.index != requested_device.index
    )
    if tensor.dtype != dtype or wrong_device:
        raise ValueError(
            f"Projection source returned the wrong dtype/device for {spec.name!r}."
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"Projection source returned non-finite {spec.name!r} weights.")
    return tensor.reshape(int(stop) - int(start), -1)


__all__ = [
    "ExpertProjectionSource",
    "ExpertProjectionSpec",
    "PrototypeCalibrationHost",
    "PrototypeCalibrationTarget",
    "normalized_parameter_distance_streaming",
    "projection_block_experts",
]
