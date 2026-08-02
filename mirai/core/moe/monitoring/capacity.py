"""Deterministic single-GPU MoE memory and transfer estimator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MoECapacitySpec:
    num_layers: int
    num_experts: int
    experts_per_token: int
    hidden_size: int
    expert_intermediate_size: int
    tokens_per_step: int
    weight_bits: float = 16.0
    adapter_rank: int = 0
    selected_expert_fraction: float = 1.0
    resident_experts_per_layer: int = 0

    def __post_init__(self) -> None:
        for name in (
            "num_layers",
            "num_experts",
            "experts_per_token",
            "hidden_size",
            "expert_intermediate_size",
            "tokens_per_step",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0.")
        if int(self.experts_per_token) > int(self.num_experts):
            raise ValueError("experts_per_token cannot exceed num_experts.")
        if float(self.weight_bits) <= 0.0:
            raise ValueError("weight_bits must be > 0.")
        if not 0.0 < float(self.selected_expert_fraction) <= 1.0:
            raise ValueError("selected_expert_fraction must be in (0, 1].")
        if not 0 <= int(self.resident_experts_per_layer) <= int(self.num_experts):
            raise ValueError(
                "resident_experts_per_layer must be in [0, num_experts]."
            )


@dataclass(frozen=True)
class MoECapacityEstimate:
    total_expert_weight_bytes: int
    expected_unique_experts_per_layer: float
    maximum_unique_experts_per_layer: int
    expected_streamed_weight_bytes_per_step: int
    maximum_streamed_weight_bytes_per_step: int
    trainable_adapter_bytes: int
    adapter_gradient_bytes: int
    adam_state_bytes: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def estimate_moe_capacity(spec: MoECapacitySpec) -> MoECapacityEstimate:
    """Estimate expert storage and PCIe traffic without allocating tensors.

    Expected unique experts assumes uniform independent routes and is therefore
    a planning estimate, not a router-quality claim. The maximum is the exact
    route-count bound for one layer.
    """
    expert_values = (
        3
        * int(spec.hidden_size)
        * int(spec.expert_intermediate_size)
    )
    bytes_per_expert = int(math.ceil(expert_values * float(spec.weight_bits) / 8.0))
    routes = int(spec.tokens_per_step) * int(spec.experts_per_token)
    experts = int(spec.num_experts)
    expected_unique = experts * (1.0 - ((experts - 1.0) / experts) ** routes)
    maximum_unique = min(experts, routes)
    resident = int(spec.resident_experts_per_layer)
    expected_streamed = max(0.0, expected_unique - resident)
    maximum_streamed = max(0, maximum_unique - resident)
    layers = int(spec.num_layers)
    selected = max(1, math.ceil(experts * float(spec.selected_expert_fraction)))
    adapter_values = (
        layers
        * selected
        * 3
        * int(spec.adapter_rank)
        * (int(spec.hidden_size) + int(spec.expert_intermediate_size))
    )
    adapter_bytes = adapter_values * 4
    return MoECapacityEstimate(
        total_expert_weight_bytes=layers * experts * bytes_per_expert,
        expected_unique_experts_per_layer=float(expected_unique),
        maximum_unique_experts_per_layer=int(maximum_unique),
        expected_streamed_weight_bytes_per_step=int(
            math.ceil(layers * expected_streamed * bytes_per_expert)
        ),
        maximum_streamed_weight_bytes_per_step=(
            layers * maximum_streamed * bytes_per_expert
        ),
        trainable_adapter_bytes=adapter_bytes,
        adapter_gradient_bytes=adapter_bytes,
        adam_state_bytes=adapter_values * 8,
    )


__all__ = ["MoECapacityEstimate", "MoECapacitySpec", "estimate_moe_capacity"]
