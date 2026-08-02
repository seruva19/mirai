"""Step-conditioned lower-layer routing width for progressive MoE sparsification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProgressiveSparsificationBand:
    """Early-training top-k override for one half-open layer interval."""

    first_layer: int
    end_layer: int
    top_k: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ProgressiveSparsificationBand":
        allowed = {"first_layer", "end_layer", "top_k"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "Unknown progressive-sparsification fields: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        missing = allowed - set(value)
        if missing:
            raise ValueError(
                "Progressive-sparsification bands require "
                + ", ".join(sorted(missing))
                + "."
            )
        return cls(
            first_layer=int(value["first_layer"]),
            end_layer=int(value["end_layer"]),
            top_k=int(value["top_k"]),
        )

    def contains(self, layer_index: int) -> bool:
        return int(self.first_layer) <= int(layer_index) < int(self.end_layer)


class ProgressiveSparsificationPolicy:
    """Switch explicit early routing widths to target widths at one step boundary."""

    def __init__(
        self,
        bands: Sequence[ProgressiveSparsificationBand],
        *,
        transition_step: int,
        target_top_k: Sequence[int],
        num_experts: int,
    ) -> None:
        transition = int(transition_step)
        targets = tuple(int(value) for value in target_top_k)
        experts = int(num_experts)
        if transition <= 0:
            raise ValueError(
                "Progressive sparsification requires a positive transition_step."
            )
        if not targets or experts < 2:
            raise ValueError(
                "Progressive sparsification requires routed layers and experts."
            )
        if any(not 1 <= value <= experts for value in targets):
            raise ValueError(
                "Progressive sparsification target top-k is outside the expert pool."
            )
        normalized = tuple(
            sorted(
                bands,
                key=lambda band: (int(band.first_layer), int(band.end_layer)),
            )
        )
        if not normalized:
            raise ValueError(
                "Progressive sparsification requires at least one early layer band."
            )
        for band in normalized:
            if not 0 <= int(band.first_layer) < int(band.end_layer) <= len(targets):
                raise ValueError(
                    "Progressive-sparsification bands require "
                    "0 <= first_layer < end_layer <= routed layers."
                )
            if not 1 <= int(band.top_k) <= experts:
                raise ValueError(
                    "Progressive-sparsification top_k is outside the expert pool."
                )
            for layer_index in range(int(band.first_layer), int(band.end_layer)):
                if int(band.top_k) <= targets[layer_index]:
                    raise ValueError(
                        "Progressive-sparsification early top_k must exceed the "
                        "target top_k for every covered layer."
                    )
        for left, right in zip(normalized, normalized[1:], strict=False):
            if int(right.first_layer) < int(left.end_layer):
                raise ValueError(
                    "Progressive-sparsification layer bands must not overlap."
                )
        self._bands = normalized
        self._transition_step = transition
        self._target_top_k = targets
        early = list(targets)
        for band in normalized:
            for layer_index in range(int(band.first_layer), int(band.end_layer)):
                early[layer_index] = int(band.top_k)
        self._early_top_k = tuple(early)

    @classmethod
    def from_model_params(
        cls,
        params: Any,
        *,
        target_top_k: Sequence[int],
        num_experts: int,
    ) -> "ProgressiveSparsificationPolicy | None":
        raw = tuple(
            getattr(params, "moe_progressive_sparsification_policy", ()) or ()
        )
        transition = int(
            getattr(
                params,
                "moe_progressive_sparsification_transition_step",
                0,
            )
        )
        if not raw and transition == 0:
            return None
        if not raw or transition <= 0:
            raise ValueError(
                "Progressive sparsification requires both a positive transition "
                "step and a non-empty policy."
            )
        if any(not isinstance(item, Mapping) for item in raw):
            raise ValueError(
                "Progressive sparsification policy must be an array of tables."
            )
        return cls(
            tuple(ProgressiveSparsificationBand.from_mapping(item) for item in raw),
            transition_step=transition,
            target_top_k=target_top_k,
            num_experts=num_experts,
        )

    @property
    def bands(self) -> tuple[ProgressiveSparsificationBand, ...]:
        return self._bands

    @property
    def transition_step(self) -> int:
        return self._transition_step

    def top_k(self, *, layer_index: int, step: int) -> int:
        index = int(layer_index)
        current_step = int(step)
        if not 0 <= index < len(self._target_top_k):
            raise IndexError("Progressive-sparsification layer index is out of range.")
        if current_step < 0:
            raise ValueError("Progressive-sparsification step must be non-negative.")
        source = (
            self._early_top_k
            if current_step < self._transition_step
            else self._target_top_k
        )
        return int(source[index])

    def is_early(self, *, step: int) -> bool:
        if int(step) < 0:
            raise ValueError("Progressive-sparsification step must be non-negative.")
        return int(step) < self._transition_step


__all__ = [
    "ProgressiveSparsificationBand",
    "ProgressiveSparsificationPolicy",
]
