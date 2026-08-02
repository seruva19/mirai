"""Depth-aware sparse-MoE router settings.

Layer-wise expert allocation is motivated by the non-uniform depth profiles in
LayerMoE, MoLA, and visual-DiT routing diagnostics:
https://arxiv.org/abs/2505.22582,
https://arxiv.org/abs/2402.08562, and
https://arxiv.org/abs/2605.19378.

Those sources do not define one joint video-DiT rule for routing width, expert
working-set fraction, and z-loss. This module therefore owns a deterministic
configuration contract for those three orthogonal controls; model providers
bind the resolved settings to their native routers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LayerRouterSettings:
    """Resolved router settings for one zero-based sparse-MoE layer."""

    top_k: int
    subset_fraction: float
    z_loss_weight: float


@dataclass(frozen=True)
class LayerRouterBand:
    """Half-open layer interval with one or more router overrides."""

    first_layer: int
    end_layer: int
    top_k: int | None = None
    subset_fraction: float | None = None
    z_loss_weight: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LayerRouterBand":
        allowed = {
            "first_layer",
            "end_layer",
            "top_k",
            "subset_fraction",
            "z_loss_weight",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "Layer router policy contains unknown keys: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        if "first_layer" not in value or "end_layer" not in value:
            raise ValueError(
                "Layer router policy bands require first_layer and end_layer."
            )
        top_k = int(value["top_k"]) if "top_k" in value else None
        subset = (
            float(value["subset_fraction"])
            if "subset_fraction" in value
            else None
        )
        z_loss = (
            float(value["z_loss_weight"])
            if "z_loss_weight" in value
            else None
        )
        return cls(
            first_layer=int(value["first_layer"]),
            end_layer=int(value["end_layer"]),
            top_k=top_k,
            subset_fraction=subset,
            z_loss_weight=z_loss,
        )

    def validate(
        self,
        *,
        num_layers: int,
        num_experts: int,
    ) -> "LayerRouterBand":
        if not 0 <= int(self.first_layer) < int(self.end_layer) <= int(num_layers):
            raise ValueError(
                "Layer router policy requires "
                "0 <= first_layer < end_layer <= num_layers."
            )
        if (
            self.top_k is None
            and self.subset_fraction is None
            and self.z_loss_weight is None
        ):
            raise ValueError("Layer router policy bands require an override.")
        if self.top_k is not None and not 1 <= int(self.top_k) <= int(num_experts):
            raise ValueError("Layer router policy top_k is outside the expert pool.")
        if self.subset_fraction is not None and not (
            0.0 < float(self.subset_fraction) <= 1.0
        ):
            raise ValueError(
                "Layer router policy subset_fraction must be in (0, 1]."
            )
        if self.z_loss_weight is not None and (
            not math.isfinite(float(self.z_loss_weight))
            or float(self.z_loss_weight) < 0.0
        ):
            raise ValueError(
                "Layer router policy z_loss_weight must be finite and non-negative."
            )
        if (
            self.top_k is not None
            and self.subset_fraction is not None
            and math.ceil(float(self.subset_fraction) * int(num_experts))
            < int(self.top_k)
        ):
            raise ValueError(
                "Layer router policy subset must contain at least top_k experts."
            )
        return self

    def contains(self, layer_index: int) -> bool:
        return int(self.first_layer) <= int(layer_index) < int(self.end_layer)


class LayerRouterPolicy:
    """Validated immutable depth-aware router policy."""

    def __init__(
        self,
        bands: Sequence[LayerRouterBand],
        *,
        num_layers: int,
        num_experts: int,
        fallback_top_k: int,
        fallback_subset_fraction: float,
        fallback_z_loss_weight: float,
    ) -> None:
        layers = int(num_layers)
        experts = int(num_experts)
        fallback = LayerRouterSettings(
            top_k=int(fallback_top_k),
            subset_fraction=float(fallback_subset_fraction),
            z_loss_weight=float(fallback_z_loss_weight),
        )
        if layers < 1 or experts < 2:
            raise ValueError("Layer router policy requires layers and experts.")
        if not 1 <= fallback.top_k <= experts:
            raise ValueError("Layer router fallback top_k is invalid.")
        if not 0.0 < fallback.subset_fraction <= 1.0:
            raise ValueError("Layer router fallback subset_fraction is invalid.")
        if not math.isfinite(fallback.z_loss_weight) or fallback.z_loss_weight < 0.0:
            raise ValueError("Layer router fallback z_loss_weight is invalid.")
        normalized = tuple(
            sorted(
                (
                    band.validate(num_layers=layers, num_experts=experts)
                    for band in bands
                ),
                key=lambda band: (band.first_layer, band.end_layer),
            )
        )
        for left, right in zip(normalized, normalized[1:], strict=False):
            if int(right.first_layer) < int(left.end_layer):
                raise ValueError("Layer router policy bands must not overlap.")
        for band in normalized:
            effective_top_k = (
                fallback.top_k if band.top_k is None else int(band.top_k)
            )
            effective_subset = (
                fallback.subset_fraction
                if band.subset_fraction is None
                else float(band.subset_fraction)
            )
            if math.ceil(effective_subset * experts) < effective_top_k:
                raise ValueError(
                    "Resolved layer router subset must contain at least top_k experts."
                )
        self._bands = normalized
        self._num_layers = layers
        self._num_experts = experts
        self._fallback = fallback

    @classmethod
    def from_model_params(cls, params: Any) -> "LayerRouterPolicy | None":
        raw = tuple(getattr(params, "moe_layer_router_policy", ()) or ())
        if not raw:
            return None
        if any(not isinstance(item, Mapping) for item in raw):
            raise ValueError("Layer router policy must be an array of tables.")
        return cls(
            tuple(LayerRouterBand.from_mapping(item) for item in raw),
            num_layers=int(getattr(params, "num_layers", 0)),
            num_experts=int(getattr(params, "num_experts", 0)),
            fallback_top_k=int(getattr(params, "experts_per_token", 0)),
            fallback_subset_fraction=float(
                getattr(params, "expert_subset_fraction", 1.0)
            ),
            fallback_z_loss_weight=float(
                getattr(params, "moe_router_z_loss_weight", 0.0)
            ),
        )

    @property
    def bands(self) -> tuple[LayerRouterBand, ...]:
        return self._bands

    def resolve(self, layer_index: int) -> LayerRouterSettings:
        index = int(layer_index)
        if not 0 <= index < self._num_layers:
            raise IndexError("Layer router policy index is out of range.")
        for band in self._bands:
            if band.contains(index):
                return LayerRouterSettings(
                    top_k=(
                        self._fallback.top_k
                        if band.top_k is None
                        else int(band.top_k)
                    ),
                    subset_fraction=(
                        self._fallback.subset_fraction
                        if band.subset_fraction is None
                        else float(band.subset_fraction)
                    ),
                    z_loss_weight=(
                        self._fallback.z_loss_weight
                        if band.z_loss_weight is None
                        else float(band.z_loss_weight)
                    ),
                )
        return self._fallback


__all__ = [
    "LayerRouterBand",
    "LayerRouterPolicy",
    "LayerRouterSettings",
]
