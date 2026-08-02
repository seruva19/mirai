"""Provider-declared diffusion-aware Expert-Choice policy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ExpertChoiceCapacityBand:
    """Capacity factor for one contiguous transformer-layer interval."""

    first_layer: int
    end_layer: int
    capacity_factor: float

    def __post_init__(self) -> None:
        if self.first_layer < 0 or self.end_layer <= self.first_layer:
            raise ValueError("Expert-Choice capacity bands require 0 <= first_layer < end_layer.")
        if self.capacity_factor <= 0.0:
            raise ValueError("Expert-Choice capacity_factor must be > 0.")

    def contains(self, layer_index: int) -> bool:
        return self.first_layer <= int(layer_index) < self.end_layer


@dataclass(frozen=True)
class ExpertChoiceCapacityStage:
    """Named training stage containing non-overlapping layer-capacity bands."""

    name: str
    bands: tuple[ExpertChoiceCapacityBand, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Expert-Choice capacity stage name must not be empty.")
        if not self.bands:
            raise ValueError("Expert-Choice capacity stage requires at least one band.")
        ordered = sorted(self.bands, key=lambda band: band.first_layer)
        for left, right in zip(ordered, ordered[1:]):
            if left.end_layer > right.first_layer:
                raise ValueError(
                    f"Expert-Choice capacity bands overlap in stage {self.name!r}."
                )

    def factor_for_layer(self, layer_index: int) -> float:
        matches = [band for band in self.bands if band.contains(layer_index)]
        if len(matches) != 1:
            raise ValueError(
                f"Expert-Choice stage {self.name!r} has no capacity band for "
                f"layer {int(layer_index)}."
            )
        return float(matches[0].capacity_factor)


@dataclass(frozen=True)
class ExpertChoiceCapacitySchedule:
    """Provider-owned per-stage and per-layer capacity schedule."""

    stages: tuple[ExpertChoiceCapacityStage, ...]
    default_stage: str

    def __post_init__(self) -> None:
        names = tuple(stage.name for stage in self.stages)
        if len(names) != len(set(names)):
            raise ValueError("Expert-Choice capacity stage names must be unique.")
        if self.default_stage not in names:
            raise ValueError(
                f"Expert-Choice default stage {self.default_stage!r} is not declared."
            )

    def factor_for(self, *, layer_index: int, stage: str | None = None) -> float:
        selected = self.default_stage if stage is None else str(stage)
        for candidate in self.stages:
            if candidate.name == selected:
                return candidate.factor_for_layer(layer_index)
        raise ValueError(f"Unknown Expert-Choice capacity stage {selected!r}.")


@dataclass(frozen=True)
class ExpertChoiceRoutingPolicy:
    """Optional diffusion-specific inputs and scheduling for Expert-Choice."""

    timestep_hidden_size: int = 0
    normalize_timestep: bool = True
    route_scale: float = 1.0
    capacity_schedule: ExpertChoiceCapacitySchedule | None = None

    def __post_init__(self) -> None:
        if self.timestep_hidden_size < 0:
            raise ValueError("timestep_hidden_size must be >= 0.")
        if self.route_scale <= 0.0:
            raise ValueError("Expert-Choice route_scale must be > 0.")

    @property
    def uses_timestep(self) -> bool:
        return self.timestep_hidden_size > 0

    def resolve_capacity_factor(
        self,
        *,
        layer_index: int,
        stage: str | None,
        fallback: float,
    ) -> float:
        if self.capacity_schedule is None:
            return float(fallback)
        return self.capacity_schedule.factor_for(layer_index=layer_index, stage=stage)

    def build_router_input(
        self,
        routing_hidden_states: Any,
        timestep_embedding: Any | None,
    ) -> Any:
        if torch is None:  # pragma: no cover
            raise RuntimeError("ExpertChoiceRoutingPolicy requires torch.")
        if routing_hidden_states.ndim != 3:
            raise ValueError("routing_hidden_states must have shape [batch, tokens, hidden].")
        if not self.uses_timestep:
            if timestep_embedding is not None:
                raise ValueError(
                    "timestep_embedding was provided but this Expert-Choice policy "
                    "does not declare a timestep channel."
                )
            return routing_hidden_states
        if timestep_embedding is None:
            raise ValueError("This Expert-Choice policy requires timestep_embedding.")
        batch, tokens, _hidden = routing_hidden_states.shape
        if timestep_embedding.ndim == 2:
            if timestep_embedding.shape != (batch, self.timestep_hidden_size):
                raise ValueError(
                    "timestep_embedding must have shape [batch, timestep_hidden_size]."
                )
            timestep = timestep_embedding[:, None, :].expand(batch, tokens, -1)
        elif timestep_embedding.ndim == 3:
            if timestep_embedding.shape != (batch, tokens, self.timestep_hidden_size):
                raise ValueError(
                    "tokenwise timestep_embedding must have shape "
                    "[batch, tokens, timestep_hidden_size]."
                )
            timestep = timestep_embedding
        else:
            raise ValueError("timestep_embedding must be rank 2 or rank 3.")
        if self.normalize_timestep:
            fp32 = timestep.float()
            inverse_rms = torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + 1e-6)
            timestep = (fp32 * inverse_rms).to(dtype=routing_hidden_states.dtype)
        else:
            timestep = timestep.to(dtype=routing_hidden_states.dtype)
        return torch.cat((routing_hidden_states, timestep), dim=-1)


@dataclass(frozen=True)
class ExpertChoiceCoverageSummary:
    """Detached aggregate of per-layer Expert-Choice token coverage."""

    mean_fraction: float
    minimum_fraction: float
    below_threshold: bool


def summarize_expert_choice_coverage(
    coverage_fractions: list[float] | tuple[float, ...],
    *,
    alarm_threshold: float,
) -> ExpertChoiceCoverageSummary | None:
    """Summarize coverage and alarm when any observed layer falls below a floor."""
    threshold = float(alarm_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("Expert-Choice coverage alarm_threshold must be in (0, 1].")
    fractions = tuple(float(value) for value in coverage_fractions)
    if not fractions:
        return None
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("Expert-Choice coverage fractions must be in [0, 1].")
    minimum = min(fractions)
    return ExpertChoiceCoverageSummary(
        mean_fraction=sum(fractions) / len(fractions),
        minimum_fraction=minimum,
        below_threshold=minimum < threshold,
    )


def resolve_capacity_schedule(
    schedule: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    step: int,
    layer_index: int,
    fallback: float,
) -> float:
    """Resolve one non-overlapping config-defined step/layer capacity band."""
    matches: list[dict[str, Any]] = []
    for band in schedule:
        start_step = int(band["start_step"])
        end_step = int(band.get("end_step", -1))
        if int(step) < start_step or (end_step >= 0 and int(step) >= end_step):
            continue
        if int(band["first_layer"]) <= int(layer_index) < int(band["end_layer"]):
            matches.append(band)
    if len(matches) > 1:
        raise ValueError(
            "Expert-Choice capacity schedule has overlapping active step/layer bands."
        )
    if not matches:
        return float(fallback)
    return float(matches[0]["capacity_factor"])
