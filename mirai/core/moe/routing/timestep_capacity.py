"""Compute-matched timestep-dependent Expert-Choice capacity."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


TIMESTEP_CAPACITY_SCHEDULES = frozenset({"disabled", "linear_reverse"})
TIMESTEP_SAMPLING_MODES = frozenset({"uniform", "logit_normal", "mode"})


def normalize_timestep_capacity_schedule(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in TIMESTEP_CAPACITY_SCHEDULES:
        raise ValueError(
            "Expert-Choice timestep capacity schedule must be one of: "
            + ", ".join(sorted(TIMESTEP_CAPACITY_SCHEDULES))
            + "."
        )
    return normalized


@dataclass(frozen=True)
class TimestepExpertChoiceCapacityPolicy:
    """Map normalized diffusion timesteps to a compute-matched capacity.

    ``linear_reverse`` adapts the reverse-linear capacity schedule from
    Expert-Choice Routing Enables Adaptive Computation in Diffusion Language
    Models (arXiv:2604.01622) to continuous video-diffusion timesteps.

    The configured sampler CDF maps post-``flow_shift`` noise levels back to a
    uniform quantile. Symmetric integer offsets around the static capacity
    therefore have zero expected value, including after nearest-integer
    rounding. This preserves the static Expert-Choice selected-slot budget in
    expectation while assigning more capacity to low-noise samples.
    """

    schedule: str = "disabled"
    capacity_factor_span: float = 0.0
    timestep_sampling: str = "uniform"
    timestep_sampling_mean: float = 0.0
    timestep_sampling_std: float = 1.0
    timestep_sampling_mode_scale: float = 1.29
    flow_shift: float = 1.0

    def __post_init__(self) -> None:
        schedule = normalize_timestep_capacity_schedule(self.schedule)
        object.__setattr__(self, "schedule", schedule)
        sampling = str(self.timestep_sampling).strip().lower()
        if sampling not in TIMESTEP_SAMPLING_MODES:
            raise ValueError(
                "Timestep capacity sampling mode must be one of: "
                + ", ".join(sorted(TIMESTEP_SAMPLING_MODES))
                + "."
            )
        object.__setattr__(self, "timestep_sampling", sampling)
        span = float(self.capacity_factor_span)
        if not math.isfinite(span) or span < 0.0:
            raise ValueError("Timestep capacity_factor_span must be finite and >= 0.")
        if schedule == "disabled" and span != 0.0:
            raise ValueError(
                "Disabled timestep capacity scheduling requires "
                "capacity_factor_span=0."
            )
        if schedule != "disabled" and span <= 0.0:
            raise ValueError(
                "Enabled timestep capacity scheduling requires "
                "capacity_factor_span > 0."
            )
        if not math.isfinite(float(self.timestep_sampling_mean)):
            raise ValueError("Logit-normal timestep mean must be finite.")
        if (
            not math.isfinite(float(self.timestep_sampling_std))
            or float(self.timestep_sampling_std) <= 0.0
        ):
            raise ValueError("Logit-normal timestep std must be finite and > 0.")
        mode_scale = float(self.timestep_sampling_mode_scale)
        mode_scale_max = 2.0 / (math.pi - 2.0)
        if (
            not math.isfinite(mode_scale)
            or mode_scale < -1.0
            or mode_scale > mode_scale_max
        ):
            raise ValueError(
                "Mode-shift timestep scale must be finite and in "
                f"[-1.0, {mode_scale_max}]."
            )
        if (
            not math.isfinite(float(self.flow_shift))
            or float(self.flow_shift) <= 0.0
        ):
            raise ValueError("Rectified-flow shift must be finite and > 0.")

    @property
    def enabled(self) -> bool:
        return self.schedule != "disabled"

    def timestep_cdf(
        self,
        noise_levels: Any,
        *,
        flow_shifts: Any | None = None,
    ) -> Any:
        """Return the configured sampler CDF at post-shift noise levels."""
        if torch is None:  # pragma: no cover
            raise RuntimeError("Timestep Expert-Choice capacity requires torch.")
        values = torch.as_tensor(noise_levels)
        if not values.dtype.is_floating_point:
            values = values.float()
        values = values.float().clamp(1e-7, 1.0 - 1e-7)
        shift = (
            float(self.flow_shift)
            if flow_shifts is None
            else torch.as_tensor(
                flow_shifts,
                device=values.device,
                dtype=values.dtype,
            )
        )
        if torch.is_tensor(shift) and tuple(shift.shape) != tuple(values.shape):
            raise ValueError("flow_shifts must match the noise-level batch shape.")
        values = (
            values
            / (shift - (shift - 1.0) * values)
        ).clamp(1e-7, 1.0 - 1e-7)
        if self.timestep_sampling == "uniform":
            return values
        if self.timestep_sampling == "logit_normal":
            logits = torch.logit(values)
            z = (
                logits - float(self.timestep_sampling_mean)
            ) / float(self.timestep_sampling_std)
            return (0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))).clamp(0.0, 1.0)

        # Mode-shift samples are T(u), with u uniform and T strictly decreasing
        # over the schema-validated scale range. Invert T without host sync;
        # F_T(t) = P[T(U) <= t] = 1 - u(t).
        lower = torch.zeros_like(values)
        upper = torch.ones_like(values)
        scale = float(self.timestep_sampling_mode_scale)
        for _ in range(32):
            uniform = (lower + upper) * 0.5
            cosine = torch.cos((math.pi / 2.0) * uniform)
            sampled = (
                1.0
                - uniform
                - scale * (cosine.square() - 1.0 + uniform)
            )
            lower = torch.where(sampled > values, uniform, lower)
            upper = torch.where(sampled > values, upper, uniform)
        return (1.0 - (lower + upper) * 0.5).clamp(0.0, 1.0)

    def capacities(
        self,
        noise_levels: Any,
        *,
        tokens_per_sample: int,
        num_experts: int,
        fallback_capacity_factor: float,
        flow_shifts: Any | None = None,
    ) -> Any:
        """Resolve one integer capacity per sample.

        The maximum deviation is expressed as a capacity-factor span and is
        reduced symmetrically when the static capacity lies near the valid
        ``[1, tokens_per_sample]`` boundary.
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("Timestep Expert-Choice capacity requires torch.")
        tokens = int(tokens_per_sample)
        experts = int(num_experts)
        factor = float(fallback_capacity_factor)
        if tokens <= 0 or experts <= 0 or not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(
                "tokens_per_sample, num_experts, and fallback_capacity_factor "
                "must be positive."
            )
        values = torch.as_tensor(noise_levels)
        if values.ndim != 1:
            raise ValueError("noise_levels must have shape [batch].")
        base_capacity = min(
            tokens,
            max(1, math.ceil(tokens * factor / experts)),
        )
        if not self.enabled:
            return torch.full(
                (int(values.shape[0]),),
                base_capacity,
                device=values.device,
                dtype=torch.int64,
            )
        requested_delta = max(
            1,
            math.ceil(
                tokens * float(self.capacity_factor_span) / experts
            ),
        )
        capacity_delta = min(
            requested_delta,
            base_capacity - 1,
            tokens - base_capacity,
        )
        if capacity_delta <= 0:
            return torch.full(
                (int(values.shape[0]),),
                base_capacity,
                device=values.device,
                dtype=torch.int64,
            )
        quantile = self.timestep_cdf(values, flow_shifts=flow_shifts)
        offsets = torch.round(
            (1.0 - 2.0 * quantile) * float(capacity_delta)
        ).to(torch.int64)
        return (base_capacity + offsets).clamp(1, tokens)
