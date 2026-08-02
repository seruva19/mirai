"""Timestep-axis adapter behaviors: T-LoRA rank schedule + per-adapter bands.

The noise coordinate is the POST-shift sigma in [0, 1] (1.0 = pure noise /
earliest timestep, 0.0 = clean), computed with the same ``shifted_sigma`` used
by the flow forward process, so bands are applied *after* flow_shift (matching
the diffusion-pipe / DiffSynth per-expert min_t/max_t convention).

Both features reduce to a single rank mask multiplied into the low-rank
intermediate (``x @ A^T``):

- T-LoRA (arXiv 2507.05964): a prefix mask over the rank axis. Masked ranks
  produce exactly-zero output and exactly-zero grads for the corresponding
  ``lora_a`` rows (down * 0 kills dL/dA) and ``lora_b`` columns (zero input
  kills dL/dB) -- the MoE-Sieve multiply-by-zero pattern.
- Timestep bands: an out-of-band sample's mask row is all-zero, so the whole
  adapter contribution (output AND grads) is exactly zero for that sample
  while the frozen-base forward and the loss are unaffected.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# "tc_gate" selects the TC-LoRA learned gate (mirai/core/models/tc_lora.py) as
# the rank-axis modulation. On this substrate it behaves like "none" for the
# schedule mask (all-ones rank prefix); the hypernetwork gate is multiplied in
# by the owning pipeline, and the timestep band still applies orthogonally.
VALID_TIMESTEP_RANK_SCHEDULES = ("none", "tlora", "tc_gate")
def tlora_rank_fraction(sigma: float, min_fraction: float) -> float:
    """Active-rank fraction for T-LoRA at post-shift noise level ``sigma``.

    Linear schedule: ``min_fraction`` of the rank is active at ``sigma == 1``
    (high noise / early timestep) growing to the full rank at ``sigma == 0``
    (clean). Clamped to ``[min_fraction, 1.0]``.
    """
    s = min(1.0, max(0.0, float(sigma)))
    floor = min(1.0, max(0.0, float(min_fraction)))
    fraction = floor + (1.0 - floor) * (1.0 - s)
    return min(1.0, max(floor, fraction))


def tlora_active_rank(sigma: float, *, rank: int, min_fraction: float) -> int:
    """Number of active rank columns (>= 1) under the tlora schedule."""
    fraction = tlora_rank_fraction(sigma, min_fraction)
    return max(1, min(int(rank), int(math.ceil(fraction * int(rank)))))


def band_is_active(sigma: float, band_min: float, band_max: float) -> bool:
    """Whether post-shift ``sigma`` falls in the adapter band ``[min, max)``.

    The top edge is inclusive when ``band_max`` is the full-range default 1.0,
    so the highest-noise samples (sigma == 1.0) are never silently dropped.
    """
    s = float(sigma)
    lo = float(band_min)
    hi = float(band_max)
    if s < lo:
        return False
    if hi >= 1.0:
        return s <= 1.0
    return s < hi


def full_range_band(band_min: float, band_max: float) -> bool:
    """True when the band is the default full range [0, 1]."""
    return float(band_min) <= 0.0 and float(band_max) >= 1.0


class TimestepAdapterPolicy:
    """Immutable per-adapter timestep policy (rank schedule + band).

    ``from_adapter_config(...)`` returns ``None`` for the disabled policy.
    """

    __slots__ = ("schedule", "min_fraction", "band_min", "band_max")

    def __init__(
        self,
        *,
        schedule: str = "none",
        min_fraction: float = 0.25,
        band_min: float = 0.0,
        band_max: float = 1.0,
    ) -> None:
        normalized = str(schedule).strip().lower()
        if normalized not in VALID_TIMESTEP_RANK_SCHEDULES:
            raise ValueError(
                "timestep_rank_schedule must be one of: "
                + ", ".join(VALID_TIMESTEP_RANK_SCHEDULES)
                + "."
            )
        if not (0.0 < float(min_fraction) <= 1.0):
            raise ValueError("timestep_rank_min_fraction must be in (0, 1].")
        if not (0.0 <= float(band_min) < float(band_max) <= 1.0):
            raise ValueError(
                "timestep band requires 0 <= band_min < band_max <= 1."
            )
        self.schedule = normalized
        self.min_fraction = float(min_fraction)
        self.band_min = float(band_min)
        self.band_max = float(band_max)

    def is_noop(self) -> bool:
        return self.schedule == "none" and full_range_band(
            self.band_min, self.band_max
        )

    @classmethod
    def from_adapter_config(cls, adapter_config: Any) -> "TimestepAdapterPolicy | None":
        """Build from an AdapterConfig-like object; None for the noop default."""
        policy = cls(
            schedule=str(
                getattr(adapter_config, "timestep_rank_schedule", "none")
            ),
            min_fraction=float(
                getattr(adapter_config, "timestep_rank_min_fraction", 0.25)
            ),
            band_min=float(getattr(adapter_config, "timestep_band_min", 0.0)),
            band_max=float(getattr(adapter_config, "timestep_band_max", 1.0)),
        )
        return None if policy.is_noop() else policy


def per_sample_rank_mask(
    sigmas: Any,
    *,
    rank: int,
    policy: TimestepAdapterPolicy,
) -> Any:
    """Per-sample rank mask ``[B, rank]`` (float32, on ``sigmas``' device).

    Row b is the prefix mask for sample b's active rank under the policy's
    schedule, zeroed entirely when sample b is outside the timestep band.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("per_sample_rank_mask requires torch.")
    sig = sigmas.detach().reshape(-1).float()
    batch = int(sig.numel())
    r = int(rank)
    columns = torch.arange(r, device=sig.device, dtype=torch.float32)
    if policy.schedule == "tlora":
        clamped = sig.clamp(0.0, 1.0)
        floor = float(policy.min_fraction)
        fraction = floor + (1.0 - floor) * (1.0 - clamped)
        active = (fraction * float(r)).ceil().clamp(min=1.0, max=float(r))
        mask = (columns.unsqueeze(0) < active.unsqueeze(1)).float()
    else:
        mask = torch.ones((batch, r), device=sig.device, dtype=torch.float32)
    if not full_range_band(policy.band_min, policy.band_max):
        in_band = sig >= float(policy.band_min)
        if float(policy.band_max) < 1.0:
            in_band = in_band & (sig < float(policy.band_max))
        else:
            in_band = in_band & (sig <= 1.0)
        mask = mask * in_band.float().unsqueeze(1)
    return mask


def batch_uniform_rank_mask(
    sigmas: Any,
    *,
    rank: int,
    policy: TimestepAdapterPolicy,
) -> Any:
    """Conservative batch-uniform mask ``[rank]``: elementwise min over samples.

    For prefix masks this equals the smallest active rank in the batch; any
    out-of-band sample zeroes the whole mask. Used where the sample axis is
    not recoverable inside the adapter (routed-expert token layouts), which
    preserves the exact zero-output/zero-grad guarantee at the cost of
    under-training mixed batches (exact per-sample when batch_size == 1).
    """
    return per_sample_rank_mask(sigmas, rank=rank, policy=policy).amin(dim=0)
