"""Token-count-aware rectified-flow timestep shifting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


FLOW_SHIFT_MODES = frozenset({"constant", "dynamic"})


def normalize_flow_shift_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in FLOW_SHIFT_MODES:
        raise ValueError(
            "Rectified-flow shift mode must be one of: "
            + ", ".join(sorted(FLOW_SHIFT_MODES))
            + "."
        )
    return normalized


@dataclass(frozen=True)
class DynamicFlowShiftPolicy:
    """Resolve the rectified-flow shift from the visual patch-token count.

    The dynamic rule is Equation 23 of *Scaling Rectified Flow Transformers
    for High-Resolution Image Synthesis* (arXiv:2403.03206):
    ``alpha(m) = alpha(n) * sqrt(m / n)``. The maximum anchor is explicit and
    validated against that relation so the schedule cannot silently become a
    different interpolation heuristic.
    """

    mode: str = "constant"
    base_shift: float = 3.0
    base_seq_len: int = 256
    max_seq_len: int = 4096
    max_shift: float = 12.0

    def __post_init__(self) -> None:
        mode = normalize_flow_shift_mode(self.mode)
        object.__setattr__(self, "mode", mode)
        base_shift = float(self.base_shift)
        max_shift = float(self.max_shift)
        base_seq_len = int(self.base_seq_len)
        max_seq_len = int(self.max_seq_len)
        if not math.isfinite(base_shift) or base_shift <= 0.0:
            raise ValueError("Rectified-flow base_shift must be finite and > 0.")
        if base_seq_len <= 0:
            raise ValueError("Rectified-flow base_seq_len must be > 0.")
        if max_seq_len < base_seq_len:
            raise ValueError(
                "Rectified-flow max_seq_len must be >= base_seq_len."
            )
        if not math.isfinite(max_shift) or max_shift <= 0.0:
            raise ValueError("Rectified-flow max_shift must be finite and > 0.")
        if mode == "dynamic":
            expected_max = base_shift * math.sqrt(
                float(max_seq_len) / float(base_seq_len)
            )
            if not math.isclose(
                max_shift,
                expected_max,
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise ValueError(
                    "Dynamic rectified-flow max_shift must equal "
                    "base_shift * sqrt(max_seq_len / base_seq_len); "
                    f"expected {expected_max:.12g}, got {max_shift:.12g}."
                )

    @property
    def enabled(self) -> bool:
        return self.mode == "dynamic"

    def shift_for_token_count(self, token_count: int) -> float:
        tokens = int(token_count)
        if tokens <= 0:
            raise ValueError("Visual patch-token count must be > 0.")
        if not self.enabled:
            return float(self.base_shift)
        bounded = min(tokens, int(self.max_seq_len))
        return float(self.base_shift) * math.sqrt(
            float(bounded) / float(self.base_seq_len)
        )

    def shifts_for_token_counts(self, token_counts: Any) -> Any:
        """Vectorized shift resolution for per-sample token counts."""
        if torch is not None and torch.is_tensor(token_counts):
            if token_counts.ndim != 1:
                raise ValueError("token_counts must have shape [batch].")
            if bool((token_counts <= 0).any()):
                raise ValueError("All visual patch-token counts must be > 0.")
            if not self.enabled:
                return torch.full(
                    token_counts.shape,
                    float(self.base_shift),
                    device=token_counts.device,
                    dtype=torch.float32,
                )
            bounded = token_counts.float().clamp(
                max=float(self.max_seq_len)
            )
            return float(self.base_shift) * torch.sqrt(
                bounded / float(self.base_seq_len)
            )
        values = tuple(int(value) for value in token_counts)
        return tuple(self.shift_for_token_count(value) for value in values)
