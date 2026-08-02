"""Static per-target LoRA allocation and scaling contracts."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import math
from typing import Any, Mapping, Sequence


RSLORA_STATE_SUFFIX = ".lora_use_rslora"
PARA_RANK_STATE_SUFFIX = ".lora_para_rank"


@dataclass(frozen=True)
class LoRATargetAllocation:
    """Resolved rank, alpha, and scaling rule for one adapter target."""

    name: str
    rank: int
    alpha: float
    use_rslora: bool = False

    @property
    def scaling_rule(self) -> str:
        return "alpha_over_sqrt_rank" if self.use_rslora else "alpha_over_rank"

    @property
    def scale(self) -> float:
        return lora_scale(self.alpha, self.rank, use_rslora=self.use_rslora)


@dataclass(frozen=True)
class LoRAAllocationPlan:
    """Deterministic target allocation resolved before model mutation."""

    allocations: Mapping[str, LoRATargetAllocation]
    total_rank: int
    rank_budget: int

    def for_target(self, name: str) -> LoRATargetAllocation:
        try:
            return self.allocations[str(name)]
        except KeyError as exc:
            raise KeyError(f"No LoRA allocation was resolved for target {name!r}.") from exc

    @property
    def fingerprint(self) -> tuple[tuple[str, int, float, bool], ...]:
        return tuple(
            (name, item.rank, item.alpha, item.use_rslora)
            for name, item in sorted(self.allocations.items())
        )


@dataclass(frozen=True)
class LoRAAllocationPolicy:
    """Resolve glob patterns to target-local ranks and alpha values."""

    default_rank: int
    default_alpha: float
    rank_pattern: Mapping[str, int]
    alpha_pattern: Mapping[str, float]
    rank_budget: int = 0
    use_rslora: bool = False

    @classmethod
    def from_adapter_config(cls, config: Any) -> LoRAAllocationPolicy:
        return cls(
            default_rank=int(getattr(config, "rank", 0)),
            default_alpha=float(getattr(config, "alpha", 0.0)),
            rank_pattern=dict(getattr(config, "rank_pattern", {}) or {}),
            alpha_pattern=dict(getattr(config, "alpha_pattern", {}) or {}),
            rank_budget=int(getattr(config, "rank_budget", 0)),
            use_rslora=bool(getattr(config, "use_rslora", False)),
        ).validate()

    def validate(self) -> LoRAAllocationPolicy:
        if int(self.default_rank) <= 0:
            raise ValueError("Default LoRA rank must be positive.")
        if not math.isfinite(float(self.default_alpha)) or float(self.default_alpha) <= 0.0:
            raise ValueError("Default LoRA alpha must be finite and positive.")
        if int(self.rank_budget) < 0:
            raise ValueError("LoRA rank_budget must be non-negative.")
        _validate_patterns(self.rank_pattern, kind="rank")
        _validate_patterns(self.alpha_pattern, kind="alpha")
        return self

    def resolve(
        self,
        target_names: Sequence[str],
        *,
        exact_ranks: Mapping[str, int] | None = None,
    ) -> LoRAAllocationPlan:
        self.validate()
        names = tuple(str(name) for name in target_names)
        if not names:
            raise ValueError("LoRA allocation requires at least one target.")
        if len(set(names)) != len(names):
            raise ValueError("LoRA target names must be unique before allocation.")
        resolved_exact_ranks = (
            None
            if exact_ranks is None
            else {str(name): int(rank) for name, rank in exact_ranks.items()}
        )
        if resolved_exact_ranks is not None:
            if self.rank_pattern:
                raise ValueError(
                    "adapter.rank_pattern cannot be combined with an adaptive rank plan."
                )
            missing = sorted(set(names) - set(resolved_exact_ranks))
            extra = sorted(set(resolved_exact_ranks) - set(names))
            if missing or extra:
                raise ValueError(
                    "Adaptive rank plan target topology mismatch: "
                    f"missing={missing}, extra={extra}."
                )
            if any(rank <= 0 for rank in resolved_exact_ranks.values()):
                raise ValueError("Adaptive rank plan target ranks must be positive.")
        used_rank_patterns: set[str] = set()
        used_alpha_patterns: set[str] = set()
        allocations: dict[str, LoRATargetAllocation] = {}
        for name in names:
            if resolved_exact_ranks is None:
                rank, rank_key = _resolve_one(
                    name,
                    self.rank_pattern,
                    default=int(self.default_rank),
                    kind="rank",
                )
            else:
                rank, rank_key = resolved_exact_ranks[name], None
            alpha, alpha_key = _resolve_one(
                name,
                self.alpha_pattern,
                default=float(self.default_alpha),
                kind="alpha",
            )
            if rank_key is not None:
                used_rank_patterns.add(rank_key)
            if alpha_key is not None:
                used_alpha_patterns.add(alpha_key)
            allocations[name] = LoRATargetAllocation(
                name=name,
                rank=int(rank),
                alpha=float(alpha),
                use_rslora=bool(self.use_rslora),
            )
        _raise_unmatched(self.rank_pattern, used_rank_patterns, kind="rank")
        _raise_unmatched(self.alpha_pattern, used_alpha_patterns, kind="alpha")
        total_rank = sum(item.rank for item in allocations.values())
        if int(self.rank_budget) > 0 and total_rank > int(self.rank_budget):
            raise ValueError(
                f"Resolved LoRA rank total {total_rank} exceeds rank_budget "
                f"{int(self.rank_budget)}."
            )
        return LoRAAllocationPlan(
            allocations=allocations,
            total_rank=total_rank,
            rank_budget=int(self.rank_budget),
        )


def lora_scale(alpha: float, rank: int, *, use_rslora: bool) -> float:
    resolved_rank = int(rank)
    if resolved_rank <= 0:
        raise ValueError("LoRA rank must be positive when computing scale.")
    denominator = math.sqrt(resolved_rank) if bool(use_rslora) else float(resolved_rank)
    return float(alpha) / denominator


def _validate_patterns(patterns: Mapping[str, Any], *, kind: str) -> None:
    for raw_pattern, raw_value in patterns.items():
        pattern = str(raw_pattern).strip()
        if not pattern:
            raise ValueError(f"LoRA {kind}_pattern keys must be non-empty.")
        if kind == "rank":
            if isinstance(raw_value, bool) or int(raw_value) <= 0:
                raise ValueError(f"LoRA rank_pattern[{pattern!r}] must be positive.")
        else:
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"LoRA alpha_pattern[{pattern!r}] must be finite and positive.")


def _resolve_one(
    name: str,
    patterns: Mapping[str, Any],
    *,
    default: Any,
    kind: str,
) -> tuple[Any, str | None]:
    matches = [str(pattern) for pattern in patterns if fnmatchcase(name, str(pattern))]
    if len(matches) > 1:
        raise ValueError(
            f"LoRA target {name!r} ambiguously matches {kind} patterns {sorted(matches)}."
        )
    if not matches:
        return default, None
    key = matches[0]
    return patterns[key], key


def _raise_unmatched(
    patterns: Mapping[str, Any], used: set[str], *, kind: str
) -> None:
    unmatched = sorted(set(str(key) for key in patterns) - used)
    if unmatched:
        raise ValueError(f"LoRA {kind} patterns matched no targets: {unmatched}.")


__all__ = [
    "RSLORA_STATE_SUFFIX",
    "LoRAAllocationPlan",
    "LoRAAllocationPolicy",
    "LoRATargetAllocation",
    "lora_scale",
]
