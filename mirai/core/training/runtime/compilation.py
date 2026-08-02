"""Model-agnostic regional compilation for repeated training modules."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}
_COMPILE_SCOPES = {"full", "regional"}


@dataclass(frozen=True)
class CompilationPolicy:
    """Validated user policy for opt-in graph compilation."""

    enabled: bool = False
    scope: str = "regional"
    mode: str = "default"
    dynamic: bool | None = None
    token_buckets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        scope = str(self.scope).strip().lower()
        mode = str(self.mode).strip().lower()
        buckets = tuple(int(value) for value in self.token_buckets)
        if scope not in _COMPILE_SCOPES:
            raise ValueError(
                "training.compile_scope must be one of: full, regional."
            )
        if mode not in _COMPILE_MODES:
            raise ValueError(
                "training.compile_mode must be one of: default, reduce-overhead, "
                "max-autotune, max-autotune-no-cudagraphs."
            )
        if any(value <= 0 for value in buckets):
            raise ValueError("training.compile_token_buckets values must be > 0.")
        if any(right <= left for left, right in zip(buckets, buckets[1:])):
            raise ValueError(
                "training.compile_token_buckets must be strictly increasing."
            )
        if buckets and scope != "regional":
            raise ValueError(
                "training.compile_token_buckets requires compile_scope='regional'."
            )
        if buckets and self.dynamic is False:
            raise ValueError(
                "training.compile_token_buckets requires compile_dynamic=true or "
                "the default automatic dynamic-shape policy."
            )
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "token_buckets", buckets)

    @classmethod
    def from_training_config(cls, training: Any) -> "CompilationPolicy":
        return cls(
            enabled=bool(getattr(training, "compile", False)),
            scope=str(getattr(training, "compile_scope", "regional")),
            mode=str(getattr(training, "compile_mode", "default")),
            dynamic=getattr(training, "compile_dynamic", None),
            token_buckets=tuple(
                int(value)
                for value in getattr(training, "compile_token_buckets", ())
            ),
        )

    def compile_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.mode != "default":
            kwargs["mode"] = self.mode
        if self.dynamic is not None:
            kwargs["dynamic"] = bool(self.dynamic)
        return kwargs


@dataclass
class CompilationRegion:
    """One provider-owned callable that core may compile and later restore."""

    name: str
    owner: Any
    attribute: str = "forward"
    _eager_callable: Callable[..., Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Compilation region name must not be empty.")
        target = getattr(self.owner, self.attribute, None)
        if not callable(target):
            raise TypeError(
                f"Compilation region '{name}' target "
                f"{type(self.owner).__name__}.{self.attribute} is not callable."
            )
        self.name = name
        self._eager_callable = target

    @property
    def eager_callable(self) -> Callable[..., Any]:
        return self._eager_callable

    def install(self, compiled_callable: Callable[..., Any]) -> None:
        setattr(self.owner, self.attribute, compiled_callable)

    def restore(self) -> None:
        setattr(self.owner, self.attribute, self._eager_callable)


@dataclass
class TokenBucketPlan:
    """Validate token shapes and hint that the token dimension may vary."""

    upper_bounds: tuple[int, ...]
    hits: Counter[int] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        policy = CompilationPolicy(token_buckets=self.upper_bounds)
        self.upper_bounds = policy.token_buckets

    def bounds_for(self, token_count: int) -> tuple[int, int]:
        count = int(token_count)
        lower = 1
        for upper in self.upper_bounds:
            if count <= upper:
                return lower, upper
            lower = upper + 1
        raise ValueError(
            f"Observed token count {count} exceeds the largest configured "
            f"compile token bucket ({self.upper_bounds[-1]})."
        )

    def mark(
        self,
        tensor: Any,
        *,
        dim: int,
        record_hit: bool = True,
    ) -> Any:
        if not self.upper_bounds:
            return tensor
        if torch is None:  # pragma: no cover
            raise RuntimeError("Torch is required for compile token bucketing.")
        normalized_dim = int(dim)
        if normalized_dim < 0:
            normalized_dim += int(tensor.ndim)
        if normalized_dim < 0 or normalized_dim >= int(tensor.ndim):
            raise ValueError(
                f"Compile token dimension {dim} is invalid for rank-{tensor.ndim} input."
            )
        token_count = int(tensor.shape[normalized_dim])
        _, upper = self.bounds_for(token_count)
        if record_hit:
            self.hits[upper] += 1
        marker = getattr(
            getattr(torch, "_dynamo", None), "maybe_mark_dynamic", None
        )
        if not callable(marker):
            raise RuntimeError(
                "This Torch runtime does not expose dynamic-dimension hints."
            )
        marker(tensor, normalized_dim)
        return tensor

    def snapshot(self) -> dict[str, Any]:
        return {
            "upper_bounds": list(self.upper_bounds),
            "hits": {
                str(bound): int(self.hits.get(bound, 0))
                for bound in self.upper_bounds
            },
        }


def _compiler_counters() -> dict[str, int]:
    if torch is None:
        return {}
    try:  # Torch exposes these diagnostics through a private compatibility API.
        from torch._dynamo.utils import counters

        return {
            "frames": int(counters["frames"]["total"]),
            "unique_graphs": int(counters["stats"]["unique_graphs"]),
            "calls_captured": int(counters["stats"]["calls_captured"]),
        }
    except Exception:  # pragma: no cover - Torch-version dependent diagnostics.
        return {}


@dataclass
class CompilationSession:
    """Installed compilation state with exact eager restoration."""

    pipeline: Any
    policy: CompilationPolicy
    forward_fn: Callable[..., Any]
    enabled: bool = False
    warning: str = ""
    regions: list[CompilationRegion] = field(default_factory=list)
    token_bucket_plan: TokenBucketPlan | None = None
    _counter_baseline: dict[str, int] = field(default_factory=dict, repr=False)

    def disable(self, *, warning: str = "") -> None:
        for region in reversed(self.regions):
            region.restore()
        self.regions.clear()
        configure_buckets = getattr(
            self.pipeline, "configure_compilation_token_buckets", None
        )
        if callable(configure_buckets):
            configure_buckets(None)
        self.forward_fn = self.pipeline.forward
        self.enabled = False
        if warning:
            self.warning = str(warning)

    def diagnostics(self) -> dict[str, Any]:
        current = _compiler_counters()
        delta = {
            name: max(0, int(value) - int(self._counter_baseline.get(name, 0)))
            for name, value in current.items()
        }
        return {
            "enabled": bool(self.enabled),
            "scope": self.policy.scope,
            "mode": self.policy.mode,
            "dynamic": self.policy.dynamic,
            "regions": [region.name for region in self.regions],
            "token_buckets": (
                self.token_bucket_plan.snapshot()
                if self.token_bucket_plan is not None
                else {"upper_bounds": [], "hits": {}}
            ),
            "compiler_counters": delta,
            "warning": str(self.warning),
        }


def prepare_training_compilation(
    *,
    pipeline: Any,
    policy: CompilationPolicy,
) -> CompilationSession:
    """Install full or provider-declared regional compilation."""

    session = CompilationSession(
        pipeline=pipeline,
        policy=policy,
        forward_fn=pipeline.forward,
        _counter_baseline=_compiler_counters(),
    )
    if not policy.enabled:
        return session
    try:
        if torch is None or not callable(getattr(torch, "compile", None)):
            raise RuntimeError("torch.compile is unavailable in this runtime.")
        kwargs = policy.compile_kwargs()
        if policy.scope == "full":
            session.forward_fn = torch.compile(pipeline.forward, **kwargs)
            session.enabled = True
            return session

        region_getter = getattr(pipeline, "get_compilation_regions", None)
        if not callable(region_getter):
            raise ValueError(
                f"{type(pipeline).__name__} does not expose compilation regions."
            )
        regions = list(region_getter())
        if not regions:
            raise ValueError(
                f"{type(pipeline).__name__} does not expose compilation regions."
            )
        if not all(isinstance(region, CompilationRegion) for region in regions):
            raise TypeError(
                "get_compilation_regions() must return CompilationRegion objects."
            )
        names = [region.name for region in regions]
        if len(names) != len(set(names)):
            raise ValueError("Compilation region names must be unique.")

        bucket_plan = (
            TokenBucketPlan(policy.token_buckets)
            if policy.token_buckets
            else None
        )
        configure_buckets = getattr(
            pipeline, "configure_compilation_token_buckets", None
        )
        if bucket_plan is not None:
            if not callable(configure_buckets):
                raise ValueError(
                    f"{type(pipeline).__name__} does not support compile token buckets."
                )
            configure_buckets(bucket_plan)
        session.token_bucket_plan = bucket_plan

        for region in regions:
            compiled = torch.compile(region.eager_callable, **kwargs)
            region.install(compiled)
            session.regions.append(region)
        session.forward_fn = pipeline.forward
        session.enabled = True
    except Exception as exc:
        session.disable(
            warning=(
                "torch.compile requested but disabled by compilation constraints: "
                f"{exc}"
            )
        )
    return session


__all__ = [
    "CompilationPolicy",
    "CompilationRegion",
    "CompilationSession",
    "TokenBucketPlan",
    "prepare_training_compilation",
]
