"""Sequential residency for independently executable pipeline components."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ComponentResidencySpec:
    name: str
    module: Any
    compute_dtype: Any | None = None


class SequentialComponentResidency:
    """Permit exactly one registered component on the compute device."""

    def __init__(self, *, compute_device: Any, idle_device: Any = "cpu") -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Sequential component residency requires torch.")
        self.compute_device = torch.device(compute_device)
        self.idle_device = torch.device(idle_device)
        self._components: dict[str, ComponentResidencySpec] = {}
        self._active: str | None = None
        self._loads = 0
        self._offloads = 0

    def register(self, spec: ComponentResidencySpec) -> None:
        name = str(spec.name).strip()
        if not name:
            raise ValueError("Sequential component residency name must not be empty.")
        if name in self._components:
            raise ValueError(f"Sequential component '{name}' is already registered.")
        spec.module.to(device=self.idle_device)
        self._components[name] = spec

    @contextmanager
    def activate(self, name: str) -> Iterator[Any]:
        key = str(name)
        if key not in self._components:
            raise KeyError(f"Unknown sequential component '{key}'.")
        if self._active is not None:
            raise RuntimeError(
                f"Sequential component '{self._active}' is already active; "
                "nested activation is not supported."
            )
        spec = self._components[key]
        kwargs = {"device": self.compute_device}
        if spec.compute_dtype is not None:
            kwargs["dtype"] = spec.compute_dtype
        spec.module.to(**kwargs)
        self._active = key
        self._loads += 1
        try:
            yield spec.module
        finally:
            if self.compute_device.type == "cuda":
                torch.cuda.synchronize(self.compute_device)
            spec.module.to(device=self.idle_device)
            self._active = None
            self._offloads += 1
            if self.compute_device.type == "cuda":
                torch.cuda.empty_cache()

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy": "sequential_component_offload",
            "compute_device": str(self.compute_device),
            "idle_device": str(self.idle_device),
            "registered_components": sorted(self._components),
            "active_component": self._active or "",
            "loads": int(self._loads),
            "offloads": int(self._offloads),
        }
