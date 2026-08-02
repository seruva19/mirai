"""Registered physical expert-weight providers for packed MoE artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from mirai.core.registry import Registry


@dataclass(frozen=True)
class PhysicalWeightProviderContext:
    """Inputs available when a packed artifact constructs a provider."""

    module_name: str
    num_experts: int
    shapes: Mapping[str, tuple[int, ...]]
    spec: Mapping[str, Any]
    tensors: Mapping[str, Any]


@runtime_checkable
class PhysicalExpertWeightProvider(Protocol):
    """Decode immutable physical expert weights without owning routing policy."""

    name: str
    num_experts: int

    def expert_weight_shape(self, key: str) -> tuple[int, ...]: ...

    def materialize_expert(
        self,
        key: str,
        expert_index: int,
        *,
        dtype: Any,
        device: Any,
    ) -> Any: ...

    def packed_tensor_names(self) -> frozenset[str]: ...

    def manifest_spec(self) -> Mapping[str, Any]: ...

    def packed_tensors(self) -> Mapping[str, Any]: ...


PhysicalWeightProviderBuilder = Callable[
    [PhysicalWeightProviderContext], PhysicalExpertWeightProvider
]
PhysicalWeightProviderRegistry: Registry[PhysicalWeightProviderBuilder] = Registry(
    "physical expert weight provider"
)


def physical_weight_provider_names(
    manifest: Mapping[str, Any],
) -> frozenset[str]:
    """Return provider names from a packed manifest without importing plugins."""
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping):
        raise ValueError("Packed state manifest has no modules object.")
    providers: set[str] = set()
    for module_name, raw_spec in modules.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Packed module {module_name!r} must be an object.")
        provider = raw_spec.get("physical_weight_provider")
        if provider is None:
            continue
        if not isinstance(provider, Mapping) or not str(provider.get("name", "")):
            raise ValueError(
                f"Packed module {module_name!r} has an invalid physical provider spec."
            )
        providers.add(str(provider["name"]))
    return frozenset(providers)


def validate_physical_weight_provider_selection(
    manifest: Mapping[str, Any],
    configured: str,
) -> frozenset[str]:
    """Require an exact match between an explicit gate and artifact providers."""
    providers = physical_weight_provider_names(manifest)
    selected = str(configured).strip().lower()
    if providers and selected == "off":
        raise ValueError(
            "Packed state contains physical expert-weight providers "
            f"{sorted(providers)!r}, but expert-weight compression is off."
        )
    if providers != ({selected} if selected != "off" else set()):
        raise ValueError(
            "Expert-weight compression mismatch: configuration requests "
            f"{selected!r}, artifact contains {sorted(providers)!r}."
        )
    return providers


def register_physical_weight_provider(
    name: str,
) -> Callable[[PhysicalWeightProviderBuilder], PhysicalWeightProviderBuilder]:
    return PhysicalWeightProviderRegistry.decorator(name)


def build_physical_weight_provider(
    name: str,
    context: PhysicalWeightProviderContext,
) -> PhysicalExpertWeightProvider:
    provider = PhysicalWeightProviderRegistry.get(str(name))(context)
    if not isinstance(provider, PhysicalExpertWeightProvider):
        raise TypeError(
            f"Physical weight provider {name!r} does not implement the provider contract."
        )
    return provider


__all__ = [
    "PhysicalExpertWeightProvider",
    "PhysicalWeightProviderBuilder",
    "PhysicalWeightProviderContext",
    "PhysicalWeightProviderRegistry",
    "build_physical_weight_provider",
    "physical_weight_provider_names",
    "register_physical_weight_provider",
    "validate_physical_weight_provider_selection",
]
