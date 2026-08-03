"""Decorator-based registries used by the training engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class RegistryError(ValueError):
    """Base registry error."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a key is registered twice."""


class MissingRegistrationError(RegistryError):
    """Raised when a key is not found."""


@dataclass
class Registry(Generic[T]):
    """Simple lowercased string-key registry."""

    kind: str
    _items: dict[str, T] = field(default_factory=dict)

    def register(self, name: str, value: T) -> T:
        key = self._normalize(name)
        if key in self._items:
            raise DuplicateRegistrationError(
                f"Duplicate {self.kind} registration for '{name}'."
            )
        self._items[key] = value
        return value

    def decorator(self, name: str) -> Callable[[T], T]:
        def _inner(value: T) -> T:
            return self.register(name, value)

        return _inner

    def get(self, name: str) -> T:
        key = self._normalize(name)
        if key not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise MissingRegistrationError(
                f"Unknown {self.kind} '{name}'. Available: {available}."
            )
        return self._items[key]

    def has(self, name: str) -> bool:
        return self._normalize(name) in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def ordered_names(self) -> list[str]:
        """Registered keys in registration order."""
        return list(self._items)

    def values(self) -> list[T]:
        """Registered values in registration order."""
        return list(self._items.values())

    def items(self) -> list[tuple[str, T]]:
        """(key, value) pairs in registration order."""
        return list(self._items.items())

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()


StrategyRegistry: Registry[type] = Registry("strategy")
LossRegistry: Registry[Callable] = Registry("loss")
ObjectiveRegistry: Registry[type] = Registry("objective")


def register_strategy(name: str) -> Callable[[type], type]:
    return StrategyRegistry.decorator(name)


def register_loss(name: str) -> Callable[[Callable], Callable]:
    return LossRegistry.decorator(name)


def register_objective(name: str) -> Callable[[type], type]:
    return ObjectiveRegistry.decorator(name)
