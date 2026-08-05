# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constructor-argument registration for the vendored MAGI-2 Preview modules.

Upstream MAGI-2 derives its VAE decoder and Flow-UniPC scheduler from the
Diffusers ``ConfigMixin``/``ModelMixin``/``SchedulerMixin`` classes. The Mirai
runtime is native-only, so this module reproduces the exact surface those
vendored classes consume -- the ``@register_to_config`` init decorator, the
``self.config.<key>`` namespace, in-place ``self.register_to_config(...)``
overrides, and ``cls.from_config(mapping)`` -- without a Diffusers dependency.

Registration semantics match the upstream decorator: values are recorded before
the decorated ``__init__`` body runs, so a body-level ``register_to_config``
call overrides the registered value; positional arguments are aligned with the
signature; and ``from_config`` keeps only keys named by the signature, dropping
private (``_``-prefixed) and unknown keys.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Iterator, Mapping
from typing import Any

__all__ = ["FrozenConfig", "NativeConfigMixin", "register_to_config"]


class FrozenConfig(Mapping):
    """Read-only mapping that also exposes its keys as attributes."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "_values", dict(values or {}))

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, "_values")[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FrozenConfig is immutable; use register_to_config().")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("FrozenConfig is immutable; use register_to_config().")

    def __repr__(self) -> str:
        return f"FrozenConfig({self._values!r})"


def _named_parameters(function: Any) -> dict[str, Any]:
    """Signature parameters that carry a name, excluding ``self`` and var-args."""
    signature = inspect.signature(function)
    named: dict[str, Any] = {}
    for index, (name, parameter) in enumerate(signature.parameters.items()):
        if index == 0 and name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        named[name] = parameter.default
    return named


def register_to_config(init):
    """Record the decorated ``__init__`` arguments on ``self.config``."""

    @functools.wraps(init)
    def inner_init(self, *args: Any, **kwargs: Any) -> None:
        if not isinstance(self, NativeConfigMixin):
            raise RuntimeError(
                f"@register_to_config was applied to {type(self).__name__}.__init__, "
                "but the class does not inherit from NativeConfigMixin."
            )
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        private_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}

        parameters = _named_parameters(init)
        registered: dict[str, Any] = {}
        for value, name in zip(args, parameters):
            registered[name] = value
        for name, default in parameters.items():
            if name in registered:
                continue
            if name in init_kwargs:
                registered[name] = init_kwargs[name]
            elif default is not inspect.Parameter.empty:
                registered[name] = default

        self.register_to_config(**{**private_kwargs, **registered})
        init(self, *args, **init_kwargs)

    return inner_init


class NativeConfigMixin:
    """Native stand-in for the Diffusers ``ConfigMixin`` surface."""

    @property
    def config(self) -> FrozenConfig:
        internal = getattr(self, "_internal_dict", None)
        if internal is None:
            raise AttributeError(
                f"{type(self).__name__} has no registered config; decorate "
                "__init__ with @register_to_config."
            )
        return internal

    def register_to_config(self, **kwargs: Any) -> None:
        values = dict(getattr(self, "_internal_dict", None) or {})
        values.update(kwargs)
        object.__setattr__(self, "_internal_dict", FrozenConfig(values))

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> Any:
        values = {
            key: value
            for key, value in dict(config).items()
            if not key.startswith("_")
        }
        values.update(kwargs)
        accepted = _named_parameters(cls.__init__)
        return cls(**{k: v for k, v in values.items() if k in accepted})
