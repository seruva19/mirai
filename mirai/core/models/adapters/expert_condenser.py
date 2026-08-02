"""Shared expert-condenser lifecycle, math, and adapter-state contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


CONDENSER_A_SUFFIX = ".condenser_a"
CONDENSER_B_SUFFIX = ".condenser_b"
CONDENSER_ALPHA_SUFFIX = ".condenser_alpha"


class ExpertCondenserHost(Protocol):
    lora_a: Any
    lora_b: Any
    condenser_rank: int
    cond_a: Any | None
    cond_b: Any | None
    condenser_alpha: Any
    _lora_scale: float
    _condenser_alpha_value: float | None

    def register_buffer(self, name: str, tensor: Any, *, persistent: bool) -> None: ...
    def _initialize_condenser_a(self, tensor: Any, init: str) -> None: ...


@dataclass(frozen=True)
class ExpertCondenserComponent:
    """Operate an optional shared low-rank term while the host owns its tensors."""

    host: ExpertCondenserHost

    def initialize_disabled(self) -> None:
        self.host.condenser_rank = 0
        self.host.cond_a = None
        self.host.cond_b = None
        self.host._condenser_alpha_value = None

    def enable(self, *, rank: int, alpha: float, init: str = "kaiming") -> None:
        rank = int(rank)
        if rank <= 0 or self.host.cond_a is not None:
            return
        in_features = int(self.host.lora_a.shape[2])
        out_features = int(self.host.lora_b.shape[1])
        self.host.condenser_rank = rank
        self.host.cond_a = nn.Parameter(torch.empty(rank, in_features))
        self.host.cond_b = nn.Parameter(torch.zeros(out_features, rank))
        self.host.register_buffer(
            "condenser_alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
            persistent=True,
        )
        self.host._initialize_condenser_a(self.host.cond_a, init)
        self.reset_cache()

    def is_enabled(self) -> bool:
        return self.host.cond_a is not None

    def reset_cache(self) -> None:
        self.host._condenser_alpha_value = None

    def delta(self, x: Any) -> Any | None:
        if self.host.cond_a is None:
            return None
        a = self.host.cond_a.to(device=x.device, dtype=x.dtype)
        b = self.host.cond_b.to(device=x.device, dtype=x.dtype)
        return F.linear(F.linear(x, a), b) * self._scale()

    def _scale(self) -> float:
        if self.host._condenser_alpha_value is None:
            self.host._condenser_alpha_value = float(
                self.host.condenser_alpha.detach().float().item()
            )
        return (
            self.host._condenser_alpha_value
            / float(self.host.condenser_rank)
            * self.host._lora_scale
        )


def write_condenser_state(
    state: dict[str, Any], *, name: str, module: Any
) -> bool:
    if getattr(module, "cond_a", None) is None:
        return False
    state[f"{name}{CONDENSER_A_SUFFIX}"] = module.cond_a.detach()
    state[f"{name}{CONDENSER_B_SUFFIX}"] = module.cond_b.detach()
    state[f"{name}{CONDENSER_ALPHA_SUFFIX}"] = module.condenser_alpha.detach().reshape(1)
    return True


def load_condenser_state(
    state: dict[str, Any], *, name: str, module: Any
) -> bool:
    a_key = f"{name}{CONDENSER_A_SUFFIX}"
    if a_key not in state:
        return False
    if getattr(module, "cond_a", None) is None:
        raise ValueError(
            f"Adapter state supplies condenser factors for '{name}' but the module "
            "has no condenser attached (set adapter.condenser_rank before loading)."
        )
    module.cond_a.copy_(_adapter_tensor(state[a_key], like=module.cond_a, name=a_key))
    b_key = f"{name}{CONDENSER_B_SUFFIX}"
    module.cond_b.copy_(_adapter_tensor(state[b_key], like=module.cond_b, name=b_key))
    alpha_key = f"{name}{CONDENSER_ALPHA_SUFFIX}"
    if alpha_key in state:
        module.condenser_alpha.copy_(
            _adapter_tensor(
                state[alpha_key], like=module.condenser_alpha, name=alpha_key
            ).reshape_as(module.condenser_alpha)
        )
    component = getattr(module, "_condenser", None)
    if isinstance(component, ExpertCondenserComponent):
        component.reset_cache()
    elif hasattr(module, "_condenser_alpha_value"):
        module._condenser_alpha_value = None
    return True


def _adapter_tensor(value: Any, *, like: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=like.dtype)
    tensor = value.detach().to(device=like.device, dtype=like.dtype)
    if tuple(tensor.shape) != tuple(like.shape):
        if int(tensor.numel()) != int(like.numel()):
            raise ValueError(
                f"Adapter tensor '{name}' has shape {tuple(tensor.shape)}, "
                f"expected {tuple(like.shape)}."
            )
        tensor = tensor.reshape_as(like)
    return tensor
