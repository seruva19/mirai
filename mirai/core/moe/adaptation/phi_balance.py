"""Population-level MoE load balancing with checkpointable EMA mirror state.

Reference: https://arxiv.org/abs/2605.15403, Algorithm 1 and Equation 12.
"""

from __future__ import annotations

from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PHI_BALANCE_STATE_SCHEMA_VERSION = 1
PHI_BALANCE_POTENTIALS = ("negative_entropy", "euclidean")


def normalize_phi_balance_potential(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in PHI_BALANCE_POTENTIALS:
        raise ValueError(
            "phi-balance potential must be one of: "
            + ", ".join(PHI_BALANCE_POTENTIALS)
            + "."
        )
    return normalized


def phi_mirror_price(
    population_mean: Any,
    *,
    potential: str,
    epsilon: float = 1e-12,
) -> Any:
    """Return ``grad phi(m)`` for a detached population-load estimate."""
    name = normalize_phi_balance_potential(potential)
    values = population_mean.detach().float()
    if values.ndim != 1 or int(values.numel()) < 2:
        raise ValueError("phi-balance population mean must be a vector of experts.")
    if not torch.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("phi-balance population mean must be finite and non-negative.")
    if name == "negative_entropy":
        return 1.0 + values.clamp_min(float(epsilon)).log()
    return values


class PhiBalanceController:
    """Maintain per-layer EMA soft loads and build the paper's auxiliary term."""

    def __init__(self, *, ema_rate: float, potential: str = "negative_entropy") -> None:
        rate = float(ema_rate)
        if not 0.0 < rate <= 1.0:
            raise ValueError("phi-balance ema_rate must be in (0, 1].")
        self.ema_rate = rate
        self.potential = normalize_phi_balance_potential(potential)
        self._means: dict[str, Any] = {}

    def loss(self, layer: str, probabilities: Any) -> Any:
        """Update detached EMA and return ``<p_t, grad phi(m_t+1)>``."""
        if torch is None or not torch.is_tensor(probabilities):
            raise RuntimeError("phi-balancing requires torch tensors.")
        if probabilities.ndim < 2 or int(probabilities.shape[-1]) < 2:
            raise ValueError("phi-balance probabilities must end in an expert dimension.")
        current = probabilities.float().reshape(-1, probabilities.shape[-1]).mean(dim=0)
        if not torch.isfinite(current).all() or bool((current < 0).any()):
            raise ValueError("phi-balance probabilities must be finite and non-negative.")
        key = str(layer)
        previous = self._means.get(key)
        detached = current.detach()
        if previous is None:
            previous = torch.zeros_like(detached)
        elif tuple(previous.shape) != tuple(detached.shape):
            raise ValueError(f"phi-balance expert topology changed for layer {key!r}.")
        updated = (1.0 - self.ema_rate) * previous.to(detached) + self.ema_rate * detached
        self._means[key] = updated.detach()
        price = phi_mirror_price(updated, potential=self.potential).to(current)
        return torch.sum(current * price.detach())

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHI_BALANCE_STATE_SCHEMA_VERSION,
            "ema_rate": self.ema_rate,
            "potential": self.potential,
            "means": {
                name: value.detach().cpu().clone()
                for name, value in sorted(self._means.items())
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", 0)) != PHI_BALANCE_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported phi-balance state schema.")
        if float(state.get("ema_rate", -1.0)) != self.ema_rate:
            raise ValueError("phi-balance checkpoint ema_rate does not match config.")
        if normalize_phi_balance_potential(str(state.get("potential", ""))) != self.potential:
            raise ValueError("phi-balance checkpoint potential does not match config.")
        raw_means = state.get("means")
        if not isinstance(raw_means, Mapping):
            raise ValueError("phi-balance checkpoint requires layer means.")
        means: dict[str, Any] = {}
        for name, raw in raw_means.items():
            value = torch.as_tensor(raw).detach().float().cpu()
            if value.ndim != 1 or int(value.numel()) < 2 or not torch.isfinite(value).all():
                raise ValueError(f"Invalid phi-balance mean for layer {name!r}.")
            means[str(name)] = value.clone()
        self._means = means


def phi_balance_checkpoint_state(controller: PhiBalanceController) -> dict[str, Any]:
    state = controller.state_dict()
    return {
        "phi_balance.schema_version": torch.tensor(
            int(state["schema_version"]), dtype=torch.int64
        ),
        "phi_balance.ema_rate": torch.tensor(float(state["ema_rate"])),
        "phi_balance.potential_code": torch.tensor(
            PHI_BALANCE_POTENTIALS.index(str(state["potential"])), dtype=torch.int64
        ),
        **{
            f"phi_balance.mean.{name}": value
            for name, value in state["means"].items()
        },
    }


def load_phi_balance_checkpoint_state(
    controller: PhiBalanceController, state: Mapping[str, Any]
) -> None:
    prefix = "phi_balance.mean."
    means = {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    schema = state.get("phi_balance.schema_version")
    if schema is None and not means:
        return
    potential_code = int(torch.as_tensor(state.get("phi_balance.potential_code")).item())
    if potential_code < 0 or potential_code >= len(PHI_BALANCE_POTENTIALS):
        raise ValueError("Invalid phi-balance checkpoint potential code.")
    controller.load_state_dict(
        {
            "schema_version": int(torch.as_tensor(schema).item()),
            "ema_rate": float(torch.as_tensor(state.get("phi_balance.ema_rate")).item()),
            "potential": PHI_BALANCE_POTENTIALS[potential_code],
            "means": means,
        }
    )


def optional_phi_balance_checkpoint_state(
    controller: PhiBalanceController | None,
) -> dict[str, Any]:
    return {} if controller is None else phi_balance_checkpoint_state(controller)


def load_optional_phi_balance_checkpoint_state(
    controller: PhiBalanceController | None, state: Mapping[str, Any]
) -> None:
    present = any(str(key).startswith("phi_balance.") for key in state)
    if controller is None:
        if present:
            raise ValueError(
                "Checkpoint contains phi-balance state but the feature is disabled."
            )
        return
    load_phi_balance_checkpoint_state(controller, state)


__all__ = [
    "PHI_BALANCE_POTENTIALS",
    "PHI_BALANCE_STATE_SCHEMA_VERSION",
    "PhiBalanceController",
    "load_phi_balance_checkpoint_state",
    "load_optional_phi_balance_checkpoint_state",
    "normalize_phi_balance_potential",
    "phi_balance_checkpoint_state",
    "optional_phi_balance_checkpoint_state",
    "phi_mirror_price",
]
