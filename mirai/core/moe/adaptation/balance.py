"""Typed MoE load-balance policy and effective training weights."""

from __future__ import annotations

from dataclasses import dataclass


MOE_BALANCE_MODES = ("aux_loss", "bias_only", "off")


def normalize_moe_balance_mode(value: str | None) -> str:
    text = str(value if value is not None else "aux_loss").strip().lower()
    aliases = {"": "aux_loss", "aux": "aux_loss", "bias": "bias_only", "none": "off"}
    normalized = aliases.get(text, text)
    if normalized not in MOE_BALANCE_MODES:
        raise ValueError(
            "model.params.moe_balance_mode must be one of: "
            + ", ".join(MOE_BALANCE_MODES)
            + "."
        )
    return normalized


@dataclass(frozen=True)
class MoEBalancePolicy:
    """Resolved balance behavior consumed by model-family pipelines."""

    mode: str
    aux_loss_weight: float
    bias_update_rate: float

    @classmethod
    def resolve(
        cls,
        mode: str | None,
        *,
        aux_loss_weight: float,
        bias_update_rate: float,
    ) -> MoEBalancePolicy:
        resolved = normalize_moe_balance_mode(mode)
        aux = float(aux_loss_weight)
        bias = float(bias_update_rate)
        if resolved == "bias_only":
            aux = 0.0
        elif resolved == "off":
            aux = 0.0
            bias = 0.0
        return cls(mode=resolved, aux_loss_weight=aux, bias_update_rate=bias)


def resolve_moe_balance_weights(
    mode: str | None,
    *,
    aux_loss_weight: float,
    bias_update_rate: float,
) -> tuple[float, float]:
    policy = MoEBalancePolicy.resolve(
        mode,
        aux_loss_weight=aux_loss_weight,
        bias_update_rate=bias_update_rate,
    )
    return policy.aux_loss_weight, policy.bias_update_rate
