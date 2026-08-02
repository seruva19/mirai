"""LoRA-Pro gradient correction and AdamW updates.

The implementation follows Equations 33--34 and Algorithm 2 of LoRA-Pro
(arXiv:2407.18242). Adam moments live in the equivalent full-weight space;
this is an essential part of the published AdamW method rather than an
implementation detail.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"LoRA-Pro requires torch: {exc}")

from mirai.core.training.optim.lora_pairs import LoRAFactorPair
from mirai.core.training.optim.stochastic_rounding import (
    stochastic_round_bfloat16,
)

def estimate_lora_pro_state_bytes(
    pairs: Iterable[LoRAFactorPair],
    *,
    moment_dtype: torch.dtype = torch.float32,
) -> int:
    """Return exact first-plus-second equivalent-moment storage."""

    element_size = torch.empty((), dtype=moment_dtype).element_size()
    return sum(
        2
        * pair.batch_size
        * pair.out_features
        * pair.in_features
        * element_size
        for pair in pairs
    )


def _regularized_gram_inverse(matrix: torch.Tensor, damping: float) -> torch.Tensor:
    eye = torch.eye(
        int(matrix.shape[-1]),
        device=matrix.device,
        dtype=matrix.dtype,
    )
    gram = matrix
    if damping > 0.0:
        gram = gram + eye * float(damping)
    return torch.linalg.pinv(gram, hermitian=True)


def solve_positive_sylvester(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    *,
    damping: float,
) -> torch.Tensor:
    """Solve ``left @ X + X @ right = value`` for symmetric PSD operands."""

    left_values, left_vectors = torch.linalg.eigh(left)
    right_values, right_vectors = torch.linalg.eigh(right)
    transformed = left_vectors.transpose(-2, -1) @ value @ right_vectors
    denominator = left_values.unsqueeze(-1) + right_values.unsqueeze(-2)
    floor = max(float(damping), torch.finfo(denominator.dtype).tiny)
    signed = torch.where(
        denominator >= 0.0,
        denominator.clamp_min(floor),
        denominator.clamp_max(-floor),
    )
    solution = transformed / signed
    return left_vectors @ solution @ right_vectors.transpose(-2, -1)


def lora_pro_correct_gradients(
    *,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    scale: float,
    damping: float,
    solve_gauge: bool,
    zero_b: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply LoRA-Pro Equations 33--34 to one rank-2 factor pair."""

    a = lora_a.float()
    b = lora_b.float()
    ga = grad_a.float()
    gb = grad_b.float()
    scale_sq = float(scale) ** 2
    aa_t = a @ a.transpose(-2, -1)
    b_t_b = b.transpose(-2, -1) @ b
    aa_t_inv = _regularized_gram_inverse(aa_t, damping)

    # The standard zero-B initialization violates the theorem's full-rank
    # assumption. The reference implementation uses this finite first-step
    # branch; after the update B becomes full rank in ordinary LoRA training.
    is_zero_b = (
        bool(zero_b)
        if zero_b is not None
        else int(torch.count_nonzero(b).item()) == 0
    )
    if is_zero_b:
        return ga, (gb @ aa_t_inv) / scale_sq

    b_t_b_inv = _regularized_gram_inverse(b_t_b, damping)
    projected_gb = gb - b @ (
        b_t_b_inv @ (b.transpose(-2, -1) @ gb)
    )
    corrected_a = (b_t_b_inv @ ga) / scale_sq
    corrected_b = (projected_gb @ aa_t_inv) / scale_sq
    if not solve_gauge:
        return corrected_a, corrected_b

    rhs = -(
        b_t_b_inv @ ga @ a.transpose(-2, -1)
    ) / scale_sq
    gauge = solve_positive_sylvester(
        b_t_b,
        aa_t,
        rhs,
        damping=damping,
    )
    return corrected_a + gauge @ a, corrected_b - b @ gauge


def lora_pro_equivalent_gradient(
    *,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Evaluate the virtual full-weight gradient from Equation 54."""

    return float(scale) * (
        grad_b.float() @ lora_a.float()
        + lora_b.float() @ grad_a.float()
    )


class LoRAProAdamW(torch.optim.Optimizer):
    """Algorithm 2 LoRA-Pro AdamW with full equivalent-weight moments."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        pairs: Iterable[LoRAFactorPair],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        damping: float = 1e-8,
        stochastic_rounding: bool = False,
    ) -> None:
        if float(lr) < 0.0:
            raise ValueError("LoRA-Pro lr must be >= 0.")
        if not 0.0 <= float(betas[0]) < 1.0:
            raise ValueError("LoRA-Pro beta1 must be in [0, 1).")
        if not 0.0 <= float(betas[1]) < 1.0:
            raise ValueError("LoRA-Pro beta2 must be in [0, 1).")
        if float(eps) < 0.0:
            raise ValueError("LoRA-Pro eps must be >= 0.")
        if float(weight_decay) != 0.0:
            raise ValueError(
                "LoRA-Pro in Mirai requires weight_decay=0 to keep frozen "
                "base weights immutable."
            )
        if float(damping) <= 0.0 or not math.isfinite(float(damping)):
            raise ValueError("LoRA-Pro damping must be finite and > 0.")
        defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "weight_decay": 0.0,
        }
        super().__init__(params, defaults)
        self.pairs = tuple(pairs)
        self.damping = float(damping)
        self.stochastic_rounding = bool(stochastic_rounding)
        self._validate_pairs()

    def _validate_pairs(self) -> None:
        grouped = {
            id(param): group
            for group in self.param_groups
            for param in group["params"]
        }
        expected: set[int] = set()
        for pair in self.pairs:
            expected.update({id(pair.lora_a), id(pair.lora_b)})
            if id(pair.lora_a) not in grouped or id(pair.lora_b) not in grouped:
                raise ValueError(
                    f"LoRA-Pro pair {pair.name!r} is absent from optimizer params."
                )
            group_a = grouped[id(pair.lora_a)]
            group_b = grouped[id(pair.lora_b)]
            for key in ("lr", "betas", "eps", "weight_decay"):
                if group_a[key] != group_b[key]:
                    raise ValueError(
                        f"LoRA-Pro pair {pair.name!r} requires identical A/B {key}."
                    )
            if float(group_a["weight_decay"]) != 0.0:
                raise ValueError(
                    "LoRA-Pro requires zero weight decay for every factor pair."
                )
        actual = set(grouped)
        if actual != expected:
            raise ValueError(
                "LoRA-Pro optimizer accepts only complete standard LoRA A/B pairs."
            )
        if not self.pairs:
            raise ValueError("LoRA-Pro optimizer found no LoRA factor pairs.")

    @staticmethod
    def _matrix_views(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor

    @staticmethod
    def _initialize_state(pair: LoRAFactorPair, state: dict[str, Any]) -> None:
        shape = (
            pair.batch_size,
            pair.out_features,
            pair.in_features,
        )
        state["step"] = torch.tensor(
            0,
            dtype=torch.int64,
            device=pair.lora_a.device,
        )
        state["exp_avg"] = torch.zeros(
            shape,
            dtype=torch.float32,
            device=pair.lora_a.device,
        )
        state["exp_avg_sq"] = torch.zeros_like(state["exp_avg"])

    @staticmethod
    def _group_for(
        parameter: nn.Parameter,
        groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        for group in groups:
            if any(candidate is parameter for candidate in group["params"]):
                return group
        raise RuntimeError("LoRA-Pro parameter group disappeared.")

    def _copy_update(
        self,
        parameter: nn.Parameter,
        gradient: torch.Tensor,
        *,
        lr: float,
    ) -> None:
        updated = parameter.detach().float() - float(lr) * gradient.float()
        if self.stochastic_rounding and parameter.dtype == torch.bfloat16:
            parameter.copy_(stochastic_round_bfloat16(updated))
        else:
            parameter.copy_(
                updated.to(device=parameter.device, dtype=parameter.dtype)
            )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for pair in self.pairs:
            grad_a = pair.lora_a.grad
            grad_b = pair.lora_b.grad
            if grad_a is None and grad_b is None:
                continue
            if grad_a is None or grad_b is None:
                raise RuntimeError(
                    f"LoRA-Pro pair {pair.name!r} has only one factor gradient."
                )
            if grad_a.is_sparse or grad_b.is_sparse:
                raise RuntimeError("LoRA-Pro does not support sparse gradients.")
            group = self._group_for(pair.lora_a, self.param_groups)
            beta1, beta2 = (float(value) for value in group["betas"])
            eps = float(group["eps"])
            lr = float(group["lr"])
            state = self.state[pair.lora_a]
            if not state:
                self._initialize_state(pair, state)
            state["step"].add_(1)
            step = int(state["step"].item())
            zero_b_first_step = step == 1 and int(
                torch.count_nonzero(pair.lora_b).item()
            ) == 0
            a_views = self._matrix_views(pair.lora_a)
            b_views = self._matrix_views(pair.lora_b)
            ga_views = self._matrix_views(grad_a)
            gb_views = self._matrix_views(grad_b)
            update_a = torch.empty_like(a_views, dtype=torch.float32)
            update_b = torch.empty_like(b_views, dtype=torch.float32)
            for index in range(pair.batch_size):
                corrected_a, corrected_b = lora_pro_correct_gradients(
                    lora_a=a_views[index],
                    lora_b=b_views[index],
                    grad_a=ga_views[index],
                    grad_b=gb_views[index],
                    scale=pair.scale,
                    damping=self.damping,
                    solve_gauge=False,
                    zero_b=zero_b_first_step,
                )
                equivalent = lora_pro_equivalent_gradient(
                    lora_a=a_views[index],
                    lora_b=b_views[index],
                    grad_a=corrected_a,
                    grad_b=corrected_b,
                    scale=pair.scale,
                )
                exp_avg = state["exp_avg"][index]
                exp_avg_sq = state["exp_avg_sq"][index]
                exp_avg.mul_(beta1).add_(equivalent, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    equivalent,
                    equivalent,
                    value=1.0 - beta2,
                )
                adam_gradient = (
                    exp_avg / (1.0 - beta1**step)
                ) / (
                    (exp_avg_sq / (1.0 - beta2**step)).sqrt() + eps
                )
                pseudo_a = (
                    float(pair.scale)
                    * b_views[index].float().transpose(-2, -1)
                    @ adam_gradient
                )
                pseudo_b = (
                    float(pair.scale)
                    * adam_gradient
                    @ a_views[index].float().transpose(-2, -1)
                )
                final_a, final_b = lora_pro_correct_gradients(
                    lora_a=a_views[index],
                    lora_b=b_views[index],
                    grad_a=pseudo_a,
                    grad_b=pseudo_b,
                    scale=pair.scale,
                    damping=self.damping,
                    solve_gauge=True,
                    zero_b=zero_b_first_step,
                )
                update_a[index].copy_(final_a)
                update_b[index].copy_(final_b)
            self._copy_update(
                pair.lora_a,
                update_a.squeeze(0) if pair.lora_a.ndim == 2 else update_a,
                lr=lr,
            )
            self._copy_update(
                pair.lora_b,
                update_b.squeeze(0) if pair.lora_b.ndim == 2 else update_b,
                lr=lr,
            )
        return loss

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["mirai_lora_pro"] = {
            "version": 1,
            "damping": self.damping,
            "stochastic_rounding": self.stochastic_rounding,
            "pairs": [pair.signature() for pair in self.pairs],
        }
        return payload

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        payload = dict(state_dict)
        metadata = payload.pop("mirai_lora_pro", None)
        if not isinstance(metadata, dict) or int(metadata.get("version", -1)) != 1:
            raise ValueError("LoRA-Pro optimizer state metadata is missing or invalid.")
        expected = [pair.signature() for pair in self.pairs]
        if metadata.get("pairs") != expected:
            raise ValueError("LoRA-Pro optimizer state factor topology does not match.")
        if float(metadata.get("damping", -1.0)) != self.damping:
            raise ValueError("LoRA-Pro optimizer damping does not match checkpoint.")
        if bool(metadata.get("stochastic_rounding", False)) != self.stochastic_rounding:
            raise ValueError(
                "LoRA-Pro stochastic-rounding policy does not match checkpoint."
            )
        saved_parameter_ids = [
            parameter_id
            for group in payload.get("param_groups", [])
            for parameter_id in group.get("params", [])
        ]
        current_parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
        ]
        if len(saved_parameter_ids) != len(current_parameters):
            raise ValueError(
                "LoRA-Pro optimizer state parameter count does not match."
            )
        saved_state = payload.get("state", {})
        exact_moments: dict[int, dict[str, torch.Tensor]] = {}
        for pair in self.pairs:
            position = next(
                (
                    index
                    for index, parameter in enumerate(current_parameters)
                    if parameter is pair.lora_a
                ),
                None,
            )
            if position is None:
                raise ValueError(
                    f"LoRA-Pro pair {pair.name!r} is absent during state load."
                )
            pair_state = saved_state.get(saved_parameter_ids[position], {})
            exact_moments[id(pair.lora_a)] = {
                key: value.detach().clone()
                for key in ("exp_avg", "exp_avg_sq")
                if isinstance((value := pair_state.get(key)), torch.Tensor)
            }
        super().load_state_dict(payload)
        for pair in self.pairs:
            state = self.state.get(pair.lora_a)
            if not state:
                continue
            expected_shape = (
                pair.batch_size,
                pair.out_features,
                pair.in_features,
            )
            for key in ("exp_avg", "exp_avg_sq"):
                value = exact_moments[id(pair.lora_a)].get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise ValueError(
                        f"LoRA-Pro optimizer state {key!r} for {pair.name!r} "
                        "has incompatible shape."
                    )
                state[key] = value.to(
                    device=pair.lora_a.device,
                    dtype=torch.float32,
                )

    @property
    def estimated_state_bytes(self) -> int:
        return estimate_lora_pro_state_bytes(self.pairs)


__all__ = [
    "LoRAProAdamW",
    "LoRAFactorPair",
    "estimate_lora_pro_state_bytes",
    "lora_pro_correct_gradients",
    "lora_pro_equivalent_gradient",
    "solve_positive_sylvester",
]
