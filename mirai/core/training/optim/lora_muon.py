"""LoRA-Muon spectral descent on standard LoRA factor pairs.

The optimizer implements Algorithm 1 and the numerical realization in
Appendix B of LoRA-Muon (arXiv:2606.12921). Mirai's factors represent
``scale * lora_b @ lora_a``; the factor directions therefore include the
corresponding inverse-scale coordinate transform.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"LoRA-Muon requires torch: {exc}")

from mirai.core.training.optim.lora_pairs import LoRAFactorPair
from mirai.core.training.optim.stochastic_rounding import (
    stochastic_round_bfloat16,
)


_POLAR_EXPRESS_COEFFICIENTS = (
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
    (1.8648, -1.2224, 0.3577),
)

_INVERSE_ROOT_COEFFICIENTS = (
    (7.424865680309214, -18.39581635618996, 12.896720413604342),
    (3.4877256051546017, -2.3300436563986993, 0.4404692168431095),
    (2.7766085124882527, -2.070643152532662, 0.46302261050004967),
    (1.9913142104341506, -1.373936700681269, 0.3875934979568538),
    (1.8754637749479246, -1.2505152090010534, 0.37505152463617264),
    (1.874999066623701, -1.2499981332141676, 0.37499906659046633),
    (1.875, -1.25, 0.375),
)

_INVERSE_ROOT_EPSILON = 1e-5
_INVERSE_ROOT_GAMMA = 1.001


def matrix_sign_reference(matrix: torch.Tensor) -> torch.Tensor:
    """Return the rectangular matrix sign ``U @ Vh`` using an SVD oracle."""

    value = matrix.float()
    if int(torch.count_nonzero(value).item()) == 0:
        return torch.zeros_like(value)
    u, _, vh = torch.linalg.svd(value, full_matrices=False)
    return u @ vh


def matrix_sign_newton_schulz(matrix: torch.Tensor) -> torch.Tensor:
    """Appendix B.3 Polar-Express matrix-sign realization."""

    value = matrix.float()
    transposed = int(value.shape[-2]) > int(value.shape[-1])
    if transposed:
        value = value.mT
    norm = torch.linalg.matrix_norm(
        value,
        ord="fro",
        dim=(-2, -1),
        keepdim=True,
    )
    value = value / (norm + 1e-20)
    for a, b, c in _POLAR_EXPRESS_COEFFICIENTS:
        gram = value @ value.mT
        value = float(a) * value + (
            float(b) * gram + float(c) * (gram @ gram)
        ) @ value
    return value.mT if transposed else value


def psd_inverse_sqrt_reference(psd: torch.Tensor) -> torch.Tensor:
    """Appendix B.4 inverse-root oracle evaluated with eigendecomposition."""

    value = psd.float()
    norm = torch.linalg.matrix_norm(
        value,
        ord="fro",
        dim=(-2, -1),
        keepdim=True,
    )
    zero = norm <= torch.finfo(value.dtype).tiny
    safe_norm = torch.where(zero, torch.ones_like(norm), norm)
    identity = torch.eye(
        int(value.shape[-1]),
        dtype=value.dtype,
        device=value.device,
    )
    normalized = value / safe_norm + _INVERSE_ROOT_EPSILON * identity
    eigenvalues, eigenvectors = torch.linalg.eigh(normalized)
    inverse_root = eigenvectors @ torch.diag_embed(
        eigenvalues.clamp_min(torch.finfo(value.dtype).tiny).rsqrt()
    ) @ eigenvectors.mT
    result = safe_norm.rsqrt() * inverse_root
    return torch.where(zero, torch.zeros_like(result), result)


def psd_inverse_sqrt_newton_schulz(psd: torch.Tensor) -> torch.Tensor:
    """Appendix B.4 seven-step polynomial inverse square root."""

    value = psd.float()
    norm = torch.linalg.matrix_norm(
        value,
        ord="fro",
        dim=(-2, -1),
        keepdim=True,
    )
    zero = norm <= torch.finfo(value.dtype).tiny
    safe_norm = torch.where(zero, torch.ones_like(norm), norm)
    identity = torch.eye(
        int(value.shape[-1]),
        dtype=value.dtype,
        device=value.device,
    )
    current_psd = value / safe_norm + _INVERSE_ROOT_EPSILON * identity
    inverse_root = identity.expand_as(current_psd).clone()
    gamma = _INVERSE_ROOT_GAMMA
    for a, b, c in _INVERSE_ROOT_COEFFICIENTS:
        squared = current_psd @ current_psd
        polynomial = (
            (float(a) / gamma) * identity
            + (float(b) / gamma**3) * current_psd
            + (float(c) / gamma**5) * squared
        )
        inverse_root = inverse_root @ polynomial
        polynomial_sq = polynomial @ polynomial
        current_psd = current_psd @ polynomial_sq
        current_psd = 0.5 * (current_psd + current_psd.mT)
    result = safe_norm.rsqrt() * inverse_root
    return torch.where(zero, torch.zeros_like(result), result)


def lora_muon_factor_directions(
    *,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    moment_a: torch.Tensor,
    moment_b: torch.Tensor,
    scale: float,
    numerical: Literal["newton_schulz", "reference"] = "newton_schulz",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the negative unit-learning-rate factor directions.

    Mirai stores ``A`` as ``[rank, in]`` and ``B`` as ``[out, rank]``.
    The paper uses two column-major factors, so its second factor and moment
    are represented here by the transpose of Mirai's ``A`` tensors.
    """

    if float(scale) <= 0.0 or not math.isfinite(float(scale)):
        raise ValueError("LoRA-Muon requires a finite positive LoRA scale.")
    inverse_root = (
        psd_inverse_sqrt_reference
        if numerical == "reference"
        else psd_inverse_sqrt_newton_schulz
    )
    matrix_sign = (
        matrix_sign_reference
        if numerical == "reference"
        else matrix_sign_newton_schulz
    )

    a = lora_a.float()
    b = lora_b.float()
    ma = moment_a.float()
    mb = moment_b.float()
    # Paper S_A belongs to Mirai B; paper S_B belongs to Mirai A.T.
    root_for_a = inverse_root(b.mT @ b)
    root_for_b = inverse_root(a @ a.mT)
    inverse_scale = 1.0 / float(scale)

    direction_b = (
        -0.5
        * inverse_scale
        * (matrix_sign(mb @ root_for_b) @ root_for_b)
    )
    paper_direction_a = (
        -0.5
        * inverse_scale
        * (matrix_sign(ma.mT @ root_for_a) @ root_for_a)
    )
    return paper_direction_a.mT, direction_b


@torch.no_grad()
def rebalance_lora_muon_gauge(
    *,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    moment_a: torch.Tensor,
    moment_b: torch.Tensor,
    alpha: float,
) -> float:
    """Apply Appendix B.1 scalar gauge rebalancing in place."""

    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("LoRA-Muon gauge-rebalance alpha must be in (0, 1].")
    norm_a = torch.linalg.matrix_norm(lora_a.float(), ord=2)
    norm_b = torch.linalg.matrix_norm(lora_b.float(), ord=2)
    if float(norm_a.item()) == 0.0 or float(norm_b.item()) == 0.0:
        return 1.0
    # Paper A is Mirai B and paper B is Mirai A.T.
    factor = float((norm_a / norm_b).pow(float(alpha) / 2.0).item())
    lora_a.div_(factor)
    lora_b.mul_(factor)
    moment_a.mul_(factor)
    moment_b.div_(factor)
    return factor


def estimate_lora_muon_state_bytes(
    pairs: Iterable[LoRAFactorPair],
    *,
    moment_dtype: torch.dtype = torch.float32,
) -> int:
    """Return exact persistent first-moment storage."""

    element_size = torch.empty((), dtype=moment_dtype).element_size()
    return sum(
        (pair.lora_a.numel() + pair.lora_b.numel()) * element_size
        for pair in pairs
    )


class LoRAMuon(torch.optim.Optimizer):
    """Algorithm 1 LoRA-Muon with optional scalar gauge rebalancing."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        pairs: Iterable[LoRAFactorPair],
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        gauge_rebalance_interval: int = 0,
        gauge_rebalance_alpha: float = 1.0,
        stochastic_rounding: bool = False,
    ) -> None:
        if float(lr) < 0.0 or not math.isfinite(float(lr)):
            raise ValueError("LoRA-Muon lr must be finite and >= 0.")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("LoRA-Muon momentum must be in [0, 1).")
        if float(weight_decay) < 0.0 or not math.isfinite(float(weight_decay)):
            raise ValueError("LoRA-Muon weight decay must be finite and >= 0.")
        if int(gauge_rebalance_interval) < 0:
            raise ValueError("LoRA-Muon gauge-rebalance interval must be >= 0.")
        if not 0.0 < float(gauge_rebalance_alpha) <= 1.0:
            raise ValueError("LoRA-Muon gauge-rebalance alpha must be in (0, 1].")
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
        }
        super().__init__(params, defaults)
        self.pairs = tuple(pairs)
        self.gauge_rebalance_interval = int(gauge_rebalance_interval)
        self.gauge_rebalance_alpha = float(gauge_rebalance_alpha)
        self.stochastic_rounding = bool(stochastic_rounding)
        self._validate_pairs()

    def _validate_pairs(self) -> None:
        grouped = {
            id(parameter): group
            for group in self.param_groups
            for parameter in group["params"]
        }
        expected: set[int] = set()
        for pair in self.pairs:
            expected.update({id(pair.lora_a), id(pair.lora_b)})
            if id(pair.lora_a) not in grouped or id(pair.lora_b) not in grouped:
                raise ValueError(
                    f"LoRA-Muon pair {pair.name!r} is absent from optimizer params."
                )
            group_a = grouped[id(pair.lora_a)]
            group_b = grouped[id(pair.lora_b)]
            for key in ("lr", "momentum", "weight_decay"):
                if group_a[key] != group_b[key]:
                    raise ValueError(
                        f"LoRA-Muon pair {pair.name!r} requires identical A/B {key}."
                    )
        if set(grouped) != expected:
            raise ValueError(
                "LoRA-Muon accepts only complete standard LoRA A/B pairs."
            )
        if not self.pairs:
            raise ValueError("LoRA-Muon found no LoRA factor pairs.")

    @staticmethod
    def _matrix_views(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0) if tensor.ndim == 2 else tensor

    @staticmethod
    def _group_for(
        parameter: nn.Parameter,
        groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        for group in groups:
            if any(candidate is parameter for candidate in group["params"]):
                return group
        raise RuntimeError("LoRA-Muon parameter group disappeared.")

    @staticmethod
    def _initialize_state(pair: LoRAFactorPair, state: dict[str, Any]) -> None:
        state["step"] = torch.tensor(
            0,
            dtype=torch.int64,
            device=pair.lora_a.device,
        )
        state["moment_a"] = torch.zeros_like(
            pair.lora_a,
            dtype=torch.float32,
            device=pair.lora_a.device,
        )
        state["moment_b"] = torch.zeros_like(
            pair.lora_b,
            dtype=torch.float32,
            device=pair.lora_b.device,
        )

    def _copy_value(
        self,
        parameter: nn.Parameter,
        value: torch.Tensor,
    ) -> None:
        if self.stochastic_rounding and parameter.dtype == torch.bfloat16:
            parameter.copy_(stochastic_round_bfloat16(value.float()))
        else:
            parameter.copy_(
                value.to(device=parameter.device, dtype=parameter.dtype)
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
                    f"LoRA-Muon pair {pair.name!r} has only one factor gradient."
                )
            if grad_a.is_sparse or grad_b.is_sparse:
                raise RuntimeError("LoRA-Muon does not support sparse gradients.")
            group = self._group_for(pair.lora_a, self.param_groups)
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            if lr * weight_decay >= 1.0:
                raise RuntimeError(
                    "LoRA-Muon split weight decay requires lr * weight_decay < 1."
                )
            state = self.state[pair.lora_a]
            if not state:
                self._initialize_state(pair, state)
            state["step"].add_(1)
            step = int(state["step"].item())
            state["moment_a"].mul_(momentum).add_(
                grad_a.float(),
                alpha=1.0 - momentum,
            )
            state["moment_b"].mul_(momentum).add_(
                grad_b.float(),
                alpha=1.0 - momentum,
            )

            a_views = self._matrix_views(pair.lora_a)
            b_views = self._matrix_views(pair.lora_b)
            moment_a_views = self._matrix_views(state["moment_a"])
            moment_b_views = self._matrix_views(state["moment_b"])
            if (
                self.gauge_rebalance_interval > 0
                and step % self.gauge_rebalance_interval == 0
            ):
                for index in range(pair.batch_size):
                    rebalance_lora_muon_gauge(
                        lora_a=a_views[index],
                        lora_b=b_views[index],
                        moment_a=moment_a_views[index],
                        moment_b=moment_b_views[index],
                        alpha=self.gauge_rebalance_alpha,
                    )

            next_a = torch.empty_like(a_views, dtype=torch.float32)
            next_b = torch.empty_like(b_views, dtype=torch.float32)
            decay_scale = math.sqrt(1.0 - lr * weight_decay)
            inverse_decay_scale = 1.0 / decay_scale
            for index in range(pair.batch_size):
                direction_a, direction_b = lora_muon_factor_directions(
                    lora_a=a_views[index],
                    lora_b=b_views[index],
                    moment_a=moment_a_views[index],
                    moment_b=moment_b_views[index],
                    scale=pair.scale,
                )
                next_a[index].copy_(
                    decay_scale * a_views[index].float()
                    + lr * inverse_decay_scale * direction_a
                )
                next_b[index].copy_(
                    decay_scale * b_views[index].float()
                    + lr * inverse_decay_scale * direction_b
                )
            self._copy_value(
                pair.lora_a,
                next_a.squeeze(0) if pair.lora_a.ndim == 2 else next_a,
            )
            self._copy_value(
                pair.lora_b,
                next_b.squeeze(0) if pair.lora_b.ndim == 2 else next_b,
            )
        return loss

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["mirai_lora_muon"] = {
            "version": 1,
            "gauge_rebalance_interval": self.gauge_rebalance_interval,
            "gauge_rebalance_alpha": self.gauge_rebalance_alpha,
            "stochastic_rounding": self.stochastic_rounding,
            "pairs": [pair.signature() for pair in self.pairs],
        }
        return payload

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        payload = dict(state_dict)
        metadata = payload.pop("mirai_lora_muon", None)
        if not isinstance(metadata, dict) or int(metadata.get("version", -1)) != 1:
            raise ValueError(
                "LoRA-Muon optimizer state metadata is missing or invalid."
            )
        if metadata.get("pairs") != [pair.signature() for pair in self.pairs]:
            raise ValueError("LoRA-Muon optimizer factor topology does not match.")
        if (
            int(metadata.get("gauge_rebalance_interval", -1))
            != self.gauge_rebalance_interval
            or float(metadata.get("gauge_rebalance_alpha", -1.0))
            != self.gauge_rebalance_alpha
        ):
            raise ValueError(
                "LoRA-Muon gauge-rebalance policy does not match checkpoint."
            )
        if (
            bool(metadata.get("stochastic_rounding", False))
            != self.stochastic_rounding
        ):
            raise ValueError(
                "LoRA-Muon stochastic-rounding policy does not match checkpoint."
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
                "LoRA-Muon optimizer state parameter count does not match."
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
                    f"LoRA-Muon pair {pair.name!r} is absent during state load."
                )
            pair_state = saved_state.get(saved_parameter_ids[position], {})
            exact_moments[id(pair.lora_a)] = {
                key: value.detach().clone()
                for key in ("moment_a", "moment_b")
                if isinstance((value := pair_state.get(key)), torch.Tensor)
            }
        super().load_state_dict(payload)
        for pair in self.pairs:
            state = self.state.get(pair.lora_a)
            if not state:
                continue
            for key, shape in (
                ("moment_a", tuple(pair.lora_a.shape)),
                ("moment_b", tuple(pair.lora_b.shape)),
            ):
                value = exact_moments[id(pair.lora_a)].get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                    raise ValueError(
                        f"LoRA-Muon state {key!r} for {pair.name!r} "
                        "has incompatible shape."
                    )
                state[key] = value.to(
                    device=pair.lora_a.device,
                    dtype=torch.float32,
                )

    @property
    def estimated_state_bytes(self) -> int:
        return estimate_lora_muon_state_bytes(self.pairs)


__all__ = [
    "LoRAMuon",
    "estimate_lora_muon_state_bytes",
    "lora_muon_factor_directions",
    "matrix_sign_newton_schulz",
    "matrix_sign_reference",
    "psd_inverse_sqrt_newton_schulz",
    "psd_inverse_sqrt_reference",
    "rebalance_lora_muon_gauge",
]
