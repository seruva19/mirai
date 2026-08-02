"""Muon-family optimizers for directly tuned grouped-expert matrices."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from mirai.core.training.optim.selected_expert_plan import (
    SelectedExpertPlanBinding,
)
from mirai.core.training.optim.stochastic_rounding import (
    stochastic_round_bfloat16,
)

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Selected-expert Muon requires torch: {exc}")


def orthogonalize_matrix_reference(matrix: Any) -> Any:
    """Return the exact polar factor ``U @ Vh`` for a matrix batch."""

    if matrix.ndim < 2:
        raise ValueError("Muon orthogonalization requires matrix-valued inputs.")
    u, _, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return u @ vh


def orthogonalize_matrix_newton_schulz(matrix: Any, *, steps: int = 5) -> Any:
    """Approximate the polar factor with the Muon quintic iteration.

    The coefficients and RMS convention follow the public Moonlight Muon
    implementation accompanying https://arxiv.org/abs/2502.16982.
    """

    if matrix.ndim < 2:
        raise ValueError("Muon orthogonalization requires matrix-valued inputs.")
    if steps < 1:
        raise ValueError("Muon Newton-Schulz steps must be >= 1.")
    a, b, c = 3.4445, -4.7750, 2.0315
    compute_dtype = (
        torch.bfloat16 if matrix.device.type == "cuda" else torch.float32
    )
    value = matrix.to(dtype=compute_dtype)
    transposed = value.shape[-2] > value.shape[-1]
    if transposed:
        value = value.mT
    value = value / (
        value.norm(dim=(-2, -1), keepdim=True) + 1e-7
    )
    for _ in range(steps):
        gram = value @ value.mT
        polynomial = b * gram + c * (gram @ gram)
        value = a * value + polynomial @ value
    if transposed:
        value = value.mT
    return value.float()


def muon_matrix_direction(
    momentum: Any,
    *,
    ns_steps: int,
    rms_target: float,
    reference: bool = False,
) -> Any:
    """Build the RMS-aligned Muon direction for independent matrices."""

    orthogonal = (
        orthogonalize_matrix_reference(momentum)
        if reference
        else orthogonalize_matrix_newton_schulz(momentum, steps=ns_steps)
    )
    rows, columns = int(momentum.shape[-2]), int(momentum.shape[-1])
    return orthogonal * (float(rms_target) * math.sqrt(max(rows, columns)))


def adamuon_matrix_direction(
    momentum: Any,
    second_moment: Any,
    *,
    beta: float,
    eps: float,
    ns_steps: int,
    rms_target: float,
    reference: bool = False,
) -> Any:
    """Apply AdaMuon Algorithm 1 to a batch of independent matrices.

    AdaMuon orthogonalizes ``sign(momentum)``, accumulates the element-wise
    second moment of that direction without bias correction, then normalizes
    every matrix to the paper's target RMS.
    """

    signed = torch.sign(momentum)
    orthogonal = (
        orthogonalize_matrix_reference(signed)
        if reference
        else orthogonalize_matrix_newton_schulz(signed, steps=ns_steps)
    )
    second_moment.mul_(float(beta)).addcmul_(
        orthogonal,
        orthogonal,
        value=1.0 - float(beta),
    )
    adapted = orthogonal / (second_moment.sqrt() + float(eps))
    matrix_elements = int(momentum.shape[-2]) * int(momentum.shape[-1])
    scale = (
        float(rms_target)
        * math.sqrt(matrix_elements)
        / (adapted.norm(dim=(-2, -1), keepdim=True) + float(eps))
    )
    return adapted * scale


class _SelectedExpertMuonBase(torch.optim.Optimizer):
    """Compact per-expert state for Muon-family matrix updates."""

    _algorithm_id = 0
    _adaptive = False

    def __init__(
        self,
        params: Iterable[Any],
        *,
        expert_ids: Iterable[int] = (),
        named_params: Iterable[tuple[str, Any]] = (),
        expert_ids_by_name: Mapping[str, Iterable[int]] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        rms_target: float = 0.2,
        stochastic_rounding: bool = False,
        reference_orthogonalization: bool = False,
    ) -> None:
        parameters = tuple(params)
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("Muon momentum must be in [0, 1).")
        if int(ns_steps) < 1:
            raise ValueError("Muon Newton-Schulz steps must be >= 1.")
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("Muon epsilon must be finite and > 0.")
        if not math.isfinite(float(rms_target)) or float(rms_target) <= 0.0:
            raise ValueError("Muon RMS target must be finite and > 0.")
        self._binding = SelectedExpertPlanBinding(
            parameters,
            expert_ids=expert_ids,
            named_params=named_params,
            expert_ids_by_name=expert_ids_by_name,
            parameter_ndim=3,
        )
        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "ns_steps": int(ns_steps),
            "eps": float(eps),
            "rms_target": float(rms_target),
        }
        super().__init__(parameters, defaults)
        self.expert_ids = self._binding.expert_ids
        self.expert_ids_by_name = self._binding.expert_ids_by_name
        self.stochastic_rounding = bool(stochastic_rounding)
        self.reference_orthogonalization = bool(reference_orthogonalization)

    @property
    def _state_tensor_keys(self) -> tuple[str, ...]:
        return (
            ("momentum_buffer", "second_moment")
            if self._adaptive
            else ("momentum_buffer",)
        )

    def _initialize_state(
        self,
        parameter: Any,
        gradient: Any,
        state: dict[str, Any],
    ) -> None:
        state["step"] = torch.tensor(0, dtype=torch.int64)
        state["algorithm_id"] = torch.tensor(
            self._algorithm_id,
            dtype=torch.int64,
        )
        state["momentum_buffer"] = torch.zeros_like(
            gradient,
            dtype=torch.float32,
        )
        if self._adaptive:
            state["second_moment"] = torch.zeros_like(
                gradient,
                dtype=torch.float32,
            )
        self._binding.initialize_state_metadata(parameter, state)

    def _validate_state(self, parameter: Any, state: Mapping[str, Any]) -> None:
        algorithm_id = state.get("algorithm_id")
        if (
            not torch.is_tensor(algorithm_id)
            or int(algorithm_id.item()) != self._algorithm_id
        ):
            raise ValueError(
                "Selected-expert Muon checkpoint algorithm mismatch."
            )
        self._binding.validate_state(
            parameter,
            state,
            tensor_keys=self._state_tensor_keys,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state only for the same algorithm and expert-row plan."""

        payload = dict(state_dict)
        metadata = payload.pop("mirai_selected_expert_muon", None)
        if not isinstance(metadata, dict) or int(metadata.get("version", -1)) != 1:
            raise ValueError(
                "Selected-expert Muon optimizer metadata is missing or invalid."
            )
        if int(metadata.get("algorithm_id", -1)) != self._algorithm_id:
            raise ValueError(
                "Selected-expert Muon checkpoint algorithm mismatch."
            )
        if (
            bool(metadata.get("stochastic_rounding", False))
            != self.stochastic_rounding
            or bool(metadata.get("reference_orthogonalization", False))
            != self.reference_orthogonalization
        ):
            raise ValueError(
                "Selected-expert Muon checkpoint execution policy mismatch."
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
                "Selected-expert Muon checkpoint parameter count mismatch."
            )
        saved_state = payload.get("state", {})
        exact_state: dict[int, dict[str, Any]] = {}
        for position, parameter in enumerate(current_parameters):
            parameter_state = saved_state.get(saved_parameter_ids[position], {})
            exact_state[id(parameter)] = {
                key: value.detach().clone()
                for key in self._state_tensor_keys
                if torch.is_tensor((value := parameter_state.get(key)))
            }

        super().load_state_dict(payload)
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if state:
                    for key in self._state_tensor_keys:
                        value = exact_state[id(parameter)].get(key)
                        if not torch.is_tensor(value):
                            raise ValueError(
                                "Selected-expert Muon checkpoint is missing "
                                f"{key}."
                            )
                        state[key] = value.to(
                            device=parameter.device,
                            dtype=torch.float32,
                        )
                    self._validate_state(parameter, state)

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["mirai_selected_expert_muon"] = {
            "version": 1,
            "algorithm_id": self._algorithm_id,
            "stochastic_rounding": self.stochastic_rounding,
            "reference_orthogonalization": self.reference_orthogonalization,
        }
        return payload

    def _direction(
        self,
        momentum: Any,
        state: dict[str, Any],
        group: Mapping[str, Any],
    ) -> Any:
        if self._adaptive:
            return adamuon_matrix_direction(
                momentum,
                state["second_moment"],
                beta=float(group["momentum"]),
                eps=float(group["eps"]),
                ns_steps=int(group["ns_steps"]),
                rms_target=float(group["rms_target"]),
                reference=self.reference_orthogonalization,
            )
        return muon_matrix_direction(
            momentum,
            ns_steps=int(group["ns_steps"]),
            rms_target=float(group["rms_target"]),
            reference=self.reference_orthogonalization,
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        indices: dict[tuple[Any, tuple[int, ...]], Any] = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "Selected-expert Muon does not support sparse gradients."
                    )
                index = self._binding.index_for(parameter, indices)
                gradient = parameter.grad.index_select(0, index).float()
                state = self.state[parameter]
                if not state:
                    self._initialize_state(parameter, gradient, state)
                self._validate_state(parameter, state)
                state["step"].add_(1)
                momentum = state["momentum_buffer"]
                momentum.mul_(float(group["momentum"])).add_(gradient)
                update_input = (
                    gradient.add(
                        momentum,
                        alpha=float(group["momentum"]),
                    )
                    if bool(group["nesterov"])
                    else momentum
                )
                direction = self._direction(update_input, state, group)
                current = parameter.index_select(0, index).float()
                if float(group["weight_decay"]):
                    current.mul_(
                        1.0
                        - float(group["lr"])
                        * float(group["weight_decay"])
                    )
                current.add_(direction, alpha=-float(group["lr"]))
                if (
                    self.stochastic_rounding
                    and parameter.dtype == torch.bfloat16
                ):
                    current = stochastic_round_bfloat16(current)
                else:
                    current = current.to(dtype=parameter.dtype)
                parameter.index_copy_(0, index, current)
        return loss


class SelectedExpertMuon(_SelectedExpertMuonBase):
    """Moonlight-style Muon over selected grouped-expert matrices."""

    _algorithm_id = 1


class SelectedExpertAdaMuon(_SelectedExpertMuonBase):
    """AdaMuon Algorithm 1 over selected grouped-expert matrices."""

    _algorithm_id = 2
    _adaptive = True


def estimate_selected_expert_muon_state_bytes(
    optimizer: _SelectedExpertMuonBase,
) -> int:
    """Return persistent tensor-state bytes owned by the optimizer."""

    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += int(value.numel()) * int(value.element_size())
    return total
