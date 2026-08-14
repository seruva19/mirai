"""Adam-mini for directly tuned grouped-expert projection rows.

The MLP partition follows Adam-mini v1.1.1
(https://arxiv.org/abs/2406.16793 and
https://github.com/zyushun/Adam-mini): each selected expert/output-neuron block
owns one scalar second moment, while the first moment retains the projection's
full shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mirai.core.training.optim.selected_expert_plan import SelectedExpertPlanBinding
from mirai.core.training.optim.stochastic_rounding import (
    stochastic_ema_bfloat16_,
    stochastic_round_bfloat16,
)

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"SelectedExpertAdamMini requires torch: {exc}")


class SelectedExpertAdamMini(torch.optim.Optimizer):
    """Apply Adam-mini's neuron partition to selected expert matrices only.

    Every parameter must have shape ``[experts, out_features, in_features]``.
    Persistent state contains a full first moment for selected rows and one
    second-moment scalar per ``[expert, out_feature]`` block. Checkpoints bind
    both the expert selection and the optimizer execution policy exactly.
    """

    def __init__(
        self,
        params: Iterable[Any],
        *,
        expert_ids: Iterable[int] = (),
        named_params: Iterable[tuple[str, Any]] = (),
        expert_ids_by_name: Mapping[str, Iterable[int]] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        stochastic_rounding: bool = False,
    ) -> None:
        parameters = tuple(params)
        self._binding = SelectedExpertPlanBinding(
            parameters,
            expert_ids=expert_ids,
            named_params=named_params,
            expert_ids_by_name=expert_ids_by_name,
            parameter_ndim=3,
        )
        beta1, beta2 = float(betas[0]), float(betas[1])
        if float(lr) < 0.0:
            raise ValueError("SelectedExpertAdamMini lr must be non-negative.")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("SelectedExpertAdamMini betas must lie in [0, 1).")
        if float(weight_decay) < 0.0:
            raise ValueError(
                "SelectedExpertAdamMini weight_decay must be non-negative."
            )
        if float(eps) < 0.0:
            raise ValueError("SelectedExpertAdamMini eps must be non-negative.")
        super().__init__(
            parameters,
            {
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "betas": (beta1, beta2),
                "eps": float(eps),
            },
        )
        self.expert_ids = self._binding.expert_ids
        self.expert_ids_by_name = self._binding.expert_ids_by_name
        self.stochastic_rounding = bool(stochastic_rounding)

    def _validate_state(self, parameter: Any, state: Mapping[str, Any]) -> None:
        self._binding.validate_state(
            parameter,
            state,
            tensor_keys=("exp_avg",),
        )
        selected = len(self._binding.ids_for(parameter))
        expected_second_shape = (selected, int(parameter.shape[1]), 1)
        second = state.get("exp_avg_sq_mean")
        if not torch.is_tensor(second) or tuple(second.shape) != expected_second_shape:
            raise ValueError(
                "Selected-expert Adam-mini checkpoint has invalid per-neuron "
                "second-moment topology."
            )

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["mirai_selected_expert_adam_mini"] = {
            "version": 1,
            "partition": "expert_output_neuron",
            "stochastic_rounding": self.stochastic_rounding,
        }
        return payload

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        payload = dict(state_dict)
        metadata = payload.pop("mirai_selected_expert_adam_mini", None)
        if not isinstance(metadata, dict) or int(metadata.get("version", -1)) != 1:
            raise ValueError(
                "Selected-expert Adam-mini optimizer metadata is missing or invalid."
            )
        if (
            str(metadata.get("partition", "")).strip().lower()
            != "expert_output_neuron"
            or bool(metadata.get("stochastic_rounding", False))
            != self.stochastic_rounding
        ):
            raise ValueError(
                "Selected-expert Adam-mini checkpoint execution policy mismatch."
            )
        saved_parameter_ids = [
            parameter_id
            for group in payload.get("param_groups", ())
            for parameter_id in group.get("params", ())
        ]
        current_parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
        ]
        if len(saved_parameter_ids) != len(current_parameters):
            raise ValueError(
                "Selected-expert Adam-mini checkpoint parameter count mismatch."
            )
        super().load_state_dict(payload)
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if state:
                    self._validate_state(parameter, state)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        indices: dict[tuple[Any, tuple[int, ...]], Any] = {}
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "SelectedExpertAdamMini does not support sparse gradients."
                    )
                index = self._binding.index_for(parameter, indices)
                gradient = parameter.grad.index_select(0, index)
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int64)
                    state["exp_avg"] = torch.zeros_like(
                        gradient,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq_mean"] = torch.zeros_like(
                        gradient.mean(dim=-1, keepdim=True),
                        memory_format=torch.preserve_format,
                    )
                    self._binding.initialize_state_metadata(parameter, state)
                self._validate_state(parameter, state)

                state["step"].add_(1)
                exp_avg = state["exp_avg"]
                exp_avg_sq_mean = state["exp_avg_sq_mean"]
                stochastic_bf16 = (
                    self.stochastic_rounding
                    and parameter.dtype == torch.bfloat16
                )
                if stochastic_bf16:
                    stochastic_ema_bfloat16_(
                        exp_avg,
                        gradient,
                        beta=float(beta1),
                    )
                    stochastic_ema_bfloat16_(
                        exp_avg_sq_mean,
                        gradient.float().square().mean(dim=-1, keepdim=True),
                        beta=float(beta2),
                    )
                else:
                    exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                    exp_avg_sq_mean.mul_(beta2).add_(
                        gradient.square().mean(dim=-1, keepdim=True),
                        alpha=1.0 - beta2,
                    )

                step = int(state["step"].item())
                denominator = exp_avg_sq_mean.sqrt().div_(
                    (1.0 - beta2**step) ** 0.5
                ).add_(group["eps"])
                current = parameter.index_select(0, index)
                if stochastic_bf16:
                    current = current.float()
                if group["weight_decay"]:
                    current.mul_(1.0 - group["lr"] * group["weight_decay"])
                current.addcdiv_(
                    exp_avg.float() if stochastic_bf16 else exp_avg,
                    denominator.float() if stochastic_bf16 else denominator,
                    value=-(group["lr"] / (1.0 - beta1**step)),
                )
                if stochastic_bf16:
                    current = stochastic_round_bfloat16(current)
                parameter.index_copy_(0, index, current)
        return loss


def estimate_selected_expert_adam_mini_state_bytes(
    optimizer: SelectedExpertAdamMini,
) -> int:
    """Return tensor bytes owned by initialized Adam-mini parameter states."""
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += int(value.numel()) * int(value.element_size())
    return total
