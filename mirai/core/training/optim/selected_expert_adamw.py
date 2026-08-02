"""Row-sparse AdamW for direct tuning of selected grouped-expert tensors."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from mirai.core.training.optim.low_bit_state import (
    SOLO_4_2_BETAS,
    SOLO_4_2_STATE_FORMAT,
    decode_solo_4_2_moments,
    encode_solo_4_2_moments,
    PackedMomentState,
)
from mirai.core.training.optim.selected_expert_plan import (
    SelectedExpertPlanBinding,
)
from mirai.core.training.optim.stochastic_rounding import (
    stochastic_round_bfloat16,
)

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"SelectedExpertAdamW requires torch: {exc}")


class SelectedExpertAdamW(torch.optim.Optimizer):
    """Update selected expert rows while storing moments only for those rows.

    Parameters must have expert id on axis zero.  The checkpoint stores selected
    ids and compact moments; loading under a different selection fails rather
    than expanding or remapping state implicitly.
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
        state_format: str = "native",
    ) -> None:
        parameters = tuple(params)
        self._binding = SelectedExpertPlanBinding(
            parameters,
            expert_ids=expert_ids,
            named_params=named_params,
            expert_ids_by_name=expert_ids_by_name,
        )
        normalized_state_format = str(state_format).strip().lower()
        if normalized_state_format not in {"native", SOLO_4_2_STATE_FORMAT}:
            raise ValueError(
                "SelectedExpertAdamW state_format must be 'native' or 'solo_4_2'."
            )
        resolved_betas = (float(betas[0]), float(betas[1]))
        if (
            normalized_state_format == SOLO_4_2_STATE_FORMAT
            and resolved_betas != SOLO_4_2_BETAS
        ):
            raise ValueError(
                "SOLO 4/2-bit selected-expert AdamW requires betas=(0.8, 0.999)."
            )
        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "betas": resolved_betas,
            "eps": float(eps),
        }
        super().__init__(parameters, defaults)
        self.expert_ids = self._binding.expert_ids
        self.expert_ids_by_name = self._binding.expert_ids_by_name
        self.stochastic_rounding = bool(stochastic_rounding)
        self.state_format = normalized_state_format

    @property
    def _low_bit_state(self) -> bool:
        return self.state_format == SOLO_4_2_STATE_FORMAT

    @staticmethod
    def _clone_packed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in payload.items()
        }

    def _validate_low_bit_state(
        self,
        parameter: Any,
        state: Mapping[str, Any],
    ) -> None:
        expected_shape = (
            len(self._binding.ids_for(parameter)),
            *tuple(int(value) for value in parameter.shape[1:]),
        )
        for key, encoding in (
            ("exp_avg", "signed_de_4bit"),
            ("exp_avg_sq", "unsigned_qema_2bit"),
        ):
            payload = PackedMomentState.from_state_dict(
                state.get(key, {}),
                device=parameter.device,
            )
            if payload.encoding != encoding or payload.shape != expected_shape:
                raise ValueError(
                    "SOLO 4/2-bit optimizer checkpoint moment topology mismatch."
                )

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload["mirai_selected_expert_adamw"] = {
            "version": 1,
            "state_format": self.state_format,
            "stochastic_rounding": self.stochastic_rounding,
        }
        return payload

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load compact moments only when their row ownership is identical."""

        payload = dict(state_dict)
        metadata = payload.pop("mirai_selected_expert_adamw", None)
        if not isinstance(metadata, dict) or int(metadata.get("version", -1)) != 1:
            raise ValueError(
                "Selected-expert AdamW optimizer metadata is missing or invalid."
            )
        if (
            str(metadata.get("state_format", "")).strip().lower()
            != self.state_format
            or bool(metadata.get("stochastic_rounding", False))
            != self.stochastic_rounding
        ):
            raise ValueError(
                "Selected-expert AdamW checkpoint execution policy mismatch."
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
                "Selected-expert AdamW checkpoint parameter count mismatch."
            )
        exact_packed: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        if self._low_bit_state:
            saved_state = payload.get("state", {})
            for position, parameter in enumerate(current_parameters):
                parameter_state = saved_state.get(saved_parameter_ids[position], {})
                first = parameter_state.get("exp_avg")
                second = parameter_state.get("exp_avg_sq")
                if not isinstance(first, Mapping) or not isinstance(second, Mapping):
                    raise ValueError(
                        "SOLO 4/2-bit optimizer checkpoint is missing packed moments."
                    )
                exact_packed[id(parameter)] = (
                    self._clone_packed_payload(first),
                    self._clone_packed_payload(second),
                )
        super().load_state_dict(payload)
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state.get(param)
                if not state:
                    continue
                tensor_keys: tuple[str, ...] = ("exp_avg", "exp_avg_sq")
                if self._low_bit_state:
                    first, second = exact_packed[id(param)]
                    state["exp_avg"] = PackedMomentState.from_state_dict(
                        first,
                        device=param.device,
                    ).state_dict()
                    state["exp_avg_sq"] = PackedMomentState.from_state_dict(
                        second,
                        device=param.device,
                    ).state_dict()
                    self._validate_low_bit_state(param, state)
                    tensor_keys = ()
                self._binding.validate_state(
                    param,
                    state,
                    tensor_keys=tensor_keys,
                )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        indices: dict[tuple[Any, tuple[int, ...]], Any] = {}
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                index = self._binding.index_for(param, indices)
                grad = param.grad.index_select(0, index)
                state = self.state[param]
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int64)
                    first = torch.zeros_like(grad)
                    second = torch.zeros_like(grad)
                    if self._low_bit_state:
                        state["exp_avg"], state["exp_avg_sq"] = (
                            encode_solo_4_2_moments(first, second)
                        )
                    else:
                        state["exp_avg"] = first
                        state["exp_avg_sq"] = second
                    self._binding.initialize_state_metadata(param, state)
                tensor_keys: tuple[str, ...] = ("exp_avg", "exp_avg_sq")
                if self._low_bit_state:
                    self._validate_low_bit_state(param, state)
                    tensor_keys = ()
                self._binding.validate_state(
                    param,
                    state,
                    tensor_keys=tensor_keys,
                )
                state["step"].add_(1)
                if self._low_bit_state:
                    exp_avg, exp_avg_sq = decode_solo_4_2_moments(
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        device=param.device,
                    )
                    grad_for_state = grad.float()
                    exp_avg = exp_avg.lerp(grad_for_state, 1.0 - beta1)
                    exp_avg_sq = exp_avg_sq.lerp(
                        grad_for_state.square(),
                        1.0 - beta2,
                    )
                    state["exp_avg"], state["exp_avg_sq"] = (
                        encode_solo_4_2_moments(exp_avg, exp_avg_sq)
                    )
                else:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad,
                        grad,
                        value=1.0 - beta2,
                    )
                step = int(state["step"].item())
                denom = exp_avg_sq.sqrt().div_((1.0 - beta2**step) ** 0.5).add_(group["eps"])
                current = param.index_select(0, index)
                stochastic_bf16 = (
                    self.stochastic_rounding
                    and param.dtype == torch.bfloat16
                )
                fp32_parameter_update = self._low_bit_state or stochastic_bf16
                if fp32_parameter_update:
                    current = current.float()
                if group["weight_decay"]:
                    current.mul_(1.0 - group["lr"] * group["weight_decay"])
                update_moment = exp_avg.float() if fp32_parameter_update else exp_avg
                update_denom = denom.float() if fp32_parameter_update else denom
                current.addcdiv_(
                    update_moment,
                    update_denom,
                    value=-(group["lr"] / (1.0 - beta1**step)),
                )
                if stochastic_bf16:
                    current = stochastic_round_bfloat16(current)
                elif current.dtype != param.dtype:
                    current = current.to(param.dtype)
                param.index_copy_(0, index, current)
        return loss
