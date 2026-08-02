"""Plan binding shared by optimizers that update selected expert rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Selected-expert optimization requires torch: {exc}")


class SelectedExpertPlanBinding:
    """Bind optimizer parameters to an exact set of expert-axis rows."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        expert_ids: Iterable[int] = (),
        named_params: Iterable[tuple[str, Any]] = (),
        expert_ids_by_name: Mapping[str, Iterable[int]] | None = None,
        parameter_ndim: int | None = None,
    ) -> None:
        parameters = tuple(params)
        ids = self._normalize_ids(expert_ids)
        plan = {
            str(name): self._normalize_ids(values)
            for name, values in dict(expert_ids_by_name or {}).items()
        }
        named = tuple((str(name), parameter) for name, parameter in named_params)
        if plan:
            if ids:
                raise ValueError(
                    "Selected-expert optimization accepts either global expert_ids "
                    "or a per-parameter plan, not both."
                )
            by_name = {name: parameter for name, parameter in named}
            if len(by_name) != len(named):
                raise ValueError(
                    "Selected-expert optimizer parameter names must be unique."
                )
            if set(by_name) != set(plan):
                raise ValueError(
                    "Selected-expert per-parameter plan must cover every named "
                    "trainable parameter exactly."
                )
            if {id(parameter) for parameter in parameters} != {
                id(parameter) for parameter in by_name.values()
            }:
                raise ValueError(
                    "Selected-expert named parameters do not match params."
                )
            ids_by_parameter = {
                id(by_name[name]): selected_ids
                for name, selected_ids in plan.items()
            }
        else:
            if not ids:
                raise ValueError(
                    "Selected-expert optimization requires non-negative expert ids."
                )
            ids_by_parameter = {id(parameter): ids for parameter in parameters}

        for parameter in parameters:
            selected_ids = ids_by_parameter.get(id(parameter), ())
            if (
                not selected_ids
                or selected_ids[0] < 0
                or parameter.ndim < 1
                or selected_ids[-1] >= int(parameter.shape[0])
            ):
                raise ValueError(
                    "Every selected-expert parameter must expose the full expert "
                    "axis at dim 0."
                )
            if parameter_ndim is not None and parameter.ndim != parameter_ndim:
                raise ValueError(
                    "Selected-expert matrix optimization requires grouped expert "
                    f"parameters with ndim={parameter_ndim}; got ndim={parameter.ndim}."
                )

        self.parameters = parameters
        self.expert_ids = ids
        self.expert_ids_by_name = plan
        self._ids_by_parameter = ids_by_parameter

    @staticmethod
    def _normalize_ids(values: Iterable[int]) -> tuple[int, ...]:
        raw_ids = tuple(int(value) for value in values)
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError("Selected expert ids must be unique.")
        ids = tuple(sorted(raw_ids))
        if ids and ids[0] < 0:
            raise ValueError("Selected expert ids must be non-negative.")
        return ids

    def ids_for(self, parameter: Any) -> tuple[int, ...]:
        try:
            return self._ids_by_parameter[id(parameter)]
        except KeyError as exc:
            raise ValueError(
                "Parameter is not owned by the selected-expert plan."
            ) from exc

    def index_for(
        self,
        parameter: Any,
        cache: dict[tuple[Any, tuple[int, ...]], Any],
    ) -> Any:
        selected_ids = self.ids_for(parameter)
        key = (parameter.device, selected_ids)
        index = cache.get(key)
        if index is None:
            index = cache[key] = torch.tensor(
                selected_ids,
                device=parameter.device,
                dtype=torch.long,
            )
        return index

    def initialize_state_metadata(
        self,
        parameter: Any,
        state: dict[str, Any],
    ) -> None:
        state["expert_ids"] = torch.tensor(
            self.ids_for(parameter),
            dtype=torch.int64,
        )

    def validate_state(
        self,
        parameter: Any,
        state: Mapping[str, Any],
        *,
        tensor_keys: Iterable[str],
    ) -> None:
        stored_ids = state.get("expert_ids")
        if not torch.is_tensor(stored_ids):
            raise ValueError(
                "Selected-expert optimizer checkpoint has no expert ids."
            )
        expected_ids = self.ids_for(parameter)
        observed_ids = tuple(int(value) for value in stored_ids.tolist())
        if observed_ids != expected_ids:
            raise ValueError(
                "Selected-expert optimizer checkpoint selection mismatch."
            )
        expected_shape = (
            len(expected_ids),
            *tuple(int(value) for value in parameter.shape[1:]),
        )
        for key in tensor_keys:
            value = state.get(key)
            if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
                raise ValueError(
                    "Selected-expert optimizer checkpoint has invalid "
                    f"{key} shape for the bound expert plan."
                )
