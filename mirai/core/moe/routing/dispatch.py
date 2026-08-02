"""Model-agnostic host contract for Expert-Choice routed execution."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExpertChoiceDispatchHost(Protocol):
    """Sparse expert host consuming canonical Expert-Choice route tensors."""

    logical_num_experts: int

    def run_expert_choice_routed(
        self,
        tokens: Any,
        expert_token_scores: Any,
        expert_token_indices: Any,
        *,
        tokens_per_sample: int,
    ) -> Any: ...


def validate_expert_choice_dispatch_host(
    host: Any, *, expected_experts: int
) -> ExpertChoiceDispatchHost:
    if not isinstance(host, ExpertChoiceDispatchHost):
        raise TypeError(
            "Expert-Choice host must implement logical_num_experts and "
            "run_expert_choice_routed()."
        )
    if int(host.logical_num_experts) != int(expected_experts):
        raise ValueError(
            "Expert-Choice host logical expert count does not match the router."
        )
    return host
