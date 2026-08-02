"""Composition seam for optional LingBot route selectors and score policies."""

from __future__ import annotations

from typing import Any

from mirai.core.moe.adaptation.diversity import DiversityAwareRoutingController
from mirai.core.moe.adaptation.dropout import ExpertDropoutController
from mirai.core.moe.adaptation.temperature import RouterTemperatureController
from mirai.core.moe.routing.dynamic_topk import BudgetedDynamicTopK
from mirai.core.moe.routing.prototypical import PrototypicalRouterExtension
from mirai.core.moe.routing.prototypical import PrototypicalRoutingSpec
from mirai.core.moe.routing.saliency import SharpMoESpec
from mirai.core.moe.routing.selective_sinkhorn import SelectiveSinkhornController


def _compose_score_policies(*policies: Any) -> Any:
    active = tuple(policy for policy in policies if policy is not None)
    if not active:
        return None

    def regularize(
        layer_name: str,
        top_indices: Any,
        top_scores: Any,
        *,
        training: bool,
    ) -> Any:
        scores = top_scores
        for policy in active:
            scores = policy.regularize(
                layer_name, top_indices, scores, training=training
            )
        return scores

    return regularize


def _route_selector(
    *,
    diversity: DiversityAwareRoutingController | None,
    selective_sinkhorn: SelectiveSinkhornController | None,
    prototypical: PrototypicalRouterExtension | None,
) -> Any:
    owners = tuple(
        name
        for name, value in (
            ("diversity", diversity),
            ("selective_sinkhorn", selective_sinkhorn),
            ("prototypical", prototypical),
        )
        if value is not None
    )
    if len(owners) > 1:
        raise ValueError(
            "Multiple policies cannot own the same route selection: "
            + ", ".join(owners)
        )
    if diversity is not None:
        def select_diverse(
            layer_name: str,
            scores: Any,
            native_top_indices: Any,
            *,
            training: bool,
            **_: Any,
        ) -> Any:
            return diversity.select(
                layer_name,
                scores,
                native_top_indices,
                training=training,
            )

        return select_diverse
    if selective_sinkhorn is not None:
        def select_sinkhorn(
            layer_name: str,
            _scores: Any,
            native_top_indices: Any,
            *,
            training: bool,
            score_logits: Any,
            native_top_weights: Any,
            valid_token_mask: Any | None,
            route_scale: float,
            **_: Any,
        ) -> Any:
            return selective_sinkhorn.select(
                layer_name,
                score_logits,
                native_top_indices,
                native_top_weights,
                valid_token_mask=valid_token_mask,
                route_scale=route_scale,
                training=training,
            )

        return select_sinkhorn
    if prototypical is not None:
        def select_prototypical(
            _layer_name: str,
            native_choice_scores: Any,
            native_top_indices: Any,
            *,
            training: bool,
            tokens: Any,
            native_gate_scores: Any,
            native_top_weights: Any,
            valid_token_mask: Any | None,
            route_scope_mask: Any | None,
            norm_topk_prob: bool,
            route_scale: float,
            choice_score_transform: Any | None,
            **_: Any,
        ) -> Any:
            return prototypical.select(
                tokens,
                native_choice_scores,
                native_gate_scores,
                native_top_indices,
                native_top_weights,
                route_scope_mask=route_scope_mask,
                valid_token_mask=valid_token_mask,
                norm_topk_prob=norm_topk_prob,
                route_scale=route_scale,
                training=training,
                choice_score_transform=choice_score_transform,
            )

        # ProMoE routing is part of the learned model at inference.  The other
        # route selectors are training-only and retain their eval fast path.
        select_prototypical._mirai_apply_in_eval = True  # type: ignore[attr-defined]
        return select_prototypical
    return None


def bind_lingbot_route_extensions(
    router: Any,
    *,
    layer_name: str,
    diversity: DiversityAwareRoutingController | None,
    expert_dropout: ExpertDropoutController | None,
    dynamic_topk: BudgetedDynamicTopK | None,
    router_temperature: RouterTemperatureController | None,
    selective_sinkhorn: SelectiveSinkhornController | None,
    prototypical: PrototypicalRouterExtension | None,
) -> None:
    router.set_router_logit_extension(
        layer_name=layer_name,
        transform=(
            router_temperature.transform_logits
            if router_temperature is not None
            else None
        ),
    )
    router.set_route_selection_extension(
        layer_name=layer_name,
        selector=_route_selector(
            diversity=diversity,
            selective_sinkhorn=selective_sinkhorn,
            prototypical=prototypical,
        ),
    )
    router.set_route_score_extension(
        _compose_score_policies(dynamic_topk, expert_dropout)
    )


def configure_lingbot_route_policy(
    pipeline: Any,
    *,
    policy_name: str,
    policy: Any,
) -> bool:
    if policy_name == "diversity_routing" and isinstance(
        policy, DiversityAwareRoutingController
    ):
        pipeline._diversity_routing_controller = policy
    elif policy_name == "expert_dropout" and isinstance(
        policy, ExpertDropoutController
    ):
        pipeline._expert_dropout_controller = policy
    elif policy_name == "router_temperature" and isinstance(
        policy, RouterTemperatureController
    ):
        pipeline._router_temperature_controller = policy
    elif policy_name == "selective_sinkhorn" and isinstance(
        policy, SelectiveSinkhornController
    ):
        pipeline._selective_sinkhorn_controller = policy
    elif policy_name == "prototypical_routing" and isinstance(
        policy, PrototypicalRoutingSpec
    ):
        pipeline._prototypical_routing_spec = policy
        pipeline.transformer._mirai_prototypical_routing_enabled = True
    elif policy_name == "sharp_moe" and isinstance(policy, SharpMoESpec):
        pipeline._sharp_moe_spec = policy
        pipeline.transformer._mirai_sharp_moe_enabled = True
    else:
        return False
    pipeline._collect_moe_router_modules()
    return True


__all__ = [
    "bind_lingbot_route_extensions",
    "configure_lingbot_route_policy",
]
