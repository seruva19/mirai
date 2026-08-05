"""Explicit composition root for Mirai-owned registry components."""

from __future__ import annotations

import importlib


# One searchable inventory replaces ambient package scanning. Modules remain
# self-registering so third-party provider_module extensions use the same small
# API, but Mirai-owned discovery order and ownership are explicit here.
BUILTIN_COMPONENT_MODULES: tuple[tuple[str, str], ...] = (
    ("prompt_rewriter", "mirai.core.models.lingbot_video.prompting"),
    ("model", "mirai.core.models.lingbot_video.pipeline"),
    ("model", "mirai.core.models.magi2_preview.pipeline"),
    ("strategy", "mirai.core.training.strategies.text_to_video"),
    ("strategy", "mirai.core.training.strategies.image_to_video"),
    ("strategy", "mirai.core.training.strategies.hybrid_conditioning"),
    ("strategy", "mirai.core.training.strategies.multi_task_video"),
    ("loss", "mirai.core.training.objectives.registry"),
    ("objective", "mirai.core.training.objectives.flow_matching"),
    ("objective", "mirai.core.training.objectives.regression"),
    ("objective", "mirai.core.training.objectives.sharp_moe"),
    ("optimizer", "mirai.core.training.optim.optimizer"),
    ("scheduler", "mirai.core.training.optim.scheduler"),
    ("training_policy", "mirai.core.training.policies.dataset_routing"),
    ("training_policy", "mirai.core.training.policies.dispersive_loss"),
    ("training_policy", "mirai.core.training.policies.domain_expert_specialization"),
    ("training_policy", "mirai.core.training.policies.diversity_routing"),
    ("training_policy", "mirai.core.training.policies.expert_dropout"),
    ("training_policy", "mirai.core.training.policies.moe_token_chunking"),
    ("training_policy", "mirai.core.training.policies.router_temperature"),
    ("training_policy", "mirai.core.training.policies.router_stage_schedule"),
    ("training_policy", "mirai.core.training.policies.router_distillation"),
    ("training_policy", "mirai.core.training.policies.simbal"),
    ("training_policy", "mirai.core.training.policies.selective_sinkhorn"),
    ("training_policy", "mirai.core.training.policies.prototypical_routing"),
    ("training_policy", "mirai.core.training.policies.sharp_moe"),
    ("training_policy", "mirai.core.training.policies.mixture_of_depths"),
    ("training_policy", "mirai.core.training.policies.preemptive_monitoring"),
)


def register_builtin_components() -> None:
    for _role, module_name in BUILTIN_COMPONENT_MODULES:
        importlib.import_module(module_name)
