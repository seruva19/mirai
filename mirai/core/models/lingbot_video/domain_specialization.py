"""LingBot binding for online domain-to-expert adapter specialization."""

from __future__ import annotations

from typing import Any

from mirai.core.moe.adaptation.domain_specialization import (
    DomainExpertSpecializationController,
)


def configure_lingbot_domain_expert_specialization(
    pipeline: Any, policy: DomainExpertSpecializationController
) -> None:
    if not isinstance(policy, DomainExpertSpecializationController):
        raise TypeError(
            "LingBot domain expert specialization requires its typed controller."
        )
    bound = 0
    for name, module in pipeline.transformer.named_modules():
        router = getattr(module, "router", None)
        experts = getattr(module, "experts", None)
        if router is None or experts is None or not hasattr(router, "num_experts"):
            continue
        compressed = getattr(experts, "expert_lora", None)
        if compressed:
            setter = getattr(experts, "set_routed_adapter_gate", None)
            policy.bind_layer(
                name,
                router=router,
                adapters=tuple(compressed.values()),
                set_route_gate=setter if callable(setter) else None,
            )
            bound += 1
            continue
        extension_getter = getattr(experts, "linear_extension", None)
        extension = (
            extension_getter() if callable(extension_getter) else None
        )
        setter = getattr(extension, "set_routed_adapter_gate", None)
        parametrizations = getattr(experts, "parametrizations", None)
        adapters = []
        for key in ("w1", "w2", "w3"):
            chain = (
                getattr(parametrizations, key)
                if parametrizations is not None and hasattr(parametrizations, key)
                else ()
            )
            if len(chain) == 1 and hasattr(chain[0], "activation_factors"):
                adapters.append(chain[0])
        if adapters:
            if not callable(setter):
                raise RuntimeError(
                    "Native domain expert specialization requires "
                    "adapter.expert_tensor_lora_backend='activation'."
                )
            policy.bind_layer(
                name,
                router=router,
                adapters=tuple(adapters),
                set_route_gate=setter,
            )
            bound += 1
    if not bound:
        raise ValueError(
            "Domain expert specialization requires routed expert LoRA adapters."
        )
