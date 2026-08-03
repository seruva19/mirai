"""Report per-layer routing agreement for configured INT8 router storage."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.loader import load_config
from mirai.config.runtime_policy import (
    validate_cli_model_contract,
    validate_native_backend_availability,
)
from mirai.config.schema import TrainingConfig
from mirai.core.models.providers import load_configured_model_provider_module
from mirai.core.moe.monitoring.agreement import compare_router_selections
from mirai.core.moe.runtime.specs import normalize_router_quantization_policy
from mirai.core.models.providers import get_model_family_provider
from mirai.core.training.trainer import _instantiate_model_pipeline
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoRouter


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(output)


def build_router_quantization_agreement(
    config: TrainingConfig,
    *,
    tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Load configured routers and compare original versus INT8 selections."""
    import torch
    import torch.nn.functional as F

    sample_count = int(tokens)
    if sample_count <= 0:
        raise ValueError("--tokens must be positive.")
    policy = normalize_router_quantization_policy(
        config.memory.router_quantization
    )
    if policy == "disabled":
        raise ValueError(
            "No quantized router is configured; set "
            "memory.router_quantization='int8_per_channel'."
        )

    validate_cli_model_contract(
        config,
        entrypoint="router-quantization-agreement",
    )
    load_configured_model_provider_module(config)
    validate_native_backend_availability(
        config,
        entrypoint="router-quantization-agreement",
    )
    provider = get_model_family_provider(config.model.type)
    if provider is None:
        raise ValueError(f"Unknown model family '{config.model.type}'.")
    model_cls = provider.require_pipeline_type()
    pipeline = _instantiate_model_pipeline(model_cls, config)
    training_model = pipeline.get_training_model()
    if training_model is None:
        raise ValueError(
            f"model.type='{config.model.type}' does not expose a training model."
        )
    routers = [
        (name, module)
        for name, module in training_model.named_modules()
        if isinstance(module, LingBotVideoRouter)
    ]
    if not routers:
        raise ValueError(
            "No LingBotVideoRouter modules were found; routing quantization "
            "agreement requires shared expert numbering on LingBot routers."
        )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sampled_tokens: dict[int, torch.Tensor] = {}
    layers: list[dict[str, Any]] = []
    for name, router in routers:
        if hasattr(router, "weight_int8"):
            raise ValueError(
                f"Router {name!r} was already quantized before its reference "
                "weight could be measured."
            )
        weight = router.weight.detach().float()
        hidden_size = int(weight.shape[1])
        if hidden_size not in sampled_tokens:
            sampled_tokens[hidden_size] = torch.randn(
                sample_count,
                hidden_size,
                generator=generator,
                dtype=torch.float32,
            )
        layer_tokens = sampled_tokens[hidden_size].to(device=weight.device)
        reference_logits = F.linear(layer_tokens, weight)
        reference_scores = (
            F.softmax(reference_logits, dim=-1)
            if router.score_func == "softmax"
            else reference_logits.sigmoid()
        )
        reference_scores = reference_scores + router.e_score_correction_bias.to(
            device=weight.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        router.weight.requires_grad_(False)
        router.enable_int8_weight()
        if not hasattr(router, "weight_int8"):
            raise ValueError(
                "No quantized router buffer was produced; set "
                "memory.router_quantization='int8_per_channel'."
            )
        candidate_weight = router._execution_weight(
            device=weight.device,
            dtype=torch.float32,
        )
        candidate_logits = F.linear(layer_tokens, candidate_weight)
        candidate_scores = (
            F.softmax(candidate_logits, dim=-1)
            if router.score_func == "softmax"
            else candidate_logits.sigmoid()
        )
        candidate_scores = candidate_scores + router.e_score_correction_bias.to(
            device=weight.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        report = compare_router_selections(
            reference_scores,
            candidate_scores,
            top_k=int(router.top_k),
            num_experts=int(router.num_experts),
        )
        layers.append(
            {
                "layer": name,
                "top_k": int(router.top_k),
                "num_experts": int(router.num_experts),
                **asdict(report),
            }
        )

    return {
        "model_type": str(config.model.type),
        "router_quantization": policy,
        "tokens": sample_count,
        "seed": int(seed),
        "layers": layers,
        "summary": {
            "layer_count": len(layers),
            "min_agreement": min(layer["agreement"] for layer in layers),
            "max_changed_token_fraction": max(
                layer["changed_token_fraction"] for layer in layers
            ),
        },
    }


def _print_report(payload: dict[str, Any]) -> None:
    print(
        f"{'layer':<48} {'experts':>7} {'top_k':>5} {'agreement':>11} "
        f"{'changed':>11} {'margin_p05':>12} {'margin_min':>12}"
    )
    for layer in payload["layers"]:
        print(
            f"{layer['layer']:<48} {layer['num_experts']:>7d} "
            f"{layer['top_k']:>5d} {layer['agreement']:>11.6f} "
            f"{layer['changed_token_fraction']:>11.6f} "
            f"{layer['margin_p05']:>12.6g} {layer['margin_min']:>12.6g}"
        )
    summary = payload["summary"]
    print(
        f"summary: layers={summary['layer_count']} "
        f"min_agreement={summary['min_agreement']:.6f} "
        "max_changed_token_fraction="
        f"{summary['max_changed_token_fraction']:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-agreement", type=float, default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.min_agreement is not None and not 0.0 <= args.min_agreement <= 1.0:
        parser.error("--min-agreement must be between 0 and 1.")

    payload = build_router_quantization_agreement(
        load_config(args.config),
        tokens=args.tokens,
        seed=args.seed,
    )
    _print_report(payload)
    if args.json:
        _write_json(args.json, payload)
    if (
        args.min_agreement is not None
        and payload["summary"]["min_agreement"] < args.min_agreement
    ):
        print(
            f"gate failed: min_agreement "
            f"{payload['summary']['min_agreement']:.6f} < "
            f"{args.min_agreement:.6f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
