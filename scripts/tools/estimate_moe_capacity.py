"""Emit model/config-specific single-GPU MoE capacity guidance."""

from __future__ import annotations

import argparse
import json

from mirai.core.moe.monitoring.capacity import MoECapacitySpec, estimate_moe_capacity


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "num-layers",
        "num-experts",
        "experts-per-token",
        "hidden-size",
        "expert-intermediate-size",
        "tokens-per-step",
    ):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--weight-bits", type=float, default=16.0)
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--selected-expert-fraction", type=float, default=1.0)
    parser.add_argument("--resident-experts-per-layer", type=int, default=0)
    args = parser.parse_args()
    spec = MoECapacitySpec(
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        experts_per_token=args.experts_per_token,
        hidden_size=args.hidden_size,
        expert_intermediate_size=args.expert_intermediate_size,
        tokens_per_step=args.tokens_per_step,
        weight_bits=args.weight_bits,
        adapter_rank=args.adapter_rank,
        selected_expert_fraction=args.selected_expert_fraction,
        resident_experts_per_layer=args.resident_experts_per_layer,
    )
    print(json.dumps(estimate_moe_capacity(spec).to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
