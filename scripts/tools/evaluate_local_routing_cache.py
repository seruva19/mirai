"""Estimate segment-cache oracle hit rates from detached MoE routing traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.moe.monitoring.local_routing import (
    build_local_routing_cache_evidence,
    save_local_routing_cache_evidence,
)


def _integers(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--segment-lengths", type=_integers, default=[1, 2, 4, 8])
    parser.add_argument("--cache-sizes", type=_integers, required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    observations = json.loads(Path(args.observations).read_text(encoding="utf-8"))
    evidence = build_local_routing_cache_evidence(
        observations,
        segment_lengths=args.segment_lengths,
        cache_sizes=args.cache_sizes,
        dataset_fingerprint=args.dataset_fingerprint,
        model_fingerprint=args.model_fingerprint,
    )
    save_local_routing_cache_evidence(args.output, evidence, overwrite=args.overwrite)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
