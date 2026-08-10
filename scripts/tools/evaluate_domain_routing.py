"""Evaluate domain-labelled MoE routes against an explicit reference router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.moe.monitoring.domain_testbed import (
    build_domain_routing_testbed_evidence,
    save_domain_routing_testbed_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    observations = json.loads(Path(args.observations).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    evidence = build_domain_routing_testbed_evidence(
        observations,
        reference,
        dataset_fingerprint=args.dataset_fingerprint,
        model_fingerprint=args.model_fingerprint,
    )
    save_domain_routing_testbed_evidence(
        args.output, evidence, overwrite=bool(args.overwrite)
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
