"""Build training cache artifacts for sparse-MoE diffusion pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.runtime_policy import runtime_policy_summary
from mirai.core.dataset.cache import build_cache_from_config
from mirai.core.dataset.registration import enforce_dataset_compliance
from mirai.core.training.runtime.cli import (
    emit_runtime_policy_notes,
    load_runtime_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to TOML config")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg, policy_notes = load_runtime_config(
        args.config,
        entrypoint="cache",
    )
    emit_runtime_policy_notes(policy_notes)
    try:
        enforce_dataset_compliance(
            dataset_path=cfg.dataset.path,
            compliance_enabled=cfg.compliance.enabled,
            usage_mode=cfg.dataset.usage_mode,
            require_provenance=cfg.compliance.require_provenance,
            require_rights_attestation=cfg.compliance.require_rights_attestation,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    payload = build_cache_from_config(cfg)
    summary = dict(payload)
    summary.pop("records", None)
    summary["runtime_policy"] = runtime_policy_summary(cfg)
    summary["runtime_policy_notes"] = list(policy_notes)
    print(json.dumps({"status": "cached", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
