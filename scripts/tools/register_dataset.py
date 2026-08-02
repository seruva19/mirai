"""Register dataset splits and compliance metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.loader import load_config
from mirai.core.dataset.registration import register_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to TOML config")
    p.add_argument(
        "--out",
        default="",
        help="Optional registration metadata output path (defaults to dataset_path/registration.json).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    out = args.out or str(Path(cfg.dataset.path) / "registration.json")
    payload = register_dataset(
        dataset_path=cfg.dataset.path,
        output_path=out,
        split_seed=cfg.dataset.split_seed,
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio,
        test_ratio=cfg.dataset.test_ratio,
        compliance_enabled=cfg.compliance.enabled,
        usage_mode=cfg.dataset.usage_mode,
        require_provenance=cfg.compliance.require_provenance,
        require_rights_attestation=cfg.compliance.require_rights_attestation,
    )
    print(json.dumps({"status": "registered", "output_path": out, **payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
