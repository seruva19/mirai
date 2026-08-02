"""Verify a downloaded MoE model snapshot before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.moe.artifacts.verification import verify_downloaded_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--manifest-name", default="download_manifest.json")
    args = parser.parse_args()

    report = verify_downloaded_snapshot(
        args.model_dir,
        expected_variant=args.variant,
        manifest_name=str(args.manifest_name),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
