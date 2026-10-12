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
from mirai.core.moe.artifacts.downloads import get_download_repo_by_variant
from mirai.core.moe.artifacts.downloads import verify_pinned_artifact_files
from mirai.core.moe.artifacts.downloads import verify_artifact_manifest_source


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
    registered_variant = str(args.variant or report.variant)
    if registered_variant in get_download_repo_by_variant():
        verify_artifact_manifest_source(
            args.model_dir,
            registered_variant,
            manifest_name=str(args.manifest_name),
        )
        verify_pinned_artifact_files(args.model_dir, registered_variant)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
