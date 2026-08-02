"""Build a versioned router-health report from Mirai metrics JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mirai.core.moe.monitoring.report import build_router_health_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.metrics)
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = build_router_health_report(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
