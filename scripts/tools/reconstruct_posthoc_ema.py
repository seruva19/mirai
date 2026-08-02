"""Reconstruct an arbitrary adapter EMA profile from Mirai snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.persistence.checkpoints import save_checkpoint
from mirai.core.training.optim.posthoc_ema import (
    load_posthoc_ema_snapshot,
    reconstruct_posthoc_ema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a power-function EMA adapter at one saved training step."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing step_XXXXXXXX.pt post-hoc EMA snapshots.",
    )
    parser.add_argument("--output", required=True, help="Output adapter .pt path.")
    parser.add_argument(
        "--std",
        required=True,
        type=float,
        help="Desired relative response standard deviation in (0, 0.289).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Saved target step; 0 selects the latest snapshot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("step_*.pt"))
    if not paths:
        raise SystemExit(f"No post-hoc EMA snapshots found in '{input_dir}'.")
    snapshots = [load_posthoc_ema_snapshot(path) for path in paths]
    result = reconstruct_posthoc_ema(
        snapshots,
        output_std=float(args.std),
        output_step=None if int(args.step) == 0 else int(args.step),
    )
    result["posthoc_ema"]["source_snapshots"] = [path.name for path in paths]
    output = save_checkpoint(Path(args.output), result)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output),
                **result["posthoc_ema"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
