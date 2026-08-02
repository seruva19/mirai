"""Compress a trained LoRA with PARA global spectral rank allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.loader import load_config
from mirai.core.lineage import sha256_file
from mirai.core.models.adapters.para import compress_lora_state_para
from mirai.core.models.adapters.para import save_para_adapter_safetensors
from mirai.core.training.adapters import load_adapter_payload
from mirai.core.training.adapters import normalize_adapter_state


def compress_adapter(
    *,
    config_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    input_format: str,
    policy: str,
    rank_preservation_ratio: float,
    energy_preservation_ratio: float,
    overwrite: bool,
) -> dict[str, object]:
    config = load_config(config_path)
    if str(config.adapter.posthoc_rank_compression).strip().lower() != "para":
        raise ValueError(
            "PARA compression requires adapter.posthoc_rank_compression='para'."
        )
    source = Path(input_path)
    if not source.is_file():
        raise ValueError(f"PARA input adapter does not exist: {source}.")
    payload = load_adapter_payload(source)
    state = normalize_adapter_state(payload, lora_format=input_format)
    result = compress_lora_state_para(
        state,
        source_adapter_sha256=sha256_file(source),
        policy=policy,
        rank_preservation_ratio=float(rank_preservation_ratio),
        energy_preservation_ratio=float(energy_preservation_ratio),
    )
    output = save_para_adapter_safetensors(
        output_path,
        result,
        overwrite=bool(overwrite),
    )
    return {
        "status": "ok",
        "output": str(output),
        **result.summary(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "kohya", "diffusers", "peft"],
    )
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--rank-ratio", type=float)
    policy.add_argument("--energy-ratio", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    resolved_policy = "rank" if args.rank_ratio is not None else "energy"
    summary = compress_adapter(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
        input_format=args.input_format,
        policy=resolved_policy,
        rank_preservation_ratio=(
            float(args.rank_ratio) if args.rank_ratio is not None else 0.25
        ),
        energy_preservation_ratio=(
            float(args.energy_ratio) if args.energy_ratio is not None else 0.99
        ),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
