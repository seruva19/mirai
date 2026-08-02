"""Export/convert adapter checkpoints across supported formats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.training.adapters import (
    adapter_metadata_from_state,
    load_adapter_payload,
    merge_adapter_into_base_scale,
    normalize_adapter_state,
    save_diffusers_adapter_safetensors,
    save_kohya_adapter_safetensors,
    save_lycoris_adapter_safetensors,
)
from mirai.core.persistence.checkpoints import save_checkpoint
from mirai.core.models.adapters.lora import expand_sparse_expert_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input adapter checkpoint/safetensors path")
    p.add_argument("--output", required=True, help="Output path")
    p.add_argument(
        "--format",
        default="kohya",
        choices=["kohya", "diffusers", "peft", "lycoris"],
        help="Export format when not using --merge",
    )
    p.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "kohya", "diffusers", "peft", "lycoris"],
        help="Input format hint",
    )
    p.add_argument("--rank", type=int, default=16, help="Adapter rank metadata")
    p.add_argument("--alpha", type=float, default=16.0, help="Adapter alpha metadata")
    p.add_argument(
        "--target-modules",
        default="transformer.attn",
        help="Comma-separated target module names",
    )
    p.add_argument("--model-path", default="./models/lingbot_video", help="Base model path metadata")
    p.add_argument("--merge", action="store_true", help="Export merged base scalar checkpoint")
    p.add_argument("--base-scale", type=float, default=1.0, help="Base scalar before merge")
    p.add_argument(
        "--adapter-type",
        default="lora",
        choices=["lora", "loha", "lokr"],
        help="Adapter type hint (used by --format lycoris).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    payload = load_adapter_payload(in_path)
    # Compact (sparse) expert-LoRA checkpoints are NOT kohya/ComfyUI-portable;
    # expand them to full-shape here so downstream portable formats are faithful
    # (unselected experts have lora_b == 0 -> zero contribution).
    payload = expand_sparse_expert_state(payload)
    adapter_state = normalize_adapter_state(payload, lora_format=args.input_format)
    target_modules = [s.strip() for s in str(args.target_modules).split(",") if s.strip()]
    metadata = adapter_metadata_from_state(
        adapter_state=adapter_state,
        model_path=str(args.model_path),
        target_modules=target_modules,
    )

    if args.merge:
        merged_state = merge_adapter_into_base_scale(
            adapter_state=adapter_state,
            base_scale=float(args.base_scale),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            out_path,
            {
                "merged_base_state": merged_state,
                "adapter_state": adapter_state,
                "input_path": str(in_path),
            },
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "merge",
                    "output": str(out_path),
                    "merged_base_scale": float(merged_state["base_scale"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.format == "kohya":
        save_kohya_adapter_safetensors(
            out_path,
            adapter_state=adapter_state,
            rank=int(args.rank),
            alpha=float(args.alpha),
            target_modules=target_modules,
            model_path=str(args.model_path),
            metadata_extra=metadata,
        )
    elif args.format == "diffusers":
        save_diffusers_adapter_safetensors(
            out_path,
            adapter_state=adapter_state,
            rank=int(args.rank),
            alpha=float(args.alpha),
            target_modules=target_modules,
            model_path=str(args.model_path),
            metadata_extra=metadata,
        )
    elif args.format == "lycoris":
        if str(args.adapter_type).strip().lower() not in {"loha", "lokr"}:
            raise SystemExit("--format lycoris requires --adapter-type loha|lokr.")
        save_lycoris_adapter_safetensors(
            out_path,
            adapter_state=adapter_state,
            adapter_type=args.adapter_type,
            rank=int(args.rank),
            alpha=float(args.alpha),
            target_modules=target_modules,
            model_path=str(args.model_path),
            metadata_extra=metadata,
        )
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            out_path,
            {
                "adapter_state": adapter_state,
                "metadata": {
                    "format": "peft",
                    "rank": int(args.rank),
                    "alpha": float(args.alpha),
                    "target_modules": target_modules,
                    "base_model_path": str(args.model_path),
                },
            },
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "convert",
                "format": str(args.format),
                "output": str(out_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
