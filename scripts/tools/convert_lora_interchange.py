"""Convert Mirai LoRA checkpoints between fused (stacked) and unfused per-expert
key layouts.

Fused is the Mirai-native grouped-expert layout (``{name}.lora_a`` = ``[E, r, in]``);
unfused is the PEFT/diffusers per-expert layout (``{name}.experts.{e}.lora_A.weight``)
that external loaders (PEFT, ComfyUI, llama.cpp, vLLM) consume. The conversion is
a lossless key/axis rename (no LoRA math), so it round-trips bit-identically and
composes with the MoE-Sieve compact sparse export.

Examples::

    python scripts/tools/convert_lora_interchange.py --input fused.safetensors \\
        --output unfused.safetensors                 # auto-detect -> unfuse
    python scripts/tools/convert_lora_interchange.py --input unfused.safetensors \\
        --output fused.safetensors --direction fuse
    python scripts/tools/convert_lora_interchange.py --input fused.safetensors --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.models.adapters.lora_interchange import convert_lora_state_dict
from mirai.core.models.adapters.lora_interchange import detect_layout


def _load_safetensors(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    from safetensors import safe_open

    tensors: dict[str, Any] = {}
    metadata: dict[str, str] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
        md = handle.metadata()
        if md:
            metadata = dict(md)
    return tensors, metadata


def _save_safetensors(
    path: Path, tensors: dict[str, Any], metadata: dict[str, str]
) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = {key: value.contiguous() for key, value in tensors.items()}
    save_file(contiguous, str(path), metadata=metadata or None)


def _key_mapping(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def _shape(value: Any) -> list[int]:
        return list(getattr(value, "shape", ()))

    return {
        "input_keys": {key: _shape(value) for key, value in sorted(before.items())},
        "output_keys": {key: _shape(value) for key, value in sorted(after.items())},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input LoRA safetensors path")
    parser.add_argument("--output", help="Output safetensors path (omit with --dry-run)")
    parser.add_argument(
        "--direction",
        default="auto",
        choices=["auto", "fuse", "unfuse"],
        help="Conversion direction (auto detects from key layout).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the key mapping without writing an output file.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not fail fast on unknown keys (they are dropped).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    in_path = Path(args.input)
    tensors, metadata = _load_safetensors(in_path)
    detected = detect_layout(tensors)
    converted, applied = convert_lora_state_dict(
        tensors, direction=str(args.direction), strict=not args.no_strict
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "input": str(in_path),
        "detected_layout": detected,
        "applied_direction": applied,
        "input_key_count": len(tensors),
        "output_key_count": len(converted),
    }

    if args.dry_run:
        summary["mode"] = "dry-run"
        summary["mapping"] = _key_mapping(tensors, converted)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if not args.output:
        raise SystemExit("--output is required unless --dry-run is set.")
    out_path = Path(args.output)
    _save_safetensors(out_path, converted, metadata)
    summary["mode"] = "convert"
    summary["output"] = str(out_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
