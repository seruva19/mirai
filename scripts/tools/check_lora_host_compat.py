"""Host compatibility checks for exported LoRA safetensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.training.adapters import (  # noqa: E402
    KOHYA_A_KEY,
    KOHYA_ALPHA_KEY,
    KOHYA_B_KEY,
    load_adapter_payload,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lora", required=True, help="Path to exported LoRA safetensors")
    p.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "compatibility" / "lora_host_compat.json"),
        help="Output JSON report path",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any host contract check fails",
    )
    return p.parse_args()


def _build_host_report(*, host: str, passed: bool, reasons: list[str]) -> dict[str, object]:
    return {"host": host, "passed": bool(passed), "reasons": reasons}


def main() -> int:
    args = parse_args()
    lora_path = Path(args.lora)
    payload = load_adapter_payload(lora_path)
    metadata = payload.get("_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    keys = {str(k) for k in payload.keys() if str(k) != "_metadata"}
    required_tensor_keys = {KOHYA_A_KEY, KOHYA_B_KEY, KOHYA_ALPHA_KEY}
    missing_tensors = sorted(required_tensor_keys - keys)

    required_md = {"format", "rank", "alpha", "target_modules", "base_model_hash"}
    md_keys = {str(k) for k in metadata.keys()}
    missing_md = sorted(required_md - md_keys)

    common_reasons: list[str] = []
    if lora_path.suffix.lower() != ".safetensors":
        common_reasons.append("Export must use .safetensors")
    if missing_tensors:
        common_reasons.append(f"Missing tensor keys: {', '.join(missing_tensors)}")
    if missing_md:
        common_reasons.append(f"Missing metadata keys: {', '.join(missing_md)}")

    a1111_reasons = list(common_reasons)
    if str(metadata.get("format", "")).lower() != "kohya":
        a1111_reasons.append("Metadata format must be 'kohya' for A1111 compatibility")
    a1111_passed = len(a1111_reasons) == 0

    invoke_reasons = list(common_reasons)
    if not str(metadata.get("target_modules", "")).strip():
        invoke_reasons.append("target_modules metadata must be non-empty")
    invoke_passed = len(invoke_reasons) == 0

    report = {
        "status": "ok" if (a1111_passed and invoke_passed) else "failed",
        "lora_path": str(lora_path.resolve()),
        "tensor_key_count": len(keys),
        "metadata_keys": sorted(md_keys),
        "hosts": [
            _build_host_report(host="A1111", passed=a1111_passed, reasons=a1111_reasons),
            _build_host_report(host="InvokeAI", passed=invoke_passed, reasons=invoke_reasons),
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.strict and report["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
