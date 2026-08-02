"""Validate LoRA load + output generation through ComfyUI HTTP API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lora", required=True, help="Path to LoRA safetensors")
    p.add_argument(
        "--workflow",
        required=True,
        help="ComfyUI workflow JSON template with a ${LORA_PATH} substitution token",
    )
    p.add_argument(
        "--server",
        required=True,
        help="ComfyUI server base URL",
    )
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "compatibility" / "comfyui_lora_check.json"),
        help="Output report path",
    )
    return p.parse_args()


def _server_alive(client: Any, server: str, timeout: int = 5) -> bool:
    try:
        r = client.get(f"{server.rstrip('/')}/object_info", timeout=timeout)
        return r.ok
    except Exception:
        return False


def _replace_tokens(payload: Any, replacements: dict[str, str]) -> Any:
    if isinstance(payload, dict):
        return {k: _replace_tokens(v, replacements) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_replace_tokens(v, replacements) for v in payload]
    if isinstance(payload, str):
        out = payload
        for key, value in replacements.items():
            out = out.replace(f"${{{key}}}", value)
        return out
    return payload


def main() -> int:
    args = parse_args()
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ComfyUI compatibility checks require the Mirai 'tools' extra."
        ) from exc

    lora_path = Path(args.lora).resolve()
    workflow_path = Path(args.workflow).resolve()
    server = args.server.rstrip("/")
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "status": "failed",
        "server": server,
        "lora_path": str(lora_path),
        "workflow_path": str(workflow_path),
        "checked_at_unix": int(time.time()),
    }

    if not lora_path.exists():
        report["error"] = f"LoRA file not found: {lora_path}"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    if not workflow_path.exists():
        report["error"] = f"Workflow file not found: {workflow_path}"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    if not _server_alive(requests, server):
        report["status"] = "skipped"
        report["error"] = f"ComfyUI server not reachable at {server}"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    template = json.loads(workflow_path.read_text(encoding="utf-8"))
    prompt_payload = _replace_tokens(template, {"LORA_PATH": str(lora_path)})
    req = requests.post(
        f"{server}/prompt",
        json={"prompt": prompt_payload},
        timeout=20,
    )
    req.raise_for_status()
    body = req.json()
    prompt_id = str(body.get("prompt_id", ""))
    if not prompt_id:
        raise RuntimeError("ComfyUI /prompt response missing prompt_id")

    deadline = time.time() + float(args.timeout_seconds)
    history_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        r = requests.get(f"{server}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        hist = r.json()
        entry = hist.get(prompt_id, {})
        if entry:
            history_payload = entry
            status = ((entry.get("status") or {}).get("status_str") or "").lower()
            if status in {"success", "error"}:
                break
        time.sleep(1.0)

    if history_payload is None:
        report["error"] = "Timed out waiting for ComfyUI history entry."
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    status_str = str(((history_payload.get("status") or {}).get("status_str") or "")).lower()
    outputs = history_payload.get("outputs", {})
    produced_files: list[str] = []
    if isinstance(outputs, dict):
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            images = node_output.get("images")
            if isinstance(images, list):
                for img in images:
                    if isinstance(img, dict):
                        filename = str(img.get("filename", "")).strip()
                        if filename:
                            produced_files.append(filename)

    report.update(
        {
            "prompt_id": prompt_id,
            "comfy_status": status_str or "unknown",
            "produced_files": produced_files,
        }
    )
    ok = status_str == "success" and len(produced_files) > 0
    report["status"] = "ok" if ok else "failed"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
