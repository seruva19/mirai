"""Eval threshold and baseline gate helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mirai.core.training.evaluation.eval_metrics import BUILTIN_MEASURED_EVAL_METRICS

ALLOWED_ABSOLUTE_THRESHOLD_METRICS = {
    "test_loss_mean",
    "loss_tail_mean",
    "val_loss_tail_mean",
}.union(BUILTIN_MEASURED_EVAL_METRICS)


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        return s == "" or "placeholder" in s or s == "todo" or s == "unset"
    return False


def validate_eval_thresholds(
    thresholds: dict[str, Any],
    *,
    allowed_metrics: set[str] | None = None,
) -> None:
    allowed = set(ALLOWED_ABSOLUTE_THRESHOLD_METRICS)
    if allowed_metrics is not None:
        allowed.update(str(v) for v in allowed_metrics)
    bad = sorted([key for key, value in thresholds.items() if _is_placeholder(value)])
    if bad:
        raise ValueError(
            "Eval thresholds contain placeholders: " + ", ".join(bad)
        )
    invalid_metric_keys: list[str] = []
    for key in thresholds:
        name = str(key)
        if name.endswith("_max"):
            metric = name[: -len("_max")]
        elif name.endswith("_min"):
            metric = name[: -len("_min")]
        else:
            invalid_metric_keys.append(name)
            continue
        if metric not in allowed:
            invalid_metric_keys.append(name)
    if invalid_metric_keys:
        raise ValueError(
            "Eval thresholds reference unsupported or non-measured metrics: "
            + ", ".join(sorted(invalid_metric_keys))
        )


def load_measured_metrics_json(path: str | Path) -> dict[str, float]:
    in_path = Path(path)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Measured metrics JSON must be an object.")
    out: dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except Exception as exc:
            raise ValueError(
                f"Measured metrics JSON field '{key}' must be numeric."
            ) from exc
    return out


def handle_baseline_mode(
    *,
    baseline_mode: str,
    baseline_path: str | Path,
    metrics_payload: dict[str, Any],
    absolute_thresholds_passed: bool,
) -> Path | None:
    mode = baseline_mode.strip().lower()
    out = Path(baseline_path)
    if mode == "establish":
        if not absolute_thresholds_passed:
            raise ValueError(
                "baseline_mode=establish requires absolute-threshold pass before baseline creation."
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            out.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
        return out
    if mode == "require_existing":
        if not out.exists():
            raise FileNotFoundError(
                "baseline_mode=require_existing but baseline file is missing."
            )
        return out
    raise ValueError(f"Unsupported baseline_mode '{baseline_mode}'.")
