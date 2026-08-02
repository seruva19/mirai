"""Shared CLI runtime-policy helpers."""

from __future__ import annotations

import sys
from typing import Any

from mirai.config.loader import load_config
from mirai.config.runtime_policy import (
    apply_runtime_policy,
    validate_native_backend_availability,
    validate_cli_model_contract,
    validate_runtime_compatibility,
)


def load_runtime_config(
    config_path: str,
    *,
    entrypoint: str,
) -> tuple[Any, list[str]]:
    cfg = load_config(config_path)
    validate_cli_model_contract(
        cfg,
        entrypoint=entrypoint,
    )
    runtime_policy_notes = apply_runtime_policy(
        cfg,
        entrypoint=entrypoint,
    )
    validate_runtime_compatibility(cfg, entrypoint=entrypoint)
    validate_native_backend_availability(cfg, entrypoint=entrypoint)
    return cfg, runtime_policy_notes


def emit_runtime_policy_notes(notes: list[str]) -> None:
    for line in notes:
        print(f"[policy] {line}", file=sys.stderr)
