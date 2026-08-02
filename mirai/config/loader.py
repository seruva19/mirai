"""Config loader with default/preset/user merge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mirai.core.models.providers import (
    get_model_family_provider,
    load_model_provider_module,
)

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - py310 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from mirai.config.schema import TrainingConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _resolve_defaults_path(
    *,
    config_dir: Path,
    user_data: dict[str, Any],
    preset_data: dict[str, Any] | None = None,
) -> Path | None:
    probe_data = _deep_merge(dict(preset_data or {}), user_data)
    model_table = probe_data.get("model", {})
    if isinstance(model_table, dict):
        model_type = str(model_table.get("type", "") or "").strip().lower()
        provider_module = str(model_table.get("provider_module", "") or "")
        load_model_provider_module(provider_module)
        provider = get_model_family_provider(model_type) if model_type else None
        defaults_name = str(getattr(provider, "config_defaults_name", "") or "").strip()
        if defaults_name:
            candidate = config_dir / "defaults" / f"{defaults_name}.toml"
            if candidate.exists():
                return candidate
        if model_type:
            return None
    return config_dir / "defaults" / "moe.toml"


def load_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    user_data = _load_toml(config_path)
    preset_name = user_data.get("preset")
    if not preset_name:
        return TrainingConfig.from_dict(user_data)

    config_dir = Path(__file__).resolve().parent
    preset_path = config_dir / "presets" / f"{preset_name}.toml"
    preset_data = _load_toml(preset_path)
    defaults_path = _resolve_defaults_path(
        config_dir=config_dir,
        user_data=user_data,
        preset_data=preset_data,
    )
    defaults_data = _load_toml(defaults_path) if defaults_path is not None else {}
    merged = _deep_merge(defaults_data, preset_data)
    merged = _deep_merge(merged, user_data)
    return TrainingConfig.from_dict(merged)
