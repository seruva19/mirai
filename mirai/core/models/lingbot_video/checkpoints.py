"""Native LingBot-Video transformer config and checkpoint loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from mirai.core.models.checkpoint_streaming import assign_module_tensor
from mirai.core.models.checkpoint_streaming import iter_safetensor_keys
from mirai.core.models.checkpoint_streaming import load_module_safetensors
from mirai.core.models.checkpoint_streaming import resolve_safetensor_files

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


from mirai.core.models.checkpoint_streaming import NativeCheckpointLoadReport as LingBotVideoCheckpointReport


def resolve_lingbot_transformer_dir(
    model_path: str | Path,
    *,
    subfolder: str = "transformer",
) -> Path | None:
    root = Path(model_path)
    if root.is_dir() and (root / "config.json").is_file():
        return root
    component = root / str(subfolder or "transformer")
    if component.is_dir() and (component / "config.json").is_file():
        return component
    transformer = root / "transformer"
    if transformer.is_dir() and (transformer / "config.json").is_file():
        return transformer
    return None


def load_lingbot_transformer_config(
    model_path: str | Path,
    *,
    subfolder: str = "transformer",
) -> dict[str, Any]:
    transformer_dir = resolve_lingbot_transformer_dir(model_path, subfolder=subfolder)
    if transformer_dir is None:
        raise FileNotFoundError(
            f"LingBot-Video transformer config not found under '{model_path}'. "
            f"Expected {str(subfolder or 'transformer')}/config.json, "
            "transformer/config.json, or a direct transformer directory."
        )
    config_path = transformer_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {key: value for key, value in payload.items() if not str(key).startswith("_")}


def _candidate_weight_files(transformer_dir: Path) -> list[Path]:
    return list(
        resolve_safetensor_files(
            transformer_dir,
            index_names=("diffusion_pytorch_model.safetensors.index.json",),
            direct_names=("diffusion_pytorch_model.safetensors",),
        )
    )


def _normalize_key(key: str) -> str:
    text = str(key)
    for prefix in ("module.", "_orig_mod.", "pipeline.", "denoiser.", "model.", "transformer."):
        while text.startswith(prefix):
            text = text[len(prefix) :]
    return text


def discover_lingbot_transformer_checkpoint_keys(
    model_path: str | Path,
    *,
    subfolder: str = "transformer",
) -> tuple[str, ...]:
    transformer_dir = resolve_lingbot_transformer_dir(model_path, subfolder=subfolder)
    if transformer_dir is None:
        raise FileNotFoundError(
            f"LingBot-Video transformer directory not found under '{model_path}'."
        )
    files = _candidate_weight_files(transformer_dir)
    if not files:
        raise FileNotFoundError(
            f"No LingBot-Video transformer safetensors found under '{transformer_dir}'."
        )
    index = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        if isinstance(weight_map, dict):
            return tuple(_normalize_key(str(key)) for key in weight_map.keys())
    keys = tuple(_normalize_key(str(key)) for key in iter_safetensor_keys(files))
    if not keys:
        raise ValueError("LingBot-Video checkpoint did not contain any tensor keys.")
    return keys


def _resolve_module_tensor(model: Any, key: str) -> tuple[Any, str]:
    if "." not in key:
        return model, key
    module_name, tensor_name = key.rsplit(".", 1)
    return model.get_submodule(module_name), tensor_name


def _assign_tensor(model: Any, key: str, value: Any) -> None:
    assign_module_tensor(model, key, value)


def load_lingbot_transformer_checkpoint(
    model: Any,
    model_path: str | Path,
    *,
    strict: bool,
    subfolder: str = "transformer",
    expected_keys: Iterable[str] | None = None,
    tensor_handlers: dict[str, Callable[[Any], None]] | None = None,
) -> LingBotVideoCheckpointReport:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to load LingBot-Video checkpoints.")
    transformer_dir = resolve_lingbot_transformer_dir(model_path, subfolder=subfolder)
    if transformer_dir is None:
        raise FileNotFoundError(
            f"LingBot-Video transformer directory not found under '{model_path}'."
        )
    files = _candidate_weight_files(transformer_dir)
    if not files:
        raise FileNotFoundError(
            f"No LingBot-Video transformer safetensors found under '{transformer_dir}'."
        )
    report = load_module_safetensors(
        model,
        files,
        component_path=transformer_dir,
        strict=bool(strict),
        normalize_key=_normalize_key,
        expected_keys=expected_keys,
        tensor_handlers=tensor_handlers,
    )
    if report.matched_keys <= 0:
        raise ValueError(
            "LingBot-Video checkpoint did not match Mirai native transformer keys."
        )
    return report
