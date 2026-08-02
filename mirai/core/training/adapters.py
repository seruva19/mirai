"""Adapter checkpoint format helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mirai.core.models.adapters.dora import DORA_MAGNITUDE_SUFFIX
from mirai.core.models.adapters.lora_allocation import RSLORA_STATE_SUFFIX

from mirai.core.persistence.checkpoints import load_checkpoint

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

KOHYA_A_KEY = "lora_unet_lora_a.weight"
KOHYA_B_KEY = "lora_unet_lora_b.weight"
KOHYA_ALPHA_KEY = "lora_unet.alpha"
DIFFUSERS_A_KEY = "unet.lora_a.weight"
DIFFUSERS_B_KEY = "unet.lora_b.weight"
DIFFUSERS_ALPHA_KEY = "unet.lora_alpha"
LYCORIS_LOHA_A_KEY = "lycoris.loha_a"
LYCORIS_LOHA_B_KEY = "lycoris.loha_b"
LYCORIS_LOHA_A1_KEY = "lycoris.loha_a1"
LYCORIS_LOHA_B1_KEY = "lycoris.loha_b1"
LYCORIS_LOHA_A2_KEY = "lycoris.loha_a2"
LYCORIS_LOHA_B2_KEY = "lycoris.loha_b2"
LYCORIS_LOKR_A_KEY = "lycoris.lokr_a"
LYCORIS_LOKR_B_KEY = "lycoris.lokr_b"
LYCORIS_LOKR_A1_KEY = "lycoris.lokr_a1"
LYCORIS_LOKR_A2_KEY = "lycoris.lokr_a2"
LYCORIS_LOKR_B1_KEY = "lycoris.lokr_b1"
LYCORIS_LOKR_B2_KEY = "lycoris.lokr_b2"
KOHYA_DOWN_SUFFIX = ".lora_down.weight"
KOHYA_UP_SUFFIX = ".lora_up.weight"
LIGHTX2V_DOWN_SUFFIX = ".lora.down.weight"
LIGHTX2V_UP_SUFFIX = ".lora.up.weight"
KOHYA_DORA_SUFFIX = ".dora_scale"
DIFFUSERS_DORA_SUFFIX = ".lora_magnitude_vector.weight"
_LIGHTWEIGHT_EXPERT_PREFIX = "lightweight_experts."
_LIGHTWEIGHT_EXPERT_SCHEMA_METADATA = (
    "mirai_lightweight_expert_schema_version"
)
_LIGHTWEIGHT_EXPERT_TOPOLOGY_METADATA = "mirai_lightweight_expert_topology"
_DECOUPLED_ROUTING_PREFIX = "decoupled_routing."
_DECOUPLED_ROUTING_SCHEMA_METADATA = "mirai_decoupled_routing_schema_version"
_DECOUPLED_ROUTING_TOPOLOGY_METADATA = "mirai_decoupled_routing_topology"
_CHAIN_OF_EXPERTS_PREFIX = "chain_of_experts."
_CHAIN_OF_EXPERTS_SCHEMA_METADATA = "mirai_chain_of_experts_schema_version"
_CHAIN_OF_EXPERTS_TOPOLOGY_METADATA = "mirai_chain_of_experts_topology"
_STRUCTURAL_ADAPTER_EXTENSIONS = (
    (
        _LIGHTWEIGHT_EXPERT_PREFIX,
        _LIGHTWEIGHT_EXPERT_SCHEMA_METADATA,
        _LIGHTWEIGHT_EXPERT_TOPOLOGY_METADATA,
    ),
    (
        _DECOUPLED_ROUTING_PREFIX,
        _DECOUPLED_ROUTING_SCHEMA_METADATA,
        _DECOUPLED_ROUTING_TOPOLOGY_METADATA,
    ),
    (
        _CHAIN_OF_EXPERTS_PREFIX,
        _CHAIN_OF_EXPERTS_SCHEMA_METADATA,
        _CHAIN_OF_EXPERTS_TOPOLOGY_METADATA,
    ),
)


def _portable_adapter_state(adapter_state: dict[str, Any]) -> dict[str, Any]:
    """Expand any compact (sparse) expert-LoRA slices to full shape.

    Portable formats (kohya/diffusers/lycoris) must never carry Mirai's compact
    ``*_selected`` / ``active_expert_*`` keys. This makes the portable export
    faithful regardless of ``adapter.sparse_expert_export`` (no-op passthrough
    when the state is already full-shape).
    """
    from mirai.core.models.adapters.sparse_expert_export import expand_sparse_expert_state

    return expand_sparse_expert_state(adapter_state)


def _to_scalar_tensor(value: Any):
    if torch is None:  # pragma: no cover
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().float().reshape(-1).mean().reshape(())
    return torch.tensor(float(value), dtype=torch.float32)


def _to_float(value: Any, default: float = 0.0) -> float:
    if torch is not None and isinstance(value, torch.Tensor):
        return float(value.detach().float().reshape(-1).mean().item())
    if value is None:
        return float(default)
    return float(value)


def _is_tensor(value: Any) -> bool:
    return bool(torch is not None and isinstance(value, torch.Tensor))


def _base_model_hash(model_path: str, *, model_component: str = "") -> str:
    identity = str(model_path)
    component = str(model_component or "").strip()
    if component:
        identity = f"{identity}#{component}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _multi_lora_prefixes(payload: dict[str, Any]) -> list[str]:
    prefixes: set[str] = set()
    for key in payload.keys():
        if key.endswith(".lora_a"):
            prefixes.add(key[: -len(".lora_a")])
        if key.endswith(".lora_b"):
            prefixes.add(key[: -len(".lora_b")])
        # Diffusers format uses capital A/B
        if key.endswith(".lora_A.weight"):
            prefixes.add(key[: -len(".lora_A.weight")])
        if key.endswith(".lora_B.weight"):
            prefixes.add(key[: -len(".lora_B.weight")])
        if key.endswith(KOHYA_DOWN_SUFFIX):
            prefixes.add(key[: -len(KOHYA_DOWN_SUFFIX)])
        if key.endswith(KOHYA_UP_SUFFIX):
            prefixes.add(key[: -len(KOHYA_UP_SUFFIX)])
        if key.endswith(LIGHTX2V_DOWN_SUFFIX):
            prefixes.add(key[: -len(LIGHTX2V_DOWN_SUFFIX)])
        if key.endswith(LIGHTX2V_UP_SUFFIX):
            prefixes.add(key[: -len(LIGHTX2V_UP_SUFFIX)])
    return sorted(prefixes)


def adapter_state_to_kohya(
    adapter_state: dict[str, Any]
) -> dict[str, Any]:
    prefixes = _multi_lora_prefixes(adapter_state)
    if prefixes:
        out: dict[str, Any] = {}
        for prefix in prefixes:
            a = adapter_state.get(f"{prefix}.lora_a", adapter_state.get(prefix + KOHYA_DOWN_SUFFIX))
            b = adapter_state.get(f"{prefix}.lora_b", adapter_state.get(prefix + KOHYA_UP_SUFFIX))
            alpha = adapter_state.get(f"{prefix}.lora_alpha", adapter_state.get(f"{prefix}.alpha", 1.0))
            magnitude = adapter_state.get(f"{prefix}{DORA_MAGNITUDE_SUFFIX}")
            if a is None or b is None:
                continue
            kohya_prefix = f"lora_unet.{prefix}"
            out[f"{kohya_prefix}{KOHYA_DOWN_SUFFIX}"] = _to_scalar_tensor(a) if not _is_tensor(a) else a.detach().cpu().contiguous()
            out[f"{kohya_prefix}{KOHYA_UP_SUFFIX}"] = _to_scalar_tensor(b) if not _is_tensor(b) else b.detach().cpu().contiguous()
            out[f"{kohya_prefix}.alpha"] = _to_scalar_tensor(alpha).reshape(1).detach().cpu().float() if torch is not None else alpha
            if magnitude is not None:
                out[f"{kohya_prefix}{KOHYA_DORA_SUFFIX}"] = (
                    _to_scalar_tensor(magnitude)
                    if not _is_tensor(magnitude)
                    else magnitude.detach().cpu().float()
                )
        if out:
            return out
    a = _to_scalar_tensor(adapter_state.get("lora_a", 0.0))
    b = _to_scalar_tensor(adapter_state.get("lora_b", 0.0))
    alpha = _to_scalar_tensor(adapter_state.get("lora_alpha", 1.0))
    if torch is None:  # pragma: no cover
        return {KOHYA_A_KEY: a, KOHYA_B_KEY: b, KOHYA_ALPHA_KEY: alpha}
    return {
        KOHYA_A_KEY: a.reshape(1, 1).detach().cpu().float(),
        KOHYA_B_KEY: b.reshape(1, 1).detach().cpu().float(),
        KOHYA_ALPHA_KEY: alpha.reshape(1).detach().cpu().float(),
    }


def adapter_state_to_diffusers(adapter_state: dict[str, Any]) -> dict[str, Any]:
    prefixes = _multi_lora_prefixes(adapter_state)
    if prefixes:
        out: dict[str, Any] = {}
        for prefix in prefixes:
            a = adapter_state.get(f"{prefix}.lora_a", adapter_state.get(prefix + KOHYA_DOWN_SUFFIX))
            b = adapter_state.get(f"{prefix}.lora_b", adapter_state.get(prefix + KOHYA_UP_SUFFIX))
            alpha = adapter_state.get(f"{prefix}.lora_alpha", adapter_state.get(f"{prefix}.alpha", 1.0))
            magnitude = adapter_state.get(f"{prefix}{DORA_MAGNITUDE_SUFFIX}")
            if a is None or b is None:
                continue
            out[f"unet.{prefix}.lora_A.weight"] = _to_scalar_tensor(a) if not _is_tensor(a) else a.detach().cpu().float()
            out[f"unet.{prefix}.lora_B.weight"] = _to_scalar_tensor(b) if not _is_tensor(b) else b.detach().cpu().float()
            out[f"unet.{prefix}.alpha"] = _to_scalar_tensor(alpha).reshape(1).detach().cpu().float() if torch is not None else alpha
            if magnitude is not None:
                out[f"unet.{prefix}{DIFFUSERS_DORA_SUFFIX}"] = (
                    _to_scalar_tensor(magnitude)
                    if not _is_tensor(magnitude)
                    else magnitude.detach().cpu().float()
                )
        if out:
            return out
    a = _to_scalar_tensor(adapter_state.get("lora_a", 0.0))
    b = _to_scalar_tensor(adapter_state.get("lora_b", 0.0))
    alpha = _to_scalar_tensor(adapter_state.get("lora_alpha", 1.0))
    if torch is None:  # pragma: no cover
        return {DIFFUSERS_A_KEY: a, DIFFUSERS_B_KEY: b, DIFFUSERS_ALPHA_KEY: alpha}
    return {
        DIFFUSERS_A_KEY: a.reshape(1, 1).detach().cpu().float(),
        DIFFUSERS_B_KEY: b.reshape(1, 1).detach().cpu().float(),
        DIFFUSERS_ALPHA_KEY: alpha.reshape(1).detach().cpu().float(),
    }


def adapter_state_to_lycoris(
    adapter_state: dict[str, Any],
    *,
    adapter_type: str,
) -> dict[str, Any]:
    a = _to_scalar_tensor(adapter_state.get("lora_a", 0.0))
    b = _to_scalar_tensor(adapter_state.get("lora_b", 0.0))
    alpha = _to_scalar_tensor(adapter_state.get("lora_alpha", 1.0))
    kind = str(adapter_type).strip().lower()
    if kind not in {"loha", "lokr"}:
        raise ValueError(f"Unsupported lycoris adapter_type '{adapter_type}'.")
    out: dict[str, Any] = {}
    if kind == "loha":
        out[LYCORIS_LOHA_A_KEY] = a
        out[LYCORIS_LOHA_B_KEY] = b
        if "loha_a1" in adapter_state and "loha_b1" in adapter_state:
            out[LYCORIS_LOHA_A1_KEY] = _to_scalar_tensor(adapter_state.get("loha_a1"))
            out[LYCORIS_LOHA_B1_KEY] = _to_scalar_tensor(adapter_state.get("loha_b1"))
        if "loha_a2" in adapter_state and "loha_b2" in adapter_state:
            out[LYCORIS_LOHA_A2_KEY] = _to_scalar_tensor(adapter_state.get("loha_a2"))
            out[LYCORIS_LOHA_B2_KEY] = _to_scalar_tensor(adapter_state.get("loha_b2"))
    else:
        out[LYCORIS_LOKR_A_KEY] = a
        out[LYCORIS_LOKR_B_KEY] = b
        if "lokr_a1" in adapter_state and "lokr_a2" in adapter_state:
            out[LYCORIS_LOKR_A1_KEY] = _to_scalar_tensor(adapter_state.get("lokr_a1"))
            out[LYCORIS_LOKR_A2_KEY] = _to_scalar_tensor(adapter_state.get("lokr_a2"))
        if "lokr_b1" in adapter_state and "lokr_b2" in adapter_state:
            out[LYCORIS_LOKR_B1_KEY] = _to_scalar_tensor(adapter_state.get("lokr_b1"))
            out[LYCORIS_LOKR_B2_KEY] = _to_scalar_tensor(adapter_state.get("lokr_b2"))
    out["lycoris.alpha"] = alpha
    if torch is None:  # pragma: no cover
        return out
    tensorized: dict[str, Any] = {}
    for key, value in out.items():
        if key == "lycoris.alpha":
            tensorized[key] = _to_scalar_tensor(value).reshape(1).detach().cpu().float()
        else:
            tensorized[key] = _to_scalar_tensor(value).reshape(1, 1).detach().cpu().float()
    return tensorized


def save_kohya_adapter_safetensors(
    path: str | Path,
    *,
    adapter_state: dict[str, Any],
    rank: int,
    alpha: float,
    target_modules: list[str],
    model_path: str,
    model_component: str = "",
    metadata_extra: dict[str, str] | None = None,
) -> Path:
    from safetensors.torch import save_file

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    portable_state = _portable_adapter_state(adapter_state)
    tensors = adapter_state_to_kohya(portable_state)
    metadata = {
        "format": "kohya",
        "rank": str(int(rank)),
        "alpha": str(float(alpha)),
        "target_modules": ",".join(target_modules),
        "base_model_hash": _base_model_hash(model_path, model_component=model_component),
    }
    if str(model_component or "").strip():
        metadata["base_model_component"] = str(model_component).strip()
    if metadata_extra:
        metadata.update(metadata_extra)
    _append_structural_adapter_safetensors_state(
        tensors, metadata, portable_state
    )
    save_file(tensors, str(out), metadata=metadata)
    return out


def save_diffusers_adapter_safetensors(
    path: str | Path,
    *,
    adapter_state: dict[str, Any],
    rank: int,
    alpha: float,
    target_modules: list[str],
    model_path: str,
    model_component: str = "",
    metadata_extra: dict[str, str] | None = None,
) -> Path:
    from safetensors.torch import save_file

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    portable_state = _portable_adapter_state(adapter_state)
    tensors = adapter_state_to_diffusers(portable_state)
    metadata = {
        "format": "diffusers",
        "rank": str(int(rank)),
        "alpha": str(float(alpha)),
        "target_modules": ",".join(target_modules),
        "base_model_hash": _base_model_hash(model_path, model_component=model_component),
    }
    if str(model_component or "").strip():
        metadata["base_model_component"] = str(model_component).strip()
    if metadata_extra:
        metadata.update(metadata_extra)
    _append_structural_adapter_safetensors_state(
        tensors, metadata, portable_state
    )
    save_file(tensors, str(out), metadata=metadata)
    return out


def save_lycoris_adapter_safetensors(
    path: str | Path,
    *,
    adapter_state: dict[str, Any],
    adapter_type: str,
    rank: int,
    alpha: float,
    target_modules: list[str],
    model_path: str,
    model_component: str = "",
    metadata_extra: dict[str, str] | None = None,
) -> Path:
    from safetensors.torch import save_file

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tensors = adapter_state_to_lycoris(
        _portable_adapter_state(adapter_state), adapter_type=adapter_type
    )
    metadata = {
        "format": "lycoris",
        "lycoris_type": str(adapter_type).strip().lower(),
        "rank": str(int(rank)),
        "alpha": str(float(alpha)),
        "target_modules": ",".join(target_modules),
        "base_model_hash": _base_model_hash(model_path, model_component=model_component),
    }
    if str(model_component or "").strip():
        metadata["base_model_component"] = str(model_component).strip()
    if metadata_extra:
        metadata.update(metadata_extra)
    save_file(tensors, str(out), metadata=metadata)
    return out


def load_adapter_payload(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() == ".safetensors":
        from safetensors import safe_open

        payload: dict[str, Any] = {}
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for key in f.keys():
                payload[key] = f.get_tensor(key)
            md = f.metadata()
            if md:
                payload["_metadata"] = dict(md)
        metadata = payload.get("_metadata")
        if isinstance(metadata, dict) and (
            str(metadata.get("mirai_adapter_transform", "")).strip().lower()
            == "para"
        ):
            from mirai.core.models.adapters.para import expand_para_adapter_payload

            return expand_para_adapter_payload(payload)
        return payload
    return load_checkpoint(p)


def normalize_adapter_state(
    payload: dict[str, Any],
    *,
    lora_format: str = "auto",
) -> dict[str, Any]:
    fmt = lora_format.strip().lower()
    if fmt == "auto" and "adapter_state" in payload and isinstance(payload["adapter_state"], dict):
        return dict(payload["adapter_state"])
    if fmt == "peft" and "adapter_state" in payload and isinstance(payload["adapter_state"], dict):
        return dict(payload["adapter_state"])

    if fmt in {"auto", "peft"} and "lora_a" in payload and "lora_b" in payload:
        out = {"lora_a": _to_scalar_tensor(payload["lora_a"]), "lora_b": _to_scalar_tensor(payload["lora_b"])}
        if "lora_alpha" in payload:
            out["lora_alpha"] = _to_scalar_tensor(payload["lora_alpha"])
        return out

    # Native multi-layer LoRA format.
    prefixes = _multi_lora_prefixes(payload)
    if fmt in {"auto", "peft", "kohya", "diffusers"} and prefixes:
        out: dict[str, Any] = {}
        for prefix in prefixes:
            normalized_prefix = str(prefix)
            if normalized_prefix.startswith("lora_unet."):
                normalized_prefix = normalized_prefix[len("lora_unet.") :]
            if normalized_prefix.startswith("unet."):
                normalized_prefix = normalized_prefix[len("unet.") :]
            a = payload.get(
                f"{prefix}.lora_a",
                payload.get(
                    f"{prefix}.lora_A.weight",
                    payload.get(
                        prefix + KOHYA_DOWN_SUFFIX,
                        payload.get(prefix + LIGHTX2V_DOWN_SUFFIX),
                    ),
                ),
            )
            b = payload.get(
                f"{prefix}.lora_b",
                payload.get(
                    f"{prefix}.lora_B.weight",
                    payload.get(
                        prefix + KOHYA_UP_SUFFIX,
                        payload.get(prefix + LIGHTX2V_UP_SUFFIX),
                    ),
                ),
            )
            alpha = payload.get(
                f"{prefix}.lora_alpha",
                payload.get(f"{prefix}.alpha", payload.get(f"unet.{prefix}.alpha")),
            )
            magnitude = payload.get(
                f"{prefix}{DORA_MAGNITUDE_SUFFIX}",
                payload.get(
                    f"{prefix}{KOHYA_DORA_SUFFIX}",
                    payload.get(f"{prefix}{DIFFUSERS_DORA_SUFFIX}"),
                ),
            )
            if a is None or b is None:
                continue
            out[f"{normalized_prefix}.lora_a"] = _to_scalar_tensor(a) if not _is_tensor(a) else a.detach().cpu().float()
            out[f"{normalized_prefix}.lora_b"] = _to_scalar_tensor(b) if not _is_tensor(b) else b.detach().cpu().float()
            if alpha is not None:
                out[f"{normalized_prefix}.lora_alpha"] = _to_scalar_tensor(alpha)
            if magnitude is not None:
                out[f"{normalized_prefix}{DORA_MAGNITUDE_SUFFIX}"] = (
                    _to_scalar_tensor(magnitude)
                    if not _is_tensor(magnitude)
                    else magnitude.detach().cpu().float()
                )
        if out:
            return _restore_mirai_scaling_metadata(out, payload)

    # Single-layer Kohya payload.
    if fmt in {"auto", "kohya"} and KOHYA_A_KEY in payload and KOHYA_B_KEY in payload:
        out = {
            "lora_a": _to_scalar_tensor(payload[KOHYA_A_KEY]),
            "lora_b": _to_scalar_tensor(payload[KOHYA_B_KEY]),
        }
        if KOHYA_ALPHA_KEY in payload:
            out["lora_alpha"] = _to_scalar_tensor(payload[KOHYA_ALPHA_KEY])
        return out

    # Single-layer Diffusers payload.
    if fmt in {"auto", "diffusers"} and DIFFUSERS_A_KEY in payload and DIFFUSERS_B_KEY in payload:
        out = {
            "lora_a": _to_scalar_tensor(payload[DIFFUSERS_A_KEY]),
            "lora_b": _to_scalar_tensor(payload[DIFFUSERS_B_KEY]),
        }
        if DIFFUSERS_ALPHA_KEY in payload:
            out["lora_alpha"] = _to_scalar_tensor(payload[DIFFUSERS_ALPHA_KEY])
        return out

    if fmt in {"auto", "lycoris"}:
        if (
            LYCORIS_LOHA_A1_KEY in payload
            and LYCORIS_LOHA_B1_KEY in payload
            and LYCORIS_LOHA_A2_KEY in payload
            and LYCORIS_LOHA_B2_KEY in payload
        ):
            a1 = _to_scalar_tensor(payload[LYCORIS_LOHA_A1_KEY])
            b1 = _to_scalar_tensor(payload[LYCORIS_LOHA_B1_KEY])
            a2 = _to_scalar_tensor(payload[LYCORIS_LOHA_A2_KEY])
            b2 = _to_scalar_tensor(payload[LYCORIS_LOHA_B2_KEY])
            out = {
                "loha_a1": a1,
                "loha_b1": b1,
                "loha_a2": a2,
                "loha_b2": b2,
                "lora_a": _to_scalar_tensor(a1 * b1),
                "lora_b": _to_scalar_tensor(a2 * b2),
                "adapter_type": "loha",
            }
            if "lycoris.alpha" in payload:
                out["lora_alpha"] = _to_scalar_tensor(payload["lycoris.alpha"])
            return out
        if (
            LYCORIS_LOKR_A1_KEY in payload
            and LYCORIS_LOKR_A2_KEY in payload
            and LYCORIS_LOKR_B1_KEY in payload
            and LYCORIS_LOKR_B2_KEY in payload
        ):
            a1 = _to_scalar_tensor(payload[LYCORIS_LOKR_A1_KEY])
            a2 = _to_scalar_tensor(payload[LYCORIS_LOKR_A2_KEY])
            b1 = _to_scalar_tensor(payload[LYCORIS_LOKR_B1_KEY])
            b2 = _to_scalar_tensor(payload[LYCORIS_LOKR_B2_KEY])
            out = {
                "lokr_a1": a1,
                "lokr_a2": a2,
                "lokr_b1": b1,
                "lokr_b2": b2,
                "lora_a": _to_scalar_tensor(a1 * a2),
                "lora_b": _to_scalar_tensor(b1 * b2),
                "adapter_type": "lokr",
            }
            if "lycoris.alpha" in payload:
                out["lora_alpha"] = _to_scalar_tensor(payload["lycoris.alpha"])
            return out
        if LYCORIS_LOHA_A_KEY in payload and LYCORIS_LOHA_B_KEY in payload:
            out = {
                "lora_a": _to_scalar_tensor(payload[LYCORIS_LOHA_A_KEY]),
                "lora_b": _to_scalar_tensor(payload[LYCORIS_LOHA_B_KEY]),
                "adapter_type": "loha",
            }
            if "lycoris.alpha" in payload:
                out["lora_alpha"] = _to_scalar_tensor(payload["lycoris.alpha"])
            return out
        if LYCORIS_LOKR_A_KEY in payload and LYCORIS_LOKR_B_KEY in payload:
            out = {
                "lora_a": _to_scalar_tensor(payload[LYCORIS_LOKR_A_KEY]),
                "lora_b": _to_scalar_tensor(payload[LYCORIS_LOKR_B_KEY]),
                "adapter_type": "lokr",
            }
            if "lycoris.alpha" in payload:
                out["lora_alpha"] = _to_scalar_tensor(payload["lycoris.alpha"])
            return out

    raise ValueError(f"Unrecognized adapter checkpoint format for lora_format='{lora_format}'.")


def _restore_mirai_scaling_metadata(
    state: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    metadata = payload.get("_metadata")
    restored = dict(state)
    if isinstance(metadata, dict):
        rule = str(metadata.get("mirai_adapter_scaling_rule", "")).strip()
        if rule and rule != "alpha_over_rank":
            if rule != "alpha_over_sqrt_rank":
                raise ValueError(
                    f"Unsupported Mirai adapter scaling rule {rule!r}."
                )
            for key in tuple(state):
                if key.endswith(".lora_a"):
                    prefix = key[: -len(".lora_a")]
                    restored[
                        f"{prefix}{RSLORA_STATE_SUFFIX}"
                    ] = _to_scalar_tensor(1).reshape(1)
        for prefix, schema_key, topology_key in _STRUCTURAL_ADAPTER_EXTENSIONS:
            schema = metadata.get(schema_key)
            topology = metadata.get(topology_key)
            extension_tensors = {
                str(key): value
                for key, value in payload.items()
                if str(key).startswith(f"{prefix}modules.")
            }
            if schema is None and topology is None and not extension_tensors:
                continue
            if schema is None or topology is None or not extension_tensors:
                raise ValueError(
                    f"Incomplete {prefix.rstrip('.')} safetensors extension."
                )
            try:
                parsed_topology = json.loads(str(topology))
                parsed_schema = int(schema)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid {prefix.rstrip('.')} safetensors metadata."
                ) from exc
            restored[f"{prefix}schema_version"] = parsed_schema
            restored[f"{prefix}topology"] = parsed_topology
            restored.update(extension_tensors)
    return restored


def _append_structural_adapter_safetensors_state(
    tensors: dict[str, Any],
    metadata: dict[str, str],
    adapter_state: dict[str, Any],
) -> None:
    for prefix, schema_metadata, topology_metadata in (
        _STRUCTURAL_ADAPTER_EXTENSIONS
    ):
        schema_key = f"{prefix}schema_version"
        topology_key = f"{prefix}topology"
        extension_tensors = {
            str(key): value.detach().cpu().contiguous()
            for key, value in adapter_state.items()
            if str(key).startswith(f"{prefix}modules.") and _is_tensor(value)
        }
        schema = adapter_state.get(schema_key)
        topology = adapter_state.get(topology_key)
        if schema is None and topology is None and not extension_tensors:
            continue
        if schema is None or topology is None or not extension_tensors:
            raise ValueError(
                f"Incomplete {prefix.rstrip('.')} adapter state."
            )
        tensors.update(extension_tensors)
        metadata[schema_metadata] = str(int(schema))
        metadata[topology_metadata] = json.dumps(
            topology,
            sort_keys=True,
            separators=(",", ":"),
        )


def adapter_metadata_from_state(
    *,
    adapter_state: dict[str, Any],
    model_path: str,
    target_modules: list[str],
    model_component: str = "",
) -> dict[str, str]:
    # Compact (sparse) states carry ``*_selected`` keys; expand so the derived
    # rank/layer-count metadata matches the full-shape tensors that are written.
    adapter_state = _portable_adapter_state(adapter_state)
    prefixes = _multi_lora_prefixes(adapter_state)
    if prefixes:
        alphas: list[float] = []
        for prefix in prefixes:
            alpha = adapter_state.get(f"{prefix}.lora_alpha", 1.0)
            alphas.append(_to_float(alpha, default=1.0))
        return {
            "rank": "multi",
            "alpha": str(sum(alphas) / max(1, len(alphas))),
            "target_modules": ",".join(target_modules),
            "base_model_hash": _base_model_hash(model_path, model_component=model_component),
            "num_lora_layers": str(len(prefixes)),
            **(
                {"base_model_component": str(model_component).strip()}
                if str(model_component or "").strip()
                else {}
            ),
        }
    return {
        "rank": "1",
        "alpha": str(_to_float(adapter_state.get("lora_alpha"), default=1.0)),
        "target_modules": ",".join(target_modules),
        "base_model_hash": _base_model_hash(model_path, model_component=model_component),
        **(
            {"base_model_component": str(model_component).strip()}
            if str(model_component or "").strip()
            else {}
        ),
    }


def merge_adapter_into_base_scale(
    *,
    adapter_state: dict[str, Any],
    base_scale: float = 1.0,
) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        merged = float(base_scale)
    else:
        a = _to_scalar_tensor(adapter_state.get("lora_a", 0.0))
        b = _to_scalar_tensor(adapter_state.get("lora_b", 0.0))
        alpha = _to_scalar_tensor(adapter_state.get("lora_alpha", 1.0))
        merged = float(base_scale) + float((a * b * alpha).item())
    if torch is None:  # pragma: no cover
        return {"base_scale": merged}
    return {"base_scale": torch.tensor(float(merged), dtype=torch.float32)}
