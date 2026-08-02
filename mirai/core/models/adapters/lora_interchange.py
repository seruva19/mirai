"""Lossless fused (stacked) <-> unfused (per-expert) LoRA state-dict interchange.

Mirai trains routed-expert LoRA on grouped/fused expert tensors, so its native
adapter state stacks every expert into one tensor: ``{name}.lora_a`` is
``[E, rank, in]`` and ``{name}.lora_b`` is ``[E, out, rank]`` (the
``ActiveExpertLoRA`` / ``LoRAExpertTensorParametrization`` layout). External
ecosystems (PEFT, diffusers, ComfyUI, llama.cpp, vLLM) expect **unfused**
per-expert modules with PEFT-style ``lora_A.weight`` / ``lora_B.weight`` keys.

This module is a pure key/axis converter -- ``torch.stack`` / ``torch.unbind``
plus key renaming, no LoRA math and no dtype casting -- so a
``fuse(unfuse(x)) == x`` round-trip is bit-identical. It also composes with the
MoE-Sieve compact sparse export (``*.lora_a_selected`` + ``active_expert_ids``):
missing experts stay representable and round-trip exactly.

Unfused (external) convention implemented -- PEFT / diffusers naming:

    {name}.experts.{e}.lora_A.weight    <- {name}.lora_a[e]      # [rank, in]
    {name}.experts.{e}.lora_B.weight    <- {name}.lora_b[e]      # [out, rank]
    {name}.experts.{e}.alpha            <- {name}.lora_alpha     # replicated
    {name}.lora_A.weight                <- {name}.lora_a         # non-expert 2D
    {name}.lora_B.weight                <- {name}.lora_b
    {name}.alpha                        <- {name}.lora_alpha

Mirai-namespaced metadata keys pass through verbatim (external loaders ignore
unknown keys); they carry the info needed for a lossless reverse:

    {name}.active_expert_mask   (sparse selection / full+mask marker)
    {name}.active_expert_ids    (compact sparse marker; present => compact form)
    {name}.condenser_a/_b/_alpha  (shared condenser term -- unfusable, passthrough)

This matches the woct0rdho qwen3-moe-fused ``convert_lora`` convention (per-expert
``experts.{e}...lora_A.weight`` on the unfused side); Mirai uses the adapter's
own ``ExpertTensorSpec`` name as the module prefix because that name already
encodes the module path.
"""

from __future__ import annotations

import re
from typing import Any

from mirai.core.models.adapters.expert_condenser import CONDENSER_A_SUFFIX
from mirai.core.models.adapters.expert_condenser import CONDENSER_ALPHA_SUFFIX
from mirai.core.models.adapters.expert_condenser import CONDENSER_B_SUFFIX
from mirai.core.models.adapters.dora import DORA_MAGNITUDE_SUFFIX
from mirai.core.models.adapters.sparse_expert_export import MASK_SUFFIX
from mirai.core.models.adapters.sparse_expert_export import SPARSE_A_SUFFIX
from mirai.core.models.adapters.sparse_expert_export import SPARSE_B_SUFFIX
from mirai.core.models.adapters.sparse_expert_export import SPARSE_IDS_SUFFIX

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# --- fused (Mirai-native) suffixes -----------------------------------------
FUSED_A_SUFFIX = ".lora_a"
FUSED_B_SUFFIX = ".lora_b"
FUSED_ALPHA_SUFFIX = ".lora_alpha"

# --- unfused (PEFT/diffusers) suffixes -------------------------------------
UNFUSED_A_SUFFIX = ".lora_A.weight"
UNFUSED_B_SUFFIX = ".lora_B.weight"
UNFUSED_ALPHA_SUFFIX = ".alpha"
UNFUSED_DORA_SUFFIX = ".lora_magnitude_vector.weight"
EXPERT_INFIX = ".experts."

# Keys preserved verbatim in either direction (Mirai-internal metadata).
_PASSTHROUGH_SUFFIXES = (
    MASK_SUFFIX,
    SPARSE_IDS_SUFFIX,
    CONDENSER_A_SUFFIX,
    CONDENSER_B_SUFFIX,
    CONDENSER_ALPHA_SUFFIX,
)

_EXPERT_A_RE = re.compile(r"^(?P<name>.+)\.experts\.(?P<idx>\d+)\.lora_A\.weight$")
_EXPERT_B_RE = re.compile(r"^(?P<name>.+)\.experts\.(?P<idx>\d+)\.lora_B\.weight$")
_EXPERT_ALPHA_RE = re.compile(r"^(?P<name>.+)\.experts\.(?P<idx>\d+)\.alpha$")
_EXPERT_DORA_RE = re.compile(
    r"^(?P<name>.+)\.experts\.(?P<idx>\d+)\.lora_magnitude_vector\.weight$"
)


class LoRAInterchangeError(ValueError):
    """Raised on malformed or ambiguous LoRA interchange input."""


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("LoRA interchange requires torch.")


def _clone(value: Any) -> Any:
    """Detach + contiguous copy, preserving dtype exactly (no math, no cast)."""
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone().contiguous()
    return value


def _tensor_equal(lhs: Any, rhs: Any) -> bool:
    if torch is not None and isinstance(lhs, torch.Tensor):
        return bool(torch.equal(lhs, rhs))
    return bool(lhs == rhs)


# ---------------------------------------------------------------------------
# Direction detection
# ---------------------------------------------------------------------------
def detect_layout(state: dict[str, Any]) -> str:
    """Return ``"fused"`` or ``"unfused"`` from the primary LoRA key casing.

    Fused keys use lowercase ``.lora_a`` / ``.lora_b`` (or the compact
    ``*_selected`` variants); unfused keys use PEFT ``.lora_A.weight`` /
    ``.lora_B.weight``. Raises when both or neither are present.
    """
    fused = any(
        key.endswith(FUSED_A_SUFFIX)
        or key.endswith(FUSED_B_SUFFIX)
        or key.endswith(SPARSE_A_SUFFIX)
        or key.endswith(SPARSE_B_SUFFIX)
        for key in state
    )
    unfused = any(
        key.endswith(UNFUSED_A_SUFFIX) or key.endswith(UNFUSED_B_SUFFIX)
        for key in state
    )
    if fused and unfused:
        raise LoRAInterchangeError(
            "Ambiguous LoRA state: mixes fused (.lora_a) and unfused "
            "(.lora_A.weight) keys."
        )
    if fused:
        return "fused"
    if unfused:
        return "unfused"
    raise LoRAInterchangeError(
        "Unrecognized LoRA state: no fused (.lora_a/.lora_b) or unfused "
        "(.lora_A.weight/.lora_B.weight) keys found."
    )


# ---------------------------------------------------------------------------
# Fused -> unfused
# ---------------------------------------------------------------------------
def _group_fused(state: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    groups: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    # Longest-suffix-first so ``.lora_a_selected`` beats ``.lora_a`` etc.
    ordered = (
        (SPARSE_A_SUFFIX, "a_sel"),
        (SPARSE_B_SUFFIX, "b_sel"),
        (SPARSE_IDS_SUFFIX, "ids"),
        (MASK_SUFFIX, "mask"),
        (CONDENSER_ALPHA_SUFFIX, "cond_alpha"),
        (CONDENSER_A_SUFFIX, "cond_a"),
        (CONDENSER_B_SUFFIX, "cond_b"),
        (DORA_MAGNITUDE_SUFFIX, "dora"),
        (FUSED_ALPHA_SUFFIX, "alpha"),
        (FUSED_A_SUFFIX, "a"),
        (FUSED_B_SUFFIX, "b"),
    )
    for key, value in state.items():
        for suffix, part in ordered:
            if key.endswith(suffix):
                name = key[: -len(suffix)]
                groups.setdefault(name, {})[part] = value
                break
        else:
            unknown.append(key)
    return groups, unknown


def unfuse_lora_state_dict(
    fused: dict[str, Any], *, strict: bool = True
) -> dict[str, Any]:
    """Convert a fused (stacked) Mirai LoRA state dict to unfused per-expert keys.

    Round-trips bit-identically through :func:`fuse_lora_state_dict`. ``strict``
    (default) fails fast on unknown keys and inconsistent shapes.
    """
    _require_torch()
    if not isinstance(fused, dict):
        raise LoRAInterchangeError("Fused LoRA state must be a dict.")
    groups, unknown = _group_fused(fused)
    if unknown and strict:
        raise LoRAInterchangeError(
            "Fused LoRA state contains unknown keys: " + ", ".join(sorted(unknown))
        )
    out: dict[str, Any] = {}
    for name, parts in sorted(groups.items()):
        _emit_unfused_group(out, name, parts)
    return out


def _emit_condenser_passthrough(out: dict[str, Any], name: str, parts: dict[str, Any]) -> None:
    for part, suffix in (
        ("cond_a", CONDENSER_A_SUFFIX),
        ("cond_b", CONDENSER_B_SUFFIX),
        ("cond_alpha", CONDENSER_ALPHA_SUFFIX),
    ):
        if part in parts:
            out[f"{name}{suffix}"] = _clone(parts[part])


def _emit_unfused_group(out: dict[str, Any], name: str, parts: dict[str, Any]) -> None:
    has_full = "a" in parts
    has_compact = "a_sel" in parts
    if has_full and has_compact:
        raise LoRAInterchangeError(
            f"Adapter '{name}' has both fused ('.lora_a') and compact "
            "('.lora_a_selected') tensors."
        )
    alpha = parts.get("alpha")
    if has_compact:
        _emit_unfused_compact(out, name, parts, alpha)
    elif has_full:
        a = parts["a"]
        if int(a.dim()) == 3:
            _emit_unfused_full_experts(out, name, parts, alpha)
        elif int(a.dim()) == 2:
            _emit_unfused_linear(out, name, parts, alpha)
        else:
            raise LoRAInterchangeError(
                f"Adapter '{name}' .lora_a has unsupported ndim {int(a.dim())}."
            )
    else:
        raise LoRAInterchangeError(f"Adapter '{name}' has no LoRA A/B tensors.")
    _emit_condenser_passthrough(out, name, parts)


def _emit_unfused_linear(
    out: dict[str, Any], name: str, parts: dict[str, Any], alpha: Any
) -> None:
    a = parts["a"]
    b = parts.get("b")
    if b is None:
        raise LoRAInterchangeError(f"Adapter '{name}' is missing '.lora_b'.")
    if int(a.shape[0]) != int(b.shape[1]):
        raise LoRAInterchangeError(
            f"Adapter '{name}' rank mismatch: lora_a rows {int(a.shape[0])} != "
            f"lora_b cols {int(b.shape[1])}."
        )
    out[f"{name}{UNFUSED_A_SUFFIX}"] = _clone(a)
    out[f"{name}{UNFUSED_B_SUFFIX}"] = _clone(b)
    if alpha is not None:
        out[f"{name}{UNFUSED_ALPHA_SUFFIX}"] = _clone(alpha)
    if "dora" in parts:
        out[f"{name}{UNFUSED_DORA_SUFFIX}"] = _clone(parts["dora"])


def _emit_unfused_full_experts(
    out: dict[str, Any], name: str, parts: dict[str, Any], alpha: Any
) -> None:
    a = parts["a"]
    b = parts.get("b")
    if b is None:
        raise LoRAInterchangeError(f"Adapter '{name}' is missing '.lora_b'.")
    _validate_stacked_shapes(name, a, b)
    experts = int(a.shape[0])
    dora = parts.get("dora")
    if dora is not None and (
        int(dora.ndim) != 2 or int(dora.shape[0]) != experts
    ):
        raise LoRAInterchangeError(
            f"Adapter '{name}' DoRA magnitude must have shape "
            f"[{experts}, out_features]."
        )
    for e in range(experts):
        _write_expert(
            out,
            name,
            e,
            a[e],
            b[e],
            alpha,
            None if dora is None else dora[e],
        )
    if "mask" in parts:
        out[f"{name}{MASK_SUFFIX}"] = _clone(parts["mask"])


def _emit_unfused_compact(
    out: dict[str, Any], name: str, parts: dict[str, Any], alpha: Any
) -> None:
    a = parts["a_sel"]
    b = parts.get("b_sel")
    ids = parts.get("ids")
    mask = parts.get("mask")
    if b is None or ids is None or mask is None:
        raise LoRAInterchangeError(
            f"Compact adapter '{name}' requires '.lora_a_selected', "
            "'.lora_b_selected', '.active_expert_ids', and '.active_expert_mask'."
        )
    _validate_stacked_shapes(name, a, b)
    if "dora" in parts:
        raise LoRAInterchangeError(
            f"Compact adapter '{name}' cannot carry DoRA magnitude state."
        )
    selected = int(a.shape[0])
    if int(ids.numel()) != selected:
        raise LoRAInterchangeError(
            f"Compact adapter '{name}' id count {int(ids.numel())} != selected "
            f"slice count {selected}."
        )
    num_experts = int(mask.numel())
    id_list = [int(i) for i in ids.reshape(-1).tolist()]
    for slot, e in enumerate(id_list):
        if e < 0 or e >= num_experts:
            raise LoRAInterchangeError(
                f"Compact adapter '{name}' expert id {e} out of range "
                f"[0, {num_experts})."
            )
        _write_expert(out, name, e, a[slot], b[slot], alpha, None)
    # Markers that make the reverse produce the compact form bit-identically.
    out[f"{name}{MASK_SUFFIX}"] = _clone(mask)
    out[f"{name}{SPARSE_IDS_SUFFIX}"] = _clone(ids)


def _write_expert(
    out: dict[str, Any],
    name: str,
    expert: int,
    a_row: Any,
    b_row: Any,
    alpha: Any,
    dora_magnitude: Any,
) -> None:
    prefix = f"{name}{EXPERT_INFIX}{expert}"
    out[f"{prefix}{UNFUSED_A_SUFFIX}"] = _clone(a_row)
    out[f"{prefix}{UNFUSED_B_SUFFIX}"] = _clone(b_row)
    if alpha is not None:
        out[f"{prefix}{UNFUSED_ALPHA_SUFFIX}"] = _clone(alpha)
    if dora_magnitude is not None:
        out[f"{prefix}{UNFUSED_DORA_SUFFIX}"] = _clone(dora_magnitude)


def _validate_stacked_shapes(name: str, a: Any, b: Any) -> None:
    if int(a.dim()) != 3 or int(b.dim()) != 3:
        raise LoRAInterchangeError(
            f"Adapter '{name}' stacked tensors must be rank-3; got "
            f"lora_a ndim {int(a.dim())}, lora_b ndim {int(b.dim())}."
        )
    if int(a.shape[0]) != int(b.shape[0]):
        raise LoRAInterchangeError(
            f"Adapter '{name}' expert-count mismatch: lora_a {int(a.shape[0])} "
            f"!= lora_b {int(b.shape[0])}."
        )
    if int(a.shape[1]) != int(b.shape[2]):
        raise LoRAInterchangeError(
            f"Adapter '{name}' rank mismatch: lora_a rank {int(a.shape[1])} != "
            f"lora_b rank {int(b.shape[2])}."
        )


# ---------------------------------------------------------------------------
# Unfused -> fused
# ---------------------------------------------------------------------------
def _group_unfused(
    state: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    groups: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    for key, value in state.items():
        m = _EXPERT_A_RE.match(key)
        if m:
            _bucket_expert(groups, m, value, "A")
            continue
        m = _EXPERT_B_RE.match(key)
        if m:
            _bucket_expert(groups, m, value, "B")
            continue
        m = _EXPERT_ALPHA_RE.match(key)
        if m:
            _bucket_expert(groups, m, value, "alpha")
            continue
        m = _EXPERT_DORA_RE.match(key)
        if m:
            _bucket_expert(groups, m, value, "dora")
            continue
        matched = False
        for suffix, part in (
            (MASK_SUFFIX, "mask"),
            (SPARSE_IDS_SUFFIX, "ids"),
            (CONDENSER_ALPHA_SUFFIX, "cond_alpha"),
            (CONDENSER_A_SUFFIX, "cond_a"),
            (CONDENSER_B_SUFFIX, "cond_b"),
            (UNFUSED_DORA_SUFFIX, "dora"),
            (UNFUSED_ALPHA_SUFFIX, "alpha"),
            (UNFUSED_A_SUFFIX, "A"),
            (UNFUSED_B_SUFFIX, "B"),
        ):
            if key.endswith(suffix):
                name = key[: -len(suffix)]
                groups.setdefault(name, {})[part] = value
                matched = True
                break
        if not matched:
            unknown.append(key)
    return groups, unknown


def _bucket_expert(
    groups: dict[str, dict[str, Any]], match: re.Match[str], value: Any, kind: str
) -> None:
    name = match.group("name")
    idx = int(match.group("idx"))
    experts = groups.setdefault(name, {}).setdefault("experts", {})
    experts.setdefault(idx, {})[kind] = value


def fuse_lora_state_dict(
    unfused: dict[str, Any], *, strict: bool = True
) -> dict[str, Any]:
    """Convert an unfused per-expert LoRA state dict back to fused (stacked) form.

    Inverse of :func:`unfuse_lora_state_dict`. ``strict`` (default) fails fast on
    unknown keys, missing experts, and inconsistent shapes.
    """
    _require_torch()
    if not isinstance(unfused, dict):
        raise LoRAInterchangeError("Unfused LoRA state must be a dict.")
    groups, unknown = _group_unfused(unfused)
    if unknown and strict:
        raise LoRAInterchangeError(
            "Unfused LoRA state contains unknown keys: " + ", ".join(sorted(unknown))
        )
    out: dict[str, Any] = {}
    for name, parts in sorted(groups.items()):
        _emit_fused_group(out, name, parts)
    return out


def _emit_fused_group(out: dict[str, Any], name: str, parts: dict[str, Any]) -> None:
    experts = parts.get("experts")
    if experts:
        _emit_fused_experts(out, name, parts, experts)
    elif "A" in parts:
        _emit_fused_linear(out, name, parts)
    elif not _has_only_metadata(parts):
        raise LoRAInterchangeError(f"Adapter '{name}' has no LoRA A/B tensors.")
    _fuse_condenser_passthrough(out, name, parts)


def _has_only_metadata(parts: dict[str, Any]) -> bool:
    return bool(parts) and all(
        key in {"mask", "ids", "cond_a", "cond_b", "cond_alpha"} for key in parts
    )


def _emit_fused_linear(out: dict[str, Any], name: str, parts: dict[str, Any]) -> None:
    a = parts["A"]
    b = parts.get("B")
    if b is None:
        raise LoRAInterchangeError(f"Adapter '{name}' is missing '.lora_B.weight'.")
    if int(a.shape[0]) != int(b.shape[1]):
        raise LoRAInterchangeError(
            f"Adapter '{name}' rank mismatch: lora_A rows {int(a.shape[0])} != "
            f"lora_B cols {int(b.shape[1])}."
        )
    out[f"{name}{FUSED_A_SUFFIX}"] = _clone(a)
    out[f"{name}{FUSED_B_SUFFIX}"] = _clone(b)
    if "alpha" in parts:
        out[f"{name}{FUSED_ALPHA_SUFFIX}"] = _clone(parts["alpha"])
    if "dora" in parts:
        out[f"{name}{DORA_MAGNITUDE_SUFFIX}"] = _clone(parts["dora"])


def _collect_expert_rows(
    name: str, experts: dict[int, dict[str, Any]], order: list[int]
) -> tuple[list[Any], list[Any], Any]:
    a_rows: list[Any] = []
    b_rows: list[Any] = []
    alpha: Any = None
    ref_shape_a: tuple[int, ...] | None = None
    ref_shape_b: tuple[int, ...] | None = None
    for e in order:
        entry = experts[e]
        if "A" not in entry or "B" not in entry:
            raise LoRAInterchangeError(
                f"Adapter '{name}' expert {e} is missing an A or B tensor."
            )
        a_e = entry["A"]
        b_e = entry["B"]
        if int(a_e.shape[0]) != int(b_e.shape[1]):
            raise LoRAInterchangeError(
                f"Adapter '{name}' expert {e} rank mismatch: "
                f"A rows {int(a_e.shape[0])} != B cols {int(b_e.shape[1])}."
            )
        if ref_shape_a is None:
            ref_shape_a = tuple(a_e.shape)
            ref_shape_b = tuple(b_e.shape)
        elif tuple(a_e.shape) != ref_shape_a or tuple(b_e.shape) != ref_shape_b:
            raise LoRAInterchangeError(
                f"Adapter '{name}' expert {e} shape "
                f"{tuple(a_e.shape)}/{tuple(b_e.shape)} differs from "
                f"{ref_shape_a}/{ref_shape_b}."
            )
        a_rows.append(a_e)
        b_rows.append(b_e)
        entry_alpha = entry.get("alpha")
        if entry_alpha is not None:
            if alpha is None:
                alpha = entry_alpha
            elif not _tensor_equal(alpha, entry_alpha):
                raise LoRAInterchangeError(
                    f"Adapter '{name}' expert {e} alpha differs from its peers; "
                    "a fused adapter shares one alpha across experts."
                )
    return a_rows, b_rows, alpha


def _emit_fused_experts(
    out: dict[str, Any],
    name: str,
    parts: dict[str, Any],
    experts: dict[int, dict[str, Any]],
) -> None:
    ids = parts.get("ids")
    mask = parts.get("mask")
    is_compact = ids is not None
    if is_compact:
        order = [int(i) for i in ids.reshape(-1).tolist()]
        if sorted(order) != sorted(experts.keys()):
            raise LoRAInterchangeError(
                f"Compact adapter '{name}' active_expert_ids {order} do not match "
                f"present experts {sorted(experts.keys())}."
            )
    else:
        order = sorted(experts.keys())
        expected = list(range(len(order)))
        if order != expected:
            raise LoRAInterchangeError(
                f"Full adapter '{name}' expert indices {order} are not the "
                f"contiguous range {expected}; a compact/sparse adapter must carry "
                "'.active_expert_ids'."
            )
    a_rows, b_rows, alpha = _collect_expert_rows(name, experts, order)
    a_stacked = torch.stack([_clone(row) for row in a_rows], dim=0)
    b_stacked = torch.stack([_clone(row) for row in b_rows], dim=0)
    if is_compact:
        out[f"{name}{SPARSE_A_SUFFIX}"] = a_stacked
        out[f"{name}{SPARSE_B_SUFFIX}"] = b_stacked
        out[f"{name}{SPARSE_IDS_SUFFIX}"] = _clone(ids)
        out[f"{name}{MASK_SUFFIX}"] = _clone(mask)
    else:
        out[f"{name}{FUSED_A_SUFFIX}"] = a_stacked
        out[f"{name}{FUSED_B_SUFFIX}"] = b_stacked
        if mask is not None:
            out[f"{name}{MASK_SUFFIX}"] = _clone(mask)
    if alpha is not None:
        out[f"{name}{FUSED_ALPHA_SUFFIX}"] = _clone(alpha)
    dora_rows = [experts[e].get("dora") for e in order]
    if any(value is not None for value in dora_rows):
        if any(value is None for value in dora_rows):
            raise LoRAInterchangeError(
                f"Adapter '{name}' has incomplete per-expert DoRA magnitude state."
            )
        reference_shape = tuple(dora_rows[0].shape)
        if any(tuple(value.shape) != reference_shape for value in dora_rows[1:]):
            raise LoRAInterchangeError(
                f"Adapter '{name}' has inconsistent DoRA magnitude shapes."
            )
        out[f"{name}{DORA_MAGNITUDE_SUFFIX}"] = torch.stack(
            [_clone(value) for value in dora_rows],
            dim=0,
        )


def _fuse_condenser_passthrough(
    out: dict[str, Any], name: str, parts: dict[str, Any]
) -> None:
    for part, suffix in (
        ("cond_a", CONDENSER_A_SUFFIX),
        ("cond_b", CONDENSER_B_SUFFIX),
        ("cond_alpha", CONDENSER_ALPHA_SUFFIX),
    ):
        if part in parts:
            out[f"{name}{suffix}"] = _clone(parts[part])


def convert_lora_state_dict(
    state: dict[str, Any], *, direction: str = "auto", strict: bool = True
) -> tuple[dict[str, Any], str]:
    """Convert ``state`` in ``direction`` ('auto'|'fuse'|'unfuse').

    Returns ``(converted, applied_direction)`` where ``applied_direction`` is the
    concrete 'fuse' or 'unfuse' that ran.
    """
    resolved = direction.strip().lower()
    if resolved == "auto":
        layout = detect_layout(state)
        resolved = "unfuse" if layout == "fused" else "fuse"
    if resolved == "unfuse":
        return unfuse_lora_state_dict(state, strict=strict), "unfuse"
    if resolved == "fuse":
        return fuse_lora_state_dict(state, strict=strict), "fuse"
    raise LoRAInterchangeError(
        f"Unknown direction '{direction}'; expected auto|fuse|unfuse."
    )
