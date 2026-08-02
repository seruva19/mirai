"""Compact sparse-expert adapter serialization and reconstruction policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SPARSE_A_SUFFIX = ".lora_a_selected"
SPARSE_B_SUFFIX = ".lora_b_selected"
SPARSE_IDS_SUFFIX = ".active_expert_ids"
MASK_SUFFIX = ".active_expert_mask"


def selection_is_active(module: Any) -> bool:
    return bool(getattr(module, "_expert_selection_active", False)) and hasattr(
        module, "active_expert_mask"
    )


@dataclass(frozen=True)
class SparseExpertExportPolicy:
    enabled: bool = False

    def write_module_state(
        self, state: dict[str, Any], *, name: str, module: Any
    ) -> bool:
        if not self.enabled or not selection_is_active(module):
            return False
        ids = torch.nonzero(
            module.active_expert_mask > 0.0, as_tuple=False
        ).reshape(-1).to(torch.long)
        state[f"{name}{SPARSE_A_SUFFIX}"] = module.lora_a.detach().index_select(
            0, ids.to(module.lora_a.device)
        )
        state[f"{name}{SPARSE_B_SUFFIX}"] = module.lora_b.detach().index_select(
            0, ids.to(module.lora_b.device)
        )
        state[f"{name}{SPARSE_IDS_SUFFIX}"] = ids.detach().cpu()
        state[f"{name}{MASK_SUFFIX}"] = module.active_expert_mask.detach()
        state[f"{name}.lora_alpha"] = module.lora_alpha.detach().reshape(1)
        return True


def load_compact_expert_adapter(
    module: Any, name: str, state: dict[str, Any]
) -> None:
    a_sel = _detach_float(state[f"{name}{SPARSE_A_SUFFIX}"])
    b_sel = _detach_float(state[f"{name}{SPARSE_B_SUFFIX}"])
    ids = _detach_long(state[f"{name}{SPARSE_IDS_SUFFIX}"])
    if not hasattr(module, "active_expert_mask"):
        raise ValueError(
            f"Compact expert adapter '{name}' loaded into a module without an "
            "active_expert_mask buffer."
        )
    mask = _adapter_tensor(
        state[f"{name}{MASK_SUFFIX}"],
        like=module.active_expert_mask,
        name=f"{name}{MASK_SUFFIX}",
    ).reshape_as(module.active_expert_mask)
    num_experts = int(module.lora_a.shape[0])
    if int(mask.numel()) != num_experts:
        raise ValueError(
            f"Compact expert adapter '{name}' mask length {int(mask.numel())} "
            f"does not match module num_experts {num_experts}."
        )
    if int(ids.numel()) != int(a_sel.shape[0]) or int(ids.numel()) != int(
        b_sel.shape[0]
    ):
        raise ValueError(
            f"Compact expert adapter '{name}' selected-id count {int(ids.numel())} "
            "does not match stored slice count."
        )
    src_ids = ids.to(device=module.lora_a.device)
    module.lora_a.index_copy_(
        0, src_ids, a_sel.to(device=module.lora_a.device, dtype=module.lora_a.dtype)
    )
    rebuilt_b = torch.zeros_like(module.lora_b)
    rebuilt_b.index_copy_(
        0,
        src_ids.to(device=module.lora_b.device),
        b_sel.to(device=module.lora_b.device, dtype=module.lora_b.dtype),
    )
    module.lora_b.copy_(rebuilt_b)
    module.active_expert_mask.copy_(mask.to(device=module.active_expert_mask.device))
    alpha_key = f"{name}.lora_alpha"
    if alpha_key in state:
        module.lora_alpha.copy_(
            _adapter_tensor(
                state[alpha_key], like=module.lora_alpha, name=alpha_key
            ).reshape_as(module.lora_alpha)
        )
    if hasattr(module, "_expert_selection_active"):
        module._expert_selection_active = bool(
            (module.active_expert_mask < 1.0).any().item()
        )


def expand_sparse_expert_state(state: dict[str, Any]) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("expand_sparse_expert_state requires torch.")
    compact_prefixes = {
        key[: -len(SPARSE_A_SUFFIX)]
        for key in state
        if key.endswith(SPARSE_A_SUFFIX)
    }
    if not compact_prefixes:
        return dict(state)
    drop_suffixes = (SPARSE_A_SUFFIX, SPARSE_B_SUFFIX, SPARSE_IDS_SUFFIX, MASK_SUFFIX)
    out: dict[str, Any] = {}
    for key, value in state.items():
        prefix = key.rsplit(".", 1)[0]
        if prefix in compact_prefixes and any(
            key.endswith(suffix) for suffix in drop_suffixes
        ):
            continue
        out[key] = value
    for prefix in compact_prefixes:
        a_sel = _detach_float(state[f"{prefix}{SPARSE_A_SUFFIX}"])
        b_sel = _detach_float(state[f"{prefix}{SPARSE_B_SUFFIX}"])
        ids = _detach_long(state[f"{prefix}{SPARSE_IDS_SUFFIX}"])
        mask = _detach_float(state[f"{prefix}{MASK_SUFFIX}"])
        num_experts = int(mask.numel())
        full_a = torch.zeros(
            (num_experts, int(a_sel.shape[1]), int(a_sel.shape[2])), dtype=a_sel.dtype
        )
        full_b = torch.zeros(
            (num_experts, int(b_sel.shape[1]), int(b_sel.shape[2])), dtype=b_sel.dtype
        )
        full_a.index_copy_(0, ids, a_sel)
        full_b.index_copy_(0, ids, b_sel)
        out[f"{prefix}.lora_a"] = full_a
        out[f"{prefix}.lora_b"] = full_b
    return out


def _adapter_tensor(value: Any, *, like: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=like.dtype)
    tensor = value.detach().to(device=like.device, dtype=like.dtype)
    if tuple(tensor.shape) != tuple(like.shape):
        if int(tensor.numel()) != int(like.numel()):
            raise ValueError(
                f"Adapter tensor '{name}' has shape {tuple(tensor.shape)}, "
                f"expected {tuple(like.shape)}."
            )
        tensor = tensor.reshape_as(like)
    return tensor


def _detach_float(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    return value.detach().cpu().float()


def _detach_long(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    return value.detach().cpu().to(torch.long)
