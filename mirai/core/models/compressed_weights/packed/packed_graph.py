"""Module-graph mutations used while restoring compressed packed states."""

from __future__ import annotations

import torch
import torch.nn as nn


def assign_packed_state_tensor(
    root: nn.Module,
    key: str,
    value: torch.Tensor,
    *,
    expert_axis_size: int | None = None,
) -> None:
    if "." not in key:
        module = root
        tensor_name = key
    else:
        module_name, tensor_name = key.rsplit(".", 1)
        module = root.get_submodule(module_name)
    if tensor_name in module._parameters:
        current = module._parameters[tensor_name]
        if current is not None and tuple(current.shape) != tuple(value.shape):
            if not _can_resize_expert_axis(
                module,
                current,
                value,
                expert_axis_size=expert_axis_size,
            ):
                raise ValueError(
                    f"compressed_weights packed residual tensor {key!r} has shape "
                    f"{tuple(value.shape)}, expected {tuple(current.shape)}."
                )
            module.num_experts = int(expert_axis_size)
        requires_grad = bool(getattr(current, "requires_grad", True))
        module._parameters[tensor_name] = nn.Parameter(
            value.detach().contiguous(), requires_grad=requires_grad
        )
        return
    if tensor_name in module._buffers:
        current = module._buffers[tensor_name]
        if current is not None and tuple(current.shape) != tuple(value.shape):
            if not _can_resize_expert_axis(
                module,
                current,
                value,
                expert_axis_size=expert_axis_size,
            ):
                raise ValueError(
                    f"compressed_weights packed residual buffer {key!r} has shape "
                    f"{tuple(value.shape)}, expected {tuple(current.shape)}."
                )
            module.num_experts = int(expert_axis_size)
        module._buffers[tensor_name] = value.detach().contiguous()
        return
    raise KeyError(f"Target module has no residual tensor {key!r}.")


def _can_resize_expert_axis(
    module: nn.Module,
    current: torch.Tensor,
    value: torch.Tensor,
    *,
    expert_axis_size: int | None,
) -> bool:
    """Authorize only manifest-declared sibling router expert-axis changes."""

    if expert_axis_size is None or int(expert_axis_size) <= 0:
        return False
    if current.ndim < 1 or current.ndim != value.ndim:
        return False
    if int(value.shape[0]) != int(expert_axis_size):
        return False
    if tuple(current.shape[1:]) != tuple(value.shape[1:]):
        return False
    module_experts = getattr(module, "num_experts", None)
    if module_experts is None:
        return False
    return int(module_experts) in {
        int(current.shape[0]),
        int(expert_axis_size),
    }


def replace_packed_child_module(
    root: nn.Module, module_name: str, replacement: nn.Module
) -> None:
    if not module_name:
        raise ValueError(
            "Cannot replace the root module from a compressed_weights packed manifest."
        )
    parent_name, child_name = (
        module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    )
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)
