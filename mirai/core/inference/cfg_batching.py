"""Single-device batched classifier-free-guidance tensor contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BatchedCfgInputs:
    sample: torch.Tensor
    timestep: torch.Tensor
    text_embeds: dict[str, torch.Tensor]


def build_batched_cfg_inputs(
    *,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
) -> BatchedCfgInputs:
    """Build an uncond-first B=2 batch, padding variable-length text safely."""
    if int(sample.shape[0]) != 1:
        raise ValueError("Batched CFG requires a single-sample latent input.")
    if conditional.ndim != 3 or unconditional.ndim != 3:
        raise ValueError("Batched CFG text contexts must have shape [1, sequence, dim].")
    if int(conditional.shape[0]) != 1 or int(unconditional.shape[0]) != 1:
        raise ValueError("Batched CFG requires one conditional and one unconditional context.")
    if int(conditional.shape[2]) != int(unconditional.shape[2]):
        raise ValueError("Batched CFG conditional and unconditional text dims must match.")

    cond_len = int(conditional.shape[1])
    uncond_len = int(unconditional.shape[1])
    max_len = max(cond_len, uncond_len)

    def _pad(value: torch.Tensor) -> torch.Tensor:
        if int(value.shape[1]) == max_len:
            return value
        padding = value.new_zeros((1, max_len - int(value.shape[1]), value.shape[2]))
        return torch.cat((value, padding), dim=1)

    context = torch.cat((_pad(unconditional), _pad(conditional)), dim=0)
    mask = torch.zeros((2, max_len), dtype=torch.bool, device=context.device)
    mask[0, :uncond_len] = True
    mask[1, :cond_len] = True
    return BatchedCfgInputs(
        sample=sample.expand(2, *sample.shape[1:]),
        timestep=timestep.expand(2),
        text_embeds={"t5": context, "text_mask": mask},
    )


def split_batched_cfg_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(unconditional, conditional)`` outputs from a B=2 forward."""
    if not torch.is_tensor(output) or output.ndim == 0 or int(output.shape[0]) != 2:
        raise RuntimeError("Batched CFG forward must return a tensor with batch size 2.")
    return output[0:1], output[1:2]
