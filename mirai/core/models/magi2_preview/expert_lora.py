"""Compact factor rows for the MAGI-2 per-expert projection path."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from mirai.core.models.compressed_weights.execution.active_expert_lora import ActiveExpertLoRA


class Magi2ExpertLoRA(ActiveExpertLoRA):
    _expert_rows: dict[int, int] | None = None

    def set_active_experts(self, expert_ids: Iterable[int]) -> None:
        ids = sorted(set(expert_ids))
        if self._expert_rows is not None and ids != list(self._expert_rows):
            raise ValueError("MAGI-2 compact LoRA selection cannot change after optimizer binding.")
        super().set_active_experts(ids)

    def compact_selected_experts(self, ids: list[int]) -> None:
        if not ids or ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] >= self.active_expert_mask.numel():
            raise ValueError("MAGI-2 compact LoRA requires sorted, unique, in-range expert IDs.")
        rows = {expert: row for row, expert in enumerate(ids)}
        if self._expert_rows is not None:
            if rows != self._expert_rows:
                raise ValueError("MAGI-2 compact LoRA selection cannot change after optimizer binding.")
            return
        self.set_active_experts(ids)
        a_ids = torch.tensor(ids, device=self.lora_a.device)
        b_ids = a_ids.to(self.lora_b.device)
        self.lora_a = nn.Parameter(self.lora_a.detach().index_select(0, a_ids))
        self.lora_b = nn.Parameter(self.lora_b.detach().index_select(0, b_ids))
        self._expert_rows = rows

    def _factor_row(self, expert_idx: int) -> int:
        if self._expert_rows is None:
            return expert_idx
        return self._expert_rows[expert_idx]

    def compact_expert_ids(self) -> tuple[int, ...] | None:
        return tuple(self._expert_rows) if self._expert_rows is not None else None
