"""Early-stop state helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopState:
    best_val_loss: float
    best_step: int
    patience_counter: int
    best_checkpoint_path: str

    def to_dict(self) -> dict:
        return {
            "best_val_loss": float(self.best_val_loss),
            "best_step": int(self.best_step),
            "patience_counter": int(self.patience_counter),
            "best_checkpoint_path": str(self.best_checkpoint_path),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EarlyStopState":
        return cls(
            best_val_loss=float(payload.get("best_val_loss", float("inf"))),
            best_step=int(payload.get("best_step", -1)),
            patience_counter=int(payload.get("patience_counter", 0)),
            best_checkpoint_path=str(payload.get("best_checkpoint_path", "")),
        )
