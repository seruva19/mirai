"""Deterministic gradient-delta harness for sparse-MoE training paths."""

from __future__ import annotations

from dataclasses import dataclass

from mirai.config.schema import TrainingConfig
from mirai.core.training.trainer import Trainer

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for gradient harness: {exc}")


@dataclass
class GradientHarnessResult:
    steps: int
    param_deltas: dict[str, float]


def run_gradient_harness(config: TrainingConfig, steps: int = 8) -> GradientHarnessResult:
    trainer = Trainer(config)
    params = list(trainer.get_trainable_parameters())
    optimizer = torch.optim.SGD(params, lr=1e-3)

    initial = [p.detach().clone() for p in params]
    batch = {
        "latents": torch.tensor([0.25, -0.1], dtype=torch.float32),
        "text_embeds": torch.tensor([1.0, 0.5], dtype=torch.float32),
    }

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = trainer.compute_loss(batch)
        loss.backward()
        optimizer.step()

    deltas: dict[str, float] = {}
    for idx, (before, after) in enumerate(zip(initial, params, strict=True)):
        deltas[f"param_{idx}"] = float((after.detach() - before).abs().sum().item())
    return GradientHarnessResult(steps=steps, param_deltas=deltas)
