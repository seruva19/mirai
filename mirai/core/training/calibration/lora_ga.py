"""Bounded single-GPU LoRA-GA gradient calibration."""

from __future__ import annotations

from typing import Any

from mirai.core.models.adapters.lora import LoRALinear
from mirai.core.training.lifecycle.training_step_pre import _build_training_batch_factory
from mirai.core.training.lifecycle.training_step_pre import resolve_step_sampling_context

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def maybe_initialize_lora_ga(session: Any) -> dict[str, Any] | None:
    adapter = session.config.adapter
    if str(adapter.lora_init).strip().lower() != "lora_ga":
        return None
    if bool(getattr(session, "resumed", False)):
        return None
    if torch is None:  # pragma: no cover
        raise RuntimeError("LoRA-GA initialization requires torch.")
    targets = [
        (name, module)
        for name, module in session.trainer.pipeline.named_modules()
        if isinstance(module, LoRALinear) and module._lora_init == "lora_ga"
    ]
    unsupported = [
        name
        for name, module in session.trainer.pipeline.named_modules()
        if getattr(module, "_lora_init", "") == "lora_ga"
        and not isinstance(module, LoRALinear)
    ]
    if unsupported:
        raise ValueError(
            "LoRA-GA supports dense LoRALinear targets only; "
            "unsupported targets: " + ", ".join(unsupported[:8])
        )
    if not targets:
        raise ValueError("LoRA-GA found no dense LoRALinear targets.")
    steps = int(adapter.lora_ga_calibration_steps)
    sampling_context = resolve_step_sampling_context(session)
    build_batch = _build_training_batch_factory(
        session=session, sampling_context=sampling_context
    )
    gradients = {
        name: torch.zeros_like(module.base.weight, device="cpu", dtype=torch.float32)
        for name, module in targets
    }
    for _, module in targets:
        module.base.weight.requires_grad_(True)
    trainer = session.trainer
    validation_state = trainer.begin_validation()
    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    completed = 0
    try:
        for step in range(steps):
            session.optimizer.zero_grad(set_to_none=True)
            loss, _ = trainer.compute_loss(build_batch(step), training=False)
            loss.backward()
            for name, module in targets:
                grad = module.base.weight.grad
                if grad is None:
                    raise RuntimeError(f"LoRA-GA target {name!r} produced no gradient.")
                gradients[name].add_(grad.detach().float().cpu(), alpha=1.0 / steps)
            completed += 1
    finally:
        session.optimizer.zero_grad(set_to_none=True)
        for _, module in targets:
            module.base.weight.requires_grad_(False)
            module.base.weight.grad = None
        if rng is not None and rng_state is not None:
            rng.setstate(rng_state)
        trainer.end_validation(validation_state)
    for name, module in targets:
        module.initialize_from_full_gradient(
            gradients[name],
            stable_gamma=float(adapter.lora_ga_stable_gamma),
        )
    report = {
        "calibration_steps": completed,
        "targets": tuple(name for name, _ in targets),
    }
    setattr(session, "lora_ga_initialization_report", report)
    return report


__all__ = ["maybe_initialize_lora_ga"]
