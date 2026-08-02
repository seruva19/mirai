"""Single-GPU, pre-optimizer GoRA calibration for model-agnostic LoRA hosts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

from mirai.core.models.adapters.lora import (
    LoRAExpertTensorParametrization,
    LoRALinear,
)
from mirai.core.models.adapters.lora_gora import (
    allocate_gora_ranks,
    gora_sensitivity_importance,
    gora_target_seed,
    initialize_gora_module,
)
from mirai.core.training.lifecycle.training_step_pre import (
    _build_training_batch_factory,
    resolve_step_sampling_context,
)

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GoRACalibrationReport:
    calibration_steps: int
    target_ranks: dict[str, int]
    target_importance: dict[str, float]
    target_cosine_similarity: dict[str, float]
    target_relative_norm: dict[str, float]
    smoothed_reference_budget: float
    smoothed_allocated_budget: float
    actual_reference_parameters: int
    actual_allocated_parameters: int
    allocation_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _GoRATarget:
    name: str
    module: Any
    base_weight: Any

    @property
    def geometry(self) -> tuple[int, int, int]:
        shape = tuple(int(value) for value in self.base_weight.shape)
        if len(shape) == 2:
            return (1, shape[0], shape[1])
        if len(shape) == 3:
            return (shape[0], shape[1], shape[2])
        raise ValueError(f"GoRA target {self.name!r} is not a matrix target.")


def maybe_initialize_gora(
    *,
    trainer: Any,
    config: Any,
    prepared_data: Any,
    compute_device: Any,
    compute_dtype: Any,
    curriculum: Any,
    rng: Any,
    run_state: Any,
    grad_accum: int,
) -> GoRACalibrationReport | None:
    """Calibrate, resize, and initialize GoRA before optimizer construction."""

    adapter = config.adapter
    if str(adapter.lora_init).strip().lower() != "gora":
        return None
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("GoRA calibration requires torch.")
    root_provider = getattr(
        trainer.pipeline,
        "get_adapter_calibration_root",
        None,
    )
    calibration_root = (
        root_provider() if callable(root_provider) else trainer.pipeline
    )
    targets = _collect_gora_targets(calibration_root)
    if not targets:
        raise ValueError("GoRA found no supported LoRA targets.")

    steps = int(adapter.gora_calibration_steps)
    reference_rank = int(adapter.rank)
    minimum_rank = (
        int(adapter.gora_min_rank)
        if int(adapter.gora_min_rank) > 0
        else max(1, reference_rank // 2)
    )
    maximum_rank = (
        int(adapter.gora_max_rank)
        if int(adapter.gora_max_rank) > 0
        else reference_rank * 4
    )
    calibration_session = SimpleNamespace(
        config=config,
        trainer=trainer,
        compute_device=compute_device,
        compute_dtype=compute_dtype,
        train_records=prepared_data.train_records,
        temporal_base_ids=prepared_data.temporal_base_ids,
        temporal_groups=prepared_data.temporal_groups,
        curriculum=curriculum,
        rng=rng,
        run_state=run_state,
        grad_accum=max(1, int(grad_accum)),
    )
    sampling_context = resolve_step_sampling_context(calibration_session)
    build_batch = _build_training_batch_factory(
        session=calibration_session,
        sampling_context=sampling_context,
    )
    gradients = {
        target.name: torch.zeros_like(
            target.base_weight,
            device="cpu",
            dtype=torch.float32,
        )
        for target in targets
    }
    base_requires_grad = {
        target.name: bool(target.base_weight.requires_grad) for target in targets
    }
    factor_requires_grad = {
        target.name: (
            bool(target.module.lora_a.requires_grad),
            bool(target.module.lora_b.requires_grad),
        )
        for target in targets
    }
    for target in targets:
        target.module.lora_a.requires_grad_(False)
        target.module.lora_b.requires_grad_(False)
        target.base_weight.requires_grad_(True)

    validation_state = trainer.begin_validation()
    rng_state = rng.getstate()
    completed = 0
    try:
        for step in range(steps):
            for target in targets:
                target.base_weight.grad = None
            loss, _ = trainer.compute_loss(build_batch(step), training=False)
            loss.backward()
            for target in targets:
                gradient = target.base_weight.grad
                if gradient is None:
                    raise RuntimeError(
                        f"GoRA target {target.name!r} produced no gradient."
                    )
                gradients[target.name].add_(
                    gradient.detach().float().cpu(),
                    alpha=1.0 / steps,
                )
                target.base_weight.grad = None
            completed += 1
    finally:
        for target in targets:
            target.base_weight.grad = None
            target.base_weight.requires_grad_(
                base_requires_grad[target.name]
            )
            old_a, old_b = factor_requires_grad[target.name]
            target.module.lora_a.requires_grad_(old_a)
            target.module.lora_b.requires_grad_(old_b)
        rng.setstate(rng_state)
        trainer.end_validation(validation_state)

    importance = {
        target.name: gora_sensitivity_importance(
            target.base_weight,
            gradients[target.name],
        )
        for target in targets
    }
    plan = allocate_gora_ranks(
        {target.name: target.geometry for target in targets},
        importance,
        reference_rank=reference_rank,
        minimum_rank=minimum_rank,
        maximum_rank=maximum_rank,
    )
    cosine: dict[str, float] = {}
    relative_norm: dict[str, float] = {}
    for target in targets:
        diagnostic = initialize_gora_module(
            target.module,
            gradients.pop(target.name).to(
                device=target.module.lora_a.device,
                dtype=torch.float32,
            ),
            rank=plan.ranks[target.name],
            stable_gamma=float(adapter.gora_stable_gamma),
            seed=gora_target_seed(int(config.training.seed), target.name),
        )
        target.module.lora_a.requires_grad_(True)
        target.module.lora_b.requires_grad_(True)
        cosine[target.name] = diagnostic.cosine_similarity
        relative_norm[target.name] = diagnostic.relative_norm

    recorder = getattr(trainer.pipeline, "record_gora_allocation", None)
    if callable(recorder):
        recorder(
            ranks=dict(plan.ranks),
            fingerprint=str(plan.fingerprint),
        )
    return GoRACalibrationReport(
        calibration_steps=completed,
        target_ranks=dict(plan.ranks),
        target_importance=dict(plan.importance),
        target_cosine_similarity=cosine,
        target_relative_norm=relative_norm,
        smoothed_reference_budget=plan.smoothed_reference_budget,
        smoothed_allocated_budget=plan.smoothed_allocated_budget,
        actual_reference_parameters=plan.actual_reference_parameters,
        actual_allocated_parameters=plan.actual_allocated_parameters,
        allocation_fingerprint=plan.fingerprint,
    )


def _collect_gora_targets(root: Any) -> tuple[_GoRATarget, ...]:
    targets: list[_GoRATarget] = []
    unsupported: list[str] = []
    seen_names: set[str] = set()
    for module_name, module in root.named_modules():
        if str(getattr(module, "_lora_init", "")).strip().lower() != "gora":
            continue
        if isinstance(module, LoRALinear):
            base_weight = getattr(module.base, "weight", None)
            name = str(module_name)
        elif isinstance(module, LoRAExpertTensorParametrization):
            base_weight = module.gora_base_weight()
            name = str(module.adapter_name)
        else:
            if bool(getattr(module, "_mirai_expert_lora_adapter", False)):
                unsupported.append(str(module_name) or type(module).__name__)
            continue
        if not name or name in seen_names:
            raise ValueError(f"GoRA target name {name!r} is missing or duplicated.")
        if not isinstance(base_weight, nn.Parameter):
            raise ValueError(
                f"GoRA target {name!r} requires a dense floating base parameter."
            )
        if base_weight.ndim not in {2, 3} or not base_weight.is_floating_point():
            raise ValueError(
                f"GoRA target {name!r} requires a floating matrix or grouped matrix."
            )
        targets.append(
            _GoRATarget(name=name, module=module, base_weight=base_weight)
        )
        seen_names.add(name)
    if unsupported:
        raise ValueError(
            "GoRA does not support the active adapter hosts: "
            + ", ".join(unsupported[:8])
        )
    return tuple(sorted(targets, key=lambda target: target.name))


__all__ = ["GoRACalibrationReport", "maybe_initialize_gora"]
