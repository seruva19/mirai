"""Checkpointed FlexMoE action-learning state machine.

The configured training objective supplies the native task-preservation term;
the registered model provider exposes grouped expert hosts without leaking a
family layout into this module. This owner implements the source-defined
discrete action sampling, linear schedules, load-sensitive cost, entropy term,
optimization state, and final action plans.

Source: Mo et al., "FlexMoE: One-for-All Nested Intra-Expert Pruning for MoE
Language Models", Equations 5-12, arXiv:2606.27866.
https://arxiv.org/abs/2606.27866
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from mirai.core.lineage import sha256_file
from mirai.core.models.compressed_weights.packed.packed_state import (
    packed_artifact_fingerprint,
)
from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.calibration.flexmoe import (
    FlexMoEActionController,
    FlexMoEActionPlan,
    FlexMoECalibrationTarget,
    FlexMoEExpertLoadObserver,
    FlexMoETaylorGradientObserver,
    action_entropy,
    clean_action_probabilities,
    global_prune_budget,
    load_sensitive_cost,
    load_ranking_evidence,
    normalize_action_ratios,
    save_action_plans,
    save_ranking_evidence,
)
from mirai.core.moe.calibration.router_repair import diffusion_router_kd_loss
from mirai.core.training.lifecycle.training_step_pre import (
    _build_training_batch_factory,
    resolve_step_sampling_context,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


FLEXMOE_ACTION_TRAINING_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FlexMoEActionLearningSpec:
    total_steps: int
    temperature_start: float
    temperature_end: float
    cost_weight_start: float
    cost_weight_end: float
    entropy_weight_start: float
    entropy_weight_end: float

    def validate(self) -> FlexMoEActionLearningSpec:
        if int(self.total_steps) < 1:
            raise ValueError("FlexMoE action learning requires total_steps >= 1.")
        for label, value in {
            "temperature_start": self.temperature_start,
            "temperature_end": self.temperature_end,
        }.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"FlexMoE {label} must be finite and positive.")
        for label, value in {
            "cost_weight_start": self.cost_weight_start,
            "cost_weight_end": self.cost_weight_end,
            "entropy_weight_start": self.entropy_weight_start,
            "entropy_weight_end": self.entropy_weight_end,
        }.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"FlexMoE {label} must be finite and non-negative.")
        return self

    def at(self, step: int) -> tuple[float, float, float]:
        self.validate()
        index = int(step)
        if index < 0 or index >= int(self.total_steps):
            raise ValueError("FlexMoE schedule step is outside the configured run.")
        fraction = (
            0.0
            if int(self.total_steps) == 1
            else float(index) / float(int(self.total_steps) - 1)
        )

        def interpolate(start: float, end: float) -> float:
            return float(start) + fraction * (float(end) - float(start))

        return (
            interpolate(self.temperature_start, self.temperature_end),
            interpolate(self.cost_weight_start, self.cost_weight_end),
            interpolate(self.entropy_weight_start, self.entropy_weight_end),
        )


@dataclass(frozen=True)
class FlexMoEActionStepReport:
    step: int
    quality_loss: float
    cost: float
    entropy: float
    total_loss: float
    temperature: float
    cost_weight: float
    entropy_weight: float


@dataclass(frozen=True)
class FlexMoERankingRunReport:
    output_path: str
    calibration_steps: int
    modules: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "output": self.output_path,
            "calibration_steps": self.calibration_steps,
            "modules": self.modules,
        }


@dataclass(frozen=True)
class FlexMoEActionRunReport:
    output_path: str
    steps: int
    global_prune_budget: float
    modules: dict[str, dict[str, float]]
    final_step: FlexMoEActionStepReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "output": self.output_path,
            "steps": self.steps,
            "global_prune_budget": self.global_prune_budget,
            "modules": self.modules,
            "final_step": asdict(self.final_step),
        }


QualityCallback = Callable[
    [Mapping[str, Any]],
    tuple[Any, Mapping[str, Any]],
]


def _validate_targets(
    raw_targets: Mapping[str, FlexMoECalibrationTarget],
) -> dict[str, FlexMoECalibrationTarget]:
    if not raw_targets:
        raise ValueError("Model provider returned no FlexMoE calibration targets.")
    targets: dict[str, FlexMoECalibrationTarget] = {}
    for raw_name, target in raw_targets.items():
        if not isinstance(target, FlexMoECalibrationTarget):
            raise TypeError(
                "Model provider FlexMoE targets must use FlexMoECalibrationTarget."
            )
        target.validate()
        name = str(raw_name)
        if name != target.name or name in targets:
            raise ValueError("FlexMoE target names must match and be unique.")
        targets[name] = target
    return targets


def _freeze_parameters(
    module: Any,
    *,
    extra_parameters: tuple[Any, ...] = (),
) -> list[tuple[Any, bool]]:
    parameters = list(module.parameters()) + list(extra_parameters)
    unique = {id(parameter): parameter for parameter in parameters}
    state = [
        (parameter, bool(parameter.requires_grad))
        for parameter in unique.values()
    ]
    for parameter, _requires_grad in state:
        parameter.requires_grad_(False)
        parameter.grad = None
    return state


def _restore_parameters(state: list[tuple[Any, bool]]) -> None:
    for parameter, requires_grad in state:
        parameter.grad = None
        parameter.requires_grad_(requires_grad)


def _source_artifact_fingerprint(config: Any) -> str:
    path = str(config.memory.frozen_weight_packed_state_path).strip()
    if not path:
        raise ValueError("FlexMoE calibration requires a packed source artifact.")
    return packed_artifact_fingerprint(path)


def run_flexmoe_ranking_session(
    session: Any,
    *,
    output_path: str | Path,
    calibration_steps: int,
    overwrite: bool = False,
) -> FlexMoERankingRunReport:
    """Collect Equation-2 Taylor evidence through the native task objective."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("FlexMoE ranking requires torch.")
    config = session.config
    if str(config.model.params.flexmoe_calibration).strip().lower() != "nested":
        raise ValueError(
            "FlexMoE ranking requires model.params.flexmoe_calibration='nested'."
        )
    if str(config.model.params.expert_weight_compression).strip().lower() != "off":
        raise ValueError("FlexMoE ranking requires the complete unpruned expert source.")
    steps = int(calibration_steps)
    if steps < 1:
        raise ValueError("FlexMoE ranking steps must be positive.")
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"FlexMoE ranking output already exists: {output}.")

    provider = get_model_family_provider(str(config.model.type))
    if provider is None or not provider.supports_flexmoe_calibration(config):
        raise ValueError(
            f"Model provider {config.model.type!r} does not support FlexMoE calibration."
        )
    targets = _validate_targets(
        provider.build_flexmoe_calibration_targets(session.trainer.pipeline)
    )
    if any(
        bool(getattr(target.host, "flexmoe_channel_mask_active", False))
        or bool(getattr(target.host, "flexmoe_taylor_observer_active", False))
        for target in targets.values()
    ):
        raise ValueError("FlexMoE ranking requires unused calibration hosts.")
    source_fingerprint = _source_artifact_fingerprint(config)
    observers = {
        name: FlexMoETaylorGradientObserver(
            num_experts=target.num_experts,
            intermediate_size=target.intermediate_size,
        )
        for name, target in targets.items()
    }
    trainer = session.trainer
    model = trainer.pipeline.get_training_model()
    if model is None:
        raise ValueError("FlexMoE ranking requires an exposed training model.")
    parameter_state = _freeze_parameters(
        model,
        extra_parameters=tuple(
            parameter
            for _name, parameter in trainer.objective.get_named_trainable_parameters()
        ),
    )
    attached: list[FlexMoECalibrationTarget] = []
    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    validation_state: dict[str, Any] | None = None
    try:
        validation_state = trainer.begin_validation()
        for name, target in targets.items():
            target.bind_taylor_observer(observers[name])
            attached.append(target)
        sampling_context = resolve_step_sampling_context(session)
        build_batch = _build_training_batch_factory(
            session=session,
            sampling_context=sampling_context,
        )
        for step in range(steps):
            batch = build_batch(step)
            inputs = trainer.prepare_objective_calibration_inputs(
                batch,
                training=False,
            )
            if not isinstance(inputs.noisy_latents, torch.Tensor):
                raise TypeError("FlexMoE ranking requires tensor noisy latents.")
            inputs.noisy_latents = inputs.noisy_latents.detach().requires_grad_(True)
            for observer in observers.values():
                observer.begin_batch(device=inputs.noisy_latents.device)
            try:
                prediction = trainer.predict_objective_calibration_inputs(
                    inputs,
                    training=False,
                )
                task = trainer.evaluate_calibration_task_loss(
                    batch=batch,
                    inputs=inputs,
                    prediction=prediction,
                ).loss_pre_accum
                if (
                    not isinstance(task, torch.Tensor)
                    or task.ndim != 0
                    or not bool(torch.isfinite(task).item())
                ):
                    raise ValueError("FlexMoE ranking requires one finite task loss.")
                task.backward()
                trainer.pipeline.finish_backward_offloads()
                for observer in observers.values():
                    observer.finish_batch()
            except BaseException:
                for observer in observers.values():
                    observer.abort_batch()
                raise
    finally:
        try:
            for target in reversed(attached):
                target.bind_taylor_observer(None)
            if validation_state is not None:
                trainer.end_validation(validation_state)
        finally:
            _restore_parameters(parameter_state)
            if rng is not None and rng_state is not None:
                rng.setstate(rng_state)

    evidence = {name: observer.evidence() for name, observer in observers.items()}
    manifest = session.manifest
    save_ranking_evidence(
        output,
        evidence,
        dataset_snapshot_id=str(manifest.dataset_snapshot_id),
        model_snapshot_id=source_fingerprint,
        config_snapshot_id=str(manifest.config_snapshot_id),
    )
    return FlexMoERankingRunReport(
        output_path=str(output),
        calibration_steps=steps,
        modules={
            name: {
                "num_experts": item.num_experts,
                "intermediate_size": item.intermediate_size,
            }
            for name, item in evidence.items()
        },
    )


def run_flexmoe_action_session(
    session: Any,
    *,
    ranking_path: str | Path,
    output_path: str | Path,
    action_ratios: tuple[float, ...],
    spec: FlexMoEActionLearningSpec,
    learning_rate: float,
    thickest_logit_margin: float,
    teacher_loss_weight: float,
    seed: int,
    overwrite: bool = False,
) -> FlexMoEActionRunReport:
    """Learn per-expert prefix actions on exact teacher/student video inputs."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("FlexMoE action learning requires torch.")
    config = session.config
    if str(config.model.params.flexmoe_calibration).strip().lower() != "nested":
        raise ValueError(
            "FlexMoE action learning requires "
            "model.params.flexmoe_calibration='nested'."
        )
    if str(config.model.params.expert_weight_compression).strip().lower() != "off":
        raise ValueError(
            "FlexMoE action learning requires the complete unpruned expert source."
        )
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("FlexMoE action learning rate must be finite and positive.")
    if not math.isfinite(float(teacher_loss_weight)) or float(teacher_loss_weight) < 0.0:
        raise ValueError("FlexMoE teacher-loss weight must be finite and non-negative.")
    if int(seed) < 0:
        raise ValueError("FlexMoE action seed must be non-negative.")
    ratios = normalize_action_ratios(action_ratios)
    schedule = spec.validate()
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"FlexMoE action output already exists: {output}.")

    ranking, ranking_lineage = load_ranking_evidence(ranking_path)
    manifest = session.manifest
    expected_lineage = {
        "dataset_snapshot_id": str(manifest.dataset_snapshot_id),
        "model_snapshot_id": _source_artifact_fingerprint(config),
        "config_snapshot_id": str(manifest.config_snapshot_id),
    }
    if ranking_lineage != expected_lineage:
        raise ValueError("FlexMoE ranking evidence does not belong to this session.")
    provider = get_model_family_provider(str(config.model.type))
    if provider is None or not provider.supports_flexmoe_calibration(config):
        raise ValueError(
            f"Model provider {config.model.type!r} does not support FlexMoE calibration."
        )
    targets = _validate_targets(
        provider.build_flexmoe_calibration_targets(session.trainer.pipeline)
    )
    if any(
        bool(getattr(target.host, "flexmoe_channel_mask_active", False))
        or bool(getattr(target.host, "flexmoe_taylor_observer_active", False))
        for target in targets.values()
    ):
        raise ValueError("FlexMoE action learning requires unused calibration hosts.")
    if set(targets) != set(ranking):
        raise ValueError("FlexMoE ranking evidence does not cover the provider targets.")
    device = session.compute_device
    controllers = {
        name: FlexMoEActionController(
            num_experts=target.num_experts,
            action_ratios=ratios,
            thickest_logit_margin=float(thickest_logit_margin),
        ).to(device=device)
        for name, target in targets.items()
    }
    for name, target in targets.items():
        evidence = ranking[name]
        if (
            evidence.num_experts != target.num_experts
            or evidence.intermediate_size != target.intermediate_size
        ):
            raise ValueError(f"FlexMoE ranking topology mismatch for {name!r}.")
    optimizer = torch.optim.AdamW(
        [controller.logits for controller in controllers.values()],
        lr=float(learning_rate),
        weight_decay=0.0,
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    learner = FlexMoEActionLearningSession(
        controllers=controllers,
        permutations={name: ranking[name].permutation() for name in targets},
        optimizer=optimizer,
        generator=generator,
        spec=schedule,
    )
    trainer = session.trainer
    model = trainer.pipeline.get_training_model()
    if model is None:
        raise ValueError("FlexMoE action learning requires an exposed training model.")
    parameter_state = _freeze_parameters(
        model,
        extra_parameters=tuple(
            parameter
            for _name, parameter in trainer.objective.get_named_trainable_parameters()
        ),
    )
    previous_observers = {
        name: target.host.get_routed_output_observer()
        for name, target in targets.items()
    }
    if any(observer is not None for observer in previous_observers.values()):
        _restore_parameters(parameter_state)
        raise ValueError(
            "FlexMoE action learning requires routed-output regularizers to be disabled."
        )
    validation_state: dict[str, Any] | None = None
    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    reports: list[FlexMoEActionStepReport] = []
    try:
        validation_state = trainer.begin_validation()
        sampling_context = resolve_step_sampling_context(session)
        build_batch = _build_training_batch_factory(
            session=session,
            sampling_context=sampling_context,
        )
        for step in range(schedule.total_steps):
            batch = build_batch(step)
            inputs = trainer.prepare_objective_calibration_inputs(
                batch,
                training=False,
            )
            with torch.no_grad():
                teacher = trainer.predict_objective_calibration_inputs(
                    inputs,
                    training=False,
                )
            if not isinstance(teacher, torch.Tensor):
                raise TypeError(
                    "FlexMoE action learning requires a tensor objective prediction."
                )
            teacher = teacher.detach()

            def quality_callback(
                masks: Mapping[str, Any],
                *,
                batch: Any = batch,
                inputs: Any = inputs,
                teacher: Any = teacher,
            ) -> tuple[Any, Mapping[str, Any]]:
                observers = {
                    name: FlexMoEExpertLoadObserver(target.num_experts)
                    for name, target in targets.items()
                }
                attached_masks: list[FlexMoECalibrationTarget] = []
                attached_observers: list[FlexMoECalibrationTarget] = []
                try:
                    for name, target in targets.items():
                        target.bind_mask(masks[name])
                        attached_masks.append(target)
                        target.bind_load_observer(observers[name])
                        attached_observers.append(target)
                    student = trainer.predict_objective_calibration_inputs(
                        inputs,
                        training=False,
                    )
                    if not isinstance(student, torch.Tensor):
                        raise TypeError(
                            "FlexMoE action learning requires tensor student predictions."
                        )
                    task_loss = trainer.evaluate_calibration_task_loss(
                        batch=batch,
                        inputs=inputs,
                        prediction=student,
                    ).loss_pre_accum
                    teacher_loss = diffusion_router_kd_loss(
                        student,
                        teacher,
                        loss_mask=inputs.loss_mask,
                    )
                    loads = {name: observer.take_load() for name, observer in observers.items()}
                    return task_loss + float(teacher_loss_weight) * teacher_loss, loads
                finally:
                    for target in reversed(attached_observers):
                        target.bind_load_observer(None)
                    for target in reversed(attached_masks):
                        target.bind_mask(None)

            reports.append(learner.step(quality_callback))
            trainer.pipeline.finish_backward_offloads()
    finally:
        try:
            for name, target in targets.items():
                target.bind_load_observer(previous_observers[name])
                target.bind_mask(None)
            if validation_state is not None:
                trainer.end_validation(validation_state)
        finally:
            _restore_parameters(parameter_state)
            if rng is not None and rng_state is not None:
                rng.setstate(rng_state)

    plans = learner.action_plans()
    ranking_snapshot_id = "sha256:" + sha256_file(ranking_path)
    save_action_plans(
        output,
        plans,
        dataset_snapshot_id=expected_lineage["dataset_snapshot_id"],
        model_snapshot_id=expected_lineage["model_snapshot_id"],
        config_snapshot_id=expected_lineage["config_snapshot_id"],
        ranking_snapshot_id=ranking_snapshot_id,
    )
    return FlexMoEActionRunReport(
        output_path=str(output),
        steps=schedule.total_steps,
        global_prune_budget=float(
            global_prune_budget(
                torch.cat(
                    [plan.retention_ratios() for plan in plans.values()],
                    dim=0,
                )
            ).item()
        ),
        modules={
            name: {
                "prune_budget": plan.prune_budget(),
                "minimum_retention": float(plan.retention_ratios().min().item()),
                "maximum_retention": float(plan.retention_ratios().max().item()),
            }
            for name, plan in plans.items()
        },
        final_step=reports[-1],
    )


class FlexMoEActionLearningSession:
    """Own exact action-logit optimizer/RNG/schedule state across resumes."""

    def __init__(
        self,
        *,
        controllers: Mapping[str, FlexMoEActionController],
        permutations: Mapping[str, Any],
        optimizer: Any,
        generator: Any,
        spec: FlexMoEActionLearningSpec,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("FlexMoE action learning requires torch.")
        if not controllers or set(controllers) != set(permutations):
            raise ValueError(
                "FlexMoE controllers and channel permutations must exactly agree."
            )
        self.controllers = dict(controllers)
        self.permutations = {
            str(name): torch.as_tensor(value).detach().cpu().to(torch.long)
            for name, value in permutations.items()
        }
        self.optimizer = optimizer
        self.generator = generator
        self.spec = spec.validate()
        self.step_index = 0
        self._load_sum = {
            name: torch.zeros(controller.num_experts, dtype=torch.float64)
            for name, controller in self.controllers.items()
        }
        self._load_observations = 0
        parameters = {
            id(parameter)
            for group in getattr(optimizer, "param_groups", ())
            for parameter in group.get("params", ())
        }
        expected = {
            id(controller.logits) for controller in self.controllers.values()
        }
        if parameters != expected:
            raise ValueError(
                "FlexMoE optimizer must own exactly the action-logit parameters."
            )
        devices = {controller.logits.device.type for controller in self.controllers.values()}
        if len(devices) != 1:
            raise ValueError("FlexMoE action controllers must share one device type.")
        generator_device = str(getattr(generator, "device", "cpu")).split(":", 1)[0]
        if generator_device not in devices:
            raise ValueError("FlexMoE generator and action logits must share a device type.")
        for name, controller in self.controllers.items():
            order = self.permutations[name]
            if tuple(order.shape)[0] != controller.num_experts:
                raise ValueError(f"FlexMoE permutation topology mismatch for {name!r}.")

    @property
    def complete(self) -> bool:
        return self.step_index >= int(self.spec.total_steps)

    def step(self, quality_callback: QualityCallback) -> FlexMoEActionStepReport:
        if self.complete:
            raise RuntimeError("FlexMoE action-learning schedule is already complete.")
        temperature, cost_weight, entropy_weight = self.spec.at(self.step_index)
        masks: dict[str, Any] = {}
        for name, controller in self.controllers.items():
            mask, _soft, _hard = controller.sampled_original_channel_masks(
                self.permutations[name],
                temperature=temperature,
                generator=self.generator,
            )
            masks[name] = mask
        quality_loss, raw_loads = quality_callback(masks)
        quality = torch.as_tensor(quality_loss)
        if quality.ndim != 0 or not quality.is_floating_point() or not bool(
            torch.isfinite(quality).item()
        ):
            raise ValueError("FlexMoE quality callback must return one finite scalar.")
        if set(raw_loads) != set(self.controllers):
            raise ValueError("FlexMoE quality callback returned incomplete expert loads.")

        cost = quality.new_zeros((), dtype=torch.float32)
        weighted_entropy = quality.new_zeros((), dtype=torch.float32)
        total_experts = sum(
            controller.num_experts for controller in self.controllers.values()
        )
        validated_loads: dict[str, Any] = {}
        for name, controller in self.controllers.items():
            probabilities = clean_action_probabilities(controller.logits)
            load = torch.as_tensor(
                raw_loads[name],
                device=probabilities.device,
                dtype=torch.float32,
            )
            layer_cost = load_sensitive_cost(
                probabilities,
                load,
                controller.action_ratios,
            )
            layer_entropy = action_entropy(probabilities)
            cost = cost + layer_cost.to(device=quality.device)
            weighted_entropy = weighted_entropy + layer_entropy.to(
                device=quality.device
            ) * (float(controller.num_experts) / float(total_experts))
            validated_loads[name] = load.detach().cpu().to(torch.float64)
        total = quality + cost_weight * cost - entropy_weight * weighted_entropy
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        for name, load in validated_loads.items():
            self._load_sum[name].add_(load)
        self._load_observations += 1
        report = FlexMoEActionStepReport(
            step=self.step_index,
            quality_loss=float(quality.detach().item()),
            cost=float(cost.detach().item()),
            entropy=float(weighted_entropy.detach().item()),
            total_loss=float(total.detach().item()),
            temperature=temperature,
            cost_weight=cost_weight,
            entropy_weight=entropy_weight,
        )
        self.step_index += 1
        return report

    def action_plans(self) -> dict[str, FlexMoEActionPlan]:
        if self._load_observations < 1:
            raise RuntimeError("FlexMoE action plans require observed routing loads.")
        return {
            name: controller.action_plan(
                self._load_sum[name] / float(self._load_observations)
            )
            for name, controller in self.controllers.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FLEXMOE_ACTION_TRAINING_STATE_SCHEMA_VERSION,
            "spec": asdict(self.spec),
            "step_index": int(self.step_index),
            "controllers": {
                name: controller.state_dict()
                for name, controller in self.controllers.items()
            },
            "optimizer": self.optimizer.state_dict(),
            "generator_state": self.generator.get_state().detach().cpu().clone(),
            "load_sum": {name: value.clone() for name, value in self._load_sum.items()},
            "load_observations": int(self._load_observations),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", 0)) != (
            FLEXMOE_ACTION_TRAINING_STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Unsupported FlexMoE action-training state schema "
                f"{state.get('schema_version')!r}."
            )
        raw_spec = state.get("spec")
        if not isinstance(raw_spec, Mapping) or FlexMoEActionLearningSpec(
            **raw_spec
        ) != self.spec:
            raise ValueError("FlexMoE action-training schedule changed across resume.")
        controller_states = state.get("controllers")
        load_sum = state.get("load_sum")
        if not isinstance(controller_states, Mapping) or set(controller_states) != set(
            self.controllers
        ):
            raise ValueError("FlexMoE controller topology changed across resume.")
        if not isinstance(load_sum, Mapping) or set(load_sum) != set(self.controllers):
            raise ValueError("FlexMoE load topology changed across resume.")
        step = int(state.get("step_index", -1))
        observations = int(state.get("load_observations", -1))
        if step < 0 or step > int(self.spec.total_steps) or observations != step:
            raise ValueError("FlexMoE action-training progress state is invalid.")
        for name, controller in self.controllers.items():
            controller.load_state_dict(controller_states[name])
            value = torch.as_tensor(load_sum[name]).to(torch.float64)
            if tuple(value.shape) != (controller.num_experts,) or not bool(
                torch.isfinite(value).all().item()
            ):
                raise ValueError(f"FlexMoE accumulated load is invalid for {name!r}.")
            self._load_sum[name].copy_(value)
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(torch.as_tensor(state["generator_state"]).cpu())
        self.step_index = step
        self._load_observations = observations


__all__ = [
    "FLEXMOE_ACTION_TRAINING_STATE_SCHEMA_VERSION",
    "FlexMoEActionRunReport",
    "FlexMoEActionLearningSession",
    "FlexMoEActionLearningSpec",
    "FlexMoEActionStepReport",
    "FlexMoERankingRunReport",
    "QualityCallback",
    "run_flexmoe_action_session",
    "run_flexmoe_ranking_session",
]
