"""Typed lifecycle and deterministic composition for training policy plugins."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Mapping, Sequence

from mirai.core.registry import Registry


@dataclass(frozen=True)
class RecordSelectionContext:
    records: Sequence[Any]
    batch_size: int
    base_seed: int
    global_batch_index: int
    step: int


@dataclass(frozen=True)
class BatchAugmentContext:
    records: Sequence[Any]
    batch: Mapping[str, Any]
    step: int
    training: bool
    global_batch_index: int | None = None


@dataclass(frozen=True)
class PreviewRequestContext:
    """Everything a preview-owning policy needs to run a preview at one step.

    ``sample_pack`` is the resolved default ``(prompt, seed, sample_name)`` pack;
    ``base_metrics`` already carries the preview-pause offload receipts. A policy
    that claims previews owns the full result list, so it can replace the
    timestep grid (override ``denoise_steps``/``sample_solver``), run multiple
    adapter variants (A/B), or skip/extend the pack.
    """

    config: Any
    trainer: Any
    step: int
    output_dir: Any
    sample_pack: tuple[tuple[str, int, str], ...]
    base_metrics: Mapping[str, Any]
    curriculum_resolution: Any
    curriculum_frame_count: Any
    force: bool


class TrainingPolicy:
    """One opt-in feature's hooks into the generic training lifecycle."""

    name = ""
    priority = 100

    def configure_pipeline(self, pipeline: Any) -> None:
        _ = pipeline

    def validate_records(self, records: Sequence[Any]) -> list[str]:
        _ = records
        return []

    def select_records(
        self, context: RecordSelectionContext
    ) -> list[Any] | None:
        _ = context
        return None

    def augment_batch(self, context: BatchAugmentContext) -> Mapping[str, Any]:
        _ = context
        return {}

    def run_previews(self, request: PreviewRequestContext) -> list[Any] | None:
        """Optionally own preview generation for this step.

        Return ``None`` (the default) to defer to the standard preview loop;
        return a list of preview results to claim the step exclusively. Only one
        active policy may claim previews at a time.
        """
        _ = request
        return None

    def before_forward(
        self,
        pipeline: Any,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> None:
        _ = (pipeline, batch, training)

    def before_optimizer_step(self, optimizer: Any) -> None:
        _ = optimizer

    def after_optimizer_step(self, optimizer: Any, *, applied: bool) -> None:
        _ = (optimizer, applied)

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError(
                f"Training policy '{self.name}' does not own persistent state."
            )

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {}


TrainingPolicyFactory = Callable[[Any], TrainingPolicy | None]
TrainingPolicyValidator = Callable[[Any], list[str]]


@dataclass(frozen=True)
class TrainingPolicyPlugin:
    name: str
    build: TrainingPolicyFactory
    validate_config: TrainingPolicyValidator


TrainingPolicyRegistry: Registry[TrainingPolicyPlugin] = Registry("training policy")


def register_training_policy(
    name: str,
    *,
    validate_config: TrainingPolicyValidator | None = None,
) -> Callable[[TrainingPolicyFactory], TrainingPolicyFactory]:
    """Register a config-driven policy factory from a builtin or plugin module."""

    def _register(factory: TrainingPolicyFactory) -> TrainingPolicyFactory:
        plugin = TrainingPolicyPlugin(
            name=str(name).strip().lower(),
            build=factory,
            validate_config=validate_config or (lambda _config: []),
        )
        TrainingPolicyRegistry.register(name, plugin)
        return factory

    return _register


def load_training_policy_modules(module_names: Sequence[str]) -> None:
    for module_name in module_names:
        text = str(module_name).strip()
        if not text:
            continue
        if not all(part.isidentifier() for part in text.split(".")):
            raise ValueError(
                f"Invalid training policy module name '{text}'; expected a "
                "dotted Python module path."
            )
        importlib.import_module(text)


def validate_training_policy_configs(config: Any) -> list[str]:
    from mirai.core.builtins import register_builtin_components

    register_builtin_components()
    load_training_policy_modules(getattr(config.training, "policy_modules", ()))
    errors: list[str] = []
    configured_options = getattr(config.training, "policy_options", {})
    unknown_options = sorted(
        set(str(name).strip().lower() for name in configured_options)
        - set(TrainingPolicyRegistry.names())
    )
    for name in unknown_options:
        errors.append(f"training.policy_options declares unknown policy '{name}'.")
    for name, plugin in TrainingPolicyRegistry.items():
        for error in plugin.validate_config(config):
            errors.append(f"training policy '{name}': {error}")
    return errors


class TrainingPolicySet:
    """Compose active policies while rejecting ambiguous hook ownership."""

    def __init__(self, policies: Sequence[TrainingPolicy] = ()) -> None:
        ordered = sorted(
            policies,
            key=lambda policy: (int(policy.priority), str(policy.name)),
        )
        names = [str(policy.name).strip().lower() for policy in ordered]
        if any(not name for name in names):
            raise ValueError("Every active training policy must have a non-empty name.")
        if len(set(names)) != len(names):
            raise ValueError("Active training policy names must be unique.")
        self._policies = tuple(ordered)

    @classmethod
    def from_config(cls, config: Any) -> TrainingPolicySet:
        from mirai.core.builtins import register_builtin_components

        register_builtin_components()
        load_training_policy_modules(getattr(config.training, "policy_modules", ()))
        active: list[TrainingPolicy] = []
        for _name, plugin in TrainingPolicyRegistry.items():
            policy = plugin.build(config)
            if policy is not None:
                active.append(policy)
        return cls(active)

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(str(policy.name) for policy in self._policies)

    def configure_pipeline(self, pipeline: Any) -> None:
        for policy in self._policies:
            policy.configure_pipeline(pipeline)

    def validation_errors(self, records: Sequence[Any]) -> list[str]:
        errors: list[str] = []
        for policy in self._policies:
            for error in policy.validate_records(records):
                errors.append(f"training policy '{policy.name}': {error}")
        return errors

    def select_records(
        self, context: RecordSelectionContext
    ) -> list[Any] | None:
        claims: list[tuple[str, list[Any]]] = []
        for policy in self._policies:
            selected = policy.select_records(context)
            if selected is not None:
                claims.append((str(policy.name), list(selected)))
        if len(claims) > 1:
            owners = ", ".join(name for name, _records in claims)
            raise ValueError(
                "Multiple training policies claimed record selection: " + owners
            )
        return claims[0][1] if claims else None

    def run_previews(
        self, request: PreviewRequestContext
    ) -> list[Any] | None:
        claims: list[tuple[str, list[Any]]] = []
        for policy in self._policies:
            produced = policy.run_previews(request)
            if produced is not None:
                claims.append((str(policy.name), list(produced)))
        if len(claims) > 1:
            owners = ", ".join(name for name, _results in claims)
            raise ValueError(
                "Multiple training policies claimed preview generation: " + owners
            )
        return claims[0][1] if claims else None

    def augment_batch(
        self,
        batch: dict[str, Any],
        *,
        records: Sequence[Any],
        step: int,
        training: bool,
        global_batch_index: int | None = None,
    ) -> dict[str, Any]:
        out = dict(batch)
        owners: dict[str, str] = {}
        for policy in self._policies:
            additions = dict(
                policy.augment_batch(
                    BatchAugmentContext(
                        records=records,
                        batch=out,
                        step=int(step),
                        training=bool(training),
                        global_batch_index=(
                            None
                            if global_batch_index is None
                            else int(global_batch_index)
                        ),
                    )
                )
            )
            for key, value in additions.items():
                if key in out:
                    owner = owners.get(key, "base batch")
                    raise ValueError(
                        f"Training policy '{policy.name}' cannot write batch key "
                        f"'{key}'; it is already owned by {owner}."
                    )
                out[key] = value
                owners[key] = f"training policy '{policy.name}'"
        return out

    def before_forward(
        self,
        pipeline: Any,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> None:
        for policy in self._policies:
            policy.before_forward(pipeline, batch, training=bool(training))

    def before_optimizer_step(self, optimizer: Any) -> None:
        for policy in self._policies:
            policy.before_optimizer_step(optimizer)

    def after_optimizer_step(self, optimizer: Any, *, applied: bool) -> None:
        for policy in self._policies:
            policy.after_optimizer_step(optimizer, applied=bool(applied))

    def state_dict(self) -> dict[str, Any]:
        states: dict[str, dict[str, Any]] = {}
        for policy in self._policies:
            state = dict(policy.state_dict())
            if state:
                states[str(policy.name)] = state
        return {
            "active": list(self.active_names),
            "state": states,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        expected = list(self.active_names)
        observed = [str(name) for name in payload.get("active", ())]
        if observed != expected:
            raise ValueError(
                "Training policy checkpoint mismatch: "
                f"checkpoint={observed}, runtime={expected}."
            )
        states = payload.get("state", {})
        if not isinstance(states, Mapping):
            raise ValueError("Training policy checkpoint state must be a mapping.")
        for policy in self._policies:
            state = states.get(str(policy.name), {})
            if not isinstance(state, Mapping):
                raise ValueError(
                    f"Training policy '{policy.name}' state must be a mapping."
                )
            policy.load_state_dict(state)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "active": list(self.active_names),
            "policies": {
                str(policy.name): dict(policy.checkpoint_metadata())
                for policy in self._policies
            },
        }
