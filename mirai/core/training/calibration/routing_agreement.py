"""Provider-driven train-versus-inference routing agreement evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.monitoring.agreement import RoutingAgreementAccumulator
from mirai.core.moe.monitoring.agreement import RoutingSelectionCapture
from mirai.core.moe.monitoring.agreement import RoutingSelectionTarget
from mirai.core.moe.monitoring.agreement import (
    build_routing_mode_agreement_evidence,
)
from mirai.core.moe.monitoring.agreement import compare_routing_capture_pairs
from mirai.core.moe.monitoring.agreement import (
    save_routing_mode_agreement_evidence,
)
from mirai.core.training.lifecycle.session_state import capture_torch_rng_state
from mirai.core.training.lifecycle.session_state import restore_torch_rng_state
from mirai.core.training.lifecycle.training_step_pre import (
    _build_training_batch_factory,
)
from mirai.core.training.lifecycle.training_step_pre import (
    resolve_step_sampling_context,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RoutingModeAgreementRunReport:
    output_path: str
    calibration_steps: int
    modules: dict[str, dict[str, Any]]
    overall: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "output": self.output_path,
            "calibration_steps": self.calibration_steps,
            "modules": self.modules,
            "overall": self.overall,
        }


def _copy_sampling_state(trainer: Any) -> dict[str, Any]:
    getter = getattr(trainer, "_sampling_state_dict", None)
    if not callable(getter):
        raise TypeError("Routing agreement requires trainer sampling-state support.")
    return deepcopy(getter())


def _restore_sampling_state(trainer: Any, state: dict[str, Any]) -> None:
    loader = getattr(trainer, "_load_sampling_state_dict", None)
    if not callable(loader):
        raise TypeError("Routing agreement requires trainer sampling-state support.")
    loader(deepcopy(state))


def _copy_policy_state(trainer: Any) -> dict[str, Any]:
    policies = getattr(trainer, "training_policies", None)
    state_dict = getattr(policies, "state_dict", None)
    return deepcopy(state_dict()) if callable(state_dict) else {}


def _restore_policy_state(trainer: Any, state: dict[str, Any]) -> None:
    policies = getattr(trainer, "training_policies", None)
    loader = getattr(policies, "load_state_dict", None)
    if callable(loader):
        loader(deepcopy(state))
    elif state:
        raise TypeError("Routing agreement cannot restore training-policy state.")


def _validate_targets(
    raw_targets: dict[str, RoutingSelectionTarget],
) -> dict[str, RoutingSelectionTarget]:
    if not raw_targets:
        raise ValueError("Model provider returned no routing agreement targets.")
    targets: dict[str, RoutingSelectionTarget] = {}
    for raw_name, target in raw_targets.items():
        if not isinstance(target, RoutingSelectionTarget):
            raise TypeError(
                "Model provider routing agreement targets must use "
                "RoutingSelectionTarget."
            )
        target.validate()
        name = str(raw_name)
        if name != target.name or name in targets:
            raise ValueError(
                "Routing agreement target names must match and be unique."
            )
        targets[name] = target
    return targets


def _capture_forward(
    trainer: Any,
    targets: dict[str, RoutingSelectionTarget],
    batch: dict[str, Any],
    *,
    training: bool,
) -> dict[str, tuple[Any, ...]]:
    with RoutingSelectionCapture(targets) as capture:
        trainer.compute_loss(batch, training=bool(training))
    return capture.snapshots()


def run_routing_mode_agreement_session(
    session: Any,
    *,
    output_path: str | Path,
    calibration_steps: int,
    overwrite: bool = False,
) -> RoutingModeAgreementRunReport:
    """Compare paired router sets without optimizer, backward, or raw trace output."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("Routing agreement calibration requires torch.")
    config = session.config
    gate = str(
        getattr(config.model.params, "moe_routing_agreement_evidence", "off")
    ).strip().lower()
    if gate != "report":
        raise ValueError(
            "Routing agreement calibration requires "
            "model.params.moe_routing_agreement_evidence='report'."
        )
    steps = int(calibration_steps)
    if steps <= 0:
        raise ValueError("Routing agreement calibration steps must be positive.")
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"Routing agreement output already exists: {output}.")

    provider = get_model_family_provider(str(config.model.type))
    if provider is None or not provider.supports_routing_mode_agreement_evidence(
        config
    ):
        raise ValueError(
            f"Model provider {config.model.type!r} does not support "
            "routing agreement evidence."
        )
    targets = _validate_targets(
        provider.build_routing_mode_agreement_targets(session.trainer.pipeline)
    )
    accumulators = {
        name: RoutingAgreementAccumulator()
        for name in targets
    }

    trainer = session.trainer
    training_model = trainer.pipeline.get_training_model()
    if training_model is None:
        raise ValueError("Routing agreement requires an exposed training model.")
    was_training = bool(getattr(training_model, "training", True))
    original_sampling = _copy_sampling_state(trainer)
    original_torch_rng = capture_torch_rng_state()
    original_policy = _copy_policy_state(trainer)
    session_rng = getattr(session, "rng", None)
    original_session_rng = (
        session_rng.getstate()
        if session_rng is not None and hasattr(session_rng, "getstate")
        else None
    )

    try:
        sampling_context = resolve_step_sampling_context(session)
        build_batch = _build_training_batch_factory(
            session=session,
            sampling_context=sampling_context,
        )
        with torch.no_grad():
            for step in range(steps):
                batch = build_batch(step)
                paired_sampling = _copy_sampling_state(trainer)
                paired_torch_rng = capture_torch_rng_state()
                paired_policy = _copy_policy_state(trainer)

                training_model.train(True)
                reference = _capture_forward(
                    trainer,
                    targets,
                    batch,
                    training=True,
                )
                next_sampling = _copy_sampling_state(trainer)
                next_torch_rng = capture_torch_rng_state()
                next_policy = _copy_policy_state(trainer)

                _restore_sampling_state(trainer, paired_sampling)
                restore_torch_rng_state(paired_torch_rng)
                _restore_policy_state(trainer, paired_policy)
                training_model.train(False)
                candidate = _capture_forward(
                    trainer,
                    targets,
                    batch,
                    training=False,
                )
                compare_routing_capture_pairs(
                    reference,
                    candidate,
                    accumulators,
                )

                _restore_sampling_state(trainer, next_sampling)
                restore_torch_rng_state(next_torch_rng)
                _restore_policy_state(trainer, next_policy)
    finally:
        _restore_sampling_state(trainer, original_sampling)
        restore_torch_rng_state(original_torch_rng)
        _restore_policy_state(trainer, original_policy)
        if (
            session_rng is not None
            and original_session_rng is not None
            and hasattr(session_rng, "setstate")
        ):
            session_rng.setstate(original_session_rng)
        training_model.train(was_training)

    manifest = session.manifest
    evidence = build_routing_mode_agreement_evidence(
        accumulators,
        calibration_steps=steps,
        dataset_snapshot_id=str(manifest.dataset_snapshot_id),
        model_snapshot_id=str(manifest.model_snapshot_id),
        config_snapshot_id=str(manifest.config_snapshot_id),
    )
    save_routing_mode_agreement_evidence(output, evidence)
    return RoutingModeAgreementRunReport(
        output_path=str(output),
        calibration_steps=steps,
        modules=dict(evidence["modules"]),
        overall=dict(evidence["overall"]),
    )


__all__ = [
    "RoutingModeAgreementRunReport",
    "run_routing_mode_agreement_session",
]
