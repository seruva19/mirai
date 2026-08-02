"""Two-phase, bounded-memory Router-KD calibration session."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.calibration.router_repair import (
    RouterRepairArtifact,
    RouterRepairLineage,
    configure_router_only_trainability,
    diffusion_router_kd_loss,
    module_state_fingerprint,
    normalize_router_repair_targets,
    restore_trainability,
    router_target_tensors,
    router_tensor_fingerprint,
)
from mirai.core.training.lifecycle.training_step_pre import (
    _build_training_batch_factory,
)
from mirai.core.training.lifecycle.training_step_pre import (
    resolve_step_sampling_context,
)
from mirai.core.training.strategies.base import TrainingInputs

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ROUTER_KD_EXAMPLE_SCHEMA = "mirai.router_kd_example"
ROUTER_KD_EXAMPLE_SCHEMA_VERSION = 1
ROUTER_KD_EXAMPLE_METADATA_KEY = "mirai_router_kd_example"


@dataclass(frozen=True)
class RouterKDFitReport:
    artifact: RouterRepairArtifact
    non_router_fingerprint: str
    train_examples: int
    holdout_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_steps": int(self.artifact.calibration_steps),
            "train_examples": int(self.train_examples),
            "holdout_examples": int(self.holdout_examples),
            "baseline_holdout_mse": float(self.artifact.baseline_holdout_mse),
            "repaired_holdout_mse": float(self.artifact.repaired_holdout_mse),
            "initial_router_fingerprint": (
                self.artifact.lineage.initial_router_fingerprint
            ),
            "repaired_router_fingerprint": (
                self.artifact.lineage.repaired_router_fingerprint
            ),
            "non_router_fingerprint": str(self.non_router_fingerprint),
        }


class _TensorTreeCodec:
    def __init__(self) -> None:
        self.tensors: dict[str, Any] = {}

    def encode(self, value: Any) -> dict[str, Any]:
        if torch is not None and isinstance(value, torch.Tensor):
            key = f"tensor_{len(self.tensors):06d}"
            self.tensors[key] = value.detach().to(device="cpu").contiguous()
            return {"kind": "tensor", "key": key}
        if isinstance(value, Mapping):
            items: list[list[Any]] = []
            for raw_key, child in value.items():
                key = str(raw_key)
                if key != raw_key:
                    raise TypeError("Router-KD mapping keys must be strings.")
                items.append([key, self.encode(child)])
            return {"kind": "mapping", "items": items}
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [self.encode(child) for child in value]}
        if isinstance(value, list):
            return {"kind": "list", "items": [self.encode(child) for child in value]}
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"kind": "scalar", "value": value}
        raise TypeError(
            f"Router-KD example cannot serialize {type(value).__name__}."
        )

    @staticmethod
    def decode(node: Mapping[str, Any], tensors: Mapping[str, Any]) -> Any:
        kind = str(node.get("kind", ""))
        if kind == "tensor":
            key = str(node.get("key", ""))
            if key not in tensors:
                raise KeyError(f"Router-KD example tensor {key!r} is missing.")
            return tensors[key]
        if kind == "mapping":
            raw_items = node.get("items")
            if not isinstance(raw_items, list):
                raise ValueError("Router-KD mapping descriptor is invalid.")
            return {
                str(key): _TensorTreeCodec.decode(child, tensors)
                for key, child in raw_items
            }
        if kind in {"tuple", "list"}:
            raw_items = node.get("items")
            if not isinstance(raw_items, list):
                raise ValueError("Router-KD sequence descriptor is invalid.")
            values = [_TensorTreeCodec.decode(child, tensors) for child in raw_items]
            return tuple(values) if kind == "tuple" else values
        if kind == "scalar":
            value = node.get("value")
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise ValueError("Router-KD scalar descriptor is invalid.")
            return value
        raise ValueError(f"Unsupported Router-KD tensor-tree node {kind!r}.")


def _inputs_payload(inputs: TrainingInputs) -> dict[str, Any]:
    return {
        "noisy_latents": inputs.noisy_latents,
        "timestep": inputs.timestep,
        "noise": inputs.noise,
        "clean_latents": inputs.clean_latents,
        "text_embeds": inputs.text_embeds,
        "objective_timestep": inputs.objective_timestep,
        "loss_mask": inputs.loss_mask,
        "extra_forward_kwargs": inputs.extra_forward_kwargs,
    }


def _payload_inputs(payload: Mapping[str, Any]) -> TrainingInputs:
    required = {
        "noisy_latents",
        "timestep",
        "noise",
        "clean_latents",
        "text_embeds",
        "objective_timestep",
        "loss_mask",
        "extra_forward_kwargs",
    }
    if set(payload) != required:
        raise ValueError("Router-KD example input fields changed.")
    if not isinstance(payload["text_embeds"], Mapping) or not isinstance(
        payload["extra_forward_kwargs"], Mapping
    ):
        raise TypeError("Router-KD text_embeds/extra_forward_kwargs must be mappings.")
    return TrainingInputs(
        noisy_latents=payload["noisy_latents"],
        timestep=payload["timestep"],
        noise=payload["noise"],
        clean_latents=payload["clean_latents"],
        text_embeds=dict(payload["text_embeds"]),
        objective_timestep=payload["objective_timestep"],
        loss_mask=payload["loss_mask"],
        extra_forward_kwargs=dict(payload["extra_forward_kwargs"]),
    )


def save_router_kd_example(
    path: str | Path,
    *,
    inputs: TrainingInputs,
    teacher_prediction: Any,
) -> Path:
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Router-KD examples require safetensors.") from exc
    if torch is None or not isinstance(teacher_prediction, torch.Tensor):
        raise TypeError("Router-KD teacher prediction must be a torch tensor.")
    codec = _TensorTreeCodec()
    descriptor = codec.encode(
        {
            "inputs": _inputs_payload(inputs),
            "teacher_prediction": teacher_prediction,
        }
    )
    output = Path(path)
    if output.exists():
        raise ValueError(f"Router-KD example already exists: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        codec.tensors,
        str(output),
        metadata={
            ROUTER_KD_EXAMPLE_METADATA_KEY: json.dumps(
                {
                    "schema": ROUTER_KD_EXAMPLE_SCHEMA,
                    "schema_version": ROUTER_KD_EXAMPLE_SCHEMA_VERSION,
                    "tree": descriptor,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )
    return output


def load_router_kd_example(path: str | Path) -> tuple[TrainingInputs, Any]:
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Router-KD examples require safetensors.") from exc
    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        raw = (handle.metadata() or {}).get(ROUTER_KD_EXAMPLE_METADATA_KEY)
    if not raw:
        raise ValueError("Router-KD example metadata is missing.")
    manifest = json.loads(raw)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != ROUTER_KD_EXAMPLE_SCHEMA
        or int(manifest.get("schema_version", 0))
        != ROUTER_KD_EXAMPLE_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported Router-KD example schema.")
    payload = _TensorTreeCodec.decode(
        manifest.get("tree", {}),
        load_file(str(source), device="cpu"),
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Router-KD example root must be a mapping.")
    prediction = payload.get("teacher_prediction")
    if torch is None or not isinstance(prediction, torch.Tensor):
        raise TypeError("Router-KD example has no teacher prediction tensor.")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("Router-KD example has no input mapping.")
    return _payload_inputs(inputs), prediction


def _move_tree(value: Any, device: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_tree(child, device) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tree(child, device) for child in value)
    if isinstance(value, list):
        return [_move_tree(child, device) for child in value]
    return value


def _move_inputs(inputs: TrainingInputs, device: Any) -> TrainingInputs:
    return _payload_inputs(_move_tree(_inputs_payload(inputs), device))


def capture_router_kd_examples(
    session: Any,
    *,
    output_dir: str | Path,
    example_count: int,
) -> list[Path]:
    """Capture teacher predictions and exact replay inputs one shard at a time."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Router-KD capture requires torch.")
    count = int(example_count)
    if count <= 0:
        raise ValueError("Router-KD capture requires example_count > 0.")
    trainer = session.trainer
    training_model = trainer.pipeline.get_training_model()
    if training_model is None:
        raise ValueError("Router-KD capture requires an exposed training model.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Router-KD capture directory must be empty.")
    was_training = bool(getattr(training_model, "training", False))
    try:
        training_model.eval()
        sampling_context = resolve_step_sampling_context(session)
        build_batch = _build_training_batch_factory(
            session=session,
            sampling_context=sampling_context,
        )
        paths: list[Path] = []
        with torch.no_grad():
            for step in range(count):
                inputs = trainer.prepare_calibration_inputs(
                    build_batch(step),
                    training=False,
                )
                prediction = trainer.predict_calibration_inputs(inputs)
                paths.append(
                    save_router_kd_example(
                        destination / f"example-{step:06d}.safetensors",
                        inputs=inputs,
                        teacher_prediction=prediction,
                    )
                )
        return paths
    finally:
        training_model.train(was_training)


def _evaluate_examples(
    trainer: Any,
    paths: list[Path],
    *,
    device: Any,
) -> float:
    if not paths:
        raise ValueError("Router-KD held-out evaluation requires examples.")
    total = 0.0
    with torch.no_grad():
        for path in paths:
            inputs, teacher_prediction = load_router_kd_example(path)
            inputs = _move_inputs(inputs, device)
            teacher_prediction = teacher_prediction.to(device=device)
            student_prediction = trainer.predict_calibration_inputs(inputs)
            total += float(
                diffusion_router_kd_loss(
                    student_prediction,
                    teacher_prediction,
                    loss_mask=inputs.loss_mask,
                ).item()
            )
    return total / float(len(paths))


def fit_router_kd_session(
    session: Any,
    *,
    example_paths: list[str | Path],
    train_examples: int,
    learning_rate: float,
    gradient_accumulation: int,
    compressed_artifact_fingerprint: str,
    teacher_model_snapshot_id: str,
) -> RouterKDFitReport:
    """Fit FP32 router masters and reject held-out output regression."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Router-KD fitting requires torch.")
    paths = [Path(path) for path in example_paths]
    train_count = int(train_examples)
    if train_count < 0 or train_count >= len(paths):
        raise ValueError(
            "Router-KD requires 0 <= train_examples < total examples."
        )
    if not (float(learning_rate) > 0.0):
        raise ValueError("Router-KD learning_rate must be positive.")
    accumulation = int(gradient_accumulation)
    if accumulation <= 0:
        raise ValueError("Router-KD gradient_accumulation must be positive.")

    config = session.config
    provider = get_model_family_provider(str(config.model.type))
    if provider is None or not provider.supports_post_compression_router_repair(
        config
    ):
        raise ValueError(
            f"Model provider {config.model.type!r} does not support Router-KD repair."
        )
    targets = normalize_router_repair_targets(
        provider.build_router_repair_targets(session.trainer.pipeline)
    )
    model = session.trainer.pipeline.get_training_model()
    if model is None:
        raise ValueError("Router-KD fitting requires an exposed training model.")
    target_ids = {id(target.parameter) for target in targets.values()}
    initial_tensors = router_target_tensors(targets)
    initial_router_fingerprint = router_tensor_fingerprint(initial_tensors)
    previous_trainability = configure_router_only_trainability(model, targets)
    non_router_before = module_state_fingerprint(
        model,
        excluded_parameter_ids=target_ids,
    )
    was_training = bool(getattr(model, "training", False))
    device = session.compute_device

    masters = {
        name: torch.nn.Parameter(
            target.parameter.detach().to(device=device, dtype=torch.float32).clone()
        )
        for name, target in targets.items()
    }
    optimizer = torch.optim.AdamW(
        list(masters.values()),
        lr=float(learning_rate),
        weight_decay=0.0,
    )

    def copy_masters_to_working() -> None:
        with torch.no_grad():
            for name, target in targets.items():
                target.parameter.copy_(
                    masters[name].to(
                        device=target.parameter.device,
                        dtype=target.parameter.dtype,
                    )
                )

    try:
        model.eval()
        copy_masters_to_working()
        train_paths = paths[:train_count]
        holdout_paths = paths[train_count:]
        baseline = _evaluate_examples(
            session.trainer,
            holdout_paths,
            device=device,
        )
        for group_start in range(0, len(train_paths), accumulation):
            group = train_paths[group_start : group_start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for path in group:
                copy_masters_to_working()
                inputs, teacher_prediction = load_router_kd_example(path)
                inputs = _move_inputs(inputs, device)
                teacher_prediction = teacher_prediction.to(device=device)
                prediction = session.trainer.predict_calibration_inputs(inputs)
                loss = diffusion_router_kd_loss(
                    prediction,
                    teacher_prediction,
                    loss_mask=inputs.loss_mask,
                ) / float(len(group))
                loss.backward()
                session.trainer.pipeline.finish_backward_offloads()
                for name, target in targets.items():
                    gradient = target.parameter.grad
                    if gradient is None:
                        raise RuntimeError(
                            f"Router-KD target {name!r} received no gradient."
                        )
                    contribution = gradient.detach().to(
                        device=device,
                        dtype=torch.float32,
                    )
                    if masters[name].grad is None:
                        masters[name].grad = contribution.clone()
                    else:
                        masters[name].grad.add_(contribution)
                    target.parameter.grad = None
            session.trainer.pipeline.prepare_optimizer_step()
            optimizer.step()
            copy_masters_to_working()
            session.trainer.pipeline.finish_optimizer_step()
        repaired = _evaluate_examples(
            session.trainer,
            holdout_paths,
            device=device,
        )
        if train_count > 0 and repaired > baseline + 1e-12:
            with torch.no_grad():
                for name, target in targets.items():
                    target.parameter.copy_(
                        initial_tensors[name].to(
                            device=target.parameter.device,
                            dtype=target.parameter.dtype,
                        )
                    )
            raise RuntimeError(
                "Router-KD increased held-out teacher-prediction MSE; "
                "the repair artifact was not emitted."
            )
        non_router_after = module_state_fingerprint(
            model,
            excluded_parameter_ids=target_ids,
        )
        if non_router_after != non_router_before:
            raise RuntimeError("Router-KD changed non-router model state.")
        repaired_tensors = router_target_tensors(targets)
        manifest = session.manifest
        artifact = RouterRepairArtifact(
            tensors=repaired_tensors,
            lineage=RouterRepairLineage(
                dataset_snapshot_id=str(manifest.dataset_snapshot_id),
                teacher_model_snapshot_id=str(teacher_model_snapshot_id),
                config_snapshot_id=str(manifest.config_snapshot_id),
                compressed_artifact_fingerprint=str(
                    compressed_artifact_fingerprint
                ),
                initial_router_fingerprint=initial_router_fingerprint,
                repaired_router_fingerprint=router_tensor_fingerprint(
                    repaired_tensors
                ),
            ),
            calibration_steps=train_count,
            baseline_holdout_mse=baseline,
            repaired_holdout_mse=repaired,
        ).validate()
        return RouterKDFitReport(
            artifact=artifact,
            non_router_fingerprint=non_router_after,
            train_examples=train_count,
            holdout_examples=len(holdout_paths),
        )
    finally:
        restore_trainability(model, previous_trainability)
        model.train(was_training)


__all__ = [
    "ROUTER_KD_EXAMPLE_METADATA_KEY",
    "ROUTER_KD_EXAMPLE_SCHEMA",
    "ROUTER_KD_EXAMPLE_SCHEMA_VERSION",
    "RouterKDFitReport",
    "capture_router_kd_examples",
    "fit_router_kd_session",
    "load_router_kd_example",
    "save_router_kd_example",
]
