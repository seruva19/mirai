"""Router-only post-compression calibration for diffusion MoE models.

Router KD updates only student routers by matching the original model's final
output distribution: https://arxiv.org/abs/2603.02217, Section 4, Eq. (1)-(3).
Video diffusion has a continuous prediction tensor rather than vocabulary
logits, so this objective minimizes prediction MSE on exactly paired
model inputs. It deliberately does not require teacher and student expert sets
or router dimensions to match.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ROUTER_KD_REPAIR_SCHEMA = "mirai.router_kd_repair"
ROUTER_KD_REPAIR_SCHEMA_VERSION = 1
ROUTER_KD_REPAIR_METADATA_KEY = "mirai_router_kd_repair"


@dataclass(frozen=True)
class RouterRepairTarget:
    """One provider-owned router parameter with a stable artifact name."""

    name: str
    parameter: Any

    def validate(self) -> "RouterRepairTarget":
        if not str(self.name).strip():
            raise ValueError("Router repair target name cannot be empty.")
        if torch is None or not isinstance(self.parameter, torch.nn.Parameter):
            raise TypeError("Router repair targets must reference torch Parameters.")
        if self.parameter.ndim < 1 or self.parameter.numel() <= 0:
            raise ValueError(
                f"Router repair target {self.name!r} must be a non-empty tensor."
            )
        return self


@dataclass(frozen=True)
class RouterRepairLineage:
    dataset_snapshot_id: str
    teacher_model_snapshot_id: str
    config_snapshot_id: str
    compressed_artifact_fingerprint: str
    initial_router_fingerprint: str
    repaired_router_fingerprint: str

    def validate(self) -> "RouterRepairLineage":
        values = {
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "teacher_model_snapshot_id": self.teacher_model_snapshot_id,
            "config_snapshot_id": self.config_snapshot_id,
            "compressed_artifact_fingerprint": self.compressed_artifact_fingerprint,
            "initial_router_fingerprint": self.initial_router_fingerprint,
            "repaired_router_fingerprint": self.repaired_router_fingerprint,
        }
        missing = [name for name, value in values.items() if not str(value).strip()]
        if missing:
            raise ValueError(
                "Router repair lineage is incomplete: " + ", ".join(sorted(missing))
            )
        for name in (
            "compressed_artifact_fingerprint",
            "initial_router_fingerprint",
            "repaired_router_fingerprint",
        ):
            if not str(values[name]).startswith("sha256:"):
                raise ValueError(f"Router repair {name} must use a sha256 fingerprint.")
        return self

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "dataset_snapshot_id": str(self.dataset_snapshot_id),
            "teacher_model_snapshot_id": str(self.teacher_model_snapshot_id),
            "config_snapshot_id": str(self.config_snapshot_id),
            "compressed_artifact_fingerprint": str(
                self.compressed_artifact_fingerprint
            ),
            "initial_router_fingerprint": str(self.initial_router_fingerprint),
            "repaired_router_fingerprint": str(self.repaired_router_fingerprint),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouterRepairLineage":
        return cls(
            dataset_snapshot_id=str(payload.get("dataset_snapshot_id", "")),
            teacher_model_snapshot_id=str(
                payload.get("teacher_model_snapshot_id", "")
            ),
            config_snapshot_id=str(payload.get("config_snapshot_id", "")),
            compressed_artifact_fingerprint=str(
                payload.get("compressed_artifact_fingerprint", "")
            ),
            initial_router_fingerprint=str(
                payload.get("initial_router_fingerprint", "")
            ),
            repaired_router_fingerprint=str(
                payload.get("repaired_router_fingerprint", "")
            ),
        ).validate()


@dataclass(frozen=True)
class RouterRepairArtifact:
    tensors: dict[str, Any]
    lineage: RouterRepairLineage
    calibration_steps: int
    baseline_holdout_mse: float
    repaired_holdout_mse: float

    def validate(self) -> "RouterRepairArtifact":
        _validate_tensor_mapping(self.tensors)
        self.lineage.validate()
        if int(self.calibration_steps) < 0:
            raise ValueError("Router repair calibration_steps cannot be negative.")
        for name, value in (
            ("baseline_holdout_mse", self.baseline_holdout_mse),
            ("repaired_holdout_mse", self.repaired_holdout_mse),
        ):
            if not (float(value) >= 0.0 and float(value) < float("inf")):
                raise ValueError(f"Router repair {name} must be finite and non-negative.")
        if (
            int(self.calibration_steps) > 0
            and float(self.repaired_holdout_mse)
            > float(self.baseline_holdout_mse) + 1e-12
        ):
            raise ValueError(
                "Router repair artifact cannot regress held-out teacher-prediction MSE."
            )
        observed = router_tensor_fingerprint(self.tensors)
        if observed != self.lineage.repaired_router_fingerprint:
            raise ValueError("Router repair payload fingerprint does not match lineage.")
        return self


def _validate_tensor_mapping(tensors: Mapping[str, Any]) -> None:
    if not tensors:
        raise ValueError("Router repair requires at least one router tensor.")
    for raw_name, tensor in tensors.items():
        name = str(raw_name)
        if not name or name != raw_name:
            raise ValueError("Router repair tensor names must be non-empty strings.")
        if torch is None or not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Router repair tensor {name!r} is not a torch Tensor.")
        if tensor.numel() <= 0:
            raise ValueError(f"Router repair tensor {name!r} is empty.")
        if not torch.isfinite(tensor.detach().float()).all():
            raise ValueError(f"Router repair tensor {name!r} contains non-finite values.")


def normalize_router_repair_targets(
    targets: Mapping[str, RouterRepairTarget],
) -> dict[str, RouterRepairTarget]:
    if not targets:
        raise ValueError("Model provider returned no router repair targets.")
    normalized: dict[str, RouterRepairTarget] = {}
    identities: set[int] = set()
    for raw_name, target in targets.items():
        if not isinstance(target, RouterRepairTarget):
            raise TypeError("Router repair targets must use RouterRepairTarget.")
        target.validate()
        name = str(raw_name)
        if name != target.name or name in normalized:
            raise ValueError("Router repair target names must match and be unique.")
        identity = id(target.parameter)
        if identity in identities:
            raise ValueError("A router parameter cannot have multiple repair names.")
        identities.add(identity)
        normalized[name] = target
    return dict(sorted(normalized.items()))


def router_target_tensors(
    targets: Mapping[str, RouterRepairTarget],
) -> dict[str, Any]:
    normalized = normalize_router_repair_targets(targets)
    return {
        name: target.parameter.detach().to(device="cpu").contiguous().clone()
        for name, target in normalized.items()
    }


def router_tensor_fingerprint(tensors: Mapping[str, Any]) -> str:
    _validate_tensor_mapping(tensors)
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(int(dim) for dim in value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def module_state_fingerprint(
    module: Any,
    *,
    excluded_parameter_ids: set[int] | frozenset[int] = frozenset(),
) -> str:
    """Hash model state while excluding only the explicitly trainable routers."""

    excluded_names = {
        name
        for name, parameter in module.named_parameters()
        if id(parameter) in excluded_parameter_ids
    }
    digest = hashlib.sha256()
    state = module.state_dict()
    for name, tensor in sorted(state.items()):
        if name in excluded_names:
            continue
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(int(dim) for dim in value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def configure_router_only_trainability(
    module: Any,
    targets: Mapping[str, RouterRepairTarget],
) -> dict[int, bool]:
    """Freeze the complete student and enable gradients only for target routers."""

    normalized = normalize_router_repair_targets(targets)
    target_ids = {id(target.parameter) for target in normalized.values()}
    all_parameters = list(module.parameters())
    if not target_ids.issubset({id(parameter) for parameter in all_parameters}):
        raise ValueError("Router repair target is not owned by the student model.")
    previous = {id(parameter): bool(parameter.requires_grad) for parameter in all_parameters}
    for parameter in all_parameters:
        parameter.requires_grad_(id(parameter) in target_ids)
        parameter.grad = None
    return previous


def restore_trainability(module: Any, previous: Mapping[int, bool]) -> None:
    for parameter in module.parameters():
        identity = id(parameter)
        if identity not in previous:
            raise ValueError("Student parameter topology changed during router repair.")
        parameter.requires_grad_(bool(previous[identity]))
        parameter.grad = None


def diffusion_router_kd_loss(
    student_prediction: Any,
    teacher_prediction: Any,
    *,
    loss_mask: Any | None = None,
) -> Any:
    """Continuous-output Router-KD objective for a video diffusion prediction."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Router repair requires torch.")
    if not isinstance(student_prediction, torch.Tensor) or not isinstance(
        teacher_prediction, torch.Tensor
    ):
        raise TypeError("Router KD predictions must be torch tensors.")
    if student_prediction.shape != teacher_prediction.shape:
        raise ValueError(
            "Router KD teacher/student final prediction shapes must match."
        )
    error = (
        student_prediction.float() - teacher_prediction.detach().float()
    ).square()
    if loss_mask is None:
        return error.mean()
    mask = torch.as_tensor(
        loss_mask,
        device=error.device,
        dtype=error.dtype,
    )
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    try:
        weighted = error * mask
    except RuntimeError as exc:
        raise ValueError("Router KD loss_mask is not broadcast-compatible.") from exc
    denominator = mask.expand_as(error).sum().clamp_min(1.0)
    return weighted.sum() / denominator


def save_router_repair_artifact(
    path: str | Path,
    artifact: RouterRepairArtifact,
    *,
    overwrite: bool = False,
) -> Path:
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Router repair artifacts require safetensors.") from exc

    artifact.validate()
    output = Path(path)
    if output.exists() and not overwrite:
        raise ValueError(f"Router repair artifact already exists: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": ROUTER_KD_REPAIR_SCHEMA,
        "schema_version": ROUTER_KD_REPAIR_SCHEMA_VERSION,
        "method": "teacher_student_final_prediction_mse_router_only",
        "calibration_steps": int(artifact.calibration_steps),
        "baseline_holdout_mse": float(artifact.baseline_holdout_mse),
        "repaired_holdout_mse": float(artifact.repaired_holdout_mse),
        "lineage": artifact.lineage.to_dict(),
        "tensor_names": sorted(artifact.tensors),
    }
    save_file(
        {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in sorted(artifact.tensors.items())
        },
        str(output),
        metadata={
            ROUTER_KD_REPAIR_METADATA_KEY: json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )
    return output


def load_router_repair_artifact(path: str | Path) -> RouterRepairArtifact:
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Router repair artifacts require safetensors.") from exc

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Router repair artifact was not found: {source}.")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        raw_manifest = (handle.metadata() or {}).get(ROUTER_KD_REPAIR_METADATA_KEY)
    if not raw_manifest:
        raise ValueError("Router repair artifact metadata is missing.")
    manifest = json.loads(raw_manifest)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != ROUTER_KD_REPAIR_SCHEMA
        or int(manifest.get("schema_version", 0))
        != ROUTER_KD_REPAIR_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported router repair artifact schema.")
    tensors = dict(load_file(str(source), device="cpu"))
    expected_names = [str(name) for name in manifest.get("tensor_names", ())]
    if sorted(tensors) != sorted(expected_names):
        raise ValueError("Router repair tensor inventory does not match metadata.")
    return RouterRepairArtifact(
        tensors=tensors,
        lineage=RouterRepairLineage.from_dict(manifest.get("lineage", {})),
        calibration_steps=int(manifest.get("calibration_steps", -1)),
        baseline_holdout_mse=float(manifest.get("baseline_holdout_mse", -1.0)),
        repaired_holdout_mse=float(manifest.get("repaired_holdout_mse", -1.0)),
    ).validate()


def apply_router_repair_artifact(
    targets: Mapping[str, RouterRepairTarget],
    artifact: RouterRepairArtifact,
    *,
    compressed_artifact_fingerprint: str,
) -> None:
    normalized = normalize_router_repair_targets(targets)
    artifact.validate()
    if (
        str(compressed_artifact_fingerprint)
        != artifact.lineage.compressed_artifact_fingerprint
    ):
        raise ValueError("Router repair artifact belongs to a different compressed base.")
    current = router_target_tensors(normalized)
    if router_tensor_fingerprint(current) != artifact.lineage.initial_router_fingerprint:
        raise ValueError("Router repair artifact initial router lineage does not match.")
    if set(current) != set(artifact.tensors):
        raise ValueError("Router repair artifact target inventory changed.")
    for name, target in normalized.items():
        value = artifact.tensors[name]
        if tuple(value.shape) != tuple(target.parameter.shape):
            raise ValueError(f"Router repair target {name!r} shape changed.")
    with torch.no_grad():
        for name, target in normalized.items():
            target.parameter.copy_(
                artifact.tensors[name].to(
                    device=target.parameter.device,
                    dtype=target.parameter.dtype,
                )
            )


__all__ = [
    "ROUTER_KD_REPAIR_METADATA_KEY",
    "ROUTER_KD_REPAIR_SCHEMA",
    "ROUTER_KD_REPAIR_SCHEMA_VERSION",
    "RouterRepairArtifact",
    "RouterRepairLineage",
    "RouterRepairTarget",
    "apply_router_repair_artifact",
    "configure_router_only_trainability",
    "diffusion_router_kd_loss",
    "load_router_repair_artifact",
    "module_state_fingerprint",
    "normalize_router_repair_targets",
    "restore_trainability",
    "router_target_tensors",
    "router_tensor_fingerprint",
    "save_router_repair_artifact",
]
