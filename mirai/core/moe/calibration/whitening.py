"""Activation-covariance evidence for truncation-aware expert factorization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


EXPERT_WHITENING_FORMAT = "mirai.moe.expert_whitening"
EXPERT_WHITENING_SCHEMA_VERSION = 1
EXPERT_WHITENING_METADATA_KEY = "mirai_moe_expert_whitening"
_PROJECTIONS = ("w1", "w2", "w3")
_COVARIANCE_GROUP = {"w1": "w13", "w2": "w2", "w3": "w13"}


@dataclass(frozen=True)
class ProjectionCovarianceEvidence:
    """Streaming ``X.T @ X`` summary for one projection input space."""

    covariance: Any
    sample_count: int

    @property
    def input_features(self) -> int:
        value = torch.as_tensor(self.covariance)
        return int(value.shape[0]) if value.ndim == 2 else 0

    def validate(self) -> ProjectionCovarianceEvidence:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Expert whitening calibration requires torch.")
        covariance = torch.as_tensor(self.covariance)
        if (
            covariance.ndim != 2
            or int(covariance.shape[0]) < 1
            or int(covariance.shape[0]) != int(covariance.shape[1])
        ):
            raise ValueError("Projection covariance must be a non-empty square matrix.")
        if int(self.sample_count) <= 0:
            raise ValueError("Projection covariance sample_count must be positive.")
        if not bool(torch.isfinite(covariance).all().item()):
            raise ValueError("Projection covariance contains non-finite values.")
        if bool((covariance.diagonal() < 0).any().item()):
            raise ValueError("Projection covariance has a negative diagonal entry.")
        tolerance = 32.0 * torch.finfo(covariance.dtype).eps
        if not bool(
            torch.allclose(
                covariance,
                covariance.transpose(0, 1),
                rtol=tolerance,
                atol=tolerance,
            )
        ):
            raise ValueError("Projection covariance must be symmetric.")
        return self

    def normalized(self, *, device: Any | None = None, dtype: Any | None = None) -> Any:
        """Return the mean outer product without changing its eigenspaces."""
        self.validate()
        resolved_dtype = torch.float64 if dtype is None else dtype
        return torch.as_tensor(self.covariance).to(
            device=device, dtype=resolved_dtype
        ) / float(self.sample_count)


@dataclass(frozen=True)
class ExpertWhiteningEvidence:
    """Projection-input covariance evidence for one grouped-expert module."""

    projections: Mapping[str, ProjectionCovarianceEvidence]

    def validate(self) -> ExpertWhiteningEvidence:
        if set(self.projections) != set(_PROJECTIONS):
            raise ValueError("Expert whitening evidence requires w1, w2, and w3.")
        validated = {
            key: self.projections[key].validate()
            for key in _PROJECTIONS
        }
        if validated["w1"].input_features != validated["w3"].input_features:
            raise ValueError("w1 and w3 whitening input dimensions must match.")
        if validated["w1"].sample_count != validated["w3"].sample_count:
            raise ValueError("w1 and w3 whitening sample counts must match.")
        if not bool(
            torch.equal(
                torch.as_tensor(validated["w1"].covariance),
                torch.as_tensor(validated["w3"].covariance),
            )
        ):
            raise ValueError("w1 and w3 must share the same routed-input covariance.")
        return self


@dataclass(frozen=True)
class ExpertWhiteningCalibrationTarget:
    """Provider-owned grouped-expert host that exposes projection inputs."""

    name: str
    host: Any
    projection_input_dims: Mapping[str, int]

    def validate(self) -> ExpertWhiteningCalibrationTarget:
        if not self.name:
            raise ValueError("Expert whitening target name must be non-empty.")
        if not callable(
            getattr(self.host, "set_whitening_calibration_observer", None)
        ) or not callable(
            getattr(self.host, "clear_whitening_calibration_observer", None)
        ):
            raise TypeError(
                "Expert whitening target host must expose whitening observer setters."
            )
        dims = {str(key): int(value) for key, value in self.projection_input_dims.items()}
        if set(dims) != set(_PROJECTIONS) or any(value <= 0 for value in dims.values()):
            raise ValueError("Expert whitening target requires positive w1/w2/w3 input dims.")
        if dims["w1"] != dims["w3"]:
            raise ValueError("Expert whitening target w1 and w3 input dims must match.")
        return self

    @property
    def covariance_bytes(self) -> int:
        """FP32 accumulator bytes, counting the shared w1/w3 matrix once."""
        self.validate()
        dims = self.projection_input_dims
        return 4 * (int(dims["w1"]) ** 2 + int(dims["w2"]) ** 2)


class ActivationCovarianceAccumulator:
    """Accumulate routed projection inputs without retaining activation samples."""

    def __init__(self, projection_input_dims: Mapping[str, int]) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Expert whitening calibration requires torch.")
        dims = {str(key): int(value) for key, value in projection_input_dims.items()}
        if set(dims) != set(_PROJECTIONS) or any(value <= 0 for value in dims.values()):
            raise ValueError("Whitening accumulator requires positive w1/w2/w3 input dims.")
        if dims["w1"] != dims["w3"]:
            raise ValueError("Whitening accumulator w1 and w3 input dims must match.")
        self._dims = {"w13": dims["w1"], "w2": dims["w2"]}
        self._covariances: dict[str, Any] = {}
        self._sample_counts = {"w13": 0, "w2": 0}

    @property
    def covariance_bytes(self) -> int:
        return 4 * sum(value * value for value in self._dims.values())

    def record(self, projections: str | tuple[str, ...], inputs: Any) -> None:
        names = (projections,) if isinstance(projections, str) else tuple(projections)
        if not names or any(name not in _PROJECTIONS for name in names):
            raise ValueError("Whitening capture projections must be w1, w2, and/or w3.")
        groups = {_COVARIANCE_GROUP[name] for name in names}
        if len(groups) != 1:
            raise ValueError("One whitening capture cannot mix different input spaces.")
        group = groups.pop()
        tensor = torch.as_tensor(inputs).detach()
        dimension = self._dims[group]
        if tensor.ndim < 2 or int(tensor.shape[-1]) != dimension:
            raise ValueError(
                f"Whitening input for {group} must end in dimension {dimension}, "
                f"got {tuple(tensor.shape)}."
            )
        flat = tensor.reshape(-1, dimension)
        if int(flat.shape[0]) == 0:
            return
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError("Whitening calibration observed non-finite activations.")
        compute = flat.to(dtype=torch.float32)
        update = compute.transpose(0, 1) @ compute
        covariance = self._covariances.get(group)
        if covariance is None:
            covariance = torch.zeros_like(update)
            self._covariances[group] = covariance
        covariance.add_(update)
        self._sample_counts[group] += int(flat.shape[0])

    def evidence(self) -> ExpertWhiteningEvidence:
        items: dict[str, ProjectionCovarianceEvidence] = {}
        for projection in _PROJECTIONS:
            group = _COVARIANCE_GROUP[projection]
            covariance = self._covariances.get(group)
            if covariance is None or self._sample_counts[group] <= 0:
                raise ValueError(
                    f"Whitening calibration observed no {projection} projection inputs."
                )
            items[projection] = ProjectionCovarianceEvidence(
                covariance=covariance.detach().to(device="cpu", dtype=torch.float32),
                sample_count=self._sample_counts[group],
            )
        return ExpertWhiteningEvidence(items).validate()


def save_expert_whitening_evidence(
    path: str | Path,
    evidence_by_module: Mapping[str, ExpertWhiteningEvidence],
    *,
    dataset_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str,
    packed_artifact_fingerprint: str,
) -> None:
    """Persist lineage-bound covariance summaries in a tensor-native artifact."""
    from safetensors.torch import save_file

    lineage = {
        "dataset_snapshot_id": str(dataset_snapshot_id).strip(),
        "model_snapshot_id": str(model_snapshot_id).strip(),
        "config_snapshot_id": str(config_snapshot_id).strip(),
        "packed_artifact_fingerprint": str(packed_artifact_fingerprint).strip(),
    }
    if not all(lineage.values()):
        raise ValueError("Expert whitening evidence requires complete lineage.")
    if not evidence_by_module:
        raise ValueError("Expert whitening evidence cannot be empty.")
    tensors: dict[str, Any] = {}
    modules: list[dict[str, Any]] = []
    for index, (name, raw_evidence) in enumerate(evidence_by_module.items()):
        evidence = raw_evidence.validate()
        prefix = f"module_{index:04d}"
        covariance_names = {
            "w13": f"{prefix}.w13_covariance",
            "w2": f"{prefix}.w2_covariance",
        }
        tensors[covariance_names["w13"]] = torch.as_tensor(
            evidence.projections["w1"].covariance,
            dtype=torch.float32,
        ).cpu().contiguous()
        tensors[covariance_names["w2"]] = torch.as_tensor(
            evidence.projections["w2"].covariance,
            dtype=torch.float32,
        ).cpu().contiguous()
        modules.append(
            {
                "name": str(name),
                "projections": {
                    "w1": {
                        "covariance": covariance_names["w13"],
                        "sample_count": evidence.projections["w1"].sample_count,
                    },
                    "w2": {
                        "covariance": covariance_names["w2"],
                        "sample_count": evidence.projections["w2"].sample_count,
                    },
                    "w3": {
                        "covariance": covariance_names["w13"],
                        "sample_count": evidence.projections["w3"].sample_count,
                    },
                },
            }
        )
    manifest = {
        "format": EXPERT_WHITENING_FORMAT,
        "schema_version": EXPERT_WHITENING_SCHEMA_VERSION,
        **lineage,
        "modules": modules,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            EXPERT_WHITENING_METADATA_KEY: json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            )
        },
    )


def load_expert_whitening_evidence(
    path: str | Path,
    *,
    expected_dataset_snapshot_id: str | None = None,
    expected_model_snapshot_id: str | None = None,
    expected_config_snapshot_id: str | None = None,
    expected_packed_artifact_fingerprint: str | None = None,
) -> tuple[dict[str, ExpertWhiteningEvidence], dict[str, str]]:
    """Load covariance evidence and enforce every requested lineage component."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    input_path = Path(path)
    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        raw_manifest = (handle.metadata() or {}).get(EXPERT_WHITENING_METADATA_KEY)
    if raw_manifest is None:
        raise ValueError("Expert whitening artifact has no Mirai manifest.")
    manifest = json.loads(raw_manifest)
    if manifest.get("format") != EXPERT_WHITENING_FORMAT:
        raise ValueError(f"Unsupported whitening format {manifest.get('format')!r}.")
    if int(manifest.get("schema_version", 0)) != EXPERT_WHITENING_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported whitening schema {manifest.get('schema_version')!r}."
        )
    lineage = {
        key: str(manifest.get(key, ""))
        for key in (
            "dataset_snapshot_id",
            "model_snapshot_id",
            "config_snapshot_id",
            "packed_artifact_fingerprint",
        )
    }
    if not all(lineage.values()):
        raise ValueError("Expert whitening artifact has incomplete lineage.")
    expected = {
        "dataset_snapshot_id": expected_dataset_snapshot_id,
        "model_snapshot_id": expected_model_snapshot_id,
        "config_snapshot_id": expected_config_snapshot_id,
        "packed_artifact_fingerprint": expected_packed_artifact_fingerprint,
    }
    for key, value in expected.items():
        if value is not None and str(value) != lineage[key]:
            raise ValueError(
                f"Expert whitening {key} mismatch: expected {value!r}, "
                f"found {lineage[key]!r}."
            )
    tensors = load_file(str(input_path), device="cpu")
    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("Expert whitening manifest has no modules.")
    loaded: dict[str, ExpertWhiteningEvidence] = {}
    for module in modules:
        name = str(module.get("name", ""))
        if not name or name in loaded:
            raise ValueError("Whitening module names must be non-empty and unique.")
        specs = module.get("projections")
        if not isinstance(specs, Mapping) or set(specs) != set(_PROJECTIONS):
            raise ValueError(f"Whitening module {name!r} has invalid projections.")
        projections: dict[str, ProjectionCovarianceEvidence] = {}
        for projection in _PROJECTIONS:
            spec = specs[projection]
            if not isinstance(spec, Mapping):
                raise ValueError(
                    f"Whitening projection {name}.{projection} must be an object."
                )
            tensor_name = str(spec.get("covariance", ""))
            if tensor_name not in tensors:
                raise KeyError(
                    f"Whitening projection {name}.{projection} is missing covariance."
                )
            projections[projection] = ProjectionCovarianceEvidence(
                covariance=tensors[tensor_name],
                sample_count=int(spec.get("sample_count", 0)),
            )
        loaded[name] = ExpertWhiteningEvidence(projections).validate()
    return loaded, lineage


__all__ = [
    "ActivationCovarianceAccumulator",
    "EXPERT_WHITENING_FORMAT",
    "EXPERT_WHITENING_METADATA_KEY",
    "EXPERT_WHITENING_SCHEMA_VERSION",
    "ExpertWhiteningCalibrationTarget",
    "ExpertWhiteningEvidence",
    "ProjectionCovarianceEvidence",
    "load_expert_whitening_evidence",
    "save_expert_whitening_evidence",
]
