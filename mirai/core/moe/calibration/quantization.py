"""Lineage-bound expert-balanced, affinity-weighted MoE calibration evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


MOE_QUANT_CALIBRATION_FORMAT = "mirai.moe.quantization_calibration"
MOE_QUANT_CALIBRATION_SCHEMA_VERSION = 1
MOE_QUANT_CALIBRATION_METADATA_KEY = "mirai_moe_quantization_calibration"


@dataclass(frozen=True)
class QuantizationCalibrationTarget:
    """Provider-owned router target exposed to generic calibration."""

    name: str
    router: Any
    num_experts: int
    logical_to_physical: tuple[int, ...] = ()

    def validate(self) -> QuantizationCalibrationTarget:
        if not self.name:
            raise ValueError("Quantization calibration target name must be non-empty.")
        if int(self.num_experts) < 2:
            raise ValueError("Quantization calibration requires at least two experts.")
        if not callable(getattr(self.router, "register_forward_hook", None)):
            raise TypeError("Quantization calibration target router must support hooks.")
        if self.logical_to_physical:
            if len(self.logical_to_physical) < int(self.num_experts):
                raise ValueError(
                    "Quantization calibration alias maps cannot have fewer logical "
                    "than physical experts."
                )
            if min(self.logical_to_physical) < 0 or max(self.logical_to_physical) >= int(
                self.num_experts
            ):
                raise ValueError("Quantization calibration alias map is out of range.")
        return self


@dataclass(frozen=True)
class ExpertAffinityEvidence:
    """Per-expert route frequency, affinity, and candidate-sample coverage."""

    selected_count: Any
    routing_mass: Any
    squared_affinity_mass: Any
    sample_routing_mass: Any

    @property
    def num_experts(self) -> int:
        return int(torch.as_tensor(self.selected_count).numel())

    @property
    def num_samples(self) -> int:
        return int(torch.as_tensor(self.sample_routing_mass).shape[0])

    def validate(self) -> ExpertAffinityEvidence:
        if torch is None:  # pragma: no cover
            raise RuntimeError("MoE quantization calibration requires torch.")
        counts = torch.as_tensor(self.selected_count)
        mass = torch.as_tensor(self.routing_mass)
        squared = torch.as_tensor(self.squared_affinity_mass)
        samples = torch.as_tensor(self.sample_routing_mass)
        if counts.ndim != 1 or int(counts.numel()) < 2:
            raise ValueError("Affinity evidence requires a one-dimensional expert axis.")
        if mass.shape != counts.shape or squared.shape != counts.shape:
            raise ValueError("Affinity evidence expert summaries have inconsistent shapes.")
        if samples.ndim != 2 or int(samples.shape[1]) != int(counts.numel()):
            raise ValueError("Affinity evidence samples must have shape [samples, experts].")
        for value in (counts, mass, squared, samples):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("Affinity evidence contains non-finite values.")
            if bool(torch.any(value < 0).item()):
                raise ValueError("Affinity evidence must be non-negative.")
        if int(counts.sum().item()) == 0 or int(samples.shape[0]) == 0:
            raise ValueError("Affinity calibration observed no routed samples.")
        return self

    def affinity_reconstruction_weights(
        self,
        *,
        epsilon: float = 1e-12,
        require_full_coverage: bool = True,
    ) -> Any:
        """Mean squared routed affinity per expert, normalized to mean one.

        Scaling a routed expert input by its gate coefficient makes its
        least-squares/Hessian contribution proportional to the squared
        coefficient.  Unobserved experts are rejected by default so a
        downstream quantizer cannot silently assign them zero importance.
        """
        self.validate()
        counts = torch.as_tensor(self.selected_count, dtype=torch.float64)
        observed = counts > 0
        if bool(require_full_coverage) and not bool(observed.all().item()):
            missing = torch.nonzero(~observed, as_tuple=False).reshape(-1).tolist()
            raise ValueError(
                "Affinity calibration has no routed evidence for experts "
                f"{missing}. Collect more balanced calibration samples."
            )
        squared = torch.as_tensor(self.squared_affinity_mass, dtype=torch.float64)
        means = squared / counts.clamp_min(1.0)
        normalizer = means[observed].mean().clamp_min(float(epsilon))
        return torch.where(observed, means / normalizer, torch.zeros_like(means))

    def balanced_sample_indices(self, budget: int) -> tuple[int, ...]:
        """Greedily maximize concave normalized expert coverage."""
        self.validate()
        limit = int(budget)
        if limit <= 0:
            raise ValueError("Expert-balanced sample budget must be positive.")
        samples = torch.as_tensor(self.sample_routing_mass, dtype=torch.float64).cpu()
        limit = min(limit, int(samples.shape[0]))
        total = samples.sum(dim=0).clamp_min(torch.finfo(torch.float64).tiny)
        normalized = samples / total.unsqueeze(0)
        coverage = torch.zeros_like(total)
        remaining = set(range(int(samples.shape[0])))
        selected: list[int] = []
        for _ in range(limit):
            best = min(
                remaining,
                key=lambda index: (
                    -float(
                        (
                            torch.sqrt(coverage + normalized[index])
                            - torch.sqrt(coverage)
                        ).sum().item()
                    ),
                    index,
                ),
            )
            selected.append(best)
            coverage += normalized[best]
            remaining.remove(best)
        return tuple(selected)


class ExpertAffinityAccumulator:
    """Accumulate detached route counts and gate affinity for one router."""

    def __init__(self, num_experts: int) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("MoE quantization calibration requires torch.")
        count = int(num_experts)
        if count < 2:
            raise ValueError("Affinity calibration requires at least two experts.")
        self.selected_count = torch.zeros(count, dtype=torch.int64)
        self.routing_mass = torch.zeros(count, dtype=torch.float64)
        self.squared_affinity_mass = torch.zeros(count, dtype=torch.float64)
        self.sample_routing_mass: list[Any] = []

    @property
    def num_experts(self) -> int:
        return int(self.selected_count.numel())

    def record(self, indices: Any, scores: Any) -> None:
        routed = torch.as_tensor(indices).detach().to(torch.int64)
        affinity = torch.as_tensor(scores).detach().to(torch.float64)
        if routed.shape != affinity.shape or routed.numel() == 0:
            raise ValueError("Route indices and affinities must be aligned and non-empty.")
        if int(routed.min().item()) < 0 or int(routed.max().item()) >= self.num_experts:
            raise ValueError("Quantization calibration observed an out-of-range expert.")
        if not bool(torch.isfinite(affinity).all().item()) or bool(
            torch.any(affinity < 0).item()
        ):
            raise ValueError("Route affinities must be finite and non-negative.")
        routed, affinity = _coalesce_routed_affinity(routed, affinity)
        routed = routed.reshape(-1)
        affinity = affinity.reshape(-1)
        active = affinity > 0
        routed = routed[active]
        affinity = affinity[active]
        if routed.numel() == 0:
            raise ValueError("Quantization calibration observed no positive route affinity.")
        counts = torch.bincount(routed, minlength=self.num_experts)
        mass = torch.zeros(
            self.num_experts,
            dtype=torch.float64,
            device=affinity.device,
        )
        squared = torch.zeros_like(mass)
        mass.scatter_add_(0, routed, affinity)
        squared.scatter_add_(0, routed, affinity.square())
        counts_cpu = counts.cpu()
        mass_cpu = mass.cpu()
        squared_cpu = squared.cpu()
        self.selected_count += counts_cpu
        self.routing_mass += mass_cpu
        self.squared_affinity_mass += squared_cpu
        self.sample_routing_mass.append(mass_cpu)

    def evidence(self) -> ExpertAffinityEvidence:
        samples = (
            torch.stack(self.sample_routing_mass, dim=0)
            if self.sample_routing_mass
            else torch.empty((0, self.num_experts), dtype=torch.float64)
        )
        return ExpertAffinityEvidence(
            selected_count=self.selected_count.clone(),
            routing_mass=self.routing_mass.clone(),
            squared_affinity_mass=self.squared_affinity_mass.clone(),
            sample_routing_mass=samples,
        ).validate()


def _coalesce_routed_affinity(indices: Any, scores: Any) -> tuple[Any, Any]:
    """Sum duplicate physical routes within each token's top-k assignment."""
    if indices.ndim < 2 or int(indices.shape[-1]) < 2:
        return indices, scores
    routed = indices.reshape(-1, int(indices.shape[-1])).clone()
    affinity = scores.reshape_as(routed).clone()
    top_k = int(routed.shape[1])
    for later in range(1, top_k):
        for earlier in range(later):
            duplicate = routed[:, later] == routed[:, earlier]
            affinity[:, earlier] = affinity[:, earlier] + torch.where(
                duplicate,
                affinity[:, later],
                torch.zeros_like(affinity[:, later]),
            )
            affinity[:, later] = torch.where(
                duplicate,
                torch.zeros_like(affinity[:, later]),
                affinity[:, later],
            )
    return routed.reshape_as(indices), affinity.reshape_as(scores)


def register_quantization_calibration_hooks(
    targets: Mapping[str, QuantizationCalibrationTarget],
    accumulators: Mapping[str, ExpertAffinityAccumulator],
) -> list[Any]:
    """Capture provider router state before the family clears it."""
    handles: list[Any] = []
    try:
        for name, target in targets.items():
            target.validate()
            if name not in accumulators:
                raise ValueError(
                    f"Quantization calibration target {name!r} has no accumulator."
                )
            accumulator = accumulators[name]
            if accumulator.num_experts != int(target.num_experts):
                raise ValueError(
                    f"Quantization calibration target {name!r} expert count does "
                    "not match its accumulator."
                )
            aliases = tuple(int(value) for value in target.logical_to_physical)

            def _hook(
                module: Any,
                _inputs: Any,
                _output: Any,
                *,
                _acc=accumulator,
                _aliases=aliases,
            ) -> None:
                indices = getattr(module, "last_top_indices", None)
                scores = getattr(module, "last_top_scores", None)
                if indices is not None and scores is not None:
                    if _aliases:
                        mapping = torch.as_tensor(
                            _aliases, device=indices.device, dtype=torch.long
                        )
                        indices = mapping.index_select(0, indices.reshape(-1)).reshape_as(
                            indices
                        )
                    _acc.record(indices, scores)

            handles.append(target.router.register_forward_hook(_hook))
    except Exception:
        for handle in reversed(handles):
            handle.remove()
        raise
    return handles


def save_quantization_calibration_evidence(
    path: str | Path,
    evidence_by_module: Mapping[str, ExpertAffinityEvidence],
    *,
    dataset_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str,
) -> None:
    """Persist tensor-native evidence with mandatory dataset/model lineage."""
    from safetensors.torch import save_file

    lineage = {
        "dataset_snapshot_id": str(dataset_snapshot_id).strip(),
        "model_snapshot_id": str(model_snapshot_id).strip(),
        "config_snapshot_id": str(config_snapshot_id).strip(),
    }
    if not all(lineage.values()):
        raise ValueError("Quantization calibration requires complete snapshot lineage.")
    if not evidence_by_module:
        raise ValueError("Quantization calibration evidence cannot be empty.")
    tensors: dict[str, Any] = {}
    modules: list[dict[str, Any]] = []
    for index, (name, evidence) in enumerate(evidence_by_module.items()):
        evidence.validate()
        prefix = f"module_{index:04d}"
        fields = {
            "selected_count": f"{prefix}.selected_count",
            "routing_mass": f"{prefix}.routing_mass",
            "squared_affinity_mass": f"{prefix}.squared_affinity_mass",
            "sample_routing_mass": f"{prefix}.sample_routing_mass",
        }
        tensors[fields["selected_count"]] = torch.as_tensor(
            evidence.selected_count, dtype=torch.int64
        ).cpu().contiguous()
        for field in ("routing_mass", "squared_affinity_mass", "sample_routing_mass"):
            tensors[fields[field]] = torch.as_tensor(
                getattr(evidence, field), dtype=torch.float64
            ).cpu().contiguous()
        modules.append({"name": str(name), "num_experts": evidence.num_experts, **fields})
    manifest = {
        "schema_version": MOE_QUANT_CALIBRATION_SCHEMA_VERSION,
        "format": MOE_QUANT_CALIBRATION_FORMAT,
        **lineage,
        "modules": modules,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            MOE_QUANT_CALIBRATION_METADATA_KEY: json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            )
        },
    )


def load_quantization_calibration_evidence(
    path: str | Path,
    *,
    expected_dataset_snapshot_id: str | None = None,
    expected_model_snapshot_id: str | None = None,
    expected_config_snapshot_id: str | None = None,
) -> tuple[dict[str, ExpertAffinityEvidence], dict[str, str]]:
    """Load evidence and reject requested lineage mismatches."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    input_path = Path(path)
    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        raw = (handle.metadata() or {}).get(MOE_QUANT_CALIBRATION_METADATA_KEY)
    if raw is None:
        raise ValueError("Quantization calibration artifact has no Mirai manifest.")
    manifest = json.loads(raw)
    if manifest.get("format") != MOE_QUANT_CALIBRATION_FORMAT:
        raise ValueError(f"Unsupported calibration format {manifest.get('format')!r}.")
    if int(manifest.get("schema_version", 0)) != MOE_QUANT_CALIBRATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported calibration schema {manifest.get('schema_version')!r}.")
    lineage = {
        key: str(manifest.get(key, ""))
        for key in ("dataset_snapshot_id", "model_snapshot_id", "config_snapshot_id")
    }
    if not all(lineage.values()):
        raise ValueError("Quantization calibration artifact has incomplete lineage.")
    expected = {
        "dataset_snapshot_id": expected_dataset_snapshot_id,
        "model_snapshot_id": expected_model_snapshot_id,
        "config_snapshot_id": expected_config_snapshot_id,
    }
    for key, value in expected.items():
        if value is not None and str(value) != lineage[key]:
            raise ValueError(
                f"Quantization calibration {key} mismatch: expected {value!r}, "
                f"found {lineage[key]!r}."
            )
    tensors = load_file(str(input_path), device="cpu")
    loaded: dict[str, ExpertAffinityEvidence] = {}
    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("Quantization calibration manifest has no modules.")
    for spec in modules:
        name = str(spec["name"])
        if not name or name in loaded:
            raise ValueError("Calibration module names must be non-empty and unique.")
        item = ExpertAffinityEvidence(
            selected_count=tensors[str(spec["selected_count"])],
            routing_mass=tensors[str(spec["routing_mass"])],
            squared_affinity_mass=tensors[str(spec["squared_affinity_mass"])],
            sample_routing_mass=tensors[str(spec["sample_routing_mass"])],
        ).validate()
        if int(spec.get("num_experts", -1)) != item.num_experts:
            raise ValueError(
                f"Calibration module {name!r} expert count does not match its tensors."
            )
        loaded[name] = item
    return loaded, lineage


__all__ = [
    "ExpertAffinityAccumulator",
    "ExpertAffinityEvidence",
    "MOE_QUANT_CALIBRATION_FORMAT",
    "MOE_QUANT_CALIBRATION_METADATA_KEY",
    "MOE_QUANT_CALIBRATION_SCHEMA_VERSION",
    "QuantizationCalibrationTarget",
    "load_quantization_calibration_evidence",
    "register_quantization_calibration_hooks",
    "save_quantization_calibration_evidence",
]
