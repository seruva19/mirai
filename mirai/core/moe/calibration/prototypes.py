"""Calibration evidence and pluggable selection for expert prototypes.

This optional extension is intentionally separate from runtime routing and the
packed-artifact transform. It observes explicit per-route outputs, computes
module-local expert distances, and resolves a registered plan-selection policy.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from mirai.core.moe.adaptation.specialization_loss import deterministic_token_sample
from mirai.core.moe.storage.aliases import PrototypeConsolidationPlan
from mirai.core.moe.storage.aliases import build_prototype_plan
from mirai.core.registry import Registry

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PROTOTYPE_CALIBRATION_FORMAT = "mirai.moe.prototype_calibration"
PROTOTYPE_CALIBRATION_SCHEMA_VERSION = 3
PROTOTYPE_CALIBRATION_METADATA_KEY = "mirai_prototype_calibration_manifest"


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Prototype calibration requires torch.")


@dataclasses.dataclass(frozen=True)
class PrototypeCalibrationEvidence:
    """Module-local contribution and replaceability evidence."""

    contribution_sum: Any
    selected_count: Any
    distance_matrix: Any
    distance_metric: str = "normalized_parameter"

    @property
    def num_experts(self) -> int:
        return int(self.contribution_sum.numel())

    def validate(self) -> PrototypeCalibrationEvidence:
        _require_torch()
        contribution = torch.as_tensor(self.contribution_sum)
        counts = torch.as_tensor(self.selected_count)
        distances = torch.as_tensor(self.distance_matrix)
        if contribution.ndim != 1 or int(contribution.numel()) < 2:
            raise ValueError("Contribution evidence requires at least two experts.")
        expected = (int(contribution.numel()), int(contribution.numel()))
        if counts.shape != contribution.shape or tuple(distances.shape) != expected:
            raise ValueError("Prototype calibration evidence shapes are inconsistent.")
        if not bool(torch.isfinite(contribution).all().item()):
            raise ValueError("Contribution evidence contains non-finite values.")
        if not bool(torch.isfinite(distances).all().item()):
            raise ValueError("Expert distance evidence contains non-finite values.")
        if bool(torch.any(contribution < 0).item()) or bool(torch.any(counts < 0).item()):
            raise ValueError("Contribution sums and selected counts must be non-negative.")
        if int(counts.sum().item()) == 0:
            raise ValueError("Prototype calibration did not observe any selected routes.")
        if bool(torch.any(distances < 0).item()):
            raise ValueError("Expert distances must be non-negative.")
        if not torch.allclose(distances, distances.T, atol=1e-10, rtol=1e-10):
            raise ValueError("Expert distance matrix must be symmetric.")
        if not torch.allclose(
            torch.diagonal(distances),
            torch.zeros(self.num_experts, dtype=distances.dtype, device=distances.device),
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError("Expert distance matrix diagonal must be zero.")
        if str(self.distance_metric) not in {
            "normalized_parameter",
            "mean_output_euclidean",
        }:
            raise ValueError(
                f"Unsupported expert distance metric {self.distance_metric!r}."
            )
        return self

    def average_contribution(self) -> Any:
        self.validate()
        contribution = torch.as_tensor(self.contribution_sum, dtype=torch.float64)
        counts = torch.as_tensor(self.selected_count, dtype=torch.float64)
        return contribution / counts.clamp_min(1.0)


class PrototypeContributionAccumulator:
    """Accumulate gate-weighted expert-output norms without retaining autograd."""

    def __init__(self, num_experts: int) -> None:
        _require_torch()
        count = int(num_experts)
        if count < 2:
            raise ValueError("Prototype calibration requires at least two experts.")
        self.contribution_sum = torch.zeros(count, dtype=torch.float64)
        self.selected_count = torch.zeros(count, dtype=torch.int64)

    @property
    def num_experts(self) -> int:
        return int(self.contribution_sum.numel())

    def record(
        self,
        expert_id: int,
        routing_weights: Any,
        expert_outputs: Any,
    ) -> None:
        expert = int(expert_id)
        if expert < 0 or expert >= self.num_experts:
            raise ValueError(f"Observed expert id {expert} is out of range.")
        weights = torch.as_tensor(routing_weights).detach().reshape(-1)
        outputs = torch.as_tensor(expert_outputs).detach()
        if outputs.ndim < 2 or int(outputs.shape[0]) != int(weights.numel()):
            raise ValueError(
                "Expert outputs must be [routes, ...] and align with routing weights."
            )
        output_device = outputs.device
        compute_dtype = torch.float32 if outputs.is_cuda else torch.float64
        weights = weights.to(device=output_device, dtype=compute_dtype)
        outputs = outputs.to(dtype=compute_dtype).reshape(int(weights.numel()), -1)
        valid = torch.isfinite(weights).all() & torch.isfinite(outputs).all()
        if not bool(valid.item()):
            raise ValueError("Prototype calibration observations must be finite.")
        if bool(torch.any(weights < 0).item()):
            raise ValueError("Prototype calibration routing weights must be non-negative.")
        norms = torch.linalg.vector_norm(outputs, ord=2, dim=1)
        contribution = torch.dot(weights, norms).detach().to(
            device="cpu", dtype=torch.float64
        )
        self.contribution_sum[expert] += contribution
        self.selected_count[expert] += int(weights.numel())

    def evidence(self, distance_matrix: Any) -> PrototypeCalibrationEvidence:
        return PrototypeCalibrationEvidence(
            contribution_sum=self.contribution_sum.clone(),
            selected_count=self.selected_count.clone(),
            distance_matrix=torch.as_tensor(distance_matrix).detach().cpu().clone(),
            distance_metric="normalized_parameter",
        ).validate()


class MeanExpertOutputAccumulator(PrototypeContributionAccumulator):
    """Stream each expert's mean output over one shared calibration population."""

    def __init__(
        self,
        num_experts: int,
        *,
        max_tokens_per_observation: int = 256,
    ) -> None:
        super().__init__(num_experts)
        token_limit = int(max_tokens_per_observation)
        if token_limit < 1:
            raise ValueError(
                "Mean-output calibration token limit must be positive."
            )
        self.max_tokens_per_observation = token_limit
        self._output_sum: Any | None = None
        self._output_count = torch.zeros(self.num_experts, dtype=torch.int64)

    def shared_calibration_inputs(self, tokens: Any) -> Any:
        """Select one deterministic bounded population shared by all experts."""
        return deterministic_token_sample(
            tokens,
            max_tokens=self.max_tokens_per_observation,
        )

    @property
    def output_tokens_per_expert(self) -> int:
        if not bool(torch.all(self._output_count == self._output_count[0]).item()):
            raise RuntimeError("Experts observed different output-token counts.")
        return int(self._output_count[0].item())

    def record_all_expert_output(
        self,
        expert_id: int,
        expert_outputs: Any,
    ) -> None:
        expert = int(expert_id)
        if expert < 0 or expert >= self.num_experts:
            raise ValueError(f"Observed expert id {expert} is out of range.")
        outputs = torch.as_tensor(expert_outputs).detach()
        if outputs.ndim < 2 or int(outputs.shape[0]) < 1:
            raise ValueError(
                "All-expert calibration outputs must have shape [tokens, ..., hidden]."
            )
        flattened = outputs.reshape(-1, int(outputs.shape[-1]))
        if not bool(torch.isfinite(flattened).all().item()):
            raise ValueError("All-expert calibration outputs must be finite.")
        vector = flattened.to(dtype=torch.float32).sum(dim=0).to(
            device="cpu", dtype=torch.float64
        )
        if self._output_sum is None:
            self._output_sum = torch.zeros(
                (self.num_experts, int(vector.numel())),
                dtype=torch.float64,
            )
        elif tuple(self._output_sum.shape) != (self.num_experts, int(vector.numel())):
            raise ValueError(
                "All-expert calibration output width changed between observations."
            )
        self._output_sum[expert] += vector
        self._output_count[expert] += int(flattened.shape[0])

    def evidence(self, distance_matrix: Any | None = None) -> PrototypeCalibrationEvidence:
        if distance_matrix is not None:
            raise ValueError(
                "Mean-output evidence computes its distance matrix from observations."
            )
        if self._output_sum is None or bool(torch.any(self._output_count == 0).item()):
            raise ValueError(
                "Mean-output calibration must observe every expert on shared inputs."
            )
        if not bool(torch.all(self._output_count == self._output_count[0]).item()):
            raise ValueError(
                "Mean-output calibration experts did not see the same token population."
            )
        means = self._output_sum / self._output_count.to(dtype=torch.float64).unsqueeze(1)
        distances = torch.cdist(means, means, p=2)
        distances.fill_diagonal_(0.0)
        return PrototypeCalibrationEvidence(
            contribution_sum=self.contribution_sum.clone(),
            selected_count=self.selected_count.clone(),
            distance_matrix=distances,
            distance_metric="mean_output_euclidean",
        ).validate()


@dataclasses.dataclass(frozen=True)
class HierarchicalOutputMergePlan:
    """Average-linkage clusters plus normalized per-expert merge weights."""

    consolidation: PrototypeConsolidationPlan
    merge_weights: tuple[float, ...]

    def validate(self) -> HierarchicalOutputMergePlan:
        if len(self.merge_weights) != self.consolidation.logical_experts:
            raise ValueError(
                "Hierarchical merge weights must match the logical expert count."
            )
        if any(not math.isfinite(value) or value < 0.0 for value in self.merge_weights):
            raise ValueError("Hierarchical merge weights must be finite and non-negative.")
        for physical_id in range(self.consolidation.physical_experts):
            total = sum(
                self.merge_weights[logical_id]
                for logical_id, mapped in enumerate(
                    self.consolidation.logical_to_physical
                )
                if mapped == physical_id
            )
            if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    "Hierarchical merge weights must sum to one inside each cluster."
                )
        return self


def select_hierarchical_output_merge(
    evidence: PrototypeCalibrationEvidence,
    reduction_ratio: float,
) -> HierarchicalOutputMergePlan:
    """Cluster mean expert outputs with deterministic average linkage."""
    evidence.validate()
    if evidence.distance_metric != "mean_output_euclidean":
        raise ValueError(
            "Hierarchical output merging requires mean_output_euclidean evidence."
        )
    ratio = float(reduction_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("reduction_ratio must be strictly between zero and one.")
    expert_count = evidence.num_experts
    cluster_count = max(1, math.floor(((1.0 - ratio) * expert_count) + 0.5))
    if cluster_count >= expert_count:
        raise ValueError(
            "reduction_ratio rounds to no physical expert reduction; increase it."
        )
    distances = torch.as_tensor(evidence.distance_matrix, dtype=torch.float64).cpu()
    clusters: list[tuple[int, ...]] = [(expert,) for expert in range(expert_count)]
    while len(clusters) > cluster_count:
        candidates: list[tuple[float, tuple[int, ...], tuple[int, ...], int, int]] = []
        for left_id, left in enumerate(clusters):
            for right_id in range(left_id + 1, len(clusters)):
                right = clusters[right_id]
                mean_distance = sum(
                    float(distances[a, b]) for a in left for b in right
                ) / float(len(left) * len(right))
                candidates.append(
                    (mean_distance, left, right, left_id, right_id)
                )
        _distance, left, right, left_id, right_id = min(candidates)
        merged = tuple(sorted((*left, *right)))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_id, right_id}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: (cluster[0], cluster))

    prototypes = tuple(cluster[0] for cluster in clusters)
    logical_to_prototype = [0] * expert_count
    merge_weights = [0.0] * expert_count
    selected = torch.as_tensor(evidence.selected_count, dtype=torch.float64).cpu()
    for prototype, cluster in zip(prototypes, clusters, strict=True):
        for expert in cluster:
            logical_to_prototype[expert] = prototype
        cluster_frequency = selected[list(cluster)]
        total = float(cluster_frequency.sum().item())
        if total > 0.0:
            normalized = cluster_frequency / total
        else:
            normalized = torch.full(
                (len(cluster),), 1.0 / float(len(cluster)), dtype=torch.float64
            )
        for expert, weight in zip(cluster, normalized.tolist(), strict=True):
            merge_weights[expert] = float(weight)
    return HierarchicalOutputMergePlan(
        consolidation=build_prototype_plan(
            logical_to_prototype,
            num_logical_experts=expert_count,
        ),
        merge_weights=tuple(merge_weights),
    ).validate()


def normalized_parameter_distance(
    expert_projections: Iterable[Any],
    *,
    epsilon: float = 1e-12,
) -> Any:
    """Mean normalized Frobenius distance across [experts, ...] projections."""
    _require_torch()
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be finite and positive.")
    total = None
    projection_count = 0
    expert_count = 0
    for raw_projection in expert_projections:
        projection = torch.as_tensor(raw_projection).detach()
        if projection.ndim < 2 or int(projection.shape[0]) < 2:
            raise ValueError("Each expert projection must have shape [experts, ...].")
        if expert_count and int(projection.shape[0]) != expert_count:
            raise ValueError("All projections must contain the same expert count.")
        expert_count = int(projection.shape[0])
        flat = projection.to(device="cpu", dtype=torch.float64).reshape(expert_count, -1)
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError("Expert projection contains non-finite values.")
        norms = torch.linalg.vector_norm(flat, ord=2, dim=1)
        pairwise = torch.cdist(flat, flat, p=2)
        normalized = (2.0 * pairwise) / (
            norms.unsqueeze(1) + norms.unsqueeze(0) + (2.0 * eps)
        )
        total = normalized if total is None else total + normalized
        projection_count += 1
    if total is None or projection_count == 0:
        raise ValueError("At least one expert projection is required.")
    distances = total / float(projection_count)
    distances.fill_diagonal_(0.0)
    return distances


PrototypeSelectionStrategy = Callable[
    [PrototypeCalibrationEvidence, float], PrototypeConsolidationPlan
]
PrototypeSelectionStrategyRegistry: Registry[PrototypeSelectionStrategy] = Registry(
    "prototype selection strategy"
)


def register_prototype_selection_strategy(
    name: str,
) -> Callable[[PrototypeSelectionStrategy], PrototypeSelectionStrategy]:
    return PrototypeSelectionStrategyRegistry.decorator(name)


def resolve_prototype_selection_strategy(name: str) -> PrototypeSelectionStrategy:
    return PrototypeSelectionStrategyRegistry.get(name)


def _minmax(values: Any, *, epsilon: float) -> Any:
    minimum = values.amin()
    return (values - minimum) / (values.amax() - minimum + float(epsilon))


@register_prototype_selection_strategy("adaptive")
def select_adaptive_prototypes(
    evidence: PrototypeCalibrationEvidence,
    reduction_ratio: float,
) -> PrototypeConsolidationPlan:
    """Select high-contribution, hard-to-replace experts and remap the rest."""
    evidence.validate()
    ratio = float(reduction_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("reduction_ratio must be strictly between zero and one.")
    expert_count = evidence.num_experts
    prototype_count = max(1, math.floor(((1.0 - ratio) * expert_count) + 0.5))
    if prototype_count >= expert_count:
        raise ValueError(
            "reduction_ratio rounds to no physical expert reduction; increase it."
        )
    distances = torch.as_tensor(evidence.distance_matrix, dtype=torch.float64).cpu()
    masked = distances.clone()
    masked.fill_diagonal_(float("inf"))
    replaceability = masked.amin(dim=1)
    epsilon = 1e-12
    scores = _minmax(evidence.average_contribution(), epsilon=epsilon) * _minmax(
        replaceability,
        epsilon=epsilon,
    )
    ranked = sorted(range(expert_count), key=lambda item: (-float(scores[item]), item))
    prototypes = tuple(sorted(ranked[:prototype_count]))
    prototype_set = set(prototypes)
    logical_to_prototype = [
        expert
        if expert in prototype_set
        else min(prototypes, key=lambda item: (float(distances[expert, item]), item))
        for expert in range(expert_count)
    ]
    return build_prototype_plan(
        logical_to_prototype,
        num_logical_experts=expert_count,
    )


def build_prototype_plans(
    evidence_by_module: Mapping[str, PrototypeCalibrationEvidence],
    *,
    reduction_ratio: float,
    strategy: str = "adaptive",
) -> dict[str, PrototypeConsolidationPlan]:
    if not evidence_by_module:
        raise ValueError("At least one module's calibration evidence is required.")
    selector = resolve_prototype_selection_strategy(strategy)
    return {
        str(module_name): selector(evidence, float(reduction_ratio))
        for module_name, evidence in evidence_by_module.items()
    }


def save_prototype_calibration_evidence(
    path: str | Path,
    evidence_by_module: Mapping[str, PrototypeCalibrationEvidence],
    *,
    dataset_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str,
    packed_artifact_fingerprint: str,
) -> None:
    """Write versioned, tensor-native evidence with mandatory lineage."""
    from safetensors.torch import save_file

    lineage = {
        "dataset_snapshot_id": str(dataset_snapshot_id).strip(),
        "model_snapshot_id": str(model_snapshot_id).strip(),
        "config_snapshot_id": str(config_snapshot_id).strip(),
        "packed_artifact_fingerprint": str(packed_artifact_fingerprint).strip(),
    }
    if not all(lineage.values()):
        raise ValueError("Prototype calibration requires complete snapshot lineage.")
    if not evidence_by_module:
        raise ValueError("At least one module's calibration evidence is required.")
    tensors: dict[str, Any] = {}
    modules: list[dict[str, Any]] = []
    for index, (module_name, evidence) in enumerate(evidence_by_module.items()):
        evidence.validate()
        prefix = f"module_{index:04d}"
        keys = {
            "contribution_sum": f"{prefix}.contribution_sum",
            "selected_count": f"{prefix}.selected_count",
            "distance_matrix": f"{prefix}.distance_matrix",
        }
        tensors[keys["contribution_sum"]] = torch.as_tensor(
            evidence.contribution_sum, dtype=torch.float64
        ).cpu().contiguous()
        tensors[keys["selected_count"]] = torch.as_tensor(
            evidence.selected_count, dtype=torch.int64
        ).cpu().contiguous()
        tensors[keys["distance_matrix"]] = torch.as_tensor(
            evidence.distance_matrix, dtype=torch.float64
        ).cpu().contiguous()
        modules.append(
            {
                "name": str(module_name),
                "num_experts": evidence.num_experts,
                "distance_metric": str(evidence.distance_metric),
                **keys,
            }
        )
    manifest = {
        "schema_version": PROTOTYPE_CALIBRATION_SCHEMA_VERSION,
        "format": PROTOTYPE_CALIBRATION_FORMAT,
        **lineage,
        "modules": modules,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            PROTOTYPE_CALIBRATION_METADATA_KEY: json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            )
        },
    )


def load_prototype_calibration_evidence(
    path: str | Path,
    *,
    expected_dataset_snapshot_id: str | None = None,
    expected_model_snapshot_id: str | None = None,
    expected_config_snapshot_id: str | None = None,
    expected_packed_artifact_fingerprint: str | None = None,
) -> tuple[dict[str, PrototypeCalibrationEvidence], dict[str, str]]:
    """Load evidence and reject requested lineage mismatches."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    input_path = Path(path)
    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        raw_manifest = metadata.get(PROTOTYPE_CALIBRATION_METADATA_KEY)
    if raw_manifest is None:
        raise ValueError("Calibration artifact is missing its Mirai manifest.")
    manifest = json.loads(raw_manifest)
    if manifest.get("format") != PROTOTYPE_CALIBRATION_FORMAT:
        raise ValueError(f"Unsupported calibration format {manifest.get('format')!r}.")
    if int(manifest.get("schema_version", 0)) != PROTOTYPE_CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported calibration schema {manifest.get('schema_version')!r}."
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
        raise ValueError("Prototype calibration artifact has incomplete lineage.")
    expected = {
        "dataset_snapshot_id": expected_dataset_snapshot_id,
        "model_snapshot_id": expected_model_snapshot_id,
        "config_snapshot_id": expected_config_snapshot_id,
        "packed_artifact_fingerprint": expected_packed_artifact_fingerprint,
    }
    for key, value in expected.items():
        if value is not None and str(value) != lineage[key]:
            raise ValueError(
                f"Prototype calibration {key} mismatch: expected {value!r}, "
                f"found {lineage[key]!r}."
            )
    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("Calibration manifest must contain a non-empty module list.")
    tensors = load_file(str(input_path), device="cpu")
    loaded: dict[str, PrototypeCalibrationEvidence] = {}
    for spec in modules:
        if not isinstance(spec, Mapping):
            raise ValueError("Calibration module manifest entry must be an object.")
        name = str(spec.get("name", ""))
        if not name or name in loaded:
            raise ValueError("Calibration module names must be non-empty and unique.")
        try:
            evidence = PrototypeCalibrationEvidence(
                contribution_sum=tensors[str(spec["contribution_sum"])],
                selected_count=tensors[str(spec["selected_count"])],
                distance_matrix=tensors[str(spec["distance_matrix"])],
                distance_metric=str(spec.get("distance_metric", "")),
            ).validate()
        except KeyError as exc:
            raise ValueError(f"Calibration tensor {exc.args[0]!r} is missing.") from exc
        if evidence.num_experts != int(spec.get("num_experts", 0)):
            raise ValueError(f"Calibration expert count mismatch for module {name!r}.")
        loaded[name] = evidence
    return loaded, lineage


__all__ = [
    "PROTOTYPE_CALIBRATION_FORMAT",
    "PROTOTYPE_CALIBRATION_METADATA_KEY",
    "PROTOTYPE_CALIBRATION_SCHEMA_VERSION",
    "PrototypeCalibrationEvidence",
    "PrototypeContributionAccumulator",
    "MeanExpertOutputAccumulator",
    "HierarchicalOutputMergePlan",
    "PrototypeSelectionStrategyRegistry",
    "build_prototype_plans",
    "load_prototype_calibration_evidence",
    "normalized_parameter_distance",
    "register_prototype_selection_strategy",
    "resolve_prototype_selection_strategy",
    "save_prototype_calibration_evidence",
    "select_hierarchical_output_merge",
    "select_adaptive_prototypes",
]
