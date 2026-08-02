"""Detached routing-selection agreement diagnostics and evidence.

``topk_set_agreement`` compares two selections of the same tokens and therefore
needs a caller that holds both, such as an offline reference-versus-transformed
comparison. The step-local probe below reports only the selection-boundary
margin, which a single forward already determines.

Train-versus-inference evidence compares the same batch, sampled noise,
timestep, token position, layer, and router invocation in two model modes. It
does not compare adjacent denoising steps, different layers, or a step-zero
snapshot. Fixed-cardinality overlap follows PR2's ``|A ∩ B| / k`` definition;
active masks additionally make route-set churn measurable when a training-only
policy changes the number of active slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_integer_matrix(value: Any, *, name: str) -> Any:
    if torch is None:
        raise RuntimeError("Routing agreement diagnostics require torch.")
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor.")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if value.dtype not in integer_dtypes:
        raise ValueError(f"{name} must use an integer dtype.")
    return value


def topk_set_agreement(
    reference_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    num_experts: int,
) -> float:
    """Mean token-wise Jaccard agreement between selected expert sets."""
    reference = _require_integer_matrix(
        reference_indices, name="reference_indices"
    )
    candidate = _require_integer_matrix(
        candidate_indices, name="candidate_indices"
    )
    if reference.shape != candidate.shape:
        raise ValueError("reference_indices and candidate_indices must have equal shape.")
    tokens, top_k = reference.shape
    if tokens <= 0 or top_k <= 0:
        raise ValueError("routing selections must contain at least one token and slot.")
    experts = int(num_experts)
    if experts <= 0:
        raise ValueError("num_experts must be positive.")
    reference = reference.to(dtype=torch.long)
    candidate = candidate.to(device=reference.device, dtype=torch.long)
    if (
        int(reference.min().item()) < 0
        or int(candidate.min().item()) < 0
        or int(reference.max().item()) >= experts
        or int(candidate.max().item()) >= experts
    ):
        raise ValueError("routing selection contains an expert id outside num_experts.")

    reference_membership = torch.zeros(
        (tokens, experts), dtype=torch.bool, device=reference.device
    )
    candidate_membership = torch.zeros_like(reference_membership)
    reference_membership.scatter_(1, reference, True)
    candidate_membership.scatter_(1, candidate, True)
    intersection = (reference_membership & candidate_membership).sum(dim=1)
    union = (reference_membership | candidate_membership).sum(dim=1)
    return float((intersection.double() / union.double()).mean().item())


def topk_overlap_fraction(
    reference_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    num_experts: int,
) -> float:
    """Return PR2's mean exact top-k overlap ``|A ∩ B| / k``."""
    reference = _require_integer_matrix(
        reference_indices, name="reference_indices"
    )
    candidate = _require_integer_matrix(
        candidate_indices, name="candidate_indices"
    )
    if reference.shape != candidate.shape:
        raise ValueError("reference_indices and candidate_indices must have equal shape.")
    tokens, top_k = reference.shape
    if tokens <= 0 or top_k <= 0:
        raise ValueError("routing selections must contain at least one token and slot.")
    reference_selection = RoutingSelection.from_tensors(
        reference,
        num_experts=num_experts,
    )
    candidate_selection = RoutingSelection.from_tensors(
        candidate,
        num_experts=num_experts,
    )
    comparison = compare_routing_selections(
        reference_selection,
        candidate_selection,
    )
    if comparison.equal_cardinality_tokens != comparison.token_count:
        raise ValueError("top-k overlap requires equal active cardinality.")
    return float(comparison.matched_cardinality_overlap)


@dataclass(frozen=True)
class RoutingSelection:
    """One router invocation's detached active expert sets."""

    indices: Any
    active_mask: Any
    num_experts: int

    @classmethod
    def from_tensors(
        cls,
        indices: Any,
        *,
        num_experts: int,
        active_mask: Any | None = None,
    ) -> "RoutingSelection":
        if torch is None:  # pragma: no cover
            raise RuntimeError("Routing agreement diagnostics require torch.")
        if not torch.is_tensor(indices) or indices.ndim < 2:
            raise ValueError("routing indices must have at least token and slot axes.")
        experts = int(num_experts)
        if experts <= 0:
            raise ValueError("num_experts must be positive.")
        flattened = indices.detach().reshape(-1, int(indices.shape[-1]))
        flattened = _require_integer_matrix(flattened, name="indices").to(
            device="cpu",
            dtype=torch.int64,
        )
        if int(flattened.shape[0]) <= 0 or int(flattened.shape[1]) <= 0:
            raise ValueError("routing selections must contain tokens and slots.")
        if active_mask is None:
            mask = torch.ones_like(flattened, dtype=torch.bool)
        else:
            if not torch.is_tensor(active_mask) or active_mask.shape != indices.shape:
                raise ValueError("active_mask must match routing indices.")
            mask = active_mask.detach().reshape_as(flattened).to(
                device="cpu",
                dtype=torch.bool,
            )
        active_values = flattened[mask]
        if int(active_values.numel()) <= 0:
            raise ValueError("every routing capture must contain active routes.")
        if (
            int(active_values.min().item()) < 0
            or int(active_values.max().item()) >= experts
        ):
            raise ValueError("routing selection contains an expert id outside num_experts.")
        membership = torch.zeros(
            (int(flattened.shape[0]), experts),
            dtype=torch.int16,
        )
        membership.scatter_add_(1, flattened, mask.to(dtype=torch.int16))
        if bool((membership > 1).any().item()):
            raise ValueError("active routing slots must not contain duplicate experts.")
        return cls(
            indices=flattened.contiguous(),
            active_mask=mask.contiguous(),
            num_experts=experts,
        )


@dataclass(frozen=True)
class RoutingSelectionTarget:
    """Provider-owned router capture target."""

    name: str
    router: Any
    num_experts: int

    def validate(self) -> "RoutingSelectionTarget":
        if not str(self.name).strip():
            raise ValueError("Routing agreement target name cannot be empty.")
        if int(self.num_experts) <= 0:
            raise ValueError("Routing agreement target num_experts must be positive.")
        register_hook = getattr(self.router, "register_forward_hook", None)
        if not callable(register_hook):
            raise TypeError("Routing agreement target router must support forward hooks.")
        return self


@dataclass(frozen=True)
class RoutingSelectionComparison:
    """Sufficient statistics for one paired router invocation."""

    token_count: int
    unchanged_tokens: int
    equal_cardinality_tokens: int
    overlap_sum: float
    jaccard_sum: float
    reference_recall_sum: float
    candidate_precision_sum: float
    reference_active_sum: int
    candidate_active_sum: int
    symmetric_difference_sum: int
    deviation_histogram: dict[int, int]

    @property
    def changed_token_fraction(self) -> float:
        return 1.0 - (float(self.unchanged_tokens) / float(self.token_count))

    @property
    def equal_cardinality_token_fraction(self) -> float:
        return float(self.equal_cardinality_tokens) / float(self.token_count)

    @property
    def matched_cardinality_overlap(self) -> float:
        if self.equal_cardinality_tokens <= 0:
            return 0.0
        return float(self.overlap_sum) / float(self.equal_cardinality_tokens)

    def to_dict(self) -> dict[str, Any]:
        tokens = float(self.token_count)
        return {
            "token_count": int(self.token_count),
            "unchanged_token_fraction": float(self.unchanged_tokens) / tokens,
            "changed_token_fraction": self.changed_token_fraction,
            "equal_cardinality_token_fraction": (
                self.equal_cardinality_token_fraction
            ),
            "matched_cardinality_overlap": self.matched_cardinality_overlap,
            "mean_jaccard": float(self.jaccard_sum) / tokens,
            "mean_reference_overlap": float(self.reference_recall_sum) / tokens,
            "mean_candidate_overlap": float(self.candidate_precision_sum) / tokens,
            "mean_reference_active_experts": (
                float(self.reference_active_sum) / tokens
            ),
            "mean_candidate_active_experts": (
                float(self.candidate_active_sum) / tokens
            ),
            "mean_symmetric_difference": (
                float(self.symmetric_difference_sum) / tokens
            ),
            "matched_cardinality_deviation_histogram": {
                str(key): int(value)
                for key, value in sorted(self.deviation_histogram.items())
            },
        }


def _selection_membership(selection: RoutingSelection) -> Any:
    membership = torch.zeros(
        (int(selection.indices.shape[0]), int(selection.num_experts)),
        dtype=torch.int16,
    )
    membership.scatter_add_(
        1,
        selection.indices,
        selection.active_mask.to(dtype=torch.int16),
    )
    return membership > 0


def compare_routing_selections(
    reference: RoutingSelection,
    candidate: RoutingSelection,
) -> RoutingSelectionComparison:
    """Compare paired active expert sets, including variable-cardinality routes."""
    if int(reference.num_experts) != int(candidate.num_experts):
        raise ValueError("Paired routing selections require shared expert numbering.")
    if int(reference.indices.shape[0]) != int(candidate.indices.shape[0]):
        raise ValueError("Paired routing selections must contain the same tokens.")
    reference_membership = _selection_membership(reference)
    candidate_membership = _selection_membership(candidate)
    reference_count = reference_membership.sum(dim=1)
    candidate_count = candidate_membership.sum(dim=1)
    if bool((reference_count <= 0).any().item()) or bool(
        (candidate_count <= 0).any().item()
    ):
        raise ValueError("Every compared token must have at least one active route.")
    intersection = (reference_membership & candidate_membership).sum(dim=1)
    union = (reference_membership | candidate_membership).sum(dim=1)
    unchanged = (reference_membership == candidate_membership).all(dim=1)
    equal_cardinality = reference_count == candidate_count
    overlap = intersection.double() / reference_count.double()
    equal_overlap = overlap[equal_cardinality]
    deviations = (
        reference_count[equal_cardinality] - intersection[equal_cardinality]
    ).to(dtype=torch.int64)
    histogram: dict[int, int] = {}
    for value in deviations.tolist():
        key = int(value)
        histogram[key] = histogram.get(key, 0) + 1
    return RoutingSelectionComparison(
        token_count=int(reference_count.numel()),
        unchanged_tokens=int(unchanged.sum().item()),
        equal_cardinality_tokens=int(equal_cardinality.sum().item()),
        overlap_sum=float(equal_overlap.sum().item()),
        jaccard_sum=float(
            (intersection.double() / union.double()).sum().item()
        ),
        reference_recall_sum=float(overlap.sum().item()),
        candidate_precision_sum=float(
            (intersection.double() / candidate_count.double()).sum().item()
        ),
        reference_active_sum=int(reference_count.sum().item()),
        candidate_active_sum=int(candidate_count.sum().item()),
        symmetric_difference_sum=int(
            (reference_count + candidate_count - (2 * intersection)).sum().item()
        ),
        deviation_histogram=histogram,
    )


@dataclass
class RoutingAgreementAccumulator:
    """Bounded aggregate over paired route captures."""

    token_count: int = 0
    unchanged_tokens: int = 0
    equal_cardinality_tokens: int = 0
    overlap_sum: float = 0.0
    jaccard_sum: float = 0.0
    reference_recall_sum: float = 0.0
    candidate_precision_sum: float = 0.0
    reference_active_sum: int = 0
    candidate_active_sum: int = 0
    symmetric_difference_sum: int = 0
    deviation_histogram: dict[int, int] = field(default_factory=dict)
    invocation_count: int = 0

    def add(self, comparison: RoutingSelectionComparison) -> None:
        self.token_count += int(comparison.token_count)
        self.unchanged_tokens += int(comparison.unchanged_tokens)
        self.equal_cardinality_tokens += int(comparison.equal_cardinality_tokens)
        self.overlap_sum += float(comparison.overlap_sum)
        self.jaccard_sum += float(comparison.jaccard_sum)
        self.reference_recall_sum += float(comparison.reference_recall_sum)
        self.candidate_precision_sum += float(comparison.candidate_precision_sum)
        self.reference_active_sum += int(comparison.reference_active_sum)
        self.candidate_active_sum += int(comparison.candidate_active_sum)
        self.symmetric_difference_sum += int(comparison.symmetric_difference_sum)
        for key, value in comparison.deviation_histogram.items():
            self.deviation_histogram[int(key)] = (
                self.deviation_histogram.get(int(key), 0) + int(value)
            )
        self.invocation_count += 1

    def merge(self, other: "RoutingAgreementAccumulator") -> None:
        comparison = RoutingSelectionComparison(
            token_count=other.token_count,
            unchanged_tokens=other.unchanged_tokens,
            equal_cardinality_tokens=other.equal_cardinality_tokens,
            overlap_sum=other.overlap_sum,
            jaccard_sum=other.jaccard_sum,
            reference_recall_sum=other.reference_recall_sum,
            candidate_precision_sum=other.candidate_precision_sum,
            reference_active_sum=other.reference_active_sum,
            candidate_active_sum=other.candidate_active_sum,
            symmetric_difference_sum=other.symmetric_difference_sum,
            deviation_histogram=dict(other.deviation_histogram),
        )
        self.add(comparison)
        self.invocation_count += int(other.invocation_count) - 1

    def report(self) -> dict[str, Any]:
        if self.token_count <= 0:
            raise ValueError("Routing agreement accumulator has no observations.")
        comparison = RoutingSelectionComparison(
            token_count=self.token_count,
            unchanged_tokens=self.unchanged_tokens,
            equal_cardinality_tokens=self.equal_cardinality_tokens,
            overlap_sum=self.overlap_sum,
            jaccard_sum=self.jaccard_sum,
            reference_recall_sum=self.reference_recall_sum,
            candidate_precision_sum=self.candidate_precision_sum,
            reference_active_sum=self.reference_active_sum,
            candidate_active_sum=self.candidate_active_sum,
            symmetric_difference_sum=self.symmetric_difference_sum,
            deviation_histogram=dict(self.deviation_histogram),
        )
        return {
            "router_invocations": int(self.invocation_count),
            **comparison.to_dict(),
        }


class RoutingSelectionCapture:
    """Temporary provider-target hooks for one forward phase."""

    def __init__(self, targets: Mapping[str, RoutingSelectionTarget]) -> None:
        if not targets:
            raise ValueError("Routing selection capture requires targets.")
        self.targets: dict[str, RoutingSelectionTarget] = {}
        for raw_name, target in targets.items():
            if not isinstance(target, RoutingSelectionTarget):
                raise TypeError("Routing selection targets must use their typed contract.")
            target.validate()
            name = str(raw_name)
            if name != target.name or name in self.targets:
                raise ValueError("Routing selection target names must match and be unique.")
            self.targets[name] = target
        self._records: dict[str, list[RoutingSelection]] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "RoutingSelectionCapture":
        if self._handles:
            raise RuntimeError("Routing selection capture is already active.")
        self._records = {name: [] for name in self.targets}
        try:
            for name, target in self.targets.items():

                def _hook(
                    module: Any,
                    _inputs: Any,
                    _output: Any,
                    *,
                    _name=name,
                    _experts=int(target.num_experts),
                ) -> None:
                    indices = getattr(module, "last_top_indices", None)
                    if indices is None:
                        return
                    active_mask = getattr(module, "last_route_active_mask", None)
                    self._records[_name].append(
                        RoutingSelection.from_tensors(
                            indices,
                            num_experts=_experts,
                            active_mask=active_mask,
                        )
                    )

                self._handles.append(target.router.register_forward_hook(_hook))
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles = []

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def snapshots(self) -> dict[str, tuple[RoutingSelection, ...]]:
        missing = [name for name, values in self._records.items() if not values]
        if missing:
            raise ValueError(
                "Routing selection capture observed no routes for: "
                + ", ".join(sorted(missing))
            )
        return {name: tuple(values) for name, values in self._records.items()}


def compare_routing_capture_pairs(
    reference: Mapping[str, tuple[RoutingSelection, ...]],
    candidate: Mapping[str, tuple[RoutingSelection, ...]],
    accumulators: Mapping[str, RoutingAgreementAccumulator],
) -> None:
    """Join captures by provider target and router invocation order."""
    if set(reference) != set(candidate) or set(reference) != set(accumulators):
        raise ValueError("Routing capture pairs and accumulators must share target names.")
    for name in reference:
        reference_calls = reference[name]
        candidate_calls = candidate[name]
        if len(reference_calls) != len(candidate_calls):
            raise ValueError(
                f"Routing target {name!r} changed invocation count between modes."
            )
        for reference_selection, candidate_selection in zip(
            reference_calls,
            candidate_calls,
            strict=True,
        ):
            accumulators[name].add(
                compare_routing_selections(
                    reference_selection,
                    candidate_selection,
                )
            )


def build_routing_mode_agreement_evidence(
    accumulators: Mapping[str, RoutingAgreementAccumulator],
    *,
    calibration_steps: int,
    dataset_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str,
) -> dict[str, Any]:
    """Build a versioned, lineage-bound aggregate evidence document."""
    lineage = {
        "dataset_snapshot_id": str(dataset_snapshot_id).strip(),
        "model_snapshot_id": str(model_snapshot_id).strip(),
        "config_snapshot_id": str(config_snapshot_id).strip(),
    }
    if not all(lineage.values()):
        raise ValueError("Routing agreement evidence requires complete snapshot lineage.")
    if int(calibration_steps) <= 0 or not accumulators:
        raise ValueError("Routing agreement evidence requires observations.")
    overall = RoutingAgreementAccumulator()
    modules: dict[str, Any] = {}
    for name, accumulator in accumulators.items():
        modules[str(name)] = accumulator.report()
        overall.merge(accumulator)
    return {
        "schema": "mirai.routing_mode_agreement",
        "schema_version": 1,
        "comparison": {
            "reference_mode": "training",
            "candidate_mode": "inference",
            "pairing": "same_batch_noise_timestep_token_layer_and_router_invocation",
            "fixed_cardinality_metric": "mean_intersection_over_k",
            "variable_cardinality_metric": "active_set_jaccard_and_exact_set_churn",
        },
        "calibration_steps": int(calibration_steps),
        "lineage": lineage,
        "modules": modules,
        "overall": overall.report(),
    }


def validate_routing_mode_agreement_evidence(payload: Mapping[str, Any]) -> None:
    """Reject malformed or lineage-free routing-mode evidence."""
    if payload.get("schema") != "mirai.routing_mode_agreement":
        raise ValueError("Unsupported routing agreement evidence schema.")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported routing agreement evidence schema version.")
    if int(payload.get("calibration_steps", 0)) <= 0:
        raise ValueError("Routing agreement evidence requires positive calibration_steps.")
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping) or not all(
        str(lineage.get(key, "")).strip()
        for key in (
            "dataset_snapshot_id",
            "model_snapshot_id",
            "config_snapshot_id",
        )
    ):
        raise ValueError("Routing agreement evidence requires complete snapshot lineage.")
    comparison = payload.get("comparison")
    if not isinstance(comparison, Mapping) or comparison.get("pairing") != (
        "same_batch_noise_timestep_token_layer_and_router_invocation"
    ):
        raise ValueError("Routing agreement evidence has incompatible pairing semantics.")
    modules = payload.get("modules")
    overall = payload.get("overall")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("Routing agreement evidence requires module reports.")
    if not isinstance(overall, Mapping):
        raise ValueError("Routing agreement evidence requires an overall report.")
    required = {
        "router_invocations",
        "token_count",
        "unchanged_token_fraction",
        "changed_token_fraction",
        "equal_cardinality_token_fraction",
        "matched_cardinality_overlap",
        "mean_jaccard",
    }
    for label, report in [*modules.items(), ("overall", overall)]:
        if not isinstance(report, Mapping) or not required.issubset(report):
            raise ValueError(f"Routing agreement report {label!r} is incomplete.")
        if int(report["router_invocations"]) <= 0 or int(report["token_count"]) <= 0:
            raise ValueError(f"Routing agreement report {label!r} is empty.")
        for key in required - {"router_invocations", "token_count"}:
            value = float(report[key])
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Routing agreement report {label!r} has invalid {key}."
                )


def save_routing_mode_agreement_evidence(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    """Atomically persist validated aggregate JSON evidence."""
    validate_routing_mode_agreement_evidence(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def selection_margin(scores: torch.Tensor, *, top_k: int) -> torch.Tensor:
    """Return each token's score gap at the top-k selection boundary."""
    if torch is None:
        raise RuntimeError("Routing agreement diagnostics require torch.")
    if not torch.is_tensor(scores) or scores.ndim != 2:
        raise ValueError("scores must be a rank-2 tensor.")
    k = int(top_k)
    if k <= 0 or k >= int(scores.shape[1]):
        raise ValueError("top_k must be between 1 and num_experts - 1.")
    ordered = torch.sort(scores, dim=1, descending=True).values
    return ordered[:, k - 1] - ordered[:, k]


@dataclass(frozen=True)
class RoutingAgreementReport:
    """Selection stability and reference-boundary diagnostics for one router."""

    agreement: float
    changed_token_fraction: float
    margin_p05: float
    margin_min: float


def compare_router_selections(
    reference_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
    *,
    top_k: int,
    num_experts: int,
) -> RoutingAgreementReport:
    """Compare top-k expert sets for the same tokens under two router weights."""
    if torch is None:
        raise RuntimeError("Routing agreement diagnostics require torch.")
    if not torch.is_tensor(reference_scores) or reference_scores.ndim != 2:
        raise ValueError("reference_scores must be a rank-2 tensor.")
    if not torch.is_tensor(candidate_scores) or candidate_scores.ndim != 2:
        raise ValueError("candidate_scores must be a rank-2 tensor.")
    if not reference_scores.is_floating_point():
        raise ValueError("reference_scores must use a floating-point dtype.")
    if not candidate_scores.is_floating_point():
        raise ValueError("candidate_scores must use a floating-point dtype.")

    reference_experts = int(reference_scores.shape[1])
    candidate_experts = int(candidate_scores.shape[1])
    experts = int(num_experts)
    if reference_experts != candidate_experts or reference_experts != experts:
        raise ValueError(
            "Router selection comparison requires a shared expert numbering: "
            "reference_scores, candidate_scores, and num_experts must describe "
            "the same experts."
        )
    if reference_scores.shape != candidate_scores.shape:
        raise ValueError(
            "reference_scores and candidate_scores must have equal shape for "
            "the same tokens."
        )
    if int(reference_scores.shape[0]) <= 0:
        raise ValueError("routing score matrices must contain at least one token.")
    k = int(top_k)
    if k < 1 or k >= experts:
        raise ValueError("top_k must be between 1 and num_experts - 1.")

    reference_indices = torch.topk(reference_scores, k=k, dim=1).indices
    candidate_indices = torch.topk(candidate_scores, k=k, dim=1).indices
    agreement = topk_set_agreement(
        reference_indices,
        candidate_indices,
        num_experts=experts,
    )
    reference_sets = torch.sort(reference_indices, dim=1).values
    candidate_sets = torch.sort(
        candidate_indices.to(device=reference_sets.device),
        dim=1,
    ).values
    changed = (reference_sets != candidate_sets).any(dim=1)
    margins = selection_margin(reference_scores, top_k=k).float()
    return RoutingAgreementReport(
        agreement=agreement,
        changed_token_fraction=float(changed.float().mean().item()),
        margin_p05=float(torch.quantile(margins, 0.05).item()),
        margin_min=float(margins.min().item()),
    )


class SelectionMarginProbe:
    """Accumulate per-layer selection-boundary margins for one step.

    A small margin means the token's route is one rounding step away from
    changing, so this is the signal that makes a later numerics change
    interpretable. Nothing is retained across steps: ``summary`` drains the
    accumulator, and no device tensor outlives the observing call.
    """

    def __init__(self) -> None:
        self._margin_p05s: list[float] = []

    def observe(self, scores: torch.Tensor, *, top_k: int) -> None:
        """Observe one layer without retaining device tensors."""
        margins = selection_margin(scores.detach(), top_k=top_k).float()
        self._margin_p05s.append(
            float(torch.quantile(margins, 0.05).detach().cpu().item())
        )

    def summary(self) -> dict[str, float]:
        """Return step scalars and clear step-local accumulators."""
        if not self._margin_p05s:
            return {}
        result = {
            "moe_selection_margin_p05": float(
                sum(self._margin_p05s) / len(self._margin_p05s)
            ),
            "moe_selection_margin_min": float(min(self._margin_p05s)),
        }
        self._margin_p05s.clear()
        return result


__all__ = [
    "RoutingAgreementAccumulator",
    "RoutingAgreementReport",
    "RoutingSelection",
    "RoutingSelectionCapture",
    "RoutingSelectionComparison",
    "RoutingSelectionTarget",
    "SelectionMarginProbe",
    "build_routing_mode_agreement_evidence",
    "compare_routing_capture_pairs",
    "compare_routing_selections",
    "compare_router_selections",
    "save_routing_mode_agreement_evidence",
    "selection_margin",
    "topk_overlap_fraction",
    "topk_set_agreement",
    "validate_routing_mode_agreement_evidence",
]
