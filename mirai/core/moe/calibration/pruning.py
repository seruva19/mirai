"""Structured expert pruning criteria and packed-artifact transform contracts.

The calibrated criteria remove low-saliency experts. AIMER instead removes high
normalized-L1/L2 experts using pretrained weights alone. Active-per-token is
fixed by top-k and unchanged; pruning shrinks packed-state / PCIe / VRAM
footprint. It is not lossless and is not in the hot training path.

The calibration criteria implement the unified expert-pruning score

``sum(route_weight ** alpha * ||expert_output|| ** beta) / count ** b``

for frequency ``(b, alpha, beta)=(0, 0, 0)``, REAP ``(1, 1, 1)``,
MAN ``(1, 0, 1)``, and MSAN ``(1, 0, 2)``. Expert outputs are observed before
router weighting. The legacy frequency path remains available for cheap
router-only calibration. Only pruning zero-frequency experts has the stronger
observed-token router-equivalence property; activation criteria make no such
claim. AIMER follows Eq. 4 of arXiv:2603.18492 and makes no claim that its LLM
quality results transfer to video.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from mirai.core.moe.calibration.projection import (
    ExpertProjectionSource,
    projection_block_experts,
)
from mirai.core.moe.runtime.specs import ExpertTensorSpec

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

EXPERT_PRUNING_CALIBRATION_FORMAT = "mirai.moe.expert_pruning_calibration"
EXPERT_PRUNING_CALIBRATION_SCHEMA_VERSION = 1
EXPERT_PRUNING_CALIBRATION_METADATA_KEY = "mirai_expert_pruning_calibration"
EXPERT_PRUNING_CALIBRATION_CRITERIA = ("frequency", "reap", "man", "msan")
EXPERT_PRUNING_CRITERIA = (*EXPERT_PRUNING_CALIBRATION_CRITERIA, "aimer")


def normalize_expert_pruning_criterion(value: str) -> str:
    """Return a supported criterion name or fail before calibration starts."""
    criterion = str(value).strip().lower()
    if criterion not in EXPERT_PRUNING_CRITERIA:
        supported = ", ".join(EXPERT_PRUNING_CRITERIA)
        raise ValueError(
            f"Unsupported expert-pruning criterion {value!r}; expected {supported}."
        )
    return criterion


def normalize_calibrated_expert_pruning_criterion(value: str) -> str:
    """Return a criterion that requires routed calibration observations."""

    criterion = normalize_expert_pruning_criterion(value)
    if criterion not in EXPERT_PRUNING_CALIBRATION_CRITERIA:
        supported = ", ".join(EXPERT_PRUNING_CALIBRATION_CRITERIA)
        raise ValueError(
            f"Expert-pruning criterion {criterion!r} is calibration-free; "
            f"calibration accepts only {supported}."
        )
    return criterion


@dataclasses.dataclass(frozen=True)
class ExpertPruningCalibrationTarget:
    """Provider-owned routed-output host exposed to generic calibration."""

    name: str
    host: Any
    num_experts: int

    def validate(self) -> ExpertPruningCalibrationTarget:
        if not self.name:
            raise ValueError("Expert-pruning calibration target name must be non-empty.")
        if int(self.num_experts) < 2:
            raise ValueError("Expert-pruning calibration requires at least two experts.")
        if not callable(getattr(self.host, "set_expert_output_observer", None)):
            raise TypeError(
                "Expert-pruning calibration hosts must expose "
                "set_expert_output_observer()."
            )
        if not callable(getattr(self.host, "get_expert_output_observer", None)):
            raise TypeError(
                "Expert-pruning calibration hosts must expose "
                "get_expert_output_observer()."
            )
        return self


@dataclasses.dataclass(frozen=True)
class ExpertPruningEvidence:
    """Per-expert sufficient statistics for one exact pruning criterion."""

    criterion: str
    score_sum: Any
    selected_count: Any

    @property
    def num_experts(self) -> int:
        return int(torch.as_tensor(self.selected_count).numel())

    def validate(self) -> ExpertPruningEvidence:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Expert-pruning calibration requires torch.")
        criterion = normalize_calibrated_expert_pruning_criterion(self.criterion)
        scores = torch.as_tensor(self.score_sum)
        counts = torch.as_tensor(self.selected_count)
        if scores.ndim != 1 or int(scores.numel()) < 2:
            raise ValueError("Expert-pruning evidence requires at least two experts.")
        if counts.shape != scores.shape:
            raise ValueError("Expert-pruning evidence shapes are inconsistent.")
        if not bool(torch.isfinite(scores).all().item()):
            raise ValueError("Expert-pruning evidence contains non-finite scores.")
        if bool(torch.any(scores < 0).item()) or bool(torch.any(counts < 0).item()):
            raise ValueError("Expert-pruning evidence must be non-negative.")
        if int(counts.sum().item()) == 0:
            raise ValueError("Expert-pruning calibration observed no active routes.")
        if criterion == "frequency" and not torch.equal(
            scores.to(torch.float64), counts.to(torch.float64)
        ):
            raise ValueError("Frequency evidence score_sum must equal selected_count.")
        return self

    def scores(self) -> Any:
        """Materialize the exact per-expert score used for ranking."""
        self.validate()
        numerator = torch.as_tensor(self.score_sum, dtype=torch.float64)
        if normalize_calibrated_expert_pruning_criterion(self.criterion) == "frequency":
            return numerator
        counts = torch.as_tensor(self.selected_count, dtype=torch.float64)
        return numerator / counts.clamp_min(1.0)


class ExpertPruningSaliencyAccumulator:
    """Accumulate route-local sufficient statistics without retaining autograd."""

    def __init__(self, num_experts: int, *, criterion: str) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Expert-pruning calibration requires torch.")
        count = int(num_experts)
        if count < 2:
            raise ValueError("Expert-pruning calibration requires at least two experts.")
        self.criterion = normalize_calibrated_expert_pruning_criterion(criterion)
        self.score_sum = torch.zeros(count, dtype=torch.float64)
        self.selected_count = torch.zeros(count, dtype=torch.int64)

    @property
    def num_experts(self) -> int:
        return int(self.selected_count.numel())

    def record(
        self,
        expert_ids: Any,
        routing_weights: Any,
        expert_outputs: Any,
    ) -> None:
        """Record aligned active routes using raw, pre-router-weight outputs."""
        ids = torch.as_tensor(expert_ids).detach().reshape(-1).to(torch.long)
        weights = torch.as_tensor(routing_weights).detach().reshape(-1)
        outputs = torch.as_tensor(expert_outputs).detach()
        if outputs.ndim < 2 or int(outputs.shape[0]) != int(ids.numel()):
            raise ValueError("Expert outputs must align with the observed routes.")
        if weights.shape != ids.shape or ids.numel() == 0:
            raise ValueError("Expert ids and routing weights must be aligned and non-empty.")
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= self.num_experts:
            raise ValueError("Expert-pruning calibration observed an out-of-range expert.")
        compute_dtype = torch.float32 if outputs.is_cuda else torch.float64
        weights = weights.to(device=outputs.device, dtype=compute_dtype)
        ids = ids.to(device=outputs.device)
        flat_outputs = outputs.to(dtype=compute_dtype).reshape(int(ids.numel()), -1)
        if not bool(torch.isfinite(weights).all().item()) or not bool(
            torch.isfinite(flat_outputs).all().item()
        ):
            raise ValueError("Expert-pruning observations must be finite.")
        if bool(torch.any(weights < 0).item()):
            raise ValueError("Expert-pruning routing weights must be non-negative.")
        active = weights > 0
        ids = ids[active]
        weights = weights[active]
        flat_outputs = flat_outputs[active]
        if ids.numel() == 0:
            return
        counts = torch.bincount(ids, minlength=self.num_experts)
        if self.criterion == "frequency":
            contributions = torch.ones_like(weights)
        else:
            norms = torch.linalg.vector_norm(flat_outputs, ord=2, dim=1)
            if self.criterion == "reap":
                contributions = weights * norms
            elif self.criterion == "man":
                contributions = norms
            else:
                contributions = norms.square()
        summary = torch.zeros(
            self.num_experts,
            device=outputs.device,
            dtype=compute_dtype,
        )
        summary.scatter_add_(0, ids, contributions)
        self.add_summary(summary, counts)

    def add_summary(self, score_sum: Any, selected_count: Any) -> None:
        scores = torch.as_tensor(score_sum).detach().to(device="cpu", dtype=torch.float64)
        counts = (
            torch.as_tensor(selected_count)
            .detach()
            .to(device="cpu", dtype=torch.int64)
        )
        if scores.shape != self.score_sum.shape or counts.shape != self.selected_count.shape:
            raise ValueError("Expert-pruning summary shape does not match the accumulator.")
        self.score_sum += scores
        self.selected_count += counts

    def evidence(self) -> ExpertPruningEvidence:
        return ExpertPruningEvidence(
            criterion=self.criterion,
            score_sum=self.score_sum.clone(),
            selected_count=self.selected_count.clone(),
        ).validate()


class ExpertPruningRoutedOutputObserver:
    """Translate dispatch route positions into criterion observations."""

    def __init__(self, accumulator: ExpertPruningSaliencyAccumulator) -> None:
        self.accumulator = accumulator
        self.abort_capture()

    @property
    def is_enabled(self) -> bool:
        return True

    def bind_routes(self, expert_indices: Any, routing_weights: Any) -> None:
        indices = torch.as_tensor(expert_indices).detach()
        weights = torch.as_tensor(routing_weights).detach()
        if indices.shape != weights.shape or indices.ndim != 2:
            raise ValueError("Bound pruning routes must be aligned [tokens, top_k].")
        self._flat_experts = indices.reshape(-1)
        self._flat_weights = weights.reshape(-1)

    def begin_routes(self, *, num_tokens: int, top_k: int, device: Any) -> None:
        del device
        if self._active:
            raise RuntimeError("Expert-pruning routed capture is already active.")
        if int(num_tokens) <= 0 or int(top_k) <= 0:
            raise ValueError("Expert-pruning routed capture requires non-empty routes.")
        self._active = True
        self._expected_slots = int(num_tokens) * int(top_k)
        self._captured_positions = 0

    def capture_routes(self, expert_outputs: Any, route_positions: Any) -> None:
        if not self._active:
            raise RuntimeError("Expert-pruning routed capture was not started.")
        self._record_positions(expert_outputs, route_positions)
        self._captured_positions += int(torch.as_tensor(route_positions).numel())

    def end_routes(self) -> None:
        if not self._active:
            raise RuntimeError("Expert-pruning routed capture was not started.")
        try:
            if self._captured_positions <= 0:
                raise ValueError("Expert-pruning routed capture observed no routes.")
        finally:
            self.abort_capture()

    def abort_capture(self) -> None:
        self._active = False
        self._expected_slots = 0
        self._captured_positions = 0
        self._flat_experts: Any | None = None
        self._flat_weights: Any | None = None

    def capture_sorted(
        self,
        expert_outputs: Any,
        sorted_positions: Any,
        *,
        num_tokens: int,
        top_k: int,
    ) -> None:
        expected = int(num_tokens) * int(top_k)
        if self._flat_experts is None or int(self._flat_experts.numel()) != expected:
            raise RuntimeError("Expert-pruning route metadata is missing or stale.")
        try:
            self._record_positions(expert_outputs, sorted_positions)
        finally:
            self.abort_capture()

    def _record_positions(self, expert_outputs: Any, route_positions: Any) -> None:
        if self._flat_experts is None or self._flat_weights is None:
            raise RuntimeError("Expert-pruning route metadata was not bound.")
        positions = torch.as_tensor(route_positions).detach().reshape(-1).to(torch.long)
        outputs = torch.as_tensor(expert_outputs).detach()
        if outputs.ndim < 2 or int(outputs.shape[0]) != int(positions.numel()):
            raise ValueError("Routed expert outputs and route positions must align.")
        if positions.numel() == 0:
            return
        if int(positions.min().item()) < 0 or int(positions.max().item()) >= int(
            self._flat_experts.numel()
        ):
            raise ValueError("Expert-pruning route position is out of range.")
        positions = positions.to(device=self._flat_experts.device)
        self.accumulator.record(
            self._flat_experts.index_select(0, positions),
            self._flat_weights.index_select(0, positions),
            outputs,
        )


def save_expert_pruning_evidence(
    path: str | Path,
    evidence_by_module: Mapping[str, ExpertPruningEvidence],
    *,
    dataset_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str,
) -> None:
    """Persist criterion evidence as a lineage-bound safetensors artifact."""
    from safetensors.torch import save_file

    if not evidence_by_module:
        raise ValueError("Expert-pruning calibration evidence cannot be empty.")
    lineage = {
        "dataset_snapshot_id": str(dataset_snapshot_id).strip(),
        "model_snapshot_id": str(model_snapshot_id).strip(),
        "config_snapshot_id": str(config_snapshot_id).strip(),
    }
    if not all(lineage.values()):
        raise ValueError("Expert-pruning calibration requires complete snapshot lineage.")
    criteria = {
        normalize_calibrated_expert_pruning_criterion(evidence.criterion)
        for evidence in evidence_by_module.values()
    }
    if len(criteria) != 1:
        raise ValueError("One pruning artifact cannot mix calibration criteria.")
    tensors: dict[str, Any] = {}
    modules: list[dict[str, Any]] = []
    for index, (name, evidence) in enumerate(evidence_by_module.items()):
        evidence.validate()
        prefix = f"module_{index:04d}"
        score_key = f"{prefix}.score_sum"
        count_key = f"{prefix}.selected_count"
        tensors[score_key] = torch.as_tensor(
            evidence.score_sum, dtype=torch.float64
        ).cpu().contiguous()
        tensors[count_key] = torch.as_tensor(
            evidence.selected_count, dtype=torch.int64
        ).cpu().contiguous()
        modules.append(
            {
                "name": str(name),
                "num_experts": evidence.num_experts,
                "score_sum": score_key,
                "selected_count": count_key,
            }
        )
    manifest = {
        "format": EXPERT_PRUNING_CALIBRATION_FORMAT,
        "schema_version": EXPERT_PRUNING_CALIBRATION_SCHEMA_VERSION,
        "criterion": next(iter(criteria)),
        **lineage,
        "modules": modules,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            EXPERT_PRUNING_CALIBRATION_METADATA_KEY: json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            )
        },
    )


def load_expert_pruning_evidence(
    path: str | Path,
    *,
    expected_criterion: str | None = None,
) -> tuple[dict[str, ExpertPruningEvidence], dict[str, str]]:
    """Load safe calibration evidence and return its mandatory lineage."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    input_path = Path(path)
    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        raw_manifest = (handle.metadata() or {}).get(
            EXPERT_PRUNING_CALIBRATION_METADATA_KEY
        )
    if raw_manifest is None:
        raise ValueError("Pruning calibration artifact is missing its Mirai manifest.")
    manifest = json.loads(raw_manifest)
    if manifest.get("format") != EXPERT_PRUNING_CALIBRATION_FORMAT:
        raise ValueError(
            f"Unsupported pruning calibration format {manifest.get('format')!r}."
        )
    if (
        int(manifest.get("schema_version", 0))
        != EXPERT_PRUNING_CALIBRATION_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported pruning calibration schema {manifest.get('schema_version')!r}."
        )
    criterion = normalize_calibrated_expert_pruning_criterion(
        str(manifest.get("criterion", ""))
    )
    if (
        expected_criterion is not None
        and criterion
        != normalize_calibrated_expert_pruning_criterion(
            expected_criterion
        )
    ):
        raise ValueError(
            f"Pruning calibration criterion mismatch: expected {expected_criterion!r}, "
            f"found {criterion!r}."
        )
    lineage = {
        key: str(manifest.get(key, ""))
        for key in ("dataset_snapshot_id", "model_snapshot_id", "config_snapshot_id")
    }
    if not all(lineage.values()):
        raise ValueError("Pruning calibration artifact has incomplete lineage.")
    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("Pruning calibration manifest has no modules.")
    tensors = load_file(str(input_path), device="cpu")
    loaded: dict[str, ExpertPruningEvidence] = {}
    for spec in modules:
        if not isinstance(spec, Mapping):
            raise ValueError("Pruning calibration module entries must be objects.")
        name = str(spec.get("name", ""))
        if not name or name in loaded:
            raise ValueError("Pruning calibration module names must be unique.")
        try:
            evidence = ExpertPruningEvidence(
                criterion=criterion,
                score_sum=tensors[str(spec["score_sum"])],
                selected_count=tensors[str(spec["selected_count"])],
            ).validate()
        except KeyError as exc:
            raise ValueError(
                f"Pruning calibration tensor {exc.args[0]!r} is missing."
            ) from exc
        if evidence.num_experts != int(spec.get("num_experts", 0)):
            raise ValueError(
                f"Pruning calibration expert count mismatch for module {name!r}."
            )
        loaded[name] = evidence
    return loaded, lineage


# The mass-accumulation bincount primitive and the deterministic hot-expert
# selector below MIRROR the routing-guided calibration seam
# (mirai/core/moe/calibration/selection.py: _add_counts / select_hot_experts). They
# are re-implemented rather than imported because the optional-feature isolation
# contract forbids one feature owner from importing another; keeping them local
# preserves that hard invariant while reusing the exact algorithm.


def _add_counts(
    accumulator: dict[str, Any], name: str, indices: Any, num_experts: int
) -> None:
    """Accumulate routed counts for ``name`` (global mass), CPU float64."""
    counts = (
        torch.bincount(
            indices.detach().reshape(-1).to(torch.long), minlength=int(num_experts)
        )
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )
    prior = accumulator.get(name)
    accumulator[name] = counts if prior is None else prior + counts


def select_hot_experts(counts: Any, fraction: float) -> list[int]:
    """Return the ``ceil(fraction*E)`` hottest expert ids, tie-break by id."""
    num_experts = int(counts.numel())
    if num_experts == 0:
        return []
    k = max(1, math.ceil(float(fraction) * num_experts))
    k = min(k, num_experts)
    values = [float(v) for v in counts.reshape(-1).tolist()]
    order = sorted(range(num_experts), key=lambda i: (-values[i], i))
    return sorted(order[:k])


@dataclasses.dataclass(frozen=True)
class AimerExpertScores:
    """Weight-only AIMER sufficient statistics for one grouped-expert module."""

    scores: Any
    absolute_sum: Any
    squared_sum: Any
    elements_per_expert: int

    def validate(self) -> AimerExpertScores:
        if torch is None:  # pragma: no cover
            raise RuntimeError("AIMER expert scoring requires torch.")
        scores = torch.as_tensor(self.scores, dtype=torch.float64)
        absolute_sum = torch.as_tensor(self.absolute_sum, dtype=torch.float64)
        squared_sum = torch.as_tensor(self.squared_sum, dtype=torch.float64)
        if scores.ndim != 1 or int(scores.numel()) < 2:
            raise ValueError("AIMER requires at least two expert scores.")
        if absolute_sum.shape != scores.shape or squared_sum.shape != scores.shape:
            raise ValueError("AIMER sufficient-statistic shapes are inconsistent.")
        if int(self.elements_per_expert) < 1:
            raise ValueError("AIMER requires a positive parameter count per expert.")
        if not bool(
            torch.isfinite(scores).all()
            and torch.isfinite(absolute_sum).all()
            and torch.isfinite(squared_sum).all()
        ):
            raise ValueError("AIMER sufficient statistics must be finite.")
        if bool(torch.any(absolute_sum < 0).item()) or bool(
            torch.any(squared_sum <= 0).item()
        ):
            raise ValueError("AIMER requires nonzero expert weights.")
        lower = 1.0 / math.sqrt(int(self.elements_per_expert))
        tolerance = 1e-6
        if bool(torch.any(scores < lower - tolerance).item()) or bool(
            torch.any(scores > 1.0 + tolerance).item()
        ):
            raise ValueError("AIMER scores violate their analytical bounds.")
        return self


def aimer_expert_scores(
    source: ExpertProjectionSource,
    *,
    max_block_elements: int,
    device: Any = "cpu",
    dtype: Any = None,
) -> AimerExpertScores:
    """Compute Eq. 4 of AIMER from bounded dense expert-projection blocks.

    The three expert projections are treated as one flattened vector.  This is
    the calibration-free ranking statistic from arXiv:2603.18492; larger scores
    are *more removable*, the opposite direction from activation saliency.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("AIMER expert scoring requires torch.")
    expert_count = int(source.num_experts)
    if expert_count < 2:
        raise ValueError("AIMER expert scoring requires at least two experts.")
    budget = int(max_block_elements)
    if budget < 1:
        raise ValueError("AIMER max_block_elements must be positive.")
    requested_device = torch.device(device)
    compute_dtype = (
        torch.float64
        if dtype is None and requested_device.type == "cpu"
        else torch.float32
        if dtype is None
        else dtype
    )
    if compute_dtype not in {torch.float32, torch.float64}:
        raise ValueError("AIMER scoring dtype must be float32 or float64.")
    specs = tuple(
        spec.validate() for spec in source.prototype_projection_specs()
    )
    if not specs:
        raise ValueError("AIMER projection source did not declare any weights.")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("AIMER projection names must be unique.")

    absolute_sum = torch.zeros(expert_count, dtype=torch.float64)
    squared_sum = torch.zeros(expert_count, dtype=torch.float64)
    elements_per_expert = sum(spec.elements_per_expert for spec in specs)
    for spec in specs:
        block_experts = projection_block_experts(
            spec,
            max_block_elements=budget,
        )
        for start in range(0, expert_count, block_experts):
            stop = min(expert_count, start + block_experts)
            block = torch.as_tensor(
                source.load_prototype_projection_block(
                    spec.name,
                    start,
                    stop,
                    device=requested_device,
                    dtype=compute_dtype,
                )
            )
            expected = (stop - start, *spec.shape)
            if tuple(block.shape) != expected:
                raise ValueError(
                    f"AIMER source returned {tuple(block.shape)} for "
                    f"{spec.name!r}; expected {expected}."
                )
            wrong_device = block.device.type != requested_device.type or (
                requested_device.index is not None
                and block.device.index != requested_device.index
            )
            if block.dtype != compute_dtype or wrong_device:
                raise ValueError(
                    f"AIMER source returned the wrong dtype/device for "
                    f"{spec.name!r}."
                )
            if not bool(torch.isfinite(block).all().item()):
                raise ValueError(
                    f"AIMER source returned non-finite {spec.name!r} weights."
                )
            flat = block.reshape(stop - start, -1)
            absolute_sum[start:stop] += (
                flat.abs().sum(dim=1).detach().to(device="cpu", dtype=torch.float64)
            )
            squared_sum[start:stop] += (
                flat.square()
                .sum(dim=1)
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )
            del block, flat

    if bool(torch.any(squared_sum <= 0).item()):
        raise ValueError("AIMER cannot rank an all-zero expert.")
    scores = absolute_sum / torch.sqrt(
        squared_sum * float(elements_per_expert)
    )
    return AimerExpertScores(
        scores=scores,
        absolute_sum=absolute_sum,
        squared_sum=squared_sum,
        elements_per_expert=elements_per_expert,
    ).validate()


# ---------------------------------------------------------------------------
# Stage 1: mass accumulation (thin wrapper over the calibration seam)
# ---------------------------------------------------------------------------


def iter_moe_routers(model: Any) -> Iterator[tuple[str, Any, int]]:
    """Yield ``(module_name, router, num_experts)`` for every MoE router.

    A router is any named module whose class name contains ``Router`` and exposes
    an integer ``num_experts``. Discovery does not require expert adapters, so
    calibration can inspect a frozen base model.
    """
    for name, module in model.named_modules():
        if "router" not in type(module).__name__.lower():
            continue
        num_experts = getattr(module, "num_experts", None)
        if num_experts is None:
            continue
        yield name, module, int(num_experts)


def _extract_top_indices(module: Any, output: Any) -> Any:
    """Best-effort top-k index tensor from a router forward.

    Prefers a stashed ``last_top_indices`` value and then checks structured or
    tuple router outputs.
    """
    top = getattr(module, "last_top_indices", None)
    if top is not None:
        return top
    top = getattr(output, "topk_indices", None)
    if top is not None:
        return top
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return None


def register_mass_hooks(model: Any, accumulator: dict[str, Any]) -> list[Any]:
    """Register removable router forward hooks that accumulate routed mass.

    Uses the local ``_add_counts`` bincount primitive (mirroring the forward-only
    calibration driver) so mass here is identical to the calibration seam's routed
    counts. Caller MUST remove the returned handles.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert pruning requires torch.")
    handles: list[Any] = []
    for name, router, num_experts in iter_moe_routers(model):

        def _hook(module, _inp, out, _name=name, _num=num_experts):
            top = _extract_top_indices(module, out)
            if top is not None:
                _add_counts(accumulator, _name, top, _num)

        handles.append(router.register_forward_hook(_hook))
    return handles


def accumulate_expert_mass(
    model: Any,
    run_forward: Callable[[int], Any],
    *,
    steps: int,
) -> dict[str, Any]:
    """Accumulate GLOBAL per-layer routed mass over ``steps`` forward-only passes.

    ``run_forward(i)`` executes one forward for calibration step ``i`` (the caller
    owns batch construction, mirroring the training calibration workflow). Returns
    ``{router_module_name: f64 CPU mass tensor [num_experts]}``. Forward-only: no
    grad, and the model's train/eval mode is preserved.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert pruning requires torch.")
    if int(steps) <= 0:
        raise ValueError("accumulate_expert_mass requires steps > 0.")
    accumulator: dict[str, Any] = {}
    was_training = bool(getattr(model, "training", False))
    handles = register_mass_hooks(model, accumulator)
    try:
        with torch.no_grad():
            for i in range(int(steps)):
                run_forward(i)
    finally:
        for handle in handles:
            handle.remove()
        if hasattr(model, "train"):
            model.train(was_training)
    return accumulator


# ---------------------------------------------------------------------------
# Stage 2: selection (deterministic, hard per-layer floor keep >= top_k)
# ---------------------------------------------------------------------------


def select_pruned_experts(
    saliency: Mapping[str, Any],
    *,
    keep_fraction: float | None = None,
    score_threshold: float | None = None,
    min_keep: int = 0,
    top_k: int,
    keep_largest: bool = True,
) -> dict[str, tuple[int, ...]]:
    """Return deterministic per-layer keep sets from arbitrary saliency scores.

    Exactly one of ``keep_fraction`` or ``score_threshold`` must be given.
    ``keep_largest=True`` retains high saliency (the calibrated criteria);
    ``False`` retains low scores (AIMER, where larger means more removable).

    Hard floor ``floor = max(top_k, min_keep)``: a layer that would keep fewer
    than ``floor`` experts raises ``ValueError`` -- a pruned layer must still be
    able to supply ``top_k`` distinct experts per token. Returns
    ``{layer: sorted tuple of kept expert ids}``.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert pruning requires torch.")
    if (keep_fraction is None) == (score_threshold is None):
        raise ValueError(
            "select_pruned_experts requires exactly one of keep_fraction / "
            "score_threshold."
        )
    if int(top_k) < 1:
        raise ValueError("select_pruned_experts requires top_k >= 1.")
    if int(min_keep) < 0:
        raise ValueError("select_pruned_experts requires min_keep >= 0.")
    if keep_fraction is not None and not (0.0 < float(keep_fraction) <= 1.0):
        raise ValueError("select_pruned_experts keep_fraction must be in (0, 1].")
    floor = max(int(top_k), int(min_keep))

    result: dict[str, tuple[int, ...]] = {}
    for layer, counts in saliency.items():
        num_experts = int(counts.numel())
        if num_experts == 0:
            raise ValueError(f"select_pruned_experts: layer {layer!r} has no experts.")
        if keep_fraction is not None:
            if bool(keep_largest):
                keep = list(select_hot_experts(counts, float(keep_fraction)))
            else:
                k = max(1, math.ceil(float(keep_fraction) * num_experts))
                values = [float(v) for v in counts.reshape(-1).tolist()]
                keep = sorted(
                    sorted(range(num_experts), key=lambda i: (values[i], i))[:k]
                )
        else:
            values = [float(v) for v in counts.reshape(-1).tolist()]
            keep = [
                expert
                for expert in range(num_experts)
                if (
                    values[expert] > float(score_threshold)
                    if bool(keep_largest)
                    else values[expert] < float(score_threshold)
                )
            ]
        if len(keep) < floor:
            raise ValueError(
                f"select_pruned_experts: layer {layer!r} would keep {len(keep)} "
                f"experts, below the hard floor {floor} (top_k={int(top_k)}, "
                f"min_keep={int(min_keep)}). Loosen the threshold/fraction."
            )
        result[layer] = tuple(sorted(int(e) for e in keep))
    return result


# ---------------------------------------------------------------------------
# Stage 3: structural pruning of the expert-spec tensor contract (pure)
# ---------------------------------------------------------------------------


def layer_key_from_module(spec: ExpertTensorSpec) -> str:
    """Default join key: the PARENT module of ``owner_module``.

    ``a.b.router`` and ``a.b.experts`` share parent ``a.b``, so a router spec and
    its sibling grouped-experts specs resolve to the same per-block keep set.
    """
    owner = str(spec.owner_module)
    return owner.rsplit(".", 1)[0] if "." in owner else ""


@dataclasses.dataclass(frozen=True)
class PrunedExpertTensors:
    """Result of :func:`prune_expert_tensors`."""

    specs: tuple[ExpertTensorSpec, ...]
    tensors: dict[str, Any]
    kept_by_layer: dict[str, tuple[int, ...]]


def _prune_axis(spec: ExpertTensorSpec) -> int | None:
    """Axis that indexes experts for ``spec`` (or ``None`` if not prunable)."""
    if "expert" in spec.layout:
        return spec.layout.index("expert")
    if spec.router and "out" in spec.layout:
        # Router weight [num_experts(out), hidden(in)] -- output rows are experts.
        return spec.layout.index("out")
    return None


def prune_expert_tensors(
    specs: Iterable[ExpertTensorSpec],
    tensors: Mapping[str, Any],
    keep_index_per_layer: Mapping[str, Sequence[int]],
    *,
    layer_key_fn: Callable[[ExpertTensorSpec], str] = layer_key_from_module,
) -> PrunedExpertTensors:
    """Structurally prune expert tensors + router output rows; renumber survivors.

    Pure tensor slicing -- no math on values. For each spec joined (via
    ``layer_key_fn``) to a keep set: ``index_select`` the expert axis (grouped
    tensors) or the router ``out`` axis (router weight rows) to the sorted kept
    ids. Survivors are renumbered contiguously ``0..K-1`` by the sorted order.
    Specs with no keep set, or with neither an expert axis nor a router role, pass
    through byte-identical. Never mutates the input tensors.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert pruning requires torch.")
    out_specs: list[ExpertTensorSpec] = []
    out_tensors: dict[str, Any] = {}
    kept_by_layer: dict[str, tuple[int, ...]] = {}
    for spec in specs:
        tensor = tensors[spec.name]
        key = layer_key_fn(spec)
        keep = keep_index_per_layer.get(key)
        axis = _prune_axis(spec)
        if keep is None or axis is None:
            out_specs.append(spec)
            out_tensors[spec.name] = tensor
            continue
        keep_idx = sorted(int(i) for i in keep)
        index = torch.as_tensor(keep_idx, dtype=torch.long, device=tensor.device)
        pruned = tensor.index_select(axis, index).contiguous()
        out_tensors[spec.name] = pruned
        out_specs.append(dataclasses.replace(spec, shape=tuple(int(d) for d in pruned.shape)))
        kept_by_layer[key] = tuple(keep_idx)
    return PrunedExpertTensors(
        specs=tuple(out_specs), tensors=out_tensors, kept_by_layer=kept_by_layer
    )


# ---------------------------------------------------------------------------
# Stage 4: re-emit a smaller compressed_weights packed-state manifest (offline)
# ---------------------------------------------------------------------------


def _parent(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else ""


def prune_packed_state(
    tensors: Mapping[str, Any],
    manifest: Mapping[str, Any],
    keep_by_module: Mapping[str, Sequence[int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-emit a ``mirai.compressed_weights.packed_state`` with cold experts removed.

    ``keep_by_module`` maps a ``grouped_experts`` module name to its sorted kept
    expert ids. For each module, axis zero of every packed tensor is sliced and
    ``num_experts`` and ``shapes`` are updated. Sibling residual tensors under
    the same block whose axis-zero length equals the source expert count are
    sliced as well. Returns transformed tensors and a transformed manifest
    without mutating either input.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert pruning requires torch.")
    import copy

    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("packed-state manifest must have a non-empty modules object.")
    new_manifest = copy.deepcopy(dict(manifest))
    new_tensors: dict[str, Any] = {str(k): v for k, v in tensors.items()}

    # Which tensor keys belong to a packed module (never treated as residual).
    packed_keys: set[str] = set()
    for spec in modules.values():
        if isinstance(spec, Mapping):
            for tname in (spec.get("tensors") or {}).values():
                packed_keys.add(str(tname))

    # Map each pruned block-parent prefix -> (old_num_experts, keep index tensor).
    residual_targets: list[tuple[str, int, Any, list[int]]] = []

    for module_name, keep in keep_by_module.items():
        spec = new_manifest["modules"].get(str(module_name))
        if not isinstance(spec, dict):
            raise ValueError(f"packed-state has no module {module_name!r} to prune.")
        if str(spec.get("kind")) != "grouped_experts":
            raise ValueError(
                f"prune_packed_state only prunes grouped_experts; {module_name!r} "
                f"is kind={spec.get('kind')!r}."
            )
        if "logical_to_physical" in spec or int(
            spec.get("logical_num_experts", spec.get("num_experts", 0))
        ) != int(spec.get("num_experts", 0)):
            raise ValueError(
                "Structural expert pruning requires an unconsolidated physical "
                "expert pool."
            )
        if spec.get("physical_weight_provider") is not None:
            raise ValueError(
                "Structural expert pruning does not rewrite physical-weight "
                "provider artifacts."
            )
        old_num = int(spec.get("num_experts"))
        keep_idx = sorted(int(i) for i in keep)
        if not keep_idx:
            raise ValueError("Structural expert pruning cannot remove every expert.")
        if len(keep_idx) != len(set(keep_idx)):
            raise ValueError("Structural expert pruning keep ids must be unique.")
        if keep_idx[0] < 0 or keep_idx[-1] >= old_num:
            raise ValueError(
                f"Structural expert pruning keep ids must be in [0, {old_num})."
            )
        index = torch.as_tensor(keep_idx, dtype=torch.long)
        tensor_names = spec.get("tensors") or {}
        for local_name, tname in tensor_names.items():
            tkey = str(tname)
            t = new_tensors[tkey]
            if str(local_name).endswith(
                ("_nf4_code", "_nf4_ncode", "_rotation")
            ):
                continue
            if t.ndim < 1 or int(t.shape[0]) != old_num:
                raise ValueError(
                    f"Packed expert tensor {local_name!r} has no declared "
                    f"expert axis of length {old_num}."
                )
            new_tensors[tkey] = t.index_select(0, index.to(t.device)).contiguous()
        spec["num_experts"] = len(keep_idx)
        shapes = spec.get("shapes")
        if isinstance(shapes, dict):
            for key, shape in list(shapes.items()):
                shape = list(shape)
                if shape:
                    shape[0] = len(keep_idx)
                shapes[key] = shape
        residual_targets.append((_parent(str(module_name)), old_num, index, keep_idx))

    # Auto-prune sibling residual tensors (router weight rows + correction bias).
    residual = new_manifest.get("residual_tensors") or {}
    for tensor_key in list(residual.values()):
        tkey = str(tensor_key)
        if tkey in packed_keys or tkey not in new_tensors:
            continue
        t = new_tensors[tkey]
        for prefix, old_num, index, _keep_idx in residual_targets:
            share_prefix = tkey.startswith(prefix + ".") if prefix else True
            if share_prefix and int(t.shape[0]) == old_num:
                new_tensors[tkey] = t.index_select(0, index.to(t.device)).contiguous()
                break

    return new_tensors, new_manifest
