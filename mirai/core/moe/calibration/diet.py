"""DIET deletion-response calibration and ODL expert selection.

This is a native PyTorch adaptation of the calibration algorithms released at
https://github.com/Eric-Zhang007/Video-DiT-MoE-Pruning (commit
e9c60419282f95b03e56a64b9b78bfd9f633366e, MIT): deletion-response signatures
(Eq. 2), cosine nearest-retained ODL (Eq. 3), and regression-guided layer
budgets.  It is an offline, explicit calibration seam; no runtime fallback
loads or trusts masks without declared source lineage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

DIET_EVIDENCE_FORMAT = "mirai.moe.diet_calibration"
DIET_EVIDENCE_SCHEMA_VERSION = 1
DIET_EVIDENCE_METADATA_KEY = "mirai_diet_calibration"


def _finite(tensor: Any, name: str, *, ndim: int | None = None) -> torch.Tensor:
    value = torch.as_tensor(tensor)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    return value


@dataclasses.dataclass(frozen=True)
class DietGroupTopology:
    num_experts: int
    top_k: int
    num_groups: int
    topk_group: int
    group_score_topk: int
    normalize_topk_prob: bool = True
    route_scale: float = 1.0
    expert_to_group: tuple[int, ...] = ()

    def validate(self) -> "DietGroupTopology":
        if self.num_experts < 2 or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("DIET topology has invalid expert count or top-k.")
        if not 1 <= self.topk_group <= self.num_groups:
            raise ValueError("DIET topology has invalid selected-group count.")
        if self.group_score_topk < 1:
            raise ValueError("DIET topology group_score_topk must be positive.")
        mapping = self.groups()
        counts = [mapping.count(group) for group in range(self.num_groups)]
        if min(counts) < self.group_score_topk:
            raise ValueError("Every DIET router group must contain enough experts.")
        if not math.isfinite(self.route_scale) or self.route_scale <= 0:
            raise ValueError("DIET route scale must be positive and finite.")
        return self

    def groups(self) -> tuple[int, ...]:
        if self.expert_to_group:
            result = tuple(int(group) for group in self.expert_to_group)
            if len(result) != self.num_experts or min(result) < 0 or max(result) >= self.num_groups:
                raise ValueError("DIET expert-to-group mapping is invalid.")
            return result
        if self.num_experts % self.num_groups:
            raise ValueError("Contiguous DIET groups require evenly divisible experts.")
        width = self.num_experts // self.num_groups
        return tuple(index // width for index in range(self.num_experts))


@dataclasses.dataclass(frozen=True)
class DietFamilyPostprocess:
    """Captured family state applied after routed expert aggregation."""

    shared_output: Any | None = None
    rms_weight: Any | None = None
    rms_epsilon: float = 1e-6
    residual_gate: Any | None = None

    def validate(self, rows: int, hidden: int) -> "DietFamilyPostprocess":
        for name, raw in (("shared_output", self.shared_output), ("residual_gate", self.residual_gate)):
            if raw is None:
                continue
            value = _finite(raw, name)
            if value.shape not in ((rows, hidden), (rows, 1), (hidden,), (1,)):
                raise ValueError(f"DIET {name} has an incompatible shape.")
        if self.rms_weight is not None and _finite(self.rms_weight, "rms_weight").shape not in ((hidden,), (1,)):
            raise ValueError("DIET RMS weight has an incompatible shape.")
        if not math.isfinite(self.rms_epsilon) or self.rms_epsilon <= 0:
            raise ValueError("DIET RMS epsilon must be positive and finite.")
        return self


@dataclasses.dataclass(frozen=True)
class DietLayerCapture:
    expert_outputs: Any
    router_scores: Any
    choice_bias: Any
    topology: DietGroupTopology
    postprocess: DietFamilyPostprocess | None = None

    def validate(self) -> "DietLayerCapture":
        outputs = _finite(self.expert_outputs, "expert_outputs", ndim=3)
        scores = _finite(self.router_scores, "router_scores", ndim=2)
        bias = _finite(self.choice_bias, "choice_bias", ndim=1)
        self.topology.validate()
        if outputs.shape[:2] != scores.shape or scores.shape[1] != self.topology.num_experts:
            raise ValueError("DIET expert outputs and router scores do not align.")
        if bias.shape != (self.topology.num_experts,):
            raise ValueError("DIET choice bias must have one value per expert.")
        if bool((scores < 0).any().item()):
            raise ValueError("DIET native sigmoid router scores must be non-negative.")
        if self.postprocess is not None:
            self.postprocess.validate(int(outputs.shape[0]), int(outputs.shape[2]))
        return self


@dataclasses.dataclass(frozen=True)
class DietPairedCapture:
    conditional: DietLayerCapture
    unconditional: DietLayerCapture
    pair_ids: Any
    token_positions: Any

    def validate(self) -> "DietPairedCapture":
        self.conditional.validate()
        self.unconditional.validate()
        if self.conditional.topology != self.unconditional.topology:
            raise ValueError("Paired DIET branches must share routing topology.")
        rows = int(torch.as_tensor(self.conditional.router_scores).shape[0])
        if torch.as_tensor(self.unconditional.router_scores).shape[0] != rows:
            raise ValueError("Paired DIET branches must contain aligned rows.")
        for name, raw in (("pair_ids", self.pair_ids), ("token_positions", self.token_positions)):
            value = torch.as_tensor(raw)
            if value.ndim != 1 or value.numel() != rows:
                raise ValueError(f"DIET {name} must align with paired rows.")
        return self


def _route(capture: DietLayerCapture, retained: torch.Tensor) -> torch.Tensor:
    capture.validate()
    scores = torch.as_tensor(capture.router_scores, dtype=torch.float64)
    outputs = torch.as_tensor(capture.expert_outputs, dtype=torch.float64)
    bias = torch.as_tensor(capture.choice_bias, dtype=torch.float64, device=scores.device)
    retained = torch.as_tensor(retained, dtype=torch.bool, device=scores.device)
    topology = capture.topology
    if retained.shape != (topology.num_experts,) or int(retained.sum().item()) < topology.top_k:
        raise ValueError("DIET retained mask has invalid shape or too few experts.")
    groups = torch.tensor(topology.groups(), device=scores.device)
    candidate = (scores + bias).masked_fill(~retained.unsqueeze(0), -torch.inf)
    group_scores = []
    for group in range(topology.num_groups):
        values = candidate[:, groups == group]
        group_scores.append(values.topk(topology.group_score_topk, dim=1).values.sum(dim=1))
    group_scores_t = torch.stack(group_scores, dim=1)
    selected_groups = group_scores_t.topk(topology.topk_group, dim=1).indices
    allowed_groups = torch.zeros_like(group_scores_t, dtype=torch.bool)
    allowed_groups.scatter_(1, selected_groups, True)
    allowed = allowed_groups.gather(1, groups.expand(scores.shape[0], -1)) & retained.unsqueeze(0)
    routed = candidate.masked_fill(~allowed, -torch.inf).topk(topology.top_k, dim=1).indices
    gates = scores.gather(1, routed)
    if topology.normalize_topk_prob:
        denominator = gates.sum(dim=1, keepdim=True)
        if bool((denominator <= 0).any().item()):
            raise ValueError("DIET replay encountered a zero unbiased gate sum.")
        gates = gates / denominator
    gates = gates * topology.route_scale
    gathered = outputs.gather(1, routed.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]))
    total = (gates.unsqueeze(-1) * gathered).sum(dim=1)
    state = capture.postprocess
    if state is not None:
        if state.shared_output is not None:
            total = total + torch.as_tensor(state.shared_output, dtype=total.dtype, device=total.device)
        if state.rms_weight is not None:
            total = total / torch.sqrt(total.square().mean(dim=-1, keepdim=True) + state.rms_epsilon)
            total = total * torch.as_tensor(state.rms_weight, dtype=total.dtype, device=total.device)
        if state.residual_gate is not None:
            total = total * torch.as_tensor(state.residual_gate, dtype=total.dtype, device=total.device)
    return total


def replay_deletion_responses(capture: DietLayerCapture) -> torch.Tensor:
    """Return output(delete e)-output(full) as ``[experts, rows, hidden]``."""

    capture.validate()
    topology = capture.topology
    full = torch.ones(topology.num_experts, dtype=torch.bool)
    baseline = _route(capture, full)
    responses = []
    for expert in range(topology.num_experts):
        retained = full.clone()
        retained[expert] = False
        responses.append(_route(capture, retained) - baseline)
    return torch.stack(responses, dim=0)


@dataclasses.dataclass(frozen=True)
class DietCalibrationEvidence:
    signatures: Any
    sample_count: int
    topology: DietGroupTopology
    cfg_mode: str
    guidance_scale: float
    token_pool: str = "mean"

    def validate(self) -> "DietCalibrationEvidence":
        signatures = _finite(self.signatures, "DIET signatures", ndim=2)
        self.topology.validate()
        if signatures.shape[0] != self.topology.num_experts or signatures.shape[1] < 1:
            raise ValueError("DIET signature shape does not match topology.")
        if self.sample_count < 1 or self.cfg_mode not in {"paired_cfg", "conditional"}:
            raise ValueError("DIET evidence has invalid sampling metadata.")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("DIET guidance scale must be finite and non-negative.")
        if self.token_pool not in {"mean", "flatten"}:
            raise ValueError("Unsupported DIET token pooling mode.")
        return self


def derive_diet_signatures(
    paired: DietPairedCapture,
    *,
    guidance_scale: float,
    token_pool: str = "mean",
) -> DietCalibrationEvidence:
    paired.validate()
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("DIET guidance scale must be finite and non-negative.")
    conditional = replay_deletion_responses(paired.conditional)
    unconditional = replay_deletion_responses(paired.unconditional)
    guided = guidance_scale * conditional + (1.0 - guidance_scale) * unconditional
    if token_pool == "mean":
        signatures = guided.mean(dim=1)
    elif token_pool == "flatten":
        signatures = guided.flatten(1)
    else:
        raise ValueError("Unsupported DIET token pooling mode.")
    return DietCalibrationEvidence(
        signatures=signatures.cpu(),
        sample_count=int(guided.shape[1]),
        topology=paired.conditional.topology,
        cfg_mode="paired_cfg",
        guidance_scale=float(guidance_scale),
        token_pool=token_pool,
    ).validate()


def odl_cost(signatures: Any, retained: Sequence[int]) -> float:
    values = _finite(signatures, "DIET signatures", ndim=2).to(torch.float64)
    kept = sorted({int(index) for index in retained})
    if not kept or kept[0] < 0 or kept[-1] >= values.shape[0]:
        raise ValueError("ODL retained indices are invalid.")
    norms = torch.linalg.vector_norm(values, dim=1).clamp_min(1e-12)
    cosine = ((values @ values.T) / (norms[:, None] * norms[None, :])).clamp(-1.0, 1.0)
    deleted = [index for index in range(values.shape[0]) if index not in set(kept)]
    if not deleted:
        return 0.0
    return float((1.0 - cosine[deleted][:, kept]).min(dim=1).values.sum().item())


def _group_feasible(kept: Sequence[int], topology: DietGroupTopology) -> bool:
    counts = [0] * topology.num_groups
    groups = topology.groups()
    for expert in kept:
        counts[groups[int(expert)]] += 1
    if any(count < topology.group_score_topk for count in counts):
        return False
    qualifying = sorted(counts)
    # Any tied set of router-selected groups must offer enough global top-k slots.
    return len(qualifying) >= topology.topk_group and sum(qualifying[: topology.topk_group]) >= topology.top_k


@dataclasses.dataclass(frozen=True)
class DietODLConfig:
    seed: int = 0
    random_restarts: int = 3
    swap_passes: int = 20
    anneal_steps: int = 1000
    initial_temperature: float = 1.0
    final_temperature: float = 0.001

    def validate(self) -> "DietODLConfig":
        if self.seed < 0 or min(self.random_restarts, self.swap_passes, self.anneal_steps) < 0:
            raise ValueError("DIET ODL iteration counts and seed must be non-negative.")
        if not all(math.isfinite(value) and value > 0 for value in (self.initial_temperature, self.final_temperature)):
            raise ValueError("DIET ODL temperatures must be positive and finite.")
        return self


@dataclasses.dataclass(frozen=True)
class DietBudgetProposal:
    layer_counts: tuple[int, ...]
    predicted_scores: tuple[float, ...]
    ridge_penalty: float


def propose_regression_layer_counts(
    evaluated_allocations: Any,
    evaluated_scores: Any,
    *,
    total_budget: int,
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    baseline_scores: Sequence[float],
    per_dimension_floors: Sequence[float],
    score_weights: Sequence[float] | None = None,
    ridge_penalty: float = 1.0,
) -> DietBudgetProposal:
    """Fit DIET's ridge surrogate and solve its constrained integer LP.

    This proposes an experimental allocation from caller-supplied evaluations;
    it does not promote or install the result.
    """

    allocations = _finite(evaluated_allocations, "evaluated allocations", ndim=2).to(torch.float64)
    scores = _finite(evaluated_scores, "evaluated scores", ndim=2).to(torch.float64)
    if allocations.shape[0] != scores.shape[0] or allocations.shape[0] < 2:
        raise ValueError("DIET budget regression requires aligned repeated evaluations.")
    layers, dimensions = allocations.shape[1], scores.shape[1]
    lower = torch.as_tensor(lower_bounds, dtype=torch.float64)
    upper = torch.as_tensor(upper_bounds, dtype=torch.float64)
    baseline = torch.as_tensor(baseline_scores, dtype=torch.float64)
    floors = torch.as_tensor(per_dimension_floors, dtype=torch.float64)
    weights = torch.ones(dimensions, dtype=torch.float64) if score_weights is None else torch.as_tensor(score_weights, dtype=torch.float64)
    if lower.shape != (layers,) or upper.shape != (layers,) or baseline.shape != (dimensions,) or floors.shape != (dimensions,) or weights.shape != (dimensions,):
        raise ValueError("DIET budget bounds, floors, and weights have inconsistent shapes.")
    if bool((lower > upper).any()) or bool((floors < 0).any()) or not math.isfinite(ridge_penalty) or ridge_penalty < 0:
        raise ValueError("DIET budget constraints or ridge penalty are invalid.")
    if not int(math.ceil(lower.sum().item())) <= total_budget <= int(math.floor(upper.sum().item())):
        raise ValueError("DIET total budget is outside the layer bounds.")
    # Match the released uncentered ridge design with an unpenalized intercept.
    centered_allocations = allocations - allocations.mean(dim=0)
    centered_scores = scores - scores.mean(dim=0)
    gram = centered_allocations.T @ centered_allocations + ridge_penalty * torch.eye(layers, dtype=torch.float64)
    coefficients = torch.linalg.solve(gram, centered_allocations.T @ centered_scores)
    intercept = scores.mean(dim=0) - allocations.mean(dim=0) @ coefficients
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ModuleNotFoundError as exc:  # pragma: no cover - optional execution seam
        raise RuntimeError("DIET regression-guided budgets require scipy.optimize.milp.") from exc
    objective = -(coefficients @ weights).numpy()
    constraints = [LinearConstraint(torch.ones((1, layers)).numpy(), [total_budget], [total_budget])]
    # score = x @ coefficients + intercept >= baseline - floor
    constraints.append(LinearConstraint(coefficients.T.numpy(), (baseline - floors - intercept).numpy(), [math.inf] * dimensions))
    result = milp(c=objective, integrality=torch.ones(layers).numpy(), bounds=Bounds(lower.numpy(), upper.numpy()), constraints=constraints)
    if not result.success or result.x is None:
        raise ValueError(f"DIET regression-guided budget is infeasible: {result.message}")
    counts = tuple(int(round(value)) for value in result.x)
    predicted = torch.as_tensor(counts, dtype=torch.float64) @ coefficients + intercept
    if sum(counts) != total_budget or bool((predicted < baseline - floors - 1e-7).any()):
        raise RuntimeError("DIET integer budget solver returned an invalid incumbent.")
    return DietBudgetProposal(counts, tuple(float(value) for value in predicted), float(ridge_penalty))


def select_odl_survivors(
    signatures: Any,
    keep_count: int,
    topology: DietGroupTopology,
    *,
    config: DietODLConfig | None = None,
) -> tuple[int, ...]:
    """Minimize cosine nearest-retained ODL with deterministic multi-start search."""

    config = (DietODLConfig() if config is None else config).validate()
    values = _finite(signatures, "DIET signatures", ndim=2)
    topology.validate()
    if values.shape[0] != topology.num_experts or not topology.top_k <= keep_count <= topology.num_experts:
        raise ValueError("ODL keep count is incompatible with routing topology.")
    rng = random.Random(config.seed)

    normalized = torch.nn.functional.normalize(values.to(torch.float32), dim=1, eps=1e-12)
    distances = (1.0 - (normalized @ normalized.T).clamp(-1.0, 1.0)).cpu()

    def cost(kept: Sequence[int]) -> float:
        kept_set = set(kept)
        deleted = [expert for expert in range(topology.num_experts) if expert not in kept_set]
        if not deleted:
            return 0.0
        return float(distances[deleted][:, list(kept)].min(dim=1).values.sum().item())

    groups = topology.groups()

    def feasible_seed(randomized: bool) -> list[int]:
        per_group = max(
            topology.group_score_topk,
            math.ceil(topology.top_k / topology.topk_group),
        )
        if per_group * topology.num_groups > keep_count:
            raise ValueError("No grouped-feasible ODL selection exists for this keep count.")
        seed: list[int] = []
        for group in range(topology.num_groups):
            members = [expert for expert, value in enumerate(groups) if value == group]
            members.sort(key=lambda expert: (float(distances[:, expert].sum()), expert))
            if randomized:
                window = members[: min(len(members), max(per_group * 2, 8))]
                rng.shuffle(window)
                members = window + members[len(window) :]
            seed.extend(members[:per_group])
        return sorted(seed)

    def greedy(randomized: bool) -> list[int]:
        kept = feasible_seed(randomized)
        while len(kept) < keep_count:
            candidates = []
            for expert in range(topology.num_experts):
                if expert in kept:
                    continue
                trial = kept + [expert]
                if _group_feasible(trial, topology):
                    candidates.append((cost(trial), (expert,)))
            if not candidates:
                raise ValueError("No grouped-feasible ODL selection exists for this keep count.")
            candidates.sort()
            width = min(8, len(candidates)) if randomized else 1
            kept.extend(candidates[rng.randrange(width)][1])
        if not _group_feasible(kept, topology):  # defensive: adding experts must preserve feasibility
            raise RuntimeError("DIET grouped seed construction produced an infeasible selection.")
        return sorted(kept)

    def improve(kept: list[int]) -> list[int]:
        current = cost(kept)
        for _ in range(config.swap_passes):
            best = (current, kept)
            for outgoing in kept:
                for incoming in range(topology.num_experts):
                    if incoming in kept:
                        continue
                    trial = sorted([x for x in kept if x != outgoing] + [incoming])
                    if _group_feasible(trial, topology):
                        best = min(best, (cost(trial), trial))
            if best[0] >= current - 1e-12:
                break
            current, kept = best
        best_kept, best_cost = kept, current
        for step in range(config.anneal_steps):
            deleted = [x for x in range(topology.num_experts) if x not in kept]
            if not deleted:
                break
            outgoing = rng.choice(kept)
            trial = sorted([x for x in kept if x != outgoing] + [rng.choice(deleted)])
            if not _group_feasible(trial, topology):
                continue
            trial_cost = cost(trial)
            fraction = step / max(config.anneal_steps, 1)
            temperature = config.initial_temperature * (config.final_temperature / config.initial_temperature) ** fraction
            if trial_cost <= current or rng.random() < math.exp((current - trial_cost) / max(temperature, 1e-12)):
                kept, current = trial, trial_cost
                if (current, kept) < (best_cost, best_kept):
                    best_kept, best_cost = kept, current
        return improve_without_anneal(best_kept)

    def improve_without_anneal(kept: list[int]) -> list[int]:
        current = cost(kept)
        for _ in range(config.swap_passes):
            candidates = [(current, kept)]
            for outgoing in kept:
                for incoming in range(topology.num_experts):
                    if incoming not in kept:
                        trial = sorted([x for x in kept if x != outgoing] + [incoming])
                        if _group_feasible(trial, topology):
                            candidates.append((cost(trial), trial))
            trial_cost, trial = min(candidates)
            if trial_cost >= current - 1e-12:
                break
            current, kept = trial_cost, trial
        return kept

    starts = [greedy(False)] + [greedy(True) for _ in range(config.random_restarts)]
    return tuple(min((improve(start) for start in starts), key=lambda kept: (cost(kept), kept)))


def uniform_layer_counts(module_experts: Mapping[str, int], keep_fraction: float) -> dict[str, int]:
    if not 0 < keep_fraction <= 1 or not module_experts:
        raise ValueError("DIET uniform budget requires modules and a fraction in (0, 1].")
    return {str(name): max(1, min(int(count), round(int(count) * keep_fraction))) for name, count in module_experts.items()}


def validate_layer_counts(counts: Mapping[str, int], topologies: Mapping[str, DietGroupTopology]) -> dict[str, int]:
    if set(counts) != set(topologies):
        raise ValueError("DIET layer counts must map every and only calibrated module.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts.values()):
        raise ValueError("DIET layer counts must be exact integers.")
    result = {str(name): int(value) for name, value in counts.items()}
    for name, count in result.items():
        topology = topologies[name].validate()
        if not topology.top_k <= count <= topology.num_experts:
            raise ValueError(f"DIET layer count for {name!r} is infeasible.")
    return result


@dataclasses.dataclass(frozen=True)
class DietLineage:
    dataset_snapshot_id: str
    model_snapshot_id: str
    config_snapshot_id: str
    calibration_manifest_id: str
    packed_state_sha256: str
    capture_fingerprint: str

    def validate(self) -> "DietLineage":
        if not all(str(value).strip() for value in dataclasses.astuple(self)):
            raise ValueError("DIET evidence requires complete snapshot and capture lineage.")
        if len(self.packed_state_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in self.packed_state_sha256
        ):
            raise ValueError("DIET packed-state lineage must be a SHA-256 fingerprint.")
        return self

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return dataclasses.asdict(self)


def _topology_dict(value: DietGroupTopology) -> dict[str, Any]:
    return dataclasses.asdict(value.validate())


def save_diet_evidence(path: str | Path, evidence_by_module: Mapping[str, DietCalibrationEvidence], *, lineage: DietLineage) -> None:
    from safetensors.torch import save_file

    lineage.validate()
    if not evidence_by_module or any(not str(name) for name in evidence_by_module):
        raise ValueError("DIET evidence mapping must have non-empty module names.")
    tensors: dict[str, torch.Tensor] = {}
    modules = []
    for index, (name, evidence) in enumerate(sorted(evidence_by_module.items())):
        evidence.validate()
        key = f"module_{index:04d}.signatures"
        tensors[key] = torch.as_tensor(evidence.signatures).cpu().contiguous()
        modules.append({"name": name, "tensor": key, "sample_count": evidence.sample_count,
                        "topology": _topology_dict(evidence.topology), "cfg_mode": evidence.cfg_mode,
                        "guidance_scale": evidence.guidance_scale, "token_pool": evidence.token_pool})
    manifest = {"format": DIET_EVIDENCE_FORMAT, "schema_version": DIET_EVIDENCE_SCHEMA_VERSION,
                "lineage": dataclasses.asdict(lineage), "modules": modules}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={DIET_EVIDENCE_METADATA_KEY: json.dumps(manifest, sort_keys=True, separators=(",", ":"))})


def load_diet_evidence(path: str | Path, *, expected_lineage: DietLineage | None = None) -> tuple[dict[str, DietCalibrationEvidence], DietLineage]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw = (handle.metadata() or {}).get(DIET_EVIDENCE_METADATA_KEY)
    if raw is None:
        raise ValueError("DIET artifact is missing its Mirai manifest.")
    manifest = json.loads(raw)
    if manifest.get("format") != DIET_EVIDENCE_FORMAT or manifest.get("schema_version") != DIET_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported DIET evidence format or schema version.")
    try:
        lineage = DietLineage(**manifest["lineage"]).validate()
    except (KeyError, TypeError) as exc:
        raise ValueError("DIET artifact has invalid lineage.") from exc
    if expected_lineage is not None and lineage != expected_lineage.validate():
        raise ValueError("DIET artifact lineage does not match the requested source.")
    tensors = load_file(str(path), device="cpu")
    loaded: dict[str, DietCalibrationEvidence] = {}
    for spec in manifest.get("modules", []):
        if not isinstance(spec, dict) or not str(spec.get("name", "")) or spec.get("tensor") not in tensors:
            raise ValueError("DIET artifact has an invalid module mapping.")
        if spec["name"] in loaded:
            raise ValueError("DIET artifact contains duplicate module names.")
        try:
            topology_data = dict(spec["topology"])
            topology_data["expert_to_group"] = tuple(
                topology_data.get("expert_to_group", ())
            )
            topology = DietGroupTopology(**topology_data)
            evidence = DietCalibrationEvidence(tensors[spec["tensor"]], int(spec["sample_count"]), topology,
                                               str(spec["cfg_mode"]), float(spec["guidance_scale"]), str(spec["token_pool"])).validate()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("DIET artifact has invalid module evidence.") from exc
        loaded[spec["name"]] = evidence
    if not loaded:
        raise ValueError("DIET artifact contains no module evidence.")
    return loaded, lineage


def validate_published_diet_mask(mask: Mapping[str, Sequence[int]], *, declared_source_lineage: Mapping[str, str]) -> dict[str, tuple[int, ...]]:
    required = {"publisher", "source_repository", "source_revision", "artifact_sha256"}
    if set(declared_source_lineage) != required or not all(str(declared_source_lineage[key]).strip() for key in required):
        raise ValueError("Published DIET masks require complete, explicitly declared source lineage.")
    if len(declared_source_lineage["artifact_sha256"]) != hashlib.sha256().digest_size * 2:
        raise ValueError("Published DIET mask artifact fingerprint must be SHA-256.")
    if not mask:
        raise ValueError("Published DIET mask cannot be empty.")
    result: dict[str, tuple[int, ...]] = {}
    for name, indices in mask.items():
        raw = tuple(int(index) for index in indices)
        if len(set(raw)) != len(raw):
            raise ValueError("Published DIET masks cannot contain duplicate expert ids.")
        result[str(name)] = tuple(sorted(raw))
    if any(not name or not indices or indices[0] < 0 for name, indices in result.items()):
        raise ValueError("Published DIET mask mappings are invalid.")
    return result


__all__ = [
    "DietCalibrationEvidence", "DietFamilyPostprocess", "DietGroupTopology", "DietLayerCapture",
    "DietBudgetProposal", "DietLineage", "DietODLConfig", "DietPairedCapture", "derive_diet_signatures", "load_diet_evidence",
    "odl_cost", "replay_deletion_responses", "save_diet_evidence", "select_odl_survivors",
    "propose_regression_layer_counts", "uniform_layer_counts", "validate_layer_counts", "validate_published_diet_mask",
]
