"""Model-agnostic primitives for FlexMoE channel-action learning.

The source method ranks each routed FFN channel from the squared first-order
Taylor terms of the attached gate, up, and down parameters, then learns one
discrete prefix-retention action per expert with straight-through
Gumbel-Softmax and a load-sensitive cost.

Source: Mo et al., "FlexMoE: One-for-All Nested Intra-Expert Pruning for MoE
Language Models", Equations 1-12, arXiv:2606.27866.
https://arxiv.org/abs/2606.27866
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


_PROJECTIONS = ("w1", "w2", "w3")
FLEXMOE_RANKING_FORMAT = "mirai.moe.flexmoe_ranking"
FLEXMOE_RANKING_SCHEMA_VERSION = 1
FLEXMOE_RANKING_METADATA_KEY = "mirai_flexmoe_ranking"
FLEXMOE_ACTION_PLAN_FORMAT = "mirai.moe.flexmoe_action_plan"
FLEXMOE_ACTION_PLAN_SCHEMA_VERSION = 1
FLEXMOE_ACTION_PLAN_METADATA_KEY = "mirai_flexmoe_action_plan"


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("FlexMoE calibration requires torch.")


def _validated_triplet(
    values: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    _require_torch()
    if set(values) != set(_PROJECTIONS):
        raise ValueError(f"FlexMoE {label} must contain exactly w1, w2, and w3.")
    tensors = {name: torch.as_tensor(values[name]) for name in _PROJECTIONS}
    w1_shape = tuple(int(value) for value in tensors["w1"].shape)
    w2_shape = tuple(int(value) for value in tensors["w2"].shape)
    w3_shape = tuple(int(value) for value in tensors["w3"].shape)
    if len(w1_shape) != 3 or w3_shape != w1_shape:
        raise ValueError(f"FlexMoE {label} w1/w3 must share [experts, intermediate, hidden].")
    expected_w2 = (w1_shape[0], w1_shape[2], w1_shape[1])
    if w2_shape != expected_w2:
        raise ValueError(f"FlexMoE {label} w2 must have shape {expected_w2}, got {w2_shape}.")
    if w1_shape[0] < 1 or w1_shape[1] < 1 or w1_shape[2] < 1:
        raise ValueError(f"FlexMoE {label} tensors must be non-empty.")
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"FlexMoE {label} {name} contains non-finite values.")
    return tensors


def channel_taylor_saliency(
    weights: Mapping[str, Any],
    gradients: Mapping[str, Any],
) -> Any:
    """Return Equation-2 channel saliency for one calibration batch.

    Shapes follow Mirai's grouped expert convention: ``w1`` and ``w3`` are
    ``[experts, intermediate, hidden]`` and ``w2`` is
    ``[experts, hidden, intermediate]``.  Each output entry is the sum of
    ``(theta * dL/dtheta)^2`` over all parameters attached to that channel.
    """

    dense = _validated_triplet(weights, label="weights")
    grads = _validated_triplet(gradients, label="gradients")
    for name in _PROJECTIONS:
        if tuple(dense[name].shape) != tuple(grads[name].shape):
            raise ValueError(f"FlexMoE gradient shape for {name} does not match its weight.")
    device = dense["w1"].device
    if any(value.device != device for value in (*dense.values(), *grads.values())):
        raise ValueError("FlexMoE weights and gradients must share one device.")
    w1 = (dense["w1"].float() * grads["w1"].float()).square().sum(dim=2)
    w3 = (dense["w3"].float() * grads["w3"].float()).square().sum(dim=2)
    w2 = (dense["w2"].float() * grads["w2"].float()).square().sum(dim=1)
    return w1 + w2 + w3


@dataclasses.dataclass
class FlexMoEChannelSaliencyAccumulator:
    """Equation-2 arithmetic mean over an explicit calibration batch set."""

    total: Any | None = None
    batches: int = 0

    def update(
        self,
        weights: Mapping[str, Any],
        gradients: Mapping[str, Any],
    ) -> None:
        batch = (
            channel_taylor_saliency(weights, gradients)
            .detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )
        if self.total is None:
            self.total = batch.clone()
        else:
            if tuple(self.total.shape) != tuple(batch.shape):
                raise ValueError("FlexMoE calibration topology changed between batches.")
            self.total.add_(batch)
        self.batches += 1

    def mean(self) -> Any:
        if self.total is None or self.batches < 1:
            raise RuntimeError("FlexMoE saliency requires at least one calibration batch.")
        return self.total.div(float(self.batches))

    def evidence(self) -> FlexMoERankingEvidence:
        return FlexMoERankingEvidence(
            saliency=self.mean().clone(),
            calibration_batches=self.batches,
        ).validate()


@dataclasses.dataclass(frozen=True)
class FlexMoERankingEvidence:
    """Equation-2 sufficient evidence for one grouped expert module."""

    saliency: Any
    calibration_batches: int

    @property
    def num_experts(self) -> int:
        return int(torch.as_tensor(self.saliency).shape[0])

    @property
    def intermediate_size(self) -> int:
        return int(torch.as_tensor(self.saliency).shape[1])

    def validate(self) -> FlexMoERankingEvidence:
        _require_torch()
        scores = torch.as_tensor(self.saliency)
        if scores.ndim != 2 or min(int(value) for value in scores.shape) < 1:
            raise ValueError("FlexMoE ranking evidence must have shape [experts, intermediate].")
        if not scores.is_floating_point() or not bool(torch.isfinite(scores).all().item()):
            raise ValueError("FlexMoE ranking evidence must be finite floating point.")
        if bool((scores < 0.0).any().item()):
            raise ValueError("FlexMoE ranking evidence cannot be negative.")
        if int(self.calibration_batches) < 1:
            raise ValueError("FlexMoE ranking evidence requires a positive batch count.")
        return self

    def permutation(self) -> Any:
        self.validate()
        return rank_channels(self.saliency)


@dataclasses.dataclass(frozen=True)
class FlexMoEActionPlan:
    """Final Equation-11 per-expert action evidence for one module."""

    action_ratios: tuple[float, ...]
    logits: Any
    expert_load: Any

    @property
    def num_experts(self) -> int:
        return int(torch.as_tensor(self.logits).shape[0])

    def validate(self) -> FlexMoEActionPlan:
        _require_torch()
        ratios = normalize_action_ratios(self.action_ratios)
        logits = torch.as_tensor(self.logits)
        load = torch.as_tensor(self.expert_load)
        if logits.ndim != 2 or tuple(logits.shape) != (
            int(load.numel()),
            len(ratios),
        ):
            raise ValueError("FlexMoE action plan logits must match experts and action ratios.")
        if not logits.is_floating_point() or not bool(torch.isfinite(logits).all().item()):
            raise ValueError("FlexMoE action-plan logits must be finite floating point.")
        if (
            load.ndim != 1
            or not load.is_floating_point()
            or not bool(torch.isfinite(load).all().item())
        ):
            raise ValueError("FlexMoE action-plan load must be a finite vector.")
        if bool((load < 0.0).any().item()) or not torch.isclose(
            load.float().sum(),
            torch.tensor(1.0, device=load.device),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                "FlexMoE action-plan load must be an assignment-frequency distribution."
            )
        return self

    def retention_ratios(self) -> Any:
        self.validate()
        return hardened_retention_ratios(self.logits, self.action_ratios)

    def prune_budget(self) -> float:
        return float(global_prune_budget(self.retention_ratios()).item())


@dataclasses.dataclass(frozen=True)
class FlexMoECalibrationTarget:
    """Provider-owned grouped expert host exposed to generic action learning."""

    name: str
    host: Any
    num_experts: int
    intermediate_size: int

    def validate(self) -> FlexMoECalibrationTarget:
        name = str(self.name).strip()
        if not name:
            raise ValueError("FlexMoE calibration target name must be non-empty.")
        if int(self.num_experts) < 1 or int(self.intermediate_size) < 1:
            raise ValueError("FlexMoE calibration target topology must be positive.")
        setter = getattr(self.host, "set_flexmoe_channel_mask", None)
        if not callable(setter):
            raise TypeError(
                "FlexMoE calibration hosts must expose set_flexmoe_channel_mask()."
            )
        shape_getter = getattr(self.host, "expert_weight_shape", None)
        if not callable(shape_getter):
            raise TypeError("FlexMoE calibration hosts must expose expert_weight_shape().")
        if not callable(getattr(self.host, "set_routed_output_observer", None)):
            raise TypeError(
                "FlexMoE calibration hosts must expose set_routed_output_observer()."
            )
        if not callable(getattr(self.host, "get_routed_output_observer", None)):
            raise TypeError(
                "FlexMoE calibration hosts must expose get_routed_output_observer()."
            )
        if not callable(getattr(self.host, "set_flexmoe_taylor_observer", None)):
            raise TypeError(
                "FlexMoE calibration hosts must expose set_flexmoe_taylor_observer()."
            )
        shape = tuple(int(value) for value in shape_getter("w1"))
        if shape[:2] != (int(self.num_experts), int(self.intermediate_size)):
            raise ValueError("FlexMoE calibration target topology disagrees with host.")
        if bool(getattr(self.host, "has_ragged_intermediate_widths", lambda: False)()):
            raise ValueError("FlexMoE action learning requires an unpruned source host.")
        return self

    def bind_mask(self, mask: Any | None) -> None:
        self.validate()
        self.host.set_flexmoe_channel_mask(mask)

    def bind_load_observer(self, observer: Any | None) -> None:
        self.validate()
        self.host.set_routed_output_observer(observer)

    def bind_taylor_observer(self, observer: Any | None) -> None:
        self.validate()
        self.host.set_flexmoe_taylor_observer(observer)


class FlexMoEExpertLoadObserver:
    """Capture Equation-10 assignment frequencies without retaining outputs."""

    capture_in_eval = True

    def __init__(self, num_experts: int) -> None:
        _require_torch()
        experts = int(num_experts)
        if experts < 1:
            raise ValueError("FlexMoE load observer requires experts.")
        self.num_experts = experts
        self._load: Any | None = None
        self._active = False

    @property
    def is_enabled(self) -> bool:
        return True

    def bind_routes(self, expert_indices: Any, routing_weights: Any) -> None:
        indices = torch.as_tensor(expert_indices).detach()
        weights = torch.as_tensor(routing_weights).detach()
        if indices.ndim != 2 or indices.shape != weights.shape:
            raise ValueError("FlexMoE load routes must be aligned [tokens, top_k].")
        active = weights.reshape(-1) != 0
        selected = indices.reshape(-1)[active].to(device="cpu", dtype=torch.long)
        if selected.numel() < 1:
            raise ValueError("FlexMoE load observation contains no active routes.")
        if int(selected.min().item()) < 0 or int(selected.max().item()) >= self.num_experts:
            raise ValueError("FlexMoE load observation contains an invalid expert id.")
        counts = torch.bincount(selected, minlength=self.num_experts).to(torch.float64)
        self._load = counts / counts.sum()

    def begin_routes(self, *, num_tokens: int, top_k: int, device: Any) -> None:
        del device
        if self._active:
            raise RuntimeError("FlexMoE load capture is already active.")
        if int(num_tokens) < 1 or int(top_k) < 1:
            raise ValueError("FlexMoE load capture requires non-empty routes.")
        self._active = True

    def capture_routes(self, expert_outputs: Any, route_positions: Any) -> None:
        if not self._active:
            raise RuntimeError("FlexMoE load capture was not started.")
        if int(torch.as_tensor(route_positions).numel()) != int(
            torch.as_tensor(expert_outputs).shape[0]
        ):
            raise ValueError("FlexMoE routed outputs and positions do not align.")

    def end_routes(self) -> None:
        if not self._active:
            raise RuntimeError("FlexMoE load capture was not started.")
        self._active = False

    def abort_capture(self) -> None:
        self._active = False

    def take_load(self) -> Any:
        if self._load is None:
            raise RuntimeError("FlexMoE forward produced no load observation.")
        value = self._load
        self._load = None
        return value


class FlexMoETaylorGradientObserver:
    """Reconstruct exact grouped parameter gradients from one routed backward."""

    def __init__(
        self,
        *,
        num_experts: int,
        intermediate_size: int,
    ) -> None:
        _require_torch()
        experts = int(num_experts)
        intermediate = int(intermediate_size)
        if experts < 1 or intermediate < 1:
            raise ValueError("FlexMoE Taylor observer topology must be positive.")
        self.num_experts = experts
        self.intermediate_size = intermediate
        self.accumulator = FlexMoEChannelSaliencyAccumulator()
        self._batch_scores: Any | None = None
        self._captures = 0
        self._completed = 0

    @property
    def is_enabled(self) -> bool:
        return True

    def begin_batch(self, *, device: Any) -> None:
        if self._batch_scores is not None:
            raise RuntimeError("FlexMoE Taylor batch is already active.")
        self._batch_scores = torch.zeros(
            self.num_experts,
            self.intermediate_size,
            device=device,
            dtype=torch.float32,
        )
        self._captures = 0
        self._completed = 0

    def capture(
        self,
        *,
        expert_index: int,
        inputs: Any,
        w1: Any,
        w2: Any,
        w3: Any,
        gate: Any,
        up: Any,
        hidden: Any,
        output: Any,
    ) -> None:
        scores = self._batch_scores
        if scores is None:
            raise RuntimeError("FlexMoE Taylor capture requires begin_batch().")
        expert = int(expert_index)
        if expert < 0 or expert >= self.num_experts:
            raise IndexError("FlexMoE Taylor capture expert is out of range.")
        tensors = {
            "inputs": torch.as_tensor(inputs),
            "w1": torch.as_tensor(w1),
            "w2": torch.as_tensor(w2),
            "w3": torch.as_tensor(w3),
            "gate": torch.as_tensor(gate),
            "up": torch.as_tensor(up),
            "hidden": torch.as_tensor(hidden),
            "output": torch.as_tensor(output),
        }
        if not tensors["output"].requires_grad:
            raise ValueError("FlexMoE Taylor output must participate in autograd.")
        expected = {
            "w1": (self.intermediate_size, int(tensors["inputs"].shape[-1])),
            "w2": (int(tensors["inputs"].shape[-1]), self.intermediate_size),
            "w3": (self.intermediate_size, int(tensors["inputs"].shape[-1])),
        }
        for key, shape in expected.items():
            if tuple(tensors[key].shape) != shape:
                raise ValueError(f"FlexMoE Taylor {key} shape mismatch.")
        if tensors["gate"].shape != tensors["up"].shape or (
            tensors["hidden"].shape != tensors["gate"].shape
        ):
            raise ValueError("FlexMoE Taylor intermediate tensors do not align.")
        self._captures += 1

        x = tensors["inputs"].detach()
        dense_w1 = tensors["w1"].detach()
        dense_w2 = tensors["w2"].detach()
        dense_w3 = tensors["w3"].detach()
        gate_value = tensors["gate"].detach()
        up_value = tensors["up"].detach()
        hidden_value = tensors["hidden"].detach()

        def record(grad_output: Any) -> Any:
            grad_y = torch.as_tensor(grad_output).float()
            compute_x = x.float()
            compute_w1 = dense_w1.float()
            compute_w2 = dense_w2.float()
            compute_w3 = dense_w3.float()
            compute_gate = gate_value.float()
            compute_up = up_value.float()
            compute_hidden = hidden_value.float()
            grad_w2 = grad_y.transpose(0, 1).matmul(compute_hidden)
            grad_hidden = grad_y.matmul(compute_w2)
            sigmoid = torch.sigmoid(compute_gate)
            silu_derivative = sigmoid * (1.0 + compute_gate * (1.0 - sigmoid))
            grad_gate = grad_hidden * compute_up * silu_derivative
            grad_up = grad_hidden * F.silu(compute_gate)
            grad_w1 = grad_gate.transpose(0, 1).matmul(compute_x)
            grad_w3 = grad_up.transpose(0, 1).matmul(compute_x)
            contribution = (
                (compute_w1 * grad_w1).square().sum(dim=1)
                + (compute_w2 * grad_w2).square().sum(dim=0)
                + (compute_w3 * grad_w3).square().sum(dim=1)
            )
            scores[expert].add_(contribution)
            self._completed += 1
            return grad_output

        tensors["output"].register_hook(record)

    def finish_batch(self) -> None:
        scores = self._batch_scores
        if scores is None:
            raise RuntimeError("FlexMoE Taylor batch was not started.")
        try:
            if self._captures < 1 or self._completed != self._captures:
                raise RuntimeError(
                    "FlexMoE Taylor backward did not visit every captured expert."
                )
            batch = scores.detach().to(device="cpu", dtype=torch.float64)
            if self.accumulator.total is None:
                self.accumulator.total = batch.clone()
            else:
                self.accumulator.total.add_(batch)
            self.accumulator.batches += 1
        finally:
            self.abort_batch()

    def abort_batch(self) -> None:
        self._batch_scores = None
        self._captures = 0
        self._completed = 0

    def evidence(self) -> FlexMoERankingEvidence:
        return self.accumulator.evidence()


def _lineage(
    *,
    dataset_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str,
) -> dict[str, str]:
    lineage = {
        "dataset_snapshot_id": str(dataset_snapshot_id).strip(),
        "model_snapshot_id": str(model_snapshot_id).strip(),
        "config_snapshot_id": str(config_snapshot_id).strip(),
    }
    if not all(lineage.values()):
        raise ValueError("FlexMoE evidence requires complete snapshot lineage.")
    return lineage


def save_ranking_evidence(
    path: str | Path,
    evidence_by_module: Mapping[str, FlexMoERankingEvidence],
    *,
    dataset_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str,
) -> None:
    """Persist immutable Equation-2 evidence with exact snapshot lineage."""

    _require_torch()
    from safetensors.torch import save_file

    if not evidence_by_module:
        raise ValueError("FlexMoE ranking evidence cannot be empty.")
    tensors: dict[str, Any] = {}
    modules: list[dict[str, Any]] = []
    for index, (name, evidence) in enumerate(evidence_by_module.items()):
        module_name = str(name).strip()
        if not module_name:
            raise ValueError("FlexMoE ranking module names must be non-empty.")
        evidence.validate()
        tensor_name = f"module_{index:04d}.saliency"
        tensors[tensor_name] = (
            torch.as_tensor(evidence.saliency)
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        )
        modules.append(
            {
                "name": module_name,
                "saliency": tensor_name,
                "num_experts": evidence.num_experts,
                "intermediate_size": evidence.intermediate_size,
                "calibration_batches": int(evidence.calibration_batches),
            }
        )
    if len({module["name"] for module in modules}) != len(modules):
        raise ValueError("FlexMoE ranking module names must be unique.")
    manifest = {
        "format": FLEXMOE_RANKING_FORMAT,
        "schema_version": FLEXMOE_RANKING_SCHEMA_VERSION,
        **_lineage(
            dataset_snapshot_id=dataset_snapshot_id,
            model_snapshot_id=model_snapshot_id,
            config_snapshot_id=config_snapshot_id,
        ),
        "modules": modules,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            FLEXMOE_RANKING_METADATA_KEY: json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )


def load_ranking_evidence(
    path: str | Path,
) -> tuple[dict[str, FlexMoERankingEvidence], dict[str, str]]:
    """Load and fully validate a ranking-evidence artifact."""

    _require_torch()
    from safetensors import safe_open
    from safetensors.torch import load_file

    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        raw_manifest = (handle.metadata() or {}).get(FLEXMOE_RANKING_METADATA_KEY)
    if raw_manifest is None:
        raise ValueError("FlexMoE ranking artifact is missing its Mirai manifest.")
    manifest = json.loads(raw_manifest)
    if manifest.get("format") != FLEXMOE_RANKING_FORMAT:
        raise ValueError(f"Unsupported FlexMoE ranking format {manifest.get('format')!r}.")
    if int(manifest.get("schema_version", 0)) != FLEXMOE_RANKING_SCHEMA_VERSION:
        raise ValueError(f"Unsupported FlexMoE ranking schema {manifest.get('schema_version')!r}.")
    lineage = {
        key: str(manifest.get(key, ""))
        for key in ("dataset_snapshot_id", "model_snapshot_id", "config_snapshot_id")
    }
    if not all(lineage.values()):
        raise ValueError("FlexMoE ranking artifact has incomplete lineage.")
    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("FlexMoE ranking artifact has no modules.")
    tensors = load_file(str(source), device="cpu")
    loaded: dict[str, FlexMoERankingEvidence] = {}
    for raw_spec in modules:
        if not isinstance(raw_spec, Mapping):
            raise ValueError("FlexMoE ranking module entries must be objects.")
        name = str(raw_spec.get("name", ""))
        if not name or name in loaded:
            raise ValueError("FlexMoE ranking module names must be unique and non-empty.")
        tensor_name = str(raw_spec.get("saliency", ""))
        if tensor_name not in tensors:
            raise ValueError(f"FlexMoE ranking tensor {tensor_name!r} for {name!r} is missing.")
        evidence = FlexMoERankingEvidence(
            saliency=tensors[tensor_name],
            calibration_batches=int(raw_spec.get("calibration_batches", 0)),
        ).validate()
        if evidence.num_experts != int(raw_spec.get("num_experts", 0)) or (
            evidence.intermediate_size != int(raw_spec.get("intermediate_size", 0))
        ):
            raise ValueError(f"FlexMoE ranking topology mismatch for module {name!r}.")
        loaded[name] = evidence
    return loaded, lineage


def save_action_plans(
    path: str | Path,
    plans_by_module: Mapping[str, FlexMoEActionPlan],
    *,
    dataset_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str,
    ranking_snapshot_id: str,
) -> None:
    """Persist final Equation-11 actions separately from ranking evidence."""

    _require_torch()
    from safetensors.torch import save_file

    ranking_id = str(ranking_snapshot_id).strip()
    if not ranking_id:
        raise ValueError("FlexMoE action plans require a ranking snapshot id.")
    if not plans_by_module:
        raise ValueError("FlexMoE action-plan artifact cannot be empty.")
    tensors: dict[str, Any] = {}
    modules: list[dict[str, Any]] = []
    retained: list[Any] = []
    for index, (name, plan) in enumerate(plans_by_module.items()):
        module_name = str(name).strip()
        if not module_name:
            raise ValueError("FlexMoE action-plan module names must be non-empty.")
        plan.validate()
        retained.append(plan.retention_ratios())
        logits_name = f"module_{index:04d}.logits"
        load_name = f"module_{index:04d}.expert_load"
        tensors[logits_name] = (
            torch.as_tensor(plan.logits).detach().to(device="cpu", dtype=torch.float32).contiguous()
        )
        tensors[load_name] = (
            torch.as_tensor(plan.expert_load)
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        )
        modules.append(
            {
                "name": module_name,
                "logits": logits_name,
                "expert_load": load_name,
                "num_experts": plan.num_experts,
                "action_ratios": list(normalize_action_ratios(plan.action_ratios)),
                "prune_budget": plan.prune_budget(),
            }
        )
    if len({module["name"] for module in modules}) != len(modules):
        raise ValueError("FlexMoE action-plan module names must be unique.")
    prune_budget = float(global_prune_budget(torch.cat(retained, dim=0)).item())
    manifest = {
        "format": FLEXMOE_ACTION_PLAN_FORMAT,
        "schema_version": FLEXMOE_ACTION_PLAN_SCHEMA_VERSION,
        **_lineage(
            dataset_snapshot_id=dataset_snapshot_id,
            model_snapshot_id=model_snapshot_id,
            config_snapshot_id=config_snapshot_id,
        ),
        "ranking_snapshot_id": ranking_id,
        "global_prune_budget": prune_budget,
        "modules": modules,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            FLEXMOE_ACTION_PLAN_METADATA_KEY: json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )


def load_action_plans(
    path: str | Path,
) -> tuple[dict[str, FlexMoEActionPlan], dict[str, str]]:
    """Load final actions and their ranking/model/data/config lineage."""

    _require_torch()
    from safetensors import safe_open
    from safetensors.torch import load_file

    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        raw_manifest = (handle.metadata() or {}).get(FLEXMOE_ACTION_PLAN_METADATA_KEY)
    if raw_manifest is None:
        raise ValueError("FlexMoE action-plan artifact is missing its Mirai manifest.")
    manifest = json.loads(raw_manifest)
    if manifest.get("format") != FLEXMOE_ACTION_PLAN_FORMAT:
        raise ValueError(f"Unsupported FlexMoE action-plan format {manifest.get('format')!r}.")
    if int(manifest.get("schema_version", 0)) != FLEXMOE_ACTION_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported FlexMoE action-plan schema {manifest.get('schema_version')!r}."
        )
    lineage = {
        key: str(manifest.get(key, ""))
        for key in (
            "dataset_snapshot_id",
            "model_snapshot_id",
            "config_snapshot_id",
            "ranking_snapshot_id",
        )
    }
    if not all(lineage.values()):
        raise ValueError("FlexMoE action-plan artifact has incomplete lineage.")
    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("FlexMoE action-plan artifact has no modules.")
    tensors = load_file(str(source), device="cpu")
    loaded: dict[str, FlexMoEActionPlan] = {}
    for raw_spec in modules:
        if not isinstance(raw_spec, Mapping):
            raise ValueError("FlexMoE action-plan module entries must be objects.")
        name = str(raw_spec.get("name", ""))
        if not name or name in loaded:
            raise ValueError("FlexMoE action-plan module names must be unique and non-empty.")
        logits_name = str(raw_spec.get("logits", ""))
        load_name = str(raw_spec.get("expert_load", ""))
        if logits_name not in tensors or load_name not in tensors:
            raise ValueError(f"FlexMoE action-plan tensors for {name!r} are missing.")
        plan = FlexMoEActionPlan(
            action_ratios=tuple(raw_spec.get("action_ratios", ())),
            logits=tensors[logits_name],
            expert_load=tensors[load_name],
        ).validate()
        if plan.num_experts != int(raw_spec.get("num_experts", 0)):
            raise ValueError(f"FlexMoE action-plan topology mismatch for {name!r}.")
        recorded_budget = float(raw_spec.get("prune_budget", float("nan")))
        if not math.isfinite(recorded_budget) or not math.isclose(
            recorded_budget,
            plan.prune_budget(),
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError(f"FlexMoE action-plan budget mismatch for {name!r}.")
        loaded[name] = plan
    recorded_global_budget = float(manifest.get("global_prune_budget", float("nan")))
    actual_global_budget = float(
        global_prune_budget(
            torch.cat(
                [plan.retention_ratios() for plan in loaded.values()],
                dim=0,
            )
        ).item()
    )
    if not math.isfinite(recorded_global_budget) or not math.isclose(
        recorded_global_budget,
        actual_global_budget,
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise ValueError("FlexMoE action-plan global budget mismatch.")
    return loaded, lineage


def rank_channels(saliency: Any) -> Any:
    """Return deterministic per-expert descending channel permutations."""

    _require_torch()
    scores = torch.as_tensor(saliency).detach()
    if scores.ndim != 2 or min(int(value) for value in scores.shape) < 1:
        raise ValueError("FlexMoE saliency must have shape [experts, intermediate].")
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("FlexMoE saliency contains non-finite values.")
    return torch.argsort(scores, dim=1, descending=True, stable=True)


def apply_channel_permutation(
    weights: Mapping[str, Any],
    permutation: Any,
) -> dict[str, Any]:
    """Apply Equation 3 without changing the full-width expert function."""

    dense = _validated_triplet(weights, label="weights")
    order = torch.as_tensor(
        permutation,
        device=dense["w1"].device,
        dtype=torch.long,
    )
    experts, intermediate, hidden = dense["w1"].shape
    if tuple(order.shape) != (int(experts), int(intermediate)):
        raise ValueError("FlexMoE permutation must have shape [experts, intermediate].")
    reference = torch.arange(intermediate, device=order.device).expand(experts, -1)
    if not torch.equal(torch.sort(order, dim=1).values, reference):
        raise ValueError("Each FlexMoE expert order must be a complete permutation.")
    row_index = order.unsqueeze(2).expand(-1, -1, hidden)
    column_index = order.unsqueeze(1).expand(-1, hidden, -1)
    return {
        "w1": dense["w1"].gather(1, row_index).contiguous(),
        "w2": dense["w2"].gather(2, column_index).contiguous(),
        "w3": dense["w3"].gather(1, row_index).contiguous(),
    }


def retained_width(intermediate: int, ratio: float) -> int:
    """Equation-4 prefix width ``ceil(r * d_ff)``."""

    width = int(intermediate)
    value = float(ratio)
    if width < 1:
        raise ValueError("FlexMoE intermediate width must be positive.")
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("FlexMoE retention ratio must be finite and in (0, 1].")
    return min(width, int(math.ceil(value * width)))


def prefix_masks(
    ratios: Any,
    *,
    intermediate: int,
    dtype: Any | None = None,
) -> Any:
    """Materialize Equation-4 token-independent per-expert prefix masks."""

    _require_torch()
    values = torch.as_tensor(ratios)
    if values.ndim != 1 or int(values.numel()) < 1:
        raise ValueError("FlexMoE ratios must be a non-empty expert vector.")
    if not bool(torch.isfinite(values).all().item()) or bool(
        ((values <= 0.0) | (values > 1.0)).any().item()
    ):
        raise ValueError("FlexMoE retention ratios must lie in (0, 1].")
    widths = torch.ceil(values.float() * int(intermediate)).to(dtype=torch.long)
    channels = torch.arange(int(intermediate), device=values.device).unsqueeze(0)
    return (channels < widths.unsqueeze(1)).to(dtype=dtype or values.dtype)


def normalize_action_ratios(values: Sequence[float]) -> tuple[float, ...]:
    """Validate the discrete Equation-5 action set without inventing values."""

    ratios = tuple(float(value) for value in values)
    if len(ratios) < 2:
        raise ValueError("FlexMoE requires at least two discrete retention actions.")
    if any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in ratios):
        raise ValueError("FlexMoE action ratios must be finite and in (0, 1].")
    if tuple(sorted(set(ratios))) != ratios:
        raise ValueError("FlexMoE action ratios must be unique and increasing.")
    if ratios[-1] != 1.0:
        raise ValueError("FlexMoE action ratios must end with the full-width action 1.0.")
    return ratios


def straight_through_gumbel_actions(
    logits: Any,
    *,
    temperature: float,
    generator: Any,
) -> tuple[Any, Any, Any]:
    """Equations 6-8 with an explicit generator for exact resume."""

    _require_torch()
    scores = torch.as_tensor(logits)
    tau = float(temperature)
    if scores.ndim != 2 or min(int(value) for value in scores.shape) < 1:
        raise ValueError("FlexMoE action logits must have shape [experts, actions].")
    if not scores.is_floating_point() or not bool(torch.isfinite(scores).all().item()):
        raise ValueError("FlexMoE action logits must be finite floating-point values.")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("FlexMoE Gumbel temperature must be finite and positive.")
    if generator is None:
        raise ValueError("FlexMoE action sampling requires an explicit generator.")
    uniform = torch.rand(
        scores.shape,
        device=scores.device,
        dtype=torch.float32,
        generator=generator,
    ).clamp_(min=torch.finfo(torch.float32).tiny, max=1.0 - torch.finfo(torch.float32).eps)
    gumbel = -torch.log(-torch.log(uniform))
    soft = F.softmax((scores.float() + gumbel) / tau, dim=-1)
    hard = F.one_hot(soft.argmax(dim=-1), num_classes=int(scores.shape[-1])).to(dtype=soft.dtype)
    straight_through = hard.detach() - soft.detach() + soft
    return soft, hard, straight_through


def clean_action_probabilities(logits: Any) -> Any:
    """Return the unnoised, untempered action distribution used by Eq. 10."""

    _require_torch()
    scores = torch.as_tensor(logits)
    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("FlexMoE action logits must have shape [experts, actions].")
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("FlexMoE action logits contain non-finite values.")
    return F.softmax(scores.float(), dim=-1)


def load_sensitive_cost(
    clean_probabilities: Any,
    expert_load: Any,
    action_ratios: Sequence[float],
) -> Any:
    """Equation-10 expected retained compute for one MoE layer."""

    _require_torch()
    probabilities = torch.as_tensor(clean_probabilities)
    load = torch.as_tensor(
        expert_load,
        device=probabilities.device,
        dtype=torch.float32,
    )
    ratios = torch.as_tensor(
        normalize_action_ratios(action_ratios),
        device=probabilities.device,
        dtype=torch.float32,
    )
    if probabilities.ndim != 2 or tuple(probabilities.shape) != (
        int(load.numel()),
        int(ratios.numel()),
    ):
        raise ValueError("FlexMoE probabilities, loads, and action set disagree.")
    if (
        load.ndim != 1
        or not bool(torch.isfinite(load).all().item())
        or bool((load < 0.0).any().item())
    ):
        raise ValueError("FlexMoE expert load must be a finite non-negative vector.")
    if not torch.isclose(load.sum(), load.new_tensor(1.0), rtol=1e-5, atol=1e-6):
        raise ValueError("FlexMoE expert load must be an assignment-frequency distribution.")
    if not bool(torch.isfinite(probabilities).all().item()) or bool(
        (probabilities < 0.0).any().item()
    ):
        raise ValueError("FlexMoE action probabilities are invalid.")
    rows = probabilities.sum(dim=-1)
    if not torch.allclose(rows, torch.ones_like(rows), rtol=1e-5, atol=1e-6):
        raise ValueError("FlexMoE action probabilities must sum to one per expert.")
    retained = probabilities.float().matmul(ratios)
    return (load * retained).sum()


def action_entropy(clean_probabilities: Any) -> Any:
    """Mean clean categorical action entropy from Equation 9."""

    _require_torch()
    probabilities = torch.as_tensor(clean_probabilities).float()
    if probabilities.ndim != 2 or int(probabilities.numel()) < 1:
        raise ValueError("FlexMoE action probabilities must be a non-empty matrix.")
    if not bool(torch.isfinite(probabilities).all().item()) or bool(
        (probabilities < 0.0).any().item()
    ):
        raise ValueError("FlexMoE action probabilities are invalid.")
    rows = probabilities.sum(dim=-1)
    if not torch.allclose(rows, torch.ones_like(rows), rtol=1e-5, atol=1e-6):
        raise ValueError("FlexMoE action probabilities must sum to one per expert.")
    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    return -(probabilities * safe.log()).sum(dim=-1).mean()


def hardened_retention_ratios(
    logits: Any,
    action_ratios: Sequence[float],
) -> Any:
    """Equation-11 deterministic deployment actions."""

    probabilities = clean_action_probabilities(logits)
    ratios = torch.as_tensor(
        normalize_action_ratios(action_ratios),
        device=probabilities.device,
        dtype=torch.float32,
    )
    if int(probabilities.shape[-1]) != int(ratios.numel()):
        raise ValueError("FlexMoE logits and action set disagree.")
    return ratios.index_select(0, probabilities.argmax(dim=-1))


def global_prune_budget(retention_ratios: Any) -> Any:
    """Equation-12 unweighted routed-parameter pruning ratio."""

    _require_torch()
    ratios = torch.as_tensor(retention_ratios).float()
    if ratios.ndim < 1 or int(ratios.numel()) < 1:
        raise ValueError("FlexMoE retention plan cannot be empty.")
    if not bool(torch.isfinite(ratios).all().item()) or bool(
        ((ratios <= 0.0) | (ratios > 1.0)).any().item()
    ):
        raise ValueError("FlexMoE retention ratios must lie in (0, 1].")
    return 1.0 - ratios.mean()


class FlexMoEActionController(nn.Module if nn is not None else object):
    """Trainable Equation-5 action logits for one grouped expert module.

    The paper specifies thickest-action initialization but not its numeric
    logit margin, so callers must supply that value explicitly.  Randomness is
    likewise caller-owned: passing an explicit generator makes checkpointed
    replay possible without consuming global Torch RNG.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        action_ratios: Sequence[float],
        thickest_logit_margin: float,
    ) -> None:
        _require_torch()
        super().__init__()
        experts = int(num_experts)
        ratios = normalize_action_ratios(action_ratios)
        margin = float(thickest_logit_margin)
        if experts < 1:
            raise ValueError("FlexMoE action controller requires experts.")
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError(
                "FlexMoE thickest-action logit margin must be finite and positive."
            )
        self.action_ratios = ratios
        initial = torch.zeros(experts, len(ratios), dtype=torch.float32)
        initial[:, -1] = margin
        self.logits = nn.Parameter(initial)

    @property
    def num_experts(self) -> int:
        return int(self.logits.shape[0])

    def sampled_original_channel_masks(
        self,
        permutation: Any,
        *,
        temperature: float,
        generator: Any,
    ) -> tuple[Any, Any, Any]:
        """Sample hard ranked prefixes and map them to original channel ids."""

        order = torch.as_tensor(
            permutation,
            device=self.logits.device,
            dtype=torch.long,
        )
        if order.ndim != 2 or int(order.shape[0]) != self.num_experts:
            raise ValueError(
                "FlexMoE channel permutation must have shape [experts, intermediate]."
            )
        reference = torch.arange(order.shape[1], device=order.device).expand_as(order)
        if not torch.equal(torch.sort(order, dim=1).values, reference):
            raise ValueError("FlexMoE action controller received an invalid permutation.")
        soft, hard, actions = straight_through_gumbel_actions(
            self.logits,
            temperature=float(temperature),
            generator=generator,
        )
        action_prefixes = prefix_masks(
            torch.tensor(
                self.action_ratios,
                device=self.logits.device,
                dtype=torch.float32,
            ),
            intermediate=int(order.shape[1]),
            dtype=torch.float32,
        )
        ranked_masks = actions.matmul(action_prefixes)
        original_masks = torch.zeros_like(ranked_masks).scatter(1, order, ranked_masks)
        return original_masks, soft, hard

    def regularization(
        self,
        expert_load: Any,
        *,
        cost_weight: float,
        entropy_weight: float,
    ) -> tuple[Any, dict[str, Any]]:
        """Return ``lambda*C - beta*H`` from Equation 9."""

        lambda_cost = float(cost_weight)
        beta = float(entropy_weight)
        if not math.isfinite(lambda_cost) or lambda_cost < 0.0:
            raise ValueError("FlexMoE cost weight must be finite and non-negative.")
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("FlexMoE entropy weight must be finite and non-negative.")
        probabilities = clean_action_probabilities(self.logits)
        cost = load_sensitive_cost(
            probabilities,
            expert_load,
            self.action_ratios,
        )
        entropy = action_entropy(probabilities)
        return lambda_cost * cost - beta * entropy, {
            "cost": cost,
            "entropy": entropy,
        }

    def action_plan(self, expert_load: Any) -> FlexMoEActionPlan:
        return FlexMoEActionPlan(
            action_ratios=self.action_ratios,
            logits=self.logits.detach().cpu().clone(),
            expert_load=torch.as_tensor(expert_load).detach().cpu().clone(),
        ).validate()


__all__ = [
    "FLEXMOE_ACTION_PLAN_FORMAT",
    "FLEXMOE_ACTION_PLAN_METADATA_KEY",
    "FLEXMOE_ACTION_PLAN_SCHEMA_VERSION",
    "FLEXMOE_RANKING_FORMAT",
    "FLEXMOE_RANKING_METADATA_KEY",
    "FLEXMOE_RANKING_SCHEMA_VERSION",
    "FlexMoEActionPlan",
    "FlexMoEActionController",
    "FlexMoEChannelSaliencyAccumulator",
    "FlexMoECalibrationTarget",
    "FlexMoEExpertLoadObserver",
    "FlexMoETaylorGradientObserver",
    "FlexMoERankingEvidence",
    "action_entropy",
    "apply_channel_permutation",
    "channel_taylor_saliency",
    "clean_action_probabilities",
    "global_prune_budget",
    "hardened_retention_ratios",
    "load_action_plans",
    "load_ranking_evidence",
    "load_sensitive_cost",
    "normalize_action_ratios",
    "prefix_masks",
    "rank_channels",
    "retained_width",
    "save_action_plans",
    "save_ranking_evidence",
    "straight_through_gumbel_actions",
]
