"""Task-affinity expert selection for Expert-Specialized Fine-Tuning.

This is a clean-room implementation of Equations 6--8 in
https://arxiv.org/abs/2407.01906.  A provider exposes router targets; core
accumulates either selected gate mass (ESFT-Gate) or the paper's ``1 / K``
token-selection mass (ESFT-Token), then chooses the smallest deterministic
per-layer prefix whose normalized relevance reaches ``selection_mass``.

Calibration observes the pretrained routing decision.  It never restricts
which experts may serve tokens: the resulting plan controls only which expert
rows the optimizer may update.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ESFT_SCORE_MODES = frozenset({"gate", "token"})


def normalize_esft_score_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode.startswith("esft_"):
        mode = mode[len("esft_") :]
    if mode not in ESFT_SCORE_MODES:
        raise ValueError("ESFT score mode must be 'gate' or 'token'.")
    return mode


@dataclass(frozen=True)
class ESFTCalibrationTarget:
    """Provider-owned router target corresponding to one grouped expert host."""

    name: str
    router: Any
    num_experts: int

    def validate(self) -> "ESFTCalibrationTarget":
        if not str(self.name).strip():
            raise ValueError("ESFT target name cannot be empty.")
        if int(self.num_experts) <= 0:
            raise ValueError("ESFT target num_experts must be positive.")
        if not callable(getattr(self.router, "register_forward_hook", None)):
            raise TypeError("ESFT target router must support forward hooks.")
        return self


@dataclass
class ESFTAffinityAccumulator:
    """Bounded CPU sufficient statistics for one routed expert layer."""

    num_experts: int
    gate_mass: Any = None
    token_mass: Any = None
    token_count: int = 0
    invocation_count: int = 0

    def __post_init__(self) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("ESFT calibration requires torch.")
        if int(self.num_experts) <= 0:
            raise ValueError("ESFT accumulator num_experts must be positive.")
        self.num_experts = int(self.num_experts)
        self.gate_mass = torch.zeros(self.num_experts, dtype=torch.float64)
        self.token_mass = torch.zeros(self.num_experts, dtype=torch.float64)

    def observe(
        self,
        indices: Any,
        gate_scores: Any,
        *,
        active_mask: Any | None = None,
    ) -> None:
        """Accumulate one fixed-cardinality top-k router invocation."""

        if not torch.is_tensor(indices) or indices.ndim < 2:
            raise ValueError("ESFT routing indices must have token and slot axes.")
        if not torch.is_tensor(gate_scores) or gate_scores.shape != indices.shape:
            raise ValueError("ESFT gate scores must match routing indices.")
        top_k = int(indices.shape[-1])
        flat_indices = indices.detach().reshape(-1, top_k).to(
            device="cpu", dtype=torch.int64
        )
        flat_scores = gate_scores.detach().reshape_as(flat_indices).to(
            device="cpu", dtype=torch.float64
        )
        if int(flat_indices.shape[0]) <= 0 or top_k <= 0:
            raise ValueError("ESFT routing observation cannot be empty.")
        if active_mask is not None:
            if not torch.is_tensor(active_mask) or active_mask.shape != indices.shape:
                raise ValueError("ESFT active_mask must match routing indices.")
            flat_active = active_mask.detach().reshape_as(flat_indices).to(
                device="cpu", dtype=torch.bool
            )
            if not bool(flat_active.all().item()):
                raise ValueError(
                    "ESFT Equations 6--8 require fixed-cardinality top-k routing; "
                    "inactive or variable-cardinality routes were observed."
                )
        if (
            int(flat_indices.min().item()) < 0
            or int(flat_indices.max().item()) >= self.num_experts
        ):
            raise ValueError("ESFT routing observation contains an invalid expert id.")
        if not bool(torch.isfinite(flat_scores).all().item()) or bool(
            (flat_scores < 0).any().item()
        ):
            raise ValueError("ESFT gate scores must be finite and non-negative.")

        self.gate_mass.scatter_add_(
            0,
            flat_indices.reshape(-1),
            flat_scores.reshape(-1),
        )
        self.token_mass.scatter_add_(
            0,
            flat_indices.reshape(-1),
            torch.full(
                (int(flat_indices.numel()),),
                1.0 / float(top_k),
                dtype=torch.float64,
            ),
        )
        self.token_count += int(flat_indices.shape[0])
        self.invocation_count += 1

    def normalized_scores(self, mode: str) -> Any:
        score_mode = normalize_esft_score_mode(mode)
        raw = self.gate_mass if score_mode == "gate" else self.token_mass
        total = float(raw.sum().item())
        if not total > 0.0:
            raise ValueError("ESFT calibration observed zero total relevance mass.")
        return raw / total


def select_esft_experts(scores: Any, *, selection_mass: float) -> tuple[int, ...]:
    """Choose the smallest stable top-score prefix reaching cumulative mass."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("ESFT calibration requires torch.")
    if not torch.is_tensor(scores) or scores.ndim != 1 or not scores.is_floating_point():
        raise ValueError("ESFT scores must be a floating rank-1 tensor.")
    mass = float(selection_mass)
    if not (0.0 < mass <= 1.0):
        raise ValueError("ESFT selection_mass must be in (0, 1].")
    if int(scores.numel()) <= 0 or not bool(torch.isfinite(scores).all().item()):
        raise ValueError("ESFT scores must be non-empty and finite.")
    values = [float(value) for value in scores.detach().cpu().tolist()]
    total = sum(values)
    if not total > 0.0:
        raise ValueError("ESFT scores must contain positive relevance mass.")
    order = sorted(range(len(values)), key=lambda expert_id: (-values[expert_id], expert_id))
    selected: list[int] = []
    cumulative = 0.0
    for expert_id in order:
        selected.append(expert_id)
        cumulative += values[expert_id] / total
        if cumulative + 1e-15 >= mass:
            break
    return tuple(sorted(selected))


@dataclass(frozen=True)
class ESFTSelectionPlan:
    """Serializable per-layer expert update plan."""

    score_mode: str
    selection_mass: float
    calibration_samples: int
    selected_experts: dict[str, tuple[int, ...]]
    normalized_scores: dict[str, tuple[float, ...]]
    token_counts: dict[str, int]
    invocation_counts: dict[str, int]

    @property
    def fingerprint(self) -> str:
        payload = {
            "score_mode": normalize_esft_score_mode(self.score_mode),
            "selection_mass": float(self.selection_mass),
            "selected_experts": {
                name: list(ids)
                for name, ids in sorted(self.selected_experts.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_mode": normalize_esft_score_mode(self.score_mode),
            "selection_mass": float(self.selection_mass),
            "calibration_samples": int(self.calibration_samples),
            "selected_experts": {
                name: list(ids)
                for name, ids in sorted(self.selected_experts.items())
            },
            "normalized_scores": {
                name: list(values)
                for name, values in sorted(self.normalized_scores.items())
            },
            "token_counts": dict(sorted(self.token_counts.items())),
            "invocation_counts": dict(sorted(self.invocation_counts.items())),
            "fingerprint": self.fingerprint,
        }


def build_esft_selection_plan(
    accumulators: Mapping[str, ESFTAffinityAccumulator],
    *,
    score_mode: str,
    selection_mass: float,
    calibration_samples: int,
) -> ESFTSelectionPlan:
    """Resolve all provider targets into one deterministic update plan."""

    mode = normalize_esft_score_mode(score_mode)
    if not accumulators:
        raise ValueError("ESFT calibration requires at least one target.")
    selected: dict[str, tuple[int, ...]] = {}
    normalized: dict[str, tuple[float, ...]] = {}
    token_counts: dict[str, int] = {}
    invocation_counts: dict[str, int] = {}
    for name, accumulator in sorted(accumulators.items()):
        if accumulator.token_count <= 0 or accumulator.invocation_count <= 0:
            raise ValueError(f"ESFT target {name!r} has no routing observations.")
        scores = accumulator.normalized_scores(mode)
        selected[str(name)] = select_esft_experts(
            scores,
            selection_mass=float(selection_mass),
        )
        normalized[str(name)] = tuple(float(value) for value in scores.tolist())
        token_counts[str(name)] = int(accumulator.token_count)
        invocation_counts[str(name)] = int(accumulator.invocation_count)
    return ESFTSelectionPlan(
        score_mode=mode,
        selection_mass=float(selection_mass),
        calibration_samples=int(calibration_samples),
        selected_experts=selected,
        normalized_scores=normalized,
        token_counts=token_counts,
        invocation_counts=invocation_counts,
    )


class ESFTCalibrationCapture:
    """Temporary provider-target hooks; no device tensor survives a hook call."""

    def __init__(self, targets: Mapping[str, ESFTCalibrationTarget]) -> None:
        if not targets:
            raise ValueError("ESFT capture requires targets.")
        self.targets: dict[str, ESFTCalibrationTarget] = {}
        for raw_name, target in targets.items():
            if not isinstance(target, ESFTCalibrationTarget):
                raise TypeError("ESFT provider targets must use ESFTCalibrationTarget.")
            target.validate()
            name = str(raw_name)
            if name != target.name or name in self.targets:
                raise ValueError("ESFT target names must match and be unique.")
            self.targets[name] = target
        self.accumulators = {
            name: ESFTAffinityAccumulator(target.num_experts)
            for name, target in self.targets.items()
        }
        self._handles: list[Any] = []

    def __enter__(self) -> "ESFTCalibrationCapture":
        if self._handles:
            raise RuntimeError("ESFT capture is already active.")
        try:
            for name, target in self.targets.items():

                def _hook(
                    module: Any,
                    _inputs: Any,
                    _output: Any,
                    *,
                    _name=name,
                ) -> None:
                    indices = getattr(module, "last_top_indices", None)
                    scores = getattr(module, "last_top_scores", None)
                    if indices is None or scores is None:
                        return
                    self.accumulators[_name].observe(
                        indices,
                        scores,
                        active_mask=getattr(module, "last_route_active_mask", None),
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


__all__ = [
    "ESFTAffinityAccumulator",
    "ESFTCalibrationCapture",
    "ESFTCalibrationTarget",
    "ESFTSelectionPlan",
    "ESFT_SCORE_MODES",
    "build_esft_selection_plan",
    "normalize_esft_score_mode",
    "select_esft_experts",
]
