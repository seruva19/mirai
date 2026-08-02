"""Per-expert projection importance from routed input second moments.

The statistic follows llama.cpp's open importance-matrix contract: each input
channel accumulates the sum of its squared activations. Routed matrix
multiplication keeps a separate vector and observation count per physical
expert. See https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_PROJECTIONS = ("w1", "w2", "w3")


@dataclass(frozen=True)
class ExpertImportanceEvidence:
    """Input-square sums and routed observation counts for one expert host."""

    input_sum_squares: Mapping[str, Any]
    observation_counts: Mapping[str, Any]

    def validate(self) -> "ExpertImportanceEvidence":
        if torch is None:  # pragma: no cover
            raise RuntimeError("Expert importance calibration requires torch.")
        if set(self.input_sum_squares) != set(_PROJECTIONS) or set(
            self.observation_counts
        ) != set(_PROJECTIONS):
            raise ValueError("Expert importance evidence requires w1, w2, and w3.")
        num_experts = 0
        for projection in _PROJECTIONS:
            values = torch.as_tensor(self.input_sum_squares[projection])
            counts = torch.as_tensor(self.observation_counts[projection])
            if values.ndim != 2 or int(values.shape[0]) < 1:
                raise ValueError("Importance values must have shape [experts, input].")
            if counts.ndim != 1 or int(counts.shape[0]) != int(values.shape[0]):
                raise ValueError("Importance counts must have shape [experts].")
            if num_experts and int(values.shape[0]) != num_experts:
                raise ValueError("Importance projections disagree on expert count.")
            num_experts = int(values.shape[0])
            if not bool(torch.isfinite(values).all().item()) or bool(
                (values < 0).any().item()
            ):
                raise ValueError("Importance values must be finite and non-negative.")
            if bool((counts < 0).any().item()):
                raise ValueError("Importance counts must be non-negative.")
        if not torch.equal(
            torch.as_tensor(self.input_sum_squares["w1"]),
            torch.as_tensor(self.input_sum_squares["w3"]),
        ) or not torch.equal(
            torch.as_tensor(self.observation_counts["w1"]),
            torch.as_tensor(self.observation_counts["w3"]),
        ):
            raise ValueError("w1 and w3 must share their routed-input evidence.")
        return self

    @property
    def num_experts(self) -> int:
        return int(torch.as_tensor(self.input_sum_squares["w1"]).shape[0])

    def mean_squares(self, projection: str, expert_id: int) -> Any:
        self.validate()
        key = str(projection)
        if key not in _PROJECTIONS:
            raise ValueError("Importance projection must be w1, w2, or w3.")
        index = int(expert_id)
        counts = torch.as_tensor(self.observation_counts[key])
        if index < 0 or index >= int(counts.shape[0]):
            raise IndexError("Importance expert_id is out of range.")
        count = int(counts[index].item())
        if count <= 0:
            raise ValueError(
                f"Importance calibration observed no inputs for expert {index} {key}."
            )
        return torch.as_tensor(self.input_sum_squares[key][index]) / float(count)


@dataclass(frozen=True)
class ExpertImportanceCalibrationTarget:
    """Provider-owned routed expert host and its exact floating-point weights."""

    name: str
    host: Any
    weights: Mapping[str, Any]

    def validate(self) -> "ExpertImportanceCalibrationTarget":
        if not str(self.name).strip():
            raise ValueError("Expert importance target name must be non-empty.")
        if not callable(
            getattr(self.host, "set_importance_calibration_observer", None)
        ) or not callable(
            getattr(self.host, "clear_importance_calibration_observer", None)
        ):
            raise TypeError(
                "Expert importance target host must expose observer setters."
            )
        if set(self.weights) != set(_PROJECTIONS):
            raise ValueError("Expert importance target requires w1, w2, and w3.")
        shapes = {
            key: tuple(int(value) for value in torch.as_tensor(weight).shape)
            for key, weight in self.weights.items()
        }
        if any(len(shape) != 3 for shape in shapes.values()):
            raise ValueError("Expert importance weights must have [E, out, in] shape.")
        experts = shapes["w1"][0]
        if experts < 2 or any(shape[0] != experts for shape in shapes.values()):
            raise ValueError("Expert importance weights disagree on expert count.")
        if shapes["w1"][2] != shapes["w3"][2]:
            raise ValueError("w1 and w3 input dimensions must match.")
        return self

    @property
    def num_experts(self) -> int:
        return int(torch.as_tensor(self.weights["w1"]).shape[0])

    @property
    def accumulator_bytes(self) -> int:
        self.validate()
        dimensions = (
            int(torch.as_tensor(self.weights["w1"]).shape[-1]),
            int(torch.as_tensor(self.weights["w2"]).shape[-1]),
        )
        return 8 * self.num_experts * sum(dimensions) + 16 * self.num_experts


class ExpertImportanceAccumulator:
    """Bounded CPU accumulator for routed per-expert input-square sums."""

    def __init__(self, *, num_experts: int, input_dims: Mapping[str, int]) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Expert importance calibration requires torch.")
        experts = int(num_experts)
        dims = {str(key): int(value) for key, value in input_dims.items()}
        if experts < 2 or set(dims) != set(_PROJECTIONS):
            raise ValueError("Importance accumulator requires experts and w1/w2/w3.")
        if any(value <= 0 for value in dims.values()) or dims["w1"] != dims["w3"]:
            raise ValueError("Importance accumulator input dimensions are invalid.")
        self.num_experts = experts
        self._dims = dims
        self._values = {
            "w13": torch.zeros((experts, dims["w1"]), dtype=torch.float64),
            "w2": torch.zeros((experts, dims["w2"]), dtype=torch.float64),
        }
        self._counts = {
            "w13": torch.zeros(experts, dtype=torch.int64),
            "w2": torch.zeros(experts, dtype=torch.int64),
        }

    def record(
        self,
        expert_id: int,
        projections: str | tuple[str, ...],
        inputs: Any,
    ) -> None:
        index = int(expert_id)
        if index < 0 or index >= self.num_experts:
            raise IndexError("Importance observer expert_id is out of range.")
        names = (projections,) if isinstance(projections, str) else tuple(projections)
        if not names or any(name not in _PROJECTIONS for name in names):
            raise ValueError("Importance observer projections must be w1/w2/w3.")
        groups = {"w13" if name in {"w1", "w3"} else "w2" for name in names}
        if len(groups) != 1:
            raise ValueError("One importance observation cannot mix input spaces.")
        group = groups.pop()
        dimension = self._values[group].shape[1]
        values = torch.as_tensor(inputs).detach()
        if values.ndim < 2 or int(values.shape[-1]) != int(dimension):
            raise ValueError("Importance observer input dimension does not match.")
        flat = values.reshape(-1, int(dimension))
        if int(flat.shape[0]) == 0:
            return
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError("Importance calibration observed non-finite inputs.")
        update = flat.float().square().sum(dim=0).to(dtype=torch.float64).cpu()
        self._values[group][index].add_(update)
        self._counts[group][index] += int(flat.shape[0])

    def evidence(self) -> ExpertImportanceEvidence:
        return ExpertImportanceEvidence(
            input_sum_squares={
                "w1": self._values["w13"].clone(),
                "w2": self._values["w2"].clone(),
                "w3": self._values["w13"].clone(),
            },
            observation_counts={
                "w1": self._counts["w13"].clone(),
                "w2": self._counts["w2"].clone(),
                "w3": self._counts["w13"].clone(),
            },
        ).validate()


__all__ = [
    "ExpertImportanceAccumulator",
    "ExpertImportanceCalibrationTarget",
    "ExpertImportanceEvidence",
]
