"""Calibration and persistence contract for mixed-precision expert storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


_FORMAT_BITS = {
    "bf16": 16,
    "fp8": 8,
    "int8": 8,
    "nf4": 4,
    "gguf_iq4": 4,
    "mxfp4": 4,
    "nvfp4": 4,
    "mxfp8_e4m3": 8,
    "gguf_iq3": 3,
}
_PROJECTIONS = ("w1", "w2", "w3")


@dataclass(frozen=True)
class ExpertPrecisionEvidence:
    """Measured reconstruction loss for one expert at each candidate format."""

    expert_id: int
    weight_numel: int
    format_error: Mapping[str, float]
    routing_frequency: float = 1.0


@dataclass(frozen=True)
class ExpertPrecisionPlan:
    """Versioned, deterministic per-expert quantization assignment."""

    schema_version: int
    formats: tuple[str, ...]
    estimated_bytes: int
    weighted_error: float
    budget_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ExpertPrecisionPlan":
        version = int(payload.get("schema_version", 0))
        if version != 1:
            raise ValueError(f"Unsupported expert precision plan version {version}.")
        formats = tuple(str(value) for value in payload.get("formats", ()))
        if not formats or any(value not in _FORMAT_BITS for value in formats):
            raise ValueError("Expert precision plan contains an unsupported format.")
        return cls(
            schema_version=version,
            formats=formats,
            estimated_bytes=int(payload["estimated_bytes"]),
            weighted_error=float(payload["weighted_error"]),
            budget_bytes=int(payload["budget_bytes"]),
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "ExpertPrecisionPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class TensorPrecisionEvidence:
    """Measured format cost for one physical expert projection."""

    module_name: str
    expert_id: int
    projection: str
    weight_numel: int
    format_error: Mapping[str, float]
    format_bytes: Mapping[str, int]
    routing_frequency: float = 1.0

    def validate(self, formats: Sequence[str]) -> "TensorPrecisionEvidence":
        if not str(self.module_name).strip():
            raise ValueError("Tensor precision evidence requires a module name.")
        if int(self.expert_id) < 0:
            raise ValueError("Tensor precision evidence expert_id must be non-negative.")
        if str(self.projection) not in _PROJECTIONS:
            raise ValueError("Tensor precision projection must be w1, w2, or w3.")
        if int(self.weight_numel) <= 0 or not math.isfinite(
            float(self.routing_frequency)
        ) or float(self.routing_frequency) < 0.0:
            raise ValueError("Tensor precision evidence has invalid size or frequency.")
        for quant_format in formats:
            if quant_format not in self.format_error:
                raise ValueError(
                    f"{self.module_name}.{self.projection}[{self.expert_id}] has "
                    f"no measured error for {quant_format!r}."
                )
            if quant_format not in self.format_bytes:
                raise ValueError(
                    f"{self.module_name}.{self.projection}[{self.expert_id}] has "
                    f"no measured byte cost for {quant_format!r}."
                )
            error = float(self.format_error[quant_format])
            byte_count = int(self.format_bytes[quant_format])
            if not math.isfinite(error) or error < 0.0 or byte_count <= 0:
                raise ValueError("Tensor precision evidence contains an invalid candidate.")
        return self


@dataclass(frozen=True)
class TensorPrecisionAssignment:
    """One schema-v2 runtime assignment."""

    module_name: str
    expert_id: int
    projection: str
    quant_format: str
    stored_bytes: int
    weighted_error: float


@dataclass(frozen=True)
class TensorPrecisionPlan:
    """Lineage-bound per-module, per-expert, per-projection precision plan."""

    schema_version: int
    assignments: tuple[TensorPrecisionAssignment, ...]
    estimated_bytes: int
    weighted_error: float
    budget_bytes: int
    dataset_snapshot_id: str
    model_snapshot_id: str
    config_snapshot_id: str
    source_weight_fingerprint: str

    def validate(self) -> "TensorPrecisionPlan":
        if int(self.schema_version) != 2:
            raise ValueError(
                f"Unsupported tensor precision plan version {self.schema_version}."
            )
        if not self.assignments:
            raise ValueError("Tensor precision plan cannot be empty.")
        if int(self.budget_bytes) <= 0 or int(self.estimated_bytes) <= 0:
            raise ValueError("Tensor precision plan has an invalid byte budget.")
        if int(self.estimated_bytes) > int(self.budget_bytes):
            raise ValueError("Tensor precision plan exceeds its byte budget.")
        if not math.isfinite(float(self.weighted_error)) or self.weighted_error < 0:
            raise ValueError("Tensor precision plan has invalid weighted_error.")
        lineage = (
            self.dataset_snapshot_id,
            self.model_snapshot_id,
            self.config_snapshot_id,
            self.source_weight_fingerprint,
        )
        if not all(str(value).strip() for value in lineage):
            raise ValueError("Tensor precision plan requires complete lineage.")
        seen: set[tuple[str, int, str]] = set()
        modules: dict[str, dict[str, set[int]]] = {}
        total_bytes = 0
        total_error = 0.0
        for assignment in self.assignments:
            key = (
                str(assignment.module_name),
                int(assignment.expert_id),
                str(assignment.projection),
            )
            if (
                not key[0]
                or key[1] < 0
                or key[2] not in _PROJECTIONS
                or key in seen
                or assignment.quant_format not in _FORMAT_BITS
                or int(assignment.stored_bytes) <= 0
                or not math.isfinite(float(assignment.weighted_error))
                or float(assignment.weighted_error) < 0.0
            ):
                raise ValueError("Tensor precision plan contains an invalid assignment.")
            seen.add(key)
            modules.setdefault(key[0], {}).setdefault(key[2], set()).add(key[1])
            total_bytes += int(assignment.stored_bytes)
            total_error += float(assignment.weighted_error)
        for module_name, projections in modules.items():
            if set(projections) != set(_PROJECTIONS):
                raise ValueError(
                    f"Tensor precision module {module_name!r} does not cover w1/w2/w3."
                )
            expected = set(range(max(projections["w1"]) + 1))
            if not expected or any(ids != expected for ids in projections.values()):
                raise ValueError(
                    f"Tensor precision module {module_name!r} has incomplete expert coverage."
                )
        if total_bytes != int(self.estimated_bytes) or not math.isclose(
            total_error,
            float(self.weighted_error),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Tensor precision plan summary does not match assignments.")
        return self

    def formats_for_module(self, module_name: str) -> dict[str, tuple[str, ...]]:
        self.validate()
        selected = [
            item
            for item in self.assignments
            if item.module_name == str(module_name)
        ]
        if not selected:
            raise KeyError(f"Tensor precision plan has no module {module_name!r}.")
        result: dict[str, tuple[str, ...]] = {}
        for projection in _PROJECTIONS:
            rows = sorted(
                (item for item in selected if item.projection == projection),
                key=lambda item: item.expert_id,
            )
            result[projection] = tuple(item.quant_format for item in rows)
        return result

    def module_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.module_name for item in self.assignments))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "assignments": [asdict(item) for item in self.assignments],
            "estimated_bytes": self.estimated_bytes,
            "weighted_error": self.weighted_error,
            "budget_bytes": self.budget_bytes,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "model_snapshot_id": self.model_snapshot_id,
            "config_snapshot_id": self.config_snapshot_id,
            "source_weight_fingerprint": self.source_weight_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TensorPrecisionPlan":
        raw_assignments = payload.get("assignments")
        if not isinstance(raw_assignments, list):
            raise ValueError("Tensor precision plan assignments must be a list.")
        assignments = tuple(
            TensorPrecisionAssignment(
                module_name=str(item["module_name"]),
                expert_id=int(item["expert_id"]),
                projection=str(item["projection"]),
                quant_format=str(item["quant_format"]),
                stored_bytes=int(item["stored_bytes"]),
                weighted_error=float(item["weighted_error"]),
            )
            for item in raw_assignments
            if isinstance(item, Mapping)
        )
        if len(assignments) != len(raw_assignments):
            raise ValueError("Tensor precision plan contains a malformed assignment.")
        return cls(
            schema_version=int(payload.get("schema_version", 0)),
            assignments=assignments,
            estimated_bytes=int(payload["estimated_bytes"]),
            weighted_error=float(payload["weighted_error"]),
            budget_bytes=int(payload["budget_bytes"]),
            dataset_snapshot_id=str(payload.get("dataset_snapshot_id", "")),
            model_snapshot_id=str(payload.get("model_snapshot_id", "")),
            config_snapshot_id=str(payload.get("config_snapshot_id", "")),
            source_weight_fingerprint=str(
                payload.get("source_weight_fingerprint", "")
            ),
        ).validate()

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "TensorPrecisionPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _format_bytes(numel: int, quant_format: str) -> int:
    bits = _FORMAT_BITS[str(quant_format)]
    return int(math.ceil(int(numel) * bits / 8.0))


def allocate_expert_precision(
    evidence: Sequence[ExpertPrecisionEvidence],
    *,
    budget_bytes: int,
    allowed_formats: Sequence[str],
) -> ExpertPrecisionPlan:
    """Minimize routing-weighted error under an exact integer byte ceiling.

    Allocation starts from the smallest representation and greedily selects the
    error reduction with the greatest benefit per additional byte.  Every
    candidate is measured; absent error entries are never guessed.
    """

    if int(budget_bytes) <= 0:
        raise ValueError("budget_bytes must be positive.")
    formats = tuple(dict.fromkeys(str(value).strip().lower() for value in allowed_formats))
    if not formats or any(value not in _FORMAT_BITS for value in formats):
        raise ValueError("allowed_formats contains an unsupported precision.")
    rows = sorted(evidence, key=lambda row: int(row.expert_id))
    if [row.expert_id for row in rows] != list(range(len(rows))):
        raise ValueError("Expert precision evidence must cover contiguous expert ids.")
    for row in rows:
        missing = [value for value in formats if value not in row.format_error]
        if missing:
            raise ValueError(
                f"Expert {row.expert_id} has no measured error for {missing}."
            )
        if row.weight_numel <= 0 or row.routing_frequency < 0:
            raise ValueError("Invalid precision-calibration evidence.")

    ordered = sorted(formats, key=lambda value: (_FORMAT_BITS[value], value))
    assignment = [ordered[0]] * len(rows)
    used = sum(_format_bytes(row.weight_numel, assignment[i]) for i, row in enumerate(rows))
    if used > int(budget_bytes):
        raise ValueError(
            f"Minimum expert representation requires {used} bytes, budget is {budget_bytes}."
        )

    while True:
        best: tuple[float, int, str, int] | None = None
        for index, row in enumerate(rows):
            current = assignment[index]
            current_bytes = _format_bytes(row.weight_numel, current)
            current_error = float(row.format_error[current]) * float(row.routing_frequency)
            for candidate in ordered:
                extra = _format_bytes(row.weight_numel, candidate) - current_bytes
                if extra <= 0 or used + extra > int(budget_bytes):
                    continue
                gain = current_error - (
                    float(row.format_error[candidate]) * float(row.routing_frequency)
                )
                if gain <= 0:
                    continue
                choice = (gain / extra, -index, candidate, extra)
                if best is None or choice > best:
                    best = choice
        if best is None:
            break
        _score, negative_index, candidate, extra = best
        index = -negative_index
        assignment[index] = candidate
        used += extra

    weighted_error = sum(
        float(row.format_error[assignment[index]]) * float(row.routing_frequency)
        for index, row in enumerate(rows)
    )
    return ExpertPrecisionPlan(
        schema_version=1,
        formats=tuple(assignment),
        estimated_bytes=int(used),
        weighted_error=float(weighted_error),
        budget_bytes=int(budget_bytes),
    )


def allocate_tensor_precision(
    evidence: Sequence[TensorPrecisionEvidence],
    *,
    budget_bytes: int,
    allowed_formats: Sequence[str],
    dataset_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str,
    source_weight_fingerprint: str,
) -> TensorPrecisionPlan:
    """Allocate exact measured packed bytes at projection granularity."""

    if int(budget_bytes) <= 0:
        raise ValueError("budget_bytes must be positive.")
    formats = tuple(
        dict.fromkeys(str(value).strip().lower() for value in allowed_formats)
    )
    if not formats or any(value not in _FORMAT_BITS for value in formats):
        raise ValueError("allowed_formats contains an unsupported precision.")
    rows = sorted(
        (row.validate(formats) for row in evidence),
        key=lambda row: (row.module_name, row.expert_id, row.projection),
    )
    keys = [(row.module_name, row.expert_id, row.projection) for row in rows]
    if len(keys) != len(set(keys)) or not rows:
        raise ValueError("Tensor precision evidence keys must be non-empty and unique.")

    def _candidate_key(row: TensorPrecisionEvidence, quant_format: str) -> tuple[int, str]:
        return int(row.format_bytes[quant_format]), str(quant_format)

    assignment = [
        min(formats, key=lambda value, row=row: _candidate_key(row, value))
        for row in rows
    ]
    used = sum(
        int(row.format_bytes[assignment[index]]) for index, row in enumerate(rows)
    )
    if used > int(budget_bytes):
        raise ValueError(
            f"Minimum tensor representations require {used} bytes, "
            f"budget is {budget_bytes}."
        )
    while True:
        best: tuple[float, int, str, int] | None = None
        for index, row in enumerate(rows):
            current = assignment[index]
            current_bytes = int(row.format_bytes[current])
            current_error = (
                float(row.format_error[current]) * float(row.routing_frequency)
            )
            for candidate in formats:
                extra = int(row.format_bytes[candidate]) - current_bytes
                if extra <= 0 or used + extra > int(budget_bytes):
                    continue
                candidate_error = (
                    float(row.format_error[candidate])
                    * float(row.routing_frequency)
                )
                gain = current_error - candidate_error
                if gain <= 0.0:
                    continue
                choice = (gain / extra, -index, candidate, extra)
                if best is None or choice > best:
                    best = choice
        if best is None:
            break
        _score, negative_index, candidate, extra = best
        index = -negative_index
        assignment[index] = candidate
        used += extra

    assignments = tuple(
        TensorPrecisionAssignment(
            module_name=row.module_name,
            expert_id=row.expert_id,
            projection=row.projection,
            quant_format=assignment[index],
            stored_bytes=int(row.format_bytes[assignment[index]]),
            weighted_error=(
                float(row.format_error[assignment[index]])
                * float(row.routing_frequency)
            ),
        )
        for index, row in enumerate(rows)
    )
    return TensorPrecisionPlan(
        schema_version=2,
        assignments=assignments,
        estimated_bytes=sum(item.stored_bytes for item in assignments),
        weighted_error=sum(item.weighted_error for item in assignments),
        budget_bytes=int(budget_bytes),
        dataset_snapshot_id=str(dataset_snapshot_id),
        model_snapshot_id=str(model_snapshot_id),
        config_snapshot_id=str(config_snapshot_id),
        source_weight_fingerprint=str(source_weight_fingerprint),
    ).validate()


def load_precision_plan(
    path: str | Path,
) -> ExpertPrecisionPlan | TensorPrecisionPlan:
    """Load the explicit schema without guessing or mutating v1 semantics."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", 0))
    if version == 1:
        return ExpertPrecisionPlan.from_dict(payload)
    if version == 2:
        return TensorPrecisionPlan.from_dict(payload)
    raise ValueError(f"Unsupported expert precision plan version {version}.")


__all__ = [
    "ExpertPrecisionEvidence",
    "ExpertPrecisionPlan",
    "TensorPrecisionAssignment",
    "TensorPrecisionEvidence",
    "TensorPrecisionPlan",
    "allocate_expert_precision",
    "allocate_tensor_precision",
    "load_precision_plan",
]
