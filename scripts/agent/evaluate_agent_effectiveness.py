"""Compare two pinned coding-agent systems on paired Mirai tasks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable


REQUIRED_FIELDS = {
    "task_id",
    "task_class",
    "run_id",
    "agent_system_id",
    "environment_id",
    "condition_order_index",
    "trajectory_ref",
    "hidden_contract_pass",
    "architecture_pass",
    "success",
    "duration_seconds",
    "input_tokens",
    "output_tokens",
    "tool_calls",
    "files_opened",
    "irrelevant_files_opened",
    "checks_run",
    "irrelevant_checks_run",
    "owner_rank",
    "invalid_edits",
    "regressions",
}
BOOLEAN_FIELDS = {"hidden_contract_pass", "architecture_pass", "success"}
NONNEGATIVE_FIELDS = {
    "duration_seconds",
    "input_tokens",
    "output_tokens",
    "tool_calls",
    "files_opened",
    "irrelevant_files_opened",
    "checks_run",
    "irrelevant_checks_run",
    "owner_rank",
    "invalid_edits",
    "regressions",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing fields: {sorted(missing)}"
            )
        key = _key(record)
        if key in keys:
            raise ValueError(f"{path}:{line_number}: duplicate paired key: {key}")
        keys.add(key)
        records.append(record)
    if not records:
        raise ValueError(f"{path}: no evaluation records")
    return records


def _key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["task_id"]), str(record["run_id"])


def _paired(
    baseline: Iterable[dict[str, Any]],
    candidate: Iterable[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    baseline_by_key = {_key(record): record for record in baseline}
    candidate_by_key = {_key(record): record for record in candidate}
    if baseline_by_key.keys() != candidate_by_key.keys():
        missing_candidate = sorted(baseline_by_key.keys() - candidate_by_key.keys())
        missing_baseline = sorted(candidate_by_key.keys() - baseline_by_key.keys())
        raise ValueError(
            "Evaluation conditions are not paired: "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )
    pairs = []
    for key in sorted(baseline_by_key):
        baseline_record = baseline_by_key[key]
        candidate_record = candidate_by_key[key]
        for field in ("task_class", "environment_id"):
            if baseline_record[field] != candidate_record[field]:
                raise ValueError(f"Paired records disagree on {field}: {key}")
        if baseline_record["agent_system_id"] == candidate_record["agent_system_id"]:
            raise ValueError(f"Paired records use the same agent system: {key}")
        if {
            baseline_record["condition_order_index"],
            candidate_record["condition_order_index"],
        } != {0, 1}:
            raise ValueError(f"Paired condition order is invalid: {key}")
        pairs.append((baseline_record, candidate_record))
    return pairs


def _validate_repetitions(records: Iterable[dict[str, Any]]) -> None:
    repetitions: dict[str, set[str]] = {}
    for record in records:
        repetitions.setdefault(str(record["task_id"]), set()).add(str(record["run_id"]))
    insufficient = {
        task_id: len(run_ids)
        for task_id, run_ids in repetitions.items()
        if len(run_ids) < 3
    }
    if insufficient:
        raise ValueError(f"Each task requires at least three repetitions: {insufficient}")


def _validate_records(records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Evaluation record is missing fields: {sorted(missing)}")
        for field in BOOLEAN_FIELDS:
            if not isinstance(record[field], bool):
                raise ValueError(f"Evaluation field {field!r} must be boolean")
        for field in NONNEGATIVE_FIELDS:
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Evaluation field {field!r} must be non-negative")
        if record["owner_rank"] < 1:
            raise ValueError("owner_rank must be at least 1")
        if record["condition_order_index"] not in (0, 1):
            raise ValueError("condition_order_index must be 0 or 1")


def _rate(records: Iterable[dict[str, Any]], field: str) -> float:
    values = [1.0 if bool(record[field]) else 0.0 for record in records]
    return fmean(values)


def _numeric(records: Iterable[dict[str, Any]], field: str) -> list[float]:
    return [float(record[field]) for record in records]


def _relative_median_reduction(
    baseline: Iterable[dict[str, Any]],
    candidate: Iterable[dict[str, Any]],
    field: str,
) -> float:
    baseline_median = median(_numeric(baseline, field))
    candidate_median = median(_numeric(candidate, field))
    if baseline_median <= 0:
        return 0.0
    return (baseline_median - candidate_median) / baseline_median


def _bootstrap_interval(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    field: str,
    samples: int = 5000,
    seed: int = 42,
) -> list[float]:
    rng = random.Random(seed)
    reductions: list[float] = []
    for _ in range(samples):
        sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        reductions.append(
            _relative_median_reduction(
                [pair[0] for pair in sampled],
                [pair[1] for pair in sampled],
                field,
            )
        )
    reductions.sort()
    return [
        reductions[int(0.025 * (samples - 1))],
        reductions[int(0.975 * (samples - 1))],
    ]


def evaluate(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    _validate_records(baseline)
    _validate_records(candidate)
    _validate_repetitions(baseline)
    _validate_repetitions(candidate)
    pairs = _paired(baseline, candidate)
    if len(pairs) > 1 and {
        pair[0]["condition_order_index"] for pair in pairs
    } != {0, 1}:
        raise ValueError("Condition order must be randomized across paired runs")
    time_reduction = _relative_median_reduction(
        baseline, candidate, "duration_seconds"
    )
    baseline_tokens = [
        {**record, "total_tokens": float(record["input_tokens"]) + float(record["output_tokens"])}
        for record in baseline
    ]
    candidate_tokens = [
        {**record, "total_tokens": float(record["input_tokens"]) + float(record["output_tokens"])}
        for record in candidate
    ]
    token_pairs = list(zip(baseline_tokens, candidate_tokens))
    token_reduction = _relative_median_reduction(
        baseline_tokens, candidate_tokens, "total_tokens"
    )

    metrics = {
        "paired_runs": len(pairs),
        "baseline_hidden_contract_pass_rate": _rate(
            baseline, "hidden_contract_pass"
        ),
        "candidate_hidden_contract_pass_rate": _rate(
            candidate, "hidden_contract_pass"
        ),
        "baseline_success_rate": _rate(baseline, "success"),
        "candidate_success_rate": _rate(candidate, "success"),
        "candidate_architecture_failures": sum(
            not bool(record["architecture_pass"]) for record in candidate
        ),
        "median_duration_reduction": time_reduction,
        "median_duration_reduction_ci95": _bootstrap_interval(
            pairs, field="duration_seconds"
        ),
        "median_token_reduction": token_reduction,
        "median_token_reduction_ci95": _bootstrap_interval(
            token_pairs, field="total_tokens"
        ),
        "baseline_mean_irrelevant_files_opened": fmean(
            _numeric(baseline, "irrelevant_files_opened")
        ),
        "candidate_mean_irrelevant_files_opened": fmean(
            _numeric(candidate, "irrelevant_files_opened")
        ),
        "baseline_mean_irrelevant_checks_run": fmean(
            _numeric(baseline, "irrelevant_checks_run")
        ),
        "candidate_mean_irrelevant_checks_run": fmean(
            _numeric(candidate, "irrelevant_checks_run")
        ),
        "baseline_mean_owner_rank": fmean(_numeric(baseline, "owner_rank")),
        "candidate_mean_owner_rank": fmean(_numeric(candidate, "owner_rank")),
        "baseline_invalid_edits": sum(
            int(record["invalid_edits"]) for record in baseline
        ),
        "candidate_invalid_edits": sum(
            int(record["invalid_edits"]) for record in candidate
        ),
        "baseline_regressions": sum(int(record["regressions"]) for record in baseline),
        "candidate_regressions": sum(
            int(record["regressions"]) for record in candidate
        ),
    }
    failures: list[str] = []
    if (
        metrics["candidate_hidden_contract_pass_rate"]
        < metrics["baseline_hidden_contract_pass_rate"]
    ):
        failures.append("hidden contract pass rate decreased")
    if metrics["candidate_success_rate"] < metrics["baseline_success_rate"]:
        failures.append("task success rate decreased")
    if metrics["candidate_architecture_failures"] != 0:
        failures.append("candidate introduced architecture failures")
    if max(time_reduction, token_reduction) < 0.2:
        failures.append("neither median duration nor token use improved by 20%")
    if (
        metrics["candidate_mean_irrelevant_files_opened"]
        > metrics["baseline_mean_irrelevant_files_opened"]
    ):
        failures.append("irrelevant file exploration increased")
    if (
        metrics["candidate_mean_irrelevant_checks_run"]
        > metrics["baseline_mean_irrelevant_checks_run"]
    ):
        failures.append("irrelevant check execution increased")
    if metrics["candidate_mean_owner_rank"] > metrics["baseline_mean_owner_rank"]:
        failures.append("owner localization rank worsened")
    if metrics["candidate_invalid_edits"] > metrics["baseline_invalid_edits"]:
        failures.append("invalid edits increased")
    if metrics["candidate_regressions"] > metrics["baseline_regressions"]:
        failures.append("regressions increased")
    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "metrics": metrics,
        "failures": failures,
    }


def _self_check() -> dict[str, Any]:
    def record(task: str, run: int, *, candidate: bool) -> dict[str, Any]:
        return {
            "task_id": task,
            "task_class": "owner_localization",
            "run_id": str(run),
            "agent_system_id": "candidate-v1" if candidate else "baseline-v1",
            "environment_id": "pinned-eval-v1",
            "condition_order_index": (
                (run + 1) % 2 if candidate else run % 2
            ),
            "trajectory_ref": f"trajectory://{task}/{run}/{'candidate' if candidate else 'baseline'}",
            "hidden_contract_pass": True,
            "architecture_pass": True,
            "success": True,
            "duration_seconds": 70 if candidate else 100,
            "input_tokens": 5600 if candidate else 8000,
            "output_tokens": 1400 if candidate else 2000,
            "tool_calls": 7 if candidate else 10,
            "files_opened": 5 if candidate else 9,
            "irrelevant_files_opened": 1 if candidate else 4,
            "checks_run": 2 if candidate else 5,
            "irrelevant_checks_run": 0 if candidate else 3,
            "owner_rank": 1,
            "invalid_edits": 0,
            "regressions": 0,
        }

    baseline = [record(task, run, candidate=False) for task in ("a", "b") for run in range(3)]
    candidate = [record(task, run, candidate=True) for task in ("a", "b") for run in range(3)]
    result = evaluate(baseline, candidate)
    if result["status"] != "passed":
        raise RuntimeError(f"agent effectiveness self-check failed: {result}")
    regression = [{**record} for record in candidate]
    regression[0]["architecture_pass"] = False
    if evaluate(baseline, regression)["status"] != "failed":
        raise RuntimeError("agent effectiveness regression guard failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--out")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        result = _self_check()
    else:
        if not args.baseline or not args.candidate:
            parser.error("--baseline and --candidate are required")
        result = evaluate(
            _load_jsonl(Path(args.baseline)),
            _load_jsonl(Path(args.candidate)),
        )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
