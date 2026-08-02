"""Run paired coding-agent evaluations in isolated Mirai workspaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA_VERSION = 1
TASK_CLASSES = {
    "owner_localization",
    "diagnosis_without_edit",
    "default_off_config_extension",
    "moe_policy_extension",
    "model_family_integration",
    "persistence_and_resume_repair",
    "optimized_reference_parity",
    "long_horizon_evolution",
}
TRACE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "tool_calls",
    "files_opened",
    "checks_run",
}
GRADE_FIELDS = {
    "hidden_contract_pass",
    "architecture_pass",
    "owner_rank",
    "invalid_edits",
    "regressions",
    "irrelevant_files_opened",
    "irrelevant_checks_run",
}
IGNORED_COPY_NAMES = {
    ".agents",
    ".dev-private",
    ".env",
    ".git",
    ".runtime",
    ".tmp",
    "__pycache__",
    "artifacts",
    "cache",
    "logs",
    "models",
    "outputs",
    "wandb",
}
IGNORED_COPY_SUFFIXES = {".ckpt", ".pem", ".pt", ".pyc", ".safetensors"}


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: expected schema_version {SCHEMA_VERSION}")
    return payload


def load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = _load_object(path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path}: tasks must be a non-empty list")
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"{path}: each task must be an object")
        required = {"task_id", "task_class", "prompt", "source_subdir", "grader_id"}
        missing = required - task.keys()
        if missing:
            raise ValueError(f"{path}: task is missing {sorted(missing)}")
        task_id = str(task["task_id"])
        if not task_id or task_id in seen:
            raise ValueError(f"{path}: invalid or duplicate task_id {task_id!r}")
        seen.add(task_id)
        if task["task_class"] not in TASK_CLASSES:
            raise ValueError(f"{path}: invalid task_class {task['task_class']!r}")
    return tasks


def load_adapter(path: Path) -> dict[str, Any]:
    adapter = _load_object(path)
    if not str(adapter.get("agent_system_id", "")).strip():
        raise ValueError(f"{path}: agent_system_id is required")
    argv = adapter.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError(f"{path}: argv must be a non-empty string list")
    timeout = adapter.get("timeout_seconds")
    if not isinstance(timeout, int) or timeout < 1:
        raise ValueError(f"{path}: timeout_seconds must be a positive integer")
    return adapter


def load_graders(path: Path) -> dict[str, list[str]]:
    payload = _load_object(path)
    graders = payload.get("graders")
    if not isinstance(graders, dict) or not graders:
        raise ValueError(f"{path}: graders must be a non-empty object")
    for grader_id, argv in graders.items():
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError(f"{path}: grader {grader_id!r} must be an argv list")
    return graders


def _resolve_task_source(root: Path, source_subdir: str) -> Path:
    root = root.resolve()
    source = (root / source_subdir).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Task source escapes source root: {source_subdir!r}") from exc
    if not source.is_dir():
        raise ValueError(f"Task source does not exist: {source}")
    return source


def _ignore_copy(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in IGNORED_COPY_NAMES
        or name.startswith(".env.")
        or Path(name).suffix.lower() in IGNORED_COPY_SUFFIXES
    }


def _tree_digest(root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in IGNORED_COPY_NAMES for part in path.parts):
            continue
        digests[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    )


def _expand_argv(argv: list[str], values: dict[str, str]) -> list[str]:
    expanded: list[str] = []
    for token in argv:
        try:
            expanded.append(token.format_map(values))
        except KeyError as exc:
            raise ValueError(f"Unknown command placeholder: {exc.args[0]}") from exc
    return expanded


def _read_fields(path: Path, required: set[str], label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{label} is missing {sorted(missing)}")
    return payload


def _require_nonnegative_integers(
    payload: dict[str, Any],
    fields: set[str],
    label: str,
) -> None:
    invalid = [
        field
        for field in fields
        if isinstance(payload[field], bool)
        or not isinstance(payload[field], int)
        or payload[field] < 0
    ]
    if invalid:
        raise ValueError(f"{label} has invalid counters: {sorted(invalid)}")


def _execute_trial_at_workspace(
    *,
    task: dict[str, Any],
    run_id: int,
    condition: str,
    condition_order_index: int,
    source_root: Path,
    adapter: dict[str, Any],
    grader_argv: list[str],
    workspace: Path,
    keep_workspace: bool,
) -> dict[str, Any]:
    source = _resolve_task_source(source_root, str(task["source_subdir"]))
    worktree = workspace / "repo"
    shutil.copytree(source, worktree, ignore=_ignore_copy)
    evidence = workspace / "evidence"
    evidence.mkdir()
    prompt_path = evidence / "prompt.txt"
    trace_path = evidence / "trace.json"
    grade_path = evidence / "grade.json"
    prompt_path.write_text(str(task["prompt"]), encoding="utf-8")
    values = {
        "workspace": str(worktree),
        "prompt_path": str(prompt_path),
        "trace_path": str(trace_path),
        "grade_path": str(grade_path),
        "task_id": str(task["task_id"]),
        "run_id": str(run_id),
        "condition": condition,
    }
    before = _tree_digest(worktree)
    started = time.perf_counter()
    agent_result = subprocess.run(
        _expand_argv(adapter["argv"], values),
        cwd=worktree,
        capture_output=True,
        text=True,
        timeout=int(adapter["timeout_seconds"]),
        check=False,
    )
    duration = time.perf_counter() - started
    if agent_result.returncode != 0:
        raise RuntimeError(
            f"Agent failed for {task['task_id']}/{condition}: "
            f"{agent_result.stderr[-2000:]}"
        )
    trace = _read_fields(trace_path, TRACE_FIELDS, "agent trace")
    _require_nonnegative_integers(trace, TRACE_FIELDS, "agent trace")
    changed = _changed_files(before, _tree_digest(worktree))
    grader_result = subprocess.run(
        _expand_argv(grader_argv, values),
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    if grader_result.returncode != 0:
        raise RuntimeError(
            f"Grader failed for {task['task_id']}/{condition}: "
            f"{grader_result.stderr[-2000:]}"
        )
    if not grade_path.is_file() and grader_result.stdout.strip():
        grade_path.write_text(grader_result.stdout, encoding="utf-8")
    grade = _read_fields(grade_path, GRADE_FIELDS, "grader result")
    for field in ("hidden_contract_pass", "architecture_pass"):
        if not isinstance(grade[field], bool):
            raise ValueError(f"grader result {field} must be boolean")
    _require_nonnegative_integers(
        grade,
        GRADE_FIELDS - {"hidden_contract_pass", "architecture_pass"},
        "grader result",
    )
    if grade["owner_rank"] < 1:
        raise ValueError("grader result owner_rank must be at least 1")
    record = {
        "task_id": str(task["task_id"]),
        "task_class": str(task["task_class"]),
        "run_id": str(run_id),
        "agent_system_id": str(adapter["agent_system_id"]),
        "environment_id": str(adapter.get("environment_id", "unspecified")),
        "condition_order_index": condition_order_index,
        "trajectory_ref": (
            f"evaluation://{task['task_id']}/{run_id}/{condition}/trace"
        ),
        "hidden_contract_pass": bool(grade["hidden_contract_pass"]),
        "architecture_pass": bool(grade["architecture_pass"]),
        "success": bool(grade["hidden_contract_pass"] and grade["architecture_pass"]),
        "duration_seconds": round(duration, 6),
        "input_tokens": int(trace["input_tokens"]),
        "output_tokens": int(trace["output_tokens"]),
        "tool_calls": int(trace["tool_calls"]),
        "files_opened": int(trace["files_opened"]),
        "irrelevant_files_opened": int(grade["irrelevant_files_opened"]),
        "checks_run": int(trace["checks_run"]),
        "irrelevant_checks_run": int(grade["irrelevant_checks_run"]),
        "owner_rank": int(grade["owner_rank"]),
        "invalid_edits": int(grade["invalid_edits"]),
        "regressions": int(grade["regressions"]),
        "changed_files": changed,
    }
    return record


def _run_trial(
    *,
    task: dict[str, Any],
    run_id: int,
    condition: str,
    condition_order_index: int,
    source_root: Path,
    adapter: dict[str, Any],
    grader_argv: list[str],
    workspace_parent: Path,
    keep_workspace: bool,
) -> dict[str, Any]:
    workspace = Path(
        tempfile.mkdtemp(
            prefix=f"mirai-agent-eval-{task['task_id']}-{condition}-",
            dir=workspace_parent,
        )
    )
    try:
        return _execute_trial_at_workspace(
            task=task,
            run_id=run_id,
            condition=condition,
            condition_order_index=condition_order_index,
            source_root=source_root,
            adapter=adapter,
            grader_argv=grader_argv,
            workspace=workspace,
            keep_workspace=keep_workspace,
        )
    finally:
        if not keep_workspace and workspace.exists():
            shutil.rmtree(workspace)


def run_evaluation(
    *,
    tasks: list[dict[str, Any]],
    source_roots: dict[str, Path],
    adapters: dict[str, dict[str, Any]],
    graders: dict[str, list[str]],
    repetitions: int,
    workspace_parent: Path,
    keep_workspaces: bool,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if repetitions < 3:
        raise ValueError("At least three repetitions per task are required")
    if adapters["baseline"]["agent_system_id"] == adapters["candidate"]["agent_system_id"]:
        raise ValueError("Baseline and candidate must identify different agent systems")
    baseline_environment = adapters["baseline"].get("environment_id", "unspecified")
    candidate_environment = adapters["candidate"].get("environment_id", "unspecified")
    if baseline_environment != candidate_environment:
        raise ValueError("Baseline and candidate must use the same environment_id")
    rng = random.Random(seed)
    records = {"baseline": [], "candidate": []}
    for task in tasks:
        grader_id = str(task["grader_id"])
        if grader_id not in graders:
            raise ValueError(f"Unknown grader_id {grader_id!r}")
        for run_id in range(repetitions):
            order = ["baseline", "candidate"]
            rng.shuffle(order)
            for order_index, condition in enumerate(order):
                records[condition].append(
                    _run_trial(
                        task=task,
                        run_id=run_id,
                        condition=condition,
                        condition_order_index=order_index,
                        source_root=source_roots[condition],
                        adapter=adapters[condition],
                        grader_argv=graders[grader_id],
                        workspace_parent=workspace_parent,
                        keep_workspace=keep_workspaces,
                    )
                )
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--baseline-source", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--baseline-adapter", required=True)
    parser.add_argument("--candidate-adapter", required=True)
    parser.add_argument("--graders", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workspace-root")
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    workspace_parent = (
        Path(args.workspace_root)
        if args.workspace_root
        else Path(tempfile.gettempdir())
    )
    workspace_parent.mkdir(parents=True, exist_ok=True)
    records = run_evaluation(
        tasks=load_tasks(Path(args.tasks)),
        source_roots={
            "baseline": Path(args.baseline_source),
            "candidate": Path(args.candidate_source),
        },
        adapters={
            "baseline": load_adapter(Path(args.baseline_adapter)),
            "candidate": load_adapter(Path(args.candidate_adapter)),
        },
        graders=load_graders(Path(args.graders)),
        repetitions=args.repetitions,
        workspace_parent=workspace_parent,
        keep_workspaces=args.keep_workspaces,
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "baseline.jsonl", records["baseline"])
    _write_jsonl(out_dir / "candidate.jsonl", records["candidate"])
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "baseline_records": len(records["baseline"]),
                "candidate_records": len(records["candidate"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
