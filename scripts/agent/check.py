"""Select and execute Mirai validation contracts for an agent change."""

from __future__ import annotations

import argparse
import fnmatch
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "agent" / "checks.json"
FEATURE_CATALOG_PATH = ROOT / "agent" / "features.json"
OUTPUT_TAIL_CHARS = 6000
COST_ORDER = {"fast": 0, "extended": 1, "gpu": 2}


def environment_fingerprint() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("torch", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _normalized_path(raw: str | Path) -> str:
    value = str(raw).replace("\\", "/").strip()
    while value.startswith("./"):
        value = value[2:]
    return value


def _path_matches(changed_path: str, owned_path: str) -> bool:
    changed = _normalized_path(changed_path)
    owned = _normalized_path(owned_path)
    if any(token in owned for token in ("*", "?", "[")):
        return fnmatch.fnmatchcase(changed, owned)
    if owned.endswith("/"):
        return changed.startswith(owned)
    return changed == owned


def _command_paths(check: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for command in check["commands"]:
        for token in shlex.split(command, posix=True):
            normalized = _normalized_path(token)
            if normalized.endswith(".py") and "/" in normalized:
                paths.add(normalized)
    return paths


@lru_cache(maxsize=1)
def _feature_owned_paths() -> dict[str, set[str]]:
    if not FEATURE_CATALOG_PATH.is_file():
        return {}
    payload = json.loads(FEATURE_CATALOG_PATH.read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for feature in payload.get("features", ()):
        check = str(feature["check"])
        result.setdefault(check, set()).update(
            {
                _normalized_path(feature["owner"]),
                *(
                    _normalized_path(path)
                    for path in feature.get("contract_paths", ())
                ),
            }
        )
    return result


def _check_matches_path(check: dict[str, Any], changed_path: str) -> bool:
    return any(
        _path_matches(changed_path, owned_path)
        for owned_path in [
            *check["paths"],
            *_command_paths(check),
            *_feature_owned_paths().get(check["name"], ()),
        ]
    )


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("agent/checks.json schema_version must be 1")
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("agent/checks.json must contain checks")
    budget = payload.get("contract_budget")
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("agent/checks.json must declare a positive contract_budget")
    if len(checks) > budget:
        raise ValueError(
            f"Validation contract budget exceeded: {len(checks)} > {budget}"
        )
    names: set[str] = set()
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("Each check must be an object")
        name = str(check.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"Invalid or duplicate check name: {name!r}")
        names.add(name)
        for field in ("paths", "commands", "invariants"):
            values = check.get(field)
            if not isinstance(values, list) or not values:
                raise ValueError(f"Check {name!r} must declare non-empty {field}")
        if not isinstance(check.get("remote_gpu"), bool):
            raise ValueError(f"Check {name!r} must declare boolean remote_gpu")
        if not isinstance(check.get("owns_paths", True), bool):
            raise ValueError(f"Check {name!r} owns_paths must be boolean")
        cost = check.get("cost")
        if cost not in COST_ORDER:
            raise ValueError(f"Check {name!r} has invalid cost: {cost!r}")
        if check["remote_gpu"] != (cost == "gpu"):
            raise ValueError(f"Check {name!r} remote_gpu and cost disagree")
    return payload


def select_checks(
    manifest: dict[str, Any],
    changed_paths: Iterable[str],
) -> list[dict[str, Any]]:
    normalized = sorted({_normalized_path(path) for path in changed_paths if str(path).strip()})
    return [
        check
        for check in manifest["checks"]
        if any(_check_matches_path(check, changed_path) for changed_path in normalized)
    ]


def select_local_checks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Select every CPU-local contract in manifest order."""

    return [
        check for check in manifest["checks"] if not bool(check["remote_gpu"])
    ]


def unowned_paths(
    manifest: dict[str, Any],
    changed_paths: Iterable[str],
) -> list[str]:
    normalized = sorted({_normalized_path(path) for path in changed_paths if str(path).strip()})
    return [
        path
        for path in normalized
        if not any(
            check.get("owns_paths", True) and _check_matches_path(check, path)
            for check in manifest["checks"]
        )
    ]


def _git_changed_paths(ref: str) -> list[str]:
    tracked = subprocess.run(
        ["git", "diff", "--name-only", ref, "--"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise RuntimeError(tracked.stderr.strip() or f"git diff {ref!r} failed")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if untracked.returncode != 0:
        raise RuntimeError(untracked.stderr.strip() or "git ls-files failed")
    return sorted(
        {
            line
            for line in [*tracked.stdout.splitlines(), *untracked.stdout.splitlines()]
            if line.strip()
        }
    )


def _command_argv(command: str) -> list[str]:
    argv = shlex.split(command, posix=True)
    if not argv:
        raise ValueError("Empty validation command")
    if argv[0] in {"python", "python3"}:
        argv[0] = sys.executable
    return argv


def _run_command(command: str) -> dict[str, Any]:
    argv = _command_argv(command)
    started = time.perf_counter()
    result = subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "argv": argv,
        "returncode": int(result.returncode),
        "duration_seconds": round(time.perf_counter() - started, 6),
        "stdout_tail": result.stdout[-OUTPUT_TAIL_CHARS:],
        "stderr_tail": result.stderr[-OUTPUT_TAIL_CHARS:],
    }


def execute_checks(
    checks: Iterable[dict[str, Any]],
    *,
    allow_remote_gpu: bool,
    max_cost: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for check in checks:
        if COST_ORDER[check["cost"]] > COST_ORDER[max_cost]:
            results.append(
                {
                    "name": check["name"],
                    "status": f"{check['cost']}_required",
                    "cost": check["cost"],
                    "invariants": check["invariants"],
                    "commands": [],
                }
            )
            continue
        if check["remote_gpu"] and not allow_remote_gpu:
            results.append(
                {
                    "name": check["name"],
                    "status": "remote_gpu_required",
                    "cost": check["cost"],
                    "invariants": check["invariants"],
                    "commands": [],
                }
            )
            continue
        command_results = [_run_command(command) for command in check["commands"]]
        results.append(
            {
                "name": check["name"],
                "status": (
                    "passed"
                    if all(item["returncode"] == 0 for item in command_results)
                    else "failed"
                ),
                "cost": check["cost"],
                "invariants": check["invariants"],
                "commands": command_results,
            }
        )
    statuses = {result["status"] for result in results}
    aggregate_status = (
        "failed"
        if "failed" in statuses
        else "incomplete"
        if any(status.endswith("_required") for status in statuses)
        else "passed"
    )
    return {
        "schema_version": 1,
        "status": aggregate_status,
        "environment": environment_fingerprint(),
        "checks": results,
    }


def _selection_payload(
    checks: Iterable[dict[str, Any]],
    *,
    changed_paths: Iterable[str],
    unowned: Iterable[str],
) -> dict[str, Any]:
    uncovered = list(unowned)
    return {
        "schema_version": 1,
        "status": "unowned_change" if uncovered else "selected",
        "changed_paths": sorted({_normalized_path(path) for path in changed_paths}),
        "unowned_paths": uncovered,
        "checks": [
            {
                "name": check["name"],
                "remote_gpu": check["remote_gpu"],
                "cost": check["cost"],
                "invariants": check["invariants"],
                "commands": check["commands"],
            }
            for check in checks
        ],
    }


def self_check() -> dict[str, Any]:
    manifest = load_manifest()
    cases = manifest.get("routing_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("agent/checks.json must declare routing_cases")
    failures: list[str] = []
    selected_counts: list[int] = []
    for case in cases:
        changed_path = str(case["path"])
        expected = set(case["checks"])
        observed = {
            check["name"] for check in select_checks(manifest, [changed_path])
        }
        selected_counts.append(len(observed))
        if observed != expected:
            failures.append(
                f"{changed_path}: expected {sorted(expected)}, observed {sorted(observed)}"
            )
    unknown = "mirai/core/unknown_seam.py"
    if unowned_paths(manifest, [unknown]) != [unknown]:
        failures.append("unknown paths must fail closed")
    total_checks = len(manifest["checks"])
    irrelevant_reduction = (
        1.0 - (sum(selected_counts) / (len(selected_counts) * total_checks))
    )
    if irrelevant_reduction < 0.8:
        failures.append(
            f"routing benchmark irrelevant-check reduction {irrelevant_reduction:.3f} < 0.8"
        )
    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "cases": len(cases),
        "check_recall": 1.0 if not failures else 0.0,
        "mean_irrelevant_check_reduction": irrelevant_reduction,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed", nargs="*", default=[])
    parser.add_argument("--changed-from")
    parser.add_argument("--check", action="append", default=[])
    parser.add_argument(
        "--all-local",
        action="store_true",
        help="Select every non-GPU contract declared by the manifest.",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-remote-gpu", action="store_true")
    parser.add_argument(
        "--max-cost",
        choices=tuple(COST_ORDER),
        default="fast",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    manifest = load_manifest()
    if args.self_check:
        payload = self_check()
    elif args.list:
        payload = {
            "schema_version": 1,
            "status": "listed",
            "checks": manifest["checks"],
        }
    else:
        changed_paths = list(args.changed)
        if args.changed_from:
            changed_paths.extend(_git_changed_paths(args.changed_from))
        requested = set(args.check)
        unknown = requested - {check["name"] for check in manifest["checks"]}
        if unknown:
            parser.error(f"unknown checks: {sorted(unknown)}")
        checks = select_local_checks(manifest) if args.all_local else []
        checks.extend(
            check
            for check in manifest["checks"]
            if check["name"] in requested and check not in checks
        )
        selected_names = {check["name"] for check in checks}
        for check in select_checks(manifest, changed_paths):
            if check["name"] not in selected_names:
                checks.append(check)
                selected_names.add(check["name"])
        if not checks:
            parser.error("no checks selected; pass --changed, --changed-from, or --check")
        uncovered = unowned_paths(manifest, changed_paths)
        payload = (
            execute_checks(
                checks,
                allow_remote_gpu=args.allow_remote_gpu,
                max_cost=args.max_cost,
            )
            if args.run and not uncovered
            else _selection_payload(
                checks,
                changed_paths=changed_paths,
                unowned=uncovered,
            )
        )

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload["status"] in {"passed", "selected", "listed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
