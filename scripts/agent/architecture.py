"""Validate executable dependency boundaries from agent/architecture.json."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_PATH = ROOT / "agent" / "architecture.json"


@dataclass(frozen=True)
class DependencyViolation:
    rule: str
    path: str
    line: int
    imported_module: str

    def render(self) -> str:
        return (
            f"{self.path}:{self.line}: dependency rule '{self.rule}' forbids "
            f"importing '{self.imported_module}'"
        )


def _normalized_path(path: Path, *, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_excluded(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _resolve_relative_import(path: Path, *, root: Path, node: ast.ImportFrom) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    package = parts[:-1] if parts[-1] != "__init__" else parts
    keep = len(package) - max(0, int(node.level) - 1)
    prefix = package[: max(0, keep)]
    if node.module:
        prefix.extend(str(node.module).split("."))
    return ".".join(prefix)


def _imports(path: Path, *, root: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((int(node.lineno), alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = (
                _resolve_relative_import(path, root=root, node=node)
                if node.level
                else str(node.module or "")
            )
            if module:
                imports.append((int(node.lineno), module))
    return imports


def _matches_prefix(module: str, prefixes: Iterable[str]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


def scan_dependency_rules(
    *,
    root: Path = ROOT,
    architecture_path: Path = ARCHITECTURE_PATH,
) -> tuple[DependencyViolation, ...]:
    payload: dict[str, Any] = json.loads(architecture_path.read_text(encoding="utf-8"))
    rules = payload.get("dependency_rules", ())
    if not isinstance(rules, list) or not rules:
        raise ValueError("agent/architecture.json must declare dependency_rules")
    violations: list[DependencyViolation] = []
    for rule in rules:
        name = str(rule.get("name", "")).strip()
        source_roots = tuple(str(value) for value in rule.get("source_roots", ()))
        target_prefixes = tuple(str(value) for value in rule.get("target_prefixes", ()))
        exclude_globs = tuple(str(value) for value in rule.get("exclude_globs", ()))
        if not name or not source_roots or not target_prefixes:
            raise ValueError(f"Invalid dependency rule: {rule!r}")
        for source_root in source_roots:
            directory = root / source_root
            if not directory.is_dir():
                raise ValueError(
                    f"Dependency rule '{name}' source root does not exist: {source_root}"
                )
            for path in sorted(directory.rglob("*.py")):
                normalized = _normalized_path(path, root=root)
                if _is_excluded(normalized, exclude_globs):
                    continue
                for line, module in _imports(path, root=root):
                    if _matches_prefix(module, target_prefixes):
                        violations.append(
                            DependencyViolation(
                                rule=name,
                                path=normalized,
                                line=line,
                                imported_module=module,
                            )
                        )
    return tuple(sorted(violations, key=lambda item: (item.path, item.line, item.rule)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Mirai's executable dependency boundaries."
    )
    parser.parse_args(argv)
    violations = scan_dependency_rules()
    if violations:
        for violation in violations:
            print(violation.render())
        return 1
    print("Architecture dependency boundaries passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
