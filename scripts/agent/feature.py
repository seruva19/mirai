"""Inspect, create, and validate Mirai feature extensions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "agent" / "features.json"

sys.path.insert(0, str(ROOT))

from mirai.core.features import EXTENSION_KITS  # noqa: E402
from mirai.core.features import FeatureDescriptor  # noqa: E402
from mirai.core.features import FeatureInvariant  # noqa: E402
from mirai.core.features import FeatureKind  # noqa: E402
from mirai.core.features import load_feature_catalog  # noqa: E402
from mirai.core.features import validate_feature_catalog  # noqa: E402


def _descriptor_payload(descriptor: FeatureDescriptor) -> dict[str, Any]:
    payload = asdict(descriptor)
    payload["kind"] = descriptor.kind.value
    payload["invariants"] = [value.value for value in descriptor.invariants]
    for field in ("symbols", "config_keys", "contract_paths", "readme_claims"):
        payload[field] = list(payload[field])
    payload["disabled_config"] = dict(descriptor.disabled_config)
    return payload


def _catalog_payload(descriptors: list[FeatureDescriptor]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "features": [
            _descriptor_payload(descriptor)
            for descriptor in sorted(descriptors, key=lambda item: item.name)
        ],
    }


def _write_catalog(path: Path, descriptors: list[FeatureDescriptor]) -> None:
    path.write_text(
        json.dumps(_catalog_payload(descriptors), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _policy_class_name(name: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", name) if part]
    return "".join(part[:1].upper() + part[1:] for part in parts) + "Policy"


def _default_symbol(name: str, kind: FeatureKind) -> str:
    base = _policy_class_name(name).removesuffix("Policy")
    return (
        base + "ModelFamilyProvider"
        if kind is FeatureKind.MODEL_FAMILY
        else base + "Policy"
    )


def _owner_template(descriptor: FeatureDescriptor) -> str:
    class_name = descriptor.symbols[0]
    if descriptor.kind is FeatureKind.MODEL_FAMILY:
        return (
            f'"""Native sparse-MoE provider for {descriptor.name}."""\n\n'
            "from __future__ import annotations\n\n"
            "from mirai.core.models.providers import ModelFamilyProvider\n"
            "from mirai.core.models.providers import register_model_family_provider\n\n\n"
            f"class {class_name}(ModelFamilyProvider):\n"
            '    """Family capability declaration."""\n\n'
            "    def __init__(self) -> None:\n"
            f'        super().__init__("{descriptor.name}", native=True, sparse_moe=True)\n\n\n'
            f'PROVIDER = register_model_family_provider("{descriptor.name}", {class_name}())\n'
        )
    return (
        f'"""Typed owner for the {descriptor.name} feature."""\n\n'
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        f"class {class_name}:\n"
        '    """Default-preserving feature policy."""\n\n'
        "    enabled: bool = False\n\n"
        "    def requests_runtime_behavior(self) -> bool:\n"
        "        return self.enabled\n"
    )


def create_feature(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
    catalog_path: Path = CATALOG_PATH,
) -> dict[str, Any]:
    descriptors = list(load_feature_catalog(catalog_path))
    if any(item.name == args.name for item in descriptors):
        raise ValueError(f"Feature {args.name!r} already exists")
    kind = FeatureKind(args.kind)
    kit = EXTENSION_KITS[kind]
    invariants = list(kit.required_invariants)
    if not args.stateful and FeatureInvariant.PERSISTENCE in invariants:
        invariants.remove(FeatureInvariant.PERSISTENCE)
    if not args.optimized and FeatureInvariant.REFERENCE_PARITY in invariants:
        invariants.remove(FeatureInvariant.REFERENCE_PARITY)
    if FeatureInvariant.DOCUMENTATION not in invariants:
        invariants.append(FeatureInvariant.DOCUMENTATION)
    descriptor = FeatureDescriptor(
        name=args.name,
        kind=kind,
        owner=args.owner,
        check=args.check,
        invariants=tuple(invariants),
        symbols=tuple(args.symbol or (_default_symbol(args.name, kind),)),
        config_keys=tuple(args.config_key),
        disabled_config=tuple(
            (key, json.loads(value))
            for key, value in (
                item.split("=", 1) for item in args.disabled_config
            )
        ),
        contract_paths=tuple(args.contract),
        default_off=not args.default_on,
        stateful=args.stateful,
        optimized=args.optimized,
        reference_path=args.reference_path,
        readme_claim=args.readme_claim,
        provenance=args.provenance,
        promotion_evidence=getattr(args, "promotion_evidence", ""),
    )
    owner_path = root / descriptor.owner
    if args.create_owner:
        if owner_path.exists():
            raise ValueError(f"Owner already exists: {descriptor.owner}")
        owner_path.parent.mkdir(parents=True, exist_ok=True)
        owner_path.write_text(_owner_template(descriptor), encoding="utf-8")
    descriptors.append(descriptor)
    _write_catalog(catalog_path, descriptors)
    issues = validate_feature_catalog(
        root=root,
        catalog=descriptors,
        checks_path=root / "agent" / "checks.json",
    )
    feature_issues = [asdict(issue) for issue in issues if issue.feature == args.name]
    return {
        "status": "created" if not feature_issues else "incomplete",
        "feature": _descriptor_payload(descriptor),
        "required_steps": list(kit.required_steps),
        "issues": feature_issues,
    }


def inspect_features(name: str | None) -> dict[str, Any]:
    descriptors = list(load_feature_catalog(CATALOG_PATH))
    if name:
        descriptors = [item for item in descriptors if item.name == name]
        if not descriptors:
            raise ValueError(f"Unknown feature {name!r}")
    return {
        "status": "inspected",
        "features": [
            {
                **_descriptor_payload(descriptor),
                "required_steps": list(EXTENSION_KITS[descriptor.kind].required_steps),
            }
            for descriptor in descriptors
        ],
    }


def validate_features(name: str | None) -> dict[str, Any]:
    all_descriptors = list(load_feature_catalog(CATALOG_PATH))
    descriptors = all_descriptors
    if name:
        descriptors = [item for item in all_descriptors if item.name == name]
        if not descriptors:
            raise ValueError(f"Unknown feature {name!r}")
    issues = validate_feature_catalog(
        root=ROOT,
        catalog=all_descriptors,
        checks_path=ROOT / "agent" / "checks.json",
    )
    if name:
        claims = set(descriptors[0].public_claims)
        issues = tuple(
            issue
            for issue in issues
            if issue.feature == name
            or (issue.feature == "README" and issue.message in claims)
        )
    return {
        "status": "passed" if not issues else "failed",
        "features": len(descriptors),
        "issues": [asdict(issue) for issue in issues],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("name", nargs="?")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("name", nargs="?")

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--kind", required=True, choices=[item.value for item in FeatureKind])
    create_parser.add_argument("--owner", required=True)
    create_parser.add_argument("--check", required=True)
    create_parser.add_argument("--config-key", action="append", default=[])
    create_parser.add_argument("--symbol", action="append", default=[])
    create_parser.add_argument(
        "--disabled-config",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
    )
    create_parser.add_argument("--contract", action="append", default=[])
    create_parser.add_argument("--stateful", action="store_true")
    create_parser.add_argument("--optimized", action="store_true")
    create_parser.add_argument("--reference-path", default="")
    create_parser.add_argument("--readme-claim", default="")
    create_parser.add_argument("--provenance", default="")
    create_parser.add_argument("--promotion-evidence", default="")
    create_parser.add_argument("--default-on", action="store_true")
    create_parser.add_argument("--create-owner", action="store_true")
    args = parser.parse_args()

    try:
        if args.command == "inspect":
            payload = inspect_features(args.name)
        elif args.command == "validate":
            payload = validate_features(args.name)
        else:
            payload = create_feature(args)
    except (OSError, ValueError) as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"passed", "inspected", "created"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
