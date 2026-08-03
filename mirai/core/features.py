"""Typed feature ownership and reusable extension-contract primitives."""

from __future__ import annotations

import ast
from dataclasses import MISSING, dataclass, fields
from enum import Enum
import json
import math
from pathlib import Path
import shlex
from typing import Any, Callable, Iterable, Mapping, TypeVar


class FeatureKind(str, Enum):
    GENERIC = "generic"
    MOE_POLICY = "moe_policy"
    ROUTING_OBJECTIVE = "routing_objective"
    COMPRESSED_WEIGHT = "compressed_weight"
    RESIDENCY = "residency"
    ADAPTER = "adapter"
    TRAINING_POLICY = "training_policy"
    MODEL_FAMILY = "model_family"
    INFERENCE = "inference"
    ARTIFACT_TRANSFORM = "artifact_transform"
    TOOLING = "tooling"
    MONITORING = "monitoring"
    RUNTIME_GUARD = "runtime_guard"
    ROUTING_TOPOLOGY = "routing_topology"


class FeatureInvariant(str, Enum):
    OWNER = "owner"
    CONFIG = "config"
    DEFAULT_PATH = "default_path"
    REFERENCE_PARITY = "reference_parity"
    GRADIENTS = "gradients"
    PERSISTENCE = "persistence"
    DOCUMENTATION = "documentation"
    NATIVE_EXECUTION = "native_execution"
    RESOURCE_BOUND = "resource_bound"
    HELD_OUT_EVIDENCE = "held_out_evidence"


@dataclass(frozen=True)
class ExtensionKit:
    kind: FeatureKind
    required_invariants: tuple[FeatureInvariant, ...]
    required_steps: tuple[str, ...]


EXTENSION_KITS: dict[FeatureKind, ExtensionKit] = {
    FeatureKind.GENERIC: ExtensionKit(
        FeatureKind.GENERIC,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.DOCUMENTATION,
        ),
        (
            "declare one searchable behavior owner",
            "attach every public claim to an executable contract",
            "keep model-family behavior behind provider capabilities",
        ),
    ),
    FeatureKind.MOE_POLICY: ExtensionKit(
        FeatureKind.MOE_POLICY,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.GRADIENTS,
            FeatureInvariant.PERSISTENCE,
        ),
        (
            "declare a typed model-agnostic policy",
            "gate it through default-preserving config",
            "integrate through router or provider capability contracts",
            "prove disabled routing and gradients are unchanged",
            "round-trip policy state when stateful",
        ),
    ),
    FeatureKind.ROUTING_OBJECTIVE: ExtensionKit(
        FeatureKind.ROUTING_OBJECTIVE,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.REFERENCE_PARITY,
            FeatureInvariant.GRADIENTS,
        ),
        (
            "declare exact objective math and reduction",
            "retain a deterministic reference implementation",
            "prove loss, input gradients, and adapter gradients",
        ),
    ),
    FeatureKind.COMPRESSED_WEIGHT: ExtensionKit(
        FeatureKind.COMPRESSED_WEIGHT,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.REFERENCE_PARITY,
            FeatureInvariant.GRADIENTS,
            FeatureInvariant.PERSISTENCE,
            FeatureInvariant.RESOURCE_BOUND,
        ),
        (
            "declare packed format and lineage",
            "retain dequantized reference math",
            "prove output and trainable-gradient parity",
            "prove frozen BF16 weights are absent from autograd",
        ),
    ),
    FeatureKind.RESIDENCY: ExtensionKit(
        FeatureKind.RESIDENCY,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.REFERENCE_PARITY,
            FeatureInvariant.RESOURCE_BOUND,
        ),
        (
            "declare exclusive tensor ownership transitions",
            "bound host and device residency",
            "prove offload preserves outputs and trainable gradients",
        ),
    ),
    FeatureKind.ADAPTER: ExtensionKit(
        FeatureKind.ADAPTER,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.GRADIENTS,
            FeatureInvariant.PERSISTENCE,
        ),
        (
            "declare typed adapter host requirements",
            "prove only intended parameters train",
            "round-trip adapter lineage and state",
        ),
    ),
    FeatureKind.TRAINING_POLICY: ExtensionKit(
        FeatureKind.TRAINING_POLICY,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.PERSISTENCE,
        ),
        (
            "implement TrainingPolicyPlugin",
            "keep construction absent when disabled",
            "round-trip deterministic policy state",
        ),
    ),
    FeatureKind.MODEL_FAMILY: ExtensionKit(
        FeatureKind.MODEL_FAMILY,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.GRADIENTS,
            FeatureInvariant.PERSISTENCE,
            FeatureInvariant.NATIVE_EXECUTION,
        ),
        (
            "implement and register ModelFamilyProvider",
            "keep tensor, prompt, latent, and VAE semantics family-owned",
            "declare sparse-MoE and memory capabilities",
            "prove one native train step and native inference",
        ),
    ),
    FeatureKind.INFERENCE: ExtensionKit(
        FeatureKind.INFERENCE,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.REFERENCE_PARITY,
            FeatureInvariant.NATIVE_EXECUTION,
        ),
        (
            "declare an inference policy contract",
            "retain deterministic reference recurrence",
            "prove conditioning order and native execution",
        ),
    ),
    FeatureKind.ARTIFACT_TRANSFORM: ExtensionKit(
        FeatureKind.ARTIFACT_TRANSFORM,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.PERSISTENCE,
            FeatureInvariant.RESOURCE_BOUND,
        ),
        (
            "declare an explicit offline config gate",
            "validate source artifact lineage and topology",
            "write a versioned output artifact without mutating the source",
            "bound calibration and transformation residency",
        ),
    ),
    FeatureKind.TOOLING: ExtensionKit(
        FeatureKind.TOOLING,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.DOCUMENTATION,
            FeatureInvariant.RESOURCE_BOUND,
        ),
        (
            "declare a typed model-agnostic input and output contract",
            "remain independent of private model-family module layouts",
            "prove deterministic bounded behavior",
        ),
    ),
    FeatureKind.MONITORING: ExtensionKit(
        FeatureKind.MONITORING,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.RESOURCE_BOUND,
            FeatureInvariant.DOCUMENTATION,
        ),
        (
            "declare bounded detached observations",
            "keep monitoring absent when disabled",
            "preserve loss, gradients, and model state",
            "round-trip only persistent monitoring state",
        ),
    ),
    FeatureKind.RUNTIME_GUARD: ExtensionKit(
        FeatureKind.RUNTIME_GUARD,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.DOCUMENTATION,
        ),
        (
            "consume a typed runtime observation",
            "remain inert in the disabled mode",
            "fail or warn before the guarded irreversible action",
        ),
    ),
    FeatureKind.ROUTING_TOPOLOGY: ExtensionKit(
        FeatureKind.ROUTING_TOPOLOGY,
        (
            FeatureInvariant.OWNER,
            FeatureInvariant.CONFIG,
            FeatureInvariant.DEFAULT_PATH,
            FeatureInvariant.GRADIENTS,
            FeatureInvariant.PERSISTENCE,
            FeatureInvariant.HELD_OUT_EVIDENCE,
        ),
        (
            "declare a typed model-agnostic routing-topology policy",
            "keep construction and behavior absent under the disabled config",
            "prove disabled routing and gradients are unchanged",
            "round-trip topology state when stateful",
            "keep the feature default-off until paired held-out A/B quality "
            "evidence is registered",
        ),
    ),
}


@dataclass(frozen=True)
class FeatureDescriptor:
    name: str
    kind: FeatureKind
    owner: str
    check: str
    invariants: tuple[FeatureInvariant, ...]
    symbols: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()
    disabled_config: tuple[tuple[str, Any], ...] = ()
    contract_paths: tuple[str, ...] = ()
    default_off: bool = True
    stateful: bool = False
    optimized: bool = False
    reference_path: str = ""
    readme_claim: str = ""
    readme_claims: tuple[str, ...] = ()
    provenance: str = ""
    promotion_evidence: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FeatureDescriptor:
        return cls(
            name=str(payload["name"]).strip(),
            kind=FeatureKind(str(payload["kind"])),
            owner=str(payload["owner"]).replace("\\", "/"),
            check=str(payload["check"]).strip(),
            invariants=tuple(
                FeatureInvariant(str(value)) for value in payload["invariants"]
            ),
            symbols=tuple(str(value) for value in payload.get("symbols", ())),
            config_keys=tuple(str(value) for value in payload.get("config_keys", ())),
            disabled_config=tuple(
                (str(key), value)
                for key, value in payload.get("disabled_config", {}).items()
            ),
            contract_paths=tuple(
                str(value).replace("\\", "/")
                for value in payload.get("contract_paths", ())
            ),
            default_off=bool(payload.get("default_off", True)),
            stateful=bool(payload.get("stateful", False)),
            optimized=bool(payload.get("optimized", False)),
            reference_path=str(payload.get("reference_path", "")).replace("\\", "/"),
            readme_claim=str(payload.get("readme_claim", "")),
            readme_claims=tuple(
                str(value) for value in payload.get("readme_claims", ())
            ),
            provenance=str(payload.get("provenance", "")),
            promotion_evidence=str(payload.get("promotion_evidence", "")).replace(
                "\\", "/"
            ),
        )

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip().lower():
            raise ValueError("FeatureDescriptor.name must be non-empty lowercase text")
        if not self.owner.endswith(".py"):
            raise ValueError(f"Feature {self.name!r} owner must be a Python module")
        if len(set(self.invariants)) != len(self.invariants):
            raise ValueError(f"Feature {self.name!r} repeats an invariant")
        claims = tuple(value for value in (self.readme_claim, *self.readme_claims) if value)
        if len(claims) != len(set(claims)):
            raise ValueError(f"Feature {self.name!r} repeats a README claim")

    @property
    def public_claims(self) -> tuple[str, ...]:
        return tuple(
            value for value in (self.readme_claim, *self.readme_claims) if value
        )


@dataclass(frozen=True)
class FeatureValidationIssue:
    feature: str
    code: str
    message: str


class QualityDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class PairedQualityObservation:
    pair_id: str
    baseline: float
    candidate: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PairedQualityObservation:
        observation = cls(
            pair_id=str(payload["pair_id"]).strip(),
            baseline=float(payload["baseline"]),
            candidate=float(payload["candidate"]),
        )
        if not observation.pair_id:
            raise ValueError("paired quality observation requires pair_id")
        if not math.isfinite(observation.baseline) or not math.isfinite(
            observation.candidate
        ):
            raise ValueError("paired quality observations must be finite")
        return observation


@dataclass(frozen=True)
class RoutingTopologyPromotionEvidence:
    """Paired held-out quality evidence required before topology promotion.

    Routing-health telemetry is deliberately insufficient: materially different
    MoE routing topologies can converge to equivalent quality (arXiv:2604.14419).
    """

    feature: str
    metric: str
    direction: QualityDirection
    held_out_split_fingerprint: str
    baseline_artifact_fingerprint: str
    candidate_artifact_fingerprint: str
    pairs: tuple[PairedQualityObservation, ...]

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> RoutingTopologyPromotionEvidence:
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("routing-topology evidence schema_version must be 1")
        evidence = cls(
            feature=str(payload["feature"]).strip(),
            metric=str(payload["metric"]).strip(),
            direction=QualityDirection(str(payload["direction"])),
            held_out_split_fingerprint=str(
                payload["held_out_split_fingerprint"]
            ).strip(),
            baseline_artifact_fingerprint=str(
                payload["baseline_artifact_fingerprint"]
            ).strip(),
            candidate_artifact_fingerprint=str(
                payload["candidate_artifact_fingerprint"]
            ).strip(),
            pairs=tuple(
                PairedQualityObservation.from_mapping(item)
                for item in payload.get("pairs", ())
            ),
        )
        for name in (
            "feature",
            "metric",
            "held_out_split_fingerprint",
            "baseline_artifact_fingerprint",
            "candidate_artifact_fingerprint",
        ):
            if not getattr(evidence, name):
                raise ValueError(f"routing-topology evidence requires {name}")
        if not evidence.pairs:
            raise ValueError("routing-topology evidence requires paired observations")
        pair_ids = tuple(item.pair_id for item in evidence.pairs)
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("routing-topology evidence pair_id values must be unique")
        return evidence

    @property
    def mean_quality_delta(self) -> float:
        sign = 1.0 if self.direction is QualityDirection.HIGHER_IS_BETTER else -1.0
        return sign * sum(
            item.candidate - item.baseline for item in self.pairs
        ) / len(self.pairs)


def load_routing_topology_evidence(
    path: Path,
) -> RoutingTopologyPromotionEvidence:
    return RoutingTopologyPromotionEvidence.from_mapping(
        json.loads(path.read_text(encoding="utf-8"))
    )


def validate_routing_topology_promotion(
    *,
    root: Path,
    feature: FeatureDescriptor,
) -> tuple[FeatureValidationIssue, ...]:
    """Fail closed when a routing topology leaves default-off without A/B evidence."""
    if feature.kind is not FeatureKind.ROUTING_TOPOLOGY:
        return ()
    evidence_ref = feature.promotion_evidence.replace("\\", "/")
    if not evidence_ref:
        if feature.default_off:
            return ()
        return (
            FeatureValidationIssue(
                feature.name,
                "missing_promotion_evidence",
                "default-on routing topology requires paired held-out A/B evidence",
            ),
        )
    evidence_path = Path(evidence_ref)
    if (
        evidence_path.is_absolute()
        or ".." in evidence_path.parts
        or not evidence_ref.startswith("agent/evidence/")
        or evidence_path.suffix.lower() != ".json"
    ):
        return (
            FeatureValidationIssue(
                feature.name,
                "invalid_promotion_evidence_path",
                evidence_ref,
            ),
        )
    resolved = root / evidence_path
    if not resolved.is_file():
        return (
            FeatureValidationIssue(
                feature.name,
                "missing_promotion_evidence",
                evidence_ref,
            ),
        )
    try:
        evidence = load_routing_topology_evidence(resolved)
    except (KeyError, TypeError, ValueError) as exc:
        return (
            FeatureValidationIssue(
                feature.name,
                "invalid_promotion_evidence",
                str(exc),
            ),
        )
    if evidence.feature != feature.name:
        return (
            FeatureValidationIssue(
                feature.name,
                "promotion_evidence_feature_mismatch",
                evidence.feature,
            ),
        )
    if (
        evidence.baseline_artifact_fingerprint
        == evidence.candidate_artifact_fingerprint
    ):
        return (
            FeatureValidationIssue(
                feature.name,
                "promotion_evidence_identical_artifacts",
                evidence.baseline_artifact_fingerprint,
            ),
        )
    if evidence.mean_quality_delta < 0.0:
        return (
            FeatureValidationIssue(
                feature.name,
                "quality_non_regression_failed",
                f"mean_quality_delta={evidence.mean_quality_delta}",
            ),
        )
    return ()


def load_feature_catalog(path: Path) -> tuple[FeatureDescriptor, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("feature catalog schema_version must be 1")
    features = tuple(
        FeatureDescriptor.from_mapping(item) for item in payload.get("features", ())
    )
    names = [feature.name for feature in features]
    if not features or len(names) != len(set(names)):
        raise ValueError("feature catalog must contain uniquely named features")
    return features


def _known_config_keys() -> set[str]:
    from mirai.config.schema import all_config_keys

    return {
        f"{section}.{key}" if section else key
        for section, keys in all_config_keys().items()
        for key in keys
    }


def _config_defaults() -> dict[str, Any]:
    from mirai.config.schema import AdapterConfig
    from mirai.config.schema import ComplianceConfig
    from mirai.config.schema import DatasetConfig
    from mirai.config.schema import InferenceConfig
    from mirai.config.schema import LoggingConfig
    from mirai.config.schema import MemoryConfig
    from mirai.config.schema import ModelConfig
    from mirai.config.schema import ModelParams
    from mirai.config.schema import OptimizerConfig
    from mirai.config.schema import StrategyConfig
    from mirai.config.schema import TrainingSection

    sections = {
        "model": ModelConfig,
        "model.params": ModelParams,
        "strategy": StrategyConfig,
        "inference": InferenceConfig,
        "training": TrainingSection,
        "optimizer": OptimizerConfig,
        "adapter": AdapterConfig,
        "dataset": DatasetConfig,
        "logging": LoggingConfig,
        "compliance": ComplianceConfig,
        "memory": MemoryConfig,
    }
    defaults: dict[str, Any] = {}
    for section, dataclass_type in sections.items():
        for item in fields(dataclass_type):
            if item.default is not MISSING:
                value = item.default
            elif item.default_factory is not MISSING:
                value = item.default_factory()
            else:
                continue
            defaults[f"{section}.{item.name}"] = value
    return defaults


def readme_feature_claims(readme: str) -> tuple[str, ...]:
    """Return public Generic/MoE feature titles in documented order."""
    claims: list[str] = []
    active = False
    for line in readme.splitlines():
        if line in {"### Generic", "### MoE"}:
            active = True
            continue
        if active and line.startswith(("## ", "### ")):
            active = False
        if not active or not line.startswith("- "):
            continue
        if line.startswith("- **"):
            end = line.find("**", 4)
            if end < 0:
                raise ValueError(f"Malformed README feature bullet: {line}")
            claim = line[4:end].strip()
        else:
            claim = line[2:].strip()
        if not claim:
            raise ValueError("README feature titles must not be empty")
        claims.append(claim)
    if len(claims) != len(set(claims)):
        duplicates = sorted(
            claim for claim in set(claims) if claims.count(claim) > 1
        )
        raise ValueError(f"README feature titles must be unique: {duplicates}")
    return tuple(claims)


def model_page_feature_claims(document: str) -> tuple[str, ...]:
    """Return titles from one model page's model-specific feature section."""
    claims: list[str] = []
    active = False
    for line in document.splitlines():
        if line == "## Model-specific features":
            active = True
            continue
        if active and line.startswith("## "):
            active = False
        if not active or not line.startswith("- **"):
            continue
        end = line.find("**", 4)
        if end < 0:
            raise ValueError(f"Malformed model feature bullet: {line}")
        claim = line[4:end].strip()
        if not claim:
            raise ValueError("Model feature titles must not be empty")
        claims.append(claim)
    if len(claims) != len(set(claims)):
        raise ValueError("Model feature titles must be unique within a model page")
    return tuple(claims)


def _defined_python_symbols(path: Path) -> set[str]:
    """Return symbols actually bound by a Python owner module."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.ClassDef):
            symbols.add(node.name)
            symbols.update(
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
    return symbols


def validate_feature_catalog(
    *,
    root: Path,
    catalog: Iterable[FeatureDescriptor],
    checks_path: Path,
) -> tuple[FeatureValidationIssue, ...]:
    checks_payload = json.loads(checks_path.read_text(encoding="utf-8"))
    checks = {str(item["name"]): item for item in checks_payload["checks"]}
    known_config = _known_config_keys()
    config_defaults = _config_defaults()
    config_reference = (root / "CONFIG_REFERENCE.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    model_page_claims: list[str] = []
    model_support = root / "model_support"
    if model_support.is_dir():
        for path in sorted(model_support.glob("*.md")):
            document = path.read_text(encoding="utf-8")
            model_page_claims.extend(model_page_feature_claims(document))
    issues: list[FeatureValidationIssue] = []

    def add(feature: FeatureDescriptor, code: str, message: str) -> None:
        issues.append(FeatureValidationIssue(feature.name, code, message))

    readme_claims = readme_feature_claims(readme)
    public_claims = (*readme_claims, *model_page_claims)
    descriptor_claims = [
        claim for feature in catalog for claim in feature.public_claims
    ]
    for claim in public_claims:
        count = descriptor_claims.count(claim)
        if count == 0:
            issues.append(
                FeatureValidationIssue(
                    feature="README",
                    code="unregistered_readme_claim",
                    message=claim,
                )
            )
        elif count > 1:
            issues.append(
                FeatureValidationIssue(
                    feature="README",
                    code="duplicate_readme_claim",
                    message=claim,
                )
            )

    for feature in catalog:
        if not feature.public_claims:
            add(
                feature,
                "missing_readme_claim",
                "feature declares no public documentation claim",
            )
        kit = EXTENSION_KITS[feature.kind]
        missing_invariants = set(kit.required_invariants) - set(feature.invariants)
        if not feature.stateful:
            missing_invariants.discard(FeatureInvariant.PERSISTENCE)
        if not feature.optimized:
            missing_invariants.discard(FeatureInvariant.REFERENCE_PARITY)
        if missing_invariants:
            add(
                feature,
                "missing_invariant",
                "missing " + ", ".join(sorted(value.value for value in missing_invariants)),
            )
        owner_path = root / feature.owner
        if not owner_path.is_file():
            add(feature, "missing_owner", feature.owner)
        elif not feature.symbols:
            add(feature, "missing_symbol_contract", feature.owner)
        else:
            owner_symbols = _defined_python_symbols(owner_path)
            for symbol in feature.symbols:
                if symbol not in owner_symbols:
                    add(feature, "missing_owner_symbol", symbol)
        if FeatureInvariant.CONFIG in feature.invariants and not feature.config_keys:
            add(feature, "missing_config_gate", "feature declares no config key")
        if not feature.contract_paths:
            add(feature, "missing_contract", "feature declares no behavioral contract")
        check = checks.get(feature.check)
        if check is None:
            add(feature, "unknown_check", feature.check)
        else:
            command_paths = {
                token.replace("\\", "/")
                for command in check["commands"]
                for token in shlex.split(str(command), posix=True)
                if token.endswith(".py")
            }
            for contract_path in feature.contract_paths:
                if not (root / contract_path).is_file():
                    add(feature, "missing_contract", contract_path)
                elif contract_path not in command_paths:
                    add(feature, "contract_not_executed", contract_path)
        for config_key in feature.config_keys:
            if config_key not in known_config:
                add(feature, "unknown_config_key", config_key)
            if f"`{config_key.split('.')[-1]}`" not in config_reference:
                add(feature, "undocumented_config_key", config_key)
        for config_key, disabled_value in feature.disabled_config:
            if config_key not in feature.config_keys:
                add(feature, "disabled_key_not_declared", config_key)
            elif config_key not in config_defaults:
                add(feature, "missing_schema_default", config_key)
            elif config_defaults[config_key] != disabled_value:
                add(
                    feature,
                    "default_gate_changed",
                    f"{config_key}: expected {disabled_value!r}, "
                    f"observed {config_defaults[config_key]!r}",
                )
        if feature.default_off and FeatureInvariant.DEFAULT_PATH not in feature.invariants:
            add(feature, "missing_default_path", "default-off feature lacks proof")
        if feature.default_off and feature.config_keys and not feature.disabled_config:
            add(
                feature,
                "missing_disabled_config",
                "default-off feature does not declare its schema gate value",
            )
        if feature.stateful and FeatureInvariant.PERSISTENCE not in feature.invariants:
            add(feature, "missing_persistence", "stateful feature lacks round-trip proof")
        if feature.optimized:
            if not feature.reference_path or not (root / feature.reference_path).exists():
                add(feature, "missing_reference", feature.reference_path)
        issues.extend(
            validate_routing_topology_promotion(root=root, feature=feature)
        )
        for claim in feature.public_claims:
            if claim not in public_claims:
                add(feature, "missing_readme_claim", claim)
    return tuple(issues)


T = TypeVar("T")


def prove_default_path(
    reference: Callable[[], T],
    disabled: Callable[[], T],
    *,
    compare: Callable[[T, T], bool],
) -> None:
    expected = reference()
    observed = disabled()
    if not compare(expected, observed):
        raise AssertionError("Disabled feature does not preserve the reference path")


def prove_reference_parity(
    reference: Callable[[], T],
    optimized: Callable[[], T],
    *,
    compare: Callable[[T, T], bool],
) -> None:
    expected = reference()
    observed = optimized()
    if not compare(expected, observed):
        raise AssertionError("Optimized feature does not match its reference")


def prove_gradients(
    reference: Callable[[], Mapping[str, T]],
    candidate: Callable[[], Mapping[str, T]],
    *,
    compare: Callable[[T, T], bool],
) -> None:
    expected = reference()
    observed = candidate()
    if expected.keys() != observed.keys():
        raise AssertionError("Gradient sets do not match")
    mismatched = [
        name for name in expected if not compare(expected[name], observed[name])
    ]
    if mismatched:
        raise AssertionError(f"Gradient parity failed for {sorted(mismatched)}")


def prove_state_roundtrip(
    source: Callable[[], T],
    snapshot: Callable[[T], Mapping[str, Any]],
    restore: Callable[[Mapping[str, Any]], T],
    observe: Callable[[T], Any],
) -> None:
    original = source()
    restored = restore(snapshot(original))
    if observe(original) != observe(restored):
        raise AssertionError("Feature state does not round-trip")


def prove_resource_bound(observed: int | float, maximum: int | float) -> None:
    if observed < 0 or maximum < 0 or observed > maximum:
        raise AssertionError(
            f"Feature resource usage {observed} exceeds declared bound {maximum}"
        )


def prove_native_execution(
    execute: Callable[[], T],
    *,
    validate: Callable[[T], bool],
) -> None:
    if not validate(execute()):
        raise AssertionError("Native feature execution contract failed")
