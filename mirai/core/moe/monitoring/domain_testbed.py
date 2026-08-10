"""Domain-labelled reference-router evaluation for sparse MoE routing.

The evaluator adapts the reference-router testbed from arXiv:2604.07030 to
offline Mirai evidence.  It compares observed token routes with an explicit
domain-to-expert assignment without changing routing or training behavior.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


DOMAIN_ROUTING_TESTBED_SCHEMA = "mirai.domain_routing_testbed"
DOMAIN_ROUTING_TESTBED_SCHEMA_VERSION = 1


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_mutual_information(pairs: Sequence[tuple[str, int]]) -> float:
    total = len(pairs)
    if total <= 0:
        raise ValueError("Domain routing evaluation requires observations.")
    domains = Counter(domain for domain, _expert in pairs)
    experts = Counter(expert for _domain, expert in pairs)
    joint = Counter(pairs)
    mutual_information = 0.0
    for (domain, expert), count in joint.items():
        probability = count / total
        mutual_information += probability * math.log(
            (count * total) / (domains[domain] * experts[expert])
        )
    domain_entropy = -sum(
        (count / total) * math.log(count / total) for count in domains.values()
    )
    expert_entropy = -sum(
        (count / total) * math.log(count / total) for count in experts.values()
    )
    denominator = math.sqrt(domain_entropy * expert_entropy)
    return 0.0 if denominator == 0.0 else mutual_information / denominator


@dataclass(frozen=True)
class DomainRoutingLayerReport:
    token_count: int
    top1_reference_accuracy: float
    selected_reference_precision: float
    reference_expert_coverage: float
    reference_regret: float
    expert_domain_purity: float
    normalized_mutual_information: float
    per_domain: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "top1_reference_accuracy": self.top1_reference_accuracy,
            "selected_reference_precision": self.selected_reference_precision,
            "reference_expert_coverage": self.reference_expert_coverage,
            "reference_regret": self.reference_regret,
            "expert_domain_purity": self.expert_domain_purity,
            "normalized_mutual_information": self.normalized_mutual_information,
            "per_domain": self.per_domain,
        }


def evaluate_domain_routing_layer(
    records: Sequence[Mapping[str, Any]],
    *,
    reference: Mapping[str, Sequence[int]],
    num_experts: int,
) -> DomainRoutingLayerReport:
    """Compare one layer's selected routes with a domain reference router."""
    experts = int(num_experts)
    if experts <= 0:
        raise ValueError("num_experts must be positive.")
    normalized_reference: dict[str, frozenset[int]] = {}
    for raw_domain, raw_experts in reference.items():
        domain = str(raw_domain).strip()
        values = tuple(int(value) for value in raw_experts)
        if not domain or not values or len(set(values)) != len(values):
            raise ValueError("Every reference domain needs unique expert ids.")
        if min(values) < 0 or max(values) >= experts:
            raise ValueError("Reference expert id is outside num_experts.")
        normalized_reference[domain] = frozenset(values)
    if not normalized_reference:
        raise ValueError("Domain routing reference cannot be empty.")

    token_count = 0
    selected_total = 0
    selected_hits = 0
    top1_hits = 0
    observed_reference: dict[str, set[int]] = defaultdict(set)
    top1_pairs: list[tuple[str, int]] = []
    expert_domains: dict[int, Counter[str]] = defaultdict(Counter)
    domain_totals: Counter[str] = Counter()
    domain_hits: Counter[str] = Counter()
    domain_top1_hits: Counter[str] = Counter()
    domain_selected: Counter[str] = Counter()

    for record in records:
        domain = str(record.get("domain", "")).strip()
        if domain not in normalized_reference:
            raise ValueError(f"Observation uses unassigned domain {domain!r}.")
        selected = tuple(int(value) for value in record.get("selected_experts", ()))
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Every observation needs unique selected_experts.")
        if min(selected) < 0 or max(selected) >= experts:
            raise ValueError("Selected expert id is outside num_experts.")
        allowed = normalized_reference[domain]
        hits = sum(expert in allowed for expert in selected)
        token_count += 1
        selected_total += len(selected)
        selected_hits += hits
        top1_hits += int(selected[0] in allowed)
        domain_totals[domain] += 1
        domain_selected[domain] += len(selected)
        domain_hits[domain] += hits
        domain_top1_hits[domain] += int(selected[0] in allowed)
        observed_reference[domain].update(expert for expert in selected if expert in allowed)
        top1_pairs.append((domain, selected[0]))
        expert_domains[selected[0]][domain] += 1

    if token_count <= 0:
        raise ValueError("Domain routing evaluation requires observations.")
    reference_slots = sum(len(values) for values in normalized_reference.values())
    covered_slots = sum(len(observed_reference[domain]) for domain in normalized_reference)
    purity = sum(max(counts.values()) for counts in expert_domains.values()) / token_count
    precision = selected_hits / selected_total
    per_domain = {
        domain: {
            "token_count": int(domain_totals[domain]),
            "top1_reference_accuracy": domain_top1_hits[domain] / domain_totals[domain],
            "selected_reference_precision": domain_hits[domain] / domain_selected[domain],
            "reference_expert_coverage": len(observed_reference[domain]) / len(allowed),
        }
        for domain, allowed in sorted(normalized_reference.items())
        if domain_totals[domain] > 0
    }
    return DomainRoutingLayerReport(
        token_count=token_count,
        top1_reference_accuracy=top1_hits / token_count,
        selected_reference_precision=precision,
        reference_expert_coverage=covered_slots / reference_slots,
        reference_regret=1.0 - precision,
        expert_domain_purity=purity,
        normalized_mutual_information=_normalized_mutual_information(top1_pairs),
        per_domain=per_domain,
    )


def build_domain_routing_testbed_evidence(
    observations: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    dataset_fingerprint: str,
    model_fingerprint: str,
) -> dict[str, Any]:
    """Build versioned, lineage-bound evidence for every observed router layer."""
    dataset_id = str(dataset_fingerprint).strip()
    model_id = str(model_fingerprint).strip()
    if not dataset_id or not model_id:
        raise ValueError("Dataset and model fingerprints are required.")
    num_experts = int(observations.get("num_experts", 0))
    layers = observations.get("layers")
    domain_map = reference.get("domains")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError("Observations require a non-empty layers mapping.")
    if not isinstance(domain_map, Mapping) or not domain_map:
        raise ValueError("Reference requires a non-empty domains mapping.")
    reports: dict[str, Any] = {}
    for raw_name, records in sorted(layers.items()):
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError("Every observed layer must contain a record sequence.")
        reports[str(raw_name)] = evaluate_domain_routing_layer(
            records,
            reference=domain_map,
            num_experts=num_experts,
        ).to_dict()
    return {
        "schema": DOMAIN_ROUTING_TESTBED_SCHEMA,
        "schema_version": DOMAIN_ROUTING_TESTBED_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_id,
        "model_fingerprint": model_id,
        "observation_fingerprint": _canonical_sha256(observations),
        "reference_fingerprint": _canonical_sha256(reference),
        "num_experts": num_experts,
        "layers": reports,
    }


def save_domain_routing_testbed_evidence(
    path: str | Path, evidence: Mapping[str, Any], *, overwrite: bool = False
) -> None:
    output = Path(path)
    if output.exists() and not overwrite:
        raise ValueError(f"Domain routing evidence already exists: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    temp.write_text(
        json.dumps(dict(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(output)


__all__ = [
    "DOMAIN_ROUTING_TESTBED_SCHEMA",
    "DOMAIN_ROUTING_TESTBED_SCHEMA_VERSION",
    "DomainRoutingLayerReport",
    "build_domain_routing_testbed_evidence",
    "evaluate_domain_routing_layer",
    "save_domain_routing_testbed_evidence",
]
