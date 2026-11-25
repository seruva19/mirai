"""Offline head-local expert deletion for MAGI-2 Preview."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from mirai.core.moe.calibration.pruning import normalize_expert_pruning_criterion, select_pruned_experts
from mirai.core.moe.calibration.router_repair import router_tensor_fingerprint

from .quantized_experts import MAGI2_ROUTED_EXPERT_TENSOR_NAMES, iter_magi2_moe_layers


_SCHEMA = "mirai.magi2_preview.head_expert_pruning"
_PARAMETERS = ("gate", *MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
_ROUTER_BUFFERS = ("expert_bias", "expert_bias_ema")


@dataclass(frozen=True)
class Magi2ExpertPruningPlan:
    criterion: str
    source_weight_fingerprint: str
    kept_by_head: dict[str, tuple[tuple[int, ...], ...]]

    def topology_digest(self) -> bytes:
        canonical = {
            name: [sorted(ids) for ids in heads] for name, heads in self.kept_by_head.items()
        }
        payload = json.dumps([_SCHEMA, self.source_weight_fingerprint, canonical], sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).digest()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "schema": _SCHEMA,
            "version": 1,
            "criterion": self.criterion,
            "source_weight_fingerprint": self.source_weight_fingerprint,
            "kept_by_head": self.kept_by_head,
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Magi2ExpertPruningPlan:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA or payload.get("version") != 1:
            raise ValueError("Unsupported MAGI-2 expert pruning plan schema.")
        kept = payload.get("kept_by_head")
        fingerprint = payload.get("source_weight_fingerprint")
        if not isinstance(kept, dict) or not kept or not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:"):
            raise ValueError("MAGI-2 expert pruning plan needs keep sets and a source fingerprint.")
        try:
            assignments = {name: tuple(tuple(ids) for ids in heads) for name, heads in kept.items()}
        except TypeError as exc:
            raise ValueError("Malformed MAGI-2 head-local keep sets.") from exc
        return cls(normalize_expert_pruning_criterion(payload.get("criterion")), fingerprint, assignments)


def _source_fingerprint(layers) -> str:
    tensors = {
        f"{name}.{key}": getattr(layer, key)
        for name, layer in layers.items()
        for key in _PARAMETERS
    }
    tensors.update({
        f"{name}.router.{key}": getattr(layer.router, key)
        for name, layer in layers.items()
        for key in _ROUTER_BUFFERS
    })
    return router_tensor_fingerprint(tensors)


def build_magi2_expert_pruning_plan(
    transformer: nn.Module, scores: dict[str, torch.Tensor], *, criterion: str, keep_fraction: float,
) -> Magi2ExpertPruningPlan:
    """Rank REAP or AIMER scores independently within each head."""
    criterion = normalize_expert_pruning_criterion(criterion)
    if criterion not in {"reap", "aimer"}:
        raise ValueError("MAGI-2 head-local score ranking requires REAP or AIMER.")
    layers = dict(iter_magi2_moe_layers(transformer))
    if not scores or set(scores) - set(layers):
        raise ValueError("MAGI-2 pruning scores must identify routed layers.")
    kept = {}
    for name, values in scores.items():
        layer = layers[name]
        if values.shape != (layer.num_heads, layer.num_experts) or not bool(torch.isfinite(values).all()):
            raise ValueError(f"Invalid per-head expert pruning scores for '{name}'.")
        selected = select_pruned_experts(
            {str(index): row for index, row in enumerate(values)},
            keep_fraction=keep_fraction, top_k=layer.topk, keep_largest=criterion != "aimer",
        )
        kept[name] = tuple(selected[str(index)] for index in range(layer.num_heads))
    return Magi2ExpertPruningPlan(criterion, _source_fingerprint({name: layers[name] for name in kept}), kept)


def build_magi2_diet_pruning_plan(
    transformer: nn.Module, kept_by_head: dict[str, tuple[tuple[int, ...], ...]],
) -> Magi2ExpertPruningPlan:
    """Bind offline DIET survivor decisions to the dense source tensors."""
    layers = dict(iter_magi2_moe_layers(transformer))
    if not kept_by_head or set(kept_by_head) - set(layers):
        raise ValueError("MAGI-2 DIET survivors must identify routed layers.")
    fingerprint = _source_fingerprint({name: layers[name] for name in kept_by_head})
    return Magi2ExpertPruningPlan("diet", fingerprint, dict(kept_by_head))


def _validated_pruning_source(transformer: nn.Module, plan: Magi2ExpertPruningPlan):
    layers = dict(iter_magi2_moe_layers(transformer))
    if not plan.kept_by_head or set(plan.kept_by_head) - set(layers):
        raise ValueError("MAGI-2 expert pruning plan does not match model topology.")
    selected = {}
    for name, heads in plan.kept_by_head.items():
        layer = layers[name]
        if layer.ep_size != 1 or len(heads) != layer.num_heads or any(
            key not in layer._parameters or layer._parameters[key] is None for key in _PARAMETERS
        ):
            raise ValueError("MAGI-2 pruning requires native dense single-device expert layers.")
        keep_count = len(heads[0])
        indices = []
        for head, ids in enumerate(heads):
            if (
                len(ids) != keep_count or len(ids) < layer.topk
                or any(type(index) is not int or not 0 <= index < layer.num_experts for index in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError(f"Invalid or unequal keep counts for '{name}', or a head falls below top-k.")
            indices.extend(head * layer.num_experts + index for index in sorted(ids))
        selected[name] = (keep_count, indices)
    if _source_fingerprint({name: layers[name] for name in selected}) != plan.source_weight_fingerprint:
        raise ValueError("MAGI-2 expert pruning plan source-weight fingerprint mismatch.")
    return layers, selected


def verify_magi2_pruning_source(transformer: nn.Module, plan: Magi2ExpertPruningPlan) -> bytes:
    """Verify the dense source before accepting a pruned model's attention adapter."""
    _validated_pruning_source(transformer, plan)
    return plan.topology_digest()


@torch.no_grad()
def apply_magi2_expert_pruning(transformer: nn.Module, plan: Magi2ExpertPruningPlan) -> None:
    layers, selected = _validated_pruning_source(transformer, plan)
    for name, (keep_count, ids) in selected.items():
        layer = layers[name]
        for key in _PARAMETERS:
            parameter = layer._parameters[key]
            index = torch.tensor(ids, device=parameter.device)
            setattr(layer, key, nn.Parameter(parameter.index_select(0, index), requires_grad=parameter.requires_grad))
        for key in _ROUTER_BUFFERS:
            buffer = getattr(layer.router, key)
            setattr(layer.router, key, buffer.index_select(0, torch.tensor(ids, device=buffer.device)))
        layer.num_experts = keep_count
        layer.config = replace(layer.config, num_experts=keep_count)
        layer.flatten_num_experts = layer.num_heads * keep_count
        layer.local_flatten_num_experts = layer.local_num_heads * keep_count
        layer._mirai_experts_pruned = True
