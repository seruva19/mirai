"""Offline head-local expert deletion for MAGI-2 Preview."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from mirai.core.moe.calibration.projection import ExpertProjectionSpec
from mirai.core.moe.calibration.pruning import (
    ExpertPruningCalibrationTarget,
    aimer_expert_scores,
    normalize_expert_pruning_criterion,
    select_pruned_experts,
)
from mirai.core.moe.calibration.router_repair import (
    router_tensor_fingerprint,
    update_router_tensor_fingerprints,
)

from .quantized_experts import MAGI2_ROUTED_EXPERT_TENSOR_NAMES, iter_magi2_moe_layers


_SCHEMA = "mirai.magi2_preview.head_expert_pruning"
_PARAMETERS = ("gate", *MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
_ROUTER_BUFFERS = ("expert_bias", "expert_bias_ema")


def _dense_pruning_layers(transformer: nn.Module) -> dict[str, nn.Module]:
    layers = dict(iter_magi2_moe_layers(transformer))
    for layer in layers.values():
        if (
            layer.ep_size != 1
            or getattr(layer, "_mirai_experts_pruned", False)
            or any(layer._parameters.get(key) is None for key in _PARAMETERS)
        ):
            raise ValueError("MAGI-2 pruning requires native dense single-device experts.")
    return layers


def build_magi2_pruning_calibration_targets(
    transformer: nn.Module,
) -> dict[str, ExpertPruningCalibrationTarget]:
    return {
        name: ExpertPruningCalibrationTarget(
            name, layer, layer.num_heads * layer.num_experts,
        )
        for name, layer in _dense_pruning_layers(transformer).items()
    }


@dataclass(frozen=True)
class _DenseMagi2ExpertProjections:
    layer: nn.Module

    @property
    def num_experts(self) -> int:
        return self.layer.num_heads * self.layer.num_experts

    def prototype_projection_specs(self) -> tuple[ExpertProjectionSpec, ...]:
        return tuple(
            ExpertProjectionSpec(name, tuple(getattr(self.layer, name).shape[1:]))
            for name in MAGI2_ROUTED_EXPERT_TENSOR_NAMES
        )

    def load_prototype_projection_block(
        self, projection_name: str, start_expert: int, stop_expert: int, *,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        weight = getattr(self.layer, projection_name).detach()
        return weight[start_expert:stop_expert].to(device=device, dtype=dtype)


@dataclass(frozen=True)
class _ShardMagi2ExpertProjections(_DenseMagi2ExpertProjections):
    name: str
    weight_map: dict[str, str]

    def load_prototype_projection_block(
        self, projection_name: str, start_expert: int, stop_expert: int, *,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        from safetensors import safe_open

        key = f"{self.name}.{projection_name}"
        target = getattr(self.layer, projection_name)
        with safe_open(self.weight_map[key], framework="pt", device="cpu") as shard:
            block = shard.get_slice(key)[start_expert:stop_expert]
        return block.to(dtype=target.dtype).to(device=device, dtype=dtype)


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
    """Rank head-major REAP or AIMER scores independently within each head."""
    criterion = normalize_expert_pruning_criterion(criterion)
    if criterion not in {"reap", "aimer"}:
        raise ValueError("MAGI-2 head-local score ranking requires REAP or AIMER.")
    layers = dict(iter_magi2_moe_layers(transformer))
    if not scores or set(scores) - set(layers):
        raise ValueError("MAGI-2 pruning scores must identify routed layers.")
    kept = _select_head_local_experts(layers, scores, criterion=criterion, keep_fraction=keep_fraction)
    return Magi2ExpertPruningPlan(criterion, _source_fingerprint({name: layers[name] for name in kept}), kept)


def _select_head_local_experts(layers, scores, *, criterion, keep_fraction):
    kept = {}
    for name, values in scores.items():
        layer = layers[name]
        if values.shape == (layer.num_heads * layer.num_experts,):
            values = values.reshape(layer.num_heads, layer.num_experts)
        if values.shape != (layer.num_heads, layer.num_experts) or not bool(torch.isfinite(values).all()):
            raise ValueError(f"Invalid per-head expert pruning scores for '{name}'.")
        selected = select_pruned_experts(
            {str(index): row for index, row in enumerate(values)},
            keep_fraction=keep_fraction, top_k=layer.topk, keep_largest=criterion != "aimer",
        )
        kept[name] = tuple(selected[str(index)] for index in range(layer.num_heads))
    return kept


def build_magi2_diet_pruning_plan(
    transformer: nn.Module, kept_by_head: dict[str, tuple[tuple[int, ...], ...]],
) -> Magi2ExpertPruningPlan:
    """Bind offline DIET survivor decisions to the dense source tensors."""
    layers = dict(iter_magi2_moe_layers(transformer))
    if not kept_by_head or set(kept_by_head) - set(layers):
        raise ValueError("MAGI-2 DIET survivors must identify routed layers.")
    fingerprint = _source_fingerprint({name: layers[name] for name in kept_by_head})
    return Magi2ExpertPruningPlan("diet", fingerprint, dict(kept_by_head))


def build_magi2_aimer_pruning_plan(
    transformer: nn.Module, *, keep_fraction: float,
    max_block_elements: int = 4_194_304, device: str | torch.device = "cpu",
) -> Magi2ExpertPruningPlan:
    """Score dense expert projections in bounded blocks before head-local pruning."""
    scores = {
        name: aimer_expert_scores(
            _DenseMagi2ExpertProjections(layer),
            max_block_elements=max_block_elements, device=device,
        ).scores
        for name, layer in _dense_pruning_layers(transformer).items()
    }
    return build_magi2_expert_pruning_plan(
        transformer, scores, criterion="aimer", keep_fraction=keep_fraction,
    )


def build_magi2_aimer_pruning_plan_from_shards(
    transformer: nn.Module, checkpoint_dir: str | Path, *, keep_fraction: float,
    max_block_elements: int = 4_194_304, device: str | torch.device = "cpu",
) -> Magi2ExpertPruningPlan:
    """Score checkpoint shards using the shapes and dtypes of a native model skeleton."""
    from safetensors import safe_open

    from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import read_safetensors_weight_map

    layers = _dense_pruning_layers(transformer)
    if not layers:
        raise ValueError("MAGI-2 AIMER requires routed layers.")
    weight_map = read_safetensors_weight_map(str(checkpoint_dir))
    inputs = {}
    for name, layer in layers.items():
        for key in _PARAMETERS:
            qualified = f"{name}.{key}"
            inputs[qualified] = (qualified, getattr(layer, key))
        for key in _ROUTER_BUFFERS:
            qualified = f"{name}.router.{key}"
            inputs[qualified] = (f"{name}.router.expert_bias_ema", getattr(layer.router, key))
    required = set(inputs) | {key for key, _ in inputs.values()}
    missing = sorted(required - weight_map.keys())
    if missing:
        raise ValueError(f"MAGI-2 checkpoint is missing AIMER source tensors: {missing[:4]}.")

    source = hashlib.sha256()
    for key, (checkpoint_key, target) in sorted(inputs.items()):
        with safe_open(weight_map[checkpoint_key], framework="pt", device="cpu") as shard:
            value = shard.get_tensor(checkpoint_key)
            if value.shape != target.shape or not value.is_floating_point():
                raise ValueError(f"MAGI-2 AIMER source tensor {checkpoint_key!r} has incompatible shape or dtype.")
            update_router_tensor_fingerprints((source,), key, value.to(dtype=target.dtype))
            del value
    scores = {
        name: aimer_expert_scores(
            _ShardMagi2ExpertProjections(layer, name, weight_map),
            max_block_elements=max_block_elements, device=device,
        ).scores
        for name, layer in layers.items()
    }
    kept = _select_head_local_experts(layers, scores, criterion="aimer", keep_fraction=keep_fraction)
    return Magi2ExpertPruningPlan("aimer", f"sha256:{source.hexdigest()}", kept)


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
