"""Offline head-local expert deletion for MAGI-2 Preview."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from mirai.core.moe.calibration.projection import ExpertProjectionSpec
from mirai.core.moe.calibration.pruning import (
    DietCalibrationTarget,
    ExpertPruningCalibrationTarget,
    aimer_expert_scores,
    normalize_expert_pruning_criterion,
    select_pruned_experts,
)
from mirai.core.moe.calibration.router_repair import (
    router_tensor_fingerprint,
    update_router_tensor_fingerprints,
)

from .quantized_experts import (
    MAGI2_NF4_STORE_ATTR,
    MAGI2_ROUTED_EXPERT_TENSOR_NAMES,
    NF4_BLOCKSIZE,
    Magi2Nf4ExpertStore,
    _new_expert_store,
    iter_magi2_moe_layers,
)

if TYPE_CHECKING:
    from mirai.core.moe.calibration.diet import DietCalibrationEvidence, DietGroupTopology, DietLayerCapture, DietODLConfig
    from mirai.core.moe.calibration.prototypes import HierarchicalOutputMergePlan, PrototypeCalibrationEvidence


_SCHEMA = "mirai.magi2_preview.head_expert_pruning"
_MERGE_SCHEMA = "mirai.magi2_preview.head_expert_consolidation"
_PARAMETERS = ("gate", *MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
_ROUTER_BUFFERS = ("expert_bias", "expert_bias_ema")


def _dense_pruning_layers(transformer: nn.Module) -> dict[str, nn.Module]:
    layers = dict(iter_magi2_moe_layers(transformer))
    for layer in layers.values():
        if (
            layer.ep_size != 1
            or getattr(layer, "_mirai_experts_pruned", False)
            or getattr(layer, "_mirai_expert_aliases", None) is not None
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


@dataclass(frozen=True)
class Magi2ExpertConsolidationPlan:
    source_weight_fingerprint: str
    heads_by_layer: dict[str, tuple[HierarchicalOutputMergePlan, ...]]

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": _MERGE_SCHEMA,
            "version": 1,
            "source_weight_fingerprint": self.source_weight_fingerprint,
            "heads_by_layer": {
                name: [
                    {
                        "logical_to_prototype": head.consolidation.logical_to_prototype,
                        "merge_weights": head.merge_weights,
                    }
                    for head in heads
                ]
                for name, heads in self.heads_by_layer.items()
            },
        }

    def topology_digest(self) -> bytes:
        return hashlib.sha256(
            json.dumps(self._payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self._payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Magi2ExpertConsolidationPlan:
        from mirai.core.moe.calibration.prototypes import HierarchicalOutputMergePlan
        from mirai.core.moe.storage.aliases import build_prototype_plan

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != _MERGE_SCHEMA or type(payload.get("version")) is not int or payload["version"] != 1:
            raise ValueError("Unsupported MAGI-2 expert consolidation schema.")
        fingerprint = payload.get("source_weight_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:") or len(fingerprint) != 71 or any(
            character not in "0123456789abcdef" for character in fingerprint[7:]
        ):
            raise ValueError("MAGI-2 consolidation plan requires a SHA-256 source fingerprint.")
        raw_layers = payload.get("heads_by_layer")
        if not isinstance(raw_layers, dict) or not raw_layers:
            raise ValueError("MAGI-2 consolidation plan requires layer assignments.")
        layers = {}
        for name, raw_heads in raw_layers.items():
            if not name or not isinstance(raw_heads, list) or not raw_heads:
                raise ValueError("MAGI-2 consolidation plan requires per-head assignments.")
            heads = []
            for raw in raw_heads:
                if not isinstance(raw, dict):
                    raise ValueError("Malformed MAGI-2 consolidation head.")
                aliases = raw.get("logical_to_prototype")
                weights = raw.get("merge_weights")
                if (
                    not isinstance(aliases, list) or any(type(value) is not int for value in aliases)
                    or not isinstance(weights, list) or any(type(value) not in {int, float} for value in weights)
                ):
                    raise ValueError("Malformed MAGI-2 consolidation aliases or weights.")
                heads.append(HierarchicalOutputMergePlan(
                    build_prototype_plan(aliases, num_logical_experts=len(aliases)),
                    tuple(float(weight) for weight in weights),
                ).validate())
            layers[name] = tuple(heads)
        return cls(fingerprint, layers)


def build_magi2_hierarchical_merge_plan(
    transformer: nn.Module, evidence: dict[str, tuple[PrototypeCalibrationEvidence, ...]],
    *, reduction_ratio: float,
) -> Magi2ExpertConsolidationPlan:
    from mirai.core.moe.calibration.prototypes import select_hierarchical_output_merge

    layers = _dense_pruning_layers(transformer)
    if not evidence or set(evidence) - set(layers):
        raise ValueError("MAGI-2 consolidation evidence must identify routed layers.")
    heads_by_layer = {}
    for name, heads in evidence.items():
        layer = layers[name]
        if len(heads) != layer.num_heads or any(item.num_experts != layer.num_experts for item in heads):
            raise ValueError(f"MAGI-2 consolidation evidence has incompatible head topology for {name!r}.")
        heads_by_layer[name] = tuple(select_hierarchical_output_merge(item, reduction_ratio) for item in heads)
    return Magi2ExpertConsolidationPlan(
        _source_fingerprint({name: layers[name] for name in heads_by_layer}), heads_by_layer,
    )


def observe_magi2_hierarchical_merge_head(
    layer: nn.Module, head: int, sampled_tokens: torch.Tensor, accumulator: Any,
) -> None:
    """Accumulate mean outputs and route frequencies on the same sampled tokens."""
    capture = capture_magi2_diet_head(layer, head, sampled_tokens)
    scores = capture.router_scores
    outputs = capture.expert_outputs
    if accumulator.num_experts != capture.topology.num_experts:
        raise ValueError("MAGI-2 hierarchical merge accumulator has incompatible expert count.")
    routes = (scores + capture.choice_bias).topk(capture.topology.top_k, dim=-1).indices
    weights = scores.gather(1, routes)
    if capture.topology.normalize_topk_prob:
        weights = torch.nn.functional.normalize(weights, p=1, dim=-1, eps=capture.topology.normalization_epsilon)
    weights = weights * capture.topology.route_scale
    for expert in range(capture.topology.num_experts):
        accumulator.record_all_expert_output(expert, outputs[:, expert])
        token_ids, slots = (routes == expert).nonzero(as_tuple=True)
        if token_ids.numel():
            accumulator.record(expert, weights[token_ids, slots], outputs[token_ids, expert])


@torch.no_grad()
def apply_magi2_hierarchical_merge(
    transformer: nn.Module, plan: Magi2ExpertConsolidationPlan,
) -> None:
    """Store merged expert matrices while preserving every logical router row."""
    layers = _dense_pruning_layers(transformer)
    layouts = _merge_layouts(layers, plan)
    if _source_fingerprint({name: layers[name] for name in layouts}) != plan.source_weight_fingerprint:
        raise ValueError("MAGI-2 consolidation source-weight fingerprint mismatch.")
    for name in layouts:
        layer = layers[name]
        if layer._mirai_moe_kernel_backend is not None or any(
            layer._parameters[key].requires_grad for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES
        ):
            raise ValueError("MAGI-2 consolidation requires frozen experts and native reference execution.")
    for name, (aliases, clusters) in layouts.items():
        layer = layers[name]
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            compact = _merge_projection(layer._parameters[key], clusters)
            setattr(layer, key, nn.Parameter(compact, requires_grad=False))
        _attach_merge_aliases(layer, aliases, len(clusters))


def _merge_layouts(layers, plan):
    from mirai.core.moe.storage.aliases import validate_logical_to_physical

    if not plan.heads_by_layer or set(plan.heads_by_layer) - set(layers):
        raise ValueError("MAGI-2 consolidation plan does not match model topology.")
    layouts = {}
    for name, heads in plan.heads_by_layer.items():
        layer = layers[name]
        if len(heads) != layer.num_heads or any(
            item.validate().consolidation.logical_experts != layer.num_experts for item in heads
        ):
            raise ValueError(f"MAGI-2 consolidation plan has incompatible head topology for {name!r}.")
        for item in heads:
            validate_logical_to_physical(
                item.consolidation.logical_to_physical,
                physical_experts=item.consolidation.physical_experts,
            )
        mapping = []
        clusters = []
        for head, item in enumerate(heads):
            aliases = item.consolidation.logical_to_physical
            mapping.extend(len(clusters) + alias for alias in aliases)
            for physical in range(item.consolidation.physical_experts):
                clusters.append([
                    (head * layer.num_experts + expert, item.merge_weights[expert])
                    for expert, alias in enumerate(aliases) if alias == physical
                ])
        layouts[name] = (tuple(mapping), clusters)
    return layouts


def _merge_projection(weight, clusters):
    compact = weight.new_empty((len(clusters), *weight.shape[1:]))
    for physical, members in enumerate(clusters):
        merged = torch.zeros_like(weight[0], dtype=torch.float32)
        for expert, coefficient in members:
            merged.add_(weight[expert].float(), alpha=coefficient)
        compact[physical].copy_(merged)
    return compact


def _attach_merge_aliases(layer, aliases, physical_count):
    layer._mirai_expert_aliases = aliases
    layer._mirai_physical_expert_count = physical_count
    layer.register_buffer(
        "_mirai_expert_alias_indices", torch.tensor(aliases, device=layer.gate.device),
        persistent=False,
    )


@torch.no_grad()
def stream_merge_magi2_experts(
    transformer: nn.Module, plan: Magi2ExpertConsolidationPlan, checkpoint_dir: str | Path,
    *, quant_format: str | None = None, blocksize: int = NF4_BLOCKSIZE,
    prepare_packed_restore: bool = False,
) -> tuple[set[str], dict[str, Magi2Nf4ExpertStore]]:
    """Read and merge one original projection at a time before publishing the new pool."""
    from safetensors import safe_open

    from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import read_safetensors_weight_map

    layers = _dense_pruning_layers(transformer)
    layouts = _merge_layouts(layers, plan)
    if prepare_packed_restore and quant_format is not None:
        raise ValueError("MAGI-2 packed consolidation restore cannot quantize the source.")
    weight_map = read_safetensors_weight_map(str(checkpoint_dir))
    sources = {}
    for name, layer in layers.items():
        if prepare_packed_restore and name not in layouts:
            continue
        for key in _PARAMETERS if name in layouts else MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            full_key = f"{name}.{key}"
            sources[full_key] = (name, layer, key, full_key)
        if name in layouts:
            for key in _ROUTER_BUFFERS:
                full_key = f"{name}.router.{key}"
                sources[full_key] = (name, layer.router, key, f"{name}.router.expert_bias_ema")
    missing = sorted((sources.keys() | {item[3] for item in sources.values()}) - weight_map.keys())
    if missing:
        raise ValueError(f"MAGI-2 checkpoint is missing consolidation source tensors: {missing[:4]}.")
    digest = hashlib.sha256()
    staged = {}
    stores = {}
    if quant_format is not None:
        for name, layer in layers.items():
            groups = len(layouts[name][1]) if name in layouts else layer.local_flatten_num_experts
            stores[name] = _new_expert_store(num_groups=groups, blocksize=blocksize, quant_format=quant_format)
    for full_key, (name, owner, key, source_key) in sorted(sources.items()):
        target = getattr(owner, key)
        with safe_open(weight_map[source_key], framework="pt", device="cpu") as shard:
            value = shard.get_tensor(source_key)
            if value.shape != target.shape or not value.is_floating_point():
                raise ValueError(f"MAGI-2 consolidation tensor {source_key!r} has incompatible shape or dtype.")
            value = value.to(dtype=target.dtype)
            if name in layouts:
                update_router_tensor_fingerprints((digest,), full_key, value)
                if key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
                    if prepare_packed_restore:
                        del value
                        continue
                    value = _merge_projection(value, layouts[name][1])
            if name in stores and key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
                stores[name].quantize_dense(key, value)
                del value
                continue
            if isinstance(target, nn.Parameter):
                merged_projection = name in layouts and key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES
                value = nn.Parameter(value, requires_grad=target.requires_grad and not merged_projection)
            staged[full_key] = value
    if f"sha256:{digest.hexdigest()}" != plan.source_weight_fingerprint:
        raise ValueError("MAGI-2 consolidation source-weight fingerprint mismatch.")
    for full_key, value in staged.items():
        _, owner, key, _ = sources[full_key]
        setattr(owner, key, value)
    for name, (aliases, clusters) in layouts.items():
        _attach_merge_aliases(layers[name], aliases, len(clusters))
    for name, store in stores.items():
        layer = layers[name]
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            del layer._parameters[key]
        layer.add_module(MAGI2_NF4_STORE_ATTR, store)
    return set(staged), stores


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


def _diet_head_topology(layer: nn.Module) -> DietGroupTopology:
    from mirai.core.moe.calibration.diet import DietGroupTopology

    if layer.config.score_func != "sigmoid" or layer.topk >= layer.num_experts:
        raise ValueError("MAGI-2 DIET requires sigmoid routing with room to delete one expert.")
    return DietGroupTopology(
        num_experts=layer.num_experts, top_k=layer.topk, num_groups=1,
        topk_group=1, group_score_topk=1, normalize_topk_prob=layer.config.route_norm,
        route_scale=layer.config.route_scale, normalization_epsilon=1e-12,
    ).validate()


@torch.no_grad()
def capture_magi2_diet_head(
    layer: nn.Module, head: int, sampled_tokens: torch.Tensor,
) -> DietLayerCapture:
    """Capture all experts for sampled tokens of one dense MAGI-2 routed head."""
    from mirai.core.moe.calibration.diet import DietLayerCapture

    topology = _diet_head_topology(layer)
    if not 0 <= head < layer.num_heads or sampled_tokens.ndim != 2 or (
        sampled_tokens.shape[0] == 0 or sampled_tokens.shape[1] != layer.d_head
    ):
        raise ValueError("MAGI-2 DIET needs a valid head and nonempty [tokens, head width] input.")
    if layer.ep_size != 1 or getattr(layer, "_mirai_experts_pruned", False) or any(
        layer._parameters.get(key) is None for key in _PARAMETERS
    ) or any(
        getattr(layer, executor, None) is not None
        for executor in ("_mirai_router_lora_executor", "_mirai_int8_router_executor")
    ):
        raise ValueError("MAGI-2 DIET capture requires the original dense expert pool and router.")
    start = head * layer.num_experts
    rows = slice(start, start + layer.num_experts)
    scores = (sampled_tokens.float() @ layer.gate[rows].float().T).sigmoid()
    outputs = []
    for index in range(rows.start, rows.stop):
        gate = (sampled_tokens @ layer.W_gate[index]).float().clamp(max=7.0)
        up = (sampled_tokens @ layer.W_up[index]).float().clamp(min=-7.0, max=7.0)
        hidden = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
        outputs.append(hidden.to(layer.W_down.dtype) @ layer.W_down[index])
    return DietLayerCapture(
        expert_outputs=torch.stack(outputs, dim=1).cpu(),
        router_scores=scores.cpu(),
        choice_bias=layer.router.expert_bias[rows].detach().to(device="cpu", copy=True),
        topology=topology,
    ).validate()


@dataclass(frozen=True)
class _Magi2DietHeadHost:
    module: nn.Module
    head: int

    def get_diet_capture_observer(self):
        entry = getattr(self.module, "_mirai_diet_callbacks", {}).get(self.head)
        return None if entry is None else entry.observer

    def set_diet_capture_observer(self, observer) -> None:
        callbacks = getattr(self.module, "_mirai_diet_callbacks", None)
        if observer is None:
            if callbacks is not None:
                callbacks.pop(self.head, None)
                if not callbacks:
                    del self.module._mirai_diet_callbacks
            return
        if callbacks is None:
            callbacks = {}
            self.module._mirai_diet_callbacks = callbacks
        callbacks[self.head] = _Magi2DietHeadCapture(self.module.moe_mlp, self.head, observer)


@dataclass(frozen=True)
class _Magi2DietHeadCapture:
    layer: nn.Module
    head: int
    observer: Any

    def __call__(self, inputs: torch.Tensor, video_rows: torch.Tensor) -> None:
        if not self.observer.is_enabled or not video_rows.numel():
            return
        indices = self.observer.sample_token_indices(inputs[:video_rows.numel()])
        if not indices.numel():
            return
        rows = video_rows.index_select(0, indices)
        width = self.layer.d_head
        head_input = inputs[:, self.head * width:(self.head + 1) * width]
        capture = capture_magi2_diet_head(self.layer, self.head, head_input.index_select(0, rows))
        self.observer.capture_layer(
            sampled_token_indices=indices, expert_outputs=capture.expert_outputs,
            router_scores=capture.router_scores, choice_bias=capture.choice_bias,
        )


def build_magi2_diet_calibration_targets(transformer: nn.Module) -> dict[str, DietCalibrationTarget]:
    """Expose each head as a DIET target with paired video-token sampling."""
    targets = {}
    for name, layer in _dense_pruning_layers(transformer).items():
        parent_name, _, _ = name.rpartition(".")
        parent = transformer.get_submodule(parent_name)
        if getattr(parent, "moe_mlp", None) is not layer:
            raise ValueError(f"MAGI-2 DIET could not find the native MoE parent for {name!r}.")
        topology = _diet_head_topology(layer)
        for head in range(layer.num_heads):
            target_name = f"{name}.head.{head}"
            targets[target_name] = DietCalibrationTarget(
                name=target_name, host=_Magi2DietHeadHost(parent, head),
                num_experts=topology.num_experts, top_k=topology.top_k, n_group=1,
                topk_group=1, group_score_topk=1, norm_topk_prob=topology.normalize_topk_prob,
                route_scale=topology.route_scale, normalization_epsilon=topology.normalization_epsilon,
            )
    return targets


def magi2_diet_source_fingerprint(transformer: nn.Module) -> str:
    """Fingerprint the full dense routed expert inventory used for DIET capture."""
    return _source_fingerprint(_dense_pruning_layers(transformer)).removeprefix("sha256:")


def select_magi2_diet_pruning_plan(
    transformer: nn.Module, evidence: dict[str, tuple[DietCalibrationEvidence, ...]],
    *, keep_fraction: float, config: DietODLConfig | None = None,
) -> Magi2ExpertPruningPlan:
    """Select equal-size survivor sets independently within each captured head."""
    from mirai.core.moe.calibration.diet import select_odl_survivors, uniform_layer_counts

    layers = _dense_pruning_layers(transformer)
    if not evidence or set(evidence) - set(layers):
        raise ValueError("MAGI-2 DIET evidence must identify routed layers.")
    counts = uniform_layer_counts({name: layers[name].num_experts for name in evidence}, keep_fraction)
    kept = {}
    for name, heads in sorted(evidence.items()):
        layer = layers[name]
        topology = _diet_head_topology(layer)
        if len(heads) != layer.num_heads or any(item.validate().topology != topology for item in heads):
            raise ValueError(f"MAGI-2 DIET evidence has incompatible head topology for {name!r}.")
        if counts[name] < topology.top_k:
            raise ValueError(f"MAGI-2 DIET expert budget is below top-k for {name!r}.")
        kept[name] = tuple(
            select_odl_survivors(item.signatures, counts[name], topology, config=config)
            for item in heads
        )
    return build_magi2_diet_pruning_plan(transformer, kept)


def build_magi2_diet_pruning_plan_from_evidence(
    transformer: nn.Module, evidence_path: str | Path, *,
    keep_fraction: float, config: DietODLConfig | None = None,
) -> Magi2ExpertPruningPlan:
    from mirai.core.moe.calibration.diet import load_diet_evidence

    evidence, lineage = load_diet_evidence(evidence_path)
    if lineage.dense_source_sha256 != magi2_diet_source_fingerprint(transformer):
        raise ValueError("DIET evidence belongs to a different MAGI-2 dense source.")
    layers = _dense_pruning_layers(transformer)
    expected = {f"{name}.head.{head}" for name, layer in layers.items() for head in range(layer.num_heads)}
    if set(evidence) != expected:
        raise ValueError("DIET evidence must cover every MAGI-2 routing head.")
    by_layer = {
        name: tuple(evidence[f"{name}.head.{head}"] for head in range(layer.num_heads))
        for name, layer in layers.items()
    }
    return select_magi2_diet_pruning_plan(transformer, by_layer, keep_fraction=keep_fraction, config=config)


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


def _pruning_indices(layers, plan: Magi2ExpertPruningPlan):
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
    return selected


def _validated_pruning_source(transformer: nn.Module, plan: Magi2ExpertPruningPlan):
    layers = _dense_pruning_layers(transformer)
    selected = _pruning_indices(layers, plan)
    _check_pruning_fingerprint(plan, _source_fingerprint({name: layers[name] for name in selected}))
    return layers, selected


def _check_pruning_fingerprint(plan: Magi2ExpertPruningPlan, fingerprint: str) -> None:
    if fingerprint != plan.source_weight_fingerprint:
        raise ValueError("MAGI-2 expert pruning plan source-weight fingerprint mismatch.")


def _resize_expert_pool(layer, keep_count: int) -> None:
    layer.num_experts = keep_count
    layer.config = replace(layer.config, num_experts=keep_count)
    layer.flatten_num_experts = layer.num_heads * keep_count
    layer.local_flatten_num_experts = layer.local_num_heads * keep_count
    layer._mirai_experts_pruned = True


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
        _resize_expert_pool(layer, keep_count)


def stream_prune_magi2_experts(
    transformer: nn.Module, plan: Magi2ExpertPruningPlan, checkpoint_dir: str | Path,
    *, quant_format: str | None = None, blocksize: int = NF4_BLOCKSIZE,
    prepare_packed_restore: bool = False,
) -> tuple[set[str], dict[str, Magi2Nf4ExpertStore]]:
    """Validate source tensors and materialize or pack the retained rows."""
    from safetensors import safe_open

    from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import read_safetensors_weight_map

    layers = _dense_pruning_layers(transformer)
    selected = _pruning_indices(layers, plan)
    if prepare_packed_restore and quant_format is not None:
        raise ValueError("MAGI-2 packed restore cannot quantize source weights.")
    weight_map = read_safetensors_weight_map(str(checkpoint_dir))
    inputs = {}
    for name, layer in layers.items():
        if prepare_packed_restore and name not in selected:
            continue
        for key in _PARAMETERS if name in selected else MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            qualified = f"{name}.{key}"
            inputs[qualified] = (name, layer, key, qualified, layer._parameters[key])
        if name in selected:
            for key in _ROUTER_BUFFERS:
                qualified = f"{name}.router.{key}"
                inputs[qualified] = (name, layer.router, key, f"{name}.router.expert_bias_ema", getattr(layer.router, key))
    missing = sorted((inputs.keys() | {entry[3] for entry in inputs.values()}) - weight_map.keys())
    if missing:
        raise ValueError(f"MAGI-2 checkpoint is missing pruning source tensors: {missing[:4]}.")
    indices = {name: torch.tensor(ids) for name, (_, ids) in selected.items()}
    stores = {}
    if quant_format is not None:
        for name, layer in layers.items():
            groups = len(selected[name][1]) if name in selected else layer.local_flatten_num_experts
            stores[name] = _new_expert_store(num_groups=groups, blocksize=blocksize, quant_format=quant_format)
    digest = hashlib.sha256()
    retained = {}
    for qualified, (name, owner, key, source_key, target) in sorted(inputs.items()):
        with safe_open(weight_map[source_key], framework="pt", device="cpu") as shard:
            value = shard.get_tensor(source_key)
            if value.shape != target.shape or not value.is_floating_point():
                raise ValueError(f"MAGI-2 pruning source tensor {source_key!r} has incompatible shape or dtype.")
            value = value.to(dtype=target.dtype)
            if name in selected:
                update_router_tensor_fingerprints((digest,), qualified, value)
            if prepare_packed_restore and key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
                shape = (len(selected[name][1]), *target.shape[1:])
                retained[qualified] = nn.Parameter(
                    torch.empty(shape, dtype=target.dtype, device="meta"), requires_grad=target.requires_grad,
                )
                del value
                continue
            if name in stores and key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
                stores[name].quantize_dense(
                    key, value, group_ids=selected[name][1] if name in selected else None,
                )
                del value
                continue
            if name in selected:
                value = value.index_select(0, indices[name])
            retained[qualified] = nn.Parameter(value, requires_grad=target.requires_grad) if isinstance(target, nn.Parameter) else value
    _check_pruning_fingerprint(plan, f"sha256:{digest.hexdigest()}")
    for qualified in retained:
        _, owner, key, _, _ = inputs[qualified]
        setattr(owner, key, retained[qualified])
    for name, (keep_count, _) in selected.items():
        _resize_expert_pool(layers[name], keep_count)
    for name, store in stores.items():
        layer = layers[name]
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            del layer._parameters[key]
        layer.add_module(MAGI2_NF4_STORE_ATTR, store)
    return {
        qualified for qualified in retained
        if not prepare_packed_restore or inputs[qualified][2] not in MAGI2_ROUTED_EXPERT_TENSOR_NAMES
    }, stores
