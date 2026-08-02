"""Online domain-to-expert affinity and exact expert-adapter update masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mirai.core.moe.adaptation.adapter_gate import RoutedAdapterGate


@dataclass(frozen=True)
class _LayerBinding:
    router: Any
    adapters: tuple[Any, ...]
    set_route_gate: Callable[[Any | None], None] | None


class DomainExpertSpecializationController:
    """Learn per-layer domain affinity from routing and mask adapter updates.

    Warmup forwards leave every expert adapter active while router hooks collect
    dispatch counts. Later forwards select experts whose affinity is within a
    relative threshold of the row maximum, with a deterministic ``min_experts``
    floor. The existing expert-adapter output mask makes inactive gradients
    exactly zero; :meth:`before_optimizer_step` additionally clears stale
    optimizer moments for those slices.
    """

    def __init__(
        self,
        *,
        warmup_steps: int,
        affinity_threshold: float,
        min_experts: int,
        momentum: float = 0.9,
        update_interval: int = 1,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be > 0.")
        if not 0.0 < affinity_threshold <= 1.0:
            raise ValueError("affinity_threshold must be in (0, 1].")
        if min_experts <= 0:
            raise ValueError("min_experts must be > 0.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if update_interval <= 0:
            raise ValueError("update_interval must be > 0.")
        self.warmup_steps = int(warmup_steps)
        self.affinity_threshold = float(affinity_threshold)
        self.min_experts = int(min_experts)
        self.momentum = float(momentum)
        self.update_interval = int(update_interval)
        self._layers: dict[str, _LayerBinding] = {}
        self._scores: dict[str, dict[str, Any]] = {}
        self._domain = ""
        self._domains: tuple[str, ...] = ()
        self._step = 0
        self._training = False
        self._handles: list[Any] = []
        self._window_step: int | None = None
        self._window_active: dict[str, Any] = {}
        self._pending_counts: dict[str, dict[str, Any]] = {}
        self._weight_decay_groups: list[tuple[dict[str, Any], float]] = []

    def bind_model(self, model: Any) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Domain expert specialization requires torch.")
        if self._handles:
            raise RuntimeError("Domain expert specialization is already bound.")
        for name, module in model.named_modules():
            router = getattr(module, "router", None)
            experts = getattr(module, "experts", None)
            adapters = getattr(experts, "expert_lora", None)
            if router is None or not adapters or not hasattr(router, "num_experts"):
                continue
            num_experts = int(router.num_experts)
            if self.min_experts > num_experts:
                raise ValueError(
                    f"min_experts={self.min_experts} exceeds {name} experts={num_experts}."
                )
            setter = getattr(experts, "set_routed_adapter_gate", None)
            self.bind_layer(
                name,
                router=router,
                adapters=tuple(adapters.values()),
                set_route_gate=setter if callable(setter) else None,
            )
        if not self._layers:
            raise ValueError(
                "Domain expert specialization requires routed expert LoRA adapters."
            )

    def bind_layer(
        self,
        name: str,
        *,
        router: Any,
        adapters: tuple[Any, ...],
        set_route_gate: Callable[[Any | None], None] | None,
    ) -> None:
        layer = str(name)
        if layer in self._layers:
            raise ValueError(f"Domain expert layer {layer!r} is already bound.")
        if not adapters:
            raise ValueError(f"Domain expert layer {layer!r} has no adapters.")
        num_experts = int(router.num_experts)
        if self.min_experts > num_experts:
            raise ValueError(
                f"min_experts={self.min_experts} exceeds {layer} experts={num_experts}."
            )
        self._layers[layer] = _LayerBinding(
            router=router,
            adapters=tuple(adapters),
            set_route_gate=set_route_gate,
        )

        def _observe(r, _inputs, _output, *, layer_name=layer, count=num_experts):
            self._observe_router(layer_name, r, count)

        self._handles.append(router.register_forward_hook(_observe))

    def bind_batch(self, *, domains: tuple[str, ...], step: int, training: bool) -> None:
        normalized = tuple(str(value).strip() for value in domains)
        if training and (not normalized or any(not value for value in normalized)):
            raise ValueError("Every training sample requires a non-empty domain.")
        unique = set(normalized)
        self._domain = normalized[0] if normalized else ""
        self._domains = normalized
        self._step = int(step)
        self._training = bool(training)
        if self._window_step != self._step:
            self._window_step = self._step
            self._window_active.clear()
        for layer, binding in self._layers.items():
            selected_by_domain = {
                domain: self._selected(layer, domain) for domain in unique
            }
            mixed = training and len(unique) > 1 and step >= self.warmup_steps
            if mixed and binding.set_route_gate is None:
                raise RuntimeError(
                    f"Layer {layer!r} does not support mixed-domain route gates."
                )
            if mixed and any(
                selected_by_domain[domain] is None for domain in unique
            ):
                missing = sorted(
                    domain
                    for domain in unique
                    if selected_by_domain[domain] is None
                )
                raise RuntimeError(
                    "Mixed-domain specialization has no learned affinity for: "
                    + ", ".join(missing)
                )
            union = sorted(
                {
                    expert
                    for selected in selected_by_domain.values()
                    if selected is not None
                    for expert in selected
                }
            )
            for adapter in binding.adapters:
                clear = getattr(adapter, "clear_active_experts", None)
                activate = getattr(adapter, "set_active_experts", None)
                selected = selected_by_domain.get(self._domain)
                if (
                    not training
                    or step < self.warmup_steps
                    or selected is None
                ):
                    if callable(clear):
                        clear()
                elif callable(activate):
                    activate(union if mixed else selected)
            if binding.set_route_gate is not None:
                if not mixed:
                    binding.set_route_gate(None)
                else:
                    mask = torch.zeros(
                        (len(normalized), int(binding.router.num_experts)),
                        dtype=torch.bool,
                    )
                    for sample, domain in enumerate(normalized):
                        mask[sample, list(selected_by_domain[domain])] = True
                    binding.set_route_gate(RoutedAdapterGate(mask))
            if training and step >= self.warmup_steps:
                active = torch.zeros(
                    int(binding.router.num_experts), dtype=torch.bool
                )
                active[union] = True
                prior = self._window_active.get(layer)
                self._window_active[layer] = active if prior is None else prior | active

    def _observe_router(self, layer: str, router: Any, num_experts: int) -> None:
        if not self._training or not self._domains:
            return
        top = getattr(router, "last_top_indices", None)
        if top is None:
            return
        rows = self._pending_counts.setdefault(layer, {})
        routes = top.detach().reshape(len(self._domains), -1).to(torch.long)
        for domain, domain_routes in zip(self._domains, routes):
            counts = torch.bincount(
                domain_routes, minlength=num_experts
            ).detach().to(device="cpu", dtype=torch.float64)
            prior = rows.get(domain)
            rows[domain] = counts if prior is None else prior + counts

    def _selected(self, layer: str, domain: str) -> tuple[int, ...] | None:
        scores = self._scores.get(layer, {}).get(domain)
        if scores is None or float(scores.max().item()) <= 0.0:
            return None
        values = [float(value) for value in scores.tolist()]
        cutoff = max(values) * self.affinity_threshold
        selected = [idx for idx, value in enumerate(values) if value >= cutoff]
        if len(selected) < self.min_experts:
            order = sorted(range(len(values)), key=lambda idx: (-values[idx], idx))
            selected = order[: self.min_experts]
        return tuple(sorted(selected))

    def before_optimizer_step(self, optimizer: Any) -> None:
        if not self._training or self._step < self.warmup_steps:
            return
        masks_by_param: dict[int, Any] = {}
        for layer, binding in self._layers.items():
            active = self._window_active.get(layer)
            if active is None:
                continue
            for adapter in binding.adapters:
                inactive = ~active
                for param in (adapter.lora_a, adapter.lora_b):
                    masks_by_param[id(param)] = active
                    state = optimizer.state.get(param, {})
                    for value in state.values():
                        if not torch.is_tensor(value) or value.numel() <= 1:
                            continue
                        if value.numel() != param.numel():
                            continue
                        view = value.reshape_as(param)
                        view[inactive.to(device=view.device)] = 0
        self._apply_selective_weight_decay(optimizer, masks_by_param)

    def _apply_selective_weight_decay(
        self, optimizer: Any, masks_by_param: Mapping[int, Any]
    ) -> None:
        self._weight_decay_groups.clear()
        with torch.no_grad():
            for group in optimizer.param_groups:
                weight_decay = float(group.get("weight_decay", 0.0))
                if weight_decay == 0.0:
                    continue
                lr = float(group["lr"])
                factor = 1.0 - lr * weight_decay
                for param in group["params"]:
                    if getattr(param, "grad", None) is None:
                        continue
                    active = masks_by_param.get(id(param))
                    if active is None:
                        param.mul_(factor)
                    else:
                        scale = torch.where(
                            active.to(device=param.device),
                            torch.tensor(factor, device=param.device),
                            torch.tensor(1.0, device=param.device),
                        ).to(dtype=param.dtype)
                        param.mul_(
                            scale.reshape(-1, *([1] * (param.dim() - 1)))
                        )
                self._weight_decay_groups.append((group, weight_decay))
                group["weight_decay"] = 0.0

    def after_optimizer_step(self, optimizer: Any, *, applied: bool) -> None:
        _ = optimizer
        for group, weight_decay in self._weight_decay_groups:
            group["weight_decay"] = weight_decay
        self._weight_decay_groups.clear()
        if applied:
            self._commit_pending_counts()
        self._pending_counts.clear()
        self._window_active.clear()
        self._window_step = None

    def _commit_pending_counts(self) -> None:
        for layer, pending_rows in self._pending_counts.items():
            rows = self._scores.setdefault(layer, {})
            for domain, counts in pending_rows.items():
                prior = rows.get(domain)
                if prior is None or self._step < self.warmup_steps:
                    rows[domain] = counts if prior is None else prior + counts
                elif (self._step - self.warmup_steps) % self.update_interval == 0:
                    rows[domain] = prior.mul(self.momentum).add(
                        counts, alpha=1.0 - self.momentum
                    )

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "scores": {
                layer: {domain: values.tolist() for domain, values in rows.items()}
                for layer, rows in self._scores.items()
            }
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        scores = state.get("scores", {})
        if not isinstance(scores, Mapping):
            raise ValueError("Domain expert specialization scores must be a mapping.")
        restored: dict[str, dict[str, Any]] = {}
        for layer, rows in scores.items():
            if not isinstance(rows, Mapping):
                raise ValueError(f"Affinity rows for layer '{layer}' must be a mapping.")
            restored[str(layer)] = {
                str(domain): torch.tensor(values, dtype=torch.float64)
                for domain, values in rows.items()
            }
        self._scores = restored

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "affinity_threshold": self.affinity_threshold,
            "min_experts": self.min_experts,
            "momentum": self.momentum,
            "update_interval": self.update_interval,
        }
