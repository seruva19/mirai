"""Optimizer factory with capability fallback policy."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
import importlib
import math
from typing import Any, Callable
from mirai.core.registry import Registry
from mirai.core.training.optim.prodigy import Prodigy

_log = logging.getLogger(__name__)

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for optimizer setup: {exc}")

_TORCH_ADAMW = torch.optim.AdamW

SELECTED_EXPERT_OPTIMIZER_TYPES = frozenset(
    {
        "selected_expert_adamw",
        "selected_expert_adamw_4_2bit",
        "selected_expert_adam_mini",
        "selected_expert_muon",
        "selected_expert_adamuon",
    }
)


@dataclass
class OptimizerBuildResult:
    optimizer: torch.optim.Optimizer
    resolved_type: str
    used_fallback: bool


@dataclass
class _OptimizerBuildContext:
    """Resolved inputs shared by every registered optimizer builder."""

    opt_params: Any
    lr: float
    weight_decay: float
    has_param_groups: bool
    allow_fallback: bool
    selected_expert_ids: tuple[int, ...] = ()
    selected_expert_plan: dict[str, tuple[int, ...]] | None = None
    selected_expert_named_params: tuple[tuple[str, Any], ...] = ()
    stochastic_rounding: bool = False
    prodigy_beta3: float = 0.0
    prodigy_decouple: bool = True
    prodigy_use_bias_correction: bool = False
    prodigy_safeguard_warmup: bool = False
    prodigy_d0: float = 1e-6
    prodigy_d_coef: float = 1.0
    prodigy_growth_rate: float = math.inf
    prodigy_slice_p: int = 1
    lora_pairs: tuple[Any, ...] = ()
    lora_pro_damping: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_eps: float = 1e-8
    muon_rms_target: float = 0.2
    lora_muon_gauge_rebalance_interval: int = 0
    lora_muon_gauge_rebalance_alpha: float = 1.0


# Builder signature: (context) -> OptimizerBuildResult. Builtins register at
# import (below); third parties add a type via ``@register_optimizer("name")``.
OptimizerBuilder = Callable[[_OptimizerBuildContext], OptimizerBuildResult]
OptimizerRegistry: Registry[OptimizerBuilder] = Registry("optimizer")


def register_optimizer(name: str) -> Callable[[OptimizerBuilder], OptimizerBuilder]:
    return OptimizerRegistry.decorator(name)


class CPUOffloadOptimizer(torch.optim.Optimizer):
    """CPU-master adapter for an already constructed optimizer.

    The wrapped optimizer owns FP32 CPU shadow parameters and state. Model
    parameters remain on their execution device and only LoRA-sized values and
    gradients cross the device boundary at each optimizer step.
    """

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._inner_optimizer = optimizer
        self._model_shadow_pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        shadow_groups: list[dict[str, Any]] = []
        for group in optimizer.param_groups:
            shadow_group = {key: value for key, value in group.items() if key != "params"}
            shadows: list[torch.nn.Parameter] = []
            for model_param in group["params"]:
                shadow = torch.nn.Parameter(
                    model_param.detach().to(device="cpu", dtype=torch.float32).clone(),
                    requires_grad=True,
                )
                shadows.append(shadow)
                self._model_shadow_pairs.append((model_param, shadow))
            shadow_group["params"] = shadows
            shadow_groups.append(shadow_group)
        super().__init__(shadow_groups, dict(getattr(optimizer, "defaults", {})))
        optimizer.param_groups = self.param_groups
        optimizer.state = self.state
        self._last_transfer_ops = 0

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if closure is not None:
            raise RuntimeError("CPUOffloadOptimizer does not support optimizer closures.")
        transfer_ops = 0
        for model_param, shadow in self._model_shadow_pairs:
            cpu_grad = getattr(model_param, "_mirai_cpu_grad", None)
            grad = cpu_grad if isinstance(cpu_grad, torch.Tensor) else model_param.grad
            if grad is None:
                shadow.grad = None
                continue
            shadow.grad = (
                grad.detach()
                if grad.device.type == "cpu" and grad.dtype == torch.float32
                else grad.detach().to(device="cpu", dtype=torch.float32)
            )
            transfer_ops += 1
        loss = self._inner_optimizer.step()
        for model_param, shadow in self._model_shadow_pairs:
            model_param.copy_(shadow.to(device=model_param.device, dtype=model_param.dtype))
            transfer_ops += 1
        self._last_transfer_ops = transfer_ops
        return loss

    def supports_offloaded_gradients(self) -> bool:
        return True

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._inner_optimizer.zero_grad(set_to_none=set_to_none)
        for model_param, _shadow in self._model_shadow_pairs:
            if hasattr(model_param, "_mirai_cpu_grad"):
                delattr(model_param, "_mirai_cpu_grad")
            if set_to_none:
                model_param.grad = None
            elif model_param.grad is not None:
                model_param.grad.zero_()

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        # Pipeline weights are restored before optimizer state. Optimizer state
        # does not contain parameter values, so refresh the CPU masters here.
        for model_param, shadow in self._model_shadow_pairs:
            shadow.copy_(model_param.detach().to(device="cpu", dtype=torch.float32))
        self._inner_optimizer.param_groups = self.param_groups
        self._inner_optimizer.state = self.state

    def consume_transfer_ops(self) -> int:
        value = int(self._last_transfer_ops)
        self._last_transfer_ops = 0
        return value


def build_cpu_offload_optimizer(optimizer: torch.optim.Optimizer) -> CPUOffloadOptimizer:
    if isinstance(optimizer, CPUOffloadOptimizer):
        return optimizer
    if optimizer.state:
        raise ValueError("CPU optimizer offload must be configured before the first update.")
    return CPUOffloadOptimizer(optimizer)


def _iter_optimizer_params(opt_params: Any) -> Any:
    for item in opt_params:
        if isinstance(item, dict):
            yield from item.get("params", [])
        else:
            yield item


def _all_params_on_cuda(opt_params: Any) -> bool:
    """True only when every optimizer parameter lives on a CUDA device.

    bitsandbytes 8-bit optimizers execute their update kernels on GPU tensors;
    if the trainable parameters are still on CPU (e.g. device dispatch withheld
    from a run), the 8-bit step fails at execution time. A ``cuda.is_available()``
    probe is therefore insufficient — the parameters themselves must be resident
    on CUDA for the 8-bit path to be usable.
    """
    params = list(_iter_optimizer_params(opt_params))
    if not params:
        return False
    return all(
        getattr(getattr(p, "device", None), "type", "cpu") == "cuda" for p in params
    )


def _safe_adamw(
    *,
    opt_params: Any,
    lr: float,
    weight_decay: float,
    has_param_groups: bool,
) -> Any:
    if has_param_groups:
        return _TORCH_ADAMW(opt_params, lr=lr)
    return _TORCH_ADAMW(opt_params, lr=lr, weight_decay=weight_decay)


def warmup_optimizer_warning(*, optimizer_type: str, warmup_steps: int) -> str | None:
    key = optimizer_type.strip().lower()
    if key == "prodigy" and warmup_steps > 0:
        return (
            "optimizer.type='prodigy' with warmup_steps > 0 applies an external warmup on top "
            "of Prodigy dynamics; this is usually discouraged. Prefer warmup_steps=0."
        )
    return None


def _is_router_parameter(name: str) -> bool:
    """Match sparse-MoE router parameters by module path segment.

    Covers both a directly tuned router weight and adapter factors attached to
    it; ``routed``/``routing`` siblings do not contain the ``router`` segment.
    """
    return "router" in name.lower()


def build_param_groups(
    *,
    named_params,
    base_lr: float,
    weight_decay: float,
    weight_decay_filter: str,
    loraplus_lr_ratio: float,
    module_lr_multipliers: dict[str, float] | None = None,
) -> list[dict]:
    named = list(named_params)
    mode = weight_decay_filter.strip().lower()
    if mode not in {"none", "lora_b_bias", "router_aware"}:
        raise ValueError(f"Unsupported weight_decay_filter '{weight_decay_filter}'.")

    grouped: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    multipliers = {
        str(k).strip().lower(): float(v)
        for k, v in dict(module_lr_multipliers or {}).items()
        if str(k).strip()
    }

    def _module_multiplier(name: str) -> float:
        key = name.lower()
        if "cross_attn" in key or ("cross" in key and "attn" in key):
            return float(multipliers.get("cross_attn", 1.0))
        if "self_attn" in key or ("attn" in key and "cross" not in key and "temporal" not in key):
            return float(multipliers.get("self_attn", 1.0))
        # Router parameters are nested under the sparse-MoE feed-forward module
        # (``ffn.router.weight``), so this branch must precede the ffn branch to
        # be reachable at all. Absent an explicit "router" entry it resolves to
        # the ffn multiplier, which keeps the ffn group's meaning unchanged.
        if _is_router_parameter(key):
            return float(multipliers.get("router", multipliers.get("ffn", 1.0)))
        if "ffn" in key or "mlp" in key:
            return float(multipliers.get("ffn", 1.0))
        if "temporal" in key:
            return float(multipliers.get("temporal", 1.0))
        return float(multipliers.get("other", 1.0))

    def _lr_for_name(name: str) -> float:
        lr = float(base_lr) * _module_multiplier(name)
        if "lora_b" in name.lower():
            lr *= float(loraplus_lr_ratio)
        return lr

    for name, param in named:
        key = name.lower()
        if key.startswith("objective.adaptive_weighting."):
            # EDM2 optimizes the uncertainty head without decoupled decay.
            decay_value = 0.0
        elif mode == "none":
            decay_value = float(weight_decay)
        elif key.endswith("bias") or "lora_b" in key:
            decay_value = 0.0
        elif mode == "router_aware" and _is_router_parameter(key):
            # Decay pulls router logits toward zero, which flattens the routing
            # distribution the frozen checkpoint encodes.
            decay_value = 0.0
        else:
            decay_value = float(weight_decay)
        lr_value = _lr_for_name(name)
        grouped.setdefault((lr_value, decay_value), []).append(param)

    out: list[dict] = []
    for (lr_value, decay_value), params_in_group in sorted(grouped.items(), key=lambda item: item[0]):
        out.append({"params": params_in_group, "lr": lr_value, "weight_decay": decay_value})
    return out


def build_optimizer(
    *,
    params,
    named_params=None,
    optimizer_type: str,
    lr: float,
    weight_decay: float,
    weight_decay_filter: str = "none",
    loraplus_lr_ratio: float = 1.0,
    module_lr_multipliers: dict[str, float] | None = None,
    allow_fallback: bool,
    selected_expert_ids: Any = (),
    selected_expert_plan: dict[str, tuple[int, ...]] | None = None,
    stochastic_rounding: bool = False,
    prodigy_beta3: float = 0.0,
    prodigy_decouple: bool = True,
    prodigy_use_bias_correction: bool = False,
    prodigy_safeguard_warmup: bool = False,
    prodigy_d0: float = 1e-6,
    prodigy_d_coef: float = 1.0,
    prodigy_growth_rate: float = math.inf,
    prodigy_slice_p: int = 1,
    lora_pairs: tuple[Any, ...] = (),
    lora_pro_damping: float = 1e-8,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    muon_eps: float = 1e-8,
    muon_rms_target: float = 0.2,
    lora_muon_gauge_rebalance_interval: int = 0,
    lora_muon_gauge_rebalance_alpha: float = 1.0,
) -> OptimizerBuildResult:
    param_groups = None
    selected_expert_optimizer = (
        str(optimizer_type).strip().lower() in SELECTED_EXPERT_OPTIMIZER_TYPES
    )
    if named_params is not None and not selected_expert_optimizer:
        param_groups = build_param_groups(
            named_params=named_params,
            base_lr=lr,
            weight_decay=weight_decay,
            weight_decay_filter=weight_decay_filter,
            loraplus_lr_ratio=loraplus_lr_ratio,
            module_lr_multipliers=module_lr_multipliers,
        )
    opt_params = param_groups if param_groups is not None else params

    ctx = _OptimizerBuildContext(
        opt_params=opt_params,
        lr=lr,
        weight_decay=weight_decay,
        has_param_groups=param_groups is not None,
        allow_fallback=allow_fallback,
        selected_expert_ids=tuple(int(value) for value in selected_expert_ids),
        selected_expert_plan={
            str(name): tuple(int(value) for value in ids)
            for name, ids in dict(selected_expert_plan or {}).items()
        },
        selected_expert_named_params=tuple(named_params or ()),
        stochastic_rounding=bool(stochastic_rounding),
        prodigy_beta3=float(prodigy_beta3),
        prodigy_decouple=bool(prodigy_decouple),
        prodigy_use_bias_correction=bool(prodigy_use_bias_correction),
        prodigy_safeguard_warmup=bool(prodigy_safeguard_warmup),
        prodigy_d0=float(prodigy_d0),
        prodigy_d_coef=float(prodigy_d_coef),
        prodigy_growth_rate=float(prodigy_growth_rate),
        prodigy_slice_p=int(prodigy_slice_p),
        lora_pairs=tuple(lora_pairs),
        lora_pro_damping=float(lora_pro_damping),
        muon_momentum=float(muon_momentum),
        muon_nesterov=bool(muon_nesterov),
        muon_ns_steps=int(muon_ns_steps),
        muon_eps=float(muon_eps),
        muon_rms_target=float(muon_rms_target),
        lora_muon_gauge_rebalance_interval=int(
            lora_muon_gauge_rebalance_interval
        ),
        lora_muon_gauge_rebalance_alpha=float(
            lora_muon_gauge_rebalance_alpha
        ),
    )
    builder = OptimizerRegistry.get(optimizer_type)
    return builder(ctx)


@register_optimizer("adamw")
def _build_adamw(ctx: _OptimizerBuildContext) -> OptimizerBuildResult:
    if ctx.stochastic_rounding:
        from mirai.core.training.optim.stochastic_rounding import (
            StochasticRoundingAdamW,
        )

        optimizer = StochasticRoundingAdamW(
            ctx.opt_params,
            lr=ctx.lr,
            weight_decay=0.0 if ctx.has_param_groups else ctx.weight_decay,
        )
        return OptimizerBuildResult(
            optimizer=optimizer,
            resolved_type="adamw",
            used_fallback=False,
        )
    opt = _safe_adamw(
        opt_params=ctx.opt_params,
        lr=ctx.lr,
        weight_decay=ctx.weight_decay,
        has_param_groups=ctx.has_param_groups,
    )
    return OptimizerBuildResult(optimizer=opt, resolved_type="adamw", used_fallback=False)


@register_optimizer("lora_pro_adamw")
def _build_lora_pro_adamw(ctx: _OptimizerBuildContext) -> OptimizerBuildResult:
    from mirai.core.training.optim.lora_pro import LoRAProAdamW

    optimizer = LoRAProAdamW(
        ctx.opt_params,
        pairs=ctx.lora_pairs,
        lr=ctx.lr,
        weight_decay=0.0 if ctx.has_param_groups else ctx.weight_decay,
        damping=ctx.lora_pro_damping,
        stochastic_rounding=ctx.stochastic_rounding,
    )
    return OptimizerBuildResult(
        optimizer=optimizer,
        resolved_type="lora_pro_adamw",
        used_fallback=False,
    )


@register_optimizer("lora_muon")
def _build_lora_muon(ctx: _OptimizerBuildContext) -> OptimizerBuildResult:
    from mirai.core.training.optim.lora_muon import LoRAMuon

    optimizer = LoRAMuon(
        ctx.opt_params,
        pairs=ctx.lora_pairs,
        lr=ctx.lr,
        momentum=ctx.muon_momentum,
        weight_decay=0.0 if ctx.has_param_groups else ctx.weight_decay,
        gauge_rebalance_interval=ctx.lora_muon_gauge_rebalance_interval,
        gauge_rebalance_alpha=ctx.lora_muon_gauge_rebalance_alpha,
        stochastic_rounding=ctx.stochastic_rounding,
    )
    return OptimizerBuildResult(
        optimizer=optimizer,
        resolved_type="lora_muon",
        used_fallback=False,
    )


@register_optimizer("selected_expert_adamw")
def _build_selected_expert_adamw(ctx: _OptimizerBuildContext) -> OptimizerBuildResult:
    return _build_selected_expert_adamw_family(ctx, low_bit=False)


def _build_selected_expert_adamw_family(
    ctx: _OptimizerBuildContext,
    *,
    low_bit: bool,
) -> OptimizerBuildResult:
    if ctx.has_param_groups:
        raise ValueError(
            "selected_expert_adamw requires ungrouped expert tensors; "
            "weight-decay and LoRA+ parameter grouping must be disabled."
        )
    from mirai.core.training.optim.selected_expert_adamw import SelectedExpertAdamW
    from mirai.core.training.optim.low_bit_state import (
        SOLO_4_2_BETAS,
        SOLO_4_2_STATE_FORMAT,
    )

    optimizer = SelectedExpertAdamW(
        ctx.opt_params,
        expert_ids=ctx.selected_expert_ids,
        named_params=ctx.selected_expert_named_params,
        expert_ids_by_name=ctx.selected_expert_plan,
        lr=ctx.lr,
        weight_decay=ctx.weight_decay,
        stochastic_rounding=ctx.stochastic_rounding,
        betas=SOLO_4_2_BETAS if low_bit else (0.9, 0.999),
        state_format=SOLO_4_2_STATE_FORMAT if low_bit else "native",
    )
    resolved_type = (
        "selected_expert_adamw_4_2bit" if low_bit else "selected_expert_adamw"
    )
    return OptimizerBuildResult(
        optimizer=optimizer,
        resolved_type=resolved_type,
        used_fallback=False,
    )


@register_optimizer("selected_expert_adamw_4_2bit")
def _build_selected_expert_adamw_4_2bit(
    ctx: _OptimizerBuildContext,
) -> OptimizerBuildResult:
    return _build_selected_expert_adamw_family(ctx, low_bit=True)


@register_optimizer("selected_expert_adam_mini")
def _build_selected_expert_adam_mini(
    ctx: _OptimizerBuildContext,
) -> OptimizerBuildResult:
    if ctx.has_param_groups:
        raise ValueError(
            "selected_expert_adam_mini requires ungrouped expert tensors; "
            "weight-decay and LoRA+ parameter grouping must be disabled."
        )
    from mirai.core.training.optim.selected_expert_adam_mini import (
        SelectedExpertAdamMini,
    )

    optimizer = SelectedExpertAdamMini(
        ctx.opt_params,
        expert_ids=ctx.selected_expert_ids,
        named_params=ctx.selected_expert_named_params,
        expert_ids_by_name=ctx.selected_expert_plan,
        lr=ctx.lr,
        weight_decay=ctx.weight_decay,
        stochastic_rounding=ctx.stochastic_rounding,
    )
    return OptimizerBuildResult(
        optimizer=optimizer,
        resolved_type="selected_expert_adam_mini",
        used_fallback=False,
    )


def _build_selected_expert_muon_family(
    ctx: _OptimizerBuildContext,
    *,
    adaptive: bool,
) -> OptimizerBuildResult:
    if ctx.has_param_groups:
        raise ValueError(
            "Selected-expert Muon requires ungrouped expert tensors; "
            "parameter grouping must be disabled."
        )
    from mirai.core.training.optim.selected_expert_muon import (
        SelectedExpertAdaMuon,
        SelectedExpertMuon,
    )

    optimizer_type = (
        "selected_expert_adamuon" if adaptive else "selected_expert_muon"
    )
    optimizer_cls = SelectedExpertAdaMuon if adaptive else SelectedExpertMuon
    optimizer = optimizer_cls(
        ctx.opt_params,
        expert_ids=ctx.selected_expert_ids,
        named_params=ctx.selected_expert_named_params,
        expert_ids_by_name=ctx.selected_expert_plan,
        lr=ctx.lr,
        weight_decay=ctx.weight_decay,
        momentum=ctx.muon_momentum,
        nesterov=ctx.muon_nesterov,
        ns_steps=ctx.muon_ns_steps,
        eps=ctx.muon_eps,
        rms_target=ctx.muon_rms_target,
        stochastic_rounding=ctx.stochastic_rounding,
    )
    return OptimizerBuildResult(
        optimizer=optimizer,
        resolved_type=optimizer_type,
        used_fallback=False,
    )


@register_optimizer("selected_expert_muon")
def _build_selected_expert_muon(
    ctx: _OptimizerBuildContext,
) -> OptimizerBuildResult:
    return _build_selected_expert_muon_family(ctx, adaptive=False)


@register_optimizer("selected_expert_adamuon")
def _build_selected_expert_adamuon(
    ctx: _OptimizerBuildContext,
) -> OptimizerBuildResult:
    return _build_selected_expert_muon_family(ctx, adaptive=True)


@register_optimizer("prodigy")
def _build_prodigy(ctx: _OptimizerBuildContext) -> OptimizerBuildResult:
    opt = Prodigy(
        ctx.opt_params,
        lr=ctx.lr,
        weight_decay=0.0 if ctx.has_param_groups else ctx.weight_decay,
        beta3=None if ctx.prodigy_beta3 == 0.0 else ctx.prodigy_beta3,
        decouple=ctx.prodigy_decouple,
        use_bias_correction=ctx.prodigy_use_bias_correction,
        safeguard_warmup=ctx.prodigy_safeguard_warmup,
        d0=ctx.prodigy_d0,
        d_coef=ctx.prodigy_d_coef,
        growth_rate=ctx.prodigy_growth_rate,
        slice_p=ctx.prodigy_slice_p,
    )
    return OptimizerBuildResult(optimizer=opt, resolved_type="prodigy", used_fallback=False)


@register_optimizer("adamw_8bit")
def _build_adamw_8bit(ctx: _OptimizerBuildContext) -> OptimizerBuildResult:
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("bitsandbytes AdamW8bit requires CUDA runtime.")
        if not _all_params_on_cuda(ctx.opt_params):
            raise RuntimeError(
                "bitsandbytes AdamW8bit requires trainable parameters resident on CUDA."
            )
        bnb = importlib.import_module("bitsandbytes")
        if not hasattr(getattr(bnb, "optim", None), "AdamW8bit"):
            raise RuntimeError("bitsandbytes install lacks AdamW8bit.")
        if not ctx.has_param_groups:
            opt = bnb.optim.AdamW8bit(ctx.opt_params, lr=ctx.lr, weight_decay=ctx.weight_decay)
        else:
            opt = bnb.optim.AdamW8bit(ctx.opt_params, lr=ctx.lr)
        return OptimizerBuildResult(
            optimizer=opt,
            resolved_type="adamw_8bit",
            used_fallback=False,
        )
    except Exception as exc:
        if not ctx.allow_fallback:
            raise RuntimeError(
                "optimizer.type='adamw_8bit' requested but bitsandbytes is unavailable."
            ) from exc
        warnings.warn(
            f"optimizer.type='adamw_8bit' unavailable ({exc}); falling back to adamw.",
            stacklevel=2,
        )
        opt = _safe_adamw(
            opt_params=ctx.opt_params,
            lr=ctx.lr,
            weight_decay=ctx.weight_decay,
            has_param_groups=ctx.has_param_groups,
        )
        return OptimizerBuildResult(optimizer=opt, resolved_type="adamw", used_fallback=True)


@register_optimizer("adafactor")
def _build_adafactor(ctx: _OptimizerBuildContext) -> OptimizerBuildResult:
    try:
        # Prefer transformers.optimization.Adafactor, then pytorch_optimizer.
        try:
            tf_opt = importlib.import_module("transformers.optimization")
            Adafactor = tf_opt.Adafactor
        except (ModuleNotFoundError, AttributeError):
            pytorch_opt = importlib.import_module("pytorch_optimizer")
            Adafactor = pytorch_opt.Adafactor
        # Adafactor manages its own LR schedule; pass lr=None for adaptive mode.
        if not ctx.has_param_groups:
            opt = Adafactor(
                ctx.opt_params,
                lr=ctx.lr,
                weight_decay=ctx.weight_decay,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            opt = Adafactor(ctx.opt_params, lr=ctx.lr, relative_step=False, scale_parameter=False)
        return OptimizerBuildResult(
            optimizer=opt, resolved_type="adafactor", used_fallback=False
        )
    except Exception as exc:
        if not ctx.allow_fallback:
            raise RuntimeError(
                "optimizer.type='adafactor' requested but neither transformers nor "
                "pytorch_optimizer is available."
            ) from exc
        warnings.warn(
            f"optimizer.type='adafactor' unavailable ({exc}); falling back to adamw.",
            stacklevel=2,
        )
        opt = _safe_adamw(
            opt_params=ctx.opt_params,
            lr=ctx.lr,
            weight_decay=ctx.weight_decay,
            has_param_groups=ctx.has_param_groups,
        )
        return OptimizerBuildResult(optimizer=opt, resolved_type="adamw", used_fallback=True)


@register_optimizer("lion")
def _build_lion(ctx: _OptimizerBuildContext) -> OptimizerBuildResult:
    try:
        pytorch_opt = importlib.import_module("pytorch_optimizer")
        LionCls = pytorch_opt.Lion
    except (ModuleNotFoundError, AttributeError):
        from mirai.core.training.optim.lion import LionFallback as LionCls  # type: ignore[assignment]
    if not ctx.has_param_groups:
        opt = LionCls(ctx.opt_params, lr=ctx.lr, weight_decay=ctx.weight_decay)
    else:
        opt = LionCls(ctx.opt_params, lr=ctx.lr)
    return OptimizerBuildResult(optimizer=opt, resolved_type="lion", used_fallback=False)


@register_optimizer("came")
def _build_came(ctx: _OptimizerBuildContext) -> OptimizerBuildResult:
    try:
        pytorch_opt = importlib.import_module("pytorch_optimizer")
        CAME = pytorch_opt.CAME
        if not ctx.has_param_groups:
            opt = CAME(ctx.opt_params, lr=ctx.lr, weight_decay=ctx.weight_decay)
        else:
            opt = CAME(ctx.opt_params, lr=ctx.lr)
        return OptimizerBuildResult(optimizer=opt, resolved_type="came", used_fallback=False)
    except (ModuleNotFoundError, AttributeError) as exc_came:
        if not ctx.allow_fallback:
            raise RuntimeError(
                "optimizer.type='came' requested but pytorch_optimizer.CAME "
                "is unavailable and optimizer.allow_fallback=false."
            ) from exc_came
        # First fallback: adafactor (same memory-efficient spirit)
        try:
            try:
                tf_opt = importlib.import_module("transformers.optimization")
                Adafactor = tf_opt.Adafactor
            except (ModuleNotFoundError, AttributeError):
                pytorch_opt2 = importlib.import_module("pytorch_optimizer")
                Adafactor = pytorch_opt2.Adafactor
            warnings.warn(
                f"optimizer.type='came' unavailable ({exc_came}); falling back to adafactor.",
                stacklevel=2,
            )
            if not ctx.has_param_groups:
                opt = Adafactor(
                    ctx.opt_params,
                    lr=ctx.lr,
                    weight_decay=ctx.weight_decay,
                    relative_step=False,
                    scale_parameter=False,
                )
            else:
                opt = Adafactor(
                    ctx.opt_params, lr=ctx.lr, relative_step=False, scale_parameter=False
                )
            return OptimizerBuildResult(
                optimizer=opt, resolved_type="adafactor", used_fallback=True
            )
        except Exception:
            warnings.warn(
                "optimizer.type='came' and adafactor both unavailable; falling back to adamw.",
                stacklevel=2,
            )
            opt = _safe_adamw(
                opt_params=ctx.opt_params,
                lr=ctx.lr,
                weight_decay=ctx.weight_decay,
                has_param_groups=ctx.has_param_groups,
            )
        return OptimizerBuildResult(
            optimizer=opt, resolved_type="adamw", used_fallback=True
        )


@register_optimizer("paged_adamw_8bit")
def _build_paged_adamw_8bit(ctx: _OptimizerBuildContext) -> OptimizerBuildResult:
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("bitsandbytes PagedAdamW8bit requires CUDA runtime.")
        if not _all_params_on_cuda(ctx.opt_params):
            raise RuntimeError(
                "bitsandbytes PagedAdamW8bit requires trainable parameters resident on CUDA."
            )
        bnb = importlib.import_module("bitsandbytes")
        optimizer_cls = getattr(getattr(bnb, "optim", None), "PagedAdamW8bit", None)
        if optimizer_cls is None:
            raise RuntimeError("bitsandbytes install lacks PagedAdamW8bit.")
        if not ctx.has_param_groups:
            opt = optimizer_cls(ctx.opt_params, lr=ctx.lr, weight_decay=ctx.weight_decay)
        else:
            opt = optimizer_cls(ctx.opt_params, lr=ctx.lr)
        return OptimizerBuildResult(
            optimizer=opt,
            resolved_type="paged_adamw_8bit",
            used_fallback=False,
        )
    except Exception as exc:
        if not ctx.allow_fallback:
            raise RuntimeError(
                "optimizer.type='paged_adamw_8bit' requested but "
                "bitsandbytes PagedAdamW8bit is unavailable."
            ) from exc
        warnings.warn(
            "optimizer.type='paged_adamw_8bit' unavailable "
            f"({exc}); falling back to adamw.",
            stacklevel=2,
        )
        opt = _safe_adamw(
            opt_params=ctx.opt_params,
            lr=ctx.lr,
            weight_decay=ctx.weight_decay,
            has_param_groups=ctx.has_param_groups,
        )
        return OptimizerBuildResult(
            optimizer=opt, resolved_type="adamw", used_fallback=True
        )


def offload_optimizer_state_to_cpu(optimizer: torch.optim.Optimizer) -> int:
    if isinstance(optimizer, CPUOffloadOptimizer):
        return optimizer.consume_transfer_ops()
    moved = 0
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.detach().cpu().float()
                moved += 1
    return moved
