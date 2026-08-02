"""Adam with Prodigy step-size estimation.

The update follows Algorithm 3 of Mishchenko and Defazio,
"Prodigy: An Expeditiously Adaptive Parameter-Free Learner"
(https://arxiv.org/abs/2306.06101). The implementation is a single-process,
clean-room adaptation informed by the MIT-licensed reference implementation at
https://github.com/konstmish/prodigy.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import torch


class Prodigy(torch.optim.Optimizer):
    """Adam moments with the Prodigy estimate of the distance scale ``d``."""

    def __init__(
        self,
        params: Iterable,
        *,
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        beta3: float | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decouple: bool = True,
        use_bias_correction: bool = False,
        safeguard_warmup: bool = False,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = math.inf,
        slice_p: int = 1,
    ) -> None:
        beta1, beta2 = (float(betas[0]), float(betas[1]))
        resolved_beta3 = math.sqrt(beta2) if beta3 is None else float(beta3)
        if not math.isfinite(float(lr)) or float(lr) <= 0.0:
            raise ValueError("Prodigy lr must be finite and positive.")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("Prodigy betas must lie in [0, 1).")
        if not 0.0 <= resolved_beta3 < 1.0:
            raise ValueError("Prodigy beta3 must lie in [0, 1).")
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("Prodigy eps must be finite and positive.")
        if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
            raise ValueError("Prodigy weight_decay must be finite and non-negative.")
        if not math.isfinite(float(d0)) or float(d0) <= 0.0:
            raise ValueError("Prodigy d0 must be finite and positive.")
        if not math.isfinite(float(d_coef)) or float(d_coef) <= 0.0:
            raise ValueError("Prodigy d_coef must be finite and positive.")
        if float(growth_rate) < 1.0 or math.isnan(float(growth_rate)):
            raise ValueError("Prodigy growth_rate must be at least one.")
        if int(slice_p) < 1:
            raise ValueError("Prodigy slice_p must be at least one.")

        defaults = {
            "lr": float(lr),
            "betas": (beta1, beta2),
            "beta3": resolved_beta3,
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "decouple": bool(decouple),
            "use_bias_correction": bool(use_bias_correction),
            "safeguard_warmup": bool(safeguard_warmup),
            "d": float(d0),
            "d0": float(d0),
            "d_max": float(d0),
            "d_numerator": 0.0,
            "d_denom": 0.0,
            "d_hat": float(d0),
            "d_coef": float(d_coef),
            "growth_rate": float(growth_rate),
            "slice_p": int(slice_p),
            "k": 0,
        }
        super().__init__(params, defaults)

    def _shared_settings(self) -> dict[str, Any]:
        reference = self.param_groups[0]
        shared_keys = (
            "betas",
            "beta3",
            "decouple",
            "use_bias_correction",
            "safeguard_warmup",
            "d0",
            "d_coef",
            "growth_rate",
            "slice_p",
            "k",
            "d",
            "d_max",
            "d_numerator",
        )
        for group in self.param_groups[1:]:
            differing = [key for key in shared_keys if group[key] != reference[key]]
            if differing:
                raise ValueError(
                    "Prodigy parameter groups must share adaptation state and settings; "
                    f"different values found for {', '.join(differing)}."
                )
        return reference

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        reference = self._shared_settings()
        beta1, beta2 = reference["betas"]
        beta3 = float(reference["beta3"])
        k = int(reference["k"])
        d = float(reference["d"])
        d_max = float(reference["d_max"])
        d_coef = float(reference["d_coef"])
        growth_rate = float(reference["growth_rate"])
        use_bias_correction = bool(reference["use_bias_correction"])
        decouple = bool(reference["decouple"])
        lr = max(float(group["lr"]) for group in self.param_groups)
        if use_bias_correction:
            bias_correction = math.sqrt(1.0 - beta2 ** (k + 1)) / (
                1.0 - beta1 ** (k + 1)
            )
        else:
            bias_correction = 1.0
        dlr = d * lr * bias_correction
        d_numerator = float(reference["d_numerator"]) * beta3
        delta_numerator = 0.0
        d_denom = 0.0
        effective_gradients: dict[torch.Tensor, torch.Tensor] = {}

        for group in self.param_groups:
            group_lr = float(group["lr"])
            if group_lr not in {0.0, lr}:
                raise ValueError(
                    "Prodigy parameter-group learning rates must be zero or share "
                    "one positive value."
                )
            decay = float(group["weight_decay"])
            d0 = float(group["d0"])
            slice_p = int(group["slice_p"])
            safeguard_warmup = bool(group["safeguard_warmup"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Prodigy does not support sparse gradients.")
                if decay and not decouple:
                    gradient = gradient.add(parameter, alpha=decay)
                effective_gradients[parameter] = gradient

                state = self.state[parameter]
                if not state:
                    sampled_parameter = parameter.detach().reshape(-1)[::slice_p]
                    state["step"] = 0
                    state["s"] = torch.zeros_like(sampled_parameter)
                    state["p0"] = sampled_parameter.clone()
                    if beta1 > 0.0:
                        state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                if group_lr == 0.0:
                    continue
                sampled_gradient = gradient.reshape(-1)[::slice_p]
                sampled_parameter = parameter.detach().reshape(-1)[::slice_p]
                s = state["s"]
                p0 = state["p0"]
                delta_numerator += (
                    (d / d0)
                    * dlr
                    * torch.dot(sampled_gradient, p0 - sampled_parameter).item()
                )
                if beta1 > 0.0:
                    state["exp_avg"].mul_(beta1).add_(
                        gradient, alpha=d * (1.0 - beta1)
                    )
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=d * d * (1.0 - beta2),
                )
                s_scale = (d / d0) * (d if safeguard_warmup else dlr)
                s.mul_(beta3).add_(sampled_gradient, alpha=s_scale)
                d_denom += float(s.abs().sum().item())

        if d_denom == 0.0:
            return loss

        global_d_numerator = d_numerator + delta_numerator
        d_hat = d_coef * global_d_numerator / d_denom
        if d == float(reference["d0"]):
            d = max(d, d_hat)
        d_max = max(d_max, d_hat)
        d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group["d_numerator"] = global_d_numerator
            group["d_denom"] = d_denom
            group["d"] = d
            group["d_max"] = d_max
            group["d_hat"] = d_hat
            decay = float(group["weight_decay"])
            eps = float(group["eps"])
            for parameter in group["params"]:
                if parameter.grad is None or float(group["lr"]) == 0.0:
                    continue
                state = self.state[parameter]
                state["step"] += 1
                denominator = state["exp_avg_sq"].sqrt().add_(d * eps)
                if decay and decouple:
                    parameter.mul_(1.0 - decay * dlr)
                if beta1 > 0.0:
                    parameter.addcdiv_(state["exp_avg"], denominator, value=-dlr)
                else:
                    parameter.addcdiv_(
                        effective_gradients[parameter], denominator, value=-dlr * d
                    )
            group["k"] = k + 1

        return loss
