"""Loss implementations returning per-sample loss values."""

from __future__ import annotations

import math

from mirai.core.registry import register_loss
from mirai.core.tensors import is_torch_tensor


@register_loss("mse")
def mse_loss(pred, target):
    if is_torch_tensor(pred):
        return (pred - target) ** 2
    return [(p - t) ** 2 for p, t in zip(pred, target, strict=True)]


@register_loss("huber")
def huber_loss(pred, target, delta: float = 1.0):
    if is_torch_tensor(pred):
        import torch

        err = (pred - target).abs()
        quadratic = 0.5 * err * err
        linear = delta * (err - 0.5 * delta)
        return torch.where(err <= delta, quadratic, linear)
    out: list[float] = []
    for p, t in zip(pred, target, strict=True):
        err = abs(p - t)
        if err <= delta:
            out.append(0.5 * err * err)
        else:
            out.append(delta * (err - 0.5 * delta))
    return out


@register_loss("pseudo_huber")
def pseudo_huber_loss(pred, target, c: float = 0.1):
    c2 = c * c
    if is_torch_tensor(pred):
        residual = (pred - target) / c
        return c2 * ((1.0 + (residual * residual)).sqrt() - 1.0)
    return [c2 * (math.sqrt(1.0 + ((p - t) / c) ** 2) - 1.0) for p, t in zip(pred, target, strict=True)]
