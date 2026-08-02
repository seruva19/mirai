"""Opt-in fp32 master copy of trainable router parameters (single owner).

Motivation: with a bf16 working router, a small optimizer update ``lr * grad``
can be below half a bf16 ULP at the weight's magnitude and round-to-nearest
truncates it to zero. The ``moe_router_underflow_fraction`` diagnostic exposes
this condition. FP32 masters retain sub-ULP updates until they cross a BF16
quantization boundary.

Mechanism (classic AMP "master weights", least-invasive form):

* Keep an fp32 master ``nn.Parameter`` for each trainable router parameter.
* The optimizer is built over the *masters* (via :meth:`substitute_named_params`
  on the named-parameter list the optimizer groups). Gradients still land on the
  bf16 *working* copies (they are what the model forward uses), so before each
  optimizer step the clipped working grad is copied up into the master's grad
  (:meth:`sync_grads_to_master`); the optimizer steps the fp32 masters, and the
  bf16 working copies are re-materialised from the masters (:meth:`materialize`).
* Effect: sub-ULP updates accumulate losslessly in fp32 and periodically cross a
  bf16 ULP, so the working copy eventually steps instead of freezing.

Scope and cost:

* Only parameters whose qualified name marks them as router parameters (small —
  the per-layer gate linears) are mastered.
* Entirely inert when no router parameter is trainable (``train_router`` off or
  absent): the manager is empty, nothing is substituted, no fp32 memory is held.
  Default-off via ``model.params.router_fp32_master``.

Checkpointing: :meth:`state_dict` / :meth:`load_state_dict` round-trip the fp32
masters so resume is exact (the working bf16 copy alone would have lost the
sub-ULP fraction that had accumulated in the master).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


def is_router_parameter(name: str, param: Any) -> bool:
    """Router-parameter predicate: name marks it as router state and it trains."""
    if not bool(getattr(param, "requires_grad", False)):
        return False
    if torch is not None and not torch.is_floating_point(param):
        return False
    return "router" in str(name).lower()


class RouterFp32Master:
    """fp32 master weights for the trainable router parameters of a model.

    Built from the model's trainable ``(name, param)`` pairs. Empty (falsy) when
    no parameter matches the router predicate, which keeps the whole feature
    inert with zero memory cost.
    """

    def __init__(
        self,
        named_params: Iterable[tuple[str, Any]],
        *,
        predicate: Callable[[str, Any], bool] = is_router_parameter,
    ) -> None:
        self._entries: list[tuple[str, Any, Any]] = []
        for name, param in named_params:
            if not predicate(name, param):
                continue
            master = nn.Parameter(param.detach().float().clone())
            self._entries.append((str(name), param, master))
        self._by_working_id = {id(working): master for _, working, master in self._entries}

    def __bool__(self) -> bool:
        return bool(self._entries)

    @property
    def mastered_names(self) -> frozenset[str]:
        """Qualified names whose effective storage is the fp32 master."""
        return frozenset(name for name, _, _ in self._entries)

    def substitute_named_params(
        self, named_params: Iterable[tuple[str, Any]]
    ) -> list[tuple[str, Any]]:
        """Return ``named_params`` with router working copies swapped for masters.

        The optimizer groups over this list so it steps the fp32 masters. Params
        that are not mastered pass through untouched.
        """
        return [
            (name, self._by_working_id.get(id(param), param))
            for name, param in named_params
        ]

    def sync_grads_to_master(self) -> None:
        """Copy each (already clipped) working grad up into its master's grad.

        Reads ``_mirai_cpu_grad`` when the trainer offloaded the grad to CPU,
        else ``.grad``. A missing grad clears the master grad so the optimizer
        skips that parameter this step.
        """
        for _, working, master in self._entries:
            grad = getattr(working, "_mirai_cpu_grad", None)
            if grad is None:
                grad = getattr(working, "grad", None)
            if grad is None:
                master.grad = None
                continue
            master.grad = grad.detach().to(device=master.device, dtype=torch.float32)

    def materialize(self) -> None:
        """Re-materialise every bf16 working copy from its fp32 master (cast down)."""
        with torch.no_grad():
            for _, working, master in self._entries:
                working.data.copy_(master.data.to(dtype=working.dtype))

    def clear_working_grads(self) -> None:
        """Drop working-copy grads after a step (the optimizer holds masters, so
        its ``zero_grad`` does not reach the working copies)."""
        for _, working, _ in self._entries:
            working.grad = None
            if hasattr(working, "_mirai_cpu_grad"):
                working._mirai_cpu_grad = None

    def state_dict(self) -> dict[str, Any]:
        return {name: master.detach().cpu().clone() for name, _, master in self._entries}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore masters from ``state`` and re-materialise the working copies."""
        with torch.no_grad():
            for name, working, master in self._entries:
                if name not in state:
                    continue
                tensor = torch.as_tensor(state[name]).to(
                    device=master.device, dtype=torch.float32
                )
                if tuple(tensor.shape) != tuple(master.shape):
                    raise ValueError(
                        f"router_fp32_master '{name}' has shape {tuple(tensor.shape)}, "
                        f"expected {tuple(master.shape)}."
                    )
                master.data.copy_(tensor)
                working.data.copy_(master.data.to(dtype=working.dtype))


__all__ = ["RouterFp32Master", "is_router_parameter"]
