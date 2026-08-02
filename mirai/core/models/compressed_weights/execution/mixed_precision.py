"""Per-expert mixed-precision frozen storage and routed reference execution."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]

from .experts import CompressedGroupedExperts


if torch is not None:

    class _FrozenBf16Projection(nn.Module):
        def __init__(self, weight: Any) -> None:
            super().__init__()
            self.register_buffer(
                "weight",
                torch.as_tensor(weight).detach().to(torch.bfloat16).contiguous(),
            )

        def materialize(self, *, dtype: Any, device: Any) -> Any:
            return self.weight.to(device=device, dtype=dtype)


    class _FrozenProjectionLinear(torch.autograd.Function):
        """Re-materialize frozen projection weights in backward."""

        @staticmethod
        def forward(ctx: Any, inputs: Any, owner: Any, projection: str) -> Any:
            weight = owner.materialize(dtype=inputs.dtype, device=inputs.device)
            output = inputs @ weight.transpose(-2, -1)
            ctx.owner = owner
            ctx.projection = str(projection)
            ctx.input_dtype = inputs.dtype
            return output

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None, None]:
            weight = ctx.owner.materialize(
                dtype=ctx.input_dtype,
                device=grad_output.device,
            )
            return grad_output.to(ctx.input_dtype) @ weight, None, None


    class _CompressedProjection(nn.Module):
        def __init__(
            self,
            weight: Any,
            *,
            projection: str,
            quant_format: str,
            group_sizes: str | int | Sequence[int] | None,
        ) -> None:
            super().__init__()
            self.projection = str(projection)
            self.host = CompressedGroupedExperts.from_empty(
                num_experts=1,
                group_sizes=group_sizes,
                expert_weight_access="active_dequant",
                quant_format=str(quant_format),
            )
            self.host.load_dense_weight(self.projection, weight.unsqueeze(0))

        def materialize(self, *, dtype: Any, device: Any) -> Any:
            return self.host._dequantize_expert(  # noqa: SLF001
                self.projection,
                0,
                dtype=dtype,
                device=device,
            )

        def set_expert_weight_access_policy(self, **kwargs: Any) -> None:
            options = dict(kwargs)
            if str(options.get("expert_weight_access", "")).strip().lower() in {
                "",
                "auto",
                "disabled",
                "full_dequant",
            }:
                options["expert_weight_access"] = "active_dequant"
            self.host.set_expert_weight_access_policy(**options)


class MixedPrecisionGroupedExperts(nn.Module):
    """A persistent heterogeneous expert pool.

    Each one-expert host owns its native packed buffers, so no padded
    highest-precision stack is materialized. The dispatcher preserves duplicate
    routes exactly and defines the mixed-precision parity contract.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        formats: Sequence[str] | Mapping[str, Sequence[str]],
        group_sizes: str | int | Sequence[int] | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("MixedPrecisionGroupedExperts requires torch.")
        super().__init__()
        experts = int(getattr(base, "num_experts"))
        self.num_experts = experts
        parametrizations = getattr(base, "parametrizations", None)
        if parametrizations is not None and any(
            hasattr(parametrizations, key) for key in ("w1", "w2", "w3")
        ):
            raise ValueError(
                "Mixed-precision expert storage requires an adapter "
                "preset that does not target grouped expert tensors."
            )
        self.hosts = nn.ModuleList()
        self.projection_hosts = nn.ModuleList()
        self.formats_by_projection: dict[str, tuple[str, ...]] = {}
        if isinstance(formats, Mapping):
            if set(formats) != {"w1", "w2", "w3"}:
                raise ValueError(
                    "Tensor precision plan must assign w1, w2, and w3."
                )
            resolved_by_projection = {
                key: tuple(str(value).strip().lower() for value in formats[key])
                for key in ("w1", "w2", "w3")
            }
            if any(len(values) != experts for values in resolved_by_projection.values()):
                raise ValueError(
                    "Tensor precision plan must assign every expert projection."
                )
            supported = {
                "bf16",
                "fp8",
                "int8",
                "nf4",
                "gguf_iq4",
                "gguf_iq3",
                "mxfp8_e4m3",
                "mxfp4",
                "nvfp4",
            }
            if any(
                value not in supported
                for values in resolved_by_projection.values()
                for value in values
            ):
                raise ValueError("Tensor precision plan contains an unsupported format.")
            for expert_id in range(experts):
                projections = nn.ModuleDict()
                for key in ("w1", "w2", "w3"):
                    source = getattr(base, key)[expert_id].detach()
                    quant_format = resolved_by_projection[key][expert_id]
                    if quant_format == "bf16":
                        projections[key] = _FrozenBf16Projection(source)
                    else:
                        projections[key] = _CompressedProjection(
                            source,
                            projection=key,
                            quant_format=quant_format,
                            group_sizes=group_sizes,
                        )
                self.projection_hosts.append(projections)
            self.formats_by_projection = resolved_by_projection
            self.formats = ()
        else:
            resolved = tuple(str(value).strip().lower() for value in formats)
            if len(resolved) != experts:
                raise ValueError(
                    "Mixed precision plan must assign every expert exactly once."
                )
            self.formats = resolved
            for expert_id, quant_format in enumerate(resolved):
                if quant_format == "bf16":
                    raise ValueError(
                        "Schema-v1 plans do not support bf16 expert storage."
                    )
                host = CompressedGroupedExperts.from_empty(
                    num_experts=1,
                    group_sizes=group_sizes,
                    expert_weight_access="active_dequant",
                    quant_format=quant_format,
                )
                for key in ("w1", "w2", "w3"):
                    source = getattr(base, key)[expert_id : expert_id + 1]
                    host.load_dense_weight(key, source)
                self.hosts.append(host)

    def forward(
        self,
        tokens: Any,
        top_scores: Any,
        top_indices: Any,
    ) -> Any:
        original_shape = tuple(tokens.shape)
        flat = tokens.reshape(-1, original_shape[-1])
        scores = top_scores.reshape(flat.shape[0], -1)
        indices = top_indices.reshape(flat.shape[0], -1)
        output = torch.zeros_like(flat)
        tensor_plan = len(self.projection_hosts) > 0
        owner_modules = self.projection_hosts if tensor_plan else self.hosts
        for expert_id, host in enumerate(owner_modules):
            positions = torch.nonzero(indices == expert_id, as_tuple=False)
            if positions.numel() == 0:
                continue
            token_ids = positions[:, 0]
            route_slots = positions[:, 1]
            selected = flat.index_select(0, token_ids)
            if tensor_plan:
                gate = _FrozenProjectionLinear.apply(selected, host["w1"], "w1")
                up = _FrozenProjectionLinear.apply(selected, host["w3"], "w3")
                hidden = torch.nn.functional.silu(gate) * up
                routed = _FrozenProjectionLinear.apply(hidden, host["w2"], "w2")
            else:
                local_scores = torch.ones(
                    (selected.shape[0], 1),
                    dtype=scores.dtype,
                    device=scores.device,
                )
                local_indices = torch.zeros(
                    (selected.shape[0], 1),
                    dtype=torch.long,
                    device=indices.device,
                )
                routed = host.run_direct_routed(
                    selected, local_scores, local_indices
                )
            weighted = routed * scores[token_ids, route_slots].unsqueeze(-1)
            output.index_add_(0, token_ids, weighted)
        return output.reshape(*original_shape)

    def run_direct_routed(self, tokens: Any, top_scores: Any, top_indices: Any) -> Any:
        return self.forward(tokens, top_scores, top_indices)

    def frozen_quantized_numel(self) -> int:
        return sum(int(buffer.numel()) for buffer in self.buffers())

    def set_expert_weight_access_policy(self, **kwargs: Any) -> None:
        options = dict(kwargs)
        if str(options.get("expert_weight_access", "")).strip().lower() in {
            "",
            "auto",
            "disabled",
            "full_dequant",
        }:
            options["expert_weight_access"] = "active_dequant"
        for host in self.hosts:
            host.set_expert_weight_access_policy(**options)
        for projections in self.projection_hosts:
            for host in projections.values():
                setter = getattr(host, "set_expert_weight_access_policy", None)
                if callable(setter):
                    setter(**options)
