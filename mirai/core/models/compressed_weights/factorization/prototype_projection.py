"""Bounded prototype-projection source for compressed_weights grouped experts."""

from __future__ import annotations

from typing import Any

from mirai.core.moe.calibration.projection import ExpertProjectionSpec

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..execution.experts import CompressedGroupedExperts


class CompressedExpertProjectionSource:
    """Adapt packed expert storage to the generic block-readable contract."""

    def __init__(self, module: CompressedGroupedExperts) -> None:
        if not isinstance(module, CompressedGroupedExperts):
            raise TypeError("Compressed projection source requires grouped experts.")
        if module.has_logical_expert_aliases():
            raise RuntimeError(
                "Prototype projection calibration requires unconsolidated experts."
            )
        self.module = module
        self.num_experts = int(module.num_experts)

    def prototype_projection_specs(self) -> tuple[ExpertProjectionSpec, ...]:
        specs: list[ExpertProjectionSpec] = []
        module = self.module
        for key in ("w1", "w2", "w3"):
            if key in module._packed_shapes:
                shape = module._packed_shapes[key]
            elif key in module._nf4_shapes:
                shape = module._nf4_shapes[key]
            elif key in module._gguf_shapes:
                shape = module._gguf_shapes[key]
            elif key in module._microscaling_shapes:
                shape = module._microscaling_shapes[key]
            elif key in module._blockwise_fp8_shapes:
                shape = module._blockwise_fp8_shapes[key]
            elif hasattr(module, f"{key}_int8"):
                shape = tuple(
                    int(value) for value in getattr(module, f"{key}_int8").shape
                )
            else:
                raise RuntimeError(
                    f"Grouped expert projection {key!r} is not loaded."
                )
            if len(shape) != 3 or int(shape[0]) != self.num_experts:
                raise RuntimeError(
                    f"Grouped expert projection {key!r} has invalid shape {shape}."
                )
            specs.append(
                ExpertProjectionSpec(
                    name=key,
                    shape=tuple(int(value) for value in shape[1:]),
                )
            )
        return tuple(specs)

    def load_prototype_projection_block(
        self,
        projection_name: str,
        start_expert: int,
        stop_expert: int,
        *,
        device: Any,
        dtype: Any,
    ) -> Any:
        key = str(projection_name)
        specs = {spec.name: spec for spec in self.prototype_projection_specs()}
        if key not in specs:
            raise ValueError(f"Unknown grouped expert projection {key!r}.")
        start = int(start_expert)
        stop = int(stop_expert)
        if start < 0 or stop <= start or stop > self.num_experts:
            raise ValueError(
                f"Invalid expert projection block [{start}, {stop}) for "
                f"{self.num_experts} experts."
            )
        resolved_device = torch.device(device)
        with torch.no_grad():
            block = torch.empty(
                (stop - start, *specs[key].shape),
                dtype=dtype,
                device=resolved_device,
            )
            for offset, expert_id in enumerate(range(start, stop)):
                dense_expert = self.module._dequantize_expert(
                    key,
                    expert_id,
                    dtype=dtype,
                    device=resolved_device,
                )
                block[offset].copy_(dense_expert)
                del dense_expert
            return block.detach()


__all__ = ["CompressedExpertProjectionSource"]
