"""Small host seam for observing already-materialized routed expert outputs."""

from __future__ import annotations

from typing import Any

import torch


class RoutedOutputObserverHost:
    """Own the optional observer lifecycle without expanding expert dispatch."""

    supports_routed_output_observer = True
    supports_routed_intermediate_observer = True
    _routed_output_observer: Any | None = None
    _routed_intermediate_observer: Any | None = None
    _routed_intermediate_chunk: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def routed_output_observer_active(self) -> bool:
        observer = self._routed_output_observer
        return observer is not None and bool(observer.is_enabled)

    def get_routed_output_observer(self) -> Any | None:
        """Return the bound observer without applying the execution-mode gate."""
        return self._routed_output_observer

    def active_routed_output_observer(self) -> Any | None:
        observer = self._routed_output_observer
        if not self.routed_output_observer_active:
            return None
        if self.training or bool(getattr(observer, "capture_in_eval", False)):
            return observer
        return None

    def set_routed_output_observer(self, observer: Any | None) -> None:
        if observer is not None:
            for name in ("begin_routes", "capture_routes", "end_routes"):
                if not callable(getattr(observer, name, None)):
                    raise TypeError(
                        "Routed output observers must implement begin_routes, "
                        "capture_routes, and end_routes."
                    )
        self._routed_output_observer = observer

    @property
    def routed_intermediate_observer_active(self) -> bool:
        observer = self._routed_intermediate_observer
        return observer is not None and bool(observer.is_enabled)

    def active_routed_intermediate_observer(self) -> Any | None:
        if self.training and self.routed_intermediate_observer_active:
            return self._routed_intermediate_observer
        return None

    def set_routed_intermediate_observer(self, observer: Any | None) -> None:
        if observer is not None:
            for name in (
                "begin_routes",
                "capture_routes",
                "end_routes",
                "capture_sorted_chunk",
            ):
                if not callable(getattr(observer, name, None)):
                    raise TypeError(
                        "Routed intermediate observers must implement route "
                        "and sorted-chunk capture."
                    )
        self._routed_intermediate_observer = observer

    def _capture_routed_outputs(
        self, expert_output: torch.Tensor, route_positions: torch.Tensor
    ) -> None:
        observer = self.active_routed_output_observer()
        if observer is not None:
            observer.capture_routes(expert_output, route_positions)

    def _capture_routed_intermediates(
        self, intermediate: torch.Tensor, route_positions: torch.Tensor
    ) -> None:
        observer = self.active_routed_intermediate_observer()
        if observer is not None:
            observer.capture_routes(intermediate, route_positions)

    def _capture_sorted_intermediate(self, intermediate: torch.Tensor) -> None:
        observer = self.active_routed_intermediate_observer()
        if observer is not None:
            observer.capture_sorted_chunk(intermediate)

    def _run_with_routed_intermediate_capture(
        self,
        runner: Any,
        *args: Any,
        capture_mask: torch.Tensor,
        route_positions: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self.routed_intermediate_observer_active:
            return runner(*args, **kwargs)
        self._routed_intermediate_chunk = (capture_mask, route_positions)
        try:
            return runner(*args, **kwargs)
        finally:
            self._routed_intermediate_chunk = None

    def _capture_chunk_intermediate(self, intermediate: torch.Tensor) -> None:
        context = self._routed_intermediate_chunk
        if context is not None:
            mask, positions = context
            self._capture_routed_intermediates(
                intermediate[mask], positions[mask]
            )


__all__ = ["RoutedOutputObserverHost"]
