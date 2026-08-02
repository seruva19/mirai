"""Memory telemetry and OOM guidance helpers."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any


_ACTIVE_PEAK_TRACKER: "ResourcePeakTracker | None" = None


class ResourcePeakTracker:
    def __init__(self, *, sample_interval_seconds: float = 0.1) -> None:
        self.sample_interval_seconds = max(0.02, float(sample_interval_seconds))
        self.process_peak_rss_mb = 0.0
        self.system_peak_used_mb = 0.0
        self.system_min_available_mb = float("inf")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ResourcePeakTracker is already started.")
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 4.0))
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self._sample()

    def _sample(self) -> None:
        process_rss = _current_process_rss_mb()
        system = _system_memory_telemetry()
        self.process_peak_rss_mb = max(self.process_peak_rss_mb, process_rss)
        self.system_peak_used_mb = max(
            self.system_peak_used_mb, float(system["system_memory_used_mb"])
        )
        self.system_min_available_mb = min(
            self.system_min_available_mb,
            float(system["system_memory_available_mb"]),
        )

    def snapshot(self) -> dict[str, float]:
        return {
            "tracked_process_peak_rss_mb": float(self.process_peak_rss_mb),
            "system_peak_used_mb": float(self.system_peak_used_mb),
            "system_min_available_mb": (
                0.0
                if self.system_min_available_mb == float("inf")
                else float(self.system_min_available_mb)
            ),
        }


@contextmanager
def track_resource_peaks(*, sample_interval_seconds: float = 0.1):
    global _ACTIVE_PEAK_TRACKER
    if _ACTIVE_PEAK_TRACKER is not None:
        raise RuntimeError("Resource peak tracking is already active.")
    tracker = ResourcePeakTracker(sample_interval_seconds=sample_interval_seconds)
    _ACTIVE_PEAK_TRACKER = tracker
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()
        _ACTIVE_PEAK_TRACKER = None


def current_vram_used_mb() -> float:
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)


def current_resource_telemetry() -> dict[str, Any]:
    """Best-effort process RAM and CUDA memory telemetry for run artifacts."""

    telemetry: dict[str, Any] = {
        "process_rss_mb": _current_process_rss_mb(),
        "process_peak_rss_mb": _peak_process_rss_mb(),
        "cuda_available": False,
        "cuda_allocated_mb": 0.0,
        "cuda_reserved_mb": 0.0,
        "cuda_peak_allocated_mb": 0.0,
        "cuda_peak_reserved_mb": 0.0,
        **_system_memory_telemetry(),
    }
    if _ACTIVE_PEAK_TRACKER is not None:
        telemetry.update(_ACTIVE_PEAK_TRACKER.snapshot())
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        return telemetry
    if not torch.cuda.is_available():
        return telemetry
    telemetry["cuda_available"] = True
    telemetry["cuda_allocated_mb"] = _bytes_to_mb(torch.cuda.memory_allocated())
    telemetry["cuda_reserved_mb"] = _bytes_to_mb(torch.cuda.memory_reserved())
    telemetry["cuda_peak_allocated_mb"] = _bytes_to_mb(torch.cuda.max_memory_allocated())
    telemetry["cuda_peak_reserved_mb"] = _bytes_to_mb(torch.cuda.max_memory_reserved())
    return telemetry


def _current_process_rss_mb() -> float:
    try:
        import psutil
    except ModuleNotFoundError:
        return 0.0
    try:
        process = psutil.Process(os.getpid())
        return _bytes_to_mb(process.memory_info().rss)
    except Exception:
        return 0.0


def _peak_process_rss_mb() -> float:
    try:
        import psutil
    except ModuleNotFoundError:
        return 0.0
    try:
        info = psutil.Process(os.getpid()).memory_info()
        return _bytes_to_mb(getattr(info, "peak_wset", info.rss))
    except Exception:
        return 0.0


def _bytes_to_mb(value: Any) -> float:
    return float(value) / (1024.0 * 1024.0)


def _system_memory_telemetry() -> dict[str, float]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "system_memory_total_mb": _bytes_to_mb(memory.total),
            "system_memory_used_mb": _bytes_to_mb(memory.total - memory.available),
            "system_memory_available_mb": _bytes_to_mb(memory.available),
        }
    except Exception:
        return {
            "system_memory_total_mb": 0.0,
            "system_memory_used_mb": 0.0,
            "system_memory_available_mb": 0.0,
        }


def oom_guidance_message(*, original_error: str) -> str:
    hints = [
        "CUDA out of memory detected.",
        "Try one or more of: reduce batch_size, increase gradient_accumulation,",
        "enable/raise blocks_to_swap, enable gradient_cpu_offload/optimizer_cpu_offload,",
        "or lower sample resolution/frame count.",
    ]
    return " ".join(hints) + f" Original error: {original_error}"
