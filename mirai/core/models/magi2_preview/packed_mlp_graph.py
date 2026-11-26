"""Bounded CUDA Graph reuse for resident packed expert inference."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import torch


@dataclass
class _CapturedMlp:
    rows: torch.Tensor
    ends: torch.Tensor
    output: torch.Tensor
    graph: torch.cuda.CUDAGraph
    weights: tuple[torch.Tensor, ...]
    done: torch.cuda.Event

    def load(self, rows, ends):
        count = rows.shape[0]
        self.rows[:count].copy_(rows)
        self.rows[count:].zero_()
        self.ends.copy_(ends)
        self.ends[-1:].fill_(self.rows.shape[0])


class PackedMlpGraphCache:
    """Reuse fixed row buckets and hold captured weight storage until eviction."""

    def __init__(self, capacity: int):
        if not 1 <= capacity <= 8:
            raise ValueError("Packed MLP graph cache capacity must be in [1, 8].")
        self._capacity = capacity
        self._entries = OrderedDict()
        self._lock = Lock()

    def run(self, rows, ends, weights, execute):
        count = rows.shape[0]
        bucket = 1 << (count - 1).bit_length()
        current = torch.cuda.current_stream(rows.device)
        key = (
            rows.device, current.cuda_stream, rows.dtype, rows.shape[1], bucket, ends.dtype, ends.numel(),
            tuple((tensor.data_ptr(), tensor.shape, tensor.stride()) for tensor in weights),
        )
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                if len(self._entries) == self._capacity:
                    _, evicted = self._entries.popitem(last=False)
                    evicted.done.synchronize()
                    del evicted
                entry = self._capture(rows, ends, weights, execute, bucket, current)
                self._entries[key] = entry
            else:
                self._entries.move_to_end(key)
            entry.load(rows, ends)
            entry.graph.replay()
            result = entry.output[:count].clone()
            entry.done.record(current)
            return result

    @staticmethod
    def _capture(rows, ends, weights, execute, bucket, current):
        static_rows = rows.new_zeros((bucket, rows.shape[1]))
        static_rows[:rows.shape[0]].copy_(rows)
        static_ends = ends.clone()
        static_ends[-1:].fill_(bucket)
        warm = torch.cuda.Stream(device=rows.device)
        warm.wait_stream(current)
        with torch.cuda.stream(warm):
            execute(static_rows, static_ends)
        warm.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=warm):
            output = execute(static_rows, static_ends)
        return _CapturedMlp(static_rows, static_ends, output, graph, weights, torch.cuda.Event())
