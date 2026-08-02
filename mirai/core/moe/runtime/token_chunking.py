"""Single-device token-chunk scheduling for routed expert execution.

The policy adapts MemFine's fine-grained chunk distribution to Mirai's
single-GPU surface: routing is computed once for the complete token set, then
the local expert dispatch/compute/combine path is recomputed one token chunk at
a time. This preserves routing topology and global auxiliary-loss statistics.

Behavioral source: https://arxiv.org/abs/2511.21431, Section 4.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ChunkRunner = Callable[[Any, Any, Any, int], Any]


@dataclass(frozen=True)
class MoETokenChunkPolicy:
    """Checkpoint local expert work in bounded contiguous token chunks."""

    token_chunk_size: int = 0

    def __post_init__(self) -> None:
        if int(self.token_chunk_size) < 0:
            raise ValueError("training.moe_token_chunk_size must be >= 0.")
        object.__setattr__(self, "token_chunk_size", int(self.token_chunk_size))

    @property
    def enabled(self) -> bool:
        return self.token_chunk_size > 0

    def chunk_count(self, token_count: int) -> int:
        count = int(token_count)
        if count < 0:
            raise ValueError("MoE token count must be >= 0.")
        if not self.enabled or count == 0:
            return 1
        return (count + self.token_chunk_size - 1) // self.token_chunk_size

    def execute(
        self,
        tokens: Any,
        top_scores: Any,
        top_indices: Any,
        *,
        runner: ChunkRunner,
        training: bool,
    ) -> Any:
        """Run one complete route plan through chunk-local expert execution."""

        if top_scores.shape != top_indices.shape or int(top_indices.ndim) != 2:
            raise ValueError(
                "MoE token chunking requires matching [tokens, top_k] routes."
            )
        if int(tokens.ndim) != 2:
            raise ValueError("MoE token chunking requires a [tokens, hidden] tensor.")
        token_count = int(tokens.shape[0])
        if int(top_indices.shape[0]) != token_count:
            raise ValueError("MoE token chunk routes do not match the token count.")
        if (
            not self.enabled
            or token_count <= self.token_chunk_size
            or not bool(training)
        ):
            return runner(tokens, top_scores, top_indices, 0)
        if torch is None:  # pragma: no cover
            raise RuntimeError("Torch is required for MoE token chunking.")

        use_checkpoint = bool(torch.is_grad_enabled())
        outputs: list[Any] = []
        for start in range(0, token_count, self.token_chunk_size):
            end = min(start + self.token_chunk_size, token_count)
            chunk_tokens = tokens[start:end]
            chunk_scores = top_scores[start:end]
            chunk_indices = top_indices[start:end]
            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint

                def run_chunk(
                    input_tokens: Any,
                    input_scores: Any,
                    input_indices: Any,
                    *,
                    token_offset: int = start,
                ) -> Any:
                    return runner(
                        input_tokens,
                        input_scores,
                        input_indices,
                        token_offset,
                    )

                output = checkpoint(
                    run_chunk,
                    chunk_tokens,
                    chunk_scores,
                    chunk_indices,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                output = runner(
                    chunk_tokens,
                    chunk_scores,
                    chunk_indices,
                    start,
                )
            outputs.append(output)
        return torch.cat(outputs, dim=0)
__all__ = ["MoETokenChunkPolicy"]
