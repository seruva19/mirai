"""Inference-only sigmoid top-k for the MAGI-2 head-local router."""

from functools import lru_cache

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


def attach_inference_router(transformer: torch.nn.Module) -> None:
    from mirai.vendors.magi2_preview.model.magi2_preview import CoreMultiHeadMoE

    for module in transformer.modules():
        if isinstance(module, CoreMultiHeadMoE) and module.config.score_func == "sigmoid":
            module._mirai_inference_router = sigmoid_topk


def sigmoid_topk(logits, bias, *, top_k: int, normalize: bool):
    heads, tokens, experts = logits.shape
    probabilities = logits.new_empty((heads, tokens, top_k))
    indices = torch.empty(probabilities.shape, dtype=torch.int64, device=logits.device)
    if tokens:
        _router_kernel()[(heads * tokens,)](
            logits.contiguous(), bias.float().contiguous(), probabilities, indices,
            tokens, experts, top_k, normalize, triton.next_power_of_2(experts),
            triton.next_power_of_2(top_k), num_warps=4,
        )
    return probabilities, indices

@lru_cache(maxsize=1)
def _router_kernel():
    if triton is None:
        raise RuntimeError("MAGI-2 Triton expert routing requires Triton.")

    @triton.jit
    def select(
        logits, bias, probabilities, indices, tokens,
        experts: tl.constexpr, top_k: tl.constexpr, normalize: tl.constexpr,
        block_experts: tl.constexpr, block_routes: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        lanes = tl.arange(0, block_experts)
        valid = lanes < experts
        scores = tl.sigmoid(tl.load(logits + row * experts + lanes, valid, other=0.0))
        head = row // tokens
        selection = scores + tl.load(bias + head * experts + lanes, valid, other=0.0)
        # Match descending Torch top-k's NaN ordering while retaining the
        # original score in the output. NaN equality cannot select a lane.
        selection = tl.where(selection != selection, float("inf"), selection)
        available = valid
        ranks = tl.arange(0, block_routes)
        chosen_scores = tl.full((block_routes,), 0, tl.float32)
        chosen_indices = tl.full((block_routes,), 0, tl.int32)
        for rank in tl.static_range(top_k):
            ranked = tl.where(available, selection, -float("inf"))
            best = tl.max(ranked, axis=0)
            chosen = tl.min(tl.where(available & (ranked == best), lanes, block_experts), axis=0)
            score = tl.sum(tl.where(lanes == chosen, scores, 0.0), axis=0)
            chosen_scores = tl.where(ranks == rank, score, chosen_scores)
            chosen_indices = tl.where(ranks == rank, chosen, chosen_indices)
            available = available & (lanes != chosen)
        if normalize:
            total = tl.sum(chosen_scores, axis=0)
            chosen_scores = chosen_scores / tl.where(total < 1.0e-12, 1.0e-12, total)
        tl.store(probabilities + row * top_k + ranks, chosen_scores, ranks < top_k)
        tl.store(indices + row * top_k + ranks, chosen_indices, ranks < top_k)

    return select
