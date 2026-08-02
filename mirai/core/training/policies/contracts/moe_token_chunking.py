"""Behavioral contracts for single-GPU MoE token chunk checkpointing."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from mirai.core.moe.runtime.token_chunking import MoETokenChunkPolicy
from mirai.vendors.lingbot_video.transformer_lingbot_video import (
    LingBotVideoSparseMoeBlock,
)


class _ChunkLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, 4, dtype=torch.float64))
        self.offsets: list[int] = []

    def run(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
        token_offset: int,
    ) -> torch.Tensor:
        self.offsets.append(int(token_offset))
        scale = scores.gather(1, indices.remainder(scores.shape[1])).sum(
            dim=1,
            keepdim=True,
        )
        return (tokens @ self.weight.transpose(0, 1)) * scale


class MoETokenChunkingContract(unittest.TestCase):
    def test_disabled_policy_is_the_exact_single_call_path(self) -> None:
        policy = MoETokenChunkPolicy()
        tokens = torch.randn(7, 4)
        scores = torch.rand(7, 2)
        indices = torch.zeros(7, 2, dtype=torch.long)
        calls: list[int] = []

        def runner(
            input_tokens,
            input_scores,
            input_indices,
            token_offset,
        ):
            calls.append(int(token_offset))
            self.assertIs(input_tokens, tokens)
            self.assertIs(input_scores, scores)
            self.assertIs(input_indices, indices)
            return input_tokens

        output = policy.execute(
            tokens,
            scores,
            indices,
            runner=runner,
            training=True,
        )

        self.assertIs(output, tokens)
        self.assertEqual(calls, [0])

    def test_chunk_checkpoint_matches_output_and_all_gradients(self) -> None:
        torch.manual_seed(91)
        reference = _ChunkLinear()
        chunked = _ChunkLinear()
        chunked.load_state_dict(reference.state_dict())
        base_tokens = torch.randn(8, 4, dtype=torch.float64)
        base_scores = torch.rand(8, 2, dtype=torch.float64)
        indices = torch.tensor(
            [[0, 1], [1, 0], [0, 0], [1, 1]] * 2,
            dtype=torch.long,
        )

        reference_tokens = base_tokens.clone().requires_grad_(True)
        reference_scores = base_scores.clone().requires_grad_(True)
        expected = reference.run(
            reference_tokens,
            reference_scores,
            indices,
            0,
        )
        expected.square().mean().backward()

        chunk_tokens = base_tokens.clone().requires_grad_(True)
        chunk_scores = base_scores.clone().requires_grad_(True)
        actual = MoETokenChunkPolicy(token_chunk_size=3).execute(
            chunk_tokens,
            chunk_scores,
            indices,
            runner=chunked.run,
            training=True,
        )
        actual.square().mean().backward()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(chunk_tokens.grad, reference_tokens.grad)
        torch.testing.assert_close(chunk_scores.grad, reference_scores.grad)
        torch.testing.assert_close(chunked.weight.grad, reference.weight.grad)
        self.assertEqual(chunked.offsets[:3], [0, 3, 6])
        self.assertGreaterEqual(len(chunked.offsets), 6)

    def test_invalid_shapes_and_sizes_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            MoETokenChunkPolicy(token_chunk_size=-1)
        policy = MoETokenChunkPolicy(token_chunk_size=2)
        with self.assertRaisesRegex(ValueError, "matching"):
            policy.execute(
                torch.randn(3, 4),
                torch.rand(3, 2),
                torch.zeros(3, 1, dtype=torch.long),
                runner=lambda *args: args[0],
                training=True,
            )

    def test_native_block_matches_output_and_all_gradients(self) -> None:
        torch.manual_seed(117)
        block = _native_block(
            hidden_size=8,
            expert_intermediate_size=16,
            shared_experts=1,
            weight_std=0.05,
        )
        base_hidden = torch.randn(2, 7, 8)

        reference_input = base_hidden.clone().requires_grad_(True)
        reference_output = block(reference_input)
        reference_output.square().mean().backward()
        reference_input_gradient = reference_input.grad.detach().clone()
        reference_parameter_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in block.named_parameters()
            if parameter.grad is not None
        }

        block.zero_grad(set_to_none=True)
        block.set_token_chunk_policy(
            MoETokenChunkPolicy(token_chunk_size=5)
        )
        chunked_input = base_hidden.clone().requires_grad_(True)
        chunked_output = block(chunked_input)
        chunked_output.square().mean().backward()

        torch.testing.assert_close(chunked_output, reference_output)
        torch.testing.assert_close(
            chunked_input.grad,
            reference_input_gradient,
        )
        self.assertEqual(
            set(reference_parameter_gradients),
            {
                name
                for name, parameter in block.named_parameters()
                if parameter.grad is not None
            },
        )
        for name, parameter in block.named_parameters():
            if name in reference_parameter_gradients:
                torch.testing.assert_close(
                    parameter.grad,
                    reference_parameter_gradients[name],
                )

    def test_native_block_reduces_saved_activation_bytes(self) -> None:
        torch.manual_seed(131)
        block = _native_block(
            hidden_size=16,
            expert_intermediate_size=128,
            shared_experts=0,
            weight_std=0.02,
        )

        def saved_bytes(policy) -> int:
            block.zero_grad(set_to_none=True)
            block.set_token_chunk_policy(policy)
            total = [0]

            def pack(tensor):
                total[0] += int(tensor.numel() * tensor.element_size())
                return tensor

            hidden = torch.randn(2, 32, 16, requires_grad=True)
            with torch.autograd.graph.saved_tensors_hooks(
                pack,
                lambda tensor: tensor,
            ):
                block(hidden).square().mean().backward()
            return total[0]

        reference_bytes = saved_bytes(None)
        chunked_bytes = saved_bytes(
            MoETokenChunkPolicy(token_chunk_size=8)
        )

        self.assertGreater(reference_bytes, 0)
        self.assertLess(chunked_bytes, reference_bytes)
        self.assertLessEqual(chunked_bytes * 2, reference_bytes)


def _native_block(
    *,
    hidden_size: int,
    expert_intermediate_size: int,
    shared_experts: int,
    weight_std: float,
) -> LingBotVideoSparseMoeBlock:
    block = LingBotVideoSparseMoeBlock(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_experts=4,
        top_k=2,
        moe_intermediate_size=expert_intermediate_size,
        score_func="softmax",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        routed_scaling_factor=1.0,
        n_shared_experts=shared_experts,
    )
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(mean=0.0, std=float(weight_std))
    return block


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
