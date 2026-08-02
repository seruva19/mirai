"""Behavioral contracts for regional training compilation."""

from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch import nn

from mirai.core.training.runtime.compilation import (
    CompilationPolicy,
    CompilationRegion,
    TokenBucketPlan,
    prepare_training_compilation,
)


class _Region(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.square() + 1.0


class _Pipeline:
    def __init__(self) -> None:
        self.region = _Region()
        self.bucket_plan = None
        self.region_queries = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.region(value)

    def get_compilation_regions(self) -> list[CompilationRegion]:
        self.region_queries += 1
        return [CompilationRegion("region", self.region)]

    def configure_compilation_token_buckets(
        self, plan: TokenBucketPlan | None
    ) -> None:
        self.bucket_plan = plan


class CompilationContract(unittest.TestCase):
    def test_disabled_path_preserves_exact_forward_and_has_no_side_effects(self) -> None:
        pipeline = _Pipeline()
        original = pipeline.forward
        session = prepare_training_compilation(
            pipeline=pipeline,
            policy=CompilationPolicy(),
        )

        self.assertFalse(session.enabled)
        self.assertEqual(session.forward_fn, original)
        self.assertEqual(pipeline.region_queries, 0)
        self.assertIsNone(pipeline.bucket_plan)

    def test_regional_install_and_runtime_fallback_are_reversible(self) -> None:
        pipeline = _Pipeline()
        original_region_forward = pipeline.region.forward

        def fake_compile(target, **kwargs):
            self.assertEqual(kwargs, {"mode": "reduce-overhead", "dynamic": True})

            def compiled(*args, **call_kwargs):
                return target(*args, **call_kwargs) + 3.0

            return compiled

        with mock.patch(
            "mirai.core.training.runtime.compilation.torch.compile",
            side_effect=fake_compile,
        ) as compiler:
            session = prepare_training_compilation(
                pipeline=pipeline,
                policy=CompilationPolicy(
                    enabled=True,
                    scope="regional",
                    mode="reduce-overhead",
                    dynamic=True,
                    token_buckets=(8, 16),
                ),
            )

        self.assertTrue(session.enabled)
        self.assertEqual(compiler.call_count, 1)
        self.assertIsNotNone(pipeline.bucket_plan)
        value = torch.tensor([2.0])
        torch.testing.assert_close(pipeline.forward(value), torch.tensor([8.0]))

        session.disable(warning="fallback")
        self.assertFalse(session.enabled)
        self.assertEqual(session.warning, "fallback")
        self.assertEqual(pipeline.region.forward, original_region_forward)
        self.assertIsNone(pipeline.bucket_plan)
        torch.testing.assert_close(pipeline.forward(value), torch.tensor([5.0]))

    def test_token_bucket_plan_validates_ranges_and_hints_dynamic_shape(self) -> None:
        plan = TokenBucketPlan((8, 16, 32))
        tensor = torch.ones((2, 12, 4))
        with mock.patch(
            "torch._dynamo.maybe_mark_dynamic",
        ) as marker:
            returned = plan.mark(tensor, dim=1)

        self.assertIs(returned, tensor)
        marker.assert_called_once_with(tensor, 1)
        self.assertEqual(plan.snapshot()["hits"], {"8": 0, "16": 1, "32": 0})
        with self.assertRaisesRegex(ValueError, "exceeds"):
            plan.mark(torch.ones((1, 33, 1)), dim=1)

    def test_invalid_compile_policies_fail_before_installation(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            CompilationPolicy(token_buckets=(16, 8))
        with self.assertRaisesRegex(ValueError, "compile_scope"):
            CompilationPolicy(scope="whole")
        with self.assertRaisesRegex(ValueError, "compile_dynamic"):
            CompilationPolicy(dynamic=False, token_buckets=(16,))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
