from __future__ import annotations

import unittest

# Colocated behavioral contract for adapter persistence and interchange.

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mirai.core.models.adapters.dora import DORA_MAGNITUDE_SUFFIX
from mirai.core.models.adapters.lora_interchange import (
    LoRAInterchangeError,
    convert_lora_state_dict,
    detect_layout,
    fuse_lora_state_dict,
    unfuse_lora_state_dict,
)
from mirai.core.models.adapters.sparse_expert_export import (
    SparseExpertExportPolicy,
    expand_sparse_expert_state,
)

def _full_expert_state(*, name: str, experts: int, rank: int, in_f: int, out_f: int):
    return {
        f"{name}.lora_a": torch.randn(experts, rank, in_f),
        f"{name}.lora_b": torch.randn(experts, out_f, rank),
        f"{name}.lora_alpha": torch.tensor([float(rank)]),
    }


def _linear_state(*, name: str, rank: int, in_f: int, out_f: int):
    return {
        f"{name}.lora_a": torch.randn(rank, in_f),
        f"{name}.lora_b": torch.randn(out_f, rank),
        f"{name}.lora_alpha": torch.tensor([16.0]),
    }


def _compact_state(*, name: str, experts: int, ids: list[int], rank: int, in_f: int, out_f: int):
    selected = len(ids)
    mask = torch.zeros(experts)
    for e in ids:
        mask[e] = 1.0
    return {
        f"{name}.lora_a_selected": torch.randn(selected, rank, in_f),
        f"{name}.lora_b_selected": torch.randn(selected, out_f, rank),
        f"{name}.active_expert_ids": torch.tensor(ids, dtype=torch.long),
        f"{name}.active_expert_mask": mask,
        f"{name}.lora_alpha": torch.tensor([float(rank)]),
    }


def _assert_bit_identical(test: unittest.TestCase, lhs: dict, rhs: dict) -> None:
    test.assertEqual(set(lhs.keys()), set(rhs.keys()))
    for key in lhs:
        a, b = lhs[key], rhs[key]
        test.assertEqual(a.dtype, b.dtype, key)
        test.assertEqual(tuple(a.shape), tuple(b.shape), key)
        test.assertTrue(torch.equal(a, b), key)


@unittest.skipIf(torch is None, "torch not installed")
class LoRAInterchangeRoundTripTests(unittest.TestCase):
    def test_dora_dense_round_trip_preserves_magnitude(self) -> None:
        fused = {
            "block.proj.lora_a": torch.randn(2, 3),
            "block.proj.lora_b": torch.randn(4, 2),
            "block.proj.lora_alpha": torch.tensor([2.0]),
            f"block.proj{DORA_MAGNITUDE_SUFFIX}": torch.randn(4),
        }
        unfused = unfuse_lora_state_dict(fused)
        self.assertIn("block.proj.lora_magnitude_vector.weight", unfused)
        restored = fuse_lora_state_dict(unfused)
        self.assertEqual(set(restored), set(fused))
        for key in fused:
            self.assertTrue(torch.equal(restored[key], fused[key]))

    def test_dora_grouped_round_trip_preserves_each_expert_magnitude(
        self,
    ) -> None:
        fused = {
            "block.experts.w1.lora_a": torch.randn(3, 2, 5),
            "block.experts.w1.lora_b": torch.randn(3, 4, 2),
            f"block.experts.w1{DORA_MAGNITUDE_SUFFIX}": torch.randn(3, 4),
        }
        unfused = unfuse_lora_state_dict(fused)
        for expert in range(3):
            self.assertIn(
                "block.experts.w1.experts."
                f"{expert}.lora_magnitude_vector.weight",
                unfused,
            )
        restored = fuse_lora_state_dict(unfused)
        for key in fused:
            self.assertTrue(torch.equal(restored[key], fused[key]))

    def test_sparse_export_writes_only_selected_rows_and_expands_losslessly(self) -> None:
        class Module:
            _expert_selection_active = True
            active_expert_mask = torch.tensor([1.0, 0.0, 1.0])
            lora_a = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
            lora_b = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
            lora_alpha = torch.tensor(2.0)

        state = {}
        self.assertTrue(
            SparseExpertExportPolicy(enabled=True).write_module_state(
                state, name="experts.w1", module=Module()
            )
        )
        self.assertEqual(
            state["experts.w1.active_expert_ids"].tolist(),
            [0, 2],
        )
        expanded = expand_sparse_expert_state(state)
        self.assertEqual(tuple(expanded["experts.w1.lora_a"].shape), (3, 2, 4))
        self.assertEqual(tuple(expanded["experts.w1.lora_b"].shape), (3, 3, 2))
        self.assertTrue(torch.equal(expanded["experts.w1.lora_a"][0], Module.lora_a[0]))
        self.assertTrue(torch.equal(expanded["experts.w1.lora_a"][2], Module.lora_a[2]))
        self.assertTrue(torch.equal(expanded["experts.w1.lora_a"][1], torch.zeros(2, 4)))

    def test_full_expert_roundtrip_bit_identical(self) -> None:
        for experts, rank, in_f, out_f in [(2, 4, 8, 16), (8, 2, 5, 3), (1, 1, 2, 2)]:
            fused = _full_expert_state(
                name="blk.0.experts.w1", experts=experts, rank=rank, in_f=in_f, out_f=out_f
            )
            unfused = unfuse_lora_state_dict(fused)
            self.assertIn("blk.0.experts.w1.experts.0.lora_A.weight", unfused)
            self.assertEqual(
                tuple(unfused["blk.0.experts.w1.experts.0.lora_A.weight"].shape),
                (rank, in_f),
            )
            back = fuse_lora_state_dict(unfused)
            _assert_bit_identical(self, fused, back)

    def test_linear_roundtrip_bit_identical(self) -> None:
        fused = _linear_state(name="attn.to_q", rank=8, in_f=32, out_f=32)
        unfused = unfuse_lora_state_dict(fused)
        self.assertIn("attn.to_q.lora_A.weight", unfused)
        self.assertIn("attn.to_q.lora_B.weight", unfused)
        self.assertIn("attn.to_q.alpha", unfused)
        self.assertNotIn("attn.to_q.experts.0.lora_A.weight", unfused)
        back = fuse_lora_state_dict(unfused)
        _assert_bit_identical(self, fused, back)

    def test_compact_sparse_roundtrip_bit_identical(self) -> None:
        fused = _compact_state(
            name="blk.1.experts.w2", experts=6, ids=[1, 3, 4], rank=2, in_f=8, out_f=4
        )
        unfused = unfuse_lora_state_dict(fused)
        # Only the selected experts get per-expert keys.
        self.assertIn("blk.1.experts.w2.experts.1.lora_A.weight", unfused)
        self.assertIn("blk.1.experts.w2.experts.4.lora_A.weight", unfused)
        self.assertNotIn("blk.1.experts.w2.experts.0.lora_A.weight", unfused)
        self.assertIn("blk.1.experts.w2.active_expert_ids", unfused)
        self.assertIn("blk.1.experts.w2.active_expert_mask", unfused)
        back = fuse_lora_state_dict(unfused)
        _assert_bit_identical(self, fused, back)
        self.assertEqual(back["blk.1.experts.w2.active_expert_ids"].dtype, torch.long)

    def test_full_expert_with_selection_mask_roundtrip(self) -> None:
        fused = _full_expert_state(
            name="blk.2.experts.w3", experts=4, rank=2, in_f=8, out_f=8
        )
        fused["blk.2.experts.w3.active_expert_mask"] = torch.tensor([1.0, 0.0, 1.0, 0.0])
        unfused = unfuse_lora_state_dict(fused)
        self.assertIn("blk.2.experts.w3.active_expert_mask", unfused)
        self.assertNotIn("blk.2.experts.w3.active_expert_ids", unfused)
        # All 4 experts still emitted (full+mask, not compact).
        self.assertIn("blk.2.experts.w3.experts.3.lora_A.weight", unfused)
        back = fuse_lora_state_dict(unfused)
        _assert_bit_identical(self, fused, back)

    def test_condenser_keys_passthrough_roundtrip(self) -> None:
        fused = _full_expert_state(
            name="blk.0.experts.w1", experts=3, rank=2, in_f=8, out_f=4
        )
        fused["blk.0.experts.w1.condenser_a"] = torch.randn(2, 8)
        fused["blk.0.experts.w1.condenser_b"] = torch.randn(4, 2)
        fused["blk.0.experts.w1.condenser_alpha"] = torch.tensor([2.0])
        unfused = unfuse_lora_state_dict(fused)
        self.assertIn("blk.0.experts.w1.condenser_a", unfused)
        back = fuse_lora_state_dict(unfused)
        _assert_bit_identical(self, fused, back)

    def test_mixed_expert_and_linear_dict_roundtrip(self) -> None:
        fused = {}
        fused.update(_full_expert_state(name="blk.0.experts.w1", experts=4, rank=2, in_f=8, out_f=4))
        fused.update(_linear_state(name="blk.0.attn.to_q", rank=4, in_f=16, out_f=16))
        fused.update(_compact_state(name="blk.1.experts.w2", experts=5, ids=[0, 2], rank=2, in_f=8, out_f=4))
        unfused = unfuse_lora_state_dict(fused)
        back = fuse_lora_state_dict(unfused)
        _assert_bit_identical(self, fused, back)

    def test_detect_layout(self) -> None:
        self.assertEqual(detect_layout({"x.lora_a": torch.zeros(2, 2, 2)}), "fused")
        self.assertEqual(detect_layout({"x.lora_a_selected": torch.zeros(1, 2, 2)}), "fused")
        self.assertEqual(detect_layout({"x.lora_A.weight": torch.zeros(2, 2)}), "unfused")

    def test_convert_auto_direction(self) -> None:
        fused = _full_expert_state(name="blk.0.experts.w1", experts=2, rank=2, in_f=4, out_f=4)
        unfused, applied = convert_lora_state_dict(fused, direction="auto")
        self.assertEqual(applied, "unfuse")
        back, applied_back = convert_lora_state_dict(unfused, direction="auto")
        self.assertEqual(applied_back, "fuse")
        _assert_bit_identical(self, fused, back)


@unittest.skipIf(torch is None, "torch not installed")
class LoRAInterchangeFailFastTests(unittest.TestCase):
    def test_unknown_fused_key_fails(self) -> None:
        fused = _full_expert_state(name="blk.0.experts.w1", experts=2, rank=2, in_f=4, out_f=4)
        fused["blk.0.experts.w1.bogus_tensor"] = torch.zeros(2)
        with self.assertRaises(LoRAInterchangeError):
            unfuse_lora_state_dict(fused)

    def test_unknown_key_dropped_with_no_strict(self) -> None:
        fused = _full_expert_state(name="blk.0.experts.w1", experts=2, rank=2, in_f=4, out_f=4)
        fused["blk.0.experts.w1.bogus_tensor"] = torch.zeros(2)
        unfused = unfuse_lora_state_dict(fused, strict=False)
        self.assertNotIn("blk.0.experts.w1.bogus_tensor", unfused)

    def test_rank_mismatch_fails(self) -> None:
        fused = {
            "x.lora_a": torch.randn(3, 4, 8),  # rank 4
            "x.lora_b": torch.randn(3, 8, 2),  # rank 2 -> mismatch
        }
        with self.assertRaises(LoRAInterchangeError):
            unfuse_lora_state_dict(fused)

    def test_expert_count_mismatch_fails(self) -> None:
        fused = {
            "x.lora_a": torch.randn(3, 4, 8),
            "x.lora_b": torch.randn(2, 8, 4),
        }
        with self.assertRaises(LoRAInterchangeError):
            unfuse_lora_state_dict(fused)

    def test_missing_expert_index_fails_on_fuse(self) -> None:
        unfused = {
            "x.experts.0.lora_A.weight": torch.randn(2, 4),
            "x.experts.0.lora_B.weight": torch.randn(4, 2),
            "x.experts.2.lora_A.weight": torch.randn(2, 4),  # gap: no expert 1
            "x.experts.2.lora_B.weight": torch.randn(4, 2),
        }
        with self.assertRaises(LoRAInterchangeError):
            fuse_lora_state_dict(unfused)

    def test_divergent_expert_alpha_fails_on_fuse(self) -> None:
        unfused = {
            "x.experts.0.lora_A.weight": torch.randn(2, 4),
            "x.experts.0.lora_B.weight": torch.randn(4, 2),
            "x.experts.0.alpha": torch.tensor([2.0]),
            "x.experts.1.lora_A.weight": torch.randn(2, 4),
            "x.experts.1.lora_B.weight": torch.randn(4, 2),
            "x.experts.1.alpha": torch.tensor([9.0]),
        }
        with self.assertRaises(LoRAInterchangeError):
            fuse_lora_state_dict(unfused)

    def test_ambiguous_layout_detection_fails(self) -> None:
        with self.assertRaises(LoRAInterchangeError):
            detect_layout({"x.lora_a": torch.zeros(2, 2, 2), "y.lora_A.weight": torch.zeros(2, 2)})

    def test_empty_layout_detection_fails(self) -> None:
        with self.assertRaises(LoRAInterchangeError):
            detect_layout({"random": torch.zeros(2)})


if __name__ == "__main__":
    unittest.main()
