"""Behavioral contracts for PARA post-hoc LoRA compression."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mirai.core.models.adapters.lora import LoRALinear
from mirai.core.models.adapters.lora import load_lora_state_dict
from mirai.core.models.adapters.lora_allocation import PARA_RANK_STATE_SUFFIX
from mirai.core.models.adapters.lora_allocation import RSLORA_STATE_SUFFIX
from mirai.core.models.adapters.para import PARA_METADATA_KEY
from mirai.core.models.adapters.para import PARA_TRANSFORM_KEY
from mirai.core.models.adapters.para import compress_lora_state_para
from mirai.core.models.adapters.para import expand_para_adapter_payload
from mirai.core.models.adapters.para import save_para_adapter_safetensors
from mirai.core.models.adapters.para import validate_para_manifest
from mirai.core.training.adapters import load_adapter_payload
from scripts.tools.compress_adapter_para import compress_adapter

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _payload(result):
    return {
        **result.tensors,
        "_metadata": {
            PARA_TRANSFORM_KEY: "para",
            PARA_METADATA_KEY: json.dumps(result.manifest),
        },
    }


@unittest.skipIf(torch is None, "torch not installed")
class ParaCompressionContracts(unittest.TestCase):
    SOURCE_SHA256 = "a" * 64

    def test_rank_policy_is_global_not_per_layer(self) -> None:
        state = {
            "strong.lora_a": torch.eye(2),
            "strong.lora_b": torch.diag(torch.tensor([4.0, 3.0])),
            "strong.lora_alpha": torch.tensor([2.0]),
            "weak.lora_a": torch.eye(2),
            "weak.lora_b": torch.diag(torch.tensor([2.0, 1.0])),
            "weak.lora_alpha": torch.tensor([2.0]),
        }
        result = compress_lora_state_para(
            state,
            source_adapter_sha256=self.SOURCE_SHA256,
            policy="rank",
            rank_preservation_ratio=0.5,
        )
        ranks = {
            item["name"]: item["retained_ranks"]
            for item in result.manifest["modules"]
        }
        self.assertEqual(ranks, {"strong": [2], "weak": [0]})
        self.assertEqual(result.manifest["retained_total_rank"], 2)

    def test_energy_policy_keeps_minimum_global_prefix(self) -> None:
        state = {
            "first.lora_a": torch.eye(2),
            "first.lora_b": torch.diag(torch.tensor([4.0, 3.0])),
            "second.lora_a": torch.eye(2),
            "second.lora_b": torch.diag(torch.tensor([2.0, 1.0])),
        }
        result = compress_lora_state_para(
            state,
            source_adapter_sha256=self.SOURCE_SHA256,
            policy="energy",
            energy_preservation_ratio=0.8,
        )
        self.assertEqual(result.manifest["retained_total_rank"], 2)
        self.assertAlmostEqual(
            result.manifest["retained_energy_fraction"],
            25.0 / 30.0,
        )

    def test_grouped_experts_are_independent_ragged_units(self) -> None:
        state = {
            "experts.w1.lora_a": torch.stack([torch.eye(2), torch.eye(2)]),
            "experts.w1.lora_b": torch.stack(
                [
                    torch.diag(torch.tensor([5.0, 0.1])),
                    torch.diag(torch.tensor([4.0, 3.0])),
                ]
            ),
            "experts.w1.lora_alpha": torch.tensor([2.0]),
        }
        result = compress_lora_state_para(
            state,
            source_adapter_sha256=self.SOURCE_SHA256,
            policy="rank",
            rank_preservation_ratio=0.75,
        )
        module = result.manifest["modules"][0]
        self.assertEqual(module["retained_ranks"], [1, 2])
        self.assertEqual(module["runtime_rank"], 2)
        expanded = expand_para_adapter_payload(_payload(result))["adapter_state"]
        self.assertEqual(tuple(expanded["experts.w1.lora_a"].shape), (2, 2, 2))
        self.assertTrue(
            torch.equal(
                expanded["experts.w1.lora_a"][0, 1],
                torch.zeros(2),
            )
        )

    def test_runtime_scaling_and_truncated_update_are_preserved(self) -> None:
        state = {
            "proj.lora_a": torch.eye(2),
            "proj.lora_b": torch.diag(torch.tensor([4.0, 1.0])),
            "proj.lora_alpha": torch.tensor([8.0]),
            f"proj{RSLORA_STATE_SUFFIX}": torch.ones(1, dtype=torch.uint8),
        }
        result = compress_lora_state_para(
            state,
            source_adapter_sha256=self.SOURCE_SHA256,
            policy="rank",
            rank_preservation_ratio=0.5,
        )
        expanded = expand_para_adapter_payload(_payload(result))["adapter_state"]
        new_rank = int(expanded[f"proj{PARA_RANK_STATE_SUFFIX}"].item())
        new_scale = float(expanded["proj.lora_alpha"].item()) / (new_rank**0.5)
        effective = expanded["proj.lora_b"] @ expanded["proj.lora_a"] * new_scale
        torch.testing.assert_close(
            effective,
            torch.diag(torch.tensor([4.0, 0.0])) * (8.0 / (2.0**0.5)),
        )

    def test_safetensors_roundtrip_and_dynamic_rank_load(self) -> None:
        state = {
            "proj.lora_a": torch.eye(2),
            "proj.lora_b": torch.diag(torch.tensor([4.0, 1.0])),
            "proj.lora_alpha": torch.tensor([2.0]),
        }
        result = compress_lora_state_para(
            state,
            source_adapter_sha256=self.SOURCE_SHA256,
            policy="rank",
            rank_preservation_ratio=0.5,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.safetensors"
            save_para_adapter_safetensors(path, result)
            payload = load_adapter_payload(path)
        expanded = payload["adapter_state"]
        self.assertEqual(int(expanded[f"proj{PARA_RANK_STATE_SUFFIX}"].item()), 1)
        root = torch.nn.Module()
        root.proj = LoRALinear(
            torch.nn.Linear(2, 2, bias=False),
            rank=2,
            alpha=2.0,
        )
        load_lora_state_dict(root, expanded)
        self.assertEqual(root.proj.rank, 1)
        torch.testing.assert_close(
            root.proj.lora_b @ root.proj.lora_a,
            expanded["proj.lora_b"] @ expanded["proj.lora_a"],
        )

    def test_offline_tool_gate_writes_loadable_lineage_bound_artifact(self) -> None:
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.toml"
            input_path = root / "input.safetensors"
            output_path = root / "output.safetensors"
            config_path.write_text(
                '[adapter]\nposthoc_rank_compression = "para"\n',
                encoding="utf-8",
            )
            save_file(
                {
                    "proj.lora_a": torch.eye(2),
                    "proj.lora_b": torch.diag(torch.tensor([4.0, 1.0])),
                    "proj.lora_alpha": torch.tensor([2.0]),
                },
                str(input_path),
            )
            expected_source_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
            summary = compress_adapter(
                config_path=config_path,
                input_path=input_path,
                output_path=output_path,
                input_format="auto",
                policy="rank",
                rank_preservation_ratio=0.5,
                energy_preservation_ratio=0.99,
                overwrite=False,
            )
            payload = load_adapter_payload(output_path)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["retained_total_rank"], 1)
        self.assertEqual(
            payload["metadata"]["manifest"]["source_adapter_sha256"],
            expected_source_sha256,
        )

    def test_fractional_rank_budget_rounds_up(self) -> None:
        state = {
            "proj.lora_a": torch.eye(3),
            "proj.lora_b": torch.diag(torch.tensor([3.0, 2.0, 1.0])),
        }
        result = compress_lora_state_para(
            state,
            source_adapter_sha256=self.SOURCE_SHA256,
            policy="rank",
            rank_preservation_ratio=0.5,
        )
        self.assertEqual(result.manifest["retained_total_rank"], 2)
        self.assertGreaterEqual(
            result.manifest["retained_rank_fraction"],
            0.5,
        )

    def test_manifest_rejects_unreferenced_or_shape_invalid_factors(self) -> None:
        state = {
            "proj.lora_a": torch.eye(2),
            "proj.lora_b": torch.diag(torch.tensor([2.0, 1.0])),
        }
        result = compress_lora_state_para(
            state,
            source_adapter_sha256=self.SOURCE_SHA256,
            policy="rank",
            rank_preservation_ratio=0.5,
        )
        with self.assertRaisesRegex(ValueError, "unreferenced"):
            validate_para_manifest(
                result.manifest,
                {
                    **result.tensors,
                    "para.unit_999999.lora_a": torch.ones(1, 1),
                },
            )
        factor_key = result.manifest["units"][0]["a_tensor"]
        malformed = dict(result.tensors)
        malformed[factor_key] = torch.ones(1, 3)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_para_manifest(result.manifest, malformed)


if __name__ == "__main__":
    unittest.main()
