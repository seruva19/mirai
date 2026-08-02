from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

# Colocated behavioral contract for adapter host capabilities.
from pathlib import Path

import pytest

from mirai.core.training.adapters import (
    adapter_state_to_diffusers,
    adapter_state_to_kohya,
    normalize_adapter_state,
    save_kohya_adapter_safetensors,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


pytestmark = pytest.mark.production_contract
ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)


@unittest.skipIf(torch is None, "torch not installed")
class AdapterHostContractTests(unittest.TestCase):
    def test_dora_magnitude_round_trips_through_portable_names(self) -> None:
        native = {
            "block.proj.lora_a": torch.randn(2, 3),
            "block.proj.lora_b": torch.randn(4, 2),
            "block.proj.lora_alpha": torch.tensor([2.0]),
            "block.proj.dora_magnitude": torch.randn(4),
        }
        for format_name, exported in (
            ("kohya", adapter_state_to_kohya(native)),
            ("diffusers", adapter_state_to_diffusers(native)),
        ):
            with self.subTest(format=format_name):
                restored = normalize_adapter_state(
                    exported,
                    lora_format=format_name,
                )
                self.assertTrue(
                    torch.equal(
                        restored["block.proj.dora_magnitude"],
                        native["block.proj.dora_magnitude"],
                    )
                )

    @staticmethod
    def _write_adapter(path: Path) -> None:
        save_kohya_adapter_safetensors(
            path,
            adapter_state={
                "lora_a": torch.tensor(0.2),
                "lora_b": torch.tensor(0.3),
                "lora_alpha": torch.tensor(1.0),
            },
            rank=1,
            alpha=1.0,
            target_modules=["attn"],
            model_path="./models/sparse_moe_test",
        )

    def test_supported_host_formats_normalize_to_native_adapter_state(self) -> None:
        cases = {
            "kohya": {
                "lora_unet_lora_a.weight": torch.tensor([[0.2]]),
                "lora_unet_lora_b.weight": torch.tensor([[0.3]]),
                "lora_unet.alpha": torch.tensor([1.0]),
            },
            "diffusers": {
                "unet.lora_a.weight": torch.tensor([[0.4]]),
                "unet.lora_b.weight": torch.tensor([[0.5]]),
                "unet.lora_alpha": torch.tensor([2.0]),
            },
            "peft": {
                "adapter_state": {
                    "lora_a": torch.tensor(0.7),
                    "lora_b": torch.tensor(0.8),
                    "lora_alpha": torch.tensor(1.0),
                }
            },
            "lycoris": {
                "lycoris.lokr_a": torch.tensor([[0.6]]),
                "lycoris.lokr_b": torch.tensor([[0.4]]),
                "lycoris.alpha": torch.tensor([1.2]),
            },
        }
        expected_a = {"kohya": 0.2, "diffusers": 0.4, "peft": 0.7, "lycoris": 0.6}
        for format_name, payload in cases.items():
            with self.subTest(format=format_name):
                state = normalize_adapter_state(payload, lora_format=format_name)
                self.assertTrue(
                    torch.allclose(
                        state["lora_a"],
                        torch.tensor(expected_a[format_name]),
                    )
                )
                self.assertIn("lora_b", state)

    def test_lightx2v_dotted_lora_keys_normalize_with_mixed_peft_keys(self) -> None:
        payload = {
            "blocks.0.attn.to_q.lora.down.weight": torch.ones(2, 3),
            "blocks.0.attn.to_q.lora.up.weight": torch.ones(4, 2),
            "blocks.0.attn.to_out.lora_A.weight": torch.ones(2, 4),
            "blocks.0.attn.to_out.lora_B.weight": torch.ones(4, 2),
        }

        state = normalize_adapter_state(payload)

        self.assertEqual(
            set(state),
            {
                "blocks.0.attn.to_q.lora_a",
                "blocks.0.attn.to_q.lora_b",
                "blocks.0.attn.to_out.lora_a",
                "blocks.0.attn.to_out.lora_b",
            },
        )
        self.assertEqual(tuple(state["blocks.0.attn.to_q.lora_a"].shape), (2, 3))

    def test_exported_adapter_passes_declared_host_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_path = root / "adapter.safetensors"
            report_path = root / "compatibility.json"
            self._write_adapter(adapter_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/tools/check_lora_host_compat.py",
                    "--lora",
                    str(adapter_path),
                    "--out",
                    str(report_path),
                    "--strict",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "ok")
        hosts = {entry["host"]: entry for entry in report["hosts"]}
        self.assertTrue(hosts["A1111"]["passed"])
        self.assertTrue(hosts["InvokeAI"]["passed"])

    @unittest.skipUnless(
        importlib.util.find_spec("requests") is not None,
        "Mirai tools extra is not installed",
    )
    def test_unreachable_comfyui_is_reported_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_path = root / "adapter.safetensors"
            workflow_path = root / "workflow.json"
            report_path = root / "report.json"
            self._write_adapter(adapter_path)
            workflow_path.write_text(
                json.dumps({"1": {"inputs": {"lora_path": "${LORA_PATH}"}}}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/tools/check_comfyui_lora_load.py",
                    "--lora",
                    str(adapter_path),
                    "--workflow",
                    str(workflow_path),
                    "--server",
                    "https://example.invalid/api",
                    "--out",
                    str(report_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, msg=result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "skipped")
        self.assertIn("not reachable", report.get("error", ""))


if __name__ == "__main__":
    unittest.main()
