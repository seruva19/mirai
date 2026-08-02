from __future__ import annotations

# Shared numerical parity contract for optimized MoE paths.

import tempfile
import unittest
from pathlib import Path

from mirai.core.parity.harness import compare_outputs, write_parity_result

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class ParityHarnessTests(unittest.TestCase):
    def test_compare_outputs_pass(self) -> None:
        ref = torch.tensor([1.0, 2.0, 3.0])
        cand = torch.tensor([1.0, 2.000001, 2.999999])
        result = compare_outputs(
            name="case-pass",
            candidate=cand,
            reference=ref,
            atol=1e-4,
            rtol=1e-4,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.count, 3)

    def test_compare_outputs_fail(self) -> None:
        ref = torch.tensor([1.0, 2.0, 3.0])
        cand = torch.tensor([1.2, 2.2, 3.2])
        result = compare_outputs(
            name="case-fail",
            candidate=cand,
            reference=ref,
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertFalse(result.passed)

    def test_write_result(self) -> None:
        result = compare_outputs(
            name="case-write",
            candidate=torch.tensor([1.0]),
            reference=torch.tensor([1.0]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "parity.json"
            write_parity_result(path, result)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
