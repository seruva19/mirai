from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import pytest
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

pytestmark = pytest.mark.moe_contract


@unittest.skipIf(torch is None, "torch not installed")
class ThinSliceE2ETests(unittest.TestCase):
    def test_cache_train_infer(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_dir = tmpdir / "cache"
            out_dir = tmpdir / "outputs"
            infer_out = out_dir / "infer_output.pt"
            data_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            torch.save(torch.tensor([0.1], dtype=torch.float32), data_dir / "sample0.pt")
            (data_dir / "sample0.txt").write_text("sample caption", encoding="utf-8")

            cfg_path = tmpdir / "config.toml"
            cfg_path.write_text(
                textwrap.dedent(
                    f"""
                    preset = "sparse_moe_test"

                    [model]
                    type = "sparse_moe_test"

                    [dataset]
                    path = "{data_dir.as_posix()}"
                    cache_path = "{(cache_dir / 'dataset_cache.pt').as_posix()}"

                    [training]
                    max_steps = 3
                    batch_size = 1
                    gradient_accumulation = 1

                    [logging]
                    output_dir = "{out_dir.as_posix()}"
                    save_every_n_steps = 1
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            cmds = [
                [
                    sys.executable,
                    "scripts/cache.py",
                    "--config",
                    str(cfg_path),                ],
                [
                    sys.executable,
                    "scripts/train.py",
                    "--config",
                    str(cfg_path),                ],
                [
                    sys.executable,
                    "scripts/infer.py",
                    "--config",
                    str(cfg_path),
                    "--adapter",
                    str(out_dir / "adapter.pt"),
                    "--prompt",
                    "a cat walking",
                    "--seed",
                    "42",
                    "--frames",
                    "4",
                    "--height",
                    "16",
                    "--width",
                    "16",
                    "--steps",
                    "1",
                    "--cfg-scale",
                    "1.0",
                    "--out",
                    str(infer_out),                ],
            ]
            for cmd in cmds:
                result = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": "",
                        "MIRAI_GPU_LEASE_PATH": str(tmpdir / ".mirai_gpu_lease.lock"),
                    },
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
                )

            self.assertTrue((cache_dir / "dataset_cache.pt").exists())
            self.assertTrue((out_dir / "checkpoints" / "last.pt").exists())
            self.assertTrue((out_dir / "adapter.pt").exists())
            self.assertTrue((out_dir / "run_manifest.json").exists())
            self.assertTrue(infer_out.exists())


if __name__ == "__main__":
    unittest.main()
