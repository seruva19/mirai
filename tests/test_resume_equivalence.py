from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
import os

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class ResumeEquivalenceTests(unittest.TestCase):
    def _write_config(
        self,
        *,
        path: Path,
        data_dir: Path,
        cache_path: Path,
        out_dir: Path,
        max_steps: int,
    ) -> None:
        path.write_text(
            textwrap.dedent(
                f"""
                preset = "sparse_moe_test"

                [model]
                type = "sparse_moe_test"

                [dataset]
                path = "{data_dir.as_posix()}"
                cache_path = "{cache_path.as_posix()}"

                [training]
                max_steps = {max_steps}
                batch_size = 1
                gradient_accumulation = 1
                seed = 999

                [logging]
                output_dir = "{out_dir.as_posix()}"
                save_every_n_steps = 1
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

    def _run(self, repo_root: Path, cmd: list[str]) -> None:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "MIRAI_GPU_LEASE_PATH": str(repo_root / ".tmp_test_resume_equivalence.lock"),
            },
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    def test_interrupted_resume_matches_uninterrupted(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            for idx, value in enumerate([0.1, 0.2, 0.3, 0.4]):
                torch.save(torch.tensor([value], dtype=torch.float32), data_dir / f"sample{idx}.pt")
                (data_dir / f"sample{idx}.txt").write_text(f"caption {idx}", encoding="utf-8")

            cache_path = tmpdir / "cache" / "dataset_cache.pt"
            full_out = tmpdir / "run_full"
            resumed_out = tmpdir / "run_resume"
            cfg_full = tmpdir / "cfg_full.toml"
            cfg_part = tmpdir / "cfg_part.toml"
            cfg_resume = tmpdir / "cfg_resume.toml"
            self._write_config(
                path=cfg_full,
                data_dir=data_dir,
                cache_path=cache_path,
                out_dir=full_out,
                max_steps=6,
            )
            self._write_config(
                path=cfg_part,
                data_dir=data_dir,
                cache_path=cache_path,
                out_dir=resumed_out,
                max_steps=3,
            )
            self._write_config(
                path=cfg_resume,
                data_dir=data_dir,
                cache_path=cache_path,
                out_dir=resumed_out,
                max_steps=6,
            )

            self._run(
                repo_root,
                [sys.executable, "scripts/cache.py", "--config", str(cfg_full)],
            )
            self._run(
                repo_root,
                [sys.executable, "scripts/train.py", "--config", str(cfg_full)],
            )
            self._run(
                repo_root,
                [sys.executable, "scripts/train.py", "--config", str(cfg_part)],
            )

            resume_ckpt = resumed_out / "checkpoints" / "step_3.pt"
            self._run(
                repo_root,
                [
                    sys.executable,
                    "scripts/train.py",
                    "--config",
                    str(cfg_resume),                    "--resume",
                    str(resume_ckpt),
                ],
            )

            full_last = torch.load(full_out / "checkpoints" / "last.pt", map_location="cpu")
            resumed_last = torch.load(resumed_out / "checkpoints" / "last.pt", map_location="cpu")

            self.assertEqual(int(full_last["global_step"]), int(resumed_last["global_step"]))
            full_pipe = full_last["trainer_state"]["pipeline"]
            resumed_pipe = resumed_last["trainer_state"]["pipeline"]
            self.assertTrue(torch.allclose(full_pipe["lora_a"], resumed_pipe["lora_a"]))
            self.assertTrue(torch.allclose(full_pipe["lora_b"], resumed_pipe["lora_b"]))


if __name__ == "__main__":
    unittest.main()
