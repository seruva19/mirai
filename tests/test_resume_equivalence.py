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
    def _assert_state_tree_equal(self, left, right, *, path: str = "root") -> None:
        if torch.is_tensor(left) or torch.is_tensor(right):
            self.assertTrue(
                torch.is_tensor(left) and torch.is_tensor(right),
                path,
            )
            self.assertTrue(torch.equal(left, right), path)
            return
        if isinstance(left, dict) or isinstance(right, dict):
            self.assertIsInstance(left, dict, path)
            self.assertIsInstance(right, dict, path)
            self.assertEqual(set(left), set(right), path)
            for key in sorted(left, key=str):
                self._assert_state_tree_equal(
                    left[key],
                    right[key],
                    path=f"{path}.{key}",
                )
            return
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            self.assertEqual(type(left), type(right), path)
            self.assertEqual(len(left), len(right), path)
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                self._assert_state_tree_equal(
                    left_item,
                    right_item,
                    path=f"{path}[{index}]",
                )
            return
        self.assertEqual(left, right, path)

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
                gradient_cpu_offload = true
                ema_enabled = true
                ema_decay = 0.95
                posthoc_ema_enabled = true
                posthoc_ema_profile_stds = [0.05, 0.1]
                posthoc_ema_snapshot_every_n_steps = 1

                [optimizer]
                type = "adamw"
                stochastic_rounding = true

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
        split_step = 4
        final_step = 24
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
                max_steps=final_step,
            )
            self._write_config(
                path=cfg_part,
                data_dir=data_dir,
                cache_path=cache_path,
                out_dir=resumed_out,
                max_steps=split_step,
            )
            self._write_config(
                path=cfg_resume,
                data_dir=data_dir,
                cache_path=cache_path,
                out_dir=resumed_out,
                max_steps=final_step,
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

            resume_ckpt = resumed_out / "checkpoints" / f"step_{split_step}.pt"
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
            state_keys = (
                "trainer_state",
                "strategy_metadata",
                "optimizer_state",
                "scheduler_state",
                "rng_state",
                "torch_rng_state",
                "skipped_steps",
                "consecutive_skipped_steps",
                "early_stop_state",
                "ema_state",
                "posthoc_ema_state",
            )
            for key in state_keys:
                self._assert_state_tree_equal(
                    full_last[key],
                    resumed_last[key],
                    path=key,
                )
            # Compare the complete post-resume trajectory, not only the endpoint.
            for step in range(split_step + 1, final_step + 1):
                full_step = torch.load(
                    full_out / "checkpoints" / f"step_{step}.pt",
                    map_location="cpu",
                )
                resumed_step = torch.load(
                    resumed_out / "checkpoints" / f"step_{step}.pt",
                    map_location="cpu",
                )
                for key in state_keys:
                    self._assert_state_tree_equal(
                        full_step[key],
                        resumed_step[key],
                        path=f"step_{step}.{key}",
                    )


if __name__ == "__main__":
    unittest.main()
