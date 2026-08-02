"""Behavioral invariants of the load-once InferenceSession.

The session must be a pure amortization of ``scripts/infer.py``: building the
model once and calling ``generate()`` many times changes *when* the fixed build
cost is paid, never the numerics. These tests pin that equivalence on the tiny
``sparse_moe_test`` validation fixture (CPU, no real assets):

  (a) session.generate() x2 (same seed) == two independent scripts/infer.py
      runs: byte-identical latents (.pt) and payloads (modulo output paths);
  (c) --batch-prompts builds the model ONCE and emits N outputs;
  (d) the resident-weight flags default OFF are byte-identical to today, and
      switching them ON does not change numerics (only offload timing).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_config(path: Path) -> None:
    path.write_text(
        'preset = "sparse_moe_test"\n\n[model]\ntype = "sparse_moe_test"\n',
        encoding="utf-8",
    )


def _write_checkpoint(path: Path, cfg_path: Path) -> None:
    from mirai.config.loader import load_config
    from mirai.core.builtins import register_builtin_components
    from mirai.core.persistence.checkpoints import save_checkpoint
    from mirai.core.training.trainer import Trainer

    register_builtin_components()
    trainer = Trainer(load_config(cfg_path))
    save_checkpoint(path, {"global_step": 0, "trainer_state": trainer.state_dict()})


_RUN_KW = dict(seed=7, frames=5, height=64, width=64, steps=2, cfg_scale=1.0)


@unittest.skipIf(torch is None, "torch not installed")
class InferenceSessionEquivalenceTests(unittest.TestCase):
    def _infer_cli(self, *, cfg_path, checkpoint, out_path) -> dict:
        result = subprocess.run(
            [
                sys.executable, "scripts/infer.py",
                "--config", str(cfg_path),
                "--checkpoint", str(checkpoint),
                "--prompt", "a red cube spinning",
                "--seed", str(_RUN_KW["seed"]),
                "--frames", str(_RUN_KW["frames"]),
                "--height", str(_RUN_KW["height"]),
                "--width", str(_RUN_KW["width"]),
                "--steps", str(_RUN_KW["steps"]),
                "--cfg-scale", str(_RUN_KW["cfg_scale"]),
                "--out", str(out_path),
            ],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return json.loads(result.stdout)

    def test_session_generate_twice_matches_two_infer_runs(self) -> None:
        """(a) Two generate() calls (same seed) on one session are byte-identical
        to each other and to two independent scripts/infer.py subprocess runs."""
        from mirai.core.inference.session import InferenceSession

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg_path = tmpdir / "config.toml"
            _write_config(cfg_path)
            checkpoint = tmpdir / "ckpt.pt"
            _write_checkpoint(checkpoint, cfg_path)

            # Two independent CLI runs (fresh process each).
            cli_a = self._infer_cli(
                cfg_path=cfg_path, checkpoint=checkpoint, out_path=tmpdir / "cli_a.mp4"
            )
            cli_b = self._infer_cli(
                cfg_path=cfg_path, checkpoint=checkpoint, out_path=tmpdir / "cli_b.mp4"
            )

            # One session, two generate() calls.
            session = InferenceSession.from_config(
                str(cfg_path), checkpoint=str(checkpoint)
            )
            sess_a = session.generate(
                prompt="a red cube spinning", out_path=tmpdir / "sess_a.mp4", **_RUN_KW
            )
            sess_b = session.generate(
                prompt="a red cube spinning", out_path=tmpdir / "sess_b.mp4", **_RUN_KW
            )

            def _latent(stem: str):
                return torch.load(
                    tmpdir / f"{stem}.pt", map_location="cpu", weights_only=True
                )

            lat_cli_a = _latent("cli_a")
            lat_cli_b = _latent("cli_b")
            lat_sess_a = _latent("sess_a")
            lat_sess_b = _latent("sess_b")

            # Byte-identical latents across all four runs.
            self.assertTrue(torch.equal(lat_cli_a, lat_cli_b))
            self.assertTrue(torch.equal(lat_sess_a, lat_sess_b))
            self.assertTrue(torch.equal(lat_cli_a, lat_sess_a))

            # Payloads identical modulo the run-specific output path.
            def _strip(p: dict) -> dict:
                q = dict(p)
                q.pop("output_path")
                return q

            self.assertEqual(_strip(cli_a), _strip(sess_a))
            self.assertEqual(_strip(cli_a), _strip(cli_b))
            self.assertEqual(_strip(sess_a), _strip(sess_b))

    def test_resident_flags_default_off_are_byte_identical(self) -> None:
        """(d) The resident flags default OFF leave the pipeline offload hooks
        untouched and produce the same latent; turning them ON keeps the latent
        byte-identical (amortization must not change numerics)."""
        from mirai.core.inference.session import InferenceSession

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg_path = tmpdir / "config.toml"
            _write_config(cfg_path)
            checkpoint = tmpdir / "ckpt.pt"
            _write_checkpoint(checkpoint, cfg_path)

            off = InferenceSession.from_config(str(cfg_path), checkpoint=str(checkpoint))
            # Default off: no instance-level offload override installed.
            self.assertNotIn("offload_text_encoder", off.pipeline.__dict__)
            self.assertNotIn("offload_vae", off.pipeline.__dict__)
            off.generate(prompt="p", out_path=tmpdir / "off.mp4", **_RUN_KW)

            on = InferenceSession.from_config(
                str(cfg_path),
                checkpoint=str(checkpoint),
                keep_text_encoder_resident=True,
                keep_vae_resident=True,
            )
            # On: offload hooks replaced by no-ops so weights stay resident.
            self.assertIn("offload_text_encoder", on.pipeline.__dict__)
            self.assertIn("offload_vae", on.pipeline.__dict__)
            on.generate(prompt="p", out_path=tmpdir / "on.mp4", **_RUN_KW)
            on.close()
            # close() restores the original hooks.
            self.assertNotIn("offload_text_encoder", on.pipeline.__dict__)
            self.assertNotIn("offload_vae", on.pipeline.__dict__)

            lat_off = torch.load(tmpdir / "off.pt", map_location="cpu", weights_only=True)
            lat_on = torch.load(tmpdir / "on.pt", map_location="cpu", weights_only=True)
            self.assertTrue(torch.equal(lat_off, lat_on))


@unittest.skipIf(torch is None, "torch not installed")
class BatchPromptsTests(unittest.TestCase):
    def test_distilled_profile_applies_four_step_adapter_defaults(self) -> None:
        generate = _load_module(
            "lingbot_generate_distilled_profile",
            "inference/lingbot_video/generate.py",
        )

        args = generate.parse_args(
            [
                "--prompt",
                '{"scene":"test"}',
                "--adapter",
                "distilled.safetensors",
                "--inference-profile",
                "distilled-4step",
            ]
        )

        self.assertEqual(args.steps, 4)
        self.assertEqual(args.scheduler, "euler")
        self.assertEqual(args.cfg_scale, 1.0)
        self.assertEqual(args.adapter_rank, 128)
        self.assertEqual(args.adapter_alpha, 128.0)
        self.assertEqual(args.adapter_preset, "attn_shared_mlp")

    def test_batch_builds_model_once_and_emits_n_outputs(self) -> None:
        """(c) --batch-prompts builds ONE session (Trainer constructed exactly
        once) and produces one output per prompt line."""
        generate = _load_module(
            "lingbot_generate_batch", "inference/lingbot_video/generate.py"
        )
        import mirai.core.inference.session as sess_mod

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg_path = tmpdir / "config.toml"
            _write_config(cfg_path)

            outs = [tmpdir / f"clip_{i}.mp4" for i in range(3)]
            batch_file = tmpdir / "prompts.jsonl"
            batch_file.write_text(
                "\n".join(
                    json.dumps({"prompt": f"prompt {i}", "out": str(outs[i])})
                    for i in range(3)
                ),
                encoding="utf-8",
            )

            build_count = {"n": 0}
            original_trainer = sess_mod.Trainer

            def _counting_trainer(cfg):
                build_count["n"] += 1
                return original_trainer(cfg)

            sess_mod.Trainer = _counting_trainer
            try:
                rc = generate.main(
                    [
                        "--batch-prompts", str(batch_file),
                        "--config", str(cfg_path),
                        "--seed", "7",
                        "--frames", "5",
                        "--height", "64",
                        "--width", "64",
                        "--steps", "2",
                        "--cfg-scale", "1.0",
                    ]
                )
            finally:
                sess_mod.Trainer = original_trainer

            self.assertEqual(rc, 0)
            # Model built exactly once for the whole batch.
            self.assertEqual(build_count["n"], 1)
            # One latent dump per prompt line.
            for out in outs:
                self.assertTrue(out.with_suffix(".pt").exists(), msg=str(out))


if __name__ == "__main__":
    unittest.main()
