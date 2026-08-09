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

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
    @staticmethod
    def _refiner_failure_pipeline(*, fail_prompt: bool = False):
        class _Pipeline:
            text_offloaded = False

            @staticmethod
            def release_base_transformer():
                return None

            @staticmethod
            def load_vae(*, device):
                _ = device

            @staticmethod
            def decode_latents_native(_latents):
                return torch.zeros(1, 3, 2, 2)

            @staticmethod
            def encode_video_native(_video, *, generator):
                _ = generator
                return torch.zeros(1, 2, 1, 4, 4)

            @staticmethod
            def offload_vae():
                return None

            @staticmethod
            def load_text_encoder(*, device):
                _ = device

            def encode_prompt(self, prompt, *, device):
                _ = (prompt, device)
                if fail_prompt:
                    raise RuntimeError("prompt failed")
                return torch.zeros(1, 1, 4)

            def offload_text_encoder(self):
                self.text_offloaded = True

        return _Pipeline()

    def test_refiner_partial_load_is_released(self) -> None:
        from mirai.core.models.lingbot_video.refiner import run_refine

        class _Refiner:
            released = False

            @staticmethod
            def has_weights():
                return True

            @staticmethod
            def load(*, device, dtype):
                _ = (device, dtype)
                raise RuntimeError("load failed")

            def release(self):
                self.released = True

        refiner = _Refiner()
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            run_refine(
                pipeline=self._refiner_failure_pipeline(),
                refiner=refiner,
                base_latent=torch.zeros(2, 1, 2, 2),
                prompt="prompt", negative_prompt="", seed=7,
                height=4, width=4, steps=2, cfg_scale=1.0,
                shift=3.0, t_thresh=0.5, sigma_tail_steps=1,
                scheduler="flow_match_euler", device="cpu",
            )
        self.assertTrue(refiner.released)

    def test_inference_moe_token_chunking_is_default_off_and_typed_when_on(self) -> None:
        from mirai.core.inference.session import InferenceSession
        from mirai.core.moe.runtime.token_chunking import MoETokenChunkPolicy

        class _Pipeline:
            policy = None

            def configure_moe_token_chunking(self, policy):
                self.policy = policy

        session = object.__new__(InferenceSession)
        session.pipeline = _Pipeline()
        session.cfg = SimpleNamespace(
            inference=SimpleNamespace(moe_token_chunk_size=0)
        )
        session._arm_moe_token_chunking()
        self.assertIsNone(session.pipeline.policy)

        session.cfg.inference.moe_token_chunk_size = 4096
        session._arm_moe_token_chunking()
        self.assertIsInstance(session.pipeline.policy, MoETokenChunkPolicy)
        self.assertEqual(session.pipeline.policy.token_chunk_size, 4096)

    def test_saved_preview_latent_can_enter_refiner_without_base_denoise(self) -> None:
        import mirai.core.inference.session as session_module
        from mirai.core.inference.session import InferenceSession

        source = torch.randn(2, 3, 4)

        class _Defaults:
            @staticmethod
            def empty_negative_prompt_warning(_prompt, *, model_type):
                _ = model_type
                return None

        class _Provider:
            model_type = "resume_test"
            inference_tasks = ("text_to_video",)

            @staticmethod
            def generation_defaults():
                return _Defaults()

            @staticmethod
            def validate_inference_prompt_rewriter(_name):
                return None

            @staticmethod
            def supports_inference_task(task):
                return task == "text_to_video"

            @staticmethod
            def supports_batched_cfg_inference():
                return False

        class _Pipeline:
            def __init__(self) -> None:
                self.refine_inputs = []
                self.discarded = 0

            def discard_refiner_context(self) -> None:
                self.discarded += 1

            @staticmethod
            def validate_refinement_request(request, *, frames, height, width):
                _ = (frames, height, width)
                return {**request, "resolved": True}

            def refine_inference_latent(self, *, base_latent, **kwargs):
                self.refine_inputs.append((base_latent.detach().clone(), kwargs))
                return base_latent + 1

        pipeline = _Pipeline()
        session = object.__new__(InferenceSession)
        session.pipeline = pipeline
        session.cfg = SimpleNamespace(
            model=SimpleNamespace(type="resume_test"),
            inference=SimpleNamespace(
                task="text_to_video",
                denoising_strength=1.0,
                prompt_rewriter="none",
                cfg_mode="sequential",
            ),
        )
        session.use_native = True
        session._expert_feature_cache = None
        session._compute_device = torch.device("cpu")
        session._compute_dtype = torch.float32
        session._base_placement_dirty = False
        session._residency_strategy = ""
        session.checkpoint = ""
        session.adapter = ""
        session.lora_scale = 1.0
        session.effective_scale = 1.0
        session.lora_format = ""
        session.merge = False
        session.inference_mode = "native"
        session.runtime_policy_notes = []

        with tempfile.TemporaryDirectory() as tmp:
            latent_path = Path(tmp) / "preview.pt"
            output_path = Path(tmp) / "refined.mp4"
            torch.save(source, latent_path)
            with (
                mock.patch.object(
                    session_module, "get_model_family_provider", return_value=_Provider()
                ),
                mock.patch.object(session_module, "runtime_policy_summary", return_value={}),
                mock.patch.object(
                    session_module,
                    "decode_pipeline_media",
                    return_value=output_path,
                ),
            ):
                payload = session.generate(
                    prompt="warm room",
                    negative_prompt="negative",
                    out_path=output_path,
                    decode_latent=str(latent_path),
                    refine={"steps": 1},
                    frames=9,
                    height=64,
                    width=64,
                )

            restored = torch.load(
                output_path.with_suffix(".pt"), map_location="cpu", weights_only=True
            )

        self.assertEqual(pipeline.discarded, 1)
        self.assertEqual(len(pipeline.refine_inputs), 1)
        torch.testing.assert_close(pipeline.refine_inputs[0][0], source)
        torch.testing.assert_close(restored, source + 1)
        self.assertTrue(payload["refined"])
        self.assertEqual(payload["decode_latent"], str(latent_path))
        self.assertTrue(payload["refiner"]["resolved"])

    def test_refiner_prompt_failure_offloads_text_encoder(self) -> None:
        from mirai.core.models.lingbot_video.refiner import run_refine

        class _Refiner:
            released = False

            @staticmethod
            def has_weights():
                return True

            @staticmethod
            def load(*, device, dtype):
                _ = (device, dtype)

            def release(self):
                self.released = True

        pipeline = self._refiner_failure_pipeline(fail_prompt=True)
        refiner = _Refiner()
        with self.assertRaisesRegex(RuntimeError, "prompt failed"):
            run_refine(
                pipeline=pipeline, refiner=refiner,
                base_latent=torch.zeros(2, 1, 2, 2),
                prompt="prompt", negative_prompt="", seed=7,
                height=4, width=4, steps=2, cfg_scale=1.0,
                shift=3.0, t_thresh=0.5, sigma_tail_steps=1,
                scheduler="flow_match_euler", device="cpu",
            )
        self.assertTrue(pipeline.text_offloaded)
        self.assertTrue(refiner.released)

    def test_refiner_offloads_vae_when_reencoding_fails(self) -> None:
        from mirai.core.models.lingbot_video.refiner import run_refine

        class _Pipeline:
            def __init__(self) -> None:
                self.base_released = False
                self.vae_loaded = False
                self.vae_offloaded = False

            def release_base_transformer(self):
                self.base_released = True

            def load_vae(self, *, device):
                self.vae_loaded = device == "cpu"

            def decode_latents_native(self, _latents):
                return torch.zeros(1, 3, 2, 2)

            def encode_video_native(self, _video, *, generator):
                _ = generator
                raise RuntimeError("encode failed")

            def offload_vae(self):
                self.vae_offloaded = True

        class _Refiner:
            @staticmethod
            def has_weights():
                return True

        pipeline = _Pipeline()
        with self.assertRaisesRegex(RuntimeError, "encode failed"):
            run_refine(
                pipeline=pipeline,
                refiner=_Refiner(),
                base_latent=torch.zeros(2, 1, 2, 2),
                prompt="prompt",
                negative_prompt="",
                seed=7,
                height=4,
                width=4,
                steps=2,
                cfg_scale=1.0,
                shift=3.0,
                t_thresh=0.5,
                sigma_tail_steps=1,
                scheduler="flow_match_euler",
                device="cpu",
            )
        self.assertTrue(pipeline.base_released)
        self.assertTrue(pipeline.vae_loaded)
        self.assertTrue(pipeline.vae_offloaded)

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
            self.assertIn("load_text_encoder", on.pipeline.__dict__)
            self.assertIn("offload_text_encoder", on.pipeline.__dict__)
            self.assertIn("offload_vae", on.pipeline.__dict__)
            on.generate(prompt="p", out_path=tmpdir / "on.mp4", **_RUN_KW)
            on.close()
            # close() restores the original hooks.
            self.assertNotIn("load_text_encoder", on.pipeline.__dict__)
            self.assertNotIn("offload_text_encoder", on.pipeline.__dict__)
            self.assertNotIn("offload_vae", on.pipeline.__dict__)

            lat_off = torch.load(tmpdir / "off.pt", map_location="cpu", weights_only=True)
            lat_on = torch.load(tmpdir / "on.pt", map_location="cpu", weights_only=True)
            self.assertTrue(torch.equal(lat_off, lat_on))

    def test_sequential_text_staging_preserves_latent(self) -> None:
        from mirai.core.inference.session import InferenceSession

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            baseline_cfg = tmpdir / "baseline.toml"
            _write_config(baseline_cfg)
            staged_cfg = tmpdir / "staged.toml"
            staged_cfg.write_text(
                baseline_cfg.read_text(encoding="utf-8")
                + "\n[inference]\nstage_text_encoder_before_denoiser = true\n",
                encoding="utf-8",
            )
            baseline = InferenceSession.from_config(str(baseline_cfg))
            baseline.generate(prompt="p", out_path=tmpdir / "baseline.mp4", **_RUN_KW)
            baseline._stage_text_encoder_before_denoiser = True
            baseline._base_placement_dirty = True
            baseline.pipeline.release_base_transformer()
            baseline.generate(prompt="p", out_path=tmpdir / "staged.mp4", **_RUN_KW)
            lat_baseline = torch.load(
                tmpdir / "baseline.pt", map_location="cpu", weights_only=True
            )
            lat_staged = torch.load(
                tmpdir / "staged.pt", map_location="cpu", weights_only=True
            )
            self.assertTrue(torch.equal(lat_baseline, lat_staged))

    def test_merge_request_is_applied_before_reporting_success(self) -> None:
        from mirai.core.inference.session import InferenceSession

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.toml"
            _write_config(cfg_path)
            session = InferenceSession.from_config(
                str(cfg_path), merge=True, place_on_device=False
            )
            self.assertTrue(session.merge)
            self.assertTrue(session.pipeline.is_adapter_merged())

    def test_dirty_base_placement_is_restored_once(self) -> None:
        from mirai.core.inference.session import InferenceSession

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.toml"
            _write_config(cfg_path)
            placements: list[dict] = []

            def _place(pipeline, **kwargs):
                placements.append(dict(kwargs))

            class _CudaDevice:
                type = "cuda"

                def __str__(self):
                    return "cuda"

            session = InferenceSession.from_config(
                str(cfg_path),
                place_fn=_place,
                device_fn=_CudaDevice,
                dtype_fn=lambda cfg: torch.bfloat16,
            )
            self.assertEqual(len(placements), 1)
            session._base_placement_dirty = True
            session._ensure_base_placement()
            session._ensure_base_placement()
            self.assertEqual(len(placements), 2)


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


@unittest.skipIf(torch is None, "torch not installed")
class FamilyGenerationDefaultsTests(unittest.TestCase):
    """A family's released sampling values reach the generic CLI through the
    provider capability, and an empty negative prompt never arrives silently."""

    def _defaults(self, model_type: str):
        from mirai.core.models.providers import resolve_family_generation_defaults

        return resolve_family_generation_defaults(model_type)

    def test_declaring_family_supplies_defaults_when_request_is_omitted(self) -> None:
        declared = self._defaults("lingbot-video")
        self.assertIsNotNone(declared.negative_prompt)
        # The declared text is the family's released asset, not a placeholder.
        parsed = json.loads(declared.resolve_negative_prompt(None))
        self.assertIn("universal_negative", parsed)
        self.assertEqual(declared.resolve_steps(None, fallback=20), declared.steps)
        self.assertEqual(
            declared.resolve_cfg_scale(None, fallback=5.0), declared.cfg_scale
        )
        self.assertEqual(
            declared.resolve_scheduler(None, fallback="euler"), declared.scheduler
        )
        # An ordinary correct run -- omitted flag, family default applied --
        # produces no degradation warning.
        self.assertIsNone(
            declared.empty_negative_prompt_warning(
                declared.resolve_negative_prompt(None), model_type="lingbot-video"
            )
        )

    def test_magi2_declares_the_released_preview_profile(self) -> None:
        declared = self._defaults("magi2-preview")
        self.assertEqual(declared.steps, 100)
        self.assertEqual(declared.cfg_scale, 5.0)
        self.assertEqual(declared.scheduler, "unipc")
        self.assertEqual(
            (declared.width, declared.height, declared.frames), (896, 512, 249)
        )
        negative = declared.resolve_negative_prompt(None)
        self.assertIn("blurred details", negative)
        self.assertIn("digital clipping", negative)

    def test_explicit_request_always_wins_and_empty_is_reported(self) -> None:
        declared = self._defaults("lingbot-video")
        self.assertEqual(declared.resolve_negative_prompt("blurry"), "blurry")
        self.assertEqual(declared.resolve_steps(3, fallback=20), 3)
        self.assertEqual(declared.resolve_cfg_scale(1.0, fallback=5.0), 1.0)
        self.assertEqual(declared.resolve_scheduler("euler", fallback="euler"), "euler")
        # Explicitly empty is honored, not overridden -- and made visible.
        self.assertEqual(declared.resolve_negative_prompt(""), "")
        warning = declared.empty_negative_prompt_warning(
            "", model_type="lingbot-video"
        )
        self.assertIsNotNone(warning)
        self.assertIn("empty", warning)

    def test_family_declaring_none_acquires_no_empty_string_default(self) -> None:
        none_declared = self._defaults("sparse_moe_test")
        self.assertIsNone(none_declared.negative_prompt)
        self.assertIsNone(none_declared.steps)
        self.assertIsNone(none_declared.cfg_scale)
        self.assertIsNone(none_declared.scheduler)
        self.assertIsNone(none_declared.width)
        self.assertIsNone(none_declared.height)
        self.assertIsNone(none_declared.frames)
        self.assertFalse(none_declared.declares_negative_prompt())
        # Caller fallbacks survive untouched.
        self.assertEqual(none_declared.resolve_steps(None, fallback=20), 20)
        self.assertEqual(none_declared.resolve_cfg_scale(None, fallback=5.0), 5.0)
        self.assertEqual(
            none_declared.resolve_scheduler(None, fallback="euler"), "euler"
        )
        self.assertEqual(none_declared.resolve_width(None, fallback=832), 832)
        self.assertEqual(none_declared.resolve_height(None, fallback=480), 480)
        self.assertEqual(none_declared.resolve_frames(None, fallback=17), 17)
        self.assertEqual(none_declared.resolve_negative_prompt(None), "")
        # A family that declares nothing cannot be degraded by an empty one.
        self.assertIsNone(
            none_declared.empty_negative_prompt_warning("", model_type="sparse_moe_test")
        )

    def test_cli_applies_declared_defaults_and_honors_explicit_request(self) -> None:
        """The generic CLI reads the capability, never a family name."""
        from mirai.core.models.providers import FamilyGenerationDefaults

        infer = _load_module("mirai_infer_defaults", "scripts/infer.py")
        source = (REPO_ROOT / "scripts" / "infer.py").read_text(encoding="utf-8")
        self.assertNotIn("lingbot", source.lower())
        self.assertNotIn("magi", source.lower())

        declared = FamilyGenerationDefaults(
            negative_prompt="declared negative",
            steps=3,
            cfg_scale=2.0,
            scheduler="euler",
            width=64,
            height=64,
            frames=5,
        )
        infer.resolve_family_generation_defaults = lambda _model_type: declared

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg_path = tmpdir / "config.toml"
            _write_config(cfg_path)

            def _run(extra: list[str], out_name: str) -> dict:
                argv = [
                    "infer.py",
                    "--config", str(cfg_path),
                    "--prompt", "a red cube spinning",
                    "--seed", "7",
                    "--out", str(tmpdir / out_name),
                ] + extra
                old_argv = sys.argv
                buffer = io.StringIO()
                try:
                    sys.argv = argv
                    with contextlib.redirect_stdout(buffer):
                        self.assertEqual(infer.main(), 0)
                finally:
                    sys.argv = old_argv
                return json.loads(buffer.getvalue())

            omitted = _run([], "omitted.mp4")
            self.assertEqual(omitted["negative_prompt"], "declared negative")

            explicit = _run(["--negative-prompt", "hazy"], "explicit.mp4")
            self.assertEqual(explicit["negative_prompt"], "hazy")

            emptied = _run(["--negative-prompt", ""], "empty.mp4")
            self.assertEqual(emptied["negative_prompt"], "")

    def test_session_warns_when_a_declared_negative_prompt_is_discarded(self) -> None:
        """The session is the backstop: every caller that reaches an empty
        negative prompt for a declaring family sees it."""
        from mirai.core.inference.session import InferenceSession
        from mirai.core.models.providers import (
            FamilyGenerationDefaults,
            get_model_family_provider,
        )

        provider = get_model_family_provider("sparse_moe_test")
        assert provider is not None
        original = type(provider).generation_defaults
        type(provider).generation_defaults = lambda _self: FamilyGenerationDefaults(
            negative_prompt="declared negative"
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                cfg_path = tmpdir / "config.toml"
                _write_config(cfg_path)
                session = InferenceSession.from_config(str(cfg_path))
                try:
                    errors = io.StringIO()
                    with contextlib.redirect_stderr(errors):
                        session.generate(
                            prompt="a red cube spinning",
                            out_path=tmpdir / "empty.mp4",
                            negative_prompt="",
                            **_RUN_KW,
                        )
                    self.assertIn("declares a default negative prompt", errors.getvalue())

                    quiet = io.StringIO()
                    with contextlib.redirect_stderr(quiet):
                        session.generate(
                            prompt="a red cube spinning",
                            out_path=tmpdir / "declared.mp4",
                            negative_prompt="declared negative",
                            **_RUN_KW,
                        )
                    self.assertNotIn("declares a default negative prompt", quiet.getvalue())
                finally:
                    session.close()
        finally:
            type(provider).generation_defaults = original

    def test_family_entrypoint_consumes_the_same_capability(self) -> None:
        """inference/lingbot_video/generate.py keeps no private copy."""
        from mirai.core.models.providers import resolve_family_generation_defaults

        generate = _load_module(
            "lingbot_generate_defaults", "inference/lingbot_video/generate.py"
        )
        declared = resolve_family_generation_defaults("lingbot-video")
        self.assertEqual(generate.default_negative_prompt(), declared.negative_prompt)
        profile = generate.DEFAULT_INFERENCE_PROFILE
        self.assertEqual(profile.steps, declared.steps)
        self.assertEqual(profile.cfg_scale, declared.cfg_scale)
        self.assertEqual(profile.scheduler, declared.scheduler)
        # Omitting the flag omits it downstream, so the generic CLI resolves it
        # through the same capability instead of receiving a second copy.
        args = generate.parse_args(["--prompt", '{"scene":"test"}'])
        self.assertNotIn(
            "--negative-prompt", generate.build_infer_argv(args, config_path="c.toml")
        )


@unittest.skipIf(torch is None, "torch not installed")
class InferenceSessionWeightResidencyTests(unittest.TestCase):
    """``inference.blocks_to_swap`` reaches the residency owner from the CLI path.

    Block swapping was reachable only from the training entrypoint; these pin
    that the shipped inference session arms the same owner, and that a config
    without the opt-in still resolves the fully resident path.
    """

    _BLOCK_SWAP_CONFIG = (
        'preset = "sparse_moe_test"\n\n'
        '[model]\ntype = "sparse_moe_test"\n\n'
        "[inference]\nblocks_to_swap = 2\nblock_swap_mode = \"async\"\n\n"
        '[memory]\nweight_residency_strategy = "block_swap"\n'
    )

    def test_default_inference_config_keeps_the_resident_path(self) -> None:
        from mirai.core.inference.session import InferenceSession

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.toml"
            _write_config(cfg_path)
            session = InferenceSession.from_config(str(cfg_path))
            self.assertEqual(session._residency_strategy, "disabled")
            self.assertIsNone(session.pipeline._block_swap_manager)

    def test_block_swap_config_arms_the_session_residency_owner(self) -> None:
        from mirai.core.inference.session import InferenceSession
        from mirai.core.training.residency.block_swap import BlockSwapManager

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.toml"
            cfg_path.write_text(self._BLOCK_SWAP_CONFIG, encoding="utf-8")
            session = InferenceSession.from_config(str(cfg_path))
            self.assertEqual(session._residency_strategy, "block_swap")
            self.assertIsInstance(
                session.pipeline._block_swap_manager, BlockSwapManager
            )
            state = session.pipeline.get_block_swap_state()
            self.assertEqual(state["blocks_to_swap"], 2)
            self.assertEqual(state["mode"], "async")
            self.assertEqual(state["weight_residency_execution_mode"], "inference")

    def test_block_swap_without_a_transport_fails_explicitly(self) -> None:
        from mirai.config.loader import load_config

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.toml"
            cfg_path.write_text(
                'preset = "sparse_moe_test"\n\n'
                '[model]\ntype = "sparse_moe_test"\n\n'
                "[inference]\nblocks_to_swap = 2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Exception, "weight_residency_strategy"):
                load_config(cfg_path)


if __name__ == "__main__":
    unittest.main()
