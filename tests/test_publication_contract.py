from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from mirai.config.loader import load_config
from scripts.download import _resolve_hf_token


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_SUFFIXES = {".md", ".py", ".toml", ".json", ".yaml", ".yml"}
IGNORED_PUBLICATION_ROOTS = {
    ".dev-private",
    ".git",
    ".agents",
    ".runtime",
    ".tmp",
    "tmp",
    "tools",
    "models",
    "outputs",
    "cache",
    "logs",
    "artifacts",
}


def _publication_relative_paths() -> list[str]:
    """Repository-relative POSIX paths that a published artifact may contain.

    Git owns the answer: tracked files plus untracked files that survive the
    standard exclude rules. Anything matched by ``.gitignore`` -- local model
    snapshots, caches, outputs -- is absent by construction, so no hand-written
    exclusion list can drift away from the real ignore semantics.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return [raw_path for raw_path in result.stdout.decode("utf-8").split("\0") if raw_path]


def _stage_publication_workspace(destination: Path) -> None:
    """Materialize the publishable file set under ``destination``."""
    for relative in _publication_relative_paths():
        source = ROOT / relative
        # Index entries can outlive the working tree (gitlinks, pending deletions).
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _public_text_files() -> list[Path]:
    files: list[Path] = []
    for relative_path in _publication_relative_paths():
        relative = Path(relative_path)
        path = ROOT / relative
        if not path.is_file() or path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
            continue
        if relative.parts[0] in IGNORED_PUBLICATION_ROOTS:
            continue
        files.append(path)
    return files


class PublicationContractTests(unittest.TestCase):
    def test_project_uses_one_mandatory_dependency_set(self) -> None:
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("[project.optional-dependencies]", metadata)
        for dependency in (
            "av>=12.0.0",
            "bitsandbytes==0.49.2",
            "pydantic-settings==2.14.1",
            "pytest>=8.4.2",
            "requests>=2.32.0",
            "ruff==0.14.8",
            "scipy>=1.11.0",
            "setuptools>=77",
            "tensorboard>=2.19.0",
            "tqdm>=4.66.0",
            "triton>=3.0.0; sys_platform == 'linux'",
            "unfoldNd==0.2.3",
            "wandb>=0.18",
            "wheel>=0.45.1",
        ):
            self.assertIn(f'"{dependency}"', metadata)

        requirements = (ROOT / "requirements-cu126.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("-e .", requirements.splitlines())
        self.assertFalse((ROOT / "requirements-cu126-linux.txt").exists())

    def test_public_model_download_does_not_require_credentials(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.dict(
            sys.modules, {"huggingface_hub": None}
        ):
            self.assertIsNone(_resolve_hf_token())

    def test_distribution_archives_preserve_the_public_runtime_surface(self) -> None:
        # TemporaryDirectory removes the tree when the block exits for any
        # reason, including an assertion failure or a build error, and reports
        # rather than swallows a removal failure.
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory) / "source"
            workspace.mkdir(parents=True)
            _stage_publication_workspace(workspace)
            distribution_directory = workspace / "dist"
            command = (
                "import setuptools.build_meta as backend; "
                "backend.build_sdist('dist'); "
                "backend.build_wheel('dist')"
            )
            result = subprocess.run(
                [sys.executable, "-c", command],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            source_archive = next(distribution_directory.glob("*.tar.gz"))
            wheel_archive = next(distribution_directory.glob("*.whl"))
            with tarfile.open(source_archive) as archive:
                source_names = {
                    "/".join(Path(name).parts[1:])
                    for name in archive.getnames()
                    if len(Path(name).parts) > 1
                }
            with zipfile.ZipFile(wheel_archive) as archive:
                wheel_names = set(archive.namelist())

            required_source_paths = {
                "README.md",
                "CONFIG_REFERENCE.md",
                "assets/mirai-logo-dark.png",
                "assets/mirai-logo-light-transparent.png",
                "assets/mirai-logo.png",
                "agent/AGENTS.md",
                "agent/architecture.json",
                "configs/lingbot_video/train_bf16.toml",
                "configs/magi2_preview/inference_offload.toml",
                "configs/magi2_preview/train_offload.toml",
                "model_support/lingbot_video.md",
                "model_support/magi2_preview.md",
                "scripts/train.py",
            }
            self.assertEqual(required_source_paths - source_names, set())
            self.assertIn("mirai/config/defaults/moe.toml", wheel_names)
            self.assertIn("mirai/config/defaults/lingbot_video.toml", wheel_names)
            self.assertIn("mirai/config/defaults/magi2_preview.toml", wheel_names)
            self.assertIn(
                "mirai/config/presets/magi2_preview_offload.toml", wheel_names
            )
            self.assertTrue(
                any(
                    name.endswith("mirai/vendors/lingbot_video/LICENSE")
                    for name in wheel_names
                )
            )
            self.assertTrue(
                any(
                    name.endswith("mirai/vendors/magi2_preview/LICENSE")
                    for name in wheel_names
                )
            )
            self.assertFalse(
                any(
                    ".dev-private" in name
                    or "/contracts/" in name
                    or name.startswith("tests/")
                    for name in wheel_names
                )
            )

    def test_fresh_git_manifest_keeps_every_public_tracked_file(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin"],
            cwd=ROOT,
            input=tracked,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertIn(ignored.returncode, {0, 1}, ignored.stderr)
        self.assertEqual(
            ignored.stdout.splitlines(),
            [],
            "These public files are tracked only because they predate an ignore rule; "
            "a fresh repository would omit them.",
        )

    def test_script_surface_is_intentional(self) -> None:
        script_root = ROOT / "scripts"
        root_entrypoints = {
            path.name for path in script_root.glob("*.py")
        }
        self.assertEqual(
            root_entrypoints,
            {
                "cache.py",
                "download.py",
                "eval.py",
                "export_adapter.py",
                "infer.py",
                "train.py",
                "verify_model_snapshot.py",
            },
        )
        self.assertEqual(
            {path.name for path in (script_root / "agent").glob("*.py")},
            {
                "architecture.py",
                "check.py",
                "evaluate_agent_effectiveness.py",
                "feature.py",
                "probe_moe_runtime.py",
                "run_agent_evaluation.py",
            },
        )
        self.assertEqual(
            {path.name for path in (script_root / "tools").glob("*.py")},
            {
                "benchmark.py",
                "calibrate_adaptive_lora_ranks.py",
                "calibrate_expert_consolidation.py",
                "calibrate_expert_pruning.py",
                "calibrate_expert_precision.py",
                "calibrate_expert_whitening.py",
                "calibrate_flexmoe.py",
                "calibrate_moe_quantization.py",
                "calibrate_router_quantization.py",
                "calibrate_routing_agreement.py",
                "check_comfyui_lora_load.py",
                "check_lora_host_compat.py",
                "compress_adapter_para.py",
                "compress_experts_flexmoe.py",
                "compress_experts_stun_sparse24.py",
                "consolidate_experts.py",
                "convert_lora_interchange.py",
                "export_compressed_weights_packed_state.py",
                "factorize_experts_mixture_basis.py",
                "factorize_experts_shared_basis.py",
                "estimate_moe_capacity.py",
                "evaluate_domain_routing.py",
                "lr_find.py",
                "plan_expert_consolidation.py",
                "prune_experts.py",
                "reconstruct_posthoc_ema.py",
                "repair_router_kd.py",
                "register_dataset.py",
                "router_health_report.py",
                "router_quantization_agreement.py",
                "upcycle_experts.py",
            },
        )

    def test_agent_architecture_map_has_resolvable_unique_owners(self) -> None:
        payload = json.loads((ROOT / "agent" / "architecture.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        names: set[str] = set()
        authorities: set[str] = set()
        for domain in payload["domains"]:
            self.assertNotIn(domain["name"], names)
            self.assertNotIn(domain["authority"], authorities)
            names.add(domain["name"])
            authorities.add(domain["authority"])
            self.assertTrue((ROOT / domain["root"]).is_dir(), domain["root"])
            self.assertTrue((ROOT / domain["authority"]).is_file(), domain["authority"])
            self.assertTrue(domain["search_terms"])

    def test_benchmark_help_is_executable(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/tools/benchmark.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("loss-tolerance", result.stdout)

    def test_user_documentation_boundary_is_explicit(self) -> None:
        self.assertFalse((ROOT / "docs").exists())
        # AGENTS.md and CLAUDE.md are agent bootstrap adapters rather than user
        # documentation; both resolve to the same contract in agent/AGENTS.md.
        bootstrap = {"AGENTS.md", "CLAUDE.md"}
        root_markdown = {
            path.name for path in ROOT.glob("*.md") if path.name not in bootstrap
        }
        self.assertEqual(root_markdown, {"README.md", "CONFIG_REFERENCE.md"})
        architecture = json.loads(
            (ROOT / "agent" / "architecture.json").read_text(encoding="utf-8")
        )
        model_pages = {
            path.stem for path in (ROOT / "model_support").glob("*.md")
        }
        self.assertEqual(model_pages, set(architecture["supported_families"]))

        agent_files = {
            path.relative_to(ROOT / "agent").as_posix()
            for path in (ROOT / "agent").rglob("*")
            if path.is_file()
        }
        self.assertEqual(
            agent_files,
            {
                "AGENTS.md",
                "architecture.json",
                "checks.json",
                "effectiveness.json",
                "features.json",
            },
        )

    def test_public_text_has_no_private_or_development_coordinates(self) -> None:
        forbidden = {
            "local service endpoint": re.compile(
                r"(?:localhost|127[.]0[.]0[.]1):[1-9]\d{3,4}", re.IGNORECASE
            ),
            "absolute Windows path": re.compile(r"(?<![A-Za-z])[A-Z]:\\"),
            "private home path": re.compile(r"/" r"home/[^/<\s]+/"),
            "private data mount": re.compile(r"/" r"mnt/data\d*/"),
            "agent scratch reference": re.compile(r"\.tmp/agent[_-]", re.IGNORECASE),
            "implementation phase": re.compile(r"\b(?:Wave|Phase)-\d+\b", re.IGNORECASE),
        }
        violations: list[str] = []
        for path in _public_text_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in forbidden.items():
                if pattern.search(text):
                    violations.append(f"{path.relative_to(ROOT)}: {label}")
        self.assertEqual(violations, [])

    def test_readme_is_model_agnostic_outside_model_support(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn('srcset="assets/mirai-logo-dark-transparent.png"', readme)
        self.assertIn('srcset="assets/mirai-logo.png"', readme)
        self.assertIn('src="assets/mirai-logo.png"', readme)
        self.assertIn("**Preview release.**", readme)
        self.assertIn("## Model support", readme)
        self.assertIn("### Generic", readme)
        self.assertIn("### MoE", readme)
        self.assertNotIn("<repository-url>", readme)
        self.assertIn("https://github.com/seruva19/mirai.git", readme)
        model_support = readme.split("## Model support", 1)[1].split("\n## ", 1)[0]
        outside_model_support = readme.replace(model_support, "")
        self.assertNotIn("lingbot", outside_model_support.lower())
        self.assertNotIn("roadmap", readme.lower())
        moe_section = readme.split("### MoE", 1)[1].split("\n## ", 1)[0]
        feature_blocks = re.split(r"(?m)^- ", moe_section)[1:]
        self.assertGreater(len(feature_blocks), 20)
        self._assert_external_feature_sources(readme, ROOT / "README.md")

        missing: list[str] = []
        for destination in re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme):
            destination = destination.strip()
            if destination.startswith(("https://", "http://", "#")):
                continue
            if not (ROOT / destination).exists():
                missing.append(destination)
        self.assertEqual(missing, [])

    def test_model_pages_have_training_inference_features_and_valid_links(self) -> None:
        for page in sorted((ROOT / "model_support").glob("*.md")):
            text = page.read_text(encoding="utf-8")
            self.assertIn("## Training", text, page)
            self.assertIn("## Inference", text, page)
            self.assertIn("## Model-specific features", text, page)
            self._assert_external_feature_sources(text, page)
            missing: list[str] = []
            for destination in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                destination = destination.strip()
                if destination.startswith(("https://", "http://", "#")):
                    continue
                if not (page.parent / destination).resolve().exists():
                    missing.append(destination)
            self.assertEqual(missing, [], page)

    def _assert_external_feature_sources(self, text: str, path: Path) -> None:
        repository_hosts = ("github.com/", "gitlab.com/", "codeberg.org/")
        for kind, destination in re.findall(
            r"\[\((paper|repo)\)\]\(([^)]+)\)",
            text,
        ):
            self.assertTrue(
                destination.startswith("https://"),
                f"{path}: ({kind}) must link to an external HTTPS source",
            )
            if kind == "repo":
                self.assertTrue(
                    any(host in destination.lower() for host in repository_hosts),
                    f"{path}: (repo) must link to an upstream source repository",
                )

    def test_all_public_examples_load_and_supported_families_are_exposed(self) -> None:
        config_root = ROOT / "configs"
        public_configs = sorted(config_root.rglob("*.toml"))
        expected = sorted(
            [
                *(config_root / "lingbot_video").glob("*.toml"),
                *(config_root / "magi2_preview").glob("*.toml"),
            ]
        )
        self.assertEqual(public_configs, expected)
        self.assertGreaterEqual(len(public_configs), 2)
        for path in public_configs:
            config = load_config(path)
            self.assertIn(config.model.type, {"lingbot-video", "magi2-preview"})
            if (
                config.model.type == "lingbot-video"
                and config.memory.frozen_weight_quantization == "none"
            ):
                self.assertEqual(
                    config.optimizer.type,
                    "adamw",
                    f"{path.relative_to(ROOT)} must remain runnable without an "
                    "optional low-bit optimizer dependency",
                )
                self.assertFalse(
                    config.memory.quantize_experts_on_load,
                    f"{path.relative_to(ROOT)} enables expert quantization without a quantized format",
                )
                self.assertIn(
                    config.memory.expert_weight_access,
                    {"auto", "disabled"},
                    f"{path.relative_to(ROOT)} selects dequantized expert access for full-precision weights",
                )
                self.assertEqual(config.memory.expert_dequant_chunk_size, 0)
                # Training-entrypoint block residency requires a compressed
                # base, so an uncompressed example never carries it. Streaming
                # uncompressed weights is an inference-only capability, and the
                # inference opt-in is the only thing that may turn a residency
                # strategy on here.
                self.assertEqual(config.training.block_swap_mode, "sync")
                self.assertEqual(config.training.blocks_to_swap, 0)
                if config.inference.blocks_to_swap > 0:
                    self.assertIn(
                        config.memory.weight_residency_strategy,
                        {"block_swap", "stream_disk"},
                        f"{path.relative_to(ROOT)} sets inference.blocks_to_swap "
                        "without a transport",
                    )
                else:
                    self.assertEqual(
                        config.memory.weight_residency_strategy, "disabled"
                    )

if __name__ == "__main__":
    unittest.main()
