from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scripts.agent.check import (
    _feature_owned_paths,
    _git_changed_paths,
    _run_command,
    environment_fingerprint,
    execute_checks,
    load_manifest,
    select_checks,
    select_local_checks,
    self_check,
    unowned_paths,
)
from scripts.agent.architecture import scan_dependency_rules
from scripts.agent.evaluate_agent_effectiveness import REQUIRED_FIELDS
from scripts.agent.evaluate_agent_effectiveness import _self_check as evaluator_self_check
from scripts.agent.run_agent_evaluation import _ignore_copy, run_evaluation
from scripts.agent.feature import create_feature
from mirai.core.features import load_feature_catalog
from mirai.core.features import _defined_python_symbols
from mirai.core.features import load_routing_topology_evidence
from mirai.core.features import EXTENSION_KITS
from mirai.core.features import FeatureDescriptor
from mirai.core.features import FeatureInvariant
from mirai.core.features import FeatureKind
from mirai.core.features import prove_default_path
from mirai.core.features import prove_gradients
from mirai.core.features import prove_native_execution
from mirai.core.features import prove_reference_parity
from mirai.core.features import prove_resource_bound
from mirai.core.features import prove_state_roundtrip
from mirai.core.features import validate_feature_catalog
from mirai.core.features import validate_routing_topology_promotion


ROOT = Path(__file__).resolve().parents[1]


class AgentValidationContractTests(unittest.TestCase):
    def test_generic_schema_does_not_branch_on_a_concrete_model_family(self) -> None:
        tree = ast.parse(
            (ROOT / "mirai" / "config" / "schema.py").read_text(encoding="utf-8")
        )
        concrete_family = "lingbot-video"
        family_comparisons = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and any(
                isinstance(value, ast.Constant)
                and value.value == concrete_family
                for value in (node.left, *node.comparators)
            )
        ]
        self.assertEqual(
            family_comparisons,
            [],
            "generic config schema must not branch on a concrete model family",
        )

    def test_training_policy_provider_seam_is_typed(self) -> None:
        base_source = (ROOT / "mirai" / "core" / "models" / "base.py").read_text(
            encoding="utf-8"
        )
        family_root = ROOT / "mirai" / "core" / "models" / "lingbot_video"
        family_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(family_root.glob("*.py"))
        )
        self.assertNotIn("def configure_training_policy(", base_source)
        self.assertNotIn("def set_training_policy_context(", base_source)
        self.assertNotIn("policy_name: str", family_sources)
        self.assertNotIn("policy: Any", family_sources)

    def test_dependency_boundaries_are_executable_and_fail_closed(self) -> None:
        self.assertEqual(scan_dependency_rules(), ())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mirai" / "core" / "runtime"
            source.mkdir(parents=True)
            (source / "leak.py").write_text(
                "from mirai.core.models.family_x.pipeline import Pipeline\n",
                encoding="utf-8",
            )
            architecture = root / "agent" / "architecture.json"
            architecture.parent.mkdir()
            architecture.write_text(
                json.dumps(
                    {
                        "dependency_rules": [
                            {
                                "name": "no_family_leak",
                                "source_roots": ["mirai/core/runtime"],
                                "target_prefixes": [
                                    "mirai.core.models.family_x",
                                    "mirai.core.models.family_y",
                                    "mirai.core.models.family_z",
                                ],
                                "exclude_globs": [],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            violations = scan_dependency_rules(
                root=root,
                architecture_path=architecture,
            )
            self.assertEqual(len(violations), 1)
            self.assertEqual(violations[0].path, "mirai/core/runtime/leak.py")

            (source / "dynamic_leak.py").write_text(
                "import importlib\n"
                "importlib.import_module('mirai.core.models.family_x.pipeline')\n",
                encoding="utf-8",
            )
            (source / "aliased_dynamic_leak.py").write_text(
                "import importlib as il\n"
                "from importlib import import_module as load\n"
                "il.import_module('mirai.core.models.family_y.pipeline')\n"
                "load('mirai.core.models.family_z.pipeline')\n",
                encoding="utf-8",
            )
            violations = scan_dependency_rules(
                root=root,
                architecture_path=architecture,
            )
            self.assertEqual(len(violations), 4)
            dynamic = next(
                item
                for item in violations
                if Path(item.path).name == "dynamic_leak.py"
            )
            self.assertEqual(
                dynamic.imported_module,
                "mirai.core.models.family_x.pipeline",
            )
            aliased = [
                item
                for item in violations
                if Path(item.path).name == "aliased_dynamic_leak.py"
            ]
            self.assertEqual(
                {item.imported_module for item in aliased},
                {
                    "mirai.core.models.family_y.pipeline",
                    "mirai.core.models.family_z.pipeline",
                },
            )

    def test_feature_catalog_is_complete_and_mechanically_valid(self) -> None:
        self.assertEqual(set(EXTENSION_KITS), set(FeatureKind))
        catalog = load_feature_catalog(ROOT / "agent" / "features.json")
        self.assertGreaterEqual(len(catalog), 4)
        issues = validate_feature_catalog(
            root=ROOT,
            catalog=catalog,
            checks_path=ROOT / "agent" / "checks.json",
        )
        self.assertEqual(issues, ())
        compressed = next(
            feature
            for feature in catalog
            if feature.name == "compressed_frozen_weights"
        )
        changed_default = replace(
            compressed,
            disabled_config=(("memory.frozen_weight_quantization", "fp8"),),
        )
        issues = validate_feature_catalog(
            root=ROOT,
            catalog=(changed_default,),
            checks_path=ROOT / "agent" / "checks.json",
        )
        self.assertIn("default_gate_changed", {issue.code for issue in issues})
        self.assertIn(
            "mirai/core/moe/routing/subset.py",
            _feature_owned_paths()["moe_routing"],
        )

    def test_feature_symbols_must_be_real_python_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            owner = Path(tmp) / "owner.py"
            owner.write_text(
                "# PhantomSymbol is not implemented.\n"
                "def container():\n"
                "    def NestedOnly():\n"
                "        pass\n"
                "class RealSymbol:\n"
                "    pass\n",
                encoding="utf-8",
            )
            symbols = _defined_python_symbols(owner)
            self.assertIn("RealSymbol", symbols)
            self.assertNotIn("PhantomSymbol", symbols)
            self.assertNotIn("NestedOnly", symbols)

    def test_feature_cli_creation_writes_the_requested_catalog_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "features.json"
            catalog_path.write_text(
                (ROOT / "agent" / "features.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            result = create_feature(
                Namespace(
                    name="contract_probe",
                    kind="training_policy",
                    owner="mirai/core/training/policies/expert_dropout.py",
                    symbol=["ExpertDropoutTrainingPolicy"],
                    check="moe_adaptation",
                    config_key=["training.policy_options"],
                    disabled_config=["training.policy_options={}"],
                    contract=[
                        "mirai/core/training/policies/contracts/expert_dropout.py"
                    ],
                    stateful=True,
                    optimized=False,
                    reference_path="",
                    readme_claim="No-token-drop expert-output dropout",
                    provenance="",
                    default_on=False,
                    create_owner=False,
                ),
                root=ROOT,
                catalog_path=catalog_path,
            )
            self.assertEqual(result["status"], "created", result["issues"])
            created = load_feature_catalog(catalog_path)
            self.assertIn("contract_probe", {feature.name for feature in created})

    def test_routing_topology_promotion_requires_paired_held_out_evidence(
        self,
    ) -> None:
        topology = FeatureDescriptor(
            name="topology_probe",
            kind=FeatureKind.ROUTING_TOPOLOGY,
            owner="mirai/core/moe/routing/topology_probe.py",
            check="moe_routing",
            invariants=(
                FeatureInvariant.OWNER,
                FeatureInvariant.CONFIG,
                FeatureInvariant.DEFAULT_PATH,
                FeatureInvariant.GRADIENTS,
                FeatureInvariant.HELD_OUT_EVIDENCE,
            ),
            default_off=True,
        )
        self.assertEqual(
            validate_routing_topology_promotion(root=ROOT, feature=topology),
            (),
        )

        promoted = replace(topology, default_off=False)
        issues = validate_routing_topology_promotion(
            root=ROOT,
            feature=promoted,
        )
        self.assertEqual(issues[0].code, "missing_promotion_evidence")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_path = root / "agent" / "evidence" / "topology_probe.json"
            evidence_path.parent.mkdir(parents=True)
            evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "feature": "topology_probe",
                        "metric": "held_out_loss",
                        "direction": "lower_is_better",
                        "held_out_split_fingerprint": "split-sha256",
                        "baseline_artifact_fingerprint": "baseline-sha256",
                        "candidate_artifact_fingerprint": "candidate-sha256",
                        "pairs": [
                            {
                                "pair_id": "seed-1",
                                "baseline": 1.0,
                                "candidate": 0.9,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            promoted = replace(
                promoted,
                promotion_evidence="agent/evidence/topology_probe.json",
            )
            self.assertEqual(
                validate_routing_topology_promotion(
                    root=root,
                    feature=promoted,
                ),
                (),
            )
            evidence = load_routing_topology_evidence(evidence_path)
            self.assertAlmostEqual(evidence.mean_quality_delta, 0.1)

            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            payload["candidate_artifact_fingerprint"] = "baseline-sha256"
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            issues = validate_routing_topology_promotion(root=root, feature=promoted)
            self.assertEqual(
                issues[0].code,
                "promotion_evidence_identical_artifacts",
            )

            payload["candidate_artifact_fingerprint"] = "candidate-sha256"
            payload["pairs"][0]["candidate"] = 1.1
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            issues = validate_routing_topology_promotion(root=root, feature=promoted)
            self.assertEqual(issues[0].code, "quality_non_regression_failed")

    def test_reusable_feature_proofs_reject_contract_drift(self) -> None:
        prove_default_path(lambda: (1, 2), lambda: (1, 2), compare=lambda a, b: a == b)
        prove_reference_parity(lambda: 3.0, lambda: 3.0, compare=lambda a, b: a == b)
        prove_gradients(
            lambda: {"input": 1.0, "adapter": 2.0},
            lambda: {"input": 1.0, "adapter": 2.0},
            compare=lambda a, b: a == b,
        )
        prove_resource_bound(7, 8)
        prove_native_execution(lambda: {"finite": True}, validate=lambda value: value["finite"])
        prove_state_roundtrip(
            lambda: {"step": 7},
            lambda value: dict(value),
            lambda state: dict(state),
            lambda value: value["step"],
        )
        with self.assertRaisesRegex(AssertionError, "reference path"):
            prove_default_path(lambda: 1, lambda: 2, compare=lambda a, b: a == b)

    def test_feature_cli_inspects_and_validates_catalog(self) -> None:
        for command in (
            ["inspect"],
            ["validate"],
            ["validate", "dispersive_representation_regularization"],
        ):
            result = subprocess.run(
                [sys.executable, "scripts/agent/feature.py", *command],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_manifest_routes_reference_cases_with_bounded_noise(self) -> None:
        result = self_check()
        self.assertEqual(result["status"], "passed", result["failures"])
        self.assertEqual(result["check_recall"], 1.0)
        self.assertGreaterEqual(result["mean_irrelevant_check_reduction"], 0.8)

    def test_unknown_change_fails_closed(self) -> None:
        manifest = load_manifest()
        unknown = "mirai/core/unowned_seam.py"
        self.assertEqual(unowned_paths(manifest, [unknown]), [unknown])

    def test_changed_from_includes_untracked_files(self) -> None:
        responses = [
            subprocess.CompletedProcess(
                [],
                0,
                stdout="mirai/config/schema.py\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                [],
                0,
                stdout="tests/test_new_contract.py\n",
                stderr="",
            ),
        ]
        with patch("scripts.agent.check.subprocess.run", side_effect=responses):
            paths = _git_changed_paths("HEAD")
        self.assertEqual(
            paths,
            ["mirai/config/schema.py", "tests/test_new_contract.py"],
        )

    def test_execution_evidence_identifies_the_environment(self) -> None:
        fingerprint = environment_fingerprint()
        self.assertRegex(fingerprint["python"], r"^\d+\.\d+\.\d+")
        self.assertTrue(fingerprint["platform"])
        self.assertIn("torch", fingerprint["packages"])

    def test_deferred_contracts_cannot_report_passed(self) -> None:
        result = execute_checks(
            [
                {
                    "name": "gpu_probe",
                    "cost": "gpu",
                    "remote_gpu": True,
                    "invariants": ["native execution"],
                    "commands": ["python -c pass"],
                }
            ],
            allow_remote_gpu=False,
            max_cost="fast",
        )
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["checks"][0]["status"], "gpu_required")

    def test_gpu_contracts_cannot_pass_without_cuda(self) -> None:
        with patch("scripts.agent.check.cuda_available", return_value=False):
            result = execute_checks(
                [
                    {
                        "name": "gpu_probe",
                        "cost": "gpu",
                        "remote_gpu": True,
                        "invariants": ["native execution"],
                        "commands": ["python -c pass"],
                    }
                ],
                allow_remote_gpu=True,
                max_cost="gpu",
            )
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["checks"][0]["status"], "gpu_unavailable")

    def test_gpu_contracts_cannot_pass_when_pytest_skips(self) -> None:
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout="1 passed, 2 skipped in 0.01s\n",
            stderr="",
        )
        with patch("scripts.agent.check.subprocess.run", return_value=completed):
            command = _run_command(
                "python -m pytest tests/test_probe.py -q",
                report_pytest_skips=True,
            )
        self.assertEqual(command["skipped_tests"], 2)
        self.assertIn("-rs", command["argv"])

        with (
            patch("scripts.agent.check.cuda_available", return_value=True),
            patch("scripts.agent.check._run_command", return_value=command),
        ):
            result = execute_checks(
                [
                    {
                        "name": "gpu_probe",
                        "cost": "gpu",
                        "remote_gpu": True,
                        "invariants": ["native execution"],
                        "commands": ["python -m pytest tests/test_probe.py -q"],
                    }
                ],
                allow_remote_gpu=True,
                max_cost="gpu",
            )
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["checks"][0]["status"], "skipped_gpu_tests")

    def test_ci_workflow_uses_executable_contract_commands(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("branches: [main]", workflow)
        self.assertIn("mirai/core/models/adapters/contracts/host.py", workflow)
        self.assertIn("-q -m production_contract", workflow)
        self.assertNotIn("pytest tests/ -q -m production_contract", workflow)
        self.assertIn("--all-local --run --max-cost extended", workflow)
        self.assertNotIn("--max-cost wave", workflow)
        self.assertEqual(workflow.count("run: python -m pip install -e ."), 3)
        self.assertNotIn('pip install -e ".[', workflow)

    def test_all_local_selection_covers_colocated_contracts_without_gpu(self) -> None:
        checks = select_local_checks(load_manifest())
        self.assertTrue(checks)
        self.assertTrue(all(not check["remote_gpu"] for check in checks))
        commands = "\n".join(
            command for check in checks for command in check["commands"]
        )
        self.assertIn(
            "mirai/core/models/adapters/contracts/expert_tensor.py",
            commands,
        )
        self.assertIn("mirai/core/moe/contracts/training_losses.py", commands)

    def test_expert_execution_spec_changes_route_to_compressed_contracts(self) -> None:
        selected = {
            check["name"]
            for check in select_checks(
                load_manifest(),
                ["mirai/core/moe/runtime/specs.py"],
            )
        }
        self.assertIn("compressed_weights", selected)
        self.assertIn("moe_routing", selected)

    def test_manifest_commands_are_executable_and_own_every_probe(self) -> None:
        payload = load_manifest()
        self.assertLessEqual(len(payload["checks"]), payload["contract_budget"])
        static = next(check for check in payload["checks"] if check["name"] == "static")
        self.assertFalse(static["owns_paths"])
        names: set[str] = set()
        owned: set[str] = set()
        non_pytest_commands: set[str] = set()
        for check in payload["checks"]:
            self.assertNotIn(check["name"], names)
            names.add(check["name"])
            self.assertIn(check["cost"], {"fast", "extended", "gpu"})
            for command in check["commands"]:
                # A standalone probe is invoked as a module rather than as a
                # file path: the runner executes argv without a shell, so only
                # ``-m`` puts the repository root on sys.path and lets the probe
                # import ``mirai`` on a checkout with no installed package. The
                # dotted name still names exactly one probe file.
                module = re.fullmatch(
                    r"python -m ((?:mirai|tests)(?:\.[A-Za-z0-9_]+)+)", command
                )
                if module is not None:
                    contract_paths = [module.group(1).replace(".", "/") + ".py"]
                else:
                    contract_paths = re.findall(
                        r"(?:tests|mirai)/[A-Za-z0-9_./-]+\.py", command
                    )
                if not contract_paths:
                    non_pytest_commands.add(command)
                for contract_path in contract_paths:
                    self.assertTrue(
                        (ROOT / contract_path).is_file(), contract_path
                    )
                    owned.add(contract_path)
        self.assertEqual(
            non_pytest_commands,
            {"ruff check .", "python scripts/agent/architecture.py"},
        )
        public_probes = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "tests").glob("test_*.py")
        }
        public_probes.update(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "mirai").glob("**/contracts/*.py")
            if path.name != "__init__.py"
        )
        self.assertEqual(public_probes, owned)

    def test_effectiveness_protocol_is_paired_and_regression_intolerant(self) -> None:
        protocol = json.loads(
            (ROOT / "agent" / "effectiveness.json").read_text(encoding="utf-8")
        )
        self.assertEqual(protocol["schema_version"], 1)
        self.assertEqual(protocol["unit_under_test"], "deployed_agent_system")
        self.assertGreaterEqual(protocol["minimum_repetitions_per_task"], 3)
        self.assertTrue(protocol["fresh_context_per_run"])
        self.assertTrue(protocol["randomize_condition_order"])
        self.assertEqual(set(protocol["required_record_fields"]), REQUIRED_FIELDS)
        acceptance = protocol["acceptance"]
        for field in (
            "agent_system_id",
            "environment_id",
            "condition_order_index",
            "trajectory_ref",
        ):
            self.assertIn(field, protocol["required_record_fields"])
        self.assertTrue(acceptance["hidden_contract_pass_rate_must_not_decrease"])
        self.assertEqual(acceptance["candidate_architecture_failures"], 0)
        self.assertTrue(acceptance["owner_rank_must_not_increase"])
        self.assertTrue(acceptance["invalid_edits_must_not_increase"])
        self.assertGreaterEqual(
            acceptance["minimum_median_time_or_token_reduction"],
            0.2,
        )
        self.assertEqual(evaluator_self_check()["status"], "passed")
        runner = protocol["runner_contract"]
        self.assertTrue(runner["hidden_graders_are_external"])
        self.assertFalse(runner["shell_execution"])
        self.assertIn("trace_path", runner["adapter_placeholders"])

    def test_paired_runner_isolates_trials_and_records_real_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_roots: dict[str, Path] = {}
            for condition in ("baseline", "candidate"):
                source_root = root / f"{condition}-source"
                task_source = source_root / "task"
                task_source.mkdir(parents=True)
                (task_source / "marker.txt").write_text(condition, encoding="utf-8")
                source_roots[condition] = source_root
            trace_code = (
                "import json,pathlib,sys;"
                "pathlib.Path(sys.argv[1]).write_text(json.dumps(dict("
                "input_tokens=10,output_tokens=2,tool_calls=1,"
                "files_opened=1,checks_run=1)))"
            )
            grade_code = (
                "import json,pathlib,sys;"
                "pathlib.Path(sys.argv[1]).write_text(json.dumps(dict("
                "hidden_contract_pass=True,architecture_pass=True,owner_rank=1,"
                "invalid_edits=0,regressions=0,irrelevant_files_opened=0,"
                "irrelevant_checks_run=0)))"
            )
            adapters = {
                condition: {
                    "agent_system_id": f"{condition}-system",
                    "environment_id": "contract-environment",
                    "argv": [sys.executable, "-c", trace_code, "{trace_path}"],
                    "timeout_seconds": 10,
                }
                for condition in ("baseline", "candidate")
            }
            records = run_evaluation(
                tasks=[
                    {
                        "task_id": "owner",
                        "task_class": "owner_localization",
                        "prompt": "Locate the owner.",
                        "source_subdir": "task",
                        "grader_id": "hidden",
                    }
                ],
                source_roots=source_roots,
                adapters=adapters,
                graders={
                    "hidden": [
                        sys.executable,
                        "-c",
                        grade_code,
                        "{grade_path}",
                    ]
                },
                repetitions=3,
                workspace_parent=root,
                keep_workspaces=False,
                seed=7,
            )
            self.assertEqual(len(records["baseline"]), 3)
            self.assertEqual(len(records["candidate"]), 3)
            self.assertTrue(
                all(
                    REQUIRED_FIELDS <= record.keys()
                    for condition in records.values()
                    for record in condition
                )
            )
            self.assertTrue(
                all(
                    record["success"]
                    for condition in records.values()
                    for record in condition
                )
            )
            self.assertFalse(list(root.glob("mirai-agent-eval-*")))

    def test_evaluation_workspace_excludes_private_and_heavy_files(self) -> None:
        ignored = _ignore_copy(
            "source",
            [
                ".env",
                ".env.local",
                ".agents",
                ".dev-private",
                "weights.safetensors",
                "checkpoint.pt",
                "module.py",
            ],
        )
        self.assertEqual(
            ignored,
            {
                ".env",
                ".env.local",
                ".agents",
                ".dev-private",
                "weights.safetensors",
                "checkpoint.pt",
            },
        )


if __name__ == "__main__":
    unittest.main()
