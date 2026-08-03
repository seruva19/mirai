from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

from mirai.config.loader import load_config
from mirai.config.schema import TrainingConfig
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.calibration.esft import (
    ESFTAffinityAccumulator,
    ESFTCalibrationCapture,
    ESFTCalibrationTarget,
    build_esft_selection_plan,
    select_esft_experts,
)
from mirai.core.training.calibration import esft as esft_runtime
from mirai.core.training.optim.optimizer import (
    build_optimizer,
    build_param_groups,
    warmup_optimizer_warning,
)
from mirai.core.training.optim.lora_muon import (
    LoRAMuon,
    estimate_lora_muon_state_bytes,
    lora_muon_factor_directions,
    matrix_sign_newton_schulz,
    matrix_sign_reference,
    psd_inverse_sqrt_newton_schulz,
    psd_inverse_sqrt_reference,
    rebalance_lora_muon_gauge,
)
from mirai.core.training.optim.lora_pairs import LoRAFactorPair
from mirai.core.training.optim.lora_pro import (
    LoRAProAdamW,
    estimate_lora_pro_state_bytes,
    lora_pro_correct_gradients,
    lora_pro_equivalent_gradient,
    solve_positive_sylvester,
)
from mirai.core.training.optim.low_bit_state import (
    SOLO_4_2_BETAS,
    SOLO_4_2_STATE_FORMAT,
    SIGNED_DE_4BIT_LEVELS,
    decode_signed_de_4bit,
    decode_unsigned_qema_2bit,
    encode_signed_de_4bit,
    encode_unsigned_qema_2bit,
    packed_solo_4_2_state_nbytes,
)
from mirai.core.training.optim.selected_expert_adamw import SelectedExpertAdamW
from mirai.core.training.optim.selected_expert_adam_mini import (
    SelectedExpertAdamMini,
    estimate_selected_expert_adam_mini_state_bytes,
)
from mirai.core.training.optim.selected_expert_muon import (
    SelectedExpertAdaMuon,
    SelectedExpertMuon,
    adamuon_matrix_direction,
    estimate_selected_expert_muon_state_bytes,
    muon_matrix_direction,
    orthogonalize_matrix_newton_schulz,
    orthogonalize_matrix_reference,
)
from mirai.core.training.runtime.contract import validate_training_runtime_config

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class OptimizerConfigurationTests(unittest.TestCase):
    def test_prodigy_controls_parse_and_validate(self) -> None:
        config = TrainingConfig.from_dict(
            {
                "optimizer": {
                    "type": "prodigy",
                    "prodigy_beta3": 0.95,
                    "prodigy_decouple": False,
                    "prodigy_use_bias_correction": True,
                    "prodigy_safeguard_warmup": True,
                    "prodigy_d0": 2e-6,
                    "prodigy_d_coef": 1.5,
                    "prodigy_growth_rate": 1.2,
                    "prodigy_slice_p": 3,
                }
            }
        )
        self.assertEqual(config.optimizer.prodigy_beta3, 0.95)
        self.assertFalse(config.optimizer.prodigy_decouple)
        self.assertTrue(config.optimizer.prodigy_use_bias_correction)
        self.assertTrue(config.optimizer.prodigy_safeguard_warmup)
        self.assertEqual(config.optimizer.prodigy_d0, 2e-6)
        self.assertEqual(config.optimizer.prodigy_d_coef, 1.5)
        self.assertEqual(config.optimizer.prodigy_growth_rate, 1.2)
        self.assertEqual(config.optimizer.prodigy_slice_p, 3)

        for key, value in {
            "prodigy_beta3": 1.0,
            "prodigy_d0": 0.0,
            "prodigy_d_coef": float("inf"),
            "prodigy_growth_rate": 0.9,
            "prodigy_slice_p": 0,
        }.items():
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    TrainingConfig.from_dict({"optimizer": {key: value}})

    def test_prodigy_warns_about_external_warmup(self) -> None:
        warning = warmup_optimizer_warning(optimizer_type="prodigy", warmup_steps=10)
        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertIn("warmup_steps=0", warning)

    def test_shipped_prodigy_preset_disables_external_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.toml"
            cfg_path.write_text(
                'preset = "sparse_moe_test"\n\n[model]\ntype = "sparse_moe_test"\n',
                encoding="utf-8",
            )
            cfg = load_config(cfg_path)
        self.assertEqual(cfg.optimizer.type, "prodigy")
        self.assertEqual(cfg.training.warmup_steps, 0)

@unittest.skipIf(torch is None, "torch not installed")
class OptimizerRuntimeTests(unittest.TestCase):
    def test_prodigy_builder_forwards_all_typed_controls(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(4))
        result = build_optimizer(
            params=[parameter],
            optimizer_type="prodigy",
            lr=0.7,
            weight_decay=0.03,
            allow_fallback=False,
            prodigy_beta3=0.91,
            prodigy_decouple=False,
            prodigy_use_bias_correction=True,
            prodigy_safeguard_warmup=True,
            prodigy_d0=3e-6,
            prodigy_d_coef=1.4,
            prodigy_growth_rate=1.1,
            prodigy_slice_p=2,
        )
        group = result.optimizer.param_groups[0]
        self.assertEqual(result.resolved_type, "prodigy")
        self.assertEqual(group["beta3"], 0.91)
        self.assertFalse(group["decouple"])
        self.assertTrue(group["use_bias_correction"])
        self.assertTrue(group["safeguard_warmup"])
        self.assertEqual(group["d"], 3e-6)
        self.assertEqual(group["d_coef"], 1.4)
        self.assertEqual(group["growth_rate"], 1.1)
        self.assertEqual(group["slice_p"], 2)

    @unittest.skipIf(torch is None, "torch is required")
    def test_selected_expert_optimizer_updates_and_persists_only_selected_rows(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(4, 3))
        parameter.grad = torch.ones_like(parameter)
        optimizer = SelectedExpertAdamW(
            [parameter],
            expert_ids=(1, 3),
            lr=0.1,
        )
        optimizer.step()
        torch.testing.assert_close(parameter[0], torch.zeros(3))
        torch.testing.assert_close(parameter[2], torch.zeros(3))
        self.assertTrue(bool((parameter[1] != 0).all()))
        self.assertTrue(bool((parameter[3] != 0).all()))
        state = optimizer.state[parameter]
        self.assertEqual(tuple(state["exp_avg"].shape), (2, 3))
        payload = optimizer.state_dict()
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = SelectedExpertAdamW(
            [restored_parameter],
            expert_ids=(1, 3),
            lr=0.1,
        )
        restored.load_state_dict(payload)
        self.assertEqual(
            tuple(restored.state[restored_parameter]["expert_ids"].tolist()),
            (1, 3),
        )

    def test_selected_expert_optimizer_owns_per_parameter_rows(self) -> None:
        first = torch.nn.Parameter(torch.zeros(4, 2))
        second = torch.nn.Parameter(torch.zeros(4, 2))
        first.grad = torch.ones_like(first)
        second.grad = torch.ones_like(second)
        optimizer = SelectedExpertAdamW(
            [first, second],
            named_params=[("first", first), ("second", second)],
            expert_ids_by_name={"first": (0, 2), "second": (1,)},
            lr=0.1,
        )
        optimizer.step()

        self.assertTrue(bool((first[[0, 2]] != 0).all()))
        self.assertTrue(bool((first[[1, 3]] == 0).all()))
        self.assertTrue(bool((second[1] != 0).all()))
        self.assertTrue(bool((second[[0, 2, 3]] == 0).all()))
        self.assertEqual(tuple(optimizer.state[first]["exp_avg"].shape), (2, 2))
        self.assertEqual(tuple(optimizer.state[second]["exp_avg"].shape), (1, 2))

        payload = optimizer.state_dict()
        incompatible_first = torch.nn.Parameter(first.detach().clone())
        incompatible_second = torch.nn.Parameter(second.detach().clone())
        incompatible = SelectedExpertAdamW(
            [incompatible_first, incompatible_second],
            named_params=[
                ("first", incompatible_first),
                ("second", incompatible_second),
            ],
            expert_ids_by_name={"first": (0,), "second": (1,)},
            lr=0.1,
        )
        with self.assertRaisesRegex(ValueError, "selection mismatch"):
            incompatible.load_state_dict(payload)

    def test_solo_4_2_codecs_match_exact_levels_and_pack_tail(self) -> None:
        first = torch.tensor(SIGNED_DE_4BIT_LEVELS).repeat(8)
        first_payload = encode_signed_de_4bit(first)
        self.assertEqual(first_payload.codes.dtype, torch.uint8)
        self.assertEqual(first_payload.codes.numel(), 64)
        torch.testing.assert_close(decode_signed_de_4bit(first_payload), first)

        tail_payload = encode_signed_de_4bit(torch.linspace(-1.0, 1.0, 130))
        self.assertEqual(tuple(decode_signed_de_4bit(tail_payload).shape), (130,))
        self.assertTrue(bool(torch.isfinite(decode_signed_de_4bit(tail_payload)).all()))

        second = torch.tensor([1.0, 0.5, 0.25, 0.125]).repeat(32)
        second_payload = encode_unsigned_qema_2bit(second)
        self.assertEqual(second_payload.codes.dtype, torch.uint8)
        self.assertEqual(second_payload.codes.numel(), 32)
        torch.testing.assert_close(decode_unsigned_qema_2bit(second_payload), second)

    def test_solo_4_2_selected_expert_state_is_compact_and_exactly_resumable(
        self,
    ) -> None:
        parameter = torch.nn.Parameter(torch.zeros(4, 8, 8))
        gradient = torch.linspace(-1.0, 1.0, parameter.numel()).view_as(parameter)
        parameter.grad = gradient.clone()
        optimizer = SelectedExpertAdamW(
            [parameter],
            expert_ids=(1, 3),
            lr=0.01,
            betas=SOLO_4_2_BETAS,
            state_format=SOLO_4_2_STATE_FORMAT,
        )
        torch.manual_seed(123)
        optimizer.step()

        torch.testing.assert_close(parameter[[0, 2]], torch.zeros(2, 8, 8))
        state = optimizer.state[parameter]
        self.assertIsInstance(state["exp_avg"], dict)
        self.assertIsInstance(state["exp_avg_sq"], dict)
        packed_bytes = packed_solo_4_2_state_nbytes(
            state["exp_avg"],
            state["exp_avg_sq"],
        )
        native_bf16_bytes = 2 * 2 * 8 * 8 * 2
        self.assertLess(packed_bytes, native_bf16_bytes)

        payload = copy.deepcopy(optimizer.state_dict())
        saved_parameter = parameter.detach().clone()
        saved_rng = torch.random.get_rng_state()
        next_gradient = gradient.flip(0)
        parameter.grad = next_gradient.clone()
        optimizer.step()
        expected = parameter.detach().clone()

        restored_parameter = torch.nn.Parameter(saved_parameter)
        restored = SelectedExpertAdamW(
            [restored_parameter],
            expert_ids=(1, 3),
            lr=0.01,
            betas=SOLO_4_2_BETAS,
            state_format=SOLO_4_2_STATE_FORMAT,
        )
        restored.load_state_dict(payload)
        torch.random.set_rng_state(saved_rng)
        restored_parameter.grad = next_gradient.clone()
        restored.step()
        torch.testing.assert_close(restored_parameter, expected, rtol=0, atol=0)

        native = SelectedExpertAdamW(
            [torch.nn.Parameter(saved_parameter.clone())],
            expert_ids=(1, 3),
            lr=0.01,
        )
        with self.assertRaisesRegex(ValueError, "execution policy mismatch"):
            native.load_state_dict(payload)

        malformed = copy.deepcopy(payload)
        parameter_id = malformed["param_groups"][0]["params"][0]
        malformed["state"][parameter_id]["exp_avg"]["shape"] = (1, 16, 8)
        incompatible = SelectedExpertAdamW(
            [torch.nn.Parameter(saved_parameter.clone())],
            expert_ids=(1, 3),
            lr=0.01,
            betas=SOLO_4_2_BETAS,
            state_format=SOLO_4_2_STATE_FORMAT,
        )
        with self.assertRaisesRegex(ValueError, "topology mismatch"):
            incompatible.load_state_dict(malformed)

    def test_solo_4_2_optimizer_registry_selects_canonical_finetune_betas(
        self,
    ) -> None:
        parameter = torch.nn.Parameter(torch.zeros(4, 8, 8))
        result = build_optimizer(
            params=[parameter],
            optimizer_type="selected_expert_adamw_4_2bit",
            lr=1e-3,
            weight_decay=0.0,
            allow_fallback=False,
            selected_expert_ids=(0, 2),
        )
        self.assertEqual(result.resolved_type, "selected_expert_adamw_4_2bit")
        self.assertEqual(result.optimizer.param_groups[0]["betas"], SOLO_4_2_BETAS)

    def test_selected_expert_adam_mini_matches_neuron_partition_oracle(
        self,
    ) -> None:
        parameter = torch.nn.Parameter(torch.zeros(4, 3, 8))
        gradient = torch.arange(1, parameter.numel() + 1, dtype=torch.float32).view_as(
            parameter
        )
        parameter.grad = gradient.clone()
        optimizer = SelectedExpertAdamMini(
            [parameter],
            expert_ids=(1, 3),
            lr=0.1,
            betas=(0.0, 0.0),
            eps=0.0,
        )
        optimizer.step()

        selected_gradient = gradient[[1, 3]]
        expected = -0.1 * selected_gradient / selected_gradient.square().mean(
            dim=-1,
            keepdim=True,
        ).sqrt()
        torch.testing.assert_close(parameter[[1, 3]], expected)
        torch.testing.assert_close(parameter[[0, 2]], torch.zeros(2, 3, 8))
        state = optimizer.state[parameter]
        self.assertEqual(tuple(state["exp_avg"].shape), (2, 3, 8))
        self.assertEqual(tuple(state["exp_avg_sq_mean"].shape), (2, 3, 1))
        native_selected_adam_bytes = 2 * selected_gradient.numel() * 4
        self.assertLess(
            estimate_selected_expert_adam_mini_state_bytes(optimizer),
            native_selected_adam_bytes,
        )

    def test_selected_expert_adam_mini_resume_and_topology_are_exact(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(4, 4, 8))
        first_gradient = torch.linspace(-1.0, 1.0, parameter.numel()).view_as(
            parameter
        )
        parameter.grad = first_gradient.clone()
        optimizer = SelectedExpertAdamMini(
            [parameter],
            expert_ids=(0, 2),
            lr=0.01,
        )
        optimizer.step()
        payload = copy.deepcopy(optimizer.state_dict())
        saved_parameter = parameter.detach().clone()

        next_gradient = first_gradient.flip(-1)
        parameter.grad = next_gradient.clone()
        optimizer.step()
        expected = parameter.detach().clone()

        restored_parameter = torch.nn.Parameter(saved_parameter.clone())
        restored = SelectedExpertAdamMini(
            [restored_parameter],
            expert_ids=(0, 2),
            lr=0.01,
        )
        restored.load_state_dict(payload)
        restored_parameter.grad = next_gradient.clone()
        restored.step()
        torch.testing.assert_close(restored_parameter, expected, rtol=0, atol=0)

        malformed = copy.deepcopy(payload)
        parameter_id = malformed["param_groups"][0]["params"][0]
        malformed["state"][parameter_id]["exp_avg_sq_mean"] = torch.zeros(2, 2, 1)
        incompatible = SelectedExpertAdamMini(
            [torch.nn.Parameter(saved_parameter.clone())],
            expert_ids=(0, 2),
            lr=0.01,
        )
        with self.assertRaisesRegex(ValueError, "second-moment topology"):
            incompatible.load_state_dict(malformed)

    def test_selected_expert_adam_mini_registry_uses_exact_plan(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(4, 3, 8))
        result = build_optimizer(
            params=[parameter],
            optimizer_type="selected_expert_adam_mini",
            lr=1e-3,
            weight_decay=0.0,
            allow_fallback=False,
            selected_expert_ids=(1, 3),
        )
        self.assertEqual(result.resolved_type, "selected_expert_adam_mini")
        self.assertIsInstance(result.optimizer, SelectedExpertAdamMini)
        self.assertEqual(result.optimizer.expert_ids, (1, 3))

    def test_selected_expert_muon_matches_reference_and_keeps_compact_state(
        self,
    ) -> None:
        parameter = torch.nn.Parameter(torch.zeros(4, 3, 2))
        gradient = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
                [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]],
                [[1.0, -1.0], [2.0, -3.0], [5.0, -8.0]],
                [[3.0, 2.0], [1.0, 4.0], [5.0, 7.0]],
            ]
        )
        parameter.grad = gradient.clone()
        optimizer = SelectedExpertMuon(
            [parameter],
            expert_ids=(1, 3),
            lr=0.1,
            momentum=0.0,
            nesterov=False,
            reference_orthogonalization=True,
        )
        expected_direction = muon_matrix_direction(
            gradient[[1, 3]],
            ns_steps=5,
            rms_target=0.2,
            reference=True,
        )
        optimizer.step()

        torch.testing.assert_close(
            parameter[[1, 3]],
            -0.1 * expected_direction,
        )
        torch.testing.assert_close(parameter[[0, 2]], torch.zeros(2, 3, 2))
        state = optimizer.state[parameter]
        self.assertEqual(tuple(state["momentum_buffer"].shape), (2, 3, 2))
        self.assertEqual(state["momentum_buffer"].dtype, torch.float32)
        self.assertEqual(
            estimate_selected_expert_muon_state_bytes(optimizer),
            2 * 3 * 2 * 4 + 4 * 8,
        )

    def test_selected_expert_adamuon_matches_algorithm_one(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(3, 2, 3))
        gradient = torch.tensor(
            [
                [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
                [[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]],
                [[-1.0, 4.0, 2.0], [3.0, -5.0, 6.0]],
            ]
        )
        parameter.grad = gradient.clone()
        optimizer = SelectedExpertAdaMuon(
            [parameter],
            expert_ids=(0, 2),
            lr=0.05,
            momentum=0.5,
            nesterov=False,
            eps=1e-8,
            reference_orthogonalization=True,
        )
        second = torch.zeros_like(gradient[[0, 2]])
        expected_direction = adamuon_matrix_direction(
            gradient[[0, 2]],
            second,
            beta=0.5,
            eps=1e-8,
            ns_steps=5,
            rms_target=0.2,
            reference=True,
        )
        optimizer.step()

        torch.testing.assert_close(
            parameter[[0, 2]],
            -0.05 * expected_direction,
        )
        torch.testing.assert_close(parameter[1], torch.zeros(2, 3))
        state = optimizer.state[parameter]
        torch.testing.assert_close(state["second_moment"], second)
        rms = expected_direction.square().mean(dim=(-2, -1)).sqrt()
        torch.testing.assert_close(rms, torch.full_like(rms, 0.2))

    def test_selected_expert_muon_plan_and_algorithm_are_checkpoint_bound(
        self,
    ) -> None:
        parameter = torch.nn.Parameter(torch.zeros(4, 2, 2))
        parameter.grad = torch.ones_like(parameter)
        optimizer = SelectedExpertMuon(
            [parameter],
            expert_ids=(0, 2),
            lr=0.01,
            reference_orthogonalization=True,
        )
        optimizer.step()
        payload = optimizer.state_dict()

        incompatible_parameter = torch.nn.Parameter(parameter.detach().clone())
        incompatible = SelectedExpertMuon(
            [incompatible_parameter],
            expert_ids=(0,),
            lr=0.01,
            reference_orthogonalization=True,
        )
        with self.assertRaisesRegex(ValueError, "selection mismatch"):
            incompatible.load_state_dict(payload)

        different_algorithm_parameter = torch.nn.Parameter(
            parameter.detach().clone()
        )
        different_algorithm = SelectedExpertAdaMuon(
            [different_algorithm_parameter],
            expert_ids=(0, 2),
            lr=0.01,
        )
        with self.assertRaisesRegex(ValueError, "algorithm mismatch"):
            different_algorithm.load_state_dict(payload)

    def test_selected_expert_muon_factory_and_matrix_contract(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(3, 2, 4))
        result = build_optimizer(
            params=[parameter],
            named_params=[("experts.w1", parameter)],
            optimizer_type="selected_expert_adamuon",
            lr=0.01,
            weight_decay=0.1,
            allow_fallback=False,
            selected_expert_plan={"experts.w1": (1,)},
            muon_momentum=0.9,
            muon_nesterov=False,
            muon_ns_steps=4,
            muon_eps=1e-7,
            muon_rms_target=0.15,
        )
        self.assertIsInstance(result.optimizer, SelectedExpertAdaMuon)
        self.assertEqual(result.resolved_type, "selected_expert_adamuon")
        self.assertEqual(result.optimizer.param_groups[0]["ns_steps"], 4)

        non_matrix = torch.nn.Parameter(torch.zeros(3, 4))
        with self.assertRaisesRegex(ValueError, "ndim=3"):
            SelectedExpertMuon([non_matrix], expert_ids=(0,))

    def test_exact_polar_reference_is_semi_orthogonal(self) -> None:
        matrix = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]],
                [[2.0, -1.0], [4.0, 3.0], [1.0, 6.0]],
            ]
        )
        polar = orthogonalize_matrix_reference(matrix)
        identity = torch.eye(2).expand(2, 2, 2)
        torch.testing.assert_close(
            polar.mT @ polar,
            identity,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_newton_schulz_preserves_reference_polar_orientation(self) -> None:
        torch.manual_seed(41)
        matrix = torch.randn(4, 5, 3)
        reference = orthogonalize_matrix_reference(matrix)
        optimized = orthogonalize_matrix_newton_schulz(matrix, steps=5)
        cosine = torch.nn.functional.cosine_similarity(
            reference.flatten(1),
            optimized.flatten(1),
            dim=1,
        )
        self.assertTrue(bool((cosine > 0.98).all()), cosine)

    def test_optional_8bit_backend_has_explicit_fallback_policy(self) -> None:
        params = [torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))]
        with patch("importlib.import_module", side_effect=ModuleNotFoundError("no bnb")):
            result = build_optimizer(
                params=params,
                optimizer_type="adamw_8bit",
                lr=1e-3,
                weight_decay=0.0,
                allow_fallback=True,
            )
        self.assertEqual(result.resolved_type, "adamw")
        self.assertTrue(result.used_fallback)

        with patch("importlib.import_module", side_effect=ModuleNotFoundError("no bnb")):
            with self.assertRaises(RuntimeError):
                build_optimizer(
                    params=params,
                    optimizer_type="adamw_8bit",
                    lr=1e-3,
                    weight_decay=0.0,
                    allow_fallback=False,
                )

    def test_removed_schedule_free_alias_fails_explicitly(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "Unknown optimizer"):
            build_optimizer(
                params=[parameter],
                optimizer_type="schedule_free_adamw",
                lr=1e-2,
                weight_decay=0.0,
                allow_fallback=True,
            )

    def test_adamw_constructor_failure_is_not_silently_reimplemented(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        with patch(
            "mirai.core.training.optim.optimizer._TORCH_ADAMW",
            side_effect=RuntimeError("constructor failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                build_optimizer(
                    params=[parameter],
                    optimizer_type="adamw",
                    lr=1e-2,
                    weight_decay=0.0,
                    allow_fallback=True,
                )


@unittest.skipIf(torch is None, "torch is required")
class LoRAProContractTests(unittest.TestCase):
    @staticmethod
    def _pair(
        *,
        grouped: bool = False,
        dtype=None,
    ) -> LoRAFactorPair:
        resolved_dtype = dtype or torch.float32
        prefix = (2,) if grouped else ()
        generator = torch.Generator().manual_seed(31)
        lora_a = torch.nn.Parameter(
            torch.randn(
                *prefix,
                2,
                4,
                generator=generator,
                dtype=resolved_dtype,
            )
        )
        lora_b = torch.nn.Parameter(
            torch.zeros(*prefix, 5, 2, dtype=resolved_dtype)
        )
        return LoRAFactorPair(
            name="target",
            lora_a=lora_a,
            lora_b=lora_b,
            scale=2.0,
        )

    def test_gradient_correction_matches_full_weight_tangent_projection(
        self,
    ) -> None:
        generator = torch.Generator().manual_seed(13)
        lora_a = torch.randn(2, 4, generator=generator)
        lora_b = torch.randn(5, 2, generator=generator)
        full_gradient = torch.randn(5, 4, generator=generator)
        scale = 1.7
        raw_a = scale * lora_b.transpose(0, 1) @ full_gradient
        raw_b = scale * full_gradient @ lora_a.transpose(0, 1)

        corrected_a, corrected_b = lora_pro_correct_gradients(
            lora_a=lora_a,
            lora_b=lora_b,
            grad_a=raw_a,
            grad_b=raw_b,
            scale=scale,
            damping=1e-8,
            solve_gauge=False,
        )
        equivalent = lora_pro_equivalent_gradient(
            lora_a=lora_a,
            lora_b=lora_b,
            grad_a=corrected_a,
            grad_b=corrected_b,
            scale=scale,
        )
        projection_b = lora_b @ torch.linalg.pinv(
            lora_b.transpose(0, 1) @ lora_b,
            hermitian=True,
        ) @ lora_b.transpose(0, 1)
        projection_a = lora_a.transpose(0, 1) @ torch.linalg.pinv(
            lora_a @ lora_a.transpose(0, 1),
            hermitian=True,
        ) @ lora_a
        identity_b = torch.eye(lora_b.shape[0])
        expected = (
            projection_b @ full_gradient
            + (identity_b - projection_b) @ full_gradient @ projection_a
        )
        torch.testing.assert_close(equivalent, expected, rtol=2e-5, atol=2e-5)

    def test_positive_sylvester_solver_satisfies_matrix_equation(self) -> None:
        left_base = torch.tensor([[2.0, -0.5], [0.25, 1.5]])
        right_base = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
        left = left_base.transpose(0, 1) @ left_base
        right = right_base.transpose(0, 1) @ right_base
        value = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
        solution = solve_positive_sylvester(
            left,
            right,
            value,
            damping=1e-8,
        )
        torch.testing.assert_close(
            left @ solution + solution @ right,
            value,
            rtol=2e-5,
            atol=2e-5,
        )

    def test_adamw_step_matches_algorithm_two_reference(self) -> None:
        generator = torch.Generator().manual_seed(73)
        lora_a = torch.nn.Parameter(torch.randn(2, 4, generator=generator))
        lora_b = torch.nn.Parameter(torch.randn(5, 2, generator=generator))
        pair = LoRAFactorPair(
            name="reference",
            lora_a=lora_a,
            lora_b=lora_b,
            scale=1.5,
        )
        raw_a = torch.randn(lora_a.shape, generator=generator)
        raw_b = torch.randn(lora_b.shape, generator=generator)
        lora_a.grad = raw_a.clone()
        lora_b.grad = raw_b.clone()
        before_a = lora_a.detach().clone()
        before_b = lora_b.detach().clone()
        damping = 1e-6
        epsilon = 1e-6
        learning_rate = 0.02

        gram_a = before_a @ before_a.transpose(0, 1)
        gram_b = before_b.transpose(0, 1) @ before_b
        identity = torch.eye(pair.rank)
        inverse_a = torch.linalg.solve(
            gram_a + damping * identity,
            identity,
        )
        inverse_b = torch.linalg.solve(
            gram_b + damping * identity,
            identity,
        )
        scale_squared = pair.scale**2
        preliminary_a = inverse_b @ raw_a / scale_squared
        preliminary_b = (
            raw_b
            - before_b @ inverse_b @ before_b.transpose(0, 1) @ raw_b
        ) @ inverse_a / scale_squared
        equivalent = pair.scale * (
            preliminary_b @ before_a + before_b @ preliminary_a
        )
        adam_gradient = equivalent / (equivalent.abs() + epsilon)
        pseudo_a = pair.scale * before_b.transpose(0, 1) @ adam_gradient
        pseudo_b = pair.scale * adam_gradient @ before_a.transpose(0, 1)
        base_a = inverse_b @ pseudo_a / scale_squared
        base_b = (
            pseudo_b
            - before_b @ inverse_b @ before_b.transpose(0, 1) @ pseudo_b
        ) @ inverse_a / scale_squared
        right_hand_side = (
            -inverse_b @ pseudo_a @ before_a.transpose(0, 1)
            / scale_squared
        )
        operator_columns = []
        for row in range(pair.rank):
            for column in range(pair.rank):
                basis = torch.zeros(pair.rank, pair.rank)
                basis[row, column] = 1.0
                operator_columns.append(
                    (gram_b @ basis + basis @ gram_a).reshape(-1)
                )
        gauge = torch.linalg.solve(
            torch.stack(operator_columns, dim=1),
            right_hand_side.reshape(-1),
        ).reshape(pair.rank, pair.rank)
        expected_a = before_a - learning_rate * (
            base_a + gauge @ before_a
        )
        expected_b = before_b - learning_rate * (
            base_b - before_b @ gauge
        )

        optimizer = LoRAProAdamW(
            [lora_a, lora_b],
            pairs=[pair],
            lr=learning_rate,
            betas=(0.0, 0.0),
            eps=epsilon,
            damping=damping,
        )
        optimizer.step()
        torch.testing.assert_close(lora_a, expected_a, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(lora_b, expected_b, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(
            optimizer.state[lora_a]["exp_avg"],
            equivalent.unsqueeze(0),
            rtol=3e-5,
            atol=3e-5,
        )

    def test_grouped_zero_b_step_uses_full_fp32_moments_and_roundtrips(
        self,
    ) -> None:
        pair = self._pair(grouped=True, dtype=torch.bfloat16)
        pair.lora_a.grad = torch.ones_like(pair.lora_a)
        pair.lora_b.grad = torch.full_like(pair.lora_b, 0.25)
        optimizer = LoRAProAdamW(
            [pair.lora_a, pair.lora_b],
            pairs=[pair],
            lr=0.01,
        )
        optimizer.step()

        state = optimizer.state[pair.lora_a]
        self.assertEqual(state["exp_avg"].dtype, torch.float32)
        self.assertEqual(tuple(state["exp_avg"].shape), (2, 5, 4))
        self.assertEqual(
            optimizer.estimated_state_bytes,
            estimate_lora_pro_state_bytes([pair]),
        )
        self.assertTrue(bool(torch.isfinite(pair.lora_b).all()))
        self.assertTrue(bool((pair.lora_b != 0).any()))

        payload = optimizer.state_dict()
        restored_pair = self._pair(grouped=True, dtype=torch.bfloat16)
        restored = LoRAProAdamW(
            [restored_pair.lora_a, restored_pair.lora_b],
            pairs=[restored_pair],
            lr=0.01,
        )
        restored.load_state_dict(payload)
        restored_state = restored.state[restored_pair.lora_a]
        self.assertEqual(restored_state["exp_avg"].dtype, torch.float32)
        self.assertEqual(restored_state["exp_avg_sq"].dtype, torch.float32)
        torch.testing.assert_close(
            restored_state["exp_avg"],
            state["exp_avg"],
        )

    def test_factory_rejects_topology_mismatch(self) -> None:
        pair = self._pair()
        result = build_optimizer(
            params=[pair.lora_a, pair.lora_b],
            optimizer_type="lora_pro_adamw",
            lr=1e-3,
            weight_decay=0.0,
            allow_fallback=False,
            lora_pairs=(pair,),
        )
        self.assertIsInstance(result.optimizer, LoRAProAdamW)

        incompatible = LoRAFactorPair(
            name="other",
            lora_a=torch.nn.Parameter(pair.lora_a.detach().clone()),
            lora_b=torch.nn.Parameter(pair.lora_b.detach().clone()),
            scale=pair.scale,
        )
        restored = LoRAProAdamW(
            [incompatible.lora_a, incompatible.lora_b],
            pairs=[incompatible],
        )
        with self.assertRaisesRegex(ValueError, "topology"):
            restored.load_state_dict(result.optimizer.state_dict())


@unittest.skipIf(torch is None, "torch is required")
class LoRAMuonContractTests(unittest.TestCase):
    @staticmethod
    def _pair(*, grouped: bool = False, dtype=None) -> LoRAFactorPair:
        resolved_dtype = dtype or torch.float32
        prefix = (2,) if grouped else ()
        generator = torch.Generator().manual_seed(101)
        lora_a = torch.nn.Parameter(
            torch.randn(
                *prefix,
                3,
                5,
                generator=generator,
                dtype=resolved_dtype,
            )
        )
        lora_b = torch.nn.Parameter(
            torch.randn(
                *prefix,
                7,
                3,
                generator=generator,
                dtype=resolved_dtype,
            )
        )
        return LoRAFactorPair(
            name="target",
            lora_a=lora_a,
            lora_b=lora_b,
            scale=2.0,
        )

    def test_newton_schulz_primitives_match_svd_and_eigh_oracles(self) -> None:
        generator = torch.Generator().manual_seed(113)
        matrix = torch.randn(9, 3, generator=generator)
        gram_source = torch.randn(6, 3, generator=generator)
        psd = gram_source.transpose(0, 1) @ gram_source
        torch.testing.assert_close(
            matrix_sign_newton_schulz(matrix),
            matrix_sign_reference(matrix),
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            psd_inverse_sqrt_newton_schulz(psd),
            psd_inverse_sqrt_reference(psd),
            rtol=3e-2,
            atol=3e-2,
        )

    def test_factor_directions_match_independent_algorithm_one_oracle(
        self,
    ) -> None:
        pair = self._pair()
        generator = torch.Generator().manual_seed(127)
        moment_a = torch.randn(pair.lora_a.shape, generator=generator)
        moment_b = torch.randn(pair.lora_b.shape, generator=generator)

        def inverse_root(value):
            norm = torch.linalg.matrix_norm(value, ord="fro")
            normalized = value / norm + 1e-5 * torch.eye(value.shape[-1])
            eigenvalues, eigenvectors = torch.linalg.eigh(normalized)
            return (
                norm.rsqrt()
                * eigenvectors
                @ torch.diag(eigenvalues.rsqrt())
                @ eigenvectors.transpose(0, 1)
            )

        def matrix_sign(value):
            u, _, vh = torch.linalg.svd(value, full_matrices=False)
            return u @ vh

        root_for_a = inverse_root(
            pair.lora_b.detach().transpose(0, 1) @ pair.lora_b.detach()
        )
        root_for_b = inverse_root(
            pair.lora_a.detach() @ pair.lora_a.detach().transpose(0, 1)
        )
        expected_b = (
            -0.5
            / pair.scale
            * matrix_sign(moment_b @ root_for_b)
            @ root_for_b
        )
        expected_a = (
            -0.5
            / pair.scale
            * matrix_sign(moment_a.transpose(0, 1) @ root_for_a)
            @ root_for_a
        ).transpose(0, 1)
        actual_a, actual_b = lora_muon_factor_directions(
            lora_a=pair.lora_a,
            lora_b=pair.lora_b,
            moment_a=moment_a,
            moment_b=moment_b,
            scale=pair.scale,
            numerical="reference",
        )
        torch.testing.assert_close(actual_a, expected_a, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual_b, expected_b, rtol=2e-5, atol=2e-5)

    def test_scalar_gauge_rebalancing_preserves_product_and_direction(
        self,
    ) -> None:
        pair = self._pair()
        generator = torch.Generator().manual_seed(131)
        moment_a = torch.randn(pair.lora_a.shape, generator=generator)
        moment_b = torch.randn(pair.lora_b.shape, generator=generator)
        before_product = pair.lora_b.detach() @ pair.lora_a.detach()
        before_a, before_b = lora_muon_factor_directions(
            lora_a=pair.lora_a,
            lora_b=pair.lora_b,
            moment_a=moment_a,
            moment_b=moment_b,
            scale=pair.scale,
            numerical="reference",
        )
        before_direction = (
            pair.lora_b.detach() @ before_a
            + before_b @ pair.lora_a.detach()
        )
        rebalance_lora_muon_gauge(
            lora_a=pair.lora_a,
            lora_b=pair.lora_b,
            moment_a=moment_a,
            moment_b=moment_b,
            alpha=1.0,
        )
        after_a, after_b = lora_muon_factor_directions(
            lora_a=pair.lora_a,
            lora_b=pair.lora_b,
            moment_a=moment_a,
            moment_b=moment_b,
            scale=pair.scale,
            numerical="reference",
        )
        after_direction = (
            pair.lora_b.detach() @ after_a
            + after_b @ pair.lora_a.detach()
        )
        torch.testing.assert_close(
            pair.lora_b.detach() @ pair.lora_a.detach(),
            before_product,
            rtol=2e-5,
            atol=2e-5,
        )
        torch.testing.assert_close(
            after_direction,
            before_direction,
            rtol=3e-4,
            atol=3e-4,
        )

    def test_zero_b_boundary_and_grouped_bf16_state_roundtrip(self) -> None:
        pair = self._pair(grouped=True, dtype=torch.bfloat16)
        pair.lora_b.data.zero_()
        pair.lora_a.grad = torch.zeros_like(pair.lora_a)
        pair.lora_b.grad = torch.full_like(pair.lora_b, 0.25)
        optimizer = LoRAMuon(
            [pair.lora_a, pair.lora_b],
            pairs=[pair],
            lr=0.01,
            momentum=0.0,
            stochastic_rounding=True,
        )
        before_a = pair.lora_a.detach().clone()
        optimizer.step()
        self.assertTrue(bool(torch.isfinite(pair.lora_b).all()))
        self.assertTrue(bool((pair.lora_b != 0).any()))
        torch.testing.assert_close(pair.lora_a, before_a)
        state = optimizer.state[pair.lora_a]
        self.assertEqual(state["moment_a"].dtype, torch.float32)
        self.assertEqual(state["moment_b"].dtype, torch.float32)
        self.assertEqual(
            optimizer.estimated_state_bytes,
            estimate_lora_muon_state_bytes([pair]),
        )

        payload = optimizer.state_dict()
        restored_pair = self._pair(grouped=True, dtype=torch.bfloat16)
        restored = LoRAMuon(
            [restored_pair.lora_a, restored_pair.lora_b],
            pairs=[restored_pair],
            lr=0.01,
            momentum=0.0,
            stochastic_rounding=True,
        )
        restored.load_state_dict(payload)
        restored_state = restored.state[restored_pair.lora_a]
        torch.testing.assert_close(
            restored_state["moment_a"],
            state["moment_a"],
        )
        torch.testing.assert_close(
            restored_state["moment_b"],
            state["moment_b"],
        )

    def test_factory_and_topology_contract(self) -> None:
        pair = self._pair()
        result = build_optimizer(
            params=[pair.lora_a, pair.lora_b],
            optimizer_type="lora_muon",
            lr=0.01,
            weight_decay=0.01,
            allow_fallback=False,
            lora_pairs=(pair,),
            muon_momentum=0.9,
            lora_muon_gauge_rebalance_interval=4,
        )
        self.assertIsInstance(result.optimizer, LoRAMuon)
        incompatible = self._pair()
        incompatible = LoRAFactorPair(
            name="other",
            lora_a=incompatible.lora_a,
            lora_b=incompatible.lora_b,
            scale=incompatible.scale,
        )
        restored = LoRAMuon(
            [incompatible.lora_a, incompatible.lora_b],
            pairs=[incompatible],
            lr=0.01,
            momentum=0.9,
            weight_decay=0.01,
            gauge_rebalance_interval=4,
        )
        with self.assertRaisesRegex(ValueError, "topology"):
            restored.load_state_dict(result.optimizer.state_dict())


@unittest.skipIf(torch is None, "torch is required")
class RouterParameterGroupTests(unittest.TestCase):
    """Router parameters are addressable independently of the ffn group.

    The router weight lives at ``blocks.<i>.ffn.router.weight``, so a group
    resolver that matches ``ffn`` first makes the router unreachable.
    """

    def _named(self):
        return [
            ("blocks.0.ffn.router.weight", torch.nn.Parameter(torch.zeros(2))),
            ("blocks.0.ffn.experts.w1.weight", torch.nn.Parameter(torch.zeros(2))),
        ]

    def _lr_of(self, groups, param):
        for group in groups:
            if any(entry is param for entry in group["params"]):
                return float(group["lr"])
        raise AssertionError("parameter missing from every group")

    def _decay_of(self, groups, param):
        for group in groups:
            if any(entry is param for entry in group["params"]):
                return float(group["weight_decay"])
        raise AssertionError("parameter missing from every group")

    def test_router_multiplier_is_independent_of_ffn(self) -> None:
        named = self._named()
        groups = build_param_groups(
            named_params=named,
            base_lr=1.0,
            weight_decay=0.0,
            weight_decay_filter="none",
            loraplus_lr_ratio=1.0,
            module_lr_multipliers={"router": 0.1, "ffn": 2.0},
        )
        self.assertAlmostEqual(self._lr_of(groups, named[0][1]), 0.1)
        self.assertAlmostEqual(self._lr_of(groups, named[1][1]), 2.0)

    def test_router_inherits_ffn_multiplier_when_unset(self) -> None:
        named = self._named()
        groups = build_param_groups(
            named_params=named,
            base_lr=1.0,
            weight_decay=0.0,
            weight_decay_filter="none",
            loraplus_lr_ratio=1.0,
            module_lr_multipliers={"ffn": 2.0},
        )
        self.assertAlmostEqual(self._lr_of(groups, named[0][1]), 2.0)

    def test_router_aware_filter_exempts_router_from_decay(self) -> None:
        named = self._named()
        groups = build_param_groups(
            named_params=named,
            base_lr=1.0,
            weight_decay=0.05,
            weight_decay_filter="router_aware",
            loraplus_lr_ratio=1.0,
        )
        self.assertEqual(self._decay_of(groups, named[0][1]), 0.0)
        self.assertAlmostEqual(self._decay_of(groups, named[1][1]), 0.05)

    def test_default_filter_still_decays_the_router(self) -> None:
        named = self._named()
        groups = build_param_groups(
            named_params=named,
            base_lr=1.0,
            weight_decay=0.05,
            weight_decay_filter="lora_b_bias",
            loraplus_lr_ratio=1.0,
        )
        self.assertAlmostEqual(self._decay_of(groups, named[0][1]), 0.05)


@unittest.skipIf(torch is None, "torch is required")
class ESFTContractTests(unittest.TestCase):
    class _FakeRouter(torch.nn.Module):
        def forward(self, value):
            self.last_top_indices = torch.tensor([[0, 1]])
            self.last_top_scores = torch.tensor([[0.8, 0.2]])
            self.last_route_active_mask = None
            return value

    @staticmethod
    def _tiny_config(*, selection: str = "esft_gate") -> TrainingConfig:
        return TrainingConfig.from_dict(
            {
                "model": {
                    "params": {
                        "variant": "tiny-video",
                        "hidden_size": 16,
                        "attention_heads": 2,
                        "num_layers": 2,
                        "num_experts": 4,
                        "experts_per_token": 2,
                        "shared_experts": 0,
                        "latent_channels": 1,
                        "patch_size": 1,
                    }
                },
                "adapter": {
                    "type": "selected_expert",
                    "expert_selection": selection,
                    "esft_selection_mass": 0.2,
                    "esft_calibration_samples": 4,
                },
                "optimizer": {
                    "type": "selected_expert_adamw",
                    "weight_decay_filter": "none",
                },
            }
        )

    def test_gate_and_token_affinity_match_esft_equations_6_and_7(self) -> None:
        accumulator = ESFTAffinityAccumulator(num_experts=3)
        accumulator.observe(
            torch.tensor([[0, 1], [1, 2]]),
            torch.tensor([[0.75, 0.25], [0.60, 0.40]]),
        )
        torch.testing.assert_close(
            accumulator.normalized_scores("gate"),
            torch.tensor([0.375, 0.425, 0.200], dtype=torch.float64),
        )
        torch.testing.assert_close(
            accumulator.normalized_scores("token"),
            torch.tensor([0.25, 0.50, 0.25], dtype=torch.float64),
        )

    def test_selection_is_minimum_stable_equation_8_prefix(self) -> None:
        scores = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
        self.assertEqual(
            select_esft_experts(scores, selection_mass=0.6),
            (0, 1),
        )
        tied = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float64)
        self.assertEqual(
            select_esft_experts(tied, selection_mass=0.65),
            (0, 1),
        )

        first = ESFTAffinityAccumulator(num_experts=3)
        second = ESFTAffinityAccumulator(num_experts=3)
        first.observe(torch.tensor([[0, 1]]), torch.tensor([[0.9, 0.1]]))
        second.observe(torch.tensor([[1, 2]]), torch.tensor([[0.2, 0.8]]))
        plan = build_esft_selection_plan(
            {"blocks.0.experts": first, "blocks.1.experts": second},
            score_mode="gate",
            selection_mass=0.75,
            calibration_samples=1,
        )
        self.assertEqual(
            plan.selected_experts,
            {
                "blocks.0.experts": (0,),
                "blocks.1.experts": (2,),
            },
        )
        self.assertEqual(len(plan.fingerprint), 64)

    def test_capture_is_temporary_and_rejects_variable_cardinality(self) -> None:
        router = self._FakeRouter()
        target = ESFTCalibrationTarget(
            name="layer.experts",
            router=router,
            num_experts=2,
        )
        with ESFTCalibrationCapture({"layer.experts": target}) as capture:
            router(torch.ones(1))
        self.assertFalse(router._forward_hooks)
        accumulator = capture.accumulators["layer.experts"]
        self.assertEqual(accumulator.token_count, 1)
        self.assertEqual(accumulator.gate_mass.device.type, "cpu")

        with self.assertRaisesRegex(ValueError, "fixed-cardinality"):
            accumulator.observe(
                torch.tensor([[0, 1]]),
                torch.tensor([[0.8, 0.2]]),
                active_mask=torch.tensor([[True, False]]),
            )

    def test_preoptimizer_driver_binds_plan_and_restores_rng(self) -> None:
        routers = {"layer.experts": self._FakeRouter()}

        class Provider:
            @staticmethod
            def supports_esft_expert_selection(_config):
                return True

            @staticmethod
            def build_esft_calibration_targets(_pipeline):
                return {
                    name: ESFTCalibrationTarget(
                        name=name,
                        router=router,
                        num_experts=2,
                    )
                    for name, router in routers.items()
                }

        class Pipeline:
            def __init__(self):
                self.plan = None

            def set_selected_expert_plan(self, plan):
                self.plan = dict(plan)

        class Trainer:
            def __init__(self):
                self.pipeline = Pipeline()
                self.calls = 0

            @staticmethod
            def begin_validation():
                return {}

            @staticmethod
            def end_validation(_state):
                return None

            def compute_loss(self, _batch, *, training):
                self.assert_not_training(training)
                self.calls += 1
                for router in routers.values():
                    router(torch.ones(1))
                return torch.tensor(0.0), {}

            @staticmethod
            def assert_not_training(training):
                if training is not False:
                    raise AssertionError("ESFT calibration must use evaluation mode.")

        config = SimpleNamespace(
            adapter=SimpleNamespace(
                expert_selection="esft_gate",
                esft_calibration_samples=3,
                esft_selection_mass=0.5,
            ),
            training=SimpleNamespace(batch_size=2),
            model=SimpleNamespace(type="fake"),
        )
        prepared = SimpleNamespace(
            train_records=[],
            temporal_base_ids=[],
            temporal_groups={},
        )
        rng = random.Random(17)
        rng_state = rng.getstate()
        trainer = Trainer()
        with (
            patch.object(
                esft_runtime,
                "get_model_family_provider",
                return_value=Provider(),
            ),
            patch.object(
                esft_runtime,
                "resolve_step_sampling_context",
                return_value=object(),
            ),
            patch.object(
                esft_runtime,
                "_build_training_batch_factory",
                return_value=lambda _step: {},
            ),
        ):
            report = esft_runtime.maybe_initialize_esft(
                trainer=trainer,
                config=config,
                prepared_data=prepared,
                compute_device=torch.device("cpu"),
                compute_dtype=torch.float32,
                curriculum=object(),
                rng=rng,
                run_state=SimpleNamespace(global_step=0),
                grad_accum=1,
            )

        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(trainer.calls, 2)
        self.assertEqual(report.requested_samples, 3)
        self.assertEqual(report.observed_samples, 4)
        self.assertEqual(trainer.pipeline.plan, {"layer.experts": (0,)})
        self.assertEqual(rng.getstate(), rng_state)

    def test_lingbot_plan_roundtrip_requires_complete_state(self) -> None:
        config = self._tiny_config()
        pipeline = LingBotVideoPipeline.from_training_config(config)
        pipeline.set_adapter_config(config.adapter)
        modules = {
            name: module
            for name, module in pipeline.transformer.named_modules()
            if bool(getattr(module, "mirai_expert_tensor_host", False))
        }
        names = sorted(modules)
        self.assertEqual(len(names), 2)
        plan = {names[0]: (0, 2), names[1]: (1,)}
        pipeline.set_selected_expert_plan(plan)
        parameter_plan = pipeline.get_selected_expert_parameter_plan()
        self.assertEqual(parameter_plan[f"{names[0]}.w1"], (0, 2))
        self.assertEqual(parameter_plan[f"{names[1]}.w1"], (1,))

        with torch.no_grad():
            modules[names[0]].w1[0].add_(1.0)
            modules[names[1]].w1[1].sub_(1.0)
        state = pipeline.state_dict()
        restored = LingBotVideoPipeline.from_training_config(config)
        restored.set_adapter_config(config.adapter)
        restored.set_selected_expert_plan(plan)
        restored.load_state_dict(state)
        restored_modules = dict(restored.transformer.named_modules())
        torch.testing.assert_close(
            restored_modules[names[0]].w1[0],
            modules[names[0]].w1[0],
        )
        torch.testing.assert_close(
            restored_modules[names[1]].w1[1],
            modules[names[1]].w1[1],
        )

        incomplete = dict(state)
        del incomplete[f"selected_expert.{names[0]}.w1"]
        with self.assertRaisesRegex(ValueError, "every selected expert tensor"):
            restored.load_state_dict(incomplete)

    def test_runtime_accepts_esft_without_manual_ids_and_rejects_expert_choice(
        self,
    ) -> None:
        config = self._tiny_config()
        validate_training_runtime_config(config)
        self.assertEqual(config.optimizer.selected_expert_ids, [])

        config.model.params.moe_routing_mode = "expert_choice"
        with self.assertRaisesRegex(ValueError, "fixed-cardinality token-choice"):
            validate_training_runtime_config(config)


if __name__ == "__main__":
    unittest.main()
