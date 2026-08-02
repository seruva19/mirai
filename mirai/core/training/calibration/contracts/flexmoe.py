from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mirai.core.moe.calibration.flexmoe import (  # noqa: E402
    FlexMoEActionController,
    FlexMoECalibrationTarget,
    FlexMoEExpertLoadObserver,
    FlexMoETaylorGradientObserver,
    channel_taylor_saliency,
    load_action_plans,
    load_ranking_evidence,
)
from mirai.core.models.compressed_weights.execution.experts import (  # noqa: E402
    CompressedGroupedExperts,
)
from mirai.core.training.calibration.flexmoe import (  # noqa: E402
    FlexMoEActionLearningSession,
    FlexMoEActionLearningSpec,
    run_flexmoe_action_session,
    run_flexmoe_ranking_session,
)
from mirai.core.training.strategies.base import TrainingInputs  # noqa: E402


def _build_session(seed: int = 13) -> FlexMoEActionLearningSession:
    controllers = {
        "blocks.0.experts": FlexMoEActionController(
            num_experts=2,
            action_ratios=(0.25, 0.5, 0.75, 1.0),
            thickest_logit_margin=1.0,
        ),
        "blocks.1.experts": FlexMoEActionController(
            num_experts=3,
            action_ratios=(0.25, 0.5, 0.75, 1.0),
            thickest_logit_margin=1.0,
        ),
    }
    optimizer = torch.optim.Adam(
        [controller.logits for controller in controllers.values()],
        lr=0.05,
    )
    return FlexMoEActionLearningSession(
        controllers=controllers,
        permutations={
            "blocks.0.experts": torch.tensor([[3, 2, 1, 0], [1, 3, 0, 2]]),
            "blocks.1.experts": torch.tensor(
                [[0, 1, 2, 3], [3, 1, 2, 0], [2, 0, 3, 1]]
            ),
        },
        optimizer=optimizer,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        spec=FlexMoEActionLearningSpec(
            total_steps=3,
            temperature_start=1.0,
            temperature_end=0.5,
            cost_weight_start=0.0,
            cost_weight_end=2.0,
            entropy_weight_start=0.3,
            entropy_weight_end=0.0,
        ),
    )


def _quality(masks):
    loss = sum(
        (mask * torch.arange(mask.shape[1], dtype=mask.dtype)).square().mean()
        for mask in masks.values()
    )
    return loss, {
        "blocks.0.experts": torch.tensor([0.75, 0.25]),
        "blocks.1.experts": torch.tensor([0.5, 0.3, 0.2]),
    }


def test_flexmoe_linear_schedule_hits_exact_endpoints() -> None:
    spec = _build_session().spec
    assert spec.at(0) == (1.0, 0.0, 0.3)
    assert spec.at(2) == (0.5, 2.0, 0.0)


def test_flexmoe_action_learning_exact_resume_matches_uninterrupted() -> None:
    uninterrupted = _build_session()
    uninterrupted_reports = [uninterrupted.step(_quality) for _ in range(3)]
    assert uninterrupted.complete

    first = _build_session()
    first_report = first.step(_quality)
    state = copy.deepcopy(first.state_dict())
    resumed = _build_session(seed=999)
    resumed.load_state_dict(state)
    resumed_reports = [first_report, resumed.step(_quality), resumed.step(_quality)]
    assert resumed.complete

    for left, right in zip(uninterrupted_reports, resumed_reports, strict=True):
        assert left == right
    for name in uninterrupted.controllers:
        torch.testing.assert_close(
            uninterrupted.controllers[name].logits,
            resumed.controllers[name].logits,
            rtol=0.0,
            atol=0.0,
        )
        left_plan = uninterrupted.action_plans()[name]
        right_plan = resumed.action_plans()[name]
        torch.testing.assert_close(left_plan.logits, right_plan.logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            left_plan.expert_load,
            right_plan.expert_load,
            rtol=0.0,
            atol=0.0,
        )


def test_flexmoe_load_observer_uses_active_assignment_frequency() -> None:
    observer = FlexMoEExpertLoadObserver(3)
    indices = torch.tensor([[0, 1], [2, 1], [0, 2]])
    weights = torch.tensor([[0.8, 0.2], [1.0, 0.0], [0.6, 0.4]])
    observer.bind_routes(indices, weights)
    observer.begin_routes(num_tokens=3, top_k=2, device=torch.device("cpu"))
    observer.capture_routes(torch.zeros(5, 4), torch.arange(5))
    observer.end_routes()
    torch.testing.assert_close(
        observer.take_load(),
        torch.tensor([0.4, 0.2, 0.4], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_flexmoe_taylor_observer_matches_explicit_parameter_gradients() -> None:
    generator = torch.Generator(device="cpu").manual_seed(29)
    weights = {
        "w1": torch.randn(2, 3, 2, generator=generator, requires_grad=True),
        "w2": torch.randn(2, 2, 3, generator=generator, requires_grad=True),
        "w3": torch.randn(2, 3, 2, generator=generator, requires_grad=True),
    }
    inputs = [
        torch.randn(4, 2, generator=generator),
        torch.randn(3, 2, generator=generator),
    ]

    reference_outputs = []
    for expert, value in enumerate(inputs):
        gate = value @ weights["w1"][expert].transpose(0, 1)
        up = value @ weights["w3"][expert].transpose(0, 1)
        hidden = torch.nn.functional.silu(gate) * up
        reference_outputs.append(hidden @ weights["w2"][expert].transpose(0, 1))
    reference_loss = reference_outputs[0].square().mean() + 0.7 * reference_outputs[
        1
    ].sin().sum()
    reference_loss.backward()
    expected = channel_taylor_saliency(
        {name: value.detach() for name, value in weights.items()},
        {name: value.grad for name, value in weights.items()},
    )

    observer = FlexMoETaylorGradientObserver(num_experts=2, intermediate_size=3)
    observer.begin_batch(device=torch.device("cpu"))
    observed_outputs = []
    for expert, value in enumerate(inputs):
        x = value.detach().requires_grad_(True)
        w1 = weights["w1"][expert].detach()
        w2 = weights["w2"][expert].detach()
        w3 = weights["w3"][expert].detach()
        gate = x @ w1.transpose(0, 1)
        up = x @ w3.transpose(0, 1)
        hidden = torch.nn.functional.silu(gate) * up
        output = hidden @ w2.transpose(0, 1)
        observer.capture(
            expert_index=expert,
            inputs=x,
            w1=w1,
            w2=w2,
            w3=w3,
            gate=gate,
            up=up,
            hidden=hidden,
            output=output,
        )
        observed_outputs.append(output)
    observed_loss = observed_outputs[0].square().mean() + 0.7 * observed_outputs[
        1
    ].sin().sum()
    observed_loss.backward()
    observer.finish_batch()
    torch.testing.assert_close(
        observer.evidence().saliency,
        expected.to(torch.float64),
        rtol=2e-6,
        atol=2e-6,
    )


def test_flexmoe_resume_rejects_changed_schedule() -> None:
    source = _build_session()
    source.step(_quality)
    changed = _build_session()
    changed.spec = FlexMoEActionLearningSpec(
        total_steps=4,
        temperature_start=1.0,
        temperature_end=0.5,
        cost_weight_start=0.0,
        cost_weight_end=2.0,
        entropy_weight_start=0.3,
        entropy_weight_end=0.0,
    )
    with pytest.raises(ValueError, match="schedule changed"):
        changed.load_state_dict(source.state_dict())


def test_flexmoe_provider_session_emits_ranking_and_action_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(97)

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = CompressedGroupedExperts.from_empty(
                num_experts=2,
                group_sizes=(4,),
                expert_weight_access="chunked_dequant",
                expert_dequant_chunk_size=2,
            )
            self.experts.load_dense_weight(
                "w1", torch.randn(2, 4, 4, generator=generator)
            )
            self.experts.load_dense_weight(
                "w2", torch.randn(2, 4, 4, generator=generator)
            )
            self.experts.load_dense_weight(
                "w3", torch.randn(2, 4, 4, generator=generator)
            )

        def forward(self, tokens):
            indices = torch.tensor([[0], [1], [0], [1]])
            scores = torch.tensor([[0.8], [0.7], [0.6], [0.9]])
            return self.experts.run_direct_routed(tokens, scores, indices)

    model = ToyModel()

    class ToyPipeline:
        def get_training_model(self):
            return model

        def finish_backward_offloads(self):
            return None

    class ToyObjective:
        @staticmethod
        def get_named_trainable_parameters():
            return ()

    class ToyTrainer:
        def __init__(self) -> None:
            self.pipeline = ToyPipeline()
            self.objective = ToyObjective()

        def begin_validation(self):
            state = bool(model.training)
            model.eval()
            return {"training": state}

        @staticmethod
        def end_validation(state):
            model.train(bool(state["training"]))

        @staticmethod
        def prepare_objective_calibration_inputs(batch, *, training=False):
            del training
            return TrainingInputs(
                noisy_latents=batch["tokens"],
                timestep=torch.zeros(4),
                noise=torch.zeros_like(batch["tokens"]),
                clean_latents=torch.zeros_like(batch["tokens"]),
                text_embeds={},
            )

        @staticmethod
        def predict_objective_calibration_inputs(inputs, *, training=False):
            del training
            return model(inputs.noisy_latents)

        @staticmethod
        def evaluate_calibration_task_loss(*, batch, inputs, prediction):
            del batch, inputs
            return SimpleNamespace(loss_pre_accum=prediction.float().square().mean())

    class ToyProvider:
        @staticmethod
        def supports_flexmoe_calibration(config):
            del config
            return True

        @staticmethod
        def build_flexmoe_calibration_targets(pipeline):
            del pipeline
            return {
                "experts": FlexMoECalibrationTarget(
                    name="experts",
                    host=model.experts,
                    num_experts=2,
                    intermediate_size=4,
                ).validate()
            }

    config = SimpleNamespace(
        model=SimpleNamespace(
            type="toy",
            params=SimpleNamespace(
                flexmoe_calibration="nested",
                expert_weight_compression="off",
            ),
        ),
        memory=SimpleNamespace(frozen_weight_packed_state_path="source.safetensors"),
    )
    session = SimpleNamespace(
        config=config,
        trainer=ToyTrainer(),
        compute_device=torch.device("cpu"),
        rng=random.Random(3),
        manifest=SimpleNamespace(
            dataset_snapshot_id="dataset",
            model_snapshot_id="base-model",
            config_snapshot_id="config",
        ),
    )
    batches = [
        {"tokens": torch.randn(4, 4, generator=generator)}
        for _index in range(4)
    ]
    monkeypatch.setattr(
        "mirai.core.training.calibration.flexmoe.get_model_family_provider",
        lambda _name: ToyProvider(),
    )
    monkeypatch.setattr(
        "mirai.core.training.calibration.flexmoe._source_artifact_fingerprint",
        lambda _config: "sha256:source",
    )
    monkeypatch.setattr(
        "mirai.core.training.calibration.flexmoe.resolve_step_sampling_context",
        lambda _session: None,
    )
    monkeypatch.setattr(
        "mirai.core.training.calibration.flexmoe._build_training_batch_factory",
        lambda **_kwargs: lambda step: batches[step],
    )

    ranking_path = tmp_path / "ranking.safetensors"
    ranking_report = run_flexmoe_ranking_session(
        session,
        output_path=ranking_path,
        calibration_steps=2,
    )
    assert ranking_report.calibration_steps == 2
    ranking, ranking_lineage = load_ranking_evidence(ranking_path)
    assert set(ranking) == {"experts"}
    assert ranking_lineage["model_snapshot_id"] == "sha256:source"

    action_path = tmp_path / "actions.safetensors"
    action_report = run_flexmoe_action_session(
        session,
        ranking_path=ranking_path,
        output_path=action_path,
        action_ratios=(0.5, 1.0),
        spec=FlexMoEActionLearningSpec(
            total_steps=2,
            temperature_start=1.0,
            temperature_end=0.5,
            cost_weight_start=0.0,
            cost_weight_end=0.2,
            entropy_weight_start=0.1,
            entropy_weight_end=0.0,
        ),
        learning_rate=0.05,
        thickest_logit_margin=1.0,
        teacher_loss_weight=1.0,
        seed=17,
    )
    assert action_report.steps == 2
    assert 0.0 <= action_report.global_prune_budget < 1.0
    actions, action_lineage = load_action_plans(action_path)
    assert set(actions) == {"experts"}
    assert action_lineage["model_snapshot_id"] == "sha256:source"
    assert action_lineage["ranking_snapshot_id"].startswith("sha256:")
