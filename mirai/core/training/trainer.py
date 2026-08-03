"""Model-agnostic sparse-MoE trainer."""

from __future__ import annotations

import inspect
from typing import Any

from mirai.config.runtime_policy import validate_native_backend_availability
from mirai.config.schema import TrainingConfig
from mirai.core.models.providers import (
    get_model_family_provider,
    load_configured_model_provider_module,
)
from mirai.core.registry import (
    LossRegistry,
    ObjectiveRegistry,
    StrategyRegistry,
)
from mirai.core.tensors import to_jsonable, to_list, to_python_scalar
from mirai.core.training.objectives.engine import (
    TrainingLossResult,
    compute_training_loss,
    evaluate_prepared_task_loss,
)
from mirai.core.training.objectives.flow_loss import apply_noise_offset
from mirai.core.training.residency.memory_safety import (
    assert_system_memory_headroom,
    configure_memory_safety,
)
from mirai.core.training.residency.activation_offload import (
    ActivationOffloadPolicy,
    prepare_activation_offload,
)
from mirai.core.training.residency.activation_compression import (
    activation_compression_context,
)
from mirai.core.training.objectives.sampling import (
    NoiseGenerator,
    build_timestep_sampler,
)
from mirai.core.training.lifecycle.session_state import (
    capture_torch_rng_state,
    restore_torch_rng_state,
)
from mirai.core.training.runtime.trainer import (
    initialize_trainer_runtime,
    resolve_weight_residency_strategy,
)
from mirai.core.training.training_policy import TrainingPolicySet
from mirai.core.training.training_policy import load_training_policy_modules

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        configure_memory_safety(config.memory)
        assert_system_memory_headroom(context="model construction")
        load_configured_model_provider_module(config)
        load_training_policy_modules(
            getattr(config.training, "policy_modules", ())
        )
        provider = get_model_family_provider(config.model.type)
        if provider is None:
            raise ValueError(
                f"Model family '{config.model.type}' has no registered provider."
            )
        family_param_errors = provider.validate_family_params(
            dict(getattr(config.model.params, "family_params", {}) or {})
        )
        if family_param_errors:
            joined = "\n".join(f"- {message}" for message in family_param_errors)
            raise ValueError(
                "[trainer] Model-family parameter validation failed before "
                f"pipeline construction:\n{joined}"
            )
        validate_native_backend_availability(config, entrypoint="trainer")
        model_cls = provider.require_pipeline_type()
        strategy_cls = StrategyRegistry.get(config.strategy.type)
        self.loss_fn = LossRegistry.get(config.training.loss_function)
        self.objective = ObjectiveRegistry.get(config.training.objective)()
        if torch is None:
            self.objective.configure(config)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.training.seed) + 10_003)
                self.objective.configure(config)

        if torch is None:
            self.pipeline = _instantiate_model_pipeline(model_cls, config)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.training.seed))
                self.pipeline = _instantiate_model_pipeline(model_cls, config)
        self.training_policies = TrainingPolicySet.from_config(config)
        self.training_policies.configure_pipeline(self.pipeline)
        assert_system_memory_headroom(context="model construction")
        runtime_setup = initialize_trainer_runtime(
            config=self.config,
            pipeline=self.pipeline,
        )
        self.runtime_policy_notes = runtime_setup.runtime_policy_notes
        self.fp8_frozen_weights_enabled = runtime_setup.fp8_frozen_weights_enabled
        self.frozen_weight_quantization = runtime_setup.frozen_weight_quantization
        self.requested_weight_residency_strategy = (
            runtime_setup.requested_weight_residency_strategy
        )
        self.weight_residency_strategy = runtime_setup.weight_residency_strategy
        self.expert_weight_access = runtime_setup.expert_weight_access
        self.expert_dequant_chunk_size = runtime_setup.expert_dequant_chunk_size
        self.quantize_experts_on_load = runtime_setup.quantize_experts_on_load
        self.router_quantization = runtime_setup.router_quantization
        self.moe_kernel_backend = runtime_setup.moe_kernel_backend
        self.trainable_parameter_offload = runtime_setup.trainable_parameter_offload
        self.memory_feature_notes = runtime_setup.memory_feature_notes
        self._forward_fn = runtime_setup.forward_fn
        self.compile_enabled = runtime_setup.compile_enabled
        self.compile_warning = runtime_setup.compile_warning
        self._compilation_session = runtime_setup.compilation_session
        self._place_objective_trainables()
        self._activation_offload_session = prepare_activation_offload(
            pipeline=self.pipeline,
            policy=ActivationOffloadPolicy.from_training_config(
                self.config.training
            ),
        )
        self.strategy = strategy_cls(config.strategy)
        prepare_params = inspect.signature(self.strategy.prepare_inputs).parameters
        self._strategy_prepare_accepts_training = "training" in prepare_params
        self._strategy_prepare_accepts_objective = "objective" in prepare_params
        self.timestep_sampler = build_timestep_sampler(
            mode=config.training.timestep_sampling,
            seed=config.training.seed,
            eps=config.training.timestep_eps,
            mean=config.training.timestep_sampling_mean,
            std=config.training.timestep_sampling_std,
            mode_scale=config.training.timestep_sampling_mode_scale,
        )
        self.noise_generator = NoiseGenerator(seed=config.training.seed + 1)

        model_errors = self.pipeline.validate_config(config)
        strategy_errors = self.strategy.validate_config()
        strategy_param_errors = self.strategy.validate_params()
        all_errors = [*model_errors, *strategy_errors, *strategy_param_errors]
        if all_errors:
            joined = "\n".join(f"- {e}" for e in all_errors)
            raise ValueError(f"Config validation failed:\n{joined}")

    @staticmethod
    def _resolve_weight_residency_strategy(config: TrainingConfig) -> str:
        return resolve_weight_residency_strategy(config)

    def _validate_input_batch(self, batch: dict[str, Any]) -> None:
        if torch is None:  # pragma: no cover
            return
        for field in ("latents", "text_embeds"):
            value = batch.get(field)
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError(
                    f"Non-finite input tensor detected in batch field '{field}'."
                )

    def _sampling_state_dict(self) -> dict[str, Any]:
        return {
            "timestep_sampler": self.timestep_sampler.state_dict(),
            "noise_generator": self.noise_generator.state_dict(),
        }

    def _load_sampling_state_dict(self, state: dict[str, Any]) -> None:
        timestep_state = state.get("timestep_sampler")
        if isinstance(timestep_state, dict):
            self.timestep_sampler.load_state_dict(timestep_state)
        noise_state = state.get("noise_generator")
        if isinstance(noise_state, dict):
            self.noise_generator.load_state_dict(noise_state)

    def _pipeline_is_training(self) -> bool:
        if hasattr(self.pipeline, "training"):
            return bool(getattr(self.pipeline, "training"))
        return bool(getattr(self.pipeline, "_is_training", True))

    def begin_validation(self) -> dict[str, Any]:
        state = {
            "sampling": self._sampling_state_dict(),
            "torch_rng": capture_torch_rng_state(),
            "pipeline_was_training": self._pipeline_is_training(),
            "objective_was_training": self._objective_is_training(),
        }
        self.pipeline.eval()
        self.objective.train(False)
        return state

    def end_validation(self, state: dict[str, Any]) -> None:
        restore_torch_rng_state(state.get("torch_rng"))
        sampling_state = state.get("sampling")
        if isinstance(sampling_state, dict):
            self._load_sampling_state_dict(sampling_state)
        if bool(state.get("pipeline_was_training", True)):
            self.pipeline.train()
        else:
            self.pipeline.eval()
        self.objective.train(
            bool(state.get("objective_was_training", True))
        )

    def compute_validation_loss(self, batch: dict[str, Any]):
        result = self.compute_loss_result(
            batch,
            training=False,
            gradient_accumulation=1,
        )
        return result.loss_pre_accum

    def _predict(self, inputs: Any) -> Any:
        try:
            with self._activation_saved_tensor_context():
                return self._forward_fn(
                    noisy_latents=inputs.noisy_latents,
                    timesteps=inputs.timestep,
                    text_embeds=inputs.text_embeds,
                    **inputs.extra_forward_kwargs,
                )
        except Exception as exc:
            if self.compile_enabled:
                warning = (
                    "training.compile=true was disabled after a compiled-region "
                    f"runtime error: {exc}"
                )
                self._compilation_session.disable(warning=warning)
                self.compile_enabled = False
                self.compile_warning = warning
                self._forward_fn = self.pipeline.forward
                with self._activation_saved_tensor_context():
                    return self._forward_fn(
                        noisy_latents=inputs.noisy_latents,
                        timesteps=inputs.timestep,
                        text_embeds=inputs.text_embeds,
                        **inputs.extra_forward_kwargs,
                    )
            else:
                raise

    def prepare_calibration_inputs(
        self,
        batch: dict[str, Any],
        *,
        training: bool = False,
    ) -> Any:
        """Build one deterministic provider-owned forward input without a loss."""

        self._validate_input_batch(batch)
        self.training_policies.before_forward(
            self.pipeline,
            batch,
            training=bool(training),
        )
        kwargs: dict[str, Any] = {
            "batch": batch,
            "pipeline": self.pipeline,
            "timestep_sampler": self.timestep_sampler,
            "noise_generator": self.noise_generator,
        }
        if self._strategy_prepare_accepts_training:
            kwargs["training"] = bool(training)
        if self._strategy_prepare_accepts_objective:
            kwargs["objective"] = self.objective
        return self.strategy.prepare_inputs(**kwargs)

    def prepare_objective_calibration_inputs(
        self,
        batch: dict[str, Any],
        *,
        training: bool = False,
    ) -> Any:
        """Prepare one input with the configured objective-side noise offset."""

        inputs = self.prepare_calibration_inputs(batch, training=training)
        offset = float(self.config.training.noise_offset)
        if offset != 0.0:
            inputs.noise = apply_noise_offset(inputs.noise, offset)
            objective_timesteps = (
                inputs.timestep
                if inputs.objective_timestep is None
                else inputs.objective_timestep
            )
            inputs.noisy_latents = self.objective.corrupt(
                pipeline=self.pipeline,
                clean_latents=inputs.clean_latents,
                noise=inputs.noise,
                timesteps=objective_timesteps,
            )
        return inputs

    def predict_calibration_inputs(self, inputs: Any) -> Any:
        """Run a previously prepared input through the configured pipeline."""

        return self._predict(inputs)

    def predict_objective_calibration_inputs(
        self,
        inputs: Any,
        *,
        training: bool = False,
    ) -> Any:
        """Run the configured objective on one already prepared input."""

        return self.objective.predict(
            inputs=inputs,
            predict=self._predict,
            pipeline=self.pipeline,
            config=self.config,
            training=bool(training),
        )

    def evaluate_calibration_task_loss(
        self,
        *,
        batch: dict[str, Any],
        inputs: Any,
        prediction: Any,
    ) -> TrainingLossResult:
        """Evaluate the native task loss without resampling the prepared input."""

        return evaluate_prepared_task_loss(
            config=self.config,
            batch=batch,
            pipeline=self.pipeline,
            strategy=self.strategy,
            objective=self.objective,
            loss_fn=self.loss_fn,
            inputs=inputs,
            prediction=prediction,
        )

    def get_compilation_diagnostics(self) -> dict[str, Any]:
        """Return current compile regions, token buckets, and compiler counters."""
        return self._compilation_session.diagnostics()

    def compute_loss_result(
        self,
        batch: dict[str, Any],
        *,
        training: bool = True,
        gradient_accumulation: int | None = None,
    ) -> TrainingLossResult:
        self._validate_input_batch(batch)
        self.training_policies.before_forward(
            self.pipeline,
            batch,
            training=bool(training),
        )
        return compute_training_loss(
            config=self.config,
            batch=batch,
            pipeline=self.pipeline,
            strategy=self.strategy,
            objective=self.objective,
            loss_fn=self.loss_fn,
            timestep_sampler=self.timestep_sampler,
            noise_generator=self.noise_generator,
            predict=self._predict,
            strategy_prepare_accepts_training=self._strategy_prepare_accepts_training,
            strategy_prepare_accepts_objective=self._strategy_prepare_accepts_objective,
            training=training,
            gradient_accumulation=gradient_accumulation,
        )

    def compute_loss(
        self,
        batch: dict[str, Any],
        *,
        training: bool = True,
        gradient_accumulation: int | None = None,
    ):
        result = self.compute_loss_result(
            batch,
            training=training,
            gradient_accumulation=gradient_accumulation,
        )
        return result.loss, result.raw_metrics()

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        result = self.compute_loss_result(batch)
        return {
            "loss": to_python_scalar(result.loss),
            "loss_pre_accum": to_python_scalar(result.loss_pre_accum),
            "per_sample_loss": to_list(result.per_sample_loss),
            "loss_weights": to_list(result.loss_weights),
            "weighted_loss": to_list(result.weighted_loss),
            "timesteps": to_list(result.timesteps),
            "auxiliary_losses": to_jsonable(result.auxiliary_losses or {}),
            "diagnostics": to_jsonable(result.diagnostics or {}),
        }

    def dry_run(self) -> dict[str, Any]:
        batch_size = self.config.training.batch_size
        batch = {
            "latents": [0.1 + (0.1 * i) for i in range(batch_size)],
            "text_embeds": [1.0] * batch_size,
        }
        return self.train_step(batch)

    def _objective_is_training(self) -> bool:
        adaptive = getattr(self.objective, "_adaptive_weighting", None)
        return bool(getattr(adaptive, "training", True))

    def _place_objective_trainables(self) -> None:
        pipeline_parameters = list(self.pipeline.get_trainable_parameters())
        if not pipeline_parameters:
            return
        self.objective.to(device=pipeline_parameters[0].device)

    def get_named_trainable_parameters(self):
        pipeline_parameters = tuple(
            self.pipeline.get_named_trainable_parameters()
        )
        objective_parameters = tuple(
            (f"objective.{name}", parameter)
            for name, parameter in self.objective.get_named_trainable_parameters()
        )
        return (*pipeline_parameters, *objective_parameters)

    def get_trainable_parameters(self):
        return tuple(
            parameter
            for _name, parameter in self.get_named_trainable_parameters()
        )

    def state_dict(self) -> dict[str, Any]:
        state = {
            "pipeline": self.pipeline.state_dict(),
            "seed": self.config.training.seed,
            "timestep_sampler": self.timestep_sampler.state_dict(),
            "noise_generator": self.noise_generator.state_dict(),
        }
        if self.training_policies.active_names:
            state["training_policies"] = self.training_policies.state_dict()
        objective_state = self.objective.state_dict()
        if objective_state:
            state["objective"] = objective_state
        return state

    def _activation_saved_tensor_context(self):
        training = self.config.training
        if bool(training.activation_compression):
            return activation_compression_context(
                enabled=True,
                rank=int(training.activation_compression_rank),
                min_bytes=int(training.activation_compression_min_mib) * 1024**2,
                seed=int(training.seed),
            )
        return self._activation_offload_session.context()

    def get_activation_offload_diagnostics(self) -> dict[str, Any]:
        return self._activation_offload_session.diagnostics()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("Checkpoint trainer_state must be a mapping.")
        self.pipeline.load_state_dict(state.get("pipeline", {}))
        timestep_sampler_state = state.get("timestep_sampler")
        if isinstance(timestep_sampler_state, dict):
            self.timestep_sampler.load_state_dict(timestep_sampler_state)
        else:
            raise ValueError("Checkpoint trainer_state has no timestep sampler state.")
        noise_state = state.get("noise_generator")
        if isinstance(noise_state, dict):
            self.noise_generator.load_state_dict(noise_state)
        else:
            raise ValueError("Checkpoint trainer_state has no noise-generator state.")
        policy_state = state.get("training_policies")
        if isinstance(policy_state, dict):
            self.training_policies.load_state_dict(policy_state)
        elif self.training_policies.active_names:
            raise ValueError(
                "Checkpoint has no training policy state but the current run has "
                f"active policies: {list(self.training_policies.active_names)}."
            )
        objective_state = state.get("objective")
        if isinstance(objective_state, dict):
            self.objective.load_state_dict(objective_state)
        elif self.objective.state_dict():
            raise ValueError(
                "Checkpoint has no objective state but the current objective "
                "owns trainable state."
            )


def _instantiate_model_pipeline(model_cls: Any, config: TrainingConfig) -> Any:
    factory = getattr(model_cls, "from_training_config", None)
    if callable(factory):
        return factory(config)
    return model_cls(config.model)
