"""Pipeline abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mirai.core.moe.routing.contracts import SparseMoECapabilities
from mirai.core.moe.runtime.specs import ExpertTensorSpec
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy
from mirai.core.moe.runtime.specs import validate_expert_tensor_specs

if TYPE_CHECKING:
    from mirai.config.schema import TrainingConfig
    from mirai.core.moe.adaptation.dataset_routing import DatasetRoutingBatch
    from mirai.core.moe.adaptation.dataset_routing import DatasetRoutingPolicy
    from mirai.core.moe.adaptation.distillation import RouterDistillationController
    from mirai.core.moe.adaptation.diversity import DiversityAwareRoutingController
    from mirai.core.moe.adaptation.domain_specialization import DomainExpertSpecializationController
    from mirai.core.moe.adaptation.dropout import ExpertDropoutController
    from mirai.core.moe.adaptation.simbal import SimBalController
    from mirai.core.moe.adaptation.stage_schedule import RouterStageScheduleController
    from mirai.core.moe.adaptation.temperature import RouterTemperatureController
    from mirai.core.moe.monitoring.preemptive import PreemptiveAttentionMonitor
    from mirai.core.moe.routing.depth import MixtureOfDepthsSpec
    from mirai.core.moe.routing.prototypical import PrototypicalRoutingSpec
    from mirai.core.moe.routing.saliency import SharpMoESpec
    from mirai.core.moe.routing.selective_sinkhorn import SelectiveSinkhornController
    from mirai.core.moe.runtime.token_chunking import MoETokenChunkPolicy
    from mirai.core.training.policies.dispersive_loss import DispersiveLossController
    from mirai.core.training.runtime.compilation import (
        CompilationRegion,
        TokenBucketPlan,
    )


@dataclass(frozen=True)
class MemoryFeatureCapabilities:
    block_swap: bool = False
    quantized_frozen_weights: bool = False
    packed_frozen_weight_state: bool = False
    weight_residency_strategy: bool = False
    runtime_offload_flush: bool = False
    expert_tensor_specs: bool = False
    expert_weight_access_policy: bool = False
    quantize_experts_on_load: bool = False
    router_quantization_policy: bool = False
    moe_kernel_backend: bool = False
    trainable_parameter_offload: bool = False


@dataclass(frozen=True)
class ModelExtensionCapabilities:
    adapter_target_presets: bool = False
    adapter_runtime_controls: bool = False
    adapter_type_controls: bool = False
    adapter_merge_unmerge: bool = False
    rank_schedule_progress: bool = False
    adapter_allocation_policy: bool = False
    adapter_initialization: bool = False
    adapter_training_policy: bool = False
    preview_latent_decode: bool = False
    validation_inference: bool = False


class BasePipeline(ABC):
    """Model abstraction the trainer depends on."""

    @abstractmethod
    def apply_noise(
        self, clean_latents: Any, noise: Any, timesteps: Any
    ) -> Any:
        """Apply model-specific noise schedule."""

    @abstractmethod
    def compute_target(
        self, noise: Any, clean_latents: Any, timesteps: Any
    ) -> Any:
        """Compute model-specific training target."""

    @abstractmethod
    def forward(
        self,
        noisy_latents: Any,
        timesteps: Any,
        text_embeds: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Forward prediction."""

    @abstractmethod
    def validate_config(self, config: TrainingConfig) -> list[str]:
        """Model-specific validation."""

    @abstractmethod
    def get_trainable_parameters(self):
        """Return trainable parameter iterable."""

    def get_named_trainable_parameters(self):
        """Return trainable parameters with stable names."""
        return [
            (f"param_{idx}", param)
            for idx, param in enumerate(self.get_trainable_parameters())
        ]

    def get_adapter_target_presets(self) -> dict[str, list[str]]:
        """Return adapter target-module presets."""
        return {}

    #: Class-level declaration of extra cache keys (beyond latent/text) a model
    #: requires per strategy, e.g. ``{"image_to_video": ["clip_embed"]}``. This is
    #: the authoritative, model-local source so core batch-schema code can resolve
    #: requirements from the registered model class (no family-name branching),
    #: even without a constructed pipeline instance.
    REQUIRED_BATCH_KEYS_BY_STRATEGY: dict[str, list[str]] = {}

    def get_required_batch_keys(self, *, strategy_type: str = "") -> list[str]:
        """Cache keys this model requires (beyond latent/text) for a strategy."""
        return list(self.REQUIRED_BATCH_KEYS_BY_STRATEGY.get(str(strategy_type), []))

    def get_model_extension_capabilities(self) -> ModelExtensionCapabilities:
        """Return optional model-extension surfaces implemented by this pipeline."""
        return ModelExtensionCapabilities()

    def validate_adapter_artifact_lineage(
        self,
        *,
        dataset_snapshot_id: str,
        model_snapshot_id: str,
        config_snapshot_id: str,
    ) -> None:
        """Validate optional adapter-shaping artifacts before optimizer creation."""
        _ = dataset_snapshot_id, model_snapshot_id, config_snapshot_id

    def preview_extra_forward_kwargs(self) -> dict[str, Any]:
        """Extra forward kwargs required to run a preview/sample for this model.

        Lets preview code stay model-agnostic: e.g. an i2v model returns its
        image-conditioning tensors here instead of preview branching on family.
        """
        return {}

    def i2v_conditioning_forward_kwargs(self, *, condition_frame_indexes: Any) -> dict[str, Any]:
        """Forward kwargs derived from i2v frame-conditioning, if the model uses them.

        Keeps the image-to-video strategy model-agnostic: a model that consumes
        frame-conditioning indices returns them here instead of the strategy
        branching on model family. Default: none.
        """
        return {}

    def i2v_conditioning_frame_dim(self, *, latents: Any) -> int:
        """Tensor dimension that represents frames for first-frame conditioning."""
        _ = latents
        return 1

    def get_memory_feature_capabilities(self) -> MemoryFeatureCapabilities:
        """Return supported training-time memory features for this pipeline."""
        return MemoryFeatureCapabilities()

    def get_sparse_moe_capabilities(self) -> SparseMoECapabilities:
        """Return sparse-MoE capabilities for true routed expert denoisers."""
        return SparseMoECapabilities()

    def get_expert_tensor_specs(self) -> list[ExpertTensorSpec]:
        """Return model-owned concrete routed/shared expert tensor declarations."""
        return []

    def get_expert_tensor_spec_map(self) -> dict[str, ExpertTensorSpec]:
        """Return expert tensor declarations keyed by checkpoint/module name."""
        return {
            spec.name: spec
            for spec in validate_expert_tensor_specs(self.get_expert_tensor_specs())
        }

    def configure_moe_optimization_policy(self, policy: MoEOptimizationPolicy) -> None:
        """Apply generic MoE expert memory/runtime policy when supported."""
        if policy.requests_runtime_behavior():
            raise ValueError(
                f"{type(self).__name__} does not implement MoE optimization policy controls."
            )

    def _unsupported_training_policy(self, name: str) -> None:
        raise ValueError(
            f"{type(self).__name__} does not implement training policy '{name}'."
        )

    def configure_mixture_of_depths(self, policy: MixtureOfDepthsSpec) -> None:
        _ = policy
        self._unsupported_training_policy("mixture_of_depths")

    def configure_dispersive_loss(self, policy: DispersiveLossController) -> None:
        _ = policy
        self._unsupported_training_policy("dispersive_loss")

    def configure_simbal(self, policy: SimBalController) -> None:
        _ = policy
        self._unsupported_training_policy("simbal")

    def configure_preemptive_monitoring(self, policy: PreemptiveAttentionMonitor) -> None:
        _ = policy
        self._unsupported_training_policy("preemptive_monitoring")

    def configure_moe_token_chunking(self, policy: MoETokenChunkPolicy) -> None:
        _ = policy
        self._unsupported_training_policy("moe_token_chunking")

    def configure_domain_expert_specialization(
        self, policy: DomainExpertSpecializationController
    ) -> None:
        _ = policy
        self._unsupported_training_policy("domain_expert_specialization")

    def configure_router_distillation(self, policy: RouterDistillationController) -> None:
        _ = policy
        self._unsupported_training_policy("router_distillation")

    def configure_router_stage_schedule(
        self, policy: RouterStageScheduleController
    ) -> None:
        _ = policy
        self._unsupported_training_policy("router_stage_schedule")

    def configure_diversity_routing(self, policy: DiversityAwareRoutingController) -> None:
        _ = policy
        self._unsupported_training_policy("diversity_routing")

    def configure_expert_dropout(self, policy: ExpertDropoutController) -> None:
        _ = policy
        self._unsupported_training_policy("expert_dropout")

    def configure_router_temperature(self, policy: RouterTemperatureController) -> None:
        _ = policy
        self._unsupported_training_policy("router_temperature")

    def configure_selective_sinkhorn(self, policy: SelectiveSinkhornController) -> None:
        _ = policy
        self._unsupported_training_policy("selective_sinkhorn")

    def configure_prototypical_routing(self, policy: PrototypicalRoutingSpec) -> None:
        _ = policy
        self._unsupported_training_policy("prototypical_routing")

    def configure_sharp_moe(self, policy: SharpMoESpec) -> None:
        _ = policy
        self._unsupported_training_policy("sharp_moe")

    def configure_dataset_routing(self, policy: DatasetRoutingPolicy) -> None:
        _ = policy
        self._unsupported_training_policy("dataset_routing")

    def set_dataset_routing_context(self, context: DatasetRoutingBatch) -> None:
        _ = context
        self._unsupported_training_policy("dataset_routing_context")

    def get_training_auxiliary_losses(self) -> dict[str, Any]:
        """Return per-forward auxiliary training losses such as router balance."""
        return {}

    def get_training_diagnostics(self) -> dict[str, Any]:
        """Return per-forward model diagnostics such as router utilization."""
        return {}

    def uses_previous_clean_routing_guidance(self) -> bool:
        """Whether denoising must feed the preceding clean prediction to routing."""

        return False

    def take_balance_gradient_probe(self) -> Any | None:
        """Consume graph references for opt-in balance/task diagnostics."""
        return None

    def supports_adapter_target_presets(self) -> bool:
        return bool(self.get_model_extension_capabilities().adapter_target_presets)

    def supports_adapter_runtime_controls(self) -> bool:
        return bool(self.get_model_extension_capabilities().adapter_runtime_controls)

    def supports_adapter_type_controls(self) -> bool:
        return bool(self.get_model_extension_capabilities().adapter_type_controls)

    def supports_adapter_merge_unmerge(self) -> bool:
        return bool(self.get_model_extension_capabilities().adapter_merge_unmerge)

    def supports_rank_schedule_progress(self) -> bool:
        return bool(self.get_model_extension_capabilities().rank_schedule_progress)

    def supports_balance_loss_schedule_progress(self) -> bool:
        return False

    def supports_preview_latent_decode(self) -> bool:
        return bool(self.get_model_extension_capabilities().preview_latent_decode)

    def supports_validation_inference(self) -> bool:
        return bool(self.get_model_extension_capabilities().validation_inference)

    def validate_refinement_request(
        self,
        request: dict[str, Any],
        *,
        frames: int,
    ) -> None:
        """Validate an optional family-owned post-denoise refinement stage."""
        _ = request, frames
        raise RuntimeError(
            f"{type(self).__name__} does not support an inference refinement stage."
        )

    def refine_inference_latent(
        self,
        *,
        base_latent: Any,
        request: dict[str, Any],
        prompt: str,
        negative_prompt: str,
        seed: int,
        device: str,
        dtype: Any | None,
    ) -> Any:
        """Run a validated family-owned refinement stage."""
        _ = base_latent, request, prompt, negative_prompt, seed, device, dtype
        raise RuntimeError(
            f"{type(self).__name__} does not support an inference refinement stage."
        )

    def supports_block_swap(self) -> bool:
        return bool(self.get_memory_feature_capabilities().block_swap)

    def supports_quantized_frozen_weights(self) -> bool:
        return bool(self.get_memory_feature_capabilities().quantized_frozen_weights)

    def supports_packed_frozen_weight_state(self) -> bool:
        return bool(self.get_memory_feature_capabilities().packed_frozen_weight_state)

    def supports_weight_residency_strategy(self) -> bool:
        return bool(self.get_memory_feature_capabilities().weight_residency_strategy)

    def supports_runtime_offload_flush(self) -> bool:
        return bool(self.get_memory_feature_capabilities().runtime_offload_flush)

    def supports_trainable_parameter_offload(self) -> bool:
        return bool(
            self.get_memory_feature_capabilities().trainable_parameter_offload
        )

    def get_block_swap_units(self) -> list[tuple[int, Any]]:
        """Return ordered runtime swap units for block-swap style residency."""
        return []

    def get_block_swap_unit_count(self) -> int:
        return int(len(self.get_block_swap_units()))

    def get_adapter_quantization_modules(self) -> list[Any]:
        """Return adapter wrappers that can quantize their frozen base."""
        return []

    def get_quantizable_frozen_linear_modules(self) -> list[tuple[str, Any]]:
        """Return standalone frozen linear modules eligible for quantization."""
        return []

    def train(self) -> None:
        """Switch to training behavior."""

    def eval(self) -> None:
        """Switch to eval behavior."""

    def set_lora_scale(self, scale: float) -> None:
        """Set inference-time adapter scale."""

    def get_lora_scale(self) -> float:
        """Get current adapter scale."""
        return 1.0

    def load_adapter_state(self, state: dict[str, Any]) -> None:
        """Load an adapter-only checkpoint payload."""
        _ = state
        raise ValueError(f"{type(self).__name__} does not support adapter state loading.")

    def set_gradient_checkpointing(self, enabled: bool | str) -> None:
        """Toggle gradient checkpointing behavior."""

    def set_adapter_config(self, adapter_config: Any) -> None:
        """Configure adapter rank/alpha/target preset when supported."""

    def get_adapter_calibration_root(self) -> Any:
        """Return the module that owns canonical adapter target names."""

        return self

    def record_gora_allocation(
        self, *, ranks: dict[str, int], fingerprint: str
    ) -> None:
        """Record finalized GoRA ranks after pre-optimizer calibration."""

        _ = ranks, fingerprint

    def get_gora_allocation_metadata(self) -> dict[str, Any]:
        """Return finalized GoRA allocation metadata when active."""

        return {}

    def prepare_model_timesteps(self, timesteps: Any, *, latents: Any) -> Any:
        """Map objective timesteps to model-conditioning timesteps.

        The default returns the original object unchanged. Providers with an
        explicit dynamic flow-shift capability may return the corresponding
        post-shift noise levels while the objective keeps the sampled
        timesteps for weighting and reproducibility.
        """
        _ = latents
        return timesteps

    def resolve_flow_shift_for_latent_shape(
        self,
        latent_shape: tuple[int, ...],
    ) -> float:
        """Resolve the scalar inference shift for one homogeneous latent batch."""
        _ = latent_shape
        model_config = getattr(self, "model_config", None)
        params = getattr(model_config, "params", None)
        return float(getattr(params, "flow_shift", 1.0))

    def set_block_swap(
        self,
        *,
        blocks_to_swap: int,
        mode: str,
        block_swap_backward: bool = True,
    ) -> None:
        """Configure block swapping behavior."""

    def set_adapter_type(self, adapter_type: str) -> None:
        """Configure adapter family (lora/loha/lokr) when supported."""

    def get_adapter_type(self) -> str:
        """Return active adapter family when supported."""
        return "lora"

    def set_adapter_runtime(
        self, *, rank_dropout: float, lora_parameter_dropout: float
    ) -> None:
        """Configure adapter runtime dropout behavior."""

    def set_rank_schedule_progress(
        self,
        *,
        step: int,
        start_step: int,
        end_step: int,
        min_scale: float,
    ) -> None:
        """Update rank-schedule progress when supported."""

    def set_balance_loss_schedule_progress(self, *, step: int) -> None:
        """Update auxiliary MoE balance-loss pressure when supported."""

    def merge_adapter(self) -> bool:
        """Merge adapter weights into the frozen base when supported."""
        return False

    def unmerge_adapter(self) -> bool:
        """Undo merge_adapter when supported."""
        return False

    def is_adapter_merged(self) -> bool:
        """Return whether the adapter is currently merged."""
        return False

    def load_vae_to_gpu(self) -> float:
        """Load preview-time VAE assets when supported."""
        return 0.18215

    def unload_vae_from_gpu(self) -> None:
        """Unload preview-time VAE assets when supported."""

    def get_vae_scaling_factor(self) -> float:
        """Return preview-time VAE scaling factor when supported."""
        return 0.18215

    def has_native_inference(self) -> bool:
        """Return True when real native inference assets are available."""
        return False

    def preview_latent_geometry(
        self, *, frame_count: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        """Return latent geometry (C, T_lat, H_lat, W_lat) for a pixel request.

        Used by the native denoise/preview loop to size the initial noise tensor
        in a model-agnostic way. Native pipelines override this with their VAE
        compression ratios and latent-channel count.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement preview_latent_geometry()."
        )

    def enable_fp8_frozen_weights(self, enabled: bool) -> None:
        """Enable or disable FP8 storage path for frozen weights."""

    def enable_quantized_frozen_weights(
        self,
        quant_type: str,
        **kwargs: Any,
    ) -> None:
        """Quantize frozen (non-adapter) weights with the given scheme.

        Parameters
        ----------
        quant_type : str
            Registry key: ``"fp8"``, ``"nf4"``, ``"int8"``, or ``"none"``.
        **kwargs
            Extra arguments for the wrapper (e.g. ``block_size`` for NF4).
        """

    def has_quantized_frozen_weights(self) -> bool:
        """Return whether frozen weights are stored in quantized form."""
        return False

    def set_compute_autocast_dtype(self, dtype: Any) -> None:
        """Record the intended mixed-precision compute dtype for forward passes."""
        self._compute_autocast_dtype = dtype

    def get_training_model(self) -> Any | None:
        """Return the module-like object that owns training weights, if any."""
        return getattr(self, "model", None)

    def get_compilation_regions(self) -> list["CompilationRegion"]:
        """Return repeated provider-owned callables safe for regional compilation."""
        return []

    def configure_compilation_token_buckets(
        self,
        plan: "TokenBucketPlan | None",
    ) -> None:
        """Install provider-owned token-axis bounds for regional compilation."""
        if plan is not None:
            raise ValueError(
                f"{type(self).__name__} does not support compile token buckets."
            )

    def place_offloaded_modules(self, *, device: Any, strategy: str) -> None:
        """Place modules for an offloaded training residency strategy."""
        _ = device, strategy

    def set_weight_residency_strategy(
        self,
        *,
        strategy: str,
        blocks_to_swap: int,
        mode: str,
        block_swap_backward: bool = True,
        offload_dir: str | None = None,
        block_residency_planner: str = "uniform",
        block_swap_prefetch_depth: int = 1,
        block_residency_priority: str = "index",
        block_swap_transfer_strategy: str = "per_tensor",
    ) -> None:
        """Configure frozen-weight residency behavior for training."""

    def flush_runtime_offloads(self) -> None:
        """Flush runtime swap/offload state at safe boundaries."""

    def finish_backward_offloads(self) -> None:
        """Release immutable weights after a completed backward pass."""

    def configure_trainable_parameter_offload(self, enabled: bool) -> None:
        """Configure host residency for trainable parameters between uses."""
        if enabled:
            raise ValueError(
                f"{type(self).__name__} does not support trainable parameter offload."
            )

    def prepare_optimizer_step(self) -> None:
        """Materialize trainable parameters for the configured optimizer."""

    def finish_optimizer_step(self) -> None:
        """Restore trainable parameter residency after optimizer execution."""

    def discard_optimizer_step(self) -> None:
        """Discard pipeline-owned gradients/state after a skipped update."""

    def get_block_swap_state(self) -> dict[str, Any]:
        """Return current block-swap/offload state when supported."""
        return {"enabled": False}

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Serializable state."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state."""
