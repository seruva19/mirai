"""Model-family provider registry and extension contracts."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import TYPE_CHECKING, Any

from mirai.core.registry import Registry
from mirai.core.moe.runtime.specs import ExpertMLPExecutionSpec

if TYPE_CHECKING:
    from mirai.config.schema import TrainingConfig
    from mirai.core.moe.calibration.esft import ESFTCalibrationTarget
    from mirai.core.moe.calibration.flexmoe import FlexMoECalibrationTarget
    from mirai.core.moe.calibration.pruning import ExpertPruningCalibrationTarget
    from mirai.core.moe.calibration.projection import PrototypeCalibrationTarget
    from mirai.core.moe.calibration.quantization import QuantizationCalibrationTarget
    from mirai.core.moe.calibration.router_quantization import (
        RouterQuantizationCalibrationTarget,
    )
    from mirai.core.moe.calibration.router_repair import RouterRepairTarget
    from mirai.core.moe.calibration.whitening import ExpertWhiteningCalibrationTarget
    from mirai.core.moe.calibration.imatrix import ExpertImportanceCalibrationTarget
    from mirai.core.moe.monitoring.agreement import RoutingSelectionTarget


@dataclass(frozen=True)
class NativeCacheEncoderConfig:
    enabled: bool
    model_type: str
    variant: str
    model_path: str
    dtype_name: str
    max_frames: int
    denoiser_subfolder: str = "transformer"
    strict_assets: bool = False
    text_encoder_path: str = ""
    enable_bucketing: bool = False
    resolution_buckets: list[tuple[int, int]] | None = None
    frame_buckets: list[int] | None = None
    bucket_resize_mode: str = "resize_crop"


@dataclass(frozen=True)
class FamilyGenerationDefaults:
    """Sampling values a model family was released with.

    Every field is ``None`` when the family declares nothing for it. That is a
    different state from a family that intends an empty unconditional: a
    generic caller applies a declared value only when the corresponding request
    field was omitted, so "omitted" and "explicitly empty" never collapse into
    the same request. A family that declares nothing leaves the caller's own
    fallback in place and never acquires an empty-string default.
    """

    negative_prompt: str | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    scheduler: str | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None

    def declares_negative_prompt(self) -> bool:
        return self.negative_prompt is not None

    def resolve_negative_prompt(self, requested: str | None) -> str:
        """Return the effective negative prompt for an omitted-or-explicit request."""
        if requested is not None:
            return str(requested)
        return str(self.negative_prompt or "")

    def resolve_steps(self, requested: int | None, *, fallback: int) -> int:
        if requested is not None:
            return int(requested)
        return int(self.steps if self.steps is not None else fallback)

    def resolve_cfg_scale(self, requested: float | None, *, fallback: float) -> float:
        if requested is not None:
            return float(requested)
        return float(self.cfg_scale if self.cfg_scale is not None else fallback)

    def resolve_scheduler(self, requested: str | None, *, fallback: str) -> str:
        if requested is not None and str(requested).strip():
            return str(requested)
        return str(self.scheduler or fallback)

    def resolve_width(self, requested: int | None, *, fallback: int) -> int:
        value = self.width if requested is None else requested
        return int(fallback if value is None else value)

    def resolve_height(self, requested: int | None, *, fallback: int) -> int:
        value = self.height if requested is None else requested
        return int(fallback if value is None else value)

    def resolve_frames(self, requested: int | None, *, fallback: int) -> int:
        value = self.frames if requested is None else requested
        return int(fallback if value is None else value)

    def empty_negative_prompt_warning(self, resolved: str, *, model_type: str) -> str | None:
        """Message for a run that discards a declared family negative prompt.

        Returns ``None`` for every run that is not degraded, so an ordinary
        correct run -- one that omitted the negative prompt and received the
        family default -- never produces it.
        """
        if not self.declares_negative_prompt() or str(resolved).strip():
            return None
        return (
            f"model family '{model_type}' declares a default negative prompt, but "
            "this run uses an empty one. Classifier-free guidance then steers away "
            "from an unconditional that carries none of the family's quality and "
            "artifact terms, which degrades predicted structure at high sigma and "
            "inflates latent magnitude. Omit the negative prompt to use the "
            "family default."
        )


@dataclass
class ModelFamilyProvider:
    """Provider-owned model-family contracts outside trainer core."""

    model_type: str
    native: bool = False
    sparse_moe: bool = False
    strict_native_assets_by_default: bool = False
    native_cache_encoding: bool = False
    asset_free_cache_encoding: bool = False
    runtime_backend_available: bool = True
    dora_supported: bool = False
    dataset_routing_policy: bool = False
    diversity_routing_policy: bool = False
    expert_dropout_policy: bool = False
    router_temperature_policy: bool = False
    layer_router_policy: bool = False
    progressive_sparsification: bool = False
    chain_of_experts: bool = False
    dispersive_loss_policy: bool = False
    simbal_policy: bool = False
    selective_sinkhorn_policy: bool = False
    prototypical_routing_policy: bool = False
    sharp_moe_policy: bool = False
    mixture_of_depths_policy: bool = False
    preemptive_monitoring: bool = False
    balance_loss_schedule: bool = False
    grouped_adjugate_experts: bool = False
    balance_gradient_ratio_telemetry: bool = False
    router_stage_schedule: bool = False
    router_distillation: bool = False
    domain_expert_specialization: bool = False
    esft_expert_selection: bool = False
    expert_pruning_calibration: bool = False
    flexmoe_calibration: bool = False
    prototype_calibration: bool = False
    moe_quantization_calibration: bool = False
    router_quantization_calibration: bool = False
    expert_whitening_calibration: bool = False
    expert_precision_calibration: bool = False
    routing_mode_agreement_evidence: bool = False
    post_compression_router_repair: bool = False
    config_defaults_name: str = ""
    # Release-evidence contract: whether this family may appear in release
    # evidence and whether it is eligible for release promotion. Family-specific rules live in
    # ``validate_release_native_smoke_evidence`` so the generic contract in
    # core/training never names a family.
    release_supported: bool = False
    release_eligible: bool = False
    inference_prompt_rewriters: tuple[str, ...] = ()
    batched_cfg_inference: bool = False
    inference_tasks: tuple[str, ...] = ("text_to_video",)
    dataset_caption_formats: tuple[str, ...] = ("raw",)
    expert_mlp_execution_spec: ExpertMLPExecutionSpec | None = None
    pipeline_type: type[Any] | None = None

    def require_pipeline_type(self) -> type[Any]:
        if self.pipeline_type is None:
            raise ValueError(
                f"Model family '{self.model_type}' registered no pipeline type."
            )
        return self.pipeline_type

    def is_native_model(self) -> bool:
        return bool(self.native)

    def is_sparse_moe_model(self) -> bool:
        return bool(self.sparse_moe)

    def is_release_supported(self) -> bool:
        return bool(self.release_supported)

    def is_release_eligible(self) -> bool:
        return bool(self.release_eligible)

    def validate_inference_prompt_rewriter(self, name: str) -> None:
        key = str(name or "none").strip().lower()
        if key == "none":
            return
        if key not in self.inference_prompt_rewriters:
            supported = ", ".join(self.inference_prompt_rewriters) or "none"
            raise ValueError(
                f"Model family '{self.model_type}' does not support inference prompt "
                f"rewriter '{key}'; supported: {supported}."
            )

    def supports_batched_cfg_inference(self) -> bool:
        return bool(self.batched_cfg_inference)

    def generation_defaults(self) -> FamilyGenerationDefaults:
        """Declare the sampling values this family was released with.

        The base contract declares nothing, so a family that never overrides
        this keeps every generic caller default untouched.
        """
        return FamilyGenerationDefaults()

    def supports_inference_task(self, task: str) -> bool:
        from mirai.core.inference.conditioning import normalize_inference_task

        return normalize_inference_task(task) in self.inference_tasks

    def validate_dataset_caption_format(self, name: str) -> None:
        key = str(name or "raw").strip().lower()
        if key not in self.dataset_caption_formats:
            supported = ", ".join(self.dataset_caption_formats)
            raise ValueError(
                f"Model family '{self.model_type}' does not support dataset "
                f"caption format '{key}'; supported: {supported}."
            )

    def resolve_dataset_caption(self, caption: str, *, caption_format: str) -> str:
        """Apply the family-owned caption contract before cache fingerprinting."""
        self.validate_dataset_caption_format(caption_format)
        return caption

    def supports_dora(self, cfg: "TrainingConfig | None" = None) -> bool:
        _ = cfg
        return bool(self.dora_supported)

    def validate_release_native_smoke_evidence(
        self,
        payload: dict[str, Any],
        *,
        runtime_policy: dict[str, Any],
        label: str = "native_smoke_report",
    ) -> list[str]:
        """Family-owned native-smoke release-evidence rules.

        The generic release-evidence contract calls this after the shared
        native-smoke checks; families override it to enforce their own snapshot
        verification requirements. Base families add no extra rules.
        """
        _ = (payload, runtime_policy, label)
        return []

    def should_force_strict_native_assets(self, cfg: "TrainingConfig") -> bool:
        _ = cfg
        return bool(self.native and self.strict_native_assets_by_default)

    def apply_runtime_policy(
        self,
        cfg: "TrainingConfig",
        *,
        entrypoint: str,
    ) -> list[str]:
        notes: list[str] = []
        if self.should_force_strict_native_assets(cfg) and not bool(
            cfg.model.params.strict_native_assets
        ):
            cfg.model.params.strict_native_assets = True
            notes.append(
                f"[{entrypoint}] forced model.params.strict_native_assets=true "
                f"for '{str(cfg.model.type).strip().lower()}'."
            )
        return notes

    def validate_runtime_compatibility(self, cfg: "TrainingConfig") -> list[str]:
        family_errors = self.validate_family_params(
            dict(getattr(cfg.model.params, "family_params", {}) or {})
        )
        if family_errors:
            return list(family_errors)
        caption_format = str(getattr(cfg.dataset, "caption_format", "raw"))
        try:
            self.validate_dataset_caption_format(caption_format)
        except ValueError as exc:
            return [str(exc)]
        inference_task = str(getattr(cfg.inference, "task", "text_to_video"))
        if not self.supports_inference_task(inference_task):
            supported = ", ".join(self.inference_tasks)
            return [
                f"Model family '{self.model_type}' does not support inference.task="
                f"'{inference_task}'; supported: {supported}."
            ]
        if (
            getattr(cfg.model.params, "moe_layer_router_policy", ())
            and not self.supports_layer_router_policy(cfg)
        ):
            return [
                f"Model family '{self.model_type}' does not support "
                "model.params.moe_layer_router_policy."
            ]
        if (
            getattr(
                cfg.model.params,
                "moe_progressive_sparsification_policy",
                (),
            )
            and not self.supports_progressive_sparsification(cfg)
        ):
            return [
                f"Model family '{self.model_type}' does not support "
                "model.params.moe_progressive_sparsification_policy."
            ]
        if (
            bool(getattr(cfg.model.params, "moe_chain_of_experts", False))
            and not self.supports_chain_of_experts(cfg)
        ):
            return [
                f"Model family '{self.model_type}' does not support "
                "model.params.moe_chain_of_experts."
            ]
        if (
            int(
                getattr(
                    cfg.model.params,
                    "moe_balance_loss_disable_step",
                    0,
                )
            )
            > 0
            and not self.supports_balance_loss_schedule(cfg)
        ):
            return [
                f"Model family '{self.model_type}' does not support "
                "model.params.moe_balance_loss_disable_step."
            ]
        return []

    def validate_family_params(self, params: dict[str, Any]) -> list[str]:
        """Validate the provider-owned ``model.params.family_params`` table."""
        if params:
            return [
                f"Model family '{self.model_type}' does not declare provider-owned "
                "model.params.family_params."
            ]
        return []

    def validate_native_backend_availability(self, cfg: "TrainingConfig") -> list[str]:
        _ = cfg
        return []

    def has_available_runtime_backend(self, cfg: "TrainingConfig | None" = None) -> bool:
        _ = cfg
        return bool(self.runtime_backend_available)

    def supports_native_cache_encoding(self, cfg: "TrainingConfig | None" = None) -> bool:
        _ = cfg
        return bool(self.native_cache_encoding)

    def supports_dataset_routing_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.dataset_routing_policy)

    def supports_diversity_routing_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.diversity_routing_policy)

    def supports_expert_dropout_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.expert_dropout_policy)

    def supports_router_temperature_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.router_temperature_policy)

    def supports_layer_router_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.layer_router_policy)

    def supports_progressive_sparsification(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.progressive_sparsification)

    def supports_chain_of_experts(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.chain_of_experts)

    def supports_dispersive_loss_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.dispersive_loss_policy)

    def supports_simbal_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.simbal_policy)

    def supports_selective_sinkhorn_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.selective_sinkhorn_policy)

    def supports_prototypical_routing_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.prototypical_routing_policy)

    def supports_sharp_moe_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.sharp_moe_policy)

    def supports_mixture_of_depths_policy(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.mixture_of_depths_policy)

    def supports_preemptive_monitoring(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.preemptive_monitoring)

    def supports_balance_loss_schedule(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.balance_loss_schedule)

    def supports_balance_gradient_ratio_telemetry(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.balance_gradient_ratio_telemetry)

    def supports_router_stage_schedule(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.router_stage_schedule)

    def supports_router_distillation(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.router_distillation)

    def supports_domain_expert_specialization(self, cfg: "TrainingConfig | None" = None) -> bool:
        return bool(self.domain_expert_specialization)

    def supports_esft_expert_selection(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.esft_expert_selection)

    def build_esft_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, "ESFTCalibrationTarget"]:
        _ = pipeline
        return {}

    def supports_prototype_calibration(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.prototype_calibration)

    def build_prototype_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, "PrototypeCalibrationTarget"]:
        _ = pipeline
        return {}

    def supports_expert_pruning_calibration(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.expert_pruning_calibration)

    def build_expert_pruning_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, "ExpertPruningCalibrationTarget"]:
        _ = pipeline
        return {}

    def supports_flexmoe_calibration(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.flexmoe_calibration)

    def build_flexmoe_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, "FlexMoECalibrationTarget"]:
        _ = pipeline
        return {}

    def supports_moe_quantization_calibration(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.moe_quantization_calibration)

    def build_moe_quantization_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, "QuantizationCalibrationTarget"]:
        _ = pipeline
        return {}

    def supports_router_quantization_calibration(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.router_quantization_calibration)

    def build_router_quantization_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, "RouterQuantizationCalibrationTarget"]:
        _ = pipeline
        return {}

    def supports_expert_whitening_calibration(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.expert_whitening_calibration)

    def build_expert_whitening_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, "ExpertWhiteningCalibrationTarget"]:
        _ = pipeline
        return {}

    def supports_expert_precision_calibration(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.expert_precision_calibration)

    def build_expert_precision_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, "ExpertImportanceCalibrationTarget"]:
        _ = pipeline
        return {}

    def supports_routing_mode_agreement_evidence(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.routing_mode_agreement_evidence)

    def build_routing_mode_agreement_targets(
        self,
        pipeline: Any,
    ) -> dict[str, "RoutingSelectionTarget"]:
        _ = pipeline
        return {}

    def supports_post_compression_router_repair(
        self, cfg: "TrainingConfig | None" = None
    ) -> bool:
        _ = cfg
        return bool(self.post_compression_router_repair)

    def build_router_repair_targets(
        self,
        pipeline: Any,
    ) -> dict[str, "RouterRepairTarget"]:
        _ = pipeline
        return {}

    def allows_asset_free_cache_encoding(self, cfg: "TrainingConfig | None" = None) -> bool:
        _ = cfg
        return bool(self.asset_free_cache_encoding)

    def build_native_cache_encoder(
        self,
        config: NativeCacheEncoderConfig,
    ) -> Any | None:
        _ = config
        return None


ModelFamilyProviderRegistry: Registry[ModelFamilyProvider] = Registry(
    "model family provider"
)
_BUILTINS_LOADED = False


def register_model_family_provider(
    model_type: str,
    provider: ModelFamilyProvider,
) -> ModelFamilyProvider:
    return ModelFamilyProviderRegistry.register(model_type, provider)


def load_model_provider_module(module_name: str) -> None:
    text = str(module_name or "").strip()
    if not text:
        return
    importlib.import_module(text)


def load_configured_model_provider_module(cfg: Any) -> None:
    model = getattr(cfg, "model", None)
    load_model_provider_module(str(getattr(model, "provider_module", "") or ""))


def ensure_builtin_model_family_providers_registered() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from mirai.core.builtins import register_builtin_components

    register_builtin_components()


def get_model_family_provider(
    model_type: str,
    *,
    load_builtins: bool = True,
) -> ModelFamilyProvider | None:
    key = str(model_type).strip().lower()
    if ModelFamilyProviderRegistry.has(key):
        return ModelFamilyProviderRegistry.get(key)
    if load_builtins:
        ensure_builtin_model_family_providers_registered()
    return (
        ModelFamilyProviderRegistry.get(key)
        if ModelFamilyProviderRegistry.has(key)
        else None
    )


def release_supported_model_types() -> list[str]:
    ensure_builtin_model_family_providers_registered()
    return sorted(
        name
        for name, provider in ModelFamilyProviderRegistry.items()
        if provider.is_release_supported()
    )


def is_registered_native_model_type(model_type: str) -> bool:
    provider = get_model_family_provider(model_type)
    return bool(provider is not None and provider.is_native_model())


def is_registered_sparse_moe_model_type(model_type: str) -> bool:
    provider = get_model_family_provider(model_type)
    return bool(provider is not None and provider.is_sparse_moe_model())


def model_supports_native_cache_encoding(
    model_type: str,
    cfg: "TrainingConfig | None" = None,
) -> bool:
    provider = get_model_family_provider(model_type)
    if provider is None:
        return False
    return bool(provider.supports_native_cache_encoding(cfg))


def model_allows_asset_free_cache_encoding(
    model_type: str,
    cfg: "TrainingConfig | None" = None,
) -> bool:
    provider = get_model_family_provider(model_type)
    if provider is None:
        return False
    return bool(provider.allows_asset_free_cache_encoding(cfg))


def resolve_family_generation_defaults(model_type: str) -> FamilyGenerationDefaults:
    """Family sampling defaults for a model type, empty when none are declared."""
    provider = get_model_family_provider(model_type)
    if provider is None:
        return FamilyGenerationDefaults()
    return provider.generation_defaults()


def build_native_cache_encoder(config: NativeCacheEncoderConfig) -> Any | None:
    provider = get_model_family_provider(config.model_type)
    if provider is None:
        return None
    return provider.build_native_cache_encoder(config)
