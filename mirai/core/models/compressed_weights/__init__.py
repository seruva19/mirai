"""Compressed frozen-weight storage and expert execution."""

from __future__ import annotations

from . import prepare
from .execution import expert_gather, linear, experts
from .quantization import blockwise_fp8, gguf_quant, learned_rotation, microscaling_quant, quant
from .quantization import structured_sparse_provider
from .packed import packed_residency, packed_state

from .quantization.gguf_quant import (
    BITS_PER_WEIGHT,
    GGUF_FORMATS,
    GGUF_TYPE_SIZE,
    _GgufMeta,
    dequantize_gguf,
    gguf_stored_bytes,
    nf4_stored_bytes,
    normalize_gguf_format,
    quantize_gguf,
)

from .quantization.quant import (
    DEFAULT_GROUP_SIZES,
    DEFAULT_QUANTIZATION_WORKSPACE_BYTES,
    CompressedWeightReport,
    NF4_BLOCKSIZE,
    QUANT_FORMATS,
    best_group_size,
    normalize_compressed_weights_strategy,
    normalize_quant_format,
    parse_group_sizes,
)
from .execution.linear import (
    CompressedLinear,
)
from .quantization.microscaling_quant import (
    MICROSCALING_FORMATS,
    MicroscalingMeta,
    dequantize_microscaling,
    microscaling_stored_bytes,
    normalize_microscaling_format,
    quantize_microscaling,
)
from .quantization.blockwise_fp8 import (
    BLOCKWISE_FP8_BLOCK_SIZE,
    BLOCKWISE_FP8_FORMATS,
    BlockwiseFP8Meta,
    blockwise_fp8_batched_linear,
    blockwise_fp8_linear,
    dequantize_blockwise_fp8_weight,
    quantize_blockwise_fp8_weight,
)
from .quantization.learned_rotation import (
    LEARNED_EXPERT_ROTATION_NAME,
    LEARNED_EXPERT_ROTATION_SCHEMA_VERSION,
    LearnedRotationResult,
    expert_weight_fingerprint,
    learn_groupwise_expert_rotation,
    validate_learned_rotation_selection,
)
from .execution.active_expert_lora import ActiveExpertLoRA
from .execution.experts import CompressedGroupedExperts
from .prepare import (
    apply_structured_2_4_experts,
    combine_compressed_weights_reports,
    prepare_compressed_weights_modules_for_checkpoint_load,
    quantize_compressed_weights_modules,
)
from .execution.expert_gather import BatchedExpertGatherStrategy
from .packed.packed_residency import LazyPackedTensorMapping
from .packed.packed_residency import PackedStateResidencyPolicy
from .packed.packed_residency import PreloadedPackedTensorMapping
from .packed.packed_residency import materialize_packed_tensors
from .factorization.shared_basis import (
    SHARED_BASIS_PROVIDER_NAME,
    SHARED_BASIS_PROVIDER_SCHEMA_VERSION,
    SharedBasisFactors,
    SharedBasisPhysicalWeightProvider,
    factorize_dense_experts,
)
from .factorization.mixture_basis import (
    MIXTURE_BASIS_ACTIVATIONS,
    MIXTURE_BASIS_PROJECTIONS,
    MIXTURE_BASIS_PROVIDER_NAME,
    MIXTURE_BASIS_PROVIDER_SCHEMA_VERSION,
    MixtureBasisFactors,
    MixtureBasisPhysicalWeightProvider,
    factorize_mixture_basis_experts,
)
from .quantization.structured_sparse_provider import (
    SPARSE24_PROVIDER_NAME,
    SPARSE24_PROVIDER_SCHEMA_VERSION,
    PackedSparse24,
    Sparse24PhysicalWeightProvider,
    pack_sparse24,
)
from .flexmoe_nested import (
    FLEXMOE_NESTED_PROVIDER_NAME,
    FLEXMOE_NESTED_PROVIDER_SCHEMA_VERSION,
    FlexMoENestedPhysicalWeightProvider,
    transform_packed_state_flexmoe_nested,
)
from .packed.packed_state import (
    DEFAULT_PACKED_SHARD_BYTES,
    COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY,
    COMPRESSED_WEIGHT_PACKED_STATE_SCHEMA_VERSION,
    export_compressed_weights_packed_state,
    get_compressed_weights_packed_state_quant_formats,
    get_compressed_weights_physical_weight_providers,
    load_compressed_weights_packed_state,
    load_compressed_weights_packed_state_file,
    load_compressed_weights_packed_tensors,
    packed_artifact_fingerprint,
    prepare_compressed_weights_modules_from_manifest,
    read_compressed_weights_packed_state_manifest,
    save_compressed_weights_packed_state,
    save_compressed_weights_packed_tensors,
)

__all__ = [
    "apply_structured_2_4_experts",
    "ActiveExpertLoRA",
    "BITS_PER_WEIGHT",
    "BLOCKWISE_FP8_BLOCK_SIZE",
    "BLOCKWISE_FP8_FORMATS",
    "BlockwiseFP8Meta",
    "BatchedExpertGatherStrategy",
    "GGUF_FORMATS",
    "GGUF_TYPE_SIZE",
    "FLEXMOE_NESTED_PROVIDER_NAME",
    "FLEXMOE_NESTED_PROVIDER_SCHEMA_VERSION",
    "FlexMoENestedPhysicalWeightProvider",
    "DEFAULT_GROUP_SIZES",
    "DEFAULT_PACKED_SHARD_BYTES",
    "DEFAULT_QUANTIZATION_WORKSPACE_BYTES",
    "COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY",
    "COMPRESSED_WEIGHT_PACKED_STATE_SCHEMA_VERSION",
    "CompressedGroupedExperts",
    "CompressedLinear",
    "CompressedWeightReport",
    "NF4_BLOCKSIZE",
    "LazyPackedTensorMapping",
    "LEARNED_EXPERT_ROTATION_NAME",
    "LEARNED_EXPERT_ROTATION_SCHEMA_VERSION",
    "LearnedRotationResult",
    "PackedStateResidencyPolicy",
    "PreloadedPackedTensorMapping",
    "QUANT_FORMATS",
    "MICROSCALING_FORMATS",
    "MIXTURE_BASIS_ACTIVATIONS",
    "MIXTURE_BASIS_PROJECTIONS",
    "MIXTURE_BASIS_PROVIDER_NAME",
    "MIXTURE_BASIS_PROVIDER_SCHEMA_VERSION",
    "MicroscalingMeta",
    "MixtureBasisFactors",
    "MixtureBasisPhysicalWeightProvider",
    "best_group_size",
    "blockwise_fp8_batched_linear",
    "blockwise_fp8_linear",
    "combine_compressed_weights_reports",
    "dequantize_gguf",
    "dequantize_blockwise_fp8_weight",
    "dequantize_microscaling",
    "export_compressed_weights_packed_state",
    "expert_weight_fingerprint",
    "get_compressed_weights_packed_state_quant_formats",
    "get_compressed_weights_physical_weight_providers",
    "gguf_stored_bytes",
    "load_compressed_weights_packed_state",
    "load_compressed_weights_packed_state_file",
    "load_compressed_weights_packed_tensors",
    "learn_groupwise_expert_rotation",
    "materialize_packed_tensors",
    "packed_artifact_fingerprint",
    "nf4_stored_bytes",
    "microscaling_stored_bytes",
    "normalize_gguf_format",
    "normalize_compressed_weights_strategy",
    "normalize_quant_format",
    "normalize_microscaling_format",
    "quantize_gguf",
    "quantize_blockwise_fp8_weight",
    "quantize_microscaling",
    "parse_group_sizes",
    "prepare_compressed_weights_modules_for_checkpoint_load",
    "prepare_compressed_weights_modules_from_manifest",
    "quantize_compressed_weights_modules",
    "read_compressed_weights_packed_state_manifest",
    "save_compressed_weights_packed_state",
    "save_compressed_weights_packed_tensors",
    "validate_learned_rotation_selection",
    "SHARED_BASIS_PROVIDER_NAME",
    "SHARED_BASIS_PROVIDER_SCHEMA_VERSION",
    "SharedBasisFactors",
    "SharedBasisPhysicalWeightProvider",
    "SPARSE24_PROVIDER_NAME",
    "SPARSE24_PROVIDER_SCHEMA_VERSION",
    "PackedSparse24",
    "Sparse24PhysicalWeightProvider",
    "factorize_dense_experts",
    "factorize_mixture_basis_experts",
    "pack_sparse24",
    "transform_packed_state_flexmoe_nested",
]
