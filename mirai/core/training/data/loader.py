"""Training-data loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.dataset.cache import (
    indexed_cache_metadata_path,
    indexed_cache_tensor_path,
    load_cache,
    load_indexed_cache_metadata,
    load_indexed_cache_records,
)
from mirai.core.training.data.schema import (
    BatchSchema,
    resolve_training_batch_schema,
    validate_records_against_schema,
)
from mirai.core.training.data.online import build_temporal_groups


@dataclass(frozen=True)
class TrainingRecordGroups:
    all_records: list[dict[str, Any]]
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]


@dataclass(frozen=True)
class PreparedTrainingData:
    all_records: list[dict[str, Any]]
    rank_records: list[dict[str, Any]]
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]
    temporal_base_ids: list[str]
    temporal_groups: dict[str, list[dict[str, Any]]]
    cache_data_access_mode: str
    indexed_cache_enabled: bool
    indexed_cache_metadata_path: str
    indexed_cache_tensor_path: str


def validate_training_records(
    records: list[dict[str, Any]],
    *,
    model_type: str,
    strategy_type: str,
    val_every_n_steps: int,
    batch_schema: BatchSchema | None = None,
) -> TrainingRecordGroups:
    all_records = list(records)
    if not all_records:
        raise ValueError("Cache has no records.")

    train_records = [r for r in all_records if str(r.get("split", "train")) == "train"]
    val_records = [r for r in all_records if str(r.get("split", "train")) == "val"]
    if not train_records:
        raise ValueError("Cache has no train records.")

    wants_validation = int(val_every_n_steps) > 0
    if wants_validation and not val_records:
        raise ValueError(
            "validation is enabled (training.val_every_n_steps > 0) but cache has no val records."
        )

    schema = batch_schema or resolve_training_batch_schema(
        model_type=model_type,
        strategy_type=strategy_type,
    )
    validate_records_against_schema(train_records, schema=schema, split_label="train")
    if wants_validation:
        validate_records_against_schema(val_records, schema=schema, split_label="val")
    return TrainingRecordGroups(
        all_records=all_records,
        train_records=train_records,
        val_records=val_records,
    )


def _normalized_cache_metadata(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v).strip() for k, v in value.items() if str(v).strip()}


def _validate_cache_lineage_metadata(
    *,
    observed_metadata: dict[str, str],
    dataset_snapshot_id: str,
    model_type: str,
    model_variant: str,
    model_component_id: str,
) -> None:
    observed_dataset_snapshot_id = str(observed_metadata.get("dataset_snapshot_id", "")).strip()
    expected_dataset_snapshot_id = str(dataset_snapshot_id).strip()
    if (
        expected_dataset_snapshot_id
        and observed_dataset_snapshot_id
        and observed_dataset_snapshot_id != expected_dataset_snapshot_id
    ):
        raise ValueError(
            "Cache dataset lineage mismatch: "
            f"cache was built from '{observed_dataset_snapshot_id}' but current dataset resolves to "
            f"'{expected_dataset_snapshot_id}'. Rebuild cache or point training at the matching dataset snapshot."
        )

    expected_model_type = str(model_type).strip().lower()
    observed_model_type = str(observed_metadata.get("model_type", "")).strip().lower()
    if expected_model_type and observed_model_type and observed_model_type != expected_model_type:
        raise ValueError(
            "Cache model lineage mismatch: "
            f"cache was built for model_type '{observed_model_type}' but current run expects "
            f"'{expected_model_type}'. Rebuild cache for the selected model family."
        )

    expected_model_variant = str(model_variant).strip().lower()
    observed_model_variant = str(observed_metadata.get("model_variant", "")).strip().lower()
    if (
        expected_model_variant
        and observed_model_variant
        and observed_model_variant != expected_model_variant
    ):
        raise ValueError(
            "Cache model variant lineage mismatch: "
            f"cache was built for model_variant '{observed_model_variant}' but current run expects "
            f"'{expected_model_variant}'. Rebuild cache for the selected model variant."
        )

    expected_model_component_id = str(model_component_id).strip()
    observed_model_component_id = str(observed_metadata.get("model_component_id", "")).strip()
    if (
        expected_model_component_id
        and observed_model_component_id
        and observed_model_component_id != expected_model_component_id
    ):
        raise ValueError(
            "Cache model component lineage mismatch: "
            f"cache was built for model_component_id '{observed_model_component_id}' but current run expects "
            f"'{expected_model_component_id}'. Rebuild cache for the selected model component."
        )


def load_prepared_training_data(
    *,
    cache_path: str | Path,
    model_type: str,
    strategy_type: str,
    val_every_n_steps: int,
    dataset_snapshot_id: str = "",
    model_variant: str = "",
    model_component_id: str = "",
    batch_schema: BatchSchema | None = None,
) -> PreparedTrainingData:
    resolved_cache_path = Path(cache_path)
    if not resolved_cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {resolved_cache_path}. Run scripts/cache.py first."
        )
    indexed_records = load_indexed_cache_records(resolved_cache_path)
    cache_data_access_mode = "full_cache"
    indexed_cache_enabled = False
    metadata_path = ""
    tensor_path = ""
    observed_metadata: dict[str, str] = {}
    if indexed_records is not None:
        all_records = list(indexed_records)
        cache_data_access_mode = "indexed_safetensors"
        indexed_cache_enabled = True
        metadata_path = str(indexed_cache_metadata_path(resolved_cache_path))
        tensor_path = str(indexed_cache_tensor_path(resolved_cache_path))
        observed_metadata = _normalized_cache_metadata(
            load_indexed_cache_metadata(resolved_cache_path)
        )
    else:
        cache = load_cache(resolved_cache_path)
        all_records = list(cache.get("records", []))
        observed_metadata = _normalized_cache_metadata(cache)
    _validate_cache_lineage_metadata(
        observed_metadata=observed_metadata,
        dataset_snapshot_id=dataset_snapshot_id,
        model_type=model_type,
        model_variant=model_variant,
        model_component_id=model_component_id,
    )
    global_record_groups = validate_training_records(
        all_records,
        model_type=model_type,
        strategy_type=strategy_type,
        val_every_n_steps=val_every_n_steps,
        batch_schema=batch_schema,
    )
    rank_records = all_records
    record_groups = global_record_groups
    temporal_base_ids, temporal_groups = build_temporal_groups(record_groups.train_records)
    return PreparedTrainingData(
        all_records=all_records,
        rank_records=rank_records,
        train_records=record_groups.train_records,
        val_records=record_groups.val_records,
        temporal_base_ids=temporal_base_ids,
        temporal_groups=temporal_groups,
        cache_data_access_mode=cache_data_access_mode,
        indexed_cache_enabled=bool(indexed_cache_enabled),
        indexed_cache_metadata_path=metadata_path,
        indexed_cache_tensor_path=tensor_path,
    )
