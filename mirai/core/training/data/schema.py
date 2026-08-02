"""Training-record batch schema authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchFieldRequirement:
    name: str
    aliases: tuple[str, ...] = ()
    reason: str = "training"
    guidance: str = ""

    def present_in(self, record: Any) -> bool:
        for key in (self.name, *self.aliases):
            if _record_has_non_empty_value(record, key):
                return True
        return False


@dataclass(frozen=True)
class BatchSchema:
    required_fields: tuple[BatchFieldRequirement, ...]

    def missing_fields(self, record: Any) -> list[BatchFieldRequirement]:
        return [field for field in self.required_fields if not field.present_in(record)]

    def requires(self, key: str) -> bool:
        normalized = str(key).strip()
        return any(field.name == normalized for field in self.required_fields)


def _record_has_non_empty_value(record: Any, key: str) -> bool:
    try:
        if key in record:
            value = record.get(key) if hasattr(record, "get") else record[key]
            if isinstance(value, str):
                return bool(value.strip())
            return value is not None
    except Exception:
        pass
    try:
        value = record.get(key)
    except Exception:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


_GUIDANCE_BY_KEY = {
    "clip_embed": (
        "Rebuild cache with strict native assets enabled and the strategy's "
        "conditioning checkpoints present."
    ),
}


def resolve_training_batch_schema(
    *,
    model_type: str,
    strategy_type: str,
    strategy: Any | None = None,
    pipeline: Any | None = None,
) -> BatchSchema:
    required = [
        BatchFieldRequirement("latent", aliases=("latent_key",)),
        BatchFieldRequirement("text_embed", aliases=("text_embed_key",)),
    ]
    for key in _required_extra_batch_keys(
        model_type=model_type,
        strategy_type=strategy_type,
        strategy=strategy,
        pipeline=pipeline,
    ):
        required.append(
            BatchFieldRequirement(
                str(key),
                aliases=(f"{key}_key",),
                reason=f"{strategy_type} strategy",
                guidance=_GUIDANCE_BY_KEY.get(str(key), ""),
            )
        )
    return BatchSchema(required_fields=tuple(required))


def _required_extra_batch_keys(
    *,
    model_type: str,
    strategy_type: str,
    strategy: Any | None,
    pipeline: Any | None,
) -> list[str]:
    """Resolve required cache keys from the model contract, not family names.

    Prefers a constructed pipeline; otherwise queries the registered model class's
    declared requirements; the strategy is a final fallback. Returns no keys if
    the model type is unknown.
    """

    if pipeline is not None:
        return [str(k) for k in pipeline.get_required_batch_keys(strategy_type=strategy_type)]

    model_cls = _registered_model_class(model_type)
    if model_cls is not None:
        declared = getattr(model_cls, "REQUIRED_BATCH_KEYS_BY_STRATEGY", {}) or {}
        return [str(k) for k in declared.get(str(strategy_type), [])]

    if strategy is not None:
        provider = getattr(strategy, "get_required_batch_keys", None)
        if callable(provider):
            return [str(k) for k in provider(model_type=model_type)]
    return []


def _registered_model_class(model_type: str) -> Any | None:
    from mirai.core.registry import ModelRegistry

    if not ModelRegistry.has(model_type):
        try:
            from mirai.core.builtins import register_builtin_components

            register_builtin_components()
        except Exception:
            return None
    return ModelRegistry.get(model_type) if ModelRegistry.has(model_type) else None


def validate_records_against_schema(
    records: list[Any],
    *,
    schema: BatchSchema,
    split_label: str,
) -> None:
    for record in records:
        missing = schema.missing_fields(record)
        if not missing:
            continue
        first = missing[0]
        if first.guidance:
            split_suffix = " validation" if str(split_label) == "val" else ""
            raise ValueError(
                f"Cache is missing {first.name} records required for {first.reason}"
                f"{split_suffix}. {first.guidance}"
            )
        sample_id = _sample_id(record)
        raise ValueError(
            f"Cache record '{sample_id}' in {split_label} split is missing "
            f"required field '{first.name}' for {first.reason}."
        )


def _sample_id(record: Any) -> str:
    try:
        value = record.get("sample_id", "")
    except Exception:
        value = ""
    text = str(value).strip()
    return text or "<unknown>"
