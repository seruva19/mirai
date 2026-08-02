"""Batch sampling and gradient-inspection helpers for training loops."""

from __future__ import annotations

import math
import random
from typing import Any

from mirai.core.dataset.cache import IndexedCacheRecord
from mirai.core.dataset.record_batch import (
    pad_text_embeddings,
    stack_latent_tensors,
)
from mirai.core.training.data.online import (
    build_temporal_groups,
    online_caption_embed,
    pick_temporal_variant,
)
from mirai.core.training.data.curriculum import record_training_task
from mirai.core.training.training_policy import TrainingPolicySet

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _record_batch_values(records: list[Any], key: str) -> list[Any]:
    return IndexedCacheRecord.batch_get(records, key)


def _shuffled_epoch_indices(
    *,
    record_count: int,
    base_seed: int,
    epoch_index: int,
) -> list[int]:
    order = list(range(max(0, int(record_count))))
    rng = random.Random(int(base_seed) + (int(epoch_index) * 100_003))
    rng.shuffle(order)
    return order


def select_deterministic_batch_records(
    records: list[Any],
    *,
    batch_size: int,
    base_seed: int,
    global_batch_index: int,
) -> tuple[list[Any], int]:
    if not records:
        raise ValueError("Cannot sample from an empty record list.")
    effective_batch_size = max(1, int(batch_size))
    record_count = len(records)
    absolute_start = int(global_batch_index) * effective_batch_size
    chosen: list[Any] = []
    epoch_orders: dict[int, list[int]] = {}
    first_epoch = int(absolute_start // record_count)
    for offset in range(effective_batch_size):
        absolute_position = absolute_start + offset
        epoch_index = int(absolute_position // record_count)
        position_in_epoch = int(absolute_position % record_count)
        order = epoch_orders.get(epoch_index)
        if order is None:
            order = _shuffled_epoch_indices(
                record_count=record_count,
                base_seed=int(base_seed),
                epoch_index=epoch_index,
            )
            epoch_orders[epoch_index] = order
        chosen.append(records[order[position_in_epoch]])
    return chosen, first_epoch


def build_batch_from_records(
    records: list[Any],
    *,
    masked_loss: bool = False,
    step: int = 0,
    online_tag_shuffle: bool = False,
    online_tag_shuffle_dropout: float = 0.0,
    online_tag_shuffle_keep_first_n_tags: int = 1,
    base_seed: int = 0,
    training_policies: TrainingPolicySet | None = None,
    training: bool = True,
    global_batch_index: int | None = None,
) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch is required for batch sampling.")
    if not records:
        raise ValueError("Cannot build a batch from an empty record list.")
    latents = stack_latent_tensors(_record_batch_values(records, "latent"))
    base_text_values = _record_batch_values(records, "text_embed")
    text_values = [
        online_caption_embed(
            record=record,
            step=step,
            seed=base_seed,
            enabled=online_tag_shuffle,
            dropout_rate=online_tag_shuffle_dropout,
            keep_first_n_tags=online_tag_shuffle_keep_first_n_tags,
            base_value=base_text_values[idx],
        )
        for idx, record in enumerate(records)
    ]
    text_embeds = pad_text_embeddings(text_values)
    out: dict[str, Any] = {"latents": latents, "text_embeds": text_embeds, "_step": step}
    if all("clip_embed" in record and record.get("clip_embed") is not None for record in records):
        out["clip_embed"] = pad_text_embeddings(_record_batch_values(records, "clip_embed"))
    if all("source_loss_multiplier" in record for record in records):
        out["source_loss_multiplier"] = torch.tensor(
            [float(record["source_loss_multiplier"]) for record in records],
            dtype=torch.float32,
        )
    if all("source_type" in record for record in records):
        source_types = [str(record["source_type"]) for record in records]
        if len(set(source_types)) == 1:
            out["source_type"] = source_types[0]
    training_tasks = [record_training_task(record) for record in records]
    if training_tasks and all(training_tasks):
        unique_tasks = set(training_tasks)
        if len(unique_tasks) != 1:
            raise ValueError(
                "A training microbatch cannot mix multiple training_task values."
            )
        out["training_task"] = training_tasks[0]
    if masked_loss and all("loss_mask" in record for record in records):
        out["loss_mask"] = torch.tensor(
            [float(record.get("loss_mask", 1.0)) for record in records],
            dtype=torch.float32,
        )
    if training_policies is not None:
        out = training_policies.augment_batch(
            out,
            records=records,
            step=step,
            training=training,
            global_batch_index=global_batch_index,
        )
    return out


def sample_batch(
    records: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
    *,
    masked_loss: bool = False,
    step: int = 0,
    epoch_index: int = 0,
    online_tag_shuffle: bool = False,
    online_tag_shuffle_dropout: float = 0.0,
    online_tag_shuffle_keep_first_n_tags: int = 1,
    base_seed: int = 0,
    online_temporal_resampling: bool = False,
    temporal_base_ids: list[str] | None = None,
    temporal_groups: dict[str, list[dict[str, Any]]] | None = None,
    training_policies: TrainingPolicySet | None = None,
    global_batch_index: int | None = None,
) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch is required for batch sampling.")
    if online_temporal_resampling:
        local_base_ids = temporal_base_ids
        local_groups = temporal_groups
        if local_base_ids is None or local_groups is None:
            local_base_ids, local_groups = build_temporal_groups(records)
        chosen: list[dict[str, Any]] = []
        for sample_pos in range(batch_size):
            base_id = rng.choice(local_base_ids)
            chosen.append(
                pick_temporal_variant(
                    local_groups[base_id],
                    epoch=epoch_index,
                    seed=int(base_seed) + int(step),
                    sample_position=sample_pos,
                )
            )
    else:
        chosen = [rng.choice(records) for _ in range(batch_size)]
    return build_batch_from_records(
        chosen,
        masked_loss=masked_loss,
        step=step,
        online_tag_shuffle=online_tag_shuffle,
        online_tag_shuffle_dropout=online_tag_shuffle_dropout,
        online_tag_shuffle_keep_first_n_tags=online_tag_shuffle_keep_first_n_tags,
        base_seed=base_seed,
        training_policies=training_policies,
        training=True,
        global_batch_index=global_batch_index,
    )


def group_records_by_bucket(
    records: list[Any],
) -> dict[str, list[Any]]:
    """Partition *records* by their 'bucket_id' field.

    Records without a 'bucket_id' (bucketing disabled) are all placed under
    the empty-string key so the non-bucketing code path continues to work.
    """
    groups: dict[str, list[Any]] = {}
    for record in records:
        bid = str(record.get("bucket_id", "")) if hasattr(record, "get") else ""
        if bid not in groups:
            groups[bid] = []
        groups[bid].append(record)
    return groups


def sample_batch_bucketed(
    records: list[Any],
    batch_size: int,
    rng: random.Random,
    *,
    masked_loss: bool = False,
    step: int = 0,
    online_tag_shuffle: bool = False,
    online_tag_shuffle_dropout: float = 0.0,
    online_tag_shuffle_keep_first_n_tags: int = 1,
    base_seed: int = 0,
    bucket_groups: dict[str, list[Any]] | None = None,
    training_policies: TrainingPolicySet | None = None,
) -> dict[str, Any]:
    """Sample a batch where all records share the same bucket_id.

    When bucketing is disabled (all records have ``bucket_id=''``), records are
    sampled uniformly.

    If *bucket_groups* is provided it is used directly (avoids re-grouping on
    every call); otherwise groups are derived from *records*.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch is required for batch sampling.")

    groups = bucket_groups if bucket_groups is not None else group_records_by_bucket(records)

    # If all records share the empty-string bucket (bucketing off), fall back
    # to flat random sampling from the full pool.
    bucket_ids = [bid for bid in groups if bid != ""]
    if not bucket_ids:
        chosen = [rng.choice(records) for _ in range(batch_size)]
    else:
        # Pick a bucket weighted by record count, then sample within it.
        weights = {bid: float(len(groups[bid])) for bid in bucket_ids}
        total = sum(weights.values())
        r = rng.random() * total
        cumulative = 0.0
        chosen_bid = bucket_ids[-1]
        for bid in sorted(bucket_ids):
            cumulative += weights[bid]
            if r <= cumulative:
                chosen_bid = bid
                break
        pool = groups[chosen_bid]
        chosen = [rng.choice(pool) for _ in range(batch_size)]

    return build_batch_from_records(
        chosen,
        masked_loss=masked_loss,
        step=step,
        online_tag_shuffle=online_tag_shuffle,
        online_tag_shuffle_dropout=online_tag_shuffle_dropout,
        online_tag_shuffle_keep_first_n_tags=online_tag_shuffle_keep_first_n_tags,
        base_seed=base_seed,
        training_policies=training_policies,
    )


def grad_norm(params: list[torch.nn.Parameter]) -> float:
    grad_sq_sum = 0.0
    for param in params:
        if param.grad is not None:
            grad_sq_sum += float((param.grad.detach().float() ** 2).sum().item())
    return math.sqrt(grad_sq_sum)


def named_grad_norms(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, float]:
    out: dict[str, float] = {}
    grouped_sq = {
        "self_attn": 0.0,
        "cross_attn": 0.0,
        "ffn": 0.0,
        "temporal": 0.0,
        "other": 0.0,
    }
    dead_layers = 0

    def _group_for_name(name: str) -> str:
        key = name.lower()
        if "self_attn" in key or ("attn" in key and "cross" not in key and "temporal" not in key):
            return "self_attn"
        if "cross_attn" in key or ("cross" in key and "attn" in key):
            return "cross_attn"
        if "ffn" in key or "mlp" in key:
            return "ffn"
        if "temporal" in key:
            return "temporal"
        return "other"

    for name, param in named_params:
        if param.grad is None:
            out[f"grad_norm_{name}"] = 0.0
            dead_layers += 1
            continue
        norm = float(param.grad.detach().float().norm().item())
        out[f"grad_norm_{name}"] = norm
        if norm == 0.0:
            dead_layers += 1
        grp = _group_for_name(name)
        grouped_sq[grp] += norm * norm
    out["grad_norm_self_attn"] = math.sqrt(grouped_sq["self_attn"])
    out["grad_norm_cross_attn"] = math.sqrt(grouped_sq["cross_attn"])
    out["grad_norm_ffn"] = math.sqrt(grouped_sq["ffn"])
    out["grad_norm_temporal"] = math.sqrt(grouped_sq["temporal"])
    out["grad_norm_other"] = math.sqrt(grouped_sq["other"])
    out["dead_layer_count"] = float(dead_layers)
    return out


def evaluate_val_loss(
    *,
    trainer: Any,
    records: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
    masked_loss: bool,
) -> float | None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch is required for validation loss evaluation.")
    _ = rng
    val_records = [r for r in records if str(r.get("split", "train")) == "val"]
    if not val_records:
        return None
    total_loss = 0.0
    total_samples = 0
    effective_batch_size = max(1, int(batch_size))
    validation_state = None
    begin_validation = getattr(trainer, "begin_validation", None)
    end_validation = getattr(trainer, "end_validation", None)
    compute_validation_loss = getattr(trainer, "compute_validation_loss", None)
    if callable(begin_validation):
        validation_state = begin_validation()
    try:
        with torch.no_grad():
            for start in range(0, len(val_records), effective_batch_size):
                batch_records = val_records[start : start + effective_batch_size]
                batch = build_batch_from_records(
                    batch_records,
                    masked_loss=masked_loss,
                    training_policies=getattr(trainer, "training_policies", None),
                    training=False,
                )
                if callable(compute_validation_loss):
                    loss = compute_validation_loss(batch)
                else:
                    loss, _ = trainer.compute_loss(batch)
                loss_value = float(loss.item()) if torch.is_tensor(loss) else float(loss)
                sample_count = len(batch_records)
                total_loss += loss_value * sample_count
                total_samples += sample_count
    finally:
        if validation_state is not None and callable(end_validation):
            end_validation(validation_state)
    if total_samples <= 0:
        return None
    return float(total_loss / total_samples)


def is_oom_runtime_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text
