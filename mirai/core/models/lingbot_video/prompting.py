"""Shared LingBot caption/prompt wrapping.

Single owner of the minimal Rewriter-schema JSON caption that aligns plain
natural-language text with the structured caption the LingBot-Video DiT was
trained on. The family inference rewriter and the model-provider cache-caption
hook both call this owner, so the two wrapping paths cannot drift apart.

The DiT consumes structured JSON captions; raw NL is out-of-distribution. At
inference an NL prompt is auto-wrapped here; at cache-encode time ``dataset.caption_format =
"lingbot_json"`` runs the same wrapper so a LoRA trains on the same JSON
conditioning it is later prompted with.
"""

from __future__ import annotations

import json

from mirai.core.inference.prompt_rewriter import (
    PromptRewriteRequest,
    PromptRewriteResult,
    register_prompt_rewriter,
)

# The canonical structured-prompt duration for a 33-frame, 24-fps clip is 1.4 s.
# Training and inference use this shared value when wrapping natural language.
DEFAULT_LINGBOT_DURATION = 1.4

CAPTION_FORMATS = ("raw", "lingbot_json")


def is_valid_json_caption(text: str) -> bool:
    """True when ``text`` already parses as JSON (an already-structured caption).

    Mirrors the inference-side check exactly: any JSON-parseable string (object,
    array, or even a bare scalar) is treated as already structured and passes
    through untouched.
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def wrap_caption_as_lingbot_json(
    text: str, *, duration: float = DEFAULT_LINGBOT_DURATION
) -> str:
    """Wrap a plain NL caption into the minimal Rewriter-schema JSON caption.

    The serialization (key order, compact separators, ``ensure_ascii=False``) is
    load-bearing: it is the exact byte string the inference CLI emits, so the
    trainer and the generator produce identical conditioning for identical NL
    input.
    """
    wrapped = {
        "caption": {
            "comprehensive_description": {
                "scene_content_description": text,
                "camera_movement_description": "",
            }
        },
        "duration": float(duration),
    }
    return json.dumps(wrapped, ensure_ascii=False, separators=(",", ":"))


def resolve_training_caption(
    caption: str,
    *,
    caption_format: str,
    duration: float = DEFAULT_LINGBOT_DURATION,
) -> str:
    """Resolve one dataset caption for cache encoding under ``caption_format``.

    - ``"raw"`` (default): the caption is returned byte-for-byte unchanged.
    - ``"lingbot_json"``: a caption that already parses as JSON passes through
      untouched; a non-empty NL caption is wrapped into the minimal JSON caption;
      an empty caption is left empty (no fabricated structure).
    """
    fmt = str(caption_format or "raw").strip().lower()
    if fmt not in CAPTION_FORMATS:
        raise ValueError(
            f"Unknown caption_format '{caption_format}'; expected one of {CAPTION_FORMATS}."
        )
    if fmt == "raw":
        return caption
    if not caption.strip():
        return caption
    if is_valid_json_caption(caption):
        return caption
    return wrap_caption_as_lingbot_json(caption, duration=duration)


@register_prompt_rewriter("lingbot_json")
def rewrite_lingbot_inference_prompts(
    request: PromptRewriteRequest,
) -> PromptRewriteResult:
    """Align raw inference prompts with LingBot's structured conditioning."""
    return PromptRewriteResult(
        prompt=resolve_training_caption(request.prompt, caption_format="lingbot_json"),
        negative_prompt=resolve_training_caption(
            request.negative_prompt, caption_format="lingbot_json"
        ),
    )
