"""Shared LingBot caption/prompt normalization.

Single owner of the caption contract that puts a Mirai prompt into the token
distribution the LingBot-Video DiT was trained on. The family inference
rewriter and the model-provider cache-caption hook both call this owner, so the
two paths cannot drift apart.

The DiT consumes structured JSON captions; raw natural language is
out-of-distribution. The caption is serialized into the VLM chat template as
text and never parsed back into fields, so the exact byte string is the
conditioning: key order, separators, and every wrapper key change the
embedding.

Two reference behaviors are reproduced here:

- A structured caption supplied as a mapping is unwrapped from its ``caption``
  envelope (or stripped of runtime-only keys) and re-serialized compactly, so
  the encoder never sees ``caption`` or ``duration`` tokens.
- A caption supplied as a string is conditioning already and is forwarded
  byte-for-byte, which is what keeps a vendored negative prompt exact.

The plain-text wrapper emits only the ``comprehensive_description`` block the
schema defines for a sentence. ``prominent_elements`` and ``camera_info`` are
produced by the reference LLM rewriter, which Mirai does not ship, so they are
not synthesized from a sentence.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from mirai.core.inference.prompt_rewriter import (
    PromptRewriteRequest,
    PromptRewriteResult,
    register_prompt_rewriter,
)

# Keys a structured prompt file carries for the runner, not for the encoder.
# They describe the requested clip, not its content, and are dropped before
# serialization.
LINGBOT_RUNTIME_CAPTION_KEYS = frozenset(
    {"duration", "fps", "height", "width", "num_frames", "resolution", "ratio"}
)

CAPTION_FORMATS = ("raw", "lingbot_json")


def is_valid_json_caption(text: str) -> bool:
    """True when ``text`` already parses as JSON (an already-structured caption)."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _dumps_caption(value: Any) -> str:
    """Serialize a caption body with the exact separators the encoder expects."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_lingbot_caption(prompt: str | Mapping[str, Any] | list) -> str:
    """Return the caption text the encoder receives for ``prompt``.

    A mapping is a structured prompt file: its ``caption`` body is unwrapped,
    or -- when the mapping is already the body -- the runtime-only keys are
    dropped. A string is conditioning already and passes through unchanged.
    """
    if isinstance(prompt, Mapping):
        if "caption" in prompt:
            caption: Any = prompt["caption"]
        else:
            caption = {
                key: value
                for key, value in prompt.items()
                if key not in LINGBOT_RUNTIME_CAPTION_KEYS
            }
        if isinstance(caption, (Mapping, list)):
            return _dumps_caption(caption)
        return str(caption)
    if isinstance(prompt, list):
        return _dumps_caption(prompt)
    if isinstance(prompt, str):
        return prompt
    raise TypeError(
        f"LingBot prompt must be a str or a mapping, not {type(prompt).__name__}."
    )


def wrap_caption_as_lingbot_json(text: str) -> str:
    """Wrap a plain NL caption into the minimal structured caption body.

    Only the ``comprehensive_description`` block is emitted, and only the
    scene content is filled: a sentence states no camera movement, so that
    field stays empty rather than inventing one. The result is a caption body,
    not a prompt file, so it carries no ``caption`` envelope and no runtime
    keys.
    """
    return _dumps_caption(
        {
            "comprehensive_description": {
                "scene_content_description": text,
                "camera_movement_description": "",
            }
        }
    )


def resolve_lingbot_prompt(prompt: str | Mapping[str, Any]) -> str:
    """Resolve any accepted prompt form to the caption text the encoder sees.

    - A mapping, or a string that parses as a structured JSON object, is real
      rewriter output: it is normalized, never re-wrapped.
    - Any other non-empty string is plain language and is wrapped.
    - An empty string stays empty; no structure is fabricated for it.
    """
    if not isinstance(prompt, str):
        return normalize_lingbot_caption(prompt)
    if not prompt.strip():
        return prompt
    try:
        parsed = json.loads(prompt)
    except (json.JSONDecodeError, ValueError):
        return wrap_caption_as_lingbot_json(prompt)
    if isinstance(parsed, dict):
        return normalize_lingbot_caption(parsed)
    # A JSON scalar or array is not a structured caption; it is conditioning
    # the caller chose verbatim, so it is forwarded unchanged.
    return prompt


def resolve_training_caption(
    caption: str | Mapping[str, Any],
    *,
    caption_format: str,
) -> str:
    """Resolve one dataset caption for cache encoding under ``caption_format``.

    - ``"raw"`` (default): the caption is returned byte-for-byte unchanged.
    - ``"lingbot_json"``: structured captions are normalized and plain
      language is wrapped, exactly as the inference path resolves a prompt.
    """
    fmt = str(caption_format or "raw").strip().lower()
    if fmt not in CAPTION_FORMATS:
        raise ValueError(
            f"Unknown caption_format '{caption_format}'; expected one of {CAPTION_FORMATS}."
        )
    if fmt == "raw":
        if not isinstance(caption, str):
            raise TypeError(
                "caption_format='raw' requires a string caption, "
                f"not {type(caption).__name__}."
            )
        return caption
    return resolve_lingbot_prompt(caption)


@register_prompt_rewriter("lingbot_json")
def rewrite_lingbot_inference_prompts(
    request: PromptRewriteRequest,
) -> PromptRewriteResult:
    """Align raw inference prompts with LingBot's structured conditioning.

    Only the positive prompt is a caption. The negative prompt is an
    unconditional the caller supplies as conditioning text already -- the
    family default is a vendored byte-exact string -- so re-serializing it
    would change the very tokens it was measured with.
    """
    return PromptRewriteResult(
        prompt=resolve_lingbot_prompt(request.prompt),
        negative_prompt=request.negative_prompt,
    )
