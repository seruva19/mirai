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

Because the caption is conditioning rather than metadata, a caption that does
not match the release schema is off-distribution input, not a cosmetic problem.
The schema is declared here as typed field specs and checked structurally
(:class:`CaptionFieldSpec`, :func:`validate_lingbot_caption`), so the check is a
contract over parsed values rather than a pattern over serialized text. Field
names carrying the wrong type are rejected; a caption that is merely
underspecified is reported and allowed through.

Schema reference: ``rewriter/system_prompts.py`` (``VIDEO_STEP2_MAP``) and
``assets/cases/`` in https://github.com/Robbyant/lingbot-video (branch
``master``).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from enum import Enum
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

# Where a caller obtains a caption that matches the schema below. Named in every
# diagnostic, because the rewriter is the only supported producer and Mirai does
# not ship it.
REWRITER_REFERENCE = (
    "the release's two-stage prompt rewriter "
    "(rewriter/system_prompts.py, VIDEO_STEP2_MAP, in "
    "https://github.com/Robbyant/lingbot-video); see "
    "model_support/lingbot_video.md for the schema and an example"
)

_MAX_REPORTED_MISSING = 8


class CaptionValueKind(Enum):
    """What the caption schema declares a field's parsed value to be."""

    STRING = "string"
    # A field whose value is free-form but not a container: the schema fixes
    # neither its Python type (timestamps are strings or numbers, counts are
    # integers, flags are booleans) nor an enum, only that it is not structure.
    SCALAR = "scalar"
    OBJECT = "object"
    OBJECT_LIST = "list of objects"


@dataclass(frozen=True)
class CaptionFieldSpec:
    """One declared caption field, and the fields nested inside it."""

    name: str
    kind: CaptionValueKind
    fields: tuple["CaptionFieldSpec", ...] = ()


def _string(name: str) -> CaptionFieldSpec:
    return CaptionFieldSpec(name, CaptionValueKind.STRING)


def _scalar(name: str) -> CaptionFieldSpec:
    return CaptionFieldSpec(name, CaptionValueKind.SCALAR)


_ACTION_FIELDS: tuple[CaptionFieldSpec, ...] = (
    _scalar("timestamp"),
    _string("action"),
)

# Human-only descriptors stay present and blank for non-human elements, so they
# are declared required: an absent key is a different token stream from an empty
# one.
_PROMINENT_ELEMENT_FIELDS: tuple[CaptionFieldSpec, ...] = (
    _string("name"),
    _string("description"),
    CaptionFieldSpec("actions", CaptionValueKind.OBJECT_LIST, _ACTION_FIELDS),
    _string("location"),
    _string("relative_size"),
    _string("shape_and_color"),
    _string("texture"),
    _string("pose"),
    _string("expression"),
    _string("clothing"),
    _scalar("is_cluster"),
    _scalar("number_of_objects"),
)

_CAMERA_INFO_FIELDS: tuple[CaptionFieldSpec, ...] = (
    _string("color"),
    _string("frame_size"),
    _string("shot_type_angle"),
    _string("lens_size"),
    _string("composition"),
    _string("lighting"),
    _string("lighting_type"),
)

LINGBOT_CAPTION_SCHEMA: tuple[CaptionFieldSpec, ...] = (
    CaptionFieldSpec(
        "comprehensive_description",
        CaptionValueKind.OBJECT,
        (_string("scene_content_description"), _string("camera_movement_description")),
    ),
    CaptionFieldSpec(
        "prominent_elements",
        CaptionValueKind.OBJECT_LIST,
        _PROMINENT_ELEMENT_FIELDS,
    ),
    CaptionFieldSpec("camera_info", CaptionValueKind.OBJECT, _CAMERA_INFO_FIELDS),
)


@dataclass(frozen=True)
class CaptionTypeDefect:
    """A declared field present under a type the schema does not allow."""

    path: str
    expected: str
    found: str

    def describe(self) -> str:
        return f"{self.path}: expected {self.expected}, found {self.found}"


@dataclass(frozen=True)
class CaptionValidation:
    """Structural verdict for one caption body."""

    defects: tuple[CaptionTypeDefect, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def is_malformed(self) -> bool:
        """True when a declared field carries a type the schema forbids."""
        return bool(self.defects)

    @property
    def is_underspecified(self) -> bool:
        """True when declared fields are absent but nothing present is wrong."""
        return not self.defects and bool(self.missing)


def _describe_value(value: Any) -> str:
    """Name the type actually found, precisely enough to locate the mistake."""
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        if not value:
            return "empty list"
        found = sorted({type(item).__name__ for item in value})
        return "list of " + "/".join(found)
    return type(value).__name__


def _matches(kind: CaptionValueKind, value: Any) -> bool:
    if kind is CaptionValueKind.STRING:
        return isinstance(value, str)
    if kind is CaptionValueKind.SCALAR:
        return not isinstance(value, (Mapping, list)) and value is not None
    if kind is CaptionValueKind.OBJECT:
        return isinstance(value, Mapping)
    return isinstance(value, list) and all(
        isinstance(item, Mapping) for item in value
    )


def _walk(
    specs: tuple[CaptionFieldSpec, ...],
    body: Mapping[str, Any],
    prefix: str,
    defects: list[CaptionTypeDefect],
    missing: list[str],
) -> None:
    for spec in specs:
        path = f"{prefix}{spec.name}"
        if spec.name not in body:
            missing.append(path)
            continue
        value = body[spec.name]
        if not _matches(spec.kind, value):
            defects.append(
                CaptionTypeDefect(path, spec.kind.value, _describe_value(value))
            )
            continue
        if not spec.fields:
            continue
        if spec.kind is CaptionValueKind.OBJECT:
            _walk(spec.fields, value, f"{path}.", defects, missing)
        elif spec.kind is CaptionValueKind.OBJECT_LIST:
            for index, item in enumerate(value):
                _walk(spec.fields, item, f"{path}[{index}].", defects, missing)


def validate_lingbot_caption(body: Mapping[str, Any]) -> CaptionValidation:
    """Check a caption body against :data:`LINGBOT_CAPTION_SCHEMA`.

    Unknown extra keys are not reported: the schema constrains what the DiT was
    trained to read, and a caller adding a field is not making a type mistake.
    """
    defects: list[CaptionTypeDefect] = []
    missing: list[str] = []
    _walk(LINGBOT_CAPTION_SCHEMA, body, "", defects, missing)
    return CaptionValidation(tuple(defects), tuple(missing))


def _render_missing(missing: tuple[str, ...]) -> str:
    shown = ", ".join(missing[:_MAX_REPORTED_MISSING])
    remainder = len(missing) - _MAX_REPORTED_MISSING
    if remainder > 0:
        shown = f"{shown}, and {remainder} more"
    return shown


class LingBotCaptionError(ValueError):
    """A caption whose declared fields carry types the schema forbids."""


class LingBotCaptionWarning(UserWarning):
    """A schema-consistent caption that omits fields the DiT was trained on."""


def enforce_lingbot_caption_contract(body: Mapping[str, Any]) -> CaptionValidation:
    """Report an off-distribution caption before it becomes conditioning.

    A wrong type under a schema field name is a mistake nobody intends, so it
    raises. An absent field is a caption the caller may deliberately have
    chosen to keep short, so it warns and proceeds. Both diagnostics name the
    fields at fault and the supported way to obtain a full caption.
    """
    verdict = validate_lingbot_caption(body)
    if verdict.is_malformed:
        detail = "; ".join(defect.describe() for defect in verdict.defects)
        raise LingBotCaptionError(
            f"LingBot caption does not match the release caption schema: {detail}. "
            "The caption is serialized verbatim into the text encoder, so an "
            "off-schema caption conditions the model outside its training "
            f"distribution. Produce the caption with {REWRITER_REFERENCE}. To "
            "send this text as conditioning anyway, bypass caption resolution "
            "with --raw (inference/lingbot_video/generate.py) or "
            "--prompt-rewriter none (scripts/infer.py)."
        )
    if verdict.is_underspecified:
        warnings.warn(
            "LingBot caption is underspecified; the schema declares "
            f"{_render_missing(verdict.missing)}, which this caption omits. The "
            "model was trained on fully specified rewriter captions and "
            "degrades measurably on shorter ones. Mirai does not synthesize "
            "prominent_elements, camera_info, or action timestamps from a "
            f"sentence; produce the caption with {REWRITER_REFERENCE}.",
            LingBotCaptionWarning,
            stacklevel=3,
        )
    return verdict


def _dumps_caption(value: Any) -> str:
    """Serialize a caption body with the exact separators the encoder expects."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_with_body(
    prompt: str | Mapping[str, Any] | list,
) -> tuple[str, Mapping[str, Any] | None]:
    """Normalize ``prompt`` and return the caption body that was serialized.

    The body is ``None`` when the result is verbatim conditioning rather than a
    structured caption, which is the only form the schema can be checked
    against.
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
        if isinstance(caption, Mapping):
            return _dumps_caption(caption), caption
        if isinstance(caption, list):
            # A list is not a caption body, so nothing the schema declares is
            # present; an empty body reports exactly that.
            return _dumps_caption(caption), {}
        return str(caption), {}
    if isinstance(prompt, list):
        return _dumps_caption(prompt), {}
    if isinstance(prompt, str):
        return prompt, None
    raise TypeError(
        f"LingBot prompt must be a str or a mapping, not {type(prompt).__name__}."
    )


def normalize_lingbot_caption(prompt: str | Mapping[str, Any] | list) -> str:
    """Return the caption text the encoder receives for ``prompt``.

    A mapping is a structured prompt file: its ``caption`` body is unwrapped,
    or -- when the mapping is already the body -- the runtime-only keys are
    dropped. A string is conditioning already and passes through unchanged.
    This is serialization only; the schema contract is enforced by
    :func:`resolve_lingbot_prompt`, which every caller resolves through.
    """
    return _normalize_with_body(prompt)[0]


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

    Every resolved caption body is checked against the release schema before it
    is returned, so an off-distribution caption is reported here -- at prompt
    resolution -- rather than after a render.
    """
    if not isinstance(prompt, str):
        text, body = _normalize_with_body(prompt)
        if body is not None:
            enforce_lingbot_caption_contract(body)
        return text
    if not prompt.strip():
        return prompt
    try:
        parsed = json.loads(prompt)
    except (json.JSONDecodeError, ValueError):
        wrapped = wrap_caption_as_lingbot_json(prompt)
        enforce_lingbot_caption_contract(json.loads(wrapped))
        return wrapped
    if isinstance(parsed, dict):
        text, body = _normalize_with_body(parsed)
        if body is not None:
            enforce_lingbot_caption_contract(body)
        return text
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
