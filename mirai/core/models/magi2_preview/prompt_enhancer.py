"""Structured MAGI-2 captions from a caller-configured chat endpoint."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from mirai.core.inference.prompt_rewriter import (
    PromptRewriteRequest,
    PromptRewriteResult,
    register_prompt_rewriter,
)


_CAPTION_FIELDS = ("scene", "action", "camera", "visual_style")
_CAPTION_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "string"} for name in _CAPTION_FIELDS},
    "required": list(_CAPTION_FIELDS),
    "additionalProperties": False,
}
_INSTRUCTIONS = (
    "Write a structured caption for a video generator using the user's request. "
    "Describe the subjects and setting in scene, the motion over time in action, "
    "the framing and camera movement in camera, and the lighting and appearance in visual_style. "
    "Keep requested details and ordering. When shown an initial frame, preserve its visible "
    "subjects and setting. Return only the requested JSON object."
)


def _image_url(image: object) -> str:
    from PIL import Image, ImageOps

    if isinstance(image, Image.Image):
        preview = ImageOps.exif_transpose(image).convert("RGB")
    elif isinstance(image, (str, os.PathLike)):
        with Image.open(image) as source:
            preview = ImageOps.exif_transpose(source).convert("RGB")
    else:
        raise ValueError("MAGI-2 prompt enhancement expects a local image path or a Pillow image.")
    preview.thumbnail((1024, 1024))
    with io.BytesIO() as buffer:
        preview.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def _format_caption(caption: object) -> str:
    if (
        not isinstance(caption, dict) or caption.keys() != set(_CAPTION_FIELDS)
        or any(not isinstance(value, str) or not value.strip() for value in caption.values())
    ):
        raise ValueError("MAGI-2 prompt enhancement returned an invalid caption.")
    return "\n".join(f"{name.replace('_', ' ').capitalize()}: {caption[name].strip()}" for name in _CAPTION_FIELDS)


def _request_caption(endpoint: str, payload: dict, key: str) -> str:
    import requests

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    for attempt in range(3):
        try:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=(5, 60), allow_redirects=False,
            )
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise RuntimeError("MAGI-2 prompt enhancement endpoint could not be reached.") from None
        else:
            if response.status_code < 200 or response.status_code >= 300:
                if attempt == 2 or (response.status_code != 429 and response.status_code < 500):
                    raise RuntimeError(f"MAGI-2 prompt enhancement failed with HTTP {response.status_code}.")
            else:
                try:
                    content = response.json()["choices"][0]["message"]["content"]
                    return _format_caption(json.loads(content))
                except (ValueError, TypeError, KeyError, IndexError):
                    if attempt == 2:
                        raise RuntimeError("MAGI-2 prompt enhancement returned no valid structured caption.") from None
        time.sleep(0.25 * (attempt + 1))
    raise AssertionError("The request loop must return or raise.")


@register_prompt_rewriter("magi2_structured")
def enhance_magi2_prompt(request: PromptRewriteRequest) -> PromptRewriteResult:
    base = os.environ.get("MIRAI_MAGI2_REWRITER_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("MIRAI_MAGI2_REWRITER_MODEL", "").strip()
    address = urlsplit(base)
    if not model or address.scheme not in {"http", "https"} or not address.hostname or address.username or address.query or address.fragment:
        raise ValueError("Set MIRAI_MAGI2_REWRITER_BASE_URL to an HTTP(S) API base URL and MIRAI_MAGI2_REWRITER_MODEL to a model name.")
    endpoint = base + "/chat/completions"
    content = [{"type": "text", "text": request.prompt}]
    if request.image is not None:
        content.append({"type": "image_url", "image_url": {"url": _image_url(request.image)}})
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": _INSTRUCTIONS}, {"role": "user", "content": content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "magi2_caption", "schema": _CAPTION_SCHEMA, "strict": True},
        },
        "max_tokens": 2048,
    }
    folder = os.environ.get("MIRAI_MAGI2_REWRITER_CACHE_DIR", "").strip()
    cache_path = None
    if folder:
        identity = json.dumps([endpoint, payload], sort_keys=True, ensure_ascii=False)
        cache_path = Path(folder) / (hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".txt")
        try:
            cached = cache_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            pass
        else:
            if cached.strip():
                return PromptRewriteResult(cached, request.negative_prompt)
    caption = _request_caption(endpoint, payload, os.environ.get("MIRAI_MAGI2_REWRITER_API_KEY", ""))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_path.parent, delete=False,
        ) as temporary:
            temporary.write(caption)
        try:
            os.replace(temporary.name, cache_path)
        finally:
            Path(temporary.name).unlink(missing_ok=True)
    return PromptRewriteResult(caption, request.negative_prompt)
