"""Shared Qwen3-VL text-encoder ownership for LingBot-Video.

This module is the single owner of the LingBot-Video prompt-embedding contract.
Both the cache-encoding path (``lingbot_video_cache``) and the native-inference
hooks (``lingbot_native_inference``) call the exact same code object here, which
is what makes inference-time ``encode_prompt`` embeddings bit-identical to the
embeddings stored by the cached-latent training path.
"""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any

from mirai.core.models.checkpoint_streaming import resolve_safetensor_files

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


TOKEN_LENGTH = 37698
HIDDEN_STATE_SKIP_LAYER = 0
PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    "or a video reference alone, generate an \"Enhanced prompt\" that provides detailed "
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

_HF_TEXT_ENCODER_HINT = (
    "Fetch the upstream LingBot-Video repository's text_encoder/ and processor/ "
    "subfolders, e.g. `hf download <lingbot-video-repo> --include 'text_encoder/*' "
    "'processor/*'`, or point model.path / model.params.text_encoder_path at a "
    "snapshot that already contains them."
)


def resolve_text_encoder_dtype(name: str) -> Any:
    """Map a config dtype name to a torch dtype (shared by cache + inference)."""
    if torch is None:  # pragma: no cover
        return None
    text = str(name or "").strip().lower()
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def load_qwen3vl_text_assets(
    *,
    text_encoder_dir: str | Path,
    processor_dir: str | Path,
    dtype: Any,
) -> tuple[Any, Any]:
    """Load the Qwen3-VL processor + text encoder, failing fast on missing assets.

    This is the single loader convention reused by both the cache path and the
    native-inference path. Missing directories or weight shards raise a
    ``FileNotFoundError`` naming the missing path and the ``hf download``
    remediation, mirroring strict native transformer loading.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for LingBot-Video text encoding.")
    te_dir = Path(text_encoder_dir)
    pr_dir = Path(processor_dir)
    if not te_dir.is_dir():
        raise FileNotFoundError(
            f"LingBot-Video native text encoding requires a Qwen3-VL text encoder "
            f"directory at '{te_dir}'. {_HF_TEXT_ENCODER_HINT}"
        )
    if not pr_dir.is_dir():
        raise FileNotFoundError(
            f"LingBot-Video native text encoding requires a processor directory at "
            f"'{pr_dir}'. {_HF_TEXT_ENCODER_HINT}"
        )
    weight_files = resolve_safetensor_files(
        te_dir,
        index_names=("model.safetensors.index.json",),
        direct_names=("model.safetensors",),
    )
    if not weight_files:
        raise FileNotFoundError(
            f"LingBot-Video text encoder weights (safetensors) were not found under "
            f"'{te_dir}'. {_HF_TEXT_ENCODER_HINT}"
        )
    try:
        from transformers import AutoProcessor
        import transformers
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for LingBot-Video native text encoding."
        ) from exc
    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        raise RuntimeError(
            "Installed transformers does not expose Qwen3VLForConditionalGeneration; "
            "use a LingBot-compatible transformers build."
        )
    processor = AutoProcessor.from_pretrained(str(pr_dir), trust_remote_code=True)
    text_encoder = model_cls.from_pretrained(
        str(te_dir),
        dtype=dtype,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    text_encoder.eval()
    return processor, text_encoder


def compute_prompt_crop_start(processor: Any) -> int:
    """Return the number of template-prefix tokens to crop before the user text."""
    marker = "<|USER_INPUT_MARKER|>"
    marked = PROMPT_TEMPLATE.format(marker)
    marker_pos = marked.find(marker)
    if marker_pos < 0:
        return 0
    prefix = processor(
        text=marked[:marker_pos],
        images=None,
        videos=None,
        return_tensors="pt",
    )
    return int(prefix["input_ids"].shape[1])


def encode_prompt_embedding(
    *,
    processor: Any,
    text_encoder: Any,
    caption: str,
    crop_start: int,
    image: Any | None = None,
) -> "torch.Tensor":
    """Produce the LingBot-Video prompt embedding for ``caption``.

    Single source of truth for the embedding format (template wrapping, hidden
    state selection, attention-mask crop, true-length slice, CPU/float32/contiguous
    output). Cache encoding and inference encoding both call this exact function,
    guaranteeing bit-identical embeddings for identical model/inputs.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for LingBot-Video text encoding.")
    caption_text = str(caption)
    images = None
    if image is not None:
        caption_text = f"{VISION_MARKER}{caption_text}"
        images = [prepare_qwen3vl_image(processor, image)]
    text = PROMPT_TEMPLATE.format(caption_text)
    inputs = processor(
        text=[text],
        images=images,
        videos=None,
        do_resize=False,
        truncation=True,
        max_length=TOKEN_LENGTH,
        padding="longest",
        return_tensors="pt",
    )
    device = next(text_encoder.parameters()).device
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = text_encoder(
            **inputs,
            output_hidden_states=HIDDEN_STATE_SKIP_LAYER is not None,
        )
    if HIDDEN_STATE_SKIP_LAYER is not None:
        embeds = outputs.hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)]
    else:
        embeds = outputs.last_hidden_state
    mask = inputs["attention_mask"].bool()
    if crop_start > 0:
        embeds = embeds[:, crop_start:]
        mask = mask[:, crop_start:]
    true_len = max(1, int(mask[0].sum().item()))
    return embeds[0, :true_len].detach().cpu().float().contiguous()


def _smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Qwen-VL compatible factor-aligned resize with bounded pixel count."""
    if height <= 0 or width <= 0:
        raise ValueError("Prompt image dimensions must be > 0.")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Prompt image aspect ratio must be <= 200.")
    factor = max(1, int(factor))
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    pixels = h_bar * w_bar
    if pixels > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif pixels < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
    return int(h_bar), int(w_bar)


def prepare_qwen3vl_image(processor: Any, image: Any) -> Any:
    """Resize one PIL-like image to the Qwen3-VL visual token grid."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("LingBot multimodal prompt encoding expects a PIL image.")
    image_processor = getattr(processor, "image_processor", processor)
    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    factor = max(1, patch_size * merge_size)
    min_pixels = int(getattr(image_processor, "min_pixels", factor * factor * 4))
    max_pixels = int(
        getattr(image_processor, "max_pixels", factor * factor * 16384)
    )
    target_h, target_w = _smart_resize(
        int(image.height),
        int(image.width),
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return image.convert("RGB").resize((target_w, target_h), Image.Resampling.BICUBIC)


class LingBotVideoTextEncoder:
    """Lazy Qwen3-VL text-encoder holder shared by cache + inference paths.

    Owns the processor/model handles and the cached crop-start. It does not own
    device residency: callers decide where the model runs (the cache path drives
    it through ``SequentialComponentResidency``; the inference path loads directly
    on the compute device). ``encode`` runs on whatever device the model already
    occupies.
    """

    def __init__(
        self,
        *,
        text_encoder_dir: str | Path,
        processor_dir: str | Path,
        dtype: Any,
    ) -> None:
        self.text_encoder_dir = Path(text_encoder_dir)
        self.processor_dir = Path(processor_dir)
        self.dtype = dtype
        self._processor: Any | None = None
        self._model: Any | None = None
        self._crop_start: int | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("LingBot-Video text encoder is not loaded.")
        return self._model

    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise RuntimeError("LingBot-Video text encoder is not loaded.")
        return self._processor

    def load(self) -> None:
        if self._model is not None:
            return
        self._processor, self._model = load_qwen3vl_text_assets(
            text_encoder_dir=self.text_encoder_dir,
            processor_dir=self.processor_dir,
            dtype=self.dtype,
        )

    def crop_start(self) -> int:
        self.load()
        if self._crop_start is None:
            self._crop_start = compute_prompt_crop_start(self._processor)
        return int(self._crop_start)

    def encode(self, caption: str, *, image: Any | None = None) -> "torch.Tensor":
        self.load()
        return encode_prompt_embedding(
            processor=self._processor,
            text_encoder=self._model,
            caption=caption,
            crop_start=self.crop_start(),
            image=image,
        )
