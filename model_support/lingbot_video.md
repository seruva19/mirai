# LingBot Video

Native training and inference support targets the sparse-MoE 30B-A3B release.
The provider loads attributed native modules directly; Diffusers is not a runtime
dependency.

## Download

Download the public snapshot without a token:

```bash
python scripts/download.py \
  --variant lingbot-video-moe-30b-a3b \
  --output-dir models/lingbot-video-moe-30b-a3b
```

Set `HF_TOKEN` only when the selected repository requires authentication.

## Presets

LingBot Video provides three family presets. They are deep-merged over the
family-neutral `defaults/moe.toml` training layer before values from the user
config are applied; model identity, paths, and architecture defaults come from
the LingBot preset itself.

### [`lingbot_video.toml`](../mirai/config/presets/lingbot_video.toml)

Resident BF16 training defaults:

- `[model]`: type, path, dtype, attention_backend
- `[model.params]`: variant, flow_shift, strict_native_assets, latent_channels,
  num_experts, experts_per_token, shared_experts, hidden_size, num_layers,
  attention_heads, patch_size, moe_aux_loss_weight, moe_router_z_loss_weight
- `[strategy]`: type
- `[training]`: batch_size, max_steps, gradient_checkpointing, loss_function,
  objective
- `[optimizer]`: type, lr
- `[logging]`: output_dir

### [`lingbot_video_offload.toml`](../mirai/config/presets/lingbot_video_offload.toml)

Memory-constrained training defaults with block swapping and frozen-weight
compression:

- `[model]`: type, dtype, attention_backend
- `[model.params]`: variant, strict_native_assets, latent_channels, flow_shift,
  num_experts, experts_per_token, shared_experts, patch_size, moe_aux_loss_type,
  moe_aux_loss_weight, moe_router_z_loss_weight, moe_bias_update_rate,
  moe_bias_centering
- `[strategy]`: type
- `[training]`: batch_size, gradient_accumulation, max_steps,
  gradient_checkpointing, gradient_cpu_offload, optimizer_cpu_offload,
  activation_cpu_offload, block_swap_mode, block_swap_backward, blocks_to_swap
- `[optimizer]`: type, allow_fallback, lr
- `[adapter]`: type, target_preset, rank, alpha
- `[dataset]`: auto_preprocess_cache, preprocess_raw_media_to_pt,
  max_cache_skip_ratio, frame_buckets
- `[logging]`: output_dir, save_every_n_steps, sample_every_n_steps
- `[memory]`: frozen_weight_quantization,
  frozen_weight_quantization_strategy, quantize_experts_on_load,
  expert_weight_access, expert_dequant_chunk_size, weight_residency_strategy,
  moe_kernel_backend, cuda_memory_fraction, minimum_system_memory_gib,
  trainable_parameter_offload

### [`lingbot_video_inference_bf16.toml`](../mirai/config/presets/lingbot_video_inference_bf16.toml)

High-VRAM inference defaults with a resident BF16 denoiser, disabled frozen-weight
quantization and offload, batched CFG, LingBot JSON prompt rewriting, and resident
text encoder and VAE across repeated session generations.

## Training

Available examples:

- [`train_bf16.toml`](../configs/lingbot_video/train_bf16.toml) — BF16 frozen
  base with native routed-expert execution.
- [`train_nf4.toml`](../configs/lingbot_video/train_nf4.toml) — NF4 compressed
  frozen base with chunked expert reconstruction. This example uses the
  bitsandbytes optimizer installed with Mirai.
- [`train_nf4_32gb.toml`](../configs/lingbot_video/train_nf4_32gb.toml) — NF4
  compressed frozen base kept **fully resident** on a 32 GiB device. Block
  swapping is disabled: host-device transfer only pays for itself when the
  weights do not fit, and the compressed base does. Unlike `train_nf4.toml`
  this example uses the dependency-free `adamw` optimizer rather than the
  Linux-only bitsandbytes path, so it runs unchanged on a Windows workstation.
  On a DGX Spark (128 GiB unified) it applies unchanged: the resident
  compressed base plus activations fit the unified pool, and
  `memory.cuda_memory_fraction` with `memory.minimum_system_memory_gib` guard
  the same pool from both directions.

Set `[model].path`, `[dataset].path`, and `[logging].output_dir`, then run:

```bash
python scripts/train.py \
  --config configs/lingbot_video/train_nf4.toml \
  --dry-run

python scripts/train.py \
  --config configs/lingbot_video/train_nf4.toml
```

Resume from an adapter checkpoint:

```bash
python scripts/train.py \
  --config configs/lingbot_video/train_nf4.toml \
  --resume <checkpoint-path>
```

Supported adapter targets include attention, router projections, shared MLPs,
and routed expert projections. Raw video may be bucketed and encoded through the
family-owned native VAE and structured text-conditioning path.

LingBot declares its routed-expert computation as `w1` gate, `w3` up, and `w2`
down with a SiLU gated-product combiner. Shared compressed execution consumes
those semantic roles rather than inferring tensor names. Packed artifacts,
learned rotations, FlexMoE transforms, and other layout-specific tools validate
the canonical LingBot graph and reject incompatible expert layouts explicitly.

LingBot's native BSHD attention and packed sequence metadata use the shared
attention-backend registry. Supported choices and capability rules are listed
in [`CONFIG_REFERENCE.md`](../CONFIG_REFERENCE.md).

The provider supports text-to-video, first-frame image-to-video, hybrid
conditioning, and progressive multi-task training. Hybrid batches must contain
a precomputed `clip_embed` compatible with the cached `text_embeds`; the native
raw-video cache path does not synthesize that external embedding.

For a staged T2I:T2V:I2V run, select `strategy.type="multi_task_video"`, add a
canonical `training_task` value to every dataset-registration sample, and
configure `training.curriculum.task_mix_schedule`. LingBot uses BCTHW latents:
T2I records must contain one latent frame, while I2V masks and preserves the
first latent frame. Resolution/frame stage changes feed their actual latent
shape into the shared dynamic flow-shift resolver when
`model.params.flow_shift_mode="dynamic"`. The raw-video cache path supplies T2V
and I2V video records; one-frame T2I records require precomputed
LingBot-compatible latent input.

```toml
[strategy]
type = "multi_task_video"
params = { first_frame_conditioning_p = 1.0 }

[training.curriculum]
enabled = true
resolution_schedule = { "0" = "256x256", "1000" = "512x512" }
frame_schedule = { "0" = 1, "1000" = 33 }

[training.curriculum.task_mix_schedule."0"]
text_to_image = 1

[training.curriculum.task_mix_schedule."1000"]
text_to_video = 6
image_to_video = 3
```

## Inference

The example [`inference_bf16.toml`](../configs/lingbot_video/inference_bf16.toml)
enables batched classifier-free guidance and keeps the text encoder and VAE
resident.

[`inference_nf4_32gb.toml`](../configs/lingbot_video/inference_nf4_32gb.toml)
is the same generation path on a 32 GiB device: NF4 compressed weights, no
block swapping, and the text encoder and VAE released between phases instead of
kept resident. Set either `keep_*_resident` back to `true` only when the
measured peak leaves room for it.

Prompts use the release’s structured JSON caption format, described under
[Prompt and negative-prompt contract](#prompt-and-negative-prompt-contract).
The family entrypoint accepts `@<path>` to read the caption from a file; the
model-agnostic `scripts/infer.py` takes the caption text on `--prompt` itself.

```bash
python inference/lingbot_video/generate.py \
  --prompt @prompt.json \
  --adapter <adapter-path> \
  --out outputs/clip.mp4
```

The family entrypoint exposes four tasks:

- `--task t2v` generates an MP4 from text.
- `--task t2i` generates one frame and writes a PNG; `--frames` is forced to
  one and the video-only refiner is rejected.
- `--task ti2v --input-image <image>` cover-resizes and center-crops the first
  frame, samples its native VAE latent, adds Qwen3-VL image tokens to both CFG
  branches, and restores the conditioned latent prefix before every model
  forward and after every solver step.
- `--task v2v --input-video <video> --denoising-strength <0..1>` samples the
  requested source frames, uses the native VAE posterior mode, mixes source and
  noise at the first retained flow sigma, and runs only the selected tail of
  the solver schedule.

```bash
python inference/lingbot_video/generate.py \
  --task t2i \
  --prompt @prompt.json \
  --out outputs/frame.png

python inference/lingbot_video/generate.py \
  --task ti2v \
  --input-image first_frame.png \
  --prompt @prompt.json \
  --out outputs/conditioned.mp4

python inference/lingbot_video/generate.py \
  --task v2v \
  --input-video source.mp4 \
  --denoising-strength 0.55 \
  --prompt @prompt.json \
  --out outputs/transformed.mp4
```

T2I and TI2V follow the released family pipeline
[(repo)](https://github.com/Robbyant/lingbot-video). V2V adapts the
source-latent and denoising-strength mechanism introduced by
[(DiffSynth-Studio PR #1545)](https://github.com/modelscope/DiffSynth-Studio/pull/1545)
behind Mirai’s provider contract.

The default generation profile is BF16, 832×480, 33 frames at 24 FPS,
Flow-UniPC with 25 steps, CFG 3, and flow shift 3. Use `--scheduler euler
--steps 40` for the Euler reference trajectory.

### Prompt and negative-prompt contract

The DiT reads its caption as text serialized into the VLM chat template, so the
exact byte string is the conditioning. Mirai resolves any accepted prompt form
to the caption body the encoder consumes
(`mirai/core/models/lingbot_video/prompting.py`):

- A structured prompt file — `--prompt @prompt.json` on the family entrypoint,
  or a JSON object passed to `--prompt` on either entrypoint — is unwrapped
  from its `caption` envelope and stripped of the runtime-only keys `duration`,
  `fps`, `height`, `width`, `num_frames`, `resolution`, `ratio`. It is never
  re-wrapped, so the encoder receives no `caption` or `duration` tokens. The
  `@<path>` form is argument handling owned by
  `inference/lingbot_video/generate.py`; `scripts/infer.py` stays
  model-agnostic and reads no files from `--prompt`.
- Plain language is wrapped into the minimal `comprehensive_description` body
  with an empty `camera_movement_description`. `prominent_elements` and
  `camera_info` come from the released LLM rewriter, which Mirai does not ship,
  and are not synthesized from a sentence — supply real rewriter output to use
  them.
- The negative prompt is conditioning text rather than a caption and is
  forwarded byte-for-byte. The vendored default is the released serialization,
  spacing included; re-serializing it would change its tokenization.

`dataset.caption_format = "lingbot_json"` runs the same resolution at
cache-encode time, so a LoRA trains on the conditioning it is later prompted
with. Captions cached before this contract resolve to different text and their
cache fingerprints will not match.

#### Caption schema

A prompt file is `{"caption": {...}, "duration": <int>}`. The caption body
carries three blocks, which the release’s rewriter fills
(`rewriter/system_prompts.py`, `VIDEO_STEP2_MAP`, and the `assets/cases/`
examples in the [repo](https://github.com/Robbyant/lingbot-video)):

| Field | Type | Contents |
|---|---|---|
| `comprehensive_description` | object | `scene_content_description` and `camera_movement_description`, both strings; the camera string may be empty. |
| `prominent_elements` | list of objects | One entry per subject: `name`, `description`, `actions` (list of `{timestamp, action}`), `location`, `relative_size`, `shape_and_color`, `texture`, `pose`, `expression`, `clothing`, `is_cluster`, `number_of_objects`. Human-only descriptors stay present and blank for non-human subjects. |
| `camera_info` | object | `color`, `frame_size`, `shot_type_angle`, `lens_size`, `composition`, `lighting`, `lighting_type`, drawn from the rewriter’s closed vocabularies. |

`mirai/core/models/lingbot_video/prompting.py` declares this schema as typed
field specs and checks the parsed caption against it whenever the caption is
resolved — that is, under `inference.prompt_rewriter = "lingbot_json"`,
`dataset.caption_format = "lingbot_json"`, and the family entrypoint without
`--raw`. Two outcomes:

- **A declared field carrying the wrong type is rejected.** Resolution raises
  and names the field, the type the schema declares, and the type found — for
  example `camera_info: expected object, found str`, or
  `prominent_elements[0].actions: expected list of objects, found str`. Right
  field names with wrong types is the common way a hand-written caption goes
  off-distribution, and it is not something a caller intends.
- **A caption missing declared fields is reported and used.** Resolution emits
  a `LingBotCaptionWarning` naming the absent fields. A bare sentence is the
  main case: it is a legitimate deliberate request, so it proceeds.

Both diagnostics are raised at prompt resolution. On the family entrypoint that
is before the model is loaded, so a malformed caption costs no load or render
time. `--raw` (family entrypoint) and `inference.prompt_rewriter = "none"` /
`--prompt-rewriter none` (`scripts/infer.py`) bypass caption resolution
entirely and send the text as conditioning unchanged.

Under `dataset.caption_format = "lingbot_json"` the same rule applies per
caption at cache-encode time, so a malformed dataset caption stops the cache
build instead of silently entering the cache.

#### Why the caption schema matters

An off-schema caption is off-distribution conditioning, not a formatting
detail. On one fixed scene and seed, holding everything but the caption
constant, decoder activations overshoot the VAE output range by:

| Caption | Decoder overshoot | Result |
|---|---|---|
| Bare sentence, wrapped into the minimal body | 9.09% | Posterized, over-saturated, flat silhouettes, banding, rainbow fringing |
| Malformed caption — schema field names, wrong types | 1.94% | Still posterized |
| Schema-valid rich caption | ~0.9% | Photorealistic |

For reference, a real encoded video sits at 0.85% under the same measurement.
Scene content stays correct in every case; what degrades is appearance. This is
the model’s response to the caption, not a Mirai defect: the release’s own
runner reproduces the same degradation on the same malformed caption (1.72%
overshoot upstream against 1.94% here), and on a schema-valid caption the two
runners are indistinguishable (mean saturation 0.3106 upstream against 0.3139
here).

The supported way to obtain a caption is the release’s two-stage LLM rewriter,
which Mirai does not ship. Mirai does not substitute for it: it will not invent
`prominent_elements`, `camera_info`, or action timestamps from a sentence,
because a fabricated caption is off-distribution content carrying Mirai’s
signature rather than the release’s.

The family declares its negative prompt, denoise steps, CFG scale, and solver
through the provider capability `ModelFamilyProvider.generation_defaults()`, so
the generic `scripts/infer.py` applies them too — driving it directly no longer
runs classifier-free guidance against an empty unconditional. The negative
prompt text lives at `mirai/core/models/lingbot_video/default_negative_prompt.json`,
read only by the provider.

`--negative-prompt`, `--steps`, `--cfg-scale`, and `--scheduler` distinguish
*omitted* from *explicitly set* in both entrypoints. Omitted takes the declared
family value; anything passed explicitly wins. An explicit
`--negative-prompt ""` is honored and prints a warning naming what degrades:
guidance then steers away from an unconditional carrying none of the family's
quality and artifact terms, which weakens predicted structure at high sigma and
inflates latent magnitude.

Use `--quant nf4` for the compressed frozen-base path. Native Euler, Flow-UniPC,
and DPM++ 2M solvers are available; refiner execution is controlled by the
entrypoint’s refiner arguments.

For the [LightLingBot-Video distilled LoRA](https://huggingface.co/lightx2v/LightLingBot-Video),
select the family-owned four-step profile:

```bash
python inference/lingbot_video/generate.py \
  --model-root <model-root> \
  --adapter <lightlingbot-v2.safetensors> \
  --inference-profile distilled-4step \
  --prompt '{"scene":"..."}' \
  --out outputs/clip.mp4
```

The profile selects four Euler steps, CFG 1, rank/alpha 128, unit LoRA scale,
and the attention-plus-shared-MLP target set. Explicit CLI values override its
defaults.

## Model-specific features

- **Native sparse-MoE pipeline** — Family-owned loading, latent validation,
  conditioning, forward loss, adapter injection, and inference hooks.
  [(paper)](https://arxiv.org/abs/2607.07675)
  [(repo)](https://github.com/Robbyant/lingbot-video)
- **Strict snapshot validation** — Shard indexes, component identities, tensor
  roles, and released dimensions are checked before execution.
  [(repo)](https://github.com/Robbyant/lingbot-video)
- **Structured prompt alignment** — Prompt normalization preserves the
  model-specific system/user schema expected by the text encoder, and the
  resolved caption is checked against that schema before the model is loaded:
  a wrong field type is rejected by name, a missing field is reported.
  [(repo)](https://github.com/Robbyant/lingbot-video)
- **Native text encoder and VAE** — Training caches and inference use native
  family components without a Diffusers load or forward dependency.
  [(repo)](https://github.com/Robbyant/lingbot-video)
- **Native T2I and TI2V conditioning** — Single-frame image generation and
  text-plus-image video generation preserve the release’s VAE first-frame and
  Qwen3-VL visual-token semantics.
  [(repo)](https://github.com/Robbyant/lingbot-video)
- **V2V schedule-strength conditioning** — Source-video latents enter a
  truncated native flow schedule at a configurable noise strength.
  [(repo)](https://github.com/modelscope/DiffSynth-Studio/pull/1545)
- **Native refiner staging** — Base denoising can hand off latents to the
  family-specific refinement stage with explicit solver and resolution control.
  [(repo)](https://github.com/Robbyant/lingbot-video)
- **Router runtime integration** — Balance objectives, dataset specialization,
  subset schedules, iterative expert communication, distillation, and routing
  diagnostics attach through LingBot-owned runtime hooks and shared policy
  objects.
- **Compressed routed experts** — Native expert modules support packed restore,
  bounded dequantization, vectorized dispatch, and adapter gradients.

## Model-specific configuration

Only keys whose accepted values or behavior are specific to LingBot Video are
listed here. Shared training, MoE, adapter, memory, and inference keys remain in
[`CONFIG_REFERENCE.md`](../CONFIG_REFERENCE.md).

| key | LingBot Video values and behavior |
|---|---|
| `model.type` | `lingbot-video`. |
| `model.path` | Snapshot root containing the native denoiser plus the released text-encoder, processor, VAE, and optional refiner component directories. |
| `model.params.variant` | Released training and inference target: `lingbot-video-moe-30b-a3b`. |
| `model.params.denoiser_subfolder` | Native denoiser component, normally `transformer`; the value is included in cache and checkpoint lineage. |
| `model.params.text_encoder_path` | Optional override for the Qwen3-VL text-encoder asset directory; empty resolves it from `model.path`. |
| `model.params.strict_native_assets` | Requires complete released native assets and snapshot validation; public model configurations set this to `true`. |
| `model.params.inference_routing_telemetry` | Enables detached inference top-k trace capture for `scripts/infer.py --routing-trace-out`. |
| `model.params.inference_routing_telemetry_layer_stride` | Retains every Nth router layer in the inference trace. |
| `model.params.moe_expert_backend` | Native routed-expert executor: `grouped_mm`, `loop`, or optional `sglang_triton`. |
| `model.params.moe_pad_backend` | Grouped-token padding path: `auto`, `loop`, or `vectorized`. |
| `model.params.moe_reorder_backend` | Route packing path: `sort` or CUDA/Triton `triton_pack`. |
| `model.params.moe_restore_backend` | Expert-output restore path: `scatter`, `chunked_scatter`, or CUDA/Triton `triton`. |
| `model.params.moe_restore_chunk_size` | Route rows per bounded `chunked_scatter` restore operation. |
| `model.params.moe_fused_qkv_linear` | Enables the native fused QKV projection when compatible with the loaded tensor layout. |
| `model.params.inference_bf16_fastmath` | Enables the family-owned optional BF16 inference fast-math path. |
| `dataset.caption_format` | `lingbot_json` resolves each training caption to the caption body the encoder consumes at cache time; structured captions are normalized rather than re-wrapped, and the caption schema is checked per caption. |
| `inference.prompt_rewriter` | `lingbot_json` applies the same family-owned caption resolution and schema check to inference prompts. |
