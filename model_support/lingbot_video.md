# LingBot Video

Native training and inference support targets the sparse-MoE 30B-A3B release.
The provider loads attributed native modules directly; Diffusers is not a runtime
dependency.

## Presets

LingBot Video provides three family presets. They are deep-merged over the
shared `defaults/moe.toml` layer before values from the user config are applied.

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
  frozen base with chunked expert reconstruction.

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

Generic packed-weight providers bind to LingBot's native expert roles without a
family-specific switch: `w1/w3` are the gate/up projections and `w2` is the
down projection. Consequently, mixture-basis artifacts optimize only the native
gate/up pair and retain the packed down projection unchanged.
Learned INT8 rotation export likewise binds one shared transform to the native
gate/up pair and a separate transform to down, allowing the generic packed
execution path to rotate the common hidden input once. The controlling key
remains generic and is therefore documented only in `CONFIG_REFERENCE.md`.
AIMER scoring consumes the same family mapping (`w1=gate`, `w3=up`, `w2=down`)
as one combined expert vector; its generic criterion and CLI controls likewise
remain in `CONFIG_REFERENCE.md`.
MXFP8-E4M3 applies through the same model-agnostic compressed-weight boundary:
all three native expert projections retain their family roles while their
frozen tensors use independent 32-value blocks and packed scale metadata. Its
shared selection key is documented only in `CONFIG_REFERENCE.md`.
DeepSeek-style block-scaled FP8 uses that same boundary with 128×128 expert
weight scales, dynamic 1×128 activation scales, and FP32 partial accumulation.
The family contributes only its native w1/w2/w3 projection mapping; the format,
artifact, execution, and gradient contracts remain model-agnostic.
LingBot's native BSHD attention and packed sequence metadata are adapted to the
shared attention-backend registry. The provider adds no family-specific backend
value; all accepted choices and capability rules remain in
`CONFIG_REFERENCE.md`.
The generic SOLO 4/2-bit and Adam-mini optimizers consume LingBot's
provider-owned per-parameter selected-expert row plan without a family branch.
Their optimizer types and state semantics remain in `CONFIG_REFERENCE.md`.
Drop-Upcycling artifacts use the same generic grouped-expert roles. The packed
loader expands the native router and expert axes together, then the provider
synchronizes its execution metadata before adapter injection; no LingBot-only
upcycling key or dispatch path exists.

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

Prompts use the release’s structured JSON format. Plain text passed through the
family entrypoint is converted to the required structure.

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
Flow-UniPC with 25 steps, CFG 3, and flow shift 3. The family entrypoint also
uses the vendored upstream negative prompt unless `--negative-prompt` overrides
it. Use `--scheduler euler --steps 40` for the Euler reference trajectory.

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
  model-specific system/user schema expected by the text encoder.
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
  diagnostics attach through typed family hooks.
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
| `dataset.caption_format` | `lingbot_json` converts raw training captions into the model’s structured prompt schema at cache time; already structured JSON passes through. |
| `inference.prompt_rewriter` | `lingbot_json` applies the same family-owned schema normalization to inference prompts. |
