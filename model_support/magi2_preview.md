# MAGI-2 Preview

Native training and inference support targets SandAI's preview-stage
[MAGI-2 Preview](https://huggingface.co/sand-ai/MAGI-2-preview) transformer. The
provider loads the attributed native modules vendored under
`mirai/vendors/magi2_preview` directly; the denoiser, text encoder, and VAE are
plain `torch.nn.Module` graphs. Diffusers is not a dependency of the family:
neither loading nor a forward pass imports it.

The reference release ships a multi-device configuration
(`engine_config.cp_size = engine_config.ep_size = 8`). Mirai forces both to `1`,
keeps the frozen checkpoint in host RAM, and transfers transformer blocks to one
GPU on demand. This requires substantial host memory and fast host-to-device
bandwidth; it does not reduce the preview-stage checkpoint's storage footprint,
which is roughly 228 GB of BF16 weights.

## Architecture

Orientation for the values the configuration surface refers to. They come from
[`magi2_preview.json`](../mirai/vendors/magi2_preview/configs/magi2_preview.json),
which is the default `model.params.family_params.config_path`.

- **Scale.** 114B total parameters, 6B activated per token. 40 transformer
  layers with `hidden_size = 3072` and `head_dim = 128`; layers `0`, `1`, `38`,
  and `39` are dense multimodal layers and the 36 layers in between are
  multi-head sparse-MoE layers.
- **Multi-head MoE.** Each MoE layer routes 12 heads independently. Every head
  owns 256 experts and selects top-6 of them per token, so a layer holds
  `12 x 256 = 3072` experts on one flattened `head * num_experts + expert` axis.
  Expert width is `d_head = 3072 / 12 = 256` and
  `expert_intermediate_size = 1280`, so the packed BF16 expert tensors are
  `W_gate`/`W_up` of shape `[3072, 256, 1280]` and `W_down` of shape
  `[3072, 1280, 256]`, with a `swiglu7` gated-product combiner.
- **Shared experts.** Two always-on paths run beside the routed experts: a
  global shared expert and a modality-specific shared expert, both of
  intermediate size 1280. The provider reports `num_shared_experts = 2`.
- **Routing.** Router logits are scored with `sigmoid`. A per-expert learned
  bias buffer (`router.expert_bias`) shifts the scores used for top-k
  *selection* only and never the returned probabilities; this is the
  aux-loss-free load-balancing form, and the family carries no auxiliary balance
  or router z-loss term. Selected probabilities are L1-normalized
  (`route_norm = true`) and then multiplied by `route_scale = 4.9`.
- **Residual stream.** MHC hyper-connections are enabled with `num_stream = 4`
  and `alpha_init = 0.01`, so each block reads and writes four residual streams
  rather than one.
- **Attention.** Attention runs with sinks enabled (`sink_token_num = 1`) and
  per-head output gating (`attn_gating.enable = true`); softcapping is off.

## Install

MAGI-2 pulls dependencies the rest of Mirai does not need. Install them with
the family extra:

```bash
pip install -e ".[magi2-preview]"
```

The extra adds Triton (the fused multi-head MoE kernels in the vendored
transformer), `tqdm` (shard loading and sampler progress), SciPy (audio-feature
resampling in the vendored inference engine), `pydantic-settings` (the typed
architecture and engine config models), and `unfoldNd` (the n-dimensional
unfold in the vendored data proxies). The last two are imported by this family
alone, so they install with the extra rather than with the base package.
Diffusers is not
part of the extra: the vendored TurboVAE decoder and Flow-UniPC scheduler carry
their own constructor-argument registration in
`mirai/vendors/magi2_preview/common/native_config.py`. `einops` and the pinned
`transformers` runtime that provides the Qwen3.5 text-encoder classes are
already base dependencies. The Triton requirement carries a
`sys_platform == "linux"` marker because the PyPI package is Linux-only; a
Windows host needs a Windows Triton build installed manually.

The optional accelerated runtime — the pinned MagiAttention, MagiCompiler, and
FlashAttention packages described by the upstream release — is deliberately not
part of the extra. Every one of those imports is guarded, and Mirai retains a
differentiable Torch reference path for training and for environments where
those kernels are unavailable. Install them manually only if you want the
accelerated attention and compile paths. MagiAttention reads
`MAGI_ATTENTION_WORKSPACE_BASE` when it is imported; Mirai never writes it for
you, so set it in the environment that starts the process if you install that
package.

## Download

`scripts/download.py` does not carry a MAGI-2 variant. Fetch the official
snapshot from the release yourself and point `model.path` at the snapshot root,
which must contain:

- `preview/` with `model.safetensors.index.json` and its shards — validated
  before execution. `model.path` may also point directly at `preview/`.
- `text_encoder/` — the Qwen3.5 text encoder used for prompt encoding and for
  training-cache text embeddings.
- `vae/Wan2.2_VAE.pth` — the native encoder used by the training cache.
- `turbo_vae/TurboV3-Wan22-TinyShallow_7_7.json` and
  `turbo_vae/checkpoint.ckpt` — the TurboVAE decoder used at inference.
- `refiner/` with `model.safetensors.index.json` and its shards — required only
  by the optional [refiner stage](#refiner-stage), and validated when that stage
  is requested. The refiner ships no `config.json`; its architecture comes from
  the vendored refiner profile.

## Presets

### [`magi2_preview_offload.toml`](../mirai/config/presets/magi2_preview_offload.toml)

The single shipped preset. It is deep-merged over
[`defaults/magi2_preview.toml`](../mirai/config/defaults/magi2_preview.toml)
before values from the user config are applied, and it defines the
host-resident single-GPU path:

- `[model]`: type, dtype
- `[model.params]`: variant, strict_native_assets, flow_shift
- `[model.params.family_params]`: audio_tokens
- `[training]`: gradient_checkpointing, blocks_to_swap, block_swap_mode
- `[adapter]`: type, target_preset, rank, alpha
- `[memory]`: weight_residency_strategy, block_swap_prefetch_depth,
  minimum_system_memory_gib, max_pinned_host_gib

## Memory strategy

The preset selects `memory.weight_residency_strategy = "block_swap"`, which
keeps immutable staging copies of the frozen transformer blocks in host RAM and
moves a block onto the execution device only for its forward/backward window.
Disk streaming (`stream_disk`) is a separate explicit choice and never an
automatic fallback.

The values that matter for planning a host:

| key | preset value | effect |
|---|---|---|
| `memory.weight_residency_strategy` | `block_swap` | RAM-to-VRAM block residency; no disk path. |
| `training.blocks_to_swap` | `40` | Every transformer layer participates, so only the swap window is device-resident. |
| `training.block_swap_mode` | `async` | Transfers overlap compute on a side stream. |
| `memory.block_swap_prefetch_depth` | `1` | One block of transfer lookahead; one extra resident swap block is reserved. |
| `memory.minimum_system_memory_gib` | `384.0` | Host-RAM guard: the run refuses to start below it. |
| `memory.max_pinned_host_gib` | `224.0` | Ceiling on page-locked host memory. |
| `training.gradient_checkpointing` | `standard` | Trades recompute for activation memory across the swap window. The vendored block carries one whole-block recompute switch, so `selective` and `aggressive` are rejected for this family. |

Page-locked staging is bounded by
`min(free_ram - minimum_system_memory_gib, max_pinned_host_gib)`; tensors over
that budget fall back to pageable host storage, which changes transfer behavior
and never tensor values. In practice the guard and the pin ceiling together
mean the whole roughly 228 GB frozen checkpoint has to fit in host RAM
alongside the 384 GiB system reserve, so this preset targets a
large-host / single-GPU machine rather than a workstation. Lower
`training.blocks_to_swap` only if the resident remainder fits in device memory.

## Grouped MoE execution

Training executes the multi-head routed experts through the vendored per-expert
reference loop by default (`memory.moe_kernel_backend` at `auto` or `torch`).
That loop remains the reference path for the grouped seam.

`memory.moe_kernel_backend = "grouped"` replaces it with grouped execution:
every `(head, expert)` pair is flattened onto the packed expert axis, routed
token slices are sorted once, and the gate/up/down projections run as grouped
GEMMs. Expert weights stay frozen — the grouped path raises if `W_gate`,
`W_up`, or `W_down` requires a gradient, and its custom autograd Function
computes only `dX` — while routing, activation, and the probability-weighted
combine remain in native autograd.

`memory.moe_gemm_backend` (and its `moe_gemm_backend_forward` /
`moe_gemm_backend_dx` role overrides) selects the grouped primitive. MAGI-2
accepts `auto`, `bmm`, and `torch_grouped`; `persistent` and `deepgemm_fp8` are
rejected because they expect the compressed-weights slice layout and
block-scaled FP8 experts respectively, and MAGI-2 experts are plain BF16.
`auto` resolves to `torch_grouped` where the torch build and the device provide
it — `torch.nn.functional.grouped_mm` is the public probe target with an SM80+
contract, and the private `torch._grouped_mm` fallback is admitted only on
SM90+ — and to `bmm` otherwise. A backend the torch build cannot provide is
rejected when the policy is configured; the device-architecture gate applies
once the execution device is known.

`torch_grouped` also requires every operand stride to be a multiple of 16
bytes, which for BF16 experts means `d_head` and `expert_intermediate_size`
must both be multiples of 8. The expert layout is checked once when the grouped
seam is attached, enumerating the real forward and dX operands rather than
guessing from shapes, so `auto` resolves to `bmm` for a layout that does not
satisfy the precondition and an explicit `torch_grouped` is rejected with the
offending operand named. There is no per-call downgrade.

`memory.moe_kernel_backend`, `memory.moe_gemm_backend`,
`memory.moe_gemm_backend_forward`, and `memory.moe_gemm_backend_dx` are the
only MoE memory-policy keys this family consumes. Every other field of the
shared policy must hold its default: a non-default value fails closed with the
key named, rather than being silently ignored. `memory.expert_weight_access`
(the family keeps native BF16 experts) and `memory.moe_gemm_backend_dw`
(frozen experts give the weight-gradient GEMM no consumer) are accepted only at
values that request no behavior.

## Training

Use [`configs/magi2_preview/train_offload.toml`](../configs/magi2_preview/train_offload.toml),
which selects the offload preset and sets `model.path`, the dataset, and the
output directory:

```bash
python scripts/train.py \
  --config configs/magi2_preview/train_offload.toml \
  --dry-run

python scripts/train.py \
  --config configs/magi2_preview/train_offload.toml
```

Adapters are LoRA only: `adapter.type` must be `lora`, and `adapter.rank` must
be positive. The packed-weight LoRA parametrization is shape-preserving and
does not support adapter dropout — a non-zero `adapter.rank_dropout` or
`adapter.lora_parameter_dropout` is rejected. Base weights are frozen; the
provider clears `requires_grad` on every transformer parameter before injecting
the parametrizations, and only the LoRA factors are saved.

Two target presets are supported:

| `adapter.target_preset` | targets |
|---|---|
| `attn_only` (default) | `.attention.linear_qkv`, `.attention.linear_proj` |
| `attn_router` | the two attention projections plus `.mlp.moe_mlp.gate` |

`attn_router` trains the router projection itself. Consider this carefully: the
family declares `emits_router_metrics = false` and carries no balance or
z-loss objective, so a router-training run produces no routing-health signal
and no load-balancing pressure. Routing drift is not observable from the
training loop.

Captions are cached in the family's `raw` format. The native cache path encodes
text through the Qwen3.5 encoder and latents through the native Wan2.2 VAE
(`vae/Wan2.2_VAE.pth`), each loaded lazily on first use. Video frames are
trimmed to `8n + 1` and to a multiple of 16 in both spatial dimensions before
encoding; precomputed `.pt` latents must have shape `[48, T, H, W]`.

The shipped config sets `dataset.auto_preprocess_cache = true`, so a missing
cache is built from `dataset.path` through that provider-owned encoder on the
first run. Because the family declares native cache encoding,
`dataset.preprocess_raw_media_to_pt` defers to the same encoder instead of the
generic media preprocessor.

Resume from an adapter checkpoint:

```bash
python scripts/train.py \
  --config configs/magi2_preview/train_offload.toml \
  --resume <checkpoint-path>
```

## Inference

Inference uses the same native denoiser and the same block-residency path as
training, so the preview transformer runs on one GPU with host-resident block
streaming. The declared inference task is `text_to_video`.

Use [`configs/magi2_preview/inference_offload.toml`](../configs/magi2_preview/inference_offload.toml)
with `scripts/infer.py`:

```bash
python scripts/infer.py \
  --config configs/magi2_preview/inference_offload.toml \
  --scheduler unipc \
  --frames 121 \
  --prompt "..." \
  --out outputs/clip.mp4
```

The example keeps `inference.keep_text_encoder_resident` and
`inference.keep_vae_resident` at `false`, so text encoding, denoising, and VAE
decoding occupy the execution device sequentially. Set either to `true` to
trade VRAM for reload time across repeated session generations.

### Generation length and output frame rate

The preview transformer is positioned on a **25 fps timeline sampled at
temporal stride 8** (`vae_stride = [8, 16, 16]`, `time_pos_fps = 25 / 8 =
3.125`). A `--frames` request is therefore expressed on that 25 fps timeline:
`T` latent frames span `8 * (T - 1) + 1` requested frames, and only counts
congruent to 1 modulo 8 are representable. A count outside the rule is rejected
rather than rounded.

**Latent `T` 32 — 249 frames, about 10 s — is the model's native horizon**: it
is the full length the preview model is trained for, and the length upstream
resolves for a ten-second request. Longer latents are not a longer preview.
`T` 63 is reached only inside the [refiner](#refiner-stage), which resamples the
preview latent to `2T - 1` frames in time, re-noises it, and runs its own short
denoise. `T` above 32 is therefore refiner space and is rejected on the preview
path, where sampling it degenerates to flat output.

The generation envelope is therefore **57 to 249 frames** (latent `T` 8 to 32).
The floor is empirical: latent `T` 5 is observed to degenerate and the
intermediate lengths were not probed, so the bound sits on the conservative
side of the measured range rather than being inferred.

Preview-only decoding is **half-rate**. The Turbo VAE expands `T` latent frames
into `4 * (T - 1) + 1` physical frames, so a 249-frame request writes 125
frames — the requested ten seconds only when played at **12.5 fps**. A file of
249 physical frames at 25 fps needs the refiner's temporal resample; see
[Refiner stage](#refiner-stage). The family declares its native output rate
through its latent layout — 12.5 fps preview-only, 25 fps once `--refine` is
resolved — so `scripts/infer.py` writes at that rate when `--fps` is not passed;
an explicit `--fps` still wins. No frames are padded or trimmed to the requested
count: the file carries exactly what the VAE decoded.

| request (`--frames`) | latent `T` | decoded frames | duration at 12.5 fps |
|---|---|---|---|
| 57 | 8 | 29 | 2.3 s |
| 121 | 16 | 61 | 4.9 s |
| 249 | 32 | 125 | 10.0 s |

The bounds apply to generation only — training forwards accept any
representable latent length, and `dataset.frame_buckets` is unaffected.
`scripts/infer.py` defaults `--frames` to 17, which this family rejects as
below the floor; pass a length inside the envelope. `logging.sample_frame_count`
must also fall inside it for a training run that emits previews; the shipped
example sets 121 for a cheap mid-length preview.

The native sampler owns its own CFG execution and its own schedule, so both
requested policies are checked rather than applied. Every denoise step packs
the conditional and unconditional branches into one single-device `B=2`
forward, which is why the shipped example sets `inference.cfg_mode = "batched"`
and passes `--scheduler unipc`: `sequential` and any other solver name are
rejected instead of being silently ignored. A training run that emits previews
sets `logging.sample_solver = "unipc"` for the same reason.

Prompt-to-video output requires the official Qwen3.5 text encoder and the
TurboVAE decoder from the same snapshot; both are validated before use and a
missing asset fails with the expected path named. Throughput and peak-memory
figures are published only after the GPU validation contract records them.

### Refiner stage

The refiner is an optional, default-off second stage that turns a finished
preview latent into a full-rate clip. It is a **separate checkpoint of a
different architecture**: a dense 30-layer transformer with no routed experts,
shipped in the `refiner/` subfolder of the snapshot. It does not re-generate the
clip — it consumes the preview latent.

What it does, in order:

1. **Resample.** The preview latent `[48, T, H, W]` is trilinearly interpolated
   to `[48, 2T - 1, H', W']` with `align_corners=True`. `align_corners` is
   load-bearing: it pins preview frame `k` to refined frame `2k`, so the
   inserted frames are true midpoints rather than a half-frame shift of the
   whole trajectory. `H'`/`W'` come from the refiner target resolution, which
   defaults to the preview resolution — the default refinement is purely
   temporal.
2. **Re-noise.** The resampled latent is corrupted once, variance-preserving, at
   a fixed index into a zero-terminal-SNR `sqrt(alphas_cumprod)` table:
   `x * sigma + n * sqrt(1 - sigma^2)`. `sigma` is the *signal* coefficient, so
   the released index keeps most of the preview and asks the refiner to restore
   detail rather than to invent content. This is not the rectified-flow blend
   the preview path uses.
3. **Denoise.** A short Flow-UniPC run over the released step count, with the
   conditional and unconditional branches evaluated as **two separate forwards**
   — unlike the preview sampler, which packs them into one `B=2` forward. The
   refiner's window attention admits batch size 1 only.
4. **Decode.** The refined latent sits on the shared Turbo VAE's own temporal
   grid (`magi2_refiner_vae_stride = [4, 16, 16]`), so `2T - 1` latent frames
   decode to `8 * (T - 1) + 1` physical frames — exactly the frames the request
   denotes, at **25 fps**.

The refiner runs with **zero audio tokens**. The released profile sets
`magi2_refiner_audio_noise_scale = -1`, which is a sentinel for "no audio track
at all" rather than a scale; the stage fails explicitly if a configured profile
makes the refiner return audio velocity.

**Cost.** The refiner is a second model load and a second denoise, so a refined
run pays a full extra stage on top of the preview. The refined latent carries
roughly twice the tokens of the preview latent at the same resolution, and more
again if the target resolution is raised, so its per-step cost is not comparable
to a preview step. The preview transformer is released off the compute device
before any refiner state is allocated, so the two never co-reside; refinement
ends the current sampling session and the next generation re-places the preview
transformer. When the run is configured for block swapping, the refiner streams
its own layers under the same policy. No latency or peak-memory figures are
claimed here; they are published only after the GPU validation contract records
them.

Enable it with `--refine`:

```bash
python scripts/infer.py \
  --config configs/magi2_preview/inference_offload.toml \
  --scheduler unipc \
  --frames 249 \
  --refine \
  --prompt "..." \
  --out outputs/clip.mp4
```

`--refine` alone runs the released refinement profile. Every refiner flag
defaults to unset, which means "the family decides"; a stated value overrides
the profile, and the run's payload reports the values that were actually
applied.

| flag | MAGI-2 behavior when unset |
|---|---|
| `--refiner-steps` | The released refiner step count. |
| `--refiner-cfg` | The released refiner guidance scale. |
| `--refiner-shift` | The released refiner flow shift, falling back to the preview `shift`. |
| `--refiner-height` / `--refiner-width` | The preview `--height` / `--width`, making the refinement purely temporal. Both must be multiples of 16, the refiner VAE spatial stride. |
| `--refiner-scheduler` | `unipc`; any other solver is rejected, as on the preview path. |
| `--refiner-t-thresh`, `--refiner-sigma-tail-steps` | Not implemented by this family and **rejected** if stated. They describe a rectified-flow tail re-entry; MAGI-2 re-noises once at a table index instead. Change it through the refiner profile. |

A refine request is validated before the base denoise starts, so an unusable
one never costs a generation. It is rejected when the `refiner/` subfolder holds
no `model.safetensors.index.json` — the error names the directory — when the
frame request falls outside the preview generation envelope, when the solver is
not `unipc`, or when a key belonging to another family's refiner is stated.

## Model-specific features

- **MAGI-2 Preview single-GPU adapter training** — Adapter-only optimization
  with native preview-transformer weights and gradients.
  [(repo)](https://github.com/SandAI-org/MAGI-2-preview)
- **Host-resident MAGI-2 block streaming** — Frozen transformer blocks move to
  the execution device only for their forward/backward window.
- **Native MAGI-2 inference** — The provider uses the same native denoiser and
  residency path for sampling; the denoiser is a plain `torch.nn.Module` and
  Diffusers is imported nowhere in loading or in a forward pass.
- **Native MAGI-2 refiner staging** — Optional default-off second stage that
  resamples the preview latent in time, re-noises it once, and short-denoises it
  through the released refiner checkpoint, producing a full-rate clip.
- **Grouped MAGI-2 expert execution** — Optional grouped-GEMM execution of the
  multi-head routed experts during training, selected by
  `memory.moe_kernel_backend="grouped"`, with the vendored per-expert loop
  retained as the reference path.

## Model-specific configuration

Only keys whose accepted values or behavior are specific to MAGI-2 Preview are
listed here. Shared training, MoE, adapter, memory, and inference keys remain in
[`CONFIG_REFERENCE.md`](../CONFIG_REFERENCE.md).

| key | MAGI-2 Preview values and behavior |
|---|---|
| `model.type` | `magi2-preview`, or the accepted alias `magi-2-preview`. |
| `model.path` | Snapshot root containing `preview/`, `text_encoder/`, `vae/`, and `turbo_vae/`; it may also point directly at `preview/`. |
| `model.params.variant` | Released target: `magi2-preview-114b-a6b`. |
| `model.params.strict_native_assets` | Requires the complete released native assets; the shipped preset sets this to `true`. |
| `model.params.flow_shift` | Preset value `7.0`, matching the release's `evaluation_config.shift`. |
| `model.params.family_params.config_path` | Override for the vendored architecture JSON; empty resolves to the shipped `magi2_preview.json`. |
| `model.params.family_params.refiner_config_path` | Override for the vendored refiner profile JSON; empty resolves to the shipped `magi2_refiner.json`. The profile states the refiner architecture, its step count, guidance scale, flow shift, VAE stride and the zero-terminal-SNR index the stage re-noises at, so changing the refinement means pointing this at a different profile. Only read when `--refine` is requested. |
| `model.params.family_params.refiner_subfolder` | Snapshot subdirectory holding the refiner shards; defaults to `refiner`. Must be relative to the snapshot root and must not traverse upwards. |
| `model.params.family_params.audio_tokens` | Length of the audio track the multimodal forward requires. MAGI-2 ships no audio encoder, so the track carries no user signal: as in the reference engine it is Gaussian noise, and it takes part in attention and MoE routing. The training forward redraws it on every call from a family-owned generator seeded from `training.seed`, so a step is reproducible under its seed and the track length never perturbs the process RNG stream; native sampling draws it once per generation from the generation generator. `-1` derives the length from the latent frame count; a non-negative value fixes it. |
| `adapter.type` | `lora` only. |
| `adapter.target_preset` | `attn_only` or `attn_router`. |
| `dataset.caption_format` | `raw`; captions are encoded by the native Qwen3.5 path at cache time. |
| `inference.task` | `text_to_video`. |
| `inference.cfg_mode` | `batched` only. The native sampler evaluates the conditional and unconditional branches together in one `B=2` forward; `sequential` is rejected. |
| `training.gradient_checkpointing` | `off` or `standard`. The vendored transformer block exposes one whole-block recompute switch, so `selective` and `aggressive` are rejected rather than collapsed onto `standard`. |
| `logging.sample_solver` / `--scheduler` | `unipc` only. The native sampler is the vendored Flow-UniPC multistep scheduler and implements no other schedule. |
