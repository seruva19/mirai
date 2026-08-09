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
bandwidth; on its own it does not reduce the preview-stage checkpoint's storage
footprint, which is roughly 228 GB of BF16 weights. Optional NF4 storage for the
routed expert stack does reduce it — see
[NF4 routed experts](#nf4-routed-experts).

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

The standard Mirai installation includes Triton (the fused multi-head MoE
kernels in the vendored transformer), `tqdm` (shard loading and sampler
progress), SciPy (audio-feature resampling in the vendored inference engine),
`pydantic-settings` (the typed architecture and engine config models), and
`unfoldNd` (the n-dimensional unfold in the vendored data proxies). Diffusers
is not a dependency: the vendored TurboVAE decoder and Flow-UniPC scheduler carry
their own constructor-argument registration in
`mirai/vendors/magi2_preview/common/native_config.py`. `einops` and the pinned
`transformers` runtime provide the Qwen3.5 text-encoder classes. The Triton
requirement carries a
`sys_platform == "linux"` marker because the PyPI package is Linux-only; a
Windows host needs a Windows Triton build installed manually.

The additional MagiAttention, MagiCompiler, and FlashAttention packages
described by the upstream release are deliberately not package dependencies.
Every one of those imports is guarded, and Mirai retains a differentiable Torch
reference path for training and for environments where those kernels are
unavailable. Install them manually only if you want the accelerated attention
and compile paths. MagiAttention reads
`MAGI_ATTENTION_WORKSPACE_BASE` when it is imported; Mirai never writes it for
you, so set it in the environment that starts the process if you install that
package.

## Download

`scripts/download.py` does not carry a MAGI-2 variant. Fetch the official
[`sand-ai/MAGI-2-preview`](https://huggingface.co/sand-ai/MAGI-2-preview)
snapshot and point `model.path` at its root, which must contain:

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

## Grouped MAGI-2 expert execution

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
only MoE memory-policy keys this family consumes while its experts are native
BF16. Every other field of the shared policy must hold its default: a
non-default value fails closed with the key named, rather than being silently
ignored. `memory.expert_weight_access` (nothing consumes an access policy for
unpacked experts) and `memory.moe_gemm_backend_dw` (frozen experts give the
weight-gradient GEMM no consumer) are accepted only at values that request no
behavior. Packing the experts adds three consumed keys, listed under
[NF4 routed experts](#nf4-routed-experts).

The grouped seam is also what exposes each layer's pre-combine expert features,
so `inference.expert_feature_cache` — cross-timestep expert-branch reuse during
sampling — requires `memory.moe_kernel_backend = "grouped"`. Its knobs are
described in [`CONFIG_REFERENCE.md`](../CONFIG_REFERENCE.md).

## NF4-packed MAGI-2 routed experts

`memory.frozen_weight_quantization = "nf4"` stores the routed expert stack in
NF4 instead of BF16. Exactly three tensors per MoE layer are packed —
`moe_mlp.W_gate`, `moe_mlp.W_up`, and `moe_mlp.W_down` — because they are the
only tensors addressed on the flattened `head * num_experts + expert` axis. The
router projection (`moe_mlp.gate`, FP32), the router bias buffers, both shared
experts, the hyper-connection state, the attention projections, and every norm
keep their released dtype and are not quantized by this family.

Storage is one NF4 quantization per group, stacked along the group axis, which
is what makes a contiguous group range independently dequantizable. NF4 requires
bitsandbytes with its CUDA 4-bit operators.

### What this changes, and what it does not

The size relation is derived from the layout, not measured: NF4 stores four bits
per weight plus double-quantized block statistics, against BF16's sixteen bits,
so a packed layer holds roughly a quarter of the bytes its BF16 expert
parameters held. Applied to a `114B-A6B` release whose roughly 228 GB is
overwhelmingly routed experts, that is the difference between a host that must
hold the whole BF16 stack and one that does not, and between a swapped block
carrying its full BF16 expert triple and one carrying the packed payload. No
throughput claim is implied by the storage ratio: dequantization remains
additional work per forward and per backward.

### Load path

`memory.quantize_experts_on_load = true` is the reason the host requirement
drops rather than merely the resident footprint. The dense expert parameters are
removed from the vendored layers before any shard is opened, the remaining
checkpoint keys load normally, and each routed tensor is then read from its
safetensors shard, packed, and released before the next one is read. Peak host
cost is one dense expert tensor rather than the expert stack.

For repeated runs, export the already-quantized stores once and set
`memory.frozen_weight_packed_state_path` to the resulting directory. The
provider-owned schema stores a versioned manifest and one NF4 safetensors shard
per routed-expert layer. Restore checks model variant, denoiser subfolder,
topology, group count, block size, shapes, dtypes, and the complete buffer
inventory. Layer shards are copied and unmapped one at a time, so direct restore
does not retain a second file-backed copy of the 53 GB payload.

```bash
python scripts/tools/export_compressed_weights_packed_state.py \
  --config configs/magi2_preview/train_offload_32gb.toml \
  --output /path/to/magi2-preview-nf4
```

On the measured 17-frame 256x448 workload, direct restore reached the first
training step in 330 seconds, with 29,563 MiB peak process VRAM, 69.42 GiB peak
host RSS, and a 3.67 s/step late median over the five-step probe. Artifact
creation is a one-time operation; it took 210 seconds and produced
56,084,859,648 packed bytes in the measured environment. These measurements
describe that workload and hardware rather than a universal throughput claim.

Without that key the family still packs the experts, but only after the released
BF16 checkpoint has been fully loaded, so the host must have been able to hold it
in the first place. That path exists for completeness; the profile that removes
the host requirement is
[`configs/magi2_preview/train_nf4.toml`](../configs/magi2_preview/train_nf4.toml).

### Execution

Packed experts leave no dense `W_gate`/`W_up`/`W_down` for the vendored
per-expert reference loop to read, so `memory.moe_kernel_backend` at `auto` or
`grouped` resolves to the grouped seam and an explicit `torch` is rejected rather
than silently reinterpreted. The grouped primitive selection
(`memory.moe_gemm_backend` and its role overrides) and the 16-byte operand
stride precondition behave exactly as in the BF16 grouped path; the layout
verdict is read from the packed store's declared shape and dtype instead of from
a dense tensor.

`memory.expert_weight_access` selects the dequantization granularity:

| value | behavior |
|---|---|
| `auto`, `disabled`, `full_dequant` | One dequantization per projection per call, covering the whole group axis. |
| `chunked_dequant` | `memory.expert_dequant_chunk_size` groups per dequantization. Contiguous group ranges own contiguous sorted-token ranges, so each segment runs its own grouped GEMM and its dense buffer is released before the next segment is produced. |
| `active_dequant`, `fused_kernel` | Rejected. They address one routed expert's operand, and the flattened head-major axis carries one weight slice per `(head, expert)` pair. |

Frozen weights never enter autograd. Every dequantization runs under
`torch.no_grad`, the dense segment is never saved on the autograd context, and
backward re-materializes the same segments from the same packed payload. NF4
dequantization is a deterministic function of the stored payload, so the input
gradient is computed against exactly the values the forward used. Expert weights
carry no gradient; routing, the swiglu7 ladder, and the probability-weighted
combine stay in native autograd, and router LoRA gradients are unaffected.

### Residency

The packed payload is registered on the MoE layer, so block residency moves it
exactly as it moved the BF16 parameters it replaced: a swapped block streams the
packed payload, and a resident block keeps it on the device. Choosing how much of
the packed model stays device-resident is therefore still
`training.blocks_to_swap`, and a device large enough for the packed model can set
it to `0`. `memory.expert_device_cache_gib` remains rejected for this family:
packed MAGI-2 experts are layer-resident state owned by the block-residency
subsystem, not per-expert operands streamed on demand, and a second byte-bounded
device cache over the same bytes would be a competing residency mechanism rather
than an additional one.

### Policy keys

With packed experts the family consumes four more shared MoE policy keys than
it does with native BF16 experts — `memory.expert_weight_access`,
`memory.expert_dequant_chunk_size`, `memory.quantize_experts_on_load`, and
`memory.moe_expert_autograd`. All four are rejected while the experts are BF16,
because nothing consumes them there. The exhaustive rejection of every remaining policy field is unchanged: a
non-default value fails closed with the key named.

`memory.moe_activation_backend` applies to both dense and NF4 grouped experts.
`torch` uses the bounded reference implementation; `triton` fuses the family's
exact FP32 SwiGLU7 forward and backward pointwise operations on CUDA.

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

[`configs/magi2_preview/train_offload_32gb.toml`](../configs/magi2_preview/train_offload_32gb.toml)
targets a 32 GiB device with 128 GiB of system RAM. It packs routed experts as
NF4 while reading the checkpoint, uses sink-aware FlexAttention, swaps 32
preview blocks with one-block
asynchronous prefetch, dequantizes 512 flattened expert groups per segment, and
rematerializes complete packed expert segments in backward instead of retaining
their wide gate/up graph. It uses batch size 1 with whole-block recomputation
and targets 33 frames at 512x512 without activation offload. The profile keeps
a 12 GiB free-RAM floor while permitting up to 48 GiB of pinned host memory.
Every choice is an explicit TOML value; the trainer
does not select a profile from detected RAM, VRAM, GPU model, or sample
dimensions.

## Optimization

Single-H100 training measurements use batch size 1, rank-16 `attn_router` LoRA,
BF16 compute, NF4 packed routed experts, whole-block recomputation, and one
CUDA device. Step time is synchronized wall time and excludes checkpoint
startup.

| Cache workload | Memory policy | Step time | Peak allocated VRAM | Peak process RSS |
|---|---|---:|---:|---:|
| 256x448x17 | 31 swapped blocks, 384-group dequantization | 3.80 s median | 29,421 MiB | 69.84 GiB |
| 512x512x33 (`[48, 9, 32, 32]` latent) | FlexAttention, 32 swapped blocks, 512-group dequantization, segmented expert rematerialization, fused Triton SwiGLU7, 40% CUDA cap | 9.26 s late-step median | 28,990 MiB | 69.49 GiB |

The 512x512x33 measurement is the median of steps 2 through 6. Block transfers,
NF4 dequantization, and backward rematerialization account for work that is not
represented by the model's 6B active-parameter count.

With a direct packed-expert artifact and
`model.hash_snapshot_contents = false`, model construction through device
residency measured 57.49 seconds on the 512x512x33 probe: 8.20 seconds to build
and restore the pipeline, 2.98 seconds to inject the adapter, and 46.32 seconds
to stage block residency. The complete one-step process, including the first
FlexAttention compile/forward/backward and final checkpoint output, took 98.38
seconds. Setting the flag to `true` restores a full bytewise hash of the model
tree before training.

**DGX Spark (128 GiB unified).** The 128/32 profile assumes separate host and
device pools. A unified-memory system needs its own measured residency budget;
the same totals do not establish that fit.

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

`attn_router` trains the router projection itself. The family carries no
balance or z-loss objective, so a router-training run has no load-balancing
pressure of its own. Routing becomes observable only through the opt-in
telemetry below, which the family declares by reporting
`emits_router_metrics = true` when — and only when — that gate is set.

### MAGI-2 routing-collapse telemetry

`model.params.moe_routing_health = true` (default `false`) arms detached
per-step routing diagnostics for this family
([`routing_collapse.py`](../mirai/core/models/magi2_preview/routing_collapse.py),
[`collapse.py`](../mirai/core/moe/monitoring/collapse.py)). MAGI-2's router
publishes nothing on its own, so the metrics come from a monitoring tap on the
one seam that sees a completed routing decision: the optional expert-execution
backend the vendored `CoreMultiHeadMoE` consults. The tap reads the selected
expert indices under `no_grad` and then delegates execution unchanged — to the
configured grouped backend when one is attached, otherwise to the same vendored
path the untapped layer would have taken. No vendored file is modified, the
forward result is unchanged, and nothing enters the loss graph.

Aggregation is per `(layer, head)`: each of the 36 MoE layers routes 12 heads
independently, so 432 routing surfaces are summarized, each reduced on device
to five scalars before anything is read back. Neither the token count nor the
256-wide expert axis reaches host memory, and cross-step state is a streak
counter plus a short ring of recent values per surface.

Emitted metrics, per step, in `diagnostics`:

| metric | meaning |
|---|---|
| `moe_minority_expert_share` | mean share of routed slots held by each router's least-used expert |
| `moe_normalized_minority_share` | the same share divided by the uniform share `1/num_experts`; `1.0` is balanced |
| `moe_dead_expert_fraction` | mean fraction of experts that received no routed slot |
| `moe_underused_expert_fraction` | mean fraction of experts below the health baseline |
| `moe_collapsed_router_fraction` / `moe_collapsed_router_count` | routers under the baseline this step |
| `moe_worst_normalized_minority_share` | the worst router's normalized minority share |
| `moe_max_collapse_duration` | longest consecutive collapsed run, in steps |
| `moe_collapse_rebound_count` | routers that crossed back above the baseline after a sustained collapse |
| `moe_collapse_plateau_fraction` | collapsed routers whose share stopped moving over the window |
| `moe_deadlocked_layer_count*` / `moe_max_deadlock_duration*` | the shared dominant-side deadlock counters, including depth quartiles, computed from the same observation |

The health baseline is a ratio to the uniform share rather than an absolute
one: the diagnosis this follows
([arXiv:2605.19378](https://arxiv.org/abs/2605.19378)) calls a layer healthy
while its minority expert holds at least 10% of the tokens against a 50%
uniform share, i.e. one fifth of uniform, and that ratio is what carries to a
router with 256 experts. The dominant-side threshold does not: a 256-expert
top-6 router cannot reach a 90% single-expert share even when almost every
expert is dead, which is why the minority side is measured separately.

Two estimators of the same study stay unavailable for this family.
`moe_expert_output_cossim` needs the complete per-token router score matrix,
which the tap does not see — only the selected top-k survives the seam, and
recomputing the full scores would repeat the router matmul. The router-update
underflow fraction rides the shared gradient-breakdown path and is unaffected
by this tap.

Eval and inference forwards observe nothing: the observer is silenced with the
module's training flag, so the denoise loop pays no reduction and advances no
cross-step state.

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

MAGI-2 declares its released preview profile through
`ModelFamilyProvider.generation_defaults()`: 896x512, 249 frames, 100 Flow-UniPC
steps, CFG 5.0, and the vendored released negative prompt. Explicit CLI values
still win. The native encoder retains the same negative-prompt fallback as a
backstop for direct family callers, but an ordinary CLI run resolves the value
before entering the pipeline and reports it in the result payload.

Use [`configs/magi2_preview/inference_offload.toml`](../configs/magi2_preview/inference_offload.toml)
with `scripts/infer.py`:

```bash
python scripts/infer.py \
  --config configs/magi2_preview/inference_offload.toml \
  --prompt "..." \
  --out outputs/clip.mp4
```

[`configs/magi2_preview/inference_offload_32gb.toml`](../configs/magi2_preview/inference_offload_32gb.toml)
selects the most conservative residency and occupancy the schema offers for
this family: all 40 blocks swapped synchronously, and neither the text encoder
nor the VAE resident. Denoiser activation size is set by `--height`, `--width`,
and `--frames`, which no config key controls, so on a 32 GiB device the request
is part of the profile — prefer a smaller spatial size and a latent length near
the floor of the generation envelope, and treat 832x480 at latent `T` 32 as a
large request rather than a default. The released 1920x1088 refiner target is
not a 32 GiB profile; use an explicit smaller `--refiner-height` /
`--refiner-width` there. `--refine` adds a second model load and a second
denoise; enable it only once the preview run's peak is known.

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
`scripts/infer.py` resolves this family's unset `--frames` to the released 249;
its generic 17-frame fallback applies only to families that declare no value.
`logging.sample_frame_count` must also fall inside the envelope for a training
run that emits previews; the shipped example sets 121 for a cheap mid-length
preview.

The native sampler owns its own CFG execution and its own schedule, so both
requested policies are checked rather than applied. Every denoise step packs
the conditional and unconditional branches into one single-device `B=2`
forward, which is why the shipped example sets `inference.cfg_mode = "batched"`
and the family declares `unipc` as its default solver: `sequential` and any
other solver name are rejected instead of being silently ignored. A training
run that emits previews sets `logging.sample_solver = "unipc"` for the same
reason.

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
   defaults to the released 1920x1088 delivery grid.
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

Keep preview and refinement as separate process stages when compiler workspaces
remain process-resident across model teardown. The preview latent is the exact
stage boundary, so restarting for the refiner neither crops nor re-encodes the
video latent.

The TurboVAE decoder chunks the temporal axis and offloads decoded chunks, but
does not spatially tile its internal activations. Mirai therefore bounds the
temporal-window volume deterministically from the latent geometry: the released
7/7 schedule remains unchanged for the 896x512 preview grid, while the complete
1920x1088 refined latent uses 2/2. This decodes the full 68x120 latent; it does
not crop or rescale it.
Set `family_params.vae_decode_chunk_size` to a positive integer to override the
automatic schedule on a device with a different memory envelope. Latent
downsampling after refinement remains diagnostic only and visibly softens the
image.

Enable it with `--refine`:

```bash
python scripts/infer.py \
  --config configs/magi2_preview/inference_offload.toml \
  --refine \
  --prompt "..." \
  --out outputs/clip.mp4
```

For the memory-bounded two-process path, preserve both deliverables:

```bash
python scripts/infer.py \
  --config configs/magi2_preview/inference_offload.toml \
  --prompt "..." \
  --out outputs/preview.mp4

python scripts/infer.py \
  --config configs/magi2_preview/inference_offload.toml \
  --decode-latent outputs/preview.pt \
  --refine \
  --prompt "..." \
  --out outputs/refined.mp4
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
| `--refiner-height` / `--refiner-width` | The released 1088 / 1920 delivery target. Both must be multiples of 16, the refiner VAE spatial stride. |
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
- **Host-resident MAGI-2 block streaming** — Frozen preview-transformer blocks
  use the configured residency policy while adapter tensors remain on device.
- **Grouped MAGI-2 expert execution** — The family-owned grouped seam evaluates
  the preview transformer's flattened head/expert axis without a dense
  per-expert Python loop.
- **NF4-packed MAGI-2 routed experts** — Routed expert stores can be restored
  from a versioned packed artifact and dequantized in bounded segments.
- **MAGI-2 routing-collapse telemetry** — Optional detached observations report
  per-layer/head expert-use health during training.
- **Trainable MAGI-2 attention** —
  `model.attention_backend = "flex"` routes packed attention through a backend
  with a backward pass instead of the dense-mask reference path autograd
  otherwise selects. Document block masking keeps packed samples isolated and
  per-head attention sinks retain their reference semantics, including sink
  parameter gradients.
  [(FlexAttention)](https://pytorch.org/blog/flexattention/)
- **Native MAGI-2 inference** — The provider uses the same native denoiser and
  residency path for sampling; the denoiser is a plain `torch.nn.Module` and
  Diffusers is imported nowhere in loading or in a forward pass.
- **Native MAGI-2 refiner staging** — Optional default-off second stage that
  resamples the preview latent in time, re-noises it once, and short-denoises it
  through the released refiner checkpoint, producing a full-rate clip. The
  single-GPU path chunks independent attention-projection, rotary, and MLP token
  rows, replacing the release's eight-rank context-parallel split without
  changing their per-row operations.
- **Native MAGI-2 refiner window attention** — The refiner's local-attention
  layers prefer the authors' single-GPU MagiAttention kernel on Hopper, including
  installs that omit its distributed communication extension. A registered
  `torch.ops.magi2` operator still takes precedence. When neither is available,
  Mirai runs the paired query/key ranges as one PyTorch FlexAttention mask, so
  MagiCompiler is not required. That portable fallback sweeps query boundaries
  into compact key-interval unions and constructs sparse block metadata directly;
  it never materializes the released 1080p query-by-key mask.
  [(FlexAttention)](https://pytorch.org/blog/flexattention/)

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
| `model.params.family_params.refiner_attention_backend` | `auto`, `native_flex`, or `vendor_eager`, selecting how the refiner's local-attention layers evaluate their paired query/key ranges. `auto` uses, in order, a registered `torch.ops.magi2.flex_flash_attn_func` operator, the authors' importable single-GPU MagiAttention kernel on Hopper, then Mirai's portable FlexAttention range-union path. `native_flex` always binds the portable path. `vendor_eager` leaves the vendored dispatch unchanged, which needs either MagiAttention or a FlashAttention-2 install. Both accelerated paths and the portable path implement the union of every key range paired with a query position under one softmax; only the final FlashAttention-2 correction fallback merges per-range partial softmaxes and can double-count keys reachable through more than one range. Only read when `--refine` is requested. |
| `model.params.family_params.refiner_block_swap_mode` | `sync` (default) or `async`. This is independent of `inference.block_swap_mode`: `sync` keeps only the current streamed refiner layer resident, while `async` also prefetches the next layer. Only read when `--refine` is requested. |
| `model.params.family_params.vae_decode_chunk_size` | Temporal Turbo VAE window size. `0` (default) selects a deterministic geometry-bounded schedule: released 7/7 for preview and 2/2 for the full 1920×1088 refined latent. A positive integer forces that size and changes decoder workspace residency. This never crops or rescales the latent. |
| `model.params.family_params.audio_tokens` | Length of the audio track the multimodal forward requires. MAGI-2 ships no audio encoder, so the track carries no user signal: as in the reference engine it is Gaussian noise, and it takes part in attention and MoE routing. The training forward redraws it on every call from a family-owned generator seeded from `training.seed`, so a step is reproducible under its seed and the track length never perturbs the process RNG stream; native sampling draws it once per generation from the generation generator. `-1` derives the length from the latent frame count; a non-negative value fixes it. |
| `model.params.moe_routing_health` | Arms the family's routing-collapse tap. MAGI-2 emits no router statistic without it, so this is the only way `emits_router_metrics` becomes true and the only routing signal an `attn_router` run has. Default off; when off no tap is attached and no diagnostic key appears. |
| `memory.frozen_weight_quantization` | `none` or `nf4`. `nf4` packs only the three routed expert tensors of each MoE layer and requires bitsandbytes; every other format is rejected. |
| `memory.expert_weight_access` | `auto`, `disabled`, and `full_dequant` dequantize the whole group axis per call; `chunked_dequant` dequantizes `memory.expert_dequant_chunk_size` groups at a time. `active_dequant` and `fused_kernel` are rejected — the flattened head-major axis carries one weight slice per `(head, expert)` pair, not a per-routed-expert operand. Any non-default value requires `frozen_weight_quantization = "nf4"`. |
| `memory.expert_dequant_chunk_size` | Groups per dequantization on the flattened `head * num_experts + expert` axis, not experts of one head. Required to be positive with `chunked_dequant`. |
| `memory.moe_activation_backend` | `torch` or `triton`. `triton` fuses the exact FP32 SwiGLU7 clamp, sigmoid, and backward operations and requires CUDA plus Triton. |
| `memory.quantize_experts_on_load` | Packs each routed expert tensor as its shard is read, so the released BF16 expert stack is never held whole. Requires `frozen_weight_quantization = "nf4"`. |
| `memory.moe_kernel_backend` | `auto`, `torch`, or `grouped`. With NF4 experts `torch` is rejected: the vendored per-expert reference loop reads dense expert tensors that packed storage replaces. |
| `adapter.type` | `lora` only. |
| `adapter.target_preset` | `attn_only` or `attn_router`. |
| `dataset.caption_format` | `raw`; captions are encoded by the native Qwen3.5 path at cache time. |
| `inference.task` | `text_to_video`. |
| `inference.cfg_mode` | `batched` only. The native sampler evaluates the conditional and unconditional branches together in one `B=2` forward; `sequential` is rejected. |
| `training.gradient_checkpointing` | `off` or `standard`. The vendored transformer block exposes one whole-block recompute switch, so `selective` and `aggressive` are rejected rather than collapsed onto `standard`. |
| `logging.sample_solver` / `--scheduler` | `unipc` only. The native sampler is the vendored Flow-UniPC multistep scheduler and implements no other schedule. |
