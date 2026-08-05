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
transformer), `tqdm` (shard loading and sampler progress), and SciPy
(audio-feature resampling in the vendored inference engine). Diffusers is not
part of the extra: the vendored TurboVAE decoder and Flow-UniPC scheduler carry
their own constructor-argument registration in
`mirai/vendors/magi2_preview/common/native_config.py`. `einops`, `unfoldNd`,
`pydantic-settings`, and the pinned
`transformers` runtime that provides the Qwen3.5 text-encoder classes are
already base dependencies. The Triton requirement carries a
`sys_platform == "linux"` marker because the PyPI package is Linux-only; a
Windows host needs a Windows Triton build installed manually.

The optional accelerated runtime — the pinned MagiAttention, MagiCompiler, and
FlashAttention packages described by the upstream release — is deliberately not
part of the extra. Every one of those imports is guarded, and Mirai retains a
differentiable Torch reference path for training and for environments where
those kernels are unavailable. Install them manually only if you want the
accelerated attention and compile paths.

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
| `training.gradient_checkpointing` | `aggressive` | Trades recompute for activation memory across the swap window. |

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
  --prompt "..." \
  --out outputs/clip.mp4
```

The example keeps `inference.keep_text_encoder_resident` and
`inference.keep_vae_resident` at `false`, so text encoding, denoising, and VAE
decoding occupy the execution device sequentially. Set either to `true` to
trade VRAM for reload time across repeated session generations.

The provider declares batch/mask parity, so `inference.cfg_mode = "batched"`
is permitted and runs the conditional and unconditional branches in one
single-device `B=2` forward. The shipped example uses `sequential`, which is
the lower-VRAM choice on a host already streaming blocks.

Prompt-to-video output requires the official Qwen3.5 text encoder and the
TurboVAE decoder from the same snapshot; both are validated before use and a
missing asset fails with the expected path named. Throughput and peak-memory
figures are published only after the GPU validation contract records them.

## Model-specific features

- **MAGI-2 Preview single-GPU adapter training** — Adapter-only optimization
  with native preview-transformer weights and gradients.
  [(repo)](https://github.com/SandAI-org/MAGI-2-preview)
- **Host-resident MAGI-2 block streaming** — Frozen transformer blocks move to
  the execution device only for their forward/backward window.
- **Native MAGI-2 inference** — The provider uses the same native denoiser and
  residency path for sampling; the denoiser is a plain `torch.nn.Module` and
  Diffusers is imported nowhere in loading or in a forward pass.
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
| `model.params.family_params.audio_tokens` | Length of the inert audio track the multimodal forward requires. `-1` derives it from the latent frame count; a non-negative value fixes it. |
| `adapter.type` | `lora` only. |
| `adapter.target_preset` | `attn_only` or `attn_router`. |
| `dataset.caption_format` | `raw`; captions are encoded by the native Qwen3.5 path at cache time. |
| `inference.task` | `text_to_video`. |
