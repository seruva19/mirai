# Agnes Video 2.0

Native integration for the unpruned
[Agnes-Video-2.0-MoE checkpoint](https://huggingface.co/Agnes-AI/Agnes-Video-2.0-MoE).
The model is a joint audio/video DiT with three routed experts and Top-1 routing
in layers 4–35. The release architecture has 28.721B stored parameters and
15.299B active parameters, excluding the text encoder and VAEs.

## Native paths

Configurations: [resident training](../configs/agnes_video/train_bf16.toml),
[RAM block-swap training](../configs/agnes_video/train_block_swap.toml), and
[inference](../configs/agnes_video/inference.toml).
Image-conditioned training uses [the I2V configuration](../configs/agnes_video/train_i2v.toml).

`model.type = "agnes-video"` selects the family provider. Joint training uses
video latents `[B,48,T,H,W]`, audio latents `[B,L,128]`, and T5Gemma hidden states
of width 3584. Audio and video use the same corruption sigma with independent
Gaussian noise. Loss weights both modalities by their valid element counts;
padded audio tokens are excluded.

Training strategies are `text_to_video` and `image_to_video`. Image-conditioned
training pins the clean first latent frame and excludes its prediction from the
loss while keeping the audio target active. The existing
`strategy.params.first_frame_conditioning_p` controls conditioning probability.

`[model].attention_backend` uses the shared `auto`, `cudnn`, `flash`, `flash3`,
`flash4`, and `flex` contract. Packed samples remain isolated. Explicit
unavailable backends fail; training `auto` selects fused attention only when
its backward capability is available.
Inference `auto` follows the released runner's per-sequence attention path:
FlashAttention 3 on Hopper, then FlashAttention 2 when installed, otherwise
PyTorch SDPA. Explicit backend choices retain the shared dispatch contract.

`model.dtype` selects BF16, FP16, or FP32 compute policy. Native mixed-precision
boundaries remain intact: expert inputs and outputs use BF16, and FP16/BF16
residual additions can promote to FP32. FP32 disables enclosing autocast.
Changing compute policy does not change an adapter's frozen-base identity.
Inference prepares frozen DiT parameters and floating buffers, including norm
weights and RoPE bands, in the selected dtype as prescribed by the released
runner. Trainable LoRA matrices and adapter alpha buffers remain FP32; training retains checkpoint storage
precision. The video VAE uses the inference dtype, while OmniVAE audio decoding
uses FP32 with scoped deterministic cuDNN execution.

LoRA target presets are `attn_only`, `attn_mlp`, and `all`. The last preset also
includes modality embedders and output heads. Modality-specific projections
have independent adapter factors. `adapter.train_router` adds router adapters;
Top-1 renormalization gives zero router task gradients, so router training
requires a positive `model.params.moe_aux_loss_weight`.

Inference evolves both modalities with independent solver histories. Native
latent artifacts retain both tensors and their timeline metadata. Decoded
output contains RGB video and mono audio; FFmpeg is required for muxing.
The native sampler is `unipc`. Guided inference uses the released shifted
sigma grid and UniPC updates; CFG scale 1 uses the released stochastic
unguided update with fresh noise at each step.

Inference tasks include `text_to_video`, `image_to_video`, `video_to_video`,
and `audio_to_video`. `inference.input_audio` supplies an optional fixed audio
track for image/video conditioning and is required for `audio_to_video`.
Image conditioning pins the encoded first frame at each denoising step.

Verification covers numerical parity, gradients, persistence, packing, cache,
and media artifacts. Released-weight CUDA contracts verify joint LoRA training,
exact checkpoint-resume state equivalence, native audio/video generation, and
adapter-scale effects. These checks establish runtime correctness, not a
fine-tuning quality benchmark or a hardware recommendation.

## Model-specific features

- **Agnes Video 2.0 native integration** — Native multimodal
  denoising, activation-space LoRA, synchronized cache tensors, and joint
  latent/audio-video output contracts.

## Checkpoints and dependencies

The pinned release is `fc3971e378b53c766baa06898bfe84d7a56653ea`. Its seven
checkpoint files are byte ranges of one PyTorch archive. The converter verifies
release sizes and hashes, uses restricted `weights_only=True` loading, and writes
indexed safetensors without replacing an existing destination:

```bash
python scripts/download.py --variant agnes-video-2.0-moe --output-dir ./models/agnes-release
python -m mirai.core.models.agnes_video.checkpoints ./models/agnes-release/checkpoint_parts ./models/agnes-video
```

`model.path` points to the converted directory. Auxiliary assets are obtained
independently: T5Gemma 9B encoder, Wan2.2 VAE, and OmniVAE audio weights and
source package. T5Gemma is governed by Gemma terms; Agnes's Apache-2.0 license
does not extend to external components. FFmpeg and FFprobe are used for
synchronized media extraction. Native forward execution has no Diffusers
dependency.

Auxiliary download and source setup:

```bash
python scripts/download.py --variant agnes-t5gemma-9b-9b-ul2 --mirror --output-dir ./models/agnes-video/text_encoder
python scripts/download.py --variant agnes-wan2.2-ti2v-5b --output-dir ./models/agnes-video/vae
python scripts/download.py --variant agnes-omnivae --output-dir ./models/agnes-video/omni
git clone https://github.com/OpenMOSS/OmniVAE ./models/agnes-video/OmniVAE
git -C ./models/agnes-video/OmniVAE checkout d30e99d2d21f9461e394237ee98fc55ae0c011c2
```

The declared public T5Gemma mirror preserves the original pinned weights and
Gemma terms. Wan and Omni downloads select only the required VAE components.
The native audio path imports the DAC source directly; installation of OmniVAE's
full training or video dependency stack is not required. Configure the downloaded
audio checkpoint directory explicitly:

```toml
[model.params.family_params]
audio_vae_path = "./models/agnes-video/omni/models/vae/audio_only/recon"
audio_vae_code_root = "./models/agnes-video/OmniVAE"
```

## Training

The video dataset includes synchronized audio tracks and text captions.
The resident configuration holds all denoiser weights on the compute device;
the block-swap configuration keeps selected frozen blocks in host RAM.
Training requires standard gradient checkpointing when backward swapping is
enabled. Frozen-weight quantization and disk streaming are not declared
capabilities of this provider.

```bash
python scripts/train.py --config configs/agnes_video/train_bf16.toml --dry-run
python scripts/train.py --config configs/agnes_video/train_bf16.toml
```

## Inference

```bash
python scripts/infer.py --config configs/agnes_video/inference.toml --prompt "A small sailboat crossing a calm lake at sunset."
```

An adapter can be supplied through the standard inference adapter argument.
When `--negative-prompt` is omitted, inference uses the released Agnes
negative conditioning text. An explicit non-empty negative prompt overrides
that default; an empty prompt is rejected because the released T5Gemma
tokenizer produces no tokens for it.
The saved `.pt` latent is a versioned joint-media payload and retains both
modalities for later decoding.

Inference sessions cast frozen DiT parameters and floating buffers to
`model.dtype`, retaining FP32 trainable adapter matrices. With automatic
attention, each packed sequence uses the released plain FlashAttention-3 path
on Hopper, FlashAttention-2 otherwise, or SDPA when those modules are absent.
An explicit attention backend uses the shared backend policy. Top-1 sparse
experts use the released padded batched GEMM during no-grad inference; unmerged
LoRA deltas remain in activation space. Training uses the native expert loop.

## Released-weight validation

The executable contracts use a configured local release snapshot and CUDA:

```bash
python -m mirai.core.models.agnes_video.contracts.native_train_step --config ./train.toml --output ./validation/train.json
python -m mirai.core.models.agnes_video.contracts.native_inference --config ./inference.toml --output ./validation/inference.json --steps 32 --frames 61
```

For manifest-driven validation, set `MIRAI_AGNES_TRAIN_CONFIG` and
`MIRAI_AGNES_TRAIN_REPORT`, plus `MIRAI_AGNES_INFERENCE_CONFIG` and
`MIRAI_AGNES_INFERENCE_REPORT`, to the corresponding config and report paths.
Missing configuration or assets produces `incomplete`, not a successful check.
The training contract compares continuous and checkpoint-resumed optimization;
inference compares adapter scales zero and one and verifies the muxed media.

## Family configuration

Keys under `[model.params.family_params]`:

| Key | Default | Meaning |
| --- | --- | --- |
| `text_encoder_path` | `<model.path>/text_encoder` | Local T5Gemma encoder and tokenizer assets |
| `video_vae_path` | `<model.path>/vae/Wan2.2_VAE.pth` | Native Wan2.2 VAE checkpoint |
| `audio_vae_path` | `<model.path>/audio_vae` | OmniVAE audio checkpoint/config directory |
| `audio_vae_code_root` | `<model.path>/OmniVAE` | Directory containing `omnivae`, directly or under `vae/` |
| `fps` | `30.0` | Shared cache and output video frame rate |
| `allow_missing_audio` | `false` | Explicitly encode silence when training media has no audio |
| `audio_guidance_scale` | `5.0` | Audio CFG scale; audio/video solver histories are independent |
| `guidance_low_scale` | `2.0` | Video CFG scale after sigma reaches 0.5; earlier steps use the requested CFG scale |

Inference requests use positive frame counts and spatial dimensions divisible
by 32. Training cache frames retain the `4n+1` constraint.
The first frame is the timeline anchor: a 61-frame request at 30 fps spans
two seconds, generates 100 OmniVAE audio tokens, and exports 60 video frames
with 96,000 audio samples at 48 kHz. Single-frame requests retain one frame.
For a five-second request at 30 fps, use 151 internal frames: 38 latent frames
decode to 149 frames, then the last frame is repeated to export 150 frames
alongside 240,000 audio samples. The requested timeline is retained in joint
latent artifacts; version 1 artifacts remain readable.
Cache preprocessing resamples video to the configured frame rate and extracts
audio from the same selected interval. Encoder weights and source identities
are included in cache lineage. Native precomputed `.pt` inputs contain
`video_latents` `[48,T,H,W]` and `audio_latents` `[L,128]`.
