# MAGI-2 Preview

Mirai supports adapter training for SandAI's
[MAGI-2 Preview](https://huggingface.co/sand-ai/MAGI-2-preview) preview-stage
transformer through its native Apache-2.0 runtime. The 114B-parameter model
activates 6B parameters per token; its 40 transformer layers contain 36
multi-head sparse-MoE layers with 256 experts and top-6 routing per head.

The reference runtime uses eight Hopper GPUs. Mirai's single-GPU path keeps
the frozen checkpoint in host RAM and transfers transformer blocks to one GPU
on demand. This requires substantial host memory and fast host-to-device
bandwidth; it does not reduce the checkpoint's roughly 228 GB preview-stage
storage requirement.

## Model-specific features

- **MAGI-2 Preview single-GPU adapter training** — Adapter-only optimization
  with native preview-transformer weights and gradients.
- **Host-resident MAGI-2 block streaming** — Frozen transformer blocks move to
  the execution device only for their forward/backward window.
- **Native MAGI-2 inference** — The provider uses the same native denoiser and
  residency path for sampling without a Diffusers model dependency.
- **Grouped MAGI-2 expert execution** — Optional grouped-GEMM execution of the
  multi-head routed experts during training, selected by
  `memory.moe_kernel_backend="grouped"`.

## Training

Install the family-specific dependencies with
`pip install -e ".[magi2-preview]"`. This keeps MAGI's pinned text-encoder
runtime independent from Mirai's other model families.

Download the official snapshot so `model.path` contains `preview/`, then use
[`configs/magi2_preview/train_offload.toml`](../configs/magi2_preview/train_offload.toml).
The supported adapter presets are `attn_only` and `attn_router`. Block swapping
is required when the checkpoint does not fit in device memory.

Training executes the multi-head routed experts through the vendored per-expert
reference loop by default. `memory.moe_kernel_backend="grouped"` replaces it
with grouped execution: every `(head, expert)` pair is flattened onto the packed
expert axis, routed token slices are sorted once, and the gate/up/down
projections run as grouped GEMMs. Expert weights stay frozen — the grouped path
raises if `W_gate`, `W_up`, or `W_down` requires a gradient — while routing,
activation, and the probability-weighted combine remain in native autograd.
`memory.moe_gemm_backend` (and its `_forward` / `_dx` role overrides) selects the
grouped primitive: `auto` uses `torch_grouped` where the torch build and device
provide it and `bmm` otherwise. `persistent` and `deepgemm_fp8` are rejected, as
is every other MoE memory policy field held at a non-default value: those four
keys and `memory.moe_kernel_backend` are the only ones this family consumes. A
backend the torch build cannot provide is rejected when the policy is
configured; the device architecture gate applies once the execution device is
known.

The optional accelerated runtime uses the pinned MagiAttention, MagiCompiler,
and FlashAttention packages described by the upstream release. Mirai retains a
differentiable Torch reference path for training and environments where those
kernels are unavailable. Triton, `pydantic-settings`, and `unfoldNd` are still
runtime dependencies.

## Inference

The same native denoiser and block-residency path is used for inference. The
preview transformer can run on one GPU with host-resident block streaming.
Prompt-to-video output additionally requires the official Qwen3.5 text encoder
and VAE assets from the same snapshot. Throughput and peak-memory figures are
published only after the GPU validation contract records them.

Use [`configs/magi2_preview/inference_offload.toml`](../configs/magi2_preview/inference_offload.toml)
with `scripts/infer.py`; text encoding, denoising, and VAE decoding occupy the
execution device sequentially by default.
