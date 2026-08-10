<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/mirai-logo-dark-transparent.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/mirai-logo.png">
    <img src="assets/mirai-logo.png" alt="Mirai" width="420">
  </picture>
</p>

<p align="center">
  Native single-GPU adapter trainer and inference runtime for dynamically routed
  sparse-MoE video diffusion models.
</p>

> [!NOTE]
> **Preview release.** Validate training and inference artifacts before
> production use.

## Install

Requirements: Python 3.10+, a CUDA GPU with compute capability SM80 or newer,
and sufficient host RAM and GPU memory for the selected frozen-weight
representation and residency policy.

```bash
git clone https://github.com/seruva19/mirai.git
cd mirai
python -m pip install -r requirements-cu126.txt
```

Mirai has one dependency set for its complete supported surface. The requirements
file selects the CUDA 12.6 Torch wheels and installs the project with that set;
platform markers select packages such as Triton where a platform-specific wheel
is required.

## Model support

- [LingBot Video](model_support/lingbot_video.md)
- [MAGI-2 Preview](model_support/magi2_preview.md)

Each model page documents its supported training and inference paths,
family-specific configuration, and native runtime capabilities.

## Training

Select an example from the supported model’s page, set the model, dataset, and
output paths, then validate and launch:

```bash
python scripts/train.py --config <training-config.toml> --dry-run
python scripts/train.py --config <training-config.toml>
python scripts/train.py --config <training-config.toml> \
  --resume <checkpoint-path>
```

Every configuration key is documented in
[`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md).

## Inference

Inference is model-provider driven and accepts an optional adapter checkpoint:

```bash
python scripts/infer.py \
  --config <inference-config.toml> \
  --adapter <adapter-path> \
  --prompt "prompt text" \
  --out outputs/clip.mp4
```

Prompt structure, supported solvers, and family-specific arguments are
documented on each model page.

## Features

The catalog below summarizes Mirai's capabilities. Many advanced features are
opt-in; experimental routing topologies are disabled by default. Their
configuration and compatibility rules are documented in
[`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md).

### Generic

- **Dataset bucketing** — Resolution and frame buckets.
- **Sequential text-encoder staging** — Inference can encode prompts before
  placing the denoiser, with optional INT8 or NF4 encoder weights, so the two
  components do not need to occupy VRAM concurrently.
- **Regional graph compilation** — Opt-in `torch.compile` targets repeated
  provider-declared transformer blocks, validates configurable token-shape
  buckets, supplies soft dynamic-shape hints, and reports compiled regions and
  graph counters in dry-run diagnostics. [(repo)](https://github.com/pytorch/pytorch)
- **Capability-gated attention backends** — A shared registry selects
  automatic PyTorch SDPA, forced cuDNN or PyTorch Flash Attention, and optional
  FlashAttention-3/4 fixed- or variable-length kernels. Packed `auto` execution
  falls back to an exact per-sequence PyTorch SDPA reference path when those
  optional kernels are absent. Explicit choices verify CUDA capability and
  package availability and fail instead of silently degrading; no backend
  speedup is claimed without workload-specific evidence.
  [(FA3 repo)](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)
  [(FA4 paper)](https://arxiv.org/abs/2603.05451)
  [(FA4 repo)](https://github.com/Dao-AILab/flash-attention)
- **Native cache pipeline** — Raw-video preprocessing and indexed latent/text
  caches.
- **Cache integrity** — Lineage, recovery, fingerprints, and sharded
  safetensors.
- **Dataset composition** — Train/validation/test splits and weighted sources.
- **Caption and temporal augmentation** — Caption variants, tag
  shuffling/dropout, temporal resampling, and curricula.
- **T2V, I2V, hybrid, and multi-task conditioning** — Registered strategies
  for fixed-task training and curriculum-selected homogeneous microbatches.
- **Progressive video curriculum** — Step-keyed resolution and frame stages can
  change the deterministic T2I:T2V:I2V sampling ratio. Every active task pool is
  validated before training; task selection derives from the restored step and
  RNG state, and dynamic flow shifting recomputes from each selected latent
  shape. [(paper)](https://arxiv.org/abs/2412.03603)
  [(paper)](https://arxiv.org/abs/2511.18870)
  [(paper)](https://arxiv.org/abs/2502.10248)
- **Training objectives** — Flow matching, regression, and opt-in recursive
  full-trajectory flow matching for saliency-guided MoE routing.
- **Contrastive flow matching** — Opt-in in-batch negative-flow repulsion with
  one model forward; requires at least two examples per microbatch.
  [(paper)](https://arxiv.org/abs/2506.05350)
  [(repo)](https://github.com/gstoica27/DeltaFM)
- **Latent wavelet supervision** — Opt-in FP32 spatial Haar loss on the
  reconstructed clean latent adds frequency-aware supervision without a VAE
  decode. Video frames remain independent in the transform.
  [(paper)](https://arxiv.org/abs/2604.12163)
- **Rectified-flow timestep sampling** — Deterministic uniform, logit-normal,
  and mode-shift distributions driven by checkpointed RNG state.
  [(paper)](https://arxiv.org/abs/2403.03206)
- **Token-count-aware flow shifting** — Opt-in square-root scaling derives the
  rectified-flow shift from visual patch-token count and uses the same
  post-shift noise coordinate for corruption, model conditioning, adapter
  timestep bands, and inference solvers.
  [(paper)](https://arxiv.org/abs/2403.03206)
- **Robust losses** — MSE, Huber, and pseudo-Huber.
- **Timestep loss weighting** — Uniform, Min-SNR, and cosine-map weighting.
- **Adaptive noise-level loss weighting** — An opt-in Fourier uncertainty head
  learns one log-variance `u(t)` over the rectified-flow noise coordinate and
  optimizes `L / exp(u) + u`. The head is shared across providers, included in
  checkpoint state, and its effective weights and log-variance range are
  reported during training. [(paper)](https://arxiv.org/abs/2312.02696)
  [(repo)](https://github.com/NVlabs/edm2)
- **Dispersive representation regularization** — Opt-in parameter-free
  InfoNCE-L2 repulsion over the physical mini-batch at a configurable
  transformer depth. Providers expose fixed-size video-token representations;
  Mirai evaluates the exact full ordered-pair objective, including its constant
  diagonal, in bounded feature chunks and requires at least two samples per
  microbatch. [(paper)](https://arxiv.org/abs/2506.09027)
  [(repo)](https://github.com/raywang4/DispLoss)
- **Loss controls** — Masks, prior preservation, noise offset, and bucket
  normalization.
- **Training-step safety** — Gradient accumulation, clipping, checkpointing,
  and non-finite-step handling.
- **Selective activation residency** — Operator-selective checkpointing,
  bounded pinned CPU offload, layer-aware defer/prefetch, and saved-view replay
  ([PyTorch DevLog](https://docs.pytorch.org/devlogs/distributed/2026-06-23-cpu-offloading/)).
- **Low-rank activation compression** — Opt-in approximate compression of
  eligible saved activations.
- **Asynchronous checkpoint serialization** — Single-flight immutable
  snapshots.
- **Optimizers** — AdamW, AdamW8bit, paged AdamW8bit, Prodigy, Adafactor, Lion,
  and CAME.
- **BF16 stochastic-rounding updates** — Opt-in AdamW, paired-LoRA, and
  selected-expert paths use unbiased mantissa dithering for BF16 parameter
  writes without persistent FP32 master weights. Supported BF16-moment paths
  apply the same unbiased writes to moment EMAs; FP32 and packed-low-bit
  moments retain their documented formats.
  [(paper)](https://arxiv.org/abs/2502.20566)
- **Learning-rate schedulers** — Constant, linear, cosine, cosine-restart,
  polynomial, and REX.
- **Training lifecycle and observability** — EMA, validation, early stopping,
  previews, TensorBoard, and Weights & Biases.
- **Checkpoint persistence and resume** — Adapter-only checkpoints, atomic
  artifacts, migrations, and replay-tested restoration of registered runtime
  scenarios rather than an exactness claim for every optional-feature
  combination.
- **LoRA lifecycle** — Targeting, rank/alpha patterns, rsLoRA, LoRA+, dropout,
  import, export, and sparse expert export. Runtime merge/unmerge is
  capability-gated and is not supported by the currently released provider.
- **Weight-decomposed low-rank adaptation (DoRA)** — DoRA trains independent
  output-channel magnitudes while LoRA updates each dense or grouped
  weight-space direction; native, Kohya, and Diffusers checkpoints preserve
  the magnitude state. [(paper)](https://arxiv.org/abs/2402.09353)
  [(repo)](https://github.com/NVlabs/DoRA)
- **Full-gradient-aligned LoRA optimization (LoRA-Pro)** — LoRA factor
  gradients are corrected in FP32 so their equivalent dense update follows
  the full-weight tangent projection. Adam moments are retained in equivalent
  weight space for dense and grouped-expert adapters.
  [(paper)](https://arxiv.org/abs/2407.18242)
  [(repo)](https://github.com/mrflogs/LoRA-Pro)
- **Momentum-anchored gradient projection** — Opt-in MAOP removes a globally
  antagonistic component from accumulated gradients using the existing FP32
  AdamW first moment, with bounded scratch memory and resumable policy state.
  [(paper)](https://arxiv.org/abs/2607.00293)
- **Gauge-invariant spectral LoRA optimization (LoRA-Muon)** — Whitened
  factor momentum receives a matrix-sign update in the geometry of the
  composed low-rank weight. Dense and grouped-expert adapters use split weight
  decay, first-moment-only state, and optional scalar gauge rebalancing.
  [(paper)](https://arxiv.org/abs/2606.12921)
- **Structured LoRA parameter dropout** — Independent Bernoulli masks over
  input-feature columns of A and output-feature rows of B; the rank axis is
  preserved. [(paper)](https://arxiv.org/abs/2404.09610)
- **Post-hoc adaptive LoRA rank compression** — PARA applies one global
  spectral rank or retained-energy budget across dense and per-expert trained
  LoRA matrices and writes a lineage-bound ragged adapter artifact.
  [(paper)](https://arxiv.org/abs/2604.27796)
- **Post-hoc power-function EMA** — Two or more CPU-FP32 adapter profiles and
  periodic adapter-only snapshots reconstruct arbitrary EMA response lengths
  after training without repeating optimization. Profile state resumes exactly,
  and snapshots contain adapter-owned state rather than frozen model weights.
  [(paper)](https://arxiv.org/abs/2312.02696)
  [(repo)](https://github.com/NVlabs/edm2)
- **LoRA interchange** — Kohya, Diffusers, LyCORIS, and ComfyUI layouts.
- **Inference solvers** — Native Euler, Flow-UniPC, and DPM++ 2M.
- **Classifier-free guidance** — Sequential execution and a capability-gated
  batched path with explicit variable-length prefix masks.
- **Batch inference sessions** — Per-request negative prompts, seeds, FPS, and
  decoding of saved latent outputs.
- **Conditioned inference** — Provider-owned T2I, text-plus-image-to-video, and
  video-to-video inputs share one conditioning contract. TI2V keeps its
  first-frame latent fixed throughout denoising; V2V truncates the solver
  schedule by an explicit denoising strength.
  [(DiffSynth PR)](https://github.com/modelscope/DiffSynth-Studio/pull/1545)
- **Expert-branch feature caching** — Opt-in cross-timestep reuse of per-routed-slot
  expert features during sampling. A layer's pre-combine expert outputs are held
  between visits; only the slots whose routed expert changed are recomputed, and
  reused features are re-weighted with the current routing probabilities. Input
  drift and an explicit reuse span bound how long an entry survives. Requires
  grouped expert execution, is lossy against the uncached path, and reports its
  per-layer reuse and invalidation counters.
  [(paper)](https://arxiv.org/abs/2606.15615)
- **Operational runtime controls** — Synthetic-step dry-run diagnostics,
  structured metrics/events, resource telemetry, and GPU leases. Dry-run still
  constructs the configured model and therefore requires its model assets and
  sufficient memory.
- **Interactive training controls** — Learning-rate finder, file/SQLite live
  control, and dataset compliance gates.

### MoE

- **Token-chunk MoE checkpointing** — Routes the complete batch once, then
  checkpoints bounded local expert-compute chunks to reduce routed-activation
  lifetime without changing routing topology.
  [(paper)](https://arxiv.org/abs/2511.21431)
- **Provider-declared expert execution** — Model providers declare expert tensor
  names, projection roles, activation, and combiner. Shared compressed execution
  supports gated three-projection and activated two-projection MLPs; specialized
  canonical-only transforms reject incompatible layouts explicitly.
- **Grouped adjugate experts** — Opt-in Grove MoE capacity expansion assigns
  one smaller trainable FFN to each explicit disjoint expert group without
  changing the native router. For a token, routes that land in the same group
  share one adjugate evaluation weighted by their summed gate mass. The output
  projection starts at zero for checkpoint-preserving upcycling; topology and
  weights round-trip with the adapter, and the configured scale is constrained
  by the paper's `scale <= groups / experts` bound.
  [(paper)](https://arxiv.org/abs/2508.07785)
  [(repo)](https://github.com/inclusionAI/GroveMoE)
- **Depth-aware router policy** — Non-overlapping layer bands independently
  override token-choice width, expert working-set fraction, and router z-loss.
  Uncovered layers retain the checkpoint-wide settings, and an empty policy
  constructs no runtime state.
  [(paper)](https://arxiv.org/abs/2505.22582)
  [(paper)](https://arxiv.org/abs/2402.08562)
  [(paper)](https://arxiv.org/abs/2605.19378)
- **Progressive lower-layer sparsification** — Early training can activate more
  experts in explicit lower routed-layer bands, then atomically restore each
  layer's target top-k at a configured step. The schedule is derived from the
  global or depth-aware target policy, remains training-only, and reports its
  active stage and mean route width.
  [(paper)](https://arxiv.org/abs/2512.16248)
  [(repo)](https://github.com/microsoft/ltp-megatron-lm)
- **Checkpoint-preserving Chain-of-Experts** — An opt-in two-step adaptation
  retains the native first expert pass, routes the inner-residual state through
  the same expert pool again, and gives the continuation step its own trainable
  low-rank router delta. A zero-initialized continuation scale preserves the
  pretrained prediction at construction. Both routing steps participate in
  auxiliary statistics, transition telemetry reports route retention and
  switching, and the added topology round-trips with the adapter. This adapter
  form does not claim the paper's scratch-pretraining fixed-compute equivalence.
  [(paper)](https://arxiv.org/abs/2506.18945)
  [(repo)](https://github.com/ZihanWang314/coe)
- **Scheduled balance-loss relaxation** — Auxiliary load-balancing pressure can
  be retained for an explicit exploration phase and disabled at one exact
  global step. The policy changes only the auxiliary balance coefficient:
  router z-loss and other objectives remain intact, and resume behavior is
  derived statelessly from the global step. This adapts DMEP's post-selection
  relaxation stage to native pretrained MoE models without claiming its
  adapter-expert pruning procedure.
  [(paper)](https://arxiv.org/abs/2604.26340)
- **Compressed frozen linears and routed experts** — Frozen dense and expert
  projections can use packed low-bit representations while adapters and router
  gradients remain trainable.
- **Mixed-precision expert storage** — A reviewed precision plan may assign a
  different packed representation to each physical expert while preserving one
  logical routed-expert interface, bounded storage accounting, and reference
  output and gradient checks.
  [(paper)](https://arxiv.org/abs/2505.05799)
- **Imatrix-calibrated per-tensor mixed-precision expert storage** — Routed
  input-square evidence measures every candidate representation independently
  for each module, physical expert, and w1/w2/w3 projection. A versioned plan
  assigns exact packed bytes under a hard ceiling without a padded
  highest-precision stack.
  [(repo)](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix)
  [(source)](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- **2:4 semi-structured frozen-expert execution** — Frozen projections retain
  exactly two values per contiguous group of four and use native
  semi-structured CUDA linear when supported, with dense reference math.
  [(repo)](https://github.com/pytorch/pytorch)
- **NF4 frozen-base training** — Blockwise NormalFloat storage reduces frozen
  weight memory while dequantizing only at the execution boundary.
  [(paper)](https://arxiv.org/abs/2305.14314)
- **Block-scaled FP8 frozen weights** — Frozen linears and routed experts use
  E4M3 weights with one FP32 scale per 128×128 tile and online E4M3 activations
  with one scale per token and 128 channels. The portable reference promotes
  every K=128 partial product into an FP32 accumulator; input gradients remain
  high precision rather than applying the unstable block-scaled Dgrad variant.
  [(paper)](https://arxiv.org/abs/2412.19437)
- **Native DeepGEMM FP8 experts** — When the SM90 and DeepGEMM capability probes
  succeed, routed FP8 expert projections execute as one M-grouped tensor-core
  GEMM while retaining Mirai's high-precision frozen-weight input-gradient and
  adapter-gradient path. The portable block-scaled implementation remains the
  reference.
  [(repo)](https://github.com/deepseek-ai/DeepGEMM)
  [(repo)](https://github.com/deepseek-ai/DeepSeek-V3)
- **GGUF low-bit storage** — IQ4_XS and IQ3_XXS artifacts provide compact,
  lineage-checked storage for frozen weights.
  [(repo)](https://github.com/ggml-org/llama.cpp)
- **Microscaling formats** — MXFP4 and NVFP4 reference formats expose
  block-scaled quantization experiments behind explicit configuration.
  [(paper)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- **MXFP8-E4M3 round-up microscaling** — Frozen linears and routed-expert
  projections can use 32-value E4M3 blocks with one UE8M0 scale per block.
  Conversion rounds `log2(amax / 448)` upward, saturates finite values, and
  applies round-to-nearest-even element encoding. Packed artifacts preserve the
  encoded bytes and scale exponents; the portable path dequantizes before GEMM,
  so the paper's Blackwell throughput and LLM-training accuracy results are not
  claimed for Mirai video training.
  [(paper)](https://arxiv.org/abs/2506.08027)
  [(specification)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- **Packed expert artifacts** — Quantized expert tensors, scales, shapes, and
  lineage are serialized as versioned artifacts whose packed payload restores
  losslessly; quantization itself is not claimed to preserve the original BF16
  weights losslessly.
- **Direct packed restore** — Compatible packed weights reconstruct the
  executable graph without first materializing the complete BF16 base.
- **Aligned expert shards** — Safetensors expert payloads can be aligned for
  bounded direct reads and deterministic shard addressing.
- **RAM-to-VRAM block swapping** — Transformer blocks move between host and
  device under an explicit synchronous or overlapped residency policy, for
  training runs and for sampling runs that stream an uncompressed base.
- **Explicit disk-backed block streaming** — Frozen tensors from non-resident
  transformer blocks are held in atomic safetensors-backed mappings and loaded
  through the same bounded residency schedule; trainable adapter state remains
  independently owned.
- **Routing-aware block residency** — Residency planning retains blocks using
  execution phase and observed routing heat under a configured declared-weight
  residency budget.
- **Bounded H2D residency ring** — Flat pinned buffers and bounded slots limit
  ring-owned in-flight expert transfers and reusable staging allocations.
- **Expert-aware activation checkpointing** — Checkpoint policies preserve
  routed-expert semantics while controlling recomputation and saved tensors.
- **Train-state CPU offload** — Activations, gradients, optimizer state, and
  trainable parameters can be offloaded independently with explicit ownership.
- **Re-dequantized expert backward** — Backward reconstructs frozen expert
  operands across registered dense, per-expert, batched, and grouped projection
  paths for input-gradient computation instead of retaining their dequantized
  forward tensors in autograd.
- **Paired and chunked dequantization** — Routed expert projections are
  reconstructed in bounded pairs or chunks per operation; this is not a bound
  on total process VRAM.
- **Routed-expert device cache** — A byte-bounded cache retains reconstructed
  INT8 experts while preserving explicit host/device ownership.
- **Unified residency ledger** — Block resident sets, transfer windows, and the
  expert-device cache reserve bytes in one configured ledger. It rejects
  declared overcommit but does not
  measure or bound activations, dequantization transients, allocator overhead,
  or total CUDA peak memory.
- **Hardware-tier planning** — Named hardware policies resolve conservative
  memory settings without changing feature defaults.
- **Explicit disk streaming** — Packed experts may stream from disk only when
  selected explicitly; disk is never an implicit offload fallback.
- **Bounded stream cache and prefetch** — Host caching and asynchronous
  prefetch use configured byte and depth limits.
- **GPUDirect Storage transport** — A capability-gated transport can place
  aligned packed reads directly into device-accessible buffers.
- **Vectorized token-choice dispatch** — Tokens are stably grouped by selected
  expert and restored to original order without dense expert execution.
- **On-device dispatch preprocessing** — Counts, offsets, and stable token
  order can be built on device to reduce route-dependent host synchronization.
- **Fused SonicMoE routing metadata** — An optional Hopper/Blackwell kernel
  builds expert counts, offsets, gather order, and inverse order on device while
  retaining Mirai's expert math, adapter path, and routing observers.
  [(paper)](https://arxiv.org/abs/2512.14080)
  [(repo)](https://github.com/Dao-AILab/sonic-moe)
- **Framework grouped matrix multiplication** — Native grouped GEMM is selected
  when the installed framework and tensor layout satisfy its contract.
  [(repo)](https://github.com/pytorch/pytorch)
- **Triton grouped GEMM** — Count-aware padded grouped kernels accelerate
  routed projections while retaining the reference dispatch path.
  [(repo)](https://github.com/triton-lang/triton)
- **Persistent grouped GEMM** — Sorted contiguous dispatch can use a
  cache-aware persistent kernel without padding inactive rows.
- **Persistent grouped-GEMM autotune warm-up** — An opt-in trainer-startup
  pass tunes provider-declared projection shapes before training activations
  occupy device memory and reuses Triton's device-local result cache.
  [(source)](https://github.com/woct0rdho/transformers-qwen3-moe-fused/pull/21)
- **Vendored fused MoE kernels** — Attributed native kernels provide fused
  forward and adapter-gradient paths behind capability checks.
  [(repo)](https://github.com/woct0rdho/transformers-qwen3-moe-fused)
- **Activation-rotated INT8 expert execution** — Frozen expert weights are
  quantized after orthogonal rotation; routed execution rotates activations,
  reconstructs FP32 GEMM operands from the stored INT8 tensors at the operation
  boundary, and preserves adapter gradients. This is compact INT8 storage, not
  a packed INT8 GEMM throughput claim.
  [(paper)](https://arxiv.org/abs/2404.00456)
- **Learned orthogonal expert quantization rotations** — Packed INT8 export can
  replace the fixed Hadamard basis with Cayley-parameterized group rotations
  learned directly from frozen expert weights. Gate/up share one transform so
  activation rotation remains hoistable; down uses its own transform. Hard
  reconstruction checkpoints cannot regress from the fixed-Hadamard baseline,
  optimization residency is bounded, and schema-v5 artifacts retain source
  fingerprints and the exact matrices used by reference dequantization and
  packed execution.
  [(paper)](https://arxiv.org/abs/2405.16406)
  [(repo)](https://github.com/facebookresearch/SpinQuant)
- **Packed INT8 expert operation** — For the canonical `w1`/`w3`/`w2` SwiGLU
  artifact layout, gate/up activation rotation, INT8-backed reference GEMM,
  scale epilogues, and active-expert LoRA share one bounded execution contract.
  Export and import reject other provider-declared layouts.
- **Independent grouped backward selection** — Forward and input-gradient
  grouped GEMMs may select different capability-checked backends.
- **Autograd-complete dispatch operations** — Routed permutation and
  duplicate-safe weighted combine have explicit inverse-gradient contracts.
- **Expert-Choice routing** — Experts select capacity-bounded token sets,
  providing an alternative to token-choice top-k routing.
  [(paper)](https://arxiv.org/abs/2202.09368)
- **Token-adaptive lightweight experts** — Opt-in zero, identity/copy, and
  learned constant-mixture routes compete with physical experts under one
  fixed-width top-k. Zero routes consume no output mass; all nonzero routes
  share the post-selection normalization. Constant routes learn
  `softmax(Wx)` mixtures of the token and an expert-specific vector. The
  null-aware balance objective treats identical zero slots as one frequency
  class.
  [(paper)](https://arxiv.org/abs/2406.13233)
  [(repo)](https://github.com/SJTU-DENG-Lab/AdaMoE)
  [(paper)](https://arxiv.org/abs/2607.24665)
- **Diffusion-aware decoupled routing** — Expert-Choice routes from the
  unmodulated normalized token state plus an independently learned projection
  of the raw timestep embedding, while expert MLPs retain the fully
  AdaLN-modulated input. The timestep projection is trained and persisted with
  the adapter.
  [(paper)](https://arxiv.org/abs/2604.12163)
  [(repo)](https://github.com/WithNucleusAI/Nucleus-Image)
- **Expert-Choice capacity schedules** — Layer and training-stage schedules
  progressively reduce capacity through validated monotone step bands without
  changing the disabled routing path.
  [(paper)](https://arxiv.org/abs/2410.02098)
- **Timestep-dependent Expert-Choice capacity** — A reverse-linear schedule
  assigns more expert capacity to low-noise samples. Sampler-CDF calibration
  supports uniform, logit-normal, and mode-shift timesteps; symmetric integer
  capacity offsets preserve the static selected-slot budget in expectation.
  [(paper)](https://arxiv.org/abs/2604.01622)
- **Expert-Choice coverage guard** — Detached mean and worst-layer token
  coverage telemetry raises a configurable alarm before capacity schedules
  silently strand tokens.
  [(paper)](https://arxiv.org/abs/2410.02098)
- **Drop-Upcycling expert splitting** — A data-free packed-artifact transform
  retains every source expert byte-for-byte, interleaves uniform physical
  copies, and statistically reinitializes one shared set of intermediate
  channels across each copy's gate, up, and down projections. Sibling router
  rows expand in the same topology; the artifact binds the source fingerprint,
  seed, ratio, masks, and module inventory. This is an opt-in adaptation for
  splitting an existing sparse expert, not a dense-to-MoE conversion.
  [(paper)](https://arxiv.org/abs/2502.19261)
  [(repo)](https://github.com/Taishi-N324/Drop-Upcycling)
- **Expert prototype consolidation** — Calibrated expert prototypes identify
  mergeable experts and preserve aliases in exported artifacts.
  [(paper)](https://arxiv.org/abs/2605.29350)
- **Output-hierarchical expert merging** — Every expert is evaluated on the
  same bounded deterministic calibration-token population. Euclidean distances
  between mean outputs drive deterministic average-linkage clustering, then
  routed-frequency weights form each physical merged expert while logical
  router rows remain available through aliases. The result is re-encoded
  through the source packed-format boundary; calibration, reviewed plan, and
  output manifest remain bound to the exact source fingerprint and module
  inventory.
  [(paper)](https://arxiv.org/abs/2410.08589)
  [(repo)](https://github.com/wazenmai/HC-SMoE)
- **Cross-expert shared-basis compression** — Expert weights are factorized
  into a shared basis plus expert-specific coefficients.
  [(paper)](https://proceedings.mlr.press/v267/li25az.html)
- **Optimized mixture-of-basis expert compression** — A data-free offline
  transform jointly fits expert-specific transformations, softmax mixture
  coefficients, and shared nonlinear bases for `w1/w3` by minimizing weight
  reconstruction error with Adam. Scale normalization is folded into the
  stored transformations, the original packed `w2` projection is preserved,
  and expert/row chunks plus explicit covariance and optimizer-memory ceilings
  bound conversion resources. The schema-v4 artifact records its exact source
  fingerprint and reconstructs only requested physical experts.
  [(paper)](https://arxiv.org/abs/2508.05257)
  [(repo)](https://github.com/inclusionAI/MoBE)
- **Truncation-aware whitened expert factorization** — Routed `w1/w3` inputs
  and post-SwiGLU `w2` inputs are summarized as bounded streaming covariance
  evidence. Shared-basis SVD is then performed in the whitened activation
  metric, supports either factor axis, and binds the result to the exact source
  artifact and calibration lineage.
  [(paper)](https://arxiv.org/abs/2403.07378)
  [(repo)](https://github.com/AIoT-MLSys-Lab/SVD-LLM)
- **Routing-aware quantization calibration** — Calibration weights error by
  expert affinity and observed routed activation statistics.
- **EAQuant-aligned router calibration** — A bounded offline pass calibrates
  symmetric per-output-channel INT8 router scales against the sum of logit MSE
  and reference-top-k probability divergence. The artifact binds dataset,
  model, config, router topology, and the exact floating-point source weights;
  runtime loading rejects any mismatch.
  [(paper)](https://arxiv.org/abs/2506.13329)
  [(repo)](https://github.com/darren-fzq1/EAQuant)
- **Expert-aware LoRA allocation** — Dense, router, shared-MLP, and rank-3
  expert-tensor targets use explicit allocation and export contracts.
- **Activation-space expert LoRA** — Adapter updates are fused into active
  expert execution without materializing dense expert deltas.
- **Routing-aware expert selection** — Calibration selects the experts whose
  adapter coverage best matches observed routing mass.
  [(paper)](https://arxiv.org/abs/2603.24044)
- **LoRA-FA adapters** — The A matrix remains fixed to reduce trainable state
  and activation memory while B receives updates.
  [(paper)](https://arxiv.org/abs/2308.03303)
- **LoftQ initialization** — Adapter initialization compensates low-bit base
  quantization error before training.
  [(paper)](https://arxiv.org/abs/2310.08659)
- **EVA and adaptive expert ranks** — Activation calibration allocates rank
  according to explained variance and an explicit global budget.
  [(paper)](https://arxiv.org/abs/2410.07170)
- **Gradient-driven adaptive expert LoRA (GoRA)** — A bounded pre-optimizer
  calibration pass scores target sensitivity, reallocates physical adapter
  ranks under the paper's smoothed budget, and pseudo-inverse-initializes the
  gradient-aligned factors. Grouped expert tensors retain one shared rank per
  fused tensor so routed execution remains vectorized.
  [(paper)](https://arxiv.org/abs/2502.12171)
  [(repo)](https://github.com/hhnqqq/MyTransformers)
- **PiSSA and LoRA-GA initialization** — Weight-principal or calibrated
  gradient subspaces initialize adapters while an exact frozen residual
  preserves the initial model function.
  [(paper)](https://arxiv.org/abs/2404.02948)
  [(paper)](https://arxiv.org/abs/2407.05000)
- **Fixed-support sparse high-rank deltas** — Persistent indices expose a
  small fraction of dense projection entries as a trainable full-rank update.
  [(paper)](https://arxiv.org/abs/2406.13175)
- **Direct selected-expert tuning with row-sparse optimizer state** — Selected
  dense expert rows update directly while optimizer state and checkpoints
  retain only the configured expert subset.
  [(paper)](https://arxiv.org/abs/2407.01906)
- **SOLO 4/2-bit selected-expert optimizer state** — The compact row-sparse
  AdamW path can persist its first moment as signed dynamic-exponent 4-bit
  codes and its second moment as unsigned logarithmic 2-bit codes. Both use
  128-value blocks with FP32 metadata; updates are evaluated in temporary FP32
  tensors and the exact state format is checkpoint-bound.
  [(paper)](https://arxiv.org/abs/2505.00347)
  [(repo)](https://github.com/MTandHJ/SOLO)
- **Adam-mini selected-expert neuron partitioning** — Directly tuned grouped
  expert projections retain a full first moment only for selected rows and one
  second-moment scalar per selected expert/output neuron. The partition is
  checkpoint-bound; its compact state round-trips with BF16 stochastic writes.
  [(paper)](https://arxiv.org/abs/2406.16793)
  [(repo)](https://github.com/zyushun/Adam-mini)
- **Matrix-geometry selected-expert optimization** — Muon applies an
  RMS-aligned polar update independently to every selected expert matrix;
  AdaMuon adds sign-stabilized orthogonal directions and an element-wise
  second moment. Unselected experts remain byte-identical, and both optimizers
  persist compact FP32 state for selected rows only.
  [(paper)](https://arxiv.org/abs/2502.16982)
  [(repo)](https://github.com/MoonshotAI/Moonlight)
  [(paper)](https://arxiv.org/abs/2507.11005)
  [(repo)](https://github.com/Chongjie-Si/AdaMuon)
- **ESFT task-affinity expert selection** — A deterministic pre-optimizer pass
  measures either average selected gate mass or token-selection ratio, then
  chooses the smallest cumulative-relevance expert set independently for every
  MoE layer. Routing remains unrestricted; only the selected rows may update.
  [(paper)](https://arxiv.org/abs/2407.01906)
  [(repo)](https://github.com/deepseek-ai/ESFT)
- **Adapter conditioning diagnostics** — Calibration records per-target
  condition number, stable rank, and allocation correlation.
  [(paper)](https://arxiv.org/abs/2506.16289)
- **Router-health release reports** — Step diagnostics consolidate into a
  versioned coverage report with explicit missing evidence.
- **Routing quantization agreement reports** — Per-layer evidence measures
  top-k expert-selection churn between original and INT8 router weights.
- **Train–inference routing agreement evidence** — An offline paired pass
  compares active expert sets for the same batch, sampled noise, timestep,
  token, layer, and router invocation in training and inference modes. Reports
  exact set churn, Jaccard, fixed-cardinality overlap/k, and route-cardinality
  changes without retaining raw token traces.
  [(R3 paper)](https://arxiv.org/abs/2510.11370)
  [(PR2 metric)](https://arxiv.org/abs/2606.00395)
- **MoE capacity estimator** — Expert storage, adapter state, unique-expert
  bounds, active parameters, host traffic, and memory pressure are estimated
  without allocating hardware.
- **Timestep-conditioned rank masking** — Adapter rank can vary over diffusion
  time using deterministic masks and checkpointed schedules.
  [(paper)](https://arxiv.org/abs/2507.05964)
- **Learned timestep rank gates** — Trainable gates select adapter rank as a
  function of the current diffusion timestep.
  [(paper)](https://arxiv.org/abs/2510.09561)
- **Always-on expert condenser** — A shared low-rank path complements sparse
  expert adapters even when individual experts are not selected.
  [(paper)](https://arxiv.org/abs/2604.23036)
- **Annealed expert working sets** — Per-layer expert subsets bound the unique
  expert working set and anneal back to full routing.
  [(paper)](https://arxiv.org/abs/2509.21892)
- **Native balance and router z-loss** — Auxiliary objectives regulate expert
  utilization and router-logit magnitude with independently gated weights.
  [(balance paper)](https://arxiv.org/abs/1701.06538)
  [(z-loss paper)](https://arxiv.org/abs/2202.08906)
- **Selective Sinkhorn routing** — An opt-in training policy replaces a small,
  deterministic fraction of native token-choice decisions with
  entropy-regularized maximum-cost optimal transport. Uniform token/expert
  marginals balance the transport plan; both top-k expert ids and normalized
  route weights come from that plan, with optional Gaussian cost noise. The
  ordinary training branch and all inference forwards retain native routing,
  and enabling the policy requires other balancing objectives to be disabled.
  [(paper)](https://arxiv.org/abs/2511.08972v2)
- **Residual prototypical routing (ProMoE adaptation)** — One learnable
  prototype per expert adds cosine-similarity guidance to the pretrained native
  router for provider-selected visual tokens. A learned scalar starts at zero,
  so expert ids and gate weights match the checkpoint router at construction;
  after training, the learned route applies during inference. The paper's
  routing contrastive loss aligns every active prototype with the mean of its
  assigned top-k token set while separating active experts. Mirai implements
  this as a checkpoint-preserving PEFT adaptation, not the paper's from-scratch
  replacement router or its conditional/unconditional partitioning stage.
  [(paper)](https://arxiv.org/abs/2510.24711)
  [(repo)](https://github.com/ali-vilab/ProMoE)
- **Saliency-harnessing trajectory routing (SharpMoE adaptation)** — A
  zero-output two-layer SiLU router adds scores derived from the preceding
  predicted clean video latent to each pretrained sparse router. Training uses
  the paper's recursive descending-timestep rollout and detaches each clean
  prediction before it guides the next route; inference carries the same state
  through sequential or batched CFG. The fixed-top-k implementation excludes
  the paper's trajectory-allocation KL term because that term is explicitly not
  applied to fixed-width token-choice DiTs. The policy is opt-in, preserves
  native routes at initialization, and stores both router state and rollout RNG.
  [(paper)](https://arxiv.org/abs/2606.26938)
- **Attention-routed Mixture-of-Depths (A-MoD adaptation)** — Selected blocks
  use the preceding dense block's mean received-attention scores to process an
  exact per-sample top-capacity subset of visual tokens. Conditioning tokens
  remain present in every routed block, omitted visual tokens follow the
  identity residual path, and the gathered token count is fixed by capacity.
  The parameter-free route is shared by training and inference, supports packed
  batches, and alternates with dense blocks so every routed decision has fresh
  attention evidence. Mirai uses the bidirectional A-MoD rule and therefore
  does not add the causal auxiliary router proposed for autoregressive MoD.
  [(paper)](https://arxiv.org/abs/2404.02258)
  [(paper)](https://arxiv.org/abs/2412.20875)
- **Similarity-preserving router balancing (SIMBAL)** — An opt-in,
  batch-independent auxiliary objective applies the paper's entrywise L1
  penalty `||W W^T - I||_1` to every effective router matrix. Under PEFT the
  frozen pretrained matrix remains immutable and gradients flow through its
  router adapter; Mirai does not apply the paper's from-scratch orthogonal
  initialization or transfer its language-model convergence claims.
  [(paper)](https://arxiv.org/abs/2506.14038)
- **Auxiliary-loss-free balancing** — Online router bias updates correct load
  imbalance without adding an auxiliary gradient objective.
  [(paper)](https://arxiv.org/abs/2412.19437)
- **Domain-labelled routing testbed** — An offline, lineage-bound evaluator
  compares observed per-layer routes with an explicit domain-to-expert reference
  and reports assignment accuracy, regret, reference coverage, expert purity,
  normalized mutual information, and per-domain breakdowns without changing the
  model or training path. Run `scripts/tools/evaluate_domain_routing.py` with an
  observations JSON (`num_experts` and per-layer `{domain, selected_experts}`
  records), a reference JSON (`domains` mapped to expert-ID lists), and the
  dataset/model fingerprints; the output is deterministic versioned evidence.
  [(paper)](https://arxiv.org/abs/2604.07030)
- **Accumulation-wide routing balance** — Expert statistics aggregate across
  the complete gradient-accumulation window rather than one microbatch,
  including models whose native auxiliary form is sequence-local.
  [(paper)](https://arxiv.org/abs/2501.11873)
- **Balance-to-task router gradient diagnostics** — Opt-in telemetry compares
  each weighted balancing objective with the task gradient on batch-collapsed
  expert-probability vectors, including a configurable dominance alarm.
  [(paper)](https://arxiv.org/abs/2509.01322)
- **Pairwise expert-combination balancing** — Router-probability correlations
  are weighted by observed expert co-selection, exposing and penalizing pair
  collapse that marginal expert utilization cannot detect.
  [(paper)](https://arxiv.org/abs/2503.16057)
- **Compute-budgeted dynamic top-k** — Route cardinality varies with token
  uncertainty under explicit minimum and mean-compute bounds while fixed-width
  dispatch tensors and gate mass remain unchanged.
- **Spatiotemporal routing consistency** — Router distributions can be
  regularized across bounded samples of adjacent video-patch tokens; model
  providers supply grid geometry and exclude conditioning tokens.
  [(paper)](https://arxiv.org/abs/2505.00792)
- **FP32 router master parameters** — Router updates are accumulated in FP32
  before synchronizing back to the execution dtype.
- **Frozen-router INT8 storage** — Immutable router matrices can use symmetric
  per-output-channel INT8 storage while routing logits are reconstructed at the
  execution boundary; trainable-router combinations fail explicitly.
- **Routing health telemetry** — Entropy, collapse, drift, depth-banded
  deadlock, underflow, and expert utilization are emitted as structured
  diagnostics.
  [(paper)](https://arxiv.org/abs/2605.19378)
- **Mechanism-driven preemptive monitoring** — The routing-health pack reads
  full-softmax per-token entropy, exact pairwise router-weight similarity, and
  centered conditioning before discrete top-k counts need to change. For
  static Q/K LoRA projections it also tracks the alpha-2 spectral entropy and
  effective rank of the exact first-order QK-product update through small QR
  cores. Update-window state resumes exactly; Mirai reports trajectories and
  does not invent an architecture-independent alarm threshold.
  [(paper)](https://arxiv.org/abs/2606.28116)
- **Fisher-Rao specialization telemetry** — The opt-in routing-health pack
  reports the exact Fisher-Rao distance between each layer's empirical marginal
  expert distribution and uniform routing, together with its normalized
  distance to the simplex boundary. Mirai intentionally does not expose the
  source paper's FHS threshold because its printed equation and proof specify
  different heterogeneity statistics.
  [(paper)](https://arxiv.org/abs/2604.14500)
- **Selection-margin telemetry** — Optional diagnostics report how close each
  token's routing decision sits to the top-k selection boundary.
- **Expert-touch guard** — Post-routing checks enforce configured bounds on the
  unique expert working set before expensive expert execution.
- **Phi-balancing** — A population-level potential shapes soft routing mass
  using a detached moving estimate of expert utilization.
  [(paper)](https://arxiv.org/abs/2605.15403)
- **Router variance regularization** — A gated objective discourages collapsed
  routing distributions while preserving the zero-weight path.
- **Expert-output orthogonality** — Co-activated expert outputs are encouraged
  to represent distinct directions using bounded sampled observations.
  [(paper)](https://arxiv.org/abs/2505.22323)
- **SwiGLU expert specialization** — Co-activated expert intermediates receive
  a similarity penalty without repeating expert execution.
  [(paper)](https://arxiv.org/abs/2602.14159)
- **Cross-layer routing coupling** — Adjacent routing distributions can be
  regularized through a bounded cross-layer consistency objective.
  [(paper)](https://arxiv.org/abs/2602.14159)
- **Diversity-aware routing** — Selection can penalize redundant co-activation
  while keeping deterministic top-k behavior under fixed seeds.
  [(paper)](https://openreview.net/forum?id=7FsbfQgti4)
- **Router temperature schedules and jitter** — Constant, linear, or sigmoid
  logit-temperature schedules can stop at an observed assignment-entropy floor;
  optional multiplicative jitter is training-only and checkpoint-replayable.
  [(paper)](https://arxiv.org/abs/2605.15484)
  [(paper)](https://arxiv.org/abs/2202.08906)
- **No-token-drop expert-output dropout** — Selected expert contributions are
  stochastically masked during training. At least one selected expert remains
  per token, and surviving gate weights are renormalized to preserve routed
  mass without discarding tokens.
  [(paper)](https://arxiv.org/abs/2606.31201)
- **Staged router adaptation** — Router trainability and optimizer membership
  change through checkpoint-safe, explicitly scheduled stages.
  [(paper)](https://arxiv.org/abs/2509.16882)
- **Router distillation** — A frozen teacher snapshot constrains trainable
  routing decisions without sharing mutable state.
- **Dataset-domain specialization** — Dataset metadata can apply soft affinity
  priors or hard expert eligibility masks per sample.
- **Structured expert pruning** — Lineage-bound offline calibration can rank
  experts by route frequency, REAP gate-weighted activation norms, MAN mean
  activation norms, or MSAN mean-squared activation norms. Deterministic keep
  sets preserve the configured top-k floor and re-emit expert tensors plus
  router rows as a smaller packed artifact.
  [(REAP paper)](https://arxiv.org/abs/2510.13999)
  [(REAP repo)](https://github.com/CerebrasResearch/reap)
  [(MAN/MSAN paper)](https://arxiv.org/abs/2606.15716)
  [(MAN/MSAN repo)](https://github.com/ZongfangLiu/unified-expert-pruning)
- **Calibration-free AIMER expert pruning** — A weight-only offline criterion
  streams bounded dense projection blocks from the selected packed artifact and
  scores each expert as the absolute mean divided by RMS across its combined
  gate, up, and down weights. Higher AIMER scores are treated as more removable;
  deterministic per-layer keep sets preserve the configured top-k floor,
  require no dataset or router observations, and persist the exact packed-source
  fingerprint, raw scores, and kept expert IDs in the output manifest. Mirai
  does not transfer the paper's language-model quality results to video.
  [(paper)](https://arxiv.org/abs/2603.18492)
  [(repo)](https://github.com/ZongfangLiu/AIMER)
- **Structured-then-semi-structured expert compression** — An offline,
  bounded-memory transform clusters experts from router-row similarity,
  retains or reconstructs one centroid representative per cluster, and stores
  the reduced projections in a compact quantized 2:4 physical-weight provider
  with on-demand dense decode. The transform fails unless its expert payload is
  smaller than the source. This is a semi-structured adaptation of STUN; the
  paper's second stage is unstructured Wanda/OWL pruning.
  [(paper)](https://aclanthology.org/2025.acl-long.671/)
- **Nested intra-expert width pruning** — FlexMoE calibration ranks linked
  `w1/w3` rows and `w2` columns from Taylor gradients, then learns one discrete
  prefix width per expert with straight-through Gumbel actions. The video
  adaptation combines the native task loss with full-width teacher-prediction
  MSE; lineage-bound ranking and action artifacts produce physically ragged
  rowwise-INT8 experts executed in equal-width buckets. All calibration
  schedules are explicit, and no language-model quality result is transferred
  to video. [(paper)](https://arxiv.org/abs/2606.27866)
- **Post-compression Router-KD repair** — An offline teacher/student pass
  updates only FP32-master router weights in a compressed model by matching the
  original model's final diffusion prediction on exactly replayed calibration
  inputs. A held-out non-regression gate and complete non-router state
  fingerprint must pass before a lineage-bound router-only patch is emitted;
  teacher and student expert counts may differ.
  [(paper)](https://arxiv.org/abs/2603.02217)
- **Sparse expert adapter checkpoints** — Exports can contain only selected
  expert adapter rows together with the mapping required for restoration.

## License

Apache-2.0. Vendored runtime source retains its upstream attribution and license
notices.
