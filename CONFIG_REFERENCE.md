# Mirai config reference

The TOML config file is the single entry point to a training run. Every
behaviour toggle is a key documented here; there are no hidden constructor
arguments. Environment variables exist only as operational overrides for a small
perf/infra subset (see the appendix), never as the primary control surface.

This file is kept exhaustive by
[`tests/test_config_reference_coverage.py`](tests/test_config_reference_coverage.py),
which asserts the documented key set equals
[`mirai.config.schema.all_config_keys()`](mirai/config/schema.py) in
both directions. An undocumented key or a documented-but-nonexistent key fails
the suite.

## Merge order

[`load_config`](mirai/config/loader.py) resolves a run in three layers,
deep-merged left to right (later wins):

```
defaults/<family>.toml  →  presets/<preset>.toml  →  your config.toml
```

- The **defaults** file is chosen by the resolved model family's
  `config_defaults_name` (falls back to `defaults/moe.toml`). It is loaded **only
  when a `preset` is set**; a preset-less config is parsed as-is with dataclass
  defaults filling the rest.
- The **preset** is `presets/<preset>.toml`, selected by the root `preset` key.
- Your file wins every conflict.

## Validation — three layers

Validation is deliberately split across three modules; a key can be rejected at
any layer.

1. **Shape / whitelist / coercion** —
   [`mirai/config/schema.py`](mirai/config/schema.py). Every table has a
   closed allowed-key set; an unknown key is a hard error with a "did you mean"
   hint. Example: `[training] batch_szie = 2` → `Unknown config key
   'training.batch_szie'. Did you mean 'batch_size'?`. Types are coerced here and
   path strings are rejected if they contain control characters.
2. **Cross-key contract rules** —
   [`mirai/core/training/runtime/contract.py`](mirai/core/training/runtime/contract.py).
   Numeric ranges, enum membership, and cross-key incompatibilities. Example:
   `training.activation_cpu_offload = true` with `training.compile = true` is
   rejected before runtime.
3. **Model-capability gating** —
   [`mirai/core/training/runtime/trainer.py`](mirai/core/training/runtime/trainer.py).
   Per-family
   rejection of memory features the resolved provider does not implement.
   Example: `memory.frozen_weight_packed_state_path` requires
   `frozen_weight_quantization = "int8"` **and** the family to declare the
   `packed_frozen_weight_state` capability; otherwise it is rejected before model
   construction.

Legend used below: **opt-in** = default is off/false/none/0/disabled;
**on** = active by default. "rc:" points to
[`runtime/contract.py`](mirai/core/training/runtime/contract.py); "tr:" points
to [`runtime/trainer.py`](mirai/core/training/runtime/trainer.py).

---

## Presets

`preset = "<name>"` selects `mirai/config/presets/<name>.toml`. Keys each preset
sets (on top of the family defaults file):

### `defaults/moe.toml` (shared MoE-video defaults, loaded before a family preset)

- `[model]`: type, path, dtype, attention_backend, hash_snapshot_contents
- `[model.params]`: variant, flow_shift, vae_chunk_size, strict_native_assets,
  num_experts, experts_per_token, shared_experts, hidden_size, num_layers,
  attention_heads, latent_channels, patch_size, expert_capacity_factor,
  moe_aux_loss_weight, moe_router_z_loss_weight
- `[strategy]`: type
- `[training]`: batch_size, gradient_accumulation, max_steps, objective,
  loss_function, gradient_checkpointing
- `[optimizer]`: type, lr, weight_decay
- `[adapter]`: type, target_preset, rank, alpha
- `[dataset]`: path, cache_path, frame_buckets
- `[logging]`: output_dir, save_every_n_steps, sample_every_n_steps
- `[memory]`: frozen_weight_quantization,
  frozen_weight_quantization_strategy, weight_residency_strategy,
  expert_weight_access, expert_dequant_chunk_size, quantize_experts_on_load,
  router_quantization, moe_kernel_backend, moe_expert_autograd,
  moe_activation_backend

Model-family preset inventories and their family-specific behavior belong in
the corresponding model-support reference.

---

## `[root]`

### `preset`

- **Type:** str?
- **Default:** none

Selects `presets/<preset>.toml` and triggers loading of the family defaults file
([`loader.py`](mirai/config/loader.py), lines 65–77). Omitting it parses your file
with dataclass defaults only.

## `[model]` — ModelConfig

### `type`

- **Type:** str
- **Default:** `"lingbot-video"`
- **Allowed / range:** must resolve to a registered sparse-MoE provider ([`runtime_policy.py`](mirai/config/runtime_policy.py))

Model-family selector; drives provider + defaults file ([`providers.py`](mirai/core/models/providers.py), [`loader.py`](mirai/config/loader.py)). Top-level `model.variant` is explicitly rejected — use `model.params.variant`.

### `path`

- **Type:** str
- **Default:** `"./models/lingbot_video"`
- **Allowed / range:** no control chars

Checkpoint root dir ([`base.py`](mirai/core/models/base.py), active model provider).

### `dtype`

- **Type:** str
- **Default:** `"bf16"`
- **Allowed / range:** `bf16`, `bfloat16`, `fp16`, `float16`, `f16`, `half`,
  `fp32`, `float32`, `f32`

Compute/param dtype ([`base.py`](mirai/core/models/base.py), models/*).

### `attention_backend`

- **Type:** str
- **Default:** `"auto"`
- **Allowed / range:** `auto`, `cudnn`, `flash`, `flash3`, `flash4`, `flex`

Model-agnostic attention kernel policy. `auto` uses PyTorch SDPA for
dense/masked attention; for packed variable-length attention it selects an
available FA4→FA3 kernel, then falls back to an exact per-sequence PyTorch SDPA
reference path. `cudnn` and `flash` force the corresponding PyTorch SDPA
backend; `flash3` and `flash4` require the official optional packages and
compatible CUDA capability. `flex` selects PyTorch FlexAttention, which has a
backward pass and therefore serves training as well as inference; packed
variable-length execution builds a document block mask so samples stay
isolated, and MAGI-2 additionally reproduces its per-head attention sinks on
that path ([`flex_attention.py`](mirai/core/models/magi2_preview/flex_attention.py)).
Explicit backends accept maskless attention only and fail rather than silently
falling back.

### `hash_snapshot_contents`

- **Type:** bool
- **Default:** `false`

Controls model-root lineage fingerprinting. `false` hashes the relative path,
byte size, and nanosecond modification time of each file without reading model
payload bytes. `true` reads every file and records a full content SHA-256 tree.
A validated `registration.json` or `download_manifest.json` remains the
preferred constant-time identity in either mode. Dataset, cache, and config
lineage retain their existing content-hash behavior; this key affects only
`model.path` ([`lineage.py`](mirai/core/lineage.py),
[`session_context.py`](mirai/core/training/lifecycle/session_context.py)).

### `provider_module`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** no control chars

Import path for an out-of-tree model provider module ([`providers.py`](mirai/core/models/providers.py), [`loader.py`](mirai/config/loader.py)).

## `[model.params]` — ModelParams

These are sparse-MoE / DiT architecture fields. Keys in this table are
interpreted by the active model family provider; support varies by family and is
validated by capability gating. Consumer citations below name the agnostic core
seam that owns the contract (for example,
[`specs.py`](mirai/core/moe/runtime/specs.py)), not any one family module.

Out-of-tree providers may define typed options under
`[model.params.family_params]`. Core preserves this table and delegates its
validation to `ModelFamilyProvider.validate_family_params`; the base provider
rejects non-empty values so unsupported options cannot be silently ignored.

### `family_params`

- **Type:** table
- **Default:** `{}`
- **Allowed / range:** provider-defined; rejected by the base provider

Provider-owned extension parameters. Values are preserved by the core config
loader and validated by the active model-family provider. The keys a shipped
family accepts under this table are listed on its page in
[`model_support/`](model_support/).

### `variant`

- **Type:** str
- **Default:** `"lingbot-video-moe-30b-a3b"`
- **Allowed / range:** provider-defined

Released architecture identifier. LingBot topology is loaded from the snapshot configuration.

### `flow_shift`

- **Type:** float
- **Default:** `3.0`
- **Allowed / range:** finite `> 0`

Constant flow-matching timestep shift, or the shift at `flow_shift_base_seq_len` when dynamic shifting is enabled.

### `flow_shift_mode`

- **Type:** str
- **Default:** `"constant"`
- **Allowed / range:** `constant`, `dynamic`

`constant` applies the configured scalar shift consistently to corruption, model conditioning, timestep-aware adapters, and preview/inference. `dynamic` derives one shift from each model-family-declared visual patch-token count and uses the same post-shift noise coordinate across those paths.

### `flow_shift_base_seq_len`

- **Type:** int
- **Default:** `256`
- **Allowed / range:** `> 0`

Visual patch-token count `n` at which dynamic shifting equals `flow_shift`. Ignored in constant mode.

### `flow_shift_max_seq_len`

- **Type:** int
- **Default:** `4096`
- **Allowed / range:** `>= flow_shift_base_seq_len`

Upper token-count clamp `m`; larger visual sequences reuse `flow_shift_max`. Ignored in constant mode.

### `flow_shift_max`

- **Type:** float
- **Default:** `12.0`
- **Allowed / range:** finite `> 0`; dynamic: `flow_shift * sqrt(flow_shift_max_seq_len / flow_shift_base_seq_len)`

Explicit upper anchor and validation guard for the square-root shift law from Equation 23 of arXiv:2403.03206.

### `vae_chunk_size`

- **Type:** int
- **Default:** `16`
- **Allowed / range:** —

VAE temporal chunk for encode/decode ([`preview.py`](mirai/core/training/preview/preview.py)).

### `lora_dropout`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** —

Dropout on the LoRA path (active model provider).

### `strict_native_assets`

- **Type:** bool
- **Default:** `True`
- **Allowed / range:** —

Require complete native model assets and strict checkpoint loading.

### `inference_routing_telemetry`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Capture detached top-k routing assignments during inference for an explicit routing-trace output. Disabled mode stores no trace.

### `inference_routing_telemetry_layer_stride`

- **Type:** int
- **Default:** `1`
- **Allowed / range:** `> 0`

Retain every Nth router layer in inference routing traces.

### `moe_expert_backend`

- **Type:** str
- **Default:** `"grouped_mm"`
- **Allowed / range:** `grouped_mm`, `loop`, `sglang_triton`

LingBot routed-expert execution backend. `sglang_triton` requires the optional SGLang MoE runtime and compatible CUDA hardware; explicit unavailability fails.

### `moe_pad_backend`

- **Type:** str
- **Default:** `"auto"`
- **Allowed / range:** `auto`, `loop`, `vectorized`

LingBot grouped-token padding implementation. `auto` uses the eager loop for training and the vectorized path for inference or compiler tracing.

### `moe_reorder_backend`

- **Type:** str
- **Default:** `"sort"`
- **Allowed / range:** `sort`, `triton_pack`

LingBot token-to-expert reorder implementation. `triton_pack` requires Triton and CUDA tensors.

### `moe_restore_backend`

- **Type:** str
- **Default:** `"chunked_scatter"`
- **Allowed / range:** `scatter`, `chunked_scatter`, `triton`

LingBot expert-output restore implementation. `triton` requires Triton; `chunked_scatter` bounds temporary restore storage.

### `moe_restore_chunk_size`

- **Type:** int
- **Default:** `128`
- **Allowed / range:** `> 0`

Route rows restored per chunk by `chunked_scatter`.

### `moe_fused_qkv_linear`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Enables the LingBot fused QKV projection when its tensor-layout contract is satisfied.

### `inference_bf16_fastmath`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Enables LingBot's optional BF16 inference fast-math path; training math is unchanged.

### `num_experts`

- **Type:** int
- **Default:** `4`
- **Allowed / range:** —

Expert count ([`specs.py`](mirai/core/moe/runtime/specs.py), models/*).

### `experts_per_token`

- **Type:** int
- **Default:** `2`
- **Allowed / range:** —

Top-k routing width (models/*).

### `shared_experts`

- **Type:** int
- **Default:** `1`
- **Allowed / range:** —

Always-on shared experts (models/*).

### `hidden_size`

- **Type:** int
- **Default:** `32`
- **Allowed / range:** —

Model width (models/*).

### `num_layers`

- **Type:** int
- **Default:** `2`
- **Allowed / range:** —

Depth (models/*).

### `attention_heads`

- **Type:** int
- **Default:** `4`
- **Allowed / range:** —

Head count (models/*).

### `latent_channels`

- **Type:** int
- **Default:** `1`
- **Allowed / range:** —

VAE latent channels (models/*, [`preview.py`](mirai/core/training/preview/preview.py)).

### `patch_size`

- **Type:** int
- **Default:** `1`
- **Allowed / range:** —

Patchify size (models/*).

### `expert_capacity_factor`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** —

Token capacity per expert (active model provider).

### `moe_routing_mode`

- **Type:** str
- **Default:** `"token_choice"`
- **Allowed / range:** `token_choice`, `expert_choice`

Selects the native token-choice router or config-driven Expert-Choice routing. The default preserves released checkpoint routing. Expert-Choice uses the pretrained router projection but changes dispatch to a capacity-bounded expert→token assignment.

### `moe_chain_of_experts`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Enables the model-agnostic two-step Chain-of-Experts adaptation. The first token-choice pass is unchanged. The second pass routes the inner-residual state `x + E₁(x)` through the same shared and routed expert pool using the native router plus an independent trainable low-rank delta; its output is added through a zero-initialized learned continuation scale. The prediction therefore equals the pretrained single-pass result at construction, while both passes contribute routing auxiliary statistics during training. State and exact topology are stored with the LoRA adapter. This adaptation executes an additional expert pass and does not promise the paper's fixed-compute comparison. It cannot compose with Expert-Choice, lightweight experts, the single-pass spatiotemporal routing objective, or non-LoRA adapters.

### `moe_chain_router_rank`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `0` while disabled; enabled: `> 0` and no greater than the provider's native hidden size or expert count

Rank of the second pass's trainable router delta. Only the continuation router owns these factors; the pretrained first-pass router remains governed by the ordinary adapter/router controls. Mirai fixes the communication-step count at two because the source ablation supports that setting and does not expose unsupported deeper chains.

### `moe_layer_router_policy`

- **Type:** array of tables
- **Default:** `[]`
- **Allowed / range:** non-overlapping half-open layer bands

Optional depth-aware router settings. Every band requires `first_layer` and `end_layer` and at least one of `top_k`, `subset_fraction`, or `z_loss_weight`. Unspecified fields inherit their global `experts_per_token`, `expert_subset_fraction`, or `moe_router_z_loss_weight` value. `top_k` and `subset_fraction` are token-choice-only; the resolved subset must contain at least `top_k` physical experts. Per-layer `top_k` cannot compose with lightweight experts or the fixed-width cross-layer coupling objective. The empty default constructs no policy and preserves uniform checkpoint routing. This is a configurable video-MoE adaptation of the depth heterogeneity reported by LayerMoE, MoLA, and visual-DiT routing diagnostics, not a claim that those papers define one joint schedule.

### `moe_progressive_sparsification_transition_step`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `0` disables; enabled: `> 0` with a non-empty progressive policy

First training step that restores every routed layer to its target top-k. The target is resolved from `moe_layer_router_policy` or `experts_per_token`. The global step is authoritative, so resume crosses the same boundary without separate mutable scheduler state.

### `moe_progressive_sparsification_policy`

- **Type:** array of tables
- **Default:** `[]`
- **Allowed / range:** non-overlapping half-open routed-layer bands; enabled with a positive transition step

Early-training routing-width overrides. Every table requires `first_layer`, `end_layer`, and `top_k`; the early width must exceed the resolved target width and fit both the expert pool and any configured expert subset. At `transition_step` all overrides are removed atomically. The mechanism is token-choice-only and cannot compose with dynamic top-k, lightweight experts, or the fixed-width cross-layer coupling objective. It adapts the progressive lower-layer sparsification schedule from [arXiv:2512.16248](https://arxiv.org/abs/2512.16248); reproducing the paper's recipe means keeping upper layers at target width and setting the transition near 90% of training.

### `moe_expert_choice_capacity_factor`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** `> 0`

Fallback Expert-Choice capacity factor. Each expert selects `ceil(tokens * factor / num_experts)` tokens per sample.

### `moe_expert_choice_capacity_schedule`

- **Type:** array of tables
- **Default:** `[]`
- **Allowed / range:** non-overlapping, monotone step/layer bands

Optional progressive Expert-Choice capacity overrides. Every table requires `start_step`, `first_layer`, `end_layer`, and `capacity_factor`; `end_step=-1` or omission means no step limit. Bands may not overlap in both their half-open step and layer ranges. For every covered layer the resolved capacity, including transitions to and from the fallback factor, must be non-increasing over training steps.

### `moe_expert_choice_timestep_capacity_schedule`

- **Type:** str
- **Default:** `"disabled"`
- **Allowed / range:** `disabled`, `linear_reverse`

Opt-in timestep-dependent Expert-Choice capacity. `linear_reverse` maps each post-`flow_shift` noise level through the CDF induced by the configured `training.timestep_sampling`, then gives low-noise samples positive and high-noise samples negative integer capacity offsets around the active fallback or step/layer capacity. This is a continuous-video adaptation of arXiv:2604.01622.

### `moe_expert_choice_timestep_capacity_span`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `0` disables; enabled: finite `> 0`

Maximum capacity-factor deviation for timestep scheduling. It is converted to expert-token slots for the current sequence length and reduced symmetrically at the valid capacity boundaries. CDF quantiles make the rounded positive and negative offsets equiprobable, preserving the static selected-slot budget in expectation for uniform, logit-normal, and mode-shift timestep sampling.

### `moe_expert_choice_coverage_alarm_threshold`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `0` disables; otherwise `(0, 1]`

Minimum acceptable per-layer Expert-Choice token coverage. A positive value requires `moe_routing_mode="expert_choice"` and emits detached mean coverage, worst-layer coverage, and an alarm bit; `0` performs no collection.

### `moe_zero_experts`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `0` disables; otherwise `> 0`

Number of logical zero-computation expert slots appended to every routed MoE layer. Added router rows are initialized deterministically from pretrained physical rows and trained with the adapter. Null selections perform no expert call; the AdaMoE null-aware balance term shares one averaged selection frequency across identical null slots. The feature is token-choice-only, requires LoRA and microbatch balance scope, and rejects routing policies/objectives whose physical-only semantics would be ambiguous.

### `moe_copy_experts`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>= 0`

Number of logical identity/copy experts. A selected copy route returns its input without executing an expert MLP.

### `moe_constant_experts`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>= 0`

Number of learned constant-mixture experts. Expert `j` returns `softmax(W_j x)[0] * x + softmax(W_j x)[1] * a_j`; both `W_j` and the expert-specific vector `a_j` are trained and persisted with the adapter.

### `moe_lightweight_top_k`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** disabled: `0`; enabled: `1..(physical experts + all lightweight experts)`

Fixed logical routing width when any zero, copy, or constant experts are enabled. Physical group limits remain active. Selected zero routes receive no output mass; physical, copy, and constant routes share one post-selection normalization. An all-zero selection bypasses the routed FFN exactly. The complete topology and learned lightweight state are persisted and must match on adapter load.

### `moe_adjugate_experts`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Enables Grove-style grouped adjugate experts. The native router and ordinary expert selection remain unchanged. Selected routes are mapped to disjoint contiguous groups; repeated group routes for one token are aggregated before the group's smaller parallel FFN is evaluated. Added output projections are zero-initialized, and the full topology and learned weights are persisted with the adapter.

### `moe_adjugate_expert_groups`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** enabled: positive divisor of the native expert count

Number of disjoint adjugate groups. It is explicit because a checkpoint's router group restriction and Grove's capacity-expansion grouping are different contracts and are not inferred from one another.

### `moe_adjugate_expert_intermediate_size`

- **Type:** int
- **Default:** `128`
- **Allowed / range:** `> 0`

Intermediate width of every provider-owned adjugate FFN.

### `moe_adjugate_expert_scale`

- **Type:** float
- **Default:** `0.05`
- **Allowed / range:** finite `> 0` and `<= groups / native experts`

Fixed multiplier applied after summing the selected ordinary-expert gate weights for a group. The upper bound is the upcycling stability restriction from Grove MoE.

### `moe_router_timestep_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `0` disables; otherwise `> 0` with `moe_routing_mode="expert_choice"`

Enables Nucleus-style decoupled routing. The pretrained router projects the unmodulated normalized token state; an independent zero-initialized trainable matrix projects the raw timestep embedding and is added with this fixed multiplier. Expert computation still consumes the fully AdaLN-modulated state. The added projection is persisted with the LoRA adapter, while `0` preserves the original modulated-input routing path exactly.

### `moe_aux_loss_type`

- **Type:** str
- **Default:** `"model_native"`
- **Allowed / range:** e.g. `sequence`

Load-balance aux-loss variant (active model provider).

### `moe_aux_loss_weight`

- **Type:** float
- **Default:** `0.01`
- **Allowed / range:** —

Aux-loss weight (active model provider).

### `moe_router_z_loss_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** —

Router z-loss weight (active model provider).

### `moe_router_similarity_loss_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `>= 0`; `0` disables

Weight for the Expert Race router-similarity objective (arXiv:2503.16057, equations 9-11). It separately normalizes individual-expert and pairwise co-selection frequencies, then weights the differentiable router-probability correlation matrix. This prevents marginally balanced routing from hiding collapsed expert combinations.

### `moe_dynamic_topk_min`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `0` disables; otherwise `1..experts_per_token`

Minimum active routes per token for compute-budgeted dynamic top-k. The router retains its native fixed-width candidate tensor; inactive route scores are zeroed before dispatch.

### `moe_dynamic_topk_average`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** disabled: `0`; enabled: `moe_dynamic_topk_min..experts_per_token`

Exact average active-route budget per router batch, subject only to integer rounding. Additional routes are assigned to tokens with the least concentrated selected routing scores, and retained scores are renormalized to preserve gate mass.

### `moe_spatiotemporal_routing_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `>= 0`; `0` disables

Weight for routing-distribution consistency across adjacent video tokens in the provider-declared temporal/spatial patch grid. Text tokens are excluded. The model-agnostic objective consumes only router probabilities, grid geometry, and video-span offsets.

### `moe_spatiotemporal_routing_max_edges`

- **Type:** int
- **Default:** `4096`
- **Allowed / range:** `> 0`

Maximum deterministic adjacency edges sampled per video sample and router layer. The sampler does not materialize the complete token graph.

### `moe_bias_update_rate`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** —

Router bias EMA update rate (active model provider).

### `moe_bias_centering`

- **Type:** bool
- **Default:** `True`
- **Allowed / range:** —

Center router bias (active model provider).

### `moe_balance_mode`

- **Type:** str
- **Default:** `"aux_loss"`
- **Allowed / range:** `aux_loss`, `bias_only`, `off`; `bias_only` needs `moe_bias_update_rate>0`

Load-balance mode ([`expert_specs.py`](mirai/core/moe/runtime/specs.py) resolver; applied by the active model provider). `aux_loss` adds the auxiliary term to the loss and applies the online bias. `bias_only` applies only the online bias. `off` disables balance pressure. Router z-loss (`moe_router_z_loss_weight`) is independent.

### `moe_balance_loss_disable_step`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `0` disables; otherwise `> 0`, `moe_balance_mode="aux_loss"`, positive aux weight, enabled aux type, zero bias update rate

Global-step boundary for DMEP-inspired balance-pressure relaxation. The configured auxiliary load-balance weight is used before this step and becomes exactly zero at and after it. Router z-loss and all other objectives are unchanged; the schedule has no mutable checkpoint state.

### `moe_balance_scope`

- **Type:** str
- **Default:** `"microbatch"`
- **Allowed / range:** `microbatch`, `global_batch`

Load-balance scope ([`moe/adaptation/global_balance.py`](mirai/core/moe/adaptation/global_balance.py); applied by the active model provider). `microbatch` (default) preserves the selected auxiliary form's local estimate byte-for-byte. `global_batch` injects the expert-selection frequency accumulated across the gradient-accumulation window while retaining the current microbatch's differentiable mean-probability factor. This applies to both token-fraction and sequence-native auxiliary forms; for the latter it intentionally replaces per-sequence frequency with the wider window. The `bias_only` online update already sums window counts; `off` is inert.

### `moe_balance_grad_ratio_telemetry`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** provider capability; incompatible with `gradient_checkpointing="aggressive"`

Enables the LongCat Equation 9 diagnostic on graph-bearing router probabilities. For each balancing objective, Mirai sums token-level probability gradients into the derivative on the batch mean and reports the norm ratio of the weighted objective gradient to the task gradient. It uses `autograd.grad` without modifying `.grad`; the extra graph traversal is absent when disabled. Standard, selective, and disabled checkpointing expose the required intermediates; reentrant aggressive checkpointing does not.

### `moe_balance_grad_ratio_threshold`

- **Type:** float
- **Default:** `0.1`
- **Allowed / range:** finite `> 0`

Dominance-alarm threshold for balance/task gradient ratios. This is telemetry only: Mirai does not rescale the loss coefficient automatically. A ratio at or above the threshold emits an alarm value of `1`.

### `moe_phi_balance_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** >=0; `0` disables

Population-level phi-balancing auxiliary-loss weight ([`moe/adaptation/phi_balance.py`](mirai/core/moe/adaptation/phi_balance.py)). Enabled mode tracks a detached per-layer EMA of pre-top-k soft routing probabilities and applies `<p, grad phi(m)>`. It cannot compose with a nonzero legacy balance-loss weight or online balance bias.

### `moe_phi_balance_ema_rate`

- **Type:** float
- **Default:** `0.01`
- **Allowed / range:** `(0, 1]` when enabled

EMA update rate in `m <- (1-eta)m + eta*p`. State is checkpointed exactly.

### `moe_phi_balance_potential`

- **Type:** str
- **Default:** `"negative_entropy"`
- **Allowed / range:** `negative_entropy`, `euclidean`

Convex population potential. Negative entropy is the paper-recommended default; Euclidean retains the reference identity mirror map.

### `moe_router_variance_loss_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** >=0; `0` disables

Weight for the token-discriminative router-score variance objective ([`moe/adaptation/specialization_loss.py`](mirai/core/moe/adaptation/specialization_loss.py)). The loss is negative mean per-expert variance across deterministically sampled pre-top-k score rows, so minimizing it discourages identical routing decisions for every token.

### `moe_expert_orthogonality_loss_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** >=0; `0` disables; requires `experts_per_token >= 2`

Weight for active-expert output orthogonality. A dispatch observer gathers only bounded complete token×top-k rows from already materialized expert outputs before weighted restore; it never repeats expert computation or builds a full unsorted output tensor. Native sorted dispatch and compressed-weight loop, vectorized/legacy batched dispatch with host or device preprocessing, Triton grouped dispatch, persistent dispatch, and framework grouped-mm are supported. Standard and aggressive checkpoint modes return the sampled term through the checkpoint boundary and clear recompute-only capture state.

### `moe_swiglu_specialization_loss_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** >=0; `0` disables; requires `experts_per_token >= 2`

Weight for the Eq. 4 intra-layer specialization objective from arXiv:2602.14159. It penalizes squared cosine similarity between bounded, same-token SwiGLU intermediates of co-activated routed experts before the down projection. The observer reuses computed activations and never repeats expert execution. Native sorted and compressed-weight loop, vectorized/legacy batched dispatch with host or device preprocessing, Triton grouped dispatch, persistent dispatch, and framework grouped-mm are supported.

### `moe_cross_layer_coupling_loss_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** >=0; `0` disables; requires at least 2 MoE layers

Weight for the adjacent-layer joint top-k probability objective from Eq. 7 of arXiv:2602.14159. Provider scores are normalized onto the expert simplex; identical bounded token rows are used across layers. This sharpens adjacent top-k pathway mass but does not require matching expert identifiers across layers.

### `moe_specialization_max_tokens`

- **Type:** int
- **Default:** `256`
- **Allowed / range:** >0

Per-layer token bound shared by expert-specialization objectives. Deterministic evenly spaced selection avoids RNG/checkpoint state and bounds retained work independently of video resolution.

### `router_fp32_master`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Opt-in FP32 master copy of trainable router parameters ([`router_fp32_master.py`](mirai/core/training/optim/router_fp32_master.py); applied by the active model provider). The master accumulates updates before re-materializing the BF16 working copy each step, preventing sub-ULP updates from being discarded. It allocates no state unless `adapter.train_router=true`; runtime validation warns otherwise. Its state is checkpointed and validated against the configured trainable routers on restore.

### `expert_subset_fraction`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** `(0, 1]`

Stochastic per-step expert-subset routing adapted from EMoE (arXiv:2509.21892). Each MoE layer routes through a hot-biased sample of `ceil(fraction*num_experts)` experts, bounding the active expert working set. `1.0` disables masking and preserves full routing. Group-limited routing (`n_group>1`) is bypassed while a subset is active; the ordinary routing path is restored at anneal completion.

### `expert_subset_pool_factor`

- **Type:** float
- **Default:** `2.0`
- **Allowed / range:** `>= 1`

Candidate pool for subset sampling = top `ceil(pool_factor*size)` hottest experts by routing mass. `1.0` = deterministic hottest-`size`; larger = more warm-tail exploration.

### `expert_subset_anneal_steps`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>= 0`

Anneal `expert_subset_fraction` → `1.0` over this many steps (`0` = constant). At/after anneal end the subset spans every expert → masking becomes a no-op.

### `expert_subset_router_kl_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `>= 0`

Reverse-KL router-consistency loss weight (EMoE hierarchical router loss). Only applied when the router is trainable (`adapter.train_router`); `0` (default) = off. `KL(subset‖full_detached)`, added as `moe_subset_router_kl`.

### `expert_pruning`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `prune`

Offline structured expert-pruning gate. `off` preserves the training path exactly. `prune` arms [`prune_experts.py`](scripts/tools/prune_experts.py): the calibrated criteria first use [`calibrate_expert_pruning.py`](scripts/tools/calibrate_expert_pruning.py), while `aimer` reads weights directly and forbids a calibration artifact. Both paths retain experts by keep fraction or score threshold with `keep >= experts_per_token`, slice grouped expert tensors and router rows, and write a new lineage-bound packed artifact without mutating the source. Numeric controls are script arguments.

### `expert_pruning_criterion`

- **Type:** str
- **Default:** `"frequency"`
- **Allowed / range:** `frequency`, `reap`, `man`, `msan`, `aimer`

Ranking criterion for opt-in expert pruning. `frequency` ranks selected-route counts. `reap` uses the conditional mean of router-weighted expert-output L2 norms; `man` removes the router weight; `msan` uses the conditional mean squared output norm. `aimer` is calibration-free: [`prune_experts.py`](scripts/tools/prune_experts.py) dequantizes bounded expert blocks from the selected packed artifact, computes `sum(abs(w)) / sqrt(numel(w) * sum(w²))` over combined `w1/w2/w3`, and prunes the largest scores. AIMER accepts `--max-block-elements` and `--metric-device cpu\|cuda`; CUDA obtains the GPU lease. The key is inert while `expert_pruning="off"`.

### `expert_consolidation`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `prototype`, `hierarchical_output`

Offline expert consolidation gate. `off` is inert. Both enabled modes require `memory.frozen_weight_packed_state_path`, use provider-owned calibration, schema-v3 tensor evidence with mandatory dataset/model/config snapshots plus the exact packed-source fingerprint, a reviewed schema-v2 JSON plan, and a schema-v3 packed output that preserves logical router rows through an explicit logical→physical map. The plan records its strategy and reduction ratio; application requires exact lineage and grouped-module inventory matches and writes the transform lineage into the output manifest. `prototype` combines routed contribution with bounded normalized-parameter distances and retains representative source tensors. `hierarchical_output` evaluates every expert on the same deterministic token sample bounded by `calibrate_expert_consolidation.py --max-output-tokens-per-observation`, clusters Euclidean mean-output distances with deterministic average linkage, physically averages every cluster by observed routing frequency, and re-encodes it through the source packed-format boundary. `--projection-block-mib` bounds the parameter-distance path. Calibration requires `memory.expert_weight_access="active_dequant"`, runs under the GPU lease, and restores RNG, sampling, and model mode. Runtime aliases activate only when the resulting artifact is selected through `memory.frozen_weight_packed_state_path`; unsupported lineage, topology, source aliases, and dispatch combinations fail explicitly.

### `expert_upcycling`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `drop`

Exact gate for the data-free physical expert-splitting transform in [`scripts/tools/upcycle_experts.py`](scripts/tools/upcycle_experts.py). `drop` requires a regular packed artifact, expert-targeted LoRA, and `adapter.train_router=true`. The source is never overwritten. Every source expert remains byte-identical while its copies reinitialize the same sampled intermediate indices across `w1/w3` rows and `w2` columns from each selected tensor's own mean and standard deviation. Router matrices receive the paper's reported uniform initialization with standard deviation 0.02; correction-bias copies start at zero. The output records exact source/model lineage and rejects a runtime config whose transform policy differs. This adapts Drop-Upcycling to existing sparse experts and does not claim dense-to-MoE conversion or the paper's LLM results for video.

### `expert_upcycling_copies`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `0` while off; `>= 1` with `expert_upcycling="drop"`

Uniform number of new physical copies created after every source expert. Uniform interleaving preserves contiguous expert-group topology. The expanded expert and router axes are restored through the normal packed-state loader.

### `expert_upcycling_reinit_ratio`

- **Type:** float
- **Default:** `0.5`
- **Allowed / range:** finite `(0, 1]`

Fraction of each copy's intermediate channels selected by `floor(r * intermediate_size)` for statistical reinitialization. The default is the strongest setting in the source paper's reported sweep; Mirai stores the exact ratio and a mask fingerprint in the output manifest.

### `expert_upcycling_seed`

- **Type:** int
- **Default:** `42`
- **Allowed / range:** `>= 0`

Root seed for stable per-module channel masks, replacement samples, and added router rows. It is part of the transform fingerprint, so a mismatch fails before training.

### `expert_weight_compression`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `flexmoe_nested`, `mixture_basis`, `shared_basis`, `stun_sparse24`

Exact gate for registered physical expert-weight providers in packed artifacts. `off` preserves schema-v1/v2/v3 packed behavior and rejects provider-bearing artifacts. `flexmoe_nested` permits a completed artifact from [`compress_experts_flexmoe.py`](scripts/tools/compress_experts_flexmoe.py): the shared `w1/w3` row and `w2` column permutation is applied, every rejected suffix is physically removed, and ragged rowwise-INT8 payloads execute in equal-width expert buckets. `shared_basis` permits schema-v4 per-module shared-basis SVD factors. `mixture_basis` permits an artifact created by [`factorize_experts_mixture_basis.py`](scripts/tools/factorize_experts_mixture_basis.py): Adam minimizes data-free reconstruction error over expert-specific transformations, softmax coefficients, and shared nonlinear (`silu`, `tanh`, or `gelu`) bases for `w1/w3`; scale normalization is folded into the transformations and the original packed `w2` tensors remain byte-preserved. The tool requires rank and basis count, defaults to 1000 steps at learning rate 0.07, and exposes expert-batch, row-chunk, checkpoint-interval, covariance-memory, optimizer-memory, storage-dtype, and maximum-error bounds. It records the exact source fingerprint and rejects a result that is not smaller than its source. `stun_sparse24` permits an artifact created by [`compress_experts_stun_sparse24.py`](scripts/tools/compress_experts_stun_sparse24.py): router-row agglomeration and centroid-nearest representative selection follow STUN's structured stage, while the published Wanda/OWL stage is deliberately replaced with compact quantized 2:4 storage and on-demand dense decode. The STUN transform additionally requires `expert_pruning="prune"`, preserves `experts_per_token` as a hard lower bound, rejects a payload that is not smaller than its source, and never overwrites its source artifact. Runtime activation remains separately opt-in through `memory.frozen_weight_packed_state_path`; unsupported providers and schema combinations fail explicitly.

### `flexmoe_calibration`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `nested`; `nested` requires `expert_weight_compression="off"` and `moe_routing_mode="token_choice"`

Offline FlexMoE calibration gate for a complete schema-v1/v2/v3 expert source. `calibrate_flexmoe.py --stage ranking` accumulates lineage-bound Taylor saliency from native video-task gradients. Its `actions` stage uses the same prepared corruption and conditioning for the full-width teacher and masked student, then optimizes straight-through Gumbel prefix actions with native task loss, teacher-prediction MSE, load-sensitive cost, and entropy. Retention ratios, thickest-action initialization margin, temperature schedule, regularization schedule, learning rate, teacher-loss weight, and seed are explicit CLI inputs because the paper does not publish a transferable video recipe. Calibration requires token-choice because exact Taylor reconstruction is attached to that dispatch seam; completed ragged artifacts can execute either token-choice or Expert-Choice through equal-width BMM buckets. `off` attaches no hooks or action state. The resulting ranking and action artifacts are consumed offline by [`compress_experts_flexmoe.py`](scripts/tools/compress_experts_flexmoe.py); runtime then selects `expert_weight_compression="flexmoe_nested"`.

### `post_compression_router_repair`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `router_kd`, `router_task`

Offline post-compression router-repair gate. Both modes arm [`scripts/tools/repair_router_kd.py`](scripts/tools/repair_router_kd.py): the original model supplies final diffusion predictions for exact replay inputs, the compressed student updates only FP32-master router weights, and a held-out prediction-MSE non-regression check plus complete non-router fingerprint must pass before output. `router_kd` minimizes final teacher/student prediction MSE, following the continuous-output video adaptation of Router KD (arXiv:2603.02217). Experimental `router_task` instead minimizes Mirai's native prepared video objective, following GEMQ's task-loss router adaptation (arXiv:2605.23078), while retaining the teacher-prediction guard. Neither mode matches router logits, so teacher and student expert counts may differ. The gate alone changes no runtime behavior.

### `router_repair_artifact_path`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** no control chars; requires `post_compression_router_repair="router_kd"` or `"router_task"`

Optional lineage-bound router-only safetensors patch emitted by [`repair_router_kd.py`](scripts/tools/repair_router_kd.py). Loading verifies the exact compressed packed-artifact fingerprint, initial router tensor fingerprint, complete target inventory, and shapes before copying repaired weights. Empty preserves the compressed base exactly. A non-empty path also requires `memory.frozen_weight_packed_state_path` at model construction.

### `expert_quantization_calibration`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `affinity`

Offline MoEQuant-style calibration gate. `affinity` lets [`scripts/tools/calibrate_moe_quantization.py`](scripts/tools/calibrate_moe_quantization.py) ask the registered model-family provider for router targets and collect lineage-bound per-expert selected counts, routed gate mass, squared affinity, and per-sample coverage. Evidence exposes deterministic expert-balanced sample selection and normalized affinity reconstruction weights. `scripts/tools/factorize_experts_shared_basis.py --quantization-calibration-evidence ...` consumes fully covered evidence to learn a population-weighted basis and rejects dataset/model lineage or module-topology mismatches. Runtime quantization remains unchanged. `off` allocates no hooks or evidence state.

### `expert_quantization_rotation`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `learned`

Offline expert-INT8 rotation gate. `learned` is consumed by [`scripts/tools/export_compressed_weights_packed_state.py`](scripts/tools/export_compressed_weights_packed_state.py) and requires `memory.frozen_weight_quantization="int8"`. Each grouped-expert module learns one shared Cayley-parameterized orthogonal transform for `w1/w3` and one for `w2`, starting from the fixed-Hadamard basis and minimizing Mirai's exact rowwise symmetric-INT8 reconstruction error. Hard-error checkpoints retain the best non-regressing candidate. `--rotation-optimization-steps`, `--rotation-learning-rate`, `--rotation-row-chunk-size`, `--rotation-checkpoint-interval`, `--rotation-device`, and `--rotation-max-workspace-gib` control bounded offline optimization. Schema-v5 packed artifacts store all projection mappings, exact matrices, per-module source-weight fingerprints, and initial/optimized errors. Runtime loading requires the same `learned` gate; reference dequantization and `rotated_int8` execution consume identical matrices. `off` preserves fixed-Hadamard quantization and rejects rotation-bearing artifacts.

### `router_quantization_calibration`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `eaquant`

Offline EAQuant routing-consistency calibration gate. `eaquant` arms [`scripts/tools/calibrate_router_quantization.py`](scripts/tools/calibrate_router_quantization.py) on floating-point source routers. Provider-owned targets expose the exact linear input and any unquantized additive logit branch; bounded deterministic token sampling is split into repeated passes when `--max-input-gib` would be exceeded. Per-output symmetric INT8 scales are coordinate-searched over `--grid-size` clipping ratios in `[--minimum-clipping-ratio, 1]` against EAQuant's logit-MSE plus KL-Top objective. `--relaxation=0` aligns only the reference top-k, matching the paper's strongest ablation; higher values include `floor(relaxation*(experts-k))` additional experts. The safetensors artifact records objective components and binds dataset/model/config lineage, router topology, and an exact fingerprint of all source router tensors. The gate is collection-only and requires `memory.router_quantization="disabled"` with no runtime calibration path.

### `expert_factorization_calibration`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `whitened`; `whitened` requires `moe_routing_mode="token_choice"` and `expert_weight_compression` equal to `off` or `shared_basis`

Offline SVD-LLM-style activation-covariance gate. With the pre-factorization source selected by `expert_weight_compression="off"`, `whitened` arms [`scripts/tools/calibrate_expert_whitening.py`](scripts/tools/calibrate_expert_whitening.py), which streams the actual routed inputs of `w1/w3` and the post-SwiGLU input of `w2` into bounded `XᵀX` summaries without retaining activation samples. `--max-covariance-gib` bounds live covariance state by splitting targets into repeated calibration passes. A transform config selects `expert_weight_compression="shared_basis"`; `factorize_experts_shared_basis.py --whitening-evidence ...` then performs shared-basis SVD in the regularized whitened metric for either factor axis and rejects dataset/model, packed-artifact, module, or projection-shape mismatches while retaining the calibration-config fingerprint as provenance. The optional `--whitening-regularization` defaults to `1e-6`. `off` preserves the existing reference-expert or affinity-weighted factorization exactly.

### `expert_precision_calibration`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `imatrix`, `alphaq`; planning requires floating experts and empty packed-state/precision-plan paths

Offline per-tensor precision-planning gate. `imatrix` arms [`scripts/tools/calibrate_expert_precision.py`](scripts/tools/calibrate_expert_precision.py): provider-owned routed expert hosts accumulate `E[x²]` separately for every physical expert's w1/w3 input and post-SwiGLU w2 input, following llama.cpp's open importance-matrix statistic. `--max-accumulator-gib` splits layers into deterministic repeated passes. `alphaq` is an experimental, calibration-free alternative inspired by [AlphaQ](https://arxiv.org/abs/2606.04980): deterministic 256×256 fixed-aspect weight blocks estimate projection-wise PL_Alpha_Hill, and smaller exponents increase the cost of quantization error. Mirai measures its actual packed-format distortion and uses its existing benefit-per-byte allocator, rather than claiming parity with AlphaQ's GPTQ noise surrogate or MILP solver. Each `--formats` candidate is encoded and decoded by Mirai's real runtime representation; `--budget-gib` is a hard ceiling. The schema-v2 JSON plan binds dataset/model/config and an exact source-weight fingerprint. `off` installs no observers and changes no runtime behavior.

### `expert_precision_alphaq_gamma`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** finite and non-negative

Curvature of the AlphaQ-inspired importance weight `(median(alpha) / alpha)^gamma`. `0.0` selects AlphaQ's data-free default `alpha_min * (alpha_max - alpha_min) / variance(alpha)`, with `1.0` used for a degenerate equal-alpha distribution. Positive values select an explicit curvature. It is ignored unless `expert_precision_calibration="alphaq"`; the CLI can override it with `--spectral-gamma`.

### `expert_precision_router_norm_fraction`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `0.0` to `1.0`

Experimental expert-protection fraction from [Efficient Quantization of Mixture-of-Experts with Theoretical Generalization Guarantees](https://arxiv.org/abs/2604.06515). A positive value requires `expert_precision_calibration="imatrix"` and `expert_precision_router_norm_min_format`. Experts are ordered by ascending final router-row L2 norm, the paper's supported surrogate when initialization weights are unavailable. A lower-ranked expert is repeatedly promoted when its maximum w1 row variance is at least three times that of the immediately higher expert. The protected prefix receives the configured precision floor; measured imatrix error still allocates the remaining byte budget. Router weights join the source fingerprint. `0.0` preserves the existing allocator exactly.

### `expert_precision_router_norm_min_format`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** empty or a supported expert precision candidate

Minimum candidate format for the router-norm-protected expert prefix. It must be present in `calibrate_expert_precision.py --formats`; an infeasible byte budget fails rather than weakening the floor. Empty is valid only when `expert_precision_router_norm_fraction=0.0`.

### `moe_routing_health`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Routing-health diagnostics pack ([`routing_health.py`](mirai/core/moe/monitoring/health.py), [`router_drift.py`](mirai/core/moe/monitoring/drift.py), [`preemptive.py`](mirai/core/moe/monitoring/preemptive.py); applied by the active model provider). Also arms [`collapse.py`](mirai/core/moe/monitoring/collapse.py); for MAGI-2 Preview this gate is the only routing signal the family has and is what makes it report `emits_router_metrics = true`. Opt-in, detached telemetry only — never touches the loss/gradient path, defaults byte-identical. Adds expert-response homogenization, deadlock duration, raw-logit reference drift, router gradient/update ratios, numerical-underflow alarms, mechanism-driven full-softmax router and Q/K update-spectrum indicators, and the exact per-layer Fisher-Rao distance from the current token population's marginal routing distribution to uniform routing (`moe_fisher_specialization_*`). The normalized Fisher value divides by `2·acos(1/sqrt(num_experts))`. Mirai does not emit the FHS>1 alarm from arXiv:2604.14500 because the paper's printed FHS equation and proof define different statistics. Default-off because the response estimator, effective-weight summaries, and factor snapshots add per-step work.

### `moe_selection_margin`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Optional detached selection-margin telemetry ([`moe/monitoring/agreement.py`](mirai/core/moe/monitoring/agreement.py); applied by the active model provider). Reports the layer-mean fifth percentile and minimum of the gap between the k-th and (k+1)-th router score, i.e. how close each token's route sits to the selection boundary. Telemetry only — never touches the loss/gradient path. `False` creates no probe state and adds no diagnostics.

### `moe_routing_agreement_evidence`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `report`

Offline train-versus-inference route-set evidence gate. `report` arms [`scripts/tools/calibrate_routing_agreement.py`](scripts/tools/calibrate_routing_agreement.py), which asks the active provider for typed router targets and runs paired forwards with the same batch, sampled noise, timestep, token positions, layers, weights, and initial RNG state. The versioned JSON artifact contains mandatory dataset/model/config lineage and aggregate exact-set churn, Jaccard, PR2 overlap/k for equal-cardinality tokens, and active-cardinality changes; raw token routes are not persisted. `off` creates no hooks or evidence state.

### `text_encoder_path`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** —

Separate text-encoder dir for families that split text assets ([`native_encode.py`](mirai/core/dataset/native_encode.py), active model provider).

### `denoiser_subfolder`

- **Type:** str
- **Default:** `"transformer"`
- **Allowed / range:** e.g. `refiner`

Subfolder owning the trainable denoiser when a root packs multiple DiT components ([`providers.py`](mirai/core/models/providers.py), [`base.py`](mirai/core/models/base.py)).

## `[strategy]` — StrategyConfig

### `type`

- **Type:** str
- **Default:** `"text_to_video"`

Training strategy selector (registry): `text_to_video`, `image_to_video`, `hybrid_conditioning`, `multi_task_video`. `multi_task_video` requires the progressive task-mix curriculum below.

### `params`

- **Type:** table
- **Default:** `{}`

Strategy-owned options; must be a table and unknown keys fail. `text_to_video` accepts `conditioning_dropout_p` and `empty_text_embed`; `image_to_video` accepts `first_frame_conditioning_p`; `hybrid_conditioning` accepts `text_weight` and `clip_weight`; `multi_task_video` accepts the union of the T2V/I2V options and delegates each homogeneous curriculum batch to the matching path (`strategies/*`).

## `[inference]` — InferenceConfig

### `task`

- **Type:** str
- **Default:** `"text_to_video"`

`text_to_video`, `text_to_image`, `image_to_video`, `video_to_video` (CLI aliases: `t2v`, `t2i`, `ti2v`/`i2v`, `v2v`). The selected model provider must declare the task. T2I forces one output frame; TI2V and V2V require per-generation `--input-image` and `--input-video` respectively ([`conditioning.py`](mirai/core/inference/conditioning.py), [`session.py`](mirai/core/inference/session.py)).

### `denoising_strength`

- **Type:** float
- **Default:** `1.0`

`[0,1]`; only configurable below `1.0` for `video_to_video`. It retains `floor(steps * strength)` solver steps and mixes the source latent with noise at the first retained rectified-flow sigma. `0` decodes the deterministic source VAE mode without a DiT forward ([`conditioning.py`](mirai/core/inference/conditioning.py), [`preview.py`](mirai/core/training/preview/preview.py)).

### `prompt_rewriter`

- **Type:** str
- **Default:** `"none"`

`none`, or a rewriter explicitly supported by the selected model-family provider. `lingbot_json` resolves the LingBot prompt to the caption body the encoder consumes: a structured JSON object is unwrapped from its `caption` envelope and stripped of runtime-only keys (`duration`, `fps`, `height`, `width`, `num_frames`, `resolution`, `ratio`), plain language is wrapped into the minimal `comprehensive_description` body, and an empty prompt stays empty. The resolved caption is checked against the family caption schema: a declared field carrying the wrong type fails the call naming the field, and a caption missing declared fields warns and proceeds. The negative prompt is conditioning text, not a caption, and is forwarded byte-for-byte ([`prompt_rewriter.py`](mirai/core/inference/prompt_rewriter.py), provider-owned prompt-rewriter hook).

### `cfg_mode`

- **Type:** str
- **Default:** `"sequential"`

`sequential`, `batched`. `batched` executes conditional and unconditional branches in one single-device B=2 forward, with explicit prefix masks for variable-length text; only providers declaring batch/mask parity may enable it ([`cfg_batching.py`](mirai/core/inference/cfg_batching.py), [`preview.py`](mirai/core/training/preview/preview.py)).

### `keep_text_encoder_resident`

- **Type:** bool
- **Default:** `False`

Keeps native text-encoder weights on their compute device across repeated `InferenceSession.generate()` calls. Increases VRAM use; the CLI BooleanOptional override is `--keep-text-encoder-resident` ([`session.py`](mirai/core/inference/session.py)).

### `stage_text_encoder_before_denoiser`

- **Type:** bool
- **Default:** `False`

Encodes T2V prompts on the compute device before placing the denoiser, then
offloads the text encoder and places the denoiser once for the complete sampling
loop. This bounds peak accelerator residency when the two components cannot
co-reside. It requires a native pipeline, is incompatible with
`keep_text_encoder_resident=true` and `--compile-mode`, and does not alter the
encoded conditioning ([`session.py`](mirai/core/inference/session.py)).

### `text_encoder_weight_quantization`

- **Type:** str
- **Default:** `"none"`
- **Allowed:** `none`, `int8`, `nf4`

Optional weight-only INT8 or NF4 storage for a native text encoder whose
provider implements it. The encoder is released after producing FP32
conditioning, so the denoiser does not inherit its storage type. Both
quantized modes are explicit lossy inference policies and require
`stage_text_encoder_before_denoiser=true`; the
unquantized encoder remains the reference path ([`native_video.py`](mirai/core/models/native_video.py),
provider-owned text encoder).

### `keep_vae_resident`

- **Type:** bool
- **Default:** `False`

Keeps native VAE weights on their compute device across repeated `InferenceSession.generate()` calls. Increases VRAM use; the CLI BooleanOptional override is `--keep-vae-resident` ([`session.py`](mirai/core/inference/session.py)).

### `vae_tiling`

- **Type:** bool
- **Default:** `False`

Enables provider-native spatial VAE tiling for inference encode and decode. This
bounds peak activation memory for high-resolution refinement at the cost of
additional overlapping tile work. The selected model-family provider must
implement the VAE-tiling hook; unsupported providers fail explicitly
([`session.py`](mirai/core/inference/session.py), provider-owned VAE runtime).

### `vae_tile_size`

- **Type:** int
- **Default:** `256`

Spatial sample-space height and width of each VAE tile. Must be greater than zero.
It is consumed only when `vae_tiling=true`.

### `vae_tile_stride`

- **Type:** int
- **Default:** `192`

Spatial sample-space height and width between adjacent VAE tiles. Must be greater
than zero and no greater than `vae_tile_size`; the difference controls overlap.
It is consumed only when `vae_tiling=true`.

### `expert_feature_cache`

- **Type:** str
- **Default:** `"off"`

`off`, `branch`. `branch` arms cross-timestep reuse of per-routed-slot expert features during sampling: each MoE layer keeps its pre-combine expert outputs, and on the next visit only the slots whose routed expert changed are recomputed while the rest are reused and re-weighted with the current routing probabilities. Requires `memory.moe_kernel_backend='grouped'`, because the per-expert reference loop and the fused kernel expose only a combined layer output. The cache never engages while autograd is enabled, so training is unaffected. It is lossy — the uncached execution is the reference path — and it holds `layers x slots x routed_slots x d_head` cached elements plus one layer input per entry, on the layer's device. Each cached layer visit reads one host-visible drift scalar, which synchronizes the compute device. Reuse counters appear as `expert_feature_cache` in the `InferenceSession.generate()` report ([`expert_feature_cache.py`](mirai/core/moe/runtime/expert_feature_cache.py), [`session.py`](mirai/core/inference/session.py), [arXiv:2606.15615](https://arxiv.org/abs/2606.15615)).

### `expert_feature_cache_drift_threshold`

- **Type:** float
- **Default:** `0.05`

`[0,1]`. Relative L2 drift of a layer's input against the cached anchor above which the whole cache entry is invalidated and every expert branch recomputed. `0` reuses only when the input is bit-identical. Inert while `expert_feature_cache="off"` ([`expert_feature_cache.py`](mirai/core/moe/runtime/expert_feature_cache.py)).

### `expert_feature_cache_max_reuse_span`

- **Type:** int
- **Default:** `2`

`>= 0`. Consecutive visits of one layer that may reuse cached branches before a full recompute is forced, bounding how far reuse error can accumulate. `0` keeps the cache armed but never reuses, which reproduces the uncached execution bit-for-bit. Inert while `expert_feature_cache="off"` ([`expert_feature_cache.py`](mirai/core/moe/runtime/expert_feature_cache.py)).

### `expert_feature_cache_slots`

- **Type:** int
- **Default:** `2`

`>= 1`. Cache entries held per MoE layer. Sequential classifier-free guidance visits each layer twice per timestep along two unrelated trajectories, so the default of `2` keeps them in separate entries; `1` halves cache residency and makes those visits evict each other ([`expert_feature_cache.py`](mirai/core/moe/runtime/expert_feature_cache.py)).

### `moe_token_chunk_size`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>= 0`

Maximum number of tokens whose routed expert branches are evaluated at once during inference. `0` preserves the unchunked path. A positive value computes routing once for the complete layer input, then executes and combines contiguous token ranges independently, bounding the large gate/up/hidden activation workspace without changing selected experts or weights. It requires a model family that implements the sparse-MoE token-chunk capability and is incompatible with `inference.expert_feature_cache`, whose cache owns the complete routed-slot layout ([`token_chunking.py`](mirai/core/moe/runtime/token_chunking.py), family grouped-MoE backend).

### `blocks_to_swap`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>= 0`

Denoiser blocks streamed from host RAM during sampling, clamped to the block count. `0` keeps the denoiser fully resident and leaves the residency owner unarmed, so an existing config is unaffected. Above `0` it is the inference-entrypoint opt-in to the same `BlockSwapManager` that serves training, armed in inference mode: the adapter and `training.gradient_checkpointing` preconditions of the training path do not apply, because a sampling forward builds no autograd graph that could read an evicted block. Requires `memory.weight_residency_strategy='block_swap'` or `'stream_disk'` for transport, and rejects `memory.block_residency_planner='phase_aware'`, whose forward-to-backward pins have no backward to serve. Unlike the training key, it works with uncompressed `memory.frozen_weight_quantization='none'` weights. `training.blocks_to_swap` governs the training entrypoint only and is ignored here. Placement never changes the compute graph, so a swapped run samples the same result as the resident one ([`device_placement.py`](mirai/core/training/residency/device_placement.py), [`block_swap.py`](mirai/core/training/residency/block_swap.py), [`session.py`](mirai/core/inference/session.py)).

### `block_swap_mode`

- **Type:** str
- **Default:** `"sync"`
- **Allowed / range:** `sync`, `async`

Transfer scheduling for `inference.blocks_to_swap`. `sync` copies each block in on the compute stream immediately before its forward; `async` issues `memory.block_swap_prefetch_depth` blocks ahead on a dedicated transfer stream so a copy overlaps the preceding block's compute, at the cost of holding the in-flight blocks on the device. Inert while `inference.blocks_to_swap = 0` ([`block_swap.py`](mirai/core/training/residency/block_swap.py)).

### `refiner_blocks_to_swap`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>= 0`

Optional block residency for a provider-owned, separately loaded refiner DiT.
At `0`, the refiner inherits the denoiser residency request. Above `0`, only the
refiner is streamed; `inference.blocks_to_swap=0` may keep the base denoiser
resident. Requires `memory.weight_residency_strategy='block_swap'` or
`'stream_disk'` and a provider that implements separate refiner residency
([`device_placement.py`](mirai/core/training/residency/device_placement.py)).

### `refiner_block_swap_mode`

- **Type:** str
- **Default:** `sync`
- **Allowed / range:** `sync`, `async`

Transfer scheduling for `inference.refiner_blocks_to_swap`. It has the same
synchronous/asynchronous semantics as `inference.block_swap_mode` and is inert
while the refiner-specific block count is zero.

## `[training]` — TrainingSection

### `seed`

- **Type:** int
- **Default:** `42`
- **Allowed / range:** —

Global RNG seed ([`training_session.py`](mirai/core/training/lifecycle/training_session.py)).

### `batch_size`

- **Type:** int
- **Default:** `2`
- **Allowed / range:** >=1

Per-step batch (training_step*.py).

### `gradient_accumulation`

- **Type:** int
- **Default:** `1`
- **Allowed / range:** >=1

Grad-accum steps ([`runtime/execution.py`](mirai/core/training/runtime/execution.py)).

### `max_steps`

- **Type:** int
- **Default:** `100`
- **Allowed / range:** >=1

Total optimizer steps ([`training_loop.py`](mirai/core/training/lifecycle/training_loop.py)).

### `loss_function`

- **Type:** str
- **Default:** `"mse"`
- **Allowed / range:** `mse`, `huber`, `pseudo_huber`

Loss function (registry) ([`flow_loss.py`](mirai/core/training/objectives/flow_loss.py), [`engine.py`](mirai/core/training/objectives/engine.py)).

### `objective`

- **Type:** str
- **Default:** `"flow_matching"`
- **Allowed / range:** registry: `flow_matching`, `regression`, `sharp_moe_trajectory`

Training objective. `sharp_moe_trajectory` is selected only with the SharpMoE policy described below and owns a recursive multi-timestep rollout (`objectives/*`).

### `contrastive_flow_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `[0, 1)`; `0` disables; enabled requires `objective="flow_matching"`, `loss_function="mse"`, `loss_weighting="uniform"`, `loss_bucket_normalization="none"`, and `batch_size >= 2`

Contrastive Flow Matching (Δ-FM): subtracts this weight times the squared distance to a uniformly sampled non-self flow target from the same microbatch. It adds no model forward, uses the checkpointed torch RNG, and reports detached positive/negative terms ([`objectives/contrastive_flow.py`](mirai/core/training/objectives/contrastive_flow.py)).

### `latent_wavelet_loss_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `>= 0`; `0` disables; enabled requires `objective="flow_matching"`, `loss_function="mse"`, `strategy.type="text_to_video"`, disabled contrastive flow, and `masked_loss=false`

Weight for single-level spatial Haar supervision on the reconstructed clean latent. The four orthonormal sub-band MSE terms are evaluated in FP32 without a VAE decode; video frames are transformed independently. This is an experimental adaptation of the high-resolution objective from [Nucleus-Image](https://arxiv.org/abs/2604.12163) ([`objectives/latent_wavelet.py`](mirai/core/training/objectives/latent_wavelet.py)).

### `loss_weighting`

- **Type:** str
- **Default:** `"uniform"`
- **Allowed / range:** `uniform`, `min_snr_gamma`, `cosmap`, `adaptive_uncertainty`

Timestep loss weighting. `adaptive_uncertainty` adds an objective-owned 128-channel Fourier head over `logit(t) / 4` and minimizes `L / exp(u) + u`; it requires flow matching, MSE, no bucket normalization, and a general-purpose optimizer. Its parameters use the main learning rate without weight decay and are included in checkpoints ([`objectives/adaptive_weighting.py`](mirai/core/training/objectives/adaptive_weighting.py)).

### `min_snr_gamma`

- **Type:** float
- **Default:** `5.0`
- **Allowed / range:** >0

Min-SNR gamma ([`flow_loss.py`](mirai/core/training/objectives/flow_loss.py)).

### `timestep_eps`

- **Type:** float
- **Default:** `1e-5`
- **Allowed / range:** in (0, 0.5)

Timestep clamp eps (objectives/*).

### `timestep_sampling`

- **Type:** str
- **Default:** `"uniform"`
- **Allowed / range:** `uniform`, `logit_normal`, `mode`

Rectified-flow training-timestep distribution; `uniform` preserves the original default sequence ([`objectives/sampling.py`](mirai/core/training/objectives/sampling.py)).

### `timestep_sampling_mean`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** finite

Mean of the normal variate before the sigmoid in `logit_normal` mode.

### `timestep_sampling_std`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** finite and > 0

Standard deviation of the normal variate before the sigmoid in `logit_normal` mode.

### `timestep_sampling_mode_scale`

- **Type:** float
- **Default:** `1.29`
- **Allowed / range:** `-1` to `2 / (pi - 2)`

Scale of the rectified-flow mode transform in `mode` sampling.

### `warmup_steps`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0

LR warmup steps ([`scheduler.py`](mirai/core/training/optim/scheduler.py)).

### `non_finite_grad_policy`

- **Type:** str
- **Default:** `"abort"`
- **Allowed / range:** `abort`, `skip_step`

Non-finite-grad handling ([`optim/gradients.py`](mirai/core/training/optim/gradients.py), [`runtime/execution.py`](mirai/core/training/runtime/execution.py)).

### `max_consecutive_skipped_steps`

- **Type:** int
- **Default:** `5`
- **Allowed / range:** >=0

Abort after N consecutive skipped steps ([`runtime/execution.py`](mirai/core/training/runtime/execution.py)).

### `moe_expert_touch_guard`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `warn`, `error`

Post-forward guard over the most saturated MoE layer; `error` aborts before backward ([`touch_guard.py`](mirai/core/moe/runtime/touch_guard.py)).

### `moe_expert_touch_max_fraction`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** in (0, 1]

Maximum permitted active-expert fraction when the guard is enabled ([`touch_guard.py`](mirai/core/moe/runtime/touch_guard.py)).

### `max_grad_norm`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** >0

Grad-clip norm ([`gradients.py`](mirai/core/training/optim/gradients.py)).

### `val_every_n_steps`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0

Validation interval; 0 = off ([`training_loop.py`](mirai/core/training/lifecycle/training_loop.py)).

### `early_stop_patience`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0

Early-stop patience; 0 = off ([`early_stop.py`](mirai/core/training/evaluation/early_stop.py), [`training_loop.py`](mirai/core/training/lifecycle/training_loop.py)).

### `ema_enabled`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Weight EMA ([`ema.py`](mirai/core/training/optim/ema.py), [`training_step_post.py`](mirai/core/training/lifecycle/training_step_post.py)).

### `ema_decay`

- **Type:** float
- **Default:** `0.999`
- **Allowed / range:** in (0, 1] when `ema_enabled`

EMA decay ([`ema.py`](mirai/core/training/optim/ema.py)).

### `posthoc_ema_enabled`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Enables CPU-FP32 power-function EMA profiles over adapter-owned state. The profiles are required checkpoint state when enabled and are periodically saved for offline reconstruction of arbitrary EMA lengths.

### `posthoc_ema_profile_stds`

- **Type:** list[float]
- **Default:** `[0.05, 0.1]`
- **Allowed / range:** at least two distinct finite values in `(0, 0.289)` when enabled

Relative response standard deviations of the tracked power-function EMA basis profiles.

### `posthoc_ema_snapshot_every_n_steps`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `0` when disabled; `> 0` when enabled

Interval for atomic adapter-only snapshots under `checkpoints/posthoc_ema/`. The final committed step is always snapshotted. Reconstruct with [`scripts/tools/reconstruct_posthoc_ema.py`](scripts/tools/reconstruct_posthoc_ema.py).

### `prior_loss_weight`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** >=0

Prior-preservation weight (prior loss path).

### `prior_ratio`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** in [0, 1]

Fraction of prior samples.

### `log_grad_breakdown_every_n_steps`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0

Grad-breakdown log interval ([`gradients.py`](mirai/core/training/optim/gradients.py)).

### `gradient_checkpointing`

- **Type:** str/bool→str
- **Default:** `"standard"`
- **Allowed / range:** `off`, `standard`, `selective`, `aggressive`

Activation checkpointing mode; `selective` caches expensive matrix/attention operators while recomputing elementwise work; TOML `true`/`false` coerce to `standard`/`off`.

### `noise_offset`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** —

Additive noise offset (objectives/*, [`flow_loss.py`](mirai/core/training/objectives/flow_loss.py)).

### `loss_bucket_normalization`

- **Type:** str
- **Default:** `"none"`
- **Allowed / range:** `none`, `per_bucket_mean`

Per-bucket loss normalization ([`engine.py`](mirai/core/training/objectives/engine.py)).

### `blocks_to_swap`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0

Transformer blocks to CPU-swap ([`block_swap.py`](mirai/core/training/residency/block_swap.py)).

### `block_swap_mode`

- **Type:** str
- **Default:** `"sync"`
- **Allowed / range:** `sync`, `async`

Block-swap sync mode ([`block_swap.py`](mirai/core/training/residency/block_swap.py)).

### `optimizer_cpu_offload`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Offload optimizer state to CPU ([`device_placement.py`](mirai/core/training/residency/device_placement.py)).

### `gradient_cpu_offload`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Offload grads to CPU ([`gradients.py`](mirai/core/training/optim/gradients.py)).

### `activation_cpu_offload`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** rejected with `compile=true`

Selectively offload saved CUDA activations to CPU ([`activation_offload.py`](mirai/core/training/residency/activation_offload.py)).

### `activation_cpu_offload_min_mib`

- **Type:** int
- **Default:** `8`
- **Allowed / range:** >= 0

Minimum saved-tensor size eligible for CPU offload.

### `activation_cpu_offload_max_gib`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** > 0 when offload is enabled

Maximum simultaneously reserved host bytes for saved activations.

### `activation_cpu_offload_pin_memory`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Use page-locked staging and non-blocking restores; counts against the host pinned-memory budget.

### `activation_cpu_offload_defer_layers`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >= 0; nonzero requires activation CPU offload

Number of provider-declared layers completed after an activation is saved before its D2H transfer is launched. `0` preserves immediate saved-tensor offload. A positive value excludes the final `defer_layers` regions because their activations would be consumed before the delay can amortize a transfer; `1` is the source-recommended MoE setting.

### `activation_cpu_offload_prefetch_layers`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >= 0; nonzero requires activation CPU offload and pinned memory

Start an offloaded layer's H2D reload this many provider-declared backward regions before its first use. `0` restores on demand. CUDA events order the transfer stream and consuming backward stream without a host synchronization.

### `activation_cpu_offload_view_replay`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** requires activation CPU offload

Offload a saved view's base storage once and reconstruct each logical view from its recorded size, stride, and storage offset after reload. `False` preserves independent per-tensor staging.

### `activation_compression`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** mutually exclusive with activation CPU offload

Compress eligible saved activation matrices with an online low-rank projection and reconstruct them during backward ([`activation_compression.py`](mirai/core/training/residency/activation_compression.py)).

### `activation_compression_rank`

- **Type:** int
- **Default:** `32`
- **Allowed / range:** > 0

Maximum saved-activation factorization rank.

### `activation_compression_min_mib`

- **Type:** int
- **Default:** `8`
- **Allowed / range:** >= 0

Minimum tensor size eligible for low-rank saved-activation compression.

### `masked_loss`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Apply a mask to the loss ([`training_step_pre.py`](mirai/core/training/lifecycle/training_step_pre.py)).

### `compile`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** rejected with `activation_cpu_offload=true`

Enable opt-in graph compilation. The disabled path calls the original pipeline forward without installing regions or token markers. Runtime compilation failures restore the eager callables exactly ([`runtime/compilation.py`](mirai/core/training/runtime/compilation.py)).

### `compile_scope`

- **Type:** str
- **Default:** `"regional"`
- **Allowed / range:** `regional`, `full`

`regional` compiles the repeated blocks declared by the model provider and leaves orchestration eager; `full` compiles `pipeline.forward` as one top-level callable.

### `compile_mode`

- **Type:** str
- **Default:** `"default"`
- **Allowed / range:** `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`

Mode forwarded to `torch.compile`; `default` omits the mode argument.

### `compile_dynamic`

- **Type:** bool or omitted
- **Default:** omitted (`None`)
- **Allowed / range:** `true`, `false`, or omit for Torch automatic dynamism

Dynamic-shape policy forwarded to `torch.compile`. Explicit token buckets require `true` or the omitted automatic policy.

### `compile_token_buckets`

- **Type:** list[int]
- **Default:** `[]`
- **Allowed / range:** strictly increasing positive upper bounds; regional scope only; incompatible with `compile_dynamic=false`

Optional validated token-shape ladder for repeated blocks. Each observed token count is assigned to the smallest containing bucket and the token axis receives a soft dynamic-shape hint; inputs are not padded or numerically changed, and the compiler may still specialize dimensions required by model semantics. Exceeding the final bound fails and restores eager execution. Dry-run diagnostics report bucket hits and compiler graph counters.

### `moe_token_chunk_size`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>= 0`; `0` disables; incompatible with `adapter.lora_parameter_dropout > 0`

Maximum number of already-routed tokens processed by one local expert-compute chunk. Positive values checkpoint each chunk with non-reentrant recomputation during training, preserving the complete-batch routing decision and auxiliary losses while bounding routed-expert activation lifetime. This is the single-GPU adaptation of MemFine FCDA; it does not implement expert-parallel communication or claim the paper's distributed speedups.

### `block_swap_backward`

- **Type:** bool
- **Default:** `True`
- **Allowed / range:** —

Swap during backward ([`block_swap.py`](mirai/core/training/residency/block_swap.py)).

### `pinned_memory_budget_fraction`

- **Type:** float
- **Default:** `0.5`
- **Allowed / range:** in (0, 1]

Pinned-memory budget fraction ([`block_swap.py`](mirai/core/training/residency/block_swap.py), [`memory_safety.py`](mirai/core/training/residency/memory_safety.py)).

### `curriculum`

- **Type:** table
- **Default:** `{}`
- **Allowed / range:** must be a table; unknown nested keys fail

Progressive resolution/frame/task schedule. `enabled=false` is inert. Supported keys are `enabled`, `resolution_schedule`, `frame_schedule`, and `task_mix_schedule`; stage keys are optimizer-step integers. Resolution values use `HxW`, frame values are positive integers, and each task-mix value is a positive-weight table over `text_to_image`, `text_to_video`, and `image_to_video`. Task mixing requires `strategy.type="multi_task_video"` and dataset-registration metadata `training_task` on every sample. Selection is deterministic by seed and global microbatch index, active pools are validated before training, and every microbatch contains one task. When dynamic flow shift is enabled, its value is derived from the selected latent shape rather than stored in the schedule ([`curriculum.py`](mirai/core/training/data/curriculum.py), [`multi_task_video.py`](mirai/core/training/strategies/multi_task_video.py)).

### `policy_modules`

- **Type:** list[str]
- **Default:** `[]`
- **Allowed / range:** dotted importable Python module names

Explicit out-of-tree training-policy plugins. Modules register factories through `register_training_policy`; an imported plugin remains inactive unless its factory returns a policy for the current config.

### `policy_options`

- **Type:** table of tables
- **Default:** `{}`
- **Allowed / range:** keys must name registered policies

Plugin-owned configuration namespaces. A plugin reads only `training.policy_options.<policy_name>` and validates its own values; unknown policy names fail before training.

Built-in `training.policy_options.momentum_anchor` enables Momentum-Anchored
Orthogonal Projection (MAOP) from [Rosetta](https://arxiv.org/abs/2607.00293).
It is default-off; set `enabled=true` with `optimizer.type="adamw"`,
`optimizer.stochastic_rounding=false`, and
`training.optimizer_cpu_offload=false`. At each optimizer boundary, after
gradient accumulation and before global norm clipping, a negative global dot
product between the mixed gradient and the existing FP32 AdamW `exp_avg` is
removed over the complete trainable parameter vector. Parameters without an
initialized moment are unchanged. `start_after_steps` delays intervention by a
non-negative number of successful updates (default `0`); `epsilon` and
`min_anchor_norm_sq` default to `1e-12`; `chunk_size` bounds temporary FP32
work to positive element chunks (default `1048576`). Applied-step and
projection counters are checkpointed. Quantized, paged, stochastic-rounding,
CPU-offloaded, sparse-gradient, and multi-device optimizer states are rejected
rather than approximating the momentum anchor.

Built-in `training.policy_options.diversity_routing` is default-off. Set
`enabled=true`, `warmup_steps` to a positive integer (default `100`), and
`ridge` to a positive covariance regularizer (default `1e-4`).
`token_chunk_size` bounds replay scratch memory (default `2048`; reduce it on
memory-constrained GPUs). During warm-up
the native top-k is returned unchanged while exact per-layer co-selection
moments are accumulated. Later training forwards replay the frozen covariance
through deterministic Mahalanobis greedy top-k. The policy is single-GPU,
checkpointed, and incompatible with stochastic expert-subset or dataset-affinity
routing.

Built-in `training.policy_options.dispersive_loss` is default-off and requires
`training.batch_size >= 2`; gradient accumulation does not enlarge the physical
pair matrix. Set `enabled=true` to add the parameter-free Dispersive InfoNCE-L2
objective to one provider-exposed video representation. `weight` and
`temperature` must be finite and positive (paper defaults: `0.5` and `0.5`).
`layer_fraction` selects the first block at or beyond that fraction of model
depth and lies in `(0, 1]` (default `0.25`; `1.0` selects the final block).
`chunk_features` is a positive upper bound on flattened features converted to
FP32 at once (default `1048576`); chunking is exact and does not change the
full ordered-pair objective. The model provider excludes variable-length text
tokens and supplies one equal-size video representation per physical sample.
The policy has no trainable or resumable state; checkpoint metadata binds its
coefficient, temperature, resolved layer, and memory bound.

Built-in `training.policy_options.simbal` is default-off. Set `enabled=true`
to regularize every sparse-MoE router with the SIMBAL objective
`||W W^T - I||_1`, where rows of effective `W` correspond to experts. `weight`
is finite and positive (paper default used by Mirai: `0.1`). The per-router
entrywise-L1 terms are evaluated in FP32 and averaged over layers before the
coefficient is applied. The objective is independent of data and batch size.
It requires `hidden_size >= num_experts`, a router adapter on every MoE layer,
and cannot be enabled with `adapter.train_router=false`. The pretrained router
base is not reinitialized or made trainable: gradients flow through the
effective router-LoRA weight, and a staged router policy suppresses the term
outside its trainable window. The policy has no resumable state; checkpoint
metadata records the coefficient and fixed norm.

Built-in `training.policy_options.selective_sinkhorn` is default-off and
training-only. Set `enabled=true` to replace each sparse token-choice layer's
native route with Selective Sinkhorn Routing on a deterministic fraction of
microbatches. `probability` is in `(0, 1]` (default `0.001`). `cost_mode` is
`softmax` (SSR-S, default) or `linear` (SSR-L). `entropy_regularization` is the
positive Sinkhorn ξ (default `0.05`); `max_iterations` is positive (default
`100`); and `tolerance` is the positive L2 marginal-residual threshold (default
`1e-4`). `noise_scale` is the non-negative multiplier for optional standard
Gaussian cost noise (default `0`, disabled). `seed` is a non-negative integer
(default: `training.seed`). A BLAKE2b-derived stream keyed by seed, absolute
microbatch index, and layer name makes branch selection and noise identical
under resume and activation-checkpoint recomputation without consuming the
global Torch RNG. The solver uses FP32 log-domain scaling with row sums `1` and
column sums `tokens / experts`; padded tokens are excluded. On an active SSR
branch, both expert ids and normalized route weights come from the transport
plan and the task loss does not backpropagate through that detached plan. All
inference and non-SSR training branches use the model's native router exactly.
The policy requires `model.params.moe_routing_mode="token_choice"`,
`model.params.moe_balance_mode="off"`, and
`model.params.moe_router_z_loss_weight=0`; it rejects population phi balancing,
expert-subset routing, dynamic top-k, lightweight experts, dataset affinity,
and the `diversity_routing`, `expert_dropout`, and `simbal` policies.

Built-in `training.policy_options.prototypical_routing` is default-off. Set
`enabled=true` to attach one FP32 learnable prototype to every physical expert
in each sparse router. For provider-selected visual tokens, Mirai adds
`beta * prototype_scale * cosine(token, prototype)` to both the native expert
selection score and the bias-free gate score. The learned FP32 scalar `beta` is
always initialized to zero; therefore enabling the policy preserves native
expert ids and route weights at construction, including native group-limited
candidate selection. The learned route remains active during inference.

`prototype_scale` is finite and positive (default `1`).
`contrastive_weight` is finite and non-negative (default `1`) and multiplies the
training-only routing contrastive loss. `contrastive_temperature` is finite and
positive (default `0.07`). The loss follows ProMoE Equation 6: each active
expert prototype is classified against the mean embeddings of all tokens that
selected that expert; with top-k routing, a token belongs to every selected
expert set. Fewer than two active experts produce a graph-connected zero.
`seed` is a non-negative integer (default: `training.seed`) used by independent
per-layer generators, so initialization does not consume the global Torch RNG.
Prototype matrices, learned beta values, layer topology, and schema version are
stored with every supported adapter artifact and must match exactly on load.

This is a checkpoint-preserving adaptation of ProMoE's prototypical routing and
routing contrastive loss, not its from-scratch prototype-only router or its
conditional/unconditional partitioning stage. It requires token-choice routing,
balance mode `off`, zero router z-loss, zero population phi balancing, and
emergent dataset routing. It cannot compose with expert subsets, dynamic top-k,
lightweight or Chain-of-Experts routing, or the `diversity_routing`,
`expert_dropout`, `router_temperature`, `selective_sinkhorn`, and `simbal`
policies. See [the paper](https://arxiv.org/abs/2510.24711) and
[official implementation](https://github.com/ali-vilab/ProMoE).

Built-in `training.policy_options.sharp_moe` is default-off. Set `enabled=true`
and `training.objective="sharp_moe_trajectory"` to attach a two-layer SiLU
saliency router to every physical sparse router and train it with recursive
full-trajectory flow matching. `trajectory_steps` is the number of descending
uniform rollout points (default `10`, minimum `2`); the first point is fixed to
`0.999`. Each later router call receives the detached clean prediction
`x_t - t·v` from the preceding point. `router_hidden_dim` bounds the intermediate
width of the saliency MLP (default `128`), and `seed` defaults to
`training.seed`. The final projection is zero-initialized, preserving pretrained
routes at construction while allowing a gradient on the first update; the input
projection uses a policy-local deterministic generator. Policy state and the
objective's rollout RNG round-trip exactly in checkpoints and adapter exports.
Inference carries the previous clean prediction through sequential and batched
CFG. This fixed-top-k adaptation implements the paper's dual-router and
recursive-trajectory components; it does not apply the trajectory-allocation KL
loss, which the paper excludes for fixed-width token-choice DiTs. The policy
requires token-choice routing, MSE, uniform weighting, no bucket normalization
or prior batches, and rejects other route-selection owners. See
[the paper](https://arxiv.org/abs/2606.26938).

### Mixture-of-Depths policy options

Built-in `training.policy_options.mixture_of_depths` is default-off and enables
the bidirectional attention-routed Mixture-of-Depths path in providers that
declare support for it. The previous dense block computes each token's mean
received attention across query positions and heads. On configured later
blocks, the provider retains the exact top-capacity visual-token subset per
sample, keeps every valid conditioning token, runs the complete attention and
MoE block on that gathered sequence, and scatters its result over an identity
residual path. The same deterministic routing is active during training and
native inference; packed batches receive independent per-sample capacities.

#### `enabled`

- **Type:** bool
- **Default:** `false`
- **Constraint:** —

Enable attention-routed block skipping.

#### `capacity_fraction`

- **Type:** float
- **Default:** `0.5`
- **Constraint:** `0 < value < 1`

Fraction of valid visual tokens processed by each routed block. Capacity is `max(1, floor(valid_visual_tokens * value))` independently per sample.

#### `first_layer`

- **Type:** int
- **Default:** `1`
- **Constraint:** `>= 1`

Zero-based first routed block. Block zero stays dense so it can supply attention evidence.

#### `layer_stride`

- **Type:** int
- **Default:** `2`
- **Constraint:** `>= 2`

Distance between routed blocks; intervening dense blocks refresh the received-attention scores.

#### `attention_query_chunk_size`

- **Type:** int
- **Default:** `128`
- **Constraint:** `>= 1`

Query stripe size used to accumulate exact attention output and received-attention scores without retaining an additional full attention matrix.

This adaptation follows A-MoD's parameter-free received-attention router: it
does not multiply the block output by a routing score and does not train or
serialize an auxiliary predictor. Configuration must match when reproducing an
inference run because the policy has no learned state of its own. Providers
reject execution modes that cannot preserve the per-sample gathered-token
contract. See the original [Mixture-of-Depths
paper](https://arxiv.org/abs/2404.02258) and the [A-MoD
paper](https://arxiv.org/abs/2412.20875).

Built-in `training.policy_options.expert_dropout` is default-off. Set
`enabled=true` and `probability` in `(0, 1)` (default `0.4`). `start_step`
defaults to `0`; `end_step=0` keeps the policy active indefinitely, otherwise
the end is exclusive and must exceed the start. During an active training
forward, selected route slots are independently masked, an all-dropped token
retains its highest-weight route, and retained weights are renormalized to the
token's original gate mass. Evaluation and out-of-window steps return the
original score tensor unchanged. The policy requires top-k routing and is
incompatible with active expert-output orthogonality because that objective
requires all selected expert outputs.

Built-in `training.policy_options.router_temperature` is default-off. Set
`enabled=true`, positive `temperature` (default `1.0`), and
`minimum_temperature` (default: `temperature`, no annealing). `schedule` is
`constant`, `linear`, or `sigmoid`; annealed schedules require non-negative
`start_step` and a larger exclusive `end_step`. `sigmoid_sharpness` defaults to
`7.0`. The provider applies the resolved temperature to router logits before
the native score function, so both gate weights and any bias-sensitive
selection may change. `entropy_floor` defaults to `0`; when positive, annealing
freezes at the current temperature after detached assignment entropy falls
below that value. `jitter_epsilon` defaults to `0` and, when positive, multiplies
training logits by uniform noise in `[1-epsilon, 1+epsilon]`; evaluation never
uses jitter. Controller state and global torch RNG state are checkpointed.
Jitter remains explicitly opt-in because ST-MoE reported a stability/quality
trade-off rather than a universal quality improvement.

Built-in `training.policy_options.router_stage_schedule` is default-off. Set
`enabled=true`, `train_start_step` (default `0`), and optional exclusive
`freeze_step` (`0` means no final freeze). The policy keeps router-LoRA
parameters in optimizer groups, disables their gradients before the adaptation
window, enables them inside it, and disables them afterward; unrelated expert
and attention adapters continue training. It requires a router-inclusive target
preset and conflicts with `adapter.train_router=false`. Evaluation does not
change parameter trainability. The step and schedule round-trip in checkpoints.

Built-in `training.policy_options.router_distillation` is default-off. Set
`enabled=true`, positive `weight` (default `0.01`) and `temperature` (default
`1.0`); optional `start_step`/exclusive `end_step` bound the active window.
`weight_schedule="constant"` preserves the fixed coefficient. The opt-in
`"linear_decay"` schedule requires `end_step` and scales the coefficient from
one at `start_step` toward zero at `end_step`, adapting the DES-MoE Eq. 4
distillation schedule while leaving Mirai's primary video objective unchanged.
Each router-LoRA student is regularized against the immutable pretrained router
projection on the same tokens using temperature-scaled forward KL. The original
router weights are not duplicated, receive no gradients, and are protected by
an exact SHA-256 topology/content fingerprint in policy checkpoints. The policy
requires a router-inclusive target preset and conflicts with
`adapter.train_router=false`. When composed with `router_stage_schedule`, its
active interval must lie entirely inside the router-adaptation interval.

Built-in `training.policy_options.domain_expert_specialization` is default-off.
Set `enabled=true`, a non-empty `domain_metadata_key`, positive `warmup_steps`
and `min_experts`, `affinity_threshold` in `(0,1]`, `momentum` in `[0,1)`, and
positive `update_interval`. Mixed-domain microbatches are supported after
warmup when every domain has learned affinity: native expert tensors require
`adapter.expert_tensor_lora_backend="activation"`, while compressed experts
require the sorted `persistent` or `torch_grouped` dispatch backend.
Homogeneous batches retain the existing backend-independent path, and different
domains may also share a gradient-accumulation window. Nonzero weight decay is
supported exactly with `optimizer.type="adamw"`; other optimizer types must use
zero decay while this policy is enabled.

## `[optimizer]` — OptimizerConfig

### `type`

- **Type:** str
- **Default:** `"adamw"`
- **Allowed / range:** registry-driven: `adamw`, `adamw_8bit`, `paged_adamw_8bit`, `prodigy`, `adafactor`, `lion`, `came`, `selected_expert_adamw`, `selected_expert_adamw_4_2bit`, `selected_expert_adam_mini`, `selected_expert_muon`, `selected_expert_adamuon`, `lora_pro_adamw`, `lora_muon`

Optimizer registry ([`optimizer.py`](mirai/core/training/optim/optimizer.py)). All `selected_expert_*` optimizers bind the exact per-parameter expert plan and store state only for selected expert-axis rows. `selected_expert_adamw_4_2bit` implements [SOLO](https://arxiv.org/abs/2505.00347) for adapter fine-tuning: signed DE4 first moments, unsigned logarithmic QEMA2 second moments, 128-value blocks, quantile 0.1, and `betas=(0.8,0.999)`. Packed codes and FP32 block metadata are persistent; each EMA and parameter update uses temporary FP32 tensors. `selected_expert_adam_mini` adapts the official [Adam-mini v1.1.1 MLP partition](https://github.com/zyushun/Adam-mini) to rank-3 grouped experts: the first moment has shape `[selected_experts,out,in]`, while the second moment has shape `[selected_experts,out,1]`, giving one adaptive rate per expert/output neuron. The exact selection, partition, and stochastic-write policy are checkpoint-bound. `selected_expert_muon` applies the RMS-aligned polar update from [Moonlight](https://arxiv.org/abs/2502.16982) independently to every selected `[out,in]` expert matrix. `selected_expert_adamuon` implements [AdaMuon](https://arxiv.org/abs/2507.11005): `sign` stabilization before the polar operator, an uncorrected element-wise second moment of the orthogonal direction, and per-matrix RMS alignment. Both require rank-3 dense grouped-expert tensors and retain FP32 compact state; they never flatten the expert axis into the matrix geometry. `lora_pro_adamw` applies [LoRA-Pro](https://arxiv.org/abs/2407.18242) Equations 33–34 and Algorithm 2 to dense and grouped-expert LoRA factors. It requires `adapter.type="lora"`, zero weight decay, equal A/B learning rates, optimizer CPU offload disabled, and fixed unmasked factor pairs; DoRA, LoRA-FA, LoRA dropout, scalar/timestep rank schedules, sparse expert selection, condenser factors, and any other unpaired trainable parameters are rejected. Its FP32 first and second moments each have the shape of the equivalent full target weight, including the expert axis, so state memory is `2 × number_of_equivalent_weight_elements × 4` bytes rather than LoRA-sized. `lora_muon` implements [LoRA-Muon](https://arxiv.org/abs/2606.12921) Algorithm 1 for the actual scaled low-rank weight. Polar-Express matrix-sign and inverse-root Newton–Schulz passes update dense and grouped-expert pairs with one FP32 first moment per factor. Standard zero-B LoRA uses the finite one-sided boundary update until both factors are full rank. LoRA-Muon accepts the same fixed, complete, unmasked standard-LoRA surface as LoRA-Pro and rejects unpaired trainables, DoRA, LoRA-FA, factor/rank masks, sparse expert selection, condenser factors, and optimizer CPU offload. Post-hoc sparse export remains supported by both optimizers.

### `lr`

- **Type:** float
- **Default:** `1e-3`
- **Allowed / range:** >0

Learning rate ([`optimizer.py`](mirai/core/training/optim/optimizer.py), [`scheduler.py`](mirai/core/training/optim/scheduler.py)).

### `weight_decay`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** >=0

Weight decay ([`optimizer.py`](mirai/core/training/optim/optimizer.py)).

### `weight_decay_filter`

- **Type:** str
- **Default:** `"lora_b_bias"`
- **Allowed / range:** `none`, `lora_b_bias`, `router_aware` ([`optimizer.py`](mirai/core/training/optim/optimizer.py))

Which params skip weight decay ([`optimizer.py`](mirai/core/training/optim/optimizer.py)). `router_aware` extends `lora_b_bias` by also exempting sparse-MoE router parameters, whose decay flattens the routing distribution carried by the frozen checkpoint.

### `loraplus_lr_ratio`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** >0; 1.0 = off

LoRA+ B/A LR ratio ([`optimizer.py`](mirai/core/training/optim/optimizer.py)).

### `prodigy_beta3`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `0` or `(0, 1)`

Prodigy distance-estimator EMA coefficient; `0` selects `sqrt(beta2)`.

### `prodigy_decouple`

- **Type:** bool
- **Default:** `True`
- **Allowed / range:** —

Use decoupled AdamW-style weight decay in Prodigy.

### `prodigy_use_bias_correction`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Apply Adam bias correction to the Prodigy step scale.

### `prodigy_safeguard_warmup`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Exclude external learning-rate warmup from the denominator of the Prodigy distance estimate.

### `prodigy_d0`

- **Type:** float
- **Default:** `1e-6`
- **Allowed / range:** finite `> 0`

Initial Prodigy distance estimate.

### `prodigy_d_coef`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** finite `> 0`

Multiplier applied to the estimated distance.

### `prodigy_growth_rate`

- **Type:** float
- **Default:** `inf`
- **Allowed / range:** `>= 1`

Per-step upper bound on multiplicative growth of the distance estimate; `inf` is unbounded.

### `prodigy_slice_p`

- **Type:** int
- **Default:** `1`
- **Allowed / range:** `>= 1`

Compute distance statistics from every Nth flattened parameter element; `1` is exact.

### `lora_pro_damping`

- **Type:** float
- **Default:** `1e-8`
- **Allowed / range:** finite and >0; used only by `lora_pro_adamw`

Positive FP32 stabilization added to LoRA-Pro Gram inverses and used as the Sylvester denominator floor ([`lora_pro.py`](mirai/core/training/optim/lora_pro.py)). It does not enable LoRA-Pro by itself.

### `muon_momentum`

- **Type:** float
- **Default:** `0.95`
- **Allowed / range:** finite, `[0,1)`; used by Muon-family optimizers

Momentum coefficient. `lora_muon` applies an EMA independently to the two factor gradients. Selected-expert Muon/AdaMuon use the published `M ← βM + G` recurrence for every selected expert matrix.

### `muon_nesterov`

- **Type:** bool
- **Default:** `True`
- **Allowed / range:** selected-expert Muon/AdaMuon

Use the source implementations' Nesterov direction `G + βM` after updating momentum. `False` follows AdaMuon Algorithm 1 directly.

### `muon_ns_steps`

- **Type:** int
- **Default:** `5`
- **Allowed / range:** `>=1`; selected-expert Muon/AdaMuon

Number of quintic Newton–Schulz polar iterations. The optimized path runs each selected expert matrix independently.

### `muon_eps`

- **Type:** float
- **Default:** `1e-8`
- **Allowed / range:** finite, `>0`; selected-expert Muon/AdaMuon

AdaMuon denominator and RMS-normalization floor. It is retained as optimizer policy for both selected-expert variants.

### `muon_rms_target`

- **Type:** float
- **Default:** `0.2`
- **Allowed / range:** finite, `>0`; selected-expert Muon/AdaMuon

Target element-wise RMS of each matrix update. `0.2` is the Moonlight/AdaMuon alignment used to share Adam-style learning-rate schedules.

### `lora_muon_gauge_rebalance_interval`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>=0`; `0` disables

Apply LoRA-Muon Appendix B.1 scalar gauge rebalancing every N optimizer steps. Rebalancing preserves the composed low-rank weight and transports both first moments by the reciprocal gauge action.

### `lora_muon_gauge_rebalance_alpha`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** finite, `(0,1]`

Damping exponent for optional scalar gauge rebalancing. It is checkpoint-bound even when the interval is zero so resume cannot silently change the optimizer policy.

### `stochastic_rounding`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** `adamw`, paired-LoRA optimizers, or any `selected_expert_*` optimizer; incompatible with `training.optimizer_cpu_offload=true`

Stochastically round BF16 parameter updates without persistent FP32 master weights. Standard AdamW, native-state selected-expert AdamW, and selected-expert Adam-mini retain BF16 moments and stochastically round their bounded FP32 EMA updates back to BF16 so sub-ULP moment changes do not deterministically stall. The SOLO 4/2-bit variant retains packed low-bit moments and uses this switch only for the final BF16 parameter write. LoRA-Pro, LoRA-Muon, selected-expert Muon, and selected-expert AdaMuon retain geometric moments in FP32 and apply stochastic rounding only when writing updated parameters. The global torch RNG is required checkpoint state so stochastic writes continue deterministically after restore.

### `allow_fallback`

- **Type:** bool
- **Default:** `True`
- **Allowed / range:** —

Allow fallback to adamw when the requested impl is unavailable ([`optimizer.py`](mirai/core/training/optim/optimizer.py)).

### `scheduler`

- **Type:** str
- **Default:** `"constant"`
- **Allowed / range:** registry-driven: `constant`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `rex`

LR scheduler ([`scheduler.py`](mirai/core/training/optim/scheduler.py)).

### `scheduler_num_cycles`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** —

Cosine-restart cycles ([`scheduler.py`](mirai/core/training/optim/scheduler.py)).

### `scheduler_power`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** —

Polynomial power ([`scheduler.py`](mirai/core/training/optim/scheduler.py)).

### `min_lr_ratio`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** in [0, 1)

Floor LR ratio ([`scheduler.py`](mirai/core/training/optim/scheduler.py)).

### `selected_expert_ids`

- **Type:** list[int]
- **Default:** `[]`
- **Allowed / range:** unique non-negative ids

Manual expert set applied to every MoE layer by the selected-expert optimizer. Required when `adapter.type="selected_expert"` and `adapter.expert_selection="all"`; must remain empty for automatic ESFT selection.

## `[adapter]` — AdapterConfig

### `type`

- **Type:** str
- **Default:** `"lora"`
- **Allowed / range:** `lora`, `sparse_delta`, `selected_expert`

Training mode. `sparse_delta` trains a fixed-support high-rank delta; `selected_expert` directly updates dense grouped-expert rows and requires a `selected_expert_*` optimizer.

### `target_preset`

- **Type:** str
- **Default:** `"attn_only"`
- **Allowed / range:** e.g. `attn_router_routed_experts`

Which module groups get adapters ([`adapters.py`](mirai/core/training/adapters.py), [`lora.py`](mirai/core/models/adapters/lora.py)).

### `rank`

- **Type:** int
- **Default:** `16`
- **Allowed / range:** >=1

LoRA rank ([`lora.py`](mirai/core/models/adapters/lora.py)).

### `alpha`

- **Type:** float
- **Default:** `16.0`
- **Allowed / range:** >0

LoRA alpha ([`lora.py`](mirai/core/models/adapters/lora.py)).

### `rank_pattern`

- **Type:** table{str:int}
- **Default:** `{}`
- **Allowed / range:** non-empty glob keys, values >0; every pattern must match exactly and targets may not match multiple patterns

Static target-level rank overrides, resolved against provider-declared concrete linear and expert-tensor targets before injection ([`lora_allocation.py`](mirai/core/models/adapters/lora_allocation.py)).

### `alpha_pattern`

- **Type:** table{str:float}
- **Default:** `{}`
- **Allowed / range:** non-empty glob keys, finite values >0; every pattern must match exactly and targets may not match multiple patterns

Static target-level alpha overrides ([`lora_allocation.py`](mirai/core/models/adapters/lora_allocation.py)).

### `rank_budget`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0; `0` disables; otherwise sum of resolved target ranks must not exceed it

Hard allocation ceiling. It rejects over-budget plans; it does not silently renormalize ranks ([`lora_allocation.py`](mirai/core/models/adapters/lora_allocation.py)).

### `adaptive_rank_plan_path`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** empty disables; otherwise a schema-v1 JSON plan

Loads a lineage-bound, content-addressed fixed-budget rank plan before LoRA injection. The plan must cover the provider's exact target topology; dataset/model/config lineage is verified before optimizer construction. It cannot be combined with `rank_pattern`. Build the artifact with [`scripts/tools/calibrate_adaptive_lora_ranks.py`](scripts/tools/calibrate_adaptive_lora_ranks.py); the configured path must already equal the requested output so config-file lineage remains stable.

### `use_rslora`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Opt-in rank-stabilized LoRA scaling `alpha/sqrt(rank)` instead of `alpha/rank`. The rule is recorded in Mirai adapter state and export metadata; mismatched checkpoint loads fail ([`lora_allocation.py`](mirai/core/models/adapters/lora_allocation.py), [`lora.py`](mirai/core/models/adapters/lora.py)).

### `use_dora`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** requires `type="lora"`, an approving model provider, unpacked weights, and `expert_tensor_lora_backend="weight_space"`

Enables [DoRA](https://arxiv.org/abs/2402.09353): LoRA represents the direction `W + sBA`, while a trainable output-channel vector represents its magnitude. Mirai evaluates the normalization in FP32 and applies runtime adapter scaling to the complete decomposed update, so scale zero restores the frozen base exactly. Dense linear and grouped expert weight-space targets are supported. Packed/compressed bases, activation-space expert execution, LoRA-FA, rank/timestep masks, sparse expert selection, condenser adapters, LoftQ, LoRA-GA, and GoRA are rejected explicitly. Native state uses `*.dora_magnitude`; Kohya `*.dora_scale` and Diffusers/PEFT `*.lora_magnitude_vector.weight` round-trip through the adapter loader ([`dora.py`](mirai/core/models/adapters/dora.py), [`lora.py`](mirai/core/models/adapters/lora.py)).

### `init_from`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** —

Initialize adapter from a checkpoint ([`session_components.py`](mirai/core/training/lifecycle/session_components.py)).

### `rank_dropout`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** in [0, 1]

Existing stochastic adapter dropout. Dense and active-expert hosts apply it to adapter inputs; the weight-space expert-tensor host applies elementwise dropout to A. It is not structured rank-channel dropout ([`lora.py`](mirai/core/models/adapters/lora.py)).

### `lora_parameter_dropout`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** finite, in [0, 1); incompatible with `use_lora_fa=true`

Structured LoRA parameter dropout from Equation 2 of [LoRA Dropout](https://arxiv.org/abs/2404.09610): independently masks input-feature columns of A and output-feature rows of B during training, without inverted-dropout rescaling or rank-axis masking. Evaluation and `0.0` preserve the original factors and consume no RNG. Applies to dense, weight-space expert-tensor, activation-space expert-tensor, and compressed active-expert LoRA hosts; shared condenser factors are intentionally unchanged ([`lora_parameter_dropout.py`](mirai/core/models/adapters/lora_parameter_dropout.py)).

### `posthoc_rank_compression`

- **Type:** str
- **Default:** `"off"`
- **Allowed / range:** `off`, `para`; `para` requires `type="lora"`

Arms the offline [PARA](https://arxiv.org/abs/2604.27796) adapter transform in [`scripts/tools/compress_adapter_para.py`](scripts/tools/compress_adapter_para.py); it never changes training by itself. PARA computes each trained `B @ A` spectrum in the rank-width QR subspace, applies one global rank-budget or retained-energy threshold across every dense and per-expert LoRA matrix, and writes a lineage-bound ragged safetensors artifact. Loading that artifact reconstructs zero-padded runtime factors at each grouped tensor's maximum retained expert rank while preserving the original effective LoRA/rsLoRA scale; zero-rank units remain exact zero updates ([`para.py`](mirai/core/models/adapters/para.py)).

### `rank_schedule_start`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0

Rank-schedule start step ([`training_step_pre.py`](mirai/core/training/lifecycle/training_step_pre.py)).

### `rank_schedule_end`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0 and >= start

Rank-schedule end step ([`training_step_pre.py`](mirai/core/training/lifecycle/training_step_pre.py)).

### `rank_schedule_min_scale`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** in (0, 1]; 1.0 = off

Min rank scale ([`training_step_pre.py`](mirai/core/training/lifecycle/training_step_pre.py)).

### `lr_multipliers`

- **Type:** table{str:float}
- **Default:** `{}`
- **Allowed / range:** must be a table

Per-group LR multipliers ([`optimizer.py`](mirai/core/training/optim/optimizer.py)). Recognized groups: `cross_attn`, `self_attn`, `router`, `ffn`, `temporal`, `other`; the first match on the parameter path wins. Router parameters are nested under the ffn path, so `router` is matched before `ffn` and falls back to the `ffn` value when unset.

### `expert_selection`

- **Type:** str
- **Default:** `"all"`
- **Allowed / range:** `all`, `routing_topk`, `esft_gate`, `esft_token`

Selection policy. `routing_topk` is MoE-Sieve masking for routed-expert LoRA and requires a supported quantized expert format. `esft_gate` and `esft_token` require `type="selected_expert"` and build a per-layer direct-update plan from downstream routing affinity before optimizer construction.

### `expert_topk_fraction`

- **Type:** float
- **Default:** `0.25`
- **Allowed / range:** in (0, 1] when `routing_topk`

Fraction of experts kept after routing calibration.

### `expert_calibration_steps`

- **Type:** int
- **Default:** `8`
- **Allowed / range:** >=1 when `routing_topk`

Routing-calibration steps.

### `esft_selection_mass`

- **Type:** float
- **Default:** `0.2`
- **Allowed / range:** finite, in `(0, 1]`

ESFT Equation 8 threshold `p`. Each layer selects the smallest stable top-score prefix whose normalized cumulative relevance reaches this mass.

### `esft_calibration_samples`

- **Type:** int
- **Default:** `32`
- **Allowed / range:** `>= 1`

Minimum number of deterministic downstream samples used by the pre-optimizer ESFT pass. The final batch may make the observed count slightly larger.

### `sparse_expert_export`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Compact sparse-expert adapter export (Mirai-native; not kohya/ComfyUI-portable without expansion) ([`adapters.py`](mirai/core/training/adapters.py), [`lora.py`](mirai/core/models/adapters/lora.py)).

### `sparse_delta_density`

- **Type:** float
- **Default:** `0.01`
- **Allowed / range:** in (0, 1]

Fraction of each matched dense projection exposed as trainable fixed-support high-rank delta values.

### `sparse_delta_selection`

- **Type:** str
- **Default:** `"magnitude"`
- **Allowed / range:** `magnitude`, `random`

Immutable support selection stored with the sparse-delta artifact.

### `timestep_rank_schedule`

- **Type:** str
- **Default:** `"none"`
- **Allowed / range:** `none`, `tlora`, `tc_gate` ([`schema.py`](mirai/config/schema.py), rc)

Rank-axis timestep modulation. `tlora` = T-LoRA timestep-dependent rank masking (arXiv 2507.05964): at high post-shift noise only a low-rank prefix is active, growing linearly to full rank at sigma 0; masked ranks give exactly-zero output+grads. `tc_gate` = TC-LoRA (arXiv 2510.09561) learned gate: a hypernetwork emits a per-rank gate g(sigma) multiplied into the low-rank intermediate, init-to-identity so it is bit-identical to static LoRA at step 0 ([`timestep_axis.py`](mirai/core/models/adapters/timestep_axis.py), [`tc_lora.py`](mirai/core/models/adapters/tc_lora.py), [`lora.py`](mirai/core/models/adapters/lora.py)).

### `timestep_rank_min_fraction`

- **Type:** float
- **Default:** `0.25`
- **Allowed / range:** in (0, 1] ([`schema.py`](mirai/config/schema.py), rc)

Fraction of rank active at sigma == 1.0 under `tlora` ([`timestep_axis.py`](mirai/core/models/adapters/timestep_axis.py)).

### `lora_init`

- **Type:** str
- **Default:** `"kaiming"`
- **Allowed / range:** registered initializer; built-ins: `kaiming`, `orthogonal`, `loftq`, `eva`, `pissa`, `lora_ga`, `gora`

LoRA initializer registry. `orthogonal` is the T-LoRA rank-prefix pairing. `loftq` is the one-iteration, fixed-quantized-base variant: after live int8/NF4/GGUF quantization, each target initializes from the top-r SVD of `W_bf16 - dequant(Q)`. `eva` is the paper's fixed-rank (`rho=1`) mode: downstream activation PCs initialize A and B stays zero. `pissa` moves the principal rank-r SVD component into trainable factors and freezes the exact residual base; already-packed expert artifacts fail because their residual cannot be rewritten. `lora_ga` accumulates bounded full-weight gradients for dense attention targets, applies A2rBr gradient-SVD initialization, subtracts the initial adapter from the frozen base, and then releases calibration gradients. `gora` performs pre-optimizer sensitivity allocation and pseudo-inverse gradient initialization without changing the frozen base. EVA supports dense and explicit compressed routed-expert hosts, runs before the first optimizer step, rejects compiled calibration and weight-parametrization experts, and does not claim shape-changing rank redistribution ([`lora_initialization.py`](mirai/core/models/adapters/lora_initialization.py), [`lora_eva.py`](mirai/core/models/adapters/lora_eva.py), [`lora_ga.py`](mirai/core/training/calibration/lora_ga.py), [`gora.py`](mirai/core/training/calibration/gora.py)).

### `lora_ga_calibration_steps`

- **Type:** int
- **Default:** `4`
- **Allowed / range:** > 0

Number of gradient-estimation batches before LoRA-GA initialization.

### `lora_ga_stable_gamma`

- **Type:** float
- **Default:** `16.0`
- **Allowed / range:** > 0

Stable LoRA-GA factor scaling constant.

### `gora_calibration_steps`

- **Type:** int
- **Default:** `64`
- **Allowed / range:** `>= 1`

Number of deterministic downstream batches whose full target-weight gradients are averaged on CPU before optimizer construction. Used only by `lora_init = "gora"`.

### `gora_min_rank`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>= 0`

GoRA rank floor. `0` resolves to `max(1, rank // 2)`, matching the paper's default rule.

### `gora_max_rank`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `0` or `>= gora_min_rank`

GoRA rank ceiling. `0` resolves to `4 * rank`, matching the paper's default rule; each target is additionally bounded by its matrix dimensions.

### `gora_stable_gamma`

- **Type:** float
- **Default:** `0.05`
- **Allowed / range:** finite, `> 0`

Initial negative-gradient step magnitude in GoRA Equation 10. GoRA requires `use_rslora = true`, unquantized calibration weights, no compiled calibration, and no competing rank allocator.

### `eva_calibration_steps`

- **Type:** int
- **Default:** `32`
- **Allowed / range:** `>= 2`

Maximum downstream-data forwards for EVA principal-vector convergence; a non-converged target fails explicitly.

### `eva_samples_per_target`

- **Type:** int
- **Default:** `256`
- **Allowed / range:** `>= 1`

Deterministic per-update activation-row cap for each dense target or physical expert. Incremental state stays on CPU and each SVD is bounded to the retained rank plus this sample cap ([`lora_eva.py`](mirai/core/models/adapters/lora_eva.py)).

### `eva_convergence_threshold`

- **Type:** float
- **Default:** `0.99`
- **Allowed / range:** in `(0, 1]`

Absolute cosine-similarity threshold applied independently to every retained right-singular vector ([`lora_eva.py`](mirai/core/models/adapters/lora_eva.py)).

### `use_lora_fa`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

LoRA-FA (arXiv:2308.03303): freeze every adapter A factor and apply the paper's corrected B gradient `grad_B @ pinv(A A^T) / scale^2`. Explicit low-rank dense and compressed-expert paths retain only the rank-width projected activation for adapter gradients. Incompatible with rank dropout and timestep rank masks because those make A step-dependent. Default `false` preserves trainable parameters, gradients, state, and metadata ([`lora_fa.py`](mirai/core/models/adapters/lora_fa.py)).

### `expert_tensor_lora_backend`

- **Type:** str
- **Default:** `"weight_space"`
- **Allowed / range:** `weight_space`, `activation`

Execution policy for provider-declared native rank-3 expert tensors. `weight_space` is the established reference and materializes `B @ A` as `[experts,out,in]`. Opt-in `activation` uses the host extension contract to evaluate `(x @ A.T) @ B.T` in grouped/loop dispatch without materializing that delta; it requires a complete w1/w2/w3 target set and fails explicitly otherwise ([`expert_tensor_lora.py`](mirai/core/models/adapters/expert_tensor_lora.py)). Compressed-weight experts already use their dedicated activation-space adapter path and are unaffected.

### `tc_gate_hidden_dim`

- **Type:** int
- **Default:** `8`
- **Allowed / range:** `>= 1` ([`schema.py`](mirai/config/schema.py), rc)

TC-LoRA (arXiv 2510.09561) gate hypernetwork hidden width; only consulted when `timestep_rank_schedule = "tc_gate"`. The gate is init-to-identity (zero-init output projection) so this value never perturbs step-0 defaults ([`tc_lora.py`](mirai/core/models/adapters/tc_lora.py), provider adapter host).

### `timestep_band_min`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** 0 <= min < max ([`schema.py`](mirai/config/schema.py), rc)

Adapter trains only on samples with post-shift sigma in [min, max): out-of-band samples get exactly-zero adapter output AND grads; frozen forward + loss unaffected ([`timestep_axis.py`](mirai/core/models/adapters/timestep_axis.py), provider adapter host).

### `timestep_band_max`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** min < max <= 1 ([`schema.py`](mirai/config/schema.py), rc)

Upper band edge, inclusive at 1.0 so pure-noise samples are kept. Complementary ranges can train separate high-noise and low-noise adapters.

### `condenser_rank`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>= 0`

Condenser LoRA guardrail (ExpertCondenser, arXiv 2604.23036): a SHARED always-on low-rank term (rank ~2-4) per routed-expert tensor key (w1/w2/w3), added uniformly to every token and NEVER masked by `expert_selection`/expert-subset — protects fragmented long-tail knowledge when the subset starves cold experts. `0` (default) = off, byte-identical. Quantized (int8/nf4) `ActiveExpertLoRA` path only; exported under `*.condenser_a`/`*.condenser_b` ([`experts.py`](mirai/core/models/compressed_weights/execution/experts.py), [`lora.py`](mirai/core/models/adapters/lora.py)).

### `condenser_alpha`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `>= 0`

Condenser scale alpha; `<= 0` falls back to `condenser_rank` (scale 1.0).

### `train_router`

- **Type:** bool/None
- **Default:** `None`
- **Allowed / range:** `true`/`false`/unset

Router policy under PEFT. `None` (default) preserves the preset's behavior (router adapted only under a router-inclusive preset e.g. `attn_router_routed_experts`) and warns when adapted. `false` force-freezes router adapters and avoids the dense all-expert forward required for exact router gradients. `true` is an explicit opt-in ([`adapters.py`](mirai/core/training/adapters.py), provider adapter host).

### ESFT direct expert selection

`adapter.type = "selected_expert"` with `expert_selection = "esft_gate"` or
`"esft_token"` runs a deterministic, forward-only calibration before optimizer
construction:

- `esft_gate` implements Equation 6 by accumulating the selected gate weights.
- `esft_token` implements Equation 7 by assigning each selected route `1 / K`.
- Both normalize relevance per layer and implement Equation 8 by choosing the
  minimum deterministic expert prefix reaching `esft_selection_mass`.
- The resulting plan may differ by layer. Every `selected_expert_*` optimizer
  stores state only for each parameter's selected rows, and the checkpoint
  stores the exact per-layer plan plus only those rows.
- Calibration uses evaluation routing and restores sampler, torch, and session
  RNG state. It does not limit runtime routing: unselected experts still serve
  tokens but remain unchanged by the optimizer.

ESFT requires dense unquantized grouped experts, fixed-cardinality token-choice
routing, `training.compile = false`, no lightweight expert topology, no optimizer
CPU offload, and an empty `optimizer.selected_expert_ids`. Mirai does not claim
the source's language-model speed or quality deltas for video training; the
implemented resource guarantee is compact optimizer/checkpoint state.

### Routing-stability metrics (`metrics.jsonl` / `step.completed`)

Emitted per step (detached; loss graph untouched), riding the same telemetry lift as `moe_balance_loss`/`moe_z_loss`. Present whenever routers produced a selection:

- `moe_routing_entropy` — mean Shannon entropy of the per-expert assignment distribution (over experts, per layer). Higher = more balanced routing.
- `moe_utilization_cv` — mean coefficient of variation of per-expert load; higher = more imbalanced.
- `moe_top1_monopoly` — mean max per-expert assignment fraction; ≈1 signals monopoly/collapse.
- `moe_routing_kl_vs_step0` — mean KL of the current per-expert assignment distribution vs a step-0 snapshot (drift from the pretrained router). Absent on the first forward (snapshot captured then).
- `moe_step_unique_experts` — mean unique experts actually touched per layer per step; this is the expert-transfer working-set count.

### Expert-Choice coverage guard (`model.params.moe_expert_choice_coverage_alarm_threshold > 0`; default off)

Detached per-step coverage telemetry from the already-materialized Expert-Choice routing decision. The guard performs no collection at the default threshold `0`.

- `moe_expert_choice_coverage_fraction` — mean fraction of tokens selected by at least one expert across observed MoE layers.
- `moe_expert_choice_min_coverage_fraction` — worst observed per-layer coverage; this prevents a healthy layer average from hiding one starved layer.
- `moe_expert_choice_coverage_alarm` — `1` when the worst-layer fraction is below the configured threshold, otherwise `0`.

### Balance/task gradient ratio (`model.params.moe_balance_grad_ratio_telemetry = true`; default off)

LongCat Equation 9 telemetry uses the task loss before auxiliary objectives are added. It compares gradients on the same graph-bearing router-probability tensors used by expert weighting; it does not substitute router-parameter gradients. Token-axis gradient sums form the derivative with respect to each layer's batch-mean expert-probability vector, and layer vectors are concatenated for the global norm. Metrics are absent when the task has no gradient path to router probabilities.

- `moe_balance_grad_ratio` — weighted native load-balance gradient norm divided by task gradient norm.
- `moe_balance_grad_ratio_max_layer` — largest defined per-layer ratio.
- `moe_balance_grad_ratio_task_norm` / `moe_balance_grad_ratio_objective_norm` — denominator and numerator norms used by the global ratio.
- `moe_balance_grad_ratio_alarm` — `1` when the ratio reaches `moe_balance_grad_ratio_threshold`.
- `moe_phi_balance_grad_ratio*` — the same per-objective breakdown when phi-balancing is active.

### Routing-health pack (`model.params.moe_routing_health = true`; default off)

Detached per-step alarms from visual-DiT routing diagnosis and router-adaptation guardrails. Absent unless the opt-in gate is set. Owners: [`mirai/core/moe/monitoring/health.py`](mirai/core/moe/monitoring/health.py) and [`mirai/core/moe/monitoring/drift.py`](mirai/core/moe/monitoring/drift.py).

- `moe_expert_output_cossim` — expert-output homogenization proxy for the paper's "global soft saturation." Exact all-pairs per-token expert-output cosine is unavailable because sparse top-k dispatch materializes only each token's selected expert outputs. The estimator samples 256 token slots per step and, for each MoE layer, averages pairwise mean-centered cosine between per-expert router-response columns before averaging over layers. Mean-centering removes the shared positive softmax offset; values near one indicate similar router-response columns. This is a router-response-space proxy observed one step upstream of output space.
- `moe_max_deadlock_duration` — longest run (in steps) of any MoE layer whose top-1 expert claimed ≥ 0.90 of routed tokens (the paper's "selective deadlock"). Cross-step per-layer streak counters; advances on training-mode forwards only.
- `moe_deadlocked_layer_count` — number of layers currently in a single-expert deadlock this step.
- `moe_deadlocked_layer_count_depth_q1` … `moe_deadlocked_layer_count_depth_q4` — the same current-step count split into shallow-to-deep quartiles over the provider's complete ordered router list. Empty quartiles report zero.
- `moe_max_deadlock_duration_depth_q1` … `moe_max_deadlock_duration_depth_q4` — longest current deadlock streak within each depth quartile. These make shallow/deep concentration observable without assuming a model-specific layer count.
- `moe_router_underflow_fraction` — fraction of router weights whose intended bf16 update `|lr*grad|` is below half a bf16 ULP at that weight's magnitude and is therefore truncated to zero. Rides `grad_breakdown`. **Telemetry only**; FP32 router master parameters are controlled independently. With `adapter.train_router=False` (recommended default) router grads are zero → reports `0.0`; when no router params exist the metric is absent.
- `moe_router_logit_drift` — layer-mean normalized RMS drift of the current per-expert raw-logit mean from the first observed training forward. The bounded reference is checkpointed with the adapter; layer topology or expert-width changes fail explicitly.
- `moe_router_logit_drift_context` — structured per-step drill-down for the same observation: timestep min/max/mean, latent video resolution, and per-layer normalized RMS plus the index and relative magnitude of the most-drifted expert. It stores summaries only, never token logits.
- `moe_router_grad_param_ratio` — global router gradient RMS divided by router parameter RMS, including CPU-offloaded gradients.
- `moe_router_update_param_ratio` — estimated optimizer-step scale `|lr| × grad_RMS / parameter_RMS`; telemetry does not mutate parameters or optimizer state.
- `moe_router_weight_similarity` — Equation 7 of arXiv:2606.28116: the exact mean cosine over ordered distinct effective router-weight rows, reduced to linear complexity in expert count. Its operating point is architecture-dependent; high similarity is not itself a collapse certificate.
- `moe_router_conditioning_ratio` — `max_i ||w_i-w̄|| / ||w̄||`, emitted only when every expert row and the common mean are nonzero. Small values identify a common-mode-dominated parameterization; softmax removes that common mode exactly.
- `moe_router_per_token_entropy` / `moe_router_per_token_entropy_fraction` — Equation 8 mean entropy of every token's complete effective routing distribution and the same value divided by `log(num_experts)`. This is distinct from `moe_routing_entropy`, which summarizes the batch's discrete top-k assignment counts.
- `moe_attention_qk_delta2_spectral_entropy` / `moe_attention_qk_delta2_effective_rank` — alpha-2 spectral shape of the exact first-order QK-product increment `Δ2=ΔWq·Wkᵀ+Wq·ΔWkᵀ`, averaged over eligible heads. Nonzero singular values are computed from QR cores without materializing the hidden-width-square operator. Min/max, head count, and changed-layer count accompany the mean.
- `moe_minority_expert_share` / `moe_normalized_minority_share` — mean share of routed slots held by each router's least-used expert, raw and divided by the uniform share `1/num_experts` (`1.0` is balanced). This is the minority side of arXiv:2605.19378's per-layer statistic; the dominant-side `moe_top1_monopoly` cannot express it for wide routers, where the top-1 share is bounded near `top_k/num_experts` while most experts are dead.
- `moe_dead_expert_fraction` / `moe_underused_expert_fraction` — mean fraction of experts that received no routed slot, and the fraction below the health baseline (one fifth of the uniform share, the paper's 10%-of-tokens line against a 50% uniform share).
- `moe_collapsed_router_fraction` / `moe_collapsed_router_count` / `moe_worst_normalized_minority_share` — routers under that baseline this step, and the worst one.
- `moe_max_collapse_duration` — longest consecutive collapsed run in steps, on the minority criterion (`moe_max_deadlock_duration` is the same idea on the ≥0.90 monopoly criterion).
- `moe_collapse_rebound_count` — routers that crossed back above the baseline after a sustained collapse; the paper's only observed self-recovery.
- `moe_collapse_plateau_fraction` — collapsed routers whose normalized minority share stopped moving over the window, the signature of a deadlock no balance pressure moves.

The Q/K monitor applies when both projections expose a static additive LoRA weight over a dense frozen linear base. Input-dependent timestep rank masks, TC-LoRA gates, DoRA normalization, and projection variants without the paper's explicit MHA `Wq/Wk` algebra are omitted rather than approximated. Its checkpointed factor snapshots define a window spanning one actually changed effective weight state. The source does not specify a numeric sampling interval or a universal healthy threshold, so Mirai emits no automatic preemptive alarm; a trajectory must be compared with a workload-specific healthy baseline.

## `[dataset]` — DatasetConfig

### `path`

- **Type:** str
- **Default:** `"./data"`
- **Allowed / range:** no control chars

Dataset root ([`loader.py`](mirai/core/training/data/loader.py),
[`cache.py`](mirai/core/dataset/cache.py)).

### `cache_path`

- **Type:** str
- **Default:** `"./cache/dataset_cache.pt"`
- **Allowed / range:** no control chars

Cache file ([`cache.py`](mirai/core/dataset/cache.py)).

### `auto_preprocess_cache`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Auto-build the cache (preprocessing/cache.py).

### `preprocess_raw_media_to_pt`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Preprocess raw media into `.pt` ([`media_preprocessor.py`](mirai/core/dataset/media/media_preprocessor.py), [`native_encode.py`](mirai/core/dataset/native_encode.py)).
Applies while `auto_preprocess_cache` builds a missing cache. Model families
that declare native cache encoding encode raw media with their own provider
encoder, so this generic step is skipped for them; a family that declares
neither native cache encoding nor asset-free projection rejects raw video here
rather than writing a pixel projection that is not a diffusion latent.

### `max_cache_skip_ratio`

- **Type:** float
- **Default:** `0.2`
- **Allowed / range:** in [0, 1]

Max fraction of skippable cache misses ([`cache.py`](mirai/core/dataset/cache.py)).

### `split_seed`

- **Type:** int
- **Default:** `42`
- **Allowed / range:** —

Train/val/test split seed ([`split.py`](mirai/core/dataset/split.py)).

### `train_ratio`

- **Type:** float
- **Default:** `0.8`
- **Allowed / range:** —

Train split fraction ([`loader.py`](mirai/core/training/data/loader.py)).

### `val_ratio`

- **Type:** float
- **Default:** `0.1`
- **Allowed / range:** —

Val split fraction ([`loader.py`](mirai/core/training/data/loader.py)).

### `test_ratio`

- **Type:** float
- **Default:** `0.1`
- **Allowed / range:** —

Test split fraction ([`loader.py`](mirai/core/training/data/loader.py)).

### `usage_mode`

- **Type:** str
- **Default:** `"internal"`
- **Allowed / range:** —

Dataset usage/licensing mode ([`registration.py`](mirai/core/dataset/registration.py)).

### `num_workers`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0

DataLoader workers ([`loader.py`](mirai/core/training/data/loader.py)).

### `prefetch_factor`

- **Type:** int
- **Default:** `2`
- **Allowed / range:** >=1

DataLoader prefetch factor ([`loader.py`](mirai/core/training/data/loader.py)).

### `persistent_workers`

- **Type:** bool
- **Default:** `True`
- **Allowed / range:** —

Persistent DataLoader workers ([`loader.py`](mirai/core/training/data/loader.py)).

### `cache_mode`

- **Type:** str
- **Default:** `"disk"`
- **Allowed / range:** —

Cache residency mode ([`cache.py`](mirai/core/dataset/cache.py)).

### `cache_compression`

- **Type:** str
- **Default:** `"none"`
- **Allowed / range:** —

Cache compression ([`cache.py`](mirai/core/dataset/cache.py)).

### `fp8_text_encoder`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

FP8 text encoder for caching (native cache encoding via the model provider).

### `clips_per_video`

- **Type:** int
- **Default:** `1`
- **Allowed / range:** —

Clips sampled per source video ([`cache.py`](mirai/core/dataset/cache.py)).

### `tag_shuffle_variants`

- **Type:** int
- **Default:** `1`
- **Allowed / range:** —

Precomputed tag-shuffle variants ([`tags.py`](mirai/core/training/data/tags.py)).

### `online_tag_shuffle`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Online tag shuffle ([`online.py`](mirai/core/training/data/online.py), [`tags.py`](mirai/core/training/data/tags.py)).

### `online_tag_shuffle_dropout`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** in [0, 1]

Online tag dropout ([`online.py`](mirai/core/training/data/online.py)).

### `online_tag_shuffle_keep_first_n_tags`

- **Type:** int
- **Default:** `1`
- **Allowed / range:** >=0

Keep first N tags when shuffling ([`online.py`](mirai/core/training/data/online.py)).

### `online_temporal_resampling`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Online temporal resampling ([`batches.py`](mirai/core/training/data/batches.py)).

### `partial_recovery`

- **Type:** bool
- **Default:** `True`
- **Allowed / range:** —

Recover a partial cache ([`cache.py`](mirai/core/dataset/cache.py), [`migrations.py`](mirai/core/persistence/migrations.py)).

### `mask_extension`

- **Type:** str
- **Default:** `".mask.pt"`
- **Allowed / range:** —

Mask-file suffix ([`cache.py`](mirai/core/dataset/cache.py), preprocessing).

### `caption_format`

- **Type:** str
- **Default:** `"raw"`
- **Allowed / range:** `raw`, `lingbot_json` ([`schema.py`](mirai/config/schema.py))

Training caption resolution at cache-encode time. `raw` = byte-identical passthrough; `lingbot_json` applies the same resolution the inference prompt rewriter applies — structured JSON captions are unwrapped and stripped of runtime-only keys, plain captions are wrapped into the minimal `comprehensive_description` body — so LingBot-Video training conditioning matches the inference prompt contract, including the caption-schema check, which stops the cache build on a caption whose declared fields carry the wrong type ([`cache.py`](mirai/core/dataset/cache.py), [`prompt_rewriter.py`](mirai/core/inference/prompt_rewriter.py), provider-owned prompt-rewriter hook).

### `enable_bucketing`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Aspect-ratio bucketing ([`bucket_resolve.py`](mirai/core/dataset/bucketing/bucket_resolve.py)).

### `bucket_resolutions`

- **Type:** list[str]
- **Default:** `[]`
- **Allowed / range:** —

Explicit bucket resolutions ([`bucket_resolve.py`](mirai/core/dataset/bucketing/bucket_resolve.py)).

### `bucket_base_resolution`

- **Type:** int
- **Default:** `512`
- **Allowed / range:** —

Base bucket resolution ([`bucket_resolve.py`](mirai/core/dataset/bucketing/bucket_resolve.py)).

### `frame_buckets`

- **Type:** list[int]
- **Default:** `[33]`
- **Allowed / range:** —

Frame-count buckets ([`bucket_resolve.py`](mirai/core/dataset/bucketing/bucket_resolve.py), [`batches.py`](mirai/core/training/data/batches.py)).

### `bucket_resize_mode`

- **Type:** str
- **Default:** `"resize_crop"`
- **Allowed / range:** —

Bucket resize policy ([`media_resize.py`](mirai/core/dataset/media/media_resize.py)).

### `bucket_round_to`

- **Type:** int
- **Default:** `16`
- **Allowed / range:** —

Round bucket dims to a multiple ([`bucket_resolve.py`](mirai/core/dataset/bucketing/bucket_resolve.py)).

## `[dataset.moe_routing]` — MoEDatasetRoutingConfig

The block is an executable opt-in training policy. `emergent` creates no active
policy object and preserves the established sampler, batch, routing, gradients,
and artifacts. `domain_balanced` is model-agnostic; affinity modes require a
provider that declares dataset-routing support. LingBot-Video is the first such
provider. Affinity affects training forwards only; validation/inference use the
model's ordinary routing without external domain context.

### `specialization_mode`

- **Type:** str
- **Default:** `"emergent"`
- **Allowed / range:** `emergent`, `domain_balanced`, `soft_affinity`, `hard_affinity`

`domain_balanced` selects domains uniformly and samples within each domain; soft affinity adds a domain prior to router logits; hard affinity masks routing to the configured expert set.

### `domain_metadata_key`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** required when not emergent

Metadata may be a direct cache-record field or a key in the record's `metadata` table. Raw cache builds preserve this key from `registration.json`. Missing values fail before training.

### `expert_affinity`

- **Type:** table{str:list[int]}
- **Default:** `{}`
- **Allowed / range:** required for affinity modes; rejected in emergent/domain-balanced

Every training domain must have an entry. Expert ids must exist in the model; hard-affinity sets must contain at least the model's `top_k` experts.

### `routing_prior_weight`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** >=0; >0 for soft affinity

Additive soft-affinity logit prior at full strength.

### `router_warmup_steps`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0

Soft affinity scales linearly from zero to full strength. Hard affinity keeps emergent routing until this step, then activates the mask.

`domain_balanced` and `dataset.online_temporal_resampling` are mutually exclusive
because both own record selection. Affinity modes and stochastic expert-subset
routing are mutually exclusive because intersecting their masks can leave fewer
than `top_k` experts.

## `[logging]` — LoggingConfig

### `output_dir`

- **Type:** str
- **Default:** `"./outputs"`
- **Allowed / range:** no control chars

Run output dir ([`artifacts.py`](mirai/core/training/lifecycle/artifacts.py)).

### `save_every_n_steps`

- **Type:** int
- **Default:** `50`
- **Allowed / range:** >=1

Checkpoint interval ([`callbacks.py`](mirai/core/training/observability/callbacks.py), [`artifacts.py`](mirai/core/training/lifecycle/artifacts.py)).

### `async_checkpoint`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Serialize periodic checkpoints on one background worker from an immutable CPU snapshot; the next save and shutdown surface failures.

### `async_checkpoint_max_gib`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** > 0 when async checkpointing is enabled

Maximum tensor bytes admitted to one asynchronous checkpoint snapshot.

### `tensorboard`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** requires `torch.utils.tensorboard`

TensorBoard logging ([`callbacks.py`](mirai/core/training/observability/callbacks.py)).

### `sample_every_n_steps`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0; >0 needs `av` or `PIL`

Preview interval ([`preview_samples.py`](mirai/core/training/preview/preview_samples.py)).

### `sample_prompt`

- **Type:** str
- **Default:** `"a cat walking"`
- **Allowed / range:** —

Single preview prompt ([`preview.py`](mirai/core/training/preview/preview.py)).

### `sample_seed`

- **Type:** int
- **Default:** `42`
- **Allowed / range:** —

Preview seed ([`preview_samples.py`](mirai/core/training/preview/preview_samples.py)).

### `sample_prompts`

- **Type:** list[str]
- **Default:** `[]`
- **Allowed / range:** must be an array

Multi-prompt previews ([`preview_samples.py`](mirai/core/training/preview/preview_samples.py)).

### `sample_seeds`

- **Type:** list[int]
- **Default:** `[]`
- **Allowed / range:** must be an array

Per-prompt preview seeds ([`preview_samples.py`](mirai/core/training/preview/preview_samples.py)).

### `sample_blocks_to_swap`

- **Type:** int
- **Default:** `-1`
- **Allowed / range:** >=-1; -1 = inherit

Block-swap during sampling ([`preview_runtime.py`](mirai/core/training/preview/preview_runtime.py)).

### `sample_cfg_scale`

- **Type:** float
- **Default:** `7.5`
- **Allowed / range:** >=0

CFG scale for previews ([`preview.py`](mirai/core/training/preview/preview.py)).

### `sample_negative_prompt`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** —

Negative prompt for previews ([`preview.py`](mirai/core/training/preview/preview.py)).

### `sample_solver`

- **Type:** str
- **Default:** `"euler"`
- **Allowed / range:** registry names: `euler`, `unipc`, `dpmpp_2m`; default `euler`

Preview flow-matching solver, registry-selected ([`preview_solvers.py`](mirai/core/training/preview/preview_solvers.py)). `unipc` is Wan-style Flow-UniPC multistep (bh2, order 2). `dpmpp_2m` implements the DPM-Solver++ second-order midpoint update using flow data prediction. Both share the Euler sigma grid and remain explicit opt-ins.

### `sample_resolution`

- **Type:** str
- **Default:** `"smallest_bucket"`
- **Allowed / range:** —

Preview resolution policy ([`preview.py`](mirai/core/training/preview/preview.py)).

### `sample_frame_count`

- **Type:** int
- **Default:** `16`
- **Allowed / range:** >=1

Preview frame count ([`preview.py`](mirai/core/training/preview/preview.py)).

### `wandb`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** requires `wandb`

Weights & Biases logging ([`callbacks.py`](mirai/core/training/observability/callbacks.py)).

### `wandb_project`

- **Type:** str
- **Default:** `"mirai"`
- **Allowed / range:** —

W&B project ([`callbacks.py`](mirai/core/training/observability/callbacks.py)).

### `wandb_run_name`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** —

W&B run name ([`callbacks.py`](mirai/core/training/observability/callbacks.py)).

## `[compliance]` — ComplianceConfig

### `enabled`

- **Type:** bool
- **Default:** `False`

Compliance gate for dataset registration ([`registration.py`](mirai/core/dataset/registration.py)).

### `require_provenance`

- **Type:** bool
- **Default:** `True`

Require provenance metadata ([`registration.py`](mirai/core/dataset/registration.py)).

### `require_rights_attestation`

- **Type:** bool
- **Default:** `True`

Require rights attestation ([`registration.py`](mirai/core/dataset/registration.py)).

## `[memory]` — MemoryConfig

MoE / quantization memory features. Each is capability-gated per family in
[`runtime/trainer.py`](mirai/core/training/runtime/trainer.py) and rejected (not silently ignored) for families that do not
implement it.

### `hardware_policy`

- **Type:** str
- **Default:** `"disabled"`
- **Allowed / range:** `disabled`, `tiered`

Opt-in resolution from `config/defaults/hardware_tiers.toml`. `tiered` matches compute capability and total VRAM, then fills only unset expert chunk, shared device-residency budget, and compatible INT8 expert-cache budget. Explicit positive values win; unknown or unavailable CUDA hardware fails explicitly. The shipped table covers compute capability 8.0-10.9; a device outside that range matches no tier and `tiered` fails closed, so a named hardware profile states its memory keys explicitly.

### `frozen_weight_quantization`

- **Type:** str
- **Default:** `"none"`
- **Allowed / range:** `none`, `fp8`, `int8`, `nf4`, `gguf_iq4`, `gguf_iq3`, `gguf_iq2`, `mxfp8_e4m3`, `mxfp4`, `nvfp4`

Frozen-weight quant scheme ([`quantization.py`](mirai/core/models/quantization.py), compressed_weights). `fp8` is DeepSeek-style E4M3 W8A8 reference execution: 128×128 FP32 weight scales, online per-token-per-128-channel activation scales, FP32 K=128 accumulation, and high-precision input gradients; packed dense and routed-expert artifacts use separate `*_fp8`/`*_fp8_scale` roles. `nf4` requires bitsandbytes; MAGI-2 Preview implements `nf4` only, and packs exactly the three routed expert tensors of each multi-head MoE layer ([`quantized_experts.py`](mirai/core/models/magi2_preview/quantized_experts.py)). `gguf_iq4`/`gguf_iq3`/`gguf_iq2` provide GGUF sub-4-bit expert storage; IQ2_XS uses canonical 74-byte blocks (2.3125 bits/weight), uniform code assignment, and imatrix-weighted calibration for per-projection format selection. It does not claim to reproduce llama.cpp's imatrix encoder. `mxfp8_e4m3` is the distinct OCP microscaling format with 32-value E4M3 blocks and round-up UE8M0 scales (`ceil(log2(amax / 448))`). `mxfp4` implements OCP E2M1 with 32-value E8M0-scaled blocks; `nvfp4` implements E2M1 with 16-value E4M3 block scales and a tensor FP32 scale. The portable paths are default-off and make no native-kernel speed claim.

### `refiner_frozen_weight_quantization`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** empty, `none`, `fp8`, `int8`, `nf4`, `gguf_iq4`, `gguf_iq3`, `gguf_iq2`, `mxfp8_e4m3`, `mxfp4`, `nvfp4`

Frozen-weight format for a separately loaded refiner DiT. Empty inherits
`memory.frozen_weight_quantization`; an explicit value lets the base and refiner
use different storage formats. The provider must support compressed on-load
construction for its refiner. Packed integer codes and FP32 scale metadata retain
their storage dtypes when the model compute dtype is applied
([`pipeline.py`](mirai/core/models/lingbot_video/pipeline.py),
[`refiner.py`](mirai/core/models/lingbot_video/refiner.py)).

### `frozen_weight_quantization_strategy`

- **Type:** str
- **Default:** `"disabled"`
- **Allowed / range:** `disabled`, `auto`, `compressed_weights`

Quantization implementation strategy. `compressed_weights` supports every format listed above, including block-scaled FP8.

### `frozen_weight_packed_state_path`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** no control chars

Prebuilt packed compressed-weight state path; requires the matching `frozen_weight_quantization` value and the `packed_frozen_weight_state` capability. Generic compressed experts use `mirai.compressed_weights.packed_state`; providers whose native expert layout cannot implement that tensor contract may declare their own versioned, lineage-checked artifact through the same capability. MAGI-2 Preview uses a manifest plus one NF4 safetensors shard per routed-expert layer so each input mapping can be released before the next layer is restored.

### `expert_precision_plan_path`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** schema-v1 or schema-v2 JSON; requires frozen expert quantization

Optional mixed-precision expert plan. Schema v1 preserves one format per physical expert across w1/w2/w3. Schema v2 assigns each `(module, expert, projection)` independently from measured imatrix-weighted reconstruction error and exact packed byte cost; it verifies complete topology and the exact floating source-weight fingerprint before quantization. Heterogeneous projections retain native packed buffers, BF16 candidates use frozen buffers, and backward re-materializes frozen weights instead of retaining a dense copy in autograd ([`precision.py`](mirai/core/models/compressed_weights/execution/mixed_precision.py), [`mixed_precision.py`](mirai/core/models/compressed_weights/execution/mixed_precision.py)).

### `expert_structured_sparsity`

- **Type:** str
- **Default:** `"disabled"`
- **Allowed / range:** `disabled`, `2:4`; no frozen quantization

Magnitude-prune frozen expert projections to exact NVIDIA 2:4 structure and dispatch through semi-structured CUDA linear when supported, with a dense mathematical fallback.

### `quantization_block_size`

- **Type:** int
- **Default:** `64`
- **Allowed / range:** —

Quant block size (nf4) (compressed_weights, [`specs.py`](mirai/core/moe/runtime/specs.py)).

### `weight_residency_strategy`

- **Type:** str
- **Default:** `"disabled"`
- **Allowed / range:** `disabled`, `block_swap`, `stream_disk`

Frozen transformer-block residency policy ([`device_placement.py`](mirai/core/training/residency/device_placement.py), [`block_swap.py`](mirai/core/training/residency/block_swap.py)). `block_swap` retains immutable staging tensors in RAM. `stream_disk` atomically writes each non-resident block to a safetensors-backed mapping under the run output directory, defaults to streaming every block when `training.blocks_to_swap=0`, and uses the same bounded forward/backward residency schedule. It requires `block_swap_transfer_strategy="per_tensor"`; trainable adapter tensors are never written into the immutable backing files.

### `expert_weight_access`

- **Type:** str
- **Default:** `"auto"`
- **Allowed / range:** `auto`, `disabled`, `full_dequant`, `active_dequant`, `chunked_dequant`, `fused_kernel` ([`specs.py`](mirai/core/moe/runtime/specs.py))

Expert weight access pattern. With `fused_kernel`, activation rotation is shared by gate/up and the stored INT8 tensors are converted to FP32 GEMM operands at the operation boundary; SwiGLU and activation-space LoRA remain in the same chunk operation. This is compact INT8 storage with reference execution, not a packed-INT8 throughput claim. It requires INT8, chunk size > 1, and kernel backend `auto` or `rotated_int8` ([`experts.py`](mirai/core/models/compressed_weights/execution/experts.py), [`rotated_int8.py`](mirai/core/models/compressed_weights/quantization/rotated_int8.py)). MAGI-2 Preview implements `full_dequant` and `chunked_dequant` over its packed group axis and rejects `active_dequant` and `fused_kernel`, which address a per-routed-expert operand its flattened head-major layout does not expose.

### `expert_dequant_chunk_size`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** —

Experts dequantized per chunk (for MAGI-2 Preview, groups of the flattened `head * num_experts + expert` axis); `0` resolves from device capabilities when `expert_weight_access="chunked_dequant"` (`compressed_weights`, [`expert_specs.py`](mirai/core/moe/runtime/specs.py)). In explicit disk mode, chunking converts singleton reads into bounded multi-expert range requests. Larger chunks reduce read operations but increase H2D traffic and transient dequantization storage; select the value from deployment measurements.

### `expert_device_cache_gib`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** >=0

Global byte-bounded LRU of immutable INT8 expert operands on the compute device. Zero creates no resident entries and preserves demand transfer. Positive values are shared across every compressed MoE layer, cache only routed physical experts and their scales, evict before crossing the exact ceiling, and never retain BF16 weights. Other quantization formats fail explicitly. MAGI-2 Preview rejects any positive value: its packed experts are layer-resident state moved by the block-residency subsystem, so device residency for them is chosen with `training.blocks_to_swap`.

### `device_residency_budget_gib`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** >=0

Optional shared VRAM-residency ceiling across the routed-expert device cache, always-resident transformer blocks, and the block-transfer window. Zero disables enforcement without changing placement. Positive values fail before training when independent reservations exceed the common byte budget.

### `quantize_experts_on_load`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** requires a supported frozen-weight quantization format

Quantize routed experts while checkpoint tensors are loaded, avoiding a persistent dense copy (compressed_weights, runtime provider). For MAGI-2 Preview this removes the dense expert parameters before any shard is opened and packs each routed tensor as it is read, so the released BF16 expert stack never exists in host memory as one copy; without it the same packing runs only after the full checkpoint has loaded.

### `router_quantization`

- **Type:** str
- **Default:** `"disabled"`
- **Allowed / range:** `disabled`, `int8_per_channel`

Frozen-router storage policy. `int8_per_channel` replaces each router matrix with symmetric per-output-channel INT8 values and FP32 scales after adapter configuration. Router-targeted adapters, direct router training, and FP32 router masters fail explicitly because quantized router weights are immutable.

### `router_quantization_calibration_path`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** no control characters; requires `router_quantization="int8_per_channel"`

Optional EAQuant calibration artifact consumed while frozen router INT8 storage is created. Loading verifies the complete router inventory, dimensions, top-k topology, and exact fingerprint of the still-floating source weights before installing calibrated scales. Empty preserves the established absmax per-output-channel quantizer exactly.

### `moe_kernel_backend`

- **Type:** str
- **Default:** `"auto"`
- **Allowed / range:** `auto`, `torch`, `rotated_int8`, `compiled_packed`, `megablocks`, `grouped` ([`kernels.py`](mirai/core/moe/runtime/kernels.py))

MoE execution backend. `rotated_int8` requires INT8 frozen weights with chunked access and moves the stored-weight rotation onto activations before batched multiplication. `compiled_packed` uses shape-specialized TorchInductor decode kernels for GGUF IQ4/IQ3 or microscaling storage (`mxfp8_e4m3`, MXFP4, NVFP4) before grouped GEMM and fails if compilation is unavailable. `megablocks` requires `megablocks.ops` and `grouped_gemm`. `grouped` selects a model-family-owned grouped-GEMM expert seam and has no generic implementation; MAGI-2 Preview implements it for training with frozen experts ([`grouped_moe.py`](mirai/core/models/magi2_preview/grouped_moe.py)) and other families reject it. With MAGI-2 NF4 experts, `auto` resolves to that seam and `torch` is rejected, because the vendored per-expert reference loop reads the dense expert tensors packed storage replaces.

### `cuda_memory_fraction`

- **Type:** float
- **Default:** `1.0`
- **Allowed / range:** in (0, 1]

CUDA memory fraction cap ([`memory_safety.py`](mirai/core/training/residency/memory_safety.py)).

### `minimum_system_memory_gib`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** >=0

Floor on **available (free)** host RAM, not a statement of total RAM required. Residency and packed-state paths abort when the free reading falls below this, and the block-swap pinned budget is `min(free_ram - this, max_pinned_host_gib)`. A value near a machine's total RAM aborts on contact. Requires `psutil` when non-zero ([`memory_safety.py`](mirai/core/training/residency/memory_safety.py)).

### `max_pinned_host_gib`

- **Type:** float
- **Default:** `12.0`
- **Allowed / range:** >0

Per-subsystem ceiling in GiB for page-locked host memory. Block-swap residency uses `min(free_ram - minimum_system_memory_gib, max_pinned_host_gib)` and packed-state preload applies the same ceiling independently. Over-budget tensors use pageable storage. This changes transfer behavior, not tensor values. If free RAM cannot be queried, the configured ceiling remains the absolute limit.

### `trainable_parameter_offload`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** requires `paged_adamw_8bit` or `optimizer_cpu_offload=true`

Offload trainable params ([`runtime/trainer.py`](mirai/core/training/runtime/trainer.py)).

### `packed_state_preload`

- **Type:** str
- **Default:** `"pinned"`
- **Allowed / range:** `pinned`, `ram`, `off` ([`specs.py`](mirai/core/moe/runtime/specs.py))

Packed-state residency mode. RAM-primary: `pinned` (page-locked, async H2D) degrades only to `ram` (pageable). `off` is explicit disk streaming for packed int8 and NF4 grouped experts: safetensors layout metadata, tiny NF4 codebooks, and read-only shard handles stay resident while requested expert ranges are read through a bounded worker pool into host staging, then batched onto a CUDA side stream. Linux suppresses access-time metadata writes with `O_NOATIME` when permitted and requests random-access read-ahead behavior; both degrade safely when unsupported. Page-locked staging is bounded by `max_pinned_host_gib`; allocation failure degrades that request to pageable staging. The reader exposes read/H2D byte, operation, latency, peak-pin, requested/unique-slice, and reordered-batch telemetry. Use `chunked_dequant` with a deployment-measured chunk size to reduce storage IOPS; `active_dequant` minimizes H2D bytes at the cost of singleton reads. Disk mode never becomes an automatic fallback.

### `packed_stream_cache_gib`

- **Type:** float
- **Default:** `0.0`
- **Allowed / range:** `>= 0`; positive requires `packed_state_preload="off"`

Byte-bounded host LRU for expert-slice and canonical active-expert-set requests in explicit disk mode ([`packed_stream_cache.py`](mirai/core/models/compressed_weights/packed/packed_stream_cache.py)). `0` preserves direct streaming with no cache allocation or cache lookup telemetry. Positive values retain immutable host ranges across repeated forward/backward requests, evict least-recent entries before crossing the ceiling, reject individually oversized requests, and expose hit/miss/byte/eviction/residency telemetry. Slice and batch namespaces are distinct; batch permutations share one sorted-set entry and caller order is restored after transfer. The cache does not change tensor values or H2D ordering and remains subject to `minimum_system_memory_gib` runtime safety checks.

### `packed_stream_backend`

- **Type:** str
- **Default:** `"staged"`
- **Allowed / range:** `staged`, `gds`; `gds` requires `packed_state_preload="off"` and `packed_stream_cache_gib=0`

Storage transport for explicit packed disk mode. `staged` uses the bounded POSIX→host→CUDA path. `gds` uses optional KvikIO/cuFile and fails closed unless compatibility is disabled, cuFile reports true GDS, every shard mount supports direct I/O, and every shard has Mirai's validated 4 KiB layout. The export script pads safetensors header whitespace for `gds`; artifacts without the aligned layout require re-export. Reads expand to aligned file/device ranges and trim to the exact tensor payload. Telemetry separates logical `gds_requested_bytes` from physical `gds_read_bytes` and reports overread/trim counts. Runtime is read-only; alignment adds at most one sequential export rewrite per shard and no SSD cache.

### `packed_stream_prefetch_depth`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `0..16`

Bounded asynchronous request slots for canonical packed expert weights. `0` preserves the established disk or RAM demand path without allocating a ring or emitting prefetch telemetry. A positive depth submits the exact `w1`/`w3`/`w2` fields for the current routed expert chunk, keeps at most `depth` transfers active, queues at most `4 × depth` identities without allocating their payloads, deduplicates identical requests, and consumes them through the established device-transfer result. Packed export and import reject provider layouts that do not use this canonical SwiGLU tensor schema. Disk mode composes with staged or GDS transport. Pageable `ram` preload gathers fields into `depth` reusable contiguous host slots and issues one side-stream H2D per field; slots are reused only after their copy event completes and page locking stays inside the residual `max_pinned_host_gib` budget. `pinned` preload directly coalesces contiguous expert runs; fully fragmented requests bypass prefetch and preserve the per-expert transfer path. Preloaded H2D is enqueued inline to avoid worker-thread overhead. This is exact schedule-aware prefetch, not predicted routing, and creates no artifact writes or SSD cache.

### `moe_dispatch`

- **Type:** str
- **Default:** `"vectorized"`
- **Allowed / range:** `vectorized`, `legacy`, `triton`, `triton_persistent`

MoE dispatch kernel. `triton` (padded count-aware grouped GEMM) and `triton_persistent` (sorted-contiguous persistent grouped GEMM, no padding) both fall back to vectorized if Triton is unavailable or the device is below SM80.

### `moe_dispatch_preprocess`

- **Type:** str
- **Default:** `"host"`
- **Allowed / range:** `host`, `device`, `sonic`

Owner of count/offset/stable-order preprocessing. `host` preserves per-chunk host assembly. `device` builds the plan with PyTorch device operations. `sonic` uses SonicMoE's fused general-routing metadata kernel and requires the optional `sonic-moe` package, Python 3.12+, PyTorch 2.7.1–2.9.1, and SM90, SM100, or SM120. Both device modes preserve the same `DispatchPlan`; explicit disk streaming should use `host` because device preprocessing includes idle experts.

### `moe_autotune_warmup_rows`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** `>= 0`; positive requires `moe_dispatch="triton_persistent"` or `moe_gemm_backend="persistent"`

Representative routed-row count used to populate the persistent grouped-GEMM Triton cache during trainer startup, before training activations occupy device memory. Shapes come from the model provider's typed expert tensor inventory; duplicate projection shapes tune once. `0` preserves lazy first-use autotuning and performs no allocation. Positive values allocate one synthetic expert projection at a time, synchronize after each key, and release its temporary storage before continuing. Triton's device-local cache remains the owner of persisted tuning results ([`autotune_warmup.py`](mirai/core/moe/runtime/autotune_warmup.py), [upstream PR](https://github.com/woct0rdho/transformers-qwen3-moe-fused/pull/21)).

### `moe_gemm_backend`

- **Type:** str
- **Default:** `"auto"`
- **Allowed / range:** `auto`, `bmm`, `persistent`, `torch_grouped`

Grouped-GEMM primitive below `moe_dispatch`. `auto` inherits dispatch selection; explicit unavailable choices fail. `deepgemm_fp8` is forward-only and therefore excluded from this shared key.

### `moe_gemm_backend_forward`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** `""` (inherit), `auto`, `bmm`, `persistent`, `torch_grouped`, `deepgemm_fp8`

Forward matmul override. `deepgemm_fp8` uses stored 128×128 FP8 weights and online 1×128 activation scales; it requires `frozen_weight_quantization="fp8"`, DeepGEMM, CUDA SM90, and expert widths divisible by 128. Input gradients remain high precision.

### `moe_gemm_backend_dx`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** `""` (inherit), `auto`, `bmm`, `persistent`, `torch_grouped`

Input-gradient backend selection; `deepgemm_fp8` is rejected.

### `moe_gemm_backend_dw`

- **Type:** str
- **Default:** `""`
- **Allowed / range:** `""` (inherit), `auto`, `bmm`, `persistent`, `torch_grouped`

Adapter-gradient backend selection; `deepgemm_fp8` is rejected.

### `moe_expert_autograd`

- **Type:** str
- **Default:** `"standard"`
- **Allowed / range:** `standard`, `segmented_recompute`

Autograd schedule for frozen routed-expert MLPs. `standard` preserves the
projection-by-projection graph. `segmented_recompute` keeps only the sorted
expert input and rematerializes each packed expert segment during backward,
bounding gate/up activation residency at the cost of extra projection work.
Support is provider-owned; MAGI-2 implements it for NF4-packed grouped experts
and rejects it for dense experts.

### `moe_activation_backend`

- **Type:** str
- **Default:** `"torch"`
- **Allowed / range:** `torch`, `triton`

Provider-owned routed-expert activation backend. `torch` is the portable
reference path. `triton` requires CUDA and an installed Triton runtime; a
provider must implement its exact activation forward and input gradients or
reject the value. MAGI-2 implements fused FP32 SwiGLU7 clamping, sigmoid, and
backward kernels without retaining extra FP32 activation state.

### `moe_batched_dequant`

- **Type:** bool
- **Default:** `True`
- **Allowed / range:** —

Enables chunk-batched expert dequantization.

### `moe_pair_dequant`

- **Type:** bool
- **Default:** `True`
- **Allowed / range:** —

Enables paired w1/w3 expert dequantization.

### `moe_batched_gather`

- **Type:** bool
- **Default:** `False`
- **Allowed / range:** —

Enables contiguous batched expert gather.

### `packed_shard_size_mb`

- **Type:** int
- **Default:** `2048`
- **Allowed / range:** >0

Packed-state export shard size in MiB.

### `int8_workspace_mb`

- **Type:** int
- **Default:** `0`
- **Allowed / range:** >=0; 0 = internal default

Compressed-weight quantization workspace in MiB.

### `block_residency_planner`

- **Type:** str
- **Default:** `"uniform"`
- **Allowed / range:** `uniform`, `phase_aware`

Block-swap residency plan ([`block_swap.py`](mirai/core/training/residency/block_swap.py), [`residency_plan.py`](mirai/core/training/residency/residency_plan.py)). `uniform` keeps the first N blocks resident and swaps the remainder uniformly. `phase_aware` retains tail blocks across the forward-to-backward transition and head blocks across the backward-to-forward transition to avoid immediate eviction and refetch. Residency does not change model math.

### `block_swap_prefetch_depth`

- **Type:** int
- **Default:** `1`
- **Allowed / range:** in [1, 4]

Asynchronous prefetch ring depth ([`block_swap.py`](mirai/core/training/residency/block_swap.py)). `1` prefetches the next block. `N` permits N swap blocks of transfer lookahead with per-slot events and reserves storage for N resident swap blocks. Applies only to `async` mode and does not change model math.

### `block_residency_priority`

- **Type:** str
- **Default:** `"index"`
- **Allowed / range:** `index`, `routing_hot`

Selects the resident block set ([`block_swap.py`](mirai/core/training/residency/block_swap.py), [`residency_plan.py`](mirai/core/training/residency/residency_plan.py)). `index` keeps the first N block indices. `routing_hot` keeps the N blocks with the highest routing concentration and uses `index` until routing statistics exist. Residency does not change model math.

### `block_swap_transfer_strategy`

- **Type:** str
- **Default:** `"per_tensor"`
- **Allowed / range:** `per_tensor`, `flat_ring`

Frozen-block H2D transfer owner. `per_tensor` preserves the established allocation path. `flat_ring` packs each immutable CPU block into one contiguous master and streams it through `block_swap_prefetch_depth + 1` reusable CUDA slots, reducing copy launches and allocator churn. CUDA-only, opt-in, and compatible only with frozen-base residency.

---

## Environment variables (appendix)

Environment variables are limited to operational integration and credentials;
model, training, inference, memory, and kernel behavior belongs in config.

| var | effect |
|---|---|
| `MIRAI_GPU_LEASE_PATH` | GPU-lease lockfile path; explicit override, absolute precedence (default derives from scope, see below) |
| `MIRAI_GPU_LEASE_SCOPE` | lease default-path scope: `global` (one `ROOT/.mirai_gpu_lease.lock`) or `per-gpu` (one file per `CUDA_VISIBLE_DEVICES` value, such as `.mirai_gpu_lease.gpu2.lock`; unpinned runs use the global lock) |
| `MIRAI_GPU_LEASE_TIMEOUT` | lease-acquire timeout in seconds (default `0`) |
| `MIRAI_GPU_LEASE_QUEUE_DIR` | FIFO queue dir for lease fairness |
| `MIRAI_JOB_ID` | job id for live-control DB / event stream |
| `MIRAI_API_DB_PATH` | SQLite path for live-control commands |
| `MIRAI_JOB_EVENTS_PATH` | JSONL event-stream output path |

| `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN` | Optional Hugging Face download credential. |
