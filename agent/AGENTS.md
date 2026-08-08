# Mirai engineering contract

Mirai trains adapters for native sparse-MoE video diffusion models. The core is
model-agnostic; model families integrate through `ModelFamilyProvider`.

## Supported surface

`README.md` states the whole platform contract: a single-GPU adapter trainer and
inference runtime for dynamically routed sparse-MoE video diffusion models,
requiring Python 3.10+ and CUDA SM80+. Training is adapter-scale, not
pretraining-scale.

## Standing clarifications

The repository does not make the assumptions below. Reason from the code and
from `CONFIG_REFERENCE.md` instead of supplying them.

- There is no required target-GPU tier or device spectrum. SM80+ is the stated
  floor. The optional `memory.hardware_policy="tiered"` profiles choose memory
  defaults from device capability and capacity; they do not narrow the supported
  surface. Hardware required by a technique remains a property of that technique.
- There is no universal host-device memory model. Weight residency, block
  swapping, and expert streaming are optional and default-off;
  `weight_residency_strategy` defaults to `disabled`.
  `configs/lingbot_video/train_bf16.toml` runs fully resident and
  `configs/lingbot_video/train_nf4.toml` runs block-swapped. Evaluate behavior
  against the resident path first, and mark anything that only applies under
  swapping as mode-specific rather than general.
- Base weights are not always frozen. Adapter training is the main path, but
  `adapter.type="selected_expert"` updates expert rows directly through the
  selected-expert optimizers in `mirai/core/training/optim/`.
- A config key can carry semantics that its call site does not restate.
  `CONFIG_REFERENCE.md` is normative: read the key's row before concluding that
  an observed behavior is a defect.

## Required invariants

1. Core orchestration never branches on a concrete family name or private module
   layout. Family behavior enters through typed provider capabilities and hooks.
2. Runtime is native-only. Vendored source is plain `torch.nn.Module`; Diffusers
   is not a load or forward dependency.
3. Config is the behavior control surface. New optional behavior is explicit,
   default-off, isolated when disabled, and documented in `CONFIG_REFERENCE.md`.
4. Unsupported combinations and artifact-lineage mismatches fail explicitly.
5. Optimized math retains a reference path and verifies outputs, loss, input
   gradients, and trainable adapter gradients as applicable.
6. Frozen BF16 weights do not survive in autograd. Normal expert offload is
   RAM-to-VRAM; disk streaming is explicit and never an automatic fallback.
7. One feature seam has one searchable owner module and one object-level
   contract. Stable orchestration depends on contracts, not implementations.
8. Comments explain constraints, tensor semantics, or provenance only. They do
   not contain development history, agent names, roadmap notes, or benchmarks.
9. README lists only implemented behavior. Every public config key is described
   in `CONFIG_REFERENCE.md`.
10. Multi-GPU functionality is outside the current supported release surface.

## Change routing

| Change | Primary owner |
|---|---|
| config | `mirai/config/schema.py`, then `CONFIG_REFERENCE.md` |
| model family | `mirai/core/models/providers.py` and family-owned module |
| MoE routing/kernel/policy | `mirai/core/moe/` |
| frozen compression/residency | `mirai/core/models/compressed_weights/` |
| training/session/runtime | `mirai/core/training/` |
| dataset/cache/lineage | `mirai/core/dataset/` |
| persistence/migrations | `mirai/core/persistence/` |
| inference policy | `mirai/core/inference/` or `inference/<family>/` |
| CLI/reporting | `scripts/` |

Search [`architecture.json`](architecture.json) before adding a new owner. Do
not duplicate an existing seam.

## Implementation rules

- Prefer typed dataclasses, enums, registries, protocols, and boundary
  validators over loose mappings, probing, or import-order magic.
- Add optional heavy imports only at their execution seam.
- Preserve unrelated worktree changes.
- Keep Python 3.10 compatibility.
- Source-derived code or algorithms retain an adjacent public attribution.
- Never claim performance without reproducible latency and peak-memory evidence.

## Validation

Validation is an architectural subsystem, not a test archive. Each seam owns one
named contract in [`checks.json`](checks.json); its commands are executable
behavioral probes for the invariants declared beside them. Use
`python scripts/agent/check.py --changed <paths>` to select the smallest sufficient
set. `--run` executes the fast gate; use `--max-cost extended` for broader
validation. GPU evidence additionally requires `--max-cost gpu
--allow-remote-gpu` in the configured remote environment. Deferred contracts
produce `incomplete`, never `passed`. An unowned path is an architecture error
and fails closed. `--all-local --run --max-cost extended` executes the complete
manifest-driven CPU gate without relying on pytest filename discovery.

Dependency boundaries in [`architecture.json`](architecture.json) are executable:
`scripts/agent/architecture.py` parses shipping imports and fails when generic
core reaches into a concrete family or the native runtime imports Diffusers.

Changes to the validation system are promoted only through the paired protocol
in [`effectiveness.json`](effectiveness.json). The deployed agent, context,
tools, and harness form the measured unit. Hidden behavioral contracts and
architecture correctness may not regress; efficiency claims require recorded
time or token improvement. `scripts/agent/run_agent_evaluation.py` executes randomized
trials from pinned baseline/candidate snapshots through shell-free adapters;
hidden graders remain external to the repository. Executable GPU validation runs
only in the configured remote GPU environment after confirming its exclusive GPU
lease is free.

## Extension protocol

### Feature admission gate

Evaluate a proposed technique before implementation. A paper's use of MoE, a
working reference implementation, or mathematical novelty is not sufficient
reason to add it. Record one of four decisions:

- `reject`: no concrete Mirai workflow, duplicates an existing mechanism, or
  its expected value does not justify its runtime and maintenance cost;
- `experimental`: a concrete workflow and technically sound integration exist,
  but Mirai-specific quality or efficiency evidence is still missing;
- `admitted`: behavioral contracts, persistence, compatibility, and measured
  resource cost pass, and a representative Mirai A/B evaluation supports the
  intended use;
- `promoted`: the feature is eligible for recommended defaults or performance
  claims under the evidence requirements below.

An admission review must answer all of the following:

1. Which released Mirai training or inference workflow benefits, and what
   observable failure or limitation does the technique address?
2. Is the mechanism intrinsically generic, native-MoE-specific, or
   model-family-specific? Classify by required architecture, not by the paper's
   title, evaluation model, or current provider coverage. A mechanism remains
   generic when another provider could implement the same contract without
   native experts or routers.
3. Does an existing owner or feature already provide materially equivalent
   behavior?
4. What new configuration, runtime state, checkpoint lineage, incompatibility,
   and migration obligations does it create?
5. What are the measured latency, peak VRAM, peak host RAM, and persistent-state
   costs on the smallest representative probe?
6. Is the implementation and long-term validation burden proportional to the
   expected benefit?
7. Which reference-parity, output, loss, input-gradient, trainable-gradient,
   save/load, disabled-path, and provider-integration checks establish
   correctness?
8. Which representative Mirai baseline/candidate evaluation could falsify the
   claimed quality or efficiency benefit?

Only `experimental`, `admitted`, and `promoted` features may enter the public
tree. Experimental features must be explicit, default-off, described without
quality or efficiency claims, and have a named evaluation path. Correctness
tests alone never advance a feature beyond `experimental`. If no practical
evaluation can be named, reject the feature instead of implementing it.

`agent/features.json` is the machine-readable feature inventory and
`mirai/core/features.py` owns its typed contract. Before adding behavior, run
`python scripts/agent/feature.py inspect`; attach the implementation to an existing
extension kit with `feature.py create`, then require `feature.py validate` to
pass. A descriptor must resolve its owner, config gates and disabled values,
behavioral contract, validation seam, persistence/reference obligations, and
README claim. Do not create a new top-level validation contract for a feature
that fits an existing kit.

Features that change token-to-expert routing topology use
`kind="routing_topology"`. They must remain `default_off=true` until
`promotion_evidence` points to a repository-owned
`agent/evidence/<feature>.json` record containing paired held-out baseline and
candidate quality observations with split and artifact fingerprints. Routing
health telemetry alone is not promotion evidence
([arXiv:2604.14419](https://arxiv.org/abs/2604.14419)).

For a model family, vendor only runtime-used attributed native modules, implement
`NativeVideoPipeline`, register one `ModelFamilyProvider`, declare capabilities,
keep prompt/latent/VAE semantics family-owned, add `configs/<family>/`, and prove
one native train step. Core orchestration must not acquire a family-name branch.

For an optional feature, create one typed owner, add a default-preserving config
gate, validate incompatible combinations, integrate through a stable contract,
keep feature state/imports absent when disabled, and prove reference parity.

For a config key, update the typed schema, coercion, range/cross-key validation,
consumer, `CONFIG_REFERENCE.md`, and public examples. Unused keys are rejected.

Persistence formats are versioned contracts. Add explicit validation and
migration; never guess when lineage or schema does not match.
