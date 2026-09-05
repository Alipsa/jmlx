# jmlx Full MLX-Parity Roadmap

## Purpose and definition of done

This is the long-range planning index for jmlx after Phases 1–5. It extends and supersedes the
forward-looking "Project Execution Plan" in `req/project-outline.md`, whose planned phases stop at
Phase 5. It is deliberately a roadmap, not an implementation plan: each milestone below must get
its own plan in `req/plans/` before code is started. Those plans should preserve the decisions and
verification style established by `initial-plan.md` and Phases 3–5.

**Priority order:** finish usable local inference first (Phase 6), then complete the
array/runtime surface, transformations and training, and finally the specialist/system
features needed for MLX parity.

This ordering is priority, not a strict ban on dependencies: Phase 6 may implement the focused
array, indexing, sort/selection, and random-key slices it needs from later groups. The inventory
records those decisions, and later phases extend the established contract rather than re-deriving it.

"Full parity" means that, for the version of MLX pinned by `scripts/bootstrap-native.sh`, a
Java caller can use every supported public mlx-c capability through a safe, idiomatic Java API
with MLX-equivalent observable semantics.  Parity is not a literal one-class-per-C-function
translation: overloads, records and managed wrappers are preferred where they make Java use safer,
but they must not remove native capability.  The raw generated FFM bindings are an implementation
asset, not themselves parity.

The target consists of independently pinned `mlx-metal` and `mlx-c` revisions declared by
`scripts/bootstrap-native.sh`. A later MLX upgrade starts a new compatibility delta: regenerate
bindings, inventory newly added or changed symbols, and add them to the appropriate unfinished or
maintenance phase. Do not silently claim parity with a newer upstream release.

Once Phase 6.0b creates it, `req/mlx-api-inventory.md` is the single readable record of the current
two-pin target and coverage; the bootstrap script remains the source of truth for the pin values.

## Current baseline (Phases 1–5)

Delivered: native loading and generated bindings; scoped `MLXArray` ownership; a focused tensor-op
surface; dtypes needed for model inference; selected fast, random, quantization and checkpoint I/O
operations; primitive reverse-mode `valueAndGrad`; a small decoder-oriented `nn` layer set; pure-Java
HF BPE tokenization and chat templates; and Llama/Qwen greedy decoding.

Phase 5 M3 (`jmlx-models`) shipped without a standalone implementation plan. Before Phase 6 adds
to that code, Phase 6.0 must create a short retrospective design record for its model-loading,
weight-mapping, generation, scope-ownership, and test contracts. It is a prerequisite, not a
request to rewrite delivered code.

This is sufficient to prove end-to-end decoder inference, but not to act as a general MLX replacement.
The binding contains far more capabilities than the public Java facade exposes.  In particular,
general indexing, convolution, FFT and linalg, broad random/statistics, higher-order transforms,
optimizers, compilation/runtime control, distributed execution, custom kernels, and most neural
network modules remain outstanding.

## Roadmap-wide rules

1. **One native inventory from Phase 6.0 onward.** Generate and maintain
   `req/mlx-api-inventory.md` from the committed bindings now, before Phases 6–7 add more public
   surface. It records every public mlx-c symbol/group, its Java facade location, status
   (`unplanned`, `planned`, `implemented`, `unsupported-by-runtime`, `derived-java-api`,
   `internal/unstable-upstream`), tests, and *both* independent native pins (`mlx-metal` version and
   `mlx-c` commit). A derived API (such as Java `grad` built from `mlx_value_and_grad`) is not a
   bound-symbol parity entry; `internal/unstable-upstream` records a generated `mlx_detail_*` binding
   intentionally excluded from the public supported surface. Add a check that fails when handwritten
   code reaches an unrecorded native type or symbol: it must cover `mlx_h` downcalls, generated struct
   layout/accessor types, and generated upcall functional interfaces, not just `mlx_h.*` text. The
   inventory is the quantitative source of the final parity claim and its per-group counts must
   inform later prioritization.
2. **Plan native uncertainty first.** A milestone plan must name the exact mlx-c declarations it
   uses and include a small Apple-Silicon probe whenever ownership, callback lifetime, error
   conventions, lazy behavior, stream behavior, or struct layout is uncertain.
3. **Keep facades coherent.** Extend `MLX`, `MLXOps`, `MLXShape`, `MLXFast`, `MLXQuant`, `MLXRandom`,
   `MLXGrad`, and `MLXIO` when their existing responsibility fits.  Create a new facade only for a
   stable family (for example `MLXLinalg`, `MLXFFT`, `MLXIndexing`, `MLXRuntime`). Production model
   code must not reach into `ffi`: this is structurally enforced today because `jmlx-core` exposes
   `jmlx-ffi` as an implementation dependency, not an API dependency. Test-only use of
   `se.alipsa.jmlx.ffi.EnabledIfNativeAvailable` remains the required native-availability gate.
4. **Safety before breadth.** Every new operation must honor `MLXScope` ownership/confinement,
   clean up temporary native vectors/maps/strings on both success and failure, and surface native
   failures as `MLXException` rather than terminating the process.
5. **MLX semantics, not convenient subsets.** Validate broadcasting, axes, dtype promotion,
   negative indices/strides, aliases/views, stream defaults and lazy evaluation against MLX.
   Java-side validation may improve error messages but must not reject valid MLX input.
6. **Test at three levels.** Add small hand-computed Java tests; differential tests against a
   pinned Python MLX oracle where practical; and integration tests using real artifacts/models when
   a feature’s value is interoperability or performance. The oracle is infrastructure with an
   explicit Phase 6.0 owner, not an assumed future dependency. Hardware-required tests retain the
   existing native-availability guard; the existing `native` job in
   `.github/workflows/ci.yml` already exercises native modules on the `macos-26` ARM64
   runner and must be extended as new native tests land.
7. **No unbounded model promise.** “Model support” is capability based, not a list of marketing
   names.  A family is supported only when a published artifact, tokenizer/template, generation
   result, and defined compatibility test all pass.
8. **Parity is a direction with product milestones.** Full pinned-surface parity is expected to be
   multi-year work, not a prerequisite for useful releases. Phase 6 alone is a defensible local
   text-LLM product and Phases 6–8 a defensible general-inference/array product. Re-evaluate later
   phase order and estimates from the inventory's per-group counts rather than treating the final
   certification gate as a near-term delivery commitment.

## Phase 6 — Complete local text-LLM inference

**Objective:** turn the current Llama/Qwen greedy proof into a stable, efficient, user-facing local
text-generation stack.  This phase is the first priority.  Its done condition is *text* inference;
vision, audio and diffusion inference are Phase 7 so that they cannot delay a dependable LLM path.

### 6.0 — Inference contract and Phase 5 M3 design record

Define the public inference API before adding model-specific code.  Specify model loading, prompt
prefill, incremental decode, token streaming, cancellation, errors, resource ownership, deterministic
generation, and model/tokenizer compatibility. Create the Phase 5 M3 retrospective design record
described above.

**Exit gate:** API javadocs and compatibility matrix exist; at least Llama and Qwen run through the
new contract without a compatibility shim; and the M3 retrospective record defines the contracts
Phase 6 will extend.

### 6.0b — Inventory, oracle, and fixture infrastructure

Generate the initial native API inventory from `jmlx-ffi`'s committed bindings and add the call-site
coverage check described in Rule 1. Establish the Python MLX oracle as owned infrastructure: decide
whether it uses the same `mlx-metal` pin and bootstrap provenance as the Java native runtime, which
goldens are committed versus calculated in CI, and how it runs on the existing `macos-26` native job.

Split fixtures into two explicit tiers. Keep tiny, randomly initialized checkpoints and expected
shape/dtype/token fixtures in-repository for correctness on every relevant PR. Run real-artifact
tests only in a separately triggered or scheduled job, with size/cache policy and authenticated
Hugging Face access where a license requires it; never make gated or multi-GB downloads a per-PR
requirement.

**Exit gate:** the generated inventory and call-site check exist; the oracle provenance is pinned;
and in-repo correctness fixtures run in the existing macOS/arm64 CI job without external model
credentials. This milestone may proceed in parallel with 6.0 and does not block the API work in
6.1–6.2.

### 6.1 — Generation and sampling

Add a `GenerationConfig` and streaming generation API supporting seeded sampling, temperature,
top-k, top-p, min-p, repetition/frequency/presence penalties, EOS/stop-token handling, maximum-token
limits, optional log probabilities, and cancellation.  Use MLX random/indexing primitives where
appropriate and define numerical corner cases (zero temperature, empty candidate sets, NaNs).

Before policy code, expose and inventory the necessary focused native slice: top-k/top-p selection
(`topk`/sort/argsort/partition as the chosen algorithm needs), categorical sampling, and explicit
random key/split semantics. Phase 6 owns the resulting per-request RNG contract; Phase 9 extends it.

**Exit gate:** greedy generation remains bit-for-bit deterministic for the same jmlx binary, pinned
MLX runtime, device, and batch configuration; bit-exactness is not claimed across changed batch
shapes. Seeded sampling is reproducible; sampling distributions and stop behavior have oracle and
integration tests.

### 6.2 — Tokenizer and prompt compatibility

Expand the tokenizer module from its byte-level BPE focus to the tokenization families required by
the supported model matrix, notably SentencePiece/Unigram and WordPiece where artifacts require them.
Complete special-token handling, truncation/padding, offset behavior where exposed, and chat-template
compatibility for target models.  Keep it pure Java unless a separately approved performance or
compatibility probe proves that impossible.

**Exit gate:** the implemented Llama/Qwen model artifacts can load tokenizer metadata/templates and
produce reference prompt goldens; ByteLevel/Metaspace BPE, non-Precompiled Unigram, and
Bert/WordPiece component fixtures match the pinned Hugging Face oracle. Tokenizer-directory and
chat-template goldens for Mistral, Gemma, Phi, and Mixtral land with their 6.3 model loaders, so a
synthetic tokenizer fixture does not prematurely upgrade an architecture's compatibility claim.

### 6.3 — Architecture and checkpoint breadth

Refactor `DecoderModel` around explicit architectural capabilities rather than Llama/Qwen branches.
First add configurable RoPE scaling: linear, dynamic-NTK, Llama 3, and YaRN forms required by the
compatibility matrix. `DecoderConfig` currently rejects `rope_scaling`, so this is a prerequisite
for Llama 3.1+ and many current checkpoints, not a family-specific enhancement. Add the decoder
components and config mappings needed for a planned compatibility matrix: Mistral, Gemma, Phi, and a
MoE family (for example Mixtral), including their normalization, attention, activation, tied-head,
sliding-window, and expert-routing differences. Support common safetensors sharding/index forms and
documented GGUF quantization variants; fail clearly for unsupported tensors or metadata rather than
partially loading a model.

**Exit gate:** an in-repo synthetic fixture covers each architecture capability and produces golden
prefill/decode token IDs; the separately triggered real-artifact suite covers one accessible,
licensed artifact per family; and unsupported checkpoint features identify the tensor/key and missing
capability.

### 6.4 — Decode performance and cache management

Make prefill and decode first-class execution paths.  Add cache capacity policies, sliding-window
eviction, cache reset/fork/reorder needed for batched decoding, cache quantization when MLX supports
the chosen representation, and batch-safe position/mask construction.  Benchmark cold load, prefill,
single-token decode, and memory growth using a published benchmark harness; optimize only after the
baseline is recorded.

**Exit gate:** repeated generation has bounded native memory; long-context/sliding-window behavior
is correct; the benchmark reports tokens/s and peak memory reproducibly for every supported family.

### 6.5 — Batched serving primitives and release hardening

Provide bounded multi-request batching primitives suitable for embedding in an application (not an
HTTP server): prompt batches, heterogeneous decode lengths, cancellation isolation, back-pressure,
and per-request deterministic RNG.  Add model download/cache guidance or a deliberately separate
artifact resolver, package smoke tests, examples, and an inference compatibility report.

**Phase 6 done:** a Java application can load a documented set of modern local text LLMs, render a
chat prompt, stream reproducible sampled output, run bounded batched generation, and close all
resources safely.  No claim is made yet for multimodal or diffusion models.

## Phase 7 — Inference breadth and conventional neural-network execution

**Objective:** cover inference workloads outside decoder-only text generation and add the missing
neural modules they require.

Milestones, to plan in dependency order:

1. **Core inference modules:** `Sequential`/containers, dropout evaluation semantics, softmax and
   common losses/activations, convolution and transpose convolution, pooling, padding, upsampling,
   normalization variants, and embedding variants.
2. **Encoder and encoder-decoder models:** BERT-style encoders, cross-attention, encoder-decoder
   generation, and classification/token-classification APIs.
3. **Vision and multimodal:** image preprocessing contract, vision transformer/CLIP-style encoders,
   projector layers, multimodal prompt/token conventions, and a tested vision-language model.
4. **Audio/diffusion only after a capability decision:** add STFT/ISTFT or scheduler/UNet/VAE support
   only when the feature inventory and a concrete reference model justify it.

**Phase 7 done:** supported non-text modalities are stated explicitly in the compatibility matrix;
the `nn` API covers their required common layers without model packages bypassing core facades.

## Phase 8 — Fundamental array, dtype, and indexing parity

**Objective:** make jmlx useful as a general lazy array library. The inventory already starts in
Phase 6.0; this phase completes its core-array groups before advanced linalg/FFT so all later work
shares correct dtype, axis, view, and indexing rules.

Milestones:

1. Complete array construction and host transfer: all native dtypes supported by the pinned MLX,
   scalar/complex construction, typed read-back, safe zero-copy entry points where ownership permits,
   `copy`, `contiguous`, and layout/stride inspection. Any zero-copy construction plan must probe and
   handle `mlx_array_new_data`'s statusless null-context failure convention.
2. Complete elementary math, comparisons, logical/bitwise functions, finite/NaN predicates, rounding,
   clipping, type promotion and scalar overload policy.
3. Complete reductions and scans: all/any, min/max/prod, arg variants, variance/std/median,
   logsumexp, cumulative operations, axis/keepdims semantics.
4. Complete shape/view manipulation: expand/squeeze variants, moveaxis, stack, tile/repeat/roll, pad,
   split variants, diagonal/diag, meshgrid, triangular/window/identity creation.
5. Complete indexing/update: slice forms including dynamic and update operations; take-along-axis,
   put-along-axis, gather/scatter and masked operations.

**Phase 8 done:** all core-array entries in the inventory have Java APIs and differential coverage,
or are recorded as a native-runtime limitation with a tested, documented failure.

## Phase 9 — Numerical, random, and specialized compute parity

**Objective:** expose MLX’s non-neural numerical computation families on the stable array foundation.

Milestones:

1. `MLXLinalg`: tensordot/einsum/kron/addmm and the complete pinned linalg decomposition/solve API,
   including result records and batched semantics.
2. `MLXFFT`: complex dtype policy, one- and multi-dimensional FFT family, shifts and normalization.
3. Complete random: explicit-key and global-key contracts, splitting, categorical/permutation/integers,
   and all native distributions with deterministic tests.
4. Add remaining specialized primitives: segment/reduction operations, sorting/partition/top-k,
   quantization variants, block/segmented matrix operations, and sparse-related APIs if present in the
   pinned C surface.

**Phase 9 done:** every linalg, FFT, random and specialized-compute inventory group is complete and
differentially tested for representative shapes, dtypes, and error cases.

## Phase 10 — Autodiff and function-transformation parity

**Objective:** make composition, differentiation, vectorization and compilation-grade function
objects safe from Java.

Milestones:

1. Extend the shipped multi-argument `valueAndGrad` and `ModuleGrad` path-keyed parameter-gradient
   support with auxiliary outputs and nested/non-flat Java pytree representations. Probe whether
   `mlx_value_and_grad` composes safely through jmlx's closure/holder/arena lifecycle before planning
   grad-of-grad; it is not assumed to be a routine extension.
2. Add Java `grad` as a documented derived convenience over `valueAndGrad`, not as a fictitious
   native-symbol binding. Add direct wrappers for VJP, JVP, and checkpointing; define ownership of
   arrays returned from callbacks and behavior if Java callbacks throw.
3. Before planning vmap or compiled-function APIs, decide whether generated `mlx_detail_vmap_*` and
   `mlx_detail_compile_*` symbols are internal/unstable upstream API or supported public surface,
   and record that decision in the inventory. Only if accepted, add vmap/custom VJP/custom JVP/custom
   function wrappers after callback-upcall lifetime, exception translation, and thread/arena behavior
   are proven by native probes.
4. Add graph inspection/export APIs that MLX exposes and a reusable differential gradient suite.

**Phase 10 done:** each pinned differentiation/transformation capability has an ergonomic Java
equivalent, with numerical-gradient and Python-MLX differential tests plus callback-lifetime tests.

## Phase 11 — Training-ready `nn` and optimizer parity

**Objective:** build the high-level layers and training loop facilities needed to use Phase 10 in
ordinary Java model development.

Milestones:

1. Complete module traversal, serialization, train/eval mode, parameter freezing, state dictionaries,
   reusable containers, initialization, and loss APIs.
2. Complete standard layers required by the pinned MLX `nn` scope: convolution/pooling, normalization,
   recurrent or attention variants if provided, embeddings, positional encodings, dropout, and common
   activation families.
3. Add optimizer families and schedulers with explicit state ownership and checkpoint round trips.
4. Provide a reference fine-tuning/training example with deterministic resume, gradient validation,
   memory-bound tests and documented limits.

**Phase 11 done:** a documented model can train, checkpoint, resume, and evaluate entirely from the
public Java API; all claimed MLX `nn`/optimizer equivalents are in the inventory.

## Phase 12 — Runtime, compilation, custom-kernel, and distributed parity

**Objective:** surface system-level MLX facilities without compromising Java lifecycle guarantees.

Milestones:

1. `MLXRuntime`: device enumeration/selection, streams, synchronization, asynchronous evaluation,
   cache/memory limits and statistics, compile mode, and Metal availability/capture controls.
2. After Phase 10's `mlx_detail_compile_*` scope decision, expose only accepted compiled-function
   capabilities plus graph export/import and cache lifetime/error semantics. An
   `internal/unstable-upstream` decision means documenting, not wrapping, those generated symbols.
3. Custom Metal kernels: safe configuration builders, template/output arguments, asynchronous error
   reporting, and minimal kernel integration tests.  CUDA exposure follows only if it is supported by
   the pinned runtime and a CI-capable environment exists.
4. Distributed collectives: first probe and record `mlx_distributed_is_available` on the supported
   CI runner. If unavailable, document that result in the inventory; if available, plan groups,
   initialization, send/receive and collective operations only after a supported multi-host or
   multi-process CI environment exists, rather than assuming one can be built on the hosted
   single-Mac runner.

**Phase 12 done:** all pinned runtime/system groups are either safely exposed and integration-tested
or documented as unavailable on the target runtime, never merely absent from the Java facade.

## Phase 13 — Parity certification and ongoing compatibility

**Objective:** turn feature completeness into a maintained claim rather than a one-time assertion.

1. Close every remaining inventory entry and publish a generated coverage report by API group; use
   its counts and effort data to publish an updated scope/estimate for any remaining work.
2. Run the full Java and Python-MLX differential suite on the pinned macOS/arm64 environment;
   preserve minimized regressions as fixtures.
3. Audit public API stability, resource ownership, exception taxonomy, documentation, examples,
   packaging, and native artifact compatibility.
4. Establish the MLX-upgrade procedure: upgrade one pinned release at a time, regenerate bindings,
   diff the inventory, add compatibility tests, and publish a migration note before claiming parity
   for that release.

**Full-parity release gate:** 100% of the pinned public mlx-c API groups are represented in the
inventory as implemented or as a verified native unavailability; all implemented entries have
semantic tests; supported platform limits are published; and the release passes macOS/arm64 CI.

## Template for every follow-on implementation plan

Each `req/plans/phase<N>-m<M>-plan.md` should contain:

1. Goal, non-goals, prerequisites, and the roadmap milestone/exit gate it satisfies.
2. Exact public Java API proposed, affected modules/files, and exact mlx-c declarations.
3. Findings from required probes, including native ownership and error behavior.
4. Detailed implementation tasks ordered so unsafe FFM/lifecycle foundations land before consumers.
5. Compatibility decisions: MLX semantics, dtype/shape/promotion rules, lazy/stream behavior, and
   backward compatibility.
6. Tests: hand-computed unit cases, differential oracle cases, integration artifacts/models, leak and
   scope/confinement tests, plus CI requirements.
7. Benchmarks when the milestone affects inference, compilation, transfers, or memory behavior.
8. Documentation, examples, inventory updates, explicit deferred items, and measurable completion
   criteria. Plans using differential or real-artifact tests must also state oracle provenance and
   fixture tier/credential requirements.

No milestone is complete merely because its methods compile.  It is complete only when its exit gate,
tests, documentation and inventory changes are all delivered.
