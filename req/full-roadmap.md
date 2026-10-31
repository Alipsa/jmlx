# jmlx Full MLX-Parity Roadmap

## Purpose and definition of done

This is the long-range planning index for jmlx after Phases 1–5. It extends and supersedes the
forward-looking "Project Execution Plan" in `req/project-outline.md`, whose planned phases stop at
Phase 5. It is deliberately a roadmap, not an implementation plan: each milestone below must get
its own plan in `req/plans/` before code is started. Those plans should preserve the decisions and
verification style established by `req/plans/initial-plan.md` and Phases 3–5.

**The phases are sequential.** Work on a phase starts only when the previous phase is done (its
"done" condition is met and its release, if it has one, is published). Within a phase, milestones
run in the listed order unless the phase plan states that two may overlap. Each phase states up
front whether it upgrades MLX and what it releases; the overview table below collects both.

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

`req/mlx-api-inventory.md` is the single readable record of the current two-pin target and
coverage; the bootstrap script remains the source of truth for the pin values.

## Phase overview

| Phase | Content | MLX upgrade | Release at the end | Status |
| --- | --- | --- | --- | --- |
| 1–5 | Native bootstrap, bindings, memory model, tensor ops, `nn`, autograd, checkpoint I/O, tokenizer, first models | — | none | delivered |
| 6 | Complete local text-LLM inference | none | none | delivered |
| 7 | Inference breadth: Qwen3, encoders, vision-language, Qwen3.8 | once, as milestone 7.5 (before Qwen3.8) | none | in progress (7.3c) |
| 8 | First releases and framework adapters | none (ships the 7.5 runtime) | first native release (milestone 8.4; the pure-Java modules are released earlier, in 8.1) | not started |
| 9 | Inference product: decode speed, prefix cache, HTTP server, Spring Boot/Quarkus, Groovy API, diffusion inference (9.6, scheduled by the 7.4 decision) | 9.0 | 0.x release | not started |
| 10 | Fundamental array, dtype and indexing parity | 10.0 | 0.x release | not started |
| 11 | Numerical, random and specialized compute parity; audio inference (11.5, scheduled by the 7.4 decision) | 11.0 | 0.x release | not started |
| 12 | Autodiff and function-transformation parity | 12.0 | 0.x release | not started |
| 13 | Training-ready `nn` and optimizer parity | 13.0 | 0.x release | not started |
| 14 | Runtime, compilation, custom-kernel and distributed parity | 14.0 | 0.x release | not started |
| 15 | Parity certification | 15.0, then frozen until 1.0 | 1.0 of the native chain | not started |

The rules behind the "MLX upgrade" and "Release" columns are in § Release and MLX upgrade plan.
Phases 10–15 were numbered 8–13 before 2026-10-30; § Revision history has the mapping.

## Current baseline (Phases 1–5)

Delivered: native loading and generated bindings; scoped `MLXArray` ownership; a focused tensor-op
surface; dtypes needed for model inference; selected fast, random, quantization and checkpoint I/O
operations; primitive reverse-mode `valueAndGrad`; a small decoder-oriented `nn` layer set; pure-Java
HF BPE tokenization and chat templates; and Llama/Qwen greedy decoding.

Phase 5 M3 (`jmlx-models`) shipped without a standalone implementation plan. Phase 6.0 created the
retrospective design record for its model-loading, weight-mapping, generation, scope-ownership and
test contracts (`req/plans/phase5-m3-retrospective.md`).

This is sufficient to prove end-to-end decoder inference, but not to act as a general MLX replacement.
The binding contains far more capabilities than the public Java facade exposes.  In particular,
general indexing, convolution, FFT and linalg, broad random/statistics, higher-order transforms,
optimizers, compilation/runtime control, distributed execution, custom kernels, and most neural
network modules remain outstanding.

## Roadmap-wide rules

1. **One native inventory from Phase 6.0 onward.** Generate and maintain
   `req/mlx-api-inventory.md` from the committed bindings. It records every public mlx-c
   symbol/group, its Java facade location, status (`unplanned`, `planned`, `implemented`,
   `unsupported-by-runtime`, `derived-java-api`, `internal/unstable-upstream`), tests, and *both*
   independent native pins (`mlx-metal` version and `mlx-c` commit). A derived API (such as Java
   `grad` built from `mlx_value_and_grad`) is not a bound-symbol parity entry;
   `internal/unstable-upstream` records a generated `mlx_detail_*` binding intentionally excluded
   from the public supported surface. A check fails when handwritten code reaches an unrecorded
   native type or symbol: it covers `mlx_h` downcalls, generated struct layout/accessor types, and
   generated upcall functional interfaces, not just `mlx_h.*` text. The inventory is the
   quantitative source of the final parity claim and its per-group counts must inform later
   prioritization.
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
   a feature’s value is interoperability or performance. Hardware-required tests retain the
   existing native-availability guard; the `native` job in `.github/workflows/ci.yml` exercises
   native modules on the `macos-26` ARM64 runner and must be extended as new native tests land.
7. **No unbounded model promise.** “Model support” is capability based, not a list of marketing
   names.  A family is supported only when a published artifact, tokenizer/template, generation
   result, and defined compatibility test all pass.
8. **Parity is a direction with product milestones.** Full pinned-surface parity is expected to be
   multi-year work, not a prerequisite for useful releases. Phases 6–9 are a defensible local
   inference product and Phases 6–10 a defensible general-inference/array product. Re-evaluate
   the scope and estimates of later phases from the inventory's per-group counts rather than
   treating the final certification gate as a near-term delivery commitment.
9. **Pulling a focused slice forward.** Sequential phases do not forbid a milestone from
   implementing a focused slice of a later phase's API when the milestone needs it (for example
   Phase 6's sampling selection ops, Phase 7.1's array ops, Phase 9.1's `mlx_async_eval` and
   `mlx_compile`). The milestone plan lists each pulled-forward symbol, the inventory records it
   where it lands, and the later phase extends that contract rather than re-deriving it. This
   pulls API forward, never a whole later milestone.
10. **The Groovy API follows the Java API.** Once Phase 9.5 publishes `jmlx-groovy`, every later
    phase that adds public Java API also adds its Groovy form, or lists the name as pending, as
    part of that phase's done condition.

## Phase 6 — Complete local text-LLM inference

**Status: delivered.** `req/phase6-inference-report.md` summarizes the result and
`req/phase6-compatibility.md` is the supported-family matrix. **MLX upgrade:** none.
**Release:** none; the remaining release steps it lists moved to Phase 8.

**Objective:** turn the current Llama/Qwen greedy proof into a stable, efficient, user-facing local
text-generation stack.  Its done condition is *text* inference; vision, audio and diffusion
inference are Phase 7 so that they cannot delay a dependable LLM path.

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
random key/split semantics. Phase 6 owns the resulting per-request RNG contract; Phase 11 extends it.

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

### 6.4.1 — Quantized KV retention and packed attention

**Status (2026-10-01): deferred pending a new attention design.** The step 1 prototype stopped:
four-bit packed attention misses its accuracy criterion because of 4-bit affine rounding of K/V, which
no attention kernel can avoid (`req/phase6-4-benchmark.md`). The unsupported-capability rejection
stays. Reopening it is a dated amendment to `req/plans/phase6-4-1-plan.md`, scheduled into a
later phase by a roadmap amendment.

The planned scope was an opt-in packed-KV attention path that attends to compressed keys and
values without materializing the entire float cache per decode step, with lower steady and peak
native memory than float retention at a documented long context.

### 6.5 — Batched serving primitives and release hardening

Provide bounded multi-request batching primitives suitable for embedding in an application (not an
HTTP server; Phase 9.3 adds that): prompt batches, heterogeneous decode lengths, cancellation
isolation, back-pressure, and per-request deterministic RNG.  Add model download/cache guidance or
a deliberately separate artifact resolver, package smoke tests, examples, and an inference
compatibility report.

**Phase 6 done:** a Java application can load a documented set of modern local text LLMs, render a
chat prompt, stream reproducible sampled output, run bounded batched generation, and close all
resources safely.  No claim is made yet for multimodal or diffusion models.

## Phase 7 — Inference breadth and conventional neural-network execution

**Objective:** cover inference workloads outside decoder-only text generation and add the missing
neural modules they require. The primary product target is the Qwen family: Qwen3 text decoders,
Qwen3-VL and the hybrid linear-attention Qwen3.8.

**MLX upgrade:** once, as milestone 7.5, after 7.4 and before 7.6. **Release:** none. The first
releases are Phase 8, so they ship a runtime that every Phase 7 family was verified on.

`req/plans/phase7-plan.md` is the phase plan: it holds each milestone's scope, the public-contract
decisions, the focused Phase 10 array slices pulled forward under rule 9, and the gates. Each
milestone also has its own sub-plan, `req/plans/phase7-<n>-plan.md`, written and reviewed before
code.

Milestones, in order:

| Milestone | Scope | Status (2026-10-31) |
| --- | --- | --- |
| 7.0 Qwen3 text decoders | `qwen3` mapping, per-head QK normalization, affine-quantized loading, Qwen3 chat template; Tier-B Qwen3-0.6B, manual Qwen3-8B 4-bit run | delivered |
| 7.1 Core inference modules | containers, dropout (eval), activations, convolution and transpose convolution, pooling, padding, upsampling, normalization variants, positional encodings, evaluation losses | implemented; acceptance pending (oracle drift, cross-host CPU fixtures) |
| 7.2 Encoders and encoder-decoder | BERT-style encoders, cross-attention, T5/Flan-T5 generation, classification and token-classification APIs | implemented; macOS 26 CI acceptance pending |
| 7.3a Vision foundation + SmolVLM | `jmlx-vision` image preprocessing, SigLIP-style tower, Idefics3 connector, multimodal prompts | delivered; macOS 26 hosted acceptance pending |
| 7.3b Qwen3-VL | dynamic-resolution preprocessing, multi-axis RoPE, vision tower, DeepStack | delivered; macOS 26 hosted Tier-B pending |
| 7.3c LLaVA-style connector | `microsoft/Phi-3.5-vision-instruct` (MIT), selected 2026-10-27 | delivered 2026-10-30 |
| 7.4 Audio/diffusion decision | a reviewed decision record, not code | decision record delivered 2026-10-31 (`req/plans/phase7-4-decision.md`), reviewed via its PR: go — audio scheduled as Phase 11 milestone 11.5, diffusion as Phase 9 milestone 9.6 |
| 7.5 MLX upgrade | paired pin upgrade to the newest release (see § Release and MLX upgrade plan) | not started |
| 7.6 Hybrid linear-attention Qwen (Qwen3.8) | `model_type` `qwen3_5`, text and image+text: Gated DeltaNet with a recurrent-state cache, gated full attention, partial rotary, HF/mlx-vlm/`mlx_lm` checkpoint layouts; Tier-B `Qwen/Qwen3.5-0.8B`, manual `mlx-community/Qwen3.8-27B-4bit` run | not started |

7.4 decides only. A "go" for audio or diffusion schedules its implementation into a later phase
by a dated roadmap amendment; it does not extend Phase 7.

**Amendment (2026-10-31, from the 7.4 decision record `req/plans/phase7-4-decision.md`):**
"go" for both modalities. Phase 9 gains milestone **9.6 — Diffusion inference** (SD 1.5-class
text-to-image in `jmlx-models`; new surface is pure-Java — noise schedulers with PNDM first, the
artifact's default, then DDPM/DDIM/Euler — plus a sampler loop, VAE wrapper, CLIP text encoder,
CLIP slow-tokenizer loader and a hash-locked diffusers reference dependency; Tier-B is
`stable-diffusion-v1-5/stable-diffusion-v1-5` under CreativeML OpenRAIL-M, **accepted in the
decision record** for a pinned read-only test artifact). Phase 11 gains milestone **11.5 —
Audio inference** (`openai/whisper-tiny`, apache-2.0; a Java 21 native-free `jmlx-audio` module
for WAV decode and the Slaney filterbank, a forward-STFT facade in `jmlx-core` on 11.2's
`MLXFFT`, and the Whisper encoder-decoder in `jmlx-models`; `jmlx-audio` is published, growing
the published-module invariants to eight). Both milestones carry their
own exit gates (9.6 and 11.5). Neither extends Phase 7; 7.5 follows 7.4 directly.

**Phase 7 done:** every milestone above is delivered, or deferred with a date; every pending
acceptance gate listed above (the 7.1 oracle and cross-host fixtures, the macOS 26 hosted runs
for 7.2, 7.3a and 7.3b) has passed or carries a dated, recorded waiver; supported non-text
modalities are stated explicitly in the compatibility matrix; and the `nn` API covers their
required common layers without model packages bypassing core facades.

## Phase 8 — First releases and framework adapters

**Objective:** publish jmlx to Maven Central for the first time, with an honest performance
statement and an integration path into the two main Java AI libraries.

**MLX upgrade:** none. The first native release ships the runtime that 7.5 installed and 7.6 was
verified on. **Release:** the pure-Java modules in milestone 8.1, and the first native release in
milestone 8.4, which ends the phase. Every module keeps its own version line (§ Versioning and
compatibility policy), so the versions below differ between modules.

New published modules (here the adapters, later the server, the Spring Boot/Quarkus integration
and the Groovy API) each start their own `0.1.0-SNAPSHOT` line. They join
`verifyReleaseScriptsMatch`, the release order and the release smoke, and none of them is
published before `jmlx-models` itself.

1. **8.1 Pure-Java release.** Extend `tools/release-smoke` so its `candidate` and `central`
   modes can verify a partial module set (they currently assume all seven). Then release
   `jmlx-jinja` 0.6.0 (continuing `hfjinja` 0.5.0 on Central), `jmlx-tokenizer` 0.1.0 and
   `jmlx-vision` 0.1.0. These modules have no native dependency, target Java 21 and are useful on
   their own. This release also proves the release procedure on low-risk modules before the native
   release.
2. **8.2 Decode baseline against `mlx_lm`.** Benchmark jmlx against `mlx_lm` on byte-identical
   weights (one dense family, one MoE family and one 4-bit checkpoint). Measure prefill tokens/s,
   decode tokens/s, time to first token and peak memory, using the existing benchmark harness and
   the reproducibility tuple from `req/phase6-4-benchmark.md`. Record per-token host overhead,
   including the native call count and FFM cost per decode step. This is a measurement only;
   optimization is Phase 9.1, so the first native release states decode speed honestly before any
   optimization.
3. **8.3 Spring AI and LangChain4j adapters.** `ChatModel`, `StreamingChatModel` and
   `EmbeddingModel` implementations for each library (the LangChain4j ones also give Quarkus users
   the quarkus-langchain4j path). Callers may run on virtual threads: the
   `BatchGenerationScheduler` worker keeps sole ownership of MLX, so the threading rule holds
   without framework cooperation. The plan states the supported version matrix (current
   Spring AI and LangChain4j releases support Java 25) and factors the request/response mapping
   so that Phase 9.3's server reuses it. **Gate:** adapter-level tests pass against the
   release-smoke repository: chat, streaming and embeddings through each library's interfaces,
   with native memory released on close.
4. **8.4 First native release.** `jmlx-native-macos-arm64` 0.1.0, `jmlx-ffi` 0.5.0, `jmlx-core`
   0.5.0, `jmlx-models` 0.1.0 and the two adapters at 0.1.0, plus a new version of any pure-Java
   module that changed since 8.1. Run `tools/release-smoke/run.sh candidate` before publishing
   and `run.sh central` after.

**Phase 8 done:** the 8.1 and 8.4 releases resolve from Maven Central, and the 8.4 release notes
include the 8.2 benchmark, the compatibility matrix and the native pins.

## Phase 9 — Inference product

**Objective:** close the gaps in jmlx's inference *product*, not its MLX parity. A survey of
other JVM front-ends on Apple Silicon (mlx-scala on MLX through FFM; lightmetal on llama.cpp;
jitLLM on TornadoVM; Jlama) showed them: decode speed relative to `mlx_lm`, reuse of prompt
prefixes between requests, a network protocol that existing clients speak, framework integration,
and a concise scripting API.

**MLX upgrade:** 9.0. **Release:** a 0.x release at the end of the phase, including the first
release of `jmlx-server`, the Spring Boot starter, the Quarkus extension and `jmlx-groovy`.

### 9.0 — MLX upgrade

As defined in § Release and MLX upgrade plan.

### 9.1 — Decode performance parity with `mlx_lm`

Pulled forward under rule 9:
- `mlx_async_eval`, from Phase 14 milestone 1 (asynchronous evaluation);
- the compile-mode controls in `mlx/c/compile.h` (`mlx_enable_compile`, `mlx_disable_compile`
  and `mlx_set_compile_mode`), from Phase 14 milestone 1 (compile mode);
- `mlx_compile` itself, from Phase 14 milestone 2.

The `mlx_detail_compile*` symbols remain under Phase 12 milestone 3's internal/unstable decision
and are not pulled forward.

1. **Re-baseline.** Re-run 8.2's benchmark on the 9.0 runtime before any optimization.
2. **Pipelined decode.** Build and `async_eval` step *n+1* while step *n* runs on the GPU. The
   plan must first probe:
   - stream and thread binding of in-flight arrays;
   - `MLXScope` ownership of arrays that are still being evaluated;
   - cancellation and scheduler cohort changes while a step is in flight;
   - how asynchronous native failures surface as `MLXException`.
3. **Compiled hot functions.** Compile the elementwise functions `mlx_lm` compiles (for example
   SiLU/SwiGLU gating) through `mlx_compile`, reusing `MLXGrad`'s closure, upcall and arena
   lifecycle. Callback-exception translation, compile-cache lifetime per thread, and shapeless
   versus shaped compilation are probed before any consumer lands.

**Exit gate:** the published benchmark reports jmlx/`mlx_lm` ratios for the families measured in
8.2. Every existing golden, precision task and cache-equivalence suite passes unchanged in both
TF32 modes, with tolerances unmodified. Each optimization is opt-out-able, and its effect is
recorded separately in the benchmark.

### 9.2 — Prompt prefix cache

Reuse already-computed KV state for a shared token prefix across requests: a multi-turn chat, a
fixed system prompt, or tool definitions. Build on `KVCache.fork`. The plan defines:

- **Keying:** token IDs plus model identity, dtype/TF32 mode and RoPE configuration. For
  multimodal requests, the key also includes image content digests and their expansion and
  position plan, including the Qwen3-VL multi-axis RoPE delta.
- **Marks:** which prefix boundaries are retained, for example message starts from the chat
  template.
- **Budget:** a bounded memory budget with LRU eviction, and ownership by the model's worker
  thread and a long-lived scope. Thread confinement makes the cache per model thread, not
  process-wide.
- **Cache policies:** interaction with SLIDING and capacity-bounded policies, which cannot be
  truncated and are reused whole or not at all, and with 7.6's recurrent-state cache (whole-prefix
  reuse only).
- **Scheduler integration:** integration with `BatchGenerationScheduler` cohorts.
- **Persistence (optional, second step):** safetensors through `MLXIO`, with a versioned header
  that rejects a cache from a different model, pin or configuration.

**Exit gate:** logits with a cache hit match a full recompute within the existing
cache-equivalence tolerances, for text and image prefixes. Memory stays within the configured
budget under a soak test. A multi-turn chat benchmark records the time-to-first-token reduction.

### 9.3 — OpenAI- and Anthropic-compatible HTTP server

A published `jmlx-server` module. It serves models through `BatchGenerationScheduler`:

- **OpenAI:** `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (backed by
  `TextEncoderModels`) and `/v1/models`.
- **Anthropic:** `/v1/messages`.

Both protocols support:

- SSE streaming in each protocol's own event format;
- cancellation on client disconnect;
- each protocol's own error schema;
- seeds;
- string stop sequences (OpenAI `stop`, Anthropic `stop_sequences`);
- image content parts for the vision-language families.

String stop sequences are new work: `GenerationConfig` has only `stopTokenIds`. A string stop
can span token boundaries, so streamed text must be held back until a partial match is ruled
out. This belongs in `jmlx-models`, so that the server and the 8.3 adapters share a single
implementation.

Tool definitions already reach chat templates through `ChatTemplateOptions.extraContext`. This
milestone adds per-family tool-call *output* parsing into each protocol's structured tool-call
form. The plan decides:

- the HTTP stack (JDK `HttpServer` with zero dependencies versus a framework);
- whether to support OpenAI's Responses API.

**Security defaults:** bind to localhost; optional bearer/API-key authentication; request-size,
`max_tokens` and concurrency limits; and no fetching of remote image URLs unless explicitly
enabled.

**Exit gate:** the official OpenAI and Anthropic SDK clients run a conformance suite against the
server (plain, streaming, tool call, image, cancellation, error cases). Server overhead over
in-process generation is benchmarked. 9.2's cache is used for multi-turn requests.

### 9.4 — Spring Boot and Quarkus integration

Published integration that makes a jmlx model an injectable, lifecycle-managed component, built
on the 8.3 adapters:

- **Spring Boot starter:** auto-configuration and typed configuration properties for the model
  path, cache policy and scheduler limits; start and stop tied to the application context;
  Actuator health and Micrometer metrics (tokens/s, queue depth, native memory).
- **Quarkus extension:** configuration, CDI producers and dev-mode behavior. The extension runs in
  JVM mode only; native-image support is out of scope until GraalVM's FFM support for this native
  stack is probed.

**Exit gate:** a sample Spring Boot application and a sample Quarkus application, each built
against the published-shape smoke repository, load a model, stream a chat completion and shut
down with native memory released. Health and metrics endpoints report correctly.

### 9.5 — Groovy API equivalent to MLX's Python API

MLX's high-level Python API (`mlx.core`, `mlx.nn`, `mlx.optimizers`) is part of MLX itself and is
built on its C++ core; mlx-c is a separate C binding. A published `jmlx-groovy` module gives
Groovy users the same concise surface on top of jmlx's *public Java API only*. It adds no native
calls and no `jmlx-ffi` dependency, so the core facades, scope and confinement rules apply
unchanged.

**Surface:**

- a Python-named static facade (`mx.matmul`, `mx.zeros`, …);
- operator overloading (`plus`, `minus`, `multiply`, `div`, `power`, `negative`, comparisons);
- `getAt`/`putAt` indexing with ranges and negative indices;
- closure-based `grad`/`valueAndGrad` and scope blocks;
- extension methods on `MLXArray`;
- `@CompileStatic` compatibility.

This milestone covers today's array, `nn` inference and grad API. Phases 10–14 extend it under
rule 10. A Python name with no Java equivalent yet is listed as pending, never emulated. Current
Groovy releases support Java 25; the plan states the supported Groovy version.

**Exit gate:** a generated table maps each covered Python name to its Groovy form. Representative
examples from the MLX Python documentation, ported to Groovy, match the Python MLX oracle's
output.

### 9.6 — Diffusion inference

Scheduled by the 2026-10-31 Phase 7.4 decision (`req/plans/phase7-4-decision.md`):
SD 1.5-class text-to-image on the delivered 7.1/7.2 surface, in `jmlx-models`
(the inference-family module home; no new published module). All new surface
is pure-Java — noise schedulers (**PNDM first** — the artifact's
`model_index.json` default — then DDPM/DDIM/Euler), the sampler loop, the VAE
decoder wrapper and the CLIP text encoder — so no new native bindings or
inventory changes are expected; the 9.6 loader additionally performs the
legacy attention-key remap (`query`/`key`/`value`/`proj_attn` →
`to_q`/`to_k`/`to_v`/`to_out.0`) that diffusers does at load time. The
`model_index.json` default scheduler is pinned for the Tier-B reference, so
the per-step noise-prediction comparison is meaningful. `jmlx-tokenizer`
gains a CLIP slow-tokenizer loader over the artifact's own `vocab.json` +
`merges.txt` (the artifact ships no `tokenizer.json`), reproducing the pinned
normalization oracle: the reference venv has no ftfy, so the slow tokenizer's
`BasicTokenizer` fallback (control-character cleaning, CJK splitting,
lowercasing) is the branch to port, plus the IGNORECASE pre-tokenizer regex,
byte encoder and `</w>` suffix. The Tier-B reference generator adds
**diffusers** to the `tools/hf-reference` hash lock (absent today), checked
against `verifyImageOracleDependencyPins`' shared Pillow/NumPy/Transformers
pins. The Tier-B artifact is
`stable-diffusion-v1-5/stable-diffusion-v1-5` (the canonical id; the
`runwayml/` id redirects to it) under CreativeML OpenRAIL-M, **accepted in the
7.4 decision record for a pinned read-only test artifact** — the milestone plan
re-records that acceptance and does not re-open it; the strict Tier-A
synthetic golden is the binding gate. Scope is text-to-image only (no
ControlNet/img2img/inpainting).

**Exit gate:** the synthetic Tier-A golden passes at strict tolerance; the
pinned Tier-B artifact's prompt embedding, per-step noise prediction under
the pinned PNDM schedule and decoded image match the CPU float32 reference
within the recorded ε.

**Phase 9 done:** every milestone's exit gate passes and the phase release is published.

## Phase 10 — Fundamental array, dtype, and indexing parity

**Objective:** make jmlx useful as a general lazy array library. This phase completes the
inventory's core-array groups before advanced linalg/FFT so all later work shares correct dtype,
axis, view, and indexing rules. Phase 7.1 already pulled forward `abs`, `minimum`, `clip`,
`floor`, `log1p`, `max_axes`, `var_axes`, `pad`, `as_strided`, `tile` and `repeat_axis`
(milestones 2–4 below); this phase extends them.

**MLX upgrade:** 10.0. **Release:** a 0.x release at the end of the phase.

Milestones:

0. MLX upgrade (§ Release and MLX upgrade plan).
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

**Phase 10 done:** all core-array entries in the inventory have Java APIs and differential coverage,
or are recorded as a native-runtime limitation with a tested, documented failure.

## Phase 11 — Numerical, random, and specialized compute parity

**Objective:** expose MLX’s non-neural numerical computation families on the stable array
foundation. Milestone 11.5 is the phase’s one non-parity milestone — audio inference,
scheduled by the 2026-10-31 Phase 7.4 decision (`req/plans/phase7-4-decision.md`) — riding
milestone 2’s `MLXFFT`.

**MLX upgrade:** 11.0. **Release:** a 0.x release at the end of the phase.

Milestones:

0. MLX upgrade (§ Release and MLX upgrade plan).
1. `MLXLinalg`: tensordot/einsum/kron/addmm and the complete pinned linalg decomposition/solve API,
   including result records and batched semantics.
2. `MLXFFT`: complex dtype policy, one- and multi-dimensional FFT family, shifts and normalization.
3. Complete random: explicit-key and global-key contracts, splitting, categorical/permutation/integers,
   and all native distributions with deterministic tests.
4. Add remaining specialized primitives: segment/reduction operations, sorting/partition/top-k,
   quantization variants, block/segmented matrix operations, and sparse-related APIs if present in the
   pinned C surface.
5. Audio inference (`openai/whisper-tiny`, the license the artifact declares:
   apache-2.0), scheduled by the 2026-10-31 Phase 7.4 decision
   (`req/plans/phase7-4-decision.md`), in three parts:
   - a new **`jmlx-audio` module (Java 21, native-free, no `jmlx-core`
     dependency — the `jmlx-vision` precedent):** 16 kHz WAV decode,
     resampling, window functions, the Slaney filterbank and log scaling. A
     Java 21 module cannot depend on Java 25 `jmlx-core` class files, so the
     native-facing half lives elsewhere.
   - a forward-STFT composition facade in `jmlx-core`, built with milestone 2's
     `MLXFFT` (forward only; no ISTFT exists in Whisper).
   - the Whisper encoder-decoder in `jmlx-models`, `api`-depending on
     `jmlx-audio` (the vision pattern), reusing the Phase 7.2 seq2seq surface
     plus the decision record's generation controls (forced decoder prefix,
     suppress tokens, 448-position cap).
   **`jmlx-audio` is published** — the project's eighth published module
   (fresh `0.1.0-SNAPSHOT` line): `verifyReleaseScriptsMatch`, its own
   byte-identical `release.sh`, release order before `jmlx-models` (alongside
   `jmlx-vision`), and the release smoke's consumer resolution all grow. Scope
   is the single fixed 30 s window; chunked long-form transcription is out of
   scope. **Exit gate:** log-mel fixtures within a recorded tolerance against
   the pinned torch-STFT oracle path (a float32 FFT through MLX is not
   expected bit-exact; the numpy branch is a different code path and not the
   oracle), a synthetic Tier-A logits golden at strict tolerance, and exact
   greedy transcript IDs on the pinned Tier-B artifact.

**Phase 11 done:** every linalg, FFT, random and specialized-compute inventory group is complete and
differentially tested for representative shapes, dtypes, and error cases, and milestone 11.5's exit
gate passes (it is the phase's only non-parity milestone).

## Phase 12 — Autodiff and function-transformation parity

**Objective:** make composition, differentiation, vectorization and compilation-grade function
objects safe from Java.

**MLX upgrade:** 12.0. **Release:** a 0.x release at the end of the phase.

Milestones:

0. MLX upgrade (§ Release and MLX upgrade plan).
1. Extend the shipped multi-argument `valueAndGrad` and `ModuleGrad` path-keyed parameter-gradient
   support with auxiliary outputs and nested/non-flat Java pytree representations. Probe whether
   `mlx_value_and_grad` composes safely through jmlx's closure/holder/arena lifecycle before planning
   grad-of-grad; it is not assumed to be a routine extension.
2. Add Java `grad` as a documented derived convenience over `valueAndGrad`, not as a fictitious
   native-symbol binding. Add direct wrappers for VJP, JVP, and checkpointing; define ownership of
   arrays returned from callbacks and behavior if Java callbacks throw.
3. Before planning vmap or compiled-function APIs, decide whether generated `mlx_detail_vmap_*` and
   `mlx_detail_compile_*` symbols are internal/unstable upstream API or supported public surface,
   and record that decision in the inventory. This decision does not cover the public
   `mlx_compile`, `mlx_enable_compile`, `mlx_disable_compile` and `mlx_set_compile_mode` in
   `mlx/c/compile.h`, which Phase 9.1 wrapped. Only if accepted, add vmap/custom VJP/custom
   JVP/custom function wrappers after callback-upcall lifetime, exception translation, and
   thread/arena behavior are proven by native probes.
4. Add graph inspection/export APIs that MLX exposes and a reusable differential gradient suite.

**Phase 12 done:** each pinned differentiation/transformation capability has an ergonomic Java
equivalent, with numerical-gradient and Python-MLX differential tests plus callback-lifetime tests.

## Phase 13 — Training-ready `nn` and optimizer parity

**Objective:** build the high-level layers and training loop facilities needed to use Phase 12 in
ordinary Java model development.

**MLX upgrade:** 13.0. **Release:** a 0.x release at the end of the phase.

Milestones:

0. MLX upgrade (§ Release and MLX upgrade plan).
1. Complete module traversal, serialization, train/eval mode, parameter freezing, state dictionaries,
   reusable containers, initialization, and loss APIs. This includes training-mode `Dropout` and
   `BatchNorm`, which Phase 7.1 deliberately left throwing.
2. Complete standard layers required by the pinned MLX `nn` scope: convolution/pooling, normalization,
   recurrent or attention variants if provided, embeddings, positional encodings, dropout, and common
   activation families. This extends the Phase 7.1 layers rather than re-deriving them.
3. Add optimizer families and schedulers with explicit state ownership and checkpoint round trips.
4. Provide a reference fine-tuning/training example with deterministic resume, gradient validation,
   memory-bound tests and documented limits.

**Phase 13 done:** a documented model can train, checkpoint, resume, and evaluate entirely from the
public Java API; all claimed MLX `nn`/optimizer equivalents are in the inventory.

## Phase 14 — Runtime, compilation, custom-kernel, and distributed parity

**Objective:** surface system-level MLX facilities without compromising Java lifecycle guarantees.

**MLX upgrade:** 14.0. **Release:** a 0.x release at the end of the phase.

Milestones:

0. MLX upgrade (§ Release and MLX upgrade plan).
1. `MLXRuntime`: device enumeration/selection, streams, synchronization, asynchronous evaluation,
   cache/memory limits and statistics, compile mode, and Metal availability/capture controls.
   Phase 9.1 already wrapped `mlx_async_eval` and the compile-mode controls.
2. Complete the public `mlx_compile` API (which Phase 9.1 wrapped for its hot functions) with
   graph export/import and cache lifetime/error semantics. Of the `mlx_detail_compile_*`
   capabilities, expose only those Phase 12 milestone 3 accepted. An `internal/unstable-upstream`
   decision means documenting, not wrapping, those generated symbols.
3. Custom Metal kernels: safe configuration builders, template/output arguments, asynchronous error
   reporting, and minimal kernel integration tests.  CUDA exposure follows only if it is supported by
   the pinned runtime and a CI-capable environment exists.
4. Distributed collectives: first probe and record `mlx_distributed_is_available` on the supported
   CI runner. If unavailable, document that result in the inventory; if available, plan groups,
   initialization, send/receive and collective operations only after a supported multi-host or
   multi-process CI environment exists, rather than assuming one can be built on the hosted
   single-Mac runner.

**Phase 14 done:** all pinned runtime/system groups are either safely exposed and integration-tested
or documented as unavailable on the target runtime, never merely absent from the Java facade.

## Phase 15 — Parity certification and ongoing compatibility

**Objective:** turn feature completeness into a maintained claim rather than a one-time assertion.

**MLX upgrade:** 15.0 is the last upgrade before 1.0; the pin is then frozen until 1.0 is
released. **Release:** 1.0 of the native chain.

0. MLX upgrade (§ Release and MLX upgrade plan), then freeze the pin.
1. Close every remaining inventory entry and publish a generated coverage report by API group; use
   its counts and effort data to publish an updated scope/estimate for any remaining work.
2. Run the full Java and Python-MLX differential suite on the pinned macOS/arm64 environment;
   preserve minimized regressions as fixtures.
3. Audit public API stability, resource ownership, exception taxonomy, documentation, examples,
   packaging, and native artifact compatibility.
4. Document the MLX-upgrade procedure (§ Release and MLX upgrade plan) as the maintained
   post-1.0 compatibility process.

**Full-parity release gate:** 100% of the pinned public mlx-c API groups are represented in the
inventory as implemented or as a verified native unavailability; all implemented entries have
semantic tests; supported platform limits are published; and the release passes macOS/arm64 CI.

## Release and MLX upgrade plan

### When releases happen

- **Phases 1–7:** nothing is released.
- **Phase 8:** the pure-Java modules in milestone 8.1 and the first native release in 8.4.
- **Phases 9–14:** every phase ends with a release: the next 0.x minor version of every published
  module the phase changed.
  A milestone that adds public API may also be released on its own, mid-phase; that is optional.
- **Phase 15:** 1.0 of the native chain (`jmlx-native-macos-arm64`, `jmlx-ffi`, `jmlx-core`,
  `jmlx-models`). Pure-Java modules and the Phase 8/9 integration modules carry no parity claim.
  They may reach 1.0 earlier on their own stability criteria, but not before any module they
  depend on.
- A bug-fix patch release is allowed at any time.

Every release follows the existing manual procedure:
- each module's `release.sh` in dependency order;
- `jmlx-jinja/req/release-checklist.md` for jinja;
- `tools/release-smoke/run.sh candidate` before publishing and `run.sh central` after;
- the next `-SNAPSHOT` bump only once Central resolves.

Every release also updates the compatibility matrix and the benchmark report, states its native
pins, and ships release notes.

### Versioning and compatibility policy

- **Independent versions.** Every module keeps its own version line, and a release raises the
  version of each module it contains, not one project-wide version. Releases are named by phase
  and milestone ("the Phase 9 release"), not by a version number. The first versions are:
  `jmlx-jinja` 0.6.0 (continuing `hfjinja` 0.5.0), `jmlx-ffi` and `jmlx-core` 0.5.0, and 0.1.0
  for every other module.
- **Before 1.0.** A minor version may change public API incompatibly, but its release notes list
  every break. Where it is cheap, a deprecated form is kept for one minor release first. A patch
  version is a compatible fix.
- **Native-pin changes.** A release that contains an MLX upgrade is a minor release of
  `jmlx-native-macos-arm64`, `jmlx-ffi`, `jmlx-core` and `jmlx-models`. Every published module
  that depends on `jmlx-models` (the adapters, the server, the Spring Boot/Quarkus integration and
  `jmlx-groovy`) is re-released against the new `jmlx-models` version in the same release.
- **From 1.0.** Strict semantic versioning.

### When MLX is upgraded

The native pins (`mlx-metal` plus its paired mlx-c commit) are upgraded at fixed points:

- **Phases 1–6:** never upgraded; the pin is `mlx-metal` 0.31.2 / mlx-c `fba4470`.
- **Phase 7:** once, as milestone 7.5, after 7.4 and before 7.6 (Qwen3.8), so 7.6's oracle
  fixtures, goldens and Tier-B evidence are produced once, on the runtime that the first native
  release (8.4) ships. On 2026-10-29 the newest pairing was `mlx-metal` 0.32.4 (with a
  `macosx_26_0` wheel) and mlx-c v0.8.0 (`a26f2d584a14ec619ae8db9a09be03becd47598a`).
- **Phase 8:** no upgrade.
- **Phases 9–14:** milestone 0 of each phase upgrades to the newest pairing that
  `scripts/checkDependencies.zsh` reports, before any other work of the phase. If there is no
  newer pairing, the milestone records that, with the date, and is done. Because the phase's
  release ships the upgrade, the gap to upstream never spans more than one phase.
- **Phase 15:** milestone 15.0 is the last upgrade. The pin is then frozen until 1.0 is
  released. Upgrades resume after 1.0, as minor releases.

An extra upgrade mid-phase is allowed only for a concrete reason: a model family or checkpoint
format that needs a newer runtime (for example the mxfp4, nvfp4 and mxfp8 variants), an upstream
bug fix, or a measured performance gain. It is planned as its own milestone of the current phase.

**Procedure.** Every upgrade is one paired pin upgrade (`mlx-metal` and its mlx-c commit, which
always move together per Decision 9 in `req/plans/initial-plan.md`) in its own PR via
`scripts/updateMlx.zsh`, never bundled with feature work. The PR regenerates the bindings, diffs
the inventory, regenerates the Python MLX oracle fixtures, re-checks golden precision, re-runs the
Tier-B suites, re-records the evidence jmlx recorded on MLX (seeded sampling streams and
pin-gated Tier-B IDs), and adds a migration note. An upgrade may skip intermediate upstream
releases; the inventory diff is then taken against the previous pin, and the migration note
covers the upstream changes of every skipped release. `req/plans/phase7-plan.md` § 7.5 is the
detailed checklist of what the first upgrade must re-verify; later upgrades reuse and update it.

## Template for every follow-on implementation plan

Each milestone plan, `req/plans/phase<N>-<M>-plan.md`, should contain:

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

## Revision history

- **2026-10-04:** milestone 7.0 (Qwen3 text decoders) inserted first in Phase 7 for product
  priority, and the permission to pull focused later-phase array slices forward extended from
  Phase 6 to Phase 7 (now rule 9).
- **2026-10-29:** a product and integration track (P1–P5), Phase 7 milestone 5 (Qwen3.8), a
  release plan and an MLX upgrade policy added.
- **2026-10-30:** made sequential. The product track became Phases 8 and 9, the old Phases 8–13
  became Phases 10–15, the Phase 7 MLX upgrade became milestone 7.5 and Qwen3.8 became 7.6, and
  each phase now states its MLX upgrade and release. Old to new:

  | Before 2026-10-30 | Now |
  | --- | --- |
  | 7.5 Qwen3.8; the MLX upgrade before it | 7.6; 7.5 |
  | P1 step 1 (baseline against `mlx_lm`) | 8.2 |
  | P4 Spring AI and LangChain4j adapters | 8.3 |
  | P1 steps 2–3 (pipelined decode, compiled functions) | 9.1 |
  | P2 prefix cache | 9.2 |
  | P3 HTTP server | 9.3 |
  | P4 Spring Boot starter and Quarkus extension | 9.4 |
  | P5 Groovy API | 9.5 |
  | Phases 8, 9, 10, 11, 12, 13 | Phases 10, 11, 12, 13, 14, 15 |
- **2026-10-31:** the Phase 7.4 decision (`req/plans/phase7-4-decision.md`) scheduled audio
  inference as Phase 11 milestone 11.5 and diffusion inference as Phase 9 milestone 9.6, both
  gated by their own exit gates.
