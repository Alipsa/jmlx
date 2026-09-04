# jmlx Phase 6 — Complete Local Text-LLM Inference

**Roadmap:** `req/full-roadmap.md` §Phase 6

## Goal

Turn `jmlx-models`' current Llama/Qwen greedy-generation proof into the first public jmlx release:
a Java application can load a documented local text model, render a chat prompt, stream reproducible
sampled output, run bounded multi-request generation, and close every native resource safely.

This is deliberately not a general MLX-parity project, an HTTP server, or multimodal inference.
Those belong to later roadmap phases. The supported platform remains macOS/Apple Silicon with the
repository's pinned MLX runtime and Java 25.

## Baseline and constraints

The existing model implementation is intentionally narrow:

- `DecoderModel.generate` owns a child scope and one `KVCache` per layer, then greedily picks
  `argmax` from the final logits.
- `LlamaModel` and `QwenModel` load safetensors, including indexed shards; `DecoderConfig` rejects
  RoPE scaling and sliding-window attention.
- `HfTokenizer` is a pure-Java byte-level BPE implementation with chat-template rendering.
- Native CI already bootstraps the pinned runtime on hosted ARM64 macOS; pure-Java tokenizer checks
  run separately on Ubuntu.

Do not extend `DecoderModel.generate` piecemeal into the public serving API. Preserve it as a
compatibility adapter during migration, then deprecate it only after the new API demonstrates the
same greedy output against the same fixture.

## Public-contract decisions

These decisions are Phase 6's first implementation task. They prevent model-specific branches from
becoming the public API.

1. Keep the public inference API in `jmlx-models`; do not add an HTTP server or a new module.
2. A loaded model is bound to a caller-owned `MLXScope`. A generation owns a child scope and releases
   its activations and caches when it completes, is cancelled, or fails. Events never expose an
   `MLXArray` or any native-owned value.
3. Add the following stable Java-facing types in `se.alipsa.jmlx.models`:

   - `TextGenerationModel extends AutoCloseable` — model metadata, `generate`, and streaming
     generation entry points. Closing it closes only resources it owns; the loading convenience API
     may create an owning scope, while the existing scope-taking loader remains available for
     advanced callers.
   - `GenerationConfig` — immutable validated configuration: max new tokens, seed, temperature,
     top-k, top-p, min-p, repetition/frequency/presence penalties, EOS/stop token IDs, optional
     log probabilities, and cache policy. Its defaults reproduce today's greedy behavior.
   - `GenerationRequest` — prompt token IDs or text plus an explicit special-token policy, config,
     and cancellation token. Chat-template rendering stays explicit rather than being silently
     inferred from arbitrary prompt text.
   - `GenerationEvent` — token ID, decoded text delta, optional log probability, and finish reason.
     A callback-based streaming method is the initial API; it runs callbacks synchronously on the
     calling thread. Reactive Streams and asynchronous executors are out of scope until an
     application needs them.
   - `GenerationResult` and `FinishReason` — complete token sequence, generated token count, final
     reason (`EOS`, `STOP_TOKEN`, `MAX_TOKENS`, `CANCELLED`), and optional collected log
     probabilities.

4. Define reproducibility precisely: greedy results must be bit-for-bit identical for one jmlx
   binary, pinned runtime, device, batch configuration, model, and request. Seeded sampling must
   reproduce under that same tuple. Do not promise equality across batch shapes or MLX/runtime
   upgrades.
5. Fail early and specifically for an unsupported model capability, tokenizer feature, checkpoint
   tensor, or generation configuration. Never partially load a checkpoint or silently ignore a
   configuration field.

## Delivery order

Each numbered milestone is a separately reviewable PR series. Milestones 6.0 and 6.0b may proceed
in parallel; 6.1 starts after the public contract is accepted. Later milestones build in order.

| Milestone | Primary modules | Exit evidence |
| --- | --- | --- |
| 6.0 Contract and M3 record | `jmlx-models`, docs | Llama/Qwen use the new API without a shim |
| 6.0b Inventory and test infrastructure | `buildSrc`, `jmlx-ffi`, CI, docs | inventory and native-call guard run in CI |
| 6.1 Generation and sampling | `jmlx-core`, `jmlx-models` | reproducible greedy and seeded sampling |
| 6.2 Tokenizer compatibility | `jmlx-tokenizer`, `jmlx-models` | every supported model has reference goldens |
| 6.3 Architectures/checkpoints | `jmlx-core`, `jmlx-models` | capability fixtures and real-artifact coverage |
| 6.4 Cache/performance | `jmlx-core`, `jmlx-models`, examples | bounded-memory benchmark evidence |
| 6.5 Batching/release hardening | `jmlx-models`, examples, CI, docs | release candidate smoke test and matrix |

## 6.0 — Contract, compatibility matrix, and Phase 5 M3 record

1. Add `req/plans/phase5-m3-retrospective.md`. Record the delivered loading, weight mapping,
   generation loop, scope ownership, checkpoint assumptions, and tests. It must name what was
   deliberately omitted: sampling, streaming, batching, RoPE scaling, broad architectures, and
   tokenizer families beyond byte-level BPE.
2. Add `req/phase6-compatibility.md` as the single human-readable support matrix. Initially it must
   distinguish `implemented`, `verified-with-synthetic-fixture`, `verified-with-real-artifact`, and
   `planned`; include model architecture, tokenizer family, chat-template status, checkpoint form,
   quantization, RoPE/scaling, context/cache policy, and license/access notes.
3. Add the public types described above, with complete resource ownership and threading javadocs.
   Implement a model factory that dispatches on `model_type`; keep `LlamaModel.load` and
   `QwenModel.load` as compatibility entry points delegating to the common loader.
4. Move current greedy generation behind `GenerationConfig.greedyDefaults()` and add a streaming
   adapter that emits the same token sequence and one terminal event. Preserve prompt token IDs in
   results and decode only generated IDs for text deltas.
5. Add tiny native tests for: Llama/Qwen common loader dispatch, greedy output equivalence with the
   old API, event order, EOS/max-token stopping, callback-thrown exception cleanup, cancellation
   before prefill and between decode steps, and scope cleanup after every terminal path.
6. Update `jmlx-models/README.md` with the API, limitations, and compatibility-matrix link.

**Gate:** both existing architectures pass through the common API with no output change; the matrix
and M3 retrospective are reviewed with the code.

## 6.0b — Native inventory, oracle, and fixtures

1. Generate `req/mlx-api-inventory.md` from the committed `mlx_h` bindings. It records both native
   pins from `scripts/bootstrap-native.sh`, native symbol/group, public facade, status, and test
   location. Generated output must be deterministic and checked in.
2. Add a `buildSrc` verification task that scans handwritten Java sources for generated FFI downcalls,
   generated layout/accessor types, and generated upcall interfaces. It fails unless every use has
   a matching inventory entry. Exclude generated bindings and narrowly document each intentional
   test-only exception.
3. Establish `tools/mlx-oracle/` as a pinned Python MLX environment. Its lock/provenance document
   states whether its MLX version matches the Java runtime pin, how it is installed on macOS ARM64,
   and which generated goldens are committed. Oracle failure must never be interpreted as a Java
   correctness result without reporting versions.
4. Add a Gradle task for small oracle fixtures. Run it in the existing `native` CI job only after
   bootstrap. Keep it deterministic, small, and credential-free.
5. Define fixture tiers:

   - Tier A: tiny configs, randomly initialized safetensors, tokenizer inputs, prompt/token goldens,
     and capability fixtures committed under module test resources. They run for every relevant PR.
   - Tier B: named real Hugging Face artifacts, hashes/revisions, licensing/access requirements, and
     expected output metadata. Run only by `workflow_dispatch` or a scheduled workflow. Never put
     multi-GB downloads or private tokens in required PR checks.

**Gate:** inventory generation and call-site verification are green in CI; a Tier-A model and
tokenizer fixture runs natively with no external credentials; the oracle's exact provenance is
recorded.

## 6.1 — Sampling and streaming generation

1. Before selecting an algorithm, probe the exact pinned mlx-c declarations for `mlx_topk`,
   `mlx_sort`/`mlx_argsort`, `mlx_partition`, categorical sampling, and explicit random-key/split
   operations. Record ownership, shape, dtype, lazy-evaluation, tie-breaking, and RNG behavior in
   the inventory and an implementation finding. If explicit key/split support differs from the
   binding expectation, stop and revise the API contract rather than falling back to global RNG.
2. Add only the core facade operations selected by that probe, probably `MLXOps.topk`/sorting and
   `MLXRandom` explicit key, split, and categorical methods. Every native temporary follows the
   existing `MLXScope`/`NativeOps` cleanup pattern; no model code reaches `ffi`.
3. Implement logits processing as a pure, ordered pipeline with tests for each stage: repetition,
   frequency, and presence penalties; temperature; top-k; top-p; min-p; and final categorical draw.
   Validate NaN/infinity logits, zero temperature, invalid probability ranges, empty candidates,
   duplicate stop IDs, and out-of-vocabulary IDs before native execution.
4. Track one explicit RNG state per request. Splitting a batch or cancelling one request must not
   perturb the random sequence of another request. Greedy mode must not consume a random key.
5. Define `logprobs` precisely: report the selected token's log probability from the post-policy
   distribution, and reject requests for unsupported alternatives rather than returning ambiguous
   values.

**Tests:** hand-calculated logits, Python-oracle distributions for fixed seeds, same-seed repeat
tests, different-seed divergence tests that avoid flaky statistical claims, stop handling,
cancellation, and Llama/Qwen tiny-checkpoint integration tests.

**Gate:** greedy output is unchanged; seeded output is reproducible for the stated tuple; all policy
corner cases are specified and tested.

## 6.2 — Tokenizer and prompt compatibility

1. Extend tokenizer parsing by tokenizer family, not by model-name conditionals. Introduce an
   internal `TokenizerModel` strategy and retain BPE behavior unchanged behind its current strategy.
2. Add SentencePiece/Unigram and WordPiece only after committing small reference artifacts and
   HF-reference encode/decode goldens. Define normalization, pre-tokenization, unknown tokens,
   byte fallback, offsets, truncation, padding, added/special tokens, and decoder behavior per
   supported strategy.
3. Make special-token policy explicit in `GenerationRequest`; use the model matrix to define each
   artifact's BOS/EOS, assistant-generation prompt, and chat-template behavior. Do not add special
   tokens twice when a rendered template already contains them.
4. Keep tokenization pure Java. Any native/tokenizers-cpp alternative requires a separately reviewed
   compatibility and performance probe and cannot silently replace the Java implementation.

**Gate:** every matrix row can load its tokenizer, render its documented chat prompt, and match
committed HF-reference token and text goldens.

## 6.3 — Architecture and checkpoint capability matrix

1. Replace Llama/Qwen branches in `DecoderModel` with explicit capability descriptors: attention
   layout/bias, normalization, MLP activation/layout, tied output, RoPE variant, sliding-window
   masking, and checkpoint-name mapping. A descriptor validates every required and forbidden tensor
   before constructing any model layer.
2. Add RoPE scaling before accepting configs that declare it. Implement and test linear,
   dynamic-NTK, Llama 3, and YaRN forms required by the approved matrix. Probe numerical agreement
   with the oracle at multiple positions; do not treat a parsed config field as support.
3. Add families one capability at a time: Mistral (sliding-window), Gemma, Phi, then one MoE family
   such as Mixtral. Each family requires a written mapping table from `config.json`/tensor names to
   capabilities, a tiny safetensors fixture, golden prefill and decode token IDs, and one Tier-B
   accessible licensed artifact.
4. Extend checkpoint validation for common indexed safetensor/shard forms and chosen GGUF
   quantizations. Error messages must identify the config field or tensor key and the missing
   capability. No best-effort loading.

**Gate:** every implemented capability has a Tier-A fixture; every claimed family has Tier-B evidence;
unsupported artifacts fail before a partial model is returned.

## 6.4 — Prefill/decode performance and cache lifecycle

1. Split generation internally into prefill and single-token decode paths. Avoid rebuilding prompt
   arrays or recomputing cached keys/values during decode.
2. Extend `KVCache` with explicit capacity, reset, fork, reorder, and sliding-window eviction
   semantics. Define whether fork/reorder copies or shares native arrays and test that closing one
   path cannot invalidate another.
3. Implement batch-safe positions and masks, then cache quantization only after an MLX capability
   probe establishes representation, accuracy, and ownership semantics.
4. Add `jmlx-examples` benchmark harnesses for cold load, prefill, one-token decode, sustained
   generation, and peak native memory. The report records jmlx commit, Java, macOS/device, both
   native pins, model revision/hash, config, batch size, context length, warm-up, and samples.
5. Add bounded-memory native tests for repeated generations, cache reset, long context, and
   sliding-window eviction. Use deterministic tiny fixtures for required CI; Tier-B benchmarks are
   manual/scheduled.

**Gate:** no unbounded native memory in repeated generation; sliding-window outputs/masks are
correct; reproducible benchmark data exists for every supported family.

## 6.5 — Batching, release evidence, and first Central release

1. Add a bounded in-process batch scheduler, not an HTTP server. It accepts prompt requests, limits
   queue depth and batch size, groups compatible decode work, handles unequal completion lengths,
   provides back-pressure, and isolates cancellation and RNG state per request.
2. Define scheduler threading explicitly. Native calls remain confined to its worker/owned scopes;
   callbacks receive immutable Java events and must not block the worker indefinitely. Start with a
   documented single-worker implementation and add concurrency only with profiling evidence.
3. Add package smoke tests that resolve the published module graph in a clean Gradle home, include
   `jmlx-native-macos-arm64`, load a Tier-A checkpoint, render a chat prompt, and stream a short
   deterministic completion.
4. Publish `jmlx-models` examples, model-cache/download guidance (or a separately scoped resolver),
   the compatibility report, benchmark method/results, licenses/notices, and explicit unsupported
   features. Do not imply support for arbitrary Hugging Face models.
5. Run the documented release order and release verification on real macOS ARM64: `jmlx-jinja` →
   `jmlx-tokenizer`; `jmlx-native-macos-arm64` → `jmlx-ffi` → `jmlx-core` → `jmlx-models`. Release
   only after all module versions and published dependencies are non-SNAPSHOT and the clean-consumer
   smoke test uses Central-resolved artifacts.

**Phase gate:** a documented Java program performs chat prompt rendering, reproducible streamed
sampling, bounded multi-request generation, and deterministic cleanup against supported models.

## Cross-cutting verification

- Required PR checks: formatting, Checkstyle, unit tests, generated-inventory check, and Tier-A
  native model/tokenizer tests on the existing macOS ARM64 job.
- Required before each model-family claim: config/tensor mapping review, tokenizer/template golden,
  synthetic prefill/decode golden, real-artifact result, and compatibility-matrix update.
- Required before release: clean published-artifact smoke test, native CI green, release POM/license
  checks, benchmark evidence, and review of every `planned`/`unsupported` matrix row.

## Explicit deferrals

- HTTP serving, network transport, authentication, and distributed scheduling.
- Vision, audio, diffusion, encoder-decoder, and embedding-model support (Phase 7).
- General array, linalg, FFT, random, transformation, optimizer, and runtime parity (Phases 8–13).
- A claim of output identity across changed model revisions, MLX pins, devices, or batch shapes.
