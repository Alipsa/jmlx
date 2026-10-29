# jmlx Full MLX-Parity Roadmap

## Purpose and definition of done

This is the long-range planning index for jmlx after Phases 1–5. It extends and supersedes the
forward-looking "Project Execution Plan" in `req/project-outline.md`, whose planned phases stop at
Phase 5. It is deliberately a roadmap, not an implementation plan: each milestone below must get
its own plan in `req/plans/` before code is started. Those plans should preserve the decisions and
verification style established by `req/plans/initial-plan.md` and Phases 3–5.

**Priority order:** finish usable local inference first (Phase 6), then complete the
array/runtime surface, transformations and training, and finally the specialist/system
features needed for MLX parity.

**Amended 2026-10-29:** a product and integration track (P1–P5, after Phase 7) adds decode-speed
parity with `mlx_lm`, a prompt prefix cache, an OpenAI/Anthropic-compatible HTTP server,
Spring Boot/Quarkus integration and a Groovy API modeled on MLX's Python API. It may be
interleaved with Phases 8–13. The same amendment adds Phase 7 milestone 5 (hybrid
linear-attention Qwen decoders, Qwen3.8) and a release plan (R1 pure-Java modules, R2 first native
release, 1.0 at full parity).

This ordering is priority, not a strict ban on dependencies: Phase 6 may implement the focused
array, indexing, sort/selection, and random-key slices it needs from later groups. The inventory
records those decisions, and later phases extend the established contract rather than re-deriving it.

**Amended 2026-10-04 (Phase 7):** this permission is extended to Phase 7 for the focused Phase 8
array slices its milestones need, as enumerated in `req/plans/phase7-plan.md`: `abs`, `minimum`,
`clip`, `floor` and `log1p` (Phase 8 milestone 2); `max_axes` and `var_axes` (milestone 3);
`pad`, `as_strided`, `tile` and `repeat_axis` (milestone 4). The Phase 7 milestone 1 layer families
themselves (convolution, pooling, padding, upsampling, normalization variants, containers, dropout,
activations) need no permission because §Phase 7 milestone 1 already names them. Each pulled-forward
slice is entered in `req/mlx-api-inventory.md` where it lands, and Phase 8 and Phase 11 milestone 2
("Complete standard layers required by the pinned MLX `nn` scope") extend this work rather than
re-deriving it.

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

### 6.4.1 — Quantized KV retention and packed attention

**Status (2026-10-01): deferred pending a new attention design.** The step 1 prototype stopped:
four-bit packed attention misses its accuracy criterion because of 4-bit affine rounding of K/V, which
no attention kernel can avoid (`req/phase6-4-benchmark.md`). The unsupported-capability rejection
stays and 6.5 proceeds.

After closing 6.4, implement an opt-in packed-KV attention path that attends to compressed keys
and values without materializing the entire float cache per decode step. The pinned runtime's SDPA
rejects packed K/V, so this requires a separately measured attention implementation (for example,
a fused native kernel or bounded-block streaming attention). Support the 16-dimensional heads used
by the synthetic family fixtures, preserve the existing FULL and SLIDING cache contracts, and leave
float retention as the default. Follow `req/plans/phase6-4-1-plan.md`.

**Exit gate:** the opt-in mode passes accuracy and cache lifecycle tests across supported families,
has lower steady and peak native memory than float retention at a documented long context, and
publishes decode throughput alongside the float baseline. 6.4.1 is time-boxed: 6.5 may start once
the plan's step 1 prototype records a proceed/stop decision (or its decision date passes, which
counts as stop) and, on a proceed, the small float-only cache-accessor refactor (step 1b) has
merged or its deadline (decision date + 14 days) has passed. The remaining 6.4.1 steps do not block
6.5 or the first Central release. A stop keeps the explicit unsupported-capability rejection and is
reopened only by a dated amendment to the plan.

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

**Amended 2026-10-04:** milestone 7.0 is inserted ahead of the listed order for product priority.
The primary product target of Phase 7 is the Qwen family — Qwen3 text decoders first, then
Qwen3-VL — and Qwen3 support is a decoder-only capability of the kind Phase 6.3 established, so it
unblocks the flagship use case (running Qwen3 locally through the existing `TextGenerationModel`
contract) without waiting on the layer work the other milestones require. It also verifies the
per-head QK-normalization capability the later milestones' model families reuse. The existing
milestone numbering below is unchanged; 7.0 slots in before it. See `req/plans/phase7-plan.md` §7.0.

Milestones, to plan in dependency order:

0. **Qwen3 text decoders** *(added 2026-10-04, first for product priority):* map `model_type`
   `qwen3` in `ArchitectureMappings`, add the per-head QK-normalization capability
   (`q_norm`/`k_norm` over `head_dim`, after projection and before RoPE) to `DecoderAttention`,
   honor explicit `head_dim`, extend affine-quantized loading to the family, and land the full
   fixture tier (synthetic logits goldens, tokenizer/chat-template goldens through the real Qwen3
   template, a pinned Qwen3-0.6B Tier-B artifact, and a recorded manual Qwen3-8B 4-bit run).

1. **Core inference modules:** `Sequential`/containers, dropout evaluation semantics, softmax and
   common losses/activations, convolution and transpose convolution, pooling, padding, upsampling,
   normalization variants, and embedding variants.
2. **Encoder and encoder-decoder models:** BERT-style encoders, cross-attention, encoder-decoder
   generation, and classification/token-classification APIs.
3. **Vision and multimodal:** image preprocessing contract, vision transformer/CLIP-style encoders,
   projector layers, multimodal prompt/token conventions, and a tested vision-language model.
4. **Audio/diffusion only after a capability decision:** add STFT/ISTFT or scheduler/UNet/VAE support
   only when the feature inventory and a concrete reference model justify it.
5. **Hybrid linear-attention Qwen decoders (Qwen3.8)** *(added 2026-10-29, product priority, and
   required by release R2 below):* Qwen3.8-27B, like Qwen3.5 and the dense Qwen3.6, uses
   `model_type` `qwen3_5`. Its stack is hybrid: three of every four layers are Gated DeltaNet
   linear attention (a causal short convolution with kernel 4 plus the gated delta-rule
   recurrence). Every fourth layer is full attention with a sigmoid output gate
   (`attn_output_gate`), partial rotary embedding (`partial_rotary_factor` 0.25) and interleaved
   multi-axis RoPE (`mrope_section [11, 11, 10]`). The checkpoint includes a Qwen3-VL-style vision
   tower (`qwen3_5`, patch 16, no DeepStack levels), so the milestone covers both text and
   image+text generation. It reuses 7.3b's vision tower and request lifecycle, and its
   `MultiAxisRope`. The milestone plan must decide:
   - **Partial rotary against the reference.** `MultiAxisRope` has a partial-rotary tail
     passthrough, but its Javadoc calls that tail "a jmlx extension, not reference behaviour",
     and it has never been tested against a reference. Transformers' `qwen3_5` code (main branch,
     2026-10-29) rotates the first `rotary_dim = head_dim * partial_rotary_factor` dimensions and
     concatenates the rest unchanged. The plan verifies the tail against the pinned reference
     version, including the inverse-frequency denominator (64 here, `2 * sum(mrope_section)`).
     7.3b's load guard `2 * sum(mrope_section) == head_dim` is specific to Qwen3-VL. The
     `qwen3_5` path checks `2 * sum(mrope_section) == head_dim * partial_rotary_factor` instead.
   - **Hugging Face and MLX-converted checkpoint layouts.** mlx-community artifacts come from
     two converters, and the model card names which one was used. The image+text artifacts,
     including `mlx-community/Qwen3.8-27B-4bit` (mlx-vlm 0.6.8), come from mlx-vlm. Text-only
     ones come from `mlx_lm`. Both converters' `qwen3_5` sanitize steps (main-branch sources,
     verified 2026-10-29) make the same four text-side changes:
     - The key prefixes change (`model.language_model.` versus `language_model.model.`).
     - The `conv1d` axes move (`[C, 1, K]` versus `[C, K, 1]`).
     - The MTP tensors are dropped.
     - `+1.0` is added to the zero-centered RMSNorm weights (`input_layernorm`,
       `post_attention_layernorm`, the final norm, `q_norm` and `k_norm`; not
       `linear_attn.norm`), which the HF model applies as `x * (1 + w)`.

     The two converters differ on the vision side:
     - `mlx_lm` discards every `vision_tower.*` and `model.visual.*` key, so its output is
       text-only.
     - mlx-vlm keeps the vision tower. It renames `model.visual.` to `vision_tower.` and
       transposes the patch embedding from `[O, C, T, H, W]` to `[O, T, H, W, C]` (the 27B has
       `vision_tower.patch_embed.proj.weight` `[1152, 2, 16, 16, 3]`).

     The descriptor must detect the layout and load all three:
     - HF;
     - mlx-vlm;
     - `mlx_lm`, which has no vision weights, so it loads as a text-only model whose metadata
       reports TEXT and rejects image requests.

     The norm offset must be applied exactly once. mlx-vlm drops MTP tensors *before* it checks
     whether to apply the offset. It then adds the offset to a norm weight when the checkpoint
     still has an unconverted `conv1d`, or when that weight's original key does not start with
     `language_model.`. So a key with the HF prefix always gets the offset. The key prefix
     (`model.language_model.` versus `language_model.model.`) is therefore the signal for
     whether the offset is already in the stored weights.

     The vision side reuses 7.3b's tower, but not its loader routing:
     - 7.3b's float path (`Qwen3VlVisionTower.visionWeights`) maps only the HF `model.visual.`
       and `model.language_model.` keys. A float mlx-vlm artifact (for example `-bf16`) would
       therefore be rejected for missing keys; this is inferred from the code, not run.
     - 7.3b's affine path expects a channels-last patch embedding and rejects any other layout
       (`validateVisionShapes`); it does not detect or convert the axis order.

     The `qwen3_5` loader must choose the prefix mapping from the keys and the patch-embedding
     axis order from the tensor shape, independently of quantization.
   - **A recurrent-state cache beside the KV cache.** It holds the convolution state and the
     delta-rule state. It cannot be truncated or slid, so its interaction with `KVCachePolicy`,
     `BatchGenerationScheduler` cohorts and P2's prefix cache (whole-prefix reuse only) must be
     defined.
   - **The delta-rule implementation.** Start with a chunked prefill and per-token decode built
     from existing ops. Pull forward custom Metal kernels (Phase 12 milestone 3, the 13
     `mlx_fast_metal_kernel_*` bindings) only if a measured benchmark shows the ops path is
     inadequate.
   - **The reference environment.** The Qwen3.8 config records Transformers 5.8.0.dev0, while
     `tools/hf-reference` pins 4.57.6. The milestone needs its own locked reference environment,
     or a reviewed upgrade of the shared one.

   Verification:
   - **Tier-A:** a synthetic hybrid checkpoint against strict float32 Hugging Face logits. Its
     config must cover what the Tier-B artifact does not and Qwen3.8-27B uses: more linear value
     heads than key heads (27B: 48 versus 16; 0.8B: 16 versus 16), an untied output head (27B is
     untied, 0.8B is tied) and `output_gate_type`. The Qwen3.8 config sets `output_gate_type` to
     `swish`, but Transformers' `qwen3_5` code on main does not read the key, so the plan
     records its effect in the pinned reference before relying on it.
   - **Tier-B:** `Qwen/Qwen3.5-0.8B` (Apache-2.0, same `model_type`), in the HF layout.
   - **Manual run:** a recorded `mlx-community/Qwen3.8-27B-4bit` run (Apache-2.0), in the
     mlx-vlm layout, with text and image+text greedy IDs matching mlx-vlm. Its quantization
     (affine 4-bit, group size 64) is a format jmlx already loads. Its tensor layout is not (see
     above).
   - **Converted layouts:** Tier-A fixtures converted from the HF-layout synthetic checkpoint
     cover `mlx_lm` (text-only) and float mlx-vlm (the case the shape-based detection exists
     for). The conversion runs the real converters, not a Java-side conversion, so the layout is
     tested independently. They run in a hash-locked environment with recorded provenance, like
     the other reference tools. `tools/qwen-vl-reference` already locks `mlx-vlm` 0.3.9 and
     `mlx-lm` 0.29.1 for 7.3b, but those versions predate `qwen3_5`. The plan either re-locks
     that tool at converter versions that support `qwen3_5` (the 27B was converted with mlx-vlm
     0.6.8) or adds a separate tool, and records the converter versions with each fixture.

   **Not included (follow-ups):**
   - the shared-expert MoE variants (Qwen3.5/3.6-35B-A3B);
   - Qwen3.8-Flash-Next (`qwen4_exp`, 512-expert MoE, about 180B parameters, non-Apache
     license);
   - multi-token-prediction heads (ignored at load; speculative decoding is separate);
   - the mxfp4, nvfp4 and mxfp8 checkpoint variants.

   `req/plans/phase7-plan.md` carries the matching amendment: the 7.5 and MLX-upgrade rows in its
   delivery order, its 7.5 section and its deferrals. The sub-plan `req/plans/phase7-5-plan.md`
   is written and reviewed before code.

**Phase 7 done:** supported non-text modalities are stated explicitly in the compatibility matrix;
the `nn` API covers their required common layers without model packages bypassing core facades.

## Release plan

**Added 2026-10-29.** Releases are tied to exit gates, not to a calendar. Every release follows
the existing manual procedure:
- each module's `release.sh` in dependency order;
- `jmlx-jinja/req/release-checklist.md` for jinja;
- `tools/release-smoke/run.sh candidate` before publishing and `run.sh central` after;
- the next `-SNAPSHOT` bump only once Central resolves.

Every release also updates the compatibility matrix and the benchmark report, states its native
pins, and ships release notes.

### Versioning and compatibility policy

- **Independent versions.** Modules keep independent version lines.
- **Before 1.0.** A minor version may change public API incompatibly, but its release notes list
  every break. Where it is cheap, a deprecated form is kept for one minor release first. A patch
  version is a compatible fix. This supersedes the additive-only rule in
  `req/plans/phase7-plan.md` (Baseline and constraints).
- **Native-pin upgrades.** From R2 onward, a native-pin upgrade follows Phase 13 milestone 4's
  procedure (regenerate, diff the inventory, migration note). It ships as a minor release of
  `jmlx-native-macos-arm64`, `jmlx-ffi`, `jmlx-core` and `jmlx-models`. Every published module
  that depends on `jmlx-models` (the server, the framework adapters and `jmlx-groovy`) is
  re-released against the new `jmlx-models` version in the same release.
- **From 1.0.** Strict semantic versioning.

### MLX upgrade policy

The native pins (`mlx-metal` plus its paired mlx-c commit) are upgraded neither with every release
nor only after full parity. An upgrade is costly: regenerated bindings, an inventory diff,
regenerated Python MLX oracle fixtures, re-checked golden precision, re-run Tier-B suites, and
re-recorded evidence that jmlx recorded on MLX (seeded sampling streams and pin-gated Tier-B IDs).
Each skipped MLX release also adds to the binding and behavior drift a later upgrade must absorb,
and newer model families and checkpoint formats (for example the mxfp4, nvfp4 and mxfp8 variants)
need a recent runtime.

An upgrade may skip intermediate upstream releases (see Phase 13 milestone 4). In that case:
- the inventory diff is taken against the previous pin;
- the migration note covers the upstream changes of every skipped release.

1. **Upgrade before Phase 7 milestone 5.** Upgrade once 7.3c has closed (delivered, or deferred
   with a date) and before Qwen3.8 work starts. Move to the newest paired release that
   `scripts/checkDependencies.zsh` reports. On 2026-10-29 that was `mlx-metal` 0.32.4 (with a
   `macosx_26_0` wheel) and mlx-c v0.8.0 (`a26f2d584a14ec619ae8db9a09be03becd47598a`). This
   skips `mlx-metal` 0.32.0–0.32.3 and mlx-c's v0.7.x tags. Its oracle fixtures, goldens and
   Tier-B evidence are then produced once, on the runtime that R2 ships.
2. **Afterwards, upgrade when there is a concrete reason.** Reasons include a model family or
   checkpoint format that needs a newer runtime, an upstream bug fix, or a measured performance
   gain. In any case, upgrade at least once per completed roadmap phase, so the gap never spans
   more than one phase. Each upgrade is its own PR via `scripts/updateMlx.zsh`, never bundled with
   feature work, and is released as described under Native-pin upgrades.
3. **Freeze for certification.** The pin is frozen from the start of Phase 13's parity
   certification until 1.0 is released against it. Upgrades resume after 1.0, as minor releases.

### R1 — Pure-Java modules

`jmlx-jinja` 0.6.0 (continuing `hfjinja` 0.5.0 on Central), `jmlx-tokenizer` 0.1.0 and
`jmlx-vision` 0.1.0. These modules have no native dependency, target Java 21 and are useful on
their own. Released as soon as the release-smoke tooling can verify a partial module set:
`candidate` and `central` modes currently assume all seven modules.

### R2 — First native release (0.x preview)

`jmlx-native-macos-arm64` 0.1.0, `jmlx-ffi` 0.5.0, `jmlx-core` 0.5.0 and `jmlx-models` 0.1.0, plus
new versions of any R1 module that changed. Gate:

1. Phase 7 is done, including milestone 5 (Qwen3.8 text and image+text).
2. P1 step 1's benchmark against `mlx_lm` is published, so decode speed is stated honestly even
   before any optimization.
3. P4's Spring AI and LangChain4j adapters are published with this release. Their own
   adapter-level tests must pass against the release-smoke repository: chat, streaming and
   embeddings through each library's interfaces, with native memory released on close. The
   rest of P4's exit gate (the Spring Boot starter, the Quarkus extension and the two sample
   applications) is not required for R2 and may follow in R3+.

### R3+ — Further 0.x releases

A release follows each completed P-milestone, and each completed Phase 8–12 milestone that adds
public API. Small milestones may be batched into one release. New P-track modules join the release
order at their first release.

### 1.0 — Full parity

The native chain (`jmlx-native-macos-arm64`, `jmlx-ffi`, `jmlx-core`, `jmlx-models`) reaches 1.0
at Phase 13's full-parity release gate. Pure-Java and P-track modules carry no parity claim. They
may reach 1.0 earlier on their own stability criteria, but not before any module they depend on.

## Product and integration track (P1–P5, alongside Phases 8–13)

**Added 2026-10-29.** A survey of other JVM front-ends on Apple Silicon (mlx-scala on MLX through
FFM; lightmetal on llama.cpp; jitLLM on TornadoVM; Jlama) showed gaps in jmlx's inference
*product*, not in its MLX parity: decode speed relative to `mlx_lm`, reuse of prompt prefixes
between requests, a network protocol that existing clients speak, framework integration, and a
concise scripting API. This track closes those gaps. Its milestones are numbered P1–P5 so that
Phases 8–13 and every plan referring to them keep their numbers. The milestones may be
interleaved with Phases 8–13 in any order consistent with the dependencies listed below. Each
milestone still needs its own plan under `req/plans/` (`product-p<N>-plan.md`) and is bound by
the roadmap-wide rules.

Native symbols pulled forward from later phases are permitted under the same conditions as the
2026-10-04 Phase 7 amendment. Each slice is entered in `req/mlx-api-inventory.md` where it
lands. The later phase then extends that contract rather than re-deriving it.

New published modules (the server, the framework adapters and the Groovy API) each start their
own `0.1.0-SNAPSHOT` line. They join `verifyReleaseScriptsMatch`, the release order and the
release smoke. None of them is published before `jmlx-models` itself.

### P1 — Decode performance parity with `mlx_lm`

**Pulled forward:**
- `mlx_async_eval`, from Phase 12 milestone 1 (asynchronous evaluation);
- the compile-mode controls in `mlx/c/compile.h` (`mlx_enable_compile`, `mlx_disable_compile`
  and `mlx_set_compile_mode`), from Phase 12 milestone 1 (compile mode);
- `mlx_compile` itself, from Phase 12 milestone 2.

The `mlx_detail_compile*` symbols remain under Phase 10 milestone 3's internal/unstable decision
and are not pulled forward.

1. **Baseline first.** Benchmark jmlx against `mlx_lm` on byte-identical weights (one dense
   family, one MoE family and one 4-bit checkpoint). Measure prefill tokens/s, decode tokens/s,
   time to first token and peak memory, using the existing benchmark harness and the
   reproducibility tuple from `req/phase6-4-benchmark.md`. Record per-token host overhead,
   including the native call count and FFM cost per decode step. No optimization starts before
   this baseline is recorded.
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
step 1. Every existing golden, precision task and cache-equivalence suite passes unchanged in both
TF32 modes, with tolerances unmodified. Each optimization is opt-out-able, and its effect is
recorded separately in the benchmark.

### P2 — Prompt prefix cache

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
  truncated and are reused whole or not at all.
- **Scheduler integration:** integration with `BatchGenerationScheduler` cohorts.
- **Persistence (optional, second step):** safetensors through `MLXIO`, with a versioned header
  that rejects a cache from a different model, pin or configuration.

**Exit gate:** logits with a cache hit match a full recompute within the existing
cache-equivalence tolerances, for text and image prefixes. Memory stays within the configured
budget under a soak test. A multi-turn chat benchmark records the time-to-first-token reduction.

### P3 — OpenAI- and Anthropic-compatible HTTP server

A published `jmlx-server` module, a deliberate amendment of Phase 6.5's "not an HTTP server"
scope. It serves models through `BatchGenerationScheduler`:

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
out. This belongs in `jmlx-models`, so that P3 and P4 share a single implementation.

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
in-process generation is benchmarked. P2's cache, when present, is used for multi-turn requests.

### P4 — Spring Boot and Quarkus integration

Published adapters that make a jmlx model an injectable, lifecycle-managed component:

- **Spring Boot starter:** auto-configuration and typed configuration properties for the model
  path, cache policy and scheduler limits; start and stop tied to the application context;
  Actuator health and Micrometer metrics (tokens/s, queue depth, native memory).
- **Spring AI:** `ChatModel`, `StreamingChatModel` and `EmbeddingModel` implementations.
- **LangChain4j:** `ChatModel`, `StreamingChatModel` and `EmbeddingModel` implementations, which
  also give Quarkus users the quarkus-langchain4j path.
- **Quarkus extension:** configuration, CDI producers and dev-mode behavior. The extension runs in
  JVM mode only; native-image support is out of scope until GraalVM's FFM support for this native
  stack is probed.

Callers may run on virtual threads. The scheduler's platform-thread worker keeps sole ownership
of MLX, so the threading rule holds without framework cooperation. Current Spring Boot, Spring AI,
LangChain4j and Quarkus releases support Java 25; the plan states the supported version matrix.
The shared request/response mapping should
be factored with P3's protocol layer rather than duplicated.

**Exit gate:** a sample Spring Boot application and a sample Quarkus application, each built
against the published-shape smoke repository, load a model, stream a chat completion and shut
down with native memory released. Health and metrics endpoints report correctly.

### P5 — Groovy API equivalent to MLX's Python API

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

**Staging:** coverage is staged with the Java surface. The first slice covers today's array,
`nn` inference and grad API. Indexing, linalg/FFT and optimizers extend it as Phases 8–11 deliver
their Java equivalents. A Python name with no Java equivalent yet is listed as pending, never
emulated. Current Groovy releases support Java 25; the plan states the supported Groovy version.

**Exit gate:** a generated table maps each covered Python name to its Groovy form. Representative
examples from the MLX Python documentation, ported to Groovy, match the Python MLX oracle's
output.

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
   and record that decision in the inventory. This decision does not cover the public
   `mlx_compile`, `mlx_enable_compile`, `mlx_disable_compile` and `mlx_set_compile_mode` in
   `mlx/c/compile.h`; product milestone P1 may already have wrapped those. Only if accepted,
   add vmap/custom VJP/custom JVP/custom function wrappers after callback-upcall lifetime,
   exception translation, and thread/arena behavior are proven by native probes.
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
2. Complete the public `mlx_compile` API (which P1 may already have wrapped for its hot
   functions) with graph export/import and cache lifetime/error semantics. This part does not
   wait for Phase 10's decision. The `mlx_detail_compile_*` capabilities do wait for it: after
   that decision, expose only the accepted ones. An `internal/unstable-upstream` decision means
   documenting, not wrapping, those generated symbols.
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
4. Establish the MLX-upgrade procedure: one paired pin upgrade (`mlx-metal` and its mlx-c
   commit, which always move together per Decision 9 in `req/plans/initial-plan.md`) per PR.
   Regenerate bindings, diff the inventory, add compatibility tests, and publish a migration
   note before claiming parity for that release. An upgrade may skip intermediate upstream
   releases. If it does, the inventory diff is taken against the previous pin and the migration
   note covers every skipped release (see § Release plan, MLX upgrade policy).

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
