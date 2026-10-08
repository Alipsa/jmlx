# jmlx Phase 7 — Inference Breadth and Conventional Neural-Network Execution

**Roadmap:** `req/full-roadmap.md` §Phase 7, as amended by step 0 below

## Goal

Extend jmlx from decoder-only text generation to the Qwen3 text family, then to text encoders,
encoder-decoder generation and image+text inference. Add the conventional `nn` layers those
workloads need to `jmlx-core`, behind its existing facades. The primary product target is the Qwen
family: Qwen3 text decoders (7.0) and Qwen3-VL (7.3b).

**Phase 7 done (roadmap gate):** supported non-text modalities are stated explicitly in the
compatibility matrix; the `nn` API covers their required common layers without model packages
bypassing core facades.

Non-goals: training (dropout in training mode, optimizers, loss gradients: Phase 11), general
array/indexing parity beyond the focused slices named below (Phase 8), FFT/linalg (Phase 9), video
input, HTTP serving, and audio/diffusion implementation (7.4 only decides).

## Baseline and constraints

- `jmlx-models` loads six float/affine-quantized decoder families through `ArchitectureDescriptor`
  → `ArchitectureMappings.tensorPlan` → `DecoderAssembler.assemble`. It dispatches on `model_type`
  in `TextGenerationModels`, and the public API is `TextGenerationModel`, `GenerationRequest`,
  `GenerationConfig`, `GenerationResult`/`GenerationEvent` and `BatchGenerationScheduler`.
- `DecoderModel`'s public `forward` overloads accept only token IDs, and no multi-axis position
  IDs exist anywhere. Internally there is a package-private `DecoderModel.EmbeddingHook`
  (`MLXArray apply(MLXArray embedded)`): a test seam that transforms activations right after the
  embedding lookup in `stepLogits`/`normalizedHiddenStatesBatch`. It is wired through
  `BatchGenerationScheduler.Hooks` and is `NONE` in production.
- `se.alipsa.jmlx.nn` has `Module`, `Linear`/`QuantizedLinear`, `Embedding`/`QuantizedEmbedding`
  (`EmbeddingLayer`), `RMSNorm`, `LayerNorm`, `SiLU`, `GELU`, attention and decoder blocks.
  `Activation` has `SILU`, `GELU` (exact, erf) and `GELU_TANH`. `ArchitectureMappings` maps the
  `gelu`/`gelu_new`/`gelu_pytorch_tanh` strings only in Gemma's `hidden_activation` branch.
  `DecoderAttention` and `MultiHeadAttention` hard-code the attention scale as `1/sqrt(headDim)`.
  `Module` has no train/eval state. There are no containers, convolutions, pooling, padding or
  upsampling.
- Inventory (`req/mlx-api-inventory.md`): `mlx_conv1d/2d/3d`, `mlx_conv_transpose1d/2d/3d`,
  `mlx_conv_general`, `mlx_pad`, `mlx_pad_symmetric`, `mlx_as_strided`, `mlx_max_axes`
  (`mlx_max_axis` and `mlx_max` also exist), `mlx_minimum`, `mlx_abs`, `mlx_clip`, `mlx_floor`,
  `mlx_log1p`, `mlx_var_axes`, `mlx_repeat_axis`, `mlx_masked_scatter`, `mlx_softmax_axes` and
  `mlx_tile` are all `unplanned`. `softmaxAxis`, `mean`, `tanh`, `erf`, `sigmoid`, `maximum`,
  `where`, `take*`, `concatenate` and `stack` already exist. The pinned `libmlx.dylib` carries only
  the `constant` and `edge` pad-mode strings; `mlx_pad_symmetric` takes a single `pad_width` for
  both sides of every axis; it is not a mode. Evidence so far: a `strings` audit of the pinned
  `libmlx.dylib` (adjacent `constant`, `edge`, `Invalid padding mode (`, with no `reflect` or
  `symmetric`) and the `mlx_pad`/`mlx_pad_symmetric` declarations in the staged
  `mlx/c/ops.h`. The 7.1 probe will confirm the accepted modes at runtime.
- The roadmap's permission to pull focused later-phase array slices forward is scoped to Phase 6
  only (`full-roadmap.md` §Purpose). 7.1's array ops are Phase 8 scope: `abs`, `minimum`,
  `clip`, `floor` and `log1p` are milestone 2; `max_axes` and `var_axes` milestone 3; `pad`,
  `as_strided`, `tile` and `repeat_axis` milestone 4. The layers themselves (convolution and
  transpose convolution, pooling, padding, upsampling, normalization variants) are already
  Phase 7 milestone 1 scope (`full-roadmap.md` §Phase 7), so they need no permission. Only the
  Phase 8 array slices they depend on are pulled forward. The `mlx_conv*` bindings (`MLXConv`)
  appear in no Phase 8 milestone. They are part of the convolution layers' Phase 7 scope, and
  Phase 11 milestone 2 ("Complete standard layers required by the pinned MLX `nn` scope", which
  also covers `Upsample`) later extends them rather than re-deriving them.
- `jmlx-tokenizer` already supports WordPiece (BERT) and SentencePiece Unigram (T5).
  `HfTokenizer.renderChat` rejects non-text message content, which multimodal templates need.
- Nothing has been published to Maven Central yet. If the first release happens mid-phase, every
  later change to a released public type must be additive (no removals or signature changes).

Roadmap rules apply throughout. In particular: name exact mlx-c declarations, probe native
uncertainty first, no `ffi` use outside `jmlx-core`, and `MLXScope` ownership and `MLXException`
for every new op.

## Public-contract decisions

1. **Modules.** Model families stay in `jmlx-models` (Phase 6 decision 1). A new published module,
   `jmlx-vision` (`se.alipsa.jmlx.vision`), holds pure-Java image decode/resize/normalize. Like
   `jmlx-tokenizer`, it targets Java 21, has no `jmlx-ffi` dependency and is not native-gated.
   `jmlx-models` declares `api project(':jmlx-vision')`.
2. **No `MLXArray` in public model results.** Encoder outputs, embeddings and classification scores
   are returned as Java arrays or records (Phase 6 decision 2 extended). `MLXArray`-level access
   stays on `nn` modules for advanced callers.
3. **One generation contract.** Seq2seq (T5) and VLM models implement `TextGenerationModel`, and
   `TextGenerationModel.generate(GenerationRequest, Consumer<GenerationEvent>)` keeps its signature.
   Sampling, streaming, cancellation and `GenerationConfig` validation are shared, but seq2seq
   history and cache accounting are redefined explicitly in 7.2 rather than inherited.
   **Scheduler rejection** reuses and tightens the existing mechanism:
   - `BatchGenerationScheduler` already rejects at start any factory model that is not a
     `DecoderModel` (`acceptModel`, `SchedulerStartException`). Seq2seq and VLM model classes
     compose a decoder rather than extend `DecoderModel`, so they are refused at start through
     that check. Its message gains the model type and the phrase "not supported by the batch
     scheduler".
   - Admission additionally rejects any request with a non-empty `images()` with
     `IllegalArgumentException`, so an image payload is never silently dropped by a text cohort.
   - No marker interface or capability method is added. If cohort support arrives later, it uses
     `ModelMetadata.inputModalities()` (decision 4) and a dated amendment.
4. **Image payload on `GenerationRequest`.** `GenerationRequest` is a final class, so images are
   carried as an additive, immutable payload on it rather than through a companion type:
   - `withImages(List<RgbImage>)` returns a copy, mirroring `withCachePolicy`, and `images()`
     returns an unmodifiable list that is empty by default. `RgbImage` is `jmlx-vision`'s decoded
     pixel record. Preprocessing is model-specific (`preprocessor_config.json`), so the model, not
     the request, turns images into tensors.
   - **Deep immutability.** An unmodifiable list does not protect the pixel buffers inside it, so
     `RgbImage` owns its data the way `GenerationRequest` already owns `promptTokenIds`:
     - Its constructor validates `width > 0`, `height > 0` and an overflow-checked
       `width * height * 3` (computed with `Math.multiplyExact`) equal to the buffer length, then
       copies the buffer.
     - Its pixel accessor returns a copy. Model code reads through a package-private or
       bulk-copy path, so preprocessing does not add one copy per access.
     - `RgbImage` is a final class, not a record, so the array component cannot leak through a
       generated accessor, and `equals`/`hashCode` compare content.

     Tests mutate the caller's buffer after construction, and mutate an accessor's returned
     buffer. Neither mutation changes the image, the request or the generated output. Zero,
     negative and overflowing dimensions are rejected, as is a buffer of the wrong length.
   - Existing constructors and the `text`/`chat` factories are unchanged. A request without images
     behaves exactly as today.
   - A model that accepts images validates before any native work that the count of image
     placeholders in its prompt equals `images().size()`. A text-only model given a non-empty
     `images()` throws `IllegalArgumentException` naming the model type. Images are never
     silently dropped.
   - `ModelMetadata` gains an additive `inputModalities()` set (`TEXT`, `IMAGE`) so callers can
     check support before building a request.
5. **Train/eval state.** `Module` gains `train(boolean)`/`isTraining()`, applied recursively.
   jmlx's default is **eval**, a deliberate deviation from MLX's `training=True` default because
   jmlx is inference-first. Record this in javadoc and the core README. `Dropout` is the
   identity in eval. In training mode it throws `UnsupportedOperationException` naming Phase 11,
   rather than silently skipping dropout.
6. **Layout.** jmlx `nn` follows MLX: channels-last (`NHWC`) activations, with conv weights shaped
   `[out, kH, kW, in]`. Checkpoint loaders own the conversion from PyTorch layout (`[out, in, kH,
   kW]`). They decide by checkpoint provenance and verify by shape. A shape that fits neither layout
   is an error. Loaders never guess.
7. **Failure policy unchanged.** An unknown `model_type`, an unsupported config field (for example
   `qwen3_moe`, video, a per-layer quantization override) or an unexpected tensor fails before a
   model is returned.
8. **Request-owned inference state.** Some tensors outlive one decode step but belong to one
   request: the T5 static cross-attention K/V (7.2), the projected image features (7.3a), the
   Qwen3-VL DeepStack features and the multimodal position state after prefill (7.3b). Under the
   current scope rules, a tensor allocated in a step's `activation` scope dangles once that step
   closes. A tensor held on the model scope instead accumulates across requests. Both are bugs, so
   the rule follows the existing `KVCache` placement in `DecoderModel.generate`:
   - Every such tensor is owned by the per-request `generation` scope, which today owns the KV
     caches. It is computed directly into that scope before the step loop, or moved there with
     `MLX.hoist` from the step that produced it. Either way the move is explicit and named in
     code, never implied by scope inference.
   - Nothing request-specific is stored on the model or its scope. Model fields hold weights only.
   - The `generation` scope closes on every terminal path: EOS, stop token, `maxNewTokens`,
     cancellation before or between steps, listener exceptions, tokenizer failures and native
     failures. The existing `try`-with-resources structure guarantees this; new code keeps the
     state inside it.
   - Leak tests reuse the `MLXMemory.activeBytes()` pattern from `BatchSchedulerMemoryTest`:
     - repeated generations with the same and with different images (and T5 sources) leave
       active memory flat;
     - within one request, growth across decode steps is bounded by KV retention. This reuses
       the existing test's slope method, which compares the late per-token slope with the early
       one. That test bounds total growth with a hard-coded constant (`4L * 1024 * 1024`,
       "generous for 2 layers x 3 rows of a tiny model"). New request-state tests use a
       per-token budget derived from layers, KV heads, head size and dtype, plus a stated
       margin. **2026-10-05 amendment:** replacing the existing scheduler test's constant is
       separate follow-up work, not a 7.2 dependency. Request-specific static tensors
       (cross-attention K/V, image and DeepStack
       features) add no per-step growth;
     - a capacity-bounded FULL cache (`KVCachePolicy.full(capacity)`) never evicts. It rejects a
       position beyond capacity, and generation validates the whole token budget up front. So it
       is tested for the same bounded growth, plus rejection before native work of a request one
       token over capacity, not for a plateau;
     - a sustained plateau is required only under sliding retention, for steps after the window
       has filled;
     - cancellation between steps and an injected failure after prefill both return to baseline.

## Delivery order

| Milestone | Primary modules | Depends on | Exit evidence |
| --- | --- | --- | --- |
| Step 0 Roadmap amendment | docs | — | amended roadmap merged |
| 7.0 Qwen3 text decoders | `jmlx-core`, `jmlx-models`, `jmlx-jinja` | step 0 | Qwen3 Tier-A logits; Qwen3-0.6B Tier-B; Qwen3-8B 4-bit run recorded |
| 7.1 Core inference modules | `jmlx-core`, `tools/mlx-oracle` | step 0 | every layer oracle-tested; inventory updated |
| 7.2 Encoders and encoder-decoder | `jmlx-core`, `jmlx-models`, `jmlx-tokenizer`, reference/Tier-B tools | 7.1 | BERT embeddings/classification; Flan-T5 generation goldens |
| 7.3a Vision foundation + SmolVLM | `jmlx-vision`, `jmlx-models`, `jmlx-tokenizer` | 7.1 (7.2 cross-attention not needed) | preprocessing goldens; SmolVLM-256M Tier-B |
| 7.3b Qwen3-VL | `jmlx-models`, `jmlx-vision` | 7.0, 7.3a | Qwen3-VL Tier-A logits and Tier-B image+text output |
| 7.3c LLaVA-style (optional) | `jmlx-models` | 7.3a | Tier-B artifact, or a dated deferral |
| 7.4 Audio/diffusion decision | docs | 7.2, 7.3a | reviewed decision record |

Step 0 merges first. 7.0 and 7.1 can then proceed in parallel. 7.3a may start once 7.1's conv,
normalization and activation layers land, even if 7.2 is still open. Each milestone is its own PR
series and its own `req/plans/phase7-<n>-plan.md`, written and reviewed before code.

## Step 0 — Roadmap amendment (its own small PR, before 7.0 or 7.1 code)

Amend `req/full-roadmap.md` with a dated note:
- add milestone 7.0 (Qwen3 text decoders) to §Phase 7 and explain the product-priority reason;
- extend the §Purpose permission ("Phase 6 may implement the focused array, indexing,
  sort/selection, and random-key slices it needs from later groups") to Phase 7, for the Phase 8
  milestone 2–4 array slices listed in the baseline. The 7.1 layers need no permission, because
  §Phase 7 milestone 1 already names them. Record that each slice is entered in the inventory,
  and that Phase 8 and Phase 11 milestone 2 ("Complete standard layers required by the pinned MLX
  `nn` scope") extend this work rather than re-deriving it.

No 7.0 or 7.1 code merges before this PR.

## 7.0 — Qwen3 text decoders

Qwen3 is a decoder-only family, so this is a 6.3-style capability addition placed first for product
priority. The 7.0 sub-plan must verify every architecture fact below against the reference
`transformers` implementation.

1. Add a per-head query/key normalization capability: `ArchitectureDescriptor.Attention` gains a
   `qkNorm` flag, and `DecoderAttention` applies `RMSNorm` over `head_dim` to Q and K after
   projection and before RoPE. Tensor plan: `self_attn.q_norm.weight`/`k_norm.weight` are required
   when the flag is set and forbidden otherwise. They are never quantized: PR #31's
   `addQuantizedCompanions` allows `.scales`/`.biases` only for `*_proj.weight`,
   `model.embed_tokens.weight` and `lm_head.weight`, and keeps norm weights float.
2. Map `model_type: qwen3` in `ArchitectureMappings`, covering: explicit `head_dim`, `attention_bias`,
   `tie_word_embeddings` (tied in small sizes, untied at 8B), RoPE theta and scaling via the
   existing scaled-RoPE paths, and rejection of `use_sliding_window=true` (as for qwen2). Dispatch
   `"qwen3"` to the existing `QwenModel` in `TextGenerationModels`' switch. Reject `qwen3_moe`
   explicitly until a dated follow-on. Window-related keys mirror qwen2 explicitly rather than by
   default:
   - the qwen3 family entry sets `acceptsMaxWindowLayers` and `acceptsLayerTypes` as qwen2 does;
   - `layer_types` is absent from current dense Qwen3 artifacts (it is a `Qwen3Config` field
     that defaults to `None` and is filled with all `full_attention`). It is accepted when absent,
     null or all `full_attention`;
   - `max_window_layers` is accepted but inert while `use_sliding_window` is false.

   The parser already treats `attention_dropout`, `use_cache` and `torch_dtype` as ignored keys.
   The sub-plan lists every real-artifact key and how it is handled, so none falls through to the
   unknown-key warning.
3. Extend PR #31's affine-quantized loading to `qwen3`, then add a quantized Tier-A case.
4. Chat template: render the real Qwen3 template through `jmlx-jinja`, with `enable_thinking`
   passed via `ChatTemplateOptions.extraContext`. Two pinned references, each with its own role:
   - **Rendering:** `tools/hf-reference`'s existing `--chat` mode (pinned `transformers` 4.57.6,
     `tokenizer.apply_chat_template`). `tools/tokenizer-oracle` only encodes and decodes through
     Rust `tokenizers` and cannot render templates. Add a committed Qwen3 tokenizer/template bundle
     under `jmlx-tokenizer/src/test/resources/families`, extend `generate.py --chat` with
     `--family qwen3`, and record rendered text plus token IDs in `goldens/chat-qwen3.json`, with
     bundle hashes in `chat_sources`. Cover `enable_thinking` true and false, with and without a
     system message, multi-turn conversations with prior assistant turns, and tool calls if the
     template supports them. The 7.0 sub-plan verifies that the pinned `transformers` version
     supports Qwen3. It records any bump as a reviewed golden diff.
   - **Encoding:** `tools/tokenizer-oracle` fixtures for the Qwen3 tokenizer's encode/decode.

   Fix any missing Jinja feature in `jmlx-jinja` along with its own CHANGELOG entry. Generated
   `<think>` content is returned as raw text; parsing it is deferred.
5. Fixtures: add a tiny Qwen3 checkpoint and logits to `tools/hf-reference` (`--family qwen3`), plus
   provenance. Tier-B: Qwen3-0.6B in `tools/tier-b` and `.github/workflows/tier-b.yml`. Record a
   Qwen3-8B 4-bit (mlx-community) local manual run and a benchmark in `req/phase6-4-benchmark.md`.
   It stays a manual, evidence-only run, not a scheduled job, unless proven to fit. The repo's
   recorded runner spec (`req/phase6-tier-b-artifacts.md`, checked 2026-09-29) is an M1 with
   7 GB RAM and a 14 GB SSD, and roughly 5 GB of packed weights plus runtime overhead is
   unverified against that. The 7.0 sub-plan measures peak native memory and download size
   locally and records whether it fits with margin. The scheduled Tier-B job stays the small
   Qwen3-0.6B either way.

**Gate:** Qwen3 synthetic logits match the HF float32 reference at the strict `1e-4` bound in
`float32GoldenTest` (tagged `full-float32`, `MLX_ENABLE_TF32=0`). In default TF32 mode, the
measured maximum error is recorded and asserted under a separate default-mode bound (see
Cross-cutting verification); both chat-template goldens pass; Qwen3-0.6B exact greedy IDs pass
Tier-B; the 8B 4-bit run and benchmark are recorded. The Qwen3 compatibility-matrix row is added
and, in the same PR, the existing rows' Quantization column (still "float weights only") is
reconciled with the MLX affine support recorded below the matrix, which landed in PR #31.

## 7.1 — Core inference modules (`jmlx-core`)

Implementation evidence (2026-10-05) is in `phase7-1-probe-findings.md`. Pinned limitations:
constant/edge padding only; native 3-D grouped convolution unsupported; grouped transpose2d with
non-unit stride explicitly rejected for a CPU/GPU correctness defect; sinusoidal dims<4 rejected
instead of returning NaNs. KL takes log-probability targets, and empty max-pool outputs preserve
the native reduction error. Acceptance remains pending legacy Phase 6 byte-exact oracle drift and
cross-host CPU fixture verification; implementation is not marked milestone-complete.

Focused Phase 8 slices land here under the permission step 0 adds, each recorded in the inventory.

1. **Probe first** (`req/plans/phase7-1-probe-findings.md`):
   - `mlx_conv2d`/`mlx_conv_general` weight layout, `groups`, dilation and output shape;
   - `mlx_conv_transpose*` `output_padding` semantics;
   - which `mlx_pad` `mode` strings are accepted (the binary strings suggest `constant`/`edge`
     only), `pad_value` dtype promotion, and what `mlx_pad_symmetric` adds over `mlx_pad`;
   - `mlx_max_axes` versus `mlx_max_axis`/`mlx_max` behavior for empty and negative axes, and
     keepdims;
   - `mlx_as_strided` (`shape`, `int64_t` strides, `size_t offset`): view ownership and
     laziness; whether strides and offset address the input's logical row-major elements or its
     physical buffer when the input is transposed, sliced or broadcast; and how negative strides,
     zero strides and out-of-range offsets behave natively. The declaration bounds none of these,
     so an unchecked call can read outside the input's buffer;
   - the error convention for invalid conv shapes (status return versus error handler).
2. **Ops (facades).** Each op names its binding.
   - `MLXOps`:
     - `abs` (`mlx_abs`), `minimum` (`mlx_minimum`), `clip` (`mlx_clip`), `floor` (`mlx_floor`),
       `log1p` (`mlx_log1p`);
     - `maxAxes` (`mlx_max_axes`) and `varAxes` (`mlx_var_axes`, with `ddof`);
     - `softmaxAxes` (`mlx_softmax_axes`);
     - `logSoftmax` (`x - logSumExpAxis`), documented as a composition in javadoc and the
       core README. Leave shared inventory records unchanged: do not add a binding-less derived
       record or remap `mlx_logsumexp_axis` from `implemented`.
   - `MLXShape`:
     - `pad` (`mlx_pad`, a mode parameter limited to what the probe confirms) and `padSymmetric`
       (`mlx_pad_symmetric`);
     - `asStrided` (`mlx_as_strided`), `tile` (`mlx_tile`), `repeatAxis` (`mlx_repeat_axis`).
   - **`asStrided` stride contract.** Pooling uses it through a package-private helper first. The
     public `MLXShape.asStrided` is exposed only once the 7.1 sub-plan defines this contract and
     its tests pass:
     - `shape` and `strides` have equal length, extents are non-negative, and `offset` is
       non-negative.
     - The lowest and highest reachable element index, `offset + Σ (extent−1)·stride` taken over
       the negative and the positive strides respectively, lies in `[0, input.size())`. It is
       computed with `Math.multiplyExact`/`addExact`, and overflow is an `IllegalArgumentException`.
       A shape containing a zero extent reaches no element; a rank-zero shape reaches the
       element at `offset`.
     - Negative and zero strides are accepted only if the probe shows MLX supports them; otherwise
       they are rejected with a named message.
     - Every input passes through `mlx_contiguous(a, false)` before striding. The probe confirms
       offset zero addresses the normalized view's start, including row-contiguous slices sharing
       a larger buffer. Enforce this with ordinary `StridedViewSafetyTest` in `check` and CI's
       required native-suite list, including on native pin changes. If this cannot be established,
       materialize a proven independent copy or reject before native striding. Pooling over
       transposed and sliced inputs is tested.

     Tests: bounds at exactly the first and last element, one past each, overflowing
     stride×extent, negative and zero strides, zero-extent shapes, scalar (rank-zero) offsets at
     the last valid element and one past it, and pooling windows over transposed
     and sliced inputs compared with the MLX oracle.
   - Flattening `mlx_repeat` stays `unplanned` until a workload needs it.
   - A new `MLXConv` facade (a stable family under roadmap rule 3) holds `conv1d/2d/3d`
     (`mlx_conv1d/2d/3d`), `convTranspose1d/2d/3d` (`mlx_conv_transpose1d/2d/3d`) and
     `convGeneral` (`mlx_conv_general`).

   All ops reuse the `NativeOps.checked`/`scopeOf` and temporary-vector cleanup helpers.
   **Inventory cleanup:** `req/mlx-api-inventory-overrides.json` lists `mlx_softmax_axis` in the
   `MLXShape` group, but `softmaxAxis` lives in `MLXOps`. Move it into an `MLXOps` record next to
   `mlx_softmax_axes` and regenerate the inventory.
3. **Modules (`se.alipsa.jmlx.nn`):**
   - containers: `Sequential` and an indexed `ModuleList` (child names `"0"`, `"1"`, …);
   - `Dropout` per decision 5;
   - activations: new `ReLU`, `LeakyReLU`, `ELU`, `SELU`, `Tanh`, `Sigmoid`, `Softplus`, `Mish`,
     `HardSwish` and `QuickGELU` (`quick_gelu`). Exact and tanh-approximate GELU already exist
     (`Activation.GELU`/`GELU_TANH`). Lift the config-string mapping out of Gemma's
     `hidden_activation` branch into one shared `hidden_act`/`hidden_activation` resolver, so the
     BERT, T5, SigLIP and CLIP strings resolve and unknown strings still fail by name.
   - `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d`;
   - `MaxPool1d/2d`, `AvgPool1d/2d` (`as_strided` + reduce, mirroring MLX's Python
     implementation);
   - `Upsample` (nearest, linear), built on `floor`/`clip`/`take` as MLX does;
   - `GroupNorm`, `BatchNorm` (eval with running statistics; training throws, per decision 5),
     `InstanceNorm`;
   - positional encodings: `SinusoidalPositionalEncoding` and `ALiBi`.
   - **Embedding variants (roadmap item):** no new embedding class is needed. Learned absolute
     positions and BERT token-type embeddings reuse `Embedding`/`QuantizedEmbedding`. T5's
     relative position bias is model-specific (7.2), and image patch embeddings are `Conv2d` or
     linear-on-patches (7.3a). Bag or sparse embeddings are deferred until a workload needs them.
4. **Losses (evaluation use):** `cross_entropy` (with label smoothing), `binary_cross_entropy`,
   `nll`, `mse`, `l1`, `smooth_l1`, `kl_div` and `cosine_similarity`, as pure Java compositions.
   Their gradients are Phase 11's concern.

**Tests:** hand-computed cases per layer; `tools/mlx-oracle` fixtures generated from the matching
MLX Python `nn` layers (conv strides/padding/groups/dilation, pooling edges, every pad mode, every
upsample mode, norm statistics); scope/confinement and leak tests for the new temporaries. New
native suites go on CI's required-suite list.

**Gate:** every listed layer matches the MLX oracle; every touched binding is `implemented` in the
inventory with tests; `verifyMlxApiCallSites` is green.

## 7.2 — Encoders and encoder-decoder generation

**Implementation update, 2026-10-05:** local BERT/T5, tokenizer and Tier-B checks pass; see
`req/phase7-2-implementation-report.md`. New macOS 26 CI acceptance remains pending. The inspected
Intel MRPC revision has no safetensors; the licensed single-sequence SST-2 fallback is selected.
Pair encoding remains implemented and oracle-verified.

**Scope amendment, 2026-10-05:** include additive tokenizer pair encoding for the selected
BERT classifier candidates and SentencePiece Precompiled normalization if required by the
pinned Flan-T5 tokenizer. These changes require tokenizer oracle fixtures, Java 21 checks,
CHANGELOG/API documentation and inclusion in jmlx-tokenizer's independent release process.
Also include safe nested artifact downloads, task-specific Tier-B manifests/tests/CI rows and
an actual pinned sentence-transformers reference dependency. The implementation sub-plan is
`req/plans/phase7-2-plan.md`; outstanding artifact inspection is a prerequisite, not evidence
that current tokenizer or Tier-B infrastructure already supports these checkpoints.

1. **Shared modules:** a bidirectional padding mask in `AttentionMask`; a `CrossAttention` module
   whose encoder keys/values are computed once and held in a static (non-appending) cache variant,
   owned by the request's `generation` scope (decision 8);
   and an encoder block with post- or pre-norm selected by descriptor.
2. **BERT-style encoder** (`model_type: bert`): word, position and token-type embeddings, post-LN
   blocks, exact GELU, pooler. Public API in `se.alipsa.jmlx.models`:
   - `TextEncoderModel`: hidden-state/embedding output with a `Pooling` policy (CLS, mean, max,
     plus optional L2 normalization), read from sentence-transformers `modules.json`/`1_Pooling`
     when present;
   - `SequenceClassifier` and `TokenClassifier`: labels from `config.json` `id2label`. Token
     results carry `TokenizerEncoding` offsets.
   Results are Java records (decision 2).
3. **T5 / Flan-T5** (`model_type: t5`): bucketed relative position bias (bidirectional in the
   encoder, causal in the decoder, computed in the first layer and shared); T5 RMS norm; unscaled
   attention (SDPA `scale = 1.0`: `DecoderAttention`/`MultiHeadAttention` and the new
   `CrossAttention` gain an explicit attention-scale constructor parameter. The current
   `1/sqrt(headDim)` remains the default, so existing callers and goldens are unchanged); dense-ReLU
   and gated-GELU FFN variants; tied/untied head with T5's `d_model^-0.5` rescale when tied;
   `decoder_start_token_id`. Generation implements `TextGenerationModel`: prompt token IDs are the
   encoder input, the encoder runs once, then the decoder decodes with a self-attention `KVCache`
   plus the static cross-attention cache. The sub-plan must state T5's float16 overflow risk and the
   dtype policy it chooses.
4. **Seq2seq generation semantics.** The decoder path must not inherit its prompt-based rules.
   Today `DecoderModel` seeds penalty frequencies from the prompt (`PenaltyInputs.frequencies`)
   and requires cache capacity for `prompt.length + maxNewTokens - 1`. Applied to T5, those rules
   would penalize encoder tokens and charge the source length against the decoder cache. The 7.2
   sub-plan defines each of the following against `transformers`' encoder-decoder `generate` and
   records any deliberate difference:
   - **Decoder history.** Repetition, frequency and presence penalties see only decoder-side
     tokens, never the encoder source. Whether `decoder_start_token_id` counts as history follows
     the tokens HF passes to its logits processors, verified rather than assumed.
   - **Start token.** It is resolved from `generation_config.json`, then `config.json`; a missing
     value fails at load. It is fed to the decoder but never emitted as an event or counted
     toward `maxNewTokens`. `GenerationResult` keeps the source as its prompt tokens and
     returns only generated decoder tokens. Text deltas decode only those generated tokens.
   - **Capacity accounting.** `GenerationCachePolicy` applies only to the decoder self-attention
     cache: start token plus `maxNewTokens - 1`. The static cross-attention cache is sized by the
     source length and validated separately against a documented maximum source length. It is not
     a `KVCachePolicy` cache and is never evicted.
   - **Unsupported cache policies.** Full and capacity-bounded decoder caches are supported.
     Sliding-window eviction is rejected with a named exception, because the relative position
     bias is not defined over an evicted cache. Cache reorder/fork is not offered for seq2seq.

   Focused tests:
   - A source token repeated many times is not penalized in the first decoder step.
   - Capacity boundaries hold with a long source and a short target, and the reverse.
   - The start token is absent from events and results, and `maxNewTokens` counts exactly.
   - Sliding policy and missing start-token configs are rejected.
   - Cancellation before encoding and between decode steps, then scope cleanup, on every terminal
     path.
5. **Fixtures:** `tools/hf-reference` gains tiny `bert`/`t5` checkpoints with encoder outputs and
   logits. Tier-B candidates are all-MiniLM-L6-v2 (embeddings), one licensed BERT classification
   artifact and Flan-T5-small. The sub-plan re-verifies licenses and pins revisions.

**Gate:** BERT embeddings/classification and Flan-T5 greedy output match HF references (Tier-A) and
pinned Tier-B runs; the item 4 seq2seq semantics are specified and their focused tests pass; the
decision 8 leak tests pass for the cross-attention cache; the
scheduler rejects seq2seq with a named exception; the matrix gains "text encoder" and "seq2seq text"
modality rows.

## 7.3a — Vision foundation and SmolVLM-256M

Implementation sub-plan: [phase7-3a-plan.md](phase7-3a-plan.md) (proposed, 2026-10-06).

**Review amendments, 2026-10-06:**

- Include Pillow-compatible three-lobe LANCZOS (`resample=1`) for the target Idefics3 processor.
  Select the slow PIL processor explicitly in both Python tools. Target exact uint8 resampling
  and complete chained preprocessing; if exactness is not achieved, record separately measured
  single-stage, full-chain and JPEG decode/full-chain bounds before acceptance. A single-resize
  tolerance does not establish a bound for the three uint8 resize stages. Verify Pillow, NumPy
  and Transformers pins agree across the image and HF oracle locks.
- Pin raw `PIL.Image.open` → slow processor as the reference loading path, with no preliminary
  RGB conversion or EXIF transpose. ImageDecoder deliberately ignores EXIF orientation and
  ICC/gAMA color management and extracts decoded samples without ColorModel conversion.
  Javadoc and README tell callers to rotate images themselves. Palette/tRNS behavior follows
  the oracle fixture, not an assumed compositing outcome.
- Preserve Pillow notices in every ported source header and in vision's binary, sources and
  Javadoc jars. Add vision-only ancillary-jar packaging and smoke content checks in the module
  PR; the existing six modules' sources/Javadoc packaging is unchanged.
- Replace masked-scatter-first with a single `take` gather over concatenated text embeddings and
  image features, indexed by the validated pure expansion map. This uses existing core facades;
  masked_scatter remains deferred unless measurements justify a separately probed wrapper.
- Ship release-smoke module/version/POM checks, release-script coverage, Ubuntu vision check and
  AGENTS.md module/architecture/release updates in the PR adding jmlx-vision and its models
  dependency, rather than postponing them to final acceptance.
- ImageTensor remains a defensively copied record with explicit content equality/hashCode,
  bounded toString and public bulk-copy access. Decoder APIs gain explicit decode limits.
- A zero-token VLM request validates syntax and expands using pure geometry, reports the expanded
  prompt length, and skips pixel preprocessing and native work; cache capacity charged is zero.
- `promptPositions` means effective input length: decoder/scheduler prompt, expanded VLM prompt,
  or T5 encoder source. It is at least the unexpanded prompt count. Adding this record component
  is an intentional pre-publication compatibility exception to decision 4's additive wording:
  constructors remain, but record patterns/equality change. Recheck publication before proceeding;
  if published, amend to compatible metadata instead. Preserve unexpanded result/abort IDs.
- Include T5 raw logits and greedy IDs in both precision-mode exact comparisons if its generation
  loop is changed by shared-code extraction.

1. **`jmlx-vision` module** (Java 21, published, release script, `verifyBytecodeLevel`,
   Ubuntu pure-Java CI job):
   - decode PNG/JPEG via `javax.imageio` into `RgbImage`, the decoded-pixel record that
     `GenerationRequest.withImages` carries (WebP is unsupported and documented);
   - convert to RGB; resize (bilinear/bicubic/LANCZOS); rescale; normalize (mean/std);
   - HF-style image tiling/splitting;
   - parse `preprocessor_config.json`;
   - output `ImageTensor` records (`float[]` plus shape and layout).

   Add a `tools/image-oracle` (hash-locked venv, provenance) with PIL/`transformers` image-processor
   goldens. Resampling is compared under a documented tolerance, because PIL's antialiased kernels
   are matched rather than assumed. The sub-plan states the exact tolerance and the reason for it.
2. **Decoder input embeddings:** generalize the existing package-private `EmbeddingHook` seam
   instead of adding a parallel path. Split the embedding lookup from the decoder stack so an
   internal forward can start from a caller-built embedding tensor. Production then uses the same
   post-embedding point the hook uses today. `EmbeddingHook` stays a test seam, re-expressed on the
   new entry point, and `BatchGenerationScheduler.Hooks` keeps working unchanged. Merge image
   features at image-token positions using a validated destination map and `take` over
   concatenated text/image rows (2026-10-06 amendment above). The token-ID path must stay
   bit-identical. The committed goldens use tolerances, so they cannot prove this. Prove it with
   the exact before/after comparison defined under Cross-cutting verification.
3. **Vision encoder and connector:** a SigLIP-style ViT (patch embedding via `Conv2d` or
   linear-on-patches, learned positions, LayerNorm/GELU blocks) and the Idefics3 connector (pixel
   shuffle + linear). Architecture facts are verified against `transformers` in the sub-plan.
4. **Multimodal prompts:** allow structured message content (`{type: image}`/`{type: text}`) in
   `HfTokenizer.renderChat`; expand image placeholders into the model's image-token layout,
   including tile row/column tokens. Implement decision 4: `GenerationRequest.withImages`/`images()`
   carrying `RgbImage` inputs, with `ModelMetadata.inputModalities()`.

   **Expansion semantics.** `GenerationRequest.text`/`chat` encode eagerly, and the token count per
   image depends on preprocessing (SmolVLM tiling, Qwen `smart_resize`). The tokenizer therefore
   cannot expand placeholders. Instead:
   - The request holds the unexpanded IDs. Each image-capable family defines its own
     **unexpanded placeholder syntax**: the exact token sequence its official chat template emits
     per image, before processor expansion. The sub-plan confirms each one against the pinned
     template and records it in the family's mapping table. For example:
     - Qwen3-VL emits `<|vision_start|><|image_pad|><|vision_end|>`, verified against
       `Qwen/Qwen3-VL-2B-Instruct`'s `chat_template.json`. The wrapper is part of the
       unexpanded syntax, and only the single `<|image_pad|>` is expanded. Two adjacent images
       render as two complete adjacent wrappers. With `add_vision_id` set, the template also emits
       plain text `Picture N: ` before each wrapper. That prefix is ordinary text, so it is neither
       part of the placeholder unit nor a validation error.
     - SmolVLM/Idefics3 emits a single `<image>` token. The processor adds the
       `<fake_token_around_image>`, tile row/column and global-image tokens during expansion.
   - Validation runs on the family's syntax, before native work:
     - The number of placeholder units must equal `images().size()`.
     - A template-produced wrapper around a single placeholder is accepted.
     - An **already-expanded** prompt is rejected with its own message. Detection is per family,
       because repetition alone is not evidence of expansion:
       - Qwen3-VL: more than one `<|image_pad|>` inside one
         `<|vision_start|>…<|vision_end|>` wrapper.
       - Idefics3/SmolVLM: any processor-only token the template never emits
         (`<fake_token_around_image>`, `<row_x_col_y>`, `<global-img>`). Adjacent `<image>`
         tokens are **not** rejected. `<image><image>` with two supplied images is valid
         unexpanded input, so each `<image>` counts as one placeholder.
     - A wrapper with no placeholder, an unbalanced wrapper, and a placeholder outside the
       wrapper (where the family requires one) are rejected.
     - `<|video_pad|>` is rejected as unsupported video input (decision 7).
   - The model expands inside `generate`, after preprocessing and before any native work. The
     expansion is a pure, package-private Java function from (IDs, per-image grid) to
     (expanded IDs, image-token positions), so it is unit-testable without native code.
   - `GenerationResult.promptTokenIds()`, `tokenIds()` and `GenerationAbortedException` report the
     request's **unexpanded** IDs, keeping `promptTokenIds() + generatedTokenIds()` consistent
     with the request a caller built. The expanded length is exposed separately, as an additive
     `promptPositions()` count or as metadata on the result, not by changing the meaning of
     `promptTokenIds()`.
   - Penalty seeding, context-length limits and `GenerationCachePolicy` capacity
     (`expanded.length + maxNewTokens - 1` for positive maxNewTokens; zero otherwise) use the
     **expanded** sequence, as HF does with its
     processor-expanded `input_ids`. The sub-plan verifies the penalty-seeding point against
     `transformers`' logits processors rather than assuming it. Capacity is validated after
     expansion, before native work, and the error names both lengths.

   Tests:
   - the placeholder count must match the image count;
   - a prompt rendered by each family's real chat template through `GenerationRequest.chat` with
     one and with two images is accepted (including Qwen3-VL's vision wrapper, two adjacent
     wrappers, and `add_vision_id=true` via `ChatTemplateOptions.extraContext`);
   - SmolVLM `<image><image>` with two images is accepted and expands to two independent image
     layouts, and the same prompt with one image fails the count check;
   - already-expanded prompts (a Qwen3-VL wrapper with several pad tokens, an Idefics3 prompt
     with processor-only markers), malformed wrappers and video placeholders are rejected with
     distinct messages;
   - a text-only model rejects a request with images;
   - a request without images is unchanged on every existing text model;
   - expansion with one image, with two images of different tile counts, and with an image
     between text segments, against the HF processor's expanded `input_ids`;
   - capacity boundaries computed from the expanded length: exactly at capacity succeeds, one over
     fails before native work, with two differently tiled images;
   - result and abort token IDs equal the request's unexpanded IDs plus generated IDs.
5. **Fixtures:** a tiny Idefics3 checkpoint plus image/logit goldens in `tools/hf-reference`, and
   a SmolVLM-256M Tier-B manifest with a committed test image and an exact greedy-ID golden.
   Synthetic goldens cover prefill plus several cached decode steps (as `tools/hf-reference`
   already records for text families), cached-versus-full-forward agreement, and prompts with
   one and with two images.
   Multimodal chat-template goldens come from the same pinned `--chat` generator, using the
   processor's `apply_chat_template`.

**Gate:** preprocessing goldens pass; SmolVLM synthetic logits match HF; SmolVLM-256M Tier-B image
description is exact; text-only decoder goldens are unchanged; the item 4 expansion, `RgbImage`
immutability and decision 8 leak tests pass; the matrix gains an "image+text → text" row.

## 7.3b — Qwen3-VL

Prerequisites: 7.0 (Qwen3 text backbone) and 7.3a. The sub-plan must verify against
`transformers`: the vision tower (2-D RoPE in the ViT, patch merger), dynamic-resolution
preprocessing (`smart_resize`, min/max pixels), interleaved multimodal RoPE with 3-axis position
IDs, and DeepStack injection of multi-level visual features into early decoder layers.

1. Extend `jmlx-vision` with Qwen's dynamic-resolution resize and patch ordering, with goldens.
2. Add multi-axis position IDs to the decoder RoPE path. The 1-D path stays bit-identical,
   verified by the exact before/after comparison (Cross-cutting verification), not by the
   tolerance goldens.
3. Implement the Qwen3-VL vision tower, merger and DeepStack hooks, plus `model_type` mapping.
   DeepStack features and the post-prefill multimodal position state are request-owned (decision
   8), and expansion follows 7.3a item 4 with grids from `smart_resize`.
   Video input and `qwen3_vl_moe` are rejected explicitly.
4. Fixtures: a tiny synthetic Qwen3-VL checkpoint. Its HF goldens record prefill logits and
   several greedy decode steps with the KV cache, including multimodal position state after
   prefill. Cases:
   - one image;
   - two images of different sizes in one prompt, so their grids and merged token counts differ;
   - an image followed by text and then a second image;
   - a text-only prompt through the same model.

   Each case also asserts cached-versus-full-forward agreement: the logits of step *n* from
   incremental decode equal a full uncached forward over the same tokens, within the
   default-mode bound, and within `1e-4` in `float32GoldenTest`.

   Tier-B: the smallest public Qwen3-VL instruct artifact the sub-plan verifies to exist and to
   be licensed, with at least one multi-image prompt. Record a manual 8B (4-bit) run and
   benchmark.

**Gate:** synthetic prefill and multi-step cached decode logits match HF for every case above, and
cached decode agrees with full forward; Tier-B image+text output is exact; the 8B run is recorded;
the matrix row is added.

## 7.3c — LLaVA-style connector (optional)

Reuses 7.3a (vision tower plus an MLP projector). Proceed only if a licensed small artifact
exists. Otherwise record a dated deferral. PaliGemma remains deferred because of Gemma-terms
access (the same Tier-B problem as Gemma v1).

## 7.4 — Audio and diffusion capability decision

Deliver `req/plans/phase7-4-decision.md`, not code. It covers:
- inventory counts for the relevant groups (`mlx_fft_*`, conv/upsample coverage after 7.1);
- one candidate reference model per modality (for example Whisper-tiny; a small UNet/VAE pipeline);
- required new surface (STFT/ISTFT, mel filterbanks, schedulers) and its Phase 9 overlap;
- Tier-B feasibility, licenses, and an estimate.

The outcome is go (which spawns its own `phase7-4-<x>-plan.md`) or a dated deferral.

**Gate:** the decision is reviewed and the roadmap is updated to match.

## Cross-cutting verification

- Required PR checks: spotless, checkstyle, unit tests, `generateMlxApiInventory` freshness,
  `verifyMlxApiCallSites`, `verifyHfReferenceGoldens`, oracle fixture verification, native suites
  on the macOS ARM64 job (added to `ci.yml`'s required-suite list), and `jmlx-vision` on the
  pure-Java job.
- **Two precision gates, never merged** (`req/phase6-golden-precision.md`):
  - *Correctness:* CPU float32 HF references are asserted at the strict `1e-4` bound only in
    `float32GoldenTest` (`full-float32` tag, `MLX_ENABLE_TF32=0`).
  - *Default mode:* the ordinary `test` task (`MLX_ENABLE_TF32=1`) asserts each new suite under a
    separately named default-mode bound. That bound is derived from errors measured and recorded
    per family and is documented next to the existing Phase 6 measurements. Default TF32 error,
    which reached about `1.2e-3` in Phase 6, never loosens a strict assertion and never triggers
    golden regeneration.
- **Exact before/after comparison** (wherever this plan requires bit identity: 7.3a item 2,
  7.3b item 2):
  - Before the change, a Gradle task records the token-ID path's prefill and decode logits as raw
    float bits, for every Tier-A decoder family in both precision modes, from the base commit.
  - After the change, the same task on the same machine and native pins asserts exact bit
    equality, together with exact greedy token IDs.
  - The recording embeds the Phase 6 decision 4 reproducibility tuple, extended with precision
    mode: jmlx commit (the binary), both native pins (the runtime), device and macOS version,
    batch configuration, model fixture and request, plus `MLX_ENABLE_TF32`. It refuses to compare
    across a different tuple, except for the jmlx commit that the comparison exists to vary.
  - The bit recordings are device-specific, so they are PR evidence, not committed CI goldens.
    The committed tolerance goldens and exact greedy-ID goldens keep running in CI.
- Every family claim requires: a mapping table, tokenizer/template golden, Tier-A logits, Tier-B
  result and a compatibility-matrix update, as in Phase 6.
- Release: `jmlx-vision` joins the published set (its own `release.sh`, `verifyReleaseScriptsMatch`,
  `tools/release-smoke`, and a release order of `jmlx-vision` → `jmlx-models`). `AGENTS.md`
  module and architecture sections are updated in the PR that adds the module.

## Explicit deferrals

- Training-mode `Dropout`/`BatchNorm`, loss gradients and optimizers (Phase 11).
- Bag/sparse embedding variants, and flattening `mlx_repeat`, until a workload needs them.
- `qwen3_moe`, `qwen3_vl_moe`, video input, and parsing of `<think>` output.
- Batched scheduling of seq2seq and image requests.
- PaliGemma, and audio/diffusion implementation (pending 7.4).
- General indexing, FFT and linalg beyond the slices named in 7.1 (Phases 8–9).
