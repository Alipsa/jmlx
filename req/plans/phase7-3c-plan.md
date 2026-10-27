# Phase 7.3c — LLaVA-style image+text inference (Phi-3.5-Vision)

- **Status:** draft, pending review. The 7.3c go/deferral discovery was performed on 2026-10-27:
  GO — [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
  (revision `12b77fb40b63a2c73c68243d3f767aab688a1b2a`, MIT) is a small, fully licensed, public
  LLaVA-style artifact (CLIP ViT-L/14@336 tower + MLP projector + decoder). WP0 formalizes the pin
  and the tensor inventory; WP1 probes the reference. No 7.3c code before this plan is reviewed.
- **Parent:** `req/plans/phase7-plan.md` §7.3c and the delivery-order row "7.3c LLaVA-style
  (optional) | `jmlx-models` | 7.3a | Tier-B artifact, or a dated deferral"; satisfies
  `req/full-roadmap.md` §Phase 7 (milestone 3 breadth, the last open Phase 7 implementation row).
- **Dependencies:** 7.1 core layers (convolution, normalization, `QUICK_GELU` activation), 7.2
  `BidirectionalAttention`/`EncoderBlock`, 7.3a multimodal request contracts (`withImages`,
  `InputModality`, per-request vision scope, structured chat validation) and 7.3b per-request
  lifecycle/injection patterns; Phase 6.3's `rope_scaling` parse infrastructure in
  `ArchitectureMappings`. Reuse, do not fork, those contracts.

## 1. Outcome and boundaries

Deliver local Phi-3.5-Vision image+text generation through `TextGenerationModels.load` and
`TextGenerationModel.generate`, reusing the existing sampling, streaming, cancellation, result and
cache-policy contracts. Support one or more images in prompt order, text between images, and
text-only requests through the same model. The model is a `TextGenerationModel` composition —
vision tower + projector + generic decoder — deliberately not a `DecoderModel`, so the existing
scheduler non-`DecoderModel` rejection stays in force without a new check.

The candidate is `microsoft/Phi-3.5-vision-instruct` (4.2B, `model_type` `phi3_v`,
`Phi3VForCausalLM`): a CLIP ViT-L/14@336 vision tower with an InternLM-XComposer2-4KHD-style
multi-crop high-resolution pipeline, a 2×2 merge, learnable newline/global separators, a two-layer
MLP projector, and a plain pre-LN RMSNorm decoder with per-dimension NTK-Su RoPE. It is the only
small, fully MIT-licensed public LLaVA-style instruct artifact found: the `llava-hf` 7B/11B
releases are Llama-terms gated (the same Tier-B blocker as Gemma v1), PaliGemma is Gemma-terms,
and the SmolVLM line is already covered by 7.3a. If WP0 or WP1 invalidates the pin or finds a
blocking reference divergence, record a dated deferral in
`req/plans/phase7-3c-reference-findings.md` and close the milestone without code (the parent's
alternative gate).

Float safetensors only, up-cast to FLOAT32 in place (the 7.3a/7.3b vision-language rule): reject
non-floating or quantized checkpoints explicitly. Exclude the original `phi3_vision` model type
(phi-3-vision-128k, a different tower), video, remote image fetching, training, image generation,
public model batching, multimodal scheduler cohorts, quantized KV, sliding image-prefill retention,
per-layer quantization overrides and MLX affine loading (no 4-bit artifact exists for this family).
No native-pin change or generated-binding edit is planned: every primitive needed (conv2d, layer
norm, `QUICK_GELU`, bidirectional attention, reshape/permute) is already bound and facade'd from
7.1/7.2.

## 2. Repository baseline and prerequisite gate

- `jmlx-models` composes SmolVLM (`SmolVlmModel`) and Qwen3-VL (`Qwen3VlModel`) as
  `TextGenerationModel` objects: a strict descriptor parse, a `TextDecoder extends DecoderModel`
  (or descriptor-assembled decoder), a dedicated vision tower, a pure prompt expander that turns
  each image part into a contiguous block of sentinel rows, per-request vision scopes evaluated and
  closed after prefill, and a row-replacement injection at the decoder embedding entry point.
  Reuse that composition; do not invent a new one.
- `TextGenerationModels.load` dispatches `t5`, `idefics3`, and strict `qwen3_vl` before the generic
  decoder mapping, then switches on `model_type` (llama/qwen2/qwen3/mistral/phi3/gemma/mixtral).
  `phi3_v` joins the strict pre-mapping dispatches: the composite loader owns the config parse, and
  a sibling or legacy name must be rejected by name rather than misrouted.
- `ArchitectureMappings` parses `rope_scaling` objects with types `linear`, `dynamic`, `llama3`,
  `yarn` (around line 646) and carries per-family `sliding_window` treatment. The `su` type and the
  `phi3_v` family entry are new branches there, not new mechanisms.
- `RopeSpec` (sealed) has `Base`, `Linear`, `DynamicNtk`, `Llama3`, `Yarn`. `Su` is a new variant.
- `Activation` already includes `QUICK_GELU` (7.1), so the CLIP tower needs no new activation.
- The 7.3a/7.3b request contract already carries image parts through structured chat messages,
  validates them, and threads `InputModality` into `ModelMetadata`
  (`VisionLanguageMetadata`). The Tier-B manifest shape (`repository`/`revision`/`task`/`license`/
  `files`/`images`/`requests`/`reference`/`recorded_pin`/`local_evidence`) is the 7.3a/7.3b one.
- `tools/hf-reference` is a generate-only, hash-locked CPython 3.12 tool (Torch 2.6.0 CPU,
  `transformers==4.57.6`, float32, eager attention) and has no `torchvision` pin. `transformers`
  4.57.6 has **no native `phi3_v` model**: this artifact runs its bundled
  trust-remote-code `modeling_phi3_v.py`/`processing_phi3_v.py`, and the bundled processor imports
  `torchvision`. The reference toolchain must therefore gain a `torchvision` pin and a
  hash-verified bundled-code execution path (WP1).
- At implementation start, record the actual base commit, the completed prerequisite suites and the
  merged state of every contract named above. Public changes after this plan's publication must
  remain additive.

## 3. Reference and artifact gate (WP0, WP1)

**WP0 — pin and inventory.** Pin revision `12b77fb40b63a2c73c68243d3f767aab688a1b2a`:

- License: MIT (`LICENSE`, 1141 B; HF tag `license:mit`); record the license evidence URL per the
  manifest convention.
- Files (sizes in bytes): `model-00001-of-00002.safetensors` 4,944,122,112;
  `model-00002-of-00002.safetensors` 3,349,208,776 (total 8,293,330,888);
  `model.safetensors.index.json` 68,901; `config.json` 3,778; `tokenizer.json` 1,851,360;
  `tokenizer_config.json` 9,519; `special_tokens_map.json` 670; `generation_config.json` 136;
  `preprocessor_config.json` 442; `processor_config.json` 119; and the bundled code
  `modeling_phi3_v.py` 88,866, `configuration_phi3_v.py` 10,712, `processing_phi3_v.py` 22,045,
  `sample_inference.py` 6,778. SHA-256 every file; verify the index covers every tensor key exactly
  once (the 7.3b 8B stale-index finding makes this a required check, not an assumption).
- Record the complete tensor-key inventory grouped by owner: decoder text weights, `lm_head`, the
  vision tower under the CLIP-vision naming, and the projector/separator parameters
  (`glb_GN`, `sub_GN`, `img_projection.{0,2}` — exact prefixes come from the index dump, not from
  this plan's memory).
- Tokenizer: `LlamaTokenizer` byte-level BPE with a plain Llama-style Jinja chat template
  (`<|role|>\n content <|end|>\n`); image parts do not appear in the template — they enter as
  `<|image_i|>` markers in message text (see §4 WP3). Verify the template renders through
  `jmlx-jinja` and that the marker strings are not special tokens.

**WP1 — reference probes.** Findings go to a new `req/plans/phase7-3c-reference-findings.md` in the
7.3b findings style (native pins, host, precision, commands, shapes, exact conclusions). It must:

1. Add the `torchvision` pin to the hash-locked `tools/hf-reference` venv (the torchvision line that
   matches Torch 2.6.0, wheel SHA-256s for both wheel platforms, recorded in `provenance.json`) and
   pin the four bundled-code files' SHA-256s (from WP0) so a generator run verifies them before use.
2. Run the bundled model on CPU, float32, **explicit eager attention** (the artifact config's
   `_attn_implementation` is `flash_attention_2`; the reference must override to eager both for the
   decoder and so the vision tower keeps its plain CLIP attention, whose FA2 swap only happens under
   fa2).
3. Pin the preprocessing pipeline exactly: per image — `convert_to_rgb`, HD transform with
   `hd_num = num_crops = 4` (transpose when `width < height`; largest `scale` with
   `scale * ceil(scale / ratio) <= 4`; torch `F.resize` bilinear to `(new_h, new_w)`; top/bottom
   white-255 padding of the 8-bit image to a multiple of 336, no left/right padding; untranspose);
   `ToTensor` + CLIP mean/std normalize; the global image is a torch **bicubic** `F.interpolate`
   to 336×336 **on the normalized float32 tensor**; crop slots are `[global, sub crops in h-major
   order]` zero-padded to `num_crops + 1 = 5`; `image_sizes` is the padded `(h, w)`; per-image token
   count is `(n + 1) * 144 + (h336 + 1) * 12 + 1` with `n = h336 * w336` (maximum 781 at
   `h336 = 4, w336 = 1`).
4. Pin the tower: the bundled `CLIP_VIT_LARGE_PATCH14_336_CONFIG` (hidden 1024, 24 layers, 16
   heads, `quick_gelu`, eps 1e-5, patch 14, image 336), class token plus 576 learned 2-D position
   embeddings, `hidden_states[-2]` (the output after block 22 of 24, i.e. 23 executed blocks,
   class token still in the attended sequence) with the class token discarded → 576×1024 per crop;
   zero-padded crop slots are computed by the reference but never read, so the Java port may skip
   them (record that they cannot influence used outputs: crops are processed independently).
5. Pin the high-resolution transform: 2×2 merge of the 24×24 grid to 144×4096; `sub_GN` newline
   rows appended after every row of the sub grid (`(h336*12) * (w336*12 + 1)` rows); per image the
   block is `[sub rows, glb_GN (1 row), global rows (12*13)]`; projection is
   `Linear(4096→3072) → QuickGELU → Linear(3072→3072)` over the concatenated rows.
6. Pin the text path: `re.split(r"<\|image_\d+\|>", text)` — chunks are tokenized **separately**
   (BPE merges never cross a marker boundary), each marker `i` expands to
   `num_img_tokens[i-1]` sentinel rows, marker indices must start at 1 and be continuous, and the
   marker count must equal the image count; the reference attention mask is all ones.
7. Pin the decoder: pre-LN blocks (`RMSNorm(1e-5)` → MHA → residual; `RMSNorm(1e-5)` → SiLU MLP
   8192 → residual), 32 layers, 32/32 heads, head_dim 96, scale `1/sqrt(96)`, causal, fp32 softmax;
   `sliding_window: 262144` is present in the config but never read by the attention code
   (accepted-inert, like the qwen2/qwen3 window keys); untied `lm_head` over vocab 32064;
   `rope_theta` 10000, `max_position_embeddings` 131072, `original_max_position_embeddings` 4096.
8. Pin NTK-Su RoPE exactly: per forward call `seq_len = max(position_ids) + 1` (prefill length for
   a fresh prefill; decode position + 1 during cached decode), factor table
   `long_factor` iff `seq_len > 4096` else `short_factor` (48 per-dimension entries each);
   `inv_freq[i] = 1 / (f[i] * 10000 ** (2i / 96))`; a global
   `scaling = sqrt(1 + ln(131072/4096) / ln(4096))` multiplies **both** cos and sin; computed in
   float32 and cast to the model dtype. Record the boundary behavior at `seq_len = 4096` (short)
   and `4097` (long) and the exact scaling constant.
9. State the resample risk plainly: both resizes are **PyTorch** operations (8-bit bilinear with
   rounding in HD transform; float32 bicubic for the global image), not PIL. The existing
   `jmlx-vision` resamplers are Pillow-exact ports for a PIL-based reference. Measure the delta on
   the WP2 fixture set; if Pillow and Torch disagree (expected), WP2 implements Torch-exact
   kernels instead and pins their provenance in `jmlx-vision/NOTICE` exactly as the 7.3a LANCZOS
   port pins its Pillow sources.

**Gate:** the findings document exists, every claim above is recorded with its probe evidence, the
`torchvision` pin and bundled-code hashes are in `provenance.json`, and no claim depends on
unpinned remote state.

**Oracle provenance and fixture tiers.** The reference oracle is the pinned artifact revision's
bundled code under the hash-locked `tools/hf-reference` venv (Torch 2.6.0 CPU + the WP1
`torchvision` pin, `transformers==4.57.6` present but not the semantic source for this family),
float32, eager attention. Tier-A is an in-repo synthetic checkpoint with no external download and
no credentials; Tier-B is the public MIT artifact, also credential-free. Python execution stays
opt-in; ordinary builds gain no Python prerequisite.

## 4. Implementation work packages

Ordered so foundations land before consumers; each WP is its own PR per the parent's delivery rule.

**WP2 — `jmlx-vision` preprocessing.** Pure Java 21, no native dependency.

- `Phi3VProcessorConfig` (parses `preprocessor_config.json`: the `num_crops`, CLIP `image_mean`/
  `image_std` and `do_convert_rgb` keys the reference reads, plus the standard inert keys
  (`auto_map`, `image_processor_type`, `processor_class`) accepted-and-ignored per the 7.3b
  full-dump precedent; value domains validated as the reference validates them),
  `Phi3VGeometryPlan` (HD transform math: transpose branch, scale-selection loop, resized and
  padded sizes, crop counts, per-image token count), `Phi3VImageProcessor` (the full pipeline of
  WP1 item 3, from a decoded `RgbImage`), and `Phi3VImageProcessorResult` (per-slot pixel
  values — global first, then sub crops in h-major order, zero-padded to five slots — `image_sizes`,
  `num_img_tokens`).
- The two Torch-exact resample modes (8-bit bilinear with the reference's rounding; float32
  bicubic) as new `jmlx-vision` resampling kernels with provenance pinned in
  `jmlx-vision/NOTICE` (source, version, SHA-256, license text), if WP1 item 9 confirms a Pillow
  delta.
- Image oracle: a new `phi3v` family (`tools/image-oracle/fixtures/phi3v/`, `run_phi3v` in
  `runner.py`) covering: a 336×336 square, a wide image (e.g. 672×336 source → `n = 2`, 457
  tokens), a tall image (336×672 source, the transpose branch, 469 tokens), a large image reaching
  `n = 4`, and a two-image input with differing sizes. Byte-exact against the pinned Torch output;
  the shared processed-pixel bins are digest-pinned in both `tools/image-oracle/provenance.json`
  and `tools/hf-reference/provenance.json` (the 7.3b WP2 pattern). `verifyImageOracleFixtureDigests`
  and the Ubuntu pure-Java job pick the family up automatically.

**WP3 — prompt expansion and tokenizer goldens.** Pure ID-space work in `jmlx-models`.

- `Phi3VPromptTokens` (the marker IDs/strings), `Phi3VPromptExpander`/`Phi3VPromptPlan`: given
  structured messages whose image parts are already geometry-planned, render each image part as
  `<|image_i|>` (i assigned in prompt order, starting at 1), split the rendered text on the marker
  pattern, tokenize **each chunk separately** through `HfTokenizer` (mirroring the reference's
  `re.split` + per-chunk tokenize, so no BPE merge crosses a marker), and replace marker `i` with
  `num_img_tokens(i)` sentinel rows. Reject: a marker index not starting at 1 or not continuous,
  marker/image count mismatch, an image part with no geometry, and an expanded-length overflow.
  Record this marker-rendering convention as the jmlx-side contract for `phi3_v` chat prompts
  (the artifact's own chat template contains no image tokens).
- Tokenizer/chat-template goldens through the real `tokenizer.json` template (the 6.3/7.3b
  convention): text-only, single image, two images with text between — expanded ID lists plus the
  per-image destination ranges, committed as fixtures.

**WP4 — `RopeSpec.Su` in `jmlx-core`.**

- New sealed variant `RopeSpec.Su(float theta, int maxPositions, int originalMaxPositions,
  float[] shortFactors, float[] longFactors)` with the WP1-item-8 semantics; factor table selection
  by the per-call `seq_len`, the per-dimension factor product, and the cos/sin scaling. Core unit
  tests are hand-computed at several positions plus the 4096/4097 boundary, and a differential
  case against the pinned Torch computation joins the existing `rope.json`-style reference
  (regenerated as a reviewed golden diff under `tools/hf-reference`, never Gradle-invoked Python).
- No `mlx_h` addition; no inventory status change (the variant is a pure Java/MLX composition over
  existing facades). `generateMlxApiInventory` must stay green unchanged.

**WP5 — model, dispatch and load guards.** `jmlx-models`.

- `Phi3VDescriptor`: strict parse of `config.json` — `model_type` exactly `phi3_v`;
  `architectures == [Phi3VForCausalLM]`; the `img_processor` block fixed to
  `name = clip_vision_model`, `model_name = openai/clip-vit-large-patch14-336`,
  `image_dim_out = 1024` (`num_img_tokens: 144` is legacy — the actual count is geometry-driven —
  so parse and ignore it, record it in the descriptor, never compute from it); the `embd_layer`
  block fixed to `projection_cls = mlp`, `use_hd_transform = true`,
  `hd_transform_order = sub_glb`, `with_learnable_separator = true` (all four, or reject); decoder
  fields (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
  `intermediate_size`, `hidden_act = silu`, `rms_norm_eps`, `rope_theta`,
  `max_position_embeddings`, `original_max_position_embeddings`, `vocab_size`,
  `tie_word_embeddings`, `bos/eos/pad_token_id`); `rope_scaling` with `type = su` and 48-entry
  `short_factor`/`long_factor` arrays (the `ArchitectureMappings` rope parse gains a `su` branch
  feeding `RopeSpec.Su`); `sliding_window` accepted-inert. Every other key is handled by name
  (accepted-inert, consumed, or rejected) so nothing falls through to an unknown-key warning.
- `Phi3VVisionTower` in `jmlx-models` (like `Idefics3VisionTower`/`Qwen3VlVisionTower`), composed
  from existing core facades only: conv2d patch embed (7.1 `MLXConv`), class token and 577×1024
  learned position embedding, 23 pre-LN blocks (7.2 `BidirectionalAttention` +
  `EncoderBlock`/explicit composition, `QUICK_GELU`), the `hidden_states[-2]` slice, the 2×2 merge
  (pure reshape/permute), newline/global separator addition, and the MLP projector. The tower
  forward runs per request in a private child scope of the request's prefill scope, evaluated and
  closed before decode (the 7.3a/7.3b per-image pattern); per-image parts are released after the
  evaluated prefill logits and retained KV; zero-padded crop slots are never computed.
- `Phi3VModel implements TextGenerationModel`: internal decoder assembled from the descriptor
  (phi3-shaped: pre-LN RMSNorm, MHA 32/32, SiLU 8192, `RopeSpec.Su`), image rows injected at the
  decoder embedding entry point by row-replacement of the sentinel rows (the 7.3a/7.3b seam; when a
  request has no images the path is bit-identical to the plain decoder — unit-verified).
  `VisionLanguageMetadata` with `InputModality` TEXT+IMAGE; the scheduler's existing
  non-`DecoderModel` rejection applies unchanged.
- Load guards: all floating weights up-cast in place to FLOAT32 and non-floating keys rejected
  (7.3a/7.3b rule); `CheckpointLoader` runs against the WP0 tensor plan with the
  `ArchitectureMappings`-side plan kept in sync with the reference generator's manifest (the
  existing generator cross-check stops on mismatch); tokenizer vocab bound against
  `vocab_size`; `processor_config.json`/`preprocessor_config.json` consistency checks mirroring
  SmolVLM's `image_seq_len`-style guard (here: `num_crops` against the descriptor).
- `TextGenerationModels.load` dispatch: strict `phi3_v` branch before the generic mapping,
  rejecting `phi3_vision` and any other variant by name.

**WP6 — Tier-A synthetic golden.** `tools/hf-reference` + `jmlx-models` tests.

- The generator (a new `--family phi3v` entry point, never Gradle-invoked, digest-pinned) builds a
  tiny `phi3_v` checkpoint with the **bundled** modeling code: it imports the pinned
  `modeling_phi3_v.py` module and reassigns its module-level
  `CLIP_VIT_LARGE_PATCH14_336_CONFIG` constant to a small `CLIPVisionConfig` (e.g. hidden 64, 3
  layers, 2 heads, patch 14, image 336 so the 24×24 grid is unchanged), leaving every code path —
  assert included — the bundled one. Decoder: hidden 128, 2 layers, 2 heads / 1 kv (GQA), head_dim
  64, SiLU 256, `rope_theta` 10000, `original_max_position_embeddings` 16,
  `max_position_embeddings` 128, untied vocab 21, plus short/long factor arrays sized for head_dim
  64. The small `original_max` makes the short→long factor switch reachable inside the golden.
- Cases (every prefill and every cached step compared against a fresh full uncached forward, the
  7.3b WP8 method): text-only; one 336×336 image (313 rows); a wide image (457 rows); a tall image
  through the transpose branch (469 rows); two images with text between; and a decode that crosses
  the synthetic rope boundary. `Phi3VGoldenTest` asserts strict `1e-4` in `float32GoldenTest`
  (`full-float32`); `Phi3VDefaultPrecisionTest` asserts a separately named default-mode bound
  derived from the locally measured TF32 error (recorded next to the Phase 6/7.3b measurements,
  never by golden regeneration).

**WP7 — lifecycle, memory and cache accounting.** `jmlx-models` tests, following the 7.3b WP7
scope: processing and tower execution sequential per image with cancellation checked between
images; only evaluated rows hoisted to the request's prefill child scope; prefill evaluates logits
and every retained KV array before that scope closes; cached decode retains only request-owned KV.
`Phi3VGenerationTest`: mixed request reuse, capacity rejection before native work, cancellation,
failure cleanup, evaluated-visual release and bounded KV growth. `Phi3VMemoryTest`: per-image peak
transient, flat active memory across repeated generations (same and different images), and a
per-token decode slope bounded by a per-token budget derived from layers, KV heads, head size and
dtype (the 7.3b WP7 budget method, not the legacy constant). FULL and capacity-bounded FULL
retention only; sliding retention rejected at the request level (7.3a/7.3b precedent).

**WP8 — Tier-B acceptance, CI row and documentation.**

- `tools/tier-b/phi3v-4b-instruct.json`: the WP0 pin plus a licensed synthetic test image
  (the SmolVLM manifest pattern: pinned PNG copied from an oracle fixture with its license), the
  required multi-image request (at least two images of different sizes with text between,
  `max_new_tokens` small), and the `reference`/`recorded_pin`/`local_evidence` sections.
- The reference generator `tools/hf-reference/phi3v_tier_b.py` (manual, digest-pinned) records CPU
  float32 eager logits and the greedy history; the manifest records the measured maximum logit
  error, the top-two gap with the acceptance `gap > 2 * epsilon + margin` arithmetic, exact greedy
  IDs and the decoded output, plus the measured allocator peak, sampled footprint peak, load and
  generation wall times. `TierBPhi3VTest` is selected by `task: phi3v-generation` and asserts
  exact greedy IDs and the end-to-end decoded output.
- `tier-b.yml` row on the self-hosted `self-macos-xl` runner (the 8.29 GB artifact and the float32
  working set exceed the standard 7 GB hosted runner; the first self-hosted run accepts the CI
  pipeline only — same acceptance semantics as the qwen3vl row — and the macOS 26 hosted execution
  remains pending, not verified). The new required suites (`Phi3VGoldenTest`,
  `Phi3VDefaultPrecisionTest`, `Phi3VGenerationTest`, `Phi3VMemoryTest`) are added to the macOS
  native job's hard-coded required-suite list in `ci.yml`. An opt-in one-off
  `benchmarkDecode` run against the Tier-B artifact (informational, no threshold, full
  reproducibility tuple) may be recorded in `req/phase6-4-benchmark.md` per the 7.3b manual
  precedent, but it is not a gate.
- Documentation: compatibility-matrix section "Phase 7.3c Phi-3.5-Vision image+text"
  (`req/phase6-compatibility.md`, including the then-outdated line-87 "Phase 7.3b / 7.3c, not
  implemented" row of `req/phase6-inference-report.md`), `req/phase6-tier-b-artifacts.md`,
  `req/phase6-tier-a-fixtures.md`, `req/phase6-inference-report.md`, `AGENTS.md` (architecture and
  status lines), the delivery table in `req/plans/phase7-plan.md`, `tools/tier-b/README.md`, and the
  findings file's §-numbered acceptance blocks.

## 5. Compatibility decisions

- **Semantics source.** The pinned revision's bundled code is the semantic source, not the
  transformers 4.57.6 core (which has no `phi3_v`). Where the bundled code is loose (legacy
  `num_img_tokens`, the `_attn_implementation` field, `sliding_window`), the reference's actual
  computed behavior decides and the decision is recorded in the findings.
- **Precision.** Weights are stored BF16 and up-cast to FLOAT32; all computation is FLOAT32,
  matching the 7.3a/7.3b vision-language rule. The RoPE tables and the cos/sin scaling are computed
  in float32 (as the reference does) and never re-derived in a lower precision.
- **Resample kernels.** Torch-exact, not Pillow-exact, for this family (WP1 item 9); provenance
  pinned in `jmlx-vision/NOTICE`. The 8-bit rounding step of the HD resize is part of the kernel
  contract, not an implementation detail.
- **Zero-padded crop slots** are never computed in Java; WP1 item 4 records why that is exact.
- **Prompt markers** are a jmlx-side rendering convention (image part → `<|image_i|>` in prompt
  order); the reference's own chat template is image-free, and the golden fixtures pin the exact
  expanded ID lists.
- **Backward compatibility.** All changes are additive: a new `RopeSpec` variant, a new
  `rope_scaling` type branch, a new family mapping, new `jmlx-vision` classes, new tests and
  fixtures. No existing overload, golden, manifest, pin or tolerance changes; the exact
  before/after bit-identity machinery of the parent's cross-cutting section is not triggered
  because no old code path changes shape.
- **Scope and streaming.** The 7.3a/7.3b per-request scope pattern and the core single-platform-
  thread rule apply unchanged; the model is single-request-per-thread with no scheduler cohorts.

## 6. Deferrals and exit gate

Deferred: the original `phi3_vision` family (phi-3-vision-128k), quantized/affine `phi3_v`
checkpoints, sliding image-prefill retention, multimodal scheduler cohorts, video, per-layer
quantization overrides, and any 8B-class artifact (none exists for this family). Each deferral is
restated in the implementation report's "remaining" section.

**Exit gate (parent row):** a `phi3v` Tier-B artifact row passes end-to-end — local eligibility
evidence recorded (exact greedy IDs, decoded output, top-two gap, peaks, wall times), the
`tier-b.yml` row green on the self-hosted runner, and the documentation set of WP8 delivered — or a
dated deferral is recorded in the findings file and the milestone closes without code. No
`Phi3VModel`, descriptor, dispatch branch or reference generator exists before its WP's PR;
no Tier-B manifest before WP8's.
