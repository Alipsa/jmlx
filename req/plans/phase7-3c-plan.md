# Phase 7.3c — LLaVA-style image+text inference (Phi-3.5-Vision)

- **Status:** draft — review round 1 (16 findings plus low-priority notes, 2026-10-27),
  review round 2 (2026-10-28: the RoPE boundary-recompute policy, the stop-token correction,
  the marker-binding wording, the relative factor-length validation, the real tiny-tower size,
  the `pre_layrnorm` key, and the low-priority items), review round 3 (2026-10-28: the
  stop-token identity — 32000 is `
`, `<|end|>` is 32007 — the boundary-recompute
  image-row retention, the 470-row Tier-A boundary config, the past-length semantics, and the
  BF16-rounded reference ordering) and review round 4 (2026-10-28: the stop token written as
  the newline token, not the two-character `\n` string, and the WP6 golden cases pinned to
  their side of the 470 boundary) applied; pending re-review. The 7.3c
  go/deferral discovery was performed on 2026-10-27:
  GO — https://huggingface.co/microsoft/Phi-3.5-vision-instruct
  (revision `12b77fb40b63a2c73c68243d3f767aab688a1b2a`, MIT) is a small, fully licensed, public
  LLaVA-style artifact (CLIP ViT-L/14@336 tower + MLP projector + decoder). WP0 formalizes the pin
  and the tensor inventory; WP1 probes the reference. No 7.3c code before this plan is reviewed.
- **Parent:** `req/plans/phase7-plan.md` §7.3c and the delivery-order row "7.3c LLaVA-style
  (optional) | `jmlx-models` | 7.3a | Tier-B artifact, or a dated deferral"; satisfies
  `req/full-roadmap.md` §Phase 7 (milestone 3 breadth, the last open Phase 7 implementation row).
- **Dependencies:** 7.1 core layers (convolution, normalization, `QUICK_GELU` and `GELU`
  activations), 7.2 `BidirectionalAttention`/`EncoderBlock`, 7.3a multimodal request contracts
  (`withImages`, `InputModality`, per-request vision scope, structured chat validation) and 7.3b
  per-request lifecycle/injection patterns; Phase 6.3's `rope_scaling` parse infrastructure in
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
MLP projector (exact `GELU`, unlike the CLIP blocks' `quick_gelu`), and a plain pre-LN RMSNorm
decoder with per-dimension NTK-Su RoPE (`su` is the old name for `longrope`). It is the only small,
fully MIT-licensed public LLaVA-style instruct artifact found: the `llava-hf` 7B/11B releases are
Llama-terms gated (the same Tier-B blocker as Gemma v1), PaliGemma is Gemma-terms, and the SmolVLM
line is already covered by 7.3a. If WP0 or WP1 invalidates the pin or finds a blocking reference
divergence, record a dated deferral in `req/plans/phase7-3c-reference-findings.md` and close the
milestone without code (the parent's alternative gate).

`phi-3-vision-128k` is **not** a separate family: it carries the same `model_type` `phi3_v` and
the same architecture (`Phi3VForCausalLM`, the same `img_processor`/`embd_layer` blocks),
differing only in the rope factors, `sliding_window` (131072) and null `pad_token_id`/dropout
fields. The descriptor (§4 WP5) handles that shape explicitly, but the support and verification
claims stay pinned to the 4.2B artifact — and there is no name-based exclusion to implement,
because no `phi3_vision` model type exists.

Float safetensors only, up-cast to FLOAT32 in place (the 7.3a/7.3b vision-language rule): reject
non-floating or quantized checkpoints explicitly. Exclude video, remote image fetching, training,
image generation, public model batching, multimodal scheduler cohorts, quantized KV, sliding
image-prefill retention, per-layer quantization overrides and MLX affine loading (no 4-bit
artifact exists for this family). No native-pin change or generated-binding edit is planned: every
primitive needed (conv2d, layer norm, `QUICK_GELU`, `GELU`, bidirectional attention,
reshape/permute) is already bound and facade'd from 7.1/7.2.

## 2. Repository baseline and prerequisite gate

- `jmlx-models` composes SmolVLM (`SmolVlmModel`) and Qwen3-VL (`Qwen3VlModel`) as
  `TextGenerationModel` objects: a strict descriptor parse, a `TextDecoder extends DecoderModel`
  (or descriptor-assembled decoder), a dedicated vision tower, a pure prompt expander that turns
  each image part into a contiguous block of sentinel rows, per-request vision scopes evaluated
  and closed after prefill, and a row-replacement injection at the decoder embedding entry point.
  Reuse that composition; do not invent a new one.
- `TextGenerationModels.load` dispatches `t5`, `idefics3`, and strict `qwen3_vl` before the
  generic decoder mapping, then switches on `model_type`
  (llama/qwen2/qwen3/mistral/phi3/gemma/mixtral). `phi3_v` joins the strict pre-mapping
  dispatches: the composite loader owns the config parse.
- `ArchitectureMappings` parses `rope_scaling` objects with types `linear`, `dynamic`, `llama3`,
  `yarn` (around line 646) and carries per-family `sliding_window` treatment. The existing `phi3`
  family entry (around line 104) is `WindowPolicy.USE` + `fusedProjections()`. The new `phi3_v`
  entry must mirror the fused flags — the bundled `Phi3MLP` uses a fused `gate_up_proj` and the
  bundled attention a fused qkv projection — while setting `WindowPolicy.IGNORE`, because the
  bundled decoder never reads `sliding_window` and `phi3`'s `USE` treatment must not leak.
- `RopeSpec` (sealed) has `Base`, `Linear`, `DynamicNtk`, `Llama3`, `Yarn`; `Su` is a new variant.
  Integration facts (`RopeSpec.java`): the default `isStatic()` is `!(this instanceof
  DynamicNtk)` (`:69`) — `Su` must override it to `false` so the short/long factor table is
  re-selected per call; the cos/sin multiplier reuses the existing `attentionScaling()` hook (the
  YaRN precedent, `:79`); `frequencies()` computes in `double` (`:19`) and casts to float32, which
  is a stated tolerance against the reference's float32 computation, not a bit-exact claim.
- `Activation` already includes `QUICK_GELU` (7.1) and `GELU`, so the tower (quick) and the
  projector (exact) need no new activation.
- The 7.3a/7.3b request contract already carries image parts through structured chat messages,
  validates them, and threads `InputModality` into `ModelMetadata`
  (`VisionLanguageMetadata`). The Tier-B manifest shape (`repository`/`revision`/`task`/`license`/
  `files`/`images`/`requests`/`reference`/`recorded_pin`/`local_evidence`) is the 7.3a/7.3b one.
- `tools/hf-reference` is a generate-only, hash-locked CPython 3.12 tool (Torch 2.6.0 CPU,
  `transformers==4.57.6`, float32, eager attention) and has no `torchvision` pin. `transformers`
  4.57.6 has **no native `phi3_v` model**: this artifact runs its bundled trust-remote-code
  `modeling_phi3_v.py`/`processing_phi3_v.py`, and the bundled processor imports `torchvision`.
  The reference toolchain must therefore gain a `torchvision` pin and a hash-verified
  bundled-code execution path (WP1).
- At implementation start, record the actual base commit, the completed prerequisite suites and
  the merged state of every contract named above. Public changes after this plan's publication
  must remain additive, with the one sealed-hierarchy exception stated in §5.

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
  `sample_inference.py` 6,778. SHA-256 every file; verify the index covers every tensor key
  exactly once (the 7.3b 8B stale-index finding makes this a required check, not an assumption).
- Record the complete tensor-key inventory grouped by owner: decoder text weights, `lm_head`, the
  vision tower under the CLIP-vision naming (including the unused final block and
  `post_layernorm`), and the projector/separator parameters (`glb_GN`, `sub_GN`,
  `img_projection.{0,2}` — exact prefixes come from the index dump, not from this plan's memory).
- Tokenizer: a SentencePiece-style BPE — a `Prepend(▁) + Replace(' ' → ▁)` normalizer and
  **no pre-tokenizer** (verified in `tokenizer.json`; it is not the byte-level-with-pre-tokenizer
  shape the 7.3b family has). `byte_fallback: true` and `fuse_unk: true` are **BPE model
  flags**, not a Precompiled normalizer — the 7.2 charsmap code does not apply to this
  artifact; the decoder is `Replace` + `ByteFallback` + `Fuse` + `Strip(start=1)`, which the
  decoded-output assertion depends on. `TemplateProcessing` prepends `<s>` (id 1) to every
  encoding, `add_bos_token` is true, and the vocabulary is 32064. The chat template is a plain
  Llama-style Jinja string (`<|role|>\n content <|end|>\n`); image parts do not appear in the
  template — they enter as `<|image_i|>` markers in message text (see §4 WP3). Verify the
  template renders through `jmlx-jinja` and that the marker strings are not special tokens.
  Special-token facts (from `tokenizer_config.json` and `special_tokens_map.json`): the eos
  **and** pad token is `
` (id 32000, `special: true`) — a special added token, distinct from the `<0x0A>`
  byte-fallback token (id 13) through which a newline in the text encodes and decodes; 32000
  itself decodes to the empty string (pinned-venv probe), so the stop is on the token id
  32000 and a newline in the output (13) does not stop generation. `<|end|>` is id 32007 —
  the token the chat template ends assistant turns with; `<s>` is 1; `</s>` is 2 and is
  **not** a special token; `generation_config.json` carries `eos_token_id: 2` **and**
  `pad_token_id: 32000`.
  `<|image|>` (id 32044) is a special token the marker regex `<\|image_\d+\|>` never matches —
  it creates no image slot in the reference; user text containing it is passed through, not
  rejected (WP3 records the decision).

**WP1 — reference probes.** Findings go to a new `req/plans/phase7-3c-reference-findings.md` in
the 7.3b findings style (native pins, host, precision, commands, shapes, exact conclusions). It
must:

1. Add the `torchvision` pin to the hash-locked `tools/hf-reference` venv (the torchvision line
   matching Torch 2.6.0, wheel SHA-256s for both wheel platforms, recorded in
   `provenance.json`) and pin the four bundled-code files' SHA-256s (from WP0) so a generator run
   verifies them before use.
2. Confirm the execution-mode gate (a code-reading conclusion — confirm by execution): the
   bundled code calls transformers cache APIs that do not exist in the pinned
   `transformers==4.57.6` (`get_usable_length` at `modeling_phi3_v.py:668` and `:1417`;
   `seen_tokens`/`get_max_length` at `:1653`–`:1661`). Line `:1417` runs whenever `use_cache` is
   true (the default), so even a fresh prefill fails. Every generator therefore runs an
   **uncached full forward with `use_cache=False` and a manual greedy loop**; no compatibility
   shim and no separate transformers pin.
3. Run the bundled model on CPU, float32, **explicit eager attention**: the artifact config's
   `_attn_implementation` is `flash_attention_2` for the decoder, and the vision tower runs
   **SDPA** by default (a bare `CLIPVisionConfig` defaults to `sdpa`, and an outer eager
   override does not reach it — probed in the pinned venv). The generator must set **and assert**
   `_attn_implementation = eager` on both the model and the tower, and record the SDPA default
   in the findings.
4. Pin the preprocessing pipeline exactly: per image — `convert_to_rgb`, HD transform with
   `hd_num = num_crops = 4` (transpose when `width < height`; largest `scale` with
   `scale * ceil(scale / ratio) <= 4`; `new_w = int(scale * 336)`, `new_h = int(new_w / ratio)` —
   a double `int()` truncation; resize through PIL (see item 12) to `(new_h, new_w)`; top/bottom
   white-255 padding of the 8-bit image to a multiple of 336, top `floor((target - h) / 2)` and
   the bottom the remainder, no left/right padding; untranspose); `ToTensor` + `Normalize`,
   computed as the division `(x - mean) / std`, not a reciprocal multiply; the global image is a
   torch **bicubic** `F.interpolate` (`align_corners=False`, no antialias, `A = -0.75`) to
   336×336 **on the normalized float32 tensor**; crop slots are `[global, sub crops in h-major
   order]` zero-padded to `num_crops + 1 = 5`; `image_sizes` is the padded `(h, w)`. The
   reference never calls `exif_transpose` → the port decodes with
   `ExifOrientationPolicy.IGNORE`. Pinned truncation example: a 300×301 source resizes to
   669 rows and pads 1 top / 2 bottom.
5. Pin the per-image token count `(n + 1) * 144 + (h336 + 1) * 12 + 1` with `n = h336 * w336`
   (padded grid in units of 336): with `num_crops = 4` the scale loop always picks
   `scale >= 2`, so a 336×336 square resizes to 672×672 (`n = 4`, **757** rows — 313 is
   unreachable); 672×336 → 457; 336×672 (transpose branch) → 469. The reachable per-image
   counts are {457, 469, 601, 625, 745, 757, 781}, maximum 781 at `h336 = 4, w336 = 1`.
6. Pin the tower: the bundled `CLIP_VIT_LARGE_PATCH14_336_CONFIG` (hidden 1024, 24 layers, 16
   heads, `quick_gelu`, eps 1e-5, patch 14, image 336); a bias-free patch conv; a class token
   plus a single 577-entry learned position table; the pre-encoder norm applied **before** the
   encoder (the input hidden state enters the first block already normalized) lives under the
   checkpoint key `pre_layrnorm` — **HF's own typo**, verified in the artifact's 592-key index
   (which has no `position_ids` buffer) and in the pinned venv's `modeling_clip.py:722`; the
   tensor plan and the load guard must use the exact misspelled key, while the never-executed
   post-encoder norm keeps the correctly spelled `post_layernorm`;
   `hidden_states[-2]` — the output **after 23 blocks**, class token still in the attended
   sequence — with the class token discarded → 576×1024 per crop. The 24th block and
   `post_layernorm` weights exist in the checkpoint but are never executed → the tensor plan
   accepts-and-ignores them by name. Zero-padded crop slots are computed by the reference but
   never read, so the Java port may skip them (record that they cannot influence used outputs:
   crops are processed independently).
7. Pin the high-resolution transform: 2×2 merge of the 24×24 grid to 144×4096 — the bundled
   `reshape_hd_patches_2x2merge` asserts `L == 24 * 24 and C == 1024`
   (`modeling_phi3_v.py:307`), and `glb_GN`/`sub_GN` are sized `image_dim_out * 4`, so the width
   is fixed at 1024 even for the tiny tower (§4 WP6); the 4096 channel order of a merged row is
   `(h_pool, w_pool, C)`, and the sub grid is laid out as `(h_crop, 12, w_crop, 12)` — rows run
   across all 12×12 blocks of a crop row (`modeling_phi3_v.py:310`–`:320`), so a transposed
   merge would surface only as a golden failure with no spec to check against; `sub_GN`
   newline rows appended after every row of the sub grid
   (`(h336 * 12) * (w336 * 12 + 1)` rows); per image the block is
   `[sub rows, glb_GN (1 row), global rows (12 * 13)]`; the projection is
   `Linear(4096 → 3072) → GELU (exact, `nn.GELU`, `modeling_phi3_v.py:174`–`:179`) →
   Linear(3072 → 3072)` over the concatenated rows — the CLIP blocks keep `quick_gelu`.
8. Pin the text path: `re.split(r"<\|image_\d+\|>", text)` — chunks are tokenized
   **separately** (BPE merges never cross a marker boundary), and each marker `i` expands to
   `num_img_tokens[i-1]` sentinel rows. The reference checks only that the unique marker ids
   are exactly `1..n` and that the unique count equals the image count
   (`processing_phi3_v.py:430`–`:433`); it does **not** check the total marker count. Images
   are bound by **position**, not marker id (`modeling_phi3_v.py:228`, `:239`–`:242`: the
   sentinel rows' `nonzero` positions receive the projected rows via `index_put`), so a
   duplicated marker emits 2N sentinel rows against an N-row projection and the reference
   **fails loudly** on the shape mismatch, while out-of-order markers (`<|image_2|>` before
   `<|image_1|>`) silently misbind — the two images are swapped at equal sizes, and at
   different sizes the features straddle both blocks. The jmlx-side rejections are a
   documented hardening, not reference behavior (§4 WP3). The reference attention mask is all
   ones.
9. Pin the tokenizer behavior: the prompt builder sets `offset = 0`
   (`processing_phi3_v.py:444`–`:446`) and strips nothing, so every chunk — including chunks
   after an image block — starts with the per-encoding `<s>` (a mid-sequence BOS, followed by
   `▁` from the space replacement). The empty-chunk case (a trailing or adjacent marker) is
   **unverified** — record the reference's actual ids for it before WP3 asserts anything.
10. Pin the decoder: pre-LN blocks (`RMSNorm(1e-5)` → MHA → residual; `RMSNorm(1e-5)` → SiLU
    MLP 8192 → residual), 32 layers, 32/32 heads, head_dim 96, scale `1/sqrt(96)`, causal,
    fp32 softmax; `sliding_window: 262144` is present in the config but never read by the
    attention code (accepted-inert, like the qwen2/qwen3 window keys); untied `lm_head` over
    vocab 32064; `rope_theta` 10000, `max_position_embeddings` 131072,
    `original_max_position_embeddings` 4096.
11. Pin NTK-Su RoPE exactly: per forward call `seq_len = kv_seq_len`
    (`modeling_phi3_v.py:669`) — the prefill length for a fresh prefill, the cached key length
    during cached decode; factor table `long_factor` iff `seq_len > 4096` else `short_factor`
    (48 per-dimension entries each); `inv_freq[i] = 1 / (f[i] * 10000 ** (2i / 96))`; a global
    `scaling = sqrt(1 + ln(131072/4096) / ln(4096))` multiplies **both** cos and sin; the
    reference computes in float32 and casts to the model dtype (against which `RopeSpec`'s
    double→float32 is the stated tolerance, §2). **Boundary recompute:** the bundled code
    deliberately discards the cache and forces a full recompute the first time a sequence
    crosses the boundary — `modeling_phi3_v.py:1650`–`:1655` drops `past_key_values` to `None`
    when `input_ids.shape[1] >= original_max_position_embeddings + 1` and the past length is
    `<= original_max_position_embeddings`. Past length is the number of cached positions;
    the reference therefore generates as if **every** key uses the long table once the
    sequence passes the boundary — including the keys written before it — and a "cached
    keys keep the table that wrote them" behavior would be a jmlx-only divergence, not
    reference behavior. Under the pinned 4.57.6 the `:1653` past-length read takes the
    `seen_tokens`/`get_max_length` branch, because `isinstance(past_key_values, Cache)` is
    true (`DynamicCache` subclasses `Cache`, `cache_utils.py:910`) while `seen_tokens` does
    not exist, so that branch raises `AttributeError` if executed — part of item 2's failure
    — and the non-Cache `past_key_values[0][0].shape[2]` fallback never runs; under the
    generator's uncached mode (`use_cache=False`) `past_key_values` is `None` and none of
    `:1650`–`:1661` runs at all. The boundary flow is therefore a code-reading conclusion
    plus the observed failure (item 13), not an executed path. The generator's per-step fresh
    forward of the full history embodies the long-table-from-crossing semantics that jmlx's
    cached decode must reproduce — including re-running the vision tower at the crossing,
    since `prepare_inputs_for_generation` passes `pixel_values`/`image_sizes` on every step
    (`:1699`–`:1707`) and `Phi3ImageEmbedding.forward` (`:228`–`:242`) re-injects the image
    rows. §5 decides the policy, and WP6 golden-tests the recompute and the image-row
    restoration.
12. Split the resample risk: the HD resize is **PIL** — torchvision's `F.resize` on a PIL input
    delegates to `Image.resize(..., BILINEAR)` — so the existing Pillow-exact bilinear kernel
    applies **if** the hand-off is confirmed (item 13; torchvision is not yet in the venv, so
    this is unverified). Only the global image's float32 bicubic needs a new Torch-exact kernel.
    Both kernels' provenance is pinned in `jmlx-vision/NOTICE` exactly as the 7.3a LANCZOS port
    pins its Pillow sources.
13. Confirm by execution what this plan marks unverified: the PIL hand-off (item 12), the
    cache-API failure (item 2), and the empty-chunk tokenization (item 9). The item-11
    boundary flow is recorded as a **code-reading conclusion** plus the failure observed when
    it is executed: the `:1653` past-length read takes the `seen_tokens` branch (the
    `isinstance(past_key_values, Cache)` check is true under 4.57.6) and raises
    `AttributeError`, while `sample_inference.py`'s own `generate` cannot complete there at
    all (item 2), so no successful cached decode exists in the pinned venv to record the flow
    against; the semantics jmlx mirrors are "past length = number of cached positions" (item
    11). The `torchvision` install is precisely what makes the first two checkable.
14. Pin the stop-token policy (answerable from the artifact; the WP1 probe only confirms):
    every `generate` call in `sample_inference.py` passes
    `eos_token_id = processor.tokenizer.eos_token_id` — **32000** (`
`, the tokenizer's
    own eos, which is also its pad token — **not** `<|end|>`, which is id 32007) — overriding
    `generation_config.json`'s
    `eos_token_id: 2` (`</s>`, not a special token). Generation therefore stops on
    **{32000}** (or `max_new_tokens`); with that stop set the model typically emits
    `<|end|>` (32007) **before** the stop, so the decoded output contains it unless special
    tokens are stripped — the Tier-B manifest records which it does (§4 WP8). The manifest
    pins stop = {32000} for the decoded-output assertion, and the probe records whether
    `sample_inference.py` adds anything else. How `Phi3VModel` documents its own default
    stop set is decided in §5.
15. Record the `phi-3-vision-128k` config diff (rope factors, `sliding_window` 131072, null
    `pad_token_id`/dropout) if the artifact is accessible; its `num_crops` is **unverified**.
    No support claim is made for it (§1).

**Gate:** the findings document exists, every claim above is recorded with its probe evidence,
the `torchvision` pin and bundled-code hashes are in `provenance.json`, the four unverified
claims (items 2, 9, 11, 12) are resolved (item 11 by code reading plus the observed failure,
per item 13), and no claim depends on unpinned remote state.

**Oracle provenance and fixture tiers.** The reference oracle is the pinned artifact revision's
bundled code under the hash-locked `tools/hf-reference` venv (Torch 2.6.0 CPU + the WP1
`torchvision` pin, `transformers==4.57.6` present but not the semantic source for this family),
float32, eager attention, uncached full forwards with manual greedy loops (WP1 item 2). Tier-A is
an in-repo synthetic checkpoint with no external download and no credentials; Tier-B is the
public MIT artifact, also credential-free. Python execution stays opt-in; ordinary builds gain no
Python prerequisite.

## 4. Implementation work packages

Ordered so foundations land before consumers; each WP is its own PR per the parent's delivery
rule.

**WP2 — `jmlx-vision` preprocessing.** Pure Java 21, no native dependency.

- `Phi3VProcessorConfig` (parses `preprocessor_config.json`: the `num_crops`, CLIP
  `image_mean`/`image_std` and `do_convert_rgb` keys the reference reads, plus the standard
  inert keys (`auto_map`, `image_processor_type`, `processor_class`, and the legacy
  `num_img_tokens`, which the reference never reads) accepted-and-ignored per the 7.3b
  full-dump precedent; value domains validated as the reference validates them; decode policy
  `DROP_ALPHA` + `ExifOrientationPolicy.IGNORE` per WP1 item 4), `Phi3VGeometryPlan` (HD
  transform math: transpose branch, scale-selection loop, the double-truncating `new_h`,
  resized and padded sizes, crop counts, per-image token count, and explicit rejection of
  extreme aspect ratios in **both** orientations: with `scale = 4` the resized long edge is
  1344, so an aspect ratio above 1344 truncates `new_h` (`new_w` in the transpose branch) to
  0 (a zero-sized resize fails in PIL; whether the same failure surfaces through the
  torchvision hand-off is unverified) — the plan rejects the image before any resize,
  regardless of the reference's failure behavior, with a unit test for each orientation),
  `Phi3VImageProcessor` (the full
  pipeline of WP1 item 4, from a decoded `RgbImage`), and `Phi3VImageProcessorResult` (per-slot
  pixel values — global first, then sub crops in h-major order, zero-padded to five slots —
  `image_sizes`, `num_img_tokens`).
- Resample kernels: the HD bilinear **reuses** the existing Pillow-exact kernel, conditional on
  WP1 item 13 confirming the PIL hand-off; the float32 bicubic (`F.interpolate` semantics:
  `align_corners = false`, no antialias, `A = -0.75`) is the one new Torch-exact kernel, with
  provenance (source, version, SHA-256, license text) pinned in `jmlx-vision/NOTICE` exactly as
  the 7.3a LANCZOS port pins its sources.
- Image oracle: a new `phi3v` family (`tools/image-oracle/fixtures/phi3v/`, `run_phi3v` in
  `runner.py`) covering: a 336×336 square (672×672 after resize, `n = 4`, **757** rows), a wide
  image (672×336 source, `n = 2`, 457 tokens), a tall image (336×672 source, the transpose
  branch, 469 tokens), and a two-image input with differing sizes. There is no separate "large
  `n = 4`" fixture: with `num_crops = 4` every aspect ratio below 2 reaches `n = 4`, and the
  square already covers it. Byte-exact against the pinned reference output; the shared
  processed-pixel bins are digest-pinned in both `tools/image-oracle/provenance.json` and
  `tools/hf-reference/provenance.json` (the 7.3b WP2 pattern). `verifyImageOracleFixtureDigests`
  and the Ubuntu pure-Java job pick the family up automatically.

**WP3 — prompt expansion and tokenizer goldens.** Pure ID-space work in `jmlx-models`.

- First, verify that `HfTokenizer` reproduces this tokenizer shape byte-for-byte — BPE with
  the `byte_fallback`/`fuse_unk` model flags and the `Replace` + `ByteFallback` + `Fuse` +
  `Strip(start=1)` decoder, no pre-tokenizer, `Prepend(▁) + Replace` normalizer (there is no
  Precompiled component — the 7.2 charsmap code does not apply here), `TemplateProcessing` BOS
  — which differs from the 7.3b byte-level-with-pre-tokenizer shape.
  If a component gap is found, extend `jmlx-tokenizer` first (its own PR), since WP3 consumes
  it.
- `Phi3VPromptTokens` (the marker IDs/strings), `Phi3VPromptExpander`/`Phi3VPromptPlan`: given
  structured messages whose image parts are already geometry-planned, render each image part as
  `<|image_i|>` (i assigned in prompt order, starting at 1), split the rendered text on the
  marker pattern, tokenize **each chunk separately** through `HfTokenizer` (mirroring the
  reference's `re.split` + per-chunk tokenize, so no BPE merge crosses a marker), and replace
  marker `i` with `num_img_tokens(i)` sentinel rows. Each chunk's encoding carries the BOS
  (`<s>`, id 1): the reference's `offset = 0` strips nothing, so the goldens pin the
  mid-sequence `<s>▁…` start after every image block, and the empty-chunk case (trailing or
  adjacent markers) is asserted only after WP1 item 13 records the reference's actual ids for
  it.
- Rejections (a documented jmlx-side hardening over the reference's two asserts): a marker
  index not starting at 1 or not continuous, marker/image count mismatch, duplicate marker
  indices (the reference fails loudly on the sentinel-count shape mismatch) or out-of-order
  marker indices (the reference silently misbinds — WP1 item 8), user text already containing
  a literal `<|image_\d+|>` (a phantom image slot through `re.split`), an image part with no
  geometry, and an expanded-length overflow. A literal `<|image|>` (id 32044) in user text is
  **not** rejected: the marker regex never matches it and it creates no image slot in the
  reference, so it is passed through as an ordinary token (recorded in the findings). Record
  this marker-rendering convention as the jmlx-side contract for `phi3_v` chat prompts (the
  artifact's own chat template contains no image tokens).
- Tokenizer/chat-template goldens through the real `tokenizer.json` template (the 6.3/7.3b
  convention): text-only, single image, two images with text between, a trailing marker, and two
  adjacent markers — expanded ID lists plus the per-image destination ranges, committed as
  fixtures.

**WP4 — `RopeSpec.Su` in `jmlx-core`.**

- New sealed variant `RopeSpec.Su(float theta, int maxPositions, int originalMaxPositions,
  float[] shortFactors, float[] longFactors)` with the WP1-item-11 semantics: factor table
  selection by the per-call sequence length (`seq_len > originalMax → long`), the per-dimension
  factor product, and the cos/sin multiplier through the existing `attentionScaling()` hook (the
  YaRN precedent; the reference multiplies both cos and sin by the same constant).
  `isStatic()` is overridden to `false` so the table is re-selected per call rather than shared
  once. `frequencies()` continues to compute in `double` and cast to float32 — a stated
  tolerance, not bit-exactness.
- Core unit tests are hand-computed at several positions plus the 4096/4097 boundary (short at
  4096, long at 4097), and a differential case against the pinned Torch computation joins the
  existing `rope.json`-style reference (regenerated as a reviewed golden diff under
  `tools/hf-reference`, never Gradle-invoked Python).
- No `mlx_h` addition; no inventory status change (the variant is a pure Java/MLX composition
  over existing facades). `generateMlxApiInventory` must stay green unchanged. Adding a sealed
  variant is the one source-compatibility exception in this milestone — the `permits` clause
  and the default-method `instanceof` chains; no in-repo exhaustive `switch` over `RopeSpec`
  exists (§5).

**WP5 — model, dispatch and load guards.** `jmlx-models`.

- `Phi3VDescriptor`: strict parse of `config.json` — `model_type` exactly `phi3_v`;
  `architectures == [Phi3VForCausalLM]`; the `img_processor` block fixed to
  `name = clip_vision_model`, `model_name = openai/clip-vit-large-patch14-336`,
  `image_dim_out = 1024` (`num_img_tokens: 144` is legacy — the actual count is
  geometry-driven — so parse and ignore it, record it in the descriptor, never compute from it;
  optional `layer_idx`/`type_feature` are accepted only at their default values or rejected);
  the `embd_layer` block fixed to `projection_cls = mlp`, `use_hd_transform = true`,
  `hd_transform_order = sub_glb`, `with_learnable_separator = true` (all four, or reject), with
  `embedding_cls` accepted-and-recorded (inert in the port); decoder fields (`hidden_size`,
  `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`,
  `hidden_act = silu`, `rms_norm_eps`, `rope_theta`, `max_position_embeddings`,
  `original_max_position_embeddings`, `vocab_size`, `tie_word_embeddings`, `bos/eos/pad_token_id`);
  `rope_scaling` with `type = su` and `short_factor`/`long_factor` arrays of exactly
  `headDim / 2` entries (`headDim = hidden_size / num_attention_heads` — 48 for the 4.2B
  artifact's head_dim 96, 32 for the WP6 tiny decoder's head_dim 64; the bundled
  `configuration_phi3_v.py:187`, `:204`, `:215` validates the same relative length and exactly
  three keys), with the `su` key allow-list exactly `{type, short_factor, long_factor}`. The
  factor-length validation sits next to the existing `DynamicNtk` check in the
  `ArchitectureDescriptor` compact constructor (`:34`); the `ArchitectureMappings` rope parse
  gains a `su` branch feeding `RopeSpec.Su` behind an explicit family gate (e.g. an
  `acceptsSuRope()` builder flag set only by the `phi3_v` entry — `parseRope` is shared by all
  families, and older Phi-3-mini-128k configs really did use `type: su`, §5); any other type —
  including `yarn`, which the bundled config class accepts — is rejected by name;
  `sliding_window` accepted-inert under the family's `WindowPolicy.IGNORE`. The descriptor
  explicitly handles the known `phi-3-vision-128k` shape (different rope factors,
  `sliding_window: 131072`, null `pad_token_id`/dropout) with no name-based exclusion — no such
  model type exists to reject — while support claims stay pinned to the 4.2B artifact. Every
  other key is handled by name (accepted-inert — `_attn_implementation`, `auto_map`,
  `torch_dtype`, `transformers_version`, the dropout fields, `use_cache` — consumed, or
  rejected) so nothing falls through to an unknown-key warning.
- `Phi3VVisionTower` in `jmlx-models` (like `Idefics3VisionTower`/`Qwen3VlVisionTower`),
  composed from existing core facades only: bias-free conv2d patch embed (7.1 `MLXConv`), the
  class token and the single 577×1024 learned position table, the pre-encoder norm (checkpoint
  key `pre_layrnorm` — HF's typo, WP1 item 6) before the
  encoder, 23 pre-LN blocks (7.2 `BidirectionalAttention` + `EncoderBlock`/explicit
  composition, `QUICK_GELU`), the `hidden_states[-2]` slice (the output after 23 blocks), the
  2×2 merge (pure reshape/permute), the newline/global separator addition, and the MLP
  projector (exact `GELU`). The tower forward runs per request in a private child scope of the
  request's prefill scope, evaluated and closed before decode (the 7.3a/7.3b per-image pattern);
  per-image parts up to the projector are released after the
  evaluated prefill logits and retained KV, but the **evaluated projection rows** — the rows
  injected at the sentinel positions, at most 781 × 3072 × 4 B ≈ 9.6 MB per image — are hoisted
  to the request scope (outliving the prefill child scope) and retained until the generation
  ends, because the boundary recompute re-prefills the full history and must re-inject them
  (WP1 item 11); zero-padded crop slots are never computed.
- `Phi3VModel implements TextGenerationModel`: internal decoder assembled from the descriptor
  (phi3-shaped: pre-LN RMSNorm, MHA 32/32, SiLU 8192, fused qkv and gate/up projections,
  `RopeSpec.Su`), image rows injected at the decoder embedding entry point by row-replacement
  of the sentinel rows (the 7.3a/7.3b seam; when a request has no images the path is
  bit-identical to the plain decoder — unit-verified). The generation loop implements the
  boundary recompute (WP1 item 11, §5): at the first decode step whose ending position exceeds
  `originalMaxPositions` while the accumulated past length — the number of
  cached positions, read from jmlx's own cache length — is `<= originalMaxPositions`, it
  discards the request's KV cache and re-prefills the full history with the long
  factor table, **re-injecting the retained per-image rows** at the sentinel positions (the
  reference re-runs the tower at the crossing — `prepare_inputs_for_generation` passes
  `pixel_values`/`image_sizes` on every step, `:1699`–`:1707`), mirroring
  `modeling_phi3_v.py:1650`–`:1655`; a unit test covers both the
  trigger (a text-only history crossing the boundary) and the non-trigger (past length already
  over the boundary at the first decode step — e.g. the 757-row image case under the Tier-A
  config — no recompute, the table never changes). `VisionLanguageMetadata` with
  `InputModality` TEXT+IMAGE; the scheduler's existing non-`DecoderModel` rejection applies
  unchanged.
- Load guards: all floating weights up-cast in place to FLOAT32 and non-floating keys rejected
  (7.3a/7.3b rule); the unused 24th-block and `post_layernorm` weights accepted-and-ignored by
  name; `CheckpointLoader` runs against the WP0 tensor plan with the `ArchitectureMappings`-side
  plan kept in sync with the reference generator's manifest (the existing generator
  cross-check stops on mismatch); tokenizer vocab bound against `vocab_size`;
  `processor_config.json`/`preprocessor_config.json` consistency checks mirroring SmolVLM's
  `image_seq_len`-style guard (here: `num_crops` against the descriptor); expanded-length
  overflow rejected before any native work (the 7.3a capacity pattern, `SmolVlmModel`).
- `TextGenerationModels.load` dispatch: strict `phi3_v` branch before the generic mapping (no
  sibling-name rejection — no other model type claims this architecture).

**WP6 — Tier-A synthetic golden.** `tools/hf-reference` + `jmlx-models` tests.

- The generator (a new `--family phi3v` entry point, never Gradle-invoked, digest-pinned)
  builds a tiny `phi3_v` checkpoint with the **bundled** modeling code: it imports the pinned
  `modeling_phi3_v.py` module and reassigns its module-level
  `CLIP_VIT_LARGE_PATCH14_336_CONFIG` constant to a small `CLIPVisionConfig`, leaving every
  code path — asserts included — the bundled one. The tower keeps `hidden 1024`: the bundled
  `reshape_hd_patches_2x2merge` (`modeling_phi3_v.py:307`) asserts `L == 24 * 24 and C == 1024`,
  and `glb_GN`/`sub_GN` are sized `image_dim_out * 4`, so the width cannot shrink; instead the
  MLP is small and the depth is 2–3 layers, which keeps the bundled assert unpatched. The real
  size at width 1024: attention alone is `4 × (1024² + 1024)` parameters ≈ **17.3 MB float32
  per CLIP layer**, and the patch embed, the 577-entry position table, the separators and the
  4096→128 projector add ≈ 7.0 MB — 2 layers ≈ 42 MB, 3 layers ≈ 59 MB float32
  (the largest existing tiny checkpoint is 17.7 MB; `goldens/` totals 67 MB). The generator
  therefore stores the tiny checkpoint as **BF16** (≈ 21 MB / ≈ 29.5 MB), halving the
  committed size and exercising the BF16→FLOAT32 in-place up-cast the 4.2B artifact's load
  path performs — and it computes every reference forward from those BF16-rounded weights: the
  freshly initialized weights are rounded to BF16 **before** saving and before any golden is
  computed, up-cast back to float32 in Torch, and only then are the forwards run, so the strict
  1e-4 gate compares Java's load path (BF16 → FLOAT32) against a reference computed from the
  identical rounded weights, not the pre-rounding float32; the ordering is recorded in the
  findings and the digest pin.
  Decoder: hidden 128, 2 layers, 2 heads / 1 kv (GQA), head_dim 64, SiLU 256,
  `rope_theta` 10000, `original_max_position_embeddings` 470, `max_position_embeddings` 2048,
  untied vocab 21, plus short/long factor arrays sized for head_dim 64.
- `max_position_embeddings` must be at least 2048: the 7.3a capacity pattern
  (`SmolVlmModel`) rejects an expanded prompt that exceeds the context length, and the
  smallest image case is 457 rows — 128 would reject every image case. `original_max = 470` is
  chosen just above the 457-row wide image, so one config covers all three regimes: a wide
  image plus short text crosses the boundary a few decode steps after prefill, the 757-row
  case starts on the long table, and text-only prompts stay on the short table. With it, the
  pinned scaling constant is `sqrt(1 + ln(2048/470) / ln(470))` (≈ 1.11321).
- Cases (every prefill and every cached step compared against a fresh full **uncached**
  forward — the reference's `use_cache=False` execution mode, WP1 item 2 — with the manual
  greedy loop, the 7.3b WP8 method); every case pins its expanded prompt length (the
  prepended `<s>`, the template tokens, the text tokens and the image rows) and therefore
  which side of 470 it starts on — the recompute pins (fired or not) are part of the case
  spec, not accidental: text-only (short table; **prompt + max_new_tokens ≤ 470**, so the
  factor table is constant across the whole generation and the recompute never fires); a
  wide image (457 rows) plus **short** text, chosen so the expanded prompt stays below 470
  while prompt + max_new_tokens exceeds it — the decode **crosses** the synthetic boundary
  after a few steps: the WP5 re-prefill fires at the crossing step, re-injecting the
  retained image rows, and from that step on both sides compute the long table over the
  full history, so the fresh-uncached comparison stays valid; WP1 item 11, §5 — this is
  the golden test of the boundary-recompute decision, including the image-row restoration;
  one 336×336 image (**757** rows — the image rows alone are over the boundary, so the long
  table is used from the first step and the recompute branch never fires during decode); a
  wide image (457 rows) plus text pinned so the expanded prompt is over 470 — the image
  rows alone are under the boundary, so this pins the table selection to the prompt length
  rather than the image-row count: long table from the first step, recompute never fires;
  a tall image through the transpose branch (469 rows) plus text pinned so the expanded
  prompt is over 470 — 469 image rows plus the prepended `<s>` alone make 470, which would
  be short at prefill with the recompute firing at the first decode step, so the text
  length is pinned explicitly: long table from the first step, recompute never fires; two
  images with text between (expanded prompt well over 470 — at least 457 + 457 image rows —
  long table from the first step, recompute never fires). The 4096/4097 table-selection
  boundary itself is covered by the WP4 unit test, and the recompute trigger's non-firing
  case by the WP5 unit test, not here.
  `Phi3VGoldenTest` asserts strict `1e-4` in `float32GoldenTest` (`full-float32`);
  `Phi3VDefaultPrecisionTest` asserts a separately named default-mode bound derived from the
  locally measured TF32 error (recorded next to the Phase 6/7.3b measurements, never by golden
  regeneration).

**WP7 — lifecycle, memory and cache accounting.** `jmlx-models` tests, following the 7.3b WP7
scope: processing and tower execution sequential per image with cancellation checked between
images; only evaluated rows hoisted to the request's prefill child scope (the retained projection
rows go to the request scope, WP5); prefill evaluates
logits and every retained KV array before that scope closes; cached decode retains only
request-owned KV and the retained projection rows. `Phi3VGenerationTest`: mixed request reuse,
capacity rejection before native work, cancellation, failure cleanup, evaluated-visual release
and bounded KV growth.
`Phi3VMemoryTest`: per-image peak transient (plus the retained per-image projection rows,
at most ≈ 9.6 MB each, held for the whole generation), flat active memory across repeated
generations (same and different images, with the retained rows released at generation end),
and a per-token decode slope bounded by a per-token budget derived
from layers, KV heads, head size and dtype (the 7.3b WP7 budget method, not the legacy
constant). FULL and capacity-bounded FULL retention only; sliding retention rejected at the
request level (7.3a/7.3b precedent).

**WP8 — Tier-B acceptance, CI row and documentation.**

- `tools/tier-b/phi3v-4b-instruct.json`: the WP0 pin plus a licensed synthetic test image
  (the SmolVLM manifest pattern: pinned PNG copied from an oracle fixture with its license), the
  required multi-image request (at least two images of different sizes with text between), with
  **prompt + max_new_tokens ≤ 4096** — the request stays on one side of the rope boundary, so
  the recompute path is exercised by the Tier-A golden case and the WP5 unit test, not the
  Tier-B run — and the `reference`/`recorded_pin`/`local_evidence` sections. The manifest pins
  the exact stop-token set for the decoded-output assertion: **{32000}** (`
`, the
  tokenizer's own eos — **not** `<|end|>`, id 32007 — which the model typically emits before
  the 32000 stop), which every `sample_inference.py` `generate` call passes as `eos_token_id`,
  overriding `generation_config`'s `eos_token_id: 2` (WP1 item 14), and it states whether the
  decoded-output assertion strips special tokens, which decides whether `<|end|>` appears in
  the recorded text.
- The reference generator `tools/hf-reference/phi3v_tier_b.py` (manual, digest-pinned) records
  CPU float32 eager uncached logits and the greedy history; the manifest records the measured
  maximum logit error, the top-two gap with the acceptance `gap > 2 * epsilon + margin`
  arithmetic, exact greedy IDs and the decoded output, plus the measured allocator peak,
  sampled footprint peak, load and generation wall times. `TierBPhi3VTest` is selected by
  `task: phi3v-generation` and asserts exact greedy IDs and the end-to-end decoded output.
- `tier-b.yml` row on the self-hosted `self-macos-xl` runner (the 8.29 GB artifact and the
  float32 working set exceed the standard 7 GB hosted runner; the first self-hosted run accepts
  the CI pipeline only — same acceptance semantics as the qwen3vl row — and the macOS 26 hosted
  execution remains pending, not verified). The new required suites (`Phi3VGoldenTest`,
  `Phi3VDefaultPrecisionTest`, `Phi3VGenerationTest`, `Phi3VMemoryTest`) are added to the macOS
  native job's hard-coded required-suite list in `ci.yml`. An opt-in one-off `benchmarkDecode`
  run against the Tier-B artifact (informational, no threshold, full reproducibility tuple) may
  be recorded in `req/phase6-4-benchmark.md` per the 7.3b manual precedent, but it is not a
  gate.
- Documentation (separate items): (a) a compatibility-matrix row for "Phase 7.3c
  Phi-3.5-Vision image+text" in `req/phase6-compatibility.md`, including keeping the text
  `phi3` row's "`longrope` rejected" wording true **and** updating the LongRoPE "planned" row
  (`req/phase6-compatibility.md:16`) to the post-7.3c status (text `phi3` still rejected;
  `phi3_v` supported); (b) the `req/phase6-inference-report.md`
  row "Qwen3-VL, LLaVA-style vision models | Phase 7.3b / 7.3c, not implemented" (line 87),
  updated to reflect the 7.3b delivery and the 7.3c state; `req/phase6-tier-b-artifacts.md`;
  `req/phase6-tier-a-fixtures.md`; `req/phase6-inference-report.md` (status sections);
  `AGENTS.md` (architecture and status lines); the delivery table in `req/plans/phase7-plan.md`;
  `tools/tier-b/README.md`; and the findings file's §-numbered acceptance blocks.

## 5. Compatibility decisions

- **Semantics source.** The pinned revision's bundled code is the semantic source, not the
  transformers 4.57.6 core (which has no `phi3_v`). Where the bundled code is loose (legacy
  `num_img_tokens`, the `_attn_implementation` field, `sliding_window`, the missing total-marker
  count check), the reference's actual computed behavior decides and the decision is recorded
  in the findings; jmlx-side rejections are a documented hardening, never a change to the
  reference behavior being matched.
- **Execution mode.** Because the bundled code cannot run against the pinned
  `transformers==4.57.6` cache layer (WP1 item 2), every reference generator runs an uncached
  full forward with `use_cache=False` and a manual greedy loop; no compatibility shim and no
  separate transformers pin. Consequence for goldens: the fresh-uncached comparison is valid at
  **every** step — before the boundary the factor table is constant, and from the first crossing
  onward the reference's own fresh forward uses the long table over the full history, exactly
  the state jmlx's re-prefill reproduces (next bullet).
- **RoPE boundary recompute (WP1 item 11).** The bundled code discards the cache and re-runs
  the full history the first time a sequence crosses `original_max_position_embeddings`
  (`modeling_phi3_v.py:1650`–`:1655`): reference generation uses the long factor table for
  **every** key once the sequence passes the boundary — cached keys do **not** silently keep
  the short table, and a stale-key behavior would be a jmlx-only divergence, not "by design".
  jmlx mirrors that: `Phi3VModel` discards the request's KV cache and re-prefills the full
  history at the first decode step whose ending position exceeds `originalMaxPositions` while
  the accumulated past length — the number of cached positions, read from jmlx's own cache
  length — is `<= originalMaxPositions`, **re-injecting the retained per-image rows** at the
  sentinel positions (WP5; the reference re-runs the tower at the crossing because
  `prepare_inputs_for_generation` passes `pixel_values`/`image_sizes` on every step,
  `:1699`–`:1707`). The trigger's non-firing case (past
  length already over the boundary — e.g. the 757-row image case under the Tier-A config) is
  unit-tested in WP5; an image request crossing the boundary is golden-tested in WP6 (the
  wide-image case); under the real artifact (`original_max = 4096`) a single-image request
  with a long generation does cross and fire, while the Tier-B request stays under the
  boundary.
- **Precision.** Weights are stored BF16 and up-cast to FLOAT32; all computation is FLOAT32,
  matching the 7.3a/7.3b vision-language rule. `RopeSpec` computes frequencies in `double` and
  casts to float32 (its existing convention), which is a stated tolerance against the
  reference's float32 computation, not a bit-exact claim; the scaling constant is a float.
- **Resample kernels.** The HD bilinear is the existing Pillow-exact kernel (the reference's
  resize goes through PIL, WP1 items 12–13); only the global image's float32 bicubic is a new
  Torch-exact kernel, with provenance pinned in `jmlx-vision/NOTICE`. The 8-bit rounding step
  of the HD resize is part of the kernel contract, not an implementation detail.
- **Zero-padded crop slots** are never computed in Java; WP1 item 6 records why that is exact.
- **Prompt markers** are a jmlx-side rendering convention (image part → `<|image_i|>` in prompt
  order); the reference's own chat template is image-free, and the golden fixtures pin the exact
  expanded ID lists, including the per-chunk BOS.
- **Stop tokens.** The caller supplies the stop set through the existing
  `GenerationConfig`/`generate` contract. The artifact-faithful default for `phi3_v` — used by
  the examples, the goldens and the Tier-B assertion — is **{32000}**: the tokenizer's own eos
  (`
`, also its pad token) — **not** `<|end|>`, which is id 32007, what the chat template
  ends assistant turns with, and what the model typically emits just before the 32000 stop.
  Every `generate` call in the artifact's `sample_inference.py` passes 32000 as
  `eos_token_id` (WP1 item 14). `generation_config.json`'s
  `eos_token_id: 2` (`</s>`, not a special token) is **not** the default stop: the reference
  script overrides it and the model is unlikely to emit it. This matches the repository
  convention where the Tier-B tests stop on `tokenizer.eosTokenId()`.
- **RoPE scoping.** `su` is the old name for `longrope`. The compatibility matrix's "Phi-3 …
  `longrope` rejected" row stays true for text `phi3`: the `su` branch in the rope parse is
  consumed only on the `phi3_v` family path, and a text `phi3` config with `rope_type: su` is
  still rejected. The gate is explicit, not positional: `parseRope` is shared by all families,
  so `su` is accepted only behind a `Family` builder flag (e.g. `acceptsSuRope()`) that only
  the `phi3_v` entry sets — this matters because older Phi-3-mini-128k configs really did use
  `type: su`.
- **Family policy.** The `phi3_v` family entry sets `WindowPolicy.IGNORE` (the decoder never
  reads `sliding_window`) and mirrors the `phi3` entry's fused-projection flags (the bundled
  `Phi3MLP` and attention use fused `gate_up_proj`/qkv).
- **Backward compatibility.** Additive except one source-compatibility exception: adding a
  variant to the sealed `RopeSpec` changes the `permits` clause. There is no in-repo exhaustive
  `switch` over the hierarchy (verified: no `case RopeSpec.*` arms) — the WP4 PR instead audits
  the default-method `instanceof` chains that classify variants (`frequencies`, `isStatic`,
  `scale`, `usesFrequencyArray`, `attentionScaling`) so `Su` is treated correctly rather than
  falling into a default branch (notably `isStatic`, whose default `!(instanceof DynamicNtk)`
  would be `true` for `Su`, hence the override). `jmlx-core` is unpublished — nothing in this
  repository has been released to Maven Central — so no published consumer is affected. No
  other overload, golden, manifest, pin or tolerance changes; the
  exact before/after bit-identity machinery of the parent's cross-cutting section is not
  triggered because no old code path changes shape.
- **Scope and streaming.** The 7.3a/7.3b per-request scope pattern and the core single-platform-
  thread rule apply unchanged; the model is single-request-per-thread with no scheduler cohorts.

## 6. Deferrals and exit gate

Deferred: the `phi-3-vision-128k` artifact (same `model_type` `phi3_v`, gated access; the
descriptor handles its known config shape, but there is no support or verification claim),
quantized/affine `phi3_v` checkpoints, sliding image-prefill retention, multimodal scheduler
cohorts, video, per-layer quantization overrides, and any 8B-class artifact (none exists for
this family). Each deferral is restated in the implementation report's "remaining" section.

**Exit gate (parent row):** a `phi3v` Tier-B artifact row passes end-to-end — local eligibility
evidence recorded (exact greedy IDs, decoded output, top-two gap, peaks, wall times), the
`tier-b.yml` row green on the self-hosted runner, and the documentation set of WP8 delivered —
or a dated deferral is recorded in the findings file and the milestone closes without code. No
`Phi3VModel`, descriptor, dispatch branch or reference generator exists before its WP's PR;
no Tier-B manifest before WP8's.
