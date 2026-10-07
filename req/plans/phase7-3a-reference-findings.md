# Phase 7.3a reference findings — SmolVLM-256M (Idefics3)

Status: WP1 evidence. Pinned reference toolchain: Transformers **4.57.6** (CPU, float32, eval mode,
`attn_implementation="eager"` recorded explicitly for every forward), the **slow** PIL image processor
(`Idefics3ImageProcessor`, `use_fast=False`), raw `PIL.Image.open → load()` path (no `load_image`, no
`exif_transpose`, no early `convert("RGB")`; EXIF orientation therefore unapplied). Probes were executed
against the pinned artifact with the `tools/hf-reference/.venv` interpreter; nothing here is source-inspection
only. Every special token below is identified by its **numeric ID** and exact **UTF-8 hex**; the human-readable
spelling of a `<|…|>`-delimited token is a convenience only and the hex is the source of truth (several of
these tokens are mangled by some text transports, so the Java implementation must resolve them by ID through
tokenizer metadata, never by copying a rendered example).

## 1. Artifact, provenance and tensor inventory

- Repo: `HuggingFaceTB/SmolVLM-256M-Instruct`.
- Pinned immutable revision (full 40-char): `7e3e67edbbed1bf9888184d9df282b700a323964`.
- License: Apache-2.0 (from `cardData.license`). There is **no LICENSE file** in the repo root; the
  Apache-2.0 grant is recorded from the model card. `NOTICE`/per-file Pillow MIT-CMU headers are added in WP2
  for the ported resize code; the model weights are Apache-2.0.
- Storage dtype: **bfloat16** for every tensor. The Java loader up-casts to float32 (reference math is float32).
- Single shard: `model.safetensors`, no index file.
- Parameter total: **256,484,928** (256 484 928). Tensor count: **471**.

### File SHA-256 / size (pinned revision)

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.safetensors` | 513,028,808 | `74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e` |
| `config.json` | 7,353 | `b70fb4bfde88df9eeebc9d8ff523733b8bf70d6c9b06c610325960e06ae9db52` |
| `tokenizer.json` | 3,548,256 | `5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a` |
| `tokenizer_config.json` | 28,249 | `36c6fd44d07d10fd8180ee6b46dcccf69fb7c06753968ff0d7e17b8bfe17b777` |
| `special_tokens_map.json` | 1,069 | `aa0ff906077086dfa9734a7f97f68c825877a48f9468807be65504495cdeef09` |
| `vocab.json` | 800,662 | `82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79` |
| `merges.txt` | 466,391 | `0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510` |
| `added_tokens.json` | 4,739 | `74135b8664b56088c0006f1c8e848d79a8eba003411f72ebf1dc2ee96227be3a` |
| `chat_template.json` | 429 | `a68ad1a42681ae44eacd109ff8dd56a840f761c03d72f4ff4c515d092f882168` |
| `processor_config.json` | 68 | `e7bff42da73ae9eec9042ef20e066e11f1ee20f025358ff79131e3c0fb549b46` |
| `preprocessor_config.json` | 486 | `6cb6e36d6fcb88ca1502c4a26750715dc3e7dedddc9a8f17b27d8d167d1457e7` |
| `generation_config.json` | 136 | `067a2a54e5f87162ecac6e0e911cc4665fc8f7f3324794ecbac0f76badb56636` |
| `README.md` | 16,400 | `0b687c656b2587d84cb85003825746f8b3a08497dff4ed51b54b1090719fa512` |

### Tensor inventory (shape, dtype, bias)

All BF16. Prefixes are the live checkpoint names (they map 1:1 to Java loader plan names).

Text (Llama / SmolLM2-135M core), no biases anywhere in the text stack:

- `lm_head.weight` — `[49280, 576]` (untied).
- `model.text_model.embed_tokens.weight` — `[49280, 576]`.
- For each of 30 layers `model.text_model.layers.{0..29}`:
  - `self_attn.q_proj.weight` — `[576, 576]`
  - `self_attn.k_proj.weight` — `[192, 576]`
  - `self_attn.v_proj.weight` — `[192, 576]`
  - `self_attn.o_proj.weight` — `[576, 576]`
  - `mlp.gate_proj.weight` — `[1536, 576]`
  - `mlp.up_proj.weight` — `[1536, 576]`
  - `mlp.down_proj.weight` — `[576, 1536]`
  - `input_layernorm.weight` — `[576]`
  - `post_attention_layernorm.weight` — `[576]`
- `model.text_model.norm.weight` — `[576]`.

Vision (SigLIP base), **every linear has a bias**:

- `model.vision_model.embeddings.patch_embedding.weight` — `[768, 3, 16, 16]`; `…bias` — `[768]`.
- `model.vision_model.embeddings.position_embedding.weight` — `[1024, 768]`.
- For each of 12 layers `model.vision_model.encoder.layers.{0..11}`:
  - `layer_norm1.{weight,bias}` — `[768]` each
  - `layer_norm2.{weight,bias}` — `[768]` each
  - `self_attn.q_proj.{weight,bias}` — `[768, 768]` / `[768]`
  - `self_attn.k_proj.{weight,bias}` — `[768, 768]` / `[768]`
  - `self_attn.v_proj.{weight,bias}` — `[768, 768]` / `[768]`
  - `self_attn.out_proj.{weight,bias}` — `[768, 768]` / `[768]`
  - `mlp.fc1.{weight,bias}` — `[3072, 768]` / `[3072]`
  - `mlp.fc2.{weight,bias}` — `[768, 3072]` / `[768]`
- `model.vision_model.post_layernorm.{weight,bias}` — `[768]` each.
- Vision tensor total: 197.

Connector (no bias):

- `model.connector.modality_projection.proj.weight` — `[576, 12288]`.

Required/optional/forbidden: `tie_word_embeddings` is `false`, so `lm_head.weight` is **required and distinct**
from `embed_tokens.weight` (a tied alias would be rejected by shape/identity checks). No text biases; all
vision linear biases are required. Absent any of the above is a load failure.

## 2. Config-key tables

### Outer (`config.json`, `model_type: "idefics3"`)

| Key | Value | Disposition |
| --- | --- | --- |
| `model_type` | `idefics3` | supported — dispatch key (checked before `ArchitectureMappings`) |
| `architectures` | `["Idefics3ForConditionalGeneration"]` | supported — informational; not the dispatch key |
| `scale_factor` | `4` | supported — connector pixel-shuffle factor |
| `image_token_id` | `49190` | supported — must equal tokenizer `<image>` ID |
| `vocab_size` | `49280` | supported — top-level, must match text_config and lm_head rows |
| `hidden_size` | `576` | supported — text hidden |
| `tie_word_embeddings` | `false` | supported — forbids tied lm_head alias |
| `torch_dtype` | `bfloat16` | supported — loader up-casts to float32 |
| `transformers_version` | (str) | inert — provenance only |

### `text_config` (Llama)

| Key | Value | Disposition |
| --- | --- | --- |
| `model_type` | `llama` | supported |
| `hidden_size` | `576` | supported |
| `intermediate_size` | `1536` | supported |
| `num_hidden_layers` | `30` | supported |
| `num_attention_heads` | `9` | supported |
| `num_key_value_heads` | `3` | supported (GQA) |
| `head_dim` | `64` | supported (`9*64=576`; `3*64=192` for K/V) |
| `max_position_embeddings` | `8192` | supported |
| `rope_theta` | `100000` | supported |
| `rope_scaling` | `null` | supported — no scaling |
| `attention_bias` | `false` | supported — must match absence of text biases |
| `hidden_act` | `silu` | supported |
| `rms_norm_eps` | `1e-5` | supported |
| `bos_token_id` | `1` | supported — resolves the generation_config conflict (see §6) |
| `eos_token_id` | `2` | supported |
| `pad_token_id` | `2` | supported |
| `vocab_size` | `49280` | supported |
| `tie_word_embeddings` | `false` | supported |
| `use_cache` | `true` | inert for shape; cache is runtime |
| `perceiver_config`, `pixel_shuffle_factor`, `use_resampler`, `qk_layer_norms`, `is_llama_config`, `transformers.js_config`, `_flash_attn_2_enabled`, `id2label`, `label2id`, and the `generation_config` junk keys carried inside `text_config` | various | **inert** — explicitly ignored; none changes numerics |

### `vision_config` (SigLIP)

| Key | Value | Disposition |
| --- | --- | --- |
| `model_type` | `siglip_vision_model` | supported |
| `use_base_siglip` | `true` | supported — selects base dimensions |
| `hidden_size` | `768` | supported |
| `image_size` | `512` | supported — tile edge |
| `patch_size` | `16` | supported — `512/16 = 32` patches/side |
| `num_hidden_layers` | `12` | supported |
| `num_attention_heads` | `12` | supported (head_dim `768/12 = 64`) |
| `intermediate_size` | `3072` | supported |
| `hidden_act` | `gelu_pytorch_tanh` | supported → `GELU_TANH` |
| `layer_norm_eps` | `1e-6` | supported |
| `size` | `{"longest_edge": 2048}` | supported — stage-1 upscale target |
| `max_image_size` | `{"longest_edge": 512}` | supported — tile longest edge |
| `projection_class`, `ignore_index`, `num_channels`, `pad_token_id`, etc. | various | inert |

### `preprocessor_config.json` (`Idefics3ImageProcessor`)

| Key | Value | Disposition |
| --- | --- | --- |
| `image_processor_type` / `processor_class` | `Idefics3ImageProcessor` / `Idefics3Processor` | supported — asserts the slow class |
| `do_convert_rgb`, `do_image_splitting`, `do_normalize`, `do_pad`, `do_rescale`, `do_resize` | all `true` | supported |
| `image_mean` / `image_std` | `[0.5,0.5,0.5]` / `[0.5,0.5,0.5]` | supported |
| `rescale_factor` | `0.00392156862745098` (= 1/255) | supported |
| `resample` | `1` | supported — `PIL.Image.Resampling.LANCZOS` |
| `size` | `{"longest_edge": 2048}` | supported — stage-1 longest edge |
| `max_image_size` | `{"longest_edge": 512}` | supported — tile longest edge |

### `processor_config.json`

| Key | Value | Disposition |
| --- | --- | --- |
| `image_seq_len` | `64` | supported — image tokens **per tile**; `64 = (512/16)^2 / 4^2 = 1024/16` |
| `processor_class` | `Idefics3Processor` | supported |

Note: `image_seq_len` is **derivable** from grid geometry and is not an independent knob in the slow path;
the per-tile `<image>` count equals 64 for a 512 tile. It is recorded, not assumed.

### `generation_config.json`

| Key | Value | Disposition |
| --- | --- | --- |
| `_from_model_config` | `true` | inert |
| `bos_token_id` | `0` | **conflict** — contradicts `text_config.bos_token_id = 1`; resolved in favor of the tokenizer (see §6). Not copied. |
| `eos_token_id` | `49279` | supported — `<end_of_utterance>` |
| `pad_token_id` | `2` | supported |

No unknown computational field merely warns: every key above is classified supported/inert/rejected. The only
cross-config conflict (bos 0 vs 1) is resolved by tokenizer metadata, and the Java loader must not silently
accept a `text_config` value that disagrees with a required tensor shape (e.g. `hidden_size` vs
`embed_tokens.weight` rows).

## 3. Image decode, color/alpha/EXIF, resize, splitting, masks

### Decode path (raw `PIL.Image.open → load()`)

The reference loads bytes with `PIL.Image.open(bytes).load()`, keeping the original `mode`, palette and
info until the slow processor's `do_convert_rgb`. No `exif_transpose` (EXIF orientation is **unapplied**),
no `load_image`, no pre-`convert("RGB")`. Alpha/palette handling is therefore defined by the processor's
`do_convert_rgb`, which composites via the palette/tRNS path and then drops alpha. Fixtures (WP2 image oracle)
determine the exact tRNS result rather than assuming white compositing. The Java `ImageDecoder` must match the
observed processor RGB conversion **before** resize.

**tRNS on truecolor/grayscale PNG (color types 0/2):** Pillow keeps the image in mode `RGB`/`L` and records
the transparency only in `info["transparency"]`; the processor's RGB conversion drops it, so the reference
decodes such files as **opaque** — a tRNS twin decodes byte-identical to its opaque original (fixtures
`rgb-trns-64x48`, `gray-trns-60x40`). The JDK's built-in PNG reader would instead turn the chunk into an
extra alpha band, so the Java `ImageDecoder` strips the `tRNS` chunk from the bytes before decoding. (PIL's
own writer emits the tRNS sample 16-bit even for 8-bit files; the decoder strips the whole chunk, so the
sample layout is irrelevant to it.)

**Oversized palette `tRNS` (more samples than palette entries):** technically malformed per RFC 2083, but
PIL (`PngImagePlugin.chunk_tRNS`) and libpng treat it as benign. PIL records the transparency as-is: the
"simple" pattern (all `0xFF` with exactly one `0x00`) becomes a single palette index — one past the palette
leaves every entry opaque — and the per-entry form keeps the whole byte string, whose samples past the
palette are ignored at lookup time. The Java `ImageDecoder` matches both: an out-of-range simple index and
the extra per-entry samples are ignored, never an error. Fixture: `palette8-trns-overflow-8x8` (per-entry
form: five samples for four entries, the fifth ignored).

**JPEG restart markers:** a scan written with a restart interval contains `RST0`–`RST7` markers inside the
entropy-coded data, and the writer (PIL's `restart_marker_blocks`, libjpeg, camera and phone encoders)
additionally emits an optional `DRI` segment (`FF DD`, length 4, restart interval) before the scan. Both are
well-formed JPEG and the reference decodes them fine; the header walk must treat in-scan `RSTn` as scan data
(not a scan terminator). The `DRI` segment needs no special case — it is a plain length-prefixed segment the
general marker path skips, and every `FF FF` outside the scan is simply a fill byte. Fixture: `jpeg-rst-96x64`
(carries both the `DRI` segment and the in-scan restart markers).

### Resize (two-stage, exact)

1. **Stage 1** — `resize(size={"longest_edge": 2048})`: `_resize_output_size_rescale_to_max_len` rescales the
   longest edge to **exactly 2048** (upscales small images), the other edge `int(longest / aspect)`, then
   `_resize_output_size_scale_below_upper_bound(…, 4096)` is a no-op here. Kernel: PIL `LANCZOS` (resample 1).
   Example: `100×80 → 1638×2048` (aspect 1.25).
2. **Stage 2** — `resize_for_vision_encoder(…, 512)`: rounds both edges **up** to multiples of 512,
   `width = ceil(w/512)*512`, `height = int(width/aspect)` then `ceil(height/512)*512`. Example: `1638×2048 →
   2048×2048`. Kernel: `LANCZOS`.

Because stage 1 always yields longest edge exactly 2048 (and 2048 is a multiple of 512), **every tile after
splitting is exactly 512×512**; there is no per-tile sub-512 padding in this artifact. The global thumbnail is
also resized to exactly 512×512.

Rounding/antialiasing: PIL `LANCZOS` (3-lobe sinc, windowed) with `Image.Resampling.LANCZOS`; intermediate
values are computed in the resize's internal float accumulator then rounded to uint8. The WP2 Java port must be
bit-exact against the pinned Pillow 12.3.0 `ImResample.c` for LANCZOS (and, for completeness of the `Resampling`
enum, BILINEAR/BICUBIC). Antialiasing on downscale follows Pillow's default.

### Splitting, ordering, padding, masks

`split_image(image, {"longest_edge": 512})`:

- If `h > 512` or `w > 512`: `num_splits_h = ceil(h/512)`, `num_splits_w = ceil(w/512)`; each crop is exactly
  512×512, iterated **row-major** over `(row, col)` with `start = (row*512, col*512)`.
- After the grid, the **global** image (original resized to 512×512) is appended **last**.
- If `h ≤ 512` and `w ≤ 512`: `num_splits = (0, 0)` and the single image is appended as-is (no global).

Tile order in `pixel_values` (one image, `gh×gw` grid): `tile(0,0), tile(0,1), …, tile(0,gw-1), tile(1,0), …,
tile(gh-1,gw-1), global`. The prompt expansion (below) lists image tokens in the **same** order, so feature
`i` maps to the `i`-th 64-token block.

`do_pad` pads all tiles in a batch to the max `(H, W)` and emits `pixel_attention_mask` (1.0 where the tile has
real content, 0.0 where zero-padded). Because all tiles are 512×512 in this artifact, the per-tile mask is all
ones; a multi-sample batch pads shorter rows with all-zero images (mask 0) which the tower drops (see §4).

Token count per tile: `(512/16)^2 = 1024` patches → after the connector pixel-shuffle (`/4^2`) → **64 image
features per tile**, emitted as 64 `<image>` (ID 49190) tokens.

### Worked example (verified)

`100×80` RGB gradient → stage 1 `1638×2048` → stage 2 `2048×2048` → `4×4` grid + 1 global = **17 tiles** →
`pixel_values` shape `(1, 17, 3, 512, 512)`, float32. Image-token count = `17 × 64 = 1088`.

## 4. Vision tower math

### Patch embedding and variable-resolution positions

`Idefics3VisionEmbeddings`:

- `patch_embedding = Conv2d(3, 768, kernel=16, stride=16)`, **has bias**, no padding. For a 512 tile this
  yields a `32×32` grid of 768-dim patch vectors, flattened row-major to `(B, 1024, 768)`.
- `position_embedding = Embedding(1024, 768)` (a `32×32` table). Positions are assigned by **bucketing**
  against the patch mask, not by a fixed square index:
  - `h = arange(nb_h)`, `w = arange(nb_w)` (valid region from the patch mask).
  - `frac_h = h / nb_h * (1 - 1e-6)`, `frac_w = w / nb_w * (1 - 1e-6)`.
  - `boundaries = arange(1/32, 1, 1/32)` (31 interior boundaries).
  - `bucket_h = bucketize(frac_h, boundaries, right=True)`, same for `w`.
  - `pos_ids = (bucket_h * 32 + bucket_w)` row-major over the valid region.
  - For a full `32×32` tile the mapping is the identity (`pos_id = row*32+col`); for a `16×32` subgrid the
    rows/cols land on even indices of the 32-grid (top-left subgrid). This variable-resolution behavior must be
    preserved — do **not** replace it with fixed square indices.
- `embeddings += position_embedding(pos_ids)`.

### Attention and block structure (12 layers)

- Attention: `q,k,v = (x @ W + b)`; `attn = softmax( (q @ k^T) * (head_dim^-0.5) + mask, dim=-1, dtype=float32)`;
  `head_dim = 64`, so scale `= 1/8`. **No causal mask** (bidirectional). `out_proj` has bias.
- 4D additive mask from the patch mask: `(1 - mask) * dtype.min`, broadcast to `(B, heads, S, S)`.
- Block: pre-LN — `x = x + attn(LayerNorm1(x))`; `x = x + mlp(LayerNorm2(x))`. `LayerNorm` eps `1e-6`, **with
  bias**.
- MLP: `fc1 (768→3072, bias) → GELU_TANH → fc2 (3072→768, bias)`. `GELU_TANH = 0.5 x (1 + tanh(sqrt(2/π)(x +
  0.044715 x^3)))`.
- Final: `post_layernorm = LayerNorm(768, eps=1e-6, with bias)`.

### Feature extraction and padding-image drop

`get_image_features(pixel_values, pixel_attention_mask)`:

- `pixel_values (B, N, 3, H, W)` → `(B*N, 3, H, W)`.
- All-zero padding images are dropped: `real = (pixel_values == 0.0).sum(-1,-2,-3) != 3*H*W`. (Real tiles are
  never all-zero because rescale/normalize maps 0 → −1.0; only pad tiles are exactly 0.0.)
- `patch_attention_mask = (pixel_attention_mask.unfold(1,16,16).unfold(2,16,16).sum() > 0)` per 16×16 patch.
- Tower → `(B*N, 1024, 768)` → `post_layernorm` → connector → `(B*N, 64, 576)`.

## 5. Connector (pixel shuffle + projection)

`pixel_shuffle(x, scale_factor=4)` on `x (B, 1024, 768)` with a `32×32` grid:

- `x.view(B, 32, 32, 768)` → `x.view(B, 32, 8, 3072)` → `permute(0,2,1,3)` → `reshape(B, 8, 8, 12288)` →
  `permute(0,2,1,3)` → `reshape(B, 64, 12288)`.
- Index result: output row `i = r*8 + c` (with `r = i//8`, `c = i%8`) concatenates the 4×4 block of patches at
  offsets `(4r+dh, 4c+dw)` for `dh,dw ∈ 0..3`, in **row-major within the block**: channel offset
  `m = dh*3072 + dw*768 + e` for `e ∈ 0..767`.
- `12288 = 16*16 * 4^2`? No — `12288 = (patch 16×16 channels regrouped) = 768 * 4^2 = 768*16`; equivalently
  `(4×4 block) × 768 = 16 × 768 = 12288`. The `scale_factor=4` divides each spatial axis by 4 (`32→8`), giving
  `8×8 = 64` output features, each of width `768*16 = 12288`.
- `modality_projection = Linear(12288 → 576, bias=False)` → tensor
  `model.connector.modality_projection.proj.weight [576, 12288]`.
- Output: `(B*N, 64, 576)`; features within a tile are ordered by `i = r*8+c` (row-major 8×8), matching the 64
  `<image>` token order per tile.

Divisibility guard: `scale_factor` (4) must divide the patch grid (32); `32/4 = 8` is integer. A `scale_factor`
that does not divide the grid is rejected.

**Index-coded micro-example (to expose permutation bugs):** with a single-channel 2×2→1×1 toy grid
(`scale_factor` applied to a `4×4`→`2×2` grid), the output cell `(0,0)` must collect block rows
`(0..3, 0..3)` in row-major: offsets `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`; cell `(0,1)` collects the
next block, etc. A shape-correct but mis-ordered port would pass shape checks and fail this ordering assertion.
The WP2 connector test encodes this with distinct marker values per patch.

## 6. Prompt template, special tokens, expansion

### Special token IDs (authoritative; hex = UTF-8)

| ID | Hex (UTF-8) | Role |
| ---: | --- | --- |
| 0 | `3c7c656e646f66746578747c3e` | `
</think>

`) — low special token (unused by template) |
| 1 | `3c7c696d5f73746172747c3e` | start-of-message (BOS-equivalent); **first** template token |
| 2 | `3c7c696d5f656e647c3e` | end special token (unused by template) |
| 49152 | `3c676c6f62616c2d696d673e` | `<global-img>` |
| 49153…49188 | `<row_r_col_c>` | row/col markers, `r,c ∈ 1..6`, row-major (`49153 = row_1_col_1`, `49188 = row_6_col_6`) |
| 49189 | `3c66616b655f746f6b656e5f61726f756e645f696d6167653e` | `<fake_token_around_image>` |
| 49190 | `3c696d6167653e` | `<image>` — the expansion marker |
| 49279 | `3c656e645f6f665f7574746572616e63653e` | `<end_of_utterance>` (also `eos_token_id`) |

`<row_r_col_c>` IDs are `49152 + (r-1)*6 + c` for 1-based `r,c` (row-major), so `49153..49188` is a contiguous
36-block. All special IDs above are resolved through tokenizer metadata (the Java tokenizer's ID lookup), not
copied from this table.

Verified encoding: these `<|…|>`-delimited tokens (IDs 0, 1, 2, 49152–49190, 49279) are registered added/special
tokens and **always** encode to their numeric ID wherever they appear as substrings (start, middle, or alone) —
confirmed for the ID-1 start token in all positions. The Java tokenizer's special-token handling must reproduce
this.

### Exact unexpanded template (length 403, hex-verified)

`<START>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}`

where `<START>` is the ID-1 token (hex `3c7c696d5f73746172747c3e`), emitted **literally** at the very start (it is
a template literal, not a `{{ bos_token }}` variable). Key behaviors:

- The template begins with the ID-1 start token, then each message.
- Role is `capitalize`d; the separator is `":"` with **no space** when the message's first content item is an
  image, and `": "` (with space) otherwise.
- Text items render their raw text; image items render the literal `<image>` (ID 49190) with no surrounding
  spaces.
- Every message ends with `<end_of_utterance>\n`.
- `add_generation_prompt` appends `Assistant:` (no trailing newline, no space).
- There is **no** separate BOS insertion by the tokenizer; the start token comes only from the template literal.

### Expansion rules (string level, then ID level)

`image_seq_len = 64` (per tile). `fake = 49189`, `global = 49152`, `img = 49190`.

- **Single image** (`rows = cols = 0`): `fake + global + img*64 + fake`.
- **Tiled image** (`gh×gw`): for each row `r` (0-based), each col `c`: `fake + <row_{r+1}_col_{c+1}> + img*64`,
  then a literal `\n` after each row; after all rows: `\n + fake + global + img*64 + fake`.

So a tile contributes `66` tokens (`1 fake + 1 row/col + 64 img`) and a row of `gw` tiles contributes `66*gw + 1`
(newline); the global block contributes `1 (\n) + 1 (fake) + 1 (global) + 64 (img) + 1 (fake) = 68`. Total
expanded length for a `gh×gw` image = `gh*(66*gw + 1) + 68`. For the verified `4×4` example: `4*(66*4+1)+68 =
4*265+68 = 1128`; plus the surrounding prompt tokens.

Verified full-prompt `input_ids` (100×80 image, single user message "What is in [image] this image?",
`add_generation_prompt=True`), head and structure:

```
[1, 11126, 42, 1812, 314, 281,            # START User : What is in
 49189, 49153, 49190×64,                  # fake row_1_col_1 <image>*64
 ...                                     # 15 more tiles, each fake+<row_r_col_c>+<image>*64, \n per row
 49189, 49152, 49190×64, 49189,           # \n fake global <image>*64 fake
 451, 2443, 47, 49279, 198,               # " this image?" EOU \n
 9519, 9531, 42]                          # "Assistant:"
```

Counts for that prompt: `<image>` (49190) = 1088, `fake` (49189) = 18, `global` (49152) = 1, `START` (1) = 1.
Total prompt length = 1141.

### Text-only / adjacent / multiple images

- **Text-only** (no `<image>` in text): the processor's text branch tokenizes as-is; the tower is not invoked
  (no `pixel_values`).
- **Adjacent/multiple images**: each `<image>` in order maps to the corresponding image in the `images` list;
  `n_images_in_text` must equal `n_images_in_images` else the processor raises. Each image's expansion is
  independent and placed at its `<image>` position.
- A sample with no `<image>` but images passed (or vice-versa) is rejected by the processor.

### bos/eos conflict resolution

`generation_config.bos_token_id = 0` contradicts `text_config.bos_token_id = 1` and the tokenizer's ID-1 start
token. The reference pipeline's actual start token is **ID 1** (from the template literal). Resolution: the Java
implementation uses the tokenizer-resolved start token (ID 1) and does **not** copy `generation_config.bos_token_id`.

## 7. Prefill/decode, cache, EOS, tower reuse

Verified end-to-end on the pinned checkpoint (float32, eager):

- **Prefill** runs the vision tower once (all `pixel_values` present) and merges image features into the
  embeddings at every `<image>` (49190) position, in feature order (row-major tiles, then global, per image).
- **Decode** steps pass `pixel_values = None` (the HF `prepare_inputs_for_generation` drops them once
  `cache_position[0] != 0`); a generated `<image>`-family token gets its ordinary embedding, not image features.
  Image features are therefore computed **once at prefill** and never recomputed during decode.
- **Cached vs uncached agreement**: with a correctly-lengthed attention mask, a cached decode step matches the
  full uncached forward to a max logit diff of `2.7e-05` (float32 rounding). A short/mismatched mask yields large
  diffs — the reference always uses a mask matching the full sequence length.
- **Greedy** on the verified prompt produced `330, 26591, 308, 1851` → `" A multicolored"`. Prefill top-5 logits:
  `[(330, 13.23230), (48596, 11.93018), (933, 11.84400), (7520, 11.44175), (17139, 10.59490)]`.
- **EOS/stop**: `eos_token_id = 49279` (`<end_of_utterance>`). Stop on that token per the generation config.
- **Text-only execution skips the tower** entirely (no `pixel_values` → no vision forward).

Image-feature reuse across a batch: features are returned shaped `(num_real_images_total, 64, 576)` and scattered
into the `<image>` positions of each sample in order; per-sample tile counts drive how many 64-blocks each sample
consumes.

## 8. WP2 Java implementation findings (JDK image decoding)

Findings from implementing `jmlx-vision`'s `ImageDecoder`/resampling against the pinned Pillow 12.3.0 oracle.
All 51 module tests are byte-exact against the committed fixtures (decode/resize/chain) with no loosened
tolerances; the normalized-tile comparison in the chain test uses the documented `1e-6` float32 tolerance.

### Raster byte layout is NOT channel order (root cause of red/blue swaps)

The JDK's built-in PNG/JPEG readers store the raster byte buffer as **B,G,R** (`TYPE_3BYTE_BGR`, 3-band) and
**A,B,G,R** (`TYPE_4BYTE_ABGR`, 4-band) — the Raster's *bands* are always in channel order R,G,B,(A), and the
`SampleModel`'s `getBandOffsets()` carries the permutation (`[2,1,0]`, `[3,2,1,0]`). 2-band gray+alpha
(`TYPE_CUSTOM`) has offsets `[0,1]` (band 0 = gray, band 1 = alpha). Addressing bytes by band index silently
swaps red and blue on every pixel. The implementation therefore addresses bytes through the
`PixelInterleavedSampleModel` (`getBandOffsets`/`getPixelStride`/`getScanlineStride`) plus
`DataBuffer.getOffset()` (no-arg; `DataBufferByte` has no `getOffset(int)`), and probes the band→channel
mapping **once through the decoded `ColorModel`**, never assuming it: per band, build a 1×1 synthetic raster
with the *real* `bandOffsets`/`pixelStride` via `Raster.createInterleavedRaster(new DataBufferByte(pixelStride),
1, 1, scanlineStride, pixelStride, bandOffsets, null)` (the 4th arg is `scanlineStride`, not a band count),
`setSample(0,0,band,255)`, read back with `(byte[]) raster.getDataElements(0,0,null)` and `cm.getRGB(comps)` —
the exact path `BufferedImage.getRGB(x,y)` uses.

JDK API traps verified against the Java 21 sources (probe scripts kept with the work session):
`ComponentColorModel` has no `getRedOffset`/`getGreenOffset`/`getBlueOffset`/`getAlphaOffset` getters;
`cm.getRGB(byte[])`/`getRed(Object)` index by *component* (R,G,B,A), not raster band, so unit-vector probes
through them test component indexing (always identity) rather than the band permutation; `getRGB(float[])`
throws `ClassCastException` on byte-transfer models; `Raster.createInterleavedRaster` needs 7 args and a
non-null `bandOffsets` with `pixelStride ≥ max(offset)+1`.

### Sub-8-bit palettes are 4-bit by default and packed, not expanded

Pillow's own default for a small palette is a **4-bit** palette PNG, so the decoder accepts palette bit depths
1/2/4/8 (sub-8-bit is rejected only for grayscale/truecolor). The JDK reader decodes sub-8-bit palettes into a
`MultiPixelPackedSampleModel` (several indices per byte, MSB-first) rather than expanding them. Do **not**
decode indices from the raw buffer with the sample model's `getBitOffset(x)`/`getOffset(x,y)`: `getBitOffset(0)`
is 0 (the low nibble) but the first sample occupies the high nibble of byte 0 (measured: `getSample(0,0)` = 0
while `(data[0] >>> 0) & 0xF` = 1). The packed branch uses `raster.getSample(x, y, 0)` per pixel — the same
path `getRGB` uses and correct for all packed depths. 8-bit palettes arrive interleaved (`PixelInterleavedSampleModel`).

### JPEG truncation

The JDK's built-in JPEG decoder is lenient about truncated scan data (it silently produces an image padded with
wrong pixels at the declared size). Truncation is detected by walking the full marker structure in the header
pre-scan, including the compressed scan as raw bytes (an `FF` followed by a byte that is neither `FF` (fill),
`00` (lenient data), nor one of the `RST0`–`RST7` restart markers — valid inside a scan written with a restart
interval — terminates it; outside the scan, every `FF FF` pair is a fill byte — the `DRI` (restart interval)
segment is a plain `FF DD` length-prefixed segment the general marker path skips — see §3), and requiring the
well-formed `EOI` marker the decoder never checks for.

### Alpha compositing

`tRNS`/RGBA/LA alpha is composited over white with Pillow's exact integer formula ported from the pinned
`src/libImaging/AlphaComposite.c` (`SHIFTFORDIV255(x) = (((x >> 8) + x) >> 8)`, 32-bit integer math); the
opaque case (`a == 255`) reduces to the identity for all 256 channel values and passes through unchanged.

### Oracle error bounds (measured)

- **Decode, all accepted fixtures**: byte-exact against the pinned Pillow output — maximum channel error
  `0`, including the color and grayscale JPEG fixtures (no JPEG-vs-libjpeg-turbo drift on the committed
  corpus; the JDK reader's own output is what the reference's non-JPEG path would also read, and the
  JPEG fixtures confirm the channel order is RGB for 3-component frames).
- **Resize/split/normalize chain**: uint8 pixels byte-exact at every stage; normalized float32 tiles within
  `1e-6` (the bound covers the reference's float64-rescale→float32 parse path, e.g. the
  `(float)(1/255)` double-parse difference ~`6e-11` relative).
- **LANCZOS**: no measured bound needed — the port is byte-exact on the committed corpus, so the plan's
  fallback (record a non-zero sin bound) never triggered.

### Documented ordering divergences

The reference resizes **and splits before** `convert_to_rgb` (its `_preprocess` order); the port decodes
straight to RGB (composite over white, palette expansion) and resizes in RGB space. Identical for opaque
RGB/grayscale input (all strict fixtures); divergent for semi-transparent RGBA (reference resizes alpha in
its native space) and palette PNG (reference resizes palette indices). The decode path is byte-exact for
both families; the byte-exact full-chain fixtures exclude them (documented in `jmlx-vision/README.md`).
