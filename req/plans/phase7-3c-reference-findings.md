# Phase 7.3c — Phi-3.5-Vision reference findings (WP0 + WP1 + WP2 kernel arithmetic)

- **Status:** WP0 and WP1 complete, 2026-10-28. Every number below was re-verified against
  the committed golden `tools/hf-reference/goldens/phi3v-wp1-probe.json`, the pinned artifact
  files (13 small files plus the two 8.3 GB shards) and the transformers 4.57.6 sources in the
  pinned venv; where a table and a golden disagree, the golden is authoritative. The probe is
  deterministic: two consecutive runs produced byte-identical goldens (same SHA-256). The 13-char
  `endoftext` literal (special token id 32000) is written by ords throughout, because message
  transport strips it from rendered text (see §7.4).
- **Plan:** `req/plans/phase7-3c-plan.md` (WP0 = §3 "pin and inventory", WP1 = §3 "reference
  probes", items 1–15; every item is resolved in the section named at its point of use). The
  2026-10-27 go/deferral discovery recorded in the plan's history — GO: this artifact is the only
  small, fully-licensed, public LLaVA-style instruct artifact — is unchanged by WP1.
- **Evidence goldens** (digest-guarded by root `check`):
  - `tools/hf-reference/goldens/phi3v-wp1-probe.json`, sha
    `06734b2207b8b418807ef178459366206ccda3307b150cc5f686f7467c753f73` (19,636 B).
  - Generator `tools/hf-reference/phi3v_probe.py`, sha
    `122af2472e5cae11eb53ce20e58440f4608eb7edee02331ac405572edca4c1f9`, pinned by
    `verifyHfReferenceGoldens` through the `phi3v_probe_sha256` provenance field.
  - Guards: `verifyHfReferenceGoldens` (golden, generator, lock, artifact provenance),
    `verifyImageOracleDependencyPins` (the four reference locks agree on Pillow, NumPy,
    Transformers and, since this milestone, torch/torchvision), `verifyImageOracleFixtureDigests`
    (the five hash-pinned bundle files under `tools/image-oracle/fixtures/phi3v/`).

## 1. WP0 record — baseline, prerequisites, publications

- **Base commit.** This work was cut from `main` at
  `9f17720ab842207de7d149d205fa3e4ee3eba0db` (2026-10-27 21:33:16 +0100, "Merge pull request #65
  from Alipsa/docs-tier-b-first-self-hosted-run"), verified by `git log` at session start. The
  commit is the merge behind the push-protected `main`, which required a green CI run to land, so
  every prerequisite suite named in the plan's §2 gate was green at the base by construction; the
  suites it touches are re-run as part of each implementation WP's local verification.
- **Publications since the plan's publication (2026-10-27):** none. Re-verified 2026-10-28
  against Maven Central: the `se.alipsa` group has 27 published artifacts and **none** of them is
  a `se.alipsa:jmlx-*` module (all seven publishable modules remain unpublished); the only
  related prior artifact in the `se/alipsa` repo1 listing is `hfjinja/` (lastUpdated 2026-08-29,
  i.e. before the plan). No post-plan publication imposes additive-only constraints on this work.
- **No competing multimodal contracts.** At the base, `main` carries the Phase 7.3a SmolVLM/
  Idefics3 and 7.3b Qwen3-VL multimodal paths. 7.3c reuses their contracts as-is — structured
  chat validation, `GenerationRequest.withImages`/`images`, `InputModality`, per-request vision
  scopes, the row-replacement injection at the decoder embedding entry point, and the shared step
  loop — and introduces no parallel version of any of them.

## 2. Pinned artifact — `microsoft/Phi-3.5-vision-instruct`

- **Revision** `12b77fb40b63a2c73c68243d3f767aab688a1b2a`. License **MIT** (`LICENSE`, 1,141 B;
  HF tag `license:mit`; evidence URL
  `https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/12b77fb40b63a2c73c68243d3f767aab688a1b2a/LICENSE`).
- **Files** (15; sizes in bytes). The probe's `pins` block records size + SHA-256 + `ok` for the
  13 non-shard files before any import; all 15 SHAs are committed in
  `tools/hf-reference/provenance.json` under `phi3v.artifact_sha256`.

  | file | bytes | sha256 |
  |---|---:|---|
  | `model-00001-of-00002.safetensors` | 4,944,122,112 | `be7bdee601478b3560aeb7122fe87a7d68445bb21b5f1ca88a764e8530b3aab8` |
  | `model-00002-of-00002.safetensors` | 3,349,208,776 | `94621eca6936d7a2fb94bd90f37edb0e28850235e7ba2c5e20445d2fc56bcaed` |
  | `model.safetensors.index.json` | 68,901 | `72d628080d85b3a0e96eac71a47f7a04a666aabf8a91269100c99a8d543f345c` |
  | `config.json` | 3,778 | `567e28c638a3e41aba1a304ae58b283845ad07e1c60e970643b0ad8fc229b382` |
  | `tokenizer.json` | 1,851,360 | `493f8ad437be806d5ccdc57e91479bdbb0c656448c1b89ff70f7653530d78518` |
  | `tokenizer_config.json` | 9,519 | `581d4654c018fbcef2504b57a805dcad37126038137b5d25d22b5a02abc1ef29` |
  | `special_tokens_map.json` | 670 | `f9535fada19315e0d8eb89521cebe111c008fee98d16e7c6744acfb50505d8c3` |
  | `generation_config.json` | 136 | `bf86d2cd9250f89f09563b7f80e25e34663f673b836f531dc207362165051dc9` |
  | `preprocessor_config.json` | 442 | `0f3d920fcec587bb425312d30b0660e1fabaf46a75961f2b75ddbb1ff92de01c` |
  | `processor_config.json` | 119 | `286a9e7fb58a489eae6825b79be69f62068c2028435c29ee52c3a485d071707b` |
  | `modeling_phi3_v.py` | 88,866 | `b223be7d7e9f01903dd8704e5da45c6030824134441aa2734c40c36597c3b41c` |
  | `configuration_phi3_v.py` | 10,712 | `18ff44c5c1f9a6523184fb2e580e03ebf722a65f37b9cc57cd1a6d620a0f2666` |
  | `processing_phi3_v.py` | 22,045 | `990ef3d1f6b5cc61b44cf8ec8e8886b19721b7e01e8bae3ebea32781b0fcf598` |
  | `sample_inference.py` | 6,778 | `49b1246f2a18d9b4a1e12892e50d19929e550263075d4c80cedeac2246546da4` |
  | `LICENSE` | 1,141 | `c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383` |

- **Index check** (required, per the 7.3b 8B stale-index finding, not assumed):
  `model.safetensors.index.json` has exactly **592** `weight_map` entries, each key appearing
  exactly once; `metadata.total_size` = **8,293,242,880**. The two shard files sum to
  8,293,330,888 B; the difference of 88,008 B is exactly the per-shard safetensors overhead
  (8-byte little-endian header length + header JSON: 78,072 B for shard 1, 9,920 B for shard 2).
  Shard 1 holds 507 keys (the complete vision tower plus decoder `embed_tokens` and layers
  0–17); shard 2 holds 85 (decoder layers 18–31, `model.norm.weight` and `lm_head.weight`).
  Reading the
  two shard headers directly: **all 592 tensors are `BF16`** — no float32 tensor exists, so the
  §1 "float safetensors only, up-cast in place" rule exercises the BF16 → FLOAT32 up-cast on the
  whole checkpoint.
- **Tensor-key inventory** (complete, grouped by owner; shapes from the shard headers):

  | owner | keys | shape (BF16) |
  |---|---:|---|
  | decoder embeddings | `model.embed_tokens.weight` (1) | `[32064, 3072]` |
  | decoder blocks | `model.layers.{0..31}.{input_layernorm.weight, post_attention_layernorm.weight, self_attn.qkv_proj.weight, self_attn.o_proj.weight, mlp.gate_up_proj.weight, mlp.down_proj.weight}` (32 × 6 = 192) | `[3072]`, `[9216, 3072]`, `[3072, 3072]`, `[16384, 3072]`, `[3072, 8192]` |
  | final norm + head | `model.norm.weight` (1), `lm_head.weight` (1, top-level, not under `model.`) | `[3072]`, `[32064, 3072]` |
  | separators | `model.vision_embed_tokens.glb_GN` (1), `model.vision_embed_tokens.sub_GN` (1) | `[1, 1, 4096]`, `[1, 1, 1, 4096]` |
  | tower embeddings | `model.vision_embed_tokens.img_processor.vision_model.embeddings.{class_embedding, patch_embedding.weight, position_embedding.weight}` (3) | `[1024]`, `[1024, 3, 14, 14]` (no bias), `[577, 1024]` |
  | tower blocks | `model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{0..23}.{layer_norm1, layer_norm2}.{weight,bias}, .mlp.{fc1,fc2}.{weight,bias}, .self_attn.{q,k,v,out}_proj.{weight,bias}` (24 × 16 = 384) | `[1024]`, `[1024]`, `[4096, 1024]`, `[1024]`, `[1024, 4096]`, `[1024]`, `[1024, 1024]` (+ `[1024]` biases on all four projections) |
  | tower pre/post norms | `model.vision_embed_tokens.img_processor.vision_model.{pre_layrnorm, post_layernorm}.{weight,bias}` (4) | `[1024]` each |
  | projector | `model.vision_embed_tokens.img_projection.{0,2}.{weight,bias}` (4) | `[3072, 4096]`, `[3072]`, `[3072, 3072]`, `[3072]` |
  | **total** | **592** | |

  Notes that bind later WPs:
  - `pre_layrnorm` is **HF's own typo** (592-key index + the pinned venv's `modeling_clip.py:722`);
    the never-executed post norm is correctly spelled `post_layernorm`. The tensor plan and load
    guard must use the exact misspelled key (WP1 item 6).
  - The checkpoint has **24** tower blocks (`encoder.layers.0..23`) and no 25th; the model
    executes 23 of them (`hidden_states[-2]`) and discards block 24's output and the
    `post_layernorm` — both weight groups exist but are never executed, so the tensor plan
    accepts-and-ignores them **by name** (WP1 item 6).
  - The decoder has **no biases anywhere** (0 of 192 keys end in `.bias`); the tower is fully
    biased. Fused `qkv_proj` (9216 = 3 × 3072) and `gate_up_proj` (16384 = 2 × 8192).
  - `glb_GN` is `[1, 1, 4096]` (one global row per image) and `sub_GN` is `[1, 1, 1, 4096]`
    (broadcast newline rows across the 2×2 merge), sized `image_dim_out * 4 = 4096`, which is
    what pins the tower width at 1024 even for the WP6 tiny tower (WP1 item 7).

## 3. Config mapping (per file)

### 3.1 `config.json` (sha `567e28c6…`, 3,778 B)

| field | value | meaning / handling |
|---|---|---|
| `architectures` | `["Phi3VForCausalLM"]` | dispatch key; `model_type` is `phi3_v` |
| `auto_map` | `AutoConfig` → `configuration_phi3_v.Phi3VConfig`, `AutoModelForCausalLM` → `modeling_phi3_v.Phi3VForCausalLM` | trust-remote-code; transformers 4.57.6 has **no native `phi3_v`** — the artifact only runs its bundled code (WP1 item 1) |
| `_attn_implementation` | `"flash_attention_2"` | the decoder's config value; **not the effective reference attention** — see §5. Accepted-inert by the loader; the reference generators set and assert `eager` (§5) |
| `model_type` | `"phi3_v"` | family key; there is no `phi3_vision` type (§1) |
| `num_hidden_layers` | 32 | decoder blocks |
| `hidden_size` | 3072 | decoder width; head_dim = 3072/32 = 96 |
| `num_attention_heads` / `num_key_value_heads` | 32 / 32 | no GQA in this artifact |
| `intermediate_size` | 8192 | SiLU MLP (`hidden_act: "silu"`); fused `gate_up_proj`/`down_proj` |
| `vocab_size` | 32064 | > tokenizer `model_vocab_size` 32000 (added tokens up to 32044) |
| `tie_word_embeddings` | false | `lm_head.weight` is separate |
| `rms_norm_eps` | 1e-5 | decoder `RMSNorm` |
| `rope_theta` | 10000.0 | NTK-Su base (WP1 item 11, §8) |
| `max_position_embeddings` | 131072 | enters the Su scaling constant |
| `original_max_position_embeddings` | 4096 | the Su short/long boundary and the boundary-recompute trigger |
| `sliding_window` | 262144 | **accepted-inert**: present in the config, never read by the bundled attention code (like the qwen2/qwen3 window keys) — descriptor `WindowPolicy.IGNORE` (WP1 item 10) |
| `rope_scaling` | `type: "su"`, 48-entry `short_factor` and `long_factor` | the full arrays are below; exactly three keys, each factor array of length `head_dim / 2 = 48` (WP1 item 11, §8) |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | 1 / 2 / 32000 | `eos_token_id: 2` is the stale value — see §3.4 and §7.4 |
| `attention_dropout`, `embd_pdrop`, `resid_pdrop` | 0.0 | accepted-inert |
| `embd_layer` | `{embedding_cls: "image", hd_transform_order: "sub_glb", projection_cls: "mlp", use_hd_transform: true, with_learnable_separator: true}` | bundled `Phi3ImageEmbedding` kwargs; the loader rejects non-default values of the two non-inert ones (WP5) |
| `img_processor` | `{model_name: "openai/clip-vit-large-patch14-336", name: "clip_vision_model", image_dim_out: 1024, num_img_tokens: 144}` | `model_name` is assert-checked by the bundled code (`modeling_phi3_v.py:138`); `num_img_tokens: 144` is a legacy constant the processor ignores (it computes per-image counts, §6); `image_dim_out` pins the tower width |
| `torch_dtype` / `use_cache` / `transformers_version` / `initializer_range` / `_name_or_path` | `bfloat16` / true / 4.38.1 / 0.02 / "Phi-3.5-vision-instruct" | accepted-inert metadata |

The two 48-entry factor arrays, verbatim from the artifact (float32-round-tripped values):

- `short_factor`: `1.08, 1.1, 1.1300000000000001, 1.2800000000000002, 1.3100000000000003,
  1.4500000000000004, 1.4500000000000004, 1.9500000000000008, 2.030000000000001,
  2.4299999999999926, 2.5699999999999896, 2.9499999999999815, 3.729999999999965,
  3.869999999999962, 4.189999999999955, 4.43999999999995, 4.6399999999999455,
  4.979999999999938, 5.159999999999934, 5.279999999999932, 5.759999999999922,
  5.889999999999919, 5.889999999999919, 5.969999999999917, 6.089999999999915,
  6.2799999999999105, 6.7699999999999, 6.8899999999998975, 7.109999999999893,
  7.129999999999892, 7.179999999999891, 7.289999999999889, 7.339999999999888,
  7.559999999999883, 7.619999999999882, 7.69999999999988, 7.879999999999876,
  7.879999999999876, 7.879999999999876, 7.939999999999875, 7.949999999999875,
  7.979999999999874, 8.19999999999987, 8.439999999999864, 8.469999999999864,
  8.589999999999861, 8.809999999999857, 8.999999999999853`
- `long_factor`: `1.0800000429153442, 1.1100000143051147, 1.1399999856948853,
  1.340000033378601, 1.5899999141693115, 1.600000023841858, 1.6200000047683716,
  2.620000123977661, 3.2300000190734863, 3.2300000190734863, 4.789999961853027,
  7.400000095367432, 7.700000286102295, 9.09000015258789, 12.199999809265137,
  17.670000076293945, 24.46000099182129, 28.57000160217285, 30.420001983642578,
  30.840002059936523, 32.590003967285156, 32.93000411987305, 42.320003509521484,
  44.96000289916992, 50.340003967285156, 50.45000457763672, 57.55000305175781,
  57.93000411987305, 58.21000289916992, 60.1400032043457, 62.61000442504883,
  62.62000274658203, 62.71000289916992, 63.1400032043457, 63.1400032043457,
  63.77000427246094, 63.93000411987305, 63.96000289916992, 63.970001220703125,
  64.02999877929688, 64.06999969482422, 64.08000183105469, 64.12000274658203,
  64.41000366210938, 64.4800033569336, 64.51000213623047, 64.52999877929688,
  64.83999633789062`

### 3.2 `preprocessor_config.json` (sha `0f3d920f…`, 442 B)

| field | value |
|---|---|
| `auto_map` | `AutoImageProcessor` → `processing_phi3_v.Phi3VImageProcessor`, `AutoProcessor` → `processing_phi3_v.Phi3VProcessor` (bundled, hash-pinned) |
| `do_convert_rgb` | true |
| `image_mean` | `[0.48145466, 0.4578275, 0.40821073]` (CLIP values) |
| `image_std` | `[0.26862954, 0.26130258, 0.27577711]` |
| `num_crops` | 4 (the HD transform's `hd_num`) |
| `num_img_tokens` | 144 (legacy; the processor computes per-image counts — §6) |
| `image_processor_type` / `processor_class` | `Phi3VImageProcessor` / `Phi3VProcessor` |

No `size`/`resample`/`padding` fields: the HD pipeline is entirely in the bundled code (§6).

### 3.3 `processor_config.json` (sha `286a9e7f…`, 119 B)

Only `auto_map` (→ `processing_phi3_v.Phi3VProcessor`) and `processor_class`; no options.

### 3.4 `generation_config.json` (sha `bf86d2cd…`, 136 B) — the stale-eos quirk

```json
{"_from_model_config": true, "bos_token_id": 1, "eos_token_id": 2,
 "pad_token_id": 32000, "transformers_version": "4.38.1"}
```

`eos_token_id: 2` is `</s>` — a **non-special** token (`special_tokens_map.json` never lists it,
and it is not in `tokenizer.json`'s special table). It is stale: every `generate` call in
`sample_inference.py` overrides it with `eos_token_id=processor.tokenizer.eos_token_id`, which
is **32000** (the `endoftext` literal, special, and also the pad token) — golden `stop`
`sample_inference_eos_lines` records all 11 generate call sites (ten two-line plus one one-line),
each carrying the override. The reference therefore stops on **{32000}**; the WP5 default stop
set and the WP8 manifest pin follow §7.4.

### 3.5 `tokenizer_config.json` (sha `581d4654…`, 9,519 B) + `special_tokens_map.json` (sha `f9535fad…`, 670 B)

- `tokenizer_class: "LlamaTokenizer"`, `add_bos_token: true`, `model_max_length: 131072`.
- Chat template (verbatim from the golden `tok.tokenizer_config.chat_template`), a plain
  Llama-style Jinja string in which image parts do **not** appear — they enter as `<|image_i|>`
  markers in message text (WP3):

  ```jinja
  {% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{- '<|assistant|>\n' -}}{% endif %}
  ```

  Rendering this template through `jmlx-jinja` is verified in WP3 (WP0's last open item), as is
  the check that the marker strings are not special tokens.
- `additional_special_tokens`: `["<|system|>", "<|end|>", "<|user|>", "<|end|>"]` (note the
  duplicated `<|end|>`).
- `special_tokens_map.json`: `bos_token` = `<s>`; `eos_token` and `pad_token` = the same
  13-char `endoftext` literal (by ords `60,124,101,110,100,111,102,116,101,120,116,124,62`,
  i.e. `<|` + `endoftext` + `|>`); `unk_token` = `<unk>`.
- `tokenizer.json` structure (golden `tok`; WP0's BPE-shape claim confirmed):
  `model_vocab_size` 32000, `merges_count` 61249, `added_tokens_count` 48,
  `added_tokens_max_id` 32044; BPE model with `byte_fallback: true` and `fuse_unk: true`
  (**BPE model flags**, not a Precompiled normalizer — the 7.2 charsmap code does not apply);
  normalizer = `Sequence[Prepend("▁"), Replace(" " → "▁")]`; `pre_tokenizer` = null;
  decoder = `Sequence[Replace("▁" → " "), ByteFallback, Fuse, Strip(start=1, stop=0, " ")]`;
  post_processor = `TemplateProcessing` whose single pair prepends `<s>` (id 1) to every
  encoding.
- Special tokens by id (golden `stop.ids` and `tok.specials`, by ords):
  1 `<s>` (special); 2 `</s>` (**not** special, ords `60,47,115,62`); 13 the newline
  byte-fallback token (decodes to the 2-char `\n` escape, ords `10`); 32000 the `endoftext`
  literal (special; eos and pad); 32007 `<|end|>` (special, ords `60,124,101,110,100,124,62`);
  32044 `<|image|>` (special — the marker regex `<\|image_\d+\|>` never matches it, so it
  creates no image slot in the reference; user text containing it is passed through, WP3
  decision).

## 4. Execution-mode gate (WP1 items 2, 13)

Probed by execution in the pinned venv (tiny eager model, random weights — no checkpoint load
needed; the safetensors shards are therefore not inputs to the probe).

- **P1 — fresh prefill with `use_cache=True`** (the default): raises
  `AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'` at
  `modeling_phi3_v.py:1417` (`past_key_values_length = past_key_values.get_usable_length(seq_length)`),
  called from `:1603` (`outputs = self.model(`). 4.57.6's `Cache` has no `get_usable_length`
  (it was added later); since `:1417` runs whenever `use_cache` is true, **even a fresh prefill
  fails**.
- **P2 — `prepare_inputs_for_generation` with a populated `DynamicCache` past that crosses the
  `original_max` boundary**: raises `AttributeError: 'DynamicCache' object has no attribute
  'seen_tokens'` at `modeling_phi3_v.py:1653` (the exact read is
  `past_length = past_key_values.seen_tokens if isinstance(past_key_values, Cache) else past_key_values[0][0].shape[2]`).
  Under 4.57.6 `isinstance(past_key_values, Cache)` is true (`DynamicCache` subclasses `Cache`,
  `cache_utils.py:910`), so the `seen_tokens` branch is taken and raises; the non-Cache
  `past_key_values[0][0].shape[2]` fallback never runs.

**Conclusions** (binding for every generator in this milestone):
1. No cached decode can complete in the pinned venv; `sample_inference.py`'s own `generate`
   cannot complete past the first step.
2. Every reference generator (WP6/WP8) therefore runs an **uncached full forward with
   `use_cache=False` and a manual greedy loop**; there is no compatibility shim and no separate
   transformers pin. Under `use_cache=False`, `past_key_values` is `None` and none of
   `modeling_phi3_v.py:1650`–`:1661` (the boundary block) runs at all.
3. The item-11 boundary semantics (§8) are therefore a **code-reading conclusion plus the
   observed failure above**, not an executed path; the uncached full-history forward per step
   embodies exactly the semantics jmlx's cached decode must reproduce.

## 5. Attention mechanism (WP1 item 3)

The 4.57.6 dispatch, source-verified in the pinned venv:

- `PretrainedConfig.__init__` only stores `self._attn_implementation = kwargs.pop("attn_implementation", None)` — a
  bare `CLIPVisionConfig()` (and the bundled module-level constant, which is built without the
  keyword, `modeling_phi3_v.py:68`–`:81`) is born **None**.
- `PreTrainedModel.__init__` sets `config._attn_implementation_internal = self._check_and_adjust_attn_implementation(config._attn_implementation, is_init_check=True)`;
  `get_correct_attn_implementation(None)` returns **"sdpa"** (falling back to "eager" only if
  `_sdpa_can_dispatch` fails).
- **Construction mutates the config in place.** Directly constructing
  `CLIPVisionModel(cfg)` with `cfg._attn_implementation = None` leaves `cfg` (the same object)
  with **both** `_attn_implementation` and `_attn_implementation_internal` equal to `"sdpa"`
  afterwards — verified by a minimal 1-layer construction in the pinned venv. Because the
  bundled `Phi3ImageEmbedding.__init__` builds the tower from the module-level constant as-is
  (`modeling_phi3_v.py:139`–`:140`: `clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG` then
  `CLIPVisionModel(clip_config)`), **the first model build in a process permanently mutates the
  shared constant** from None to sdpa. Any later code reading
  `CLIP_VIT_LARGE_PATCH14_336_CONFIG._attn_implementation` sees "sdpa", not the pristine None.
- Probe-ordering consequence (this milestone's P4 fix): an earlier probe draft ran P1/P2 first,
  and P1/P2 swap `model_mod.CLIP_VIT_LARGE_PATCH14_336_CONFIG` for a tiny **eager** tower
  before P4 read the constant, so the first two goldens recorded `bundled_clip_constant_attn_implementation:
  "eager"` — a contamination, not the pristine value. The final probe runs **P4 first** and
  records the constant before any build; the two superseded goldens (shas `119d07e1…`,
  `4b398d0e…`) are void and the final golden's P4 record below is authoritative.
- If the **decoder** config is `flash_attention_2`, the bundled code additionally swaps every
  tower layer's `self_attn` for `CLIPAttentionFA2` (`modeling_phi3_v.py:145`–`:149`); with the
  generator's eager decoder config this swap does not happen.

Golden `p4` (authoritative):

```json
{
 "artifact_config_attn_implementation": "flash_attention_2",
 "bare_clipvisionconfig_attn_implementation": null,
 "bundled_clip_constant_attn_implementation": null,
 "clipvisionmodel_default_attn_implementation": null,
 "tower_default_self_attn_class": "CLIPAttention",
 "tower_config_attn_after_build": "sdpa",
 "tower_attn_internal_after_build": "sdpa",
 "eager_construction": {
  "decoder_self_attn_class": "Phi3Attention",
  "tower_self_attn_class": "CLIPAttention",
  "model_config_attn_after_build": "eager",
  "tower_config_attn_after_build": "eager",
  "failed": false
 }
}
```

(`clipvisionmodel_default_attn_implementation` is null because `CLIPVisionModel` has no
`_default_attn_implementation` class attribute in 4.57.6; the eager-construction block is the
WP6 generator path — tiny decoder config with `_attn_implementation="eager"` plus a tiny eager
tower constant swapped into the module attribute — and it succeeds with both attention classes
plain, non-FA2.)

**Generator rule** (binding): the reference generators set **and assert**
`_attn_implementation = "eager"` on both the model config and the tower config, and never rely
on the bundled constant's value (which is process-state dependent per the mutation above). The
effective reference attention for this family is therefore eager on both sides; the artifact's
`flash_attention_2` field is accepted-inert metadata (§3.1).

## 6. Preprocessing pipeline (WP1 items 4, 5, 12, 13)

### 6.1 The HD transform (bundled `processing_phi3_v.py:83`–`:104`, verbatim semantics)

```python
def HD_transform(img, hd_num=16):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * 336)
    new_h = int(new_w / ratio)
    img = torchvision.transforms.functional.resize(img, [new_h, new_w])
    img = padding_336(img)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)
    return img
```

with `padding_336` (`:62`–`:71`): `tar = int(np.ceil(height / 336) * 336)`,
`top_padding = int((tar - height) / 2)`, bottom the remainder — white-255 top/bottom padding
only, no left/right. The call site passes `hd_num = self.num_crops = 4`. With `hd_num = 4` the
scale loop **always** picks `scale >= 2`, so a 336×336 square resizes to 672×672 (`n = 4` crop
blocks; the 313-row `n = 1` outcome is unreachable). `new_h` is a **double `int()`
truncation** (`int(new_w / ratio)` on the float division), which is what makes the pinned
truncation example hold: a 300×301 source → `new_h = 669` rows, padded 1 top / 2 bottom
(golden `p6.image_cases[trunc-300x301]`).

### 6.2 The PIL hand-off — verified (WP1 items 12, 13)

torchvision 0.21.0's `F.resize(pil_image, [h, w])` with the default interpolation is
`torchvision_default_interpolation: "BILINEAR"` (golden `p5`) and **delegates to
`PIL.Image.resize((w, h), Image.Resampling.BILINEAR)`**: golden `p5.cases` records five seeded
RGB inputs (seeds 201–205) — the transposed 300×301 truncation case, the 336×336 square, the
672×336 wide, a 251×137 → 784×168 odd upscale, and a 511×223 → 400×909 odd downscale — for which
the torchvision and PIL outputs are **byte-identical** in every case (`identical_bytes: true`,
per-case SHA-256 pairs in the golden). The existing Pillow-exact bilinear kernel therefore
covers the HD resize; only the global image's float32 bicubic
(`F.interpolate(..., size=(336,336), mode="bicubic")` on the **normalized** tensor) needs a new
Torch-exact kernel (WP2). Both kernels' provenance goes into `jmlx-vision/NOTICE` in the 7.3a
pattern.

### 6.3 Pinned preprocessing cases (golden `p6.image_cases`)

Seeded random RGB inputs (seeds 101–104; two-image case reuses the square + wide):

| case | source (w×h) | `image_sizes` (padded h, w) | `num_img_tokens` | `pixel_values` sha256 |
|---|---|---|---:|---|
| `square-336x336` | 336×336 | `[[672, 672]]` | 757 | `0292f014488f681947391a5daf5d805c6900c2098d16e171b9228c199ee7f7e7` |
| `wide-672x336` | 672×336 | `[[336, 672]]` | 457 | `6cdac6a70c10c927b3345a5cd3fe683f96d8acd0a0e368854ca9ff257cc71732` |
| `tall-336x672` | 336×672 | `[[672, 336]]` | 469 | `3869d7af80146c160ce639a95a4ad34f831feee3177acd5e677c12848febe396` |
| `trunc-300x301` | 300×301 | `[[672, 672]]` | 757 | `ca8760988fa296c40e768702c00d43894a02f1e6f8fc9fbf6738758263aa0375` |
| `two-image-square+wide` | 336×336, 672×336 | `[[672, 672], [336, 672]]` | [757, 457] | `546e53f45a5e79fd9f7bf6a4cf5dc047133d7e879e26903c5db11ef69a6233a5` |

Every case: `pixel_values` shape `(n_images, 5, 3, 336, 336)`, dtype `torch.float32` (the
processor output, pre-model). Per-slot SHA-256s are in the golden (`slot_sha256`, image-major
then slot). Slot 0 is the global bicubic; slots 1–4 are the sub crops in h-major order,
zero-padded to `num_crops + 1 = 5` — the wide and tall cases' unused slots 3–4 share the
all-zero tensor sha `90ef90f5e407dee7383f0cbcf712dee990ed52e1621df372b49432db66231fd2`.
**Zero-padded crop slots are computed by the reference but never read** by the tower (crops are
processed independently and the merge only consumes the `n` real slots), so the Java port may
skip computing them without affecting any used output.

The per-image token formula is `(n + 1) * 144 + (h336 + 1) * 12 + 1` with `n = h336 * w336`
(the padded grid in units of 336) — confirmed for all five cases; with `num_crops = 4` the
reachable per-image counts are {457, 469, 601, 625, 745, 757, 781}, maximum 781 at
`h336 = 4, w336 = 1` (WP1 item 5).

### 6.4 Pipeline facts that bind the port

- `BatchFeature(return_tensors="pt")` **auto-tensorizes** `image_sizes` and `num_img_tokens`
  (the probe records them back as plain ints; the golden comment pins this behavior) — the WP2
  result record carries them as plain `int[]`/`int[][]`.
- The reference **never calls `exif_transpose`** anywhere in the decode or processor path, so
  the port decodes with `ExifOrientationPolicy.IGNORE` (the 7.3b WP9 policy choice carried over
  — WP1 item 4).
- Normalization is the division `(x - mean) / std` (not a reciprocal multiply), on the
  `ToTensor` output in [0, 1]; the global 336×336 bicubic runs on the **normalized float32
  tensor**, not on pixels (WP1 item 4).
- The processor's `calc_num_image_tokens` (`:172`–`:194`, call site `:269`) computes the
  formula from the **padded** sizes, which is why `image_sizes` (the padded grid) is what the
  text expander consumes (§7, WP3).

### 6.5 The torch bicubic kernel's real arithmetic — pinned-wheel FMA contraction (WP2)

The global-image bicubic ported in WP2 is **not** a plain `*`/`+` transcription of the staged
PyTorch v2.6.0 source. The pinned CPU wheels are compiled with floating-point contraction
enabled (the C++ default), so every C subexpression of the form `a * b + c` in the kernel
executes as a single-rounded hardware fused multiply-add (AArch64 `fmla s`): the product
`a * b` is formed exactly and only the final sum is rounded to float32. Verified on the
37×41 → 17×23 probe (1,173 output pixels, `RandomState(0)` input, bit-exact comparison
against the pinned venv's `F.interpolate(mode='bicubic', align_corners=False,
antialias=False)`); the winning model is fully specified in `TorchBicubic.java`:

- **Real source index** (`area_pixel_compute_source_index`, align_corners=false):
  `real = fmaf(scale, (float)(i + 0.5f), -0.5f)`; `input_index = min(floorf(real), in-1)`;
  `lambda = clamp(real - input_index, 0, 1)`; `scale = (float)in / (float)out` plain.
- **Cubic polynomials** (UpSample.h:400–407, `A = -0.75`; the constants `A+2 = 1.25`,
  `A+3 = 2.25`, `-5A = 3.75`, `8A = -6.0`, `-4A = 3.0` are all exact in f32):
  `cubic_convolution1(x)`: `t = fmaf(1.25f, x, -2.25f); t *= x; return fmaf(t, x, 1.0f)`;
  `cubic_convolution2(x)`: `t = fmaf(-0.75f, x, 3.75f); t = fmaf(t, x, -6.0f);
  return fmaf(t, x, 3.0f)`.
- **Weights** (`get_cubic_upsample_coefficients`): `x1 = lambda`, `x2 = 1.0f - x1` (plain),
  `[c2(x1 + 1.0f), c1(x1), c1(x2), c2(x2 + 1.0f)]`.
- **Separable accumulation** (`Interpolate<2>`/`<1>`, height-outer/width-inner): the first
  term is a bare product; each later term at both the width-row and height level is
  `acc = fmaf(term, w, acc)`.

A plain `*`/`+` transcription of the staged source deviates from the pinned wheels by up to
~881 float ulps (max relative error ~8e-5) on the probe; the fused form matches bit-exactly
(1173/1173). Java has no float FMA, so `TorchBicubic.fma32` emulates it bit-exactly (exact
double product — a binary32×binary32 product has at most 48 significand bits — plus an
error-free two-sum plus an exact adjacent-float midpoint comparison with ties-to-even).
Edge cases verified against libm `fmaf`: `0 * inf → NaN`; `inf + same-sign inf = inf`,
`inf + opposite-sign inf = NaN`; zero signs are read from raw sign bits because
`-0.0f < 0.0f` is false in Java; `z + z` keeps the common sign when the two zeros agree,
`+0` when they differ. `fma32` stress evidence: 600,034 cases (uniform, cancellation,
extreme exponents, targeted) plus a 50,096-case zero-sign zoo, zero mismatches against
libm; the 1,173-pixel Java probe is bit-exact against the venv's torch. The module's own
`PillowResample` BICUBIC (Pillow's 8-bit fixed-point `a = -0.5` pipeline) is a different
algorithm and is untouched. Earlier candidates (plain/FMA accumulation variants, f64
tables, f64 accumulators, flat 16-term orders) were all falsified by the probe and are
superseded by the model above.

## 7. Text path, tokenizer, stop tokens (WP1 items 8, 9, 14)

### 7.1 Prompt expansion (golden `p6.text_cases`)

`_convert_images_texts_to_inputs` (`processing_phi3_v.py:409`–`:463`) splits on
`re.split(r"<\|image_\d+\|>", text)`, tokenizes **each chunk separately** (BPE merges never
cross a marker boundary), and expands marker `i` to `num_img_tokens[i-1]` sentinel rows of
`-i`. All four probe cases (attention mask min = max = 1, i.e. all ones):

| case | text (markers shown) | input_ids (compact) | total |
|---|---|---|---:|
| `mid-marker` | `A<\|image_1\|>B` | `[1, 319, -1 × 757, 1, 350]` | 761 |
| `leading-marker` | `<\|image_1\|>tail` | `[1, -1 × 757, 1, 12464]` | 760 |
| `trailing-marker` | `mid<\|image_1\|>` | `[1, 7145, -1 × 757, 1]` | 760 |
| `adjacent-markers` | `<\|image_1\|><\|image_2\|>` | `[1, -1 × 757, 1, -2 × 457, 1]` | 1217 |

Key facts:
- The prompt builder sets `offset = 0` (`:444`–`:446`) and strips nothing, so **every chunk —
  including chunks after an image block — starts with the per-encoding `<s>` (id 1)**, i.e. a
  mid-sequence BOS (followed by `▁` from the space replacement when the chunk text has a
  leading space). Golden `p3`: empty string → `[1]`; `"hello"` → `[1, 22172]`; `" hello"` →
  `[1, 29871, 22172]` — 29871 is the standalone `▁` token (BPE does not merge it into `hello`
  in this vocabulary), and `tok.decode([29871])` is `""` because the decoder's
  `Strip(start=1, stop=0, " ")` drops its leading space.
- **The empty-chunk case — the plan's WP1 item 9 gap — is resolved: an empty chunk encodes to
  exactly `[1]` (the BOS only).** Both the trailing-marker and the adjacent-markers cases
  exhibit it (the second chunk in `trailing-marker` and the chunk between the two markers in
  `adjacent-markers` are empty, and each contributes a single `1`).
- Images bind to sentinel rows **by position, not marker id** (`modeling_phi3_v.py:228`,
  `:239`–`:242`: the sentinel rows' `nonzero` positions receive the projected rows via
  `index_put`); the reference checks only that the unique marker ids are exactly `1..n` and that
  the unique count equals the image count (`:430`–`:433`), not the total marker count. A
  duplicated marker fails loudly (2N sentinel rows vs an N-row projection); out-of-order
  markers silently misbind. The jmlx-side rejections are a documented hardening (WP3).

### 7.2 Tokenizer coverage check (WP0's last open item)

The pinned `tokenizer.json` shape — SentencePiece-style BPE with
`Prepend("▁") + Replace(" " → "▁")` normalizer, **no pre-tokenizer**, `byte_fallback`/`fuse_unk`
as BPE model flags, `Replace + ByteFallback + Fuse + Strip(start=1)` decoder,
`TemplateProcessing` BOS prepend — is covered by `jmlx-tokenizer`'s Metaspace BPE +
`TextNormalizer` + `AddedTokenSplitter` components as they stand; the WP3 golden (this family's
prompt-expansion ids, including the per-chunk BOS) is the executable proof, and the chat
template's `jmlx-jinja` render is WP3's second case.

### 7.3 Stop tokens (golden `stop`)

- `tok_bos_token_id = 1`, `tok_eos_token_id = 32000`, `tok_pad_token_id = 32000`.
- `decode([32000])` → the 13-char `endoftext` literal with `skip_special_tokens=False` (ords
  `60,124,101,110,100,111,102,116,101,120,116,124,62` — written by ords here because transport
  strips the rendered form) and `""` with `skip_special_tokens=True`.
- `decode([32007])` → `<|end|>` (ords `60,124,101,110,100,124,62`) / `""`;
  `decode([13])` → the 2-char `\n` escape (ords `10`) — a newline in the output does **not**
  stop generation.
- `sample_inference.py` passes `eos_token_id=processor.tokenizer.eos_token_id` (32000) at all
  11 generate call sites (§3.4), overriding `generation_config.json`'s stale `eos_token_id: 2`;
  it adds nothing else.

**Decision (plan §5, confirmed by the probe):** the default stop set is **{32000}**, carried in
`GenerationConfig.eosTokenIds` (not `stopTokenIds`), matching the repo's Tier-B convention of
stopping on the tokenizer's own eos. With that stop set the model typically emits `<|end|>`
(32007) **before** the stop; `generatedText()` never contains either 32007 (special → stripped
by the shared loops' hardcoded `tokenizer.newIncrementalDecoder(true)`) or 32000 (it stays in
the generated IDs). The WP8 manifest pins `stop = {32000}` and the
`expected_text = decode(greedy_ids, skip_special_tokens=True)` rule.

## 8. NTK-Su RoPE and the boundary recompute (WP1 item 11)

Code-reading conclusion (confirmed by the observed P2 failure, §4), binding for WP4/WP5/WP6:

- Per forward call, `seq_len = kv_seq_len` (`modeling_phi3_v.py:669`) — the full prefill length
  for a fresh prefill, the cached key length during cached decode. The factor table is
  `long_factor` iff `seq_len > original_max_position_embeddings` (4096 in the artifact), else
  `short_factor` — 48 per-dimension entries each. `inv_freq[i] = 1 / (f[i] * 10000 ** (2i / 96))`;
  a global `scaling = sqrt(1 + ln(131072 / 4096) / ln(4096))` (≈ 1.19023809) multiplies **both**
  cos and sin. The reference computes in float32 and casts to the model dtype; `RopeSpec`'s
  double → float32 is the stated tolerance (plan §2).
- **Boundary recompute:** `modeling_phi3_v.py:1650`–`:1655` drops `past_key_values` to `None`
  when `input_ids.shape[1] >= original_max_position_embeddings + 1` and the past length
  (number of cached positions) is `<= original_max_position_embeddings`, forcing a full
  recompute. The reference therefore generates as if **every** key uses the long table once the
  sequence passes the boundary — including keys written before it — and a "cached keys keep the
  table that wrote them" behavior would be a jmlx-only divergence. The past-length read at
  `:1653` takes the `seen_tokens` branch under 4.57.6 and would raise `AttributeError` if
  executed (observed, P2); under the generators' `use_cache=False` mode none of
  `:1650`–`:1661` runs at all. The per-step fresh forward of the full history embodies the
  long-table-from-crossing semantics that jmlx's cached decode must reproduce — including
  re-running the vision tower at the crossing, since `prepare_inputs_for_generation` passes
  `pixel_values`/`image_sizes` on every step (`:1699`–`:1707`) and `Phi3ImageEmbedding.forward`
  re-injects the image rows (§5 decision: retain the evaluated per-image projection rows —
  ≤ 781 × 3072 × 4 B ≈ 9.6 MB/image — in the request scope and re-inject them).
- `Phi3VConfig.__init__` (bundled `configuration_phi3_v.py`) validates the rope_scaling block:
  exactly three keys (`type`, `short_factor`, `long_factor`), `type ∈ {su, yarn}`, and both
  factor arrays of length `hidden_size // num_attention_heads // 2` (48 for the artifact; the
  WP6 tiny checkpoint's head_dim 64 → 32). The `yarn` branch in the bundled code is a
  straight-pass to the same per-dimension factors; the jmlx descriptor consumes only `su` and
  rejects `yarn` by name on this family (and keeps the text `phi3` longrope rejection — the
  shared parse branch is gated by an explicit family flag, plan §5).

## 9. `phi-3-vision-128k` (WP1 item 15)

Golden `deferred_128k`: **not accessible (gated); `num_crops` unverified; no support claim.**
The config diff recorded in the plan (different rope factors, `sliding_window` 131072, null
`pad_token_id`/dropout) is therefore evidence of the shared `model_type` only; support and
verification claims stay pinned to the 4.2B artifact (§1), and the descriptor (WP5) handles the
128k shape by the same rules without a name-based exclusion (no `phi3_vision` type exists).

## 10. Toolchain changes (WP1 item 1)

- **torchvision 0.21.0** (the torchvision matching torch 2.6.0) added to **both**
  `tools/hf-reference` and `tools/image-oracle` hash-locked venvs (installed and verified:
  torch 2.6.0 / torchvision 0.21.0 in each; `verify_environment.py` now checks both packages).
  Wheel SHA-256s, recorded in `tools/hf-reference/provenance.json` under
  `phi3v.torchvision.wheel_sha256` and verified against the actual downloaded files:
  - `torchvision-0.21.0-cp312-cp312-macosx_11_0_arm64.whl`
    `97a5814a93c793aaf0179cfc7f916024f4b63218929aee977b645633d074a49f`
  - `torchvision-0.21.0+cpu-cp312-cp312-linux_x86_64.whl`
    `d6874431e678ba107b60a83f255c33f3755f06bad587b1b919aa514ec325dcd8`
  Both locks keep the platform-marked torch lines (`2.6.0` darwin / `2.6.0+cpu` linux) and the
  single `torchvision==0.21.0` line; the provenance scalars store the base versions
  (`"2.6.0"` / `"0.21.0"`), and `verify_environment.py` strips the `+cpu` marker before
  comparing. Final lock SHAs: hf-reference `d370b90e337c1304aff7a858ea2a4546b49122368e7bdf2e034a275709bf9bc9`
  (recorded in `phase72.requirements_lock_sha256`), image-oracle
  `ddc97209e0a9b643696daf4a761808dcc9b84e87e855edcfe4932d3c1756427f`
  (not digest-pinned anywhere; cross-checked by `verifyImageOracleDependencyPins`).
- **Bundled-code fixture:** the five trust-remote-code bundle files
  (`configuration_phi3_v.py`, `modeling_phi3_v.py`, `processing_phi3_v.py`,
  `sample_inference.py`, `preprocessor_config.json`) are committed under
  `tools/image-oracle/fixtures/phi3v/`, byte-identical to the artifact (all five SHAs in §2),
  digest-checked by `verifyImageOracleFixtureDigests` (exact-set) and re-verified by
  `phi3v_probe.py` before import.
- **Guards:** `verifyHfReferenceGoldens` gained the probe generator's input and the
  `phi3v_probe_sha256` check (the `qwen3vl_wp8` pattern); `verifyImageOracleDependencyPins` now
  also requires the image-oracle and hf-reference locks to agree on torch and torchvision
  (last-match-wins per package; both locks resolve to the identical version + marker).
- **Known pre-existing quirk (not fixed, recorded):** `generate.py update_provenance` rewrites
  `tools/hf-reference/provenance.json` wholesale on regeneration, retaining only the base keys
  plus `chat_sources` — the phase keys (`phase72`, `qwen3vl_*`, `phi3v`) would be dropped and
  `verifyHfReferenceGoldens` fails loudly on it. The same convention already applied to 7.3b;
  regeneration of the phi3v goldens must be followed by re-adding the phase provenance blocks
  (as done for this WP).

## 11. Determinism, environment, command archive

- **Probe environment** (golden `env`): python 3.12.16, torch 2.6.0 (6 threads), torchvision
  0.21.0, transformers 4.57.6, numpy 2.5.3, pillow 12.3.0, Darwin/arm64. Host: the local M5 Max
  (macOS 27.0.1, build 26A434) — the same host that later runs the WP8 local Tier-B evidence.
- **Determinism:** two consecutive probe runs produced byte-identical goldens
  (`06734b2207b8b418807ef178459366206ccda3307b150cc5f686f7467c753f73` both times).
- **Command archive** (re-runnable from a checkout with the 13 small WP0 files in
  `<dir>`; the shards are not needed):

  ```sh
  cd tools/hf-reference
  .venv/bin/python phi3v_probe.py --directory <dir>   # rewrites goldens/phi3v-wp1-probe.json, prints its sha256
  ```

  Golden regeneration is the only step that rewrites committed evidence; `verifyHfReferenceGoldens`
  then re-checks the digest.

- **WP1 gate — resolved claims.** Item 2 (cache-API failures) by execution (P1/P2, §4); item 9
  (empty chunk = `[1]`) by execution (§7.1); item 11 (boundary flow) by code reading plus the
  observed failure (§8); item 12 (PIL hand-off) by execution (P5, §6.2). The torchvision pin and
  all bundled-code hashes are in `provenance.json` (§10). No claim depends on unpinned remote
  state: the 128k artifact is explicitly not used (§9).

## 12. WP4 implementation evidence — `RopeSpec.Su`

- **Semantics landed** (from §8 into `se.alipsa.jmlx.nn.RopeSpec.Su`): the factor table is long
  only when the per-call sequence length is strictly greater than `originalMaxPositions`
  (4096 short, 4097 long); the reference's per-dimension division of `inv_freq` becomes a
  per-dimension **multiplication** of the MLX period, since jmlx's `freqs` array holds periods
  (`periods[i] = f[i] * theta^(2i/96)`); the static scaling `sqrt(1 + ln(max/orig)/ln(orig))`
  (1.190238… for the artifact; 1.0 when the context is not extended) is supplied through
  `attentionScaling()` — the reference multiplies both cos and sin, and `RopeSpec.apply`
  post-multiplies the rotated output by the same constant; `isStatic()` is false so
  `stepFrequencies` re-selects the table per call; `apply(x, dims, offset, null)` selects the
  table at `offset + T` (the kv length), the reference's `seq_len`. The boundary recompute
  (dropping the cache and re-prefilling with the long table) stays a model-level generation-loop
  concern for WP5, not a `RopeSpec` behavior.
- **Differential golden** `tools/hf-reference/goldens/phi3v-rope.json` (generated by the
  digest-pinned `tools/hf-reference/phi3v_rope.py`, never Gradle-invoked; it verifies the
  SHA-256 of the pinned artifact's `config.json` and both bundled modules first, then executes
  the bundled `Phi3SuScaledRotaryEmbedding` and `apply_rotary_pos_emb` exactly, as a synthetic
  package import of the pinned files). It records the selected `inv_freq` for the kv lengths
  128 / 4096 / 4097 / 131072 and single-position rotations at offsets 0 / 17 / 4095 / 4096 /
  131071 (head_dim 96, input `i/96`, 4095 the last short position, 4096 the first long one).
- **Tolerances:** the pure period check uses `max(1e-5, period * 1e-6)` (the stated
  double-vs-float32 tolerance, plan §2); the rotated check uses the golden's per-sample
  `tolerance = max(2e-4, 4 * ulp32(max angle))`, because both the reference (a float32
  position×inv_freq matmul) and the MLX kernel round the angle to float32.
- **Measured** (local M5 Max, native): max |Δrotated| = 4.77e-7 (offset 0), 5.96e-7 (17),
  4.11e-5 (4095), 2.97e-4 (4096), 8.00e-3 (131071), against tolerances 2e-4 / 2e-4 /
  9.77e-4 / 9.77e-4 / 3.125e-2.
- **Core unit tests** are hand-computed (theta 10000, dims 8: base periods 1, 10, 100, 1000;
  short factors [2, 3, 4, 5] → periods [2, 30, 400, 5000]; the 4096/4097 boundary; max 8192 /
  original 1024 → scaling `sqrt(1 + ln 8 / ln 1024) = sqrt(1.3)`; scaling 1.0 when max ≤
  original; constructor and factor-length rejections).
- **No `mlx_h` addition**; `generateMlxApiInventory` stays green unchanged. The
  length-dependent batch guard (`RopeSpec.apply`'s unequal-position path and `DecoderModel`'s
  batch cache check) is extended to `Su` alongside `DynamicNtk`.

## 13. WP5 implementation evidence — `decode∘encode` is lossy: the prompt text is not recoverable from IDs

- **Finding (a property of the pinned tokenizer, not a jmlx defect).** The reference
  `tokenizers` encoder (jmlx's `HfTokenizer` reproduces its IDs exactly — differential below)
  is lossy at `special: true` added-token boundaries:
  - a whitespace flanking a special token is **duplicated**: `"mid <s> literal"` →
    `[1, 7145, 29871, 1, 29871, 16333]` (the `29871` = `▁`-token appears once per side),
    which decodes (`skip_special_tokens=false`) to `"mid <s>  literal"` — one extra space;
  - a **trailing newline after the final special token is dropped**: `"<|assistant|>\n"` and
    `"<|assistant|>"` encode to the identical `[1, 32001]` — the ID stream does not
    distinguish them, so the rendered text is provably not recoverable from its IDs.
    `"<|user|>\n…<|assistant|>\n"` (every rendered chat prompt of this artifact ends this way)
    encodes to `[1, 32010, 29871, 13, …, 32007, 29871, 13, 32001]` with no token for the last
    newline;
  - the image markers are unaffected — `<|image_i|>` is ordinary BPE text, not an added
    token: `"<|image_1|>\n"` decodes unchanged, `"A <|image_1|> B"` round-trips exactly.
  A mid-string newline after a special token is recoverable (`"<|user|>\n"` →
  `[1, 32010, 29871, 13]` vs `"<|user|> "` → `[1, 32010, 259]`); only the trailing one is
  lost, but that one decides the last chunk's encoding, so per-chunk re-tokenization from
  a recovered text could never match the reference's `expanded_ids` for a chat prompt.
- **Differential (reference vs jmlx, WP3 fixture tokenizer).** For `"mid <s> literal"`,
  `"<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"`,
  `"<s>leading bos literal"` and `"A <|image_1|> B"`: jmlx `encode(text, addBos=true)`
  equals the reference's IDs exactly and jmlx `decode(ids, skipSpecials=false)` equals the
  reference's decoded string exactly — including the duplicated space and the dropped
  trailing newline. jmlx is faithful; the round trip is lossy on both sides.
- **Decision (plan §2 additive-public-change rule).** `decode∘encode` cannot carry the
  rendered prompt, so `Phi3VModel` does not recover text from `request.promptTokenIds()`.
  Instead `GenerationRequest` gains a nullable `promptText` accessor, additive: the
  `text` factory retains its prompt, the `chat` factory retains its rendered text,
  pre-tokenized requests retain `null` (and are rejected by `Phi3VModel`, whose marker
  expansion needs the tokenizer's per-chunk encoding). The request's ID stream is still
  validated and still what a CANCELLED result reports; the ADD/OMIT policy no longer
  constrains expansion, because the text — not the IDs — is the expansion input.

### WP5 native implementation evidence

- **Frequency-seed sentinel wrap.** The reference keeps the `-i` sentinel rows in its
  `input_ids` tensor for the whole run; torch's negative indexing wraps them to
  `vocab_size - i` wherever a penalty gathers over that tensor. `Phi3VModel` therefore feeds
  the step loop a frequency seed with the sentinels wrapped (`vocab_size + (-i)`), while the
  *embedding* inputs clamp them to row 0 — the merge hook replaces those rows, so the clamp
  value is inert. Greedy (no penalties) never reads the seed's sentinel counts; the wrap
  matters for seeded and penalty requests and is the faithful reading of the reference.
- **Tiny-tokenizer byte-level space.** The 21-token `phi3v-tiny-tokens` fixture stores the
  space as the byte-level escape `Ġ` (U+0120), not a literal U+0020: the ByteLevel
  pre-tokenizer maps byte 32 to `Ġ` before the BPE vocab lookup, so a literal-space key is
  never matched and every space in an encoding is silently dropped (probed: `"toast "` →
  `[t,o,a,s,t]`, 5 IDs instead of 6). With the escape key, `"toast "` encodes to 6 IDs and
  the boundary/context arithmetic of `Phi3VModelTest` holds (456-token prompt at 76 × 6).
- **Pinned-index typo.** The artifact's `model.safetensors.index.json` confirms the
  typo'd `pre_layrnorm` key (`vision_model.pre_layrnorm.{weight,bias}`); the tiny
  checkpoint fixture writes the same spelling.
- **HD-transform reference alignment.** `add_image_newline` concatenates the `[n, h, w,
  4096]` features with the `[n, h, 1, 4096]` newline along the **width axis** (torch
  `dim=2`), appending one `sub_GN` column at the end of every row — not a trailing channel
  slice. The `sub_glb` extend order is three **2D** row blocks (the per-image sub rows
  `.squeeze(0)`, `glb_GN.squeeze(0)` = `[1, 4096]`, and the per-image global row indexed
  from `[1, 156, 4096]`), concatenated along dim 0; `Phi3VVisionTower.hdRows` reproduces
  that exactly and the projector therefore sees `[rows, 4096]` (the model reshapes the
  concatenated per-image blocks to `[1, R, 4096]` for the row-replacement merge hook).
- **Boundary recompute vs independent reference.** `Phi3VModelTest.boundaryStepRePrefillsUnderTheLongFactorTable`
  (456 prompt tokens, maxNew 16 — decode step 15 first reaches total width 471) matches a
  from-scratch per-step fresh-forward reference (fresh `KVCache`s, one full `stepLogits` per
  step, greedy on the last row, short table ≤ 470 / long table > 470) exactly;
  `promptBeyondOriginalMaxNeverRecomputes` (474 prompt tokens) confirms no re-prefill fires
  when the prompt already exceeds 470.

## 14. WP6 implementation evidence — Tier-A synthetic golden

- **Shared embedding, 77 keys / 76 tensors.** The bundled `Phi3VModel.__init__` (line 1351)
  builds `Phi3ImageEmbedding(config, wte=self.embed_tokens, ...)`, so a fresh model's
  `state_dict()` reports 77 keys for 76 unique tensors: `model.embed_tokens.weight` and
  `model.vision_embed_tokens.wte.weight` are two keys sharing one storage (the commented-out
  weight-copy at lines 1352–1353 confirms the intent). The generator asserts the 77-key set
  equals the expected 76 plus the duplicate key, asserts `data_ptr()` equality, stores only 76
  tensors (the shared embedding once, under `model.embed_tokens.weight`, matching the
  `TinyCheckpoints` layout), and the reference's strict `load_state_dict` re-supplies the
  duplicate key from the same tensor. Every text row in every forward — including the image
  path's own `self.wte(input_ids)` — embeds from `model.embed_tokens`, so Java's
  single-lookup path is the reference's own path. Recorded in the golden provenance
  `shared_embedding_note`.
- **`pad_token_id` and Dropout.** The bundled `Phi3VModel.__init__` (line 1341) builds
  `nn.Embedding(vocab, hidden, padding_idx=config.pad_token_id)` with the config default
  32000, which crashes on vocab 21; the checkpoint `config.json` therefore pins
  `pad_token_id: 0`. Its `__init__` also builds `self.dropout = nn.Dropout(0.0)`, the
  identity in both eval and train modes, so no generator-side dropout handling is needed.
- **Encoding pins (no BOS).** The tiny tokenizer has no added tokens, so no `<s>`/BOS is ever
  prepended — the plan's "prepended `<s>`" wording is stale and the final prompt lengths are
  computed without it. Verified encodings: `"toast "` → `[20,18,13,19,20,12]`, `"toast"` →
  `[20,18,13,19,20]`, `"code"` → `[15,18,16,17]` (an earlier note said `[15,16,17,14]`; the
  generator pins were corrected against the venv before the first run), `"toast code"` →
  `[20,18,13,19,20,12,15,18,16,17]`, `"c d "` → `[15,12,16,12]`, `" e s"` → `[12,17,12,19]`.
- **BF16 ordering.** Every stored weight is rounded to BF16 before saving; the reference is a
  fresh model build with `_attn_implementation="eager"` (bundled
  `Phi3VPreTrainedModel._supports_sdpa = False`, so it is set explicitly and asserted on
  `model.config` and `img_processor.config` after loading) plus strict
  `load_state_dict({k: v.float()})`. The load spot-check probes the **row-1 slice** of
  `model.embed_tokens.weight` (row 0 is the zeroed pad row; a first-draft `flatten()[5:9]`
  probe hit row 0 and its non-zero guard fired) plus a non-zero guard.
- **Determinism.** `SEED = 20261025`, `torch.set_num_threads(1)`,
  `torch.use_deterministic_algorithms(True)`, `torch.manual_seed(SEED)` before the init-model
  build; a dedicated `torch.Generator(SEED)` perturbs, in `model.modules()` order, the 8
  `nn.LayerNorm`s (weight N(1,0.1), bias N(0,0.05)) and 5 bundled `Phi3RMSNorm`s (weight
  N(1,0.1)), with the counts asserted `{"layer_norm": 8, "phi3_rms_norm": 5}`, then draws the
  `glb_GN` and `sub_GN` tokens N(0,1) from the same generator. Two consecutive runs produced
  byte-identical golden and checkpoint (diffed full SHA lists).
- **`hd_feature_transform` applies `img_projection` internally** (bundled lines 295–297, one
  call over the `torch.cat` of the per-image `[sub, glb_GN, global]` blocks), returning
  `(R, text_hidden)`. The generator asserts `rows.shape == (sum(num_img_tokens), 128)` before
  flattening; Java's per-image `hdRows` concat plus one `projector().forward` call equals the
  reference's single call because the projection is a row-independent `Linear`.
- **Reference forwards are fresh, full, uncached**:
  `ref(input_ids=[expanded + history incl. negative sentinels], pixel_values (n,5,3,336,336)
  or None, image_sizes (n,2) long or None, attention_mask=ones(1,L,long), use_cache=False)`;
  logits = `out.logits[0, -1]`. The bundled boundary recompute (lines 1650–1655) is never
  exercised by the golden — it only justifies Java's re-prefill design. The all-ones mask is
  correct because every row of every forward is a real token (sentinels stay in the tensor as
  clamped IDs, never masked).
- **Scope-lifetime fix (found while writing the golden, before any run).** `Phi3VModel`'s
  `prefill` originally hoisted the projected rows into the prefill **activation** scope, which
  is closed at the end of prefill — so the single re-prefill crossing would read freed rows
  after `activation.close()`. Fixed by hoisting to `activation.parent()` (the request's
  generation scope) with a `requireNonNull` guard and updated javadoc; the regression
  `Phi3VModelTest.imageBoundaryGenerateRetainsTheProjectedRowsForTheRePrefill` drives a full
  `generate` across the boundary and passes.
- **Cases and measured evidence.** Six cases (P = expanded length, no BOS): text-only
  (P=120, maxNew 10, greedy all 0), wide-cross-boundary (P=465, maxNew 8, **recompute step
  6** = 471−465, greedy all 8), square (P=768, maxNew 6, greedy 13,20,13,20,13,20),
  wide-long-prompt (P=473, maxNew 6, greedy all 3 — images alone are 457 < 470, so prompt
  length pins the table selection), tall (P=481, maxNew 6, greedy all 8), two-image
  (P=1231, maxNew 4, greedy 13,20,13,20; wide then square). The observed greedy token set
  {0,3,8,13,20} varies across cases, so the golden's non-degenerate `greedy_id_note` branch
  was taken. Java side: `Phi3VGoldenTest` (strict `full-float32`, bound `1e-4`) measured
  maximum logit/projected-row error `1.5497208e-6`; `Phi3VDefaultPrecisionTest` (TF32 on,
  named bound `0.0012`) measured `9.266138e-4` — recorded in
  `req/phase6-golden-precision.md`. Both suites are registered in the macOS CI
  required-suite assertion and the result-XML upload.
