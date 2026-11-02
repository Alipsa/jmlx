# Phase 9.8 — Z-Image Base and FLUX.2 Klein Base 4B image generation decision

Status: decision record (reviewed via PR; the PR review is the decision gate)
Date: 2026-11-02
Base commit: `65e5306` (main after PR #74)

This record only decides; it does not start code. It settles the four questions that
`req/full-roadmap.md` § 9.8 gates the milestone on: the license, where the code lives, the
reference toolchain, and the quantization and LoRA scope. Requirements come from
`req/jmlx-z-image-flux2-klein-base-requirements.md` (DIFF-001 to DIFF-011, ZIMG-001 to ZIMG-003,
FKB-001 to FKB-003, ART-001 to ART-004). A milestone plan, `req/plans/phase9-8-plan.md`, follows
once this record is merged and 9.7 is delivered.

**Method.** Desk analysis only; no model has been run. The sources are:

- The Hugging Face API, configs, tokenizer files and safetensors headers of
  `Tongyi-MAI/Z-Image` at revision `04cc4abb7c5069926f75c9bfde9ef43d49423021` and
  `black-forest-labs/FLUX.2-klein-base-4B` at revision
  `a3b4f4849157f664bdbc776fd7453c2783562f4d`, read on 2026-11-02.
- A per-tensor byte comparison of Z-Image's `text_encoder/model-00003-of-00003.safetensors`
  against `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c` (§1).
- The diffusers source in the released wheel `diffusers-0.42.1-py3-none-any.whl` (SHA-256
  `9154faa704ce00e0b77f568e0d64786064368528aab5fc14eeea92f87d788014`, the PyPI digest), the
  newest release on 2026-11-02: `pipeline_z_image.py`, `pipeline_flux2_klein.py`,
  `transformer_z_image.py`, `transformer_flux2.py`, `autoencoder_kl_flux2.py` and
  `scheduling_flow_match_euler_discrete.py`. Diffusers main was not compared.
- The jmlx code at the base commit.

The probes that need the multi-gigabyte artifacts or a running reference environment are listed in
§9 as the milestone plan's first work packages. Facts marked *(desk estimate)* are calculated, not
measured; facts marked *unverified* are inferences a WP1 probe confirms.

## 1. Artifact facts (verified at the pinned revisions)

**Z-Image** (`model_index.json`: `ZImagePipeline`)

| Component | Class | Files | Bytes | Stored dtype |
| --- | --- | --- | --- | --- |
| transformer | `ZImageTransformer2DModel` | 2 shards | 12,309,873,872 | BF16 (521 tensors) |
| text_encoder | `Qwen3Model` (config `Qwen3ForCausalLM`) | 3 shards | 8,044,982,000 | BF16 (398 tensors) |
| vae | `AutoencoderKL` | 1 file | 167,666,902 | BF16 (244 tensors) |
| scheduler | `FlowMatchEulerDiscreteScheduler` | config only | — | — |
| tokenizer | `Qwen2Tokenizer` (incl. `tokenizer.json`) | 4 files | — | — |

All 21 files total 20,547,479,575 B; without the 8.98 MB `teaser.jpg`, the README and
`.gitattributes` (1,751 B) the download is 20,538,488,559 B.

- **Transformer config:** `dim` 3840, 30 layers plus 2 noise-refiner and 2 context-refiner
  layers, 30 heads × 128 (`n_kv_heads` 30), `qk_norm`, `in_channels` 16, `all_patch_size` [2],
  `cap_feat_dim` 2560, `axes_dims` [32, 48, 48], `axes_lens` [1536, 512, 512], `rope_theta` 256,
  `t_scale` 1000, `norm_eps` 1e-5. 6,154,908,736 parameters.
- **Text-encoder config:** `model_type: qwen3`, hidden 2560, 36 layers, 32 heads / 8 KV heads ×
  128, vocabulary 151,936, `tie_word_embeddings: true`. 4,022,468,096 parameters.
- **VAE config:** the FLUX.1 autoencoder (`_name_or_path: ../checkpoints/flux-dev`):
  `latent_channels` 16, `block_out_channels` [128, 256, 512, 512], `layers_per_block` 2,
  `scaling_factor` 0.3611, `shift_factor` 0.1159, no `quant_conv`/`post_quant_conv`.
- **Scheduler config:** `use_dynamic_shifting: false`, `shift` 6.0.

**FLUX.2 Klein Base 4B** (`model_index.json`: `Flux2KleinPipeline`)

| Component | Class | Files | Bytes | Stored dtype |
| --- | --- | --- | --- | --- |
| transformer | `Flux2Transformer2DModel` | 1 file | 7,751,109,744 | BF16 (169 tensors) |
| text_encoder | `Qwen3ForCausalLM` | 2 shards | 8,044,981,992 | BF16 (398 tensors) |
| vae | `AutoencoderKLFlux2` | 1 file | 168,120,878 | BF16 (250 tensors) plus one I64 |
| scheduler | `FlowMatchEulerDiscreteScheduler` | config only | — | — |
| tokenizer | `Qwen2TokenizerFast` (incl. `tokenizer.json`, `chat_template.jinja`) | 7 files | — | — |

All 25 files total 23,740,007,506 B. That includes a second copy of the transformer in Black
Forest Labs' own layout (`flux-2-klein-base-4b.safetensors`, 7,751,105,712 B, 149 tensors, the
same 3,875,544,576 parameters) and three sample images. jmlx loads only the diffusers layout, so
the needed download is 15,980,131,711 B.

- **Transformer config:** 5 double-stream and 20 single-stream layers, 24 heads × 128 (inner
  dimension 3072), `mlp_ratio` 3, `in_channels` 128, `patch_size` 1, `joint_attention_dim` 7680,
  `axes_dims_rope` [32, 32, 32, 32], `rope_theta` 2000, `guidance_embeds: false`, `eps` 1e-6.
  Every linear layer is bias-free. 3,875,544,576 parameters.
- **Text-encoder config:** the same Qwen3 shape and parameter count as Z-Image's.
- **VAE config:** `latent_channels` 32, `patch_size` [2, 2], `block_out_channels`
  [128, 256, 512, 512], `layers_per_block` 2, `use_quant_conv` and `use_post_quant_conv`,
  `batch_norm_eps` 1e-4. The extra I64 tensor is presumably BatchNorm's `num_batches_tracked`.
- **Scheduler config:** `use_dynamic_shifting: true`, `time_shift_type: exponential`,
  `base_shift` 0.5, `max_shift` 1.15, `base_image_seq_len` 256, `max_image_seq_len` 4096.

**Shared files.** Both repositories' `tokenizer.json` (11,422,654 B), `vocab.json` and
`merges.txt` are the same blobs as `Qwen/Qwen3-4B`'s. Z-Image's `tokenizer_config.json` is the
same blob as Qwen3-4B's, and its embedded chat template is byte-identical to FLUX.2's
`chat_template.jinja` (4,168 B). Both use pad token `<|endoftext|>`, no BOS.

**The text encoders are Qwen3-4B, with one exception.**

- Z-Image's shards 1 and 2 are the same blobs as Qwen3-4B's. Shard 3 has the same size and
  header, but two tensors differ: `model.layers.35.mlp.down_proj.weight` and
  `model.layers.35.mlp.up_proj.weight` (the last layer). They are not zeroed (84% and 41% of their
  elements are non-zero).
- The pipeline conditions on `hidden_states[-2]` (§2). By the Transformers convention that entry
  is layer 34's output, so layer 35 never contributes. *Unverified* until the WP1 probe, whose
  success criterion is a regression test, not just an observation (run in the reference, one
  process, CPU float32):
  1. load the official Z-Image encoder and run the reference prompt conditioning;
  2. replace `model.layers.35`'s two tensors with Qwen3-4B's (revision `1cfa9a7`, downloaded in
     WP0, at least shard 3);
  3. assert the selected conditioning output (`hidden_states[-2]`, real-token rows) is
     **bit-identical** — same process and dtype leave no tolerance to absorb a real
     contribution from layer 35.
- FLUX.2's encoder is sharded differently, so blobs cannot be compared. Whether its weights equal
  Qwen3-4B's (or Z-Image's) is *unverified*; WP0 compares per-tensor digests after the download.
  If they match on layers 0–26, one resident encoder could serve both models. Whether the
  truncated encoder loads reproduce the reference's selected entries is asserted in WP2, where
  those loads are built: the 27-layer FLUX.2 load must reproduce a full 36-layer forward's
  `hidden_states[9/18/27]` (all 512 rows) and the 35-layer Z-Image load must reproduce
  `hidden_states[-2]` (real-token rows), each within the recorded conditioning ε, so neither
  pipeline's conditioning ever depends on a layer it does not load.

## 2. Reference-pipeline facts (diffusers 0.42.1)

These facts change what jmlx must build, and several differ between the two models in ways a
shared abstraction must not hide:

| | Z-Image | FLUX.2 Klein Base 4B |
| --- | --- | --- |
| Chat template | `apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` | the same template with `enable_thinking=False` |
| Tokenization | padded to `max_sequence_length` 512 | the same |
| Conditioning | `hidden_states[-2]`, then **only the real-token rows** (the attention-mask selection drops padding) | `hidden_states[k]` for k = 9, 18, 27, stacked to 3 × 2560 = 7680 per token; **all 512 rows, padding included**, with no text mask in the transformer |
| Layers needed | 0–34 (35 of 36) | 0–26 (27 of 36) |
| Defaults | 50 steps, `guidance_scale` 5.0, `cfg_normalization` off, `cfg_truncation` 1.0, negative prompt `""` | 50 steps, `guidance_scale` 4.0, negative prompt `""` |
| Guidance on | when `guidance_scale > 0` | when `guidance_scale > 1` (and the pipeline is not distilled) |
| Guidance formula | `pos + g · (pos − neg)` | `neg + g · (pos − neg)` |
| Guidance passes | cond and uncond batched into one transformer call | two transformer calls |
| Sigmas | `linspace(1, 1/n, n)`, then the static shift 6.0 (`mu` is computed but unused) | `linspace(1, 1/n, n)`, then the exponential dynamic shift with `mu = compute_empirical_mu(seq_len, n)` |
| Transformer timestep input | `(1000 − t) / 1000`, and the output is negated | `t / 1000` |
| Latent | `(B, 16, H/8, W/8)`, float32 | `(B, 128, H/16, W/16)` in the conditioning dtype, packed to `(B, H·W/256, 128)`. The starting noise is drawn by `randn_tensor` **directly in this already-patchified shape** (`pipeline_flux2_klein.py:499-503`); no patchify step is applied to it, and teacher-forced runs inject the reference's starting latent in exactly this shape |
| Before VAE decode | `latents / 0.3611 + 0.1159` | unpack; `latents · sqrt(running_var + 1e-4) + running_mean` (the VAE's BatchNorm statistics); 2×2 unpatchify to 32 channels |
| Size constraint | height and width divisible by 16; the pipeline **raises `ValueError`** otherwise (`pipeline_z_image.py:402-411`) | divisible by `vae_scale_factor · 2` = 16, but the pipeline only **warns and rounds down** (`pipeline_flux2_klein.py:558-566`, `:493-494`). jmlx deliberately does not copy that: DIFF-005 requires validating rather than silently changing dimensions, so jmlx **rejects** sizes that are not multiples of 16 for both models |
| Output | RGB | RGB |

The model cards add recommendations the pipelines do not encode. Z-Image's card recommends 28–50
steps, guidance 3.0–5.0 and 512² to 2048² total pixel area, and its example uses guidance 4. The
FLUX.2 card's example uses 50 steps and guidance 4.0.

Further facts:

- **The same guidance number means different strengths.** Z-Image's `g` corresponds to `g + 1`
  in FLUX.2's formula. A guidance value is therefore not portable between the two models, and
  the capability descriptor (DIFF-006) must carry each model's own semantics.
- **Z-Image's two extra guidance options are exposed, as floats.** The capability descriptor
  (DIFF-006) carries `cfg_truncation` and `cfg_normalization` as optional float parameters
  with the pipeline defaults (1.0 and 0.0 — off when ≤ 0), and DIFF-010's persisted metadata
  records their effective values. They are floats, not booleans: despite its boolean
  declaration, the pipeline uses `cfg_normalization` as a float factor
  (`max_new_norm = ori_pos_norm * float(self._cfg_normalization)`,
  `pipeline_z_image.py:554-557`), and jmlx models both options as floats so the reference's
  full range is reproducible. FLUX.2 has neither option; its descriptor omits them.
- **FLUX.2's padding rows are conditioning.** Their hidden states come from a right-padded
  forward under Transformers' padding mask, so each pad query should attend to every real key
  before it (*unverified*; WP1 probes it). jmlx's batched decoder path pads on the left
  (`DecoderModel.stepLogits`), so reproducing these rows is new work.
- **Initial noise** comes from `randn_tensor` on a `torch.Generator` in both pipelines. MLX random
  cannot reproduce it.
- **Both VAEs use diffusers' standard `Encoder`/`Decoder`** (`models/autoencoders/vae.py`), the
  same classes as SD 1.5's `AutoencoderKL`, with the same `block_out_channels` and
  `layers_per_block`. Only the latent width, the convolutions around it and the latent
  normalization differ.
- **Precision-sensitive modules** named by the source: Z-Image skips `t_embedder` and
  `cap_embedder` when casting layers; FLUX.2 skips `pos_embed` and every `norm`.
- **Architecture, Z-Image:** a single-stream DiT. Text and image tokens are each padded to a
  multiple of 32 with learned pad tokens and refined by their own two-layer stacks, then joined.
  Blocks use RMSNorm, QK RMSNorm, a SwiGLU feed-forward and adaLN modulation with tanh gates;
  rotary embedding is applied as a complex product over three axes. The final layer is a
  non-affine LayerNorm with adaLN.
- **Architecture, FLUX.2:** an MMDiT. Double-stream blocks keep separate text and image weights
  under joint attention; single-stream blocks use one fused QKV-plus-MLP projection. Both use
  SwiGLU, QK RMSNorm and bias-free shared modulation, with a four-axis rotary embedding in
  interleaved-pair form and a bias-free `AdaLayerNormContinuous` output.
- **LoRA loaders exist for both** (`ZImageLoraLoaderMixin`, `Flux2LoraLoaderMixin`), so the
  reference can apply an adapter the way users of the pinned diffusers do.

## 3. What exists versus the gaps

| Component | jmlx today | Gap |
| --- | --- | --- |
| Text encoder | `qwen3` dispatch (7.0; Tier-B on Qwen3-0.6B with tied embeddings); a package-private `DecoderModel.PostBlockHook` that fires after each block, before the final norm | an encoder-only load that stops at the deepest needed layer and has no output projection; capture of selected layers' outputs; right padding with the Transformers padding mask (FLUX.2); a public seam for all of this if the code moves to another module (§6) |
| Tokenizer and template | `HfTokenizer`; Qwen3 chat templates through `jmlx-jinja` (7.0); `ChatTemplateOptions.extraContext`; `Padding` with `RIGHT`/`LEFT` | none expected: `enable_thinking` is passed as template context and padding to 512 is right-sided. WP1 confirms both against the reference. |
| Z-Image transformer | `RMSNorm`, `LayerNorm`, `SwiGLU`, attention with explicit masks, `MLXFast` RoPE (traditional and split forms), `MultiAxisRope` (Qwen3-VL sections) | the S3-DiT blocks, refiners, learned pad tokens, adaLN with tanh gates, three-axis complex RoPE ([32, 48, 48], theta 256) with per-axis position ids, patchify and unpatchify |
| FLUX.2 transformer | as above | double- and single-stream blocks, fused projection, shared modulation, four-axis interleaved RoPE (theta 2000), packing and position ids |
| Scheduler and sampler | 9.6 contracts; 9.7's flow-match Euler with the exponential dynamic shift | the static shift; FLUX.2's `compute_empirical_mu`; two guidance formulas and thresholds; Z-Image's timestep inversion and output negation |
| VAE decoder | 9.6's SD 1.5 `AutoencoderKL` decoder (same `Decoder` class and topology); `BatchNorm`, `Conv2d`, `GroupNorm`, `Upsample` (7.1) | latent widths 16 and 32; FLUX.2's `post_quant_conv`, BatchNorm-statistics de-normalization and 2×2 unpatchify; Z-Image's scale and shift |
| Image output | 9.7's PNG writer (RGBA) | an RGB mode |
| Quantization | affine `QuantizedLinear`, `QuantizedEmbedding`, `MLXQuant` (group 64); 9.7's Q4-resident path | Q4 conversion for both new transformers and the Qwen3 encoder (§8) |
| LoRA | none | adapter loading, key mapping, rank/alpha scaling, application next to dense and Q4 weights (§8) |
| Jobs, metadata, variations | 9.7's asynchronous jobs and cancellation (IMG-010) | the DIFF-010 metadata schema; seed lists (ART-002) |

No new native binding is expected. The milestone plan re-checks this against
`req/mlx-api-inventory.md` at whatever pin is current.

## 4. Memory feasibility and precision paths *(desk estimate)*

Weights only; activations are measured during the milestone. Q4 is affine group 64, about
4.5 bits per parameter. The encoder rows count only the layers each pipeline uses (§2) plus the
389M-parameter embedding; the VAE stays as stored (BF16) except on the FLOAT32 path.

**VAE precision policy, aligned with 9.7.** 9.7 keeps its Qwen-Image VAE in F32 on every path
because that VAE *is stored* F32 (1.35 GB); this milestone's two VAEs are stored BF16, and the
policy is the same rule — each VAE runs at its stored precision unless a path's bound says
otherwise. Neither pipeline reads `force_upcast: true` from its VAE config, so the reference
does not decide this. The one open point is recorded, not glossed over: whether a BF16 VAE
meets the Q4 path's bound is *unverified*; WP7 compares the BF16 VAE decode against the
float32 reference under a named bound and, if it fails, up-casts the VAE to F32 on the BF16
and Q4 paths (about +0.17 GB, still inside the §4 Q4 resident figures' headroom).

| Component | BF16 resident | FLOAT32 resident | Q4 resident |
| --- | --- | --- | --- |
| Z-Image transformer (6.15B) | 12.31 GB | 24.62 GB | ~3.46 GB |
| Z-Image encoder, layers 0–34 (3.92B) | 7.84 GB | 15.69 GB | ~2.21 GB |
| FLUX.2 transformer (3.88B) | 7.75 GB | 15.50 GB | ~2.18 GB |
| FLUX.2 encoder, layers 0–26 (3.11B) | 6.23 GB | 12.46 GB | ~1.75 GB |
| either VAE (0.084B) | 0.17 GB | 0.34 GB | not quantized (0.17 GB) |

| Path | Z-Image `RESIDENT` | Z-Image `STAGED` | FLUX.2 `RESIDENT` | FLUX.2 `STAGED` |
| --- | --- | --- | --- | --- |
| FLOAT32 | ~40.6 GB | ~25.0 GB | ~28.3 GB | ~15.8 GB |
| BF16 | ~20.3 GB | ~12.5 GB | ~14.2 GB | ~7.9 GB |
| Q4 | ~5.8 GB | ~3.6 GB | ~4.1 GB | ~2.4 GB |

**On the 24 GB target:**

- **Q4 is the selected low-memory inference path and is expected to fit within 24 GB at
  512 × 512 for both models, resident.** That expectation is a desk estimate of resident
  weights only; it is confirmed by WP7's measured peak memory (allocator and process
  footprint, guidance passes included) and successful repeated inference on the M4 Air, which
  is exactly DIFF-007's acceptance run. Until those measurements exist, the fit is an
  expectation, not a fact.
- **FLUX.2 BF16 resident (~14.2 GB) and Z-Image BF16 staged (~12.5 GB) may also fit**, but they
  are not assumed to. They are measured on the M4 Air only if the Q4 runs leave clear headroom.
- **Activations grow with resolution and with guidance.** At 1024² both models attend over 4,096
  image tokens plus text (Z-Image: the real tokens padded to a multiple of 32; FLUX.2: 512). Z-Image
  batches the two guidance passes into one call, which doubles its activations; FLUX.2 runs them
  one after the other.

**Three weight-residency paths, as in 9.7,** all measured against one CPU float32 reference:

- **FLOAT32**, the up-cast pattern, for the strict gates. Each Tier-A synthetic checkpoint stores
  BF16 weights, so the same checkpoint also exercises BF16-resident loading. The Tier-B tight-ε
  runs use this path on the M5 Max; even Z-Image `RESIDENT` (~40.6 GB) fits there.
- **BF16 resident** and **Q4 resident**, each with a separately named bound.

**Reference memory.** The CPU float32 reference runs component by component; its largest stage
holds the Z-Image transformer (~24.6 GB). The M5 Max host has 128 GiB (9.7 record §4); WP0
re-records it.

## 5. Decision 1 — license

**Facts (verified 2026-11-02).**

- Both repositories declare `license: apache-2.0` in their model cards, and the API reports
  `license:apache-2.0` for both. Neither repository is gated.
- FLUX.2 ships `LICENSE.md`, the Apache License 2.0 text. Its card states that "Outputs can be
  used for commercial purposes", that BFL released the Klein 4B models under Apache 2.0 and the
  Klein 9B models under a non-commercial license, and that BFL "encourage[s]" deployers of the
  4B models to filter NSFW and protected content.
- Z-Image ships no license file; its card declares Apache 2.0.
- Neither repository has a `NOTICE` file, so Apache 2.0 §4(d) adds no notice to carry.
- Z-Image's text encoder is Qwen3-4B (Apache 2.0) with a modified last layer; FLUX.2's has the
  same shape, and its identity is checked in WP0 (§1).

**Decision: accept both, for pinned, read-only Tier-B test artifacts.** Unlike 9.7, the verified
use case carries no non-commercial restriction: a consumer may use either checkpoint
commercially under Apache 2.0. The documentation still states:

- jmlx never bundles the weights; users download them from the model owners.
- jmlx ships no safety filter for these models, as 9.6 ships none for SD 1.5. BFL's
  encouragement to filter applies to deployers.
- FLUX.2 Klein **9B** is under a different, non-commercial license and is not covered.
- Each LoRA adapter has its own license, reviewed separately (§8).

**What may and may not enter the public repository.**

- **No file from either artifact is committed:** no configs, tokenizer files, weights or
  converted or quantized derivatives. Apache 2.0 would allow redistribution, but committing
  multi-gigabyte artifacts is not wanted, and keeping one rule for every artifact keeps the
  repository simple. Tier-B reads the runner-local download at run time.
- **Manifests record facts about the artifacts** (revision, file names, sizes, SHA-256s, tensor
  shapes and dtypes): the existing Tier-B pattern.
- **Committed Tier-B goldens are model outputs.** Size budget: **at most 12 MiB per model**, as
  little-endian float32. At 512², one Z-Image latent is 256 KiB and one FLUX.2 packed latent is
  512 KiB; each teacher-forced step stores its input plus the conditional and unconditional
  outputs, so guidance arithmetic is checked separately from the transformer.
  - Z-Image: the real-token conditioning rows (10 KiB per token), 8 steps (6 MiB), the final
    latent and the 512² RGB decode (3 MiB): about 9.5 MiB plus the conditioning.
  - FLUX.2: one prompt's full conditioning is 512 × 7680 floats = 15 MiB, over budget on its own.
    The golden stores the real-token rows plus a fixed subset of padding rows for both prompts,
    4 steps (6 MiB), the final latent and the decode: about 9.5 MiB plus the conditioning. As in
    9.7, the Java transformer consumes its own conditioning; only the latent is injected.

  WP0 fixes the step sets (always including the first and last steps) and the padding-row
  subset within the budget. ZIMG-001's internal-layer parity and FKB-001's per-block parity
  never ride these goldens: one float32 internal activation at 1024² is 60 MiB (1,024 × 3,840
  for Z-Image at 512² alone is 15 MiB; FLUX.2's joint stream at 512² is 18 MiB), over the
  per-model cap on its own. They are Tier-A synthetic checks instead (§ 9.8 exit gate), where
  small dimensions make them cheap and "at full precision" still holds.
- **Tier-A fixtures use no artifact file.** Each synthetic checkpoint is generated, with a
  committed hand-written fixture tokenizer and a minimal chat template that renders the
  `enable_thinking` branches the way the real template does.
- **The strict Tier-A synthetic goldens stay the binding gates.**

**If the review declines either checkpoint,** 9.8 continues with the other alone, by a dated
amendment.

## 6. Decision 2 — module home

**Decision: a new published module, `jmlx-diffusion`, created at the start of 9.8.** It takes
the diffusion layer: the scheduler, sampler and VAE contracts and 9.6's and 9.7's pipelines,
moved at 9.8's start, plus the two new pipelines, the LoRA loader, the job API and the image
output. The conditioning seams stay in `jmlx-models` and become public API there.

**Explicit ownership.**

| Component | Module |
| --- | --- |
| MLX arrays and native operations | `jmlx-core` / `jmlx-ffi` |
| Conditioning encoders — CLIP (9.6), Qwen3-VL (`Qwen3VlModel`, 7.3b; text-only load per 9.7), Qwen3 — and their now-public seams | `jmlx-models` |
| Generic diffusion schedulers and samplers | `jmlx-diffusion` |
| Denoiser models: SD 1.5's UNet, Qwen-Image's, and the Z-Image and FLUX.2 DiTs | `jmlx-diffusion` |
| VAE implementations: SD 1.5's, Qwen-Image's, Z-Image's and FLUX.2's | `jmlx-diffusion` |
| Diffusion pipelines: 9.6's, 9.7's, `ZImagePipeline` and `Flux2KleinPipeline` | `jmlx-diffusion` |
| Public image-generation API | `jmlx-diffusion` |
| LoRA application for diffusion models | `jmlx-diffusion` |

**Dependency direction.** `jmlx-diffusion` declares `api project(':jmlx-core')` and
`implementation project(':jmlx-models')`.

- **`jmlx-core` is `api`**: the public scheduler, sampler and VAE contracts that
  `jmlx-diffusion` owns take and return tensors (`MLXArray`), and IMG-007's ownership
  requirement (`AutoCloseable` and `MLXScope` ownership) puts `jmlx-core` types on the public
  API, so they must be visible to consumers.
- **`jmlx-models` is `implementation`**: the pipelines consume the conditioning encoders
  internally, and no `jmlx-models` type is part of `jmlx-diffusion`'s public API. The one
  candidate, cancellation, is settled below.
- **Cancellation is its own type.** DIFF-008/IMG-010 need a cancellation token and the
  existing one is `se.alipsa.jmlx.models.CancellationToken`; `jmlx-diffusion` defines its own
  rather than promote `jmlx-models` to `api` for one interface, and a job bridges the two
  internally.
- **`jmlx-vision` follows the image result.** 9.7's public API with PNG output is delivered by
  9.7 WP7 and gated by 9.7's exit gate, so the image-result code already exists before 9.8:
  if its result type reuses `jmlx-vision`'s `RgbImage`, `jmlx-vision` is `api` (`implementation`
  on `jmlx-models` would not put it on consumers' compile classpath); if the result carries
  its own pixel type, no `jmlx-vision` declaration is needed at all — it is still reached
  transitively through `jmlx-models`. WP0 records which of the two applies when moving 9.7's
  image-result code, and the WP0 row in §9 lists it.
- (The 9.6/9.7 contracts these bullets infer from are not written yet; if they land with a
  genuine `jmlx-models`, `jmlx-vision` or `jmlx-tokenizer` type on the surface, the
  milestone plan amends this paragraph.)

**`jmlx-models` must never depend on `jmlx-diffusion` — not at compile time, not at runtime.**
Gradle already fails a compile-time project cycle; WP0 adds the missing runtime guard, a check
task asserting that `jmlx-models`' runtime classpath contains no `:jmlx-diffusion`, so the
one-way direction cannot silently invert later.

**Why reopen now.** 9.7's record §6 made the split conditional: "a third image family would
make jmlx-models carry diffusion code that most of its users don't need." 9.8 brings the
third and fourth families, and the condition now holds:

- The interfaces have two in-module users, which is the validation 9.7 wanted.
- Most diffusion code is still to come: two transformers, LoRA, jobs and metadata. Without a
  split, every text-generation consumer of `jmlx-models` would carry it.
- **The diffusion surface is not yet published — and must stay so until the move.**
  `jmlx-models` itself publishes in Phase 8 (8.4's native chain), but 9.6's and 9.7's
  diffusion API is new surface, and Phase 9 releases at the phase's end, so moving it at 9.8's
  start breaks no published coordinate. Until the move, **no `jmlx-models` release — mid-phase,
  per-milestone or bug-fix patch — is cut from a commit carrying 9.6's or 9.7's diffusion
  code**; if a release is wanted earlier, it waits for 9.8's move. A delegating-stub migration
  is not an alternative: stubs in `jmlx-models` that delegate to the moved code would make
  `jmlx-models` depend on `jmlx-diffusion`, which this §6 forbids, Gradle's cycle check
  rejects, and WP0's runtime-classpath check would fail.
- The requirements suggest this split (their §2 is explicitly "subject to JMLX's existing
  module conventions" and would also allow the pipelines in `jmlx-models` or an adapter
  module); this record adopts the `jmlx-diffusion` home rather than that alternative, for the
  costs and benefits below.

**What it costs.**

- Another published module: its own `0.1.0-SNAPSHOT` line, a byte-identical `release.sh`
  in `verifyReleaseScriptsMatch`, a place in the release order after `jmlx-models`, and
  release-smoke invariants (the synthetic diffusion cases run through it).
- The conditioning seams must be public in `jmlx-models` rather than package-private. They
  become part of the published `jmlx-models` API.
- A mechanical move of 9.6's and 9.7's diffusion code, done before any new 9.8 code lands.

**The alternative considered** is to keep everything in `jmlx-models` (9.7's outcome). It
avoids the module overhead and keeps the seams package-private, but it ships every diffusion
family to every text-generation consumer for good, and a later split would move published API.

**If the review prefers the alternative,** 9.8 stays in `jmlx-models` and this section is
amended; nothing else in this record depends on the module boundary.

## 7. Decision 3 — reference toolchain

**Decision:**

- **Extend 9.7's `tools/image-gen-reference/`.** 9.7 creates it, pinning `diffusers==0.42.1`
  (which contains both pipelines, §2) under one Transformers 5.x and one torch release. 9.8 adds
  one manual generator script per family, digest-pinned in the same `provenance.json` and verified
  by the same `verifyImageGenReferenceGoldens` task. If 9.7's chosen Transformers or torch
  release cannot run both pipelines, WP1 adds a second lock and records why.
- **MFLUX is a cross-check, not the reference.** It runs on MLX, so agreement with it is not
  independent evidence, which every jmlx reference must be. It may be used to debug, never to
  set a bound.
- **Hidden-state indexing is probed, not assumed.** WP1 asserts the Transformers entries the
  pipelines use: `hidden_states[-2]` excluding layer 35 (Z-Image), entries 9, 18 and 27
  (FLUX.2), and FLUX.2's padding rows under the padding mask.
- **Reference precision and comparison mode** follow 9.7: CPU float32, per component.
  1. encoder → conditioning, including the negative prompt `""`;
  2. transformer, teacher-forced from the reference's own latent at each compared step, with
     the conditional and unconditional outputs stored separately; the initial latent is
     injected;
  3. VAE decode of the reference's final latent.

  The binding ε applies to the FLOAT32 path. The BF16 and Q4 paths have separately named bounds.
  A free-running 50-step trajectory is reported, not gated. Seeded reproducibility of jmlx's own
  MLX-random noise is a separate, Java-only check.
- **Defaults come from the pinned pipeline code**, not the model cards: Z-Image 50 steps and
  guidance 5.0, FLUX.2 50 steps and guidance 4.0 (§2). The capability descriptor reports them
  with each model's guidance formula; the cards' ranges are documented, not enforced.

## 8. Decision 4 — quantization and LoRA scope

**Quantization: one canonical representation, converted locally.**

- **MLX affine Q4, group 64** (`MLXQuant`), produced by jmlx's conversion tooling from the
  pinned official BF16 checkpoints, written only to the user's local directory. This is 9.7's
  Q4-resident path.
- **Quantized:** the linear layers of both transformers' blocks and of the Qwen3 encoder's used
  layers, and the encoder embedding through `QuantizedEmbedding`.
- **Kept at their stored precision:** the modules the source marks precision-sensitive (§2), the
  remaining norms, the timestep and caption embedders, the output layers and the VAE. WP7
  records the exact list with each path's bound.
- **Prequantized community checkpoints** (MFLUX, mlx-community, GGUF K-quants) **are not loaded
  in 9.8.** Their layouts differ per tool and are not interchangeable with MLX affine Q4. A
  later milestone may add loaders for specific ones.

**LoRA: inference only, one adapter at a time, never merged.**

- Applied as `y = W·x + s·B(A·x)` next to the base weight, on every path. A Q4 base weight cannot
  absorb a merged adapter without requantizing, and one code path keeps the three residency paths
  comparable. When `s == 0` the adapter branch is skipped entirely — never multiplied by zero —
  so ART-001's scale-0 check is exact in every case (multiplication would not be: `0 · Inf` is
  `NaN`, so a literal `s·B(A·x)` is only exactly zero when the adapter output is finite).
- **Formats:** the key layouts of the two adapters WP0 selects (one per model), mapped
  explicitly. Any other layout is rejected with a diagnostic naming the unmapped keys.
- **Licenses:** WP0 selects adapters under licenses that permit use in a public test (Apache
  2.0, MIT or CC-BY), records each in the Tier-B manifest, and commits no adapter file.
- **Reference:** the pinned diffusers loaders (§2), at the same scale.
- Multi-adapter composition and hot-swapping wait until single-adapter correctness is proven
  (ART-001).

## 9. Analysis, estimate and recommendation

**Go, as Phase 9 milestone 9.8, after 9.7.** The reasons:

- Every native primitive it needs exists today.
- 9.6 and 9.7 lay the scheduler, sampler, VAE, staging and Q4 foundations; the VAE decoder
  topology is 9.6's.
- Both text encoders are Qwen3-4B-shaped `qwen3` models, a family jmlx already loads.
- Both checkpoints are Apache 2.0.

**Estimate** *(desk, in the 7.3x milestone shape)*: **10 work packages, about 10–14 weeks.**

| WP | Content |
| --- | --- |
| WP0 | Pin both artifacts; re-record the license acceptance; create `jmlx-diffusion` (release script, release order, smoke stub, the runtime-classpath direction check of §6) and move 9.6's and 9.7's diffusion code, recording the `jmlx-vision` scope (§6: `api` if 9.7's image result reuses `RgbImage`, otherwise no declaration); download Qwen3-4B (revision `1cfa9a7`, at least shard 3) and compare the two encoders' per-tensor digests; select the two LoRA adapters; fix the golden step and row sets; re-record the host's memory |
| WP1 | Reference scripts and probes: hidden-state indexing, FLUX.2 padding rows, both `enable_thinking` renderings against `HfTokenizer`, both sigma schedules, both guidance formulas, both RoPE layouts. The FLUX.2 padding-row probe is an explicit, independently reviewed checkpoint that must pass before WP4 starts: because FLUX.2 conditions on the padded positions' hidden-state values themselves (not just a mask), a wrong answer here would contaminate every later FLUX.2 numerical comparison |
| WP2 | The public conditioning seam on the Qwen3 decoder: truncated encoder-only load, layer capture, right padding with the padding mask. The truncated loads are validated per §1: the 27-layer FLUX.2 load reproduces a full forward's `hidden_states[9/18/27]` (all 512 rows), the 35-layer Z-Image load reproduces `hidden_states[-2]` (real-token rows) |
| WP3 | Scheduler extensions: static shift, empirical `mu`, per-model guidance |
| WP4 | FLUX.2 transformer, VAE path and Tier-A golden (built first, per the requirements' order) |
| WP5 | Z-Image transformer, VAE path and Tier-A golden |
| WP6 | Public pipelines and capability descriptors; Tier-B FLOAT32 parity for both |
| WP7 | Q4 conversion and BF16/Q4 paths with their bounds; the M4 Air 512² runs |
| WP8 | LoRA inference for both models |
| WP9 | Jobs, metadata, seed variations, release-smoke cases, benchmarks, the ZIMG-003 gallery and review protocol |

The largest risks are FLUX.2's padding-row conditioning, which jmlx has never reproduced, and the
Q4 quality bounds of two new transformers.

**Scope:** P0 and P1, text-to-image. P2 (FKB-003, ART-003, ART-004) is deferred to later
milestones.

**Exit gate:** as stated in `req/full-roadmap.md` § 9.8.

## 10. Gate

The decision is reviewed in this record's PR, and merging the PR accepts §5–§8. The roadmap
entry (`req/full-roadmap.md` § 9.8, the overview row and the revision history) and the
requirements document (`req/jmlx-z-image-flux2-klein-base-requirements.md`) land in the same PR
and state the decided outcome. If the review changes a decision, the PR updates all three
documents together.
