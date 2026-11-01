# Phase 9.7 — Qwen-Image-2.1 image generation decision

Status: draft decision record (reviewed via PR; the PR review is the decision gate)
Date: 2026-11-01
Base commit: `0ed5dd4` (main after PR #69)

This record only decides; it does not start code. It settles the three questions that
`req/full-roadmap.md` § 9.7 gates the milestone on: the license, where the code lives, and the
reference toolchain. Requirements come from `req/jmlx-image-generation-roadmap.md`
(IMG-001 to IMG-016). A milestone plan, `req/plans/phase9-7-plan.md`, follows once this record
is merged and 9.6 is delivered.

**Method.** Desk analysis only; no executable probe has been run yet. The sources are:

- The Hugging Face API, configs and safetensors headers of `Qwen/Qwen-Image-2.1` at revision
  `d26bb61231c349cf6b7896fa83353113880e1ba3`.
- The diffusers source at commit `86ef420b17363c28e36ebad6b58555e9aa8feda2` (main,
  2026-10-31). `pipeline_qwenimage21.py` and `transformer_qwenimage21.py` are byte-identical in
  the released tags v0.41.0 and v0.42.1.
- The jmlx code at the base commit.

The probes that need the multi-gigabyte artifact or a Transformers 5 environment are listed in
§8 as the milestone plan's first work packages. Facts marked *(desk estimate)* are calculated,
not measured.

## 1. Artifact facts (verified at the pinned revision)

| Component | Class | Files | Bytes | Stored dtype |
| --- | --- | --- | --- | --- |
| transformer | `QwenImage21Transformer2DModel` | 2 shards | 14,230,284,408 | BF16 |
| text_encoder | `Qwen3VLForConditionalGeneration` | 4 shards | 17,534,339,488 | BF16 |
| vae | `AutoencoderKLQwenImage21` | 1 file | 1,350,989,512 | F32 (238 tensors) |
| scheduler | `FlowMatchEulerDiscreteScheduler` | config only | — | — |
| processor | `Qwen3VLProcessor` (incl. `tokenizer.json`) | 9 files | — | — |

All files at the revision total 33,134,949,561 B. The API's `usedStorage` (47.4 GB) includes
git history. The API reports 7,115,124,736 BF16 parameters for the transformer.

- **Transformer config:**
  - 32 layers, 32 heads × 128 (inner dimension 4096), `mlp_ratio` 3
  - `in_channels`/`out_channels` 64, `patch_size` 1, `context_in_dim` 4096
  - `axes_dims_rope` [16, 56, 56], `eps` 1e-6, `causal_condition: true`
- **Text-encoder config:**
  - `model_type: qwen3_vl`, the same dispatch jmlx's `Qwen3VlModel` loads
  - text: hidden 4096, 36 layers, 8 KV heads, vocabulary 151,936
  - `mrope_interleaved: true`, `mrope_section` [24, 20, 20], untied embeddings
  - vision: depth 27, hidden 1152, DeepStack indexes [8, 16, 24]
- **VAE config:**
  - `z_dim` 64, `in_channels`/`out_channels` 4
  - `scale_factor_spatial` 16
  - per-channel `latents_mean`/`latents_std` (64 entries each)
- **Scheduler config:**
  - `use_dynamic_shifting: true`, `time_shift_type: exponential`
  - `base_shift` 0.5, `max_shift` 0.9, `base_image_seq_len` 256, `max_image_seq_len` 8192
  - `shift_terminal` 0.02
  
  `model_index.json` sets no `sample_sigmas`, so the pipeline default of 40 steps applies.

## 2. Reference-pipeline facts (diffusers `86ef420`, identical in v0.42.1)

These facts change what jmlx must build, and some differ from what the model card implies:

- **Conditioning is the pre-norm last hidden state.**
  - The pipeline feeds the transformer `hidden_states[-1]` of the text encoder, captured
    *before* the final RMSNorm.
  - From Transformers 5.0, that entry is tied to the normalized `last_hidden_state`. The
    pipeline therefore installs a forward hook on `norm` that returns the norm's input.
  - A reference run without that hook produces wrong conditioning, and so does a jmlx seam
    that returns the post-norm state.
  - Transformers PR #48087, merged 2026-09-17, adds a config switch to untie the two.
    *Unverified:* which release first ships it; the pipeline comment says 5.18.
- **Prompt construction.** The text-to-image prompt is a raw template string with a fixed
  system message ("Comprehend and analyze the provided prompt."). It is passed straight to the
  processor, not through `apply_chat_template`; the pipeline notes that the two tokenize
  differently. The leading system-message tokens are dropped from the hidden states. Their
  count (`_drop_idx`) comes from tokenizing the system message through the chat template.
- **Attention is block-causal.** The mask is `(q_idx >= kv_idx) or same_image_block`. The joint
  text/image sequence is causal, text tokens are strictly causal, and each image block (every
  condition image and the target) is bidirectional inside itself. With `causal_condition`, the
  timestep modulation carries an extra row for condition tokens, modulated from t = 0.
- **VAE.** Its "causal 3-D" convolution is specialized to a `Conv2d` for single-frame images,
  so text-to-image decoding is 2-D. Upsampling uses `nearest-exact` interpolation. Latents are
  de-normalized as `latents * std + mean` before decoding. The decode produces 4 channels
  (RGBA). The alpha-over-white flattening applies only to the vision-encoder copy of an input
  image, which is an editing path, not text-to-image.
- **Sampling.**
  - Flow-match Euler, with `mu` computed from the image sequence length (`calculate_shift`).
  - Defaults: 40 steps, `true_cfg_scale` 1.0 (no guidance).
  - Initial noise comes from `randn_tensor` on a `torch.Generator`. MLX random cannot
    reproduce it.
- **Model card vs released diffusers.** The card asks for diffusers from git main. The released
  wheels from 0.41.0 on contain the identical pipeline and transformer, so a hash-lockable
  release can be pinned.

## 3. What exists versus the gaps

| Component | jmlx today | Gap |
| --- | --- | --- |
| Conv, norm, upsample | `conv1d/2d/3d`, `convTranspose1d/2d/3d`, `pad`, `GroupNorm`, `Upsample` (`NEAREST`/`LINEAR`), all 7.1 | Verify that `NEAREST` matches `nearest-exact` at the VAE's integer scale factors; desk check only so far |
| Text encoder | `Qwen3VlModel` (7.3b), `qwen3_vl` dispatch, interleaved `MultiAxisRope` | Pre-norm last-hidden-state output seam; raw template and `_drop_idx`; text-only load that skips the vision tower; BF16/Q4-resident weights (the current loader up-casts everything to FLOAT32) |
| Tokenizer | `HfTokenizer` over Qwen `tokenizer.json` (7.3b) | none expected |
| Transformer | `DecoderAttention` with explicit masks; MLX `scaled_dot_product_attention` | Single-stream block, block-causal mask builder, `causal_condition` timestep modulation, three-axis RoPE with `axes_dims_rope` [16, 56, 56] (a different layout from the Qwen3-VL text rotary, to verify), latent pack/unpack |
| Scheduler and sampler | 9.6 contracts (PNDM first; not yet delivered) | `FlowMatchEulerDiscreteScheduler` with exponential dynamic shift and `shift_terminal` |
| VAE decoder | 9.6 VAE-wrapper contract | Wan-style residual decoder (`base_dim` 96, `decoder_base_dim` 144, `dim_mult` [1, 2, 4, 8, 8]); de-normalization; RGBA output |
| Image output | `jmlx-vision` decodes images but has no encoder | PNG/RGBA writing (JDK `ImageIO` or a small pure-Java encoder) |
| Quantization | affine `QuantizedLinear`/`MLXQuant` (group 64) | Q4 conversion of both large components; GGUF K-quants are out of scope |

No new native binding is expected. The milestone plan re-checks this against
`req/mlx-api-inventory.md` at whatever pin is current (Phase 9.0 upgrades it first).

## 4. Memory feasibility for the 24 GB M4 Air target *(desk estimate)*

Weights only; activations are measured during the milestone. Q4 is affine group 64, about
4.5 bits per parameter.

| Component | BF16 resident | FLOAT32 resident (today's loader) | Q4 resident |
| --- | --- | --- | --- |
| transformer (7.12B) | 14.2 GB | 28.5 GB | ~4.0 GB |
| text encoder (~8.77B) | 17.5 GB | 35.1 GB | ~4.9 GB |
| VAE (~0.34B, stored F32) | 0.68 GB | 1.35 GB | not quantized |

- **BF16 doesn't fit.** Keeping BF16 everywhere needs 33 GB, which does not fit in 24 GB.
- **Staged loading is required.** Encode, release the encoder, then load the transformer. Even
  staged, a BF16 encoder alone (17.5 GB) leaves little headroom beside a desktop session.
- **The target therefore depends on Q4 for both large components** (IMG-008).
- **The FLOAT32 up-cast cannot be used.** The 7.3b/7.3c loaders up-cast all weights to
  FLOAT32, which is out of the question here. 9.7 loads BF16 or Q4 weights as they are stored,
  and records separately named tolerances beside the strict float32 gate.

**Reference memory.** A CPU float32 reference holds about 35 GB of encoder weights in its
largest stage, run component by component. It is planned on the M5 Max host. *Unverified:* that
host's installed memory is not recorded in the repo; the largest run so far had a 51.4 GB
footprint (Phi-3.5-Vision Tier-B).

## 5. Decision 1 — license

**Facts (verified 2026-11-01).** The artifact, and the Turbo variant at `fc13ba1`, are under the
"Qwen RESEARCH LICENSE AGREEMENT" (API `license_name: qwen-research`):

- §1.i defines Non-Commercial as "for research or evaluation purposes only".
- §2.a grants rights "FOR NON-COMMERCIAL PURPOSES ONLY".
- §2.b requires a separate commercial license for any commercial use.
- §3 requires passing on the agreement and a notice file on redistribution.
- §4.b requires "Built with Qwen" or "Improved using Qwen" attribution when outputs are used to
  create or improve a distributed AI model.
- The governing law is that of China, with the courts of Hangzhou.

**Recommendation: accept, for a pinned, read-only Tier-B test artifact only.** This follows the
7.4 precedent for SD 1.5:

- jmlx is MIT-licensed code and never bundles, converts for publication, or redistributes these
  weights.
- Tier-B downloads the pinned revision into runner-local storage. That is evaluation use and
  not redistribution.
- Q4 conversion tooling writes to the user's local directory only.
- jmlx's documentation states that using the weights, including in a downstream application such
  as an image-studio front end, is governed by the model's license. A commercial consumer needs
  its own license from the model owner.
- The strict Tier-A synthetic golden stays the binding gate. Its weights are generated, not
  derived from the artifact.

**If the review declines:** 9.7 is removed by a dated amendment. The alternative considered is
retargeting to `Qwen/Qwen-Image` (apache-2.0, revision `75e0b4b`, about 20.4B BF16 parameters).
That is a different and larger architecture, which would need its own scoping and is not assumed
to meet the 24 GB target.

## 6. Decision 2 — module home

**Recommendation: `jmlx-models`, extending 9.6's contracts. No new published module.** The
7.4 decision's reasons still hold:

- Every inference family so far lives in `jmlx-models`.
- The text encoder is `Qwen3VlModel`, which is already there.
- A separate module would add a release script, a place in the release order, and release-smoke
  invariants, with no consumer that needs diffusion without `jmlx-models`.

The scheduler and sampler interfaces stay model-independent within `jmlx-models` (IMG-015), so
splitting them out later remains a mechanical move.

**Reopen when** a second, architecturally distinct image family (IMG-015) or an external
consumer needs the diffusion layer without the model families. That is recorded as a later
decision, not assumed.

## 7. Decision 3 — reference toolchain

**Recommendation:**

- **A new hash-locked tool, `tools/image-gen-reference/`.** It is manual, never Gradle-invoked,
  and its scripts and goldens are digest-pinned. It is separate from `tools/hf-reference`
  because:
  - `tools/hf-reference` pins Transformers 4.57.6, whose Qwen3-VL goldens must not move.
  - The pipeline needs `transformers>=5.17`.
  
  It pins:
  - CPython 3.12
  - `diffusers==0.42.1` (a released wheel, identical to `86ef420` for the pipeline and
    transformer)
  - an exact Transformers 5.x release; the newest on 2026-11-01 is 5.22.1
  - torch
  
  The new lock is outside `verifyImageOracleDependencyPins`' shared-pin set; that guard's scope
  is restated in its own documentation. Its digests join `verifyHfReferenceGoldens`, or an
  equivalent root-`check` task.
- **The reference keeps the pipeline's pre-norm hook.** Alternatively it sets the untie config
  switch, once the pinned Transformers release is shown to contain PR #48087. A probe asserts
  that the captured conditioning differs from the post-norm `last_hidden_state`.
- **Reference precision.** CPU float32, run per component:
  1. encoder → conditioning
  2. transformer, per step, with an *injected* initial latent
  3. VAE decode
  
  Each stage writes a little-endian float32 binary golden, following the Phi-3.5-Vision Tier-B
  precedent. Seeded reproducibility of jmlx's own MLX-random noise is a separate, Java-only
  check.

## 8. Analysis, estimate and recommendation

**Go, as Phase 9 milestone 9.7, after 9.6.** The reasons:

- Every native primitive it needs exists today.
- The genuinely new work is the transformer, the flow-match scheduler, the VAE decoder, the
  conditioning seam and resident low-bit loading, all pure Java.
- 9.6 lays the scheduler, sampler and VAE contracts first.

**Estimate** *(desk, in the 7.3x milestone shape)*: **8–9 work packages, about 8–12 weeks.**

| WP | Content |
| --- | --- |
| WP0 | Pin the artifact; re-record the license acceptance |
| WP1 | Reference environment, plus probes: pre-norm hook, `_drop_idx` and template tokenization, block-causal mask, sigma schedule, `nearest-exact` vs `NEAREST`, three-axis RoPE layout |
| WP2 | Flow-match scheduler and sampler on the 9.6 contract |
| WP3 | Conditioning seam on `Qwen3VlModel`: text-only load, resident BF16 weights |
| WP4 | Transformer: blocks, mask, modulation, RoPE |
| WP5 | VAE decoder and RGBA output |
| WP6 | Tier-A synthetic golden (strict) |
| WP7 | Public API, PNG output, Q4 path and memory bounds, Tier-B |
| WP8 | Asynchronous jobs and cancellation on a single platform worker thread; benchmarks; the recorded manual M4 Air run |

The largest risks are the memory target (§4) and the Q4 quality bound for the encoder, which no
jmlx milestone has measured.

**Scope:** P0 and P1 (IMG-001 to IMG-012), text-to-image. P2 (IMG-013 to IMG-016) is deferred
to later milestones.

**Exit gate:** as stated in `req/full-roadmap.md` § 9.7.

## 9. Gate

The decision is reviewed in this record's PR. The roadmap entry (`req/full-roadmap.md` § 9.7,
the overview row and the revision history) and the requirements document
(`req/jmlx-image-generation-roadmap.md`) land in the same PR and are amended to match the merged
outcome. If the review changes a decision, the PR updates all three documents together.
