# Phase 9.7 — Qwen-Image-2.1 image generation decision

Status: decision record (reviewed via PR; the PR review is the decision gate)
Date: 2026-11-01
Base commit: `0ed5dd4` (main after PR #69), record revised after review round 1 (2026-11-01)

This record only decides; it does not start code. It settles the three questions that
`req/full-roadmap.md` § 9.7 gates the milestone on: the license, where the code lives, and the
reference toolchain. Requirements come from `req/jmlx-image-generation-roadmap.md`
(IMG-001 to IMG-016). A milestone plan, `req/plans/phase9-7-plan.md`, follows once this record
is merged and 9.6 is delivered.

**Method.** Desk analysis only; no executable probe has been run yet. The sources are:

- The Hugging Face API, configs and safetensors headers of `Qwen/Qwen-Image-2.1` at revision
  `d26bb61231c349cf6b7896fa83353113880e1ba3`.
- The diffusers source at commit `86ef420b17363c28e36ebad6b58555e9aa8feda2` (main,
  2026-10-31). The released tag v0.42.1 is byte-identical to it for `pipeline_qwenimage21.py`,
  `transformer_qwenimage21.py` and `autoencoder_kl_qwenimage21.py`. v0.41.0 has the same
  pipeline and autoencoder, but its transformer differs: it splits heads differently for
  tensor parallelism, adds a `_tp_plan`, and builds block ids differently (1,018 vs 1,034
  lines). *Unverified:* the changes look numerically equivalent on a single device.
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
- **Model card vs released diffusers.** The card asks for diffusers from git main. Released
  0.42.1 is identical to `86ef420` for all three Qwen-Image-2.1 files, so a hash-lockable release
  can be pinned. 0.41.0 is not identical (its transformer differs, as noted under Method).
- **Tokenizer files.** The artifact's `processor/vocab.json` is the same blob as the Apache-2.0
  `Qwen/Qwen3-VL-2B-Instruct` tokenizer (revision `8964489`) that 7.3b uses. Its `merges.txt`
  and `tokenizer.json` are different blobs. No gate relies on the 2B tokenizer: Tier-B loads the
  artifact's own `tokenizer.json`, and Tier-A uses a committed fixture tokenizer (§5). The WP1
  probe checks jmlx's `HfTokenizer` against the reference processor on the artifact's tokenizer,
  including the raw template and `_drop_idx`.

## 3. What exists versus the gaps

| Component | jmlx today | Gap |
| --- | --- | --- |
| Conv, norm, upsample | `conv1d/2d/3d`, `convTranspose1d/2d/3d`, `pad`, `GroupNorm`, `Upsample` (`NEAREST`/`LINEAR`), all 7.1 | none for upsampling: when upscaling, `Upsample` `NEAREST` uses `rint((i + 0.5)·n/m − 0.5)` (`Upsample.java:77`). That can differ from `nearest-exact`'s `floor((i + 0.5)·n/m)` only when `(i + 0.5)/s` is an integer, which never happens for an integer scale `s`. So the two agree at the VAE's 2× stages. A unit test pins this. |
| Text encoder | `Qwen3VlModel` (7.3b), `qwen3_vl` dispatch, interleaved `MultiAxisRope` | Pre-norm last-hidden-state output seam; raw template and `_drop_idx`; text-only load that skips the vision tower and `lm_head`; BF16- and Q4-resident loading next to today's FLOAT32 up-cast (§4) |
| Tokenizer | `HfTokenizer` over Qwen `tokenizer.json` (7.3b) | none expected. Tier-B loads the artifact's own `tokenizer.json` from the download at run time. Tier-A uses a committed fixture tokenizer (§5). |
| Transformer | `DecoderAttention` with explicit masks; MLX `scaled_dot_product_attention` | Single-stream block, block-causal mask builder, `causal_condition` timestep modulation, three-axis RoPE with `axes_dims_rope` [16, 56, 56] (a different layout from the Qwen3-VL text rotary, to verify), latent pack/unpack |
| Scheduler and sampler | 9.6 contracts (PNDM first; not yet delivered) | `FlowMatchEulerDiscreteScheduler` with exponential dynamic shift and `shift_terminal` |
| VAE decoder | 9.6 VAE-wrapper contract | Wan-style residual decoder (`base_dim` 96, `decoder_base_dim` 144, `dim_mult` [1, 2, 4, 8, 8]); de-normalization; RGBA output |
| Image output | `jmlx-vision` decodes images but has no encoder | PNG/RGBA writing (JDK `ImageIO` or a small pure-Java encoder) |
| Quantization | affine `QuantizedLinear`, `QuantizedEmbedding`, `MLXQuant` (group 64) | Q4 conversion of both large components; GGUF K-quants are out of scope |

No new native binding is expected. The milestone plan re-checks this against
`req/mlx-api-inventory.md` at whatever pin is current (Phase 9.0 upgrades it first).

## 4. Memory feasibility and precision paths *(desk estimate)*

Weights only; activations are measured during the milestone. Q4 is affine group 64, about
4.5 bits per parameter.

| Component | BF16 resident | FLOAT32 resident | Q4 resident |
| --- | --- | --- | --- |
| transformer (7.12B) | 14.2 GB | 28.5 GB | ~4.0 GB |
| text encoder, full checkpoint (~8.77B) | 17.5 GB | 35.1 GB | ~4.9 GB |
| text encoder, text-only load (~7.75B) | ~15.5 GB | ~31.0 GB | ~4.4 GB |
| VAE (~0.34B, stored F32) | 0.68 GB | 1.35 GB | not quantized |

The text-only row drops the vision tower (about 0.4B) and the unused `lm_head` (about 0.62B,
untied). Text-to-image conditioning uses neither. `embed_tokens` (about 0.62B) can use the
existing `QuantizedEmbedding`. The full-checkpoint row is the conservative bound.

**On the 24 GB target:**
- **BF16 resident doesn't fit.** It needs about 30–33 GB.
- **Q4 for both large components fits.** About 9.75 GB of weights (text-only encoder, F32 VAE)
  can stay resident together, so the target depends on Q4 (IMG-008).
- **Staging is a memory-budget policy, not a requirement.** The policy is set on the loaded
  model object, and a request always runs against that object (IMG-007):
  - `RESIDENT` (the default): every component stays loaded between jobs.
  - `STAGED`: the encoder and the transformer are **never resident together**. Each job loads
    the encoder, encodes, and releases it; then it loads the transformer and denoises. The VAE
    (1.35 GB) stays loaded. The transformer stays loaded after the job until the next job's
    encode starts.
    
    The peak is therefore the larger of the two plus the VAE, and every job re-reads both large
    components.

  | Path | `RESIDENT` peak weights | `STAGED` peak weights | `STAGED` re-read per job |
  | --- | --- | --- | --- |
  | FLOAT32 | ~60.9 GB | ~32.4 GB (encode stage) | ~29.7 GB of BF16 files, up-cast |
  | BF16 | ~30.4 GB | ~16.2 GB | ~29.7 GB |
  | Q4 | ~9.75 GB | ~5.75 GB | ~8.4 GB |
  
  BF16 `STAGED` is not assumed to fit the 24 GB target with activations and a desktop session;
  it is measured only on the M5 Max. The 24 GB path is Q4.

**Three weight-residency paths, all measured against one CPU float32 reference:**
- **FLOAT32**, the existing up-cast pattern. It runs the strict gates:
  - The Tier-A synthetic checkpoint stores its weights as BF16, as 7.3b WP8 (104 BF16 tensors)
    and 7.3c WP6 (76 BF16 tensors) do. The strict gate runs through the FLOAT32 up-cast at
    load, and the same checkpoint also exercises BF16-resident loading.
  - The Tier-B tight-ε run uses this path with `STAGED` on the M5 Max, peaking at about 32.4 GB
    of weights in the encode stage.
- **BF16 resident** and **Q4 resident**: Q4 is the path for the 24 GB target. Each has a
  separately named bound against the same reference, following the 7.3c precedent of strict and
  TF32 bounds side by side.

**Reference memory.** The CPU float32 reference runs component by component; its largest stage
holds about 35 GB of encoder weights (full checkpoint). It is planned on the M5 Max host.
*Unverified:* that host's installed memory is not recorded in the repo. The largest run so far
had a 51.4 GB footprint (Phi-3.5-Vision Tier-B).

## 5. Decision 1 — license

**Facts (verified 2026-11-01).** The artifact, and the Turbo variant at `fc13ba1`, are under the
"Qwen RESEARCH LICENSE AGREEMENT" (API `license_name: qwen-research`):

- §1.i defines Non-Commercial as "for research or evaluation purposes only".
- §2.a grants rights "FOR NON-COMMERCIAL PURPOSES ONLY".
- §2.b requires a separate commercial license for any commercial use.
- §1.f/g define the "Materials" broadly enough to include the configuration files and
  documentation shipped with the weights.
- §3 requires passing on the agreement and a notice file on redistribution.
- §4.b requires "Built with Qwen" or "Improved using Qwen" attribution when outputs are used to
  create or improve a distributed AI model.
- The governing law is that of China, with the courts of Hangzhou.

**Decision: accept, for a pinned, read-only Tier-B test artifact only.**

**How this differs from 7.4.** The test-artifact handling is the same as 7.4's acceptance of SD
1.5, but the consequence is not:

- CreativeML OpenRAIL-M permits commercial use, subject to use-based restrictions.
- The Qwen Research License forbids commercial use outright.
- 9.7 therefore ships a published, MIT-licensed `jmlx-models` API whose only verified
  checkpoint cannot be used commercially without a separate license from the model owner. The
  code carries no restriction, but the verified use case does.
- The milestone's documentation (the compatibility matrix and the model's Javadoc) states this
  plainly, so no consumer mistakes "verified with jmlx" for "licensed for your product".

**What may and may not enter the public repository.** This is an *unverified legal assessment*,
which the PR review accepts or overrides:

- **No file from the artifact is committed.** That covers configs, `model_index.json`,
  `processor/*`, `tokenizer.json`, weights, and any converted or quantized derivative. The
  Tier-B suite reads them from the runner-local download at run time.
- **The manifest records facts about the artifact** (revision, file names, sizes, SHA-256s,
  tensor shapes and dtypes), not its files. That is the existing Tier-B manifest pattern.
- **Committed Tier-B goldens are model outputs, not Materials**: conditioning tensors, per-step
  denoiser outputs and the VAE decode. §4.b's attribution duty applies only when outputs are
  used to create or improve a distributed AI model, which a test golden does not.
- **Tier-A fixtures use no artifact file.** The synthetic checkpoint is generated. Its tokenizer
  is a committed, hand-written tiny fixture tokenizer, the same pattern as the 2,031-byte
  21-token tokenizers of 7.3b WP8 and 7.3c WP6. It carries the template's special tokens
  (`<|im_start|>`, `<|im_end|>`) and the system/user/assistant role markers. It comes with a
  minimal hand-written chat template that renders the system message the way the artifact's
  does, so the raw template and `_drop_idx` are still exercised.
  
  WP1 checks that rendering against the artifact's own template, which is not committed. The
  real 151,936-token tokenizer would add a 7 MB `tokenizer.json` and a 151,936-row embedding to
  the synthetic checkpoint, and it would still not match Tier-B, which runs the artifact's own,
  different tokenizer.
- **Size budget for committed Tier-B goldens: at most 10 MiB**, as little-endian float32. At
  512×512, one latent tensor is 1,024 tokens × 64 channels, which is 256 KiB. The budget covers:
  - the conditioning tensor (under 1 MiB for a short prompt);
  - teacher-forced input/output pairs at a fixed set of 8 steps (4 MiB), always including the
    first and last steps (WP0 chooses the set);
  - the reference's final latent and its float32 RGBA decode (about 4.25 MiB).
  
  Storing all 40 steps (about 20 MiB) is not chosen.
- **jmlx never bundles, converts for publication, or redistributes the weights.** Q4 conversion
  tooling writes only to the user's local directory.
- **The strict Tier-A synthetic golden stays the binding gate.**

**If the review declines:** 9.7 is removed by a dated amendment. The alternative considered is
retargeting to `Qwen/Qwen-Image` (apache-2.0, revision `75e0b4b`, about 20.4B BF16 parameters).
That is a different and larger architecture, which would need its own scoping and is not assumed
to meet the 24 GB target.

## 6. Decision 2 — module home

**Decision: `jmlx-models`, extending 9.6's contracts. No new published module.** The 7.4
decision's reasons still hold:

- Every inference family so far lives in `jmlx-models`.
- The text encoder is `Qwen3VlModel`, which is already there.
- A separate module would add a release script, a place in the release order, and release-smoke
  invariants, with no consumer that needs diffusion without `jmlx-models`.

9.6 (UNet, CLIP, PNDM) and 9.7 (single-stream DiT, Qwen3-VL conditioning, flow-match) are
already two architecturally distinct families. They are the first two users of the
model-independent scheduler and sampler interfaces (IMG-015), and both stay in `jmlx-models` on
purpose: two in-module users are what validate the interfaces before a module boundary freezes
them.

**Reopen when** an external consumer needs the diffusion layer without the model families, or a
third image family would make `jmlx-models` carry diffusion code that most of its users don't
need. Either is recorded as a later decision, not assumed.

## 7. Decision 3 — reference toolchain

**Decision:**

- **A new hash-locked tool, `tools/image-gen-reference/`.** It is manual, never Gradle-invoked,
  and its scripts and goldens are digest-pinned. It is separate from `tools/hf-reference`
  because:
  - `tools/hf-reference` pins Transformers 4.57.6, whose Qwen3-VL goldens must not move.
  - The pipeline needs `transformers>=5.17`.
  
  It pins:
  - CPython 3.12
  - `diffusers==0.42.1` (a released wheel, identical to `86ef420` for all three Qwen-Image-2.1
    files)
  - one exact Transformers 5.x release; the newest on 2026-11-01 is 5.22.1
  - one exact torch release
  
  WP1 picks the exact Transformers and torch versions and records them in the lock. This record
  does not.
- **The pin guard does not cover the new lock.** `verifyImageOracleDependencyPins` takes an
  explicit list of locks (image oracle, fast-comparison, `tools/qwen-vl-reference` and
  `tools/hf-reference`), so the new lock is outside its shared Pillow/NumPy/Transformers set by
  construction. WP1 records that in the guard's description.
- **A new digest task.** The tool's scripts, manifest and goldens are SHA-256-pinned in its own
  `provenance.json`, verified by a new pure-Java root-`check` task,
  `verifyImageGenReferenceGoldens`, built the way `verifyHfReferenceGoldens` is.
- **Two diffusers pins, on purpose.** 9.6 adds diffusers to the `tools/hf-reference` lock for SD
  1.5, under Transformers 4.57.6. 9.7 pins `diffusers==0.42.1` in the new lock, under
  Transformers 5.x. The two serve different pipelines in separate environments, and nothing
  needs them to agree, so no cross-check is added.
- **The reference keeps the pipeline's pre-norm hook.** Alternatively it sets the untie config
  switch, once the pinned Transformers release is shown to contain PR #48087. A probe asserts
  that the captured conditioning differs from the post-norm `last_hidden_state`.
- **Reference precision and comparison mode.** The reference is CPU float32, run per component:
  1. encoder → conditioning
  2. transformer, *teacher-forced*: each compared step's input is the reference's own latent at
     that step, so errors do not accumulate across steps. The initial latent is injected.
  3. VAE decode of the reference's final latent
  
  Each stage writes a little-endian float32 binary golden within the §5 budget, following the
  Phi-3.5-Vision Tier-B precedent.
  
  The binding ε applies to the FLOAT32 path (§4). The BF16 and Q4 paths have separately named
  bounds. A free-running 40-step trajectory, which drifts in BF16 and Q4, is reported but not
  gating. Seeded reproducibility of jmlx's own MLX-random noise is a separate, Java-only check.

## 8. Analysis, estimate and recommendation

**Go, as Phase 9 milestone 9.7, after 9.6.** The reasons:

- Every native primitive it needs exists today.
- The genuinely new work is the transformer, the flow-match scheduler, the VAE decoder, the
  conditioning seam and resident low-bit loading, all pure Java.
- 9.6 lays the scheduler, sampler and VAE contracts first.

**Estimate** *(desk, in the 7.3x milestone shape)*: **9 work packages, about 8–12 weeks.**

| WP | Content |
| --- | --- |
| WP0 | Pin the artifact; re-record the license acceptance; fix the golden step set |
| WP1 | Reference environment and digest task, plus probes: pre-norm hook; `HfTokenizer` vs the reference processor on the artifact's tokenizer, with the raw template and `_drop_idx`; the fixture chat template's system-message rendering; block-causal mask; sigma schedule; three-axis RoPE layout |
| WP2 | Flow-match scheduler and sampler on the 9.6 contract |
| WP3 | Conditioning seam on `Qwen3VlModel`: text-only load, FLOAT32 and resident BF16 weights |
| WP4 | Transformer: blocks, mask, modulation, RoPE |
| WP5 | VAE decoder and RGBA output; the `NEAREST`/`nearest-exact` unit test |
| WP6 | Tier-A synthetic golden: BF16-stored weights; strict via the FLOAT32 up-cast; BF16-resident load under its named bound; fixture tokenizer |
| WP7 | Public API with the `RESIDENT`/`STAGED` policy, PNG output, Q4 path and memory bounds, Tier-B |
| WP8 | Asynchronous jobs and cancellation on a single platform worker thread; the release-smoke synthetic case; benchmarks; the recorded manual M4 Air run |

The largest risks are the memory target (§4) and the Q4 quality bound for the encoder, which no
jmlx milestone has measured.

**Scope:** P0 and P1 (IMG-001 to IMG-012), text-to-image. P2 (IMG-013 to IMG-016) is deferred
to later milestones.

**Exit gate:** as stated in `req/full-roadmap.md` § 9.7. It lists which acceptance criteria are
gating and which are recorded only.

## 9. Gate

The decision is reviewed in this record's PR, and merging the PR accepts §5–§7. The roadmap
entry (`req/full-roadmap.md` § 9.7, the overview row and the revision history) and the
requirements document (`req/jmlx-image-generation-roadmap.md`) land in the same PR and state the
decided outcome. If the review changes a decision, the PR updates all three documents together.
