# Phase 7.4 — Audio and diffusion capability decision

Status: decision record (reviewed via PR; the PR review is the decision gate)
Date: 2026-10-31
Base commit: `8cc938c` (main after PR #66), record revised same day after review rounds 1–4
Decides only. A "go" schedules implementation into a later phase by a dated
roadmap amendment (`req/full-roadmap.md` § Phase 7); it does not extend Phase 7
and does not start code. 7.5 (MLX pin upgrade) follows 7.4 directly.

Method: desk analysis of `req/mlx-api-inventory.md`, the phase plans and the
installed transformers 4.57.6 modeling source (`modeling_whisper.py`,
`modeling_clip.py`), plus the Hugging Face API for artifact facts (sizes,
licenses, repo redirects), plus two executable reference probes run manually in
the locked `tools/hf-reference` venv (torch 2.6.0, transformers 4.57.6, CPython
3.12.16, Darwin arm64, seed 20261031). The probes are digest-pinned in
`tools/hf-reference/provenance.json` and enforced by `verifyHfReferenceGoldens`
in root `build.gradle` (a wrong or missing digest fails the build), are never
Gradle-invoked, and write no golden. Each probe states which of its output
blocks are measured and which are desk mappings; the full JSON output is
recorded verbatim in §6.

## 1. Inventory counts (mandate: relevant groups)

From `req/mlx-api-inventory.md` at the base commit:

- **FFT family — entirely unplanned.** All 16 downcalls
  (`mlx_fft_fft`, `mlx_fft_fft2`, `mlx_fft_fftn`, `mlx_fft_ifft`, `mlx_fft_ifft2`,
  `mlx_fft_ifftn`, `mlx_fft_rfft`, `mlx_fft_rfft2`, `mlx_fft_rfftn`,
  `mlx_fft_irfft`, `mlx_fft_irfft2`, `mlx_fft_irfftn`, `mlx_fft_fftshift`,
  `mlx_fft_ifftshift`, `mlx_fft_fftfreq`, `mlx_fft_rfftfreq`), the three FFT-norm
  constants (`MLX_FFT_NORM_FORWARD/BACKWARD/ORTHO`) and `MLX_COMPLEX64` are
  `unplanned`. `MLXFFT` (complex dtype policy, FFT family, shifts, normalization)
  is Phase 11 milestone 2 scope (`req/full-roadmap.md` § Phase 11).
- **Convolution/upsample coverage after 7.1 — complete for a UNet/VAE.**
  `mlx_conv1d/2d/3d`, `mlx_conv_transpose1d/2d/3d`, `mlx_conv_general`,
  `mlx_pad`, `mlx_pad_symmetric` and `mlx_as_strided` are all `implemented`
  (10/10). The 7.1 `nn` layers built on them (`Conv1d/2d/3d`,
  `ConvTranspose1d/2d/3d`, `MaxPool/AvgPool`, `Upsample` nearest/linear,
  `GroupNorm`) are delivered.
- **Encoders/attention (7.2) — implemented.** `BidirectionalAttention`,
  `CrossAttention`, `EncoderBlock`, BERT/T5-family loading and generation.

## 2. Candidate reference model per modality (mandate)

- **Audio: `openai/whisper-tiny`.** Facts verified against the Hugging Face API
  on 2026-10-31: `model.safetensors` is **151,061,672 bytes (fp32)**; the
  license the artifact declares is **apache-2.0** (the OpenAI GitHub code repo
  is MIT — the Tier-B pin states the artifact's declared license). 39 M params,
  4+4 layers, vocab 51 865. Encoder-decoder, reusing the Phase 7.2 seq2seq
  findings (`CrossAttention`, `StaticKVCache`, T5 generation loop). The probe
  (§6.1) computes the log-mel front end: a small fixed contract — 16 kHz mono
  in, fixed (1, 80, 3000) float32 out (25 ms window, 10 ms hop, Slaney 80-band
  mel, clamp/log10, drop the last frame `[:, :-1]`, dynamic-range compress,
  normalize).
- **Diffusion: a small SD-class UNet/VAE pipeline.** Canonical artifact
  `stable-diffusion-v1-5/stable-diffusion-v1-5` (the `runwayml/` id now
  307-redirects there; the milestone WP0 pins the canonical id and a revision).
  fp32 safetensors component sizes, verified via the HF API on 2026-10-31:
  UNet 3,438,167,540 B + VAE 334,643,276 B + text encoder 492,265,874 B =
  **4,265,076,690 B**. License: **CreativeML OpenRAIL-M** — decided in §7, not
  deferred. The strict Tier-A gate needs no external artifact at all: a tiny
  synthetic checkpoint, the pattern every 7.3x milestone used.

## 3. Required new surface and Phase 11 overlap (mandate)

- **Audio.** New: 16 kHz mono PCM/WAV decode (no audio module exists;
  `jmlx-vision` is image-only — a small new pure-Java `jmlx-audio` module),
  resampling for non-16 kHz input, **forward STFT only** (Whisper never
  inverts; there is no ISTFT anywhere in the pipeline), and the Slaney mel
  filterbank (pure-Java port; the filterbank math needs no FFT).
  **Phase 11 overlap: STFT sits on `mlx_fft_*`, Phase 11 milestone 2.** Two
  honest options: (a) pull `MLXFFT` forward into the audio milestone,
  7.1-style, doubling its scope; or (b) schedule audio as Phase 11 milestone
  11.5, after 11.2, so FFT is built once in its own phase with its own
  differential tests. Option (b) is recommended (§7).
  **Precision reality (added in review round 2):** a float32 FFT through MLX
  will not match torch's `torch.stft` (or pocketfft) bit-for-bit — unlike the
  vision byte-exact gates, which were pure-Java ports of deterministic kernels.
  The 11.5 exit gate must therefore be a **recorded tolerance on the log-mel
  tensor** (like the logits ε), not byte-exactness. The oracle path must also
  be pinned explicitly: transformers has two STFT branches with different
  floating-point results — the torch branch drops the last frame on the
  complex STFT output before `abs()²` (`stft[..., :-1]`); the numpy branch
  drops `log_spec[:, :-1]` after log10. The probe measured the **torch branch**
  (transformers selects it when torch is installed), and that is the oracle.
  **Module ownership (revised in review round 3):** the Java 21 and
  native-facing halves split, following the `jmlx-vision` precedent —
  - **`jmlx-audio` (Java 21, native-free, no `jmlx-core` dependency):** WAV
    decode, resampling, window functions, the Slaney mel filterbank and the
    log scaling. All the byte/tolerance-critical preprocessing math lives
    here, exactly where `jmlx-vision` puts image preprocessing.
  - **`jmlx-core`:** a forward-STFT composition facade over milestone 2's
    `MLXFFT` (built once with the FFT family; forward only — Whisper has no
    ISTFT). Being core code it is Java 25 like the rest of the native chain.
  - **`jmlx-models`:** the Whisper model itself, `api`-depending on
    `jmlx-audio` (the vision pattern) and composing the two halves.
  **STFT-to-mel hand-off (added in review round 4):** the `jmlx-core`
  facade computes the power spectrum on MLX — the pinned torch branch's
  `stft[..., :-1].abs() ** 2`, i.e. the last frame is dropped on the complex
  STFT output *before* the power — and that evaluated power spectrum (a
  Java `float[]`) is the single MLX→Java hand-off; `jmlx-audio` then does
  the mel matmul and the log scaling in pure Java. The consequence is
  recorded, not optional: the end-to-end log-mel tolerance fixture lives in
  `jmlx-models` (native-free `jmlx-audio` cannot run the STFT), and
  `jmlx-audio`'s own tests start from oracle power spectra.
  This keeps `jmlx-audio` compilable on the Java 21 toolchain — a Java 21
  module cannot depend on Java 25 `jmlx-core` class files.
  **Publishing:** `jmlx-audio` **is published** (a fresh `0.1.0-SNAPSHOT`
  line, like vision): the project goes from seven to eight published
  modules, and the invariants grow with it — `verifyReleaseScriptsMatch`, its
  own byte-identical `release.sh`, a place in the release order (before
  `jmlx-models`, alongside `jmlx-vision`), and the release smoke's consumer
  resolution.
  Tokenizer: whisper's `tokenizer.json` (no normalizer, ByteLevel pre-tokenizer,
  BPE, TemplateProcessing) fits jmlx-tokenizer's existing components. The
  generation-side controls are model-level, not tokenizer-level: the forced
  decoder prefix (startoftranscript + optional language/task/notimestamps),
  `suppress_tokens`/`begin_suppress_tokens` from `generation_config`, and the
  448-position decode cap (`max_target_positions`).
- **Diffusion.** New and **entirely pure-Java**: noise schedulers, the sampler
  loop, the VAE decoder wrapper (scaling constant 0.18215 is a literal), and
  the CLIP text encoder stack. **No Phase 11 overlap: the pipeline exercises
  zero FFT/linalg family members.** Scheduler (corrected in review round 3,
  narrowed in review round 4): the artifact's `model_index.json` (sha
  `451f4fe`) names **`PNDMScheduler`** as the default, so PNDM heads the list
  with DDPM/DDIM/Euler as the common alternatives. The pinned
  `scheduler/scheduler_config.json` (same sha) sets `skip_prk_steps: true` —
  the Runge–Kutta warm-up **never runs**, so the 9.6 port needs only the PLMS
  path with its lower-order first steps (there is no PRK phase to scope or
  estimate) — plus `steps_offset: 1`, `set_alpha_to_one: false`,
  `beta_schedule: "scaled_linear"`, `beta_start: 0.00085` and
  `beta_end: 0.012`, which the 9.6 scheduler must port (steps_offset and
  set_alpha_to_one change the timestep grid and the final alpha). The 9.6
  Tier-B reference **pins the artifact's default scheduler**, so the exit
  gate's per-step noise-prediction comparison is meaningful. Checkpoint
  layout (verified; the remap scope was corrected in review round 4): the
  pinned **VAE** stores its mid-block AttentionBlock attention under the
  **legacy key names `query`/`key`/`value`/`proj_attn`**, which diffusers
  remaps to `to_q`/`to_k`/`to_v`/`to_out.0` at load — the 9.6 loader
  implements that **VAE-only** remap (the UNet's attention already uses the
  current `to_q`/`to_k`/`to_v`/`to_out.0` names: 0 of its 687 keys end in
  `.query.weight` at that sha); the VAE mid-block weights are stored as
  **2-D Linear** tensors (`[512, 512]`), math-equivalent to the 1×1 convs the
  architecture implies.
  Tier-B reference tooling: the CPU float32 reference for the real UNet/VAE/
  pipeline needs **diffusers**, which is not in `tools/hf-reference`'s hash
  lock — 9.6 adds it as a hash-locked dependency, checked against
  `verifyImageOracleDependencyPins`' shared Pillow/NumPy/Transformers pins.
  CLIP text encoder, per the installed `modeling_clip.py` (4.57.6): pre-LN
  (`layer_norm1/2` before attention/MLP), a **causal** mask
  (`_create_4d_causal_attention_mask`) and **QuickGELU** — BERT's post-LN
  bidirectional `EncoderBlock` is not a drop-in. SD 1.5 uses `CLIPTextModel`:
  it conditions on the full `last_hidden_state` (77×768 after
  `final_layer_norm`) — **no pooling, no `text_projection`** (those belong to
  `...WithProjection`, used by contrastive image-text models, not SD). The
  milestone must verify how causal masking is wired over an encoder stack
  (`DecoderAttention`-style reuse versus a causal option on an encoder block).
  Tokenizer (corrected in review round 2): the pinned SD artifact ships **no
  `tokenizer.json`** — only `tokenizer/{vocab.json, merges.txt,
  special_tokens_map.json, tokenizer_config.json}` (slow-tokenizer files,
  verified against the HF API at sha `451f4fe`). Two paths: **(a) a
  slow-tokenizer loader in `jmlx-tokenizer` over vocab.json + merges.txt**
  — this record's recommendation, since the artifact itself ships
  exactly those files; or (b) pin `openai/clip-vit-large-patch14`'s
  `tokenizer.json` as a second artifact and add `Split.invert=true` support —
  rejected by jmlx-tokenizer today in **both** `TokenizerJsonLoader` and
  `PreTokenizerPipeline` — plus a `RobertaProcessing` post-processor, with an
  equivalence check against the artifact's own files. Note that the
  clip-vit-large-patch14 `tokenizer.json` is **not loadable by jmlx-tokenizer
  as-is** because of that `invert=true` step, so "covers the tokenizer parts"
  was wrong in round 1; the only component overlap is the BPE model itself.
  Option (a) is not just "small wrapper logic" (round-2 correction): the slow
  tokenizer's normalization **depends on whether ftfy is installed**
  (`tokenization_clip.py`: `ftfy.fix_text` if importable, else a vendored BERT
  `BasicTokenizer(strip_accents=False, do_split_on_punc=False)`). The
  `tools/hf-reference` lock has **no ftfy** (verified), so the reference takes
  the BasicTokenizer fallback — **that fallback is the pinned oracle branch**,
  and the Java port must reproduce its full 4.57.6 step order (added in
  review round 4): control-character cleaning (`_clean_text`), CJK character
  splitting (`_tokenize_chinese_chars`), **NFC Unicode normalization**
  (`unicodedata.normalize("NFC", text)` — without it, decomposed input
  tokenizes differently from the oracle), whitespace tokenization,
  lowercasing (no accent stripping, `strip_accents=False`), the IGNORECASE
  `'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+` pre-tokenizer,
  the whitespace re-collapse after the split
  (`whitespace_tokenize(" ".join(split_tokens))`), the byte encoder, and the
  `</w>` end-of-word suffix (already supported in jmlx-tokenizer's
  `TokenizerModels`/`TokenizerDefinition`). BOS 49406 / EOS
  49407 and the 77-token pad/truncate remain wrapper logic. This
  normalization port is counted in the 9.6 tokenizer work package.

## 4. What exists versus the gaps

Audio (desk mapping, checked against the installed `modeling_whisper.py`; the
front-end row is probe-measured):

| Whisper component | jmlx today | Gap |
| --- | --- | --- |
| log-mel front end | — | WAV decode + forward STFT + mel filterbank (§3) |
| Conv1d stem | `MLXConv.conv1d` (7.1) | none — `conv1` is stride 1, only `conv2` is stride 2 |
| encoder blocks | LayerNorm/GELU facades | none — whisper is pre-LN **LayerNorm (with bias) + GELU**, not RMSNorm |
| cross-attention decoder | `CrossAttention` + `StaticKVCache` (7.2) | none — decoder blocks are also LayerNorm + GELU |
| attention bias detail | — | `k_proj` is bias-free while `q_proj`/`v_proj`/`out_proj` keep bias; the loader must read per-projection bias presence |
| positional tables | `Embedding`/sinusoid composition | encoder positions: frozen sinusoidal table capped at 1500; decoder learned positions capped at 448 |
| autoregressive decode | T5 generation path (7.2) | none — plus the model-level prefix/suppression controls of §3 |

Diffusion (desk mapping against the inventory and `nn` facades; the probe
executes an SD-shaped UNet/VAE forward and measures its shapes/checksums —
see §6.2 `measured_vs_desk`): conv2d, GroupNorm, LayerNorm, SiLU,
self-attention, cross-attention text conditioning, GEGLU feed-forward
(Linear + chunk + **exact GELU** on the gate per diffusers `activations.py` —
`F.gelu`, `approximate="none"`, *not* QuickGELU; a 9.6 implementation built
on QuickGELU would fail a strict golden), linear time
embedding, the Transformer2D wrapper (GroupNorm + 1×1 `proj_in`/`proj_out`
convs), residual and concat skips, and nearest-interpolate-plus-conv
upsampling (SD's Upsample2D **and** all three VAE decode stages use this — no
`ConvTranspose2d` anywhere; the VAE mid block is single-head attention with no
feed-forward, per diffusers `AttentionBlock`) are all implemented. Gaps: the
pure-Java schedulers (**PNDM first** — the artifact's default — then
DDPM/DDIM/Euler) and sampler loop, the VAE wrapper, the CLIP text stack with
its causal masking question, the VAE's legacy attention-key remap
(`query`/`key`/`value`/`proj_attn` → `to_q`/`to_k`/`to_v`/`to_out.0`; the
UNet uses the current names) in the 9.6 loader, and the CLIP slow-tokenizer
loader with its pinned BasicTokenizer
normalization port (§3).

## 5. Tier-B feasibility, licenses, estimate (mandate)

- **Audio / whisper-tiny.** Artifact 151 MB fp32 — about **340× smaller** than
  the 51 GB Phi-3.5-Vision Tier-B run's measured footprint, the largest the
  project has executed; any existing runner fits it. License apache-2.0, as
  declared by the artifact (§2). Tier-B reference is a CPU float32 eager
  forward per greedy step (the `tools/hf-reference` pattern). Estimate, in the
  7.3x milestone shape and **including the tokenizer/generation-control
  surface of §3**: **4–5 work packages, ~3–4 weeks** — front end + oracle
  fixtures (tolerance-based, per §3), decoder wiring reusing 7.2, synthetic
  Tier-A golden, Tier-B pin. Effort concentrates in the front end and its
  precision evidence (the pinned torch-STFT oracle path and the `stft[...,
  :-1]` frame drop).
- **Diffusion / SD-class.** Artifact 4.27 GB fp32 total — below the largest
  existing Tier-B run; fits the self-hosted runner. **License, decided here
  (not deferred):** the Tier-B pin accepts CreativeML OpenRAIL-M for a pinned,
  read-only test artifact — no jmlx code or shipped artifact carries the
  license; the strict Tier-A synthetic golden remains the binding gate. If the
  review disagrees, the fallback is synthetic-only Tier-B, which keeps the
  milestone shippable. Estimate, same shape and **including the CLIP
  slow-tokenizer loader with its BasicTokenizer normalization port (§3) and
  the hash-locked diffusers addition to the hf-reference venv (§3)**:
  **5–6 work packages, ~4–6 weeks** — scheduler + sampler (PNDM first), VAE,
  CLIP text, UNet assembly, tokenizer loader, synthetic golden, Tier-B.
  Larger than audio because three model components land at once, but none
  requires new native surface.

## 6. Probe findings (verbatim)

### 6.1 Audio — `whisper_logmel_probe.py`

```json
{
  "probe": "phase7-4-audio-whisper-logmel",
  "measured_vs_desk": "the frontend_contract and sample_values_checksum blocks are computed by this probe; jmlx_surface_mapping and loader_details are desk mappings checked against the installed transformers 4.57.6 modeling source, not executed",
  "environment": {
    "python": "3.12.16",
    "torch": "2.6.0",
    "transformers": "4.57.6",
    "numpy": "2.5.3",
    "host_os": "Darwin",
    "host_arch": "arm64",
    "device": "cpu",
    "seed": 20261031
  },
  "waveform": {
    "samples": 480000,
    "sample_rate": 16000,
    "duration_s": 30.0,
    "synthesis": "sum of 0.15*sin sines at 220/440/1760 Hz plus 0.005 gaussian noise, np.random.default_rng(20261031)"
  },
  "frontend_contract": {
    "input_features_shape": [
      1,
      80,
      3000
    ],
    "input_features_dtype": "torch.float32",
    "frame_window_ms": 25,
    "hop_ms": 10,
    "mel_filters": 80,
    "fft_bins": 201,
    "mel_basis_source": "transformers.audio_utils.mel_filter_bank (pure numpy, Slaney)",
    "stft_path_measured": "torch branch: torch.stft(...) then stft[..., :-1].abs() ** 2 \u2014 the last frame is dropped on the complex STFT output BEFORE the power, then the mel matmul, clamp(1e-10), log10",
    "stft_path_not_measured": "numpy branch (pocketfft, no torch installed): log_spec[:, :-1] drops the last frame AFTER log10 \u2014 a different code path with different floating-point results; a Java port must pin which branch is the oracle",
    "log_scaling": "torch path: clamp(mel, min=1e-10).log10(); dynamic-range compress to max-8: mel = max(mel, mel.max() - 8.0); normalize: mel = (mel + 4.0) / 4.0",
    "fixed_chunk": "30 s -> fixed (1, 80, 3000) float32 feature tensor regardless of audio length (pad/truncate)"
  },
  "sample_values_checksum": {
    "input_features_sum": 33672.32073581219,
    "input_features_min": -0.7975344657897949,
    "input_features_max": 1.1794712543487549
  },
  "jmlx_surface_mapping": {
    "already_available": {
      "conv1d_encoder_stem": "mlx_conv1d implemented (MLXConv, jmlx-core, Phase 7.1); whisper uses Conv1d(80->d, k=3, stride=1) then Conv1d(d->d, k=3, stride=2) \u2014 only the second conv downsamples",
      "transformer_encoder": "whisper encoder blocks are pre-LN with LayerNorm (with bias) and GELU ff (ACT2FN['gelu']); LayerNorm/GELU facades exist in jmlx-core nn (RMSNorm is NOT what whisper uses)",
      "cross_attention_decoder": "CrossAttention + StaticKVCache implemented (Phase 7.2); whisper decoder layers are also pre-LN LayerNorm + GELU",
      "seq2seq_decode": "generation is autoregressive token decode through the cross-attention decoder, the same surface as T5 (Phase 7.2); no CTC/loss surface is needed"
    },
    "gaps": {
      "audio_decode": "no audio module exists (jmlx-vision is image-only); 16 kHz mono PCM/WAV decode is pure-Java surface (a jmlx-audio or jmlx-vision extension)",
      "stft": "needs mlx_fft_* (all 16 downcalls + MLX_COMPLEX64 unplanned in req/mlx-api-inventory.md; MLXFFT is Phase 11 milestone 2). Whisper is FORWARD STFT only \u2014 spectrogram, mel matmul, then the [:, :-1] frame drop \u2014 no ISTFT anywhere in the pipeline",
      "mel_filterbank": "pure-Java port (Slaney mel scale + triangular filters); depends on the STFT magnitude output, not on FFT directly for the filterbank math",
      "resample": "arbitrary sample-rate resampling to 16 kHz is pure-Java surface if non-16 kHz input is accepted (whisper requires 16 kHz)"
    },
    "loader_details": {
      "k_proj_bias": "WhisperAttention sets k_proj with bias=False while q_proj/v_proj/out_proj keep bias=True \u2014 the checkpoint loader must read per-projection bias presence, not assume uniformity",
      "positional_tables": "encoder embed_positions is a frozen sinusoidal table capped at max_source_positions=1500; decoder learned positions capped at max_target_positions=448 \u2014 decode beyond 448 tokens is unsupported by the architecture itself",
      "generation_controls": "whisper decode needs the forced decoder prefix (startoftranscript + optional language/task/notimestamps special tokens) and suppress_tokens/begin_suppress_tokens (from generation_config) applied by the sampler \u2014 new model-level surface, no native dependency"
    }
  },
  "reference_model": {
    "repo": "openai/whisper-tiny",
    "license": "apache-2.0 \u2014 the license the Hugging Face artifact declares (the OpenAI GitHub code repo is MIT); the Tier-B pin must state the artifact's declared license",
    "size_bytes_safetensors": 151061672,
    "params_M": 39,
    "encoder_layers": 4,
    "decoder_layers": 4,
    "vocab_size": 51865,
    "checkpoint_format": "safetensors on the HF repo (fp32; ~75 MB fp16 GGUF exists in the whisper.cpp ecosystem); original OpenAI release is .pt"
  }
}
```

### 6.2 Diffusion — `diffusion_unet_probe.py`

```json
{
  "probe": "phase7-4-diffusion-unet-vae",
  "measured_vs_desk": "the shapes and value_checksums blocks are computed by this probe; unet_primitive_map, vae_primitive_map and the *_gap blocks are desk mappings against jmlx's inventory and nn facades, not executed",
  "environment": {
    "python": "3.12.16",
    "torch": "2.6.0",
    "transformers": "4.57.6",
    "host_os": "Darwin",
    "host_arch": "arm64",
    "device": "cpu",
    "seed": 20261031
  },
  "shapes": {
    "latents_in": [
      1,
      4,
      16,
      16
    ],
    "unet_noise_pred_out": [
      1,
      4,
      16,
      16
    ],
    "context_text_embedding": [
      1,
      77,
      768
    ],
    "resolution_path": "16x16 -> downsample 8x8 (Transformer2D cross-attn here) -> upsample back to 16x16",
    "vae_pixels_out": [
      1,
      3,
      128,
      128
    ],
    "vae_decode_factor": "16x16 latents -> 128x128 pixels (3 nearest-upsample stages, 8x, as SD's block_out_channels [128,256,512,512] implies)"
  },
  "value_checksums": {
    "noise_pred_sum": -8.594521004706621,
    "pixels_sum": -5508.250665940905,
    "pixels_range": [
      -1.8475260734558105,
      2.129082202911377
    ]
  },
  "unet_primitive_map": {
    "conv2d": {
      "used": "stem/res/out/downsample convs",
      "jmlx": "implemented (MLXConv.conv2d, Phase 7.1)"
    },
    "groupnorm": {
      "used": "res block norms, num_groups=32",
      "jmlx": "implemented (GroupNorm, Phase 7.1)"
    },
    "layernorm": {
      "used": "Transformer2D block norms (cross-attn block)",
      "jmlx": "implemented (LayerNorm, Phase 4)"
    },
    "silu": {
      "used": "res block activations",
      "jmlx": "implemented (SiLU, Phase 4)"
    },
    "self_attention": {
      "used": "Transformer2D self-attention",
      "jmlx": "implemented (MultiHeadAttention/attention facades)"
    },
    "cross_attention_text_conditioning": {
      "used": "Transformer2D cross-attention to the CLIP text embedding (how the UNet takes the prompt)",
      "jmlx": "implemented (CrossAttention, Phase 7.2)"
    },
    "geglu_feed_forward": {
      "used": "Transformer2D FF: linear to 2x, chunk, x * exact_gelu(y) per diffusers activations.py GEGLU (F.gelu, approximate='none' \u2014 NOT QuickGELU), project",
      "jmlx": "composable from Linear + chunk + GELU (GELU.java exists) + multiply; no new primitive"
    },
    "transformer2d_wrapper": {
      "used": "GroupNorm + 1x1 conv proj_in around the attention block, 1x1 conv proj_out + residual (use_linear_projection=false)",
      "jmlx": "implemented (GroupNorm + MLXConv 1x1 convs)"
    },
    "linear_time_embedding": {
      "used": "sinusoids -> MLP -> add to res blocks",
      "jmlx": "implemented (Linear; sin/cos basic ops)"
    },
    "residual_skip": {
      "used": "1x1 conv identity skips + concat skips",
      "jmlx": "implemented"
    },
    "nearest_upsample_plus_conv": {
      "used": "Upsample2D: nearest interpolate then conv, inside the net",
      "jmlx": "implemented (Upsample nearest + MLXConv, Phase 7.1)"
    }
  },
  "vae_primitive_map": {
    "mid_single_head_attention": {
      "used": "VAE decoder mid block: GroupNorm(32), qkv projections, heads=1 attention, proj_out + residual \u2014 NO feed-forward (diffusers AttentionBlock)",
      "jmlx": "implemented (attention facades; single-head is a head-count, not a new primitive)"
    },
    "nearest_upsample_plus_conv": {
      "used": "all three decode stages (SD's VAE has NO ConvTranspose2d \u2014 nearest interpolate + conv)",
      "jmlx": "implemented (Upsample nearest + MLXConv)"
    },
    "latent_scaling_constant": {
      "used": "latents / 0.18215 before decode",
      "jmlx": "pure-Java constant, no native surface needed"
    }
  },
  "checkpoint_layout_note": {
    "model_index": "scheduler: diffusers PNDMScheduler; text_encoder: transformers CLIPTextModel; tokenizer: transformers CLIPTokenizer (verified from model_index.json at artifact sha 451f4fe)",
    "scheduler_pinning": "PNDM is the artifact's default \u2014 the 9.6 Tier-B reference must pin the artifact's default scheduler for the per-step noise-prediction comparison to be meaningful; its pinned scheduler/scheduler_config.json has skip_prk_steps: true (the Runge-Kutta warm-up never runs \u2014 only the PLMS path with its lower-order first steps) plus steps_offset: 1, set_alpha_to_one: false, beta_schedule 'scaled_linear', beta_start 0.00085, beta_end 0.012, which the 9.6 scheduler must port (steps_offset and set_alpha_to_one change the timestep grid and the final alpha)",
    "vae_attention_keys": "VAE-ONLY: the pinned VAE stores its mid-block AttentionBlock attention under the LEGACY names query/key/value/proj_attn, which diffusers remaps to to_q/to_k/to_v/to_out.0 at load time \u2014 the 9.6 loader must implement that remap (the UNet's attention already uses the current to_q/to_k/to_v/to_out.0 names: 0 of its 687 keys end in .query.weight at sha 451f4fe)",
    "vae_attention_weight_shapes": "stored as 2-D Linear weights ([512, 512] in the mid block), math-equivalent to the 1x1 convs the architecture implies \u2014 a loader expectation, not a primitive gap"
  },
  "note_on_convtranspose": "an earlier draft of this probe used ConvTranspose2d; SD 1.5's actual VAE decoder and UNet Upsample2D use nearest interpolation plus conv, which the probe now mirrors \u2014 convTranspose remains available in MLXConv either way",
  "scheduler_gap": {
    "finding": "no scheduler surface exists anywhere in jmlx; the artifact's default is PNDM \u2014 with its pinned skip_prk_steps: true the Runge-Kutta warm-up never runs, so the port needs only the PLMS path with its lower-order first steps \u2014 with DDPM/DDIM/Euler the common alternatives; all are small pure-Java state machines (alphas_cumprod table + step update + sigma conversion); PNDM's multistep history makes it a real state machine, not a one-line step update",
    "native_dependency": "none"
  },
  "text_encoder_gap": {
    "finding": "SD 1.5 conditions on the CLIP ViT-L/14 TEXT encoder's last_hidden_state (77x768 after final_layer_norm), via CLIPTextModel \u2014 no pooling and no text_projection (those belong to ...WithProjection, used by contrastive image-text models, not SD)",
    "architecture": "CLIP text is pre-LN (LayerNorm before attention and MLP) with a CAUSAL mask (_create_4d_causal_attention_mask in modeling_clip.py) and QuickGELU activation \u2014 BERT's post-LN bidirectional EncoderBlock is NOT a drop-in; DecoderAttention-style causal masking over an encoder stack, or a causal option on an encoder block, is new wiring to verify at milestone start",
    "native_dependency": "none beyond existing attention facades"
  },
  "tokenizer_gap": {
    "clip": "the pinned SD artifact ships NO tokenizer.json \u2014 only tokenizer/{vocab.json, merges.txt, special_tokens_map.json, tokenizer_config.json} (slow-tokenizer files, verified against the HF API at sha 451f4fe). Two paths: (a) a slow-tokenizer loader in jmlx-tokenizer over vocab.json+merges.txt (BpeMerger already exists; BOS <|startoftext|> 49406 / EOS <|endoftext|> 49407 and 77-token pad/truncate are small wrapper logic) \u2014 the record's recommendation, since the artifact itself ships these files; or (b) pin openai/clip-vit-large-patch14's tokenizer.json as a second artifact and add Split.invert=true support (rejected today in BOTH TokenizerJsonLoader and PreTokenizerPipeline) plus a RobertaProcessing post-processor, with an equivalence check against the artifact's own files. NOTE: the clip-vit-large-patch14 tokenizer.json is NOT loadable by jmlx-tokenizer as-is \u2014 its Split pre-tokenizer step has invert=true, which the loader rejects",
    "whisper": "whisper's tokenizer.json (no normalizer, ByteLevel pre-tokenizer, BPE, TemplateProcessing) fits jmlx-tokenizer's existing components; the generation-side controls (forced decoder prefix with language/task/notimestamps, suppress_tokens, begin_suppress_tokens, 448-position cap) are model-level, recorded in the audio probe's loader_details"
  },
  "fft_relevance": "none: UNet/VAE/CLIP-text exercise no FFT family member; conv+attention+upsample cover the whole pipeline",
  "reference_model_candidates": {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": {
      "license": "CreativeML OpenRAIL-M (not OSI; use restrictions)",
      "revision": "pin at milestone WP0 (runwayml/stable-diffusion-v1-5 now 307-redirects to this canonical id)",
      "size_bytes_fp32_safetensors": {
        "unet": 3438167540,
        "vae": 334643276,
        "text_encoder": 492265874,
        "total": 4265076690
      },
      "notes": "canonical SD 1.5; license differs from every existing Tier-B pin (MIT/Apache-2.0)"
    },
    "tiny_synthetic_checkpoint": {
      "license": "generated in-repo (Tier-A path, as 7.3a-7.3c)",
      "notes": "strict golden without any external artifact; the binding gate"
    }
  }
}
```

## 7. Analysis and recommendation

**Audio — go, scheduled as Phase 11 milestone 11.5** (new milestone after
11.4, riding 11.2's `MLXFFT`): build FFT once in its own phase with its own
differential coverage, then audio rides it. Pulling `mlx_fft_*` forward into a
Phase 7-style milestone would split FFT ownership and re-test effort;
Whisper-tiny is not urgent enough to justify that (nothing in Phases 7–10
depends on it). This is a sequencing choice, not a condition. If the review
wants audio earlier, the alternative is option (a) of §3 — a dated amendment
pulling `MLXFFT` into a pre-Phase-11 audio milestone — with the stated cost.
**Scope:** single fixed 30 s window transcription, the only shape Whisper's
architecture takes. Chunked long-form transcription (windowing with
conditioning on previous text) is **out of scope** for 11.5 — a possible
follow-up milestone, recorded here so the boundary is explicit.
**Publishing:** `jmlx-audio` is published (eighth module; §3) — the
release-order and release-smoke invariants grow with it.

**Diffusion — go, scheduled as Phase 9 milestone 9.6** (new milestone after
9.5): every native primitive is already implemented and verified, the remaining
surface is pure-Java, and it gives Phase 9's inference product a second
modality without touching the Phase 8 release-critical path. Decisions made
here: (1) the Tier-B license is **accepted** as described in §5 — the milestone
plan re-records the acceptance but does not re-open it; (2) scope — text-to-
image only, SD 1.5-class, no ControlNet/img2img/inpainting in the first
milestone; (3) module home — the pipeline lands in `jmlx-models` (the pattern
for every inference family so far); a new published module is *not* created.
The one genuinely unverified wiring risk — causal masking for the CLIP text
stack — is named in §3 as a milestone-start verification, not hidden.

**Alternative considered for both: dated deferral** (both modalities deferred
until after Phase 9, review again at the Phase 10 boundary). Reasonable if the
review judges release focus paramount; the cost is that 7.1's conv/upsample
investment stays text/vision-only for several more phases.

## 8. Draft roadmap amendment (applies on "go")

Dated amendment to `req/full-roadmap.md` and `req/plans/phase7-plan.md`,
effective on this record's merge (applied in the same PR):

- Phase 9 gains milestone **9.6 — Diffusion inference** (module home
  `jmlx-models`): pure-Java schedulers (**PNDM first** — the artifact's
  default — then DDPM/DDIM/Euler) and sampler, VAE decoder wrapper with the
  legacy attention-key remap, CLIP text encoder with causal masking, UNet
  assembly from 7.1/7.2 facades; a CLIP slow-tokenizer loader (vocab.json +
  merges.txt — the artifact ships no `tokenizer.json`) with the pinned
  BasicTokenizer normalization port, in `jmlx-tokenizer`. Tier-B:
  `stable-diffusion-v1-5/stable-diffusion-v1-5` under CreativeML OpenRAIL-M
  (accepted in this record, §5); the reference generator pins the artifact's
  default scheduler and adds diffusers to the hf-reference hash lock
  (checked against `verifyImageOracleDependencyPins`); Tier-A synthetic
  golden is the binding gate.
  Exit gate: the synthetic golden passes at strict tolerance; the pinned
  Tier-B artifact's prompt embedding, per-step noise prediction under the
  pinned PNDM schedule and decoded image match the CPU float32 reference
  within the recorded ε.
- Phase 11 gains milestone **11.5 — Audio inference**, in three parts per §3:
  a new **`jmlx-audio` module (Java 21, native-free)** holding WAV decode,
  resampling, window functions, the Slaney filterbank and log scaling; a
  forward-STFT facade in `jmlx-core` built with milestone 2's `MLXFFT`; and
  the Whisper model in `jmlx-models` (`api`-depending on `jmlx-audio`, the
  vision pattern) with the §3 generation controls. **`jmlx-audio` is
  published** — the project's eighth published module: `verifyReleaseScriptsMatch`,
  its own `release.sh`, release order before `jmlx-models` (alongside
  `jmlx-vision`), and the release smoke all grow accordingly; Phase 11's
  release line names `jmlx-audio`'s first release, like Phase 9's line names
  `jmlx-server`'s (applied in the same PR). Tier-B:
  `openai/whisper-tiny` (apache-2.0). Exit gate: log-mel fixtures within a
  recorded tolerance against the pinned torch-STFT oracle path (§3), a
  synthetic Tier-A logits golden at strict tolerance, and exact greedy
  transcript IDs on the pinned Tier-B artifact. Scope: single 30 s window;
  chunked long-form transcription is out of scope.
- Neither extends Phase 7; 7.5 follows 7.4 directly. In the roadmap, the
  "Phase 11 done" line is updated so 11.5 is gated at phase level; the
  "Phase 9 done" line already gates every milestone's exit gate, so it needs
  no change.

## 9. Gate

The decision is reviewed in this record's PR and the roadmap is updated to
match the merged outcome (status rows in `req/full-roadmap.md` and
`req/plans/phase7-plan.md`, modality statement in `req/phase6-compatibility.md`,
and the dated amendment above on "go"). The probe digests are enforced by
`verifyHfReferenceGoldens`, so the record's evidence cannot silently drift
from the scripts that produced it.
