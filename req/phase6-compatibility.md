# Phase 6 compatibility matrix

This is the human-readable source of truth for local text inference support. “Implemented” means
the capability is present in a released API only after the Phase 6 gate says so; a synthetic test is
not evidence that an arbitrary Hugging Face artifact will load.

| Architecture | Status | Verification | Tokenizer / chat template | Checkpoint | Quantization | RoPE / context/cache policy | License / access notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama pre-norm decoder | implemented | verified-with-real-artifact on recorded Tier-B pin; scaled RoPE verified with synthetic fixtures | ByteLevel BPE; directory metadata/templates and tokenizer-backed text streaming | float safetensors, including index shards | float + MLX affine 4-bit (synthetic verification) | base, linear, dynamic NTK, Llama 3, YaRN RoPE; FULL cache, optionally capacity-bounded | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Qwen2 pre-norm GQA decoder | implemented | verified-with-real-artifact on recorded Tier-B pin; scaled RoPE verified with synthetic fixtures | ByteLevel BPE; directory metadata/templates and tokenizer-backed text streaming | float safetensors, including index shards | float + MLX affine 4-bit (synthetic verification) | base, linear, dynamic NTK, Llama 3, YaRN RoPE; FULL cache, optionally capacity-bounded; `use_sliding_window=true` rejected | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Qwen3 pre-norm GQA decoder (per-head QK norm) | implemented | verified-with-real-artifact on recorded Tier-B pin; strict float32 HF reference on the synthetic fixture | ByteLevel BPE; real Qwen3 chat template (live-verified SHA-256) embedded in the synthetic family bundle | float safetensors (0.6B tied embeddings, 8B untied) | float + MLX affine 4-bit (synthetic verification; local 8B 4-bit run) | explicit head dimension, base RoPE; FULL cache, optionally capacity-bounded; `use_sliding_window=true` rejected | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Mistral | implemented | verified-with-real-artifact on recorded Tier-B pin; synthetic Hugging Face tiny-checkpoint logits and window references | Real-artifact chat text/IDs and synthetic family tokenizer golden | float safetensors, including index shards | float + MLX affine 4-bit (synthetic verification) | FULL cache by default; explicit sliding eviction only when checkpoint `sliding_window` is non-null | Public Apache-2.0 artifact; exact run in Tier-B manifest |
| Gemma v1 | implemented | verified-with-synthetic-fixture against Hugging Face tiny-checkpoint logits | Synthetic family tokenizer/template; text/ID golden | float safetensors, including index shards | float + MLX affine 4-bit (synthetic verification) | explicit head dimension and base RoPE; FULL cache, optionally capacity-bounded | Tier-B terms acceptance pending |
| Phi-3 | implemented | verified-with-synthetic-fixture against Hugging Face tiny-checkpoint logits | Synthetic family tokenizer/template; text/ID golden | float safetensors with fused QKV and gate/up projections | float + MLX affine 4-bit (synthetic verification) | FULL cache by default; sliding eviction requires a checkpoint window; `longrope` rejected | Public MIT artifact; hosted-runner RAM limits Tier-B execution |
| Mixtral / MoE | implemented | verified-with-synthetic-fixture against Hugging Face tiny-checkpoint logits | Synthetic family tokenizer with Mistral template; text/ID golden | float safetensors with expert tensors | float weights only | FULL cache by default; sliding eviction requires a checkpoint window; gathered top-k expert compute | Tier-B artifact/runner pending |
| Gemma 2/3, Phi-2, LongRoPE, shared-expert MoE | planned | no verification | — | — | — | unsupported capability named during config parsing | — |

The supported runtime is macOS on Apple Silicon with Java 25 and the MLX pins in
`scripts/bootstrap-native.sh`. Artifact licensing and access requirements are recorded alongside
each future Tier-B fixture; this matrix intentionally makes no claim that every public Hugging Face
checkpoint is compatible.

The independent pure-Java tokenizer module additionally verifies synthetic/reference fixtures for
Metaspace BPE, Metaspace Unigram, and Bert/WordPiece. Phase 6.3 family tokenizer bundles are synthetic
and establish component and chat-rendering behavior, not byte-level parity with a production tokenizer.
SentencePiece `Precompiled` normalizers decode the real Darts charsmap (Phase 7.2); malformed
binary tables are rejected at load with a named `Precompiled: malformed ...` error. The [2026-09-29 macOS native CI run](https://github.com/Alipsa/jmlx/actions/runs/36627571854)
executed the Phase 6.3 attention, MoE, Llama/Qwen, Mistral, Gemma, Phi-3 and Mixtral suites;
the workflow checks their JUnit XML to prevent a silent native skip. The
[Llama](https://github.com/Alipsa/jmlx/actions/runs/36640308134),
[Qwen](https://github.com/Alipsa/jmlx/actions/runs/36640308134), and
[Mistral](https://github.com/Alipsa/jmlx/actions/runs/36640308134) pinned real-artifact jobs
asserted 16 greedy IDs on the recorded runner pin; the other rows retain
synthetic-fixture status.

MLX affine quantization (`quantization: {group_size, bits}`, tensors as `.weight`/`.scales`/`.biases`)
loads for Llama, Qwen2, Qwen3, Mistral, Gemma v1 and Phi-3, with the packed embedding table also
serving as the tied output head. Verification is **synthetic plus one local real artifact**:
`QuantizedDecoderTest`
compares each family's quantized logits against the same weights dequantized to float, and
`mlx-community/Llama-3.2-1B-Instruct-4bit` (not a pinned Tier-B artifact) produced the same greedy
token IDs as an independent MLX-Python reference on three prompts (24, 8 and 16 tokens). The
`mlx-community/Qwen3-8B-4bit` local run (throughput and memory, not Tier-B evidence) is recorded in
`req/phase6-4-benchmark.md`. Mixtral
quantization, per-layer overrides, non-affine modes, GPTQ/AWQ and GGUF are rejected with the key
named. Quantized KV cache remains unsupported.

## Phase 7.2 text tasks (2026-10-05)

| Task/family | Tier-A | Tier-B | Limits |
| --- | --- | --- | --- |
| Text encoder, BERT | Tiny HF encoder hidden states including padded rows; CLS/mean/max/L2 | all-MiniLM-L6-v2, actual sentence-transformers CPU vector | Absolute positions, GELU, unquantized; one supported Pooling plus optional Normalize |
| Sequence classification, BERT | Tiny HF sequence logits and labels | Licensed bert-base-uncased-sst2 fallback; stable-margin exact class | Regression rejected; pair API separately oracle-verified |
| Token classification, BERT | Tiny HF token logits for every row | No real task-head claim | Softmax/sigmoid selected by problem_type; no CRF |
| Seq2seq text, T5/Flan-T5 | Tied ReLU/untied gated-GELU; encoder, full/cached logits, buckets, generation | flan-t5-small complete greedy IDs, EOS and every CPU logit history | FULL cache only; no scheduler, beam search, decoder prefixes or quantized weights |

All three new real-artifact checks passed locally on M5 Max/macOS 27.0.1 with TF32 disabled.
Their first supported macOS 26 CI executions remain pending. See `phase7-2-implementation-report.md`
for pins, margins and measured resource use; synthetic head checks do not imply other model families.

## Phase 7.3a image+text task (2026-10-13)

| Task/family | Tier-A | Tier-B | Limits |
| --- | --- | --- | --- |
| Image+text → text, SmolVLM / Idefics3 | Byte-exact Pillow image oracle fixtures (decode, resize, full preprocess chain); tiny Idefics3 checkpoint with strict 1e-4 CPU reference logits for patch/positioned embeddings, tower, connector, merged embeddings, prefill/cached/full-history steps and exact greedy IDs | [SmolVLM-256M-Instruct pin](phase6-tier-b-artifacts.md#phase-73a-wp7-imagetext-candidate-2026-10-11): 16 original / 347 expanded prompt IDs, exact greedy IDs `[378, 2443, 21559, 354]`, full logits at prefill and three cached histories within 1e-4 of CPU float32 eager | float safetensors only (storage FP16/BF16 up-cast; quantized and per-layer-overridden vision checkpoints rejected); per request: ≤16 images, ≤67,108,864 decoded pixels and ≤256 tiles in total, plus the context-length budget; FULL and capacity-bounded KV retention, sliding image-prefill retention rejected; no VLM batching or scheduler cohorts |

The row above is evidence-backed but its real-artifact check is **local eligibility evidence**: the
recorded runs used M5 Max/macOS 27.0.1 with TF32 disabled. The weekly Tier-B row targets the
self-hosted `self-macos-xl` runner (GitHub's larger `macos-26-xlarge` tier is billed per minute
even in public repos), which is the same M5 Max host (macOS 27.0.1) that produced the local
evidence; its first executions with persistent model storage ran green on 2026-10-26 (runs
40403387275 and 40409087523, evidence in `tools/tier-b/README.md`), accepting the CI
pipeline, not independent host, OS or memory-fit evidence, and the first supported macOS 26
execution remains pending. See
`phase7-3a-implementation-report.md` for pins, margins and measured resource use; the synthetic
checks do not imply other vision families.

## Phase 7.3b Qwen3-VL image+text (2026-10-26)

| Task/family | Tier-A | Tier-B | Limits |
| --- | --- | --- | --- |
| Image+text → text, Qwen3-VL (dense) | WP8 tiny full-model checkpoint (three decoder layers, two DeepStack levels, `mrope_section [16, 8, 8]`, 21-token tied vocabulary, position-table interpolation beyond the learned 8x8 table): expansion, grids, processed-pixel hashes, three-axis positions/delta, tower merged/DeepStack rows, and prefill plus every cached and forced-marker step's logits against a fresh full uncached CPU float32 forward — strict `1e-4` in `Qwen3VlGoldenTest` (measured maximum `2.3245811e-6`) and default TF32 bound `0.0025` in `Qwen3VlDefaultPrecisionTest` (measured `0.0019590855`); shared EOS/stop/sampling/streaming/abort contracts verified on the composed model against the WP6 checkpoint (WP7) | Pinned locally (2026-10-26): `tools/tier-b/qwen3vl-2b-instruct.json` — the required multi-image prompt (152 expanded tokens, delta −110), byte-exact preprocessing, exact greedy IDs under the recorded stability gate (minimum top-two gap 0.1028, ε 2e-4 / margin 1e-4 against a measured maximum error 1.0228e-4), measured allocator peak 9.83 GB, sampled footprint peak 11.98 GB, load ~843 ms, generation ~247 ms, 4.27 GB artifact disk. CI: first self-hosted run with persistent model storage recorded 2026-10-26 (run 40403387275, first all-arms green run with persistent model storage, merged-head confirmation 40409087523, both `device=gpu`; max logit error, native allocator peak, greedy IDs and decoded output identical to the local evidence, wall times differ — `tools/tier-b/README.md`): a `tier-b.yml` row exists on the self-hosted `self-macos-xl` runner (the standard 7 GB runner cannot hold the measured footprint, and GitHub's larger `macos-26-xlarge` tier is billed per minute even in public repos); the runner is the same M5 Max host (macOS 27.0.1) as the local evidence, so its first run accepted the CI pipeline, not independent host, OS or memory-fit evidence, and the macOS 26 hosted execution remains pending | float safetensors (storage BF16/FP16 up-cast) plus MLX affine 4-bit text projections (synthetic verification; the manual quantized 8B 4-bit run is recorded in `req/phase6-4-benchmark.md`, evidence-only); per request: FULL or capacity-bounded FULL retention, sliding rejected; single request per thread, no scheduler cohorts; `qwen3_vl_moe` rejected by name |

The Tier-A row is **local acceptance evidence**: the recorded runs used M5 Max/macOS 27.0.1 (MLX
0.31.2, mlx-c fba4470); PR #61's macOS 26 CI run 321 executed both required suites with no skips or
failures, but both modes measured the identical `2.3245811e-6` there, so TF32 was not active on
that runner and the `0.0025` default-mode bound is supported only by the local M5 Max measurement.
The exact greedy-ID assertion is degenerate on the 21-token checkpoint (recorded in the golden's
provenance); the full logit-vector comparison is the binding gate. See the reference findings §11.8
for pins, the measured maxima (`req/phase6-golden-precision.md`) and the shared-contract coverage
decision; the synthetic checks do not imply Tier-B support for the pinned 2B/8B artifacts.

## Phase 7.3c Phi-3.5-Vision image+text (2026-10-30)

| Task/family | Tier-A | Tier-B | Limits |
| --- | --- | --- | --- |
| Image+text → text, Phi-3.5-Vision (LLaVA-style, dense) | WP6 tiny synthetic `Phi3VModel` checkpoint (`goldens/checkpoints/phi3v-wp6/`, 76 BF16 tensors, single shared embedding): byte-exact HD-transform + normalized bicubic preprocessing, per-image and expanded prompt IDs with the sentinel marker runs, the CLIP tower's patch/positioned embeddings, the reference-exact hd-transform rows, and prefill plus every cached step's fresh full uncached CPU float32 logits — strict `1e-4` in `Phi3VGoldenTest` (measured maximum `1.5497208e-6`) and default TF32 bound `0.0012` in `Phi3VDefaultPrecisionTest` (measured `9.266138e-4`); per-image evaluate-and-release peak, cancellation boundaries, capacity/sliding rejection and the 4096-boundary re-prefill verified on the tiny checkpoint (WP7) | Pinned locally (2026-10-30): `tools/tier-b/phi3v-4b-instruct.json` (MIT, 8,295,395,355 bytes, 15 files) — the required multi-image prompt (a 672x336 and a 336x336 image, 1241 expanded tokens, both under the 4096 original max so no NTK-Su re-prefill fires), byte-exact preprocessing, 24 exact greedy IDs under the recorded stability gate (minimum top-two gap 0.003166, ε 2e-4 / margin 1e-4 against a measured maximum error `1.0299683e-4`), measured allocator peak 31.1 GB, sampled footprint peak 51.4 GB, load ~1133 ms, generation ~5423 ms. CI: a `tier-b.yml` row exists on the self-hosted `self-macos-xl` runner (the standard 7 GB runner cannot hold the measured footprint, and GitHub's larger `macos-26-xlarge` tier is billed per minute even in public repos); its first self-hosted run accepts the CI pipeline, not independent host, OS or memory-fit evidence, and the macOS 26 hosted execution remains pending | float safetensors (storage BF16/FP16 up-cast; quantized keys rejected by name); per request ≤16 images / ≤80 sentinel slots and the context-length budget; NTK-Su RoPE (48-factor short/long tables) with the 4096-boundary recompute that selects the long table; FULL and capacity-bounded KV retention, sliding image-prefill retention rejected; no VLM batching or scheduler cohorts |

The row above is **local acceptance evidence**: the recorded runs used M5 Max/macOS 27.0.1 (MLX
0.31.2, mlx-c fba4470) with TF32 disabled for the allocator/footprint and the strict Tier-A bound.
`Phi3VGoldenTest` and `Phi3VDefaultPrecisionTest` are required by the macOS CI native-suite
assertion; the Tier-B row's first self-hosted execution remains pending, and the macOS 26 hosted
execution of all four WP6/WP7/WP8 suites remains pending. See
`phase7-3c-reference-findings.md` §16 for the WP8 acceptance evidence and
`phase6-tier-b-artifacts.md` for the pin and the reference-generator contract.

## Phase 7.4 audio/diffusion decision (2026-10-31)

Phase 7 ships **no audio or diffusion support**; this matrix makes no claim for either
modality. The 7.4 decision record (`req/plans/phase7-4-decision.md`, reviewed via its PR)
schedules both by dated roadmap amendment: **audio inference** (`openai/whisper-tiny`, the
license the artifact declares: apache-2.0) as Phase 11 milestone 11.5 — a Java 21
native-free `jmlx-audio` module (WAV decode, Slaney filterbank), a forward-STFT facade in
`jmlx-core` on 11.2's `MLXFFT` (no ISTFT exists in Whisper), and the Whisper model in
`jmlx-models`; `jmlx-audio` joins the published set, and the release-script, release-order
and release-smoke invariants grow with it — and
**diffusion inference** (SD 1.5-class text-to-image in `jmlx-models`, new surface entirely
pure-Java: schedulers, VAE, CLIP text encoder; a CLIP slow-tokenizer loader for the
artifact's own vocab.json + merges.txt in `jmlx-tokenizer` — the artifact ships no
`tokenizer.json`) as Phase 9 milestone 9.6. The diffusion Tier-B artifact
(`stable-diffusion-v1-5/stable-diffusion-v1-5`, CreativeML OpenRAIL-M) is accepted in the
decision record for a pinned read-only test artifact. Until those milestones deliver, audio
and diffusion remain unsupported and unverified.

## Phase 7.6 hybrid Qwen3.5 image+text (2026-11-03)

| Task/family | Tier-A | Tier-B | Limits |
| --- | --- | --- | --- |
| Image+text → text, Qwen3.5 / Qwen3.8 (hybrid Gated DeltaNet linear attention + gated full attention) | Four-layer hybrid checkpoint (three Gated DeltaNet layers + one gated full-attention layer, partial interleaved MRoPE, tiny no-DeepStack tower): text prefill/cached/fresh-forward logits and retained state, all five storage layouts (HF float, mlx-vlm float/affine, mlx-lm float/affine, plus tied/equal-head and eight-layer variants), and vision expansion/grids/pixels/positions/tower rows with prefill+cached steps — strict `1e-4` in `Qwen35TextGoldenTest`/`Qwen35PackedGoldenTest`/`Qwen35VisionGoldenTest` (full float32; packed text measured maximum Java/HF `7.1525574e-7`) and the separately named default-TF32 bound `0.005` in `Qwen35DefaultPrecisionTest` (measured `0.0014960468`); generation contracts, the public-decode evaluation counter and exact memory ownership in the ordinary-mode suites | `tools/tier-b/qwen35-08b.json` (Qwen/Qwen3.5-0.8B, Apache-2.0, revision `2fc0636…`): text and image cases, 8 greedy tokens each (EOS stopping disabled), exact greedy IDs and decoded output, every prefill/cached logit vector under predeclared ε = 0.001 / margin 1e-4 (measured maxima 8.321e-5 text / 4.034e-4 image), native peak 5.12 GB and sampled footprint 6.65 GB under predeclared 8 GB/12 GB limits — automated `tier-b.yml` arm on `self-macos-xl`. The 27B affine artifact (`tools/tier-b/qwen38-27b-4bit.json`, mlx-vlm 0.6.8 converter) is a resource-qualified manual run, not a hosted-runner claim: measured maxima 0.0447 text / 0.0997 image under ε = 0.1006 (0.001 arithmetic + the measured 0.0996 stock normalization contribution), exact IDs and decoded output, native peak 18.0 GiB / footprint 21.1 GiB | dense `qwen3_5` only (MTP execution, MoE, video, speculative decoding and sliding retention rejected); float weights up-cast to FLOAT32, uniform affine 4-bit group-64 packed projections (conv/norm/recurrent state never quantized); TEXT capability from a text-only inventory (mlx-lm) that rejects images, a partial tower rejected; FULL and capacity-bounded KV retention; the batch scheduler rejects the hybrid model at startup (it is a `TextGenerationModel`, not a `DecoderModel`); the shared Qwen3-VL processor caps (≤16 images, 4096 raw-row budget, effective pixel cap) |

The row above is **local acceptance evidence**: the recorded runs used M5 Max / macOS 27.2
(MLX 0.32.4, mlx-c a26f2d5) with TF32 disabled. All seven new model suites and the five new
core suites are required by the macOS CI native-suite assertion; hosted CI has not run this
branch. The `tier-b.yml` qwen35 arm targets the self-hosted `self-macos-xl` runner (the
measured 6.65 GB footprint sits within ~400 MB of the standard runner's entire 7 GB once the
OS, test JVM and Gradle daemon share it; GitHub's larger `macos-26-xlarge` tier is billed per
minute even in public repos); its first execution accepts the CI pipeline, not independent
host, OS or memory-fit evidence, and the supported macOS 26 hosted execution remains pending.
See `req/plans/phase7-6-plan.md`, `req/plans/phase7-6-worklog.md` and
`req/phase6-golden-precision.md` for the full evidence; the synthetic checks do not imply
other hybrid families.
