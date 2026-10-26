# Phase 7.3a implementation evidence

## WP6 — local synthetic implementation, 2026-10-11

WP1–WP5 were delivered in PRs #41–#44. Branch `phase7-3a-wp6` already contained the tiny
Idefics3 checkpoint/numerics, tower/connector and shared decoder generation-loop extraction
(commits f2644b1, d38f2d8, 9713a65, 3b735c4 and 6fc5b8c). WP6 completes strict checkpoint
assembly and `SmolVlmModel` generation through `TextGenerationModels.load`.

The family descriptor rejects unknown computation fields and quantized/per-layer overrides,
validates nested Llama and vision dimensions, complete outer tensor names and loaded shapes,
image IDs and tied aliases. The pinned real config is preserved with SHA verification in a pure
Java test. Storage FP16/BF16 weights are evaluated/cast to FLOAT32; patch weights are explicitly
transposed OIHW to OHWI. No native pins, generated bindings or goldens were changed by this
completion work. The earlier branch commits generated the tiny fixtures with reviewed provenance.

Generation checks image/pixel/tile/context and cache budgets before pixel/native work. Geometry
and expansion also run for zero-token requests, which skip native generation scope allocation.
Features are evaluated/hoisted out of shorter vision children, merged using one take over
concatenated token/feature rows, and released with the prefill child after logits and retained KV
are evaluated. Cached decode uses token lookup only. Full and bounded-full retention are supported;
sliding image-prefill retention is explicitly rejected pending equivalence evidence. Limits are
16 images, 67,108,864 total decoded pixels, 256 tiles and the configured context length.

Local native execution used macOS 27.0.1 on Apple Silicon with the existing staged MLX 0.31.2 /
mlx-c fba4470 pins. Native tests executed without skips:

- `SmolVlmGoldenTest`: strict absolute 1e-4 CPU reference comparisons for all five cases, patch/
  positioned embeddings, tower, connector, merged embeddings, prefill/cached scores, three
  cached/full-history steps, Java-preprocessor prefill and exact five-token greedy IDs.
- `SmolVlmGenerationTest`: loader dispatch, zero/one/many tokens, capacity equality/one-over,
  original/expanded accounting, actual preprocessing/tower/connector counts, request isolation,
  EOS/stop, cancellation before vision/between steps, sampled streaming/log probabilities,
  listener/tokenizer/vision/evaluation failures, reuse after failures, FP16/BF16 weight up-casts,
  malformed patch rejection and feature-scope ownership. A zero-step request also succeeds with
  the native loading scope already closed, proving it does not touch native scope/weights.
- `SmolVlmDefaultPrecisionTest`: independent default TF32 bound, measured maximum logit error
  9.0524554e-4 versus CPU references through Java preprocessing; visual scope closes before decode.
- `Idefics3VisionTowerSmokeTest`: finite native forward and scope/shape contracts.

`SmolVlmDescriptorTest` and `Idefics3VisionMathTest` cover pure validation and permutation/position
math, including the real 512/16/4 geometry and the configured learned position grid for rectangular
validity. macOS CI now explicitly requires all four native suites in their matching precision tasks.

Both exact-bit comparisons passed against an archive of `origin/main` (`3a8a2f0`), on the same host
and native pins, in TF32-on and TF32-off modes. The recorder covers all 14 decoder variants, direct
and scheduler paths and non-null embedding hooks. Raw recordings live under
`jmlx-models/build/exact-bits/tf32-{1,0}`; they are local PR evidence, not portable goldens. The archive
used the workspace git directory solely for recorder provenance commands, so its git metadata
identifies the branch HEAD with a dirty tree; its recorded DecoderModel SHA identifies the actual
unmodified `origin/main` source. Verification compares float bits and IDs, with no tolerance.

## Local verification

The complete root `build` passed, including both model precision suites, formatting/checkstyle,
Java 21 bytecode, release-script equality, native call-site/inventory guards, HF golden digests
and image-oracle source/dependency pins. `tools/release-smoke/run.sh ci` passed all seven runtime,
source and Javadoc artifact/license checks, POM-versus-metadata resolution and seeded scheduler
consumer output. Executable image and tokenizer oracle environment/fixture verification passed.
MLX oracle verification passed its compatible Phase 7.1 profile; it explicitly skipped the
Phase 6 profile because this host is macOS 27, while that profile requires exactly macOS 26.
No Torch regeneration, Tier-B download, Maven publication or version bump was performed.

## Remaining acceptance

This host's successful native runs are local evidence. The plan's original macOS 26 GPU profile
and required new-suite CI gate remain pending until PR CI runs. WP7 local memory and real-artifact work is recorded below; its required macOS 26 Tier-B
execution/hosted fit remains pending. WP8 owns final seven-artifact consumer/capability
acceptance. The compatibility matrix gains no verified real image+text family claim from WP6.
No Maven publication or version bump is included.

## WP7 — memory, precision, Tier-B and CI gates, 2026-10-11

WP7 builds on merged WP6 (`9c59e23`, PR #45). Existing tiny Idefics3 intermediates, identical
oracle pixels, merged embeddings, prefill/cached/full-history logits and greedy goldens remain
unchanged. Their strict float32 bound stays `1e-4`; the independently measured TF32 bound stays
`0.002`. No further decoder/shared-generation extraction or T5 change was needed.

`VisionLanguageMemoryTest` uses evaluated arrays and `MLXMemory.activeBytes`, excluding cached
free storage. Twelve rounds across all five synthetic cases cover same/different images, changing
tile counts, text-only requests, full and bounded-full caches, exact/one-over capacity,
cancellation before vision/between decode steps/between images, listener and tokenizer failures,
injected vision/post-prefill failures and an actual native invalid-transpose error. Every request
returns to the warmed baseline; measured baseline/max were 847,360 bytes with late growth zero.
The post-prefill observer remains package-private and production uses the existing no-op observer.
A failing observer preserves cache poisoning before the enclosing request scope closes.

The lifetime test evaluates features after their vision activation children close, then evaluates
prefill logits and every retained KV array before closing prefill. Features become unusable with
the closed prefill scope; cached decode continues successfully. KV becomes unusable when the
request scope closes. The tiny decoder derives 512 FLOAT32 KV bytes per position (2 layers,
2 KV heads, head dimension 16, K/V) and 5,120 projected-feature bytes for 20 rows × hidden 64.
Prefill closure releases 723,122 bytes of total visual/activation storage in the isolated run;
retained KV measures 32,768 bytes for 46 positions (logical 23,552). Decode retention samples at
8/64 additional positions were 32,768/57,088 bytes, inside the derived KV budget with a fixed
16 KiB allocator envelope. Maximum observed retained-minus-logical overhead in the combined
suite was 11,264 bytes; the envelope does not grow with tokens or requests. The early/late slope
check spans 56 positions to distinguish growth from a plateau despite allocator rounding.
Only full/bounded-full policies are accepted; sliding image-prefill remains explicitly rejected,
so no sliding plateau or equivalence acceptance is claimed.

`TierBVisionLanguageTest` is selected by manifest task `image-text-generation`, independently of
model type, in the opt-in TF32-off JVM. Its manifest pins SmolVLM-256M's immutable revision,
all 13 checkpoint/config/tokenizer/template/model-card files, aggregate 517,903,046-byte cap,
licensed local RGB input and structured prompt. It verifies rendered chat, 16 original / 347
expanded IDs, full logits at prefill and three cached histories, original result IDs, effective
positions, finish reason and exact greedy IDs `[378, 2443, 21559, 354]`. Native features are
computed only at prefill, then the test closes the visual scope before the teacher-forced cached
steps. The highest all-vocabulary error is `5.2928924560546875e-5` versus CPU float32 eager,
below the independent `1e-4` bound. Gaps are `[1.51336098, 4.38764763, 0.92834282, 0.26448631]`;
every gap exceeds `2 * allowedError + margin = 0.0003`. The suite recomputes both top IDs and
gaps from the full logits. Actual public generation independently matches every greedy ID.

Measured native allocator peak was 5,737,416,464 bytes; in the same 2026-10-12 run the test
process's phys_footprint, sampled every 20 ms through libproc's `proc_pid_rusage(RUSAGE_INFO_V2)`,
peaked at 7,129,368,904 bytes while test-JVM RSS (`ps -o rss`) sampled 471,728 KiB. The RSS figure
counts only private resident pages and misses the MLX/Metal allocations the process holds -- it
sampled below the checkpoint's own 513 MB weight file -- so it is not useful memory evidence, and
no bound was ever asserted on either number. After review the suite now samples the process
footprint (which the hosted runner's RAM must actually hold) and gates both figures:
`MLXMemory.peakBytes() <= 6,261,704,464` (recorded peak plus 500 MiB, below one extra float32
weight copy) and `footprint <= 8 GiB` (about 20% above the measured peak). Because the 7.1 GB
measured footprint alone exceeds the standard `macos-26` arm64 hosted runner's 7 GB of RAM, the
Tier-B smolvlm row runs on a runner with more RAM (originally the billed `macos-26-xlarge` tier,
arm64 M2, 14 GB; since 2026-10-26 the self-hosted `self-macos-xl` runner, 128 GB — larger-runner
usage is billed per minute even in public repos); the bounds are a regression detector, not a
guarantee the runner survives a large regression: a regression that surfaces while the process
still fits in the RAM fails the suite explicitly, but the figures are read after the suite
finishes, so a large enough regression still OOM-kills the job before the assertion runs.
These measurements precede addition of the weekly/on-demand SmolVLM CI row. Hosted macOS 26
execution/fit remains pending. An earlier 17-tile square-input probe also passed, with maximum
error `8.58306884765625e-5` and native peak 5,739,185,936 bytes; the committed case uses the
five-tile wide opaque RGB fixture. Neither existing golden values nor strict tolerances were
changed to choose that case.

The generate-only `vision_tier_b.py` uses the existing hash-locked CPython 3.12/Pillow 12.3.0/
NumPy 2.5.3/Transformers 4.57.6/Torch 2.6.0 environment, raw PIL open/load and explicit slow
Idefics3 processing. It records pixels, sources, template/checkpoint hashes and full CPU scores.
New logits live in `goldens/smolvlm-tier-b-logits.json`. Additive provenance pins that golden
and the manifest/image/generator; `verifyHfReferenceGoldens` checks them without Torch execution.
Regeneration is manual and resets eligibility before remeasurement. Existing download containment,
size and SHA checks are reused without buildSrc changes. No ordinary test downloads an artifact.

CI requires the memory suite in ordinary `test` XML, alongside WP6's default-precision/generation
suites; strict `SmolVlmGoldenTest` stays required in `float32GoldenTest`. Ubuntu retains vision
Java 21 bytecode/style/preprocessing/tokenizer/oracle gates. Native and Tier-B workflows upload
JUnit evidence even on failure. The compatibility matrix receives no final verified VLM claim
until the pending host gates and WP8 consumer/capability work close.

Local verification passed root `build`, both exact-bit comparisons using the retained WP5/WP6
same-host baseline, all three memory tests with no skips, strict/default synthetic native suites,
opt-in pinned Tier-B, executable image/tokenizer oracles, supported Phase 7.1 MLX oracle and
seven-artifact release-smoke `ci`. Raw bit recordings remain under
`jmlx-models/build/exact-bits/tf32-{1,0}` as local PR evidence; baseline provenance caveats are
unchanged from WP6 above. The Phase 6 MLX oracle explicitly skips this macOS 27 host because it
requires macOS 26. No native pins, bindings, module versions or publication were changed.

## WP8 — final consumer evidence and capability documentation, 2026-10-13

WP8 builds on merged WP7 (PR #46, merge `0d648e1`). It changes no API, version, native pin,
generated binding, golden, manifest or tolerance; it re-runs the plan's required final checks on
the merged tree in fresh JVMs and finalizes the capability documentation. Host: Apple M5 Max,
macOS 27.0.1, MLX 0.31.2 / mlx-c `fba4470`, Java 25.

- Root `./gradlew build` green: every module `check`, `spotlessCheck`/checkstyle,
  `verifyBytecodeLevel`, `verifyReleaseScriptsMatch`, `verifyMlxApiCallSites`, the
  `req/mlx-api-inventory.md` freshness check, `verifyHfReferenceGoldens` and
  `verifyImageOracleDependencyPins`.
- Fresh full-suite native runs with no native-gate skips: `:jmlx-core:test` 46 suites / 405 tests
  (the single skip is the pre-existing `MoeMlpBenchmarkTest` benchmark probe, excluded by design)
  and `:jmlx-models:test` 47 suites / 337 tests, zero failures; fresh
  `:jmlx-core:float32GoldenTest` (19 tests) and `:jmlx-models:float32GoldenTest` (31 tests) green
  with TF32 disabled.
- Oracle verification under supported profiles: image oracle environment and all three committed
  fixtures (Python 3.12, Pillow 12.3.0, NumPy 2.5.3, Transformers 4.57.6, Darwin/arm64); tokenizer
  oracle environment and all four fixtures (tokenizers 0.23.2); MLX oracle on the Phase 7.1
  profile (Python 3.12.15, mlx 0.31.2) with its fixtures. The Phase 6 MLX profile is explicitly
  skipped on this macOS 27 host because it requires exactly macOS 26.
- Both exact-bit comparisons (TF32 on and off) are bit-identical to the retained WP6 same-host
  baseline (14 applicable variants, 126 captures per mode, recorded at `6fc5b8c`); only the git
  commit differs, as the verifier permits.
- Opt-in pinned Tier-B `smolvlm-256m-instruct` re-run: exact greedy IDs
  `[378, 2443, 21559, 354]`, maximum all-vocabulary logit error `5.2928924560546875e-5`
  (allowed `1e-4`), minimum top-two gap `0.26448631286621094` (required `> 0.0003`), native
  allocator peak 5,737,416,464 bytes and sampled process peak 7,146,752,688 bytes, under the
  suite's gates. Margin eligibility passed; the run is local macOS 27 evidence.
- Seven-artifact release-smoke `ci`: seven coordinates resolved from the disposable repository
  only, POM-only and module-metadata runtime classpaths equal (11 components), licenses in all
  seven runtime jars, the native jar payload, the three `jmlx-vision` jars' Pillow `NOTICE`
  content, seven Javadoc and seven sources jars, a fresh native extraction with no path override,
  and the seeded two-request batch equal to the committed golden.

Documentation finalized: the [compatibility matrix](phase6-compatibility.md) gains the Phase 7.3a
image+text row with the evidence and limits above; the [Tier-A fixtures](phase6-tier-a-fixtures.md)
and [Tier-B artifacts](phase6-tier-b-artifacts.md) documents record the vision fixtures and the
WP8 re-run status; the [inference report](phase6-inference-report.md) covers the seven-module
release chain and the 7.2/7.3a surface; the root README documents the SmolVLM image+text entry
point.

### Separation of evidence

- **Local evidence (this host):** everything above. It proves preprocessing, synthetic
  prefill/cached logits, real-artifact exact greedy output, unchanged text bits, request/result
  contracts, cleanup and the consumer smoke on the merged tree.
- **macOS 26 CI acceptance (pending):** the first supported hosted execution of the WP6/WP7 native
  suites in the new-CI required-suite assertions, and the first Tier-B SmolVLM row execution (the
  row now targets the self-hosted `self-macos-xl` runner; a GitHub-hosted execution would require
  the account's larger-runner billing to be configured — `tools/tier-b/README.md`). Until both
  land, the compatibility-matrix row is local eligibility evidence, never a hosted-verified claim.
- **Deferred capabilities:** sliding image-prefill KV retention (rejected pending equivalence
  evidence), VLM batching/multimodal scheduler cohorts, Qwen3-VL (Phase 7.3b), LLaVA-style models
  (Phase 7.3c or a dated deferral), and the Phase 6 rows with pending artifact pins (Gemma v1,
  Phi-3, Mixtral). No Maven publication or version bump is part of WP8.
