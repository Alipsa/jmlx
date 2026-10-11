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
and required new-suite CI gate remain pending until PR CI runs. WP7 still owns measured allocator
margins/leak and retention-slope tests, pinned SmolVLM-256M Tier-B dispatch, real-artifact stable
ID/logit-gap/peak-memory evidence and its CI row. WP8 owns final seven-artifact consumer/capability
acceptance. The compatibility matrix gains no verified real image+text family claim from WP6.
No Maven publication or version bump is included.
