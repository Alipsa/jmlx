# Agents instructions

This file provides guidance to AI agents when working with code in this repository.

## What this is

`jmlx` is a pure, idiomatic Java 25 framework for Apple Silicon GPU tensor operations, wrapping Apple's
native MLX (`mlx-c`) with zero-copy FFM (Project Panama) bindings. The **v0.1 vertical slice** described
in `req/initial-plan.md` (native bootstrap, generated bindings, memory management, a handful of tensor
ops) is done, and two further phases are built on top of it: `req/phase3-plan.md` (broadcast-compatible
ops, relaxed matmul, `slice`, a batched `eval`) and `req/phase4-plan.md` (`se.alipsa.jmlx.nn` -- `Module`,
`Linear`, `QuantizedLinear`, normalization/activation layers, `MultiHeadAttention` with RoPE and a KV
cache, and reverse-mode autograd via `MLXGrad`/`ModuleGrad`), both delivered. `req/plans/phase5-m1-plan.md`
(`MLXIO` -- safetensors/GGUF checkpoint I/O) and `req/plans/phase5-m2-plan.md` (`jmlx-tokenizer` --
byte-level BPE tokenizer + `hfjinja` chat-template rendering) are also delivered. `jmlx-jinja` (the
migrated former `hfjinja` project) is a dependency-free Java 21+ Hugging Face Jinja subset for
chat-template rendering. `req/project-outline.md` describes the full multi-phase vision (autograd,
`se.alipsa.jmlx.nn`, safetensors/tokenizers, and model loading). Phase 5 M3 is implemented as
`jmlx-models` (local Hugging Face safetensors loading and generation).

Phase 6 (`req/plans/phase6-plan.md`, complete local text-LLM inference and the first Maven Central
release) is built in milestones, each with its own plan under `req/plans/`:

- 6.0b: MLX API inventory, Python MLX oracle, and Tier-A fixtures.
- 6.1: seeded sampling, penalties, and top-k/top-p/min-p filtering.
- 6.2: component-based tokenizers (Metaspace BPE, Unigram, WordPiece) and prompt compatibility.
- 6.3: descriptor-driven Llama, Qwen2, Mistral, Gemma v1, Phi-3, and Mixtral decoders, with linear,
  dynamic NTK, Llama 3, and YaRN RoPE scaling.
- 6.4: explicit prefill/decode, `KVCachePolicy` (full, capacity-bounded, sliding), and benchmarks.
- 6.4.1: quantized KV retention, prototyped and stopped.
- 6.5: `BatchGenerationScheduler`, per-thread MLX streams, and release hardening.

`req/phase6-compatibility.md` is the source of truth for which families are verified and how
(real Tier-B artifact versus synthetic Tier-A fixture). `req/phase6-inference-report.md` summarizes
the Phase 6 state and lists the remaining release steps. Nothing has been published to Maven
Central yet.

Requires macOS on Apple Silicon, macOS 26+, and a Java 25 toolchain. The native bootstrap also needs
the Xcode Command Line Tools (`git`, `cc`, `otool`) and `cmake` (`brew install cmake`); see
`README.md`.

## One-time native bootstrap

The native runtime (mlx-c compiled against a pinned `mlx-metal` wheel) is not checked in and must be
staged before anything native will run:

```sh
./scripts/bootstrap-native.sh
```

This is idempotent and downloads/verifies (trust-on-first-use, pinned SHA-256) a pinned `mlx-metal`
wheel and jextract build, clones `mlx-c` at a pinned commit, builds it against the wheel's MLX, and
stages the result as a **flat** directory at `native/install/lib/` (`libmlxc.dylib`, `libmlx.dylib`,
`libjaccl.dylib`, `mlx.metallib`). Flatness is a runtime invariant, not incidental: MLX finds
`mlx.metallib` and dyld resolves `@rpath` siblings by colocation with `libmlx.dylib`/`libmlxc.dylib`;
splitting into `lib/`/`share/` breaks both.

The Java bindings in `jmlx-ffi/src/main/generated/java` are committed jextract output, not generated
on the fly. Only regenerate them if the pinned mlx-c commit changes:

```sh
./scripts/regen-bindings.sh
```

After running it, `git diff --exit-code jmlx-ffi/src/main/generated/java` must be clean if nothing
should have changed — this is the bindings-drift check.

```sh
# Required after binding or native-pin changes; root :check verifies it is current.
# Update req/mlx-api-inventory-overrides.json first when a binding gains an explicit status.
./gradlew generateMlxApiInventory

# Verifies every handwritten generated-binding use has an explicit non-unplanned mapping.
./gradlew verifyMlxApiCallSites

# Native-only: verifies the staged public mlx-c headers still match the committed bindings report.
# Regenerate with generateMlxApiHeaderCoverage after native pin or binding changes.
./gradlew verifyMlxApiHeaderCoverage
```

The Python MLX oracle is an explicit macOS/arm64-only fixture tool. It is not invoked by ordinary
Java tests. Install its CPython 3.12 environment from the hash-locked frontend/backend wheels, then
verify the recorded environment and canonical fixtures:

```sh
./tools/mlx-oracle/install.sh
./gradlew verifyMlxOracle verifyMlxOracleFixtures
```

Oracle commands automatically select compatible profiles: macOS 26 runs all families; newer hosts
run Phase 7.1 CPU fixtures and report Phase 6 GPU fixtures as skipped. Phase 6 requires macOS 26.

Only `./gradlew generateMlxOracleFixtures` rewrites the committed oracle JSON. Review that diff and
the recorded provenance whenever native pins change. Tier-A inputs and Java goldens are documented
in `req/phase6-tier-a-fixtures.md`; `req/phase6-tier-b-artifacts.md` is the opt-in real-artifact
manifest.

The Python Hugging Face `tokenizers` oracle (`tools/tokenizer-oracle/`) is the equivalent fixture
tool for `jmlx-tokenizer`: a hash-locked CPython 3.12 venv pinning `tokenizers==0.23.2`, used to
generate ground truth for `TokenizerOracleTest` and to verify committed fixtures under
`tools/tokenizer-oracle/fixtures/` (`*.tokenizer.json` inputs, `*.input.json` test-case specs,
`*.expected.json` canonical oracle output). Install and verify with:

```sh
./tools/tokenizer-oracle/install.sh
./gradlew verifyTokenizerOracle verifyTokenizerOracleFixtures
```

Only `./gradlew generateTokenizerOracleFixtures` rewrites a fixture's committed `*.expected.json`;
review that diff whenever a fixture input or tokenizer JSON changes. Every `*.tokenizer.json` and
`*.input.json` fixture source is SHA-256-pinned in `tools/tokenizer-oracle/provenance.json`, checked
by `runner.py` before use -- a tampered or stale fixture source fails the build rather than silently
comparing against drifted input.

The Python Pillow/Transformers image-preprocessing oracle (`tools/image-oracle/`) is the equivalent
fixture tool for `jmlx-vision`: a hash-locked CPython 3.12 venv pinning Pillow 12.3.0, NumPy 2.5.3
and Transformers 4.57.6, used to generate and verify the decode/resize/normalize/full-processor
fixtures under `tools/image-oracle/fixtures/` (`*.input.json` specs, `*.expected.json` canonical
oracle output, the fixture images and `preprocessor_config.json`) and, since Phase 7.3b WP2,
the Qwen3-VL `qwen3vl` family (`fixtures/qwen3vl/`), whose shared processed-patch bins under
`fixtures/qwen3vl/patches/` are digest-pinned in both `tools/image-oracle/provenance.json` and
`tools/hf-reference/provenance.json`. Install and verify with:

```sh
./tools/image-oracle/install.sh
./gradlew verifyImageOracle verifyImageOracleFixtures
```

Only `./gradlew generateImageOracleFixtures` rewrites a fixture's committed `*.expected.json`;
review that diff whenever a fixture input image or spec changes. Every fixture source is
SHA-256-pinned in `tools/image-oracle/provenance.json`, re-verified by the pure-Java
`verifyImageOracleFixtureDigests` (part of root `check`) and by `runner.py` before use.
`verifyImageOracleDependencyPins` (also part of root `check`) keeps the image and HF reference locks
in agreement on the shared Pillow/NumPy/Transformers pins. Python execution stays an explicit,
venv-dependent step -- ordinary builds gain no Python prerequisite.

`tools/hf-reference/` is a generate-only Python tool (Hugging Face `transformers`, CPU, float32)
that produced the tiny synthetic checkpoints and logits goldens under `tools/hf-reference/goldens/`
that the decoder-family tests compare against. Gradle never runs Torch: `verifyHfReferenceGoldens`
(part of root `check`) only verifies the committed files' SHA-256 against `provenance.json`.
Regeneration is a manual, reviewed step; see `tools/hf-reference/README.md`.

`scripts/checkDependencies.zsh` is a read-only report of available updates (Gradle plugins/deps, the
wrapper, the pinned `mlx-metal` version, and — since mlx-c versions independently of MLX — the mlx-c
tag that pairs with a newer wheel). `scripts/updateMlx.zsh <mlx-c-commit-sha>` repins `MLX_C_COMMIT` in
`bootstrap-native.sh` and re-runs bootstrap + regen for you; it takes a full 40-char SHA, not a short
one.

## Build, test, run

```sh
./gradlew build                # compiles jmlx-ffi, jmlx-core, jmlx-tokenizer, jmlx-jinja, jmlx-vision, jmlx-models, jmlx-examples, jmlx-benchmarks, jmlx-native-macos-arm64
./gradlew :jmlx-core:test       # memory lifecycle, numeric correctness, native error path
./gradlew test --tests "se.alipsa.jmlx.core.MLXArrayTest"   # a single test class
./gradlew :jmlx-examples:run    # runs HelloMLX end-to-end on real GPU hardware
./gradlew :jmlx-tokenizer:test  # pure-Java tokenizer tests (no native bootstrap needed)
./gradlew :jmlx-jinja:test      # pure-Java Jinja tests (no native bootstrap needed)
./gradlew :jmlx-vision:test     # pure-Java image decode/resample/processor tests (no native bootstrap needed)
```

**Gradle enforces numerical-reference precision.** Tests tagged `full-float32` run in
`:jmlx-core:float32GoldenTest` or `:jmlx-models:float32GoldenTest`
(part of `check`/`build`) in separate JVMs with `MLX_ENABLE_TF32=0`. This covers CPU Hugging
Face references, Mixtral sorted-prefill comparison, and strict core attention/MoE oracles.
Preserve their existing assertions; do not regenerate goldens to accommodate native reduced
precision. Tag new strict float32 references `full-float32` to select the enforced mode
automatically. The ordinary `test` tasks in both modules force `MLX_ENABLE_TF32=1` and exclude
that tag, retaining default-mode inference coverage.
`BatchStepEquivalenceTest` runs in both tasks, with its mode-specific tolerance. Both modes
are explicit task inputs, so the developer's shell cannot accidentally select the wrong mode.
Run `./gradlew :jmlx-models:check` for both, or target `float32GoldenTest --tests ...` for a
reference suite. The native CI execution assertion checks the corresponding result directories;
update its required-suite list when adding a native reference suite.
`jmlx-benchmarks:benchmarkTf32` controls its own workers' precision modes independently.
See `req/phase6-golden-precision.md` and `req/phase6-batch-equivalence.md` for the evidence,
and `jmlx-models/README.md` for precision selection in external applications.

Every module's native-dependent tests are **skipped, not failed**, when `native/install/lib/mlx.metallib`
is absent — see `@EnabledIfNativeAvailable` (a `jmlx-ffi` test fixture, shared via `testFixtures`,
delegating to `NativeLoader.ensureLoaded()` itself so the skip gate can never diverge from the real
loader logic). Don't add a separate existence check to decide whether native tests should run.
`jmlx-native-macos-arm64`'s `stageNativeResources` task is deliberately always actionable: when
native files are absent it removes stale generated resources before making an empty local jar, while
`verifyPackagedNativeResources` skips its content assertion. `jmlx-ffi`'s
`extractionFallbackTest` (see below) follows the same unstaged-checkout invariant, so `./gradlew
build` still succeeds for a contributor who has not bootstrapped.

`jmlx-ffi` also has a `loaderGuardTest` task (wired into `check`) that exercises `NativeLoaderMissingMetallibTest`
in its own JVM against a disposable copy of the native dir with `mlx.metallib` excluded — it's excluded
from the regular `test` task because `NativeLoader.ensureLoaded()` caches its outcome, so it would race
every other native test over the real staging directory if run in the same JVM. `extractionFallbackTest`
(also wired into `check`) is the same pattern for `NativeLoaderExtractionFallbackTest`, forcing
`NativeLoader` onto `ClasspathNativeExtractor`'s classpath-extraction fallback (blank
`jmlx.library.path`, `JMLX_LIBRARY_PATH` removed, the real `jmlx-native-macos-arm64` jar on its
classpath) to prove extraction works against the real, published-shape artifact — see
`ClasspathNativeExtractorTest` for the same mechanics exercised against tiny fake fixtures instead.

`jmlx-models` uses the same forked-JVM pattern for its thread probes, which `test` excludes:
`threadProbeTest` (`WorkerStreamProbeTest`, wired into `check`) needs a JVM where no other test has
used MLX first, and `concurrentStreamProbe` (`ConcurrentStreamProbeTest`) is evidence only. It races
two MLX threads and may crash its JVM, so it is never part of `check`.

Opt-in tasks that are not part of `check`:

```sh
# Real-artifact (Tier-B) smoke: download a pinned checkpoint, then run against it.
./gradlew downloadTierBArtifact -PtierBManifest=tools/tier-b/<manifest>.json -PtierBTarget=<model-dir>
JMLX_TIER_B_MODEL_DIR=<model-dir> JMLX_TIER_B_MANIFEST=tools/tier-b/<manifest>.json \
  ./gradlew :jmlx-models:tierBTest   # optional: JMLX_TIER_B_MODEL_TYPE=<model_type> to assert it
./gradlew :jmlx-benchmarks:benchmarkTf32 --args='...'       # TF32 enabled vs disabled (fresh JVMs)
./gradlew :jmlx-examples:benchmarkDecode --args='...'       # DecodeBenchmark (direct path)
./gradlew :jmlx-examples:benchmarkBatchDecode --args='...'  # BatchDecodeBenchmark (direct vs batched)
./gradlew :jmlx-examples:runSchedulerExample --args='...'   # SchedulerExample
```

`.github/workflows/tier-b.yml` runs `tierBTest` weekly and on demand, for the pinned artifacts in
`req/phase6-tier-b-artifacts.md`.

**CI checks that native tests actually ran.** The macOS native job in `.github/workflows/ci.yml`
reads the JUnit XML for a hard-coded list of suites, the "Assert required native suites executed"
step, and fails if any of them was skipped. When you add a native test class that must run, add it
to that list. Otherwise a native skip passes CI unnoticed.

## Releasing a module

Seven modules publish to Maven Central independently of each other and of the root project's
version: `jmlx-jinja`, `jmlx-tokenizer`, `jmlx-vision`, `jmlx-native-macos-arm64`, `jmlx-ffi`,
`jmlx-core`, and `jmlx-models`. Each has its own `release.sh` (all seven byte-identical, enforced
by `verifyReleaseScriptsMatch`). `jmlx-examples` (an application demo) and `jmlx-benchmarks`
(opt-in performance experiments) are unpublished.

```sh
./jmlx-jinja/release.sh              # publishes se.alipsa:jmlx-jinja
./jmlx-tokenizer/release.sh          # publishes se.alipsa:jmlx-tokenizer
./jmlx-vision/release.sh             # publishes se.alipsa:jmlx-vision
./jmlx-native-macos-arm64/release.sh # publishes se.alipsa:jmlx-native-macos-arm64
./jmlx-ffi/release.sh                # publishes se.alipsa:jmlx-ffi
./jmlx-core/release.sh               # publishes se.alipsa:jmlx-core
./jmlx-models/release.sh             # publishes se.alipsa:jmlx-models
```

Every script refuses to publish a `-SNAPSHOT`, and refuses a module that has no `version` line of
its own (which would mean it is silently riding the root version). To release: set the module's
version to a release value, run its `release.sh`, then bump it to the next `-SNAPSHOT`. Version
bumping is deliberately manual.

**Release order matters, along two independent chains.** `jmlx-tokenizer` depends on `jmlx-jinja`
via `api project(...)`; `jmlx-core` depends on `jmlx-ffi` via `implementation project(...)`;
`jmlx-models` depends on `jmlx-core`, `jmlx-tokenizer` and `jmlx-vision` via `api project(...)`.
Gradle publishes every one of these as a concrete coordinate, and `verifyNoSnapshotDependencies`
fails a release while any of them is still a SNAPSHOT. Release jinja before tokenizer, and ffi
before core before models. `jmlx-vision` itself has no published project dependency, but models
needs it, so release vision before models too. `jmlx-native-macos-arm64` has no published-POM
dependency relationship to either chain, and jmlx-core deliberately does not depend on it (see
Decision 5 in
`req/plans/native-artifact-packaging-plan.md`); it can therefore be released independently. Its
artifact is nevertheless a test-fixture build dependency of `jmlx-ffi`'s `extractionFallbackTest`,
so `jmlx-ffi:check` and any ffi release build the native jar. Release each chain's modules while the
ones it depends on are still checked out at their released version; only after every release in a
chain should its modules be bumped back to their next `-SNAPSHOT` versions.

Signing and Central credentials come from Gradle properties (`signing.keyId`,
`sonatypeUsername`, `sonatypePassword`), normally in `~/.gradle/gradle.properties`. Signing is
inert when `signing.keyId` is absent, so `check` and CI stay keyless. Releasing is a local,
credentialed, manual action — CI verifies but never publishes. `main` is protected by
repository push rules — a direct push is rejected with "push declined due to repository rule
violations" — so every change, including a one-line doc or status note, must ride a PR that
passes CI (branch, `gh pr create`, then stop and wait for the human's review: a green CI is
not a merge — run `gh pr merge <n> --merge` only after the human has reviewed the PR and
explicitly approved it). A local `main` can therefore sit
ahead of `origin/main` with a commit it cannot push; reset it to `origin/main` and carry the
commit on a feature branch instead.

`release.sh` runs `check` (via the `release` task's own dependencies), but not `jmlx-jinja`'s
heavier `releaseVerification` matrix (isolated Gradle home, candidate-vs.-two-clean-archives diff,
dependency-update review) — that is not part of `check` and stays a separate, manual step.
`jmlx-jinja/req/release-checklist.md` is jinja's full release procedure; follow it, not just
`release.sh`, before an actual jinja release.

## Release smoke

`tools/release-smoke/` is an independent Gradle consumer (no `project(...)` deps) that proves the
published shape: `./tools/release-smoke/run.sh ci` publishes the seven modules to a disposable
`build/smoke-repo` (each module's opt-in `smoke` Maven repository,
`publishMavenPublicationToSmokeRepository`), then resolves `jmlx-models` plus the runtime-only native
jar from it under a fresh `GRADLE_USER_HOME` with `exclusiveContent` for `se.alipsa`, a disposable
`jmlx.native.cache.path`, and no native path override. It also checks the POM-only runtime classpath
matches the Gradle-metadata one, every jar's license, the three vision jars' `META-INF/NOTICE`
content (the pinned Pillow source references, SHA-256s and MIT-CMU license text), the native jar's
payload, and a seeded two-request batch against `goldens/mistral-sampled.properties`. The smoke
gates the scheduler so both requests share one cohort, then asserts `cohortSizes() == [2]`.
`--record` rewrites the golden and is accepted in `ci` mode only. The native CI job runs `run.sh ci`.
`run.sh candidate` is the same with SNAPSHOT suffixes stripped (release versions set first), and
`run.sh central` resolves from Maven Central after all seven are published. The smoke tokenizer in
`fixtures/` is hand-written to pair with the synthetic Mistral checkpoint (see its `PROVENANCE.md`).

## Code style

Hand-written sources are Google Java Style, 2-space indent, 100-column width. Formatting is
enforced by running `googleJavaFormat(...)` directly from root `build.gradle`'s Spotless block (no
separate Spotless config file); linting is `config/checkstyle/checkstyle.xml`, derived from
checkstyle's own bundled `google_checks.xml` with deviations documented in a comment at the top of
that file. Checkstyle runs with `maxWarnings = 0` in both builds: the config reports violations at
`warning` severity, but any violation fails `check` (for `buildSrc`, run
`./gradlew -p buildSrc check`; see below), the same as a Spotless difference. Fix a violation
(rename, add the Javadoc) rather than leaving it. Never raise `maxWarnings`. Use a targeted
suppression (`@SuppressWarnings("checkstyle:<Check>")` or a
`// CHECKSTYLE.SUPPRESS: <Check> for +N lines` comment) only where a name is genuinely externally
constrained, with a comment saying why. The generated jextract bindings under
`jmlx-ffi/src/main/generated/java` are exempt from both and must stay byte-identical to
`scripts/regen-bindings.sh`'s output — never hand-edit them.
`buildSrc` (shared Gradle build-logic helpers, e.g. `PublishedPomDependencies`) is its own isolated
Gradle build, so root `build.gradle`'s `subprojects{}` conventions never reach it -- it applies the
same checkstyle/Spotless configuration independently in its own `buildSrc/build.gradle` instead
(duplicating the two tool versions, since it can't read the main build's
`gradle/libs.versions.toml` either). That check is not wired into the main build's own
`check`/`build` -- buildSrc is a separate Gradle build with no automatic cross-invocation -- so run
it explicitly: `./gradlew -p buildSrc check`.

```sh
./gradlew spotlessCheck                                      # verify formatting
./gradlew spotlessApply                                       # reformat in place
./gradlew checkstyleMain checkstyleTest checkstyleTestFixtures # style/lint (part of build/check)
```

## Architecture

```
jmlx-benchmarks  Tf32Benchmark, Tf32Worker       opt-in performance experiments

jmlx-examples    HelloMLX, SchedulerExample,       demos and opt-in benchmarks
                 DecodeBenchmark, BatchDecodeBenchmark
       |
jmlx-core        se.alipsa.jmlx.nn                 Module, Linear, QuantizedLinear,
                                                    RMSNorm/LayerNorm/SiLU/GELU/Embedding,
                                                    MultiHeadAttention, GroupedQueryAttention,
                                                    DecoderAttention, DecoderBlock, AttentionMask,
                                                    RopeSpec, KVCache, KVCachePolicy, GatedMlp,
                                                    SwiGLU, MoeMlp, SwitchGlu, ModuleGrad,
                                                    MultiAxisRope, MultiAxisRotation
                 se.alipsa.jmlx.core                MLX, MLXOps, MLXShape, MLXFast, MLXQuant,
                                                    MLXRandom, MLXGrad, MLXIO, MLXMemory, MLXArray,
                                                    DType, MLXException
                 se.alipsa.jmlx.memory              MLXScope
       |
jmlx-ffi         se.alipsa.jmlx.ffi.*              committed jextract output
                 NativeLoader                      hand-written, tries jmlx.library.path /
                                                    JMLX_LIBRARY_PATH, then classpath extraction
                 ClasspathNativeExtractor           extracts jmlx-native-macos-arm64's bundled
                                                    binaries to a per-pin cache dir on disk
       |
native/install/lib/  libmlxc.dylib                 built by scripts/bootstrap-native.sh
                     libmlx.dylib  libjaccl.dylib   staged from the mlx-metal wheel
                     mlx.metallib  native-pin.properties  staged from the mlx-metal wheel

jmlx-native-macos-arm64  se/alipsa/jmlx/native/macos-aarch64/  the same 4 binaries + pin file,
                                                    packaged as classpath resources -- optional
                                                    runtime dependency, not a build-time one

jmlx-tokenizer   se.alipsa.jmlx.tokenizer           HfTokenizer, IncrementalTokenDecoder,
                                                    TokenizerEncoding/EncodingOptions/Padding/
                                                    Truncation, ChatTemplateRenderer/
                                                    ChatTemplateOptions, TokenizerJson/
                                                    TokenizerJsonLoader, TokenizerMetadata,
                                                    Vocabulary, BpeMerger, ByteLevel* components,
                                                    AddedTokenSplitter, TextNormalizer,
                                                    PostProcessorApplier, TokenizerException
                                                    (byte-level BPE, Metaspace BPE, Unigram,
                                                    WordPiece)
       |
jmlx-jinja       se.alipsa.jmlx.jinja              Template, chat-template Jinja rendering
                 pure Java; no dependency on jmlx-ffi or native/install/lib

jmlx-vision      se.alipsa.jmlx.vision             ImageDecoder, RgbImage, ImageTensor, Layout,
                                                    PillowResample, ImageTransforms, Resampling,
                                                    ImageDecodeLimits, AlphaPolicy,
                                                    ExifOrientationPolicy,
                                                    SmolVlmImageProcessor/
                                                    SmolVlmProcessorConfig/
                                                    SmolVlmImageProcessorResult,
                                                    Qwen3VlImageProcessor/
                                                    Qwen3VlProcessorConfig/
                                                    Qwen3VlGeometryPlan/
                                                    Qwen3VlImageProcessorResult
                                                    pure Java; Pillow-exact bilinear/bicubic/
                                                    LANCZOS resampling; SmolVLM/Idefics3 and
                                                    Qwen3-VL (Phase 7.3b WP2) image
                                                    processors; no dependency on jmlx-ffi or
                                                    native/install/lib

jmlx-models      se.alipsa.jmlx.models             TextGenerationModel, TextGenerationModels,
                                                    GenerationConfig/Request/Result/Event,
                                                    GenerationCachePolicy, CancellationToken,
                                                    FinishReason, GenerationAbortedException,
                                                    ModelMetadata, DecoderModel, LlamaModel,
                                                    QwenModel, MistralModel, GemmaModel, Phi3Model,
                                                    MixtralModel, ArchitectureDescriptor/
                                                    ArchitectureMappings, DecoderAssembler,
                                                    DecoderConfig, TensorPlan,
                                                    BatchGenerationScheduler (+ BatchSchedulerConfig,
                                                    BatchRequestHandle, Scheduler*/BatchAdmission*
                                                    exceptions)
                 depends on jmlx-core + jmlx-tokenizer + jmlx-vision;
                 native inference and generation
```

Three native modules, deliberately: the jextract output for `mlx/c/mlx.h` is a large generated blob. Isolating
it in `jmlx-ffi` means it compiles once and stays untouched by day-to-day iteration on `jmlx-core`,
keeping incremental builds fast and generated code out of review diffs. `jmlx-core` depends on
`jmlx-ffi` as `implementation`, not `api` — `MLXArray`/`MLX` wrap raw `MemorySegment` handles behind
plain Java types and never expose `jmlx-ffi` on their own public surface.

**`jmlx-tokenizer`, `jmlx-jinja` and `jmlx-vision` sit outside the native chain entirely.** None of
them depends on `jmlx-ffi`; they never touch `native/install/lib` and do not call
`NativeLoader.ensureLoaded()`. They are not all independent of each other, though: `jmlx-tokenizer`
declares `api project(':jmlx-jinja')` and `ChatTemplateRenderer` returns `jmlx-jinja`'s own `Template`
type on its public API, since `jmlx-tokenizer` (Phase 5 M2, `req/plans/phase5-m2-plan.md`) is a
from-scratch Java port of the byte-level-BPE pipeline that renders HF `chat_template` strings through
`jmlx-jinja`, the migrated former `hfjinja` project. `jmlx-vision` (Phase 7.3a,
`req/plans/phase7-3a-plan.md`) is a pure-Java image decode/resample/normalize library plus the
SmolVLM/Idefics3 image processor, whose only external dependency is Jackson (matching
`jmlx-tokenizer`) and which has no project dependencies at all; its resampling kernels are a
Pillow-ported implementation pinned by SHA-256 in `jmlx-vision/NOTICE`. See `jmlx-jinja/README.md`
for usage and its own `upstreamVerify` / Node-oracle verification tasks, and `jmlx-vision/README.md`
for the image oracle's fixture and precision story. None is part of the "Loading order matters"
native-guard discussion below (which is specific to `MLX`, `MLXScope`, `NativeOps`, `MLXGrad`,
`MLXIO`, and `MLXMemory`).

**`jmlx-models` is in the native chain.** It depends on `jmlx-core` for MLX inference, on
`jmlx-tokenizer` for prompt encoding/decoding, and on `jmlx-vision` (as `api`) for image
preprocessing, so model loading and generation require the native bootstrap and participate in the
same loading-order guarantees as `jmlx-core`.

`jmlx-tokenizer`, `jmlx-jinja` and `jmlx-vision` override the toolchain to **Java 21** rather than
inheriting the root's Java 25 — that 25 exists for `jmlx-core`/`jmlx-ffi`'s Panama FFM, and none of
those pure-Java modules needs it, so targeting 21 keeps their published artifacts usable by Java 21
consumers. `jmlx-tokenizer`'s and `jmlx-vision`'s `verifyBytecodeLevel` tasks enforce this.

**Seven modules are published**, each carrying its own version independent of the root's
`0.5.0-SNAPSHOT` (`jmlx-examples` and `jmlx-benchmarks` are unpublished applications): `jmlx-jinja` is
`0.6.0-SNAPSHOT` (continuing the archived hfjinja project's line), `jmlx-tokenizer`,
`jmlx-native-macos-arm64` and `jmlx-vision` are all `0.1.0-SNAPSHOT` (none has ever been
published), `jmlx-ffi` and `jmlx-core` are both `0.5.0-SNAPSHOT` (an explicit line carrying forward
the number they already had via the root's default), and `jmlx-models` is `0.1.0-SNAPSHOT` (also
never published). `jmlx-core`/`jmlx-models`/`jmlx-vision` disable Javadoc's `missing` doclint
category (`-Xdoclint:all,-missing`) rather than retrofitting `@param`/`@return`/`@throws` tags onto
an already prose-documented public surface — the root's reactive `maven-publish` wiring's `-Werror`
still gates every other Javadoc problem. `jmlx-ffi` instead restricts Javadoc's (and `checkstyleMain`'s) source
to its hand-written tree, excluding the committed jextract bindings entirely, for the same reason
those bindings are exempt from Spotless/checkstyle (see "Code style" above). See "Releasing a
module" above.

**Loading order matters.** jextract binds each downcall's method handle lazily, in a private
per-function holder class, the first time that function is called — and that first call fails unless
the dylib is already loaded by then. `MLX`, `MLXScope`, `NativeOps`, `MLXGrad`, `MLXIO`, and
`MLXMemory` each have a static initializer that calls `NativeLoader.ensureLoaded()` for exactly this
reason; any of the six can be the first one touched, so each guards independently rather than
relying on load order. `NativeOps`'s
guard covers `MLXOps`/`MLXShape`/`MLXFast`/`MLXQuant`/`MLXRandom` transitively, since every op in those
classes reaches native only through `NativeOps`'s own `checked`/`binaryOp`/`unaryOp`/`shapeOp`-family
helpers — none of those classes needs (or has) its own guard. `MLXGrad` and `MLXIO` both need their own
regardless: `MLXGrad` calls `mlx_h.*` directly at 17 call sites (`mlx_closure_*`/`mlx_value_and_grad` and
their frees), and `MLXIO` calls `mlx_h.*` directly for safetensors/GGUF I/O and their five
non-`mlx_array` struct types (`mlx_string`, `mlx_vector_string`, `mlx_map_string_to_array`,
`mlx_map_string_to_string`, `mlx_io_gguf`) — both bypass `NativeOps`'s helpers entirely, so neither can
rely on `NativeOps`'s guard the way the op facades do.

**`NativeLoader`** loads `libmlxc.dylib` via `System.load(absolutePath)`, not
`SymbolLookup.libraryLookup`: the bindings are generated *without* a `-l` flag, so their internal
`SYMBOL_LOOKUP` is `loaderLookup().or(defaultLookup())`, which only sees libraries registered with the
calling classloader — exactly what `System.load` does and `libraryLookup()` does not. It resolves the
native library directory from the `jmlx.library.path` system property first, then `JMLX_LIBRARY_PATH`.
The root `build.gradle` wires every `Test` task's `jmlx.library.path` to the real
`native/install/lib`; `jmlx-examples`'s `run` task does the same, but its `installDist`/`distZip` start
scripts deliberately do *not* get that property baked in (it's an absolute build-machine path) — a
distributed launcher elsewhere is expected to set `JMLX_LIBRARY_PATH` instead.

mlx-c's default error handler is `printf` + `exit(-1)` on failure, which would kill the JVM before any
status code is observable. `NativeLoader` installs a replacement handler (an FFM upcall stub, kept
alive in a process-lifetime `Arena` since mlx-c may call it any time) that just records the message on
a thread-local, since the mlx-c error convention fires synchronously on the calling thread.
`NativeOps.checked` (moved from `MLX.checked` in an earlier phase) clears that thread-local
immediately before every native call (not just after a failure) and wraps a
non-zero status into `MLXException`, attaching the recorded native message when present. A few mlx-c
entry points (e.g. `mlx_array_new_data`) have no status return at all and only signal failure via the
error handler plus a null `ctx` in the returned struct — those call sites check for that explicitly.

**Memory model:** `MLXScope` owns every native `mlx_array` handle allocated through it (it implements
`SegmentAllocator` so it can be passed directly to mlx-c's struct-returning constructors) and frees them
in reverse insertion order on `close()`. A `Cleaner` backstop exists in case a scope is never explicitly
closed, but the cleanup action is registered against a private `Holder` object, never the `MLXScope`
itself — capturing the scope in the cleanup action would make it permanently reachable and the action
would never run. `MLXScope` and every `MLXArray` allocated from it are confined to the thread that
created the scope (checked via `ensureOpen()`/thread-identity checks on every access, including
centrally in `MLXArray`); the sole exception is the JVM Cleaner thread, which only ever touches the
`Holder`, never the scope. An op's result is allocated into the innermost scope among every non-null
`MLXArray` operand it's given (`NativeOps.scopeOf`), not just its first — a rule that rejects two
operands only when their scopes are *unrelated* (siblings, or two independent roots), not merely
different: a related ancestor/descendant pair is legal and resolves to the descendant (e.g.
`Linear.forward`'s `add(y, bias)` across a model scope and a step scope, or
`MLXQuantTest.dequantizeWithBiasesInAChildScopeOfWAllocatesIntoTheChild`). `MLX.array(scope, ...)` and
a few other ops (creation ops, `MLXRandom.normal`/`uniform`, `MLXGrad.Fn#apply`) take the scope
explicitly since they have no operand to infer one from; `MLX.hoist`, `MLXShape.transpose(MLXArray,
MLXScope)`, and `MLXQuant.quantizedMatmul`'s explicit-target overload instead override `scopeOf`'s own
answer, letting a caller push a result toward an ancestor or descendant scope on purpose. See `MLX`'s
own class javadoc for the full, authoritative version of this rule.

**Lazy evaluation:** every op in `MLX` only builds mlx's lazy computation graph; nothing runs on
GPU/CPU until `MLX.eval(...)` (or the implicit eval inside `MLXArray.toFloatArray()`) forces it.
`toFloatArray()` also forces contiguity first (`mlx_contiguous`, row-major) before reading the raw data
pointer — a lazy op like `transpose` can otherwise yield a strided view, and reading raw data without
that step would return plausible-looking values in the wrong order instead of crashing.

Shape is plain `int[]`; there is no `Shape` type in this slice. `DType` covers `FLOAT32`, `INT32`,
`BOOL`, `UINT32` (the packed-weight dtype `QuantizedLinear` validates against), `FLOAT16` and
`BFLOAT16`. `defaultDevice()` is resolved once from mlx-c's own default and cached for the process
lifetime — this slice doesn't expose device switching. Streams are **not** process-wide: MLX streams
are thread-bound ("There is no Stream(gpu, N) in current thread"; MLX's default GPU stream only
worked on the first thread to use it), so each `MLXScope` carries its owner thread's own stream
(`MLXScope.stream()`, created once per thread by the first root scope, inherited by children), and
every op runs on its result scope's stream. Arrays must therefore be evaluated on the thread that
built them. The default CPU stream (`NativeOps.DEFAULT_CPU_STREAM`, used by `MLXIO`) works from any
thread.

**Threading rule:** use MLX from at most one thread at a time per process. The full contract is in
the `se.alipsa.jmlx.core` package Javadoc (`package-info.java`).

- Each thread's stream is created once and never freed (about 60 KB per thread), so many short-lived
  MLX threads leak streams. Prefer one long-lived MLX thread.
- MLX binds streams to the OS thread through a C++ `thread_local`, and a virtual thread can move
  between OS threads. `MLXScope` therefore rejects a root scope created on a virtual thread with
  `IllegalStateException`.
- To serve concurrent callers, use `BatchGenerationScheduler` in `jmlx-models`. Its single
  platform-thread worker owns the model and decodes compatible requests together as one cohort.
  Callers submit and cancel from any thread. Its Javadoc defines the lifecycle, the per-row versus
  per-cohort failure rules, and the optional cohort gate with its `maxBatchWait`.
- The scheduler can only block a second scheduler within one classloader. It cannot stop direct
  `generate` calls from other threads, so the one-thread rule there is enforced by documentation
  only.

## Native version pinning

mlx-c is versioned independently of MLX itself (mlx-c's own `CMakeLists.txt` pins an exact MLX tag via
`GIT_TAG`), so a newer `mlx-metal` wheel does not imply any particular mlx-c commit works with it — the
two are only bumped together, via `scripts/checkDependencies.zsh` (which cross-references mlx-c's tags
against the wheel version) and `scripts/updateMlx.zsh`. Current pin: mlx-c `fba4470` (tracks `v0.6.0`)
against `mlx-metal==0.31.2` (`macosx_26_0_arm64`) — see Decision 9 in `req/initial-plan.md` for why the
latest 0.32.0 wheel was deliberately not used.

`scripts/jextract-overrides/mlx/c/half.h` is a trimmed override copied over the staged header before
running jextract: the real header's `__bf16` typedef (the raw C half-precision arithmetic type) is a
hard parse error for jextract on this target. This only strips that C-level typedef, not
bfloat16 support itself — `DType.BFLOAT16` (mapping to mlx-c's own `MLX_BFLOAT16` enum constant) is
fully supported by this slice; jextract only ever needed to parse past the C type, never bind it.
`regen-bindings.sh` runs jextract twice — an unfiltered
discovery pass, then a second pass restricted to symbols whose `--dump-includes` header path is under
`mlx/c/` — both to trim Darwin system-header spillover and because (independently) an unfiltered
whole-umbrella run silently drops `mlx_array_new`/`mlx_array_free`/`mlx_array_new_data`/
`mlx_array_new_data_managed` from the generated Java with no warning. See Decision 10 in
`req/initial-plan.md` for the full story if either script needs changing.

## Requirements docs

`req/initial-plan.md`, `req/phase3-plan.md`, and `req/phase4-plan.md` are living design documents, not
just history — they record the *why* behind non-obvious decisions (native library loading strategy,
jextract's two-pass generation, memory confinement rules, the `se.alipsa.jmlx.nn` module framework and
autograd design) that the code comments themselves point back to. `req/plans/phase4-m4-plan.md` is
`QuantizedLinear`'s own implementation plan, and `req/plans/phase5-m1-plan.md` is `MLXIO`'s, each
amended in place (not rewritten) as post-merge/post-implementation review found and fixed real gaps in
it. When touching `NativeLoader`, `MLXScope`, `MLX`, `NativeOps`, `MLXIO`, anything under
`se.alipsa.jmlx.nn`, or the bootstrap/regen scripts, check whether the relevant decision is already
recorded in one of these before re-deriving it. `req/project-outline.md` is the original,
broader multi-phase vision; where it conflicts with `req/initial-plan.md` on scope or naming, the
latter is authoritative for what's actually being built now. `req/full-roadmap.md` extends that
outline after Phase 5: use it to select future phase/milestone scope and exit gates, then write the
corresponding detailed plan under `req/plans/` before implementation.

Phase 5 and 6 documents:

- `req/phase5-plan.md` and `req/plans/phase5-m3-retrospective.md`.
- `req/plans/phase6-plan.md`, with one plan per milestone (`phase6-0b-plan.md` through
  `phase6-5-plan.md`) plus their probe findings, all under `req/plans/`.
- Living reports:
  - `req/phase6-compatibility.md`: the supported-family matrix. Update it whenever verification
    status changes.
  - `req/phase6-inference-report.md`: Phase 6 status and the remaining release steps.
  - `req/phase6-4-benchmark.md`: decode benchmarks.
  - `req/phase6-tier-a-fixtures.md` and `req/phase6-tier-b-artifacts.md`: fixtures and pinned
    artifacts.
  - `req/phase6-2-tokenizer-components.md`: tokenizer component coverage.
  - `req/mlx-api-inventory.md` and `req/mlx-api-header-coverage.md`: binding coverage, generated by
    the `*MlxApi*` Gradle tasks.
- `req/plans/phase7-3a-plan.md` (the vision/multimodal plan, WP1-WP8) and
  `req/plans/phase7-3a-reference-findings.md` (the pinned reference findings it builds on).


## Phase 7.2 additions

Core now includes BidirectionalAttention, CrossAttention, EncoderBlock and StaticKVCache.
Models adds TextEncoderModels, SequenceClassifiers, TokenClassifiers, Pooling, defensive result
records and T5Model/T5LoadOptions. Tokenizer adds BERT pair encoding and Precompiled charsmap
normalization. These do not change native pins or generated bindings. EncoderDefaultPrecisionTest
runs in ordinary test; BertGoldenTest, T5GoldenTest and EncoderAttentionTest use full-float32.
Task-specific Tier-B suites are selected by manifest task, independently of model_type.
Java downloadTierBArtifact owns SHA verification and does not declare Gradle outputs. Its buildSrc
tests remain explicitly separate from root check. Python execution remains opt-in for reference
and oracle tools; ordinary builds gain no Python prerequisite. See the Phase 7.2 implementation
report for local evidence and the pending new macOS 26 CI gate. Versions and publication are unchanged.

## Phase 7.3a additions

`jmlx-vision` is a new pure-Java 21 module (no native dependency, Jackson only): image decoding
(PNG/JPEG, including palette PNGs), Pillow-exact bilinear/bicubic/LANCZOS resampling, normalization,
and the SmolVLM/Idefics3 image processor, with its byte-exactness evidence coming from the
`tools/image-oracle/` fixtures. `jmlx-models` now `api`-depends on it, so seven modules are
published (jmlx-vision starts a fresh `0.1.0-SNAPSHOT` line; its `release.sh` is byte-identical to
the others and covered by `verifyReleaseScriptsMatch`). CI: the Ubuntu job runs
`:jmlx-vision:check` plus the executable image-oracle verification; root `check` carries the
pure-Java pin/digest guards; the release smoke asserts the three vision jars' Pillow NOTICE
content. The Pillow port provenance lives in `jmlx-vision/NOTICE` (pinned source SHA-256s plus the
full MIT-CMU license text) and is carried in all three vision jars. Vision's documented
operation-order divergences from the reference (decode-then-resize in RGB space vs the reference's
resize/split before `convert_to_rgb`) are recorded in `jmlx-vision/README.md` and the reference
findings; strict full-chain fixtures use opaque RGB inputs where the two orders agree.


Phase 7.3a WP6 adds `SmolVlmModel` (composition of an internal Llama decoder, Idefics3 vision
tower and connector) and strict `idefics3` loader dispatch before decoder-only mappings. All
floating source weights are converted to FLOAT32; vision/connector checkpoint keys and the
PyTorch OIHW-to-MLX OHWI patch conversion are explicit. Per-request visual scopes close after
evaluated prefill logits/KV; subsequent tokens use the shared decoder generation loop. Metadata
reports TEXT+IMAGE, results retain original IDs and charge expanded promptPositions, and a zero-
token request performs geometry/expansion validation without pixel or native work. Full and bounded
full retention are supported; sliding image-prefill retention is rejected. Native suites are
`SmolVlmGoldenTest` (full-float32), `SmolVlmGenerationTest`, `SmolVlmDefaultPrecisionTest` and
`Idefics3VisionTowerSmokeTest`, all required by macOS CI. WP7 adds `VisionLanguageMemoryTest` (required in ordinary macOS CI) and opt-in
`TierBVisionLanguageTest`, selected by manifest task `image-text-generation`, with pinned
SmolVLM-256M CPU logits, licensed local image, expanded IDs, stable greedy IDs and memory
measurements. `verifyHfReferenceGoldens` also verifies the vision Tier-B manifest/image/generator
digests; neural regeneration remains manual via `tools/hf-reference/vision_tier_b.py`. Local WP7
evidence passes. WP8 is implemented via PR #47, with local consumer/capability evidence in
`req/phase7-3a-implementation-report.md`; the macOS 26 hosted CI/Tier-B acceptance remains pending.

## Phase 7.3b additions

Phase 7.3b is the Qwen3-VL image+text milestone; `req/plans/phase7-3b-plan.md` and
`req/plans/phase7-3b-reference-findings.md` are the authoritative records for this section. The
landed prerequisites: WP0 pinned the `Qwen/Qwen3-VL-2B-Instruct` 2B artifact (revision plus full
file/tensor inventory), and WP1 recorded the executable reference probes, including the
fast-vs-slow preprocessing comparison under `tools/image-oracle/fast-comparison/` and the WP1
probe goldens. In `jmlx-vision`, WP2 part 1 landed the pure-Java `Qwen3Vl*` preprocessing port
(`Qwen3VlProcessorConfig`, `Qwen3VlGeometryPlan`, `Qwen3VlImageProcessor`,
`Qwen3VlImageProcessorResult`) plus `ExifOrientationPolicy` and the decode-level
`exif-orientation` oracle family (the WP9 policy: `APPLY` for the Qwen file path, `IGNORE` kept
for SmolVLM). WP2 part 2 (2026-10-18) landed the `qwen3vl` image-oracle family (21 cases,
`run_qwen3vl` in `tools/image-oracle/runner.py`), the `Qwen3VlImageProcessor.processFile(Path)`
and `processEncoded(byte[])` file-to-processor entry points (decode `DROP_ALPHA` + `APPLY`), and
the shared processed-patch
fixture under `tools/image-oracle/fixtures/qwen3vl/patches/`, whose manifest and bins are
digest-pinned in both `tools/image-oracle/provenance.json` and `tools/hf-reference/
provenance.json`; it also pinned the 4.57.6 path-input invariant that the slow processor's
`exif_transpose` runs only on the string-input load path (PIL-image inputs skip it), verified
against the rotated fixtures (findings §4.8). No native pins or generated bindings changed, and
no new Gradle tasks were added — the existing `verifyImageOracleFixtureDigests` and
`verifyHfReferenceGoldens` guards cover the new fixtures, and `ImageOracleQwen3VlTest` (part of
`:jmlx-vision:test`) is the Java-side byte verification.

Phase 7.3b WP4 (2026-10-20) landed the frozen `MultiAxisRope`/`MultiAxisRotation`
core components in `jmlx-core`'s `se.alipsa.jmlx.nn` (the Qwen3-VL interleaved
three-axis RoPE kernel with its `cachePosition + delta` decode coordinates), the
additive request-supplied rotary-state / post-block injection seam on the
`DecoderModel` path (existing overloads untouched; production decoders use no-op
hooks; `RopeSpec`'s sealed hierarchy and `MLXFast` are byte-identical — verified by
empty diff), and the package-private `VisualFeatureInjection` compositions in
`jmlx-models` (row-gather image-row replacement plus masked DeepStack residual add,
composed from existing core facades over the Idefics3 gather-index algorithm — no new
primitive, no FFI from model code). Old-path bit identity in both TF32 modes is
verified against the pre-change exact-bits baseline (`exactBitsVerifyTf32On`/`Off`,
14 variants × 9 captures), and the pinned-vendor probe pair
(`tools/hf-reference/qwen3vl_rope_interleaved_check.{py,jsh}` — manually run in their
locked environments, never Gradle-invoked) cross-checks the component. The CI
required-suite list addition landed in PR #57 (review round 2), and `Qwen3VlModel`,
its loader dispatch and the DeepStack consumption of `VisualFeatureInjection` are
WP5/WP6. No native pins, generated bindings or build files changed.

Phase 7.3b WP5 (2026-10-22) landed the Qwen3-VL vision tower in `jmlx-models`:
`Qwen3VlVisionTower` (patch embed + 3-D interpolated position embeddings, N
`Qwen3VlVisionBlock`s — each a LayerNorm/attention/LayerNorm/MLP two-statement residual
block, matching the pinned 4.57.6 `Qwen3VLVisionBlock.forward` sequential semantics
where `norm2` sees the updated residual — plus DeepStack level merges) composed with
`jmlx-core`'s new `PatchMerger`, `TwoAxisRope`, `TwoAxisRotation` and
`TwoAxisRotaryAttention` components (all with their own core tests), returning
`Qwen3VlVisionFeatures` (merged rows + per-level DeepStack rows) hoisted into the
caller's per-request target scope (`forward(MLXScope, List)`, the `SmolVlmModel`
per-request pattern — the tower's module scope receives no per-forward handles, and
multi-image per-image parts are released after the evaluated concatenation); each
encoder block additionally runs in a private per-block child scope of the image scope
that is evaluated and closed before the next block starts, and the embedding stage
(patch embed + position-embedding gathers) runs in its own child scope of the image
scope that is evaluated and closed before block 0 starts (the per-block and
embedding-stage evaluate-and-release loops of the 2026-10-23 review round), so one
block's intermediates never accumulate across blocks and the per-image peak neither
scales with depth nor carries the embedding stage while the encoder runs. The strict
HF intermediate-feature golden
`tools/hf-reference/goldens/qwen3vl-vision-wp5.json` plus its tiny synthetic
checkpoint, image and generator script (`tools/hf-reference/qwen3vl_vision.py`,
digest-pinned in `tools/hf-reference/provenance.json`) drive
`Qwen3VlVisionTowerGoldenTest` (full-float32), and `Qwen3VlVisionPeakTest`
(ordinary `test`) measures the real-2B-geometry per-image peak — the tower forward's
own transient 1.25 GiB (all three depth probes bit-for-bit equal, depth 4 − depth 1
= 0 B against the test's 64 MiB depth-independence bound), sustained 16 MiB, and the
isolated attention-bracket transient 0.31 GiB, all under the plan §4 5 GiB bound —
recorded in the reference findings §11.4, which documents the pre-fix cause
(per-image scope Java handles pinning every block's intermediates, slope
1,174,405,136 B = 70 × 16 MiB + 16 B per block) and now covers the fix with the
depth-independence regression test. Both suites are required by the macOS CI
assertion. The review round added the opt-in
`exactBitsRecord/VerifyVision` Gradle tasks (never part of `check`) for the
bit-identity evidence; no native pins or generated bindings changed.

Phase 7.3b WP6 (2026-10-25) landed the composed `Qwen3VlModel` (the WP4 seam decoder plus the
WP5 vision tower, whose DeepStack levels are consumed through the WP4 `VisualFeatureInjection`
seam; a `TextGenerationModel`, not a `DecoderModel`, so the scheduler's existing
non-DecoderModel check rejects it), `qwen3_vl` dispatch before the generic decoder config parse
with `qwen3_vl_moe` rejected by name, the `2 * sum(mrope_section) == head_dim` load guard, the
rotary width derived from `head_dim`, and the in-place BF16/FP16-to-FLOAT32 up-cast of every
stored floating weight (the affine plan keeps the packed U32 tensors and up-casts the floating
sidecars). Local acceptance evidence: `Qwen3VlModelLoadTest` and `Qwen3VlModelAffineLoadTest`
(ordinary CI; the affine test packs 4-bit group-64 in Java and compares the packed decoder
against the identical dequantized weights) plus the full-float32 `Qwen3VlModelFloatGoldenTest`
(strict HF logits golden `qwen3vl-model-wp6.json`, four cases including text-only) and
`Qwen3VlTextOnlyEquivalenceTest` (seam multi-axis versus plain-base text-only logits, one and
the same angle table) — all four required by the macOS CI assertion. The plan §7 one-off
`benchmarkDecode seam-comparison` (the seam's explicit-mask path versus the old implicit-causal
path on bit-identical text weights) is recorded with the full reproducibility tuple in
`req/phase6-4-benchmark.md`: clean commit `c8150b8`, 31 paired samples after 5 warmups,
median seam/old ratio 1.1621 (about 16% slower), p10/p90 0.9572/1.2016. Raw per-iteration
nanos are committed under `req/benchmarks/`; the spread is descriptive, not a confidence
interval or evidence of noise-level cost. This is a one-off fixture run, not a CI gate.
The wide-attention path also has strict HF float32 logits coverage (hidden 256, attention
width 512; asymmetric BF16-exact projections, text/image prefill and cached decode).
No native pins or generated bindings
changed.


Phase 7.3b WP7 (2026-10-25) verifies Qwen3-VL request ownership and full-cache accounting.
Processing and tower execution run sequentially per image, checking cancellation between
images; only evaluated main/DeepStack rows are hoisted to the request's prefill child scope.
Prefill evaluates logits and every retained KV array before closing that scope. Cached decode
retains only request-owned KV and the request-local expansion/position plan; generated image
markers do not consume images or rerun vision. `Qwen3VlGenerationTest` and `Qwen3VlMemoryTest`
cover mixed request reuse, capacity rejection, cancellation, failure cleanup and evaluated
visual release/KV growth. `Qwen3VlCacheEquivalenceTest` runs in both precision tasks and
compares four layouts with greedy/forced-marker histories against recomputed full forwards
(strict `1e-4`, separately measured TF32 `0.003`). All three suites are required by CI;
findings §11.7 records local evidence and the remaining WP8 independent-reference/artifact
acceptance gates. No native pin, generated binding, public API or golden changed.

Phase 7.3b WP8 (2026-10-26) delivers the Tier-A acceptance: `tools/hf-reference/qwen3vl_wp8.py`
(never Gradle-invoked, digest-pinned) generates a tiny full Qwen3-VL checkpoint (three decoder
layers, two DeepStack levels, `mrope_section [16, 8, 8]`, 21-token tied vocabulary, position
interpolation beyond the learned table, 24 distinct norm affines) and a six-case golden (the
four plan cases plus two forced generated-marker continuations carrying every sentinel and a
wrapper-looking run). `Qwen3VlGoldenTest` (`full-float32`, strict `1e-4`) and
`Qwen3VlDefaultPrecisionTest` (separately named TF32 bound `0.0025`) compare every prefill,
cached and forced step plus the tower's merged/DeepStack rows against the golden's fresh, full,
uncached HF forward; measured maxima (`2.3245811e-6` strict / `0.0019590855` TF32) are recorded
in `req/phase6-golden-precision.md`. The model's test observer now emits one `deepstack` event
per injected level and the golden test pins that count after every step; the exact greedy-ID
assertion is degenerate on this checkpoint (recorded in the golden's `greedy_id_note`) and the
full logit-vector comparison is the binding gate. Local evidence and the shared-contract
coverage decision are in the reference findings §11.8. PR #61's macOS 26 CI run 321 executed
both required suites with no skips or failures; both modes measured the identical
`2.3245811e-6` there, so TF32 was not active on that runner and the `0.0025` default-mode
bound is supported only by the local M5 Max measurement. The remainder of WP8 closed locally
on 2026-10-26 (findings §11.8): `tools/tier-b/qwen3vl-2b-instruct.json` pins the 2B Tier-B
(the required multi-image prompt, the CPU float32 reference generator
`tools/hf-reference/qwen3vl_tier_b.py` with its binary logits golden and the case's
`expected_text` decoded-output pin, `TierBQwen3VlTest` selected by task
`qwen3-vl-generation`, which asserts the exact greedy IDs **and** the end-to-end
`generate`'s exact decoded output; measured allocator peak 9,828,890,788 B / sampled
footprint peak 11,979,116,480 B / load ~843 ms / generation ~247 ms); the `tier-b.yml`
qwen3vl row runs on the self-hosted `self-macos-xl` runner (GitHub's larger
`macos-26-xlarge` tier is billed per minute even in public repos, so the row does not
target it); the runner executes as a dedicated non-privileged macOS user with a per-job
`GRADLE_USER_HOME`, the repo requires approval for every external fork-PR run, and the
runner is the same M5 Max host (macOS 27.0.1) as the local evidence; the row's first CI
runs with persistent model storage completed 2026-10-26 (run 40403387275, merged-head
confirmation 40409087523, `tools/tier-b/README.md`) with maximum logit error, native
allocator peak, greedy IDs and decoded output matching the local runs (wall times differ),
accepting the CI pipeline — not independent host, OS or memory-fit evidence — and the
macOS 26 hosted execution remains pending, not verified. The manual 8B
4-bit run (`Qwen3Vl8bManualProbeTest`, opt-in via `JMLX_8B_PROBE_DIR`) recorded its
image+text output, the artifact's stale-index finding, the full-dump
`preprocessor_config.json` acceptance in `Qwen3VlProcessorConfig` and asserts the
exact greedy-ID match against the independently re-run mlx-vlm probe (warm-up plus
five measured sequences, medians with spread, on a clean committed SHA) in
`req/phase6-4-benchmark.md`. No native pin, generated binding or public API changed.

## Phase 7.3c additions

Phase 7.3c is the Phi-3.5-Vision image+text milestone; `req/plans/phase7-3c-plan.md` and
`req/plans/phase7-3c-reference-findings.md` are the authoritative records for this section.
WP0 pinned `microsoft/Phi-3.5-vision-instruct` at revision
`12b77fb40b63a2c73c68243d3f767aab688a1b2a` (MIT, 15 files, 8,295,395,355 bytes, 592 stored
BF16 tensors — including the artifact's bundled trust-remote-code
`configuration/modeling/processing_phi3_v.py`, which transformers 4.57.6 has no native
`phi3_v` for, so the reference runs the bundled code). WP1 recorded the executable reference
probes: the `use_cache=False` uncached execution-mode gate (findings §4), the eager-attention
assertion on both the decoder and the CLIP tower (§5), the HD-transform preprocessing pipeline
including the pinned-wheel torch bicubic kernel's real FMA-contracted arithmetic (§6.5), the
per-chunk marker-split tokenization (§7.1), the stop-token facts (the tokenizer's own eos
32000, not `<|end|>` 32007, overriding `generation_config`'s `eos_token_id: 2`, §7.3), the
48-entry short/long NTK-Su factor tables and the 4096-boundary recompute semantics (§8), and
the `phi-3-vision-128k` shape (§9). No native pin or generated binding changed in any WP.

Phase 7.3c WP2 landed the pure-Java `Phi3V*` preprocessing port in `jmlx-vision`
(`Phi3VProcessorConfig`, `Phi3VGeometryPlan` — HD transform math with explicit rejection of
extreme aspect ratios in both orientations — `Phi3VImageProcessor` (decode `DROP_ALPHA` +
`ExifOrientationPolicy.IGNORE`), `Phi3VImageProcessorResult`), with one new Torch-exact
kernel: float32 bicubic (`F.interpolate` semantics: `align_corners=false`, no antialias,
`A=-0.75`), its provenance pinned in `jmlx-vision/NOTICE` exactly as the 7.3a LANCZOS port
pins its sources; the HD bilinear reuses the existing Pillow-exact kernel. The `phi3v`
image-oracle family (`tools/image-oracle/fixtures/phi3v/`, `run_phi3v` in `runner.py`; square
757 rows, wide 457, tall 469 through the transpose branch, and a two-image case) is
byte-exact, with the shared processed-pixel bins digest-pinned in both
`tools/image-oracle/provenance.json` and `tools/hf-reference/provenance.json`; no new Gradle
tasks were added — the existing `verifyImageOracleFixtureDigests` and
`verifyHfReferenceGoldens` guards pick the family up, and `ImageOraclePhi3VTest` (part of
`:jmlx-vision:test`) is the Java-side byte verification. WP3 landed the prompt-expansion
contract in `jmlx-models` (`Phi3VPromptTokens`/`Phi3VPromptExpander`/`Phi3VPromptPlan`): each
image part renders as `<|image_i|>` (i assigned in prompt order from 1), the rendered text is
split on the marker pattern and each chunk is tokenized separately through `HfTokenizer` (no
BPE merge crosses a marker; the reference's `offset = 0` strips nothing, so the goldens pin
the mid-sequence `<s>` start), and marker `i` is replaced with `num_img_tokens(i)` sentinel
rows, plus the jmlx-side hardening rejections (non-continuous marker indices, marker/image
count mismatch, duplicate or out-of-order markers, a user-literal `<|image_\d+|>`, a part
with no geometry, expanded-length overflow); a literal `<|image|>` is passed through as an
ordinary token (the marker regex never matches it). The artifact's tokenizer shape
(BPE with `byte_fallback`/`fuse_unk`, `Replace`+`ByteFallback`+`Fuse`+`Strip(start=1)`
decoder, no pre-tokenizer, `Prepend(▁)+Replace` normalizer, `TemplateProcessing` BOS) is
verified byte-for-byte against it.

Phase 7.3c WP4 landed the sealed `RopeSpec.Su` variant in `jmlx-core`'s `se.alipsa.jmlx.nn`
(per-call factor-table selection against `originalMaxPositions`, the per-dimension factor
product, the cos/sin multiplier through the existing `attentionScaling()` hook,
`isStatic() = false` so the table is re-selected per call rather than shared once;
`frequencies()` computes in `double` and casts to float32 — a stated tolerance, not
bit-exactness). Core unit tests are hand-computed at several positions plus the 4096/4097
boundary with a differential case against a regenerated `tools/hf-reference` golden (never
Gradle-invoked). Adding the sealed variant is the one source-compatibility exception in the
milestone (the `permits` clause and the default-method `instanceof` chains; no in-repo
exhaustive `switch` over `RopeSpec` exists). No `mlx_h` addition, no inventory status change.

Phase 7.3c WP5 landed `Phi3VDescriptor` (the 31-key strict top-level allow-list with the
quantization keys rejected by name through it, the pinned `img_processor` and `embd_layer`
block validation, the five-key strip and re-parse through `ArchitectureMappings` with a
throwing unknown-key sink, the `su`-rope requirement behind a `phi3_v`-only family gate
— `parseRope` is shared by all families and older Phi-3-mini-128k configs really did use
`type: su` —, the relative-key visionKeys mapping to the tower parameter paths, and the
tensor plan: 11 fixed + executed-layer keys required, the final layer and `post_layernorm`
optional, `patch_embedding.bias` forbidden), `Phi3VVisionTower` (composed from existing core
facades only: bias-free conv2d patch embed, the class token and the 577×1024 learned position
table, the `pre_layrnorm` pre-encoder norm (HF's typo), 23 pre-LN `BidirectionalAttention`
blocks with `QUICK_GELU`, the `hidden_states[-2]` slice, the 2×2 merge, the newline/global
separator addition, and the exact-`GELU` MLP projector; the per-image staged evaluate-and-
release loop of WP7), and `Phi3VModel` (a `TextGenerationModel`, not a `DecoderModel`, so the
scheduler's existing non-`DecoderModel` rejection applies; image rows are injected at the
decoder embedding entry point by row-replacement of the sentinel rows — the 7.3a/7.3b seam,
bit-identical to the plain decoder when a request has no images). The generation loop
implements the 4096-boundary recompute (mirroring `modeling_phi3_v.py:1650`–`:1655`): at the
first decode step whose ending position exceeds `originalMaxPositions` while the accumulated
past length (jmlx's own cache length) is `<= originalMaxPositions`, it discards the request's
KV cache and re-prefills the full history with the long factor table, re-injecting the
retained per-image projection rows (the reference re-runs the tower at the crossing; the
evaluated rows, at most ≈ 9.6 MB per image, are hoisted to the request scope and retained
until generation ends because of this). Load guards: all floating weights up-cast in place to
FLOAT32, non-floating keys rejected, the unused 24th-block and `post_layernorm` weights
accepted-and-ignored by name, `TextGenerationModels.load` strict `phi3_v` dispatch before the
generic mapping (no sibling-name rejection — no other model type claims this architecture).

Phase 7.3c WP6 (2026-10-28) landed the Tier-A synthetic golden: `tools/hf-reference/
phi3v_wp6.py` (never Gradle-invoked, digest-pinned, seed 20261025) builds the tiny
`Phi3VModel` checkpoint with the bundled modeling code (the module-level CLIP config
constant reassigned to a small `CLIPVisionConfig`, leaving every bundled code path — asserts
included — the one the artifact runs; the tower keeps hidden 1024 at shallow depth because
the bundled `reshape_hd_patches_2x2merge` asserts `L == 24 × 24 and C == 1024`; decoder
hidden 128, 2 layers, 2 heads / 1 KV, head_dim 64, `original_max_position_embeddings` 470,
`max_position_embeddings` 2048, untied vocab 21): 21-key `config.json`, 76 BF16 tensors
(77 state-dict keys under the single shared `embed_tokens`/`vision_embed_tokens.wte`
storage), `pad_token_id: 0` (the bundled default 32000 crashes on vocab 21). Every stored
weight is rounded to BF16 before saving and before any golden is computed, up-cast back to
float32 in Torch before the forwards run, so the strict gate compares Java's load path
against a reference computed from the identical rounded weights; the six-case golden
`phi3v-wp6.json` (schema `jmlx-phase7-3c-wp6-phi3v-model-v1`: text-only, wide-cross-boundary,
square, wide-long-prompt, tall, two-image — every prefill and every cached step compared
against a fresh, full, uncached forward) pins which side of 470 each case starts on, and the
wide-cross-boundary case (P = 465) exercises the WP5 re-prefill at decode step 6, including
image-row restoration. `Phi3VGoldenTest` (`full-float32`, strict `1e-4`, measured maximum
`1.5497208e-6`) and `Phi3VDefaultPrecisionTest` (default TF32, measured `9.266138e-4` under
the separately named bound `0.0012`) are both required by the macOS CI assertion, recorded in
`req/phase6-golden-precision.md`.

Phase 7.3c WP7 verifies request ownership and full-cache accounting in the 7.3b WP7 pattern
(per-image sequential processing and tower execution with cancellation checked between
images; only evaluated rows hoisted to the request's prefill child scope, the retained
projection rows to the request scope; prefill evaluates logits and every retained KV array
before that scope closes; cached decode retains only request-owned KV and the retained rows;
FULL and capacity-bounded retention only, sliding rejected at the request level). It found
and fixed a production bug: the tower originally built the whole per-image CLIP graph lazily
in the caller's scope, pinning every intermediate (787,398,705 B per-image peak); restructured
to the 7.3b WP5 per-stage evaluate-and-release pattern (per-image peak 407,175,193 B, inside
its committed 477,644,032 B bound), byte-identical in both precision modes.
`Phi3VGenerationTest` (capacity/sliding rejection, the boundary re-prefill event with its
exact width, cancellation boundaries, finish reasons) and `Phi3VMemoryTest` (per-image peak +
retention, flat active memory across repeated generations and fault-injected failures,
per-token decode slope under the 7.3b budget method) are both required by the macOS CI
assertion. No native pin, generated binding or golden changed.

Phase 7.3c WP8 (2026-10-30) delivers the Tier-B acceptance: `tools/tier-b/
phi3v-4b-instruct.json` pins the 4B artifact (the required multi-image prompt: the committed
672x336 and 336x336 images under `tools/tier-b/images/` — derived from the hash-pinned
image-oracle `phi3v` fixtures — with text between the two markers, 24 greedy tokens; the HD
grids are 457 and 757 image tokens, 35 unexpanded / 1241 expanded tokens, under the 4096
original max so the NTK-Su boundary re-prefill never fires in the Tier-B run — it is
exercised by the WP6 golden case and the WP5 unit test), with the exact stop-token set
{32000} carried in `eosTokenIds` (not `stopTokenIds`, so the final token stays in the
generated IDs the way HF keeps it) and `expected_text = decode(greedy_ids,
skip_special_tokens=True)` (both 32000 and 32007 are `special: true` in the pinned
`tokenizer.json`, and jmlx's generation loops hardcode `newIncrementalDecoder(true)`). The
CPU float32 reference generator `tools/hf-reference/phi3v_tier_b.py` (manual, digest-pinned)
loads the artifact's bundled trust-remote-code processor and model, up-casts the 592 stored
BF16 tensors to float32 (exactly the Java loader's path), and records the full-vocabulary
logit vector at every expected greedy history step as the little-endian float32 binary
golden `tools/hf-reference/goldens/phi3v-tier-b-logits.bin` (3,078,144 bytes, one full
32,064-vector per step — a documented size-driven deviation from the JSON logits goldens,
following the Qwen3-VL patch-bin precedent); every greedy step is a fresh, full, uncached
forward (the bundled modeling cannot run a cached forward under 4.57.6, findings §4).
Measured local evidence (M5 Max / macOS 27.0.1, TF32 off, pinned mlx-metal 0.31.2 / mlx-c
fba4470): maximum Java-vs-reference logit error 1.02996826171875e-4 (stable across repeated
runs), hence the recorded ε = 2e-4 / margin = 1e-4 (eligibility bound 2ε + margin = 5e-4,
about 6× below the recorded minimum top-two gap 0.003166); MLX allocator peak 31,100,376,165
B (deterministic across runs; suite bound peak + 500 MiB); sampled process footprint peak
51,421,089,584 B (suite bound 52,000,000,000 B, ~552 MB above the measured peak); load
~1133 ms, one-request generation ~5423 ms. `TierBPhi3VTest` (selected by task
`phi3v-generation`) re-verifies the manifest's image digests, re-processes each image through
`jmlx-vision`'s Phi-3.5-Vision processor and asserts the reference's per-image
`pixel_values_sha256`, the HD geometry, the rendered prompt, unexpanded and expanded IDs,
compares every cached decode/prefill logit vector under the recorded ε, re-derives the
reference's own recorded gaps and applies the stability gate to both the reference vectors
and the Java run's own measured margins, and asserts the exact greedy IDs **and** the
end-to-end `generate`'s exact decoded output. The `tier-b.yml` phi3v row runs on the
self-hosted `self-macos-xl` runner (the 8.3 GB artifact and the 51.4 GB measured footprint
exceed the standard 7 GB hosted runner, and GitHub's larger `macos-26-xlarge` tier is billed
per minute even in public repos, so the row does not target it); the runner executes as a
dedicated non-privileged macOS user with a per-job `GRADLE_USER_HOME`, the repo requires
approval for every external fork-PR run, and the runner is the same M5 Max host (macOS
27.0.1) as the local evidence. The row's first self-hosted run accepts the CI pipeline — not
independent host, OS or memory-fit evidence — and the macOS 26 hosted execution remains
pending, not verified. The documentation pass updated `req/phase6-compatibility.md` (a
per-phase 7.3c section — VLMs get per-phase sections, not matrix rows — while the text
`phi3` row's `longrope rejected` wording stays true: `longrope` is a rope_type still
rejected, `phi3_v`'s NTK-Su being implemented as `RopeSpec.Su`), `req/phase6-inference-
report.md`, `req/phase6-tier-a-fixtures.md`, `req/phase6-tier-b-artifacts.md`,
`tools/tier-b/README.md` and `req/plans/phase7-plan.md`. No native pin, generated binding
or public API changed.

The PR #68 review round (2026-10-30) landed: `Phi3VRopeReferenceTest` joined the CI
required-suite list; `RopeSpec.Su` is defensive-copy immutable with value equality (the
record default compared its factor arrays by identity); the top-level
`original_max_position_embeddings` key left the shared `CONSUMED` list and is accepted
only behind the phi3_v family's `acceptsSuRope` capability, so a llama/qwen2/phi3/mistral
config carrying it reaches the unknown-key sink instead of silently running plain RoPE;
`TorchBicubic`'s hand-written single-rounding `fma32` is `Math.fma` (Java 21's float
overload is specified as the same single-rounding fused multiply-add, byte-exact on the
`phi3v` oracle family); the WP5 load reads the CLIP intermediate size before releasing the
plan-optional last layer's weights, so a one-layer CLIP checkpoint no longer fails on a
closed handle; and `GenerationRequest.chat` retains its messages and options so
`Phi3VModel` can re-flatten structured text/image content parts to `<|image_i|>` markers
(the artifact's chat template string-concatenates `message['content']`, which structured
parts cannot survive) — plain-string content keeps the rendered-text path, and the Tier-B
manifest's literal-marker requests are unchanged. The model's re-tokenization still
always applies the reference processor's per-chunk `<s>` regardless of the request's
`PromptSpecialTokens` policy (the Tier-B logits pin; documented on the class), with
`GenerationResult.promptTokenIds()` retaining the request's original IDs per that
contract. The boundary re-prefill's projected rows, the re-prefill history and the
per-step native token readback are now conditional on a decode step actually being able
to reach `original_max_position_embeddings + 1` (`expandedLength <= original_max <=
expandedLength + maxNewTokens - 1`): when unreachable, the rows die with the prefill
scope, `hookRef` stays null and decode steps skip the readback and boxed history; and the
pixel buffer's real-slot trim moved from a Java-side `Arrays.copyOf` to a native row
slice. No native pin or generated binding changed.

The second PR #68 review round (2026-10-30) landed on the same branch: `GenerationRequest.chat`
deep-copies its messages (maps and content lists recursively, like the images' pixel buffers), the
re-flattened messages pass every message field through to the re-render (not just role/content),
and the messages path renders and encodes with the checkpoint's own tokenizer — the same tokenizer
as every other expansion path, so one prompt can never tokenize two ways; the structured-chat test
loads the model from a checkpoint copy carrying the artifact's template, mutates the caller's
message structures after request creation and still expects the unchanged expansion. The boundary
re-prefill's `BoundaryState` (active/hook/retainedRows) replaced the hook-reference array: the
crossable predicate `boundaryCrossable` is extracted and off-by-one-fixed (only `maxNewTokens - 1`
decode steps exist, so the fire condition is `expandedLength <= original_max <= expandedLength +
maxNewTokens - 2`), and the moment the re-prefill fires its retained rows are closed, the hook
dropped and the state deactivated — later steps skip the readback and boxed history. `MLX.array`
gained a `(scope, float[], offset, length, shape)` overload uploading a right-sized prefix view via
`MemorySegment.copy` (the pixel path uploads only the real slots, not the padded buffer), and
`pixels()` uses it. `BatchFacts.dynamicNtk` became `lengthDependentRope = !rope.isStatic()` in both
the scheduler cohort key and the batch preflight, covering Su and future length-dependent variants.
The dead `request.tokenizer() == null` branches in the cancellation path are gone, and the
ASCII-only `\d` deviation from Python's Unicode `\d` is recorded on `Phi3VPromptTokens.MARKER_PATTERN`.

The third PR #68 review round (2026-10-30) landed: the chat-message deep copy wraps in
insertion-order `Collections.unmodifiableMap(new LinkedHashMap<>(...))` /
`Collections.unmodifiableList(new ArrayList<>(...))` — `Map.copyOf`/`List.copyOf` rejected the
null values chat fields legitimately carry and lost key order; `RopeSpec.Su.frequencies` reads the
factor-table fields directly (the cloning accessors are for callers, not the per-step hot path);
`MLX.array`'s offset overload allocates with the `JAVA_FLOAT` element layout so the alignment is
explicit; `pixels()`'s comment now states that the one Java-heap copy happens inside
`Phi3VImageProcessorResult.values()`'s documented defensive contract; the release smoke's
`requiredNotice` also pins PyTorch's `UpSampleKernel.cpp`/`UpSample.h` SHA-256s and the BSD
3-Clause text in the three vision jars; and `ci.yml` uploads `Phi3VRopeReferenceTest`'s JUnit XML
with the other Phi3V suites.

A fourth small PR #68 follow-up landed: the chat-message deep copy preserves nested-map keys as
they are (no String cast — tool-call payloads with non-string keys never fail with a
ClassCastException; today's jinja renderer rejects them at the render boundary first), and
`RopeSpec.apply`'s batched unequal-position check uses `!isStatic()` like `DecoderModel`'s batch
preflight instead of an `instanceof DynamicNtk || instanceof Su` list.

A fifth small PR #68 follow-up landed: a checkpoint whose single plan-optional CLIP layer omits its
tensors fails `Phi3VModel.load` with a descriptive `IllegalArgumentException` rather than an opaque
NullPointerException; `DecoderModel.DecodeStep` now receives the fed token as a Java `int` (the loop
already holds it), so the Phi-3.5-Vision re-prefill history no longer reads the token back from a
native array each step; and `RopeSpec.Su.toString` prints the factor tables with `Arrays.toString`
instead of array identity hashes.

A sixth PR #68 review round landed: `Phi3VModel.load` requires the discovered CLIP depth to equal
the pinned ViT-L/14-336 depth (24) — a checkpoint that dropped the plan-optional final layer would
otherwise run the tower one block too few with silently wrong image features — and the tiny
synthetic checkpoints opt out explicitly via the `jmlx.phi3v.expectedClipLayers` test property set
on `test`/`float32GoldenTest` in `jmlx-models/build.gradle` (tierB deliberately not covered). The
boundary re-prefill now evaluates every retained K/V array like the regular prefill's contract, not
just the logits. Smaller items: the no-op full-range slice in `Phi3VVisionTower.hdRows` is gone;
the re-prefill history is pre-sized and immutable when unused; the expander's image-count mismatch
has one message and a repeated marker (`1,2,1`) is reported as a duplicate; `GenerationRequest.chat`
builds through `tokenizerBacked`; a `rope_type`-carrying su config names the
newer-transformers form in its rejection. Deferred per the reviewer's own triage: precomputing the
two Su frequency tables (performance only) and the vision-language models' shared helper
extraction (follow-up PR).

A seventh PR #68 review round landed: the `jmlx.phi3v.expectedClipLayers` system property is gone —
production loads (the public `Phi3VModel.load` and the `TextGenerationModels` dispatch) always
require the pinned 24; the tiny three-layer checkpoints load through a package-private
`load(scope, dir, expectedClipLayers)` overload, and the one dispatch test installs a package-private
`setExpectedClipLayersForTest` override around its single load. `runner.py` skips the arm64-only
`phi3v` image-oracle family with a logged skip on other architectures (the pinned torch wheel's
FMA-contracted bicubic bytes are AArch64-specific), and the macOS 26 native CI job now also runs
`verifyImageOracle`/`verifyImageOracleFixtures` so the family stays reference-verified in CI on
arm64. `ChatTemplateOptions` deep-copies `extraContext` recursively (it was one level deep, so a
nested tools-list mutation after the request was built could diverge the Phi-3V re-expansion from
`promptTokenIds`); `GenerationRequest.chat` keeps its eager build-time message deep copy — it is
the snapshot `promptTokenIds` were encoded from, and deferring it breaks the pinned mutation tests —
with the rationale documented on the method. `Phi3VImageProcessorResult` gained a ranged
`copyValuesTo` and the prefill uploads only the real-slot prefix (~2.7 MB, not the ~23 MB padded
row); the re-prefill history is a pre-sized `int[]` + length counter on `BoundaryState` (no
boxing); the marker-text-in-plain-string rule is documented as per-path on `GenerationRequest.chat`
and `Phi3VModel`; the `Phi3VModelTest` boundary test is restructured (seam reference first, no
Checkstyle suppression); and the `embd_layer.embedding_cls` presence-only check was verified
against the digest-pinned bundled modeling (the value is indexed but never read). The
vision-language shared-helper extraction remains a deferred follow-up. No native pin, generated
binding or golden changed.

An eighth PR #68 review round landed: the `phi3v` image-oracle gate in `runner.py` now matches the
exact recording platform — `is_reference_platform()` is `sys.platform == "darwin" and
platform.machine() == "arm64"` — because round 7's `is_arm64()` also matched a Linux aarch64 host,
which installs the pinned `torch==2.6.0+cpu` build (unverified for FMA contraction) rather than the
macOS arm64 wheel the committed bins were recorded with; widening the gate past the recording
platform is deferred until that Linux `+cpu` build is checked against the committed bins.
`Phi3VModel.expectedClipLayersForTest`'s Javadoc was reworded to acknowledge it is a mutable
process-wide static that production never sets (the finding was an overstatement, not a bug).
`ChatTemplateOptions` deep-copies object arrays element-wise and clones the eight primitive array
types (verified against `jmlx-jinja`'s `Values.fromHostValue`: the renderer accepts
String/Boolean/Number/array/List/Map and rejects `Set`/other collections at the render boundary),
and its Javadoc states the resulting snapshot contract; `ChatTemplateRenderOptionsTest` gained a
nested-array mutation case. `GenerationRequest.chat`'s Javadoc dropped a wrong "(round-6 design
review)" label. No native pin, generated binding or golden changed.

A ninth PR #68 review round landed: round 8's type-preserving object-array copy threw
`ArrayStoreException` for a concretely typed array (an `ArrayList[]` can't hold the unmodifiable
wrapper copy of its elements), so `ChatTemplateOptions` now converts every array — object and
primitive alike — into an unmodifiable list (the renderer converts both to the same array value),
which also stops the copy being mutable through `extraContext()`'s accessor; the recursive copy
gained an identity cycle check throwing `IllegalArgumentException("chat context value contains a
cycle")` (it previously `StackOverflowError`'d before the renderer's clear error) and an
original→copy map so a repeated reference is copied once and shared, matching the renderer's
per-render conversion. The duplicated copy logic in `GenerationRequest` was removed in favor of a
single public `ChatTemplateOptions.copyContextValue` shared by both snapshots (the message copy's
pinned null/order/non-String-key/mutation behavior is unchanged); the Javadoc dropped the wrong
"immutable" for the shared scalars, since `Number` also covers mutable types such as
`AtomicInteger`, which the renderer accepts. No native pin, generated binding or golden changed.

A tenth PR #68 review round landed: the round-9 generic `copyContextValue` (an unchecked `(T)`
cast whose "container shape" justification no longer held, since arrays come back as lists and
concrete container types as unmodifiable wrappers) was unsafe — the compiler accepted call sites
like `String[] s = copyContextValue(new String[]{...})` that then threw `ClassCastException`. It
is replaced by two typed public methods whose return type is always true, `copyContext(Map<String,
?>) -> Map<String, Object>` and `copyMessages(List<? extends Map<String, ?>>) ->
List<Map<String, Object>>`, both delegating to one package-private `copyValue` (cycle/dedup/shape
logic unchanged), so no unchecked cast or `@SuppressWarnings` remains; the reviewer's unsafe
probes now fail to compile against the published jars. The public `GenerationRequest.chat`
Javadoc gained a sentence recording that the message snapshot now copies arrays into unmodifiable
lists and rejects cycles (a backstop — the render above the copy reaches that check first). No
native pin, generated binding or golden changed.
