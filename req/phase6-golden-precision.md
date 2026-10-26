# M5 golden-test precision investigation

The nine default-mode failures seen while validating PR #30 are explained by
MLX 0.31.2's reduced-precision float32 GPU paths on the Apple M5 Max. They are
not evidence of a Java decoder, sliding-cache or sorted-expert routing defect:
all original tests pass with `MLX_ENABLE_TF32=0`, without changing checkpoints,
reference outputs, assertions or production Java code.

## Controlled measurements

On the same staged mlx-c fba4470 / MLX 0.31.2 runtime, temporary diagnostics
visited every array comparison instead of stopping at the first failing value.
The two runs changed only the TF32 environment setting and used fresh test JVMs.
All outputs were finite. The diagnostics were removed after measurement.

| Suite | Arrays compared | Default maximum absolute error | Full float32 maximum absolute error |
| --- | ---: | ---: | ---: |
| DecoderRefactorGoldenTest (Llama, Qwen2) | 6 | 1.1740e-3 | 1.6391e-7 |
| DecoderRefactorGoldenTest (Qwen3) | 3 | 7.7340e-4 | 2.3097e-7 |
| MistralModelTest | 3 | 9.1095e-4 | 1.6391e-7 |
| GemmaModelTest | 3 | 8.9169e-4 | 3.5763e-7 |
| Phi3ModelTest | 3 | 9.9596e-4 | 1.7881e-7 |
| Llama31ModelTest | 3 | 8.2982e-4 | 1.7881e-7 |
| MixtralModelTest | 43 | 1.0424e-3 | 2.3842e-7 |
| DecoderSlidingWindowTest | 26 | 9.1095e-4 | 1.7881e-7 |

The 87 comparisons include full prefill logits, both HF decode steps,
40 token-by-token Mixtral comparisons against a sorted long prefill, Mistral
window-boundary cases, long cached decode, and full-cache versus sliding-cache
comparisons. No element exceeds the original `1e-4` bound in full-float32 mode.
The HF fixtures were generated on CPU using float32 eager attention, as recorded
in `tools/hf-reference/README.md`.

Phase 7.0 (Qwen3, measured 2026-10-05 on the same staged mlx-c fba4470 / MLX
0.31.2 runtime and Apple M5 Max) added the qwen3 row above with the same
temporary-diagnostic method: prefill and the two decode steps of the committed
qwen3 golden, removed after measurement. The default-mode maximum (7.7340e-4)
is below the `2e-3` bound `BatchStepEquivalenceTest` uses in default mode, and
the full-float32 maximum is below the original `1e-4` golden bound.

## Native mechanism

The pinned version's
[matmul dispatch](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/matmul.cpp)
selects NAX kernels for float32 when NAX hardware is available and
`env::enable_tf32()` is true. Its
[full-attention dispatch](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/backend/metal/scaled_dot_product_attention.cpp)
uses the same precision switch. Matmul shapes and sorted gather paths can select
different kernels, explaining both prefill-vs-decode and sorted-vs-unsorted
precision differences. The experiment disables both native reduced-precision
paths; it does not isolate their individual contributions.

## Enforced test policy

The CPU float32 golden assertions remain at `1e-4`. Gradle's dedicated
`:jmlx-models:float32GoldenTest` task forces `MLX_ENABLE_TF32=0` in a separate
JVM and is wired into `check`/`build`. The ordinary `test` task excludes those
seven tagged suites and forces `MLX_ENABLE_TF32=1` for default-mode inference coverage.
`BatchStepEquivalenceTest` runs in both tasks to verify both documented bounds.
Both precision settings are explicit task inputs. CI checks the golden suites'
new results directory and requires batch equivalence to execute in both tasks.

Changing production inference precision globally would affect performance and
is not required to verify these references. Regenerating the CPU fixtures or
loosening their assertions would obscure the distinction between float32
correctness and the native default precision mode.

Run both modes with no shell environment setup:

```sh
./gradlew :jmlx-models:check
# Golden references and strict batch equivalence only:
./gradlew :jmlx-models:float32GoldenTest
```

The full module checks passed with the original assertions. Existing Checkstyle
naming warnings are unrelated. Temporary diagnostic reruns used `--rerun-tasks`
when switching the inherited environment before Gradle enforcement was added;
normal verification now selects and tracks each task's precision mode explicitly.

## Core references found during build verification

The default-mode full build also failed eleven pre-existing strict comparisons
in `DecoderAttentionTest`, `MultiHeadAttentionTest`, `MoeMlpTest` and
`SwitchGluTest`. Examples included `3.6225288` versus `3.6220703` for windowed
attention, and `4.99756` versus `4.9960938` for a composed multi-head reference.
All original core tests passed with `MLX_ENABLE_TF32=0`. The supposedly infinite
unselected-expert case still produced finite actual values; its failure was the
strict comparison with a differently computed dense oracle, not a nonfinite output.

These eleven methods now carry the `full-float32` JUnit tag and run in
`:jmlx-core:float32GoldenTest` with TF32 disabled. Other methods in the same
classes remain in the ordinary `test` JVM with TF32 explicitly enabled. The
models reference suites use the same tag at class level; its golden task also
includes the `batch-equivalence` tag. Both modules wire the golden task into
`check`/`build`, and CI requires the relevant native suites in both result
directories. New strict numerical references should use `full-float32` to select
the full-precision JVM automatically.

## Phase 7.2 precision evidence (2026-10-05)

BertGoldenTest, T5GoldenTest and EncoderAttentionTest run only in the strict float32 result
directory, with the existing absolute `1e-4` bound. CPU goldens were not rewritten to fit Metal.
EncoderDefaultPrecisionTest runs in ordinary TF32 mode with an independent `0.03` bound. On
Apple M5 Max / macOS 27.0.1 / aarch64 the measured maximum over BERT hidden states/task heads and tied-ReLU
and untied-gated T5 encoder/logit fixtures was `0.020420074462890625`. This is a default-mode
bound, not a replacement for strict reference assertions or an all-checkpoint precision claim.
Real-artifact Tier-B uses TF32 off: maximum errors were MiniLM `1.9371509552001953e-7`, SST-2
`4.76837158203125e-7`, Flan-T5 `3.814697265625e-5`, all below the separate recorded `1e-4`
real-checkpoint bound. Class/greedy IDs additionally require recorded top-two gaps above
`2 * epsilon + margin` (`margin=1e-4`); host mismatch never downgrades numerical comparison.
The new suites still need their first macOS 26 CI execution; local evidence is explicitly dated.

Per-family maxima including cached T5 histories: BERT `1.4835596084594727e-4`, T5 ReLU
`0.006804823875427246`, T5 gated `0.020420074462890625`. Each remains below the independent
default-mode bound; strict suites keep their original bound.


## Phase 7.3a WP6 local synthetic precision (2026-10-11)

`SmolVlmGoldenTest` uses the pinned tiny Idefics3 checkpoint, CPU Transformers oracle pixels,
and absolute `1e-4` for patch embeddings, positioned embeddings, tower output, connector,
merged embeddings, prefill/cached logits and three cached/full-history comparisons. The same
strict test separately composes Java preprocessing with prefill at `1e-4`. It runs only with
TF32 disabled; no goldens or strict tolerances were changed.

`SmolVlmDefaultPrecisionTest` runs the five cases through Java preprocessing and prefill plus
two cached score comparisons, including closing the visual scope before decode. The observed
maximum absolute logit error was `9.0524554e-4` with TF32 enabled on macOS 27.0.1 / Apple Silicon,
MLX 0.31.2 / mlx-c fba4470. Its separately named TF32 logit bound is `0.002`, with margin for
device variation. This is local synthetic evidence, not a real-checkpoint error bound or the
required macOS 26 CI acceptance. WP7 establishes the real-artifact gaps/error/memory gates.

## Phase 7.3a WP7 real-artifact precision (2026-10-11)

The pinned `SmolVLM-256M-Instruct` Tier-B case runs the Java image decoder and full slow-compatible
resize/split/normalize chain with FLOAT32 checkpoint computation and TF32 off. Maximum absolute
logit error versus CPU float32 eager was `5.2928924560546875e-5` over all 49,280 vocabulary scores
at prefill and three cached histories. The independent strict allowed error stays `1e-4`.
The smallest recorded top-two gap is `0.26448631286621094`; every greedy ID is exact and every gap
exceeds `2 * 1e-4 + 1e-4`. The test recomputes gaps/top-two IDs from full golden logits, so the
eligibility check cannot accept inconsistent summary fields. These are combined Java-preprocessor
end-to-end errors, separately measured from the identical-oracle-pixel synthetic model-math tests.
The opaque RGB image stays within the existing bit-exact PNG/resize/full-chain acceptance domain.
The synthetic TF32 bound remains `0.002` and strict model-math assertions remain `1e-4`.

This local run used macOS 27.0.1 / Apple Silicon, MLX 0.31.2 and mlx-c
`fba4470b89073180056c9ea46c443051375f7399`. macOS 26 CI acceptance is pending; host differences
never replace these numerical/exact-ID gates with structural checks. No existing golden was changed.


## Phase 7.3b WP7 cached/full precision (2026-10-25)

`Qwen3VlCacheEquivalenceTest` runs in both TF32 modes on the pinned two-DeepStack-level WP6
checkpoint, comparing four prompt layouts with nine greedy and forced-marker cached steps
against a fresh full-history forward (images recomputed at every reference step). Maximum
absolute cached/full error is `5.662441e-7` with TF32 off under the strict `1e-4` gate, and
`0.0021870732` with TF32 on under a separate Qwen3-VL cached/full bound of `0.003`. All greedy
IDs agree. The latter bound includes margin for reduced-precision kernel variation and does
not weaken the HF-reference gate. Independent WP6 HF goldens and their digests are
unchanged. WP8's expanded independent HF/default-mode reference acceptance is recorded
in the Phase 7.3b WP8 section below (local evidence 2026-10-26,
plus its macOS 26 CI execution).

Host: macOS 27.0.1, Apple M5 Max, MLX 0.31.2, mlx-c fba4470. Request ownership, endpoint
memory and reproduction commands are recorded in the reference findings §11.7. Supported
macOS 26 execution is checked separately by the implementation PR's required CI suites.


PR #60's [macOS 26 CI run 316](https://github.com/Alipsa/jmlx/actions/runs/40227040860)
also executed `Qwen3VlCacheEquivalenceTest` in both modes with no skips or failures (uploaded
`vision-language-native-results` artifact `12612020846`). Both modes measured the same maximum
absolute cached/full error `5.364418e-7` on that runner, so TF32 was not active there and its
default-mode run was effectively strict float32: the `1e-4` strict bound passed on a second
host, while the `0.003` default-mode bound is still supported only by the local M5 Max
measurement. The run does not establish an error bound for every Apple GPU generation.

## Phase 7.3b WP8 local acceptance precision (2026-10-26)

`Qwen3VlGoldenTest` (full float32, TF32 off) runs all six cases of the pinned WP8 acceptance
golden — one image, two different-size images, image/text/image/text, text-only, and the two
forced generated-marker continuations — through the Java processor, vision tower, prefill and
cached decode, comparing every prefill, decode and forced-step logit vector and every merged/
DeepStack feature row against the golden's fresh, full, uncached CPU-HF float32 reference. The
measured maximum absolute error is `2.3245811e-6` under the strict `1e-4` gate. The exact
greedy-ID assertion is degenerate on this 21-token tied checkpoint (token 7 at every base-case
step; top-two margins 0.4851 to 1.1170, recorded in the golden's provenance), so the full
logit-vector comparison is the binding gate.

`Qwen3VlDefaultPrecisionTest` runs the same six cases in the default (TF32 on) JVM without the
greedy-ID assertion (the reduced-precision margin is not pinned). The measured maximum absolute
logit/feature error is `0.0019590855`, under the separately named Qwen3-VL WP8 TF32 bound
`0.0025` with about 28% margin. This is local synthetic acceptance evidence against the CPU HF
reference, not a real-checkpoint error bound, and the strict float32 gate was not weakened.

Host: macOS 27.0.1, Apple M5 Max, Java 25, MLX 0.31.2, mlx-c
`fba4470b89073180056c9ea46c443051375f7399`. Commands:

```sh
./gradlew :jmlx-models:float32GoldenTest --tests "se.alipsa.jmlx.models.Qwen3VlGoldenTest"
./gradlew :jmlx-models:test --tests "se.alipsa.jmlx.models.Qwen3VlDefaultPrecisionTest"
```

PR #61's [macOS 26 CI run 321](https://github.com/Alipsa/jmlx/actions/runs/40294915661)
executed `Qwen3VlGoldenTest` and `Qwen3VlDefaultPrecisionTest` with no skips or failures
(uploaded `vision-language-native-results` artifact `12638440258`). Both modes measured the
same maximum logit/feature error `2.3245811e-6` on that runner, so TF32 was not active there
and the default-mode run was effectively full float32: the strict `1e-4` gate passed on a
second host, while the `0.0025` default-mode bound is still supported only by the local
M5 Max measurement. The run does not establish an error bound for every Apple GPU generation.
