# Phase 6 inference report

Status of local text inference in jmlx at the end of Phase 6.5, on branch `phase6-5-batching`. It
summarises evidence recorded elsewhere and links to it; nothing here upgrades a synthetic result to a
real-artifact one. **The first Maven Central release has not happened**: see
[Release status](#release-status).

## Supported surface

- Six float-safetensors decoder families: Llama, Qwen2, Mistral, Gemma v1, Phi-3, Mixtral. Support
  and verification level per row is in the [compatibility matrix](phase6-compatibility.md).
- Pure-Java Hugging Face tokenizer (`jmlx-tokenizer`) and Jinja chat templates (`jmlx-jinja`).
  `HfTokenizer` is immutable and safe for concurrent use, backed by `HfTokenizerConcurrencyTest`.
- Direct generation (`TextGenerationModel.generate`) with greedy and seeded sampling, penalties and
  top-k/top-p/min-p filtering.
- `BatchGenerationScheduler` (Phase 6.5): one MLX-owning worker batching requests into cohorts, with
  bounded admission, per-request cancellation, row-level failure isolation and a worker-owned model.
- Phase 7.2: BERT text encoder, sequence and token classification, and T5/Flan-T5 generation
  (`TextEncoderModels`, `SequenceClassifiers`, `TokenClassifiers`). See
  [phase7-2-implementation-report.md](phase7-2-implementation-report.md).
- Phase 7.3a: the published pure-Java `jmlx-vision` module (PNG/JPEG decode, Pillow-exact
  bilinear/bicubic/LANCZOS resampling, Idefics3 processor) and `SmolVlmModel` in `jmlx-models`:
  structured chat content with image parts, `GenerationRequest.withImages`, placeholder expansion,
  tower/connector prefill and cached image+text generation through `TextGenerationModels.load`
  (`model_type=idefics3`). See [phase7-3a-implementation-report.md](phase7-3a-implementation-report.md).

## Verification evidence

| Tier | What it proves | Where |
| --- | --- | --- |
| A: synthetic Hugging Face tiny checkpoints | Logits parity against the Python reference for all six families, plus text/ID goldens | [Tier-A fixtures](phase6-tier-a-fixtures.md); native CI job |
| B: pinned real artifacts | Exact 16-token greedy IDs for Llama, Qwen2 and Mistral on the recorded device/macOS/MLX pin | [Tier-B manifest](phase6-tier-b-artifacts.md); scheduled workflow |
| Scheduler | Batched rows equal independent direct runs (all six families); seeded rows independent of their companion; row-level and cohort-wide failure semantics; bounded repeated-batch memory | `jmlx-models` `BatchGenerationSchedulerTest`, `BatchSchedulerFailureTest`, `BatchSchedulerLifecycleTest`, `BatchSchedulerMemoryTest`, `BatchStepEquivalenceTest` |
| Threading | Each `MLXScope` carries its owner thread's stream; the worker stream is usable and a foreign thread's is not | `WorkerStreamProbeTest` (in `check`); `ConcurrentStreamProbeTest` is evidence only |
| Package shape | The published jars resolve and run from a clean consumer | `tools/release-smoke/run.sh ci` (CI mode); see below |

Gemma v1, Phi-3 and Mixtral remain `verified-with-synthetic-fixture`: their Tier-B artifacts are
gated, too large for the hosted runner, or not yet chosen. The scheduler has **not** been run on a
real Tier-B artifact; its equivalence tests use the synthetic checkpoints.

## Benchmarks

Method, command and results are in the
[Phase 6.5 section of the benchmark report](phase6-4-benchmark.md#phase-65-direct-versus-batched-decode),
recorded on an Apple M2 Max with the pins below. Batched throughput was between 0.88x and 1.22x the
direct path over the seven synthetic families (two below 1). Those checkpoints are tiny, so the numbers show
correctness and the absence of leaks (active bytes return to baseline), not production-size gains. A
Tier-B-size run is outstanding. No speed threshold is enforced in CI.

## Runtime pins

| Component | Pin |
| --- | --- |
| `mlx-metal` wheel | 0.31.2 (`macosx_26_0_arm64`) |
| mlx-c | `fba4470b89073180056c9ea46c443051375f7399` (tracks v0.6.0) |
| Platform | macOS 26+ on Apple Silicon, Java 25 |

## Notices and licenses

jmlx and every published module are MIT. Every published jar now carries `META-INF/LICENSE` (the
module's own, else the root's), and `META-INF/NOTICE` where the module has one:

- `jmlx-jinja`: attribution for the vendored `@huggingface/jinja` 0.5.9 source (MIT).
- `jmlx-ffi`: bindings generated from mlx-c.
- `jmlx-native-macos-arm64`: MLX and mlx-c (MIT); `libmlx.dylib` also embeds Apache-2.0 metal-cpp.
- `jmlx-vision`: the ported Pillow resampling and alpha-composite source (MIT-CMU), with
  pinned source SHA-256s and the full license text.

The release smoke asserts the license in all seven runtime jars, the notice and native payload in
the native jar, and the Pillow `NOTICE` content in the three `jmlx-vision` jars. Model weights are
never redistributed; see the download guidance in the
[`jmlx-models` README](../jmlx-models/README.md).

## Unsupported features

Each is rejected with a named error at config parsing or load time, not silently approximated.

| Feature | Behaviour |
| --- | --- |
| GGUF, GPTQ/AWQ and per-layer or non-affine MLX quantization; any quantization on Mixtral | Rejected at load with the offending key named; float and uniform MLX affine safetensors load |
| Packed (quantized) KV cache | Not supported: the pinned runtime rejects packed K/V in SDPA (see the [quantized KV probe](phase6-4-benchmark.md#quantized-kv-retention-probe)) |
| Gemma 2/3, Phi-2, Phi-3 `longrope`, shared-expert MoE | `planned`; the unsupported `model_type` or `rope_type` is named |
| Qwen2 `use_sliding_window=true` | Config key named |
| Any quantized Idefics3/SmolVLM checkpoint (affine, non-affine or per-layer) | Rejected at load ("quantized vision-language checkpoints are unsupported" for a plain affine block; malformed, per-layer or root-level blocks name the key); float safetensors only, storage FP16/BF16 up-cast |
| Sliding KV retention for image-prefill (vision-language models) | Rejected pending image-prefill equivalence evidence; FULL and capacity-bounded retention only |
| VLM batching and multimodal scheduler cohorts | Not supported; `BatchGenerationScheduler` admission rejects images before native work |
| Qwen3-VL, LLaVA-style vision models | Phase 7.3b / 7.3c, not implemented |
| Malformed SentencePiece `Precompiled` charsmap binary tables | Rejected at tokenizer load with a named `Precompiled: malformed ...` error; well-formed charsmaps decode (Phase 7.2) |
| Multi-GPU, non-Apple platforms | No native artifact |
| Concurrent MLX use from several threads | Unsupported; use the scheduler's single worker. The `Cleaner` backstops are the one known exception (see the `se.alipsa.jmlx.core` package Javadoc) |
| Mixing the direct `generate` API with a running scheduler | Documentation-only rule; not enforced |

Every `planned` or `unsupported` row in the compatibility matrix was reviewed for this report; none
claims support.

## Release status

The first release is **open**. Nothing has been published to Maven Central.

| Order | Module | Version | Smoke repository (CI mode) | Central |
| --- | --- | --- | --- | --- |
| 1 | `jmlx-jinja` | 0.6.0 | published and resolved | not published |
| 2 | `jmlx-tokenizer` | 0.1.0 | published and resolved | not published |
| 3 | `jmlx-native-macos-arm64` | 0.1.0 | published and resolved | not published |
| 4 | `jmlx-ffi` | 0.5.0 | published and resolved | not published |
| 5 | `jmlx-core` | 0.5.0 | published and resolved | not published |
| 6 | `jmlx-vision` | 0.1.0 | published and resolved | not published |
| 7 | `jmlx-models` | 0.1.0 | published and resolved | not published |

`jmlx-vision` (Phase 7.3a) starts a fresh `0.1.0` line and is the seventh published module; it
publishes before `jmlx-models`, which `api`-depends on it.

`tools/release-smoke/run.sh ci` passed locally on 2026-10-02 (Apple M2 Max, Java 25.0.3): six
coordinates from the disposable repository only, the POM-only and module-metadata runtime classpaths
equal (10 components), licenses, native payload, Javadoc and sources jars present, a fresh native
extraction into a disposable cache with no path override, and the seeded two-request batch equal to
`tools/release-smoke/goldens/mistral-sampled.properties`. Tampering with the golden fails the run.
WP8 (2026-10-13) re-ran the same `ci` consumer against all **seven** artifacts on the merged tree
(Apple M5 Max, macOS 27.0.1): seven coordinates from the disposable repository only, the POM-only
and module-metadata runtime classpaths equal (11 components), licenses across all seven runtime
jars, the native payload, the three vision jars' Pillow `NOTICE` content, seven Javadoc and seven
sources jars, a fresh native extraction, and the seeded two-request batch equal to the committed
golden. See the [WP8 section](phase7-3a-implementation-report.md) for the full run.

Still to do, by hand on macOS ARM64 with credentials (CI never publishes):

1. Set each module to its release version, in the order above, and run `run.sh candidate`.
2. For each module follow `release.sh`, plus `jmlx-jinja/req/release-checklist.md` for jinja, and
   confirm Central availability before releasing its dependents.
3. Run `run.sh central`, which resolves all seven from Central under a fresh Gradle home.
4. Only then move each module to its next `-SNAPSHOT` version, and record the final coordinates and
   the Central smoke result here.

Found and fixed while preparing the release: `jmlx-tokenizer`'s Javadoc failed under `-Werror`
(`TokenizerEncoding`'s secondary constructor had no `@param` tags), which would have failed its
first publish.
