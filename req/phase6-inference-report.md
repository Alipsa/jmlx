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
| `mlx-metal` wheel | 0.32.4 (`macosx_26_0_arm64`), SHA-256 `514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936` |
| mlx-c | `a26f2d584a14ec619ae8db9a09be03becd47598a` (tag v0.8.0, tracks MLX v0.32.4) |
| Platform | macOS 26+ on Apple Silicon, Java 25 |

Upgraded at the Phase 7.5 milestone (2026-11-01, `req/plans/phase7-5-plan.md`); the full
migration record is in [Phase 7.5 pin upgrade](#phase-75-pin-upgrade-2026-11-01) below.

## Phase 7.5 pin upgrade (2026-11-01)

The paired native pin moved from `mlx-metal` 0.31.2 / mlx-c `fba4470b89073180056c9ea46c443051375f7399`
(tracking MLX v0.31.2 / mlx-c v0.6.0) to `mlx-metal` 0.32.4 / mlx-c
`a26f2d584a14ec619ae8db9a09be03becd47598a` (tag v0.8.0, whose CMakeLists.txt pins MLX v0.32.4).
Wheel `mlx_metal-0.32.4-py3-none-macosx_26_0_arm64.whl`, SHA-256
`514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936`; the oracle's paired wheel is
`mlx-0.32.4-cp312-cp312-macosx_26_0_arm64.whl`, SHA-256
`7928869076db6267ab3ebdb253b2c80be1e738b3032c783170bd1d59d460c578`. The regenerated committed
jextract bindings, the API inventory and the header-coverage report moved with it in the same step.

### Skipped upstream releases

There is no MLX 0.31.3 or 0.31.4 release: the tags jump from v0.31.2 straight to v0.32.0. The
skipped MLX releases, and the changes that matter to jmlx:

- **v0.32.0** — the thread-local stream approach was applied to the CPU too (#3537): streams are
  now per-thread, `mlx_default_cpu_stream_new` returns the calling thread's default stream, and
  the `new_thread_unsafe_stream` API was added (#3578). NAX kernels require
  `MACOSX_DEPLOYMENT_TARGET=26.2`.
- **v0.32.1** — an inverse-CDF path was added to `categorical` sampling (#4177), selected for
  inputs whose total size equals the axis size (every non-axis extent 1, ndim > 1); 1-D inputs
  still run the gumbel-max path. The NAX attention Q@K.T loop gained `unroll_count(4)` (#3843).
- **v0.32.2** — a `force_fused` option was added to `scaled_dot_product_attention` (#4185).
- **v0.32.3** — Gated DeltaNet Metal kernels (#4020) and SDPA dispatch changes (head-dim-256
  fused path, NAX attention for long unmasked D72/D80 inputs).
- **v0.32.4** — SDPA VJP (#4563), gated delta net VJP (#4565), and the cumsum fix for a
  non-contiguous bool input (#4590) — the cumsum path jmlx calls.

mlx-c moved v0.6.0 → v0.7.0 → v0.8.0. The complete, authoritative symbol list is the regenerated
`req/mlx-api-inventory.md` (733 → 801 entries; implemented 195 → 196, the recorded `MLX_CPU`
constant) and
`req/mlx-api-header-coverage.md` (721 → 788 declarations; declared-and-bound 710 → 777). The
changes besides jmlx's two retargets (jmlx calls none of these today — the hazard is a later
milestone calling an old name that still compiles but now means a different operation):

- **Removed symbols:** `mlx_distributed_group_rank` → `mlx_distributed_group_get_rank`;
  `mlx_distributed_group_size` → `mlx_distributed_group_get_size`.
- **Same name, different semantics** (the old name still compiles but now means a no-axis /
  flattened scan; the new `_axis`/`_axes` variant carries the old per-axis behaviour):
  `mlx_cumsum` (+`mlx_cumsum_axis`), `mlx_cummax` (+`mlx_cummax_axis`), `mlx_cummin`
  (+`mlx_cummin_axis`), `mlx_cumprod` (+`mlx_cumprod_axis`), `mlx_logcumsumexp`
  (+`mlx_logcumsumexp_axis`), `mlx_median` (+`mlx_median_axes`), `mlx_trace` (+`mlx_trace_axes`).
- **Same name, new parameters:** `mlx_gather_qmm` gains `const mlx_array global_scale` before
  `sorted_indices` (matters for later quantized-MoE work); `mlx_detail_compile_clear_cache` and
  `mlx_detail_compile_erase` gain parameters; `mlx_fast_scaled_dot_product_attention` gains
  `force_fused` (this one is called — retargeted below).
- **Changed in place:** the `mlx_io_vtable` function pointers in `io_types.h`.
- **Added and unused (later-milestone candidates):** e.g. `mlx_cummax_axis`, `mlx_cummin_axis`,
  `mlx_cumprod_axis`, `mlx_logcumsumexp_axis`, `mlx_median_axes`, `mlx_trace_axes`, and
  `mlx_fast_gated_delta_update` (the Qwen3.8 Gated DeltaNet kernel — relevant to 7.6, not wired
  in here). The v0.8.0 regenerated headers add many further symbols (the header diff counts 61
  new function names).

Also upstream and Python-side: 0.32.4's `mx.random.uniform` now takes
`(low, high, shape, ...)` instead of `(shape, ...)` — jmlx calls the C API and is unaffected;
only the re-locked Python environments (`tools/mlx-oracle`, `tools/qwen-vl-reference`) needed to
account for it.

### Hand-written Java changes (six)

The plan anticipated the two call-site retargets; the migration ended at six hand-written changes.
All preserve the public API; all but the last preserve the pre-upgrade observable behaviour:

1. **`MLXOps.cumulativeSumAxis`** retargeted to `mlx_cumsum_axis` with the optional dtype
   parameter absent, reproducing the previous input-inferred dtype (the same-named
   `mlx_cumsum` is now a no-axis scan).
2. **`MLXFast.scaledDotProductAttention`** passes the new `force_fused` parameter as `false` —
   its C default — reproducing the pre-v0.8.0 call exactly; upstream's fused full-attention path
   is opt-in and must not silently change jmlx's SDPA bits.
3. **`NativeOps.resolveDefaultCpuStream`** now builds the `(cpu, 0)` device with
   `mlx_device_new_type` and creates the stream with `mlx_stream_new_thread_unsafe`. Because
   0.32.4 streams are per-thread (v0.32.0, #3537), the previous
   `mlx_default_cpu_stream_new`-sourced, process-wide cached loader stream threw "There is no
   Stream(cpu, N) in current thread" when used from any thread other than its creating one; the
   thread-unsafe stream is registered in MLX's global table, usable from any thread, and safe on
   CPU (no per-thread Metal state) under jmlx's one-MLX-thread-at-a-time rule.
4. **`MoeMlp`** (and its test-only `DenseMoeReference`) wrap the route's `takeAlongAxis`
   indices in `MLXOps.stopGradient`: 0.32.4's `GatherAxis::vjp` throws "Cannot calculate VJP
   with respect to indices" for in-graph indices, where 0.31.2 tolerated them. This is the same
   guard `SwitchGlu` already applies; the router gradient reaches the indices only through the
   gathered probabilities, so the change is numerically neutral (`MoeMlpTest`'s
   `gradientsFlowOnlyToSelectedExperts` passes unchanged in both precision modes).
   `MLXShape.takeAlongAxis`/`takeAxis` Javadoc documents the 0.32.4 constraint.
5. **`MLXRandom.categorical`** routes single-row inputs (ndim > 1 with `size == shape[axis]`)
   through a 1-D view and reshapes the result back to the axis-reduced shape. 0.32.1's
   inverse-CDF sampler (#4177) is selected for exactly those shapes and draws a *different
   token* from gumbel-max for identical values and key (probe on the identical 0.32.4 dylib:
   logits `[1, 3, 2, -inf]`, draw key `(255383827, 267815257)` selects 1 in 1-D and 2 in
   `[1,1,4]`). jmlx's decode rows are `[1,1,V]`, so they would have moved to the new sampler,
   while the 1-D oracle reference inputs stayed on gumbel-max; 0.31.2 had only the gumbel-max
   path, which is why the two agreed until now. Routing through the 1-D view reproduces the
   pre-upgrade call exactly — the same principle as change 2 — bit-identical for these shapes.

6. **`SinusoidalPositionalEncoding`** rejects odd dims at construction. 0.32.4's
   `nn.SinusoidalPositionalEncoding` requires the dims to be "positive and even"; 0.31.2 — and
   jmlx until this change — returned `dims - 1` features for odd dims. The constructor guard gains
   `dims % 2 != 0` (the message is now `SinusoidalPositionalEncoding: dims >= 4 and even, positive
   frequencies required`), and `Phase71ContractsTest` asserts the odd-dims rejection next to the
   existing dims-2 one. jmlx still rejects `dims == 2` on purpose (0.32.4 special-cases it; the
   pre-upgrade reference returned NaNs and the Phase 7.1 probe rejected it deliberately), so the
   remaining divergence from 0.32.4 is jmlx being stricter, never looser.

### Oracle fixture input changes

Two `phase7-1-core.input.json` cases needed input-level edits:

- **`sinusoidal-60`** (dims 5) removed: 0.32.4 rejects odd-dimension
  `SinusoidalPositionalEncoding` inputs; jmlx's constructor now matches (hand-written change 6).
- **`repeat_axis-105`** shape moves `[0]` → `[0,3]`: 0.32.4 keeps the untouched axis's extent in
  the repeat result (no Java change).

### Re-recorded goldens — pin lines and host measurements only

**No golden value moved.** The seeded sampling stream is bit-identical across pins, so every
"re-record" below changes only pin lines and host measurements:

- `phase6-1-sampling` and `phase6-tier-a-array` oracle fixtures: regenerated on the `macos-26`
  native job (run 41347355181, the only host satisfying the GPU profile) with the re-locked
  0.32.4 oracle — **byte-identical** to the committed 0.31.2 fixtures. MLX reproduced the GPU
  sampling and Tier-A array outputs bit-for-bit on that host.
- `phase7-1-core.expected.json`: regenerated on the 26 runner; carries only the two
  input-driven changes above (byte-identical to a local macOS 27 regeneration — the CPU
  full-float32 fixture is host-reproducible).
- `LlamaModelTest`'s seeded constants (`EXPECTED_SEED_42` / `EXPECTED_SEED_7`): **unchanged**;
  the test passes with the committed values.
- `tools/release-smoke/goldens/mistral-sampled.properties`: `run.sh ci --record` under the new
  pins rewrote only the `native.*` pin lines; `generatedIds` is bit-identical.
- The four text Tier-B manifests (qwen2.5-0.5b, smollm2-135m, lite-mistral-150m-v2,
  qwen3-0.6b): `recorded_pin` re-recorded from the `macos-26` tier-B dispatch (run 41352378744,
  all ten arms green) to
  `Apple M1 (Virtual)|macOS=26.6.2|arch=aarch64|mlx-metal=0.32.4|mlx-c=a26f2d584a14ec619ae8db9a09be03becd47598a|batch=1`,
  with that run's `observed_peak_test_jvm_rss_kib` and `observed_run`. The run's
  `generated_ids` were identical to the recorded `expected_token_ids` — **no ID moved**, and the
  exact-ID gate re-arms on the new pin.
- The VLM Tier-B manifests' `recorded_pin` sits beside their dated `local_evidence` blocks and
  stays 0.31.2 evidence; `TierBReferences.report` now appends `runtime_pin` (the same tuple
  `TierBSmokeTest` compares, now shared) so a report line can no longer be misread as
  describing a different runtime.

### Memory bounds and accounting

The allocator-tier size sweep is **byte-identical** between pins (exact ≤ 16 KiB, rounding up to
the next 16 KiB tier above, exact cleanup), so no memory-suite or memory-bound adjustment was
needed (WP6 closed with no commit).

### Verification

Local (M5 Max / macOS 27.0.1, both precision modes): `:jmlx-core:test` 444/444,
`:jmlx-models:test` 581/581, `:jmlx-core:float32GoldenTest` 30/30,
`:jmlx-models:float32GoldenTest` 56/56 — zero failures. The tokenizer and image oracle
fixtures pass unchanged; all six strict HF-referenced Tier-B runs (MiniLM, BERT, Flan-T5,
SmolVLM, Qwen3-VL, Phi-3.5-Vision) pass unchanged locally; the `macos-26` dispatch run
41352378744 ran all ten tier-B arms green on the new pin. The final gate is the `macos-26`
native CI run of `ci.yml` on the final tree (`-PmlxOracleRequireAllProfiles=true`,
required-native-suites assertion, no skips); see the PR description for the run link.

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
| `qwen3_vl_moe` and other MoE vision-language variants | Rejected at load with the `model_type` named; only the dense `qwen3_vl` and `phi3_v` decoders load (Phases 7.3b / 7.3c) |
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
