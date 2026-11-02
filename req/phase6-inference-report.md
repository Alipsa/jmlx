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
   `dims % 2 != 0`, and `Phase71ContractsTest` asserts the odd-dims rejection next to the existing
   dims-2 one. jmlx kept the pre-upgrade `dims == 2` rejection at merge time (0.31.2 returned NaNs
   there and the Phase 7.1 probe rejected them deliberately). The post-merge follow-up (2026-11-02)
   relaxed that to 0.32.4's special case, where the one frequency pair's single frequency is the
   maximum frequency (the `i == 0` limit of the `1 - i / (half - 1)` schedule): the guard now
   rejects only odd or non-positive dims (message
   `SinusoidalPositionalEncoding: even positive dims and positive frequencies required`), the
   dims-2 `assertThrows` became a value assertion (position 1 with the default max frequency 1 has
   angle exactly 1.0 rad), and the oracle gained case `sinusoidal-60` (dims 2, min 0.01 / max 2)
   pinning the frequency choice — jmlx's dims validation is now exactly 0.32.4's.

### Oracle fixture input changes

Two `phase7-1-core.input.json` cases needed input-level edits:

- **`sinusoidal-60`** (dims 5) removed: 0.32.4 rejects odd-dimension
  `SinusoidalPositionalEncoding` inputs; jmlx's constructor now matches (hand-written change 6).
  Post-merge, the number was re-used for a dims-2 case pinning 0.32.4's single-frequency special
  case (the follow-up recorded in change 6).
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
needed (WP6 closed with no commit). The WP6 re-measurement of every measured bound, recorded next
to the old value, is the subsection below.

### Re-measured bounds (WP6 gate)

Every bound WP6 lists was re-measured on the new pins on 2026-11-02, on the Apple M5 Max host
(macOS 27.2 — the 0.31.2 recordings are dated to macOS 27.0.1, the host having updated in
between), on the final tree under `mlx-metal` 0.32.4 / mlx-c `a26f2d5`: each Tier-B suite in its
own fresh `:jmlx-models:tierBTest` JVM, and the precision/memory suites (all in
`se.alipsa.jmlx.models`) — `Qwen3VlDefaultPrecisionTest`, `Phi3VDefaultPrecisionTest`,
`SmolVlmDefaultPrecisionTest`, `Qwen3VlCacheEquivalenceTest`, `EncoderDefaultPrecisionTest`
and `VisionLanguageMemoryTest` — selected by their `--tests` filters, in one fresh
`:jmlx-models:test` JVM. No bound was adjusted;
the strict `full-float32` bounds were not touched. The VLM manifests'
`local_evidence`/`recorded_pin` blocks remain the dated 0.31.2 recordings — no manifest
was re-measured, so the WP6 manifest re-measure rule does not trigger — and this
subsection is the new-pin record.

**Default-mode (TF32) maxima.** Every maximum is bit-identical to its 0.31.2 recording in
`req/phase6-golden-precision.md`; each named bound held.

| Suite | Named bound | 0.31.2 (macOS 27.0.1) | 0.32.4 (macOS 27.2) |
| --- | --- | ---: | ---: |
| `Qwen3VlDefaultPrecisionTest` | `0.0025` | `0.0019590855` | `0.0019590855` |
| `Qwen3VlCacheEquivalenceTest` (TF32) | `0.003` | `0.0021870732` | `0.0021870732` |
| `SmolVlmDefaultPrecisionTest` | `0.002` | `9.0524554e-4` | `9.0524554e-4` |
| `Phi3VDefaultPrecisionTest` | `0.0012` | `9.266138e-4` | `9.266138e-4` |
| `EncoderDefaultPrecisionTest` — BERT | `0.03` (shared) | `1.4835596084594727e-4` | `1.4835596084594727e-4` |
| `EncoderDefaultPrecisionTest` — T5 ReLU | | `0.006804823875427246` | `0.006804823875427246` |
| `EncoderDefaultPrecisionTest` — T5 gated | | `0.020420074462890625` | `0.020420074462890625` |

The controlled-measurement default-mode decoder-family maxima in
`req/phase6-golden-precision.md` (which sized `BatchStepEquivalenceTest`'s default-mode `2e-3`)
are not re-recorded here: those suites print pass/fail only — their maxima came from a temporary
diagnostic removed after measurement — so for them this re-measurement is pass/fail, green in both
precision modes on the new pins (see Verification), not a re-recorded maximum.

**Tier-B native-memory gates.** All three suites pass in full (numerical comparison, margin
eligibility, exact greedy IDs and decoded output); every bound held.

| Suite | Measured | 0.31.2 (macOS 27.0.1) | 0.32.4 (macOS 27.2) | Bound |
| --- | --- | ---: | ---: | --- |
| `TierBQwen3VlTest` (Qwen3-VL 2B) | native allocator peak | 9,828,890,788 B | 9,828,892,732 B | peak + 500 MiB |
| | sampled process footprint | 11,979,116,480 B | 11,962,879,864 B | 12,500,000,000 B |
| | max logit error | `1.0228157043457031e-4` | `1.0228157043457031e-4` | ε 2e-4, margin 1e-4 |
| `TierBVisionLanguageTest` (SmolVLM 256M) | native allocator peak | 5,737,416,464 B | 5,737,416,464 B | peak + 500 MiB |
| | sampled process footprint | 7,129,368,904 B | 7,197,411,704 B | 8 GiB |
| | max logit error | `5.2928924560546875e-5` | `5.2928924560546875e-5` | ε 1e-4, margin 1e-4 |
| `TierBPhi3VTest` (Phi-3.5-Vision 4B) | native allocator peak | 31,100,376,165 B | 30,202,745,961 B | peak + 500 MiB |
| | sampled process footprint | 51,421,089,584 B | 50,465,672,648 B | 52,000,000,000 B |
| | max logit error | `1.02996826171875e-4` | `8.0108642578125e-5` | ε 2e-4, margin 1e-4 |

The tightest bound — Phi-3.5-Vision's footprint, ~552 MiB above the 0.31.2 measurement — gained
margin: 52,000,000,000 − 50,465,672,648 = 1,534,327,352 B (~1.43 GiB) above the new peak, and
its max logit error shrank from `1.02996826171875e-4` to `8.0108642578125e-5`.

**Wall times (not gated).** During re-measurement the 0.32.4 Tier-B generation
wall times sat above the 0.31.2 recordings — Qwen3-VL 247 → 855 ms (batched run), 953
and 584 ms (isolated re-runs); Phi-3.5-Vision 5423 → 31,673 ms (batched), 20,427 and
15,338 ms (isolated re-runs); loads 843 → 1326/1311/1111 ms and 1133 → 1311/1370/1255
ms — and three 0.31.2 pin-isolated controls (the pre-Phase-7.5 tree `46cb325`, the
`dims==2` work a no-op for these decoders, with the 0.31.2 pin staged, re-run three
times per model on the same macOS 27.2 host against the same checkpoints) sat at the
recorded level (Qwen3-VL 243/248/253 ms against a recorded 247; Phi-3.5-Vision
2,121/2,165/2,161 against a recorded 5,423), which at the time looked like a
2.3–3.9× / 7.1–14.9× pin effect. It did not reproduce: all the 0.32.4 runs fell in a
single 10:53–11:29 UTC window, all the 0.31.2 controls in a clean 12:23 /
13:10–13:11 UTC window, and a back-to-back alternating session (new, old, new, old,
both models, 14:10–14:12 UTC, `main` @ `801e11a` against the `46cb325` worktree)
shows no measurable difference — Qwen3-VL 261/247 ms new vs. 243/248 old,
Phi-3.5-Vision 2,175/2,118 ms new vs. 2,324/2,145 old, i.e. ratios of 0.94–1.07,
with logit errors and native peaks at their normal values in all eight runs. The
most likely cause is host state during the 10:53–11:29 UTC window (unmeasured — no
thermal, load, memory or process records exist for it; the Φ3V new-pin runs' gradual
recovery across it, 31,673 → 20,427 → 15,338, is consistent with that), not the pin;
whether 0.32.4 nonetheless degrades disproportionately under a degraded host state —
the only prediction a pin×host interaction would make — and whether a per-pin
first-use warm-up cost after the macOS 27.2 update instead explains the window,
were both settled and ruled out the same day; the resolution is recorded at the
end of this section, and issue #73 closes with it.
Every control's native peak and logit error is identical to its recorded run; the
`46cb325` test predates the `runtime_pin` report field (added by PR #71) and only
echoes the manifest's `recorded_pin`, so the running pin of these XMLs rests on the
(since removed) worktree's staged `native-pin.properties` (0.31.2 / `fba4470`) and is
corroborated by the peaks and errors being byte-identical to the 0.31.2 recordings.
Small-model decode likewise shows no evidence of a regression: the controlled
same-host decode benchmark (plan §7) —
`:jmlx-examples:benchmarkDecode` on the 259 MB llama checkpoint (`model.safetensors`
SHA-256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`), prompt
`1,7,42,3,19,5`, 32 tokens, 7 samples, 2 warm-ups, FULL policy, new-pin commit `dcd6960`
against the old-pin tree `46cb325`; the three JSON reports, which record commit, git
status and pins, are committed under `req/benchmarks/` — shows the new pin at 385.1 tok/s
median sustained throughput (median one-step decode 2.32 ms, max 2.51 ms over the 7
samples) against 62.1 and 284.2 tok/s for two old-pin runs (median one-step decode 10.9
and 3.16 ms; min/max 2.70/19.25 and 2.49/30.57 ms); the old-pin runs are too noisy (62.1
vs 284.2 tok/s) to bound a regression, and the benchmark does not exercise the VLM
prefill/vision path. Both decode benchmarks ran inside the 10:53–11:29 UTC window
(new pin 11:24, old pin 11:26 / 11:29 UTC, by the local output-file mtimes — the
JSONs carry no timestamps): the old pin stalled there too, on this light workload,
while the new-pin small decode ran clean — see issue #73 for what this implies for
the host-state and first-use explanations. All thirty-four Tier-B run XMLs and
the two issue-#73 log files (host state, Metal-cache forensics) are committed
alongside the JSONs — the six old-pin controls, the six new-pin re-measurement
runs, the eight alternating-session runs, the twelve degraded-session runs
(four clean, eight under the GPU burner) and the two cold-cache A/B runs — so
every figure above is backed by a raw record.
The new-pin XMLs' file names state a commit provenance that is consistent with the
timestamps: the isolated runs (11:17–11:29 UTC) fall between the `dcd6960` (11:07 UTC)
and `596c2b2` (11:36 UTC) commits, and the same session's benchmark JSON records
`dcd6960` with an empty git status at its own run time, while the batched runs
(10:53–10:54 UTC) predate `dcd6960` and ran on `2e553c4`, which differs from it only
in `req/` documents. The committed evidence files are as produced except the XML `hostname`
attribute (recorded as `M5-Max` rather than the local machine name) and the old-pin
JSONs' `checkpoint` path (recorded as the relative `.tier-b/llama`). Nothing is gated
on wall time. Issue #73 is resolved by the record below; the roadmap's 8.2 baseline
must still measure in a quiet host state, record the host state with the numbers,
and use the alternating methodology before attributing a VLM prefill figure to the
runtime.

**Issue #73 resolution (2026-11-02, 16:07–16:19 UTC, same M5 Max host).** A
Swift/Metal compute kernel — now committed under `tools/gpu-burn/` for the Phase
8.2 methodology — sustained GPU load while both pins alternated on both models,
with the per-run host-state recording (load, thermal-warning level, free
memory, resident background processes) that the original window lacked (the
scrubbed per-run record, the machine/process context and the burner continuity
note are committed as `req/benchmarks/phase7-5-wp6-issue73-host-state.txt`).
The burner holds the GPU saturated for the default settings (64 in-flight
command buffers of 262,144-thread × 200,000-FMA dispatches cannot drain inside
the 25 ms off-phase — over 1e14 FMA/s would be needed), so this is a constant
~100 % compute-contention regime, not a tunable duty cycle. Under it both pins
degrade by the same factor — Qwen3-VL 243/245 ms clean → 2,145/2,057 ms new vs.
2,064/2,011 ms old (8.2–8.8×, new/old 1.00–1.07); Phi-3.5-Vision 3,538/3,658 ms
clean → 13,526/13,587 ms new vs. 14,555/14,273 ms old (3.8–4.0×, new/old
0.93–0.95) — no pin×host interaction under that regime: within ±7 %, nowhere
near the window's 2.3–14.9× per-model asymmetry. The burner was one process
with unchanged settings from the 16:09:19 UTC calibration run through the
16:15:36 UTC last loaded run, so the calibration run is part of the same loaded
regime. No thermal or performance warning level was recorded in any run (load
1.3–2.2, 41–71 % free memory; the resident omlx-server local GPU model serving
and Chrome's GPU process ran throughout, as during the original window). Even
without the burner the clean phase's heavy Phi-3.5-Vision workload ran 1.5–1.7×
its 14:10 baseline on both pins (3,538/3,658 ms vs. 2,118–2,324 ms, new/old
ratio 0.97) while the light Qwen3-VL workload sat at baseline, so this live
host's ordinary ambient state already degrades heavy workloads
disproportionately, equally on both pins. All fourteen runs' logit errors and
native peaks are at their normal values. The per-pin first-use alternative is
ruled out two ways. First, forensics (raw record:
`req/benchmarks/phase7-5-wp6-issue73-metal-cache-forensics.txt`): the JDK-keyed
Metal pipeline cache (`net.java.openjdk.java/com.apple.metal{,fe}` in the
per-user cache path) received zero writes during the 10:53–11:29 UTC window —
the only post-update first-use artifact is an ~18 MB `metalfe` PCM (mtime
08:48:41 UTC, two hours before the window, in a fresh FE-hash directory;
attributed to the resident omlx-server, the only long-lived resident Java
process, started 07:43:59 UTC, 46 s after the 07:43:13 UTC boot) — and the
0.31.2 controls at 13:10–13:11 UTC wrote 4.8 MB of fresh pipeline data (mtime
13:10:45/46 UTC) while both completed at baseline (2,165/2,161 ms). Second, a
cold-cache A/B bounds what first-use can cost inside the measured walls: with
the JDK's two Metal caches moved aside — the coldest state the pin can be in —
the cold Qwen3-VL run took 5.081 s total test time against 3.802 s warm; of the
~1.3 s cold cost, ≈1.1 s lands in the untimed logit-comparison loop, which runs
the full prefill and every decode step in the same JVM before the timed
generate, so decoder-path pipeline compilation structurally cannot reach
generate_wall_ms at all, and only 86 ms (load_wall_ms 959 vs. 873) and 54 ms
(generate_wall_ms 299 vs. 245) of the cold cost fall in the measured walls —
at most 140 ms in total, below the smallest of Qwen3-VL's own window deltas
(331 ms generate / 268 ms load) and orders of magnitude below the
Phi-3.5-Vision delta (13.2–29.5 s). Question 1 — what degraded the host in the
original window — is not recoverable retroactively (nothing was recorded). The
leading candidate is GPU contention from the live host's own activity (the
resident omlx-server serving, started at the 07:43 UTC reboot): the burner
shows that GPU contention degrades both pins by the same large factor and
reaches the window's severity on the lighter workload, so the mechanism is
sufficient — but the burner's per-model pattern is inverted relative to the
window (Qwen3-VL 8.2–8.8× vs. Phi-3.5-Vision 3.8–4.0×; the window showed
2.3–3.9× vs. 7.1–14.9×), a constant compute stream against an unknown real
contention mix, so it does not identify omlx-server specifically as the cause.
The allocator-peak note is recharacterized: on 0.31.2 the native peak is
byte-identical per model across every run; on 0.32.4 the Qwen3-VL peak takes
three values ~2 KB apart (9,828,890,684 / 9,828,892,732 / 9,828,892,836 B) and
the Phi-3.5-Vision peak moves ±2 KB, the new-pin values differing from run to
run without tracking cache state or load (in this session the clean and
burner-loaded Qwen3-VL runs all landed on 9,828,890,684 while the A/B's
"warm" re-run landed on 9,828,892,732) — cause unknown, a ~0.00003 %
planning/allocator marker, not a performance concern. The macOS 27.2 update
itself is estimated from `/System/Volumes/Data`'s mtime at ≈ 2026-11-01 23:09
UTC (the sealed System volume's mtimes are build times and unusable). The
fourteen runs are committed under `req/benchmarks/` (hostname scrubbed) as
`phase7-5-wp6-tierb-degraded-clean-{new-65e5306,old-46cb325}-{qwen3vl,phi3v}.xml`
(four),
`phase7-5-wp6-tierb-degraded-loaded-{new-65e5306,old-46cb325}-{qwen3vl,phi3v}-{1,2}.xml`
(eight; the loaded Qwen3-VL new-pin run 1 is the burner calibration) and
`phase7-5-wp6-tierb-coldcache-new-65e5306-qwen3vl-{cold,warm}.xml` (two).

**`VisionLanguageMemoryTest`'s `FIXED_MARGIN`.** The 16 KiB margin is unchanged; the
retained-minus-logical overhead it was sized from measured 0 B on 0.32.4 — `VLM retention:
KV_bytes_per_position=512, features_bytes=5120, visual_release=709210, prefill_retained=23552,
early=24064, mid=47104, late=56320`, exactly 512 B × positions (46/47/92/110) at every endpoint —
against at most 11,264 B on the 0.31.2 recordings.

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
