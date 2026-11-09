# jmlx-benchmarks

Opt-in benchmarks for local Apple Silicon inference. This unpublished module
hosts performance experiments independently of examples and library tests.
Building or checking it never runs an inference benchmark or downloads a model.
Java 25 and the staged native runtime are required to execute benchmarks.

## TF32 cost

```sh
./gradlew :jmlx-benchmarks:benchmarkTf32 --args='tools/hf-reference/goldens/checkpoints/llama build/benchmarks/tf32-llama'
```

Arguments:

```text
checkpoint output-directory [prompt-length decode-steps samples warmups forks batch matmul-size]
```

Defaults are a 32-token prompt, 16 cached decode steps, 100 timed samples after
20 warmup samples per JVM, 5 pairs of fresh JVMs, batch size 1, and square matrix size 2048. Each pair runs
with `MLX_ENABLE_TF32=1` and `0`, reversing order on alternating pairs. All MLX
work stays on one platform thread; workers run sequentially. The coordinator
explicitly sets each worker's precision environment before native initialization.

For a larger workload or batched decode, use a local checkpoint and tune dimensions:

```sh
./gradlew :jmlx-benchmarks:benchmarkTf32 --args='/path/to/model build/benchmarks/tf32-model 512 64 20 5 3 4'
```

Keep prompt length plus decode steps within the model's supported context.
Inputs are deterministic token IDs, identical across precision modes. Fixed
continuations prevent a numerical change from selecting different tokens and
altering the workload. Caches are fresh for every sample.

The report contains separate float32 matrix-multiplication, prefill and cached-decode timings.
The matrix case multiplies two deterministic `[size,size]` float32 arrays and
forces GPU evaluation. Allocation and input preparation are excluded, and each
sample constructs a fresh operation. This provides a GPU-focused comparison even
when a tiny model is dominated by Java and synchronization overhead. `DecoderModel.forward`
evaluates logits and caches before returning, so these measure graph construction
plus synchronized GPU execution. Prefill projects **all** prompt positions; this
is not the last-position-only prefill used by `generate`. Input preparation,
model loading, warmup and host logit reads are outside the timed regions. Decode
time is the sum of the timed forward calls for all requested continuation steps;
it excludes scope cleanup and host logit reads between steps.

Outputs:

- `comparison.json`: checkpoint hashes, native pin, CPU/OS/Java metadata, workload,
  raw samples, per-fork logit checksums and paired ratios.
- `comparison.md`: median timings and disabled/enabled ratios.
- `pair-<n>-tf32-<0|1>.json`: individual worker results.

A ratio of `1.25x` means disabling TF32 took 25% longer for that phase. The ratio
is the median of paired JVM median ratios, reducing sensitivity to drift between
pairs. Displayed times pool the samples. Inspect raw timings and repeat runs to
assess noise; there is no performance pass/fail threshold. Use a new output
directory for each experiment, and run without competing GPU workloads.

The tiny committed checkpoint is useful for smoke tests and measuring overhead.
Use representative models and shapes before drawing conclusions about inference
performance. The switch may have no effect on older hardware or non-float32
operations. The benchmark records the output dtype but does not prove which
native kernel ran or assert numerical equivalence from its checksums.

The existing decoder and batch-scheduler benchmarks remain available in
`jmlx-examples`; future performance experiments should go in this module.

## Initial M5 Max measurement

With the committed tiny float32 Llama checkpoint and the defaults above, staged
MLX 0.31.2 / mlx-c fba4470, the median paired disabled/enabled ratios were:

| Workload | Ratio |
| --- | ---: |
| 2048 × 2048 float32 matrix multiplication | 2.731x |
| 32-token prefill | 1.017x |
| 16 cached decode steps | 1.007x |

The matrix case shows a substantial precision cost on this hardware. The tiny
decoder was dominated by other overhead; its small timing differences do not
establish a meaningful inference slowdown. Results are workload-specific.
The raw local report is `build/benchmarks/tf32-m5-max/comparison.json`.

## Decode baseline (phase 8.2)

```sh
./gradlew :jmlx-benchmarks:benchmarkBaseline --args='path/to/spec.json'
```

The phase 8.2 decode-baseline launcher (frozen contract in
`req/plans/phase8-2-reference-findings.md` section 13). It consumes one spec JSON,
validates it and the frozen `tools/decode-baseline/workloads.json` manifest against
the repository, forks one worker process per spec entry in the spec's fork order
(sequentially), and writes the session report. The session directory must not
pre-exist and must be outside the repository; the launcher creates it. Every worker
gets its own launcher-created directory and job file, and failures or timeouts
produce a failed report plus a nonzero launcher exit — never a partial summary
presented as success.

Spec fields:

```text
out_root             new directory, outside the repository
repo_dir             the repository checkout
precision_env        "0" or "1"; applied to each worker process environment
java_worker_classpath  classpath of the Java worker; the alias "@models-test"
                     resolves to this task's jmlx.baseline.models.test.classpath
                     property, which points at :jmlx-models' test runtime
                     classpath (BaselineWorker is benchmark-only test source,
                     never a published API)
artifact_dirs        case name → checkpoint directory
orchestration        optional: spec-order (the default) or alternating
                     (consecutive workers must use different runtimes)
workers[]            {runtime java|python, adapter controlled|stock, case,
                     prompt_length (one of the manifest's frozen lengths),
                     mode headline|fixed-continuation|reference-pair,
                     instrumentation optional none|binding-count|tracing
                     (Java workers only; the launcher wires the jmlx-core
                     binding agent as a -javaagent for binding-count)}
```

Layout under `out_root`: one directory per worker
(`<runtime>-<adapter>-<case>-<length>-<mode>/`, with a
`-<instrumentation>` suffix when the worker is instrumented — the
instrumentation mode is part of the cell identity — holding
`worker-claim`, `worker-ready` and `report.json`), `jobs/<dir>.job.json` (the
frozen job file handed to the worker), `stderr/<dir>.log`, `native-cache/`
(the isolated classpath-extraction cache for the forked Java workers, pinned
by `-Djmlx.native.cache.path`; a guard, not a live path — every worker
command sets `jmlx.library.path` first and `NativeLoader` resolves it before
the classpath-extraction fallback, so in a bootstrapped checkout no
extraction runs and the directory stays empty; retained on a successful
session, deleted on session failure), the verbatim
`spec.json`, and `session.json` (schema `jmlx-decode-baseline-1`: the session
block, the embedded worker reports, and the summary with adapter/mode-scoped
type-7 quantiles and the exclusion list).

Instrumented workers (the spec's per-worker `instrumentation`) additionally
write, in their `report.json`: per-request `binding_counts` (binding-count
mode) or `tracing` (tracing mode), a `memory` block, and the `identity.host`
record; the worker block carries `binding_signatures` (the JVM-form
descriptors, provenance only) and `binding_descriptors` (each recorded
binding's native `FunctionDescriptor` — what the FFM estimator
classifies) in binding-count mode.
Instrumented workers are excluded from the headline aggregation by
construction (exclusion reason `instrumented`). The frozen contract and the
WP2 record are in `req/plans/phase8-2-reference-findings.md` sections 13 and
15.

Exit codes: 0 all workers ok; 1 spec validation failed (nothing forked, no
`session.json`); 2 any worker non-ok (`session.json` is still written, the failed
cells are recorded as exclusions). The Java controlled worker needs the staged
native runtime; the Python stock worker needs the pinned venv from
`tools/decode-baseline/install.sh` (installed explicitly, never by a Gradle
task). Use a new `out_root` for every experiment, and run without competing GPU
workloads.

## FFM calibration (phase 8.2 WP2)

```sh
./gradlew :jmlx-benchmarks:benchmarkFfmCalibration --args='build/ffm-cal'
```

Builds a no-op FFM calibration dylib under the output directory
(`cc -shared -O2`; a machine without `cc` fails with a clear error) and
measures the Project Panama downcall transition cost of the 27 binding
shapes the controlled binding-count workloads actually call (the union
of the native descriptor classes of the called bindings, findings
section 15.7 item A) in fresh JVMs. Arguments:

```text
output-directory [iterations calls forks warmup]
```

Defaults are 200 iterations, 2000 calls per sample, 5 forks and 2000 warmup
calls. The reported quantiles are per sample (one sample is `calls`
consecutive downcalls); the transition estimator divides by `calls` for
per-call costs. Each shape is a non-elidable no-op C function with the
same parameter/return register shape as a class of the generated
downcalls
(see `src/main/resources/ffm_cal.c` for the shapes and the carrier-fidelity
argument — the M parameters are the 8-byte structs the pinned header passes
by value, declared on the Java side with the struct's 8-byte layout exactly
as the generated wrappers do), invoked through the generated wrappers' own
`invokeExact` form: one private holder class per shape binds its handle in a
`static final` field, so each call site references a JIT-trusted constant.
The argument carriers use the same process-lifetime shared-arena
allocation kind production `MLXScope` uses, and the struct-return
results are allocated through a per-sample allocator that mirrors a
production step scope's lifecycle (created before every sample's timed
batch and closed after it, outside the timed region), so the measured
per-call cost is production's. The dylib and the extracted
C source are the only build artifacts; both stay under the output
directory. Every external-process wait is bounded
(`cc` compile 300 s, version probe 30 s, worker fork 300 s + 1 s per
million downcalls). Outputs: `build-info.json` (compiler, flags,
source/binary SHA-256s), `raw-fork-<n>.json` (the per-fork raw samples) and
`ffm-calibration.json` (per-shape p10/median/p90, min/max and every raw
sample, plus a warmed-up no-op timing-floor entry).

To estimate the per-decode-step FFM transition budget from a
binding-count worker report (the decode baseline) and a calibration summary,
run this module's `FfmTransitionEstimate` main class with its main runtime
classpath:

```text
FfmTransitionEstimate worker-report.json ffm-calibration.json output-json
```

The estimate accepts only controlled worker reports that carry
`binding_descriptors` (the stock adapter's single whole-request window
cannot be split into prefill and per-step phases, and a report
predating the native-descriptor recording is a re-run, not a
misclassification) and aggregates only the `measured` requests — one
step-budget sample per (measured request, position), so the per-step
figures never scale with the request count. It refuses a calibration
summary without a positive `workload.calls`: the calibration quantiles
are per sample, and every per-call cost divides them by `calls`. Every
summary node reports two ranges: `p10_ns`/`median_ns`/`p90_ns` are the
request-to-request spread of the per-request budgets, and
`calibration_p10_ns`/`calibration_p90_ns` are the calibration's own
sampling range. It is a calibrated estimate, not a causal partition of
the step's time: the output carries the explicit assumptions, the
coverage of classified calls and the uncovered signatures. The contract
and the local measurement baseline are in
`req/plans/phase8-2-reference-findings.md` section 15.4.
