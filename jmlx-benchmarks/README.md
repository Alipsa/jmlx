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
workers[]            {runtime java|python, adapter controlled|stock, case,
                     prompt_length (one of the manifest's frozen lengths),
                     mode headline|fixed-continuation|reference-pair}
```

Layout under `out_root`: one directory per worker
(`<runtime>-<adapter>-<case>-<length>-<mode>/` holding `worker-claim`,
`worker-ready` and `report.json`), `jobs/<dir>.job.json` (the frozen job file
handed to the worker), `stderr/<dir>.log`, the verbatim `spec.json`, and
`session.json` (schema `jmlx-decode-baseline-1`: the session block, the embedded
worker reports, and the summary with mode-scoped type-7 quantiles and the
exclusion list).

Exit codes: 0 all workers ok; 1 spec validation failed (nothing forked, no
`session.json`); 2 any worker non-ok (`session.json` is still written, the failed
cells are recorded as exclusions). The Java controlled worker needs the staged
native runtime; the Python stock worker needs the pinned venv from
`tools/decode-baseline/install.sh` (installed explicitly, never by a Gradle
task). Use a new `out_root` for every experiment, and run without competing GPU
workloads.
