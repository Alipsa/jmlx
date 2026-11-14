# Decode baseline tooling (Phase 8.2)

The frozen workload manifest, Python worker sources, hash-locked Python environment
and pre-downloaded checkpoints for the Phase 8.2 decode baseline against `mlx_lm`
(plan: `req/plans/phase8-2-plan.md`; measurement narrative:
`req/plans/phase8-2-reference-findings.md` sections 15–17; published results:
`req/phase8-2-benchmark.md`, with the raw evidence under `req/benchmarks/phase8-2/`).

This is a measurement tool, not a product: nothing here runs in CI's ordinary
`check` (the only in-`check` consumer is the pure-Java
`verifyDecodeBaselineManifest` digest guard), and no performance claim of any
release comes from it — Phase 8.2 publishes measurements only; optimization is
Phase 9.1.

## Contents

| Path | Role |
| --- | --- |
| `workloads.json` | Frozen workload manifest, schema `jmlx-decode-baseline-workloads-1`, SHA-256 `e8c7730178a2e8f1bf1ace461db14119d2394fff4f60fa90deca34ef4857a270` (re-hashed by every worker and by the pure-Java `verifyDecodeBaselineManifest` guard). Pins: the three text checkpoints (repository, revision, per-file SHA-256, bytes, license), the two VLM follow-up cases (re-pinned through their Tier-B manifests), the frozen prompts with canonical SHA-256s, the run policy, and the worker source digests (`worker_sources`). |
| `worker.py` | Frozen Python worker (digest `8f002eac…`, `worker_sources` in the manifest). Implements the controlled and stock adapters, the request mix, the report schema and the per-worker identity/digest verification. Runs isolated (`-I`) from the pinned venv. |
| `controlled.py` | Frozen Python controlled adapter (digest `1c5e32da…`): single full-prompt forward, last-position-only projection, 127 cached steps. |
| `install.sh` / `verify.sh` | Create / verify the hash-locked CPython 3.12 venv (see Setup). |
| `requirements.in` / `requirements.lock` | The pinned environment: `mlx-lm==0.32.2`, `mlx==0.32.4`, `mlx-metal==0.32.4`, `transformers==5.22.1`, `Pillow==12.3.0`, `numpy==2.5.3`, `safetensors==0.8.0`, `tokenizers==0.23.2` (hash-locked wheels, macOS arm64). |
| `installed-source-digests.json` | The WP0 source freeze of the installed `mlx_lm` package: every `*.py` file's SHA-256 and the complete file set. `verify.sh` and every Python worker re-check it **before** importing `mlx_lm`, so a drifted or tampered tree fails without being loaded. |
| `artifacts/<case>/` | The three text checkpoints, downloaded in an explicit setup step (see below) and byte-pinned by the manifest. Git-ignored (like every other checkpoint download in this repo); no network access at run time — every worker re-hashes every file it consumes and aborts on mismatch. |
| `.venv/` | Local, git-ignored venv created by `install.sh`. |

## Requirements

- macOS on Apple Silicon (the venv locks are `macosx_*_arm64` only); Java 25 and
  the staged native runtime (`./scripts/bootstrap-native.sh`, i.e.
  `native/install/lib/mlx.metallib` present).
- CPython 3.12 on `PATH` as `python3.12` (override with `PYTHON_BIN=...`).
- Xcode Command Line Tools `cc` for the FFM calibration task only.
- A quiet host (plan section 4: no intentional competing benchmark/build/serving
  activity; every worker is bracketed by `tools/gpu-burn/hostrec.sh` records —
  observations, not guarantees).

## Setup

```sh
./tools/decode-baseline/install.sh    # creates tools/decode-baseline/.venv from the hash lock
./tools/decode-baseline/verify.sh     # re-checks pins, the mlx_lm source freeze, records device info
```

`verify.sh` prints the pinned distribution versions, re-verifies every installed
`mlx_lm` source digest against the freeze, imports the stack and records the
default device and `device_info` (the WP0 device record in the findings §4.1
came from this command).

The three text checkpoints are **not committed** (they are git-ignored, like every
other checkpoint download in this repo). Download them in an explicit setup step,
pinned to the exact revision recorded in the manifest, with the Hugging Face CLI
(`pip install -U "huggingface_hub[cli]"` for the `hf` entry point):

```sh
hf download Qwen/Qwen3-0.6B        --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir tools/decode-baseline/artifacts/qwen3-0.6b
hf download M4-ai/TinyMistral-6x248M --revision 2b4c0793c2e1dc58df51b9a6ce6852ba28e1a72c --local-dir tools/decode-baseline/artifacts/tinymistral-6x248m
hf download mlx-community/Qwen3-8B-4bit --revision 545dc4251c05440727734bcd94334791f6ab0192 --local-dir tools/decode-baseline/artifacts/qwen3-8b-4bit
```

The WP4 VLM checkpoints are the Tier-B artifacts named in the manifest's
`vlm_cases` (re-pinned through `tools/tier-b/qwen3vl-2b-instruct.json` and
`tools/tier-b/phi3v-4b-instruct.json`); obtain them the same way at their pinned
revisions. Every worker re-hashes every checkpoint file it consumes against the
manifest's per-file digests and aborts on mismatch, so a wrong or drifted download
fails the worker rather than producing a report.

## Running the text baseline (WP3)

```sh
./gradlew :jmlx-benchmarks:benchmarkBaseline --args='<spec.json>'
```

The launcher (`se.alipsa.jmlx.benchmarks.BaselineBenchmark`) consumes one spec
JSON, validates it and the manifest against the repository **before any fork**,
then forks one worker process per spec entry, sequentially, in the spec's fork
order. Exit codes: `0` all workers ok, `1` spec validation failure (nothing
forked), `2` at least one worker failed/timed out (the session is assembled from
the reports it could read and the failures are recorded).

Spec fields (`jmlx-decode-baseline-spec`):

| Field | Rule |
| --- | --- |
| `out_root` | required; the session output root (one directory per worker). |
| `repo_dir` | required; a checkout containing `tools/decode-baseline/workloads.json`. |
| `precision_env` | `"1"` or `"0"` (`MLX_ENABLE_TF32`); set explicitly in every fresh worker before native initialization. |
| `java_worker_classpath` | the Java worker's runtime classpath, or the `"@models-test"` alias, which the `benchmarkBaseline` task resolves to the `jmlx-models` test runtime classpath (the benchmark-only adapters are test classes, deliberately not published API). |
| `orchestration` | `spec-order` (default) or `alternating` (consecutive workers must use different runtimes). |
| `artifact_dirs` | map of case name → directory of the pre-downloaded checkpoint (must exist). |
| `workers` | nonempty array of `{runtime: java|python, adapter: controlled|stock, case, prompt_length, mode: headline|fixed-continuation|reference-pair, instrumentation: none|binding-count|tracing}`. Python workers are never instrumented. A `binding-count` Java worker additionally requires the `jmlx.baseline.binding.agent.jar` property (set by the `benchmarkBaseline` task); its absence is a spec failure. |

The Python worker is forked as `tools/decode-baseline/.venv/bin/python -I
worker.py <job.json>` (the launcher requires the venv python to exist). Each
worker re-hashes the manifest, every consumed checkpoint file, the frozen prompt
and the `mlx_lm` source freeze, records its own identity block (code SHA, pins,
precision, device, effective limits) and writes one `report.json` plus the
frozen request mix. The frozen run policy (manifest `run_policy`): 1 first
request + 5 warmups + 20 measured requests per worker, 128 emitted tokens,
6 pairs per cell, `worker_timeout_s 1800`, `cell_time_budget_s 14400`,
`session_time_budget_s 43200` — missing or nonpositive limits are a spec
failure, never a silent default.

## Running the VLM first-inference session (WP4)

```sh
./gradlew :jmlx-benchmarks:benchmarkVlmFirstUse --args='<spec.json>'
```

The launcher (`VlmFirstUseBenchmark`) is the both-pin, both-VLM
first-inference session (plan section 5.2): it forks one
`VlmFirstUseWorker` per workload × pair × tree, where the two trees are the
current checkout (new native pin) and an isolated old-pin compatibility
worktree built at the old commit with its own staged runtime. Each worker forks
from its own tree's `installDist`, so both trees run their **own** framework
code against their **own** pin — the session is a code-and-pin comparison, not
an isolated native-pin effect. Spec fields (schema `jmlx-vlm-first-use-spec-1`):

| Field | Rule |
| --- | --- |
| `session` | required; session name. |
| `out_root` | required; must not pre-exist and must sit outside the repository and both trees. |
| `repo_root` | required; the new-tree checkout. |
| `hostrec_script` | required; the per-fork host-state recorder (the `tools/gpu-burn/hostrec.sh` conventions). |
| `trees.old` / `trees.new` | `{path, expected_tree_sha, class_dir, native_dir}`; the launcher verifies each tree's `git rev-parse` tree SHA, its `installDist` classes and its staged native lib/pin before the first fork. The two trees must differ. |
| `workloads` | `[{name, manifest, manifest_sha256, model_dir, stop_token_ids}]`; each workload is one digest-pinned Tier-B manifest (prompt, image, cache and generation policy owned by the manifest; the spec re-pins the manifest file digest). `stop_token_ids` must be explicit and nonempty per workload. |
| `pairs`, `warmups`, `measured` | positive/nonnegative; workers per tree per workload = `pairs`. |
| `worker_timeout_s`, `session_time_budget_s` | positive; finite limits, no silent defaults. |
| `precision_env`, `precision_value` | e.g. `MLX_ENABLE_TF32` / `0`. |

Per worker: model load, the **first** image+text generation captured
immediately after load in the fresh JVM (lazy loader work stays inside the
first request), correctness verification of the first request against the
Tier-B manifest reference, then the warmups and measured repetitions. All
requests are greedy under `MLX_ENABLE_TF32=0`; every measured repetition must
reproduce the first request's IDs (divergence is recorded, not hidden).

## FFM calibration (WP2 input)

```sh
./gradlew :jmlx-benchmarks:benchmarkFfmCalibration --args='output-directory [iterations calls forks warmup]'
# defaults: 200 2000 5 2000
```

Builds a no-op FFM calibration dylib under the output directory (`cc -shared
-O2`; a machine without `cc` fails with a clear error) and measures the
downcall transition cost of representative generated-binding shapes in fresh
JVMs, one shape per no-op symbol. The Phase 8.2 calibration record (5 forks ×
200 iterations × 2000 calls, warmup 2000, 1000 pooled samples per shape; 27
shapes) is embedded in `req/benchmarks/phase8-2/evidence-index.json`
(`sources.wp3.ffm_calibration_record`, with the source and binary SHA-256s);
the `build/ffm-cal/` artifact of the measurement-time run is not committed.
The estimates built from it are calibrated estimates, not a causal partition
(findings §15.4 assumptions).

## Verifying the evidence

```sh
# Root check guard (pure Java, cross-platform): re-hashes workloads.json against the pinned digest.
./gradlew verifyDecodeBaselineManifest

# Raw-to-summary verifier (pure Java, part of :jmlx-benchmarks:test, so it also runs on the Linux CI job):
./gradlew :jmlx-benchmarks:test --tests "se.alipsa.jmlx.benchmarks.Phase82SummaryVerifierTest"
```

The verifier re-derives every value in
`req/benchmarks/phase8-2/summary.json` from the committed raw trees under
`req/benchmarks/phase8-2/raw/` (458 WP3 files + 181 WP4 files, digest-pinned in
`req/benchmarks/phase8-2/evidence-index.json`; pre-normalization digests in
`req/plans/phase8-2-wp3-raw-digests.json` and
`req/plans/phase8-2-wp4-raw-digests.json`): type-7 quantiles, pair sets,
greedy-identity blocks, the FFM budget medians within the per-call
recording bound, and the WP4 pooling and paired ratios. It also asserts the
raw tree is exactly the committed digest-pinned set and contains no username
or absolute local path (one retained `rsswatch.log` line naming the
GitHub-hosted runner user, recorded in the evidence index, excepted).

## Timing boundaries (frozen; plan section 4)

Every timer is monotonic with explicit inclusion/exclusion notes in the report.
Stock and controlled results are never pooled.

- **Controlled** (both runtimes; `controlled.py` on the Python side, the
  benchmark-only adapter on the Java side): one single full-prompt forward with
  last-position-only output projection (no chunking, no next-token lookahead,
  no periodic cache clearing), then exactly 127 cached decode steps — for 128
  emitted tokens, one evaluated step per input token, each step's time and
  context position retained. `ttft_s` / `prefill_s` are the measured
  prefill/decode contract; `steps` is one entry per cached step. Token 1 is
  emitted even if it is an EOS ID (no stop termination in the controlled mode).
  Tokenization, text rendering and stdout are excluded (pretokenized input).
- **Stock** (the shipped loop on both sides): the installed `generate_step`
  split prefill (default chunk 2048: all-but-last prompt tokens as cache-only
  forwards, then one step on the last token), retaining the n+1 lookahead
  pipeline, the stock cache clearing (after every prefill chunk and at
  `n % 256 == 0`), the shipped cache with its clearing policy and the stock
  `wired_limit` context. Host-visible events are observed and all pending work
  is drained at full completion; the extra queued terminal forward is counted
  and reported, never removed. Stock TTFT includes building/submitting token 2
  before yielding token 1 and the `tree_reduce` byte-size sum of the
  `wired_limit` context entry — its boundary differs from controlled TTFT.
  Stock retains normal stop behavior on the per-case stop set (actual emitted
  counts and finish reasons are recorded) and shipped detokenization.
- **Prime (cache regime `primed-per-request`)**: before every request, a
  sequential full read of the checkpoint (the prime). `prime_s` is recorded per
  request with `prime_excluded: true` and is excluded from every measured
  interval — first, warmup and measured alike. The WP4 observed-cache regime
  primes nothing and does not clear or inspect the OS/Metal caches: the cache
  state is **observed, not reset**.
- **Streaming decode** (headline): `(N - 1) / (last_event_time -
  first_event_time)` over the host-visible token events 2 through N; token 1
  is never counted as a cached decode; all event timestamps are retained.
- **Full generation**: generation entry to terminal completion, including
  iterator/context exit and GPU drain; actual N, unused lookahead work and
  total seconds are reported separately from event-based throughput.
- **Memory**: native active/cached/peak counters and process peak RSS are
  collected separately with units, sampling interval and source; allocator
  peaks are reset at the declared boundaries after draining prior work (the
  WP4 schema `-2` worker uses `MLXMemory.resetPeak()` between the after_load
  and after_first samples; schema `-1` counters are lifetime-cumulative).
  Native bytes and RSS overlap under unified memory: never added, never called
  total process memory.

Statistics: type-7 linear-interpolation quantiles; per-pair warm median over
the 20 measured requests; cell quantiles over the pair medians of the cell's
pair set; ratio = type-7 median over the pairs of the per-pair java/python
median ratio. Never divide unrelated percentile summaries, and never treat
120 requests from six workers as 120 independent process samples.

## Report schema (frozen; findings section 13)

Job file `jmlx-decode-baseline-job-1` (launcher → worker): `schema`,
`adapter`, `runtime`, `mode`, `case`, `prompt_length`, `artifact_dir`,
`manifest_path`, `manifest_sha256`, `repo_dir`, `out_dir`, `prompt_ids` (with
canonical `sha256`), `precision_env`/value, request mix and limits (the frozen
`run_policy` values), `instrumentation` (Java only) and the worker-side host
recording hooks. Report `jmlx-decode-baseline-1`: `worker.identity` (code
SHA/branch/dirty, pins, precision, device info, effective limits, the
manifest and worker-source digests), `worker.host_state` (per-fork hostrec
records with full command outputs and missing-sample status),
`worker.binding_signatures` (binding-count workers only), `stop_token_ids`,
`load_s`/`load_state` (materialized vs lazy), and `requests[]` — each with
`index`, `kind` (first / warmup / measured), `prime_s`/`prime_excluded`,
`ttft_s`, `prefill_s`, `steps[]` (`position`, `s`; fixed-continuation adds
`argmax_id`, `top2_margin`), `event_s`, `tokens`,
`streaming_decode_tps`, `full_generation_s`, `unused_lookahead` and
divergence records. The WP4 session report schemas are
`jmlx-vlm-first-use-session-1`/`-2` with the corresponding worker report
schemas (the `-2` worker adds `workload_inputs`, the full `token_event_times_s`
arrays and phase-local peak resets).

## Licenses

The tooling itself is part of this MIT-licensed repository. The consumed
model artifacts (all byte-pinned per file in `workloads.json` with the
license recorded per case):

| Case | Repository @ revision | License |
| --- | --- | --- |
| `qwen3-0.6b` (dense) | `Qwen/Qwen3-0.6B` @ `c1899de2…` | Apache-2.0 |
| `tinymistral-6x248m` (MoE) | `M4-ai/TinyMistral-6x248M` @ `2b4c0793…` | Apache-2.0 |
| `qwen3-8b-4bit` (quantized 4-bit) | `mlx-community/Qwen3-8B-4bit` @ `545dc425…` | MIT |
| `qwen3vl-2b` (VLM, WP4) | `Qwen/Qwen3-VL-2B-Instruct` @ `89644892…`, re-pinned through `tools/tier-b/qwen3vl-2b-instruct.json` | Apache-2.0 |
| `phi3v-4b` (VLM, WP4) | `microsoft/Phi-3.5-vision-instruct` @ `12b77fb4…`, re-pinned through `tools/tier-b/phi3v-4b-instruct.json` | MIT |

The WP4 test images are synthetic project images (project MIT license,
provenance recorded in the Tier-B manifests). The pinned Python distributions
carry their own licenses in the venv (`requirements.lock` pins the exact
wheel digests).

## Capacity

- Checkpoints on disk: the three text artifacts under `artifacts/` are
  1,503,300,328 + 4,012,327,872 + 4,607,835,174 bytes of safetensors plus the
  pinned support files (~10.1 GB total); the WP4 VLM checkpoints are the
  Tier-B artifacts (max 4,266,647,442 B and 8,295,395,355 B per manifest).
  Total ≈ 23 GB of model files plus the venv and the session outputs.
- The 4.6 GB 4-bit checkpoint is memory-mapped. On the 128 GiB measurement
  host its page-cache pages are evicted during the 2048-length runs, and
  weight re-reads fall to SSD speed (~1.75 s per decode step versus ~13 ms
  compute-bound at 512 — findings §7.3). Those cells are marked `io_affected`
  in the manifest and in the published report; running the 2048-length 4-bit
  cells on a host without adequate free RAM/page cache reproduces the same
  regime and is not a defect of the workers.
- Measured MLX allocator peaks (the published report §4/§6): up to
  21,570 MB at `qwen3-8b-4bit/2048` (text, binding-count worker) and
  31,100,376,165 B for the phi3v-4b first forward (WP4). The WP4 sessions
  need the 128 GiB-class host.
- Session outputs are small relative to the inputs: 458 WP3 files and 181
  WP4 files (~104 MiB committed, digest-pinned).

## Cold / first-use terminology

- **First request**: the generation captured immediately after load in a
  fresh JVM, before any warmup or probe — lazy loader work (weight
  materialization, shader compilation, allocator setup) stays inside it by
  construction, because no prior inference in the process has triggered it.
- **Warm**: the post-warmup measurements of the same process (the 20 measured
  requests, or the 5 measured repetitions in WP4).
- **No cold-system claim**: no result in the published report establishes a
  cold-system bound. The OS and Metal caches were neither cleared nor
  inspected in the WP4 observed-cache regime (they are recorded as observed,
  not reset), and the WP3 prime excludes checkpoint-read time from every
  measured interval. "First use" in the report titles is the first-in-process
  sense above, and the findings §17 attribution limits apply to every
  first-use comparison.
