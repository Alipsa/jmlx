# Phase 8.2 — Decode baseline against mlx_lm

- **Status:** proposed implementation plan, 2026-11-04; no benchmark executed for this plan.
- **Parent:** [Full roadmap, Phase
  8](../full-roadmap.md#phase-8--first-releases-and-framework-adapters),
  milestone 8.2; the issue #73 follow-up in that milestone is part of the exit gate.
- **Prerequisite:** reconcile Phase 7 completion and its remaining acceptance gates using
  [Phase 8.1 WP0](phase8-1-plan.md#wp0--confirm-readiness-and-record-the-release-inputs).
  Record passing evidence or dated owner waivers; do not infer readiness from implementation
  merges. The owner authorization recorded there for 8.1 tooling does not authorize 8.2
  implementation. Writing/reviewing this plan may proceed before readiness is established.
  The 8.1 publication freeze ran from the 8.1 release merge until the WP5
  development-bump PR merged (per `phase8-1-plan.md`); that PR (#84) merged at
  `42e6ff0` on 2026-11-05, so the freeze is lifted. WP0 records that merge commit as the
  freeze-lift evidence. The roadmap does not require Central publication of the pure-Java
  modules to measure a local checkout; benchmark work does not publish or change
  module versions.

## 1. Goal and scope

Produce a reproducible comparison of jmlx and Python `mlx_lm` on byte-identical checkpoint
files: one dense family, one MoE family and one separate 4-bit checkpoint. Report synchronized
prefill tokens/s, cached decode tokens/s, time to first token (TTFT), peak memory and per-token
host overhead, including generated-binding downcall counts and an explicitly qualified FFM
transition-cost estimate. Measure the current implementation without optimizing it to improve
these numbers. Phase 9.1 owns optimization.

Also complete the Phase 7.5 first-use methodology: measure actual first inference and warmed
generation on both native pins for both Qwen3-VL and Phi-3.5-Vision workloads. This is a Java
pin comparison, not an `mlx_lm` vision comparison. It must not reuse a Tier-B timer that runs
correctness forwards before the measured generation.

Deliver benchmark tooling, pinned inputs/environment records, raw samples and a release-ready
report at `req/phase8-2-benchmark.md`. Success means credible measurements and clearly bounded
claims, not beating `mlx_lm` or meeting a speed threshold. The report is an input to 8.4's
release notes and to 9.1's optimization choices.

Non-goals: native upgrades, binding regeneration, new model-family support, new quantization,
KV-cache redesign, graph compilation changes, batching/scheduler comparisons,
framework adapter work, publication, or changing existing oracle goldens/tolerances.
Ordinary users gain no Python prerequisite. Large artifacts and native benchmarking
remain opt-in.

## 2. Repository baseline and design constraints

These are inspected code/document findings, not results from executing this plan:

- `jmlx-examples`' `DecodeBenchmark` and `:jmlx-examples:benchmarkDecode` already measure
  loads, prefill, decode, generation and native memory. Direct mode emits
  `jmlx-decode-benchmark-4`; seam mode has its own schema. Preserve their existing interfaces
  and report meanings. Both direct forward timings and the TF32 worker project all prompt
  positions, unlike generation's last-position-only projection. They cannot be compared to
  another runtime's generation prefill without matching that work.
- `jmlx-benchmarks` is an unpublished module. `Tf32Benchmark` already launches fresh workers
  in alternating order and retains raw samples; reuse that orchestration/reporting approach.
  Its checkpoint hashing happens before workers, so it is not a cold page-cache harness.
- `DecoderModel.generate` uses package-private `stepLogits` for prefill and a private/internal
  cached decode path. `GenerationLoop` emits synchronous token events. A benchmark must not
  widen these methods into public API merely to insert timers.
- `MLXMemory` reports process-wide native active, peak-active and allocator-cached bytes, not
  process RSS. Reads do not evaluate lazy graphs. Its Javadoc also documents delayed Metal
  completion-handler frees; immediate post-eval active bytes need not be steady-state bytes.
- `jmlx-core`'s test-only `BindingCounterAgent` instruments actual generated downcall methods
  at runtime. It counts calls beyond `NativeOps.checked`, including creation/free helpers.
  Its existing contract rejects counting-run timings as performance evidence. Reuse/generalize
  this mechanism; never hand-edit committed generated bindings or count Java facade calls as
  if each represented one native call.
- `tools/qwen35-reference/mlx/requirements.in` pins `mlx-lm==0.32.2`, `mlx==0.32.4` and
  `mlx-metal==0.32.4`, and also `mlx-vlm==0.6.8`, with a hash lock. Qualify the compute
  pins in a dedicated text environment that omits `mlx-vlm`; the VLM dependency is not needed
  for this comparison. This existing lock is not proof that the adapters/families work.
- The current bootstrap pins MLX Metal 0.32.4 with mlx-c
  `a26f2d584a14ec619ae8db9a09be03becd47598a`. The Phase 7.5 old pair is Metal 0.31.2 with
  mlx-c `fba4470b89073180056c9ea46c443051375f7399`. Read staged pin properties and loaded
  library paths in every worker rather than trusting these strings or a checkout label alone.
- [Phase 6.4 benchmark evidence](../phase6-4-benchmark.md) supplies the reproducibility tuple
  and examples of scope/precision limitations. [The inference report](../phase6-inference-report.md)
  records issue #73's limits: the old cold/warm-cache experiment timed generation after
  correctness work, did not cover both pins/workloads, and cannot bound actual first inference.
  `tools/gpu-burn/hostrec.sh` and its README supply per-run host observation conventions.

**Harness decision:** the roadmap asks to use the existing benchmark harness. This plan
reuses its worker orchestration, measurements and reporting utilities, but deliberately adds
a new launcher and schema because matched Java/Python boundaries and adapter identities do
not fit the legacy report contracts. Preserve the old commands/schemas; record this extension
in the implementation report and 8.4 handoff rather than describing it as unchanged reuse.

**Public API and native contract:** no new public Java API or production native symbol is
planned. Keep benchmark adapters in test/benchmark source sets, including any same-package
access needed for generation-equivalent steps. A small benchmark-only C calibration library
is permitted in §5; it is not shipped in a native artifact. Preserve scopes, lazy evaluation,
streams and error handling. If a benchmark reveals a production defect, record it and fix it
in a separate reviewed change before freezing a new baseline; do not silently tune the code.

## 3. Inputs, environments and correctness qualification

### 3.1 Frozen workload manifest

Add `tools/decode-baseline/workloads.json` and its documented schema. WP0 must replace candidate
selections with exact repository revisions, file sizes and SHA-256 values before measurement.
Record all consumed configs, tokenizer/template files, indexes and shards, model license and
conversion provenance. Reuse existing digest-pinned Tier-B artifacts when appropriate; there
is no current `tools/tier-b/` Mixtral manifest to assume supplies the MoE case.

| Required case | Initial selection to qualify in WP0 | Acceptance constraint |
| --- | --- | --- |
| Dense, unquantized | Qwen3-0.6B from `tools/tier-b/qwen3-0.6b.json` | Both runtimes load the same pinned files and use matched effective compute dtype |
| MoE | First assess `M4-ai/TinyMistral-6x248M`, already evaluated in `req/phase6-tier-b-artifacts.md`; compare a standard Mixtral-8x7B if needed | Float weights only; qualify a real compatible Mixtral layout, routing, dtype, and RAM/disk/page-cache fit; synthetic fixtures alone do not satisfy this row |
| Separate 4-bit | `mlx-community/Qwen3-8B-4bit` at `545dc4251c05440727734bcd94334791f6ab0192`, with all consumed file hashes in `req/phase6-4-benchmark.md` | Same packed tensors, scale/bias values, group size and unquantized leaves; requalify on the current pin and independently cross-check outputs |
| Pin follow-up | Existing Qwen3-VL-2B and Phi-3.5-Vision-4B Tier-B manifests and their image fixtures | Same image/prompt/config bytes and generation policy for old and new native pairs |

These are qualification candidates, not newly verified artifacts. The 4-bit checkpoint is
already revision/digest-pinned; its gaps are the old 0.31.2 / `fba4470` execution pin and the
absence of an independent output comparison, not missing provenance. Carry its existing pins
into the new manifest and verify them locally before current-pin qualification.

**MoE feasibility gate:** jmlx currently supports only Mixtral-family MoE with float weights;
quantized Mixtral is rejected and `qwen3_moe` is deferred. A quantized standard Mixtral is not
a fallback. The Tier-B document records TinyMistral as Apache-2.0/public with a 4.01 GB weight
file, but leaves revision, digests, execution cost and measured peak pending. WP0 must qualify
its actual config/tensor layout and precision, or explicitly reject it with a reason; its
name and size alone do not establish loader support or representative standard-Mixtral speed.
If accepted, label results as that smaller real MoE, not Mixtral-8x7B performance.

For each MoE candidate, record parameter/header-derived resident weight bytes, load/casting
transients, worst-case KV/activation/scratch for P=2048 plus decode, runtime/OS headroom,
Metal recommended working set and wired-memory policy. Budget free disk for source shards,
setup downloads/caches and any derivation, plus reports; avoid assuming hardlinks or cache
entries consume a second full copy. Verify actual physical RAM/free disk in WP0. As a planning
estimate only, ~46.7B parameters at two bytes each imply ~93.4 GB (~87 GiB) for a standard
8x7B's BF16 weights, before all overhead; this is not an inspected artifact size. On a 128 GiB
host that leaves limited headroom, and a float32 realization would exceed RAM for weights
alone. Run only one model worker at a time and account for residual processes and page cache;
neither two resident runtimes nor two disjoint checkpoint copies can be assumed to fit.

Record expected storage reads and page-cache residency per load, measure disk-read deltas/
page faults where available, and label whether load and first-request time are likely I/O-bound.
Identical source files may share cached pages; zero-copy/mapped loading does not guarantee
residency, and load/materialization may fault data at different phases. Do not interpret I/O
or swapping as FFM cost. Freeze the load/cache policy and its limitations before timing; no
cold-disk comparison is claimed without evidence. If no compatible real MoE fits, record the
capacity blocker or a dated owner scope amendment, never substitute a synthetic fixture.

Download only in an explicit setup step using `hf download --revision <full-sha>` or the
existing manifest downloader. Workers load local, verified inputs offline. Disable remote
model code. Verify digests before accepting reports, but keep hashing/downloading outside
measured phases and state their page-cache effects. Never quantize, rewrite config, cast and
resave tensors, or repair an index differently for the two sides without a reviewed common
input derivation with complete source/output digests. File identity alone is insufficient:
record loader casts, effective parameter dtypes, tensor mappings and quantized coverage.

For each controlled text case use batch 1, full retention, no prefix reuse, greedy selection,
no penalties, no stop/EOS termination, and exactly 128 emitted tokens. Start with exact pretokenized
prompt ID arrays of lengths 32, 512 and 2048, within both models' supported context including
decode. Freeze actual IDs, hashes and validation in the manifest; no runtime-specific chat rendering
or prompt truncation. Record cache layout/capacity and prefill chunking on both sides. For the
controlled comparison disable optional compilation, speculative decoding and chunked prefill
in the custom adapter described below; this cannot be achieved merely by raising the stock
loop's `prefill_step_size`. Stock-adapter results retain their native behavior and are labeled
separately, including actual token counts and early EOS termination.

### 3.2 Python and native environment

Add a dedicated `tools/decode-baseline/` CPython 3.12 environment with direct pins, complete
hash lock, install/verify scripts and recorded installed distributions/source hashes. Qualify
the existing 0.32.2 `mlx_lm` pin against MLX/Metal 0.32.4 first; do not copy a large VLM lock
without reviewing its dependencies. Exclude `mlx-vlm` from this dedicated environment.
Ordinary Gradle checks verify committed manifests/digests without installing or executing Python. Do
not modify the other oracle environments.

Use the same MLX/Metal version for the main Java/Python comparison. Pin backend wheels and
record actual imported paths; verify that Python and Java load their intended libraries.
Set `MLX_ENABLE_TF32` explicitly in each fresh process before native initialization. Headline
runs use `1` on both sides, with effective dtypes recorded; use `0` for strict float32
qualification or a separately labeled sensitivity run. BF16 matmuls are not affected by TF32;
only cells with confirmed float32 compute can support a TF32 sensitivity interpretation.
No shell-inherited precision mode may silently select the run.

Before timing, WP0 records a per-case Java/Python dtype inventory: stored weights, effective
attention/MLP/expert matmul operands, quantized sidecars and unquantized leaves, plus any cast.
Initial expectations to verify are BF16 dense Qwen3 weights/compute, the header-declared float
dtype for the chosen MoE (not an assumed up-cast), and UINT32-packed 4-bit weights with the
artifact's floating sidecar/activation dtypes. `DecoderAssembler` only checks that each MoE
layer's expert tensors share one dtype; it neither enforces a model-wide dtype nor proves an
end-to-end float32 conversion; inspect loaded arrays and operand paths on both sides. Do
not reuse the VLM up-cast assumption for text decoders. A mismatch must be resolved or
explicitly separated before a matched-work ratio is accepted.

**Known Python loop differences and adapter decision.** The installed
`tools/qwen35-reference/mlx/.venv/lib/python3.12/site-packages/mlx_lm/generate.py`
(0.32.2, inspected 2026-11-04) establishes these differences; WP0 records that file's digest
and its relevant model/cache/helper sources in the new environment:

| Stock 0.32.2 behavior | Consequence for this comparison |
| --- | --- |
| `generate_step` consumes all but the last prompt token as cache-only forwards, then calls `_step` on the last token; default chunk size 2048 | For all planned P>1 there are at least two forwards; P=2048 is 2047+1 even with a large chunk setting. This is not Java's single full-prompt last-position prefill |
| It constructs and `async_eval`s the next token before yielding the current one | Stock TTFT includes constructing/submitting token 2 before token 1 is yielded, unlike controlled TTFT. A max-128 run constructs 128 cached forwards, including one unused final result; record evaluated work and drain it at completion |
| Host graph construction for n+1 overlaps GPU execution for n | This pipeline differs from synchronous Java generation and can affect throughput/host overhead independently of FFM |
| `stream_generate` enters `wired_limit`; `utils.maybe_set_recommended_wired_limit` sets the device's recommended limit when available and the context later restores it | Java currently has no `mlx_set_wired_limit` call in core/models/examples. Record actual policy/limit, recommended working set, and any unmatched setting |
| Cache clearing follows every prefill chunk and `n % 256 == 0`, including n=0 after the first yield | Stock cached-byte counters and allocator reuse differ from a no-clear controlled loop |

Implement two explicit Python adapters over the same pinned model weights:

- **Controlled:** a custom synchronous loop using pinned model/cache objects. Process the full
  prompt in one forward with last-position-only output projection, matching Java `stepLogits`;
  then run exactly 127 cached forwards for 128 emitted tokens. No next-token lookahead,
  `async_eval` pipeline or periodic cache clearing. Evaluate required logits/cache state at
  matched boundaries. If the pinned model projects all positions before slicing, implement
  and qualify a benchmark-only adapter that slices the last-position hidden state before
  calling the loaded model's own output head. For tied Qwen3-0.6B use its embedding's
  `as_linear`; for untied Qwen3-8B use its loaded `lm_head`, preserving any quantized module.
  Verify the 4-bit artifact's actual head quantization/sidecars in WP0 rather than assuming
  them. Never reimplement projection, substitute a dense/dequantized head or reconstruct a
  module from copied weights. Apply the same own-head rule to the chosen MoE. This mirrors
  Java `stepLogits` slicing before `embedding.project` or `lmHead.forward`. Record all custom
  source hashes and compare logits/cache evolution against the shipped model on tiny and
  real inputs. No upstream source monkeypatching.
- **Stock:** execute installed `stream_generate` without changing its loop/model source;
  use exact prompt IDs and greedy settings, but retain its split prefill, pipelining,
  lookahead, wired-limit context, cache clears and detokenization/finalization. Record normal
  EOS behavior (the resolved `TokenizerWrapper.eos_token_ids` stop set) and actual emitted
  count; the 128-token budget is a maximum here. Label these results `stock-default-loop`
  with all explicit request options, not controlled equivalence. The `wired_limit` context's
  exit synchronizes only when the device reports a recommended working-set size
  (`maybe_set_recommended_wired_limit` returned a previous limit); on the no-limit branch it
  yields without `mx.synchronize`, and `stream_generate` leaves the `generate_step` generator
  suspended rather than closed, so the unused final forward can still be pending when the
  iterator ends. End the full-generation timer with an explicit `mx.synchronize()` after the
  iterator finishes, whichever branch ran; last-token arrival and full completion are distinct
  timestamps. Count the token in the final `GenerationResponse` exactly once; its finish
  reason does not make it an extra token event.

**Stock Java pair path.** The Java side of a stock cell is the public
`DecoderModel.generate(GenerationRequest, Consumer)` with the checkpoint's own `HfTokenizer`
attached to the request, so the loop's incremental detokenization runs and is timed, matching
the stock loop's always-detokenizing behavior (`GenerationLoop` decodes text only when the
request carries a tokenizer). The request carries the pretokenized prompt IDs, full retention,
`greedyDefaults(128, eosTokenIds)` and a minimal event listener, with no extra `stopTokenIds`
and `eosTokenIds` exactly the stock worker's resolved `TokenizerWrapper.eos_token_ids` — the
union of `generation_config.json`/`config.json` `eos_token_id` and the tokenizer's
`eos_token_id` — recorded per artifact in the manifest during WP0 so both sides stop on the
same set. Both stock loops keep their own internals, so the stock table is an end-to-end
stock-versus-stock comparison: the paired ratio describes each runtime's default generation
loop, not matched per-step cost. Record actual emitted counts and finish reasons on both sides
and do not promote a stock ratio to a matched-work claim.

Run controlled and stock comparisons in separate fresh workers/matrices; do not pool their
statistics. The controlled default is that neither runtime calls a wired-limit setter:
Python does not enter `wired_limit` or call `mx.set_wired_limit`, matching Java's existing
behavior. Record the policy as `not set` on both sides and capture `mx.device_info()` (including
recommended working set) in the environment evidence. `Not set` means no explicit setter call,
not an effective limit of zero or unlimited. Effective limit remains `unknown` unless observed
independently; do not call the setter merely to query it, since `mx.set_wired_limit` returns
only the previous value and changes the setting. No benchmark-side Java wired-limit binding
call is needed. A different controlled policy requires a reviewed, separately labeled run.
For stock mode retain the shipped limit context and record its requested/previous/restored
values where observable, with effective value `unknown` where unobservable. A source-derived
requested value alone is not proof of the effective limit.
Apply the same explicit rule to allocator cache clearing; preserve stock behavior and record
its events/policy in diagnostic runs. An unmatched wired-memory regime, especially for MoE,
precludes claiming a pure language/FFM comparison.

Upstream's [benchmark source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/benchmark.py)
and [generation source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
are design references only; mutable `main` URLs are not execution pins. Freeze installed
sources in WP0. CLI throughput fields do not replace the external timing contract.

### 3.3 Untimed qualification

Before accepting timing samples, compare evaluated logits on the same prompt and fixed decode
continuation, greedy token IDs and selected-token margins. Freeze per-case dtype-appropriate
absolute/relative tolerances and the divergence policy before headline runs. Enforced float32
references retain their existing assertions. Record every mismatch; no golden regeneration or
tolerance widening to accommodate speed results.

Use both a fixed-continuation step workload (identical IDs and context growth on both sides)
and actual greedy generation. If greedy outputs diverge, reject the matched-generation ratio
for that cell pending review; retain the mismatch and the qualified fixed-continuation timing
with its narrower meaning. Stock early-EOS results remain valid observations of that loop,
but cannot supply an equal-length generation ratio. Qualification must happen in separate
processes from first-use measurements. After a first-use sample is saved, validate its outputs
against the same records.

## 4. Timing, memory and statistical contract

Add a versioned schema, initially `jmlx-decode-baseline-1`, distinct from existing benchmark
schemas. Every timer uses a monotonic clock and records exact inclusion/exclusion notes.
The controlled adapter evaluates required outputs/cache state before stopping each step timer.
Stock generation retains asynchronous overlap; observe its host-visible events and drain all
pending work at full completion, without inserting per-token synchronizations. Its internal
prompt/decode fields are stock metrics, not the controlled prefill/decode measurements below.
Report controlled matched work and stock default-loop behavior in separate tables.

| Metric | Required boundary and denominator |
| --- | --- |
| Process startup | Parent launch to worker-ready handshake; separate from model work |
| Load/setup | Worker model-load entry to loader return, with lazy/materialized state recorded; separate explicit weight materialization if performed |
| First request TTFT | Generation entry with supplied IDs to the first emitted, host-visible token; includes lazy weight realization and first-use work left by the loader policy. Stock also includes building/submitting token 2 before yielding token 1, and the `wired_limit` context entry's `tree_reduce` sum of every model array's byte size; its boundary differs from controlled TTFT |
| Warm TTFT | Same request boundary after declared warmups, fresh request cache |
| Prefill | Controlled generation-equivalent last-position projection through evaluated logits/cache, excluding sampling; prompt length divided by elapsed seconds |
| Cached decode | Fixed continuation after prefill, one evaluated step per input token; step count divided by total step time; retain each step's time and context position |
| Streaming decode | Actual greedy events 2 through N: `(N - 1) / (last_event_time - first_event_time)`; retain all event timestamps; never count token 1 as a cached decode |
| Full generation | Generation entry to terminal completion, including iterator/context exit and GPU drain; report actual N, unused lookahead work and total seconds separately from event-based throughput |

For 128 emitted tokens there are 127 cached generation steps after prefill. Controlled
continuation runs declare their own explicit number of step inputs; neither controlled
runtime may perform an extra terminal forward. The stock loop is the explicit exception:
count/report its extra queued forward rather than removing it or hiding its cost at process
exit. Controlled runs emit token 1 even if it is an EOS ID under normal settings; stock retains
normal stop behavior. Tokenization, text rendering and stdout are excluded from controlled
pretokenized timing. Stock accepts pretokenized input but retains shipped detokenization;
label that cost, including finalization of the last response. User-facing text TTFT, if added,
is a separate metric including template/tokenization work identically on both sides.

Use a minimal event listener for headline generation. Per-token memory polling, profiling,
binding instrumentation and detailed phase tracing belong in separate diagnostic processes;
mark those reports ineligible for headline throughput. Input construction, cache initialization
and cleanup boundaries must agree or be listed as differences. No production public API is
added for timing: qualify benchmark-only same-package adapters against ordinary generation
outputs and retain the actual public-generation TTFT/streaming measurements.

For memory runs, collect native active/cached/peak counters and process peak RSS separately,
with units, sampling interval and source. Reset allocator peaks at declared load and generation
boundaries after draining prior work; retain the whole-process peak too. Record weights-only
baseline, prefill peak, decode/generation peak and post-request active samples. Use bounded
post-eval stabilization for steady-state checks outside timers; report timeout/unstable status.
Python's counters must be matched against the pinned native counter semantics. Native bytes
and RSS overlap under unified memory: never add them or call native peak total process memory.
RSS sampling is a lower bound unless a platform peak counter establishes the maximum.

For each text workload cell and each Python adapter, run six fresh Java/Python process pairs,
alternating which runtime leads each pair so each runtime leads equally, serially on one host.
The Java member of a controlled pair is the `stepLogits`-equivalent adapter of §3.2; the Java
member of a stock pair is the public `generate` path named in §3.2. Each worker records
the first request, then five discarded warmups and 20 measured warm requests with fresh
request caches.
Preserve raw per-process/per-request/per-step values and iteration order. Report medians,
p10/p90 with a specified quantile convention, per-process summaries and paired ratios computed
from each process pair's warm medians. Do not divide unrelated percentile summaries or treat
120 requests from six workers as 120 independent process samples. First-request distributions
come from the fresh processes and are separate from warm statistics.

Record UTC start/end, device/chip, physical RAM, macOS version/build, Java vendor/version/flags,
Python environment, code SHA/branch/dirty status, workload/manifest hashes, runtime/library pins,
precision, allocator/cache settings, actual token counts, warmups and fork order. Require clean
committed benchmark code for accepted runs. Archive raw outputs outside the verified checkout;
a later documentation PR records that SHA without editing the measured tree during a run.

Capture host state before and after every worker: load averages, full thermal/power status,
memory pressure/swap, resident GPU-using processes and known concurrent work. Reuse the
`tools/gpu-burn/hostrec.sh` conventions but retain full command outputs and missing-sample
status, not just its warning-line count. Quiet means no intentional competing benchmark/build/
serving activity; process CPU/load observations cannot prove GPU idleness. Set observation and
retry policy and finite time limits from §4.1 before the session. Keep contaminated/failed
samples with exclusion reasons; reruns must include both members of every affected pair,
never just a favorable member. A >10% shift
between adjacent fork medians, a thermal/memory warning, missing host record, or surprising
VLM prefill triggers another alternating session before runtime attribution. The 10% rule
is an investigation trigger, not a performance acceptance bound or permission to discard data.
The trigger is subject to §4.1's rerun cap; it never creates an unlimited measurement loop.

### 4.1 Cost budget and stopping rules

WP0 must commit a per-cell wall-clock/disk budget before the full matrix, derived from a
bounded qualification pilot: measure one load, P-dependent prefill and a short decode sequence
for each real case/adapter, then extrapolate requests, process launches and data-read costs.
Initial WP0 estimates may use the existing harness and installed stock loop; refresh them
with the implemented controlled adapter during WP2 before committing the WP3/WP4 budgets.
Pilot results are planning/qualification evidence, not headline samples. For one adapter the
base matrix is 3 artifacts × 3 lengths × 6 pairs × 2 runtimes × 26 requests = 2,808 requests,
or 359,424 emitted-token slots. Both adapter comparisons budget 216 workers, 5,616 requests
and up to 718,848 tokens; stock early stops may reduce that count, not the reserved budget.
At a hypothetical aggregate 10/50/100 tokens/s this is 20.0/4.0/2.0 device-hours of decode alone,
before 5,616 prefills, loads, validation, fixed-continuation, memory and diagnostic runs.
These are arithmetic scenarios, not measured throughput or a promise of elapsed runtime.

| WP | Initial planning allowance (wall clock, not measured) | Required refinement before execution |
| --- | --- | --- |
| WP0 | 1–2 working days; up to 2 device-hours of bounded pilots, excluding network download time | Record artifact download bytes/bandwidth estimate, per-candidate memory/I/O fit and cell-time estimates |
| WP1 | 3–5 working days of implementation/verification | Include both adapters, model-output projection qualification and failure/report tests |
| WP2 | 2–3 working days; up to 2 device-hours of calibration/pilots | Budget counting, tracing, memory passes and calibration separately; freeze run counts |
| WP3 | Reserve 1–3 days of host time initially; actual request matrix may exceed this on MoE | Use pilot-based sums for both adapters, fixed-continuation, qualification and diagnostics; state a finite total time budget and per-cell limits |
| WP4 | 1–2 days including compatible builds and at least 24 VLM workers (2 workloads × 6 old/new pairs) | Freeze per-worker request count, load/first/warm estimates and any extra cache-regime budget |
| WP5 | 1–2 working days | Include archive transfer, raw-to-summary verification and review preparation; human review latency excluded |

Before WP3/WP4, the manifest/run policy must contain finite `worker_timeout`,
`cell_time_budget` and `session_time_budget` values derived from these pilots, including load
and iterator cleanup; the launcher rejects missing/nonpositive limits. Use one initial session
and at most two additional alternating sessions per cell (three total), with the same sample
counts and complete pairs in each session; a rerun repeats the cell's fixed matrix rather
than selecting individual favorable observations. A timeout/resource failure stops that
session; any restart consumes one of the remaining sessions and the same time budget.
Stop the cell at budget exhaustion or the three-session limit. Preserve partial output and
failure/timeout logs. No expensive run is started if its estimated cost exceeds the
remaining budget.

If a cell remains variable, finish it as `unstable`, retain every session and show ranges and
host observations without a settled performance ratio or runtime attribution. A timeout,
capacity failure or missing minimum sample count is `incomplete`, not `unstable` success.
Do not let MoE or a repeated >10% trigger block WP3 indefinitely. Additional sessions or a
larger budget require a recorded run-policy/plan revision before resuming, not an automatic
retry. A complete, qualified but unstable cell can close measurement collection with that
limitation; incomplete required cells need completion or a dated owner scope amendment.

## 5. Host overhead and first-use follow-up

### 5.1 Per-token host and FFM diagnostics

Run diagnostics on the same fixed inputs/configuration as qualified headline workloads, in
fresh processes. Record phase/context-local counts for prefill and each cached decode step:
all generated binding calls, creation/free, vector helpers, eval, scalar/data reads and metadata
queries. Reuse the runtime agent with explicit recording windows; initialization, report reads
and diagnostics' own counter queries are outside model windows or separately labeled. Fail
if agent installation/transformation fails or counts are unexpectedly empty. Counts describe
attempted mlx-c API downcalls, not Metal dispatches, kernels or GPU synchronizations.

Collect host thread CPU time and wall time around graph construction, selection/event handling
and synchronization in a separate tracing mode. Identify whether each measured interval includes
native CPU work or GPU waits. Thread CPU time omits background/native worker threads; wall
minus GPU time is not a valid measurement of Java overhead when work overlaps. Include scope/
array lifecycle overhead in the accounting and preserve the unassigned residual. Stock
`mlx_lm`'s n+1 host-graph / n GPU pipeline is a known scheduling difference with a potential
benefit over Java's synchronous loop, not an FFM saving. Analyze controlled synchronous
diagnostics first; stock overlap, extra forwards and allocator policy belong in a separate
attribution table.

Measure FFM transition cost in a benchmark-only calibration using a small C library with
non-elidable no-op/identity functions for representative downcall signatures, built with a
recorded compiler/flags and source/binary hashes. Use Java 25 FFM and matched carrier/layout/
linker options, warm the handles, consume results, measure loop/timer baselines and retain raw
samples across fresh forks. This library is generated under `build/`, never staged or published
as part of MLX. Time representative real native calls separately through existing core
facades only, labeling their native work; benchmark/test Java must not add direct `mlx_h.*`
call sites. The calibration shim's own FFM handles are separate from MLX bindings. If an
existing facade cannot expose a required measurement, record that limit or amend the plan
before adding a direct call: such a change needs an explicit mapping in
`req/mlx-api-inventory-overrides.json`, `generateMlxApiInventory` and call-site verification.
Do not describe the duration of `mlx_eval` or graph construction as pure FFM cost.

Combine measured signature-class counts with calibration ranges into an **estimated FFM
transition budget per decode step**, showing assumptions and uncovered signatures. It is a
calibrated estimate, not a causal partition of end-to-end decode time; counting and calibration
instrumentation perturb execution. Keep wall time in real native calls, Java CPU observations
and transition estimates separate. If attribution remains uncertain, state that uncertainty;
do not infer that all Java/Python throughput difference is FFM. The exit gate requires counts,
calibration and the qualified estimate, not an unsupported exact decomposition.

### 5.2 Both-pin, both-VLM first inference

Prepare isolated old/new native directories and matching bindings/source adaptations.
Place compatibility worktrees, including the old-pin tree, outside the checkout and record
their resolved absolute paths in evidence. Do not nest a worktree in an otherwise scanned
repository directory:
`verifyMlxApiCallSites` scans handwritten Java across benchmark/test source sets and would
misclassify the nested tree's old generated bindings. Verify worktree placement before root
checks; the main checkout's inventory scan must not include either compatibility tree. Do not
point current generated bindings at the old dylib: Phase 7.5 changed cumsum and SDPA signatures.
Use reviewed worktrees/builds sharing the benchmark measurement logic and workload bytes,
with only required pin/API compatibility differences. Record full source diffs, tree SHAs,
staged and actually loaded pin identities. Otherwise label a run as a code-and-pin comparison,
not an isolated native-pin effect. Do not repin the main checkout or overwrite its staged runtime.

For each of the two VLM workloads and each native pair, capture the first image+text generation
call immediately after the declared loader/setup policy in a fresh JVM, before correctness
forwards, warmups, probes or another image case. Retain load, first TTFT, prefill/vision stages
where observable, full generation and subsequent warm measurements. Lazy loader work stays
inside the first request; if a loader necessarily evaluates weights during load, record it.
Use one fixed case per process so case 2 cannot borrow case 1's first-use claim.

Distinguish fresh process with observed OS/Metal caches from controlled first-use-cache tests.
The required baseline is actual first-in-process inference followed by warm generation on both
pins/workloads. A controlled shader-cache experiment is additional work if drawing conclusions
about compilation or persistent-cache effects. For that experiment, use a dedicated test
account or a proven process/cache isolation mechanism; record the isolated cache scope and
before/after inventory for each pin. Do not delete the normal user's global caches, assume
`clearCache()` clears shader pipelines, or call a fresh JVM a cold machine. If isolation cannot
be established, record that limit and leave shader-cache attribution unresolved; it does not
invalidate a correctly labeled first-in-process measurement. OS page cache and system
shader-cache state remain observed/uncontrolled unless separately demonstrated otherwise.

Collect at least six alternating old/new process pairs per VLM for the observed-cache baseline
and for each additional cache regime used to support an attribution claim.
Apply §4's host records, correctness-after-capture and raw/paired statistics. Repeat surprising
VLM prefill results back to back before attribution. Controlled contention with `gpu-burn`
is optional follow-up, separately labeled with exact settings and drain behavior; it does
not replace quiet-host baseline evidence or establish hardware utilization. Preserve issue
#73's unresolved historical attribution: new measurements cannot reconstruct its missing host
state or prove which resident process caused the original slowdown.

## 6. Ordered work packages

### WP0 — Record readiness and freeze the comparison contract

- Reconcile Phase 7 gates and any owner scope decisions; record the 8.1 merge freeze as
  lifted by the development-bump PR's merge at `42e6ff0` (2026-11-05).
- Select and digest-pin the three real text artifacts, both VLM cases, exact input IDs/images,
  generation settings, precision/qualification tolerances, cache regimes and host capacity.
  Record TinyMistral acceptance/rejection and float-only MoE RAM/disk/page-cache/I/O budgets;
  list expected and observed effective dtypes on both sides before measurement.
- Qualify the Python lock and installed source; inspect actual timing/cache semantics. Record
  both Python adapter contracts, wired/cache policy, pilot-derived cost/time limits, native
  pin compatibility work, source-diff limits and proposed output schema in
  `req/plans/phase8-2-reference-findings.md`.

**Gate:** no unpinned artifact, unbudgeted MoE run or unreviewed timing/precision mismatch enters
measurement. Missing capacity/readiness remains explicit, not silently waived.

### WP1 — Implement benchmark workers and reports

- Add the orchestration/reporting in `jmlx-benchmarks`, with an opt-in
  `:jmlx-benchmarks:benchmarkBaseline` launcher. Reuse existing harness utilities where their
  semantics match; preserve `benchmarkDecode`, `benchmarkBatchDecode` and `benchmarkTf32`.
- Add benchmark-only generation/step adapters in model test sources with a dedicated opt-in
  worker/task where package access is required; do not publish test internals as API.
- Add `tools/decode-baseline/` controlled and stock Python workers, manifest validation and
  offline execution; exclude `mlx-vlm` from their dedicated environment.
  Install/verify remains explicit. Worker failures/timeouts produce failed reports and a
  nonzero launcher exit, never a partial summary presented as success.
- Both workers consume the same reviewed manifest. Validate arguments, version/digest/policy
  mismatches and workload dimensions before launching expensive work. Use unique output
  directories; reject accidental overwrite and do not reuse Gradle task outputs as new runs.

**Gate:** tiny synthetic fixtures prove boundaries, exact step/token counts, public-generation
adapter equivalence and report validation. They make no real-model performance claim.

### WP2 — Add counting, host observations and FFM calibration

- Generalize/reuse the test-only agent without changing generated sources. Add explicit
  instrumentation-mode metadata and reject diagnostic records in headline aggregation.
- Implement phase-local counts, thread CPU/wall diagnostics, calibration C source/build and
  raw-to-summary FFM estimate with coverage/uncertainty. Keep real native-call measurements
  behind existing core facades and run `verifyMlxApiCallSites` / `verifyMlxApiInventory`.
  Any plan amendment adding direct MLX binding uses must update the inventory overrides and
  regenerate the inventory before these checks; generated sources remain untouched.
- Add per-worker host records and reproducible alternating orchestration, cleanup/timeouts,
  including restoration of any task-owned isolated cache resources on failure.

**Gate:** counters cover direct allocation/free paths as well as op facades; an intentional
instrumentation failure fails the run. Calibrations consume outputs and report measurement
baselines. No production native artifact or API changes.

### WP3 — Qualify and run the text baseline

- Run correctness qualification for all real cases, then collect §4's uninstrumented paired
  baseline, separate memory runs and §5.1 diagnostics from clean committed code.
- Check manifest/runtime/adapter identities, raw sample completeness, host-state records and
  paired summary reproduction. Apply §4.1's budgets and rerun cap; retain unstable/incomplete
  outcomes and all sessions rather than indefinitely rerunning for a favorable stable result.

**Gate:** both adapters and all three artifact categories have qualified, complete measurements
for each required prompt length, or a dated owner scope amendment. Incomplete-cell blockers
are recorded but do not pass this gate.
Complete unstable cells are reported as unstable, with no settled ratio or attribution.
No ratio from mismatched work is promoted.

### WP4 — Complete both-pin VLM first-use measurements

- Build/verify isolated compatible pin trees, qualify image/prompt identities, and execute
  §5.2 for both workloads in the observed-cache regime. Add controlled cache experiments only
  where required to support a shader-cache/compilation claim.
- Preserve first-request outputs before validation and warmup. Archive paired host/timing/cache
  evidence and any instrumentation limits. Claims about controlled cache behavior need their
  own isolation evidence; no first-process result establishes a cold-system bound.

**Gate:** first-inference and warm results exist for both pins/workloads; a post-qualification
generation timer cannot satisfy this requirement.

### WP5 — Review and publish the evidence in the repository

- Add `req/phase8-2-benchmark.md`, raw JSON and digests under `req/benchmarks/phase8-2/`, and a
  digest-pinned evidence index identifying inputs, source commits, environment and host logs.
  Large traces/binaries stay in an explicitly located durable archive with hashes and retrieval
  instructions; an ignored local path alone is not an archive. Exclude credentials/usernames.
- Include exact runnable setup/execution/report-verification commands in
  `tools/decode-baseline/README.md`, after interfaces are implemented. Document schema fields,
  timing boundaries, licenses, capacity requirements and cold/first-use terminology.
- Update the roadmap's 8.2 status, the inference report's issue #73 follow-up, benchmark module
  documentation and `AGENTS.md` with actual execution limits and 8.4/9.1 handoff links.
  Do not retroactively rewrite historical results or claim native publication.
- Carry changes through PR/CI; merge only after explicit human review/approval.

**Gate:** a reviewer can reproduce every summary from raw records, identify the exact measured
code and input bytes, and distinguish measured facts, calibrated estimates and open limitations.

## 7. Verification and completion

| Check | Required evidence |
| --- | --- |
| Manifest/environment | Wrong digest, missing shard, incompatible pin, source drift or unsupported configuration fails before accepted measurement; both workers record effective identities |
| Timing semantics | Controlled single-prefill/127-step boundaries and stock split/pipelined/extra-forward behavior verified separately on tiny fixtures; stock iterator/context fully drained; legacy schemas preserved |
| Python policies | Source-hashed adapters reuse each loaded model's own tied/untied/quantized head after slicing; controlled wired policy is `not set` on both sides with device info/effective-limit uncertainty recorded; stock TTFT includes token-2 submission; allocator clears, detokenization and early EOS recorded |
| Feasibility/budget | Float-only MoE candidate accepted/rejected with reason and RAM/disk/page-cache/I/O budget; per-case dtypes and pilot cost estimates recorded; finite timeouts, three-session cap and unstable/incomplete reports tested |
| Correctness | Per-case qualified logits/margins and actual IDs; divergence retained and excluded from matched-generation ratios; existing strict goldens unchanged |
| Report/statistics | Known raw fixtures verify quantiles and per-pair ratios, fork order and units; missing/failed/instrumented samples cannot enter successful headline summaries |
| Host/first-use | Timestamped records for every worker; no prior inference before first request; both pins × both VLMs with observed cache state; extra cache-attribution claims require explicit isolation evidence |
| Memory | Active/cached/peak counters separated from RSS; peaks reset at stated phases; bounded stabilization failures visible; no throughput ratio from memory-polling runs |
| Host/FFM attribution | Actual downcall counts, phase/context windows, agent failure test, consumed calibration results, signature coverage and qualified transition budget; no GPU-kernel-count claim |
| Integration | Relevant Java module checks/Spotless/Checkstyle and root inventory/call-site checks pass; no new direct MLX calls without explicit inventory mapping/regeneration; old-pin worktree path is outside checkout; `buildSrc` checks if changed; native suites remain green; ordinary `check` adds no downloads/Python execution |
| Evidence | Clean measured SHA, exact commands and manifest/environment hashes; raw-to-summary verifier passes; durable archive and 8.4/9.1 report links recorded |

Add focused parser/schema/statistics/worker-failure tests and small native adapter/counting
checks. Native smoke checks test report correctness, never speed thresholds; if a new native
suite is required in CI, add it to the required-suite execution assertion. Do not make full
real-model runs a normal CI prerequisite. Linux may verify pure report/manifest code only;
no Linux or Windows inference performance is claimed. Java 25/macOS Apple Silicon native
measurements require the staged runtime; a newer development OS does not establish macOS 26
performance. Report actual hosts rather than extrapolating hosted runner fit.

Completion requires qualified real dense, MoE and separate 4-bit controlled and stock
measurements (including honestly labeled complete-but-unstable cells), the host/FFM
measurements, both-pin/both-VLM first-use follow-up, reproducible archived evidence and reviewed
status/report updates. Any narrower required artifact, adapter or first-inference matrix
requires a dated owner amendment with its effect on release claims; incomplete required
measurements are not a completed baseline. An unstable result supports a variability statement,
not a stable speed claim.
The 8.4 handoff states exactly what can be said about performance on the measured host/settings.
The 9.1 handoff lists observed bottlenecks and testable hypotheses without implementing them.
Unresolved shader-cache attribution must remain explicit, even when the required first-process
and warm measurements are complete.
