# Phase 8.2 — decode baseline and VLM first-inference evidence

Published measurement evidence for Phase 8.2 (plan: `req/plans/phase8-2-plan.md`;
measurement narrative, host observations and review history:
`req/plans/phase8-2-reference-findings.md` sections 15–17). This report carries the
**rounded display values**; the full-precision aggregates are in
`req/benchmarks/phase8-2/summary.json` (schema `jmlx-phase8-2-summary-1`), and every
aggregate in it is recomputed from the committed raw trees by the pure-Java verifier
`Phase82SummaryVerifierTest` (part of `:jmlx-benchmarks:test`, so it runs on the
cross-platform Linux CI job as well).

Rounding: seconds to 3 decimals, throughput to 1 decimal, ratios to 4 decimals,
microseconds to 1 decimal, megabytes (10^6 B) to 1 decimal.

This phase is a measurement only. No native artifact was published, no performance
claim about any release is made here, and optimization is Phase 9.1 (section 9 below).

## 1. What was measured, and on what

| | |
| --- | --- |
| Measured code (WP3) | clean tree at `cc0c603455caca69d434963990843aca98e6e606`, branch `phase8-2-wp3`, no untracked files; pinned by every worker's own `identity` block |
| Measured code (WP4) | a **code-and-pin comparison, not an isolated native-pin effect**: the old tree is `46cb325bec9b4878534cc5c6bda1a569dfe4aa4c` (156 commits before the new tree) and the new trees are `e4cb1f0f43ddd90242237f7e155a399ca5c84cfd` (session 1) and `ecc69a50e7a7263e955ac20c716b71344ce86c34` (session 2); the two trees differ in 156 commits, not only the native pin (findings 17.1). The measurement worker source is byte-identical in both trees of each session (SHA-256 re-verified by the launcher before the first fork) |
| Native pins (WP4) | old `0.31.2` / mlx-c `fba4470b89073180056c9ea46c443051375f7399`; new `0.32.4` / mlx-c `a26f2d584a14ec619ae8db9a09be03becd47598a` |
| Host | one host for all measurements: Apple M5 Max (`Mac17,6`), 128 GiB RAM, macOS 27.2.0 (`Darwin 27.2.0`); timestamps UTC |
| Java | Eclipse Adoptium `25.0.4.1` worker JVMs |
| Python | the hash-locked dedicated CPython 3.12 venv of `tools/decode-baseline` (mlx 0.32.4 / mlx-metal 0.32.4, matching the staged Java runtime of WP3; findings 9) |
| Precision | WP3 headline and diagnostics: `MLX_ENABLE_TF32=1` on both runtimes; WP4: `MLX_ENABLE_TF32=0` |
| Workload manifest | `tools/decode-baseline/workloads.json`, SHA-256 `e8c7730178a2e8f1bf1ace461db14119d2394fff4f60fa90deca34ef4857a270` (digest-pinned; re-hashed by every worker and by the pure-Java `verifyDecodeBaselineManifest` guard) |

Text cases (one dense, one quantized 4-bit, one MoE, all with float32 KV caches):
`qwen3-0.6b` (`Qwen/Qwen3-0.6B` @ `c1899de2…`, Apache-2.0),
`qwen3-8b-4bit` (`mlx-community/Qwen3-8B-4bit` @ `545dc425…`, MIT),
`tinymistral-6x248m` (`M4-ai/TinyMistral-6x248M` @ `2b4c0793…`, Apache-2.0).
VLM cases (both digest-pinned via their Tier-B manifests): `qwen3vl-2b`
(`Qwen/Qwen3-VL-2B-Instruct` @ `89644892…`, Apache-2.0) and `phi3v-4b`
(`microsoft/Phi-3.5-vision-instruct` @ `12b77fb4…`, MIT).

Protocol (frozen manifest `run_policy`): per worker, 1 first request + 5 warmups +
20 measured requests, 128 emitted tokens each, cache regime **primed-per-request**
(sequential full read of the checkpoint before every request; the prime time is
recorded per request and excluded from every measured interval). Statistics:
type-7 linear-interpolation quantiles; per-pair warm median over the 20 measured
requests; cell quantiles over the pair medians of the cell's pair set; ratio =
type-7 median over the pairs of the per-pair java/python median ratio. Quantile
arrays in `summary.json` are `[p10, median, p90]`. Worker counts: 344 WP3 workers,
52 WP4 workers — every raw file of both sessions is committed under
`req/benchmarks/phase8-2/raw/` (639 files) and digest-pinned in
`req/benchmarks/phase8-2/evidence-index.json`.

The two adapter contracts differ by design (see `tools/decode-baseline/README.md`
for the full schema and timing boundaries): the **controlled** adapter runs a
single full-prompt forward with last-position-only output projection and exactly
127 cached decode steps (its `ttft_s`/`prefill_s`/`steps` are the measured
prefill/decode contract); the **stock** adapter runs the shipped `mlx_lm`
loop (split prefill, n+1 lookahead pipeline, stock cache clearing) and reports
host-visible event times only. Stock and controlled results are never pooled.

## 2. WP3 — controlled text baseline (headline)

Greedy ID identity: in all 9 cells the Java and Python controlled adapters emit
identical 128-token sequences in every session (session A: 120/120 per cell = 6
pairs × 20 measured; A2: 60/60 per compared cell; A3: 40/40 and 20/20), so the
ratios below are matched-generation ratios.

Settled cells (`full_generation_s` p10/median/p90 over the cell's pair medians;
ratio = java/python):

| cell | pairs | java full_gen (s) | python full_gen (s) | ratio full_gen | ratio tps | ratio ttft | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b/32 | 6 | 0.557/0.558/0.559 | 0.491/0.494/0.503 | 1.1318 | 0.8912 | 2.4044 | stable |
| qwen3-0.6b/512 | 6 | 0.654/0.657/0.658 | 0.516/0.518/0.526 | 1.2655 | 0.7963 | 1.9192 | stable |
| qwen3-0.6b/2048 | 9 | 0.969/0.969/0.976 | 0.606/0.612/0.906 | 1.5863 | 0.6241 | 1.3456 | stable, pending ratification of the A2 reduced-session deviation |
| qwen3-8b-4bit/32 | 9 | 1.372/1.377/2.184 | 1.267/1.270/1.983 | 1.0841 | 0.9220 | 1.0605 | stable, pending ratification of the A2 reduced-session deviation |
| tinymistral-6x248m/32 | 6 | 0.580/0.582/0.585 | 0.516/0.517/0.517 | 1.1278 | 0.8851 | 1.0570 | stable |
| tinymistral-6x248m/512 | 6 | 0.631/0.632/0.634 | 0.538/0.538/0.541 | 1.1734 | 0.8487 | 1.0763 | stable |
| tinymistral-6x248m/2048 | 6 | 0.874/0.876/0.876 | 0.664/0.664/0.664 | 1.3185 | 0.7301 | 1.0835 | stable |

Unsettled cells (no ratio attributed; type-7 p10–p90 over the complete pair
medians, every degraded member retained with its host observation):

- **qwen3-8b-4bit/512** — `unstable` (A + A2 + A3, 11 complete pairs, io_affected):
  full_gen java 1.789–6.046 s / python 1.405–3.551 s; tps java 22.4–78.0 /
  python 38.3–101.4 t/s.
- **qwen3-8b-4bit/2048** — `incomplete` by the plan section 4.1 timeout default
  (one timed-out worker), `unstable` per the recorded run rule (A + A2 + A3,
  10 complete pairs, io_affected): full_gen java 3.077–9.623 s / python
  1.967–2.237 s; tps java 14.5–52.2 / python 87.6–95.2 t/s.

`io_affected` marks `qwen3-8b-4bit/2048` (and the unstable 512 cell, which ran in
the same degraded host windows): the 4.6 GB checkpoint is memory-mapped and its
page-cache pages are evicted during the run at 2048, so weight re-reads fall to
SSD speed (findings 7.3).

First requests of session A (the first request of each worker, fresh JVM per
worker; separate from the warm statistics — min/median/max full_generation_s, and
median load_s):

| cell | java first full_gen (s) | python first full_gen (s) | java load_s med (s) | python load_s med (s) |
| --- | --- | --- | --- | --- |
| qwen3-0.6b/32 | 0.755/0.758/2.677 | 0.510/0.511/0.520 | 0.036 | 0.299 |
| qwen3-0.6b/512 | 0.846/0.849/0.856 | 0.537/0.541/0.545 | 0.034 | 0.304 |
| qwen3-0.6b/2048 | 1.267/1.273/1.292 | 0.625/0.627/0.633 | 0.034 | 0.300 |
| qwen3-8b-4bit/32 | 1.645/1.651/5.593 | 1.272/1.278/4.747 | 0.055 | 0.411 |
| qwen3-8b-4bit/512 | 2.015/2.031/6.898 | 1.408/1.417/5.631 | 0.052 | 0.408 |
| qwen3-8b-4bit/2048 | 3.498/3.608/19.628 | 1.908/1.915/1.985 | 0.052 | 0.425 |
| tinymistral-6x248m/32 | 0.740/0.744/0.755 | 0.526/0.530/0.532 | 0.196 | 0.187 |
| tinymistral-6x248m/512 | 0.794/0.800/0.803 | 0.546/0.549/0.551 | 0.183 | 0.176 |
| tinymistral-6x248m/2048 | 1.077/1.083/1.088 | 0.676/0.677/0.678 | 0.184 | 0.175 |

The Python maxima in the 4-bit rows (4.7–6.9 s first full_gen) are the
degraded-state first requests of the same memfree-strike windows that degrade the
warm cells; they are retained with their host observations, not discarded.

## 3. WP3 — stock text baseline (headline)

The stock adapter runs the shipped `mlx_lm` loop. Its greedy outputs **diverge
between the runtimes in five cells** (0/120 identical IDs in every stock session
for `qwen3-0.6b` at all lengths and for `qwen3-8b-4bit/32` and `/512`; identical
for the three tinymistral cells and for `qwen3-8b-4bit/2048`), so the ratio in
the divergent cells is a **stock-loop timing observation, not an accepted
matched-generation ratio** (plan section 3.3). All stock requests emitted 128
tokens with finish `length` (no early stop on the manifest stop sets).

| cell | pairs | java full_gen (s) | python full_gen (s) | ratio full_gen | ratio tps | ratio ttft | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b/32 | 6 | 0.532/0.533/0.535 | 0.395/0.395/0.397 | 1.3485 | 0.7372 | 1.3530 | greedy-divergent, ratio rejected, timing recorded |
| qwen3-0.6b/512 | 6 | 0.632/0.633/0.635 | 0.423/0.424/0.426 | 1.4935 | 0.6592 | 1.2198 | greedy-divergent, ratio rejected, timing recorded |
| qwen3-0.6b/2048 | 6 | 0.946/0.947/0.949 | 0.509/0.512/0.519 | 1.8516 | 0.5150 | 1.2482 | greedy-divergent, ratio rejected, timing recorded |
| qwen3-8b-4bit/32 | 18 | 1.356/1.374/1.452 | 1.187/1.195/1.299 | 1.1375 | 0.8604 | 0.7429 | greedy-divergent, ratio rejected, timing recorded |
| qwen3-8b-4bit/512 | 12 | 1.779/1.785/1.915 | 1.329/1.342/1.489 | 1.3269 | 0.7207 | 0.9408 | greedy-divergent, ratio rejected, timing recorded |
| tinymistral-6x248m/32 | 6 | 0.561/0.561/0.563 | 0.457/0.458/0.458 | 1.2272 | 0.7754 | 0.5024 | stable (greedy-identical) |
| tinymistral-6x248m/512 | 6 | 0.611/0.612/0.612 | 0.467/0.468/0.468 | 1.3078 | 0.7402 | 0.8457 | stable (greedy-identical) |
| tinymistral-6x248m/2048 | 6 | 0.855/0.856/0.857 | 0.600/0.600/0.602 | 1.4245 | 0.6387 | 0.9602 | stable (greedy-identical) |
| **qwen3-8b-4bit/2048 (io_affected)** | 10 | 3.046/3.063/3.156 | 1.923/2.032/2.140 | 1.5021 | 0.5313 | 0.9017 | **blocked-pending-owner-decision** |

The blocked cell: 10 complete pairs (B pairs 1–5 + B2 pairs 7–11). Two further
java members were strike-excluded by the first-request event check
(`b`-pair0 and `b2`-pair6; each validated full_gen median 3.504 s / 3.475 s,
tps 44.9 / 45.6 t/s over its 20 measured requests) and the memfree strikes fall
inside their windows. Counting the excluded members' validated medians, the
>10% adjacent-fork trigger fires, so plan section 4 requires one further
alternating session before attribution; it was not run, and the cell is
**blocked** (not a settled measurement). Including the excluded members, the
12-pair ratio is 1.5226 (tps 0.5272) — recorded as `with_excluded_ratio`, not
as the cell's ratio.

## 4. WP3 — memory and per-token host diagnostics (ineligible for headline)

From the `diag-5-1` session: 18 instrumented workers (binding-count and tracing
instrumentation), one per cell and runtime-instrumentation; instrumented workers
are excluded from every headline aggregate by construction. Tracing values are
medians over the 20 measured requests of the tracing worker; memory values are
medians over the 20 measured requests of the binding-count worker (the table
values, findings 16.6). `residual_*` = the adapter call's span minus the sum of
the instrumented windows (the windows never change the work).

| cell | wall_ms | cpu_ms | res_wall_ms | res_cpu_ms | prefill_sync_us | step constr/sync/sel/rel (us) | req_release_us |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b/32 | 559.5 | 236.9 | 2.0 | 1.6 | 4818.6 | 366.0/3557.2/252.7/116.6 | 811.2 |
| qwen3-0.6b/512 | 662.9 | 243.3 | 1.9 | 1.5 | 14393.7 | 386.5/4264.4/254.4/124.9 | 829.0 |
| qwen3-0.6b/2048 | 975.9 | 266.1 | 2.2 | 1.8 | 72328.9 | 394.7/6286.5/262.1/127.0 | 842.5 |
| qwen3-8b-4bit/32 | 1555.5 | 320.3 | 2.5 | 1.9 | 48764.7 | 475.9/10847.3/295.0/161.1 | 1083.2 |
| qwen3-8b-4bit/512 | 1865.6 | 342.6 | 2.9 | 2.2 | 172164.8 | 482.6/12129.0/268.7/175.8 | 1087.0 |
| qwen3-8b-4bit/2048 | 3138.3 | 402.4 | 4.1 | 3.1 | 669804.2 | 552.9/16980.4/275.0/224.0 | 1112.6 |
| tinymistral-6x248m/32 | 586.7 | 213.9 | 1.6 | 1.4 | 15320.9 | 334.1/3804.9/209.9/100.0 | 356.1 |
| tinymistral-6x248m/512 | 637.5 | 211.5 | 1.6 | 1.4 | 27473.9 | 320.0/4108.5/212.1/101.1 | 352.9 |
| tinymistral-6x248m/2048 | 883.9 | 224.5 | 1.9 | 1.6 | 128775.8 | 328.2/5220.4/230.0/100.2 | 354.3 |

The residual is small (1.4–4.1 ms wall over a ~0.6–3.1 s request), i.e. the
instrumented windows cover essentially the whole adapter span. The per-decode
step is dominated by the step sync window (3.5–17.0 ms/step, model- and
length-dependent); construction, selection and release are sub-millisecond.

| cell | active (MB) | peak (MB) | cached (MB) | rss bc (MB) | rss tracing (MB) | suspect |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b/32 | 1503.3 | 1574.3 | 794.1 | 3032.1 | 2740.2 | no |
| qwen3-0.6b/512 | 1503.3 | 2659.9 | 7338.6 | 3047.5 | 2731.0 | no |
| qwen3-0.6b/2048 | 1503.3 | 5869.9 | 26190.1 | 3029.6 | 2727.2 | no |
| qwen3-8b-4bit/32 | 4607.7 | 5040.1 | 4174.1 | 6166.2 | 5867.9 | no |
| qwen3-8b-4bit/512 | 4607.7 | 8846.3 | 28043.3 | 6156.7 | 5871.5 | no |
| qwen3-8b-4bit/2048 | 4607.7 | 21570.4 | 103258.5 | 1290.4 | 5864.1 | **yes** |
| tinymistral-6x248m/32 | 4012.4 | 4111.2 | 716.4 | 2070.4 | 1661.8 | no |
| tinymistral-6x248m/512 | 4012.4 | 5725.9 | 3804.7 | 2067.5 | 1659.0 | no |
| tinymistral-6x248m/2048 | 4012.4 | 11034.5 | 14563.1 | 2072.6 | 1659.8 | no |

The `qwen3-8b-4bit/2048` binding-count RSS peak (1290.4 MB) is far below the
tracing worker's (5864.1 MB) and the recorded host memfree troughs; the block is
flagged `suspect` and its memory values are not attributed (findings 16.6).
Active/peak/cached are the MLX allocator counters; RSS is a process-level lower
bound. The separate memory runs with Python memory counters required by the
plan are **not done** (open item, section 7).

## 5. WP3 — calibrated FFM transition estimates (calibrated estimates, not measured facts)

`FfmTransitionEstimate` (schema `jmlx-ffm-transition-estimate-1`) over the
committed binding-count reports of the diag-5-1 session, measured requests only,
with the WP2 calibration (27 per-shape no-op downcall medians × the workload's
2000 calls; the committed calibration record stores the per-call medians to 2
decimals, the four struct-return shapes to 1 decimal). Coverage is 1.0 in all 9
cells (zero uncovered bindings; the 39 descriptor forms map into the 27
calibration shapes). The per-request budget is length-independent (identical at
32/512/2048 within a model — the verifier asserts the identity across the three
committed reports per case). These are **calibrated estimates, not a causal
partition** (the findings 15.4 assumptions apply).

| case | per_decode_step (µs) | prefill (µs) | request_total (ms) | total calls, 20 measured |
| --- | --- | --- | --- | --- |
| qwen3-0.6b | 106.68 | 100.23 | 13.649 | 16,671,420 |
| qwen3-8b-4bit | 111.12 | 102.37 | 14.214 | 16,241,340 |
| tinymistral-6x248m | 76.65 | 85.54 | 9.820 | 11,771,900 |

Per-request downcall counts (uniform across the 20 measured requests):

| case | prefill constr/sync/sel/rel | step constr/sync/sel/rel | request release |
| --- | --- | --- | --- |
| qwen3-0.6b | 4,948/3/14/1,145 | 5,315/3/14/1,183 | 56 |
| qwen3-8b-4bit | 4,567/3/14/1,215 | 5,062/3/14/1,269 | 72 |
| tinymistral-6x248m | 3,980/3/14/1,009 | 3,719/3/14/859 | 24 |

The verifier recomputes the budget medians from the committed record within the
per-call recording bound (0.05 ns/call for the four one-decimal shapes,
0.005 ns/call for the rest, times the budget's per-request call count). The
calibration **ranges** recorded beside each median in `summary.json`
(`recorded_not_reproducible_ranges`) are **not reproducible**: the
measurement-time per-shape p10/p90 samples are not archived.

## 6. WP4 — both-pin, both-VLM first inference

Two alternating sessions, 24 workers each (2 trees × 6 pairs per workload, both
workloads), observed-cache regime (no cache clearing or priming; the OS/Metal
cache state is observed, not controlled). Per process: model load, the first
image+text generation captured immediately after load in the fresh JVM (before
any warmup or probe — lazy loader work stays inside the first request),
correctness verification of the first request against the Tier-B manifest, then
3 warmups and 5 measured repetitions. Every first request in both sessions
matched the manifest reference (IDs, finish reason, prompt positions, rendered
prompt), and every measured repetition reproduced the first request's IDs
(greedy determinism under TF32=0). Stop sets (inert for the frozen cases, both
finish MAX_TOKENS): qwen3vl-2b `{151643, 151645}`, phi3v-4b `{32000}`.

Session 1 (`wp4-observed-cache-1`, 2026-11-11 15:38:49–15:46:07 UTC, worker
source SHA-256 `405a839c…` in both trees; session report schema
`jmlx-vlm-first-use-session-1`, worker schema `-1`, peak counters
**lifetime-cumulative**):

- Host state: **not** the plan section 4 quiet-host baseline — the resident
  serving-model server (omlx) ran at up to 97.9% in four bursts
  (97.9/95.2/83.2/75.6%) bracketing the qwen3vl pairs 2–5; from the phi3v phase
  it was at or below 0.6% (48 per-fork hostrec records, digest-pinned).
- qwen3vl-2b: first TTFT old 0.249/0.449/0.691 s, new 0.475/0.509/0.673 s;
  first full_gen old 0.613/0.776/1.139 s, new 0.777/0.827/1.047 s; load wall
  medians 1.090/1.360 s. Warm (pooled n=30, p10/median/p90): TTFT old
  0.252/0.329/0.353 s, new 0.145/0.324/0.352 s; full_gen old
  0.495/0.605/0.627 s, new 0.266/0.598/0.630 s. Paired per-pair warm-median
  ratios (new/old), full_gen: 0.9940/1.0662/0.4777/0.9049/0.9731/0.7893
  (median 0.9390); TTFT: 0.9379/1.1172/0.5246/0.8927/0.9768/0.9631 (median
  0.9505). The new pair 2 dip (0.4777) is a single low pair bracketed by the
  omlx bursts — recorded, not attributed.
- phi3v-4b: first TTFT old 1.340/1.573/2.470 s, new 1.270/1.354/2.184 s; first
  full_gen old 2.422/3.124/5.776 s, new 2.371/2.448/3.856 s; load wall medians
  1.082/0.846 s. Warm: TTFT old 1.381/1.572/2.416 s, new 1.396/1.508/2.103 s;
  full_gen old 2.268/2.448/4.023 s, new 2.255/2.397/3.670 s. Paired full_gen
  ratios: 0.9091/0.9130/1.0504/0.9741/0.9764/0.9798 (median 0.9753); TTFT:
  0.8497/0.8819/1.0819/0.9545/0.9622/0.9730 (median 0.9584). Pairs 0–1 are slow
  in **both trees at the same session position** (~3.6–4.0 s), stepping down
  from pair 2 — a synchronized two-tree step consistent with a host
  cache-regime transition; the controlled shader-cache experiment that would
  attribute it was not run.
- Memory (lifetime-cumulative peaks; spans over the 12 workers per tree, exact
  bytes in `summary.json`): qwen3vl-2b after load peak 8,514,863,104–8,514,865,152 B
  in both trees (active 8,510,144,512–8,510,146,560 B); after the first forward
  peak 9,828,890,684–9,828,892,836 B — consistent with the Phase 7.3b WP8
  measurement of 9,828,890,788 B for the same checkpoint and case. phi3v-4b
  after load peak 16,643,123,200–16,643,127,296 B in both trees (active
  16,536,107,008–16,536,111,104 B); after the first forward the trees diverge:
  old 31,100,376,165–31,100,380,261 B, new 30,202,745,961–30,202,748,009 B —
  the new tree peaks ~897.6 MB lower on the identical checkpoint and case.
  Under the code-and-pin label this is **not attributable to the native pin
  alone**; the old tree's number matches the Phase 7.3c WP8 measurement
  (31,100,376,165 B) within one page.

Session 2 (`wp4-observed-cache-2`, 2026-11-11 21:00:17–21:07:00 UTC, worker
source SHA-256 `fede0c00…` in both trees; session report schema
`jmlx-vlm-first-use-session-2`, worker schema `-2`, peak counters
**phase-local** via `MLXMemory.resetPeak()` after the after_load and after_first
samples — the session 1 cumulative and session 2 phase-local peaks agree within
the quoted spans, confirming the peak is reached in the first forward):

- Host state: omlx at 89.4% in the pre record, decaying monotonically across
  the first four qwen3vl forks (the owner's pre-launch activity, recorded
  without causal attribution) and at or below 0.7% from qwen3vl old:p4 to the
  end; the entire phi3v phase ran on the quiet host.
- qwen3vl-2b: first TTFT old 0.204/0.500/0.765 s, new 0.233/0.526/0.922 s;
  first full_gen old 0.335/0.835/1.327 s, new 0.366/0.839/1.350 s; load wall
  medians 1.366/1.391 s (new pair 2's load is an outlier: 2.845 s). Warm:
  TTFT old 0.110/0.239/0.378 s, new 0.111/0.338/0.389 s; full_gen old
  0.224/0.392/0.654 s, new 0.226/0.616/0.674 s. Paired full_gen ratios:
  1.1901/1.0630/0.9944/2.3792/1.0793/1.0055 (median 1.0711); TTFT:
  1.3029/1.0683/0.9857/2.3280/1.0361/1.0013 (median 1.0522). Pairs 1, 2, 4 and
  5 sit in 0.994–1.079; the p3 ratio (2.3792) is an intra-pair ordering
  artifact — the host's fast-regime boundary fell between the pair's new and
  old members. Only pairs 4–5 ran after the observed omlx activity decayed
  (1.0793 and 1.0055); these two quiet pairs do not support a workload-wide
  code-and-pin attribution.
- phi3v-4b: first TTFT old 1.355/1.535/1.655 s, new 1.316/1.424/1.817 s; first
  full_gen old 2.472/3.795/6.568 s, new 2.447/2.696/9.755 s (slow first
  requests occur in both trees — the lazy loader work inside the first request
  — recorded without attribution); load wall medians 1.094/1.035 s. Warm:
  TTFT old 1.416/1.617/1.869 s, new 1.348/1.733/1.855 s; full_gen old
  2.332/2.491/2.794 s, new 2.234/2.672/2.864 s. Paired full_gen ratios:
  0.9023/1.0256/0.9922/1.1201/1.0582/1.0181 (median 1.0219); TTFT:
  0.8295/1.0128/0.9635/1.1666/1.0267/1.0336 (median 1.0198) — on the fully
  quiet host. The session 1 early-pair slowness **did not recur**: session 2's
  warm full is flat at 2.136–2.815 s across all pairs, consistent with a
  host-side state tied to the first use of the checkpoint after the 09:03
  reboot. "Consistent with" is the strongest claim; the attribution
  experiment was not run.
- Memory (phase-local peaks; exact byte spans in `summary.json`): qwen3vl-2b
  after load 8,514,863,104–8,514,865,152 B both trees; after first forward
  9,828,890,684–9,828,892,836 B (byte-identical to session 1's span and to the
  Phase 7.3b WP8 value); after the warm/measured phases the peak re-touches the
  same maximum buffer (it never exceeds the first forward's peak). phi3v-4b
  after load 16,643,123,200–16,643,125,248 B; after first forward old
  31,100,376,165–31,100,378,213 B, new 30,202,744,781–30,202,748,009 B — the
  ~897.6 MB tree-to-tree gap is unchanged across sessions.
- `after_final` note (schema -2): the after-final peaks equal the after-first
  peaks in every cell — the warm phase re-touches the same maximum buffer
  rather than growing it; the memory table's `after_final` span is therefore
  the same buffer, not additional growth.

**Cell closure (plan section 4.1): qwen3vl-2b = `unstable`; phi3v-4b =
`unstable`.** Both sessions have all required worker samples and are retained
with their ranges and host records, but neither cell has a settled performance
ratio or runtime attribution. Session 3 was not run: section 4 requires another
alternating session before runtime attribution, and ending collection after two
sessions, before the three-session limit, is a recorded deviation from
section 4.1's stopping rule. Any later attribution needs another complete
alternating session or a recorded plan revision. The shader-cache attribution
remains explicit and unresolved (the controlled experiment was not run; plan
section 5.2 marks it additional work).

**Terminology (cold / first-use).** This report uses "first request" for the
generation captured immediately after load in a fresh JVM, and "warm" for the
post-warmup measurements of the same process. It makes **no cold-system
claim**: no first-process result here establishes a cold-system bound, the OS
and Metal caches were neither cleared nor inspected (findings 17.1/17.6), and
the observed-cache regime records the cache state as observed, not as reset.

## 7. Open items and limitations

WP3 (owner decisions required before the WP3 gate passes on these cells;
findings 16.7):

1. Stock greedy divergence (five cells): the matched-generation ratios stay
   rejected pending review of the stock `mlx_lm` loop divergence.
2. A2 (3 pairs, 2/1 leaders) and A3 (2 pairs) reduced-size deviations pending
   ratification; A2 decides controlled 0.6b/2048 and 4bit/32.
3. The post-timeout A3 pair10 counts as a fourth session for the two
   controlled 4bit cells under the plan section 4.1 restart rule — over the
   three-session cap.
4. Controlled 4bit/2048 status: `incomplete` by the plan section 4.1 timeout
   default, `unstable` per the recorded run rule.
5. Stock 4bit/2048 blocked: a further alternating session is required before
   attribution and was not run; the memfree-strike and event-check-exclusion
   decisions are open.
6. The separate memory runs with Python memory counters are not done.
7. The 4bit memory-consumer hypothesis (worker allocator pool) is open.

WP4 (findings 17.6):

1. Both VLM cells closed unstable: no settled performance ratio and no runtime
   attribution; any later attribution needs another complete alternating
   session or a recorded plan revision.
2. Shader-cache attribution remains explicit and unresolved: the controlled
   shader-cache experiment was not run.

FFM:

1. Calibrated estimate, not a causal partition; the calibration p10/p90 samples
   are not archived, so the recorded ranges are not bit-reproducible (the
   medians are, within the 2-decimal per-call recording bound).

General: all numbers are host- and setting-specific (M5 Max, 128 GiB, macOS
27.2.0, the stated Java/Python runtimes and precision flags, the pinned native
runtime). A newer development OS does not establish macOS 26 performance, and
nothing here is a claim about hosted-runner fit or about any published
artifact.

## 8. Handoff to 8.4 — what the first native release may state

What the first native release (milestone 8.4) may state about performance, on
the measured host and settings only:

- jmlx's **controlled** single-prefill/127-step generation is 1.08–1.59× the
  wall time of the `mlx_lm` Python reference (same host, same pinned MLX
  runtime for both sides, TF32=1, primed-per-request, 128 tokens): 1.1318–1.5863
  over the settled dense/MoE cells and 1.0841 for the settled 4-bit cell; the
  4-bit 2048-length cells are io_affected (page-cache-limited) and the
  4-bit/512 cell is unstable, so no decode-speed statement should be made for
  the 4-bit checkpoint at 2048 (and the 512 statement is unstable).
- The **stock** `mlx_lm` loop comparison (1.1375–1.8516× full-gen wall) is a
  timing observation: in five of nine cells the two runtimes' greedy outputs
  diverge, so those ratios must not be presented as matched-generation
  results.
- Per-token host overhead (calibrated estimate): the Java FFM transition
  budget is 76.65–111.12 µs per decode step, 85.54–102.37 µs of prefill, and
  9.820–14.214 ms per 128-token request, over 11.77M–16.67M downcalls per 20
  measured requests.
- First-inference and warmed generation for both VLMs were measured on both
  pins, but **no settled cross-pin ratio exists** (both cells unstable); the
  first release must not state a pin-effect in either direction, and the
  ~897.6 MB phi3v after-first-forward peak difference (new tree lower) is a
  code-and-pin observation, not a pin claim.
- None of these figures may be quoted for another host, OS or precision mode
  than the one recorded in section 1, and none of them is a claim about a
  published artifact (no native publication happened in 8.2).

## 9. Handoff to 9.1 — observed bottlenecks and testable hypotheses

Observed, with the evidence in this report and findings 15–17; **none of these
is implemented in 8.2** (optimization is Phase 9.1):

1. **Per-decode-step host overhead.** The step sync window dominates the step
   budget (3.557–16.980 ms/step), while construction/selection/release stay
   sub-millisecond (210–553 µs construction, 210–295 µs selection, 100–224 µs
   release). The calibrated FFM transition budget is 76.65–111.12 µs per step
   over 5,062–5,315 construction + 1,183–1,269 release downcalls per step.
   Hypothesis (testable): fewer/larger downcalls per step (e.g. batched or
   fused transitions) reduces the measured step wall time by roughly the
   calibrated budget.
2. **Prefill sync.** The prefill sync window is 4.8 ms (0.6b/32) to 669.8 ms
   (4bit/2048) — at 2048 it is the single largest instrumented window of the
   request. Hypothesis (testable): prefill at long prompts is sync-bound at the
   full-prompt forward; chunking or overlapping the last-position projection
   is a candidate only for 9.1.
3. **4-bit page-cache regime (io_affected cells).** The 4.6 GB mmap'd
   checkpoint's page-cache pages are evicted during 2048 runs, and weight
   re-reads fall to SSD speed (~1.75 s per decode step vs ~13 ms
   compute-bound at 512; findings 7.3). Hypothesis (testable): retention
   (mmap pinning, larger OS cache, or an explicit warm read inside the prime)
   restores the compute-bound step rate; the 4bit memory-consumer hypothesis
   (worker allocator pool) is the open alternative explanation for the memfree
   troughs.
4. **VLM first-use regime transition.** The phi3v early-pair slowness of
   session 1 (both trees, one session position) did not recur on the quiet
   host in session 2 — consistent with host state after reboot (OS page cache /
   system shader cache warming). Testable hypothesis: the controlled
   shader-cache experiment (plan section 5.2 additional work) isolates it;
   until then no first-use or pin attribution is supported.
5. **Stock-loop divergence.** The five greedy-divergent stock cells (section 3)
   are a correctness observation of the two stock loops, and a candidate 9.1
   investigation if matched stock comparison is ever required.

## 10. Reproducing the evidence

- Raw trees: `req/benchmarks/phase8-2/raw/wp3/` (33 session.json + 344
  report.json + 33 spec.json, hostrec/thermal records, the rsswatch log) and
  `req/benchmarks/phase8-2/raw/wp4/` (both full sessions, the post-fix smoke,
  session specs and hostrec logs) — 639 files in total.
- Digests: `req/benchmarks/phase8-2/evidence-index.json` (SHA-256 of every
  committed raw file, the workload manifest, the FFM calibration record and the
  Tier-B input pins); the pre-normalization digests are
  `req/plans/phase8-2-wp3-raw-digests.json` (458 files) and
  `req/plans/phase8-2-wp4-raw-digests.json` (181 files). No file in the raw
  trees contains a username or absolute local path other than one retained
  `rsswatch.log` line naming the GitHub-hosted runner user (`ghrunner`),
  recorded in the evidence index.
- Summary: `req/benchmarks/phase8-2/summary.json` at full double precision.
- Verifier:

  ```sh
  ./gradlew :jmlx-benchmarks:test --tests "se.alipsa.jmlx.benchmarks.Phase82SummaryVerifierTest"
  ```

  The verifier re-derives every summary value from the raw bytes (type-7
  quantiles, pair sets, identity blocks, the FFM budget within the calibration
  recording bound, the WP4 pooling and paired ratios), pins the measured code
  SHAs and tree SHAs, and asserts the raw tree is exactly the committed
  digest-pinned set. It is pure Java and part of `:jmlx-benchmarks:test`, so it
  also runs on the Linux CI job.
- Tooling and how the measurements were taken: `tools/decode-baseline/README.md`.
