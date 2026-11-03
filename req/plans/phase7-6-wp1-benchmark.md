# Phase 7.6 WP1 — Delta rule benchmark and default review

Recorded 2026-11-03 from clean detached source commit
`7f315e02c2dc8e16dca67f2cb196c8923160ee54`. All Java records report an empty Git status. The benchmark checkout uses
the existing staged native runtime and an ignored native directory; the unrelated untracked
user script in the main checkout is untouched. Full core checks and the reference provenance
guard pass locally. Model integration and hosted CI remain pending.

## Scope and numerical qualification

These are evaluated **delta-rule-only** measurements on identical prepared FLOAT32 Q/K/V,
decay, beta and nonzero initial state. They exclude learned projections, convolution, norm,
vision, tokenization and model loading. An aggregate sample repeats 48 independent layer calls
at the 27B head dimensions; it is not a full 27B network forward or a tokens-per-second claim.
The same saved input SHA-256s are recorded across Java and all Python candidates. State axes
and linear/log gate conventions are adapted explicitly.

HF 5.22.1 CPU FLOAT32 chunked/recurrent outputs were exported for all 12 shapes/lengths.
Before warmed timings, both Java prototypes compare output and final state directly against
those files at the unchanged `1e-4` bound. The maximum observed Java/HF error across the matrix
is `1.847744e-06`. See [HF qualification](../../tools/qwen35-reference/evidence/benchmark-hf-qualification.json)
and [Java matrix](../../tools/qwen35-reference/evidence/delta-java.json).
This prepared-input check contains no mlx-vlm Q/K normalization discrepancy; the full-model
normalization attribution is a separate record and remains outside these timing claims.

## Evaluated aggregate times

Median milliseconds for 48 calls, three samples per path. Java includes evaluation and scope
cleanup. Python evaluates output and final state. Each frontend alternates/rotates its own
candidate order; frontend runs are sequential, so ratios are descriptive rather than a paired
statistical inference about language/runtime overhead.

| Tokens | Java chunked | Java recurrent | mlx-vlm chunked | mlx-vlm recurrent | mlx-vlm custom Metal |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 147.525 | 707.635 | 114.145 | 144.548 | 13.462 |
| 256 | 533.157 | 2782.478 | 138.780 | 534.071 | 28.673 |
| 1024 | 2122.804 | 11099.083 | 250.743 | 2105.646 | 93.311 |
| 1280 | 2550.557 | 13848.364 | 304.105 | 2676.893 | 114.528 |

At 1,280 tokens the Java sample ranges and allocator increments were:

- `chunked_ops`: 2548.381–2602.323 ms; maximum transient allocator increment 178,550,212 B.
- `token_recurrence`: 13847.108–13862.053 ms; maximum transient allocator increment 78,626,816 B.

Three-sample ranges are descriptive, not confidence intervals. The observed additional Java
ops cost cannot be attributed to FFM or GPU dispatch from these records alone. The Java
prototype solves/evaluates bounded chunks separately; mlx-vlm batches intra-chunk preprocessing.
The custom Metal path is the reference package's own kernel, not MLX's built-in delta op.

MLX's built-in op is separately measured and numerically ineligible at all four real-width
cases: output or final-state error exceeds `1e-4`. Its rejected results remain in the raw
reference record; no timing is offered as an eligible alternative and no bound was loosened.

## Measured native API calls

Separate instrumented JVM, same clean source and qualified inputs; its timings are invalid.
Counts below cover 48 layer calls at 1,280 tokens, including explicit scope cleanup. The agent
counts generated MethodHandle downcall bodies at runtime and does not edit generated bindings.

| Path | Total native downcalls | `mlx_matmul` | `mlx_eval` |
| --- | ---: | ---: | ---: |
| chunked_ops | 3,159,936 | 66,240 | 1,008 |
| token_recurrence | 11,676,096 | 184,320 | 61,488 |

These are actual C API downcalls, **not Metal dispatch counts**. Dispatch counts were not
observed. The raw records retain the full per-function breakdown. A source-derived operation
estimate must not be presented as a measured GPU dispatch count.

## Host and memory limits

Each sample has a timestamped CPU load, top process list and VM snapshot. GPU utilization,
all competing GPU work and thermal state are not established. The first-call records describe
one call per shape within that JVM after input preparation; they do not represent an untouched
machine's first model inference. Warmed aggregate records contain 48 calls. Logical retained
state is separate from allocator/process peaks; the reported allocator increments exclude
already active prepared/reference tensors and are not model memory-fit bounds.

Independent native tests cover repeated request cleanup, exact parameter-scope handle stability,
a 4,128-position bounded-graph case, split prefills, convolution history and failure poisoning.
The ordinary and enforced FLOAT32 core check tasks pass. Full model/request/vision ownership
and first-inference benchmarks remain later work packages.

## Review required before selecting a default

The current proposal is **chunked ops**, retaining recurrence for decode and avoiding a new
native/kernel facade in this milestone. It passes the independent HF checks and improves the
measured Java aggregate cost substantially, while still costing about 2.55 s of delta-rule work
at the 1,280-token/48-layer shape. No production default is selected yet.

An alternative is to amend the plan for a custom Metal implementation before selecting a
default; the roughly 115 ms reference aggregate provides measured motivation, not an existing
Java implementation or proof of its integration cost. The built-in native op cannot be selected
under the present numerical qualification. Per the [WP1 gate](phase7-6-plan.md#7-ordered-work-packages-and-evidence),
review this choice before WP2/integration proceeds.

Raw records:

- [Java timings](../../tools/qwen35-reference/evidence/delta-java-48-layers.json),
  [single-layer matrix](../../tools/qwen35-reference/evidence/delta-java.json).
- [Reference four-path timings/errors](../../tools/qwen35-reference/evidence/delta-benchmark-48-layers.json).
- [Java per-case downcall counts](../../tools/qwen35-reference/evidence/delta-java-counts.json),
  [measured 48-layer counts](../../tools/qwen35-reference/evidence/delta-java-48-layers-counts.json).
