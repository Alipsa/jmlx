# Phase 6.4 decoder benchmark

The opt-in `:jmlx-examples:benchmarkDecode` task measures a local checkpoint. It downloads
nothing. The first model load in a fresh JVM is timed before any checkpoint hashing; this is a
first-in-process measurement with the OS page cache in its observed state, not a flushed-cache
measurement. Public `forward` evaluates its outputs before each prefill and one-token decode
timer stops. That forward computes logits for every prompt position, whereas generation projects
only the last hidden state, so the reported `prefill_ns` and prefill peak memory overstate real
prefill cost (the report records this as `"prefill_logits": "all_positions"`); compare them only
with each other. Sustained generation samples active native bytes after every emitted token and reports
the allocator's peak active bytes separately from cached bytes.

On a bootstrapped macOS Apple Silicon host, run each synthetic family in a fresh JVM:

```sh
./scripts/bootstrap-native.sh
for family in llama llama31 qwen2 mistral gemma phi3 mixtral; do
  ./gradlew :jmlx-examples:benchmarkDecode \
    --args="${PWD}/tools/hf-reference/goldens/checkpoints/${family} ${PWD}/build/phase6-4-${family} 1,7,42,3,19,5 32 5 2 full"
done
```

Each run writes `<output-prefix>.json` and `<output-prefix>.md`. The JSON records the checkpoint
file hashes, runtime pins, sample distributions, tokens/s, load/prefill/decode times, peak active
native memory, and per-token active-memory samples. These tiny checkpoints measure synthetic
fixture performance; their numbers do not represent production-size models. Repeat with pinned
Tier-B Llama, Qwen2, and Mistral directories when they are available on the runner, saving raw
JSON, commands, hardware, and checkpoint hashes. Inspect the Tier-B Mistral `sliding_window`
setting before running or labeling an eviction benchmark.

The implementation environment for this change is Linux without the pinned macOS MLX runtime.
Consequently no native throughput or memory results are recorded here. The macOS CI benchmark
smoke step validates output fields without speed thresholds; performance comparisons remain an
explicit native-run acceptance item.

## Quantized KV retention probe

The pinned macOS runtime probe ran in [CI run 125](https://github.com/Alipsa/jmlx/actions/runs/36777592834)
and the padded-head follow-up in [CI run 126](https://github.com/Alipsa/jmlx/actions/runs/36778017501).
Both native jobs and the required-suite assertions passed. Their `kv-cache-quantization-probe`
artifacts contain the raw JUnit XML and measurements. All seven tiny HF reference checkpoints use
16-dimensional K/V heads.

| Cache shape and affine setting | Packed bytes for one 64-token K | Float bytes | Maximum round-trip error | Direct packed SDPA |
| --- | ---: | ---: | ---: | --- |
| D=16, group=16, 4 or 8 bits | — | 4,096 | — | Quantize rejected: native groups are 32, 64, 128 |
| D=16 padded to 32, group=32, 4 bits | 1,536 | 4,096 | 0.00832210 | Not attempted; packed last dimension is 4, not the query's 16 |
| D=32, group=32, 4 bits | 1,536 | 8,192 | 0.00924163 | Rejected: packed last dimension 4, query dimension 32 |
| D=64, group=64, 4 bits | 2,560 | 16,384 | 0.0214118 | Rejected: packed last dimension 8, query dimension 64 |

Eight-bit D=32/64 cases also packed successfully, with smaller round-trip error (0.000670463
for group 32), but direct SDPA rejected their packed dimensions. The padded D=16 probe observed a
26,128-byte process-wide peak during dequantization; that figure includes the still-live source
and padding arrays, so it is **not** a controlled float-versus-quantized peak comparison. The
representation itself proves the limiting transient: a D=16 cache padded to D=32 needs 1,536
packed bytes plus an 8,192-byte dequantized tensor for each 64-token K, before attention's other
buffers. A float D=16 K needs 4,096 bytes. For D=32/64, the dequantized tensor alone is the size
of the original float cache, and packed storage remains live alongside it. Full-cache
dequantization also adds work to every token.

The probe therefore does not establish a useful decode-memory or throughput path on the pinned
runtime. `GenerationCachePolicy.quantized(bits, groupSize)` and `KVCachePolicy.quantized(bits,
groupSize)` fail immediately with a named unsupported-capability exception. No request silently
uses float retention. Append, reorder, fork, and scope ownership of a quantized cache were not
probed because the representation fails the attention-memory gate first. A future fused packed-KV
attention kernel would require a new capability probe and accuracy/ownership tests before enabling
the policy. A preallocated `slice_update` buffer or sliding ring buffer remains a post-baseline
float-cache optimization candidate.

## Phase 6.4.1 step 1: packed attention prototype and decision (2026-10-01)

**Decision: stop.** No candidate meets the frozen prototype criterion P1 at 4 bits, and the cause is
the quantizer, not the attention path. 6.4.1 is deferred pending a new attention design (or an
amendment that changes what is being quantized). The explicit rejection in
`GenerationCachePolicy.quantized` and `KVCachePolicy.quantized` stays, and 6.5 proceeds. Step 1b is
skipped, so 6.5 code uses `keys()`/`values()` and a future reopening must migrate those consumers
(plan, "Sequence and blocking").

Host Apple M2 Max, macOS 26.7, pinned mlx-metal 0.31.2 / mlx-c `fba4470`, commit `6e4cb43`
plus the probe. Probe: `PackedAttentionProbeTest` (random normal K/V/Q seeded 20261001,
B=1, 2 kv heads x 4 query heads, S = 2048/4096, L = 1 and 256, causal/sliding/left-padded batch
masks). Raw output: `req/data/phase6-4-1-probe/packed-attention-probe.txt`. Candidate A is packed
`quantized_matmul` attention with GQA folded into the query axis (so the packed K/V need no head
broadcast), masks applied with `where(mask, scores, lowest)`, and zero-padded query dimensions.

| Criterion | Result |
| --- | --- |
| P1 accuracy (max error as % of float output max abs; limit 1% for 8-bit, 5% for 4-bit) | **4-bit fails everywhere**: D16 f32 6.4-21.0%, D64 f32 6.1-16.5%, D64 bf16 6.2-23.1% (0 of 27 cases pass). 8-bit: D16 f32 0.5-1.3% (6/9 pass), D64 f32 0.4-1.0% (8/9), D64 bf16 0.8-2.3% (1/9) |
| P2 peak below float at S=4096 | pass for all six configurations (for example D64 f32 4-bit: 1.78 MB packed against 25.4 MB float) |
| P3 scaling bound | pass for all six; peak growth per token was identical across the 5 runs (spread 0), so no configuration was inconclusive |
| P4 time <= 2.0 x float | **passes, but bf16 4-bit is borderline and run-mode dependent.** Two harness defects were fixed after review: calls accumulated in one scope, and all float calls were timed before all packed calls (so warm-up and GPU clock state favoured one side and the verdict depended on run order). Float and packed calls are now alternated, which one leads flipping each iteration, with 30 warmup and 60 timed calls (the plan's P4 text says 20 timed; this is a deliberate increase for noise, not a threshold change), and each line records `warmup=`/`timed=`. Seven runs of the final harness (3 `decodeTime` alone, 4 whole class incl. the committed one): float32 1.17-1.73 (all pass); bf16 8-bit 1.54-1.65 (all pass); bf16 4-bit 1.58-1.68 in 4 runs, 1.99 in one whole-class run and 2.05 and 2.09 in two of the three `decodeTime`-alone runs, so it fails 2 of 7. Superseded earlier figures: 0.84-2.00 from the accumulating harness, and 2.02-2.36 (bf16 4-bit) / 1.98-2.14 (bf16 8-bit) from the sequential harness, whose fails were an ordering artifact. Read P4 as inconclusive for bf16 4-bit (timing-state dependent), a pass elsewhere; it does not change the stop, which P1 decides |
| P5 host synchronizations | 0 (lazy graph) |

**Why P1 fails, and why B and C cannot fix it.** `quantizerFloorDiagnostic` compares three
outputs on identical inputs: float SDPA over the original K/V, float SDPA over the
quantize-then-dequantize K/V, and candidate A. For float32 the attention path adds 0.0000-0.0003%
over float SDPA on the dequantized K/V, so all of A's error is in the packed K/V. For bfloat16 the
added share is 0.56-0.90% (precise-softmax and bf16 matmul rounding) on top of a 1.1-13% floor.

Candidate B was measured at the only level that decides P1: the diagnostic's `viaDequant` is B with
a single block (dequantize K/V, then float SDPA). Splitting it into blocks with an online softmax
changes only reduction order, so B's P1 error is that floor (9-17% at 4 bits). B's P2-P5 were not
measured, because P1 already decides the stop; this is a ruling, not a skipped rule. C reads the
same packed K/V and was not built, for the same reason.

Under a sharper attention pattern (queries scaled by 6) the 4-bit error reaches 33-68% and 8-bit
reaches 2-4%, because rounding the keys moves which tokens the softmax picks. Sharper, more
realistic attention therefore makes this worse, not better. Two caveats on the numbers: draws differ
between tests (the same D16 4-bit S=2048 L=256 causal case reads 21.0% in `accuracy` and 9.0% in
the floor diagnostic), so the per-case range is noisy and the decision rests on the 4-bit minimum
(6.1%); and the batched cells include padded query rows whose outputs are raw V values, which
enlarges the denominator, so they read lower and flatter the 8-bit pass counts. Neither affects the
stop. The plan does not allow loosening P1 after a result.

**Other facts recorded.**

- Affine scales and biases follow the input dtype: float32, bfloat16 and float16 inputs give
  float32, bfloat16 and float16 scales and biases, and UINT32 packed words. The D2 accounting for
  bfloat16/float16 is confirmed.
- GQA folding needs no broadcast of packed K/V and no `repeatKeyValueHeads` on them.
- Append multiplier `a` on packed storage (concatenate, evaluate, float sources released first):
  2.06-2.33 (0 spread over 5 runs). The float append multiplier measured 2.0 for float32 and 4.0 for
  bfloat16; the bfloat16 value is unexplained and informational only.
- Rounding-flip warning (K/V perturbed by 4 ulps float32 or 1 ulp bfloat16 in 10% of elements,
  seed 12345): float32 flipShare 0.0001%, no warning; bfloat16 1.5% (8-bit) and 9.9% (4-bit),
  both above the 25% warning level. A one-ulp bfloat16 change is large next to bfloat16's
  quantization step, so this mostly shows how little accuracy margin bfloat16 K/V has.
- D4 scope variants (handles retained, released early, intermediate `eval`) were not measured: they
  matter only to candidate B, which was not built.
- Unmeasured because the decision came first: `quantized_matmul` with a genuinely broadcast batch
  axis (not needed by the folded layout), chunked-prefill memory, and every model-level gate.

**What a reopening would have to change.** The error is the 4-bit affine rounding of K/V itself,
and sharper attention makes it worse, so changing P1's inputs is not a path to 4-bit keys. Both
options below are untested hypotheses, not measurements:

- Keep keys in float and quantize only the values. Value rounding enters the output linearly,
  whereas key rounding moves which tokens the softmax selects.
  This gives up about half of the saving. Per token per head for D=64 bfloat16 at 4 bits, group 64
  (D2 formula, not a measurement): float K and V are 128 + 128 = 256 bytes, packed K and V are
  36 + 36 = 72 (28%), but float keys with packed values are 128 + 36 = 164 (64%).
- 8-bit only, which is marginal against P1 as written (15 of 27 cases pass), and under the plan's
  pass condition cannot pass on its own, because that condition requires the 16-dimension float32
  4-bit setting.

Any reopening is a dated amendment to the plan that scopes the new design, measures P1 on K/V
captured from a real Tier-B checkpoint instead of random-normal inputs, and migrates 6.5's
`keys()`/`values()` consumers. It also needs a real checkpoint for accuracy gates: six of the seven
synthetic fixtures were too tie-prone for the flip limit in step 0.

## Phase 6.5 direct versus batched decode

`:jmlx-examples:benchmarkBatchDecode` runs the same heterogeneous requests once sequentially
through the direct path and once through a `BatchGenerationScheduler`, each in the same JVM with the
two runs alternating (model load is outside both clocks). It downloads nothing and writes
`<prefix>.json` (per-sample wall times, per-request first-token and total latency, peak and
leaked active bytes) and `<prefix>.md`. Memory counters are read only while no other thread uses
MLX: the direct scope and the scheduler are both closed first.

Command, per family, each in a fresh JVM (3 requests, prompt lengths 6/3/5, new tokens 24/8/16,
7 samples after 2 warm-ups, max batch 4, all submitted back to back so queue depth is 3):

```sh
export JMLX_BENCH_DEVICE="$(sysctl -n machdep.cpu.brand_string)"
for family in llama llama31 qwen2 mistral gemma phi3 mixtral; do
  ./gradlew -q :jmlx-examples:benchmarkBatchDecode \
    --args="${PWD}/tools/hf-reference/goldens/checkpoints/${family} ${PWD}/build/p65/${family} 1,7,42,3,19,5;1,9,4;1,2,3,4,5 24,8,16 7 2 4"
done
```

Run on Apple M2 Max, Java 25.0.3, `mlx-metal==0.31.2`, mlx-c `fba4470`, commit `1a98704` (the
working tree also held the uncommitted benchmark class). Median tokens/s over the 48 generated
tokens:

| Family | Direct | Batched | Speedup | Peak active bytes (direct / batched) | Leaked after run |
| --- | ---: | ---: | ---: | --- | ---: |
| llama | 1149 | 1128 | 0.98 | 454,648 / 739,043 | 0 |
| llama31 | 1155 | 1331 | 1.15 | 454,680 / 753,414 | 0 |
| qwen2 | 1120 | 1047 | 0.94 | 461,816 / 780,006 | 0 |
| mistral | 1102 | 1365 | 1.24 | 454,784 / 754,177 | 0 |
| gemma | 883 | 1169 | 1.32 | 483,592 / 933,579 | 0 |
| phi3 | 1168 | 1238 | 1.06 | 454,648 / 753,382 | 0 |
| mixtral | 817 | 918 | 1.12 | 1,179,648 / 1,557,003 | 0 |

How to read this: these are the tiny synthetic fixtures, so each step is dominated by fixed
per-call overhead, not arithmetic, and the speedups (0.94-1.32) are within run-to-run noise for
some families. They show that batching is correct, leak-free (active bytes return to the baseline in
every run) and not slower in aggregate, **not** what it buys on a production-size model; that needs
a Tier-B run on real weights. The batched peak is the whole run's peak, including the cohort's
cache compaction, and is higher than the direct peak because the cohort holds three rows' caches at
once; it is not broken out per phase. No speed threshold is enforced anywhere in CI.
