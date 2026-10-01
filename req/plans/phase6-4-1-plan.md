# Phase 6.4.1 implementation plan — quantized KV retention and packed attention

**Sequence and blocking.** Phase 6.4 is merged (PR #27, `3eee0cc`). 6.4.1 is time-boxed. Exactly
two dated gates can delay 6.5, and each has a deadline set in step 0, so nothing here is open-ended:

1. **The step 1 decision** (**proceed** or **stop**), due by the **decision date**. Missing it is a
   **stop**.
2. **On a proceed only, step 1b** (the float-only neutral-accessor refactor of D3), due by the **1b
   deadline = decision date + 14 calendar days**, so that 6.5's batching code is written against
   `isEmpty()`/`evalArrays()` rather than float `keys()`/`values()`. If 1b has not merged by its
   deadline, 6.5 starts anyway and migrating its consumers becomes a 6.5 task.

6.5's start is therefore the decision on a stop, and the earlier of 1b's merge and the 1b deadline
on a proceed. **Steps 2–7 never gate 6.5 or the first Central release**; they run alongside it.

- **A stop is final for sequencing.** Later "proceed" evidence does not reopen it silently:
  reopening needs a dated amendment to this file stating the new evidence, a new decision date, and
  a deliverable that **migrates every 6.5 consumer of `keys()`/`values()`** to the neutral accessors
  (6.5 code written after a stop, or after a missed 1b deadline, will use float `keys()`/`values()`
  and would otherwise break on a quantized cache). A reopening never re-blocks 6.5.
- Quantized KV is opt-in, and `GenerationCachePolicy.quantized(bits, groupSize)` /
  `KVCachePolicy.quantized(bits, groupSize)` already fail with a named unsupported-capability
  exception; that is a safe state to ship in.

**Goal:** make compressed K/V storage useful during decode: attend without reconstructing the whole
float cache, reduce steady and peak native memory at a fixed long context, and preserve the default
float generation path.

**Evidence:** [Phase 6.4 benchmark](../phase6-4-benchmark.md) records the pinned macOS CI probes.
Native affine quantization accepts groups 32/64/128 and (as probed) 4 and 8 bits, while every
synthetic family fixture has 16-dimensional heads. Padding D=16 to 32 packs successfully, but the
pinned `scaled_dot_product_attention` rejects packed K/V. Materializing all float K/V before SDPA
defeats the peak-memory goal. The probe's 64-token K for D=16 padded to 32 at 4 bits, group 32,
measured 1,536 bytes: 24 bytes per token, i.e. 16 packed plus 8 for float32 scale and bias. So
float32 scale/bias dtype is confirmed for float32 input; bfloat16/float16 input is not. The probe
did not test cache lifecycle because no useful attention path existed.

## Decisions fixed by this plan

These are settled before any prototype so the prototype measures the real integration surface.

### D1. Policy shape: retention and quantization are orthogonal

`KVCachePolicy.Mode` stays `FULL | SLIDING_WINDOW`; `evicts()` and `requireCapacity` keep branching
on it unchanged. Quantization is a separate nullable component, never a third mode.

```java
// jmlx-core, se.alipsa.jmlx.nn
public record KVQuantization(int bits, int groupSize) {}   // bits in {4, 8}; group in {32, 64, 128}
public record KVCachePolicy(Mode mode, int limit, KVQuantization quantization) {
  public KVCachePolicy withQuantization(int bits, int groupSize);   // keeps mode and limit
  public boolean isQuantized();
}
// jmlx-models, se.alipsa.jmlx.models
public record GenerationCachePolicy(Mode mode, int limit, KVQuantization quantization) {
  public GenerationCachePolicy withQuantization(int bits, int groupSize);
}
```

- Existing factories (`full()`, `full(int)`, `slidingWindow(int)`, `slidingWindowFromModel()`) pass
  `null`. The only constructor call sites today are those factories, so changing the canonical
  constructor needs no compatibility shim.
- `GenerationCachePolicy.resolve(Integer descriptorWindow)` resolves retention exactly as today
  (`SLIDING_WINDOW_FROM_MODEL` takes the checkpoint window) and carries `quantization` through
  unchanged. `slidingWindowFromModel().withQuantization(4, 32)` is the quantized Mistral window-4
  case.
- `DecoderModel.requireCompatiblePolicy` compares policies with `equals`, so mixed
  quantized/float layer caches are rejected with no new code.
- The two named-unsupported `quantized(bits, groupSize)` factories stay throwing until the exit gate
  passes. They cannot carry a retention, so on passing they are **removed** and replaced by
  `withQuantization`; `KVCachePolicyTest` and `GenerationCachePolicyTest` change in the same commit.

### D2. Parameter validation rejects combinations that cost more than float

Quantization groups run along the head dimension D per token. D is padded up to a multiple of the
group size, so per token per head:

`packed = Dpad * bits / 8 + 2 * (Dpad / group) * scaleBytes`, `float = D * elementBytes`,
`Dpad = ceil(D / group) * group`.

Scales and biases follow the input dtype: float32 is confirmed (see Evidence); bfloat16/float16 is
**unconfirmed until the prototype probes it**, and the accounting must use the observed dtype.

| D | Dtype | Setting | Packed + scales/biases | Float | Verdict |
| ---: | --- | --- | ---: | ---: | --- |
| 16 | float32 | 4-bit, group 32 | 16 + 8 = 24 | 64 | accept (37.5%) |
| 16 | float32 | 8-bit, group 32 | 32 + 8 = 40 | 64 | accept (62.5%) |
| 16 | float32 | 4-bit, group 64 | 32 + 8 = 40 | 64 | accept (62.5%) |
| 16 | float32 | 4-bit, group 128 | 64 + 8 = 72 | 64 | **reject** |
| 16 | bfloat16 | 4-bit, group 32 | 16 + 4 = 20 | 32 | accept (62.5%) |
| 16 | bfloat16 | 8-bit, group 32 | 32 + 4 = 36 | 32 | **reject** |
| 64 | bfloat16 | 4-bit, group 64 | 32 + 4 = 36 | 128 | accept (28%) |
| 64 | bfloat16 | 8-bit, group 64 | 64 + 4 = 68 | 128 | accept (53%) |

Rule: reject any combination where `packed >= float`, any unsupported bits/group, any non-float K/V
dtype, and any native-capability failure, always **before** prefill or any cache mutation. Never
fall back silently to float retention. The error names the computed byte counts.

Where the dtype comes from. `jmlx-models` has no "compute dtype" concept today, and
`GenerationCachePolicy.resolve(Integer)` receives only the sliding window. Therefore:

1. **Authoritative check, at `KVCache.append` before mutation**, using the actual K/V dtype and
   head dim. This is the only check that can be exact, and it also covers callers that build their
   own caches and call `forward`.
2. **Optional early check in `DecoderModel`** (called by `generate()` before prefill, and by
   `resolveCachePolicy`), using the embedding weight's dtype as the expected activation dtype.
   Step 0 verifies that this equals the K/V dtype actually produced for every fixture and Tier-B
   checkpoint. If it can differ (for example a quantized embedding), drop the early check rather
   than guess; `resolve(Integer)` keeps its current signature.
3. The check itself is a pure function (`KVQuantization.requireSmallerThanFloat(int headDim, DType
   dtype)`) in `jmlx-core`. The bfloat16 and float16 cases are **unit tests of that function**, not
   end-to-end tests, because every fixture checkpoint is float32.

### D3. Cache accessor contract: `keys()`/`values()` stay float-only

For a quantized cache `keys()`/`values()` throw `UnsupportedOperationException` naming the typed
API. They do **not** return `null`: that is the "empty cache" signal `DecoderModel` already branches
on, so a `null` would be silently misread as empty. One public `KVCache` type remains
(`DecoderModel` and its callers take `List<KVCache>`); the storage representation is an internal
strategy (float vs packed), not a second public cache class.

| New `KVCache` accessor | Replaces / purpose |
| --- | --- |
| `isEmpty()` | `keys() == null` checks |
| `kvHeads()`, `headDim()` (logical, unpadded) | `keys().shape()[1]`, `[3]`. **Both return 0 while the cache is empty**, matching `batchSize()`, `length()` and `rowLength(row)`, because `KVCache(MLXScope, KVCachePolicy)` receives no shape before the first `append`. Callers must test `isEmpty()` first (`validateCacheShape` already returns early on an empty cache); Javadoc says so, and 1b sets this behavior for 6.5 |
| `batchSize()` (already exists) | `keys().shape()[0]` |
| `evalArrays()` (every owned native array, packed or float) | `keys()`/`values()` pairs for the step boundary |
| `PackedKv packedView(int bufferOffset, int count)` | typed read of packed K/V + scales + biases. `bufferOffset` indexes the **retained buffer**, never an absolute position (`KVCache.startPosition(row)` is already an absolute per-row position); `PackedKv` is a named record of the six arrays plus logical shape |
| `dequantizedKeys(...)`/`dequantizedValues(...)` | **package-private**, diagnostic only, bounded by a documented block size. Core tests in `se.alipsa.jmlx.nn` reach them; models-level tests compare logits instead and never materialize |

Call sites to change (verified at `3eee0cc`):

- `DecoderModel` ~L286 (`cache.keys() != null && …shape()[0] != batch`): silently skipped if
  `keys()` returned `null`; use `isEmpty()`/`batchSize()`.
- `DecoderModel` ~L313–317 (`validateCacheShape`): would treat a `null` as an empty cache; use
  `isEmpty()`, `kvHeads()`, `headDim()`.
- `DecoderModel` ~L340–344 (`cacheArrays`): this array is what the sampler's step boundary
  evaluates (also `forward` at ~L97/L119). Without `evalArrays()` packed storage would never be
  evaluated at the boundary.
- `DecoderAttention` ~L204–205: the only consumer that gains a packed path.
- `GroupedQueryAttention` ~L126–129 and `MultiHeadAttention` ~L121–124: **not** extended. In both,
  `cache.append(k, v)` runs *before* `cache.keys()`, so a rejection placed at the read would leave a
  quantized cache already mutated. They reject a quantized cache **before `append`** with an
  `IllegalArgumentException`, and a test asserts the cache's length, positions and poisoned flag are
  unchanged afterwards.

### D4. Block intermediates are scoped and measured, not assumed free

`MLXScope` keeps every handle until `close()` (or an explicit `MLXScope.free`), and generation uses
one activation scope per step (`DecoderModel` ~L456). A streaming design that creates one
dequantized block per scope handle may keep every block resident until the step boundary, which is
close to a full float cache. Any streaming candidate must build block intermediates in a child scope
closed (or freed) as soon as the block is chained into the accumulator. Closing a handle does not by
itself release a lazy array's buffer before evaluation, so the prototype measures peak bytes under
three variants: handles retained, handles released early, and an intermediate `eval` per block
(which costs a host synchronization). Unverified until measured.

### D5. Memory-gate definition, chunked prefill, and long-context checkpoint

- **Metrics.** The model-level gates use the **per-decode-step peak** active bytes at context N
  (`MLXMemory.resetPeak()` immediately before one decode step after the context is built, as
  `decodePeak` does today), plus steady retained bytes after the step.
  `peak_generation_active_bytes` includes prefill and is **not** a gate metric. **N = 4096, batch
  1.**
- **Scaling check (proves "no full float materialization").** A strictly-lower peak is not enough: a
  quantized step that dequantized the whole cache can still sit below a float step that holds old
  and new cache during `append` (illustration for D=64, bfloat16, 4-bit, per token per head: float
  step peak ≈ 2 × 128 = 256 bytes versus a full-dequantizing quantized step ≈ 2 × 36 + 128 = 200
  bytes; this is arithmetic from the D2 formula, not a measurement). So also measure the
  per-decode-step peak at N₁ = 2048 and N₂ = 4096 and compute the growth per token
  `s = (peak(N₂) − peak(N₁)) / (N₂ − N₁)`.

  The bound must be built from the **failure pattern itself**, not from the float size and not from
  the float run's measurement (the float path's `repeatKeyValueHeads` is `broadcastTo` then
  `flatten`, which probably copies, unmeasured, so a float multiplier could be 5–6 rather than 2). A
  fixed `s_quant < floatBytesPerToken` bound is wrong: it charges the quantized side for the
  `append` transient but compares with a single float buffer, so it rejects a correct path whenever
  packed is more than half of float (D=16 float32 8-bit is accepted by D2 at 62.5% yet would cost
  `2 × 80 + 16 = 176 > 128`). The ratified bound is:

  `s_quant < a × packedBytesPerToken + 0.5 × floatKeptBytesPerToken + scoreKeptBytesPerToken`

  - `a` is the **measured append multiplier**: peak growth per token of an append-only step on
    packed storage (concatenate and evaluate, no attention) divided by `packedBytesPerToken`. It is
    about 2 if `append` transiently holds old and new buffers and about 1 if it does not. Its value
    depends on the append implementation and on whether MLX frees each layer's old buffer as it
    goes, so it is measured at **two levels** and each check uses its own value:
    - *Step 1, for P3:* on the prototype's append, recorded in `req/phase6-4-benchmark.md` as the
      **first** act of step 1, before any attention candidate is measured.
    - *Step 6, for G5:* re-measured on the **step 2 `KVCache`** as an append-only step through
      **all layers** at the same N₁ and N₂, recorded before G5's quantized run. G5 never reuses
      the step 1 value: if step 2 changes the append strategy (for example a preallocated buffer
      with slice-update, which would make `a` about 1), the re-measurement is what counts.
  - `layersKept` is the number of layers whose intermediates are live at the same time at the
    step's peak. It is the layer count unless D4's measured variants show that the step releases a
    layer's intermediates earlier, in which case the **measured** count is used. It is counted from
    the **final graph** (the step 4 integration, not the prototype) and recorded with the G5 run.
  - `floatKeptBytesPerToken = floatBytesPerToken × layersKept / layers`: the float size of the
    layers a full-dequantizing path would hold at once. It is retained the same way as the score
    term because a path that dequantizes one layer and frees it before the next adds only
    `float / layers` per layer kept, not the whole float term. With `layersKept = layers` this is
    `floatBytesPerToken`; with 2 layers and `layersKept = 1` it is half of it, so the
    unconditional `0.5 × float` slack would sit exactly on the boundary and let that path through.
  - `0.5` is the ratified, never-tuned slack factor. Full dequantization adds a whole
    `floatKeptBytesPerToken` over a correct path, so the two are separated by
    `0.5 × floatKeptBytesPerToken` for every setting D2 accepts, independent of score size.
  - `scoreKeptBytesPerToken = scoreIntermediates × layersKept × scoreDtypeBytes × queriesPerKv × L`.
    Candidate A builds **at least three** score-sized arrays (raw scores, scaled scores, softmax
    weights; four with the masked copy), and D4 says `MLXScope` keeps every handle until the step
    scope closes, so they may all be live, and across every layer rather than one.
    `scoreIntermediates` is counted from the final graph. Never assume one tensor in one layer.
  - Bytes per token are summed over K and V and logical kv heads (and over all layers at model
    level), with `packedBytesPerToken` from the D2 formula.

  **Error rule.** The bound separates a correct path from full dequantization by exactly
  `0.5 × floatKeptBytesPerToken`, so the *signed total error* of its inputs,
  `err = (a_used − a_real) × packed + (scoreKept_used − scoreKept_real)` plus
  `0.5 × (floatKept_used − floatKept_real)`, must stay strictly inside `(−0.5, +0.5)` times the
  **counted** `floatKeptBytesPerToken` (built from the counted `layersKept`, not from a used or
  assumed value). An overestimate of more than `+0.5 × floatKept` lets full dequantization through
  (the plan's own D=16 f32 8-bit case: assuming `a = 2` where the real value is 1 gives
  `err = 80 > 64`); an underestimate of more than `−0.5 × floatKept` rejects a correct path. So the
  inputs are measured or counted, never chosen as a "safe" upper bound, and the report states each
  one's measured value and its residual uncertainty. The measured `s_quant` is itself noisy: it is
  the median of 5 repeated step measurements, reported with its min–max spread. If the residual
  uncertainty of the inputs **plus the `s_quant` spread** could reach `0.5 × floatKept`, the
  model-level check is recorded as **inconclusive**. An inconclusive G5 is **not** a pass by default
  and is resolved by one fixed rule: it passes only if P3, the per-call check, **also passes on the
  integrated step 4 `DecoderAttention` path** (not only on the prototype) for **every distinct layer
  configuration in the model** (kv heads, head dim, sliding window), with both results recorded.
  That P3 run uses `a` **re-measured per call on the step 2 `KVCache`** (an append-only call at the
  same S₁/S₂), not the step 1 prototype's, under the same error rule: if step 2 changed the append
  strategy, the old `a` is wrong by `(a_old − a_new) × packed`, enough to let full dequantization
  through (80 > 64 in the D=16 f32 8-bit case). If any of those is also inconclusive or fails, G5
  fails. This is expected when `layersKept` is small and `floatKept` is a small share of the step
  peak, so noise can cover it; how noisy the allocator's peak is has not been measured. A per-call
  P3 is a single layer, so it separates a correct path from full dequantization by `0.5 × float` of
  that layer, with no model-level dilution.

  Report the float run's measured multiplier for information only.

  Worked check (arithmetic from the D2 formula, not a measurement): per kv-head-token, K and V
  together, one layer (so `layersKept = layers` and `floatKept = float`), `queriesPerKv = 4`,
  `L = 1`, float32 scores (16 bytes per intermediate), three intermediates (48), and `a = 2`.

  | Setting | packed / float | Bound | Correct path (`2 × packed + 48`) | Full dequantization (`+ float`) |
  | --- | ---: | ---: | ---: | ---: |
  | D=16 f32, 4-bit, g32 | 48 / 128 | 96 + 64 + 48 = 208 | 144 (passes) | 272 (**rejected**) |
  | D=16 f32, 8-bit, g32 | 80 / 128 | 160 + 64 + 48 = 272 | 208 (passes) | 336 (**rejected**) |
  | D=64 bf16, 4-bit, g64 | 72 / 256 | 144 + 128 + 48 = 320 | 192 (passes) | 448 (**rejected**) |

  The previous float-size bound would have failed the first row's correct path (144 > 128) once
  three retained intermediates were counted.
- **Chunked prefill needs a named entry point.** Today `DecoderModel.prefill` (private, ~L540) runs
  `normalizedHiddenStates` over the entire prompt and projects only the last position, so it does
  not chunk; the benchmark builds its context through public `forward` (`DecodeBenchmark` ~L101),
  which computes logits for every position (`"prefill_logits": "all_positions"`), a
  `[1, 4096, vocab]` tensor at N = 4096 that would swamp the comparison. Add:
  - `public final MLXArray DecoderModel.prefillChunked(MLXArray tokenIds, List<KVCache> caches, int
    chunkSize)`: processes `ceil(len / chunkSize)` chunks and returns last-position logits only.
    At each boundary it makes one host synchronization (per chunk, none per layer): for every chunk
    **before the last** it evaluates `evalArrays()` only, which already forces every layer's K/V for
    the chunk. Forcing the hidden state as well would run the last layer's MLP and the final norm
    for a result that is discarded. Only the **last** chunk also evaluates its last-position
    logits. It is named `prefillChunked` so it cannot be confused with the existing private
    `prefill(ids, caches)` (~L540), which stays the unchunked path.
    - **Each chunk is built in its own child scope**, closed right after that chunk's evaluation.
      `MLXScope` frees nothing before `close()`, and `generate()` runs a whole step in one
      activation scope (`DecoderModel` ~L456); if every chunk's hidden states and per-layer score
      tensors lived in that scope they would all be held until the step ended and peak memory would
      grow with the full prompt, exactly as unchunked prefill does. Cache storage stays in the
      cache's own scope, as `KVCache.append` already arranges by hoisting. Only the last chunk's
      last-position logits are hoisted out, to `work` (see *Scopes* below).
    - **Capacity and shape are checked for the whole prompt before chunk 1** (`preflight` over the
      full `tokenIds`, including `requireCapacity(position + length)`), so a `FULL(limit)` overflow
      is a clean `IllegalArgumentException` before any cache write. Without that, an overflow found
      mid-prompt would arrive after earlier chunks mutated the caches and poison the cache set.
      Per-chunk preflight then re-checks positions only.
    - Poisoning is otherwise as in `forward`: a native or eval failure after the first chunk's
      write poisons the whole cache set.
  - `GenerationRequest.withPrefillChunk(int chunkSize)`, stored as an `OptionalInt` so "not set" is
    distinct from every explicit value. `chunkSize` must be positive; there is **no** `0` sentinel.
    Resolution at `generate()`:
    - Unset + float policy → today's unchunked prefill, so the float default and its goldens are
      unchanged.
    - Unset + quantized policy → chunk size C = 256.
    - Explicit `n` → chunked with `n` for either policy (so a float run can match a quantized run's
      chunking for G1–G3, as D5 requires). Setting it on a float request is allowed.
    - **Unchunked quantized prefill is not selectable**: with no `0` sentinel, a quantized request
      is always chunked, which bounds the transient float K/V and score tensors that unchunked
      prefill would create. An explicit `n` larger than the prompt is the (single-chunk) escape
      hatch and is allowed.
  - **Order: append, then attend, on the packed path too.** `DecoderAttention` appends the chunk's
    K/V to the cache before reading it, and the packed path keeps that order, so *every* query,
    including a chunk's own tokens, attends to the packed keys read back from the cache. The chunk's
    float K/V exist only transiently, as the input to the append. Chunking therefore does **not**
    change what the quantized path attends to: quantization is per token, so the packing rule does
    not depend on the chunk size. The packed values are not bit-identical across chunk sizes,
    though: reduction-order differences upstream shift the float K/V slightly, and in rare elements
    that flips a quantization rounding bucket, a one-step (about `scale`) difference.
  - **Matched chunk size.** **G1–G3 and the memory runs use the same explicit chunk size for float
    and quantized**, for two reasons: float chunked and unchunked prefill differ in reduction order,
    which would add noise to the accuracy comparison, and per-step peak depends on the chunk's
    transient float and score tensors. The chunk size is recorded in the report. Chunked prefill
    with a sliding policy (chunk larger than the window) gets its own test.
  - **Step 5 tolerance (chunked versus single-chunk quantized prefill).** It uses a frozen
    tolerance, not "float epsilon": max `|Δlogit|` ≤ `0.25 × deltaCap` of the setting (0.25% of the
    float logit range for 8-bit and 1.25% for 4-bit), proposed here and ratified in step 0. Step 0
    also records the float chunked-versus-unchunked noise, which does not need `prefillChunked`
    (built in step 4): it is emulated with the existing public `forward(tokenIds, caches)` called
    once per chunk on the same caches (`preflight` accepts a populated, unpadded cache), compared
    against one `forward` over the whole prompt, using the **last-position** logits of each
    (`forward` returns all positions). **Rule, per fixture:** the tolerance is defined per setting
    (8-bit or 4-bit) but applied per fixture. If a fixture's recorded float noise is at or above
    **half** the setting's tolerance, step 0 raises the tolerance **for that fixture only** to `2 ×`
    its noise before ratifying it, so step 5 cannot be the first to hit it. The other fixtures keep
    `0.25 × deltaCap`. A fixture's tolerance never exceeds `deltaCap`: if `2 ×` its noise would,
    chunked prefill is too noisy to compare against on that fixture, and the plan is amended (dated,
    with the evidence) before step 1 begins.
  - **Rounding-flip probe (step 1; informational warning, not a criterion).** The 4-bit threshold is
    unmeasured: one rounding flip moves an element by a whole quantization step (about 1/15 of its
    group's range), and whether mixing across positions, heads and layers keeps that small is not
    known. A one-layer probe is given its K/V, so chunking can matter only through tiny float
    differences in them. The probe therefore perturbs a seeded random **10%** of the float K/V
    elements (fixed seed `12345`, independent random signs; ratified in step 0) by **±k ulps of the
    K/V dtype**, with `k = 4` for float32 and `k = 1` for bfloat16 and float16 (proposed, ratified
    in step 0). `k` is per dtype because one bfloat16 step is about `2^-8` (0.39% of the value), so
    4 steps would be about 1.6%: more than a whole 8-bit quantization step (roughly 0.4–0.8% of the
    group's largest value), which would move most elements across a bucket and measure the
    perturbation instead of rounding flips. At `k = 1` and 10% of elements, the perturbation is the
    size of real reduction-order noise (at most about one step, in some elements); in float32,
    `k = 4` is about 5e-7 relative, so only elements near a boundary flip (arithmetic from the dtype
    spacing, not a measurement). Definition, so two implementations agree: the ulp is the dtype's
    spacing at `|x|` (its exponent's step), `x' = x ± k × ulp(|x|)` computed in float32 and cast
    back to the K/V dtype, and for `x = 0` the ulp is the dtype's smallest normal value. The seed
    and sign pattern make the warning reproducible between CI runs, which matters because whether
    any element crosses a rounding boundary depends on which elements sit near one.
    The probe quantizes both versions and runs the same attention call (`L = 256` chunk, S = 2048,
    4- and 8-bit, each dtype in the probe matrix). It also runs **float SDPA on the original and the
    perturbed K/V**, so four outputs exist: `attnQ(K, V)`, `attnQ(K', V')`, `attnF(K, V)` and
    `attnF(K', V')`. It records three numbers in **P1's units** (a max abs difference as a % of the
    float output's max abs value): `quantizedDiff = max|attnQ' − attnQ|`,
    `floatDiff = max|attnF' − attnF|`, and the warning metric
    `flipShare = max over elements of |(attnQ' − attnQ) − (attnF' − attnF)|`. `flipShare` is
    computed element by element, not as `quantizedDiff − floatDiff`: that subtraction is never
    larger than `flipShare` (triangle inequality) and cancels a real flip effect when the two errors
    peak at different elements. `flipShare` is the part of the quantized path's response to the
    perturbation that the float path does not show. All three are recorded side by side. The warning
    level applies to `flipShare` only and is compared with P1's own bound for that setting: **above
    25% of P1's bound** (0.25% for 8-bit, 1.25% for 4-bit) is recorded as a warning, to prompt a
    look at the step 5 result. This is **not** comparable with the step 5 tolerance, which is a
    logit difference over the whole model, and the plan defines no mapping between them. **The
    step 5 tolerance stays as ratified** whatever the probe shows, and is never loosened after a
    quantized result. The probe runs as its own test method, since it is small (`L = 256` queries
    against S keys, no 4096-query single chunk) and must not disturb P2/P3's peak-memory numbers.
  - **Scope of `prefillChunked` (stated so 6.5 does not assume more):**
    - *Unpadded only.* It uses the same `preflight`, which calls `requireUnpadded`, and has no
      `validLengths` variant. A left-padded batched prefill is **6.5 work**, not part of this plan.
    - *No cancellation polling.* The public contract (`req/plans/phase6-plan.md` 6.4 task 1) is
      that cancellation is observed before prefill and between decode steps, not during prefill,
      and `generate()` polls only between steps. A 4096-token prompt in 16 chunks is therefore not
      cancellable until prefill finishes. Chunk boundaries do **not** poll the token: polling would
      need the token as a parameter and a reviewed contract change, and is left to 6.5 if its
      scheduler needs it. The javadoc says so.
    - *Scopes.* Let `work = MLXScope.innermost(tokenIds.scope(), scope())`, where `scope()` is the
      model's scope. Every layer op allocates into the innermost scope among its operands, which
      includes the weights, and two unrelated scopes are rejected. A chunk scope created as
      `tokenIds.scope().newChild()` would be a *sibling* of the model scope whenever the model scope
      is nested deeper than `tokenIds`, and every layer op would fail. So each chunk scope is
      `work.newChild()`, computed (with the whole-prompt capacity check) in `preflight` before any
      cache write, so unrelated `tokenIds` and model scopes fail as a clean
      `IllegalArgumentException`, not after chunk 1 has poisoned the cache set. The returned
      last-position logits are hoisted to `work` (the same scope `forward`'s result would land in).
      The caches' scope must be an ancestor of `work`, because `KVCache.append` hoists into it;
      `generate()` satisfies this (cache scope `generation`, `work` is the step's activation scope).
      An optional preflight check that the cache scope is an ancestor of `work`, so that a bad setup
      fails before chunk 1 and not mid-prompt, is welcome but not required: `forward` has the same
      exposure today. The per-chunk child scopes and their intermediates are closed before it
      returns.
- **Long-context checkpoint.** All seven synthetic fixtures declare `max_position_embeddings: 128`
  and the main code does not enforce that limit outside RoPE scaling (`ArchitectureMappings` uses it
  only for dynamic-NTK/YaRN). Running a tiny fixture past 128 gives memory numbers but meaningless
  accuracy. Memory at N may use any fixture; **accuracy at N (G3) requires a real checkpoint.** See
  step 6 for the pinned Tier-B candidates, prompt, test, and job.

## Attention candidates (ordered by cost; stop at the first that passes the prototype criteria)

**A. Packed `quantized_matmul` attention (baseline; needs no new binding).**
`MLXQuant.quantizedMatmul` already wraps `mlx_quantized_matmul`, which multiplies without
dequantizing the packed operand (`MLXQuant.java` ~L190 and ~L226).

1. `scores = quantizedMatmul(q, Kq, scales, biases, transpose=true, group, bits)`, multiplied by the
   **existing `DecoderAttention.scale`** (`1/sqrt(logical headDim)`), so the two paths cannot drift;
   the packed path lives in `DecoderAttention` (or a package-private helper that takes `scale`).
2. **Masking.** `AttentionMask.slidingWindow` and `AttentionMask.batched` return **BOOL** masks
   (true = attend) and SDPA consumes them as such, so the manual path applies
   `MLXOps.where(mask, scores, lowest)`, where `lowest` is the **scores dtype's lowest finite
   value** (not `-inf`); adding a BOOL is wrong. For plain causal attention
   `DecoderAttention.attendWithMask` passes `causal = (mask == null)` and SDPA builds the mask
   internally, and `AttentionMask` has **no causal builder**. Multi-token and chunked queries would
   therefore be unmasked in candidate A. Add
   `AttentionMask.causal(scope, queryStart, queryLength, keyLength)` (BOOL, absolute positions,
   sharing the arange logic of the sliding builder) and use it whenever `L > 1` and no other mask
   applies. The batched mask is `[B,1,T,S]`; the 5-D GQA layout below needs `[B,1,1,T,S]` (an extra
   axis). A row with **no** attendable key (a padded query, a key row that is entirely left padding,
   or a sliding window with no keys left) must not produce NaN: with `-inf` the softmax of an
   all-masked row is NaN, whereas the dtype's lowest finite value yields a finite (uniform) row that
   downstream code then ignores or masks. This protects every mask kind, not just padded queries. (I
   recall `mlx-lm` masking quantized attention with `mx.finfo(dtype).min` for this reason; that is
   an unverified external claim.) Reuse `batched()`'s padded-query handling and test padded queries,
   all-padding key rows, and empty sliding windows; the float-vs-quantized comparison must treat
   such rows as "don't care", not as agreement.
3. `MLXOps.softmaxAxis(scores, -1, precise=true)` (float32 reduction).
4. `out = quantizedMatmul(weights, Vq, scales, biases, transpose=false, group, bits)`, then slice
   the padded value columns off.

Zero-padding `q` to `Dpad` makes padded K dimensions contribute nothing to the dot product. Padding
never becomes an attendable feature or key position. Grouped-query attention must broadcast query
heads over packed K/V without `repeatKeyValueHeads` on packed tensors (which would copy them):
reshape queries to `[B, kvHeads, queriesPerKv, L, Dpad]` and expand the packed operands' head axis
(confirm batched-broadcast support of `quantized_matmul` on the pinned runtime). The one transient
is the float score tensor `[B, H, L, S]`: small for decode (`L = 1`), bounded for prefill by
chunking (D5). I recall `mlx-lm` using this structure for quantized KV attention; that is an
unverified external claim, and the prototype decides.

**B. Bounded-block streaming attention (only if A fails the prototype criteria).** Dequantize
fixed-size key/value blocks and combine with an online (running max/sum) softmax so only one block's
float form is live. Subject to D4.

**C. Fused packed-KV Metal kernel (only if A and B fail).** `mlx_fast_metal_kernel*` is already in
the generated bindings but is marked `unplanned` in `req/mlx-api-inventory.md` (~L63–66, ~L338+).
Choosing it means a handwritten wrapper plus an entry in `req/mlx-api-inventory-overrides.json`,
`./gradlew generateMlxApiInventory`, `verifyMlxApiCallSites`, `verifyMlxApiHeaderCoverage`, a
`NativeLoader`/ownership review, and package tests. Do not start it without recording why A and B
failed.

## Acceptance criteria (frozen before any measurement)

These are proposed starting values. Ratify or change them in this file **before** step 1 runs.
Afterwards they change only by a dated amendment here that states the evidence, never by adjusting
them to fit a result.

### Prototype criteria (attention-op level; the step 1 / stop-rule decision)

Measured by a standalone `jmlx-core` probe on a single attention call, not on a model. Shapes:
`B=1`, GQA (`kvHeads=2`, `queriesPerKv=4`), D=64 float32 and bfloat16, plus **D=16 float32 (padded
to 32)** which is mandatory, at S = 2048 and 4096, `L=1` and a 256-token chunk, 4- and 8-bit, and
every mask kind (causal, sliding, left-padded batch with unequal positions).

| Id | Criterion |
| --- | --- |
| P1 | max abs error of the attention output vs float SDPA on identical inputs: 8-bit ≤ 1% and 4-bit ≤ 5% of the float output's max abs value; no NaN/Inf, including padded query rows |
| P2 | peak active bytes during one decode-step attention including its `append` (retained cache plus transient, same conditions as the float run) strictly below float's at S = 4096 |
| P3 | the D5 scaling check, applied per attention call between S = 2048 and 4096 (not model-level), with its single failure-pattern bound `s_quant < a × packed + 0.5 × floatKept + scoreKept` (measured append multiplier `a`, counted score intermediates; one layer, so `floatKept = float`), for one kv-head-token |
| P4 | median time per call over 20 timed calls after warmup ≤ 2.0 × float SDPA's at S = 4096, `L=1` |
| P5 | host synchronizations inside the attention op: zero for candidate A (lazy graph, caller evals once); counted and reported for B and C |

A candidate **passes** when P1–P5 hold for D=16 float32 4-bit group 32 and for at least one D=64
setting. Memory and time are measured against a float cache holding the same S tokens.

### Model-level gates (the exit gate; fixtures and Tier-B)

| Gate | Threshold |
| --- | --- |
| G1 accuracy, 8-bit, all seven fixtures | **teacher-forced** over 32 positions (the float run's greedy tokens are fed to both runs): max `\|Δlogit\|` ≤ `deltaCap` = 1% of that position's float logit range at every position; top-1 flips (raw, over all positions) ≤ `floor(excluded / 2)` |
| G2 accuracy, 4-bit, all seven fixtures | same teacher-forcing; max `\|Δlogit\|` ≤ `deltaCap` = 5% of float range at every position; top-1 flips (raw, over all positions) ≤ `floor(excluded / 2)`; first-divergence index reported |
| G3 accuracy at N = 4096, real checkpoint | teacher-forced, 64 positions; max `\|Δlogit\|` ≤ `deltaCap` = 5% of float range **(its own cap, same as G2)**; top-1 flips (raw) ≤ `floor(excluded / 2)`; max and mean error reported |
| G4 retained memory | steady retained KV bytes at N strictly below float and within 10% of the D2 formula (no hidden float copy) |
| G5 peak memory | per-decode-step peak at N = 4096 strictly below float **and** the D5 scaling check passes, using `a` re-measured on the step 2 `KVCache`, `scoreIntermediates`/`layersKept` counted from the final graph and the `floatKept` term built from `layersKept`, with the D5 error rule (including the `s_quant` spread) satisfied, or, if that check is inconclusive, the integrated per-call P3 passing for every layer configuration with `a` re-measured on the step 2 `KVCache` (an inconclusive G5 is otherwise a failure; G4 and "strictly lower" alone do not prove no full materialization) |
| G6 float default | float goldens byte-identical (the primary regression check); decode tokens/s within ±5% **measured at step 7 as interleaved A/B runs on the same host**: the merge-base build (pre-6.4.1, `3eee0cc` or the later merge base) against the candidate branch, alternating, 5 pairs. The step 0 baseline is informational only, since hosted-runner variance is of the same order and no candidate build exists at step 0 |
| G7 quantized throughput | median tokens/s at N ≥ 50% of float, same interleaved A/B method; the actual ratio is published whichever way it lands |

**Exclusion rule (fixed before measuring, never derived from the observed error).** A position `p`
is excluded from top-1 agreement iff `floatMargin(p) ≤ 2 × deltaCap(p)`, where `floatMargin` is the
float run's top-1 minus top-2 logit and `deltaCap` is the gate's own threshold above (in logit
units, from that position's float range). An observed-error rule would hide every disagreement: to
flip the top-1 the quantized logits must close the float gap, so the observed error is at least half
of it, and a "margin ≤ 2 × observed error" test would exclude exactly the positions that
disagree, leaving agreement at 100% regardless. The ratified rule instead excludes only positions
whose margin is within what the *permitted* error could flip. Consequences, stated so the gates
cannot be read as stronger than they are:

- **The Δ cap is the operative accuracy bound.** At a non-excluded position the margin exceeds
  2 × `deltaCap`, so while max `|Δlogit|` ≤ `deltaCap` the top-1 cannot flip. A top-1 disagreement
  at a non-excluded position therefore *is* a Δ-cap violation and is reported as one. Agreement
  restricted to non-excluded positions is implied by the Δ cap and is **not** the gate.
- **The flip limit is the only agreement check, and it is what the exclusion cap protects.** Flips
  can happen only at excluded positions, so a plain "agreement ≥ 90%" threshold would be implied by
  the 10% exclusion cap and could never fail. The gates therefore limit flips to at most half of the
  excluded positions (`floor(excluded / 2)`), counted over all positions. With no excluded
  position, any flip fails.
- At most **10% of positions** may be excluded **for a fixture or checkpoint that is under the flip
  limit**. More means it is too tie-prone to be informative, and the gate fails as "inconclusive"
  instead of passing. Step 0 checks that the caps can pass before they are ratified (next bullet),
  so "inconclusive" should not arise from a fixture that was never examined.
- **Feasibility is measured in step 0, float-only.** The seven fixtures are 2-layer random-weight
  models (hidden 64, vocab 128), so their logits may be close together and many positions may have a
  top-1/top-2 margin below `2 × deltaCap`; with 32 positions the 10% cap allows only 3 exclusions.
  Step 0 therefore records, **for each fixture, the float-only margin distribution** over the
  teacher-forced positions and the number of positions that would be excluded for each `deltaCap`
  (1% and 5%). This uses no quantized result, so it does not break "frozen before measurement".
  Ratify the caps knowing they can pass. For a fixture where they cannot, change the **inputs**
  before any quantized run: more positions (for example 64 or 128, up to the 128-position fixture
  limit minus the prompt) or a different committed prompt chosen only by its float margin
  distribution. The caps and `deltaCap` values themselves are not loosened for a fixture.
  **Last resort, flip limit only.** If no prompt or length keeps a tiny random-weight fixture's
  exclusions within 10%, step 0 may drop **that fixture from the flip limit only**. It keeps the
  Δ cap (the operative bound), and its dropped status, the margin distribution that justified it
  and the date are recorded in this file **before any quantized run**. A dropped fixture is gated
  by the Δ cap alone: with no flip limit the 10% exclusion cap has nothing to protect and does not
  apply to it. The cap applies only to fixtures still under the flip limit, and is never relaxed for
  those nor for G3. This applies to the seven synthetic fixtures only: G3's real checkpoint keeps
  its flip limit, and an infeasible G3 is "inconclusive" under the normal rule. Whether any fixture
  needs this is known only after step 0's float-only measurement.
- Every excluded `p` is listed in the report with its margin, `deltaCap`, and whether it flipped.

Fixtures (seven, not six): `llama`, `llama31` (Llama 3 RoPE scaling, which matters because RoPE is
applied before K is packed), `qwen2`, `mistral` (window 4), `gemma`, `phi3`, `mixtral` under
`tools/hf-reference/goldens/checkpoints/`.

## Stop rule

The decision is made at step 1 on the **prototype criteria P1–P5**, which the probe can measure. It
does not use G4/G5/G7, which need a full model at N = 4096 and are model-level gates.

- If candidate A passes P1–P5, record **proceed** with A.
- If A fails any of P1–P5 (including P1: a candidate that saves memory but misses accuracy is
  rejected here), measure B; if B fails, decide on C only with a written reason.
- If no candidate passes, record **stop**: keep the named rejection, record measurements and the
  blocker in `req/phase6-4-benchmark.md`, mark 6.4.1 "deferred pending a new attention design" in
  the roadmap, and 6.5 proceeds.
- The decision date set in step 0 is binding: no decision by that date is a **stop**, and a stop is
  reopened only by a dated amendment (see "Sequence and blocking").
- The model-level gates then decide **enablement** (step 7), not the proceed/stop decision. If they
  fail after a proceed, the rejection stays, the blocker is recorded, and 6.5 is already unaffected.

`req/full-roadmap.md` (6.4.1 section) and `req/plans/phase6-plan.md` (6.4.1 section) are updated in
the same commit as this plan so no document contradicts it.

## Implementation work

0. **Freeze decisions and thresholds.** Ratify D1–D5, the prototype criteria and the model-level
   gates in this file (including the D5 slack factor 0.5 and each `deltaCap`), and set the
   **decision date** for step 1 (the 1b deadline is that date + 14 days). Verify the embedding-dtype
   assumption of D2. **Record each fixture's float-only top-1/top-2 margin distribution** and the
   resulting exclusion counts (and the float chunked-versus-unchunked noise via sequential `forward`
   calls, for the D5 step 5 tolerance), and fix the positions and prompts (see the exclusion rule's
   feasibility bullet) before ratifying the G1–G2 caps. Record an **informational** pre-change float
   baseline on the target macOS host (raw JSON committed or attached); it is not the G6 comparison,
   which is the step 7 interleaved A/B run against the merge-base build. Add `llama31` to the
   benchmark document's family loop.
1. **Prototype attention (native, macOS) and decide.** `PackedAttentionProbeTest` in `jmlx-core`,
   following the `KVCacheQuantizationProbeTest` pattern (CI artifact upload, required-suite entry),
   plus `AttentionMask.causal`. **First**, measure and record the append multiplier `a` (D5) on
   packed storage, before any attention candidate. Then run the P1–P5 matrix against float SDPA for
   candidate A; record the K/V-perturbation rounding-flip warning (D5); confirm the scale/bias dtype
   for bfloat16/float16, batched broadcast, the D4 scope variants, and the counted score
   intermediates. Publish the choice, raw numbers, and rejected alternatives in
   `req/phase6-4-benchmark.md` and record **proceed/stop** before touching the public policy.
1b. **Land the neutral accessors (D3) as a float-only refactor (proceed only).** `isEmpty()`,
    `kvHeads()`, `headDim()` (0 while empty), `evalArrays()`; migrate the `DecoderModel` sites, with
    float tests unchanged and green and goldens byte-identical. Independent of quantization. Due by
    the 1b deadline (decision date + 14 days); 6.5 starts at the earlier of its merge and that
    deadline (see "Sequence and blocking"). On a stop it is skipped.
2. **Core policy and storage.** `KVQuantization`, `KVCachePolicy.withQuantization`, the D2
   validator, and `packedView`/diagnostic accessors; then packed storage in `KVCache`. Quantization
   groups lie along D, so token-axis operations act directly on packed tensors without requantizing:
   append is a token-axis concatenation of packed/scales/biases; sliding eviction and `trimToLast`
   are token-axis slices (`MLXShape.slice`, as the float path now does); `reorder` with duplicate
   indices is a batch-axis gather. Define and test: append and chunked prefill, zero-length sliding
   retention, capacity checks (unchanged `policy.requireCapacity`), reset, `fork`/`reorder` into a
   destination scope (independently evaluated; source-scope close must not affect the copy), poison
   and reuse, and left-padded rows. For padded slots the requirement is that they **dequantize to
   finite values and are masked** (not that scale is exactly zero: MLX's affine quantization may
   clamp the scale with an epsilon, which is unchecked on the pinned runtime). All packed K/V,
   scales and biases live in request-owned scopes. **No float tail by default:** a per-token append
   needs none, since groups never span tokens. Add an uncompressed buffer only if the prototype
   shows it amortizes packing cost over chunked appends, bounded independently of context length.
3. **Reject quantized caches in MHA/GQA** before `append` (D3), with the unchanged-state test.
4. **Integrate attention, chunked prefill, and sampling.** In `DecoderAttention`, feed packed
   storage (append, then attend: every query reads packed keys from the cache, D5) to the chosen
   path, preserving per-row masks, absolute positions, the "build the graph before evicting" order,
   and sliding semantics. Add `DecoderModel.prefillChunked(...)` and
   `GenerationRequest.withPrefillChunk(int)` (D5; `OptionalInt`, no `0` sentinel). Evaluate newly
   packed storage at the sampler step boundary with the selected logits via `evalArrays()`; no host
   synchronization per layer or block (per chunk during prefill only). Public `forward` keeps its
   documented evaluate-all behavior and poisoning: a native or eval failure after the first mutation
   poisons the entire cache set. Float goldens stay byte-identical.
5. **Verify accuracy and lifecycle.** Native suites `QuantizedKVAttentionTest`,
   `QuantizedKVCacheTest`, `QuantizedKVCacheMemoryTest`, `DecoderQuantizedGenerationTest`, plus the
   probe, each added to the per-module required-suite list in `.github/workflows/ci.yml`. Compare
   float and quantized prefill and decode on all seven fixtures (including Mistral window 4 via
   `slidingWindowFromModel().withQuantization(...)`) against G1/G2, recording token IDs and max/mean
   logit error. Lifecycle: append, trim, reset, fork, reorder (duplicates), source-scope close,
   repeated generation with bounded memory, cancellation, injected eval failure (poisoning), mixed
   quantized/float caches, the MHA/GQA rejection, chunked-prefill-with-sliding, and chunked versus
   single-chunk quantized prefill (max `|Δlogit|` ≤ the fixture's ratified tolerance, default
   `0.25 × deltaCap`; D5). D2 rejections are **validator unit tests**, including 4-bit group 128 and
   bfloat16 8-bit group 32 (not end-to-end, since all fixtures are float32).
6. **Long-context accuracy (G3) and the memory/throughput benchmark.**
   - *Checkpoint.* Use the pinned Tier-B Llama (`HuggingFaceTB/SmolLM2-135M-Instruct`, rev
     `12fd25f7…`) and, if its hosted-runner memory allows, Qwen2.5-0.5B-Instruct, from
     `req/phase6-tier-b-artifacts.md` (hashes in `tools/tier-b/*.json`). Confirm each pinned
     config's context limit is ≥ 4096 + 64, plus head dim and dtype, and choose the setting via D2
     before running.
   - *Prompt.* A committed text under `tools/tier-b/prompts/` (repo-authored or public-domain, with
     its provenance and licence noted), with its SHA-256 recorded in a new `prompt_sha256` field of
     the Tier-B manifest. The prompt is a committed file, not a download, so **the test alone checks
     the hash**; `tools/tier-b/download.py` is unchanged (it only fetches and hashes the manifest's
     remote `files`, and a separate local-file check there would add nothing). The test tokenizes
     the prompt with the checkpoint's own tokenizer and takes exactly N = 4096 tokens (failing if
     shorter), so the input is reproducible.
   - *Test.* A new `TierBQuantizedKvTest` beside `TierBSmokeTest` in the Tier-B source set
     (`jmlx-models/src/tierB/java/se/alipsa/jmlx/models/`, run by `:jmlx-models:tierBTest`): chunked
     prefill of the 4096-token prompt, 64 teacher-forced positions, the G3 comparison with the
     fixed exclusion rule, plus G4/G5 (including the D5 scaling check at 2048 and 4096, with `a`
     re-measured on the step 2 `KVCache` first, `scoreIntermediates`/`layersKept` counted from
     the final graph and `floatKept` built from `layersKept`).
   - *Workflow.* A `quantized-kv` job (or matrix entry with a `kv_quantization` input) in
     `.github/workflows/tier-b.yml`, running on the same schedule and `workflow_dispatch`, with the
     same artifact download and cap, an "assert the test executed" step like the smoke test's, and
     its observations and raw JSON uploaded as an artifact. It is not a required PR check.
   - *Benchmark.* Extend `DecodeBenchmark` with an optional trailing quantization argument (for
     example `none|q4g32|q8g32`) and additive fields (`kv_quantization`, `prefill_chunk`, per-step
     peak at N₁ and N₂, steady retained KV bytes). Additive fields keep schema
     `jmlx-decode-benchmark-1`; change the schema only if an existing field's meaning changes.
     **Prefill path.** Today's benchmark builds context through public `forward` (`DecodeBenchmark`
     ~L101, `"prefill_logits": "all_positions"`), which at N = 4096 is a `[1, 4096, vocab]` logits
     tensor. Whenever a prefill chunk size is set (always for a quantized run, per D5, and
     explicitly for the matched float run) the benchmark builds context with
     `DecoderModel.prefillChunked` and records `"prefill_logits": "last_position"`; with no chunk
     size it keeps `forward` and `all_positions`, so existing reports stay comparable. The one-token
     decode measurement stays on `forward`. Prefill timings and peaks are comparable only between
     runs with the same `prefill_logits` value. The argument parser currently rejects more than
     seven arguments (`args.length > 7`, ~L42), so raise the limit and extend the usage text for the
     quantization and chunk-size arguments (the chunk size as an optional trailing argument, for
     example `... full q4g32 256`). Compare float vs quantized at matched checkpoint, prompt, N,
     chunk size, batch 1, and hardware, as interleaved A/B over 5 runs. Hash checkpoint files
     outside the load timer. Report K and V storage separately plus temporary attention buffers. Add
     a CI smoke step (in `ci.yml`) that runs a quantized tiny fixture and validates the new fields
     without timing thresholds.
7. **Enable and document.** Only if every model-level gate passes: remove the two named-unsupported
   factories (D1), update `req/phase6-compatibility.md` (Quantization column),
   `req/phase6-4-benchmark.md` (accuracy envelope, long-context results, exact commands, checkpoint
   hashes, runtime pins), `req/mlx-api-inventory*.md` if a binding was added, and the README and
   CLAUDE.md summaries.

## Step 0 record (2026-10-01, host Apple M2 Max, 64 GB, macOS 26.7, base `54f031f`)

Ratified unchanged: D1–D5, P1–P5, G1–G7, the 0.5 slack, `deltaCap` 1% (8-bit) and 5% (4-bit),
N = 4096, C = 256, the step 5 tolerance `0.25 × deltaCap`, and the probe parameters (k = 4 float32,
k = 1 bfloat16/float16, 10%, seed 12345, 25% warning level). **Decision date: 2026-10-08. 1b
deadline: 2026-10-22.**

**Embedding dtype.** All seven fixtures are float32 weights, so the early D2 check (embedding dtype
equals K/V dtype) holds for them. The Tier-B checkpoints are checked in step 6.

**Float chunk noise** (last-position logits, 4-token sequential `forward` versus one `forward`
over 31 tokens; `QuantizedKvFeasibilityProbeTest`): 3e-6 % to 1.5e-5 % of the logit range for every
fixture, orders of magnitude below half of the smallest tolerance (0.25 %). No per-fixture raise.

**Float-only margin feasibility** (32 teacher-forced positions; positions excluded when
`floatMargin ≤ 2 × deltaCap`; the 10% cap allows 3):

| Fixture | Excluded at 1% | Excluded at 5% | Flip limit |
| --- | ---: | ---: | --- |
| llama | 11 | 27 | dropped |
| llama31 | 6 | 20 | dropped |
| qwen2 | 5 | 28 | dropped |
| mistral (window 4) | 4 | 26 | dropped |
| gemma | 0 | 0 | kept (0 excluded: any flip fails) |
| phi3 | 8 | 27 | dropped |
| mixtral | 10 | 31 | dropped |

The random-weight fixtures have top-1/top-2 margins around 0.01–0.1 against a logit range near
0.8. The exclusion threshold is a fraction of the range, so more positions or a different prompt
change the count of excluded positions in proportion, not the fraction, so more positions would
not bring six of the seven under 10% (reasoned from the distribution above, not re-measured with
other prompts). **Last resort applied (recorded before any quantized run):** llama, llama31,
qwen2, mistral, phi3 and mixtral are dropped from the flip limit for G1 and G2 and gated by the
Δ cap alone. gemma keeps both. G3 (real checkpoint) keeps its flip limit and the 10% cap.

**Informational float baseline** (tokens/s median of 5 samples, 32 tokens, prompt 1,7,42,3,19,5,
full policy; raw JSON in `req/data/phase6-4-1-baseline/`): llama 1076.9, llama31 1099.6,
qwen2 1080.0, mistral 1089.1, gemma 830.6, phi3 1097.4, mixtral 762.0. Not the G6 comparison.

## Exit gate (enabling the opt-in policy)

Enablement requires **all** of:

- All required native suites pass, including D=16 heads and nonuniform batch positions.
- G1–G3 accuracy, G4–G5 memory at N = 4096 (absolute bytes and percentage difference for steady
  retained bytes and per-decode-step peak, **plus the scaling check showing growth tracks packed,
  not float, bytes**), and G6–G7 throughput all meet the frozen thresholds, with memory bounded
  across sustained decode and repeated generation.
- The accuracy envelope, token comparisons, checkpoint and prompt provenance, runtime pins, and
  exact commands are committed.

Only then replace the named unsupported factories with working opt-in policies. If the gates are not
met, the rejection stays and the blocker is recorded. Phase 6.5 is unaffected either way once the
sequencing gates in "Sequence and blocking" are met (the step 1 decision, and on a proceed step 1b
or its deadline).
