# Phase 6.4.1 implementation plan — quantized KV retention and packed attention

**Sequence:** finish the Phase 6.4 PR first. Complete this milestone before Phase 6.5 batching.
Phase 6.4's `GenerationCachePolicy.quantized(bits, groupSize)` and
`KVCachePolicy.quantized(bits, groupSize)` deliberately throw until this milestone passes its gate.

**Goal:** make compressed K/V storage useful during decode: attend without reconstructing the whole
float cache, reduce both steady and peak native memory at a documented long context, and preserve
the default float generation path.

**Evidence:** [Phase 6.4 benchmark](../phase6-4-benchmark.md) records the pinned macOS CI probes.
Native affine quantization accepts groups 32/64/128, while every synthetic family fixture has
16-dimensional heads. Padding D=16 to 32 packs successfully, but the pinned SDPA rejects packed
K/V. Materializing all float K/V before SDPA defeats the peak-memory goal. The existing probe did
not test quantized cache lifecycle because no useful attention path existed yet.

## Contracts and design gate

- Quantization is **opt in**; existing `FULL` and `SLIDING_WINDOW` policies, default generation,
  greedy goldens, and float-cache ownership remain unchanged. A quantized policy carries bit width,
  group size, and the resolved retention policy. Reject unsupported combinations before prefill;
  never fall back silently to float retention.
- Preserve absolute and per-row positions, left-padded batch layout, mask semantics, capacity
  preflight, sliding-window eviction, and the cache-set poisoning rule. RoPE is applied before K is
  packed. Head padding is internal; it must never become an attendable feature or key position.
- Store packed K/V, scales, biases, shape/padding metadata, and any small uncompressed tail in
  request-owned scopes. A decode step may unpack a **bounded block** into temporary storage, but
  must not materialize an entire float cache. Standalone fork/reorder must return independently
  evaluated storage; reset and eviction release every owned representation. The public `keys()` and
  `values()` contract needs an explicit decision before implementation: either provide a clearly
  documented bounded diagnostic materialization method, or make compressed storage accessible
  through a new typed cache API without implying it is a float tensor.
- Select an attention design only after a native prototype on the pinned runtime demonstrates its
  accuracy and peak-memory behavior. A fused packed-KV native/Metal kernel and bounded-block
  streaming attention are candidates. If adding a native binding, update the API inventory,
  generated header coverage, build/bootstrap pins, package tests, and ownership Javadocs. Avoid
  one host synchronization per block or layer; the step should retain the single sampler eval
  boundary unless measured evidence justifies a change.

## Implementation work

1. **Prototype and choose the attention path.** On macOS CI, compare fused or streaming candidates
   against float SDPA for single-token and multi-token queries, causal/full and sliding masks,
   grouped-query attention, unequal batch positions, D=16 padded heads, and D=32/64 heads. Use a
   numerically stable softmax reduction. Capture raw logits/attention error, active and peak native
   bytes, synchronization count, and tokens/s. Publish the choice and rejected alternatives in the
   benchmark report before changing the public cache policy.
2. **Implement compressed ownership.** Add the resolved core policy and request-level policy, then
   store packed K/V and metadata with the existing cache's scope rules. Define append/chunked
   prefill, zero-length sliding retention, trim, capacity, reset, fork, reorder with duplicate
   indices, and poison/reuse behavior. Evaluate standalone copies before returning. Validate dtype,
   bit width, group size, head padding, and native capability before generation mutates a cache.
   Keep any float tail bounded independently of total context length.
3. **Integrate attention and sampling.** Feed packed cache blocks to the chosen attention path.
   Preserve per-row masks and absolute positions, avoid full-cache dequantization, and evaluate
   newly packed storage at the existing step boundary with selected logits. Public forward keeps
   its documented synchronization and poison behavior. A native/eval failure after first mutation
   poisons the entire cache set.
4. **Verify accuracy and lifecycle.** Add native suites named `QuantizedKVAttentionTest`,
   `QuantizedKVCacheTest`, `QuantizedKVCacheMemoryTest`, and
   `DecoderQuantizedGenerationTest`; list each in CI's per-module required-suite assertion.
   Compare float and quantized prefill/decode for all six synthetic families, including the
   Mistral window-4 fixture, and report token IDs, max/mean logits error, and any divergence.
   State a measured accuracy envelope rather than promising bit-identical logits. Exercise
   append, trim, reset, fork, reorder, source-scope close, repeated generation, cancellation, and
   injected eval failure. Keep existing float goldens byte-identical.
5. **Benchmark the actual memory tradeoff.** Add a provenance-tracked fixture or accessible real
   checkpoint with a context well beyond the current 128-position tiny-fixture limit. In fresh
   processes, compare float and quantized steady active bytes, per-token peak active bytes, cached
   allocator bytes, and median tokens/s at matched prompt, context, batch size, and hardware. Hash
   checkpoint files outside the load timer. Report both K and V storage plus temporary attention
   buffers; do not infer a peak win from packed tensor size alone. Run the deterministic native CI
   smoke test without timing thresholds, and publish the reproducible long-context measurements.

## Exit gate before 6.5

- The quantized path passes all required native suites, including D=16 heads and nonuniform batch
  positions, and no decode step materializes the entire float K/V cache.
- At the documented long context, both steady retained **and peak active** native bytes are below
  the float baseline; the report gives the absolute values and percentage difference for each.
  Memory remains bounded across sustained decode and repeated generation. Decode throughput is
  measured and any regression is stated; the float default has no performance or output regression.
- The published accuracy envelope, token comparisons, checkpoint provenance, runtime pins, and
  exact benchmark commands are committed. Only then replace the named unsupported factories with
  working opt-in policies and update the compatibility matrix. If the prototype cannot meet these
  gates, keep the explicit rejection, record the blocker, and leave 6.5 blocked pending a revised
  6.4.1 design.
