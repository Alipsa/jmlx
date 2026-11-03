# Phase 7.6 WP1 — Structural optimization qualification

The 2026-11-03 review selected chunked prefill, deferred custom Metal and required structural
optimization before WP2. The closest comparison is mlx-vlm's ordinary chunked ops. The clean
source commit for the new records is `2c3f9da4b30c1147cae11d6072320683bb9511f1`.

## Result and scope

The 1,280-token / 48-call median is **474.313 ms**, versus **2,550.557 ms** before restructuring
and **304.105 ms** for mlx-vlm's own chunked ops: about **5.38× faster than the original Java
structure** and **1.56× the reference ops time**. This meets the stated approximately 600 ms /
2× target without custom Metal, a new facade or a relaxed numerical bound.

These are delta-rule-only prepared-input measurements, excluding learned projections,
convolution, output gating, vision and loading. The same 48 independent layer calls and exact
input hashes are used; this is not an end-to-end 27B prefill or throughput measurement.
Three samples per path, with first-per-shape calls kept separate and timestamped host snapshots.

| Tokens | Original Java ms | Optimized Java ms | mlx-vlm chunked ms | Optimized/reference |
| --- | ---: | ---: | ---: | ---: |
| 64 | 147.525 | 146.265 | 114.145 | 1.28× |
| 256 | 533.157 | 168.749 | 138.780 | 1.22× |
| 1024 | 2122.804 | 327.721 | 250.743 | 1.31× |
| 1280 | 2550.557 | 474.313 | 304.105 | 1.56× |

All original baseline records remain committed. Frontend runs are sequential, with candidate
orders alternated within each frontend. Ratios and three-sample ranges are descriptive, not a
causal attribution to FFM or GPU dispatch. GPU utilization/thermal state is not established.

## Structure and numerical qualification

- Invert the 64×64 unit-lower-triangular system with stable forward substitution, then apply
  it to the RHS once. The expensive solve is no longer repeated against 256-wide RHS data.
- Batch every state-independent quantity across at most 16 chunks (1,024 positions). Only
  the carried-state/output update scans chunks sequentially. Preprocessing scopes release
  temporary handles; inverse-row intermediates are retired promptly.
- Evaluate preprocessing roots and bounded eight-chunk scan windows. Final concatenation stays
  lazy and is included in the caller evaluation. Decode has a dedicated single-token path.
- Cache replacement can retain provisional lazy inference state; the session must evaluate
  output and every retained state root once before committing its step or poison on failure.
- Cache shape metadata locally in row/window helpers; no global MLXArray metadata cache was added.

Both Java algorithms still qualify directly against independent HF output/final-state files at
`1e-4` on all 12 benchmark cases; maximum observed error is about **1.85e-6**. Existing component
and split-prefill goldens are unchanged. Additional HF fixtures cover correlated and near-correlated
keys, beta near one, strong decay, batch size two and preprocessing boundaries through 2,049
positions. All pass. The original repeated-request, exact parameter-handle and 4,128-position
memory ceiling also passes unchanged. A larger preprocessing batch was rejected when it exceeded
that ceiling; the selected batch remains 16 chunks.

The separately recorded Torch FLOAT32 doubling-inverse experiment fails on the correlated
limiting system (inverse maximum error 4,294,967,296). It never supplies goldens and is not
selected. This is a numerical stability reason for using the stable small inverse, not a
qualification of any custom kernel. MLX's built-in delta op remains numerically excluded.

## Measured calls and decode boundary

Separate instrumented run; its timings are invalid. At 48 calls / 1,280 tokens:

| Chunked path | Original | Optimized |
| --- | ---: | ---: |
| Total native C downcalls | 3,159,936 | 430,032 |
| `mlx_matmul` | 66,240 | 9,408 |
| `mlx_eval` | 1,008 | 288 |
| `mlx_array_ndim` | 1,118,208 | 110,256 |
| `mlx_array_shape` | 545,856 | 60,528 |

These are API calls, not GPU dispatch counts. The complete DeltaNet decode fixture, including
cache replacement, records **zero internal `mlx_eval` calls** and **one caller-boundary call**
for output plus convolution/recurrent state. Decode output and both states still match HF at
`1e-4`. The counter distinguishes internal calls from the explicit session synchronization.

## Evidence and next package

- [Optimized aggregate](../../tools/qwen35-reference/evidence/optimized/delta-java-48-layers.json),
  [full matrix](../../tools/qwen35-reference/evidence/optimized/delta-java.json).
- [Aggregate calls](../../tools/qwen35-reference/evidence/optimized/delta-java-48-layers-counts.json),
  [decode calls](../../tools/qwen35-reference/evidence/optimized/delta-java-counts-decode.json).
- [Independent stress provenance](../../tools/qwen35-reference/goldens/delta-stress.json),
  [original baseline](phase7-6-wp1-benchmark.md).

The user's WP1 structural gate is met. Chunked ops is the prefill algorithm; single-token
recurrence is the decode path. Proceed to WP2. Full model, cancellation/ownership integration,
Tier-B and hosted CI acceptance remain pending. Custom Metal remains deferred in this milestone.
