# Phase 6.0b.1 probe findings

**Runtime:** macOS/Apple Silicon; Java 25; `mlx-metal` 0.31.2; `mlx-c`
fba4470b89073180056c9ea46c443051375f7399.

**Command:**

```text
./scripts/bootstrap-native.sh
./gradlew :jmlx-core:test --tests se.alipsa.jmlx.core.SelectionAndRandomProbeTest --info
```

## Observations

The probe uses logits `[[1, 3, 3, 2], [5, 5, 4, 0]]` and selects axis 1.

| Operation | Native dtype / shape | Observed values |
| --- | --- | --- |
| `mlx_argmax_axis` | UINT32 / [2] | [1, 0] |
| `mlx_topk_axis(k=2)` | FLOAT32 / [2, 2] | [3, 3, 5, 5] |
| `mlx_sort_axis` | FLOAT32 / [2, 4] | [1, 2, 3, 3, 0, 4, 5, 5] |
| `mlx_argsort_axis` | UINT32 / [2, 4] | [0, 3, 1, 2, 3, 2, 0, 1] |
| `mlx_partition_axis(kth=1)` | FLOAT32 / [2, 4] | [1, 2, 3, 3, 0, 4, 5, 5] |
| `mlx_argpartition_axis(kth=1)` | UINT32 / [2, 4] | [0, 3, 1, 2, 3, 2, 0, 1] |
| `mlx_random_key(42)` | UINT32 / [2] | [0, 42] |
| `mlx_random_split_num(key, 3)` | UINT32 / [3, 2] | [-1160419002, -561808247, -548466209, 894150801, 801545058, -1931765865] |
| `mlx_random_categorical` | UINT32 / [2] | [1, 0] |

The signed values in the split row are the Java INT32 readback of the UINT32 bit patterns.

## Decisions

- Greedy decoding retains the existing axis-aware `mlx_argmax_axis` path. Equal maxima select the
  first index along the axis for this pinned tuple. `MLXOps.argmaxAxis` intentionally converts the
  native UINT32 result to INT32; the direct probe establishes the native dtype.
- Phase 6.1 must use axis-aware selection on decoder logits, not the flattening operations. The flat
  variants and two-way split remain pending a follow-up probe.
- `mlx_random_key` plus `mlx_random_split_num` produces a stable explicit-key sequence and is viable
  for per-request state. Phase 6.1 must keep one key per request and must not use global
  `MLXRandom.seed` in generation.
- These results establish evaluated values only. A dedicated lazy-evaluation/ownership probe remains
  required before accepting any facade operation.

The existing README claim of deterministic greedy generation is consistent with the observed argmax
tie-breaking for this pinned tuple; no Phase 6.0 contract correction is needed.
