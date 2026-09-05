# Phase 6.0b.1 probe findings

**Runtime:** macOS/Apple Silicon; Java 25; `mlx-metal` 0.31.2; `mlx-c`
fba4470b89073180056c9ea46c443051375f7399.

**Command:**

```text
./scripts/bootstrap-native.sh
./gradlew :jmlx-core:test --tests se.alipsa.jmlx.core.SelectionAndRandomProbeTest --info
```

## Observations

The primary selection probe uses logits `[[1, 3, 2, 0], [5, 4, 3, 0]]` (distinct values per row)
and selects axis 1. Dedicated tie probes cover both index-0 and nonzero first-tied indices.

| Operation | Native dtype / shape | One observed instance |
| --- | --- | --- |
| `mlx_argmax_axis` | UINT32 / [2] | [1, 0] |
| `mlx_topk_axis(k=2)` | FLOAT32 / [2, 2] | [2, 3, 4, 5] |
| `mlx_sort_axis` | FLOAT32 / [2, 4] | [0, 1, 2, 3, 0, 3, 4, 5] |
| `mlx_argsort_axis` | UINT32 / [2, 4] | [3, 0, 2, 1, 3, 2, 1, 0] |
| `mlx_partition_axis(kth=1)` | FLOAT32 / [2, 4] | [0, 1, 2, 3, 0, 3, 4, 5] |
| `mlx_argpartition_axis(kth=1)` | UINT32 / [2, 4] | [3, 0, 2, 1, 3, 2, 1, 0] |
| `mlx_random_key(42)` | UINT32 / [2] | [0, 42] |
| `mlx_random_split_num(key, 3)` | UINT32 / [3, 2] | [-1160419002, -561808247, -548466209, 894150801, 801545058, -1931765865] |
| `mlx_random_categorical` | UINT32 / [2] | [1, 0] |

The signed values in the split row are the Java INT32 readback of the UINT32 bit patterns. The
partition and argpartition rows are one observed ordering only; the committed probe asserts their
partition invariants rather than exact ordering. The table records observations made while running
the command above, not an assertion that every displayed ordering is regression-guarded.

`mlx_topk_axis` returned each row's top-2 values in ascending order for this pinned tuple, but MLX's
topk is partition-based and does not document any order guarantee among the k largest. The committed
test (`SelectionAndRandomProbeTest.assertTopK`) therefore only asserts that the result is exactly the
correct multiset of the k largest values per row (order-insensitive), not a specific order — asserting
the observed ascending order would encode an accident of this pinned build, not a guarantee.

A distinct second probe (`SelectionAndRandomProbeTest.recordsArgmaxTieBreaking`), using logits
`[[3, 1, 3, 0], [5, 5, 4, 0]]` where each row's maximum is repeated (row 0: value 3 at indices 0 and 2;
row 1: value 5 at indices 0 and 1), specifically exercises argmax tie-breaking. Observed
`mlx_argmax_axis` result: `[0, 0]` — the first tied index in each row.

`SelectionAndRandomProbeTest.recordsFirstTiedIndexAwayFromZero` separately uses
`[1, 3, 3, 0]` and observes `[1]`, confirming that first-index tie-breaking is not an index-0 bias.

## Follow-up probe coverage

`SelectionAndRandomProbeTest` now also contains the remaining acceptance probes. They use the same
`[[1, 3, 2, 0], [5, 4, 3, 0]]` input and distinguish the flat entry points from their axis-aware
counterparts by asserting one-dimensional result shapes. Flat sort has the exact expected result
`[0, 0, 1, 2, 3, 3, 4, 5]`; top-k is checked as an order-independent multiset; argsort is checked as
a permutation whose indexed values are nondecreasing; and both partition variants are checked by
their pivot invariants. Valid boundary cases cover `topk(k=8)`, `partition(kth=0)`, and
`argpartition(kth=7)`.

The two-output `mlx_random_split` result is compared both with `mlx_random_split_num(key, 2)` and
with the committed UINT32 golden. Its Java signed-INT32 representation is
`[-1829035798, -615737125, 255383827, 267815257]`. The probe also checks that the siblings differ
and the parent remains `[0, 42]`. This expectation follows from the pinned MLX implementation, where
the two-output overload splits the `split(key, 2)` array in order; the native test remains the
runtime regression evidence.

Categorical coverage now repeats the same call with the same key, checks the pinned `[1, 0]` result,
and rejects an axis outside the logits rank. A separate ownership test creates flat-sort,
two-output-split, and categorical results in a parent scope, closes the child scope containing every
input before first evaluation, and only then evaluates and reads the results. Successful execution
therefore establishes both lazy graph ownership and cleanup through the ordinary `MLXScope` paths.

PR #22's macOS/Apple-Silicon native job executed all four probes successfully against the recorded
MLX v0.31.2 pins. The observations below are therefore accepted native evidence rather than the
Linux-provisional results recorded while this follow-up was authored.

## Decisions

- Greedy decoding retains the existing axis-aware `mlx_argmax_axis` path. Equal maxima select the
  first index along the axis, evidenced by the dedicated tie probes above. `MLXOps.argmaxAxis`
  intentionally converts the native UINT32 result to INT32; the direct probe establishes the native
  dtype.
- Phase 6.1 must use axis-aware selection on decoder logits, not the flattening operations. Flat
  variants and two-way split have committed assertions that passed PR #22's native gate.
- `mlx_random_key` plus `mlx_random_split_num` produces a stable explicit-key sequence and is viable
  for per-request state. Phase 6.1 must keep one key per request and must not use global
  `MLXRandom.seed` in generation.
- Lazy-evaluation/ownership assertions deliberately evaluate only after their input scope closes;
  they passed PR #22's native gate before the facade operations were accepted.

The existing README claim of deterministic greedy generation is consistent with the observed argmax
tie-breaking; no Phase 6.0 contract correction is needed.
