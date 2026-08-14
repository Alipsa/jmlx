# jmlx Phase 4 — M4 implementation plan (`QuantizedLinear`)

**Spec:** `req/phase4-plan.md` §8 (the three quantization entry points and their parameter-shape
table), §3 (the two remaining generic op-body helpers this plan builds: `optInt`, `optDtype` --
`vectorOutOp` already exists, built in M3, and needs no change), §2 (`scopeOf`/`innermost`, and the
rule that every multi-operand op -- including nullable operands -- resolves its target through it),
§5 (`Module`/layer conventions: `param`/`child`, the checkpoint-layout contract `Linear` already
established), Decision 8 (test goldens; composition-identity fallback when a direct golden would
encode mlx's internal packing layout), §9 (Documentation -- folded into this plan; see the note
below), Testing approach (`dequantize(quantize(w)) ≈ w` as "the only tractable `QuantizedLinear`
assertion").

Branch: `worktree-phase4-m4`, off `main` at `59c9afe` (PR #10 merged, M3 done). Native bootstrap
staged in this worktree (`native/install/lib/mlx.metallib` exists) -- `@EnabledIfNativeAvailable`
tests run for real, against real Apple Silicon GPU hardware.

**§9 (Documentation) is folded into this plan, not deferred again.** `req/phase4-plan.md`'s own
Status table note on §9 reads "stays `NOT STARTED` until M4 also lands" -- M1/M2/M3 each deferred it
for that reason; M4 is the merge point the note is waiting for, so this plan's Task 3 does it rather
than opening a fifth "not yet" entry. §10 (CI, self-hosted runner) is a separate, already-tracked
follow-up with no dependency on M4 and stays out of scope here, same as every prior plan.

## Findings from this plan's pre-work (Verification item 0d, resolved)

Per `req/phase4-plan.md`'s Verification item 0d, `mlx_quantize`'s exact output shapes/dtypes and
`mlx_quantized_matmul`'s `transpose` semantics are not fully pinned down by the header alone. Probed
with a scratch test class (`ScratchQuantizeProbe`, in `se.alipsa.jmlx.core` for package-private access
to `NativeOps`/`nullableHandle`/`cstr`), compiled and run against this worktree's real bindings, then
deleted -- same precedent as M2/M3's probes. All findings below are measured output, not derived.

* **`mlx_quantize`'s three outputs, `[1,64]` FLOAT32 input, `group_size=32`, `bits=4`,
  `mode="affine"`, `global_scale` absent.** Status `0`, exactly 3 outputs (`vectorOutOp`'s existing
  shape, unchanged from `split`'s):
  * `w_q`: shape `[1, 8]`, dtype `3` == `mlx_h.MLX_UINT32()` -- confirms the packing formula
    `packedCols = cols * bits / 32` (`64 * 4 / 32 = 8`), and confirms `DType.UINT32`'s existing native
    value needs no change to read this output correctly.
  * `scales`, `biases`: each shape `[1, 2]`, dtype `10` == `mlx_h.MLX_FLOAT32()` -- confirms
    `numGroups = cols / groupSize` (`64 / 32 = 2`), one scale and one bias term per group.
* **`mlx_dequantize` with `dtype` absent (`has_value=false`) returns FLOAT32**, matching `scales`'
  own dtype -- output shape `[1, 64]`, round-trips the original data with **max absolute error
  ≈0.1067** against the probe's `[-3.2 .. 3.1]`-range input (16 representable levels per 32-element
  group at `bits=4`, consistent with a ≈0.4/16 ≈ 0.107 step size for that group's local range). This
  is the empirical basis for `MLXQuantTest`'s round-trip tolerance and for the Testing table's
  `dequantize(quantize(w)) ≈ w` row.
* **`mlx_quantized_matmul(x, w_q, scales, biases, transpose=true, ...)` treats `w_q` as the packed
  form of a checkpoint-layout `[out, in]` weight and computes `x @ w^T`** -- exactly `Linear`'s own
  weight convention, with the transpose fused into the native call rather than a separate op.
  Confirmed: `x` = the same `[1,64]` row used to build `w_q` (so the expected unquantized result is
  `dot(x,x) = 218.56`); actual quantized result `219.67467`, a ≈0.5% relative difference consistent
  with 4-bit quantization noise on `w_q` alone (`x` itself is unquantized FLOAT32 in this call).
  Output shape `[1, 1]`, matching `x`'s batch dim and `w_q`'s logical `out=1`.
* **`group_size`/`bits` both absent (`has_value=false`) does not error -- native applies defaults.**
  Same `[1,64]` input: `w_q` shape is still `[1, 8]` (bits default `4`, unchanged packing), but
  `scales` shape is `[1, 1]` (**one** group covering the whole row) where the explicit-`32` run
  produced `[1, 2]` -- so the native default `group_size` is `64` (the full row), not `32`. This path
  has no `QuantizedLinear` caller (the layer always supplies explicit values, per Global Constraint 6
  below) but is exercised directly by `MLXQuantTest` for `optInt`'s absent-value branch.
* **`mlx_gather_qmm`, `mlx_qqmm` and `mlx_gather_mm` also exist in this `ops.h`** (a gathered/MoE
  quantized-matmul family) but are not named anywhere in spec §8, which lists exactly three entry
  points matching `project-outline.md`'s single `QuantizedLinear` deliverable. Out of scope --
  no consumer in this codebase needs gathered/indexed quantized matmul.
* **Four `QuantizationMode` values exist upstream** (`wheel/mlx/include/mlx/primitives.h:155`:
  `Affine, Mxfp4, Mxfp8, Nvfp4`), all reachable through the same `const char* mode` parameter this
  plan's ops expose. Only `"affine"` is exercised by any test in this plan or used by
  `QuantizedLinear` -- consistent with `req/phase4-plan.md`'s own Research findings ("`mlx_quantize`
  with `mode="affine"`" is the only mode that document's Decisions ever reference). The other three
  are unvalidated by this codebase; see Deliberately not covered.

## Global Constraints

1. **Package boundaries are unchanged from M1-M3.** `optInt`/`optDtype` (`NativeOps`) and `quantize`/
   `dequantize`/`quantizedMatmul` (`MLXQuant`) live in `se.alipsa.jmlx.core`; `QuantizedLinear` lives
   in `se.alipsa.jmlx.nn` and depends only on the public `core` surface. `grep -rn
   'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` must stay empty (spec
   Verification #6a).
2. **`NativeOps` keeps its zero-sibling-dependency invariant.** `optInt`/`optDtype` are built directly
   from the generated `mlx_optional_int_`/`mlx_optional_dtype_` struct accessors, exactly mirroring
   `optFloat`'s existing shape -- no call into `MLX`, `MLXQuant`, or any other sibling.
3. **Only the two remaining generic helpers are added: `optInt`, `optDtype`.** `vectorOutOp` (§3's
   sixth helper) already exists, built in M3 for `split`, and needs zero changes: `mlx_quantize`'s
   `(mlx_vector_array* res, ...)` shape is identical, so `MLXQuant.quantize` calls the existing helper
   with a new lambda body, the same way `MLXShape.split` already does.
4. **`mode` is a generic `String` parameter on all three `MLXQuant` methods**, matching spec §8's
   named signatures (`quantize(w, gs, bits, mode)` etc.) and mirroring `MLXFast.scaledDotProductAttention`'s
   `maskMode` -- the facade exposes native's real parameter rather than hiding it behind a single
   hardcoded literal. `QuantizedLinear` itself always calls with the literal `"affine"`: the only mode
   with any documented rationale in this repo (Research findings: "`mlx_fast_affine_quantize` does
   not exist ... filled by `mlx_quantize` with `mode="affine"`"). `Mxfp4`/`Mxfp8`/`Nvfp4` remain
   reachable through `MLXQuant` for a future caller but are not exercised by any test here.
5. **`groupSize`/`bits` are boxed `Integer`, nullable for "absent"** on all three `MLXQuant` methods --
   mirroring `MLXFast.rope`'s existing `Float base` parameter for the same by-value-optional-struct
   shape. `QuantizedLinear`'s own constructor takes plain non-null `int groupSize, int bits`: a
   persisted layer must remember the exact values it was built with (native's own defaults, found
   above, are a moving target this layer must not silently depend on), so the "absent" path is real
   API surface on `MLXQuant` (tested directly by `MLXQuantTest`) but has no `QuantizedLinear` caller.
6. **`quantizedMatmul` needs no explicit-target overload the way `Linear.forward`'s `transpose(W)`
   did (§2's fifth sub-hazard).** That hazard is specifically about a *single-operand* op on a
   parameter -- `scopeOf(op, a)` degenerates to `a.scope()`, which is the model scope when `a` is a
   weight. `quantizedMatmul(x, w, scales, biases, ...)` always has `x` (an activation, never a
   parameter) as a real second operand, so `scopeOf` resolves to the innermost of `x`'s scope and the
   parameter scopes -- correctly the step scope -- with no special-casing. Stating this now so a later
   reader does not "fix" a leak that was never there by adding an overload with no consumer.
7. **Every multi-operand `MLXQuant` op resolves its target via `NativeOps.scopeOf(...)` over every
   array operand, including nullable ones** -- `quantize`'s `(w, globalScale)`, `dequantize`'s `(w,
   scales, biases, globalScale)`, `quantizedMatmul`'s `(x, w, scales, biases)`. `quantize`'s case is
   the one worth naming explicitly: `MLXShape.split` (its `vectorOutOp` precedent) passes `a.scope()`
   directly as the target because `split` has no nullable operand, but `quantize` does (`globalScale`)
   -- so its target must be `scopeOf("quantize", w, globalScale)`, not `w.scope()`, or a `globalScale`
   from a strictly inner scope than `w` would be allocated into the wrong (outer) scope without ever
   running its own `checkAccess()`. `MLXQuantTest` includes a dedicated cross-scope row for this (Task
   1, Step 1f).
8. **Style:** run `./gradlew spotlessApply` before finishing, then `./gradlew build` (Spotless,
   Checkstyle, full test suite including native tests and the forked `loaderGuardTest`) must succeed.
   Report actual test counts from the build output.
9. **Testing style, matching every existing test file:** `@EnabledIfNativeAvailable` on the test
   class; `try (MLXScope scope = new MLXScope())`; `assertArrayEquals(expected, actual.toFloatArray(),
   EPS)`. Quantization error is not float rounding error, so the round-trip/composition-identity rows
   use a tolerance derived from the Findings section above, not the facade's usual `1e-5f`/`1e-3f`.
10. **`MLXQuant` gets a new, dedicated test file (`MLXQuantTest`), not new methods on
    `MLXNumericTest`/`MLXFastTest`.** Matches how `MLXFastTest` was created new when `MLXFast` was
    born at M1, not folded into `MLXNumericTest` -- a new op-family class gets a new test file; only
    *additions* to an existing class's op family go into that class's existing test file (M3's Global
    Constraint 7).

## Native surface this plan uses (confirmed against the generated bindings and the probe above)

| Java call | C signature | Notes |
|---|---|---|
| `mlx_h.mlx_quantize(res, w, group_size, bits, mode, global_scale, s)` | `int mlx_quantize(mlx_vector_array* res, const mlx_array w, mlx_optional_int group_size, mlx_optional_int bits, const char* mode, const mlx_array global_scale, const mlx_stream s)` (`ops.h`) | `res` is the existing `vectorOutOp` shape (built in M3 for `mlx_split`, unchanged here). `group_size`/`bits` are by-value `optInt` structs (new helper). `global_scale` is a by-value nullable struct via the existing `nullableHandle`. Output shapes/dtypes for a `[1,64]`/`32`/`4` call: see Findings. |
| `mlx_h.mlx_dequantize(res, w, scales, biases, group_size, bits, mode, global_scale, dtype, s)` | `int mlx_dequantize(mlx_array* res, const mlx_array w, const mlx_array scales, const mlx_array biases /* may be null */, mlx_optional_int group_size, mlx_optional_int bits, const char* mode, const mlx_array global_scale /* may be null */, mlx_optional_dtype dtype, const mlx_stream s)` (`ops.h`) | The densest single call in this plan: two by-value optional structs (`group_size`/`bits` via `optInt`, `dtype` via the new `optDtype`) plus two nullable-array structs (`biases`, `global_scale`) in one call. `dtype` absent -> FLOAT32 output (Findings). |
| `mlx_h.mlx_quantized_matmul(res, x, w, scales, biases, transpose, group_size, bits, mode, s)` | `int mlx_quantized_matmul(mlx_array* res, const mlx_array x, const mlx_array w, const mlx_array scales, const mlx_array biases /* may be null */, bool transpose, mlx_optional_int group_size, mlx_optional_int bits, const char* mode, const mlx_stream s)` (`ops.h`) | `transpose=true` is the `Linear`-compatible convention: `w`/`scales`/`biases` describe a `[out, in]`-layout original weight, result is `x @ w^T` (Findings). No array output beyond the single `res` -- a plain `scopeOf`-resolved hand-rolled body, same shape as `MLXFast.rmsNorm`. |
| `mlx_optional_dtype_` | `struct { mlx_dtype value; bool has_value; }`, 8 bytes -- confirmed by reading the generated binding directly; byte-identical layout to `mlx_optional_int_`/`mlx_optional_float_`. No native constructor function -- allocate and set both fields, `value` via the package-private `DType.nativeValue()`. |

## Task 1: `NativeOps.optInt`/`optDtype` + `MLXQuant` ops (`se.alipsa.jmlx.core`)

**Files:**
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/NativeOps.java`
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXQuant.java`
- New: `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXQuantTest.java`

### Step 1a: `NativeOps` — `optInt`

Add next to `optFloat` (`NativeOps.java:250-255`), same shape:

```java
/**
 * The native encoding of an {@code mlx_optional_int} by-value struct (req/phase4-plan.md §8's
 * Native surface table: 8 bytes, {@code {int value; bool has_value;}}, no native constructor
 * function -- same shape as {@link #optFloat}, for {@code mlx_quantize}/{@code mlx_dequantize}/
 * {@code mlx_quantized_matmul}'s {@code group_size}/{@code bits} parameters). {@code value == null}
 * encodes "absent" ({@code has_value=false}); native applies its own default in that case (confirmed
 * empirically, req/plans/phase4-m4-plan.md's Findings section) -- this facade does not duplicate it.
 */
static MemorySegment optInt(Arena tmp, Integer value) {
  MemorySegment seg = mlx_optional_int_.allocate(tmp);
  mlx_optional_int_.value(seg, value != null ? value : 0);
  mlx_optional_int_.has_value(seg, value != null);
  return seg;
}
```

### Step 1b: `NativeOps` — `optDtype`

```java
/**
 * The native encoding of an {@code mlx_optional_dtype} by-value struct -- same 8-byte {@code
 * {int value; bool has_value;}} shape as {@link #optInt}/{@link #optFloat} (confirmed by reading the
 * generated {@code mlx_optional_dtype_} binding directly: byte-identical layout, only the field's C
 * type name differs). {@code value == null} encodes "absent"; {@code mlx_dequantize} then produces
 * its own default dtype -- FLOAT32 in every case this facade can construct (confirmed empirically,
 * req/plans/phase4-m4-plan.md's Findings section: absent {@code dtype} matches {@code scales}' own
 * dtype, which this facade only ever produces as FLOAT32 via {@link MLXQuant#quantize}).
 */
static MemorySegment optDtype(Arena tmp, DType value) {
  MemorySegment seg = mlx_optional_dtype_.allocate(tmp);
  mlx_optional_dtype_.value(seg, value != null ? value.nativeValue() : 0);
  mlx_optional_dtype_.has_value(seg, value != null);
  return seg;
}
```

### Step 1c: `MLXQuant` — `quantize`

Replace the placeholder body (currently just a private constructor) with:

```java
public final class MLXQuant {

  private MLXQuant() {}

  /**
   * Affine quantization ({@code mlx_quantize}): packs {@code w} into a {@code UINT32}-encoded weight
   * plus per-group {@code scales}/{@code biases} for reconstruction (see {@link #dequantize}). The
   * three-array result order is native's own: {@code [w_q, scales, biases]}. {@code groupSize}/{@code
   * bits} are Java {@code null} for "let native pick its own default" (see this class's Findings
   * reference in {@link NativeOps#optInt} -- {@code QuantizedLinear} never uses this path, since a
   * persisted layer must remember its own exact values). {@code globalScale} is a Java {@code null}
   * for "none" -- a legitimate call, not an error, exactly like {@link MLXFast#rmsNorm}'s {@code
   * weight}.
   *
   * <p>The result's scope is resolved via {@link NativeOps#scopeOf} over <em>both</em> array
   * operands, {@code w} and {@code globalScale} -- not just {@code w.scope()} the way {@link
   * MLXShape#split}'s {@code vectorOutOp} call does, because {@code split} has no nullable operand
   * and this op does (req/plans/phase4-m4-plan.md Global Constraint 7).
   */
  public static MLXArray[] quantize(
      MLXArray w, Integer groupSize, Integer bits, String mode, MLXArray globalScale) {
    MLXScope target = NativeOps.scopeOf("quantize", w, globalScale);
    MemorySegment modeStr = NativeOps.cstr(mode);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment gs = NativeOps.optInt(tmp, groupSize);
      MemorySegment b = NativeOps.optInt(tmp, bits);
      MemorySegment gsHandle = NativeOps.nullableHandle(globalScale, tmp);
      return NativeOps.vectorOutOp(
          "quantize",
          target,
          (vec, stream) -> mlx_h.mlx_quantize(vec, w.handle(), gs, b, modeStr, gsHandle, stream));
    }
  }

  // dequantize, quantizedMatmul follow in 1d/1e
}
```

**`NativeOps.cstr` is called with a caller-supplied `mode` here, not a `private static final` field
the way `MLXFast`'s `MASK_MODE_*` constants are.** `cstr`'s own javadoc names both patterns as legal
(it is backed by an intern cache keyed on the string, "even an accidental per-call use never grows
`FACADE_ARENA` beyond one segment per distinct literal ever seen") -- `MLXQuant`'s three methods take
`mode` as a parameter (Global Constraint 4) rather than a closed compile-time enum the way SDPA's
`causal`/`maskArr` booleans pick from `MASK_MODE_CAUSAL`/`_ARRAY`/`_NONE`, so there is no fixed set of
`private static final` fields to declare here. `QuantizedLinear` (Task 2) passes the same Java string
literal `"affine"` on every call, so `cstr`'s cache still allocates that one segment exactly once.

### Step 1d: `MLXQuant` — `dequantize`

```java
/**
 * Reconstructs a float array from {@code w}/{@code scales}/{@code biases} (the output of {@link
 * #quantize}, or an equivalent triple loaded from a checkpoint). {@code biases}/{@code globalScale}
 * are each a Java {@code null} for "none" (mlx-c's own nullable-array convention, same as {@link
 * #quantize}'s {@code globalScale}). {@code dtype} is a Java {@code null} for "let native pick its
 * own default" -- confirmed empirically to be FLOAT32 for every input this facade can construct
 * (req/plans/phase4-m4-plan.md Findings).
 */
public static MLXArray dequantize(
    MLXArray w,
    MLXArray scales,
    MLXArray biases,
    Integer groupSize,
    Integer bits,
    String mode,
    MLXArray globalScale,
    DType dtype) {
  MLXScope scope = NativeOps.scopeOf("dequantize", w, scales, biases, globalScale);
  MemorySegment modeStr = NativeOps.cstr(mode);
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment gs = NativeOps.optInt(tmp, groupSize);
    MemorySegment b = NativeOps.optInt(tmp, bits);
    MemorySegment dt = NativeOps.optDtype(tmp, dtype);
    MemorySegment biasesHandle = NativeOps.nullableHandle(biases, tmp);
    MemorySegment globalScaleHandle = NativeOps.nullableHandle(globalScale, tmp);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "dequantize",
        () ->
            mlx_h.mlx_dequantize(
                res,
                w.handle(),
                scales.handle(),
                biasesHandle,
                gs,
                b,
                modeStr,
                globalScaleHandle,
                dt,
                NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }
}
```

### Step 1e: `MLXQuant` — `quantizedMatmul`

```java
/**
 * Fused quantized matrix multiply ({@code mlx_quantized_matmul}): {@code x @ w^T} when {@code
 * transpose=true}, without ever materializing a dequantized {@code w} -- the operation {@link
 * se.alipsa.jmlx.nn.QuantizedLinear} is built on. {@code transpose=true} is the convention that
 * matches {@code Linear}'s own checkpoint layout: {@code w}/{@code scales}/{@code biases} describe
 * the packed form of an {@code [out, in]} weight (confirmed empirically,
 * req/plans/phase4-m4-plan.md Findings). {@code biases} is a Java {@code null} for "none", same
 * nullable-array convention as {@link #dequantize}.
 *
 * <p>Needs no explicit-target overload the way {@code Linear.forward}'s {@code transpose(W, x.scope())}
 * does (req/phase4-plan.md §2's fifth sub-hazard): {@code x} is always a genuine second array
 * operand here, never absent, so {@link NativeOps#scopeOf} alone already resolves to the step scope
 * whenever {@code x} is an activation and {@code w}/{@code scales}/{@code biases} are model-scope
 * parameters (Global Constraint 6).
 */
public static MLXArray quantizedMatmul(
    MLXArray x,
    MLXArray w,
    MLXArray scales,
    MLXArray biases,
    boolean transpose,
    Integer groupSize,
    Integer bits,
    String mode) {
  MLXScope scope = NativeOps.scopeOf("quantizedMatmul", x, w, scales, biases);
  MemorySegment modeStr = NativeOps.cstr(mode);
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment gs = NativeOps.optInt(tmp, groupSize);
    MemorySegment b = NativeOps.optInt(tmp, bits);
    MemorySegment biasesHandle = NativeOps.nullableHandle(biases, tmp);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "quantizedMatmul",
        () ->
            mlx_h.mlx_quantized_matmul(
                res,
                x.handle(),
                w.handle(),
                scales.handle(),
                biasesHandle,
                transpose,
                gs,
                b,
                modeStr,
                NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }
}
```

### Step 1f: `MLXQuantTest.java` (new file)

`@EnabledIfNativeAvailable`; `EPS_ROUNDTRIP = 0.15f` (loosely above the Findings section's measured
`≈0.1067` max absolute error, for one shared constant across the round-trip rows rather than a
recomputed per-test bound); `EPS_EXACT = 1e-5f` for shape/dtype/status-level assertions that involve
no quantization at all.

| Test | Proves |
|---|---|
| `quantizeProducesUint32PackedWeightAndFloat32ScalesAndBiases` | The exact shapes/dtypes from Findings, asserted as a regression pin: `[1,64]`/`32`/`4` -> `w_q` `[1,8]` UINT32, `scales`/`biases` each `[1,2]` FLOAT32 |
| `dequantizeRoundTripsWithinQuantizationError` | `dequantize(quantize(w)) ≈ w` within `EPS_ROUNDTRIP` -- the Testing-table row named in the top-level spec |
| `quantizedMatmulMatchesDequantizedFloatMatmul` | `quantizedMatmul(x, quantize(w)..., transpose=true)` vs. `matmul(x, transpose(dequantize(quantize(w))))` computed from the **same** quantized components -- a composition identity (Decision 8) tighter than comparing against the original unquantized `w`, since both sides now do identical math fused vs. unfused and should agree near-exactly, isolating "is `quantizedMatmul` correct" from "how lossy is 4-bit quantization" |
| `quantizeWithNullGlobalScaleAndAbsentGroupSizeBitsSucceeds` | The nullable/absent-optional paths this facade supports but `QuantizedLinear` never exercises (Global Constraint 5) -- mirrors `MLXFastTest`'s null-`weight` `rmsNorm` coverage |
| **`quantizeWithGlobalScaleInAChildScopeOfWAllocatesIntoTheChild`, and the unrelated-scope negative** | Global Constraint 7's `scopeOf(w, globalScale)` rule, discriminating the case a `w.scope()`-only target would silently pass: build `w` in a parent, `globalScale` in a child, assert the result lands in the child; then `globalScale` in an unrelated root, assert `IllegalArgumentException` -- same two-row shape as M3's SDPA `q`/`k`/`v` scope test |
| `dequantizeWithNullBiasesAndNullGlobalScaleSucceeds` | The two nullable-array parameters unique to `dequantize` are independently optional, not a package deal |

## Task 2: `QuantizedLinear` (`se.alipsa.jmlx.nn`)

**Files:**
- New: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/QuantizedLinear.java`
- New: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/QuantizedLinearTest.java`

**Design decision: the constructor takes already-quantized components, not a float weight to quantize
on construction.** This mirrors `Linear`'s own constructor (`MLXArray weight`, not "an initializer
spec") and is the only shape compatible with Phase 5's eventual checkpoint loader: a real quantized
checkpoint (the outline's own stated GGUF/MLX-quantization use case) stores `weight`/`scales`/`biases`
directly on disk -- there is no plaintext weight at load time to quantize. `MLXQuant.quantize` is the
tool a caller uses *before* construction (as the test in this task does, to build a fixture from a
known float weight) or never, if loading a real checkpoint. A `QuantizedLinear.from(Linear, groupSize,
bits)` convenience factory is a natural follow-on but has no caller yet -- see Deliberately not
covered.

```java
package se.alipsa.jmlx.nn;

import java.util.Arrays;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * A quantized fully-connected layer: {@code y = quantizedMatmul(x, weight, scales, biases,
 * transpose=true) + bias}, {@code mode} fixed to {@code "affine"} (the only mode with a documented
 * rationale in this codebase -- req/phase4-plan.md, Research findings). {@code weight} is the
 * {@code UINT32}-packed form of a checkpoint-layout {@code [out, in]} weight -- see {@link
 * MLXQuant#quantize} to build one from a float weight, or load one directly from a quantized
 * checkpoint (Phase 5). Unlike {@link Linear}, {@code forward} needs no explicit-target overload for
 * its weight-bearing op: {@link MLXQuant#quantizedMatmul}'s javadoc explains why.
 */
public final class QuantizedLinear extends Module implements UnaryModule {

  private static final String MODE = "affine";

  private final int groupSize;
  private final int bits;
  private final boolean hasBias;

  /**
   * {@code weight} is {@code [out, packedIn]} `UINT32` (packed columns = {@code in * bits / 32});
   * {@code scales}/{@code biases} are each {@code [out, in / groupSize]`}; {@code bias}, if non-null,
   * is {@code [out]}. Native validates the packing/group-size relationship on the first {@link
   * #forward} call; this constructor validates only the shape relationships it can check without
   * unpacking {@code weight} -- the same division of labor {@link Linear}'s own constructor draws.
   */
  public QuantizedLinear(
      MLXScope scope,
      MLXArray weight,
      MLXArray scales,
      MLXArray biases,
      MLXArray bias,
      int groupSize,
      int bits) {
    super(scope);
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException(
          "QuantizedLinear: weight must be rank 2 [out, packedIn], got shape "
              + Arrays.toString(weight.shape()));
    }
    if (scales.ndim() != 2 || !Arrays.equals(scales.shape(), biases.shape())) {
      throw new IllegalArgumentException(
          "QuantizedLinear: scales and biases must have the same rank-2 shape, got "
              + Arrays.toString(scales.shape())
              + " and "
              + Arrays.toString(biases.shape()));
    }
    if (scales.shape()[0] != weight.shape()[0]) {
      throw new IllegalArgumentException(
          "QuantizedLinear: scales/biases first dimension must match weight's out dimension ("
              + weight.shape()[0]
              + "), got "
              + scales.shape()[0]);
    }
    if (bias != null && (bias.ndim() != 1 || bias.shape()[0] != weight.shape()[0])) {
      throw new IllegalArgumentException(
          "QuantizedLinear: bias must be rank 1 with length weight.shape()[0]="
              + weight.shape()[0]
              + ", got shape "
              + Arrays.toString(bias.shape()));
    }
    if (groupSize <= 0 || bits <= 0) {
      throw new IllegalArgumentException(
          "QuantizedLinear: groupSize and bits must be positive, got groupSize="
              + groupSize
              + ", bits="
              + bits);
    }
    param("weight", weight);
    param("scales", scales);
    param("biases", biases);
    hasBias = bias != null;
    if (hasBias) {
      param("bias", bias);
    }
    this.groupSize = groupSize;
    this.bits = bits;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    MLXArray y =
        MLXQuant.quantizedMatmul(
            x, param("weight"), param("scales"), param("biases"), true, groupSize, bits, MODE);
    return hasBias ? MLXOps.add(y, param("bias")) : y;
  }
}
```

**`groupSize`/`bits` are plain `int` fields, not `MLXArray` parameters** -- same reasoning as
`Linear`'s `hasBias` boolean (`Module`'s own javadoc: "a cached scalar field ... has no such hazard,
since it is never an `MLXArray` and is never rebound"). They are structural configuration of the
layer, not a value `update`/`rebind` ever touches.

**No memory-leak-loop test names a specific hazard here, unlike `Linear`'s.** `Linear.forward`'s own
`MLXMemoryLeakTest`-shaped test exists specifically because `transpose(W)` is a single-operand op on a
parameter that *could* allocate into the model scope if written wrong (§2's fifth sub-hazard).
`QuantizedLinear.forward` has no such op: `quantizedMatmul` is multi-operand with `x` always present
(Global Constraint 6), and `add(y, bias)` resolves through the existing `binaryOp`/`scopeOf` path from
`y` (already in the step scope) and `bias` (the model scope) exactly like `Linear`'s own `add` call.
A leak-loop test is still added, as a general regression guard consistent with every other layer
having one, but its javadoc says explicitly that it targets no named hazard, rather than inventing one
to match the pattern.

### `QuantizedLinearTest.java` (new file)

`@EnabledIfNativeAvailable`; fixture: `[2, 64]` float weight (`out=2`, `in=64` -- the smallest shape
divisible by the `groupSize=32` used throughout Task 1, doubled to `out=2` so shape-mismatch tests on
`bias`/`scales` have a non-trivial dimension to get wrong), built with deterministic non-uniform
values (e.g. `(i % 7 - 3) * 0.3f`) so quantization error is neither zero nor a single repeated value.

| Test | Proves |
|---|---|
| `forwardMatchesLinearOnTheDequantizedWeight` | Composition identity, tight `EPS = 1e-3f` (this is fused-vs-unfused arithmetic on identical quantized components, not a lossy comparison -- see `MLXQuantTest`'s equivalent row): `new QuantizedLinear(wQ, scales, biases, bias, ...).forward(x)` vs. `new Linear(dequantize(wQ, scales, biases, ...), bias).forward(x)` |
| `forwardWithoutBiasComputesTheQuantizedTransformOnly` | The `hasBias ? add : y` branch's `false` side, mirroring `LinearTest`'s equivalent row |
| `parametersExposesWeightScalesBiasesAndOptionalBiasInTheCheckpointLayout` | `parameters()` returns `weight`/`scales`/`biases`/`bias` at the exact dotted paths a Phase 5 loader will write into -- the same "pins the layout contract" reasoning as `Linear`'s own `parametersReturnsWeightInTheCheckpointsOutInShapeNotItsTranspose` test |
| `constructorRejectsScalesAndBiasesWithMismatchedShapes`, `constructorRejectsScalesWithWrongOutDimension`, `constructorRejectsABiasOfTheWrongLength`, `constructorRejectsNonPositiveGroupSizeOrBits` | Each shape/value guard in the constructor, mirroring `LinearTest`'s `constructorRejects*` precedent |
| `activeMemoryDoesNotGrowWithPerIterationChildScopeUnderALongLivedParent` | General regression guard (see the class-level note above on why no specific hazard is named); same warmup/measured-iteration/threshold shape as `LinearTest`'s equivalent |

## Task 3: Documentation (§9)

No production op or layer code in this task -- only the four items spec §9 names, each checked against
the current tree rather than assumed still-needed.

* **`req/project-outline.md`'s Phase 4 section (`:83-91`): mark delivered**, with the same explicit
  reconciliation Phase 3 used for "thread-safe" rather than a bare checkmark -- state that `nn` is
  built on a public op surface living in `core` (not inside `nn` itself, contra a literal reading of
  the outline's own package name), and that autograd (`MLXGrad`) shipped in M2 even though the
  outline lists no autograd deliverable anywhere under Phase 4.
* **The architecture diagram (`req/project-outline.md:13`) still lists `Stream`, `Device` and
  `Autograd` as `core` types.** Amend the `core` row: `Autograd` is real, as `MLXGrad`; `Stream` and
  `Device` still do not exist (this slice caches a single process-lifetime default of each -- see
  `MLX.defaultDevice()`/`defaultStream()` -- and exposes no switching). Amend the line rather than
  leaving it aspirational, per §9's own instruction.
* **`MLX.java`'s class javadoc already cites `req/phase4-plan.md` §1** (added during M0a, verified
  against the current tree while writing this plan) **-- no change needed here.** Recorded so a
  reader checking off §9's bullet list does not re-add a citation that is already present.
* **`MLXQuant.java`'s own class javadoc is still the M0a placeholder** ("Empty until M4 ... created
  now, during M0a's pure-motion facade split"). Task 1 replaces it with a real description of
  `quantize`/`dequantize`/`quantizedMatmul` plus the "see `MLX`'s javadoc for the index of every
  sibling" back-pointer every other split class (`MLXOps`, `MLXShape`, `MLXFast`, `MLXRandom`) already
  carries -- verified against the current tree that `MLXQuant` is the one split class still missing
  it, precisely because it had no real ops to describe until now.
* **`jmlx-examples/HelloMLX`: add a `Linear` forward pass inside a child scope.** Current `HelloMLX`
  (`jmlx-examples/src/main/java/se/alipsa/jmlx/examples/HelloMLX.java`) only exercises M0a/M0b-era ops
  (`add`/`matmul`, cross-shape broadcast) in a single flat scope -- neither of Phase 4's two
  behavioural changes (cross-scope ops resolving via `scopeOf`, child-scope lifetime) is visible in
  the demo. Add, after the existing `broadcastSum` block: build a `Linear` with a weight/bias in the
  outer (model) scope, open a child scope, build an activation `x` in the child, call
  `linear.forward(x)`, print the result, let the child close. This demonstrates exactly the shape
  Task 2's own `QuantizedLinear` forward relies on (weight in an ancestor scope, activation in a
  descendant), without requiring `HelloMLX` to depend on quantization specifically.

## Task 4: full verification pass

Mirrors `req/plans/phase4-m3-plan.md`'s own closing task. No new production code -- this task only
runs checks and fixes anything they surface.

- [ ] `./gradlew build` -- Spotless, Checkstyle, the forked `loaderGuardTest`, plus every module's test
  suite. Report actual test counts.
- [ ] `./gradlew :jmlx-core:test --tests '*MLXMemoryLeakTest*'` -- confirms this plan's new op surface
  introduced no regression in the existing leak suite.
- [ ] `./gradlew :jmlx-core:test --tests '*MLXGpuVerificationTest*'` -- kernels still dispatch to GPU.
- [ ] `./gradlew :jmlx-examples:run` -- `HelloMLX` prints correct values for the new `Linear` forward
  pass in a child scope (Task 3's last bullet).
- [ ] `scripts/regen-bindings.sh` then `git diff --exit-code jmlx-ffi/src/main/generated/java` -- must
  stay clean. This plan adds zero new mlx-c symbols (`mlx_quantize`/`mlx_dequantize`/
  `mlx_quantized_matmul` and `mlx_optional_dtype_` are all already bound, confirmed while researching
  this plan) -- the last of `req/phase4-plan.md`'s four merge points to prove Phase 4 needed no
  binding regeneration at all.
- [ ] `grep -rn 'mlx_array_new(' jmlx-core/src/main` -- every call site's allocator is `scopeOf(...)`'s
  result, an explicit `MLXScope` parameter, or a confined `Arena` freed by a local `finally` (spec
  Verification #6). `dequantize`/`quantizedMatmul`'s hand-rolled bodies from Task 1 must fall into the
  first case.
- [ ] `grep -rn 'mlx_vector_array_new' jmlx-core/src/main` -- every call site's allocator confirmed
  **by eye** to be a confined `Arena`, never an `MLXScope` (spec Verification #7). `quantize`
  introduces no new call site here at all -- it reuses `vectorOutOp`, whose own `mlx_vector_array_new`
  call was already verified in M3.
- [ ] `grep -rn 'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` -- zero
  matches (spec Verification #6a).
- [ ] Update the Status table at the top of `req/phase4-plan.md`: mark M4 `Done` and §9 `Done`, each
  with a commit hash and a short findings note (matching M1-M3's own entries) summarizing this plan's
  empirically-confirmed quantization facts, so a later reader does not have to re-derive them.

## Deliberately not covered by this plan

- **§10 (CI, self-hosted runner)** -- an unrelated, already-tracked follow-up; not touched here, same
  as every prior plan.
- **`Mxfp4`/`Mxfp8`/`Nvfp4` quantization modes** -- reachable through `MLXQuant`'s generic `mode`
  string parameter but exercised by no test; their packing layout is not `affine`'s and is not
  verified against this mlx-c version at all (Findings section).
- **`mlx_gather_qmm`/`mlx_qqmm`/`mlx_gather_mm`** (gathered/MoE quantized matmul) -- not named in spec
  §8; no consumer in this codebase.
- **`QuantizedLinear.from(Linear, groupSize, bits)`, or any other quantize-on-construction
  convenience** -- no caller needs it yet (see Task 2's design-decision note); the two-step
  `MLXQuant.quantize` then `new QuantizedLinear(...)` path this plan's own test uses is not hidden
  behind a factory until a second caller makes the duplication worth naming.
- **Quantization-aware training** (gradients through `quantizedMatmul`, or a `ModuleGrad` that treats
  quantized weights specially) -- `QuantizedLinear` is an inference-shaped layer; nothing in Phase 4's
  scope trains one.
- **Safetensors/GGUF loading of real quantized checkpoints** -- Phase 5, per `req/phase4-plan.md`'s
  own Out of scope section. `QuantizedLinear`'s components-based constructor (Task 2) is the hook Phase
  5 will load into, exactly as `Linear`'s own constructor already is.
