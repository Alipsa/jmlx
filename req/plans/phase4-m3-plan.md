# jmlx Phase 4 — M3 implementation plan (RoPE, MultiHeadAttention, KV cache)

**Spec:** `req/phase4-plan.md` §7 (the op list this merge point adds), §2 (`scopeOf`/`innermost`, the
`vectorOutOp` two-allocator shape, `hoist`'s "freeing a handle only decrements a refcount" safety
proof), §3 (the six generic op-body helpers — this plan builds the three M3 actually needs), §5
(`Module`/layer conventions, the "forward() must never allocate into `scope()`" rule), Decision 8 (test
goldens; composition-identity fallback), Testing approach (the exact fixture size for layer tests: "1
batch, 2 heads, dim 4, 3 tokens"). M4 (`QuantizedLinear`) and §9 (Documentation) are out of scope — §9
covers the whole phase and stays `NOT STARTED` until M4 also lands (M1/M2 didn't touch it either; see
`req/phase4-plan.md`'s Status table).

Branch: `worktree-phase4-m3`, off `main` at `8bb08ad` (PR #8 merged, M2 done). Native bootstrap staged
in this worktree (`native/install/lib/mlx.metallib` exists) — `@EnabledIfNativeAvailable` tests run for
real, against real Apple Silicon GPU hardware.

## Findings from this plan's pre-work (Verification item 0e, resolved; three more facts settled beyond it)

Per `req/phase4-plan.md`'s Verification item 0, RoPE and SDPA's exact behavior is not fully specified
by the header alone and was probed with scratch classes compiled and run against the real bindings
(same precedent as Probes 0b/0f), then deleted. Findings, all confirmed against real hardware output:

* **RoPE's rotation formula (item 0e).** For `traditional=false`, dims `d`, pairs are the **half-split**
  form `(x_i, x_{i+d/2})` for `i` in `[0, d/2)` — **not** the interleaved `(x_{2i}, x_{2i+1})` form.
  `freq_i = base^(-2i/d)`; `angle_i = offset * scale * freq_i`; `x_i' = x_i*cos(angle_i) -
  x_{i+d/2}*sin(angle_i)`; `x_{i+d/2}' = x_i*sin(angle_i) + x_{i+d/2}*cos(angle_i)`. Confirmed: `x =
  [1,0,1,0]`, `dims=4`, `base=10000`, `scale=1.0`, `offset=1` → `[-0.30116868, 0.0, 1.3817731, 0.0]`
  (pair `(x0,x2)=(1,1)` rotates by `angle_0=1` rad; pair `(x1,x3)=(0,0)` is fixed by any rotation).
  `offset=0` on the same input is the identity (`angle=0` ⇒ `cos=1,sin=0`), confirmed as
  `[1.0, 0.0, 1.0, 0.0]`.
* **`mlx_fast_rope` requires at least one of `base`/`freqs`.** Passing `base.has_value=false` with
  `freqs` null throws a clean `MLXException` ("Neither base nor freqs has a value") — not a crash, not
  a silent default. No extra Java-side guard is needed (native's message already names the problem);
  `MLXFast.rope`'s javadoc documents the requirement rather than re-deriving the check.
* **`mlx_fast_rope` requires the input to have at least 3 dimensions** (`[..., T, dims]`) — a 2-D
  input (`[T, dims]`) throws "Input must have at least 3 dimensions". `MultiHeadAttention` always calls
  it on a 4-D `[B, H, T, Dh]` tensor, so this never bites M3's real caller, but it means a *direct*
  `MLXFastTest` unit test for `rope` must use at least a 3-D shape (this plan uses `[1, 1, 4]`).
* **SDPA's `mask_mode="causal"` and the composed reference agree exactly.** `q=k=v=[[1,0],[0,1]]`
  (`B=1,H=1,T=2,Dh=2`), `scale=1.0`, causal → `[1.0, 0.0, 0.2689414, 0.7310586]`. Hand check: row 0 can
  only attend to position 0 (masked score row `[1,-inf]` → softmax `[1,0]` → output `v0=[1,0]`); row 1
  attends to both (`[0,1]` → softmax `[0.26894, 0.73106]` → `0.26894*v0+0.73106*v1=[0.26894,0.73106]`).
  Matches to float precision.
* **`mask_mode="array"` is additive for a `FLOAT32` mask and boolean (`true`=attend, `false`=mask out)
  for a `BOOL` mask; both reproduce the causal result exactly when built to encode the same causal
  constraint.** Additive mask `[0,-1e9,0,0]` on the same `q,k,v` → identical
  `[1.0, 0.0, 0.2689414, 0.7310586]`. Boolean mask `true` exactly at the strictly-upper cell (`col >
  row`) → `[0.0, 1.0, 0.5, 0.5]`: row 0's only `true` cell is column 1, so it attends *only* to `v1`
  (`[0,1]`); row 1 has no `true` cell at all, and an all-masked row degenerates to a uniform softmax
  (`[0.5,0.5]`), giving `0.5*v0+0.5*v1=[0.5,0.5]`. Both match exactly — this plan does not need to
  guess mask polarity.
* **The built-in `"causal"` mode correctly bottom-aligns when the query is shorter than the
  key/value (the KV-cache decode shape).** `Tq=1` (`q=[1,1]`), `Tk=3` (`k=v=[[1,0],[0,1],[1,1]]`),
  causal, `scale=1.0` → `[0.78805846, 0.78805846]`. Hand check: the single query position is treated
  as absolute position `Tk-1=2`, so it attends unmasked to all three keys: scores `[1,1,2]` → softmax
  `[0.21194,0.21194,0.57612]` → `0.21194*[1,0]+0.21194*[0,1]+0.57612*[1,1] = [0.78806,0.78806]`.
  **This is what makes `MultiHeadAttention.forward` correct during KV-cache decode with zero extra
  masking code** — the same `causal=true` flag used for prefill also works for a single-token decode
  step against the accumulated cache, because mlx's own causal mask is defined relative to `Tk`, not
  `Tq`.
* **`mlx_split(res, a, num_splits, axis, s)` splits into `num_splits` equal-size parts along `axis`, in
  order.** Confirmed: `a=[1,2,3,4,5,6]` shape `[1,6]`, `num_splits=3`, `axis=1` → three `[1,2]`-shaped
  parts `[1,2]`, `[3,4]`, `[5,6]`.

## Global Constraints

1. **Package boundaries are unchanged from M1/M2.** New op-surface additions (`NativeOps` helpers,
   `MLXShape`/`MLXOps`/`MLXFast` methods) live in `se.alipsa.jmlx.core`. `KVCache` and
   `MultiHeadAttention` live in `se.alipsa.jmlx.nn` and depend only on the public `core` surface.
   `grep -rn 'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` must stay
   empty (spec Verification #6a).
2. **`NativeOps` keeps its zero-sibling-dependency invariant** (see its own class javadoc: "depends on
   nothing else here"). The new `vectorInOp`/`vectorOutOp` helpers build their `mlx_vector_array`
   directly from `mlx_h` calls plus the already-present `copyHandlesInto` — they must **not** call
   `MLX.newVectorArray`, even though the logic is a few lines longer as a result. `MLXGrad` (a normal
   sibling, not the foundation layer) is free to call `MLX.newVectorArray`, as it already does; that
   precedent does not extend to `NativeOps` itself.
3. **Only the generic helpers with a real M3 consumer are added: `axisOp`, `vectorInOp`, `vectorOutOp`,
   `optFloat`, `cstr`.** `optInt`/`optDtype` from spec §3's original six-plus-two list are **not**
   added — nothing in M3 needs them (they are `quantize`/`quantizedMatmul` parameters, M4's job). This
   is the same principle the M0d Status note already established for the original six helpers: build
   each alongside its first real consumer, not speculatively.
4. **Every multi-operand op resolves its target via `NativeOps.scopeOf(...)` over *every* array
   operand, including nullable ones** — `rope`'s `(x, freqs)`, SDPA's `(queries, keys, values,
   maskArr, sinks)` (the exact 5-operand, 2-nullable shape spec §2's operand-count table names for
   `mlx_fast_scaled_dot_product_attention`).
5. **Style:** run `./gradlew spotlessApply` before finishing, then `./gradlew build` (Spotless,
   Checkstyle, full test suite including native tests and the forked `loaderGuardTest`) must succeed.
   Report actual test counts from the build output.
6. **Testing style, matching every existing test file:** `@EnabledIfNativeAvailable` on the test class;
   `try (MLXScope scope = new MLXScope())`; hand-computed or empirically-confirmed goldens (this plan's
   "Findings" section above supplies the confirmed ones); `assertArrayEquals(expected,
   actual.toFloatArray(), EPS)`; `EPS = 1e-5f` for exact-rational goldens, `EPS = 1e-3f` for goldens
   involving `cos`/`sin`/`exp`/`sqrt` (matching `MLXFastTest`'s own precedent for `rmsNorm`/`layerNorm`).
7. **Op tests are added to the existing flat test classes, not new per-op files** — `MLXNumericTest`
   for `MLXOps`/`MLXShape` additions, `MLXFastTest` for `MLXFast` additions — matching how every
   existing op test already lives (`MLXNumericTest`'s own javadoc: "every facade op against
   hand-computed values", one flat class, not one class per op). `KVCache` and `MultiHeadAttention`
   are new `nn` classes and get new, dedicated test files, matching `LinearTest`/`RMSNormTest`'s own
   one-class-per-layer precedent.

## Native surface this plan uses (confirmed against the generated bindings and the probes above)

| Java call | C signature | Notes |
|---|---|---|
| `mlx_h.mlx_fast_rope(res, x, dims, traditional, base, scale, offset, freqs, s)` | `int mlx_fast_rope(mlx_array* res, const mlx_array x, int dims, bool traditional, mlx_optional_float base, float scale, int offset, const mlx_array freqs, const mlx_stream s)` (`fast.h:169-178`) | `base`/`freqs` each may be omitted but not both (see Findings). `base`/`freqs` are by-value structs — `base` via `optFloat`, `freqs` via `nullableHandle`. |
| `mlx_h.mlx_fast_scaled_dot_product_attention(res, queries, keys, values, scale, mask_mode, mask_arr, sinks, s)` | `int mlx_fast_scaled_dot_product_attention(mlx_array* res, const mlx_array queries, const mlx_array keys, const mlx_array values, float scale, const char* mask_mode, const mlx_array mask_arr, const mlx_array sinks, const mlx_stream s)` (`fast.h:189-198`) | `mask_mode` ∈ `{"", "causal", "array"}`, allocated once per value in `NativeOps.FACADE_ARENA` via `cstr`. `mask_arr`/`sinks` are by-value nullable structs via `nullableHandle`. |
| `mlx_h.mlx_split(res, a, num_splits, axis, s)` | `int mlx_split(mlx_vector_array* res, const mlx_array a, int num_splits, int axis, const mlx_stream s)` (`ops.h:1075-1080`) | `res` is a `mlx_vector_array*` out-param — the `vectorOutOp` shape. |
| `mlx_h.mlx_concatenate_axis(res, arrays, axis, s)` | `int mlx_concatenate_axis(mlx_array* res, const mlx_vector_array arrays, int axis, const mlx_stream s)` (`ops.h:213-217`) | `arrays` is an `mlx_vector_array` **by value** — the `vectorInOp` shape. |
| `mlx_h.mlx_expand_dims/_tril/_triu(res, a, int, s)` | all `int mlx_*(mlx_array* res, const mlx_array x, int k_or_axis, const mlx_stream s)` (`ops.h:407-411`, `:1232-1233`) | Identical `(res, a, int, stream)` shape — the new `axisOp` helper. |
| `mlx_h.mlx_flatten(res, a, start_axis, end_axis, s)` | `int mlx_flatten(mlx_array* res, const mlx_array a, int start_axis, int end_axis, const mlx_stream s)` (`ops.h:420-425`) | Identical shape to `mlx_swapaxes` — reuses the **existing** `NativeOps.axis2Op`, no new helper. |
| `mlx_h.mlx_less/_less_equal/_greater/_greater_equal/_equal(res, a, b, s)` | plain `(res, a, b, stream)` | Fit the existing `NativeOps.binaryOp` exactly. |
| `mlx_h.mlx_where(res, condition, x, y, s)` | `int mlx_where(mlx_array* res, const mlx_array condition, const mlx_array x, const mlx_array y, const mlx_stream s)` (`ops.h:1267-1272`) | Three array operands, none nullable — hand-rolled via `NativeOps.scopeOf`, same shape as `MLXOps.inner`/`outer`'s hand-rolled bodies. |
| `mlx_h.mlx_softmax_axis(res, a, axis, precise, s)` | `int mlx_softmax_axis(mlx_array* res, const mlx_array a, int axis, bool precise, const mlx_stream s)` (`ops.h:1058-1063`) | Extra `bool` beyond `axisOp`'s 3-param shape (same reason `MLXOps.sum(a)` is hand-rolled beyond `unaryOp`) — hand-rolled. |
| `mlx_optional_float_` | `struct { float value; bool has_value; }`, 8 bytes (`optional.h:32-35`, confirmed against the generated `mlx_optional_float_.java`) | No native constructor function — allocate via `mlx_optional_float_.allocate(tmp)` and set both fields, exactly as spec §6 describes for the sibling `mlx_optional_int`. |

## Task 1: `NativeOps` helpers + `MLXShape`/`MLXOps` op additions (`se.alipsa.jmlx.core`)

**Files:**
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/NativeOps.java`
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXShape.java`
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXOps.java`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXNumericTest.java` (append new `@Test` methods)

**Interfaces produced (for Tasks 2 and 4):**
- `NativeOps.axisOp(String, MLXArray, int, AxisOp)`, `NativeOps.vectorInOp(String, MLXArray[], int,
  VectorInOp)`, `NativeOps.vectorOutOp(String, MLXScope, VectorOutOp)`, `NativeOps.optFloat(Arena,
  Float)`, `NativeOps.cstr(String)` — all package-private, used by `MLXFast` in Task 2.
- `MLXShape.concatenate(MLXArray[], int)`, `MLXShape.split(MLXArray, int, int)` (returns `MLXArray[]`),
  `MLXShape.expandDims(MLXArray, int)`, `MLXShape.flatten(MLXArray, int, int)`,
  `MLXShape.tril(MLXArray, int)`, `MLXShape.triu(MLXArray, int)`.
- `MLXOps.less/lessEqual/greater/greaterEqual/equal(MLXArray, MLXArray)`, `MLXOps.where(MLXArray,
  MLXArray, MLXArray)`, `MLXOps.softmaxAxis(MLXArray, int, boolean)` — used by Task 4's test.

### Step 1a: `NativeOps` — `axisOp`

Add next to `axis2Op`:

```java
/**
 * Wraps the {@code (res, a, int, stream)} native shape shared by {@code mlx_expand_dims}, {@code
 * mlx_tril} and {@code mlx_triu} (req/phase4-plan.md §7's Native surface table). Unlike {@code
 * axis2Op}, this is a single caller-supplied int -- an axis for {@code expandDims}, a diagonal
 * offset {@code k} for {@code tril}/{@code triu} -- not necessarily an axis in every case, so the
 * parameter is named generically in the functional interface below.
 */
static MLXArray axisOp(String opName, MLXArray a, int param, AxisOp op) {
  MLXScope scope = a.scope();
  MemorySegment res = mlx_h.mlx_array_new(scope);
  checked(opName, () -> op.apply(res, a.handle(), param, DEFAULT_STREAM));
  return new MLXArray(scope, res);
}

@FunctionalInterface
interface AxisOp {
  int apply(MemorySegment res, MemorySegment a, int param, MemorySegment stream);
}
```

### Step 1b: `NativeOps` — `vectorInOp`

Add next to `copyHandlesInto`. Builds its own `mlx_vector_array` directly (Global Constraint 2 — must
not call `MLX.newVectorArray`):

```java
/**
 * Wraps an op taking an {@code mlx_vector_array} <em>input</em> by value (e.g. {@code
 * mlx_concatenate_axis}) plus a caller-supplied {@code axis}. Resolves the result's scope via {@link
 * #scopeOf} over every element of {@code xs} -- the ancestor rule generalized to N operands
 * (req/phase4-plan.md §3). Builds the {@code mlx_vector_array} itself via {@link #copyHandlesInto}
 * rather than delegating to {@code MLX.newVectorArray}: this class depends on nothing else in this
 * package (see the class javadoc), and calling into {@code MLX} here would be the one exception.
 */
```

> **Amended after execution.** This body leaks: `mlx_vector_array_new_data`'s vector is never freed,
> leaking one refcount per operand on every call -- invisible in a single call, `O(N^2)` active
> memory in a decode-loop shape (discovered during Task 3, fixed in commit `ffb042c`). The shipped
> version wraps the call in `try { ... } finally { mlx_h.mlx_vector_array_free(vec); }` -- see
> `NativeOps.vectorInOp` in the actual source for the correct body. Left here unmodified as a record
> of what shipped originally; do not copy this block.

```java
static MLXArray vectorInOp(String opName, MLXArray[] xs, int axis, VectorInOp op) {
  MLXScope scope = scopeOf(opName, xs);
  MemorySegment[] handles = new MemorySegment[xs.length];
  for (int i = 0; i < xs.length; i++) {
    handles[i] = xs[i].handle();
  }
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment buf = copyHandlesInto(handles, tmp);
    // mlx_vector_array_new_data is statusless -- same null-ctx-on-failure hazard MLX.array and
    // MLX.newVectorArray already guard against explicitly.
    NativeLoader.clearLastNativeError();
    MemorySegment vec = mlx_h.mlx_vector_array_new_data(tmp, buf, handles.length);
    if (mlx_vector_array_.ctx(vec).address() == 0) {
      throw nativeFailure("mlx_vector_array_new_data");
    }
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(opName, () -> op.apply(res, vec, axis, DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }
}

@FunctionalInterface
interface VectorInOp {
  int apply(MemorySegment res, MemorySegment arrays, int axis, MemorySegment stream);
}
```

Add the import `se.alipsa.jmlx.ffi.mlx_vector_array_;` to `NativeOps.java` (not yet imported there;
`MLX.java` already imports it for the same struct).

### Step 1c: `NativeOps` — `vectorOutOp`

Add next to `vectorInOp`. This is spec §3's own code block, verbatim in shape (the two-allocator
hazard: `tmp` for the intermediate vector, `target` for each unpacked element):

```java
/**
 * Wraps an op producing an {@code mlx_vector_array} <em>output</em> (e.g. {@code mlx_split}).
 * {@code target} is an explicit parameter, not inferred from an operand, purely for legibility
 * (req/phase4-plan.md §3: "its inputs may be none" was a false justification in an earlier draft of
 * that document -- {@code mlx_split}'s {@code a} operand does exist -- the real reason is that this
 * helper's whole hazard is two allocators with opposite correct answers on adjacent lines, and
 * naming the target in the signature keeps that contrast visible at the call site).
 *
 * <p>{@code op} is invoked with only {@code (vec, stream)}; any other native parameters (the source
 * array, {@code num_splits}, an axis, ...) are captured by the caller's lambda, the same pattern
 * {@link #checked(String, IntSupplier)} already uses throughout this class.
 */
static MLXArray[] vectorOutOp(String opName, MLXScope target, VectorOutOp op) {
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment vec = mlx_h.mlx_vector_array_new(tmp); // tmp -- NOT target
    try {
      checked(opName, () -> op.apply(vec, DEFAULT_STREAM));
      long n = mlx_h.mlx_vector_array_size(vec);
      MLXArray[] out = new MLXArray[(int) n];
      for (int i = 0; i < n; i++) {
        MemorySegment h = mlx_h.mlx_array_new(target); // target -- NOT tmp
        final long idx = i;
        checked(opName, () -> mlx_h.mlx_vector_array_get(h, vec, idx));
        out[i] = new MLXArray(target, h);
      }
      return out;
    } finally {
      mlx_h.mlx_vector_array_free(vec);
    }
  }
}

@FunctionalInterface
interface VectorOutOp {
  int apply(MemorySegment vec, MemorySegment stream);
}
```

### Step 1d: `NativeOps` — `optFloat` and `cstr`

Add next to `nullableHandle`:

```java
/**
 * The native encoding of an {@code mlx_optional_float} by-value struct (req/phase4-plan.md §7's
 * Native surface table: 8 bytes, {@code {float value; bool has_value;}}, no native constructor
 * function -- allocate and set both fields, the same shape spec §6 describes for the sibling {@code
 * mlx_optional_int}). {@code value == null} encodes "absent" ({@code has_value=false}); the {@code
 * value} field is left at {@code 0f} in that case, which mlx-c never reads (see {@code
 * mlx_optional_float_.value}'s getter contract -- it is caller-supplied storage, not itself
 * consulted for validity).
 */
static MemorySegment optFloat(Arena tmp, Float value) {
  MemorySegment seg = mlx_optional_float_.allocate(tmp);
  mlx_optional_float_.value(seg, value != null ? value : 0f);
  mlx_optional_float_.has_value(seg, value != null);
  return seg;
}

/**
 * Allocates a NUL-terminated C string once, in {@link #FACADE_ARENA} -- for the closed-set {@code
 * const char*} parameters this facade's mlx-c surface uses (SDPA's {@code mask_mode}), which never
 * need per-call allocation (req/phase4-plan.md §7's Native surface table). Callers store the result
 * in a {@code private static final MemorySegment} field per distinct literal, exactly like {@link
 * #DEFAULT_STREAM} -- this method allocates on every call, so it must only ever be invoked from a
 * static initializer, never per-op.
 */
static MemorySegment cstr(String s) {
  return FACADE_ARENA.allocateFrom(s);
}
```

Add the import `se.alipsa.jmlx.ffi.mlx_optional_float_;` to `NativeOps.java`.

### Step 1e: `MLXShape` additions

```java
/** Concatenates {@code arrays} along {@code axis}; every array must agree on every other axis. */
public static MLXArray concatenate(MLXArray[] arrays, int axis) {
  if (arrays.length == 0) {
    throw new IllegalArgumentException("concatenate: requires at least one array");
  }
  return NativeOps.vectorInOp("concatenate", arrays, axis, mlx_h::mlx_concatenate_axis);
}

/** Splits {@code a} into {@code numSplits} equal-size parts along {@code axis}, in order. */
public static MLXArray[] split(MLXArray a, int numSplits, int axis) {
  return NativeOps.vectorOutOp(
      "split",
      a.scope(),
      (vec, stream) -> mlx_h.mlx_split(vec, a.handle(), numSplits, axis, stream));
}

/** Inserts a new size-1 axis at position {@code axis}. */
public static MLXArray expandDims(MLXArray a, int axis) {
  return NativeOps.axisOp("expandDims", a, axis, mlx_h::mlx_expand_dims);
}

/**
 * Merges the axes from {@code startAxis} to {@code endAxis} (inclusive) into a single axis. Fits
 * the existing {@link NativeOps#axis2Op} exactly -- same {@code (res, a, int, int, stream)} shape as
 * {@code mlx_swapaxes} -- so no new helper is needed for this op.
 */
public static MLXArray flatten(MLXArray a, int startAxis, int endAxis) {
  return NativeOps.axis2Op("flatten", a, startAxis, endAxis, mlx_h::mlx_flatten);
}

/** Zeroes every element strictly above the {@code k}-th diagonal (the upper triangle kept is {@code
 * k} diagonals above the main one; {@code k=0} keeps the main diagonal). */
public static MLXArray tril(MLXArray a, int k) {
  return NativeOps.axisOp("tril", a, k, mlx_h::mlx_tril);
}

/** Zeroes every element strictly below the {@code k}-th diagonal -- the complement of {@link
 * #tril}. {@code triu(ones(shape, BOOL), 1)} is the standard strictly-upper causal mask. */
public static MLXArray triu(MLXArray a, int k) {
  return NativeOps.axisOp("triu", a, k, mlx_h::mlx_triu);
}
```

### Step 1f: `MLXOps` additions

```java
/** Elementwise {@code a < b}, broadcasting per NumPy's rules. Result dtype is {@code BOOL}. */
public static MLXArray less(MLXArray a, MLXArray b) {
  requireBroadcastCompatible(a, b, "less");
  return NativeOps.binaryOp("less", a, b, mlx_h::mlx_less);
}

/** Elementwise {@code a <= b}. Result dtype is {@code BOOL}. */
public static MLXArray lessEqual(MLXArray a, MLXArray b) {
  requireBroadcastCompatible(a, b, "lessEqual");
  return NativeOps.binaryOp("lessEqual", a, b, mlx_h::mlx_less_equal);
}

/** Elementwise {@code a > b}. Result dtype is {@code BOOL}. */
public static MLXArray greater(MLXArray a, MLXArray b) {
  requireBroadcastCompatible(a, b, "greater");
  return NativeOps.binaryOp("greater", a, b, mlx_h::mlx_greater);
}

/** Elementwise {@code a >= b}. Result dtype is {@code BOOL}. */
public static MLXArray greaterEqual(MLXArray a, MLXArray b) {
  requireBroadcastCompatible(a, b, "greaterEqual");
  return NativeOps.binaryOp("greaterEqual", a, b, mlx_h::mlx_greater_equal);
}

/** Elementwise {@code a == b}. Result dtype is {@code BOOL}. */
public static MLXArray equal(MLXArray a, MLXArray b) {
  requireBroadcastCompatible(a, b, "equal");
  return NativeOps.binaryOp("equal", a, b, mlx_h::mlx_equal);
}

/**
 * Elementwise select: {@code x} where {@code condition} is nonzero, {@code y} otherwise. Three array
 * operands, none nullable -- resolves its target via {@link NativeOps#scopeOf} across all three,
 * same shape as this class's other hand-rolled bodies ({@link #inner}, {@link #outer}).
 */
public static MLXArray where(MLXArray condition, MLXArray x, MLXArray y) {
  MLXScope scope = NativeOps.scopeOf("where", condition, x, y);
  MemorySegment res = mlx_h.mlx_array_new(scope);
  NativeOps.checked(
      "where",
      () ->
          mlx_h.mlx_where(
              res, condition.handle(), x.handle(), y.handle(), NativeOps.DEFAULT_STREAM));
  return new MLXArray(scope, res);
}

/**
 * Softmax along {@code axis}. {@code precise} requests float32 accumulation regardless of {@code
 * a}'s dtype (mlx-c's own {@code precise} flag) -- carries an extra {@code bool} beyond {@link
 * NativeOps#axisOp}'s {@code (res, a, int, stream)} shape, so it is hand-rolled rather than forced
 * through that helper, the same reason {@link #sum(MLXArray)} is hand-rolled beyond {@code unaryOp}.
 */
public static MLXArray softmaxAxis(MLXArray a, int axis, boolean precise) {
  MLXScope scope = a.scope();
  MemorySegment res = mlx_h.mlx_array_new(scope);
  NativeOps.checked(
      "softmaxAxis",
      () -> mlx_h.mlx_softmax_axis(res, a.handle(), axis, precise, NativeOps.DEFAULT_STREAM));
  return new MLXArray(scope, res);
}
```

### Step 1g: tests (append to `MLXNumericTest.java`)

```java
@Test
void concatenateJoinsArraysAlongTheGivenAxis() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray a = MLX.array(scope, new float[] {1, 2}, new int[] {1, 2});
    MLXArray b = MLX.array(scope, new float[] {3, 4, 5}, new int[] {1, 3});
    MLXArray result = MLXShape.concatenate(new MLXArray[] {a, b}, 1);
    assertArrayEquals(new int[] {1, 5}, result.shape());
    assertArrayEquals(new float[] {1, 2, 3, 4, 5}, result.toFloatArray(), EPS);
  }
}

/** Proves {@code vectorInOp} resolves via {@code scopeOf} over every element, not just the first. */
@Test
void concatenateAcrossParentAndChildScopeAllocatesIntoTheChild() {
  try (MLXScope parent = new MLXScope()) {
    MLXArray a = MLX.array(parent, new float[] {1, 2}, new int[] {1, 2});
    try (MLXScope child = parent.newChild()) {
      MLXArray b = MLX.array(child, new float[] {3, 4}, new int[] {1, 2});
      MLXArray result = MLXShape.concatenate(new MLXArray[] {a, b}, 1);
      assertSame(child, result.scope());
      assertArrayEquals(new float[] {1, 2, 3, 4}, result.toFloatArray(), EPS);
    }
  }
}

@Test
void splitDividesIntoEqualPartsInOrder() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4, 5, 6}, new int[] {1, 6});
    MLXArray[] parts = MLXShape.split(a, 3, 1);
    assertEquals(3, parts.length);
    assertArrayEquals(new float[] {1, 2}, parts[0].toFloatArray(), EPS);
    assertArrayEquals(new float[] {3, 4}, parts[1].toFloatArray(), EPS);
    assertArrayEquals(new float[] {5, 6}, parts[2].toFloatArray(), EPS);
  }
}

@Test
void expandDimsInsertsASizeOneAxis() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
    assertArrayEquals(new int[] {1, 3}, MLXShape.expandDims(a, 0).shape());
    assertArrayEquals(new int[] {3, 1}, MLXShape.expandDims(a, 1).shape());
  }
}

@Test
void flattenMergesTheGivenAxisRange() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray a = MLX.array(scope, new float[8], new int[] {1, 2, 2, 2});
    assertArrayEquals(new int[] {1, 4, 2}, MLXShape.flatten(a, 1, 2).shape());
  }
}

@Test
void triuZeroesBelowTheKthDiagonal() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray ones = MLX.ones(scope, new int[] {3, 3}, DType.FLOAT32);
    MLXArray result = MLXShape.triu(ones, 1);
    assertArrayEquals(new float[] {0, 1, 1, 0, 0, 1, 0, 0, 0}, result.toFloatArray(), EPS);
  }
}

@Test
void trilZeroesAboveTheKthDiagonal() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray ones = MLX.ones(scope, new int[] {3, 3}, DType.FLOAT32);
    MLXArray result = MLXShape.tril(ones, 0);
    assertArrayEquals(new float[] {1, 0, 0, 1, 1, 0, 1, 1, 1}, result.toFloatArray(), EPS);
  }
}

@Test
void comparisonsProduceElementwiseBooleanResults() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
    MLXArray b = MLX.array(scope, new float[] {2, 2, 2}, new int[] {3});
    assertEquals(DType.BOOL, MLXOps.less(a, b).dtype());
    assertArrayEquals(
        new float[] {1, 0, 0}, MLX.astype(MLXOps.less(a, b), DType.FLOAT32).toFloatArray(), EPS);
    assertArrayEquals(
        new float[] {0, 1, 1},
        MLX.astype(MLXOps.greaterEqual(a, b), DType.FLOAT32).toFloatArray(),
        EPS);
    assertArrayEquals(
        new float[] {0, 1, 0}, MLX.astype(MLXOps.equal(a, b), DType.FLOAT32).toFloatArray(), EPS);
  }
}

@Test
void whereSelectsBetweenTwoArraysByCondition() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray a = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
    MLXArray b = MLX.array(scope, new float[] {2, 2, 2}, new int[] {3});
    MLXArray condition = MLXOps.less(a, b);
    MLXArray ifTrue = MLX.full(scope, new int[] {3}, 100f, DType.FLOAT32);
    MLXArray ifFalse = MLX.full(scope, new int[] {3}, -1f, DType.FLOAT32);
    assertArrayEquals(
        new float[] {100, -1, -1}, MLXOps.where(condition, ifTrue, ifFalse).toFloatArray(), EPS);
  }
}

/** Reuses this plan's own empirically-confirmed SDPA finding: softmax([0,1]) = [0.2689414, 0.7310586]. */
@Test
void softmaxAxisNormalizesAlongTheGivenAxis() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray a = MLX.array(scope, new float[] {0, 1}, new int[] {1, 2});
    MLXArray result = MLXOps.softmaxAxis(a, 1, true);
    assertArrayEquals(new float[] {0.2689414f, 0.7310586f}, result.toFloatArray(), 1e-3f);
  }
}
```

Add the `import se.alipsa.jmlx.core.DType;`-style statics already present in this file as needed
(`DType` is in the same package, no import required; `assertEquals` is already statically imported by
this file's existing `import static org.junit.jupiter.api.Assertions.assertEquals;`).

### Step 1h: verify

Run `./gradlew :jmlx-core:test --tests "se.alipsa.jmlx.core.MLXNumericTest"`. All new and existing
cases pass. Run `./gradlew spotlessApply checkstyleMain checkstyleTest`.

### Step 1i: commit

```bash
git add jmlx-core/src/main/java/se/alipsa/jmlx/core/NativeOps.java \
        jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXShape.java \
        jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXOps.java \
        jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXNumericTest.java
git commit -m "phase4 M3: op-surface additions (concatenate, split, expandDims, flatten, tril/triu, comparisons, where, softmaxAxis)"
```

## Task 2: `MLXFast.rope` and `MLXFast.scaledDotProductAttention` (`se.alipsa.jmlx.core`)

**Depends on:** Task 1 (`NativeOps.optFloat`, `NativeOps.cstr`, `NativeOps.nullableHandle` already
exists from M1).

**Files:**
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXFast.java`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXFastTest.java` (append new `@Test` methods)

**Interfaces produced (for Task 4):**
- `MLXFast.rope(MLXArray x, int dims, boolean traditional, Float base, float scale, int offset,
  MLXArray freqs)`.
- `MLXFast.scaledDotProductAttention(MLXArray queries, MLXArray keys, MLXArray values, float scale,
  boolean causal, MLXArray maskArr, MLXArray sinks)`.

### Step 2a: `rope`

```java
/**
 * Rotary position embedding ({@code mlx_fast_rope}). {@code dims} is the (even) width of the head
 * dimension to rotate -- for {@code traditional=false} (the half-split/NeoX form this facade
 * exposes), pairs are {@code (x_i, x_{i+dims/2})} for {@code i} in {@code [0, dims/2)}, each rotated
 * by {@code angle_i = offset * scale * base^(-2i/dims)} (confirmed empirically against this mlx-c
 * version; req/plans/phase4-m3-plan.md's Findings section). {@code x}'s second-to-last axis is the
 * position axis: {@code offset} is the position of {@code x}'s first row along that axis, so a
 * single call over a {@code [..., T, dims]} tensor rotates row {@code t} by position {@code
 * offset+t} -- this is what lets a KV-cache decode step pass just the new token's own row with
 * {@code offset = cache.offset()} rather than needing per-position calls.
 *
 * <p>{@code base} and {@code freqs} are each a Java {@code null} for "absent" -- but mlx-c requires
 * at least one of the two to be present, and throws a clean {@link MLXException} naming that
 * ("Neither base nor freqs has a value") if both are {@code null}; this method does not duplicate
 * that check.
 */
public static MLXArray rope(
    MLXArray x, int dims, boolean traditional, Float base, float scale, int offset, MLXArray freqs) {
  MLXScope scope = NativeOps.scopeOf("rope", x, freqs);
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment baseStruct = NativeOps.optFloat(tmp, base);
    MemorySegment freqsHandle = NativeOps.nullableHandle(freqs, tmp);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "rope",
        () ->
            mlx_h.mlx_fast_rope(
                res,
                x.handle(),
                dims,
                traditional,
                baseStruct,
                scale,
                offset,
                freqsHandle,
                NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }
}
```

### Step 2b: `scaledDotProductAttention`

```java
private static final MemorySegment MASK_MODE_NONE = NativeOps.cstr("");
private static final MemorySegment MASK_MODE_CAUSAL = NativeOps.cstr("causal");
private static final MemorySegment MASK_MODE_ARRAY = NativeOps.cstr("array");

/**
 * Scaled dot-product attention ({@code mlx_fast_scaled_dot_product_attention}): {@code softmax(scale
 * * queries @ keys^T + mask) @ values}, batched over every leading axis. {@code causal} and {@code
 * maskArr} are mutually exclusive Java-side conveniences over mlx-c's single {@code mask_mode}
 * string -- {@code causal=true} sends {@code "causal"}; a non-null {@code maskArr} sends {@code
 * "array"}; neither sends {@code ""} (unmasked).
 *
 * <p>{@code maskArr}, if given, is either {@code BOOL} ({@code true} = attend, {@code false} = mask
 * out) or an inexact dtype added directly to the scaled scores (a large negative value, e.g. {@code
 * -1e9}, masks a position out) -- both confirmed empirically to reproduce {@code causal}'s own result
 * exactly when built to encode the same constraint (req/plans/phase4-m3-plan.md's Findings section).
 *
 * <p>{@code causal} bottom-aligns when {@code queries} has fewer rows than {@code keys}/{@code
 * values}: row {@code i} of {@code queries} is treated as absolute position {@code
 * keys.shape()[-2] - queries.shape()[-2] + i}, not position {@code i} -- this is what makes a single
 * new token's decode-step attention over a longer KV cache correct without a custom mask (confirmed
 * empirically, same Findings section).
 *
 * @throws IllegalArgumentException if both {@code causal} and a non-null {@code maskArr} are given
 */
public static MLXArray scaledDotProductAttention(
    MLXArray queries,
    MLXArray keys,
    MLXArray values,
    float scale,
    boolean causal,
    MLXArray maskArr,
    MLXArray sinks) {
  if (causal && maskArr != null) {
    throw new IllegalArgumentException(
        "scaledDotProductAttention: causal and an explicit maskArr are mutually exclusive");
  }
  MLXScope scope = NativeOps.scopeOf("scaledDotProductAttention", queries, keys, values, maskArr, sinks);
  MemorySegment maskMode =
      causal ? MASK_MODE_CAUSAL : (maskArr != null ? MASK_MODE_ARRAY : MASK_MODE_NONE);
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment maskHandle = NativeOps.nullableHandle(maskArr, tmp);
    MemorySegment sinksHandle = NativeOps.nullableHandle(sinks, tmp);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked(
        "scaledDotProductAttention",
        () ->
            mlx_h.mlx_fast_scaled_dot_product_attention(
                res,
                queries.handle(),
                keys.handle(),
                values.handle(),
                scale,
                maskMode,
                maskHandle,
                sinksHandle,
                NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }
}
```

Add imports to `MLXFast.java`: none beyond what M1 already added (`java.lang.foreign.Arena`,
`MemorySegment` are already imported; `mlx_h` and `MLXScope` too).

### Step 2c: tests (append to `MLXFastTest.java`)

```java
@Test
void ropeRotatesHalfSplitPairsByPositionDependentAngle() {
  try (MLXScope scope = new MLXScope()) {
    // dims=4: pair (x0,x2)=(1,1) rotates by angle_0 = 1*1.0*10000^0 = 1 rad; pair (x1,x3)=(0,0)
    // is fixed by any rotation. Confirmed empirically (this plan's Findings section).
    MLXArray x = MLX.array(scope, new float[] {1, 0, 1, 0}, new int[] {1, 1, 4});
    MLXArray result = MLXFast.rope(x, 4, false, 10000f, 1.0f, 1, null);
    assertArrayEquals(new float[] {-0.30116868f, 0f, 1.3817731f, 0f}, result.toFloatArray(), EPS);
  }
}

@Test
void ropeWithOffsetZeroIsTheIdentity() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray x = MLX.array(scope, new float[] {1, 0, 1, 0}, new int[] {1, 1, 4});
    MLXArray result = MLXFast.rope(x, 4, false, 10000f, 1.0f, 0, null);
    assertArrayEquals(new float[] {1, 0, 1, 0}, result.toFloatArray(), EPS);
  }
}

/**
 * Exercises BOTH frequency components (freq_0=1, freq_1=base^-0.5=0.01) with every input element
 * nonzero, hand-derived from the confirmed formula: angle_0=1 rad (cos=0.5403023, sin=0.8414710),
 * angle_1=0.01 rad (cos=0.9999500, sin=0.0099998).
 */
@Test
void ropeRotatesBothFrequencyComponentsWhenEveryElementIsNonzero() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {1, 1, 4});
    MLXArray result = MLXFast.rope(x, 4, false, 10000f, 1.0f, 1, null);
    assertArrayEquals(
        new float[] {-1.9841106f, 1.9599007f, 2.4623779f, 4.0197997f}, result.toFloatArray(), EPS);
  }
}

@Test
void ropeWithNeitherBaseNorFreqsThrowsAnMlxException() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray x = MLX.array(scope, new float[] {1, 0, 1, 0}, new int[] {1, 1, 4});
    assertThrows(MLXException.class, () -> MLXFast.rope(x, 4, false, null, 1.0f, 1, null));
  }
}

@Test
void scaledDotProductAttentionCausalMatchesHandComputedGolden() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray q = MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray k = MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray v = MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray result = MLXFast.scaledDotProductAttention(q, k, v, 1.0f, true, null, null);
    assertArrayEquals(new float[] {1f, 0f, 0.2689414f, 0.7310586f}, result.toFloatArray(), 1e-3f);
  }
}

@Test
void scaledDotProductAttentionArrayModeAdditiveMaskMatchesCausal() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray q = MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray k = MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray v = MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray additiveMask = MLX.array(scope, new float[] {0f, -1e9f, 0f, 0f}, new int[] {1, 1, 2, 2});
    MLXArray result = MLXFast.scaledDotProductAttention(q, k, v, 1.0f, false, additiveMask, null);
    assertArrayEquals(new float[] {1f, 0f, 0.2689414f, 0.7310586f}, result.toFloatArray(), 1e-3f);
  }
}

@Test
void scaledDotProductAttentionArrayModeBooleanMaskTrueMeansAttend() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray q = MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray k = MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray v = MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray rowIdx = MLX.array(scope, new float[] {0, 0, 1, 1}, new int[] {1, 1, 2, 2});
    MLXArray colIdx = MLX.array(scope, new float[] {0, 1, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray boolMask = MLXOps.greater(colIdx, rowIdx); // true exactly at the strictly-upper cell
    MLXArray result = MLXFast.scaledDotProductAttention(q, k, v, 1.0f, false, boolMask, null);
    assertArrayEquals(new float[] {0f, 1f, 0.5f, 0.5f}, result.toFloatArray(), 1e-3f);
  }
}

@Test
void scaledDotProductAttentionCausalBottomAlignsAShorterQuery() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray q = MLX.array(scope, new float[] {1, 1}, new int[] {1, 1, 1, 2});
    MLXArray k = MLX.array(scope, new float[] {1, 0, 0, 1, 1, 1}, new int[] {1, 1, 3, 2});
    MLXArray v = MLX.array(scope, new float[] {1, 0, 0, 1, 1, 1}, new int[] {1, 1, 3, 2});
    MLXArray result = MLXFast.scaledDotProductAttention(q, k, v, 1.0f, true, null, null);
    assertArrayEquals(new float[] {0.78805846f, 0.78805846f}, result.toFloatArray(), 1e-3f);
  }
}

@Test
void scaledDotProductAttentionRejectsCausalTogetherWithAnExplicitMask() {
  try (MLXScope scope = new MLXScope()) {
    MLXArray q = MLX.array(scope, new float[] {1, 0, 0, 1}, new int[] {1, 1, 2, 2});
    MLXArray k = q;
    MLXArray v = q;
    MLXArray mask = MLX.zeros(scope, new int[] {1, 1, 2, 2}, DType.BOOL);
    assertThrows(
        IllegalArgumentException.class,
        () -> MLXFast.scaledDotProductAttention(q, k, v, 1.0f, true, mask, null));
  }
}
```

`assertThrows` needs `import static org.junit.jupiter.api.Assertions.assertThrows;` added to
`MLXFastTest.java` (not yet imported there — only `assertArrayEquals`/`assertSame` are).

### Step 2d: verify

`./gradlew :jmlx-core:test --tests "se.alipsa.jmlx.core.MLXFastTest"`, then
`./gradlew spotlessApply checkstyleMain checkstyleTest`.

### Step 2e: commit

```bash
git add jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXFast.java \
        jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXFastTest.java
git commit -m "phase4 M3: MLXFast.rope and scaledDotProductAttention"
```

## Task 3: `KVCache` (`se.alipsa.jmlx.nn`)

**Depends on:** Task 1 (`MLXShape.concatenate`), M0a's `MLX.hoist`.

**Files:**
- Create: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/KVCache.java`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/KVCacheTest.java`

**Interfaces produced (for Task 4):**
- `new KVCache(MLXScope scope)`, `cache.offset()`, `cache.keys()`, `cache.values()`,
  `cache.append(MLXArray k, MLXArray v)`.

**Design.** `KVCache` is **not** a `Module` — the accumulated keys/values are not trainable
parameters, they are activations that change identity every decode step, and `Module.rebind`'s
value-swap contract has no meaning for them. It owns a long-lived `MLXScope` (supplied by the caller)
that **must be an ancestor of every scope `k`/`v` are allocated in** when passed to `append` — the same
requirement `MLX.hoist` already enforces and reports clearly (`IllegalArgumentException: hoist: target
must be a's own scope or an ancestor of it`), so this class adds no separate check for it.

**The memory-growth hazard this class exists to avoid, stated explicitly (extends req/phase4-plan.md
§2's Research findings to a new case).** Each `append` must replace the accumulated `keys`/`values`
with a freshly concatenated array — but if the *previous* generation's handle is never freed, active
memory grows by one full copy of "everything appended so far" **every** step: after `N` appends, the
still-alive handles alone would sum to `keys_1 + keys_2 + ... + keys_N` element counts, `O(N^2)` in
total appended length, even though only the final `keys_N` is ever read. This is exactly the shape
Context ¶2 opens `req/phase4-plan.md` with ("a 1000-token decode loop... retains ~300k arrays"), one
level more concrete.

**The fix is safe by the same proof `hoist` already relies on.** `req/phase4-plan.md` §2's Research
findings establish that an `ArrayDesc` owns its graph inputs *by value*, and each wrapper is a
`shared_ptr<ArrayDesc>` — so after `concatenate([oldKeys, k], axis)` builds a new array whose graph
holds its own copy of `oldKeys`'s wrapper (sharing the same `ArrayDesc`, refcount now 2), **closing the
Java-side `oldKeys` handle only decrements that refcount to 1** — it cannot touch the descriptor the
new concatenated (and now-hoisted) array's graph still references. `append` therefore explicitly closes
the previous `keys`/`values` handles immediately after hoisting their replacements, every step:

```java
package se.alipsa.jmlx.nn;

import java.util.Objects;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Accumulates the key/value tensors of a multi-head-attention layer across decode steps. Not a
 * {@link Module}: the accumulated tensors are activations, not trainable parameters -- {@code
 * Module.rebind}'s value-swap contract has no meaning for them.
 *
 * <p>{@link #append}'s {@code k}/{@code v} may live in any scope that is this cache's own {@code
 * scope} or a <em>descendant</em> of it (a per-step child scope, in the normal decode-loop shape) --
 * {@link MLX#hoist} enforces this and reports a clear {@link IllegalArgumentException} if violated,
 * so this class adds no separate check.
 *
 * <p>{@code append} explicitly closes the superseded {@code keys}/{@code values} handle after
 * hoisting each replacement -- safe because an mlx {@code ArrayDesc} owns its graph inputs by value
 * (req/phase4-plan.md §2, Research findings): the freshly concatenated array's graph holds its own
 * copy of the superseded array's wrapper, sharing the same descriptor, so closing this class's own
 * handle only decrements a refcount it does not solely own. Without this, active memory would grow by
 * one full copy of "everything appended so far" every step -- {@code O(N^2)} in total appended length
 * after {@code N} steps, not {@code O(N)} -- exactly the shape req/phase4-plan.md's Context section
 * opens with.
 */
public final class KVCache {

  private final MLXScope scope;
  private MLXArray keys;
  private MLXArray values;
  private int offset;

  /** Creates an empty cache whose accumulated tensors live in {@code scope}. */
  public KVCache(MLXScope scope) {
    this.scope = Objects.requireNonNull(scope, "KVCache: scope must not be null");
  }

  /** The number of positions already accumulated -- the position of the next appended row. */
  public int offset() {
    return offset;
  }

  /** The accumulated keys, shape {@code [..., offset(), headDim]}, or {@code null} before the first
   * {@link #append}. */
  public MLXArray keys() {
    return keys;
  }

  /** The accumulated values, shape {@code [..., offset(), headDim]}, or {@code null} before the
   * first {@link #append}. */
  public MLXArray values() {
    return values;
  }

  /**
   * Appends {@code k}/{@code v} (shape {@code [..., T, headDim]}, second-to-last axis the sequence
   * position -- matching {@code MLXFast.rope}'s own position-axis convention) and advances {@link
   * #offset()} by their sequence length. On the first call, hoists {@code k}/{@code v} directly
   * (nothing to concatenate against); on every later call, concatenates against the existing
   * accumulated tensor, hoists the result into this cache's own scope, then closes the superseded
   * handle -- see this class's javadoc for why that close is both necessary and safe.
   *
   * @throws NullPointerException if {@code k} or {@code v} is {@code null}
   * @throws IllegalArgumentException if {@code k}/{@code v}'s scope is neither this cache's own
   *     scope nor a descendant of it (via {@link MLX#hoist}), or (from {@link
   *     MLXShape#concatenate}) if their shape disagrees with the existing accumulated tensor on any
   *     axis but the sequence axis
   */
  public void append(MLXArray k, MLXArray v) {
    Objects.requireNonNull(k, "KVCache.append: k must not be null");
    Objects.requireNonNull(v, "KVCache.append: v must not be null");
    int seqAxis = k.ndim() - 2;
    int newLength = k.shape()[seqAxis];
    if (keys == null) {
      keys = MLX.hoist(k, scope);
      values = MLX.hoist(v, scope);
    } else {
      MLXArray concatenatedKeys = MLXShape.concatenate(new MLXArray[] {keys, k}, seqAxis);
      MLXArray concatenatedValues = MLXShape.concatenate(new MLXArray[] {values, v}, seqAxis);
      MLXArray hoistedKeys = MLX.hoist(concatenatedKeys, scope);
      MLXArray hoistedValues = MLX.hoist(concatenatedValues, scope);
      keys.close();
      values.close();
      keys = hoistedKeys;
      values = hoistedValues;
    }
    offset += newLength;
  }
}
```

**Test file** (`KVCacheTest.java`):

```java
package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.NativeMemoryProbe;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class KVCacheTest {

  private static final float EPS = 1e-5f;

  @Test
  void firstAppendHoistsDirectlyAndAdvancesOffset() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope);
      MLXArray k = MLX.array(scope, new float[] {1, 2}, new int[] {1, 1, 1, 2});
      MLXArray v = MLX.array(scope, new float[] {3, 4}, new int[] {1, 1, 1, 2});
      cache.append(k, v);
      assertEquals(1, cache.offset());
      assertArrayEquals(new float[] {1, 2}, cache.keys().toFloatArray(), EPS);
      assertArrayEquals(new float[] {3, 4}, cache.values().toFloatArray(), EPS);
    }
  }

  @Test
  void secondAppendConcatenatesAlongTheSequenceAxisAndAdvancesOffset() {
    try (MLXScope scope = new MLXScope()) {
      KVCache cache = new KVCache(scope);
      cache.append(
          MLX.array(scope, new float[] {1, 2}, new int[] {1, 1, 1, 2}),
          MLX.array(scope, new float[] {3, 4}, new int[] {1, 1, 1, 2}));
      cache.append(
          MLX.array(scope, new float[] {5, 6}, new int[] {1, 1, 1, 2}),
          MLX.array(scope, new float[] {7, 8}, new int[] {1, 1, 1, 2}));
      assertEquals(2, cache.offset());
      assertArrayEquals(new int[] {1, 1, 2, 2}, cache.keys().shape());
      assertArrayEquals(new float[] {1, 2, 5, 6}, cache.keys().toFloatArray(), EPS);
      assertArrayEquals(new float[] {3, 4, 7, 8}, cache.values().toFloatArray(), EPS);
    }
  }

  /**
   * A step scope closing after {@code append} must not invalidate the cache's own copy -- proves the
   * hoist-then-close discipline in {@link KVCache#append}'s javadoc actually decouples the two.
   */
  @Test
  void cacheSurvivesTheStepScopeClosingAfterAppend() {
    try (MLXScope modelScope = new MLXScope()) {
      KVCache cache = new KVCache(modelScope);
      try (MLXScope step = modelScope.newChild()) {
        cache.append(
            MLX.array(step, new float[] {1, 2}, new int[] {1, 1, 1, 2}),
            MLX.array(step, new float[] {3, 4}, new int[] {1, 1, 1, 2}));
      }
      try (MLXScope step2 = modelScope.newChild()) {
        cache.append(
            MLX.array(step2, new float[] {5, 6}, new int[] {1, 1, 1, 2}),
            MLX.array(step2, new float[] {7, 8}, new int[] {1, 1, 1, 2}));
      }
      assertArrayEquals(new float[] {1, 2, 5, 6}, cache.keys().toFloatArray(), EPS);
    }
  }

  @Test
  void appendFromAnUnrelatedScopeThrows() {
    try (MLXScope cacheScope = new MLXScope();
        MLXScope unrelated = new MLXScope()) {
      KVCache cache = new KVCache(cacheScope);
      MLXArray k = MLX.array(unrelated, new float[] {1, 2}, new int[] {1, 1, 1, 2});
      MLXArray v = MLX.array(unrelated, new float[] {3, 4}, new int[] {1, 1, 1, 2});
      assertThrows(IllegalArgumentException.class, () -> cache.append(k, v));
    }
  }

  /**
   * The memory-growth hazard {@link KVCache}'s own javadoc names: without the {@code close()} calls
   * in {@code append}, active memory after N appends would sum every superseded generation --
   * roughly {@code N/2} times larger than the correctly-freed figure. A generous-but-discriminating
   * multiple over the correct (linear) figure catches that shape without tripping on ordinary
   * allocator overhead.
   */
  @Test
  void activeMemoryGrowsLinearlyNotQuadraticallyAcrossManyAppends() {
    int elementsPerToken = 50_000; // ~200 KB per tensor per token (float32)
    int appends = 30;
    try (MLXScope modelScope = new MLXScope()) {
      KVCache cache = new KVCache(modelScope);
      long baseline = NativeMemoryProbe.activeMemoryBytes();
      for (int i = 0; i < appends; i++) {
        try (MLXScope step = modelScope.newChild()) {
          MLXArray k = MLX.array(step, new float[elementsPerToken], new int[] {1, 1, 1, elementsPerToken});
          MLXArray v = MLX.array(step, new float[elementsPerToken], new int[] {1, 1, 1, elementsPerToken});
          cache.append(k, v);
          MLX.eval(cache.keys(), cache.values());
        }
      }
      long after = NativeMemoryProbe.activeMemoryBytes();
      long grew = after - baseline;
      long expectedLinear = 2L * appends * elementsPerToken * 4;
      assertTrue(
          grew <= expectedLinear * 4,
          "active memory grew by "
              + grew
              + " bytes over "
              + appends
              + " appends (expected roughly "
              + expectedLinear
              + " bytes for correctly-freed superseded generations -- this suggests"
              + " KVCache.append is not closing the previous keys/values handle after hoisting)");
    }
  }
}
```

### Step 3a-3d: write test first, run to see it fail on a missing class, implement, run to see it
pass, `spotlessApply`, commit.

```bash
git add jmlx-core/src/main/java/se/alipsa/jmlx/nn/KVCache.java \
        jmlx-core/src/test/java/se/alipsa/jmlx/nn/KVCacheTest.java
git commit -m "phase4 M3: KVCache"
```

## Task 4: `MultiHeadAttention` (`se.alipsa.jmlx.nn`)

**Depends on:** Tasks 1-3.

**Files:**
- Create: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/MultiHeadAttention.java`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/MultiHeadAttentionTest.java`

**Design.** A fused-QKV-projection layer, matching the "split (fused QKV projection)" op this merge
point's own spec §7 names. `qkvWeight` has shape `[3*embedDim, embedDim]`; `outWeight` has shape
`[embedDim, embedDim]`. `embedDim` is derived from `outWeight`, matching `Linear`'s own
checkpoint-layout convention (never transposed before registration). Does **not** implement {@code
UnaryModule} — its `forward` takes a cache and a causal flag in addition to `x`, and `UnaryModule`'s
own javadoc names exactly this class as the reason that interface exists separately from `Module`.

**RoPE parameters are fixed** (`traditional=false`, `base=10000f`) rather than constructor arguments —
YAGNI: nothing in this plan needs a caller-configurable base or the interleaved form, and both can be
added later without breaking this constructor's signature (an overload, not a change).

```java
package se.alipsa.jmlx.nn;

import java.util.Arrays;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Multi-head self-attention with a fused QKV projection and optional KV caching: {@code x -> qkv ->
 * split -> per-head RoPE -> (optional cache append) -> causal-or-not SDPA -> merge heads -> output
 * projection}. See req/phase4-plan.md §7.
 *
 * <p>Not a {@link UnaryModule}: {@link #forward} takes a {@link KVCache} and a causal flag in
 * addition to {@code x} -- {@code UnaryModule}'s own javadoc names this class as the reason that
 * interface is not folded into {@link Module} itself.
 */
public final class MultiHeadAttention extends Module {

  private static final float ROPE_BASE = 10000f;

  private final int numHeads;
  private final int headDim;
  private final float scale;
  private final Linear qkvProj;
  private final Linear outProj;

  /**
   * Creates a layer with {@code numHeads} heads. {@code qkvWeight} is {@code [3*embedDim,
   * embedDim]} (the checkpoint layout for a fused QKV projection); {@code outWeight} is {@code
   * [embedDim, embedDim]}. {@code embedDim} is derived from {@code outWeight} and must be evenly
   * divisible by {@code numHeads}.
   */
  public MultiHeadAttention(
      MLXScope scope,
      int numHeads,
      MLXArray qkvWeight,
      MLXArray qkvBias,
      MLXArray outWeight,
      MLXArray outBias) {
    super(scope);
    if (outWeight.ndim() != 2 || outWeight.shape()[0] != outWeight.shape()[1]) {
      throw new IllegalArgumentException(
          "MultiHeadAttention: outWeight must be square [embedDim, embedDim], got shape "
              + Arrays.toString(outWeight.shape()));
    }
    int embedDim = outWeight.shape()[0];
    if (qkvWeight.ndim() != 2
        || qkvWeight.shape()[0] != 3 * embedDim
        || qkvWeight.shape()[1] != embedDim) {
      throw new IllegalArgumentException(
          "MultiHeadAttention: qkvWeight must be [3*embedDim, embedDim] = ["
              + (3 * embedDim)
              + ", "
              + embedDim
              + "], got shape "
              + Arrays.toString(qkvWeight.shape()));
    }
    if (numHeads <= 0 || embedDim % numHeads != 0) {
      throw new IllegalArgumentException(
          "MultiHeadAttention: numHeads (" + numHeads + ") must evenly divide embedDim (" + embedDim + ")");
    }
    this.numHeads = numHeads;
    this.headDim = embedDim / numHeads;
    this.scale = (float) (1.0 / Math.sqrt(headDim));
    qkvProj = child("qkvProj", new Linear(scope, qkvWeight, qkvBias));
    outProj = child("outProj", new Linear(scope, outWeight, outBias));
  }

  /**
   * Computes this layer's output for {@code x} (shape {@code [batch, seq, embedDim]}). If {@code
   * cache} is non-null, its accumulated keys/values (from earlier calls) are attended over too, RoPE
   * positions start at {@code cache.offset()}, and {@code cache} is advanced by this call's {@code
   * seq} positions -- {@code cache} may be {@code null} for a one-shot forward pass with no
   * carried-over state (RoPE positions then start at {@code 0}). {@code causal}, together with
   * {@code cache}, is what makes both a full-sequence prefill and a single-token decode step correct
   * with the same flag: mlx's causal mask bottom-aligns a shorter query against a longer key/value
   * (see {@link MLXFast#scaledDotProductAttention}'s javadoc).
   *
   * <p>{@code cache}'s own scope must be this layer's {@code scope()} or an ancestor of it -- see
   * {@link KVCache}'s javadoc; violating this throws from inside {@link KVCache#append}.
   */
  public MLXArray forward(MLXArray x, KVCache cache, boolean causal) {
    int[] xShape = x.shape();
    int batch = xShape[0];
    int seq = xShape[1];
    int offset = cache != null ? cache.offset() : 0;

    MLXArray qkv = qkvProj.forward(x); // [batch, seq, 3*embedDim]
    MLXArray[] parts = MLXShape.split(qkv, 3, 2);
    MLXArray q = toHeads(parts[0], batch, seq);
    MLXArray k = toHeads(parts[1], batch, seq);
    MLXArray v = toHeads(parts[2], batch, seq);

    q = MLXFast.rope(q, headDim, false, ROPE_BASE, 1.0f, offset, null);
    k = MLXFast.rope(k, headDim, false, ROPE_BASE, 1.0f, offset, null);

    if (cache != null) {
      cache.append(k, v);
      k = cache.keys();
      v = cache.values();
    }

    MLXArray attn = MLXFast.scaledDotProductAttention(q, k, v, scale, causal, null, null);
    MLXArray merged = MLXShape.flatten(MLXShape.transpose(attn, new int[] {0, 2, 1, 3}), 2, 3);
    return outProj.forward(merged);
  }

  /** {@code [batch, seq, embedDim] -> [batch, seq, numHeads, headDim] -> [batch, numHeads, seq,
   * headDim]}. */
  private MLXArray toHeads(MLXArray part, int batch, int seq) {
    MLXArray reshaped = MLXShape.reshape(part, new int[] {batch, seq, numHeads, headDim});
    return MLXShape.transpose(reshaped, new int[] {0, 2, 1, 3});
  }
}
```

**Test file** (`MultiHeadAttentionTest.java`) — uses the spec's own named fixture size (1 batch, 2
heads, headDim 4, 3 tokens, so `embedDim=8`) and Decision 8's composition-identity fallback: `qkvWeight`
is three stacked `embedDim`x`embedDim` identity blocks (so `q=k=v=x` exactly) and `outWeight` is a plain
identity (so the layer's output *is* the merged attention output), which keeps the fixture's weights
trivial while the test's **reference** is computed independently -- per-position single-length `rope`
calls (instead of trusting the layer's one multi-position call) and manual `matmul`+`triu`+`where`+
`softmaxAxis` causal masking (instead of the built-in `causal` flag) -- so the test genuinely
cross-checks two different code paths rather than re-running the layer's own arithmetic:

```java
package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class MultiHeadAttentionTest {

  private static final float EPS = 1e-3f;
  private static final int BATCH = 1;
  private static final int HEADS = 2;
  private static final int HEAD_DIM = 4;
  private static final int SEQ = 3;
  private static final int EMBED_DIM = HEADS * HEAD_DIM; // 8

  private static final float[] X_DATA = {
    1, 2, 3, 4, 5, 6, 7, 8,
    8, 7, 6, 5, 4, 3, 2, 1,
    2, 4, 6, 8, 1, 3, 5, 7
  };

  private static MLXArray stackedIdentityQkvWeight(MLXScope scope) {
    float[] data = new float[3 * EMBED_DIM * EMBED_DIM];
    for (int block = 0; block < 3; block++) {
      for (int i = 0; i < EMBED_DIM; i++) {
        data[(block * EMBED_DIM + i) * EMBED_DIM + i] = 1f;
      }
    }
    return MLX.array(scope, data, new int[] {3 * EMBED_DIM, EMBED_DIM});
  }

  private static MLXArray identityWeight(MLXScope scope) {
    float[] data = new float[EMBED_DIM * EMBED_DIM];
    for (int i = 0; i < EMBED_DIM; i++) {
      data[i * EMBED_DIM + i] = 1f;
    }
    return MLX.array(scope, data, new int[] {EMBED_DIM, EMBED_DIM});
  }
```

> **Amended after execution.** The `headOut` line below (`MLXOps.matmul(weights, ropedHead)`) is
> wrong: it reuses `ropedHead` (RoPE'd) as the V operand, but RoPE must never be applied to V --
> `MultiHeadAttention.forward` correctly never RoPEs `v`. Found during Task 4 by independent
> cross-check, fixed in the shipped test to `MLXOps.matmul(weights, head)` (the raw, un-roped
> slice). Left here unmodified as a record of what shipped originally; do not copy this method.

```java
  /** Per-head causal attention computed independently of {@link MultiHeadAttention#forward}: a
   * separate per-position rope call per head/position, and a manual matmul+triu+where+softmaxAxis
   * causal mask instead of the built-in {@code causal} flag -- Decision 8's composition-identity
   * fallback, since a literal hand-derived golden for a full multi-head attention pass is
   * impractical. */
  private static float[] composedReference(MLXScope scope, MLXArray x) {
    float[] expected = new float[BATCH * SEQ * EMBED_DIM];
    for (int h = 0; h < HEADS; h++) {
      MLXArray head =
          MLXShape.slice(
              x, new int[] {0, 0, h * HEAD_DIM}, new int[] {BATCH, SEQ, (h + 1) * HEAD_DIM});
      MLXArray[] roped = new MLXArray[SEQ];
      for (int t = 0; t < SEQ; t++) {
        MLXArray pos =
            MLXShape.slice(head, new int[] {0, t, 0}, new int[] {BATCH, t + 1, HEAD_DIM});
        roped[t] = MLXFast.rope(pos, HEAD_DIM, false, 10000f, 1.0f, t, null);
      }
      MLXArray ropedHead = MLXShape.concatenate(roped, 1); // [BATCH, SEQ, HEAD_DIM]
      MLXArray scoresRaw =
          MLXOps.matmul(ropedHead, MLXShape.transpose(ropedHead, new int[] {0, 2, 1}));
      MLXArray scaled =
          MLXOps.multiply(
              scoresRaw,
              MLX.full(scope, new int[0], 1f / (float) Math.sqrt(HEAD_DIM), DType.FLOAT32));
      MLXArray maskBool = MLXShape.triu(MLX.ones(scope, new int[] {BATCH, SEQ, SEQ}, DType.BOOL), 1);
      MLXArray negInf = MLX.full(scope, new int[] {BATCH, SEQ, SEQ}, -1e9f, DType.FLOAT32);
      MLXArray zero = MLX.full(scope, new int[] {BATCH, SEQ, SEQ}, 0f, DType.FLOAT32);
      MLXArray additiveMask = MLXOps.where(maskBool, negInf, zero);
      MLXArray weights = MLXOps.softmaxAxis(MLXOps.add(scaled, additiveMask), 2, true);
      MLXArray headOut = MLXOps.matmul(weights, ropedHead); // [BATCH, SEQ, HEAD_DIM]
      float[] headOutData = headOut.toFloatArray();
      for (int t = 0; t < SEQ; t++) {
        for (int d = 0; d < HEAD_DIM; d++) {
          expected[t * EMBED_DIM + h * HEAD_DIM + d] = headOutData[t * HEAD_DIM + d];
        }
      }
    }
    return expected;
  }

  @Test
  void forwardWithoutCacheMatchesPerHeadComposedReferenceWithCausalMasking() {
    try (MLXScope scope = new MLXScope()) {
      MultiHeadAttention mha =
          new MultiHeadAttention(
              scope, HEADS, stackedIdentityQkvWeight(scope), null, identityWeight(scope), null);
      MLXArray x = MLX.array(scope, X_DATA, new int[] {BATCH, SEQ, EMBED_DIM});

      float[] expected = composedReference(scope, x);
      MLXArray actual = mha.forward(x, null, true);

      assertArrayEquals(expected, actual.toFloatArray(), EPS);
    }
  }

  /**
   * Decodes the same three tokens one at a time through a shared {@link KVCache} and asserts the
   * newest token's output matches the equivalent row of a fresh full-sequence prefill -- the
   * defining correctness property of KV caching. Each step's activations live in their own child
   * scope of the cache's scope, matching the real decode-loop shape {@link KVCache}'s javadoc
   * describes.
   */
  @Test
  void incrementalDecodeWithKvCacheMatchesFullPrefillForTheNewestPosition() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray qkvWeight = stackedIdentityQkvWeight(scope);
      MLXArray outWeight = identityWeight(scope);

      MultiHeadAttention fullMha = new MultiHeadAttention(scope, HEADS, qkvWeight, null, outWeight, null);
      MLXArray x = MLX.array(scope, X_DATA, new int[] {BATCH, SEQ, EMBED_DIM});
      float[] fullOut = fullMha.forward(x, null, true).toFloatArray();
      float[] lastRowExpected = Arrays.copyOfRange(fullOut, 2 * EMBED_DIM, 3 * EMBED_DIM);

      MultiHeadAttention decodeMha =
          new MultiHeadAttention(scope, HEADS, qkvWeight, null, outWeight, null);
      KVCache cache = new KVCache(scope);
      for (int t = 0; t < 2; t++) {
        try (MLXScope step = scope.newChild()) {
          MLXArray xt =
              MLX.array(
                  step,
                  Arrays.copyOfRange(X_DATA, t * EMBED_DIM, (t + 1) * EMBED_DIM),
                  new int[] {BATCH, 1, EMBED_DIM});
          decodeMha.forward(xt, cache, true);
        }
      }
      float[] lastActual;
      try (MLXScope step = scope.newChild()) {
        MLXArray xLast =
            MLX.array(
                step,
                Arrays.copyOfRange(X_DATA, 2 * EMBED_DIM, 3 * EMBED_DIM),
                new int[] {BATCH, 1, EMBED_DIM});
        lastActual = decodeMha.forward(xLast, cache, true).toFloatArray();
      }

      assertArrayEquals(lastRowExpected, lastActual, EPS);
    }
  }

  @Test
  void constructorRejectsANonSquareOutWeight() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray outWeight = MLX.array(scope, new float[EMBED_DIM * 3], new int[] {EMBED_DIM, 3});
      MLXArray qkvWeight = stackedIdentityQkvWeight(scope);
      assertThrows(
          IllegalArgumentException.class,
          () -> new MultiHeadAttention(scope, HEADS, qkvWeight, null, outWeight, null));
    }
  }

  @Test
  void constructorRejectsANumHeadsThatDoesNotDivideEmbedDim() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray outWeight = identityWeight(scope);
      MLXArray qkvWeight = stackedIdentityQkvWeight(scope);
      assertThrows(
          IllegalArgumentException.class,
          () -> new MultiHeadAttention(scope, 3, qkvWeight, null, outWeight, null));
    }
  }
}
```

### Step 4a-4d: write test first, run to see it fail on a missing class, implement, run to see it
pass, `spotlessApply`, commit.

```bash
git add jmlx-core/src/main/java/se/alipsa/jmlx/nn/MultiHeadAttention.java \
        jmlx-core/src/test/java/se/alipsa/jmlx/nn/MultiHeadAttentionTest.java
git commit -m "phase4 M3: MultiHeadAttention"
```

## Task 5: full verification pass

Mirrors `req/plans/phase4-m2-plan.md`'s own closing task. No new production code — this task only
runs checks and fixes anything they surface.

- [ ] `./gradlew build` — Spotless, Checkstyle, and the forked `loaderGuardTest`, plus every module's
  test suite. Report actual test counts.
- [ ] `./gradlew :jmlx-core:test --tests '*MLXMemoryLeakTest*'` — confirms this plan's new op surface
  (concatenate/split/hoist inside `KVCache`) introduced no regression in the existing leak suite.
- [ ] `./gradlew :jmlx-core:test --tests '*MLXGpuVerificationTest*'` — kernels still dispatch to GPU.
- [ ] `grep -rn 'mlx_array_new(' jmlx-core/src/main` — every call site's allocator is `scopeOf(...)`'s
  result, an explicit `MLXScope` parameter, or a confined `Arena` freed by a local `finally` (spec
  Verification #6). The new `axisOp`/`vectorInOp`/`vectorOutOp` bodies from Task 1 must each fall into
  one of these three; confirm by eye against the diff, not just by running the existing suite (a
  fourth-case regression has no test coverage per spec Verification #6's own scope note).
- [ ] `grep -rn 'mlx_vector_array_new' jmlx-core/src/main` — every call site's allocator confirmed **by
  eye** to be a confined `Arena`, never an `MLXScope` (spec Verification #7). This is the exact
  invariant this plan's own scratch-probe crash (a wrong-type-delete against a `mlx_vector_array`
  allocated through a scope) demonstrated the cost of getting wrong — confirm `vectorInOp`'s
  `mlx_vector_array_new_data(tmp, ...)` and `vectorOutOp`'s `mlx_vector_array_new(tmp)` both use the
  confined `tmp`, never `target` or `scope`.
- [ ] `grep -rn 'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` — zero
  matches (spec Verification #6a).
- [ ] Update the Status table at the top of `req/phase4-plan.md`: mark M3 `Done`, with the commit hash
  and a short findings note (matching M1/M2's own entries) summarizing this plan's empirically-confirmed
  RoPE/SDPA facts, so a later reader does not have to re-derive them.

## Deliberately not covered by this plan

- **M4 (`QuantizedLinear`)** — a separate merge point; `optInt`/`optDtype`/`cstr`'s "mode" string reuse
  belong there, alongside their real consumer.
- **§9 (Documentation)** — covers the whole phase (outline reconciliation, `HelloMLX` demo update) and
  stays `NOT STARTED` until M4 also lands, matching M1/M2's own precedent of not touching it.
- **§10 (CI)** — an unrelated, already-tracked follow-up; not touched here.
- **`mlx_fast_rope_dynamic`** (an array-valued `offset`) — explicitly out of scope per spec §7: "out of
  scope until a compiled decode loop exists". `MultiHeadAttention`'s decode-step usage only ever needs
  a plain `int` offset (`cache.offset()`), which `mlx_fast_rope` already takes.
- **Grouped-query attention / a separate KV head count from query head count** — not named anywhere in
  spec §7; `MultiHeadAttention` uses one head count for all three of q/k/v, matching the outline's
  named layer list.
- **Sliding-window or size-bounded KV cache eviction** — `KVCache` accumulates without bound; nothing
  in this plan's scope needs eviction, and adding it now would be speculative.
