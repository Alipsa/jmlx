# jmlx Phase 4 — M2 implementation plan (`MLXGrad` / `ModuleGrad`)

**Spec:** `req/phase4-plan.md` §6 (`MLXGrad`/`ModuleGrad` design, the exception-safety protocol, the
rebinding mechanism), §3 (the `copyHandlesInto` shape this plan finally builds), Research findings
(closure round-trip, `mlx_vector_array_set_data` vs `_new_data`), Status section (Probe 0b/0f findings
this plan proceeds on). M3 (RoPE/attention/KV cache) and M4 (`QuantizedLinear`) are out of scope.
Where this plan gives an exact signature, that is authoritative; where it defers to the spec's prose,
the spec is authoritative.

Branch: `worktree-phase4-m2`, off `main` at `0855ef5` (PR #7 merged, M1 done). Native bootstrap already
staged in this worktree (`native/install/lib/mlx.metallib` exists) — `@EnabledIfNativeAvailable` tests
run for real.

## Why this plan skips a separate scratch probe for Verification item 0c

The spec's Verification item 0 asks for "one scratch class, compiled *and* run against the real
bindings" before writing §6, to catch defects a header-reading pass cannot ("Java written but never
compiled or run", `phase3-plan.md:838-843`). For M1's two hardest, least-header-derivable questions —
upcall exception-safety threading (0b) and the rebinding mechanism itself (0f) — that scratch-class
step was essential and was done, with findings recorded in `req/phase4-plan.md`'s Status section.

0c (`mlx_value_and_grad` on `sum(x*x)`) is different in kind: every fact it would exercise —
`mlx_closure_new_func`'s upcall shape, `mlx_value_and_grad`'s by-value `fun` parameter, ownership of
the two resulting closures, `mlx_vector_array_set_data` vs `_new_data` — was independently confirmed
by reading the actual generated bindings (`mlx_closure_new_func$fun.java`,
`mlx_closure_value_and_grad_new_func$fun.java`) and the mlx-c source under
`native/scratch/mlx-c/mlx/c/{closure,transforms}.cpp` and `private/closure.h`, not assumed. Task 1
below **is** the compile-and-run step for this mechanism — its test (`sum(x*x)` at `[1,2,3]` →
`[2,4,6]`, the exact 0c fixture) is written and run before any later task depends on `MLXGrad`
compiling correctly, and it is a permanent regression test rather than a deleted scratch file, which
is strictly stronger evidence than the scratch-and-delete precedent 0b/0f used. If Task 1's test fails
on real hardware, this plan is wrong somewhere and must be revised before Task 2 proceeds.

## Global Constraints

1. **Package is forced, not chosen.** `MLXGrad` lives in `se.alipsa.jmlx.core` (needs `MLXArray`'s
   package-private constructor to wrap raw handles arriving from the upcall). `ModuleGrad` lives in
   `se.alipsa.jmlx.nn` (needs `Module`). Never the other direction —
   `grep -rn 'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` must stay
   empty (spec Verification #6a, already relied on by M1).
2. **Every native call goes through `NativeOps.checked(...)`**, matching every existing op body. The
   one exception is inside the upcall body itself (Task 1), which must catch `Throwable` around
   *everything*, including its own `checked(...)` calls — never let one escape past the upcall frame.
3. **`mlx_vector_array_set_data`, never `_new_data`, for the upcall's `res` out-param** — `_new_data`
   returns a struct by value and would leave `*res` untouched (spec Research findings, `vector.h:31`
   vs `:33`).
4. **Nullable-array-by-value / `optInt` / `cstr` machinery from spec §3 is irrelevant here** — no op
   in this plan takes a nullable array or an optional scalar. Do not import `nullableHandle`.
5. **Style:** run `./gradlew spotlessApply` before finishing, then `./gradlew build` (Spotless,
   Checkstyle, full test suite including native tests) must succeed. Report actual test counts from
   the build output.
6. **Testing style, matching every existing test file:** `@EnabledIfNativeAvailable` on the test class;
   `try (MLXScope scope = new MLXScope())`; hand-computed goldens; `assertArrayEquals(expected,
   actual.toFloatArray(), EPS)`; `EPS = 1e-5f` for exact-rational goldens.

## Native surface this plan uses (confirmed against the generated bindings, not assumed)

| Java call | C signature | Notes |
|---|---|---|
| `mlx_h.mlx_closure_new_func(SegmentAllocator, MemorySegment fun)` | `mlx_closure mlx_closure_new_func(int (*fun)(mlx_vector_array*, const mlx_vector_array))` | Returns the closure **by value**; wraps our upcall stub. |
| `mlx_closure_new_func$fun.allocate(Function, Arena)` | — | `Function.apply(MemorySegment res, MemorySegment in)`: `res` is `mlx_vector_array*` (ADDRESS — forward its `.address()`, never read/write it directly as bytes in Java); `in` is `mlx_vector_array` **by value** (a properly-sized, directly readable 8-byte segment). |
| `mlx_h.mlx_value_and_grad(MemorySegment res, MemorySegment fun, MemorySegment argnums, long n)` | `int mlx_value_and_grad(mlx_closure_value_and_grad* res, const mlx_closure fun, const int* argnums, size_t n)` | `fun` is passed **by value** (the same segment `mlx_closure_new_func` returned). `res` must be a pre-allocated null-ctx struct from `mlx_closure_value_and_grad_new(allocator)`, exactly the `mlx_array_new`-before-`mlx_add` idiom already used everywhere in this codebase. Confirmed (`transforms.cpp:99-115`) that this call does **not** touch or take ownership of `fun`'s ctx — it constructs an independent heap object, so the original plain closure needs its own free. |
| `mlx_h.mlx_closure_value_and_grad_apply(MemorySegment res0, MemorySegment res1, MemorySegment cls, MemorySegment input)` | `int mlx_closure_value_and_grad_apply(mlx_vector_array *res_0, mlx_vector_array *res_1, mlx_closure_value_and_grad cls, const mlx_vector_array input)` | Two out-vectors: `res0` = values, `res1` = grads (spec §6). `cls` by value. |
| `mlx_h.mlx_closure_free(MemorySegment)` / `mlx_h.mlx_closure_value_and_grad_free(MemorySegment)` | — | **Both** required in `Fn.close()`, confirmed via `native/scratch/mlx-c/mlx/c/private/closure.h`: each closure kind owns a distinct heap `std::function`; freeing one never frees the other. |
| `mlx_h.mlx_vector_array_set_data(MemorySegment vec, MemorySegment data, long size)` | `int mlx_vector_array_set_data(mlx_vector_array* vec, const mlx_array* data, size_t size)` | `data` is a raw contiguous `mlx_array[]` buffer — exactly `NativeOps.copyHandlesInto`'s return value (Task 1), **not** an already-built `mlx_vector_array`. |
| `mlx_h.mlx_vector_array_size(MemorySegment vec)` / `mlx_h.mlx_vector_array_get(MemorySegment res, MemorySegment vec, long idx)` | both take `vec` **by value** | Used both to unpack `in` inside the upcall and to unpack `res0`/`res1` in `Fn.apply`. |

## Task 1: `NativeOps.copyHandlesInto` + `MLXGrad` (`se.alipsa.jmlx.core`)

**Files:**
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/NativeOps.java` (add `copyHandlesInto`)
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLX.java` (widen `newVectorArray`, refactor onto
  `copyHandlesInto`)
- Create: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXGrad.java`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXGradTest.java`

### Step 1a: extract `copyHandlesInto`

`MLX.newVectorArray` already builds a raw `mlx_array[]` buffer before handing it to
`mlx_vector_array_new_data`. `MLXGrad`'s upcall needs exactly that buffer-building half, but must feed
it to `mlx_vector_array_set_data` instead — a different native call, same buffer shape. Extract the
buffer-building loop into `NativeOps` (package-private, both `MLX` and `MLXGrad` are in this package):

```java
// NativeOps.java, next to nullableHandle
/**
 * Copies {@code handles} into a freshly allocated contiguous {@code mlx_array[]} buffer -- the raw
 * struct-array shape both {@code mlx_vector_array_new_data} ({@link MLX#eval}'s vector construction)
 * and {@code mlx_vector_array_set_data} ({@link MLXGrad}'s upcall, writing an existing vector's ctx)
 * accept as their {@code data} parameter. A byte-for-byte struct copy, not a ctx get/set round-trip, so
 * this stays correct if {@code mlx_array_} ever gains a field. {@code allocator} must be a confined
 * {@link Arena} (never an {@link MLXScope}) for the same reason {@link MLX#eval}'s vector allocator
 * must be: the buffer is not an {@code mlx_array} handle this method's caller will ever pass to
 * {@code mlx_array_free}, so allocating it through a scope would corrupt that scope's handle-tracking
 * invariant.
 */
static MemorySegment copyHandlesInto(MemorySegment[] handles, SegmentAllocator allocator) {
  int n = handles.length;
  MemorySegment buf = mlx_array_.allocateArray(n, allocator);
  long elementSize = mlx_array_.sizeof();
  for (int i = 0; i < n; i++) {
    MemorySegment.copy(handles[i], 0L, buf, i * elementSize, elementSize);
  }
  return buf;
}
```

Then in `MLX.java`, replace the body of `newVectorArray` (keep the method, **remove `private`** — package-private, `MLXGrad` needs to call it too — and update its javadoc's "Factored out of eval" sentence to also name `MLXGrad`):

```java
static MemorySegment newVectorArray(MemorySegment[] handles, Arena allocator) {
  MemorySegment buf = NativeOps.copyHandlesInto(handles, allocator);
  // mlx_vector_array_new_data is statusless and returns a null-ctx struct
  // on failure (vector.cpp:41-54) -- unchanged from before this refactor.
  NativeLoader.clearLastNativeError();
  MemorySegment vec = mlx_h.mlx_vector_array_new_data(allocator, buf, handles.length);
  if (mlx_vector_array_.ctx(vec).address() == 0) {
    throw NativeOps.nativeFailure("mlx_vector_array_new_data");
  }
  return vec;
}
```

No behavior change for `eval` — this is pure extraction. Run `./gradlew :jmlx-core:test --tests
'*MLXEvalTest*' '*MLXMemoryLeakTest*'` after this step alone to confirm.

### Step 1b: `MLXGrad`

Create `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXGrad.java`:

```java
package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.function.Function;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_closure_new_func$fun;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Primitive-only autograd over a flat primal vector: {@code mlx_value_and_grad} wrapped as a Java
 * closure. Deliberately has no {@code Module}-aware overload -- see {@code se.alipsa.jmlx.nn.ModuleGrad}
 * and req/phase4-plan.md §6 for why that lives in {@code nn} instead of here (this class would have to
 * import {@code nn} to add one, inverting the one-way package dependency).
 */
public final class MLXGrad {

  private MLXGrad() {}

  static {
    NativeLoader.ensureLoaded();
  }

  /**
   * Stashes a {@link Throwable} that escaped the upcall body, so {@link Fn#apply} can re-surface the
   * ORIGINAL exception after mlx-c reports the resulting native failure as a generic {@link MLXException}
   * (req/phase4-plan.md §6, the three-step exception-safety protocol). Thread-local, not a per-{@code Fn}
   * field: cleared immediately before every {@code apply}, mirroring {@code NativeOps.checked}'s identical
   * clear-before-call rule for {@code NativeLoader}'s native-error thread-local, for the same reason -- a
   * stale value from a previous failure must never be misattributed to the next one.
   */
  private static final ThreadLocal<Throwable> ESCAPED = new ThreadLocal<>();

  /**
   * Differentiates {@code body} with respect to the primal indices in {@code argnums}. {@code argnums}
   * must be non-empty and strictly increasing; the upper-bound check (every index {@code < primals.length})
   * happens per-{@link Fn#apply} call, once {@code primals.length} is known.
   */
  public static Fn valueAndGrad(Function<MLXArray[], MLXArray[]> body, int[] argnums) {
    validateArgnumsShape(argnums);
    return new Fn(body, argnums);
  }

  private static void validateArgnumsShape(int[] argnums) {
    if (argnums.length == 0) {
      throw new IllegalArgumentException("valueAndGrad: argnums must not be empty");
    }
    for (int i = 0; i < argnums.length; i++) {
      if (argnums[i] < 0) {
        throw new IllegalArgumentException("valueAndGrad: argnums[" + i + "] = " + argnums[i] + " is negative");
      }
      if (i > 0 && argnums[i] <= argnums[i - 1]) {
        throw new IllegalArgumentException(
            "valueAndGrad: argnums must be strictly increasing, got " + Arrays.toString(argnums));
      }
    }
  }

  /**
   * A live {@code mlx_closure_value_and_grad} plus the upcall stub backing it. Reusable across many
   * {@link #apply} calls -- {@code target} is a per-call argument, not bound at construction, because
   * grads must land in the per-iteration step scope (req/phase4-plan.md §6: binding it at construction
   * forces a choice between a closed target by iteration two, or one upcall stub per step).
   */
  public static final class Fn implements AutoCloseable {

    private final Arena arena = Arena.ofConfined();
    private final Function<MLXArray[], MLXArray[]> body;
    private final int[] argnums;
    private final MemorySegment plainClosure;
    private final MemorySegment vgClosure;
    private MLXScope applyTarget;
    private boolean closed;

    private Fn(Function<MLXArray[], MLXArray[]> body, int[] argnums) {
      this.body = body;
      this.argnums = argnums.clone();
      mlx_closure_new_func$fun.Function upcall = this::onClosureInvoked;
      MemorySegment funcPtr = mlx_closure_new_func$fun.allocate(upcall, arena);
      this.plainClosure = mlx_h.mlx_closure_new_func(arena, funcPtr);
      MemorySegment vg = mlx_h.mlx_closure_value_and_grad_new(arena);
      try (Arena tmp = Arena.ofConfined()) {
        MemorySegment nativeArgnums = tmp.allocateFrom(ValueLayout.JAVA_INT, this.argnums);
        NativeOps.checked("valueAndGrad",
            () -> mlx_h.mlx_value_and_grad(vg, plainClosure, nativeArgnums, this.argnums.length));
      }
      this.vgClosure = vg;
    }

    /**
     * Runs the closure, landing traced primals, values and grads in {@code target}. {@code primals}
     * must have at least {@code argnums[argnums.length - 1] + 1} elements.
     */
    public Result apply(MLXScope target, MLXArray[] primals) {
      ensureOpen();
      if (argnums[argnums.length - 1] >= primals.length) {
        throw new IllegalArgumentException("valueAndGrad: argnums " + Arrays.toString(argnums)
            + " out of range for " + primals.length + " primal(s)");
      }
      MemorySegment[] handles = new MemorySegment[primals.length];
      for (int i = 0; i < primals.length; i++) {
        handles[i] = primals[i].handle();
      }
      applyTarget = target;
      ESCAPED.remove();
      try (Arena tmp = Arena.ofConfined()) {
        MemorySegment inputVec = MLX.newVectorArray(handles, tmp);
        MemorySegment res0 = mlx_h.mlx_vector_array_new(tmp);
        MemorySegment res1 = mlx_h.mlx_vector_array_new(tmp);
        try {
          NativeOps.checked("valueAndGrad.apply",
              () -> mlx_h.mlx_closure_value_and_grad_apply(res0, res1, vgClosure, inputVec));
        } catch (MLXException nativeFailure) {
          rethrowEscapedOr(nativeFailure);
          throw nativeFailure;
        } finally {
          mlx_h.mlx_vector_array_free(inputVec);
        }
        try {
          MLXArray[] values = unpackVector(res0, target);
          MLXArray[] grads = unpackVector(res1, target);
          return new Result(values, grads);
        } finally {
          mlx_h.mlx_vector_array_free(res0);
          mlx_h.mlx_vector_array_free(res1);
        }
      } finally {
        applyTarget = null;
      }
    }

    private static void rethrowEscapedOr(MLXException nativeFailure) {
      Throwable escaped = ESCAPED.get();
      ESCAPED.remove();
      if (escaped == null) {
        return;
      }
      escaped.addSuppressed(nativeFailure);
      if (escaped instanceof RuntimeException re) {
        throw re;
      }
      if (escaped instanceof Error e) {
        throw e;
      }
      // body is a Function<MLXArray[], MLXArray[]>, which declares no checked
      // exceptions -- unreachable in practice, kept only because catch
      // (Throwable) below can technically observe one via sneaky-throw.
      throw new MLXException("valueAndGrad: body threw a checked exception", escaped);
    }

    /**
     * The upcall body: {@code int fun(mlx_vector_array* res, const mlx_vector_array in)}. Runs
     * synchronously on the {@link #apply} caller's thread (confirmed: req/phase4-plan.md Probe 0b(c)),
     * so the {@link MLXArray}s it builds are confined to that thread like any other. Never lets a
     * {@link Throwable} escape past this frame -- catches everything, stashes it on {@link #ESCAPED},
     * and returns {@code 1}, a supported non-leaking mlx-c error path (spec Research findings,
     * {@code closure.cpp:44-51}).
     */
    private int onClosureInvoked(MemorySegment res, MemorySegment in) {
      try {
        MLXArray[] primalsIn = unpackVector(in, applyTarget);
        MLXArray[] result = body.apply(primalsIn);
        if (result == null || result.length == 0 || result[0].ndim() != 0) {
          int rank = result == null || result.length == 0 ? -1 : result[0].ndim();
          throw new IllegalArgumentException("valueAndGrad: body's first returned array must be rank-0 "
              + "(a reduced scalar loss), got rank " + rank);
        }
        MemorySegment[] handles = new MemorySegment[result.length];
        for (int i = 0; i < result.length; i++) {
          handles[i] = result[i].handle();
        }
        try (Arena tmp = Arena.ofConfined()) {
          MemorySegment buf = NativeOps.copyHandlesInto(handles, tmp);
          NativeOps.checked("valueAndGrad", () -> mlx_h.mlx_vector_array_set_data(res, buf, handles.length));
        }
        return 0;
      } catch (Throwable t) {
        ESCAPED.set(t);
        return 1;
      }
    }

    private static MLXArray[] unpackVector(MemorySegment vec, MLXScope target) {
      long n = mlx_h.mlx_vector_array_size(vec);
      MLXArray[] out = new MLXArray[(int) n];
      for (int i = 0; i < n; i++) {
        MemorySegment h = mlx_h.mlx_array_new(target);
        final long idx = i;
        NativeOps.checked("valueAndGrad", () -> mlx_h.mlx_vector_array_get(h, vec, idx));
        out[i] = new MLXArray(target, h);
      }
      return out;
    }

    /**
     * Frees both closures, then the arena backing the upcall stub. Order is load-bearing
     * (req/phase4-plan.md §6): both closures' underlying {@code std::function}s hold the raw stub
     * pointer until freed, so closing the arena first would leave a dangling stub inside a live closure.
     */
    @Override
    public void close() {
      if (closed) {
        return;
      }
      closed = true;
      NativeOps.checked("valueAndGrad.close", () -> mlx_h.mlx_closure_value_and_grad_free(vgClosure));
      NativeOps.checked("valueAndGrad.close", () -> mlx_h.mlx_closure_free(plainClosure));
      arena.close();
    }

    private void ensureOpen() {
      if (closed) {
        throw new IllegalStateException("MLXGrad.Fn is closed");
      }
    }
  }

  /** {@code values[0]} is {@code body}'s rank-0 loss; {@code grads[i]} corresponds to {@code argnums[i]}. */
  public record Result(MLXArray[] values, MLXArray[] grads) {}
}
```

### Tests — `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXGradTest.java`

```java
package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import org.junit.jupiter.api.Test;

@EnabledIfNativeAvailable
class MLXGradTest {

  private static final float EPS = 1e-5f;

  @Test
  void gradOfSumOfSquares() {
    try (MLXScope model = new MLXScope()) {
      try (MLXGrad.Fn fn = MLXGrad.valueAndGrad(
          xs -> new MLXArray[] {MLXOps.sum(MLXOps.multiply(xs[0], xs[0]))}, new int[] {0})) {
        try (MLXScope step = model.newChild()) {
          MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {3});
          MLXGrad.Result r = fn.apply(step, new MLXArray[] {x});
          assertEquals(1, r.values().length);
          assertEquals(1, r.grads().length);
          assertArrayEquals(new float[] {14f}, r.values()[0].toFloatArray(), EPS);
          assertArrayEquals(new float[] {2, 4, 6}, r.grads()[0].toFloatArray(), EPS);
        }
      }
    }
  }

  @Test
  void fnReusedAcrossTwoStepScopesLandsGradsInEachOwnScope() {
    try (MLXScope model = new MLXScope();
        MLXGrad.Fn fn = MLXGrad.valueAndGrad(
            xs -> new MLXArray[] {MLXOps.sum(MLXOps.multiply(xs[0], xs[0]))}, new int[] {0})) {
      MLXArray grad1;
      try (MLXScope step1 = model.newChild()) {
        MLXArray x1 = MLX.array(step1, new float[] {1, 2, 3}, new int[] {3});
        grad1 = fn.apply(step1, new MLXArray[] {x1}).grads()[0];
        assertSame(step1, grad1.scope());
      }
      // step1 closed; grad1 is now unusable, but a second apply into a fresh
      // step scope must still work -- proves target moved to apply(), not the constructor.
      try (MLXScope step2 = model.newChild()) {
        MLXArray x2 = MLX.array(step2, new float[] {2, 2, 2}, new int[] {3});
        MLXArray grad2 = fn.apply(step2, new MLXArray[] {x2}).grads()[0];
        assertSame(step2, grad2.scope());
        assertArrayEquals(new float[] {4, 4, 4}, grad2.toFloatArray(), EPS);
      }
    }
  }

  @Test
  void argnumsMustNotBeEmpty() {
    assertThrows(IllegalArgumentException.class,
        () -> MLXGrad.valueAndGrad(xs -> xs, new int[0]));
  }

  @Test
  void argnumsMustBeStrictlyIncreasing() {
    assertThrows(IllegalArgumentException.class,
        () -> MLXGrad.valueAndGrad(xs -> xs, new int[] {1, 1}));
    assertThrows(IllegalArgumentException.class,
        () -> MLXGrad.valueAndGrad(xs -> xs, new int[] {1, 0}));
  }

  @Test
  void argnumsMustBeNonNegative() {
    assertThrows(IllegalArgumentException.class,
        () -> MLXGrad.valueAndGrad(xs -> xs, new int[] {-1}));
  }

  @Test
  void argnumsOutOfRangeForPrimalsThrows() {
    try (MLXScope scope = new MLXScope();
        MLXGrad.Fn fn = MLXGrad.valueAndGrad(
            xs -> new MLXArray[] {MLXOps.sum(xs[0])}, new int[] {1})) {
      MLXArray x = MLX.array(scope, new float[] {1}, new int[] {1});
      assertThrows(IllegalArgumentException.class, () -> fn.apply(scope, new MLXArray[] {x}));
    }
  }

  @Test
  void nonRankZeroLossThrowsNamingTheRank() {
    try (MLXScope scope = new MLXScope();
        MLXGrad.Fn fn = MLXGrad.valueAndGrad(xs -> new MLXArray[] {xs[0]}, new int[] {0})) {
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      IllegalArgumentException ex =
          assertThrows(IllegalArgumentException.class, () -> fn.apply(scope, new MLXArray[] {x}));
      assertTrue(ex.getMessage().contains("rank 1"), ex.getMessage());
    }
  }

  private static final class BodyBoom extends RuntimeException {
    BodyBoom(String message) {
      super(message);
    }
  }

  @Test
  void exceptionThrownInsideBodySurfacesAsThatExceptionJvmAlive() {
    try (MLXScope scope = new MLXScope();
        MLXGrad.Fn fn = MLXGrad.valueAndGrad(xs -> {
          throw new BodyBoom("boom");
        }, new int[] {0})) {
      MLXArray x = MLX.array(scope, new float[] {1}, new int[] {1});
      // Asserts the exception TYPE, not just "throws something" -- without
      // step 3 of the exception-safety protocol, a generic MLXException is
      // thrown too, which a type-blind assertThrows(RuntimeException.class)
      // would not distinguish from the broken version.
      assertThrows(BodyBoom.class, () -> fn.apply(scope, new MLXArray[] {x}));
    }
    // JVM survives to reach this line and every later test in the suite --
    // that is the assertion this test exists to make, not any single line in it.
  }

  @Test
  void hoistOutOfLossStillCloses() {
    // Regression guard for the Fn.close() free-ordering rule: exercised
    // implicitly by every other test's try-with-resources on Fn, but this
    // test's sole purpose is a double-close (idempotency), matching MLXArray/
    // MLXScope's own "safe to call more than once" contract.
    try (MLXScope scope = new MLXScope()) {
      MLXGrad.Fn fn = MLXGrad.valueAndGrad(xs -> new MLXArray[] {MLXOps.sum(xs[0])}, new int[] {0});
      MLXArray x = MLX.array(scope, new float[] {1}, new int[] {1});
      fn.apply(scope, new MLXArray[] {x});
      fn.close();
      fn.close();
      assertNotSame(fn, null);
    }
  }
}
```

Run: `./gradlew :jmlx-core:test --tests '*MLXGradTest*'`. Expected: all pass on real hardware. This is
the compile-and-run gate for the whole closure mechanism — do not proceed to Task 2 until it is green.

## Task 2: `ModuleGrad` (`se.alipsa.jmlx.nn`)

**Files:**
- Create: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/ModuleGrad.java`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/ModuleGradTest.java`

**Interfaces consumed:** `MLXGrad.valueAndGrad(Function<MLXArray[],MLXArray[]>, int[])`,
`MLXGrad.Fn.apply(MLXScope, MLXArray[])`, `MLXGrad.Result(MLXArray[] values, MLXArray[] grads)`
(Task 1); `Module.freeze()`, `Module.parameters()`, `Module.rebind(SequencedMap<String,MLXArray>)` (M1,
already merged).

```java
package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.SequencedMap;
import java.util.function.BiFunction;
import java.util.stream.IntStream;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXGrad;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Module-aware autograd: wraps {@link MLXGrad.Fn} and rebinds {@code tree}'s parameters to the traced
 * primals around each {@code loss} call, so the differentiated graph actually runs over the model's own
 * ops rather than over disconnected primal arrays (req/phase4-plan.md §6). Lives here, not in
 * {@code se.alipsa.jmlx.core}, so that package never has to import {@code Module} -- see the class
 * javadoc on {@link MLXGrad}.
 */
public final class ModuleGrad implements AutoCloseable {

  private final Module tree;
  private final BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss;
  private final List<String> paramPaths;
  private final MLXGrad.Fn fn;
  private boolean closed;

  private ModuleGrad(Module tree, BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss) {
    this.tree = tree;
    this.loss = loss;
    tree.freeze();
    this.paramPaths = List.copyOf(tree.parameters().keySet());
    if (paramPaths.isEmpty()) {
      throw new IllegalStateException("ModuleGrad: tree has no parameters to differentiate");
    }
    int[] argnums = IntStream.range(0, paramPaths.size()).toArray();
    this.fn = MLXGrad.valueAndGrad(this::body, argnums);
  }

  /**
   * Freezes {@code tree} and captures {@code tree.parameters().keySet()} -- the ORDER only, not the
   * values (spec §6: re-reading values every {@link #apply} is what keeps grads current after an
   * {@code update}). {@code loss} receives {@code (params, inputs)} and must return a rank-0 loss as
   * element 0 -- see {@link MLXGrad.Fn#apply} for what happens if it does not.
   */
  public static ModuleGrad of(Module tree, BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss) {
    return new ModuleGrad(tree, loss);
  }

  /**
   * The closure body handed to {@link MLXGrad#valueAndGrad}: splits the traced primal vector into
   * params (indices {@code [0, paramPaths.size())}, differentiated) and inputs (the rest, passed
   * through unchanged), rebinds {@code tree} onto the traced params for the duration of {@code loss},
   * and restores the pre-call bindings in a {@code finally} -- including when {@code loss} throws, so a
   * later call outside the traced region never reads into a scope that has since closed.
   */
  private MLXArray[] body(MLXArray[] tracedPrimals) {
    int paramCount = paramPaths.size();
    MLXArray[] tracedParams = Arrays.copyOfRange(tracedPrimals, 0, paramCount);
    MLXArray[] inputs = Arrays.copyOfRange(tracedPrimals, paramCount, tracedPrimals.length);
    SequencedMap<String, MLXArray> saved = tree.parameters();
    SequencedMap<String, MLXArray> traced = new LinkedHashMap<>();
    for (int i = 0; i < paramCount; i++) {
      traced.put(paramPaths.get(i), tracedParams[i]);
    }
    tree.rebind(traced);
    try {
      return loss.apply(tracedParams, inputs);
    } finally {
      tree.rebind(saved);
    }
  }

  /**
   * Per-iteration: {@code target} is this step's scope (grads and the returned loss value land there),
   * {@code inputs} is this batch. Re-reads {@code tree.parameters()}' current VALUES on every call (spec
   * §6: snapshotting them once in {@link #of} would differentiate against stale weights after the first
   * {@code update}) and throws if the key SET has drifted since {@link #of}.
   */
  public Result apply(MLXScope target, MLXArray[] inputs) {
    ensureOpen();
    SequencedMap<String, MLXArray> current = tree.parameters();
    List<String> currentKeys = List.copyOf(current.keySet());
    if (!currentKeys.equals(paramPaths)) {
      throw new IllegalStateException("ModuleGrad: parameter key set changed since of() -- expected "
          + paramPaths + " but tree.parameters() now has " + currentKeys);
    }
    MLXArray[] primals = new MLXArray[paramPaths.size() + inputs.length];
    int i = 0;
    for (String path : paramPaths) {
      primals[i++] = current.get(path);
    }
    for (MLXArray input : inputs) {
      primals[i++] = input;
    }
    MLXGrad.Result r = fn.apply(target, primals);
    SequencedMap<String, MLXArray> grads = new LinkedHashMap<>();
    for (int p = 0; p < paramPaths.size(); p++) {
      grads.put(paramPaths.get(p), r.grads()[p]);
    }
    return new Result(r.values()[0], java.util.Collections.unmodifiableSequencedMap(grads));
  }

  /** {@code value} is the rank-0 loss for this call; {@code grads} is keyed by dotted parameter path. */
  public record Result(MLXArray value, SequencedMap<String, MLXArray> grads) {}

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    fn.close();
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("ModuleGrad is closed");
    }
  }
}
```

**On the negative rebinding-disabled test.** The spec's Testing table asks for grads matching hand-computed
values with rebinding on, and NOT matching with rebinding disabled. That negative was already run as
Probe 0f (req/phase4-plan.md Status section: "Rebinding off ... grads came back as exactly `0.0`/`0.0`")
against a hand-rolled toy model, and its finding is recorded permanently in the spec's prose. This plan
does not re-derive it as a committed test, for the same reason M1 did not re-commit probes 0b/0f
themselves: doing so would require shipping a second, rebinding-disabled code path in production
`ModuleGrad` for the sole purpose of a test disproving it — a deliberately-broken branch with no other
caller, which is its own maintenance hazard. The positive test below (`gradsMatchHandComputedValues`)
is the permanent regression guard; the negative finding stays a documented, already-verified fact.

### Tests — `jmlx-core/src/test/java/se/alipsa/jmlx/nn/ModuleGradTest.java`

Fixture: a single `Linear` layer, `weight = [[1,1,1]]` (shape `[1,3]`), `bias = [0]`, batch `x = [[1,2,3]]`
(shape `[1,3]`), target `= [[0]]`. `forward(x) = x @ weight.T + bias = [[6]]`. `loss = sum(square(forward(x)
- target)) = 36` (rank-0 via `MLXOps.sum`'s no-axes overload). Hand-derived grads:
`d(loss)/d(pred) = 2*(pred-target) = 12`; `d(pred)/d(weight_j) = x_j` so `dweight = 12*[1,2,3] =
[12,24,36]`; `d(pred)/d(bias) = 1` so `dbias = [12]`.

```java
package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import java.util.List;
import java.util.Map;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import org.junit.jupiter.api.Test;

@EnabledIfNativeAvailable
class ModuleGradTest {

  private static final float EPS = 1e-4f;

  private static MLXArray[] mseLoss(MLXArray[] params, MLXArray[] inputs) {
    Linear probe = null; // params[0]=weight, params[1]=bias -- forward computed inline, no Module needed
    MLXArray weightT = se.alipsa.jmlx.core.MLXShape.transpose(params[0], inputs[0].scope());
    MLXArray pred = MLXOps.add(MLXOps.matmul(inputs[0], weightT), params[1]);
    MLXArray diff = MLXOps.subtract(pred, inputs[1]);
    return new MLXArray[] {MLXOps.sum(MLXOps.multiply(diff, diff))};
  }

  @Test
  void gradsMatchHandComputedValues() {
    try (MLXScope model = new MLXScope()) {
      MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
      MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
      Linear linear = new Linear(model, weight, bias);
      try (ModuleGrad mg = ModuleGrad.of(linear, ModuleGradTest::mseLoss)) {
        try (MLXScope step = model.newChild()) {
          MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray target = MLX.array(step, new float[] {0}, new int[] {1, 1});
          ModuleGrad.Result r = mg.apply(step, new MLXArray[] {x, target});
          assertArrayEquals(new float[] {36f}, r.value().toFloatArray(), EPS);
          assertEquals(List.of("weight", "bias"), List.copyOf(r.grads().keySet()));
          assertArrayEquals(new float[] {12, 24, 36}, r.grads().get("weight").toFloatArray(), EPS);
          assertArrayEquals(new float[] {12}, r.grads().get("bias").toFloatArray(), EPS);
        }
        // Restore path: after the step scope closed, forward() must still
        // work -- proves rebind's finally restored the model-scope arrays.
        try (MLXScope after = model.newChild()) {
          MLXArray x2 = MLX.array(after, new float[] {1, 0, 0}, new int[] {1, 3});
          assertArrayEquals(new float[] {1f}, linear.forward(x2).toFloatArray(), EPS);
        }
      }
    }
  }

  @Test
  void restoreSurvivesAThrowingLoss() {
    try (MLXScope model = new MLXScope()) {
      MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
      MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
      Linear linear = new Linear(model, weight, bias);
      try (ModuleGrad mg = ModuleGrad.of(linear, (params, inputs) -> {
        throw new RuntimeException("loss boom");
      })) {
        try (MLXScope step = model.newChild()) {
          MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray target = MLX.array(step, new float[] {0}, new int[] {1, 1});
          assertThrows(RuntimeException.class, () -> mg.apply(step, new MLXArray[] {x, target}));
        }
      }
      try (MLXScope after = model.newChild()) {
        MLXArray x2 = MLX.array(after, new float[] {1, 0, 0}, new int[] {1, 3});
        assertArrayEquals(new float[] {1f}, linear.forward(x2).toFloatArray(), EPS);
      }
    }
  }

  @Test
  void reusedAcrossDifferentInputsGradsDifferAccordingly() {
    try (MLXScope model = new MLXScope()) {
      MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
      MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
      Linear linear = new Linear(model, weight, bias);
      try (ModuleGrad mg = ModuleGrad.of(linear, ModuleGradTest::mseLoss)) {
        float[] grads1;
        try (MLXScope step1 = model.newChild()) {
          MLXArray x1 = MLX.array(step1, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray t1 = MLX.array(step1, new float[] {0}, new int[] {1, 1});
          grads1 = mg.apply(step1, new MLXArray[] {x1, t1}).grads().get("weight").toFloatArray();
        }
        try (MLXScope step2 = model.newChild()) {
          MLXArray x2 = MLX.array(step2, new float[] {2, 2, 2}, new int[] {1, 3});
          MLXArray t2 = MLX.array(step2, new float[] {0}, new int[] {1, 1});
          float[] grads2 = mg.apply(step2, new MLXArray[] {x2, t2}).grads().get("weight").toFloatArray();
          assertNotEquals(grads1[0], grads2[0]);
        }
      }
    }
  }

  @Test
  void updateThenApplyReflectsNewWeights() {
    try (MLXScope model = new MLXScope()) {
      MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
      MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
      Linear linear = new Linear(model, weight, bias);
      try (ModuleGrad mg = ModuleGrad.of(linear, ModuleGradTest::mseLoss)) {
        float[] gradsBefore;
        try (MLXScope step1 = model.newChild()) {
          MLXArray x = MLX.array(step1, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray t = MLX.array(step1, new float[] {0}, new int[] {1, 1});
          gradsBefore = mg.apply(step1, new MLXArray[] {x, t}).grads().get("weight").toFloatArray();
        }
        linear.update(Map.of("weight", MLX.array(model, new float[] {2, 2, 2}, new int[] {1, 3})));
        try (MLXScope step2 = model.newChild()) {
          MLXArray x = MLX.array(step2, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray t = MLX.array(step2, new float[] {0}, new int[] {1, 1});
          float[] gradsAfter = mg.apply(step2, new MLXArray[] {x, t}).grads().get("weight").toFloatArray();
          assertNotEquals(gradsBefore[0], gradsAfter[0]);
        }
      }
    }
  }

  @Test
  void treeWithNoParametersThrows() {
    try (MLXScope model = new MLXScope()) {
      Module empty = new Module(model) {};
      assertThrows(IllegalStateException.class,
          () -> ModuleGrad.of(empty, (params, inputs) -> inputs));
    }
  }
}
```

Remove the unused `Linear probe = null;` line from `mseLoss` before committing (leftover from drafting
— Checkstyle's unused-variable check would otherwise fail the build).

Run: `./gradlew :jmlx-core:test --tests '*ModuleGradTest*'`.

## Task 3: full verification pass

1. `./gradlew spotlessApply` then `./gradlew build` — Spotless, Checkstyle, full test suite (including
   `loaderGuardTest`), must succeed. Record the actual `BUILD SUCCESSFUL` output and per-module test
   counts.
2. `grep -rn 'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` — must stay
   empty (Global Constraint 1 / spec Verification #6a).
3. `grep -rn 'mlx_vector_array_new\b\|mlx_vector_array_set_data\b' jmlx-core/src/main` — confirm by eye
   every allocator is a confined `Arena`, never an `MLXScope` (spec Verification #7, extended to this
   plan's two new call sites).
4. `./gradlew :jmlx-examples:run` — HelloMLX still runs (no `MLXGrad`/`ModuleGrad` demo added yet; this
   just confirms Task 1/2 did not regress the example).
5. Update `req/phase4-plan.md`'s Status table: M2 done, commit hash, and a short findings note the same
   shape as M0b/M1's entries if anything here diverged from the spec's assumptions.

## Deliberately not covered by this plan

* `MLXFast.rope`/SDPA (M3) and `QuantizedLinear` (M4) are untouched.
* No optimizer step, no `Module.update` interaction with `hoist` for updated weights — both are named
  Open Questions in the spec for a later phase, not M2 entry conditions.
* `jmlx-examples/HelloMLX` is not extended with a training-loop demo — §9 (Documentation) scopes that
  separately and is not part of this merge point.
