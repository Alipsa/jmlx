package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_closure_;
import se.alipsa.jmlx.ffi.mlx_closure_new_func$fun;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Primitive-only autograd over a flat primal vector: {@code mlx_value_and_grad} wrapped as a Java closure. Deliberately
 * has no {@code Module}-aware overload -- see {@code ModuleGrad} in the neural-network package and req/phase4-plan.md
 * §6 for why that lives there instead of here (this class would otherwise have to import that package, inverting its
 * one-way dependency onto this one).
 */
public final class MLXGrad {

  private MLXGrad() {}

  static {
    NativeLoader.ensureLoaded();
  }

  /**
   * Stashes a {@link Throwable} that escaped the upcall body, so {@link Fn#apply} can re-surface the ORIGINAL exception
   * after mlx-c reports the resulting native failure as a generic {@link MLXException} (req/phase4-plan.md §6, the
   * three-step exception-safety protocol). Thread-local, not a per-{@code Fn} field: cleared immediately before every
   * {@code apply}, mirroring {@code NativeOps.checked}'s identical clear-before-call rule for {@code NativeLoader}'s
   * native-error thread-local, and for the same reason -- a stale value from a previous failure must never be
   * misattributed to the next one.
   */
  private static final ThreadLocal<Throwable> ESCAPED = new ThreadLocal<>();

  /**
   * Differentiates {@code body} with respect to the primal indices in {@code argnums}. {@code argnums} must be
   * non-empty and strictly increasing; the upper-bound check (every index {@code < primals.length}) happens per
   * {@link Fn#apply} call, once {@code primals.length} is known.
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
   * A live {@code mlx_closure_value_and_grad} plus the upcall stub backing it. Reusable across many {@link #apply}
   * calls -- {@code target} is a per-call argument, not bound at construction, because grads must land in the
   * per-iteration step scope (req/phase4-plan.md §6: binding it at construction forces a choice between a target
   * closed by iteration two, or one upcall stub churned per step). Confined to its constructing thread, the same as
   * {@link MLXScope}/{@link MLXArray}: {@code arena} and both closures below are only ever touched from that thread.
   */
  public static final class Fn implements AutoCloseable {

    private final Thread owner = Thread.currentThread();
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
      // If anything below throws, this Fn is never returned to valueAndGrad's caller and has no
      // Cleaner backstop (unlike MLXScope) -- so arena, and whichever of the two closures already
      // exist, must be released here rather than leaking for the rest of the process.
      try {
        mlx_closure_new_func$fun.Function upcall = this::onClosureInvoked;
        MemorySegment funcPtr = mlx_closure_new_func$fun.allocate(upcall, arena);
        MemorySegment plain = mlx_h.mlx_closure_new_func(arena, funcPtr);
        // mlx_closure_new_func is statusless: on failure it calls the error handler and returns a
        // null-ctx struct (closure.cpp's catch block), the same hazard class MLX.array/
        // MLX.newVectorArray already guard against explicitly.
        if (mlx_closure_.ctx(plain).address() == 0) {
          throw NativeOps.nativeFailure("mlx_closure_new_func");
        }
        this.plainClosure = plain;
        MemorySegment vg = mlx_h.mlx_closure_value_and_grad_new(arena);
        try (Arena tmp = Arena.ofConfined()) {
          MemorySegment nativeArgnums = tmp.allocateFrom(ValueLayout.JAVA_INT, this.argnums);
          NativeOps.checked("valueAndGrad",
              () -> mlx_h.mlx_value_and_grad(vg, plainClosure, nativeArgnums, this.argnums.length));
        }
        this.vgClosure = vg;
      } catch (Throwable t) {
        arena.close();
        throw t;
      }
    }

    /**
     * Runs the closure, landing traced primals, values and grads in {@code target}. {@code primals} must have at least
     * {@code argnums[argnums.length - 1] + 1} elements.
     */
    public Result apply(MLXScope target, MLXArray[] primals) {
      ensureOpen();
      if (argnums[argnums.length - 1] >= primals.length) {
        throw new IllegalArgumentException(
            "valueAndGrad: argnums " + Arrays.toString(argnums) + " out of range for " + primals.length + " primal(s)");
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
        // res0/res1 each heap-allocate on the native side the moment mlx_vector_array_new
        // constructs them (private/vector.h: new std::vector<array>(...)), regardless of whether
        // mlx_closure_value_and_grad_apply below succeeds -- a failure returns 1 without touching
        // them (closure.cpp), so this method still owns both and must free them on every exit path,
        // not just the success path. One finally covering the whole block, not two, is what makes
        // that true even when checked(...) throws before result unpacking ever runs.
        try {
          NativeOps.checked("valueAndGrad.apply",
              () -> mlx_h.mlx_closure_value_and_grad_apply(res0, res1, vgClosure, inputVec));
          List<MLXArray> values = unpackVector(res0, target);
          List<MLXArray> grads = unpackVector(res1, target);
          return new Result(values, grads);
        } catch (MLXException nativeFailure) {
          rethrowEscapedOr(nativeFailure);
          throw nativeFailure;
        } finally {
          mlx_h.mlx_vector_array_free(inputVec);
          mlx_h.mlx_vector_array_free(res0);
          mlx_h.mlx_vector_array_free(res1);
        }
      } finally {
        applyTarget = null;
        // Defensive, not load-bearing under the current control flow (every path that sets ESCAPED
        // is already drained by rethrowEscapedOr on the failure path): clearing it here too means a
        // stashed Throwable can never outlive one apply() call and pin the MLXArrays/scopes it may
        // reference via the thread-local.
        ESCAPED.remove();
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
      // body is a Function<MLXArray[], MLXArray[]>, which declares no checked exceptions -- unreachable in
      // practice, kept only because the catch (Throwable) below can technically observe one via sneaky-throw.
      throw new MLXException("valueAndGrad: body threw a checked exception", escaped);
    }

    /**
     * The upcall body: {@code int fun(mlx_vector_array* res, const mlx_vector_array in)}. Runs synchronously on the
     * {@link #apply} caller's thread (confirmed: req/phase4-plan.md Probe 0b(c)), so the {@link MLXArray}s it builds
     * are confined to that thread like any other. Never lets a {@link Throwable} escape past this frame -- catches
     * everything, stashes it on {@link #ESCAPED}, and returns {@code 1}, a supported non-leaking mlx-c error path
     * (req/phase4-plan.md Research findings, {@code closure.cpp:44-51}).
     */
    private int onClosureInvoked(MemorySegment res, MemorySegment in) {
      try {
        List<MLXArray> primalsIn = unpackVector(in, applyTarget);
        MLXArray[] result = body.apply(primalsIn.toArray(new MLXArray[0]));
        if (result == null || result.length == 0 || result[0].ndim() != 0) {
          int rank = result == null || result.length == 0 ? -1 : result[0].ndim();
          throw new IllegalArgumentException("valueAndGrad: body's first returned array must be rank-0 (a reduced "
              + "scalar loss), got rank " + rank);
        }
        MemorySegment[] handles = new MemorySegment[result.length];
        for (int i = 0; i < result.length; i++) {
          if (result[i] == null) {
            throw new IllegalArgumentException("valueAndGrad: body's returned array[" + i + "] is null");
          }
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

    private static List<MLXArray> unpackVector(MemorySegment vec, MLXScope target) {
      long n = mlx_h.mlx_vector_array_size(vec);
      MLXArray[] out = new MLXArray[(int) n];
      for (int i = 0; i < n; i++) {
        MemorySegment h = mlx_h.mlx_array_new(target);
        final long idx = i;
        NativeOps.checked("valueAndGrad", () -> mlx_h.mlx_vector_array_get(h, vec, idx));
        out[i] = new MLXArray(target, h);
      }
      return List.of(out);
    }

    /**
     * Confined to the constructing thread, matching {@link MLXScope}/{@link MLXArray}. Frees both closures, then the
     * arena backing the upcall stub -- order is load-bearing (req/phase4-plan.md §6): both closures' underlying
     * {@code std::function}s hold the raw stub pointer until freed, so closing the arena first would leave a dangling
     * stub inside a live closure. Each step runs even if an earlier one throws, so a failure never abandons the later
     * frees (or the arena) permanently -- {@code close()} is otherwise a one-shot: once {@link #closed} flips, a
     * partially-failed cleanup is not retried.
     */
    @Override
    public void close() {
      checkThread();
      if (closed) {
        return;
      }
      closed = true;
      try {
        NativeOps.checked("valueAndGrad.close", () -> mlx_h.mlx_closure_value_and_grad_free(vgClosure));
      } finally {
        try {
          NativeOps.checked("valueAndGrad.close", () -> mlx_h.mlx_closure_free(plainClosure));
        } finally {
          arena.close();
        }
      }
    }

    private void ensureOpen() {
      checkThread();
      if (closed) {
        throw new IllegalStateException("MLXGrad.Fn is closed");
      }
    }

    private void checkThread() {
      Thread current = Thread.currentThread();
      if (current != owner) {
        throw new IllegalStateException("MLXGrad.Fn is confined to " + owner + " but was accessed from " + current);
      }
    }
  }

  /**
   * {@code values} holds {@code body}'s rank-0 loss (element 0); {@code grads.get(i)} corresponds to
   * {@code argnums[i]}.
   */
  public record Result(List<MLXArray> values, List<MLXArray> grads) {}
}
