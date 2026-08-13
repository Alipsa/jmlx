package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_closure_;
import se.alipsa.jmlx.ffi.mlx_closure_new_func$fun;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Primitive-only autograd over a flat primal vector: {@code mlx_value_and_grad} wrapped as a Java
 * closure. Deliberately has no {@code Module}-aware overload -- see {@code ModuleGrad} in the
 * neural-network package and req/phase4-plan.md §6 for why that lives there instead of here (this
 * class would otherwise have to import that package, inverting its one-way dependency onto this
 * one).
 */
public final class MLXGrad {

  private MLXGrad() {}

  static {
    NativeLoader.ensureLoaded();
  }

  /**
   * Stashes a {@link Throwable} that escaped the upcall body, so {@link Fn#apply} can re-surface
   * the ORIGINAL exception after mlx-c reports the resulting native failure as a generic {@link
   * MLXException} (req/phase4-plan.md §6, the three-step exception-safety protocol). Thread-local,
   * not a per-{@code Fn} field: cleared immediately before every {@code apply}, mirroring {@code
   * NativeOps.checked}'s identical clear-before-call rule for {@code NativeLoader}'s native-error
   * thread-local, and for the same reason -- a stale value from a previous failure must never be
   * misattributed to the next one.
   */
  private static final ThreadLocal<Throwable> ESCAPED = new ThreadLocal<>();

  /**
   * Differentiates {@code body} with respect to the primal indices in {@code argnums}. {@code
   * argnums} must be non-empty and strictly increasing; the upper-bound check (every index {@code <
   * primals.length}) happens per {@link Fn#apply} call, once {@code primals.length} is known.
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
        throw new IllegalArgumentException(
            "valueAndGrad: argnums[" + i + "] = " + argnums[i] + " is negative");
      }
      if (i > 0 && argnums[i] <= argnums[i - 1]) {
        throw new IllegalArgumentException(
            "valueAndGrad: argnums must be strictly increasing, got " + Arrays.toString(argnums));
      }
    }
  }

  /**
   * A live {@code mlx_closure_value_and_grad} plus the upcall stub backing it. Reusable across many
   * {@link #apply} calls -- {@code target} is a per-call argument, not bound at construction,
   * because grads must land in the per-iteration step scope (req/phase4-plan.md §6: binding it at
   * construction forces a choice between a target closed by iteration two, or one upcall stub
   * churned per step). Confined to its constructing thread, the same as {@link MLXScope}/{@link
   * MLXArray}: {@code Holder} and both closures it owns are only ever touched from that thread,
   * except for the {@link Cleaner} backstop below.
   */
  public static final class Fn implements AutoCloseable {

    private static final Cleaner CLEANER = Cleaner.create();

    /**
     * Owns the two native closures and the arena backing the upcall stub. This -- not {@code Fn} --
     * is what gets registered with the {@link Cleaner}, for the same capture-avoidance reason
     * {@code MLXScope}'s own private {@code Holder} exists: the cleanup action must never hold a
     * reference path back to the {@code Fn} it cleans up after, or {@code Fn} could never become
     * unreachable and the action would never run.
     */
    private static final class Holder {
      private final Arena arena;
      private final MemorySegment plainClosure;
      private final MemorySegment vgClosure;
      private volatile boolean closed;

      Holder(Arena arena, MemorySegment plainClosure, MemorySegment vgClosure) {
        this.arena = arena;
        this.plainClosure = plainClosure;
        this.vgClosure = vgClosure;
      }

      /**
       * Frees both closures, then the arena backing the upcall stub -- order is load-bearing
       * (req/phase4-plan.md §6): both closures' underlying {@code std::function}s hold the raw stub
       * pointer until freed, so closing the arena first would leave a dangling stub inside a live
       * closure. Each step runs even if an earlier one throws, so a failure never abandons the
       * later frees (or the arena) permanently. Synchronized and idempotent: the owning {@code
       * Fn}'s explicit {@code close()} and the {@link Cleaner} backstop can both reach this method,
       * and must not double-free.
       *
       * <p>Open question (req/initial-plan.md, Open questions), inherited unresolved from {@code
       * MLXScope}'s own {@code Holder}: whether these native frees are safe to run from the Cleaner
       * thread rather than {@code Fn}'s owning thread. Implemented as specified pending that
       * determination.
       */
      synchronized void closeAll() {
        if (closed) {
          return;
        }
        closed = true;
        try {
          NativeOps.checked(
              "valueAndGrad.close", () -> mlx_h.mlx_closure_value_and_grad_free(vgClosure));
        } finally {
          try {
            NativeOps.checked("valueAndGrad.close", () -> mlx_h.mlx_closure_free(plainClosure));
          } finally {
            arena.close();
          }
        }
      }
    }

    /**
     * The upcall target: holds {@code body} and the current {@code applyTarget}, but -- unlike
     * {@code Fn} itself -- nothing that references {@code Fn}. This is what actually backs the
     * upcall stub, via a bound reference to {@link #onClosureInvoked}, so the reference chain the
     * FFM {@code Linker.upcallStub} contract creates -- arena (kept alive by {@link Holder}) to
     * stub to the bound {@code MethodHandle} to its target -- never loops back to {@code Fn}. Had
     * the target instead been {@code Fn::onClosureInvoked}, {@code Fn} would stay strongly
     * reachable for as long as {@link Holder}'s arena is (unverified against JDK source, but this
     * is the documented {@code upcallStub} contract), which is exactly what {@link Holder} exists
     * to remain reachable for -- {@code Fn} could then never become unreachable, and the {@link
     * Cleaner} backstop above would never run.
     */
    private static final class Upcall {
      private final Function<MLXArray[], MLXArray[]> body;
      private MLXScope applyTarget;

      Upcall(Function<MLXArray[], MLXArray[]> body) {
        this.body = body;
      }

      /**
       * The upcall body: {@code int fun(mlx_vector_array* res, const mlx_vector_array in)}. Runs
       * synchronously on the {@link Fn#apply} caller's thread (confirmed: req/phase4-plan.md Probe
       * 0b(c)), so the {@link MLXArray}s it builds are confined to that thread like any other, and
       * {@code applyTarget} needs no synchronization despite {@code Fn} being reachable from the
       * Cleaner thread too: nothing here is ever touched off {@code Fn}'s owning thread. Never lets
       * a {@link Throwable} escape past this frame -- catches everything, stashes it on {@link
       * #ESCAPED}, and returns {@code 1}, a supported non-leaking mlx-c error path
       * (req/phase4-plan.md Research findings, {@code closure.cpp:44-51}).
       */
      int onClosureInvoked(MemorySegment res, MemorySegment in) {
        try {
          MLXArray[] primalsIn = unpackVector(in, applyTarget);
          MLXArray[] result = body.apply(primalsIn);
          if (result == null || result.length == 0 || result[0].ndim() != 0) {
            int rank = result == null || result.length == 0 ? -1 : result[0].ndim();
            throw new IllegalArgumentException(
                "valueAndGrad: body's first returned array must be rank-0 (a reduced "
                    + "scalar loss), got rank "
                    + rank);
          }
          MemorySegment[] handles = new MemorySegment[result.length];
          for (int i = 0; i < result.length; i++) {
            if (result[i] == null) {
              throw new IllegalArgumentException(
                  "valueAndGrad: body's returned array[" + i + "] is null");
            }
            handles[i] = result[i].handle();
          }
          try (Arena tmp = Arena.ofConfined()) {
            MemorySegment buf = NativeOps.copyHandlesInto(handles, tmp);
            NativeOps.checked(
                "valueAndGrad", () -> mlx_h.mlx_vector_array_set_data(res, buf, handles.length));
          }
          return 0;
        } catch (Throwable t) {
          ESCAPED.set(t);
          return 1;
        }
      }
    }

    private final Thread owner = Thread.currentThread();
    private final Upcall upcall;
    private final Holder holder;
    private final Cleaner.Cleanable cleanable;
    private final int[] argnums;
    private boolean closed;

    private Fn(Function<MLXArray[], MLXArray[]> body, int[] argnums) {
      this.argnums = argnums.clone();
      this.upcall = new Upcall(body);
      // If anything below throws, this Fn is never returned to valueAndGrad's caller and has no
      // Holder yet to register with the Cleaner -- so arena, and whichever of the two closures
      // already exist, must be released here rather than leaking for the rest of the process.
      // plain/vg are plain locals (not Holder fields) precisely so this catch block can see which
      // of them got as far as being created, independent of whether a Holder is ever built.
      // Shared, not confined: Holder.closeAll() below runs on the Cleaner thread when the backstop
      // fires, and every operation there -- both closures' by-value struct marshalling (which reads
      // vgClosure/plainClosure) and arena.close() itself -- requires the segment/arena to permit
      // access from a thread other than whichever one constructed it (confirmed empirically: both
      // throw WrongThreadException off-thread against a confined arena, neither does against a
      // shared one). A WrongThreadException thrown there would be silently swallowed by the
      // Cleaner (its cleaning-action contract ignores exceptions with no trace), so a confined
      // arena here would have made the backstop above permanently inert -- closed still flips to
      // true on the first line of closeAll(), so it would never even retry -- while looking, from
      // every angle short of an off-thread empirical check, exactly like a working one.
      Arena arena = Arena.ofShared();
      MemorySegment plain = null;
      MemorySegment vg = null;
      try {
        mlx_closure_new_func$fun.Function upcallFn = upcall::onClosureInvoked;
        MemorySegment funcPtr = mlx_closure_new_func$fun.allocate(upcallFn, arena);
        // mlx_closure_new_func is statusless: on failure it calls the error handler and returns a
        // null-ctx struct (closure.cpp's catch block), the same hazard class MLX.array/
        // MLX.newVectorArray already guard against explicitly. clearLastNativeError() must run
        // immediately before the call, not just after a failure: without it, a stale message left
        // by some earlier, unrelated statusless failure would still be sitting there and get
        // misattributed to this call (NativeOps.checked's javadoc documents the identical hazard).
        NativeLoader.clearLastNativeError();
        plain = mlx_h.mlx_closure_new_func(arena, funcPtr);
        if (mlx_closure_.ctx(plain).address() == 0) {
          throw NativeOps.nativeFailure("mlx_closure_new_func");
        }
        MemorySegment plainForLambda = plain;
        vg = mlx_h.mlx_closure_value_and_grad_new(arena);
        MemorySegment vgForLambda = vg;
        try (Arena tmp = Arena.ofConfined()) {
          MemorySegment nativeArgnums = tmp.allocateFrom(ValueLayout.JAVA_INT, this.argnums);
          NativeOps.checked(
              "valueAndGrad",
              () ->
                  mlx_h.mlx_value_and_grad(
                      vgForLambda, plainForLambda, nativeArgnums, this.argnums.length));
        }
      } catch (Throwable t) {
        // Mirrors Holder.closeAll()'s own free order (vg before plain) before releasing the arena:
        // the same ordering rule that method's javadoc calls load-bearing applies here too --
        // closing the arena while a closure's std::function still holds the upcall stub pointer
        // leaves that closure dangling, even on a failure path where nothing will ever invoke it
        // again. mlx_closure_free/mlx_closure_value_and_grad_free both no-op on a null ctx
        // (private/closure.h), so the guards below only need to ask "did construction get this
        // far", not "is the ctx itself valid".
        try {
          if (vg != null) {
            mlx_h.mlx_closure_value_and_grad_free(vg);
          }
          if (plain != null) {
            mlx_h.mlx_closure_free(plain);
          }
        } finally {
          arena.close();
        }
        throw t;
      }
      this.holder = new Holder(arena, plain, vg);
      this.cleanable = CLEANER.register(this, holder::closeAll);
    }

    /**
     * Runs the closure, landing traced primals, values and grads in {@code target}. {@code primals}
     * must have at least {@code argnums[argnums.length - 1] + 1} elements.
     */
    public Result apply(MLXScope target, MLXArray[] primals) {
      ensureOpen();
      if (argnums[argnums.length - 1] >= primals.length) {
        throw new IllegalArgumentException(
            "valueAndGrad: argnums "
                + Arrays.toString(argnums)
                + " out of range for "
                + primals.length
                + " primal(s)");
      }
      MemorySegment[] handles = new MemorySegment[primals.length];
      for (int i = 0; i < primals.length; i++) {
        handles[i] = primals[i].handle();
      }
      upcall.applyTarget = target;
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
          NativeOps.checked(
              "valueAndGrad.apply",
              () -> mlx_h.mlx_closure_value_and_grad_apply(res0, res1, holder.vgClosure, inputVec));
          MLXArray[] values = unpackVector(res0, target);
          MLXArray[] grads = unpackVector(res1, target);
          return new Result(List.of(values), List.of(grads));
        } catch (MLXException nativeFailure) {
          rethrowEscapedOr(nativeFailure);
          throw nativeFailure;
        } finally {
          mlx_h.mlx_vector_array_free(inputVec);
          mlx_h.mlx_vector_array_free(res0);
          mlx_h.mlx_vector_array_free(res1);
        }
      } finally {
        upcall.applyTarget = null;
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
      // body is a Function<MLXArray[], MLXArray[]>, which declares no checked exceptions --
      // unreachable in practice, kept only because the catch (Throwable) below can technically
      // observe one via sneaky-throw.
      throw new MLXException("valueAndGrad: body threw a checked exception", escaped);
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
     * Confined to the constructing thread, matching {@link MLXScope}/{@link MLXArray}. {@code
     * close()} is otherwise a one-shot: once {@link #closed} flips, {@link Holder#closeAll} is not
     * retried even if it previously threw partway through -- {@link Holder#closeAll} is itself
     * idempotent, so calling it again from the {@link Cleaner} backstop after this method's
     * explicit call is always safe.
     */
    @Override
    public void close() {
      checkThread();
      if (closed) {
        return;
      }
      closed = true;
      holder.closeAll();
      cleanable.clean();
    }

    // Unlike MLXScope.ensureOpen(), does not also consult a Holder-level closed flag: that
    // check exists on MLXScope to catch a parent scope's cascade closing a still-reachable
    // child's Holder out from under it. Fn has no such cascade -- its Holder is only ever closed
    // by this method or by the Cleaner, and the Cleaner cannot have fired while this method is on
    // the stack, since that requires Fn to already be unreachable. So `closed` alone is
    // sufficient here.
    private void ensureOpen() {
      checkThread();
      if (closed) {
        throw new IllegalStateException("MLXGrad.Fn is closed");
      }
    }

    private void checkThread() {
      Thread current = Thread.currentThread();
      if (current != owner) {
        throw new IllegalStateException(
            "MLXGrad.Fn is confined to " + owner + " but was accessed from " + current);
      }
    }
  }

  /**
   * {@code values} holds {@code body}'s rank-0 loss (element 0); {@code grads.get(i)} corresponds
   * to {@code argnums[i]}.
   */
  public record Result(List<MLXArray> values, List<MLXArray> grads) {}
}
