package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.ref.WeakReference;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Tests for {@link MLXGrad}, the primitive autograd wrapper (req/phase4-plan.md §6,
 * req/plans/phase4-m2-plan.md Task 1).
 */
@EnabledIfNativeAvailable
class MLXGradTest {

  private static final float EPS = 1e-5f;

  @Test
  void gradOfSumOfSquares() {
    try (MLXScope model = new MLXScope();
        MLXGrad.Fn fn =
            MLXGrad.valueAndGrad(
                xs -> new MLXArray[] {MLXOps.sum(MLXOps.multiply(xs[0], xs[0]))}, new int[] {0})) {
      try (MLXScope step = model.newChild()) {
        MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {3});
        MLXGrad.Result r = fn.apply(step, new MLXArray[] {x});
        assertEquals(1, r.values().size());
        assertEquals(1, r.grads().size());
        assertArrayEquals(new float[] {14f}, r.values().get(0).toFloatArray(), EPS);
        assertArrayEquals(new float[] {2, 4, 6}, r.grads().get(0).toFloatArray(), EPS);
      }
    }
  }

  @Test
  void fnReusedAcrossTwoStepScopesLandsGradsInEachOwnScope() {
    try (MLXScope model = new MLXScope();
        MLXGrad.Fn fn =
            MLXGrad.valueAndGrad(
                xs -> new MLXArray[] {MLXOps.sum(MLXOps.multiply(xs[0], xs[0]))}, new int[] {0})) {
      try (MLXScope step1 = model.newChild()) {
        MLXArray x1 = MLX.array(step1, new float[] {1, 2, 3}, new int[] {3});
        MLXArray grad1 = fn.apply(step1, new MLXArray[] {x1}).grads().get(0);
        assertSame(step1, grad1.scope());
      }
      // step1 closed; a second apply into a fresh step scope must still work --
      // proves target moved to apply(), not the constructor.
      try (MLXScope step2 = model.newChild()) {
        MLXArray x2 = MLX.array(step2, new float[] {2, 2, 2}, new int[] {3});
        MLXArray grad2 = fn.apply(step2, new MLXArray[] {x2}).grads().get(0);
        assertSame(step2, grad2.scope());
        assertArrayEquals(new float[] {4, 4, 4}, grad2.toFloatArray(), EPS);
      }
    }
  }

  @Test
  void argnumsMustNotBeEmpty() {
    assertThrows(IllegalArgumentException.class, () -> MLXGrad.valueAndGrad(xs -> xs, new int[0]));
  }

  @Test
  void argnumsMustBeStrictlyIncreasing() {
    assertThrows(
        IllegalArgumentException.class, () -> MLXGrad.valueAndGrad(xs -> xs, new int[] {1, 1}));
    assertThrows(
        IllegalArgumentException.class, () -> MLXGrad.valueAndGrad(xs -> xs, new int[] {1, 0}));
  }

  @Test
  void argnumsMustBeNonNegative() {
    assertThrows(
        IllegalArgumentException.class, () -> MLXGrad.valueAndGrad(xs -> xs, new int[] {-1}));
  }

  @Test
  void argnumsOutOfRangeForPrimalsThrows() {
    try (MLXScope scope = new MLXScope();
        MLXGrad.Fn fn =
            MLXGrad.valueAndGrad(xs -> new MLXArray[] {MLXOps.sum(xs[0])}, new int[] {1})) {
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

  @Test
  void nullSecondaryResultThrowsNamingTheIndex() {
    try (MLXScope scope = new MLXScope();
        MLXGrad.Fn fn =
            MLXGrad.valueAndGrad(xs -> new MLXArray[] {MLXOps.sum(xs[0]), null}, new int[] {0})) {
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      IllegalArgumentException ex =
          assertThrows(IllegalArgumentException.class, () -> fn.apply(scope, new MLXArray[] {x}));
      assertTrue(ex.getMessage().contains("[1]"), ex.getMessage());
    }
  }

  // Regression guard: a null at index 0 must go through the same named-index diagnostic as any
  // other index, not NPE out of the rank check (which used to run before the null check).
  @Test
  void nullPrimaryResultThrowsNamingTheIndex() {
    try (MLXScope scope = new MLXScope();
        MLXGrad.Fn fn = MLXGrad.valueAndGrad(xs -> new MLXArray[] {null}, new int[] {0})) {
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      IllegalArgumentException ex =
          assertThrows(IllegalArgumentException.class, () -> fn.apply(scope, new MLXArray[] {x}));
      assertTrue(ex.getMessage().contains("[0]"), ex.getMessage());
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
        MLXGrad.Fn fn =
            MLXGrad.valueAndGrad(
                xs -> {
                  throw new BodyBoom("boom");
                },
                new int[] {0})) {
      MLXArray x = MLX.array(scope, new float[] {1}, new int[] {1});
      // Asserts the exception TYPE, not just "throws something" -- without step 3 of the
      // exception-safety protocol, a generic MLXException is thrown too, which a type-blind
      // assertion would not distinguish from the broken version.
      BodyBoom caught = assertThrows(BodyBoom.class, () -> fn.apply(scope, new MLXArray[] {x}));
      // The native failure that step 2 of the protocol produced must not be silently dropped --
      // it is attached as a suppressed exception on the original.
      assertEquals(1, caught.getSuppressed().length);
      assertInstanceOf(MLXException.class, caught.getSuppressed()[0]);
    }
    // Reaching this line (and every later test in the suite) is the real assertion: the JVM
    // survived a Throwable escaping the upcall body.
  }

  @Test
  void closeIsIdempotentAndPostCloseApplyThrows() {
    try (MLXScope scope = new MLXScope()) {
      MLXGrad.Fn fn = MLXGrad.valueAndGrad(xs -> new MLXArray[] {MLXOps.sum(xs[0])}, new int[] {0});
      MLXArray x = MLX.array(scope, new float[] {1}, new int[] {1});
      fn.apply(scope, new MLXArray[] {x});
      fn.close();
      fn.close();
      assertThrows(IllegalStateException.class, () -> fn.apply(scope, new MLXArray[] {x}));
    }
  }

  @Test
  void crossThreadApplyThrows() throws InterruptedException {
    try (MLXScope scope = new MLXScope();
        MLXGrad.Fn fn =
            MLXGrad.valueAndGrad(xs -> new MLXArray[] {MLXOps.sum(xs[0])}, new int[] {0})) {
      MLXArray x = MLX.array(scope, new float[] {1}, new int[] {1});
      Throwable[] caught = new Throwable[1];
      Thread other =
          new Thread(
              () -> {
                try {
                  fn.apply(scope, new MLXArray[] {x});
                } catch (Throwable t) {
                  caught[0] = t;
                }
              });
      other.start();
      other.join();
      assertInstanceOf(IllegalStateException.class, caught[0]);
    }
  }

  /**
   * Mirrors {@code MLXScopeTest.cleanerBackstopFreesTheEscapedScopesNativeMemory}'s reachability
   * half: if {@code Fn}'s {@code Holder}/{@code Upcall} split were broken and something captured
   * {@code Fn} itself (directly or transitively) instead of just {@code Holder}'s arena, {@code Fn}
   * could never become phantom-reachable and this loop would time out.
   *
   * <p>Unlike that test, there is deliberately no {@code NativeMemoryProbe}-based second half here:
   * {@code Holder.closeAll()} frees two process-heap {@code std::function}s and the small FFM
   * upcall trampoline backing them, none of which mlx-c's active-memory counter ({@code
   * mlx_get_active_memory}) tracks -- that counter only reflects device/array buffers, and {@code
   * Fn} retains none between calls. Whether {@code Holder.closeAll()} itself completes without
   * throwing when run off the constructing thread -- the actual defect fixed alongside this test,
   * {@code Arena.ofConfined()} vs {@code Arena.ofShared()} for {@code Holder}'s arena -- was
   * instead confirmed empirically outside this suite: a confined arena's {@code MemorySegment} read
   * and {@code Arena.close()} both threw {@code WrongThreadException} when invoked from a thread
   * other than the arena's owner, a shared arena's did neither. That check cannot be folded into
   * this test because {@link java.lang.ref.Cleaner} itself swallows any exception thrown by a
   * cleaning action without a trace (also confirmed empirically) -- so a {@code
   * WrongThreadException} thrown by a wrongly-confined arena on the Cleaner thread is invisible to
   * any test, this one included, and would still let this reachability check pass. Only using the
   * right arena type guarantees the fix; this test only guards the unrelated capture failure.
   */
  @Test
  @Timeout(value = 30, unit = TimeUnit.SECONDS)
  void cleanerBackstopRunsForAnEscapedFn() throws InterruptedException {
    WeakReference<MLXGrad.Fn> ref = new WeakReference<>(createDetachedFn());

    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
    while (ref.get() != null && System.nanoTime() < deadline) {
      System.gc();
      Thread.sleep(50);
    }
    assertNull(ref.get(), "escaped Fn was not collected within 20s");
  }

  // Isolated in its own frame so no local variable in the calling test method keeps fn (or the
  // scope its one real apply() call runs against) reachable after this method returns.
  private static MLXGrad.Fn createDetachedFn() {
    MLXGrad.Fn fn = MLXGrad.valueAndGrad(xs -> new MLXArray[] {MLXOps.sum(xs[0])}, new int[] {0});
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      fn.apply(scope, new MLXArray[] {x});
    }
    return fn;
  }
}
