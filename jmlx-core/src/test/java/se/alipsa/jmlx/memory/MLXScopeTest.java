package se.alipsa.jmlx.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.ref.WeakReference;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.mlx_h;

/**
 * See req/initial-plan.md, Testing approach, "Memory lifecycle" (scope half) and "Testing the Cleaner backstop".
 */
@EnabledIfNativeAvailable
class MLXScopeTest {

  @Test
  void closeIsIdempotent() {
    MLXScope scope = new MLXScope();
    scope.close();
    assertDoesNotThrow(scope::close);
  }

  @Test
  void allocateAfterCloseThrows() {
    MLXScope scope = new MLXScope();
    scope.close();
    assertThrows(IllegalStateException.class, () -> scope.allocate(8, 8));
  }

  @Test
  void closeFromWrongThreadThrowsAndLeavesScopeOpen() throws InterruptedException {
    MLXScope scope = new MLXScope();
    AtomicReference<Throwable> caught = new AtomicReference<>();
    Thread other = new Thread(() -> {
      try {
        scope.close();
      } catch (Throwable t) {
        caught.set(t);
      }
    });
    other.start();
    other.join();

    assertInstanceOf(IllegalStateException.class, caught.get());
    // The rejected cross-thread close must not have actually closed
    // anything: the owning thread can still use the scope, and its own
    // close() still succeeds.
    assertDoesNotThrow(() -> MLX.array(scope, new float[] {1f}, new int[] {1}));
    assertDoesNotThrow(scope::close);
  }

  // The escaped array must be large enough that "never freed" and "freed"
  // are clearly distinguishable through mlx-c's caching allocator: a tiny
  // array's footprint is noise-level regardless of whether it was actually
  // freed, which is exactly why an earlier version of this test (see the
  // javadoc below) passed even with the real backstop mutated away.
  private static final int ARRAY_ELEMENTS = 1_000_000; // 4 MB float32
  private static final long LEAK_THRESHOLD_BYTES = 500_000; // << the 4 MB array size

  /**
   * "The action provably captures a holder, not the scope" is enforced structurally (a static-nested holder, not the
   * scope, is registered with the Cleaner), not directly assertable. What is observable is the consequence: if the
   * capture rule were violated, the scope could never become unreachable and this loop would never observe a cleared
   * referent, tripping the timeout instead.
   *
   * <p>
   * The referent clearing only proves the scope became unreachable, not that {@code holder.closeAll()} actually ran and
   * freed the native array -- a separate, independent {@link java.lang.ref.Cleaner} registration on the same scope was
   * tried here first, but multiple Cleaners on one referent fire independently with no defined order, so that
   * approach's latch passed even with this class's own backstop registration deliberately removed. Sampling mlx-c's own
   * active memory ({@code mlx_get_active_memory}, same as {@link se.alipsa.jmlx.core.MLXMemoryLeakTest}) before and
   * after is the one check that actually depends on the real backstop having executed -- confirmed by the same mutation
   * (temporarily registering a no-op in place of {@code holder::closeAll}): this version of the test times out, the
   * latch-based version did not.
   */
  @Test
  @Timeout(value = 30, unit = TimeUnit.SECONDS)
  void cleanerBackstopFreesTheEscapedScopesNativeMemory() throws InterruptedException {
    long baseline = activeMemoryBytes();
    WeakReference<MLXScope> ref = new WeakReference<>(createDetachedScope());

    while (ref.get() != null) {
      System.gc();
      Thread.sleep(50);
    }

    // The referent clearing and the Cleaner thread actually invoking the
    // registered action are not synchronized with each other, so poll for
    // the memory drop too rather than sampling once right after the loop
    // above exits; @Timeout above is the only bound, same as that loop.
    long after = activeMemoryBytes();
    while (after - baseline > LEAK_THRESHOLD_BYTES) {
      System.gc();
      Thread.sleep(50);
      after = activeMemoryBytes();
    }

    assertTrue(after - baseline <= LEAK_THRESHOLD_BYTES, "cleaner backstop did not free the escaped scope's native"
        + " memory (baseline=" + baseline + ", after=" + after + ")");
  }

  // Isolated in its own frame so no local variable in the calling test
  // method keeps the scope reachable after this method returns. Allocates
  // through MLX.array(), not scope.allocate() directly: the latter is only
  // valid when given a real mlx_array-shaped constructor call, and a raw
  // uninitialized allocation would make mlx_array_free undefined behavior
  // once the backstop runs.
  private static MLXScope createDetachedScope() {
    MLXScope scope = new MLXScope();
    MLX.array(scope, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
    return scope;
  }

  private static long activeMemoryBytes() {
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment out = tmp.allocate(ValueLayout.JAVA_LONG);
      int status = mlx_h.mlx_get_active_memory(out);
      if (status != 0) {
        throw new IllegalStateException("mlx_get_active_memory failed with status " + status);
      }
      return out.get(ValueLayout.JAVA_LONG, 0);
    }
  }
}
