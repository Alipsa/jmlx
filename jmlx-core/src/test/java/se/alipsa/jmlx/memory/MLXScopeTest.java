package se.alipsa.jmlx.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.ref.WeakReference;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.NativeMemoryProbe;

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
  // Named for what it discriminates here -- freed vs. never-freed on a
  // single allocation -- not "leak", which means something different (an
  // accumulation budget across many iterations) in MLXMemoryLeakTest's own
  // same-named-looking constant.
  private static final long FREED_DETECTION_SLACK_BYTES = 500_000; // << the 4 MB array size
  // Strictly less than @Timeout below: a poll loop bounded only by the test
  // timeout fails as a bare TimeoutException with none of the baseline/after
  // numbers the assertion below exists to report. Bounding the loop itself
  // leaves the timeout as a backstop for a genuinely hung JVM, not the
  // mechanism that reports an ordinary failure.
  private static final long POLL_DEADLINE_SECONDS = 20;

  /**
   * "The action provably captures a holder, not the scope" is enforced structurally (a static-nested holder, not the
   * scope, is registered with the Cleaner), not directly assertable. What is observable is the consequence: if the
   * capture rule were violated, the scope could never become unreachable and the first loop below would never observe a
   * cleared referent.
   *
   * <p>
   * The referent clearing only proves the scope became unreachable, not that {@code holder.closeAll()} actually ran and
   * freed the native array -- a separate, independent {@link java.lang.ref.Cleaner} registration on the same scope was
   * tried here first, but multiple Cleaners on one referent fire independently with no defined order, so that
   * approach's latch passed even with this class's own backstop registration deliberately removed. Sampling mlx-c's own
   * active memory ({@link NativeMemoryProbe}, shared with {@link se.alipsa.jmlx.core.MLXMemoryLeakTest}) before and
   * after is the one check that actually depends on the real backstop having executed -- confirmed by the same mutation
   * (temporarily registering a no-op in place of {@code holder::closeAll}): this version of the test fails the
   * assertion below, the latch-based version did not.
   */
  @Test
  @Timeout(value = 30, unit = TimeUnit.SECONDS)
  void cleanerBackstopFreesTheEscapedScopesNativeMemory() throws InterruptedException {
    // A throwaway allocation before sampling baseline: MLX's first-ever use
    // (default device/stream resolution, first Metal allocation) is a
    // one-time cost that would otherwise land entirely in "after" rather
    // than "baseline" if this happened to be the first test in the JVM to
    // touch MLX at all -- MLXMemoryLeakTest's own warmup phase exists for
    // the same reason, just amortized over many iterations instead of one.
    try (MLXScope warmup = new MLXScope()) {
      MLX.array(warmup, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
    }

    long baseline = NativeMemoryProbe.activeMemoryBytes();
    WeakReference<MLXScope> ref = new WeakReference<>(createDetachedScope());

    long refDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(POLL_DEADLINE_SECONDS);
    while (ref.get() != null && System.nanoTime() < refDeadline) {
      System.gc();
      Thread.sleep(50);
    }
    assertNull(ref.get(), "escaped scope was not collected within " + POLL_DEADLINE_SECONDS + "s");

    // The referent clearing and the Cleaner thread actually invoking the
    // registered action are not synchronized with each other, so poll for
    // the memory drop too rather than sampling once right after the loop
    // above exits.
    long memoryDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(POLL_DEADLINE_SECONDS);
    long after = NativeMemoryProbe.activeMemoryBytes();
    while (after - baseline > FREED_DETECTION_SLACK_BYTES && System.nanoTime() < memoryDeadline) {
      System.gc();
      Thread.sleep(50);
      after = NativeMemoryProbe.activeMemoryBytes();
    }

    assertTrue(after - baseline <= FREED_DETECTION_SLACK_BYTES, "cleaner backstop did not free the escaped scope's"
        + " native memory (baseline=" + baseline + ", after=" + after + ")");
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
}
