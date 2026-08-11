package se.alipsa.jmlx.memory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.ref.Cleaner;
import java.lang.ref.WeakReference;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

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

  /**
   * "The action provably captures a holder, not the scope" is enforced structurally (a static-nested holder, not the
   * scope, is registered with the Cleaner), not directly assertable. What is observable is the consequence: if the
   * capture rule were violated, the scope could never become unreachable and this loop would never observe a cleared
   * referent, tripping the timeout instead.
   *
   * <p>
   * Reaching a cleared referent only proves the scope became unreachable, not that the backstop's cleanup actually ran
   * -- both can be true well before the JVM's Cleaner thread gets around to invoking the registered action. A second,
   * independent {@link Cleaner} registered on the same scope makes that run itself observable via a latch, since
   * multiple Cleaners may be registered on one object and each fires its own action once that object becomes
   * phantom-reachable.
   */
  @Test
  @Timeout(value = 30, unit = TimeUnit.SECONDS)
  void cleanerBackstopAllowsAnEscapedScopeToBecomeUnreachable() throws InterruptedException {
    CountDownLatch cleanupRan = new CountDownLatch(1);
    WeakReference<MLXScope> ref = new WeakReference<>(createDetachedScope(cleanupRan));

    while (ref.get() != null) {
      System.gc();
      Thread.sleep(50);
    }

    assertNull(ref.get());
    assertTrue(cleanupRan.await(30, TimeUnit.SECONDS),
        "cleaner backstop did not run after the scope became unreachable");
  }

  // Isolated in its own frame so no local variable in the calling test
  // method keeps the scope reachable after this method returns. Allocates
  // through MLX.array(), not scope.allocate() directly: the latter is only
  // valid when given a real mlx_array-shaped constructor call, and a raw
  // uninitialized allocation would make mlx_array_free undefined behavior
  // once the backstop runs.
  private static MLXScope createDetachedScope(CountDownLatch cleanupRan) {
    MLXScope scope = new MLXScope();
    MLX.array(scope, new float[] {1f, 2f, 3f}, new int[] {3});
    Cleaner.create().register(scope, cleanupRan::countDown);
    return scope;
  }
}
