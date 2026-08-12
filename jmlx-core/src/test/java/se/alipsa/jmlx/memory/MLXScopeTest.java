package se.alipsa.jmlx.memory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.ref.WeakReference;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
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

  @Test
  void arrayAccessFromWrongThreadThrowsAndOwningThreadCloseStillWorks() throws InterruptedException {
    MLXScope scope = new MLXScope();
    MLXArray array = MLX.array(scope, new float[] {1f}, new int[] {1});
    AtomicReference<Throwable> caught = new AtomicReference<>();
    Thread other = new Thread(() -> {
      try {
        array.shape();
      } catch (Throwable t) {
        caught.set(t);
      }
    });
    other.start();
    other.join();

    assertInstanceOf(IllegalStateException.class, caught.get());
    // The rejected foreign-thread read must not have wedged anything: the
    // owning thread can still read the array and close the scope.
    assertDoesNotThrow(array::shape);
    assertDoesNotThrow(scope::close);
  }

  @Test
  void arrayAccessAfterScopeCloseThrowsInsteadOfReadingFreedMemory() {
    MLXScope scope = new MLXScope();
    MLXArray array = MLX.array(scope, new float[] {1f}, new int[] {1});
    // MLXScope.close() frees handles via Holder.closeAll() and never
    // touches any MLXArray, so array's own `closed` flag is still false
    // here -- this is exactly the path that discriminates checkAccess()
    // (option 1, checkThread() + ensureOpen()) from an array-local
    // Thread owner (option 3, which can only reproduce the thread half).
    // Without the ensureOpen() half, shape() below would read a freed
    // handle instead of throwing.
    scope.close();

    assertThrows(IllegalStateException.class, array::shape);
  }

  // req/phase4-plan.md §2: child scopes, scopeOf/innermost, hoist/keep.

  @Test
  void newChildHasTheCorrectParentAndDepthAndAncestry() {
    try (MLXScope root = new MLXScope()) {
      assertNull(root.parent());
      assertEquals(0, root.depth());
      try (MLXScope child = root.newChild()) {
        assertSame(root, child.parent());
        assertEquals(1, child.depth());
        try (MLXScope grandchild = child.newChild()) {
          assertEquals(2, grandchild.depth());
          assertTrue(root.isAncestorOf(grandchild));
          assertTrue(child.isAncestorOf(grandchild));
          assertTrue(root.isAncestorOf(root)); // reflexive
          assertTrue(!grandchild.isAncestorOf(root));
        }
      }
    }
  }

  @Test
  void newChildFromWrongThreadThrowsAndLeavesParentOpen() throws InterruptedException {
    MLXScope parent = new MLXScope();
    AtomicReference<Throwable> caught = new AtomicReference<>();
    Thread other = new Thread(() -> {
      try {
        parent.newChild();
      } catch (Throwable t) {
        caught.set(t);
      }
    });
    other.start();
    other.join();

    assertInstanceOf(IllegalStateException.class, caught.get());
    assertDoesNotThrow(parent::newChild).close();
    parent.close();
  }

  @Test
  void innermostReturnsTheChildRegardlessOfArgumentOrder() {
    try (MLXScope parent = new MLXScope(); MLXScope child = parent.newChild()) {
      assertSame(child, MLXScope.innermost(parent, child));
      assertSame(child, MLXScope.innermost(child, parent));
    }
  }

  @Test
  void innermostRejectsTwoIndependentRootScopes() {
    try (MLXScope a = new MLXScope(); MLXScope b = new MLXScope()) {
      assertThrows(IllegalArgumentException.class, () -> MLXScope.innermost(a, b));
    }
  }

  @Test
  void innermostRejectsSiblingScopes() {
    try (MLXScope parent = new MLXScope(); MLXScope childA = parent.newChild(); MLXScope childB = parent.newChild()) {
      assertThrows(IllegalArgumentException.class, () -> MLXScope.innermost(childA, childB));
    }
  }

  /**
   * The use-after-free the cascade introduces (req/phase4-plan.md §2). {@code parent.close()} frees {@code child}'s
   * handles via the {@code Holder} cascade without ever touching {@code child}'s own {@code closed} field -- without
   * {@code holder.isClosed()} in {@code MLXArray}/{@code MLXScope}'s {@code ensureOpen()}, {@code a.shape()} below
   * would read freed memory instead of throwing.
   */
  @Test
  void parentCascadeThenChildAndItsArraysThrowInsteadOfReadingFreedMemory() {
    MLXScope parent = new MLXScope();
    MLXScope child = parent.newChild();
    MLXArray a = MLX.array(child, new float[] {1f}, new int[] {1});

    parent.close();

    assertThrows(IllegalStateException.class, a::shape);
    assertThrows(IllegalStateException.class, () -> child.allocate(8, 8));
    // The child's own explicit close(), arriving after the cascade already
    // freed its handles, must still be a harmless no-op -- not a double free
    // and not an exception.
    assertDoesNotThrow(child::close);
    assertDoesNotThrow(parent::close);
  }

  @Test
  void keepOnAnArrayInARootScopeThrowsIllegalStateExceptionNotNullPointerException() {
    try (MLXScope root = new MLXScope()) {
      MLXArray a = MLX.array(root, new float[] {1f}, new int[] {1});
      assertThrows(IllegalStateException.class, () -> MLX.keep(a));
    }
  }

  @Test
  void hoistIntoItsOwnScopeReturnsTheSameArray() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1f}, new int[] {1});
      assertSame(a, MLX.hoist(a, scope));
    }
  }

  /**
   * Pins the reflexive fast path: a copy-returning implementation of the {@code target == source} case would also pass
   * a same-values assertion, so this asserts identity (above) paired with a repetition-count that would show growth if
   * hoist ever allocated a fresh handle per call in that case.
   */
  @Test
  void hoistIntoItsOwnScopeRepeatedlyShowsNoMemoryGrowth() {
    try (MLXScope warmup = new MLXScope()) {
      MLX.array(warmup, new float[] {0f}, new int[] {1});
    }
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
      for (int i = 0; i < 50; i++) {
        MLX.hoist(a, scope);
      }
      long baseline = NativeMemoryProbe.activeMemoryBytes();
      for (int i = 0; i < 1000; i++) {
        MLX.hoist(a, scope);
      }
      long after = NativeMemoryProbe.activeMemoryBytes();
      assertTrue(after - baseline <= FREED_DETECTION_SLACK_BYTES,
          "active memory grew from " + baseline + " to " + after + " bytes over 1000 same-scope hoist() calls");
    }
  }

  /**
   * Asserts the {@code array.h}/{@code private/array.h} refcounting reasoning (req/phase4-plan.md §2, Research
   * findings) rather than merely inferring it: builds {@code y = multiply(exp(x), x)} in a child scope, {@code
   * keep}s {@code y} into the parent, closes the child (freeing {@code x} and the un-kept {@code exp(x)} intermediate),
   * then evaluates {@code y} and checks its values against a hand-computed golden -- which only succeeds if mlx's own
   * {@code ArrayDesc} kept the graph alive independently of this facade's handle bookkeeping.
   */
  @Test
  void hoistSurvivesChildScopeCloseAndStillEvaluatesCorrectly() {
    try (MLXScope warmup = new MLXScope()) {
      MLX.array(warmup, new float[] {0f}, new int[] {1});
    }
    long baseline = NativeMemoryProbe.activeMemoryBytes();
    try (MLXScope parent = new MLXScope()) {
      MLXArray kept;
      try (MLXScope child = parent.newChild()) {
        MLXArray x = MLX.array(child, new float[] {1f, 2f, 3f}, new int[] {3});
        MLXArray y = MLXOps.multiply(MLXOps.exp(x), x);
        kept = MLX.keep(y);
      }
      MLX.eval(kept);
      float[] expected = {(float) (1 * Math.exp(1)), (float) (2 * Math.exp(2)), (float) (3 * Math.exp(3))};
      assertArrayEquals(expected, kept.toFloatArray(), 1e-2f);
    }
    long after = NativeMemoryProbe.activeMemoryBytes();
    assertTrue(after - baseline <= FREED_DETECTION_SLACK_BYTES, "active memory did not return to baseline after the"
        + " parent closed (baseline=" + baseline + ", after=" + after + ")");
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
  // mechanism that reports an ordinary failure. Shared by both poll loops
  // below via one deadline computed once: two independent 20s budgets would
  // let total polling sum past the 30s @Timeout if the first loop ran long,
  // reintroducing the bare-timeout failure mode this constant exists to
  // avoid.
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
    // A single element suffices: baseline=0 in practice shows the array's
    // own footprint contributes nothing to active memory, so only the
    // one-time init path this warmup targets needs exercising.
    try (MLXScope warmup = new MLXScope()) {
      MLX.array(warmup, new float[] {0f}, new int[] {1});
    }

    long baseline = NativeMemoryProbe.activeMemoryBytes();
    WeakReference<MLXScope> ref = new WeakReference<>(createDetachedScope());

    // One deadline shared by both loops below, not one per loop: two
    // independent budgets could sum past the outer @Timeout if the first
    // loop ran long, reintroducing the bare-timeout failure this test's
    // deadline bounding exists to avoid.
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(POLL_DEADLINE_SECONDS);
    while (ref.get() != null && System.nanoTime() < deadline) {
      System.gc();
      Thread.sleep(50);
    }
    assertNull(ref.get(), "escaped scope was not collected within " + POLL_DEADLINE_SECONDS + "s");

    // The referent clearing and the Cleaner thread actually invoking the
    // registered action are not synchronized with each other, so poll for
    // the memory drop too rather than sampling once right after the loop
    // above exits.
    long after = NativeMemoryProbe.activeMemoryBytes();
    while (after - baseline > FREED_DETECTION_SLACK_BYTES && System.nanoTime() < deadline) {
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
