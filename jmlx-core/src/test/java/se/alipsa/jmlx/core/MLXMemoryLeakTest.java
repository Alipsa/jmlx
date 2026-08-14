package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.NativeMemoryProbe;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * See req/initial-plan.md, Testing approach, "Leak test, specified concretely".
 *
 * <p>"Memory does not grow monotonically" is unfalsifiable: MLX uses a caching allocator, so memory
 * legitimately grows then plateaus. This queries mlx-c's own <b>active</b> memory (not cached, not
 * process RSS, via {@code mlx_get_active_memory} -- {@code memory.h}), runs a warmup phase excluded
 * from measurement so the allocator's steady-state caching doesn't register as a leak, then asserts
 * active memory after a further N iterations does not exceed a fixed threshold above the
 * post-warmup reading. A memory <i>decrease</i> is not a leak, so the assertion is one-sided:
 * {@code after - baseline <= threshold}, not {@code Math.abs(after - baseline) <= threshold}.
 *
 * <p>Each iteration allocates a several-hundred-KB array in its own scope and frees it,
 * specifically so that a real per-iteration leak would accumulate to something this test can
 * actually catch, rather than being lost in noise. Two variants exercise the two ways an array's
 * native memory gets freed: {@link #activeMemoryDoesNotGrowWhenScopeClosesArrays} relies solely on
 * the scope's close(); the other, {@link #activeMemoryDoesNotGrowWhenArraysAreClosedExplicitly},
 * closes each {@link MLXArray} itself before its scope closes -- otherwise nothing in this suite
 * would notice if {@code MLXArray.close()} silently stopped freeing anything (a broken {@code
 * MLXScope.Holder.freeOne}, say): every {@code MLXArrayTest} case only asserts that closing doesn't
 * throw.
 */
@EnabledIfNativeAvailable
class MLXMemoryLeakTest {

  private static final int WARMUP_ITERATIONS = 50;
  private static final int MEASURED_ITERATIONS = 200;
  private static final int ARRAY_ELEMENTS = 100_000; // ~400 KB float32 per iteration

  // Generous relative to one iteration's ~400 KB, but far too small to
  // hide a real per-iteration leak accumulated over 200 iterations.
  private static final long LEAK_THRESHOLD_BYTES = 2_000_000;

  @Test
  void activeMemoryDoesNotGrowWhenScopeClosesArrays() {
    assertNoGrowthOver(MLXMemoryLeakTest::runIterationRelyingOnScopeClose);
  }

  @Test
  void activeMemoryDoesNotGrowWhenArraysAreClosedExplicitly() {
    assertNoGrowthOver(MLXMemoryLeakTest::runIterationClosingArraysExplicitly);
  }

  /**
   * A missing {@code mlx_vector_array_free} in {@link NativeOps#vectorInOp} would leak a refcount
   * on every operand of every {@link MLXShape#concatenate} call -- invisible in a single call, but
   * unbounded across iterations. This is the regression guard for exactly that bug.
   */
  @Test
  void activeMemoryDoesNotGrowWithConcatenateInTheLoop() {
    assertNoGrowthOver(MLXMemoryLeakTest::runIterationWithConcatenate);
  }

  /**
   * The variant that matters for req/phase3-plan.md §4's {@code mlx_vector_array}-based joint
   * {@code eval}: a missing {@code mlx_vector_array_free} inside {@code MLX.eval} would show up
   * here as {@code activeMemoryBytes()} growth across iterations, and a double-free of the vector
   * (or of a handle copied into its backing buffer) would abort the JVM outright -- which is itself
   * the assertion, since no Java exception could report it.
   */
  @Test
  void activeMemoryDoesNotGrowWithMultiArrayEvalInTheLoop() {
    assertNoGrowthOver(MLXMemoryLeakTest::runIterationWithMultiArrayEval);
  }

  /**
   * See req/phase4-plan.md §2's {@code Holder} cascade + {@code removeChild} machinery: unlike the
   * variants above, each iteration here creates a genuine {@link MLXScope#newChild()} of one
   * long-lived parent (not a fresh root) and does a cross-scope {@code add} against a weight living
   * in that parent, matching the shape of a real per-step decode loop. A missing {@code
   * removeChild} call would accumulate dead {@code Holder} objects in the parent's children set --
   * a Java-heap leak this native-memory probe cannot see directly -- but a broken cascade or free
   * path in that same machinery would still show up here as active-memory growth.
   */
  @Test
  void activeMemoryDoesNotGrowWithPerIterationChildScopeUnderALongLivedParent() {
    try (MLXScope parent = new MLXScope()) {
      MLXArray weight = MLX.array(parent, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
      for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        runChildScopeIteration(parent, weight);
      }

      long baseline = NativeMemoryProbe.activeMemoryBytes();

      for (int i = 0; i < MEASURED_ITERATIONS; i++) {
        runChildScopeIteration(parent, weight);
      }

      long after = NativeMemoryProbe.activeMemoryBytes();

      assertTrue(
          after - baseline <= LEAK_THRESHOLD_BYTES,
          "active memory grew from "
              + baseline
              + " to "
              + after
              + " bytes over "
              + MEASURED_ITERATIONS
              + " per-iteration child scopes (threshold "
              + LEAK_THRESHOLD_BYTES
              + " bytes)");
    }
  }

  private static void runChildScopeIteration(MLXScope parent, MLXArray weight) {
    try (MLXScope child = parent.newChild()) {
      MLXArray x = MLX.array(child, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
      MLXArray y = MLXOps.add(x, weight);
      MLX.eval(y);
    }
  }

  private static void assertNoGrowthOver(Runnable iteration) {
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
      iteration.run();
    }

    long baseline = NativeMemoryProbe.activeMemoryBytes();

    for (int i = 0; i < MEASURED_ITERATIONS; i++) {
      iteration.run();
    }

    long after = NativeMemoryProbe.activeMemoryBytes();

    assertTrue(
        after - baseline <= LEAK_THRESHOLD_BYTES,
        "active memory grew from "
            + baseline
            + " to "
            + after
            + " bytes over "
            + MEASURED_ITERATIONS
            + " iterations (threshold "
            + LEAK_THRESHOLD_BYTES
            + " bytes)");
  }

  private static void runIterationRelyingOnScopeClose() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
      MLXArray b = MLXOps.exp(a);
      MLX.eval(b);
    }
  }

  private static void runIterationClosingArraysExplicitly() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
      MLXArray b = MLXOps.exp(a);
      MLX.eval(b);
      b.close();
      a.close();
    }
  }

  private static void runIterationWithConcatenate() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
      MLXArray b = MLX.array(scope, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
      MLXArray c = MLXShape.concatenate(new MLXArray[] {a, b}, 0);
      MLX.eval(c);
    }
  }

  private static void runIterationWithMultiArrayEval() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[ARRAY_ELEMENTS], new int[] {ARRAY_ELEMENTS});
      MLXArray b = MLXOps.exp(a);
      MLXArray c = MLXOps.log(b);
      MLX.eval(a, b, c);
    }
  }
}
