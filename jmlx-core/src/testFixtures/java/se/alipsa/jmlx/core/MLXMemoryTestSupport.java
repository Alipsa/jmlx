package se.alipsa.jmlx.core;

import java.util.concurrent.TimeUnit;

/**
 * Shared memory-measurement support for tests that read {@link MLXMemory#activeBytes()}.
 *
 * <p>MLX Metal frees buffers through GPU completion handlers that run after {@code eval} or a
 * scope's {@code close()} has already returned, so a plain {@code MLXMemory.activeBytes()} read
 * taken right after work can race those delayed frees and report bytes that are already dead. The
 * settled read below is the sanctioned way for a memory-measurement test to sample the counter; do
 * not inline a plain read in a measurement assertion.
 */
public final class MLXMemoryTestSupport {
  private static final long SETTLE_INTERVAL_MS = 2;
  private static final long SETTLE_TAIL_MS = 16;
  private static final long SETTLE_TIMEOUT_MS = 1000;

  private MLXMemoryTestSupport() {}

  /**
   * {@link MLXMemory#activeBytes()} once its in-flight Metal completion-handler frees (see {@link
   * MLXMemory}'s class javadoc) have drained: polls until the value has been unchanged for {@value
   * #SETTLE_TAIL_MS} milliseconds. Throws {@link IllegalStateException} if the counter is still
   * moving {@value #SETTLE_TIMEOUT_MS} milliseconds after the first read, instead of silently
   * returning an unsettled value.
   */
  public static long settledActiveBytes() throws InterruptedException {
    long previous = MLXMemory.activeBytes();
    long lastChange = System.nanoTime();
    long deadline = lastChange + TimeUnit.MILLISECONDS.toNanos(SETTLE_TIMEOUT_MS);
    while (System.nanoTime() < deadline) {
      Thread.sleep(SETTLE_INTERVAL_MS);
      long current = MLXMemory.activeBytes();
      if (current != previous) {
        previous = current;
        lastChange = System.nanoTime();
      } else if (System.nanoTime() - lastChange >= TimeUnit.MILLISECONDS.toNanos(SETTLE_TAIL_MS)) {
        return current;
      }
    }
    throw new IllegalStateException(
        "MLXMemory.activeBytes() did not settle within "
            + SETTLE_TIMEOUT_MS
            + " ms; last value: "
            + previous);
  }
}
