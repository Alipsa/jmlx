package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/** Config validation; pure Java, so it runs without the native runtime. */
class BatchSchedulerConfigTest {

  @Test
  void nonPositiveLimitsAreRejected() {
    assertThrows(IllegalArgumentException.class, () -> new BatchSchedulerConfig(0, 4, 8192, 4096));
    assertThrows(IllegalArgumentException.class, () -> new BatchSchedulerConfig(4, 0, 8192, 4096));
    assertThrows(IllegalArgumentException.class, () -> new BatchSchedulerConfig(4, 4, 0, 4096));
    assertThrows(IllegalArgumentException.class, () -> new BatchSchedulerConfig(4, 4, 8192, 0));
  }

  @Test
  void anOverflowingPermitTotalIsRejectedWithIllegalArgumentException() {
    // Math.addExact would throw ArithmeticException; the documented contract is
    // IllegalArgumentException when the total admission permits do not fit an int.
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> new BatchSchedulerConfig(4, Integer.MAX_VALUE - 3, 8192, 4096));
    assertEquals(
        "maxBatchSize + maxQueuedRequests overflows an int: 4 + " + (Integer.MAX_VALUE - 3),
        e.getMessage());
  }

  @Test
  void hugeButFittingPermitTotalIsAccepted() {
    // The admission queue is bounded without pre-allocation, so this must not throw here or in
    // start(): previously an ArrayBlockingQueue of this size OOMed with "Requested array size
    // exceeds VM limit".
    BatchSchedulerConfig config =
        new BatchSchedulerConfig(Integer.MAX_VALUE / 2, Integer.MAX_VALUE / 2, 8192, 4096);
    assertEquals(Integer.MAX_VALUE - 1, config.admissionPermits());
  }
}
