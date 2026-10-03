package se.alipsa.jmlx.models;

/**
 * Immutable limits for a {@link BatchGenerationScheduler}.
 *
 * @param maxBatchSize most requests decoded together in one cohort
 * @param maxQueuedRequests most requests waiting to start; further submissions are rejected
 * @param maxPromptTokenBudget most prefill work in one cohort, counted as rows times the maximum
 *     padded prompt width (left padding included, not the sum of valid prompt lengths)
 * @param maxNewTokensPerRequest largest {@code maxNewTokens} a request may ask for
 */
public record BatchSchedulerConfig(
    int maxBatchSize, int maxQueuedRequests, int maxPromptTokenBudget, int maxNewTokensPerRequest) {

  /**
   * Validates that every limit is positive and that the total admission permits fit an int.
   *
   * @throws IllegalArgumentException if a limit is not positive or the permits overflow
   */
  public BatchSchedulerConfig {
    requirePositive("maxBatchSize", maxBatchSize);
    requirePositive("maxQueuedRequests", maxQueuedRequests);
    requirePositive("maxPromptTokenBudget", maxPromptTokenBudget);
    requirePositive("maxNewTokensPerRequest", maxNewTokensPerRequest);
    if ((long) maxBatchSize + maxQueuedRequests > Integer.MAX_VALUE) {
      // The documented contract is IllegalArgumentException (not Math.addExact's
      // ArithmeticException) when the total permits do not fit an int.
      throw new IllegalArgumentException(
          "maxBatchSize + maxQueuedRequests overflows an int: "
              + maxBatchSize
              + " + "
              + maxQueuedRequests);
    }
  }

  /** Modest limits suitable for a local demo: 4 rows, 16 waiting, 8192 prefill tokens. */
  public static BatchSchedulerConfig defaults() {
    return new BatchSchedulerConfig(4, 16, 8192, 4096);
  }

  /** Total admission permits: every accepted request holds one until its stage is dispatched. */
  int admissionPermits() {
    return maxBatchSize + maxQueuedRequests;
  }

  private static void requirePositive(String name, int value) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive");
    }
  }
}
