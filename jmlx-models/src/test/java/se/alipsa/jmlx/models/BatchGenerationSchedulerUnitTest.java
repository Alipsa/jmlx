package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Pure-Java unit tests for the scheduler's gating and cohort accounting: no model load, no native
 * library, so they run on every platform (the native-gated classes are skipped elsewhere).
 */
class BatchGenerationSchedulerUnitTest {
  private static final int CAPACITY = BatchGenerationScheduler.CohortSizeWindow.CAPACITY;

  @Test
  void aNonPositiveGateWaitIsRejectedAtStart() {
    // Rejected before any worker starts, so it is safe to call twice back to back.
    BatchGenerationScheduler.ModelFactory factory =
        root -> TextGenerationModels.load(root, SchedulerFixtures.checkpoint("llama"));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            BatchGenerationScheduler.start(
                SchedulerFixtures.config(2, 4), factory, waiting -> true, Duration.ZERO));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            BatchGenerationScheduler.start(
                SchedulerFixtures.config(2, 4), factory, waiting -> true, Duration.ofMillis(-1)));
  }

  @Test
  void aGateWaitThatDoesNotFitInNanosIsRejectedAtStart() {
    BatchGenerationScheduler.ModelFactory factory =
        root -> TextGenerationModels.load(root, SchedulerFixtures.checkpoint("llama"));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            BatchGenerationScheduler.start(
                SchedulerFixtures.config(2, 4),
                factory,
                waiting -> true,
                Duration.ofSeconds(Long.MAX_VALUE)));
  }

  @Test
  void theCohortWindowKeepsTheLastCohortsInStartOrderAndCountsEverything() {
    BatchGenerationScheduler.CohortSizeWindow window =
        new BatchGenerationScheduler.CohortSizeWindow();
    assertEquals(List.of(), window.recent());
    assertEquals(0, window.cohorts());
    assertEquals(0, window.rows());

    int total = CAPACITY + 17;
    for (int i = 1; i <= total; i++) {
      window.record((i % 3) + 1); // sizes cycle 2, 3, 1
    }

    List<Integer> expected = new ArrayList<>(CAPACITY);
    for (int i = total - CAPACITY + 1; i <= total; i++) {
      expected.add((i % 3) + 1);
    }
    assertEquals(expected, window.recent(), "the last cohorts, in start order");
    assertEquals(total, window.cohorts(), "cohorts beyond the window are still counted");
    long rows = 0;
    for (int i = 1; i <= total; i++) {
      rows += (i % 3) + 1;
    }
    assertEquals(rows, window.rows());

    // A window with fewer cohorts than CAPACITY keeps all of them.
    BatchGenerationScheduler.CohortSizeWindow small =
        new BatchGenerationScheduler.CohortSizeWindow();
    small.record(3);
    small.record(1);
    assertEquals(List.of(3, 1), small.recent());
    assertEquals(2, small.cohorts());
    assertEquals(4, small.rows());
  }
}
