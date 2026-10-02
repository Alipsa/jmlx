package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static se.alipsa.jmlx.models.SchedulerFixtures.await;
import static se.alipsa.jmlx.models.SchedulerFixtures.config;
import static se.alipsa.jmlx.models.SchedulerFixtures.gated;
import static se.alipsa.jmlx.models.SchedulerFixtures.greedy;
import static se.alipsa.jmlx.models.SchedulerFixtures.start;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/** Native active-memory plateaus: across repeated cohorts, and within one long cohort. */
@EnabledIfNativeAvailable
class BatchSchedulerMemoryTest {
  private static final int[][] PROMPTS = {{1, 7, 42, 3, 19, 5}, {1, 9, 4}, {1, 2, 3, 4, 5}};
  private static final long SLACK = 8L * 1024 * 1024;

  /**
   * Runs one gated cohort of all three prompts and appends {@code MLXMemory.activeBytes()} read
   * from the worker (the only thread allowed to read it while the scheduler runs) on every token of
   * the first row.
   */
  private static void cohort(BatchGenerationScheduler scheduler, int newTokens, List<Long> samples)
      throws Exception {
    List<BatchRequestHandle> handles = new ArrayList<>();
    for (int i = 0; i < PROMPTS.length; i++) {
      boolean sampler = i == 0;
      handles.add(
          scheduler.submit(
              greedy(PROMPTS[i], newTokens),
              e -> {
                if (sampler && e.tokenId() != null) {
                  samples.add(MLXMemory.activeBytes());
                }
              }));
    }
    for (BatchRequestHandle handle : handles) {
      await(handle);
    }
  }

  @Test
  void repeatedCohortsDoNotGrowActiveMemory() throws Exception {
    // The gate is one-shot; later cohorts run ungated. Every cohort has the same shape, so the
    // last sample of the first and of the last cohort are at the same point in the lifecycle.
    List<Long> samples = new ArrayList<>();
    try (BatchGenerationScheduler scheduler = start("llama", config(4, 8), gated(3))) {
      cohort(scheduler, 6, samples);
      long afterFirst = samples.getLast();
      for (int i = 0; i < 12; i++) {
        cohort(scheduler, 6, samples);
      }
      long growth = samples.getLast() - afterFirst;
      assertTrue(
          growth <= SLACK,
          "12 further cohorts retained " + growth + " native bytes after their scopes closed");
    }
  }

  @Test
  void activeMemoryPlateausWithinOneLongCohort() throws Exception {
    // A per-step activation leak cannot hide behind end-of-cohort cleanup: sample active bytes on
    // the worker, from a callback, at every step of one 200-token cohort.
    int steps = 200;
    List<Long> samples = new ArrayList<>();
    AtomicLong stepCount = new AtomicLong();
    try (BatchGenerationScheduler scheduler = start("llama", config(4, 8), gated(3))) {
      List<BatchRequestHandle> handles = new ArrayList<>();
      for (int i = 0; i < PROMPTS.length; i++) {
        boolean sampler = i == 0;
        handles.add(
            scheduler.submit(
                greedy(PROMPTS[i], steps),
                e -> {
                  if (sampler && e.tokenId() != null) {
                    stepCount.incrementAndGet();
                    samples.add(MLXMemory.activeBytes());
                  }
                }));
      }
      for (BatchRequestHandle handle : handles) {
        await(handle);
      }
    }
    assertEquals(steps, samples.size());
    // The KV cache legitimately grows by a fixed amount per token; a leak would grow far faster.
    // Compare the late slope with the early slope instead of an absolute bound.
    long early = samples.get(60) - samples.get(20);
    long late = samples.get(190) - samples.get(150);
    long perTokenBudget = 4L * 1024 * 1024; // generous for 2 layers x 3 rows of a tiny model
    assertTrue(
        samples.get(steps - 1) - samples.get(20) <= perTokenBudget * 4,
        "active bytes climbed from " + samples.get(20) + " to " + samples.get(steps - 1));
    assertTrue(
        late <= Math.max(early, 0) * 2 + SLACK,
        "per-step growth is not accelerating: early=" + early + " late=" + late);
  }
}
