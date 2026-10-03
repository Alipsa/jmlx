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
import java.util.concurrent.atomic.AtomicBoolean;
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
   * Runs one cohort of all three prompts, then one single-row probe whose first token (sampled on
   * the worker, the only thread allowed to read {@code MLXMemory} while the scheduler runs) is the
   * fixed measurement point: the cohort's scope, caches, and activations are closed, and only the
   * probe's own single-row state is alive at every sample.
   */
  private static void cohortAndProbe(
      BatchGenerationScheduler scheduler, int newTokens, List<Long> samples) throws Exception {
    List<BatchRequestHandle> cohort = new ArrayList<>();
    for (int[] prompt : PROMPTS) {
      cohort.add(scheduler.submit(greedy(prompt, newTokens), e -> {}));
    }
    for (BatchRequestHandle handle : cohort) {
      await(handle);
    }
    AtomicBoolean sampled = new AtomicBoolean();
    BatchRequestHandle probe =
        scheduler.submit(
            greedy(PROMPTS[0], newTokens),
            e -> {
              if (e.tokenId() != null && sampled.compareAndSet(false, true)) {
                samples.add(MLXMemory.activeBytes());
              }
            });
    await(probe);
  }

  @Test
  void repeatedCohortsDoNotGrowActiveMemory() throws Exception {
    // The gate is persistent (every cohort needs three rows) with the fixture's short wait, so
    // every cohort is exactly three rows whatever the worker's wake-up timing (a one-shot gate
    // would leave the later cohorts to run as 1+2 or 1+1+1, so row 0's last sample would include
    // a different amount of live KV cache in each cohort), and the one-row probe after each
    // cohort is released by the wait instead of being held until close().
    List<Long> samples = new ArrayList<>();
    try (BatchGenerationScheduler scheduler = start("llama", config(4, 8), gated(3))) {
      for (int i = 0; i < 13; i++) {
        cohortAndProbe(scheduler, 6, samples);
      }
      long growth = samples.getLast() - samples.getFirst();
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
