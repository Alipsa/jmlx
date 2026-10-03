package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static se.alipsa.jmlx.models.SchedulerFixtures.await;
import static se.alipsa.jmlx.models.SchedulerFixtures.config;
import static se.alipsa.jmlx.models.SchedulerFixtures.failureOf;
import static se.alipsa.jmlx.models.SchedulerFixtures.gated;
import static se.alipsa.jmlx.models.SchedulerFixtures.greedy;
import static se.alipsa.jmlx.models.SchedulerFixtures.start;
import static se.alipsa.jmlx.models.SchedulerFixtures.tokens;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.nn.KVCache;

/** Failure classification: row-level, cohort-wide, and a failed worker. */
@EnabledIfNativeAvailable
class BatchSchedulerFailureTest {
  private static final int[][] PROMPTS = {{1, 7, 42, 3, 19, 5}, {1, 9, 4}, {1, 2, 3, 4, 5}};

  /**
   * Multiplies row {@code victim} of the embedded batch by NaN on the first (prefill) call only.
   */
  private static DecoderModel.EmbeddingHook nanInRow(int victim, AtomicInteger calls) {
    return embedded -> {
      if (calls.getAndIncrement() != 0) {
        return embedded;
      }
      int[] shape = embedded.shape();
      float[] mask = new float[shape[0]];
      java.util.Arrays.fill(mask, 1f);
      mask[victim] = Float.NaN;
      MLXArray factor = MLX.array(embedded.scope(), mask, new int[] {shape[0], 1, 1});
      return MLXOps.multiply(embedded, factor);
    };
  }

  @ParameterizedTest
  @ValueSource(strings = {"llama", "qwen2", "mistral", "gemma", "phi3", "mixtral"})
  void nonFiniteRowFailsAloneAndNeverTouchesTheOthers(String family) throws Exception {
    // Reference: the same 3-row cohort with the same schedule (row 1 leaves after its first token,
    // compacting 3 -> 2 rows at step 0), but cancelled instead of poisoned.
    List<Integer> refA;
    List<Integer> refC;
    AtomicBoolean cancelRow1 = new AtomicBoolean();
    try (BatchGenerationScheduler scheduler = start(family, config(4, 4), gated(3))) {
      BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[0], 6), e -> {});
      scheduler.submit(
          new GenerationRequest(
              PROMPTS[1], GenerationConfig.greedyDefaults(6, java.util.Set.of()), cancelRow1::get),
          e -> {
            if (e.tokenId() != null) {
              cancelRow1.set(true);
            }
          });
      BatchRequestHandle c = scheduler.submit(greedy(PROMPTS[2], 6), e -> {});
      refA = tokens(await(a));
      refC = tokens(await(c));
    }

    List<GenerationEvent> victimEvents = new CopyOnWriteArrayList<>();
    try (BatchGenerationScheduler scheduler =
        start(family, config(4, 4), gated(3).withEmbeddingHook(nanInRow(1, new AtomicInteger())))) {
      final BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[0], 6), e -> {});
      final BatchRequestHandle victim = scheduler.submit(greedy(PROMPTS[1], 6), victimEvents::add);
      final BatchRequestHandle c = scheduler.submit(greedy(PROMPTS[2], 6), e -> {});

      GenerationAbortedException aborted =
          assertInstanceOf(GenerationAbortedException.class, failureOf(victim));
      assertEquals("non-finite logits", aborted.stage());
      assertTrue(
          aborted.getCause().getMessage().contains("finite"), aborted.getCause().getMessage());
      assertTrue(victimEvents.isEmpty(), "a failed row emits no events and no terminal event");

      // Exactly the reference's tokens: same batch shapes, same compaction step.
      assertEquals(refA, tokens(await(a)), family + " row A");
      assertEquals(refC, tokens(await(c)), family + " row C");
      assertEquals(BatchGenerationScheduler.State.RUNNING, scheduler.state());
    }
  }

  @Test
  void throwingForwardPoisonsTheCohortButTheWorkerKeepsServing() throws Exception {
    AtomicInteger calls = new AtomicInteger();
    DecoderModel.EmbeddingHook failing =
        embedded -> {
          if (calls.getAndIncrement() == 1) { // the first decode step of the first cohort
            throw new IllegalStateException("injected forward failure");
          }
          return embedded;
        };
    try (BatchGenerationScheduler scheduler =
        start("llama", config(4, 4), gated(2).withEmbeddingHook(failing))) {
      BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[0], 6), e -> {});
      BatchRequestHandle b = scheduler.submit(greedy(PROMPTS[1], 6), e -> {});
      for (BatchRequestHandle handle : List.of(a, b)) {
        GenerationAbortedException aborted =
            assertInstanceOf(GenerationAbortedException.class, failureOf(handle));
        assertEquals("batch step", aborted.stage());
        assertEquals("injected forward failure", aborted.getCause().getMessage());
        assertEquals(1, aborted.generatedTokenIds().size(), "failed after the prefill token");
      }
      assertEquals(BatchGenerationScheduler.State.RUNNING, scheduler.state());
      // The model is still valid: a later cohort starts with fresh caches and succeeds.
      assertEquals(
          4, await(scheduler.submit(greedy(PROMPTS[0], 4), e -> {})).generatedTokenIds().size());
    }
  }

  @Test
  void cohortSetupFailureFailsTheCohortButTheWorkerKeepsServing() throws Exception {
    AtomicBoolean faulted = new AtomicBoolean();
    try (BatchGenerationScheduler scheduler =
        start(
            "llama",
            config(4, 4),
            gated(2)
                .withSetupFault(
                    () -> {
                      if (faulted.compareAndSet(false, true)) {
                        throw new IllegalStateException("injected cohort setup failure");
                      }
                    }))) {
      BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[0], 6), e -> {});
      BatchRequestHandle b = scheduler.submit(greedy(PROMPTS[1], 6), e -> {});
      for (BatchRequestHandle handle : List.of(a, b)) {
        GenerationAbortedException aborted =
            assertInstanceOf(GenerationAbortedException.class, failureOf(handle));
        assertEquals("cohort setup", aborted.stage());
        assertEquals("injected cohort setup failure", aborted.getCause().getMessage());
        assertEquals(0, aborted.generatedTokenIds().size(), "failed before any token");
      }
      assertEquals(BatchGenerationScheduler.State.RUNNING, scheduler.state());
      // The model is still valid: a later cohort starts with fresh caches and succeeds.
      assertEquals(
          4, await(scheduler.submit(greedy(PROMPTS[0], 4), e -> {})).generatedTokenIds().size());
    }
  }

  @Test
  void aFailureEscapingTheDecodeLoopIsLabelledCohortNotSetup() throws Exception {
    // The fault fires between steps, after setup finished: it must be attributed to stage
    // "cohort", not "cohort setup", even though it fails the cohort the same way.
    AtomicInteger faults = new AtomicInteger();
    try (BatchGenerationScheduler scheduler =
        start(
            "llama",
            config(4, 4),
            gated(2)
                .withDecodeFault(
                    () -> {
                      if (faults.incrementAndGet() == 2) { // the second decode step
                        throw new IllegalStateException("injected decode loop failure");
                      }
                    }))) {
      BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[0], 6), e -> {});
      BatchRequestHandle b = scheduler.submit(greedy(PROMPTS[1], 6), e -> {});
      for (BatchRequestHandle handle : List.of(a, b)) {
        GenerationAbortedException aborted =
            assertInstanceOf(GenerationAbortedException.class, failureOf(handle));
        assertEquals("cohort", aborted.stage());
        assertEquals("injected decode loop failure", aborted.getCause().getMessage());
        assertEquals(2, aborted.generatedTokenIds().size(), "failed after the second token");
      }
      assertEquals(BatchGenerationScheduler.State.RUNNING, scheduler.state());
      // The model is still valid: a later cohort starts with fresh caches and succeeds.
      assertEquals(
          4, await(scheduler.submit(greedy(PROMPTS[0], 4), e -> {})).generatedTokenIds().size());
    }
  }

  @Test
  void jointEvaluationFailureIsAttributedToTheSharedCohort() throws Exception {
    AtomicInteger evaluations = new AtomicInteger();
    AtomicReference<DecoderModel> model = new AtomicReference<>();
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            config(4, 4),
            root -> {
              DecoderModel m =
                  (DecoderModel)
                      TextGenerationModels.load(root, SchedulerFixtures.checkpoint("llama"));
              m.setStepBoundaryEvaluatorForTest(
                  arrays -> {
                    if (evaluations.incrementAndGet() == 2) {
                      throw new IllegalStateException("injected joint evaluation failure");
                    }
                    MLX.eval(arrays);
                  });
              model.set(m);
              return m;
            },
            gated(2))) {
      BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[0], 6), e -> {});
      BatchRequestHandle b = scheduler.submit(greedy(PROMPTS[1], 6), e -> {});
      for (BatchRequestHandle handle : List.of(a, b)) {
        GenerationAbortedException aborted =
            assertInstanceOf(GenerationAbortedException.class, failureOf(handle));
        assertEquals("batch step", aborted.stage());
        assertEquals("injected joint evaluation failure", aborted.getCause().getMessage());
      }
      assertEquals(BatchGenerationScheduler.State.RUNNING, scheduler.state());
      assertNotNull(await(scheduler.submit(greedy(PROMPTS[0], 3), e -> {})));
    }
  }

  @Test
  void anInteriorLayerCompactionFailureResetsEveryLayerAndFailsTheSurvivors() throws Exception {
    List<KVCache> olds = new CopyOnWriteArrayList<>();
    List<KVCache> news = new CopyOnWriteArrayList<>();
    AtomicInteger layersSeen = new AtomicInteger();
    CacheReorderer failingAtLayerOne =
        (old, layer, indices, destination) -> {
          layersSeen.incrementAndGet();
          if (layer == 1) {
            throw new IllegalStateException("injected interior reorder failure");
          }
          KVCache replacement = old.reorder(indices, destination);
          olds.add(old);
          news.add(replacement);
          return replacement;
        };
    try (BatchGenerationScheduler scheduler =
        start("llama", config(4, 4), gated(3).withReorderer(failingAtLayerOne))) {
      // Row 1 stops after two tokens, forcing one compaction while rows 0 and 2 survive.
      BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[0], 8), e -> {});
      BatchRequestHandle leaver = scheduler.submit(greedy(PROMPTS[1], 2), e -> {});
      BatchRequestHandle c = scheduler.submit(greedy(PROMPTS[2], 8), e -> {});
      assertEquals(FinishReason.MAX_TOKENS, await(leaver).finishReason());
      for (BatchRequestHandle handle : List.of(a, c)) {
        GenerationAbortedException aborted =
            assertInstanceOf(GenerationAbortedException.class, failureOf(handle));
        assertEquals("cache compaction", aborted.stage());
        assertEquals("injected interior reorder failure", aborted.getCause().getMessage());
      }
      assertEquals(2, layersSeen.get(), "layer 0 reordered, layer 1 failed");
      assertEquals(1, olds.size());
      assertNull(olds.getFirst().keys(), "the old layer 0 was reset after its replacement");
      assertNull(news.getFirst().keys(), "the installed new layer 0 was reset on failure");
      assertEquals(BatchGenerationScheduler.State.RUNNING, scheduler.state());
      assertNotNull(await(scheduler.submit(greedy(PROMPTS[0], 3), e -> {})));
    }
  }

  @Test
  void failedWorkerCompletesEveryAcceptedStageAndRejectsLaterSubmits() throws Exception {
    AtomicInteger calls = new AtomicInteger();
    DecoderModel.EmbeddingHook fatal =
        embedded -> {
          if (calls.getAndIncrement() == 1) {
            throw new AssertionError("injected fatal error");
          }
          return embedded;
        };
    Thread.UncaughtExceptionHandler previous = Thread.getDefaultUncaughtExceptionHandler();
    AtomicReference<Throwable> uncaught = new AtomicReference<>();
    Thread.setDefaultUncaughtExceptionHandler((t, e) -> uncaught.set(e));
    try {
      BatchGenerationScheduler scheduler =
          start(
              "llama", config(1, 4), BatchGenerationScheduler.Hooks.NONE.withEmbeddingHook(fatal));
      List<GenerationEvent> events = new CopyOnWriteArrayList<>();
      List<BatchRequestHandle> handles = new ArrayList<>();
      for (int i = 0; i < 3; i++) { // one active (batch size 1), two queued
        handles.add(scheduler.submit(greedy(PROMPTS[i], 20), events::add));
      }
      for (BatchRequestHandle handle : handles) {
        GenerationAbortedException aborted =
            assertInstanceOf(GenerationAbortedException.class, failureOf(handle));
        assertInstanceOf(SchedulerFailedException.class, aborted.getCause());
        assertInstanceOf(AssertionError.class, aborted.getCause().getCause());
      }
      assertTrue(
          events.stream().noneMatch(e -> e.finishReason() != null),
          "no listener receives a terminal event on failure");
      SchedulerFailedException failed =
          assertThrows(
              SchedulerFailedException.class,
              () -> scheduler.submit(greedy(PROMPTS[0], 2), e -> {}));
      assertInstanceOf(AssertionError.class, failed.getCause());
      scheduler.close(); // returns normally and does not rethrow the failure
      assertEquals(BatchGenerationScheduler.State.FAILED, scheduler.state());
      assertTrue(scheduler.failure().isPresent());
      assertInstanceOf(AssertionError.class, scheduler.failure().get());
      assertInstanceOf(AssertionError.class, uncaught.get(), "an Error is not swallowed");
      try (BatchGenerationScheduler again = start("llama", config(2, 2))) {
        assertEquals(BatchGenerationScheduler.State.RUNNING, again.state());
      }
    } finally {
      Thread.setDefaultUncaughtExceptionHandler(previous);
    }
  }

  @Test
  void distinctRejectionTypesRaceCorrectly() throws Exception {
    BatchSchedulerConfig tiny = new BatchSchedulerConfig(1, 1, 4096, 256);
    // The gate (waiting >= 1000) holds the queue; close() in the finally reaches the worker first.
    BatchGenerationScheduler scheduler = start("llama", tiny, gated(1000));
    try {
      scheduler.submit(greedy(PROMPTS[1], 2), e -> {});
      assertThrows(
          BatchAdmissionRejectedException.class,
          () -> {
            // Permits are batch(1) + queued(1) = 2; the third submission is over capacity.
            scheduler.submit(greedy(PROMPTS[1], 2), e -> {});
            scheduler.submit(greedy(PROMPTS[1], 2), e -> {});
          });
    } finally {
      scheduler.close();
    }
    assertThrows(
        SchedulerClosedException.class, () -> scheduler.submit(greedy(PROMPTS[1], 2), e -> {}));
  }
}
