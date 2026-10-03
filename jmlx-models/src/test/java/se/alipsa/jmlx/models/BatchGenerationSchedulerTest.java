package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static se.alipsa.jmlx.models.SchedulerFixtures.await;
import static se.alipsa.jmlx.models.SchedulerFixtures.config;
import static se.alipsa.jmlx.models.SchedulerFixtures.gated;
import static se.alipsa.jmlx.models.SchedulerFixtures.greedy;
import static se.alipsa.jmlx.models.SchedulerFixtures.sampled;
import static se.alipsa.jmlx.models.SchedulerFixtures.start;
import static se.alipsa.jmlx.models.SchedulerFixtures.tokens;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Behavior of the batch scheduler: parity with the direct path, batching, and row isolation. */
@EnabledIfNativeAvailable
class BatchGenerationSchedulerTest {
  private static final int[][] PROMPTS = {{1, 7, 42, 3, 19, 5}, {1, 9, 4}, {1, 2, 3, 4, 5}};

  /** The single-request direct path on this thread, as the reference. */
  private static GenerationResult direct(String family, GenerationRequest request)
      throws Exception {
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model =
          (DecoderModel) TextGenerationModels.load(scope, SchedulerFixtures.checkpoint(family));
      return model.generate(request, ignored -> {});
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"llama", "qwen2", "mistral", "gemma", "phi3", "mixtral"})
  void singleRequestMatchesTheDirectPath(String family) throws Exception {
    GenerationRequest request = greedy(PROMPTS[0], 6);
    GenerationResult expected = direct(family, request);
    try (BatchGenerationScheduler scheduler = start(family, config(2, 4))) {
      GenerationResult actual = await(scheduler.submit(request, e -> {}));
      assertEquals(expected.generatedTokenIds(), actual.generatedTokenIds(), family);
      assertEquals(expected.finishReason(), actual.finishReason());
      assertEquals(expected.promptTokenIds(), actual.promptTokenIds());
    }
  }

  @Test
  void mixedCohortMatchesIndependentDirectRunsAndUsesFewerForwards() throws Exception {
    int[] newTokens = {7, 3, 5};
    List<GenerationRequest> requests = new ArrayList<>();
    List<GenerationResult> expected = new ArrayList<>();
    for (int i = 0; i < PROMPTS.length; i++) {
      GenerationRequest request = greedy(PROMPTS[i], newTokens[i]);
      requests.add(request);
      expected.add(direct("llama", request));
    }
    // Gated: all three requests must share one cohort whatever the worker's wake-up timing.
    try (BatchGenerationScheduler scheduler = start("llama", config(4, 8), gated(3))) {
      List<BatchRequestHandle> handles = new ArrayList<>();
      for (GenerationRequest request : requests) {
        handles.add(scheduler.submit(request, e -> {}));
      }
      for (int i = 0; i < handles.size(); i++) {
        GenerationResult actual = await(handles.get(i));
        assertEquals(expected.get(i).generatedTokenIds(), actual.generatedTokenIds(), "row " + i);
        assertEquals(newTokens[i], actual.generatedTokenIds().size(), "uneven lengths");
      }
      long perRequest = 7 + 3 + 5;
      assertTrue(
          scheduler.forwardCalls() < perRequest,
          "a cohort must use fewer forwards than one per request step: "
              + scheduler.forwardCalls()
              + " vs "
              + perRequest);
    }
  }

  @Test
  void seededSampledRowsAreIndependentOfTheirCompanion() throws Exception {
    GenerationRequest a = sampled(PROMPTS[0], 6, 11);
    List<Integer> withB;
    List<Integer> withC;
    // Gated so A really decodes in a cohort with its companion, not alone.
    try (BatchGenerationScheduler scheduler = start("llama", config(2, 4), gated(2))) {
      BatchRequestHandle ha = scheduler.submit(a, e -> {});
      scheduler.submit(sampled(PROMPTS[1], 6, 12), e -> {});
      withB = tokens(await(ha));
    }
    try (BatchGenerationScheduler scheduler = start("llama", config(2, 4), gated(2))) {
      BatchRequestHandle ha = scheduler.submit(a, e -> {});
      scheduler.submit(sampled(PROMPTS[1], 6, 99), e -> {});
      withC = tokens(await(ha));
    }
    assertEquals(withB, withC, "a different companion seed must not change row A");
  }

  @Test
  void aPublicCohortGateHoldsTheFirstRequestUntilTheSecondArrives() throws Exception {
    // The public gate overload (not the test-seam Hooks) pins the batch shape, the way the
    // release smoke needs: whatever the worker's wake-up timing, both requests share one cohort.
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            config(2, 4),
            root -> TextGenerationModels.load(root, SchedulerFixtures.checkpoint("llama")),
            waiting -> waiting >= 2,
            Duration.ofSeconds(30))) {
      BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[0], 4), e -> {});
      BatchRequestHandle b = scheduler.submit(greedy(PROMPTS[1], 4), e -> {});
      assertEquals(4, await(a).generatedTokenIds().size());
      assertEquals(4, await(b).generatedTokenIds().size());
      assertEquals(
          List.of(2), scheduler.cohortSizes(), "the gated requests share one cohort of two");
    }
  }

  @Test
  void cancellingARequestHeldByTheGateCompletesItWithoutAnotherRequest() throws Exception {
    // The gate holds single requests for far longer than the await timeout, so the only thing
    // that can complete the first request is its cancellation waking the worker: before the fix
    // the stage stayed pending (and held its permit) until the next submit or close(). The gate
    // then opens, proving the worker kept serving.
    AtomicBoolean hold = new AtomicBoolean(true);
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            config(2, 4),
            root -> TextGenerationModels.load(root, SchedulerFixtures.checkpoint("llama")),
            // While holding, admit only cohorts of two; once released, admit whatever waits.
            waiting -> !hold.get() || waiting >= 2,
            Duration.ofMinutes(10))) {
      BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[0], 4), e -> {});
      a.cancel();
      GenerationResult result = await(a);
      assertEquals(FinishReason.CANCELLED, result.finishReason());
      hold.set(false);
      // The worker kept serving afterwards.
      assertEquals(
          4, await(scheduler.submit(greedy(PROMPTS[1], 4), e -> {})).generatedTokenIds().size());
    }
  }

  @Test
  void tokenCancellationIsDrainedAfterTheGateWait() throws Exception {
    // Tokens have no waker. Set cancellation after the gate is reached, then verify the
    // deadline wakes drainWaiting() and removes the request before it forms a cohort.
    CountDownLatch gated = new CountDownLatch(1);
    AtomicBoolean token = new AtomicBoolean();
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            config(2, 4),
            root -> TextGenerationModels.load(root, SchedulerFixtures.checkpoint("llama")),
            waiting -> {
              gated.countDown();
              return waiting >= 2;
            },
            Duration.ofMillis(500))) {
      BatchRequestHandle a =
          scheduler.submit(
              new GenerationRequest(
                  PROMPTS[0], GenerationConfig.greedyDefaults(4, Set.of()), token::get),
              e -> {});
      assertTrue(gated.await(5, TimeUnit.SECONDS));
      token.set(true);
      GenerationResult result = await(a);
      assertEquals(FinishReason.CANCELLED, result.finishReason());
      assertEquals(
          0, scheduler.cohortCount(), "cancelled request is drained before cohort formation");
    }
  }

  @Test
  void tokenPollingAfterTheGateDoesNotBlockAdmission() throws Exception {
    AtomicBoolean inGate = new AtomicBoolean();
    CountDownLatch polling = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            config(2, 4),
            root -> TextGenerationModels.load(root, SchedulerFixtures.checkpoint("llama")),
            waiting -> {
              inGate.set(true);
              return false;
            },
            Duration.ofMillis(500))) {
      BatchRequestHandle first =
          scheduler.submit(
              new GenerationRequest(
                  PROMPTS[0],
                  GenerationConfig.greedyDefaults(4, Set.of()),
                  () -> {
                    if (inGate.get()) {
                      polling.countDown();
                      try {
                        release.await();
                      } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new IllegalStateException(e);
                      }
                      return true;
                    }
                    return false;
                  }),
              e -> {});
      try {
        assertTrue(polling.await(5, TimeUnit.SECONDS));
        var admission =
            java.util.concurrent.CompletableFuture.supplyAsync(
                () -> scheduler.submit(greedy(PROMPTS[1], 4), e -> {}));
        admission.get(5, TimeUnit.SECONDS).cancel();
      } finally {
        release.countDown();
      }
      assertEquals(FinishReason.CANCELLED, await(first).finishReason());
    }
  }

  @Test
  void tokenThatThrowsOnceAfterTheGateWaitAbortsTheRequest() throws Exception {
    AtomicBoolean inGate = new AtomicBoolean();
    AtomicBoolean threw = new AtomicBoolean();
    IllegalStateException cause = new IllegalStateException("token boom");
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            config(2, 4),
            root -> TextGenerationModels.load(root, SchedulerFixtures.checkpoint("llama")),
            waiting -> {
              inGate.set(true);
              return false;
            },
            Duration.ofMillis(500))) {
      BatchRequestHandle bad =
          scheduler.submit(
              new GenerationRequest(
                  PROMPTS[0],
                  GenerationConfig.greedyDefaults(4, Set.of()),
                  () -> {
                    if (inGate.get() && threw.compareAndSet(false, true)) {
                      throw cause;
                    }
                    return false;
                  }),
              e -> {});
      GenerationAbortedException aborted =
          assertInstanceOf(GenerationAbortedException.class, SchedulerFixtures.failureOf(bad));
      assertEquals(cause, aborted.getCause());
      assertEquals("cancellation token", aborted.stage());
      assertTrue(aborted.generatedTokenIds().isEmpty());
      assertTrue(threw.get());
      assertEquals(0, scheduler.cohortCount(), "the failed request never runs");
      assertEquals(
          4, await(scheduler.submit(greedy(PROMPTS[1], 4), e -> {})).generatedTokenIds().size());
    }
  }

  @Test
  void aGateRejectedQueueIsReleasedWhenTheWaitExpires() throws Exception {
    // waiting >= 2 never passes for a single request; the wait must release it instead of holding
    // it until close().
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            config(2, 4),
            root -> TextGenerationModels.load(root, SchedulerFixtures.checkpoint("llama")),
            waiting -> waiting >= 2,
            Duration.ofMillis(500))) {
      long began = System.nanoTime();
      GenerationResult result = await(scheduler.submit(greedy(PROMPTS[0], 4), e -> {}));
      long heldMillis = (System.nanoTime() - began) / 1_000_000;
      assertEquals(4, result.generatedTokenIds().size());
      assertTrue(
          heldMillis >= 400,
          "the request must be held until the wait expires: " + heldMillis + " ms");
      assertEquals(List.of(1), scheduler.cohortSizes(), "it ran alone once the wait expired");
    }
  }

  @Test
  void aGateThatThrowsReleasesTheCohortInsteadOfFailingTheScheduler() throws Exception {
    // A throwing predicate must not fail the worker or strand the queue: it is treated as
    // "release now".
    AtomicBoolean threw = new AtomicBoolean();
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            config(2, 4),
            root -> TextGenerationModels.load(root, SchedulerFixtures.checkpoint("llama")),
            waiting -> {
              if (threw.compareAndSet(false, true)) {
                throw new IllegalStateException("gate boom");
              }
              return true;
            },
            Duration.ofSeconds(30))) {
      BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[0], 4), e -> {});
      assertEquals(4, await(a).generatedTokenIds().size());
      assertTrue(threw.get(), "the gate actually threw");
      assertEquals(BatchGenerationScheduler.State.RUNNING, scheduler.state());
      assertEquals(1, scheduler.cohortCount());
      assertEquals(
          4, await(scheduler.submit(greedy(PROMPTS[1], 4), e -> {})).generatedTokenIds().size());
    }
  }

  @Test
  void listenerFailureAbortsOnlyItsOwnRow() throws Exception {
    try (BatchGenerationScheduler scheduler = start("llama", config(2, 4))) {
      List<GenerationEvent> healthyEvents = new CopyOnWriteArrayList<>();
      BatchRequestHandle bad =
          scheduler.submit(
              greedy(PROMPTS[0], 6),
              e -> {
                throw new IllegalStateException("listener boom");
              });
      BatchRequestHandle good = scheduler.submit(greedy(PROMPTS[1], 5), healthyEvents::add);
      Throwable failure = SchedulerFixtures.failureOf(bad);
      GenerationAbortedException aborted =
          assertInstanceOf(GenerationAbortedException.class, failure);
      assertEquals("listener", aborted.stage());
      assertEquals(1, aborted.generatedTokenIds().size(), "aborted on its first token");
      GenerationResult result = await(good);
      assertEquals(5, result.generatedTokenIds().size());
      assertEquals(FinishReason.MAX_TOKENS, result.finishReason());
      GenerationEvent last = healthyEvents.getLast();
      assertEquals(FinishReason.MAX_TOKENS, last.finishReason(), "one terminal event");
      assertEquals(1, healthyEvents.stream().filter(e -> e.finishReason() != null).count());
    }
  }

  @Test
  void stageDependentSeesACompletionExceptionWrappingTheRealCause() throws Exception {
    try (BatchGenerationScheduler scheduler = start("llama", config(2, 4))) {
      BatchRequestHandle bad =
          scheduler.submit(
              greedy(PROMPTS[0], 4),
              e -> {
                throw new IllegalStateException("boom");
              });
      Throwable seen =
          bad.stage()
              .handle((r, t) -> t)
              .toCompletableFuture()
              .get(SchedulerFixtures.TIMEOUT_SECONDS, java.util.concurrent.TimeUnit.SECONDS);
      assertInstanceOf(java.util.concurrent.CompletionException.class, seen);
      assertInstanceOf(GenerationAbortedException.class, seen.getCause());
    }
  }

  @Test
  void cancellingOneRowLeavesTheOtherRunningToItsOwnLimit() throws Exception {
    try (BatchGenerationScheduler scheduler = start("llama", config(2, 4))) {
      List<GenerationEvent> events = Collections.synchronizedList(new ArrayList<>());
      BatchRequestHandle[] cancelled = new BatchRequestHandle[1];
      BatchRequestHandle victim =
          scheduler.submit(
              greedy(PROMPTS[0], 20),
              e -> {
                events.add(e);
                if (e.tokenId() != null && events.size() == 3) {
                  cancelled[0].cancel();
                }
              });
      cancelled[0] = victim;
      final BatchRequestHandle survivor = scheduler.submit(greedy(PROMPTS[1], 12), e -> {});
      GenerationResult cancelledResult = await(victim);
      assertEquals(FinishReason.CANCELLED, cancelledResult.finishReason());
      assertTrue(cancelledResult.generatedTokenIds().size() < 20);
      assertEquals(
          FinishReason.CANCELLED, events.getLast().finishReason(), "one CANCELLED terminal event");
      GenerationResult survived = await(survivor);
      assertEquals(12, survived.generatedTokenIds().size());
      assertEquals(FinishReason.MAX_TOKENS, survived.finishReason());
    }
  }

  @Test
  void cancellationTokenThatThrowsFailsOnlyItsOwnRequest() throws Exception {
    try (BatchGenerationScheduler scheduler = start("llama", config(2, 4))) {
      GenerationRequest flaky =
          new GenerationRequest(
              PROMPTS[0],
              GenerationConfig.greedyDefaults(8, Set.of()),
              () -> {
                throw new IllegalStateException("token boom");
              });
      BatchRequestHandle bad = scheduler.submit(flaky, e -> {});
      BatchRequestHandle good = scheduler.submit(greedy(PROMPTS[1], 4), e -> {});
      GenerationAbortedException aborted =
          assertInstanceOf(GenerationAbortedException.class, SchedulerFixtures.failureOf(bad));
      assertEquals("cancellation token", aborted.stage());
      assertEquals(4, await(good).generatedTokenIds().size());
    }
  }

  @Test
  void stopAndEosTokensEndTheirOwnRowOnly() throws Exception {
    GenerationResult probe = direct("llama", greedy(PROMPTS[0], 6));
    int third = probe.generatedTokenIds().get(2);
    try (BatchGenerationScheduler scheduler = start("llama", config(2, 4))) {
      BatchRequestHandle stops =
          scheduler.submit(greedy(PROMPTS[0], 6, Set.of(), Set.of(third)), e -> {});
      BatchRequestHandle eos =
          scheduler.submit(greedy(PROMPTS[0], 6, Set.of(third), Set.of()), e -> {});
      final BatchRequestHandle plain = scheduler.submit(greedy(PROMPTS[1], 6), e -> {});
      GenerationResult stopped = await(stops);
      assertEquals(FinishReason.STOP_TOKEN, stopped.finishReason());
      assertEquals(probe.generatedTokenIds().subList(0, 2), stopped.generatedTokenIds());
      GenerationResult ended = await(eos);
      assertEquals(FinishReason.EOS, ended.finishReason());
      assertEquals(probe.generatedTokenIds().subList(0, 3), ended.generatedTokenIds());
      assertEquals(6, await(plain).generatedTokenIds().size());
    }
  }

  @Test
  void zeroNewTokensCompletesWithoutPrefill() throws Exception {
    try (BatchGenerationScheduler scheduler = start("llama", config(2, 4))) {
      GenerationResult result = await(scheduler.submit(greedy(PROMPTS[0], 0), e -> {}));
      assertEquals(FinishReason.MAX_TOKENS, result.finishReason());
      assertTrue(result.generatedTokenIds().isEmpty());
      assertEquals(0, scheduler.forwardCalls(), "no prefill for maxNewTokens == 0");
    }
  }

  @Test
  void requestsThatCanNeverBeServedAreRejectedAtSubmitWithoutAPermit() throws Exception {
    // batch 1 + queued 2 = 3 permits, exactly the number of invalid submissions below, so a leaked
    // permit would exhaust them. Both valid submissions fit the queue whatever the worker's timing.
    BatchSchedulerConfig small = new BatchSchedulerConfig(1, 2, 8, 10);
    try (BatchGenerationScheduler scheduler = start("llama", small)) {
      assertThrows(
          IllegalArgumentException.class,
          () -> scheduler.submit(greedy(new int[] {1, 999}, 2), e -> {}),
          "token id outside the vocabulary");
      assertThrows(
          IllegalArgumentException.class,
          () -> scheduler.submit(greedy(PROMPTS[0], 11), e -> {}),
          "maxNewTokens above the scheduler's cap");
      assertThrows(
          IllegalArgumentException.class,
          () -> scheduler.submit(greedy(new int[9], 2), e -> {}),
          "prompt over the token budget even alone");
      // None of those took a permit: all three permits are still free.
      BatchRequestHandle a = scheduler.submit(greedy(PROMPTS[1], 2), e -> {});
      BatchRequestHandle b = scheduler.submit(greedy(PROMPTS[1], 2), e -> {});
      assertNotNull(await(a));
      assertNotNull(await(b));
    }
  }
}
