package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static se.alipsa.jmlx.models.SchedulerFixtures.await;
import static se.alipsa.jmlx.models.SchedulerFixtures.config;
import static se.alipsa.jmlx.models.SchedulerFixtures.greedy;
import static se.alipsa.jmlx.models.SchedulerFixtures.start;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Start, close, restart, failure, and the threading guard. */
@EnabledIfNativeAvailable
class BatchSchedulerLifecycleTest {
  private static final int[] PROMPT = {1, 7, 42, 3, 19, 5};

  @Test
  void closeThenRestartWorks() throws Exception {
    List<Integer> first;
    try (BatchGenerationScheduler scheduler = start("llama", config(2, 2))) {
      first = SchedulerFixtures.tokens(await(scheduler.submit(greedy(PROMPT, 5), e -> {})));
      assertEquals(BatchGenerationScheduler.State.RUNNING, scheduler.state());
    }
    try (BatchGenerationScheduler again = start("llama", config(2, 2))) {
      assertEquals(
          first, SchedulerFixtures.tokens(await(again.submit(greedy(PROMPT, 5), e -> {}))));
    }
  }

  @Test
  void closedStateIsNeverVisibleWhileTheGuardIsStillHeld() throws Exception {
    BatchGenerationScheduler scheduler = start("llama", config(2, 2));
    scheduler.close();
    assertEquals(BatchGenerationScheduler.State.CLOSED, scheduler.state());
    // Whoever sees CLOSED can start again immediately: the guard was released before shutdown.
    try (BatchGenerationScheduler next = start("llama", config(2, 2))) {
      assertEquals(BatchGenerationScheduler.State.RUNNING, next.state());
    }
  }

  @Test
  void secondStartWhileAnotherWorkerIsAliveIsRejected() throws Exception {
    try (BatchGenerationScheduler first = start("llama", config(2, 2))) {
      assertThrows(SchedulerAlreadyRunningException.class, () -> start("llama", config(2, 2)));
      assertEquals(BatchGenerationScheduler.State.RUNNING, first.state());
    }
  }

  @Test
  void closeIsIdempotentAndLaterSubmitsAreNamedClosed() throws Exception {
    BatchGenerationScheduler scheduler = start("llama", config(2, 2));
    scheduler.close();
    scheduler.close();
    assertThrows(
        SchedulerClosedException.class, () -> scheduler.submit(greedy(PROMPT, 2), e -> {}));
  }

  @Test
  void closeCancelsEveryQueuedAndActiveRequestWithOneTerminalEventEach() throws Exception {
    BatchGenerationScheduler scheduler = start("llama", config(1, 4));
    CountDownLatch firstToken = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    List<List<GenerationEvent>> events = new ArrayList<>();
    List<BatchRequestHandle> handles = new ArrayList<>();
    for (int i = 0; i < 3; i++) {
      List<GenerationEvent> mine = java.util.Collections.synchronizedList(new ArrayList<>());
      events.add(mine);
      int index = i;
      handles.add(
          scheduler.submit(
              greedy(PROMPT, 200),
              e -> {
                mine.add(e);
                if (index == 0 && e.tokenId() != null) {
                  firstToken.countDown();
                  try {
                    release.await(30, TimeUnit.SECONDS);
                  } catch (InterruptedException ex) {
                    Thread.currentThread().interrupt();
                  }
                }
              }));
    }
    assertTrue(firstToken.await(60, TimeUnit.SECONDS));
    Thread closer = new Thread(scheduler::close, "closer");
    closer.start();
    // close() has stopped admission; the worker is still inside the first callback.
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
    while (scheduler.state() != BatchGenerationScheduler.State.CLOSING
        && System.nanoTime() < deadline) {
      Thread.sleep(5);
    }
    assertThrows(
        SchedulerClosedException.class, () -> scheduler.submit(greedy(PROMPT, 2), e -> {}));
    release.countDown();
    closer.join(TimeUnit.SECONDS.toMillis(60));
    assertFalse(closer.isAlive(), "close() returned");
    assertEquals(BatchGenerationScheduler.State.CLOSED, scheduler.state());
    for (int i = 0; i < 3; i++) {
      GenerationResult result = await(handles.get(i));
      assertEquals(FinishReason.CANCELLED, result.finishReason(), "request " + i);
      GenerationEvent last = events.get(i).getLast();
      assertEquals(FinishReason.CANCELLED, last.finishReason(), "terminal event " + i);
      assertEquals(1, events.get(i).stream().filter(e -> e.finishReason() != null).count());
    }
  }

  @Test
  void closeFromACallbackOnTheWorkerDoesNotWaitAndStillShutsDownCleanly() throws Exception {
    BatchGenerationScheduler scheduler = start("llama", config(2, 2));
    AtomicBoolean closedFromWorker = new AtomicBoolean();
    BatchRequestHandle handle =
        scheduler.submit(
            greedy(PROMPT, 50),
            e -> {
              if (e.tokenId() != null && !closedFromWorker.getAndSet(true)) {
                scheduler.close(); // would self-join if it waited
              }
            });
    assertEquals(FinishReason.CANCELLED, await(handle).finishReason());
    // An outside close() still waits for the drain, and afterwards a restart is accepted.
    scheduler.close();
    assertEquals(BatchGenerationScheduler.State.CLOSED, scheduler.state());
    try (BatchGenerationScheduler again = start("llama", config(2, 2))) {
      assertNotNull(await(again.submit(greedy(PROMPT, 2), e -> {})));
    }
  }

  @Test
  void stageObserverClosingTheSchedulerFromTheDispatcherLeavesNoLiveThread() throws Exception {
    BatchGenerationScheduler scheduler = start("llama", config(2, 2));
    BatchRequestHandle handle = scheduler.submit(greedy(PROMPT, 3), e -> {});
    CountDownLatch observed = new CountDownLatch(1);
    handle
        .stage()
        .thenRun(
            () -> {
              scheduler.close(); // on the dispatcher: cannot join itself, must not throw or hang
              observed.countDown();
            });
    assertTrue(observed.await(60, TimeUnit.SECONDS));
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(60);
    while (scheduler.state() != BatchGenerationScheduler.State.CLOSED
        && System.nanoTime() < deadline) {
      Thread.sleep(5);
    }
    assertEquals(BatchGenerationScheduler.State.CLOSED, scheduler.state());
    assertNoSchedulerThreads();
  }

  @Test
  void anInterruptedCloseKeepsWaitingAndRestoresTheFlagWithoutThrowing() throws Exception {
    BatchGenerationScheduler scheduler = start("llama", config(2, 2));
    CountDownLatch inCallback = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    scheduler.submit(
        greedy(PROMPT, 100),
        e -> {
          inCallback.countDown();
          try {
            release.await(30, TimeUnit.SECONDS);
          } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
          }
        });
    assertTrue(inCallback.await(60, TimeUnit.SECONDS));
    AtomicBoolean flagAfter = new AtomicBoolean();
    AtomicReference<Throwable> thrown = new AtomicReference<>();
    CountDownLatch returned = new CountDownLatch(1);
    Thread closer =
        new Thread(
            () -> {
              try {
                Thread.currentThread().interrupt();
                scheduler.close();
                flagAfter.set(Thread.currentThread().isInterrupted());
              } catch (Throwable t) {
                thrown.set(t);
              } finally {
                returned.countDown();
              }
            });
    closer.start();
    assertFalse(returned.await(500, TimeUnit.MILLISECONDS), "close() must keep waiting");
    release.countDown();
    assertTrue(returned.await(60, TimeUnit.SECONDS));
    assertEquals(null, thrown.get());
    assertTrue(flagAfter.get(), "the interrupt flag is restored");
  }

  @Test
  void startInterruptedDuringASlowFactoryStillReturnsAUsableScheduler() throws Exception {
    AtomicReference<BatchGenerationScheduler> started = new AtomicReference<>();
    AtomicBoolean flag = new AtomicBoolean();
    AtomicReference<Throwable> thrown = new AtomicReference<>();
    CountDownLatch inFactory = new CountDownLatch(1);
    CountDownLatch finishFactory = new CountDownLatch(1);
    Thread starter =
        new Thread(
            () -> {
              try {
                started.set(
                    BatchGenerationScheduler.start(
                        config(2, 2),
                        root -> {
                          inFactory.countDown();
                          finishFactory.await(30, TimeUnit.SECONDS);
                          return TextGenerationModels.load(
                              root, SchedulerFixtures.checkpoint("llama"));
                        }));
                flag.set(Thread.currentThread().isInterrupted());
              } catch (Throwable t) {
                thrown.set(t);
              }
            });
    starter.start();
    assertTrue(inFactory.await(60, TimeUnit.SECONDS));
    starter.interrupt();
    // start() cannot have returned yet: the ready latch is only released once the factory
    // returns, and the factory is still blocked below, so this is a sanity check that can never
    // fail -- the "uninterruptible" part of the contract is proven by the flag being restored
    // and the scheduler being usable after the join below.
    assertNull(started.get(), "sanity check only: start() must keep waiting through the interrupt");
    finishFactory.countDown();
    starter.join(TimeUnit.SECONDS.toMillis(60));
    assertEquals(null, thrown.get());
    assertTrue(flag.get(), "the interrupt flag is restored");
    try (BatchGenerationScheduler scheduler = started.get()) {
      assertNotNull(await(scheduler.submit(greedy(PROMPT, 2), e -> {})));
    }
  }

  @Test
  void everyStartupFailureReleasesTheGuardBeforeStartReturns() throws Exception {
    // A checked exception, an unchecked one, a wrong type, and a model in an unrelated scope.
    assertThrows(
        SchedulerStartException.class,
        () ->
            BatchGenerationScheduler.start(
                config(2, 2),
                root -> {
                  throw new java.io.IOException("no such checkpoint");
                }));
    assertThrows(
        IllegalStateException.class,
        () ->
            BatchGenerationScheduler.start(
                config(2, 2),
                root -> {
                  throw new IllegalStateException("unchecked factory failure");
                }));
    assertThrows(
        SchedulerStartException.class,
        () -> BatchGenerationScheduler.start(config(2, 2), root -> null));
    assertThrows(
        SchedulerStartException.class,
        () ->
            BatchGenerationScheduler.start(
                config(2, 2),
                root -> {
                  // A model loaded into a scope the worker's root does not own.
                  MLXScope unrelated = new MLXScope();
                  return TextGenerationModels.load(
                      unrelated, SchedulerFixtures.checkpoint("llama"));
                }));
    // An immediate retry after each failure is accepted: start() joined the worker.
    try (BatchGenerationScheduler ok = start("llama", config(2, 2))) {
      assertEquals(BatchGenerationScheduler.State.RUNNING, ok.state());
    }
  }

  @Test
  void throwingCleanupFailsTheSchedulerButDoesNotBlockARestart() throws Exception {
    BatchGenerationScheduler scheduler =
        SchedulerFixtures.start(
            "llama",
            config(2, 2),
            BatchGenerationScheduler.Hooks.NONE.withCleanupFault(
                () -> {
                  throw new IllegalStateException("injected close failure");
                }));
    assertNotNull(await(scheduler.submit(greedy(PROMPT, 2), e -> {})));
    scheduler.close();
    assertEquals(BatchGenerationScheduler.State.FAILED, scheduler.state());
    assertTrue(scheduler.failure().isPresent());
    assertSame(IllegalStateException.class, scheduler.failure().get().getClass());
    // Every submit after the failure throws its own SchedulerFailedException: the shared
    // instance carries the worker's stack trace and callers on different threads could modify
    // it concurrently (addSuppressed from try-with-resources, initCause).
    SchedulerFailedException first =
        assertThrows(
            SchedulerFailedException.class, () -> scheduler.submit(greedy(PROMPT, 2), e -> {}));
    SchedulerFailedException second =
        assertThrows(
            SchedulerFailedException.class, () -> scheduler.submit(greedy(PROMPT, 2), e -> {}));
    assertNotSame(first, second);
    assertSame(first.getCause(), second.getCause());
    scheduler.close(); // returns normally after a failure, state stays FAILED
    assertEquals(BatchGenerationScheduler.State.FAILED, scheduler.state());
    // The worker has exited, so it is not a concurrent MLX thread: a restart is accepted.
    try (BatchGenerationScheduler again = start("llama", config(2, 2))) {
      assertEquals(BatchGenerationScheduler.State.RUNNING, again.state());
    }
  }

  private static void assertNoSchedulerThreads() throws InterruptedException {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
    while (System.nanoTime() < deadline) {
      boolean alive =
          Thread.getAllStackTraces().keySet().stream()
              .anyMatch(
                  t ->
                      t.isAlive()
                          && (t.getName().equals("jmlx-batch-worker")
                              || t.getName().equals("jmlx-batch-completion")));
      if (!alive) {
        return;
      }
      Thread.sleep(20);
    }
    throw new AssertionError("a scheduler thread is still alive");
  }
}
