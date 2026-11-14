package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static se.alipsa.jmlx.models.SchedulerFixtures.await;
import static se.alipsa.jmlx.models.SchedulerFixtures.checkpoint;
import static se.alipsa.jmlx.models.SchedulerFixtures.config;
import static se.alipsa.jmlx.models.SchedulerFixtures.greedy;
import static se.alipsa.jmlx.models.SchedulerFixtures.start;

import java.util.concurrent.CompletionStage;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/** Worker operations must share admission, ownership and shutdown with generation. */
@EnabledIfNativeAvailable
class SchedulerOperationTest {
  @Test
  void operationsUseTheSameThreadAsChatAndFailuresLeaveWorkerUsable() throws Exception {
    AtomicReference<Thread> generationThread = new AtomicReference<>();
    try (BatchGenerationScheduler scheduler = start("mistral", config(2, 4))) {
      await(
          scheduler.submit(
              greedy(new int[] {1, 3}, 2), e -> generationThread.set(Thread.currentThread())));
      assertEquals(
          generationThread.get(),
          scheduler
              .execute(root -> Thread.currentThread())
              .toCompletableFuture()
              .get(60, TimeUnit.SECONDS));
      CompletionStage<Object> bad =
          scheduler.execute(
              root -> {
                throw new IllegalArgumentException("bad encoder");
              });
      assertThrows(
          java.util.concurrent.ExecutionException.class,
          () -> bad.toCompletableFuture().get(60, TimeUnit.SECONDS));
      assertEquals(
          42, scheduler.execute(root -> 42).toCompletableFuture().get(60, TimeUnit.SECONDS));
      await(scheduler.submit(greedy(new int[] {1, 3}, 2), e -> {}));
    }
  }

  @Test
  void closeRejectsQueuedOperationsAndCompletesEveryAcceptedStage() throws Exception {
    CountDownLatch entered = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    BatchGenerationScheduler scheduler = start("mistral", config(1, 2));
    try {
      final CompletionStage<Integer> active =
          scheduler.execute(
              root -> {
                entered.countDown();
                if (!release.await(60, TimeUnit.SECONDS)) {
                  throw new IllegalStateException("test timed out");
                }
                return 7;
              });
      assertTrue(entered.await(60, TimeUnit.SECONDS));
      final CompletionStage<Integer> pending = scheduler.execute(root -> 8);
      final CompletionStage<Integer> other = scheduler.execute(root -> 9);
      assertThrows(BatchAdmissionRejectedException.class, () -> scheduler.execute(root -> 10));
      Thread closer = Thread.ofVirtual().start(scheduler::close);
      try {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(60);
        while (scheduler.state() == BatchGenerationScheduler.State.RUNNING
            && System.nanoTime() < deadline) {
          Thread.onSpinWait();
        }
        assertEquals(BatchGenerationScheduler.State.CLOSING, scheduler.state());
      } finally {
        release.countDown();
      }
      closer.join(TimeUnit.SECONDS.toMillis(60));
      assertFalse(closer.isAlive());
      assertEquals(7, active.toCompletableFuture().get(60, TimeUnit.SECONDS));
      var firstFailure =
          assertThrows(
              java.util.concurrent.ExecutionException.class,
              () -> pending.toCompletableFuture().get(60, TimeUnit.SECONDS));
      var otherFailure =
          assertThrows(
              java.util.concurrent.ExecutionException.class,
              () -> other.toCompletableFuture().get(60, TimeUnit.SECONDS));
      org.junit.jupiter.api.Assertions.assertNotSame(
          firstFailure.getCause(), otherFailure.getCause());
      assertThrows(SchedulerClosedException.class, () -> scheduler.execute(root -> 10));
    } finally {
      release.countDown();
      scheduler.close();
    }
  }

  @Test
  void workerRootReleasesEncoderWeightsOnClose() throws Exception {
    long before = MLXMemory.activeBytes();
    try (BatchGenerationScheduler scheduler = start("mistral", config(1, 4))) {
      long active =
          scheduler
              .execute(
                  root -> {
                    TextEncoderModel encoder =
                        TextEncoderModels.load(root.newChild(), checkpoint("bert"));
                    encoder.encode(
                        new se.alipsa.jmlx.tokenizer.TokenizerEncoding(
                            java.util.List.of(2, 4, 3),
                            java.util.List.of(0, 0, 0),
                            java.util.List.of(1, 1, 1),
                            java.util.List.of(1, 0, 1),
                            java.util.Collections.nCopies(
                                3, se.alipsa.jmlx.tokenizer.TokenOffset.NONE)));
                    return MLXMemory.activeBytes();
                  })
              .toCompletableFuture()
              .get(60, TimeUnit.SECONDS);
      assertTrue(active > before);
    }
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
    while (MLXMemory.activeBytes() != before && System.nanoTime() < deadline) {
      Thread.sleep(1);
    }
    assertEquals(before, MLXMemory.activeBytes());
  }
}
