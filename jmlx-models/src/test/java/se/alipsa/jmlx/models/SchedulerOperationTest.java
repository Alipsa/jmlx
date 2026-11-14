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

  @Test
  void embeddingBatchRetainsAdmissionWhenGenerationFillsTheQueue() throws Exception {
    CountDownLatch firstText = new CountDownLatch(1);
    CountDownLatch continueBatch = new CountDownLatch(1);
    java.util.List<String> trace = new java.util.concurrent.CopyOnWriteArrayList<>();
    AtomicReference<TextEncoderModel> encoder = new AtomicReference<>();
    var tokenizer =
        se.alipsa.jmlx.tokenizer.HfTokenizer.fromDirectory(
            java.nio.file.Path.of(
                System.getProperty("jmlx.repository.root"),
                "tools",
                "release-smoke",
                "fixtures",
                "bert"));
    var input =
        tokenizer.encode("hello world", se.alipsa.jmlx.tokenizer.EncodingOptions.unbounded(true));
    BatchGenerationScheduler scheduler = start("mistral", config(1, 4));
    try {
      final var batch =
          scheduler.<float[]>executeSequence(
              java.util.List.of(
                  root -> {
                    encoder.set(TextEncoderModels.load(root.newChild(), checkpoint("bert")));
                    final float[] vector = encoder.get().embed(input);
                    trace.add("embedding 0");
                    firstText.countDown();
                    if (!continueBatch.await(60, TimeUnit.SECONDS)) {
                      throw new IllegalStateException("test timed out");
                    }
                    return vector;
                  },
                  root -> {
                    trace.add("embedding 1");
                    return encoder.get().embed(input);
                  },
                  root -> {
                    trace.add("embedding 2");
                    return encoder.get().embed(input);
                  }),
              CancellationToken.NONE);
      assertTrue(firstText.await(60, TimeUnit.SECONDS));
      // One batch is the configured encoder share (one quarter of four queue slots).
      assertThrows(
          BatchAdmissionRejectedException.class,
          () -> scheduler.executeSequence(java.util.List.of(root -> true), CancellationToken.NONE));
      java.util.List<BatchRequestHandle> generation = new java.util.ArrayList<>();
      for (int i = 0; i < 3; i++) {
        generation.add(
            scheduler.submit(
                greedy(new int[] {1, 3}, 1),
                event -> {
                  if (event.tokenId() != null) {
                    trace.add("generation");
                  }
                }));
      }
      assertThrows(
          BatchAdmissionRejectedException.class,
          () -> scheduler.submit(greedy(new int[] {1, 3}, 1), event -> {}));
      continueBatch.countDown();
      var vectors = batch.toCompletableFuture().get(60, TimeUnit.SECONDS);
      assertEquals(3, vectors.size());
      org.junit.jupiter.api.Assertions.assertArrayEquals(vectors.get(0), vectors.get(2));
      assertTrue(trace.indexOf("generation") < trace.indexOf("embedding 2"), trace.toString());
      for (BatchRequestHandle handle : generation) {
        await(handle);
      }
      // The single reservation was returned at completion; another batch can start immediately.
      assertEquals(
          java.util.List.of(42),
          scheduler
              .executeSequence(java.util.List.of(root -> 42), CancellationToken.NONE)
              .toCompletableFuture()
              .get(60, TimeUnit.SECONDS));
    } finally {
      continueBatch.countDown();
      scheduler.close();
    }
  }

  @Test
  void cancellingRunningSequenceSkipsRemainingTextsAndReturnsItsReservation() throws Exception {
    CountDownLatch started = new CountDownLatch(1);
    CountDownLatch finish = new CountDownLatch(1);
    java.util.concurrent.atomic.AtomicBoolean cancelled =
        new java.util.concurrent.atomic.AtomicBoolean();
    java.util.concurrent.atomic.AtomicBoolean laterText =
        new java.util.concurrent.atomic.AtomicBoolean();
    BatchGenerationScheduler scheduler = start("mistral", config(1, 4));
    try {
      final var batch =
          scheduler.executeSequence(
              java.util.List.of(
                  root -> {
                    started.countDown();
                    if (!finish.await(60, TimeUnit.SECONDS)) {
                      throw new IllegalStateException("test timed out");
                    }
                    return 1;
                  },
                  root -> {
                    laterText.set(true);
                    return 2;
                  }),
              cancelled::get);
      assertTrue(started.await(60, TimeUnit.SECONDS));
      cancelled.set(true);
      finish.countDown();
      assertThrows(
          java.util.concurrent.ExecutionException.class,
          () -> batch.toCompletableFuture().get(60, TimeUnit.SECONDS));
      assertFalse(laterText.get());
      assertEquals(
          java.util.List.of(7),
          scheduler
              .executeSequence(java.util.List.of(root -> 7), CancellationToken.NONE)
              .toCompletableFuture()
              .get(60, TimeUnit.SECONDS));
    } finally {
      finish.countDown();
      scheduler.close();
    }
  }
}
