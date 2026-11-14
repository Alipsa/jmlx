package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static se.alipsa.jmlx.models.SchedulerFixtures.checkpoint;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

@EnabledIfNativeAvailable
class SchedulerInferenceClientTest {
  @TempDir Path temporary;

  private Path fixture(String name) {
    return Path.of(System.getProperty("jmlx.repository.root"), "tools", "release-smoke", "fixtures")
        .resolve(name);
  }

  private Path encoderDirectory() throws Exception {
    Path encoder = Files.createDirectory(temporary.resolve("bert"));
    for (String name : List.of("config.json", "model.safetensors")) {
      Files.copy(checkpoint("bert").resolve(name), encoder.resolve(name));
    }
    for (String name : List.of("tokenizer.json", "tokenizer_config.json")) {
      Files.copy(fixture("bert").resolve(name), encoder.resolve(name));
    }
    return encoder;
  }

  private SchedulerInferenceClient client(BatchGenerationScheduler scheduler, Path encoder) {
    return new SchedulerInferenceClient(
        scheduler,
        HfTokenizer.fromDirectory(fixture("")),
        "tiny",
        GenerationConfig.greedyDefaults(8, Set.of()),
        encoder,
        new Pooling(Pooling.Mode.MEAN, true));
  }

  private static BatchGenerationScheduler scheduler() {
    return BatchGenerationScheduler.start(
        BatchSchedulerConfig.defaults(),
        root -> TextGenerationModels.load(root, checkpoint("mistral")));
  }

  private static List<Map<String, Object>> messages() {
    return List.of(Map.of("role", "user", "content", "the fox and the hen"));
  }

  @Test
  void repeatedEmbeddingsReleaseScratchAndConcurrentChatUsesSameRuntime() throws Exception {
    Path encoder = encoderDirectory();
    BatchGenerationScheduler scheduler = scheduler();
    try (SchedulerInferenceClient client = client(scheduler, encoder)) {
      var chat = client.chat(messages(), client.defaults(), text -> {});
      var first =
          client
              .embed(List.of("hello world", "another sentence"))
              .toCompletableFuture()
              .get(60, TimeUnit.SECONDS);
      assertEquals(8, first.inputTokens());
      assertEquals(16, first.vectors().getFirst().length);
      assertEquals(
          8,
          chat.stage().toCompletableFuture().get(60, TimeUnit.SECONDS).generatedTokenIds().size());
      long before =
          scheduler
              .execute(root -> MLXMemory.activeBytes())
              .toCompletableFuture()
              .get(60, TimeUnit.SECONDS);
      for (int i = 0; i < 12; i++) {
        var next =
            client
                .embed(List.of("hello world", "another sentence"))
                .toCompletableFuture()
                .get(60, TimeUnit.SECONDS);
        assertArrayEquals(first.vectors().getFirst(), next.vectors().getFirst());
      }
      long after =
          scheduler
              .execute(root -> MLXMemory.activeBytes())
              .toCompletableFuture()
              .get(60, TimeUnit.SECONDS);
      assertTrue(after <= before + 1024 * 1024, "encoder scratch grew across requests");
    }
  }

  @Test
  void throwingConsumerFailsItsRequestAndLeavesWorkerUsable() throws Exception {
    try (SchedulerInferenceClient client = client(scheduler(), null)) {
      var failed =
          client.chat(
              messages(),
              client.defaults(),
              text -> {
                throw new IllegalArgumentException("bad consumer");
              });
      ExecutionException failure =
          assertThrows(
              ExecutionException.class,
              () -> failed.stage().toCompletableFuture().get(60, TimeUnit.SECONDS));
      assertEquals("bad consumer", failure.getCause().getMessage());
      assertEquals(
          8,
          client
              .chat(messages(), client.defaults(), text -> {})
              .stage()
              .toCompletableFuture()
              .get(60, TimeUnit.SECONDS)
              .generatedTokenIds()
              .size());
    }
  }

  @Test
  void slowConsumerCannotBlockWorkerOrAccumulateUnboundedDeltas() throws Exception {
    BatchGenerationScheduler scheduler = scheduler();
    CountDownLatch releaseConsumer = new CountDownLatch(1);
    CountDownLatch consumerEntered = new CountDownLatch(1);
    AtomicBoolean first = new AtomicBoolean(true);
    try (SchedulerInferenceClient client = client(scheduler, null)) {
      var call =
          client.chat(
              messages(),
              GenerationConfig.greedyDefaults(256, Set.of()),
              text -> {
                if (first.getAndSet(false)) {
                  consumerEntered.countDown();
                  try {
                    if (!releaseConsumer.await(60, TimeUnit.SECONDS)) {
                      throw new IllegalStateException("consumer test timed out");
                    }
                  } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(failure);
                  }
                }
              });
      assertTrue(consumerEntered.await(60, TimeUnit.SECONDS));
      // Queued encoder work must run even while a callback is blocked. Enough output accumulates
      // first to exercise the 64-delta bound and cancel the generation.
      scheduler
          .execute(
              root -> {
                releaseConsumer.countDown();
                return true;
              })
          .toCompletableFuture()
          .get(60, TimeUnit.SECONDS);
      ExecutionException error =
          assertThrows(
              ExecutionException.class,
              () -> call.stage().toCompletableFuture().get(60, TimeUnit.SECONDS));
      assertTrue(error.getCause().getMessage().contains("64 queued deltas"));
    } finally {
      releaseConsumer.countDown();
      scheduler.close();
    }
  }

  @Test
  void completedGenerationsStillCountTowardBoundWhileConsumersAreBlocked() throws Exception {
    BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            new BatchSchedulerConfig(1, 1, 4096, 256),
            root -> TextGenerationModels.load(root, checkpoint("mistral")));
    CountDownLatch entered = new CountDownLatch(2);
    CountDownLatch release = new CountDownLatch(1);
    try (SchedulerInferenceClient client = client(scheduler, null)) {
      java.util.function.Consumer<String> blocked =
          text -> {
            entered.countDown();
            try {
              if (!release.await(60, TimeUnit.SECONDS)) {
                throw new IllegalStateException("consumer test timed out");
              }
            } catch (InterruptedException failure) {
              Thread.currentThread().interrupt();
              throw new IllegalStateException(failure);
            }
          };
      InferenceCall first = client.chat(messages(), client.defaults(), blocked);
      // Wait for the first delivery thread before submitting into the single waiting slot.
      long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(60);
      while (entered.getCount() == 2 && System.nanoTime() < deadline) {
        Thread.sleep(1);
      }
      InferenceCall second = client.chat(messages(), client.defaults(), blocked);
      try {
        assertTrue(entered.await(60, TimeUnit.SECONDS));
        // Both native generations can finish, but their blocked delivery threads remain admitted.
        scheduler.execute(root -> true).toCompletableFuture().get(60, TimeUnit.SECONDS);
        assertThrows(
            BatchAdmissionRejectedException.class,
            () -> client.chat(messages(), client.defaults(), text -> {}));
      } finally {
        release.countDown();
      }
      first.stage().toCompletableFuture().get(60, TimeUnit.SECONDS);
      second.stage().toCompletableFuture().get(60, TimeUnit.SECONDS);
    } finally {
      release.countDown();
      scheduler.close();
    }
  }

  @Test
  void failedEncoderLoadDoesNotPoisonGenerationOrRetainPartialWeights() throws Exception {
    Path invalid = Files.createDirectory(temporary.resolve("invalid"));
    BatchGenerationScheduler scheduler = scheduler();
    try (SchedulerInferenceClient client = client(scheduler, invalid)) {
      for (int i = 0; i < 2; i++) {
        assertThrows(
            ExecutionException.class,
            () -> client.embed(List.of("hello")).toCompletableFuture().get(60, TimeUnit.SECONDS));
      }
      assertEquals(
          8,
          client
              .chat(messages(), client.defaults(), text -> {})
              .stage()
              .toCompletableFuture()
              .get(60, TimeUnit.SECONDS)
              .generatedTokenIds()
              .size());
    }
  }
}
