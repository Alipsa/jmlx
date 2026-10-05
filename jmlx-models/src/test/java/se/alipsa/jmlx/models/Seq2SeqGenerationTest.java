package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class Seq2SeqGenerationTest {
  static Path checkpoint() {
    return BertGoldenTest.root().resolve("tools/hf-reference/goldens/checkpoints/t5-gated");
  }

  static GenerationRequest request(int tokens) {
    return new GenerationRequest(
        new int[] {5, 7, 9, 1},
        GenerationConfig.greedyDefaults(tokens, Set.of()),
        CancellationToken.NONE);
  }

  @Test
  void capacitySourceLimitCancellationAndListenerSemantics() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      T5Model model = T5Model.load(scope, checkpoint());
      List<GenerationEvent> events = new ArrayList<>();
      GenerationResult result = model.generate(request(3), events::add);
      assertEquals(3, result.generatedTokenIds().size());
      assertEquals(4, events.size());
      assertEquals(
          result.generatedTokenIds(),
          events.stream().filter(e -> e.tokenId() != null).map(GenerationEvent::tokenId).toList());
      int first = result.generatedTokenIds().getFirst();
      GenerationRequest overlap =
          new GenerationRequest(
              new int[] {5, 7, 9, 1},
              GenerationConfig.greedyDefaults(3, Set.of(first), Set.of(first)),
              CancellationToken.NONE);
      assertEquals(FinishReason.EOS, model.generate(overlap, ignored -> {}).finishReason());
      GenerationRequest stop =
          new GenerationRequest(
              new int[] {5, 7, 9, 1},
              GenerationConfig.greedyDefaults(3, Set.of(), Set.of(first)),
              CancellationToken.NONE);
      assertEquals(List.of(), model.generate(stop, ignored -> {}).generatedTokenIds());
      assertEquals(FinishReason.STOP_TOKEN, model.generate(stop, ignored -> {}).finishReason());
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  request(3).withCachePolicy(GenerationCachePolicy.full(2)), ignored -> {}));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  request(3).withCachePolicy(GenerationCachePolicy.slidingWindow(2)),
                  ignored -> {}));
      T5Model limited = T5Model.load(scope, checkpoint(), new T5LoadOptions(2));
      assertThrows(
          IllegalArgumentException.class, () -> limited.generate(request(1), ignored -> {}));
      AtomicBoolean cancelled = new AtomicBoolean();
      GenerationRequest cancellation =
          new GenerationRequest(
              new int[] {5, 7, 9, 1}, GenerationConfig.greedyDefaults(5, Set.of()), cancelled::get);
      result =
          model.generate(
              cancellation,
              event -> {
                if (event.tokenId() != null) {
                  cancelled.set(true);
                }
              });
      assertEquals(FinishReason.CANCELLED, result.finishReason());
      assertEquals(1, result.generatedTokenIds().size());
      cancelled.set(false);
      model.projectionObserver((layer, cache) -> cancelled.set(true));
      result = model.generate(cancellation, ignored -> {});
      assertEquals(FinishReason.CANCELLED, result.finishReason());
      assertEquals(List.of(), result.generatedTokenIds());
      model.projectionObserver((layer, cache) -> {});
      GenerationAbortedException aborted =
          assertThrows(
              GenerationAbortedException.class,
              () ->
                  model.generate(
                      request(3),
                      event -> {
                        if (event.tokenId() != null) {
                          throw new IllegalStateException("listener");
                        }
                      }));
      assertEquals(1, aborted.generatedTokenIds().size());
      assertEquals(
          1,
          model
              .generate(
                  request(1),
                  event -> {
                    if (event.tokenId() == null) {
                      throw new IllegalStateException("terminal");
                    }
                  })
              .generatedTokenIds()
              .size());
    }
  }

  @Test
  void samplingIsRequestLocalAndReportsFilteredProbabilities() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      T5Model model = T5Model.load(scope, checkpoint());
      GenerationConfig config =
          new GenerationConfig(
              5,
              OptionalLong.of(42),
              0.7f,
              5,
              0.8f,
              0.1f,
              1.3f,
              0.2f,
              0.1f,
              Set.of(),
              Set.of(),
              true);
      GenerationRequest sampled =
          new GenerationRequest(new int[] {5, 7, 9, 1}, config, CancellationToken.NONE);
      GenerationResult first = model.generate(sampled, ignored -> {});
      GenerationResult second = model.generate(sampled, ignored -> {});
      assertEquals(first.generatedTokenIds(), second.generatedTokenIds());
      assertEquals(first.logProbabilities(), second.logProbabilities());
      assertEquals(5, first.logProbabilities().size());
      assertTrue(first.logProbabilities().stream().allMatch(p -> Double.isFinite(p) && p <= 0));
    }
  }

  @Test
  void penaltiesTrackDecoderStartAndEmittedTargetsOnly() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      T5Model model = T5Model.load(scope, checkpoint());
      List<java.util.Map<Integer, Integer>> histories = new ArrayList<>();
      model.penaltyObserver(histories::add);
      GenerationConfig policy =
          new GenerationConfig(
              3, OptionalLong.empty(), 0, 0, 1, 0, 1.4f, 0.3f, 0.2f, Set.of(), Set.of(), false);
      GenerationResult result =
          model.generate(
              new GenerationRequest(new int[] {5, 5, 5, 7, 1}, policy, CancellationToken.NONE),
              ignored -> {});
      java.util.Map<Integer, Integer> expected = new java.util.HashMap<>();
      expected.put(0, 1);
      for (int i = 0; i < histories.size(); i++) {
        assertEquals(expected, histories.get(i));
        expected.merge(result.generatedTokenIds().get(i), 1, Integer::sum);
      }
      assertEquals(3, histories.size());
    }
  }

  @Test
  void unreadableConfigJsonFailsWithIOException(@TempDir Path directory) throws Exception {
    // A malformed or missing config.json is a checked IOException on every architecture, not an
    // unchecked Jackson 3 exception leaking from the loader internals.
    Files.writeString(directory.resolve("config.json"), "{\"model_type\": ");
    try (MLXScope scope = new MLXScope()) {
      IOException malformed =
          assertThrows(IOException.class, () -> TextGenerationModels.load(scope, directory));
      assertTrue(malformed.getMessage().contains("failed to read"));
      Files.delete(directory.resolve("config.json"));
      assertThrows(IOException.class, () -> TextGenerationModels.load(scope, directory));
      Files.writeString(directory.resolve("config.json"), "{\"model_type\": ");
      assertThrows(
          IOException.class,
          () -> TextGenerationModels.load(scope, directory, T5LoadOptions.defaults()));
      assertThrows(IOException.class, () -> T5Model.load(scope, directory));
    }
  }

  @Test
  void schedulerNamesUnsupportedModelAndReleasesGuard() throws Exception {
    SchedulerStartException exception =
        assertThrows(
            SchedulerStartException.class,
            () ->
                BatchGenerationScheduler.start(
                    BatchSchedulerConfig.defaults(), scope -> T5Model.load(scope, checkpoint())));
    assertTrue(exception.getMessage().contains("t5"));
    assertTrue(exception.getMessage().contains("not supported by the batch scheduler"));
    assertThrows(
        SchedulerStartException.class,
        () -> BatchGenerationScheduler.start(BatchSchedulerConfig.defaults(), scope -> null));
  }
}
