package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

/** Public generation, request reuse, cancellation boundaries and full-capacity contracts. */
@EnabledIfNativeAvailable
class Qwen3VlGenerationTest {
  @Test
  void mixedRequestsKeepTheirOwnDeltaAndFullCapacityNeverEvicts() throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model = Qwen3VlModel.load(root, Qwen3VlModelFloatGoldenTest.directory());
      var requests = Qwen3VlTestSupport.requests(5);
      List<GenerationResult> expected = new ArrayList<>();
      for (var request : requests) {
        expected.add(model.generate(request, ignored -> {}));
      }
      assertTrue(
          model.plan(requests.get(0).promptTokenIds(), requests.get(0).images()).positionDelta()
              != model
                  .plan(requests.get(1).promptTokenIds(), requests.get(1).images())
                  .positionDelta());
      for (int index : new int[] {0, 3, 2, 1, 3, 0, 2}) {
        var request = requests.get(index);
        int length = model.plan(request.promptTokenIds(), request.images()).expandedLength();
        List<String> stages = new ArrayList<>();
        model.setObserverForTest(stages::add);
        List<GenerationEvent> events = new ArrayList<>();
        assertEquals(
            expected.get(index),
            model.generate(
                request.withCachePolicy(GenerationCachePolicy.full(length + 4)), events::add));
        assertEquals(6, events.size());
        assertEquals(request.images().size(), stages.stream().filter("vision"::equals).count());
        assertEquals(4, stages.stream().filter("decode"::equals).count());
        stages.clear();
        var error =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    model.generate(
                        request.withCachePolicy(GenerationCachePolicy.full(length + 3)),
                        ignored -> {}));
        assertTrue(error.getMessage().contains("expanded prompt " + length));
        assertTrue(error.getMessage().contains("unexpanded " + request.promptTokenIds().length));
        assertTrue(stages.isEmpty());
        for (var sliding :
            List.of(
                GenerationCachePolicy.slidingWindow(16),
                GenerationCachePolicy.slidingWindowFromModel())) {
          assertTrue(
              assertThrows(
                      IllegalArgumentException.class,
                      () -> model.generate(request.withCachePolicy(sliding), ignored -> {}))
                  .getMessage()
                  .contains("sliding"));
        }
      }
    }
  }

  @Test
  void cancellationStopsBeforePixelsBetweenImagesBeforePrefillAndBetweenSteps() throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model = Qwen3VlModel.load(root, Qwen3VlModelFloatGoldenTest.directory());
      var base = Qwen3VlTestSupport.requests(5).get(1);
      for (String boundary :
          List.of(
              "initial",
              "preprocess",
              "image-closed",
              "before-prefill",
              "prefill-closed",
              "decode",
              "listener")) {
        AtomicBoolean cancelled = new AtomicBoolean(boundary.equals("initial"));
        List<String> stages = new ArrayList<>();
        model.setObserverForTest(
            stage -> {
              stages.add(stage);
              if (boundary.equals(stage)) {
                cancelled.set(true);
              }
            });
        var request =
            new GenerationRequest(base.promptTokenIds(), base.config(), cancelled::get)
                .withImages(base.images());
        var result =
            model.generate(
                request,
                event -> {
                  if (boundary.equals("listener") && event.tokenId() != null) {
                    cancelled.set(true);
                  }
                });
        assertEquals(FinishReason.CANCELLED, result.finishReason(), boundary);
        if (boundary.equals("initial")) {
          assertTrue(stages.isEmpty());
        } else if (boundary.equals("image-closed")) {
          assertEquals(1, stages.stream().filter("vision"::equals).count());
          assertEquals(1, stages.stream().filter("preprocess"::equals).count());
        }
        assertTrue(result.generatedTokenIds().size() < 5);
      }
    }
  }

  @Test
  void eosStopSamplingStreamingAndAbortUseTheSharedContracts() throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model = Qwen3VlModel.load(root, Qwen3VlModelFloatGoldenTest.directory());
      var base = Qwen3VlTestSupport.requests(5).getFirst();
      int first = model.generate(base, ignored -> {}).generatedTokenIds().getFirst();
      for (boolean eos : List.of(true, false)) {
        var config =
            GenerationConfig.greedyDefaults(5, eos ? Set.of(first) : Set.of(), Set.of(first));
        var result =
            model.generate(
                new GenerationRequest(base.promptTokenIds(), config, CancellationToken.NONE)
                    .withImages(base.images()),
                ignored -> {});
        assertEquals(eos ? FinishReason.EOS : FinishReason.STOP_TOKEN, result.finishReason());
        assertEquals(eos ? List.of(first) : List.of(), result.generatedTokenIds());
      }
      var tokenizer = HfTokenizer.fromDirectory(Qwen3VlModelFloatGoldenTest.directory());
      var policy =
          new GenerationConfig(
              4,
              OptionalLong.of(42),
              0.7f,
              10,
              0.9f,
              0.05f,
              1.2f,
              0.2f,
              0.1f,
              Set.of(),
              Set.of(),
              true);
      var request =
          GenerationRequest.text(
                  tokenizer,
                  "<|vision_start|><|image_pad|><|vision_end|>text",
                  PromptSpecialTokens.OMIT,
                  policy,
                  CancellationToken.NONE)
              .withImages(base.images());
      List<GenerationEvent> events = new ArrayList<>();
      var sampled = model.generate(request, events::add);
      assertEquals(sampled, model.generate(request, ignored -> {}));
      assertEquals(4, sampled.logProbabilities().size());
      assertEquals(5, events.size());
      assertTrue(sampled.generatedText() != null);
      var aborted =
          assertThrows(
              GenerationAbortedException.class,
              () ->
                  model.generate(
                      base,
                      event -> {
                        if (event.tokenId() != null) {
                          throw new IllegalStateException("listener");
                        }
                      }));
      assertEquals(List.of(first), aborted.generatedTokenIds());
      assertEquals(
          java.util.Arrays.stream(base.promptTokenIds()).boxed().toList(),
          aborted.promptTokenIds());
      assertEquals(5, model.generate(base, ignored -> {}).generatedTokenIds().size());
    }
  }
}
