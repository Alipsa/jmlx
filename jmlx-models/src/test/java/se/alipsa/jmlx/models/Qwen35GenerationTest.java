package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.RgbImage;

/** Public hybrid generation contracts, capability and cancellation in default precision. */
@EnabledIfNativeAvailable
class Qwen35GenerationTest {
  private static final Path ROOT = Path.of("..", "tools", "qwen35-reference");

  @Test
  void dispatchCapabilitiesAndTextOnlySchedulerRejectionFollowActualInventory() throws Exception {
    for (String kind : new String[] {"hf", "vlm-float", "vlm-affine", "lm-float", "lm-affine"}) {
      Path directory = ROOT.resolve("fixtures/" + kind);
      try (MLXScope scope = new MLXScope()) {
        TextGenerationModel loaded = TextGenerationModels.load(scope, directory);
        Qwen35Model model = assertInstanceOf(Qwen35Model.class, loaded);
        assertFalse(loaded instanceof DecoderModel);
        assertEquals("qwen3_5", model.metadata().modelType());
        assertEquals(
            kind.startsWith("lm")
                ? Set.of(InputModality.TEXT)
                : Set.of(InputModality.TEXT, InputModality.IMAGE),
            model.metadata().inputModalities());
        if (kind.startsWith("lm")) {
          assertThrows(
              IllegalArgumentException.class, () -> model.generate(imageRequest(0), ignored -> {}));
        }
      }
      SchedulerStartException error =
          assertThrows(
              SchedulerStartException.class,
              () ->
                  BatchGenerationScheduler.start(
                      BatchSchedulerConfig.defaults(),
                      scope -> TextGenerationModels.load(scope, directory)));
      assertTrue(error.getMessage().contains("not supported by the batch scheduler"));
    }
  }

  @Test
  void zeroTokenAndInvalidRequestsValidateCapacityWithoutRunningNativeInference() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Qwen35Model model = Qwen35Model.load(scope, ROOT.resolve("fixtures/hf"));
      List<String> stages = new ArrayList<>();
      model.observerForTest(stages::add);
      GenerationRequest request = imageRequest(0);
      List<GenerationEvent> events = new ArrayList<>();
      GenerationResult result = model.generate(request, events::add);
      assertEquals(List.of(), stages);
      assertEquals(List.of(GenerationEvent.finished(FinishReason.MAX_TOKENS)), events);
      assertEquals(List.of(), result.generatedTokenIds());
      int length = result.promptPositions();
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  request.withCachePolicy(GenerationCachePolicy.full(length - 1)), ignored -> {}));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  request.withCachePolicy(GenerationCachePolicy.slidingWindow(32)), ignored -> {}));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  request.withCachePolicy(GenerationCachePolicy.slidingWindowFromModel()),
                  ignored -> {}));
      assertThrows(
          IllegalArgumentException.class,
          () -> model.generate(text(new int[] {121}, 0), ignored -> {}));
      assertThrows(
          IllegalArgumentException.class,
          () -> model.generate(text(new int[] {120}, 0), ignored -> {}));
      assertThrows(
          IllegalArgumentException.class,
          () -> model.generate(text(new int[] {128}, 0), ignored -> {}));
      assertEquals(List.of(), stages);
    }
  }

  @Test
  void sharedSamplingEventsStopsAbortAndTokenizerContractsRemainRequestLocal() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Path directory = ROOT.resolve("fixtures/lm-affine");
      Qwen35Model model = Qwen35Model.load(scope, directory);
      GenerationRequest request = text(new int[] {1, 3, 4, 5}, 3);
      GenerationResult baseline = model.generate(request, ignored -> {});
      int first = baseline.generatedTokenIds().getFirst();
      List<GenerationEvent> eosEvents = new ArrayList<>();
      GenerationResult eos =
          model.generate(
              new GenerationRequest(
                  request.promptTokenIds(),
                  GenerationConfig.greedyDefaults(3, Set.of(first), Set.of(first)),
                  CancellationToken.NONE),
              eosEvents::add);
      assertEquals(FinishReason.EOS, eos.finishReason());
      assertEquals(List.of(first), eos.generatedTokenIds());
      assertEquals(2, eosEvents.size());
      GenerationResult stop =
          model.generate(
              new GenerationRequest(
                  request.promptTokenIds(),
                  GenerationConfig.greedyDefaults(3, Set.of(), Set.of(first)),
                  CancellationToken.NONE),
              ignored -> {});
      assertEquals(FinishReason.STOP_TOKEN, stop.finishReason());
      assertEquals(List.of(), stop.generatedTokenIds());
      GenerationConfig sampled =
          new GenerationConfig(
              5, OptionalLong.of(42), .8f, 8, .9f, .01f, 1.1f, .1f, .1f, Set.of(), Set.of(), true);
      GenerationRequest sampling =
          new GenerationRequest(request.promptTokenIds(), sampled, CancellationToken.NONE);
      GenerationResult a = model.generate(sampling, ignored -> {});
      GenerationResult b = model.generate(sampling, ignored -> {});
      assertEquals(a, b);
      assertEquals(5, a.logProbabilities().size());
      assertThrows(
          GenerationAbortedException.class,
          () ->
              model.generate(
                  request,
                  event -> {
                    if (event.tokenId() != null) {
                      throw new IllegalStateException("injected token listener failure");
                    }
                  }));
      assertEquals(baseline, model.generate(request, ignored -> {}));
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
      GenerationRequest encoded =
          GenerationRequest.text(
              tokenizer,
              "<bos>AB",
              PromptSpecialTokens.OMIT,
              GenerationConfig.greedyDefaults(3, Set.of()),
              CancellationToken.NONE);
      List<GenerationEvent> events = new ArrayList<>();
      GenerationResult decoded = model.generate(encoded, events::add);
      assertEquals(
          decoded.generatedText(),
          events.stream()
              .map(GenerationEvent::textDelta)
              .collect(java.util.stream.Collectors.joining()));
    }
  }

  @Test
  void cancellationBetweenImagesChunksAndDecodeStepsReleasesRequestState() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Qwen35Model model = Qwen35Model.load(scope, ROOT.resolve("fixtures/hf"));
      AtomicBoolean cancelled = new AtomicBoolean();
      AtomicInteger images = new AtomicInteger();
      model.observerForTest(
          stage -> {
            if (stage.equals("image-closed") && images.incrementAndGet() == 1) {
              cancelled.set(true);
            }
          });
      List<RgbImage> two = List.of(image(), image());
      GenerationRequest imageRequest =
          new GenerationRequest(
                  new int[] {1, 118, 120, 119, 37, 118, 120, 119, 38},
                  GenerationConfig.greedyDefaults(3, Set.of()),
                  cancelled::get)
              .withImages(two);
      GenerationResult imageResult = model.generate(imageRequest, ignored -> {});
      assertEquals(FinishReason.CANCELLED, imageResult.finishReason());
      assertEquals(List.of(), imageResult.generatedTokenIds());
      assertEquals(1, images.get());
      AtomicBoolean armed = new AtomicBoolean();
      AtomicInteger checks = new AtomicInteger();
      model.observerForTest(
          stage -> {
            if (stage.equals("before-prefill")) {
              armed.set(true);
            }
          });
      int[] longPrompt = new int[129];
      java.util.Arrays.fill(longPrompt, 37);
      GenerationResult chunkResult =
          model.generate(
              new GenerationRequest(
                  longPrompt,
                  GenerationConfig.greedyDefaults(1, Set.of()),
                  () -> armed.get() && checks.incrementAndGet() >= 6),
              ignored -> {});
      assertEquals(FinishReason.CANCELLED, chunkResult.finishReason());
      assertTrue(checks.get() >= 6);
      assertEquals(List.of(), chunkResult.generatedTokenIds());
      model.observerForTest(ignored -> {});
      cancelled.set(false);
      GenerationResult decodeResult =
          model.generate(
              new GenerationRequest(
                  new int[] {1, 3, 4, 5},
                  GenerationConfig.greedyDefaults(5, Set.of()),
                  cancelled::get),
              event -> {
                if (event.tokenId() != null) {
                  cancelled.set(true);
                }
              });
      assertEquals(FinishReason.CANCELLED, decodeResult.finishReason());
      assertEquals(1, decodeResult.generatedTokenIds().size());
      assertEquals(
          3,
          model
              .generate(text(new int[] {1, 3, 4, 5}, 3), ignored -> {})
              .generatedTokenIds()
              .size());
    }
  }

  private static GenerationRequest text(int[] ids, int maximum) {
    return new GenerationRequest(
        ids, GenerationConfig.greedyDefaults(maximum, Set.of()), CancellationToken.NONE);
  }

  private static GenerationRequest imageRequest(int maximum) throws Exception {
    return text(new int[] {1, 118, 120, 119, 37, 38}, maximum).withImages(List.of(image()));
  }

  private static RgbImage image() throws Exception {
    return ImageDecoder.decode(ROOT.resolve("fixtures/images/rgb-73x45.png"));
  }
}
