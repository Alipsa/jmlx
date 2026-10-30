package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/** Public generation, request reuse, capacity, cancellation boundaries and failure contracts. */
@EnabledIfNativeAvailable
class Phi3VGenerationTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void mixedRequestsKeepTheirOwnCapacityAndSlidingIsRejected() throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model = Phi3VModel.load(root, Phi3VTestSupport.directory());
      var requests = Phi3VTestSupport.requests(5);
      List<GenerationResult> expected = new ArrayList<>();
      for (var request : requests) {
        expected.add(model.generate(request, ignored -> {}));
      }
      Set<Integer> lengths = new HashSet<>();
      for (var request : requests) {
        lengths.add(expandedLength(model, request));
      }
      assertEquals(5, lengths.size(), "the mixed set must exercise five distinct prompt lengths");
      for (int index : new int[] {0, 3, 2, 1, 3, 0, 2}) {
        var request = requests.get(index);
        int length = expandedLength(model, request);
        List<String> stages = new ArrayList<>();
        model.setObserverForTest(stages::add);
        List<GenerationEvent> events = new ArrayList<>();
        // The demand is expanded length plus the budget minus one, so exactly one capacity
        // value fits the budget: the request is reusable against it, one below is rejected
        // before any stage fires.
        assertEquals(
            expected.get(index),
            model.generate(
                request.withCachePolicy(GenerationCachePolicy.full(length + 4)), events::add));
        assertEquals(6, events.size());
        assertEquals(request.images().size(), stages.stream().filter("vision"::equals).count());
        assertEquals(
            request.images().size(), stages.stream().filter("image-closed"::equals).count());
        assertEquals(
            request.images().isEmpty() ? 0 : 1, stages.stream().filter("project"::equals).count());
        assertEquals(1, stages.stream().filter("before-prefill"::equals).count());
        assertEquals(1, stages.stream().filter("prefill-evaluated"::equals).count());
        assertEquals(1, stages.stream().filter("prefill-closed"::equals).count());
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
        assertTrue(error.getMessage().contains("exceeds capacity"));
        assertTrue(stages.isEmpty(), "capacity rejection must precede every stage");
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
      model.setObserverForTest(ignored -> {});
    }
  }

  @Test
  void capacityBoundedFullRetentionFitsTheBoundaryRePrefill() throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model = Phi3VModel.load(root, Phi3VTestSupport.directory());
      HfTokenizer tokenizer = Phi3VTestSupport.tokenizer();
      List<RgbImage> wide = List.of(Phi3VTestSupport.wide());
      // 465 prompt tokens, budget 8: the demand is 472 and decode step 6 first reaches total
      // width 471, so a capacity of exactly 472 must fit the re-prefill's full-history width.
      var request =
          GenerationRequest.text(
                  tokenizer,
                  "c d <|image_1|> e s",
                  PromptSpecialTokens.ADD,
                  GenerationConfig.greedyDefaults(8, Set.of()),
                  CancellationToken.NONE)
              .withImages(wide);
      List<String> stages = new ArrayList<>();
      model.setObserverForTest(stages::add);
      GenerationResult bounded =
          model.generate(request.withCachePolicy(GenerationCachePolicy.full(472)), ignored -> {});
      assertEquals(1, stages.stream().filter("re-prefill"::equals).count());
      assertEquals(7, stages.stream().filter("decode"::equals).count());
      assertEquals(bounded, model.generate(request, ignored -> {}));
      stages.clear();
      var error =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.generate(
                      request.withCachePolicy(GenerationCachePolicy.full(471)), ignored -> {}));
      assertTrue(error.getMessage().contains("expanded prompt 465"));
      assertTrue(stages.isEmpty());
      model.setObserverForTest(ignored -> {});
    }
  }

  @Test
  void cancellationStopsAtEveryBoundary() throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model = Phi3VModel.load(root, Phi3VTestSupport.directory());
      var base = Phi3VTestSupport.requests(5).get(4); // two-image: wide then square
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
            GenerationRequest.text(
                    Phi3VTestSupport.tokenizer(),
                    base.promptText(),
                    base.promptSpecialTokens(),
                    base.config(),
                    cancelled::get)
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
        } else if (boundary.equals("preprocess")) {
          assertEquals(
              List.of("preprocess"), stages, "cancellation must precede pixel/native work");
        } else if (boundary.equals("image-closed")) {
          assertEquals(1, stages.stream().filter("vision"::equals).count());
          assertEquals(1, stages.stream().filter("image-closed"::equals).count());
          assertFalse(stages.contains("before-prefill"));
          assertFalse(stages.contains("prefill-evaluated"));
          assertFalse(stages.contains("prefill-closed"));
          assertFalse(stages.contains("decode"));
        } else if (boundary.equals("before-prefill")) {
          assertEquals(2, stages.stream().filter("vision"::equals).count());
          assertEquals(2, stages.stream().filter("image-closed"::equals).count());
          assertEquals(1, stages.stream().filter("project"::equals).count());
          assertFalse(stages.contains("prefill-evaluated"));
          assertFalse(stages.contains("prefill-closed"));
          assertFalse(stages.contains("decode"));
        } else if (boundary.equals("prefill-closed") || boundary.equals("listener")) {
          assertEquals(1, stages.stream().filter("prefill-evaluated"::equals).count());
          assertEquals(1, stages.stream().filter("prefill-closed"::equals).count());
          assertFalse(stages.contains("decode"));
          assertEquals(1, result.generatedTokenIds().size());
        } else if (boundary.equals("decode")) {
          assertEquals(1, stages.stream().filter("decode"::equals).count());
          assertEquals(2, result.generatedTokenIds().size(), "no further decode step may start");
        }
        if (List.of("initial", "preprocess", "image-closed", "before-prefill").contains(boundary)) {
          assertTrue(result.generatedTokenIds().isEmpty());
        }
        assertTrue(result.generatedTokenIds().size() < 5);
      }
      model.setObserverForTest(ignored -> {});
    }
  }

  @Test
  void eosStopSamplingStreamingAndAbortUseTheSharedContracts() throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model = Phi3VModel.load(root, Phi3VTestSupport.directory());
      var base =
          Phi3VTestSupport.request(
              Phi3VTestSupport.tokenizer(),
              "toast <|image_1|>toast",
              List.of(Phi3VTestSupport.square()),
              5);
      int first = model.generate(base, ignored -> {}).generatedTokenIds().getFirst();
      // Pre-tokenized requests are rejected by this family (the expansion needs the text), so the
      // variant requests are rebuilt through the text factory with the same prompt.
      for (boolean eos : List.of(true, false)) {
        var config =
            GenerationConfig.greedyDefaults(5, eos ? Set.of(first) : Set.of(), Set.of(first));
        var result =
            model.generate(
                GenerationRequest.text(
                        Phi3VTestSupport.tokenizer(),
                        base.promptText(),
                        base.promptSpecialTokens(),
                        config,
                        CancellationToken.NONE)
                    .withImages(base.images()),
                ignored -> {});
        assertEquals(eos ? FinishReason.EOS : FinishReason.STOP_TOKEN, result.finishReason());
        assertEquals(eos ? List.of(first) : List.of(), result.generatedTokenIds());
      }
      var tokenizer = Phi3VTestSupport.tokenizer();
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
                  "toast <|image_1|>toast",
                  PromptSpecialTokens.OMIT,
                  policy,
                  CancellationToken.NONE)
              .withImages(List.of(Phi3VTestSupport.square()));
      List<GenerationEvent> events = new ArrayList<>();
      var sampled = model.generate(request, events::add);
      assertEquals(sampled, model.generate(request, ignored -> {}));
      assertEquals(4, sampled.logProbabilities().size());
      assertEquals(5, events.size());
      assertNotNull(sampled.generatedText());
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
      assertEquals(Arrays.stream(base.promptTokenIds()).boxed().toList(), aborted.promptTokenIds());
      assertEquals(5, model.generate(base, ignored -> {}).generatedTokenIds().size());
    }
  }

  @Test
  void structuredChatContentExpandsThroughTheRetainedMessages(@TempDir Path directory)
      throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model = Phi3VModel.load(root, Phi3VTestSupport.directory());
      // The WP6 checkpoint ships no chat template; pair its tiny tokenizer with the artifact's
      // string-concatenating template so the documented GenerationRequest.chat form is exercised.
      HfTokenizer chatTokenizer = chatTokenizer(directory);
      GenerationRequest structured =
          GenerationRequest.chat(
                  chatTokenizer,
                  List.of(
                      Map.of(
                          "role",
                          "user",
                          "content",
                          List.of(
                              Map.of("type", "text", "text", "toast "),
                              Map.of("type", "image"),
                              Map.of("type", "text", "text", " toast")))),
                  ChatTemplateOptions.defaults(true),
                  GenerationConfig.greedyDefaults(5, Set.of()),
                  CancellationToken.NONE)
              .withImages(List.of(Phi3VTestSupport.square()));
      // The artifact's template string-concatenates message['content'], so the request cannot
      // retain proper markers in its rendered text; the model must re-flatten the retained
      // messages. The equivalent literal-marker request must generate identically.
      GenerationRequest marked =
          Phi3VTestSupport.request(
              Phi3VTestSupport.tokenizer(),
              "<|user|>\ntoast <|image_1|> toast<|end|>\n<|assistant|>\n",
              List.of(Phi3VTestSupport.square()),
              5);
      GenerationResult fromParts = model.generate(structured, ignored -> {});
      GenerationResult fromMarkers = model.generate(marked, ignored -> {});
      assertEquals(fromMarkers.generatedTokenIds(), fromParts.generatedTokenIds());
      assertEquals(fromMarkers.promptPositions(), fromParts.promptPositions());
    }
  }

  private static HfTokenizer chatTokenizer(Path directory) throws Exception {
    Files.copy(
        Phi3VTestSupport.directory().resolve("tokenizer.json"),
        directory.resolve("tokenizer.json"));
    ObjectNode config = MAPPER.createObjectNode();
    config.put("chat_template", artifactChatTemplate());
    Files.writeString(directory.resolve("tokenizer_config.json"), config.toString());
    return HfTokenizer.fromDirectory(directory);
  }

  private static String artifactChatTemplate() throws Exception {
    return JsonFiles.read(
            Path.of(
                System.getProperty("jmlx.repository.root"),
                "jmlx-models/src/test/resources/phi3v-tokens/tokenizer_config.json"))
        .required("chat_template")
        .asString();
  }

  private static int expandedLength(Phi3VModel model, GenerationRequest request) {
    return model.plan(request.promptText(), request.images()).expandedLength();
  }
}
