package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.TokenizerException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/** Native public-generation contracts, including zero-work and failure/reuse paths. */
@EnabledIfNativeAvailable
class SmolVlmGenerationTest {
  @Test
  void tiedWeightsUseNumericalDeviceEquality() {
    try (MLXScope root = new MLXScope()) {
      MLXArray embedding = MLX.array(root, new float[] {1f, 0f}, new int[] {1, 2});
      MLXArray equal = MLX.array(root, new float[] {1f, -0f}, new int[] {1, 2});
      MLXArray different = MLX.array(root, new float[] {2f, 0f}, new int[] {1, 2});
      MLXArray nan = MLX.array(root, new float[] {Float.NaN, 0f}, new int[] {1, 2});
      assertTrue(SmolVlmModel.sameWeights(equal, embedding));
      assertFalse(SmolVlmModel.sameWeights(different, embedding));
      assertFalse(SmolVlmModel.sameWeights(nan, nan));
    }
  }

  @Test
  void missingRequestMarkersAreArgumentErrors(@TempDir Path dir) throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      for (String marker : List.of("<image>", "<row_1_col_1>")) {
        var mapper = new ObjectMapper();
        var tree =
            mapper.readTree(SmolVlmGoldenTest.directory().resolve("tokenizer.json").toFile());
        ((ObjectNode) tree.required("model").required("vocab")).remove(marker);
        var added = (ArrayNode) tree.required("added_tokens");
        for (int i = added.size() - 1; i >= 0; i--) {
          if (marker.equals(added.get(i).required("content").asString())) {
            added.remove(i);
          }
        }
        Path file = dir.resolve("tokenizer.json");
        Files.writeString(file, mapper.writeValueAsString(tree));
        HfTokenizer tokenizer = HfTokenizer.fromFile(file);
        GenerationRequest request =
            GenerationRequest.text(
                tokenizer,
                "text",
                PromptSpecialTokens.OMIT,
                GenerationConfig.greedyDefaults(0, Set.of()),
                CancellationToken.NONE);
        IllegalArgumentException failure =
            assertThrows(
                IllegalArgumentException.class, () -> model.generate(request, ignored -> {}));
        assertEquals("request tokenizer does not match model image markers", failure.getMessage());
        assertInstanceOf(TokenizerException.class, failure.getCause());
      }
    }
  }

  @Test
  void zeroTokenPathDoesNotTouchEvenAClosedNativeScope() throws Exception {
    SmolVlmModel model;
    try (MLXScope root = new MLXScope()) {
      model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
    }
    GenerationRequest zero =
        SmolVlmGoldenTest.request(SmolVlmGoldenTest.golden().required("cases").get(0), 0);
    assertEquals(46, model.generate(zero, ignored -> {}).promptPositions());
  }

  @Test
  void tokenizerFailureReportsUnexpandedPrompt(@TempDir Path dir) throws Exception {
    JsonNode c = SmolVlmGoldenTest.golden().required("cases").get(0);
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      GenerationRequest base = SmolVlmGoldenTest.request(c, 1);
      int first = model.generate(base, ignored -> {}).generatedTokenIds().getFirst();
      ObjectMapper mapper = new ObjectMapper();
      var tree = mapper.readTree(SmolVlmGoldenTest.directory().resolve("tokenizer.json").toFile());
      var vocab = (ObjectNode) tree.required("model").required("vocab");
      String missing =
          vocab.properties().stream()
              .filter(e -> e.getValue().asInt() == first)
              .map(Map.Entry::getKey)
              .findFirst()
              .orElseThrow();
      vocab.remove(missing);
      Path file = dir.resolve("tokenizer.json");
      Files.writeString(file, mapper.writeValueAsString(tree));
      HfTokenizer tokenizer = HfTokenizer.fromFile(file);
      GenerationRequest request =
          GenerationRequest.text(
                  tokenizer,
                  c.required("prompt").asString(),
                  PromptSpecialTokens.OMIT,
                  base.config(),
                  CancellationToken.NONE)
              .withImages(base.images());
      GenerationAbortedException failure =
          assertThrows(
              GenerationAbortedException.class, () -> model.generate(request, ignored -> {}));
      assertEquals("output decoder", failure.stage());
      assertEquals(Arrays.stream(base.promptTokenIds()).boxed().toList(), failure.promptTokenIds());
      assertEquals(List.of(first), failure.generatedTokenIds());
      assertEquals(1, model.generate(base, ignored -> {}).generatedTokenIds().size());
    }
  }

  @Test
  void halfStorageIsUpcastAndBadPatchShapesAreRejected(@TempDir Path dir) throws Exception {
    try (var files = Files.list(SmolVlmGoldenTest.directory())) {
      for (Path source : files.filter(Files::isRegularFile).toList()) {
        if (!source.toString().endsWith(".safetensors")) {
          Files.copy(source, dir.resolve(source.getFileName()));
        }
      }
    }
    try (MLXScope root = new MLXScope()) {
      var original =
          MLXIO
              .loadSafetensors(
                  root, SmolVlmGoldenTest.directory().resolve("model.safetensors").toString())
              .tensors();
      for (var dtype : List.of(DType.FLOAT16, DType.BFLOAT16)) {
        Map<String, MLXArray> half = new LinkedHashMap<>();
        original.forEach((key, value) -> half.put(key, MLX.astype(value, dtype)));
        MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), half, Map.of());
        SmolVlmModel model = SmolVlmModel.load(root, dir);
        assertTrue(
            model.towerForTest().parameters().values().stream()
                .allMatch(a -> a.dtype() == DType.FLOAT32));
        assertTrue(
            model.decoderForTest().parameters().values().stream()
                .allMatch(a -> a.dtype() == DType.FLOAT32));
        GenerationRequest request =
            SmolVlmGoldenTest.request(SmolVlmGoldenTest.golden().required("cases").get(0), 1);
        assertEquals(1, model.generate(request, ignored -> {}).generatedTokenIds().size());
      }
      Map<String, MLXArray> wrong = new LinkedHashMap<>(original);
      wrong.put(
          "model.vision_model.embeddings.patch_embedding.weight",
          MLX.array(root, new float[64 * 8 * 8 * 3], new int[] {64, 8, 8, 3}));
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), wrong, Map.of());
      assertThrows(IllegalArgumentException.class, () -> SmolVlmModel.load(root, dir));
    }
  }

  @Test
  void dispatchZeroOneManyAndRequestIsolation() throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model =
          (SmolVlmModel) TextGenerationModels.load(root, SmolVlmGoldenTest.directory());
      assertEquals(
          Set.of(InputModality.TEXT, InputModality.IMAGE), model.metadata().inputModalities());
      List<String> executions = new ArrayList<>();
      model.setObserverForTest(executions::add);
      for (JsonNode c : SmolVlmGoldenTest.golden().required("cases")) {
        GenerationRequest zero =
            SmolVlmGoldenTest.request(c, 0).withCachePolicy(GenerationCachePolicy.full(1));
        int positions = model.plan(zero.promptTokenIds(), zero.images()).expandedLength();
        executions.clear();
        GenerationResult empty = model.generate(zero, ignored -> {});
        assertEquals(positions, empty.promptPositions());
        assertTrue(empty.generatedTokenIds().isEmpty());
        assertTrue(executions.isEmpty(), "zero-step must skip pixel and native work");
        for (int count : new int[] {1, 5}) {
          executions.clear();
          List<GenerationEvent> events = new ArrayList<>();
          GenerationRequest request =
              SmolVlmGoldenTest.request(c, count)
                  .withCachePolicy(GenerationCachePolicy.full(positions + count - 1));
          GenerationResult result = model.generate(request, events::add);
          assertEquals(count, result.generatedTokenIds().size());
          assertEquals(count + 1, events.size());
          assertEquals(positions, result.promptPositions());
          assertEquals(
              Arrays.stream(request.promptTokenIds()).boxed().toList(), result.promptTokenIds());
          assertEquals(
              c.required("tiles").size(), executions.stream().filter("vision"::equals).count());
          assertEquals(
              c.required("tiles").size(), executions.stream().filter("connector"::equals).count());
          assertEquals(
              request.images().size(), executions.stream().filter("preprocess"::equals).count());
          executions.clear();
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.generate(
                      request.withCachePolicy(GenerationCachePolicy.full(positions + count - 2)),
                      ignored -> {}));
          assertTrue(executions.isEmpty(), "capacity rejected before pixels");
        }
      }
    }
  }

  @Test
  void eosStopCancellationAndAbortKeepOriginalPrompt() throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      JsonNode c = SmolVlmGoldenTest.golden().required("cases").get(0);
      GenerationRequest base = SmolVlmGoldenTest.request(c, 5);
      int first = model.generate(base, ignored -> {}).generatedTokenIds().getFirst();
      GenerationRequest eos =
          new GenerationRequest(
                  base.promptTokenIds(),
                  GenerationConfig.greedyDefaults(5, Set.of(first), Set.of(first)),
                  CancellationToken.NONE)
              .withImages(base.images());
      GenerationResult stopped = model.generate(eos, ignored -> {});
      assertEquals(FinishReason.EOS, stopped.finishReason());
      assertEquals(List.of(first), stopped.generatedTokenIds());
      GenerationRequest stop =
          new GenerationRequest(
                  base.promptTokenIds(),
                  GenerationConfig.greedyDefaults(5, Set.of(), Set.of(first)),
                  CancellationToken.NONE)
              .withImages(base.images());
      stopped = model.generate(stop, ignored -> {});
      assertEquals(FinishReason.STOP_TOKEN, stopped.finishReason());
      assertTrue(stopped.generatedTokenIds().isEmpty());
      AtomicBoolean cancelled = new AtomicBoolean(true);
      GenerationRequest cancellable =
          new GenerationRequest(base.promptTokenIds(), base.config(), cancelled::get)
              .withImages(base.images());
      List<String> executions = new ArrayList<>();
      model.setObserverForTest(executions::add);
      assertEquals(
          FinishReason.CANCELLED, model.generate(cancellable, ignored -> {}).finishReason());
      assertTrue(executions.isEmpty());
      cancelled.set(false);
      model.setObserverForTest(
          stage -> {
            if (stage.equals("preprocess")) {
              cancelled.set(true);
            }
          });
      assertEquals(
          FinishReason.CANCELLED, model.generate(cancellable, ignored -> {}).finishReason());
      cancelled.set(false);
      model.setObserverForTest(executions::add);
      GenerationResult partial = model.generate(cancellable, event -> cancelled.set(true));
      assertEquals(FinishReason.CANCELLED, partial.finishReason());
      assertEquals(1, partial.generatedTokenIds().size());
      List<Integer> original = Arrays.stream(base.promptTokenIds()).boxed().toList();
      GenerationAbortedException aborted =
          assertThrows(
              GenerationAbortedException.class,
              () ->
                  model.generate(
                      base,
                      event -> {
                        throw new IllegalStateException("listener");
                      }));
      assertEquals(original, aborted.promptTokenIds());
      assertEquals(1, aborted.generatedTokenIds().size());
      model.setObserverForTest(
          stage -> {
            if (stage.equals("connector")) {
              throw new IllegalStateException("vision failure");
            }
          });
      assertThrows(IllegalStateException.class, () -> model.generate(base, ignored -> {}));
      model.setObserverForTest(ignored -> {});
      model
          .decoderForTest()
          .setStepBoundaryEvaluatorForTest(
              arrays -> {
                throw new IllegalStateException("evaluation failure");
              });
      assertThrows(IllegalStateException.class, () -> model.generate(base, ignored -> {}));
      model.decoderForTest().setStepBoundaryEvaluatorForTest(StepBoundaryEvaluator.NATIVE);
      assertEquals(5, model.generate(base, ignored -> {}).generatedTokenIds().size());
    }
  }

  @Test
  void zeroStepStillRejectsMalformedPromptsAndLimits() throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      JsonNode c = SmolVlmGoldenTest.golden().required("cases").get(0);
      GenerationRequest base = SmolVlmGoldenTest.request(c, 0);
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  new GenerationRequest(
                      base.promptTokenIds(), base.config(), CancellationToken.NONE),
                  ignored -> {}));
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(SmolVlmGoldenTest.directory());
      for (String marker : List.of("<fake_token_around_image>", "<global-img>", "<row_1_col_1>")) {
        int[] prompt =
            tokenizer.encode(marker, false).stream().mapToInt(Integer::intValue).toArray();
        assertThrows(
            IllegalArgumentException.class,
            () ->
                model.generate(
                    new GenerationRequest(prompt, base.config(), CancellationToken.NONE),
                    ignored -> {}));
      }
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  base.withImages(Collections.nCopies(17, base.images().getFirst())),
                  ignored -> {}));
      int[] tooLong = new int[257];
      Arrays.fill(tooLong, 32);
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  new GenerationRequest(tooLong, base.config(), CancellationToken.NONE),
                  ignored -> {}));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  base.withCachePolicy(GenerationCachePolicy.slidingWindow(16)), ignored -> {}));
    }
  }

  @Test
  void sampledStreamingUsesRequestTokenizerAndExpandedPenalties() throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      JsonNode c = SmolVlmGoldenTest.golden().required("cases").get(0);
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(SmolVlmGoldenTest.directory());
      GenerationConfig policy =
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
      GenerationRequest request =
          GenerationRequest.text(
                  tokenizer,
                  c.required("prompt").asString(),
                  PromptSpecialTokens.OMIT,
                  policy,
                  CancellationToken.NONE)
              .withImages(SmolVlmGoldenTest.images(c));
      List<GenerationEvent> events = new ArrayList<>();
      GenerationResult a = model.generate(request, events::add);
      GenerationResult b = model.generate(request, ignored -> {});
      assertEquals(a, b);
      assertEquals(4, a.logProbabilities().size());
      assertFalse(a.generatedText() == null);
      assertEquals(5, events.size());
    }
  }

  @Test
  void featuresSurviveVisionChildAndDieWithPrefillScope() throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      GenerationRequest request =
          SmolVlmGoldenTest.request(SmolVlmGoldenTest.golden().required("cases").get(0), 1);
      MLXArray features;
      try (MLXScope prefill = root.newChild()) {
        features = model.features(request, prefill);
        assertEquals(20 * 64, features.toFloatArray().length);
      }
      assertThrows(IllegalStateException.class, features::toFloatArray);
    }
  }
}
