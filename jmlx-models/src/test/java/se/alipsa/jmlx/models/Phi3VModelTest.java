package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.RgbImage;

/**
 * Native Phi-3.5-Vision contracts: dispatch and metadata, text-only identity against the plain
 * decoder, the NTK-Su boundary recompute against a fresh-forward reference, the rejection guards
 * and the image pipeline smoke.
 *
 * <p>The tiny checkpoint (21-token vocabulary, two text layers, three CLIP layers, original max 470
 * / max 2048, short factors 1.0 and long factors 2.0) makes the short/long table selection
 * observable: every width at or below 470 runs the short table, every width above it the long
 * table, and the tiny "toast " word tokenizes to six single-letter IDs.
 */
@EnabledIfNativeAvailable
class Phi3VModelTest {

  private static final long SEED = 20261029L;

  private static Phi3VModel load(Path dir, MLXScope scope) throws Exception {
    TinyCheckpoints.randomPhi3V(dir, SEED);
    return Phi3VModel.load(scope, dir);
  }

  private static GenerationRequest textRequest(
      HfTokenizer tokenizer, String prompt, int maxNewTokens) {
    return GenerationRequest.text(
        tokenizer,
        prompt,
        PromptSpecialTokens.ADD,
        GenerationConfig.greedyDefaults(maxNewTokens, Set.of()),
        CancellationToken.NONE);
  }

  @Test
  void dispatchLoadsPhi3VWithItsMetadata(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      TinyCheckpoints.randomPhi3V(dir, SEED);
      TextGenerationModel loaded = TextGenerationModels.load(scope, dir);
      Phi3VModel model = assertInstanceOf(Phi3VModel.class, loaded);
      ModelMetadata metadata = model.metadata();
      assertEquals("phi3_v", metadata.modelType());
      assertEquals(TinyCheckpoints.PHI3V_VOCAB, metadata.vocabSize());
      assertEquals(TinyCheckpoints.PHI3V_TEXT_LAYERS, metadata.numHiddenLayers());
      assertEquals(TinyCheckpoints.PHI3V_CLIP_LAYERS, metadata.numEncoderLayers());
      assertEquals(Set.of(InputModality.TEXT, InputModality.IMAGE), metadata.inputModalities());
    }
  }

  @Test
  void textOnlyGenerationMatchesThePlainDecoder(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Phi3VModel model = load(dir, scope);
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
      GenerationRequest request = textRequest(tokenizer, "toast ".repeat(3), 5);
      List<Integer> viaModel = model.generate(request, ignored -> {}).generatedTokenIds();
      List<Integer> viaDecoder =
          model.decoderForTest().generate(request, ignored -> {}).generatedTokenIds();
      assertEquals(viaDecoder, viaModel);
      assertEquals(5, viaModel.size());
    }
  }

  @Test
  void boundaryStepRePrefillsUnderTheLongFactorTable(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Phi3VModel model = load(dir, scope);
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
      // 6 tokens per "toast ": 456 prompt tokens, so prefill plus decode steps 1-14 stay at or
      // below 470 (short table); decode step 15 first reaches total width 471 and must re-prefill
      // under the long table.
      String prompt = "toast ".repeat(76);
      int maxNew = 16;
      List<Integer> actual =
          model.generate(textRequest(tokenizer, prompt, maxNew), ignored -> {}).generatedTokenIds();
      assertEquals(referenceGreedy(scope, model, tokenizer, prompt, maxNew), actual);
    }
  }

  @Test
  void imageBoundaryGenerateRetainsTheProjectedRowsForTheRePrefill(@TempDir Path dir)
      throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Phi3VModel model = load(dir, scope);
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
      byte[] pixels = new byte[672 * 336 * 3];
      for (int i = 0; i < pixels.length; i++) {
        pixels[i] = (byte) (i % 251);
      }
      RgbImage wide = new RgbImage(672, 336, pixels);
      // The wide image is 457 rows; "c d " (4) + 457 + " e s" (4) = 465 < 470, so the prefill
      // runs the short table and decode step 6 first reaches total width 471: the boundary
      // re-prefill re-injects the projected rows retained from step 0, whose activation scope
      // runStepLoop closes after the step -- the rows must live in the request's generation
      // scope or this generate() dies with "MLXScope is closed" at the re-prefill.
      String prompt = "c d <|image_1|> e s";
      int maxNew = 8;
      List<String> events = new ArrayList<>();
      model.setObserverForTest(events::add);
      // The generate() result must be retained until the seam-driven reference below has built
      // its own list; the distance in between is the seam itself, not avoidable storage.
      // CHECKSTYLE.SUPPRESS: VariableDeclarationUsageDistance
      List<Integer> viaModel =
          model
              .generate(
                  GenerationRequest.text(
                          tokenizer,
                          prompt,
                          PromptSpecialTokens.ADD,
                          GenerationConfig.greedyDefaults(maxNew, Set.of()),
                          CancellationToken.NONE)
                      .withImages(List.of(wide)),
                  ignored -> {})
              .generatedTokenIds();
      assertEquals(1, events.stream().filter("re-prefill"::equals).count(), "re-prefill events");
      // The seam-driven reference: one caller-owned activation scope kept open for the whole
      // request, stepping the retained hook exactly as the production loop does.
      Phi3VPromptPlan plan = model.plan(prompt, List.of(wide));
      List<Integer> history = new ArrayList<>();
      for (int id : plan.expandedIds()) {
        history.add(Math.max(id, 0));
      }
      List<Integer> viaSeam = new ArrayList<>();
      try (MLXScope generation = scope.newChild()) {
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < model.metadata().numHiddenLayers(); i++) {
          caches.add(new KVCache(generation));
        }
        DecoderModel.EmbeddingHook[] hookRef = {null};
        try (MLXScope activation = generation.newChild()) {
          MLXArray logits =
              model
                  .prefill(
                      CancellationToken.NONE,
                      List.of(wide),
                      plan,
                      activation,
                      caches,
                      hookRef,
                      true)
                  .logits();
          viaSeam.add(argmax(logits.toFloatArray()));
          for (int step = 1; step < maxNew; step++) {
            MLXArray ids =
                MLX.array(activation, new int[] {viaSeam.get(step - 1)}, new int[] {1, 1});
            MLXArray stepLogits = model.decodeStep(ids, caches, history, hookRef, true);
            MLX.eval(DecoderModel.stepCacheArrays(caches));
            viaSeam.add(argmax(stepLogits.toFloatArray()));
          }
        }
      }
      assertEquals(viaSeam, viaModel, "generate() must survive the image boundary re-prefill");
    }
  }

  @Test
  void promptBeyondOriginalMaxNeverRecomputes(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Phi3VModel model = load(dir, scope);
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
      // 474 prompt tokens: already beyond 470, so the prefill uses the long table outright and
      // no decode step ever sits exactly at the switch point.
      String prompt = "toast ".repeat(79);
      int maxNew = 5;
      List<Integer> actual =
          model.generate(textRequest(tokenizer, prompt, maxNew), ignored -> {}).generatedTokenIds();
      assertEquals(referenceGreedy(scope, model, tokenizer, prompt, maxNew), actual);
    }
  }

  @Test
  void preTokenizedRequestsAreRejected(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Phi3VModel model = load(dir, scope);
      GenerationRequest request =
          new GenerationRequest(
              new int[] {1}, GenerationConfig.greedyDefaults(1, Set.of()), CancellationToken.NONE);
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class, () -> model.generate(request, ignored -> {}));
      assertEquals(
          "phi3_v generation requires a tokenizer-backed request: the retained prompt text is"
              + " needed to expand the image markers",
          failure.getMessage());
    }
  }

  @Test
  void slidingRetentionIsRejected(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Phi3VModel model = load(dir, scope);
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
      GenerationRequest request =
          textRequest(tokenizer, "toast", 2)
              .withCachePolicy(GenerationCachePolicy.slidingWindow(64));
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class, () -> model.generate(request, ignored -> {}));
      assertEquals(
          "sliding phi3_v retention is not yet supported (the boundary re-prefill equivalence"
              + " is unproven)",
          failure.getMessage());
    }
  }

  @Test
  void boundedFullCapacityIsEnforced(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Phi3VModel model = load(dir, scope);
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
      // 18 + 90 - 1 = 107 > 50.
      GenerationRequest request =
          textRequest(tokenizer, "toast ".repeat(3), 90)
              .withCachePolicy(GenerationCachePolicy.full(50));
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class, () -> model.generate(request, ignored -> {}));
      assertEquals("generation (expanded prompt 18 tokens) exceeds capacity", failure.getMessage());
    }
  }

  @Test
  void contextLengthIsEnforced(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Phi3VModel model = load(dir, scope);
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
      // 600 + 1500 - 1 = 2099 > 2048.
      GenerationRequest request = textRequest(tokenizer, "toast ".repeat(100), 1500);
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class, () -> model.generate(request, ignored -> {}));
      assertEquals(
          "generation exceeds context length: expanded prompt 600, unexpanded 600",
          failure.getMessage());
    }
  }

  @Test
  void tokenizerVocabularyIsCheckedAgainstTheCheckpoint(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      TinyCheckpoints.randomPhi3V(dir, SEED);
      var config = dir.resolve("config.json");
      java.nio.file.Files.writeString(
          config,
          java.nio.file.Files.readString(config)
              .replace("\"vocab_size\": 21", "\"vocab_size\": 20"));
      IllegalArgumentException failure =
          assertThrows(IllegalArgumentException.class, () -> Phi3VModel.load(scope, dir));
      assertEquals("tokenizer vocabulary exceeds checkpoint vocabulary", failure.getMessage());
    }
  }

  @Test
  void preprocessorImageTokenCountIsCrossChecked(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      TinyCheckpoints.randomPhi3V(dir, SEED);
      var preprocessor = dir.resolve("preprocessor_config.json");
      java.nio.file.Files.writeString(
          preprocessor,
          java.nio.file.Files.readString(preprocessor)
              .replace("\"num_img_tokens\": 144", "\"num_img_tokens\": 143"));
      IllegalArgumentException failure =
          assertThrows(IllegalArgumentException.class, () -> Phi3VModel.load(scope, dir));
      assertEquals(
          "preprocessor num_img_tokens 143 disagrees with config img_processor num_img_tokens 144",
          failure.getMessage());
    }
  }

  @Test
  void imageRequestRunsTheFullPipeline(@TempDir Path dir) throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Phi3VModel model = load(dir, scope);
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
      byte[] pixels = new byte[336 * 336 * 3];
      for (int i = 0; i < pixels.length; i++) {
        pixels[i] = (byte) (i % 251);
      }
      RgbImage image = new RgbImage(336, 336, pixels);
      List<String> events = new ArrayList<>();
      model.setObserverForTest(events::add);
      GenerationRequest request =
          textRequest(tokenizer, "bad <|image_1|> toast", 4).withImages(List.of(image));
      GenerationResult result = model.generate(request, ignored -> {});
      assertEquals(4, result.generatedTokenIds().size());
      assertTrue(result.promptPositions() > 456);
      // The per-image vision scope, the shared projection and the prefill close in order.
      int preprocess = events.indexOf("preprocess");
      int vision = events.indexOf("vision");
      int project = events.indexOf("project");
      int closed = events.indexOf("prefill-closed");
      assertTrue(preprocess >= 0 && vision > preprocess && project > vision && closed > project);
    }
  }

  /**
   * The independent reference: at every step a fresh, uncached full forward of the whole history
   * (its width selects the NTK-Su table exactly as the checkpoint does), greedy on the last row.
   * Step scopes are children of the model scope, since the weights live there.
   */
  private static List<Integer> referenceGreedy(
      MLXScope modelScope,
      Phi3VModel model,
      HfTokenizer tokenizer,
      String prompt,
      int maxNewTokens) {
    DecoderModel decoder = model.decoderForTest();
    KVCachePolicy policy = decoder.resolveCachePolicy(GenerationCachePolicy.full());
    int layers = decoder.config().numHiddenLayers();
    List<Integer> history = new ArrayList<>(tokenizer.encode(prompt, true));
    List<Integer> generated = new ArrayList<>();
    for (int step = 0; step < maxNewTokens; step++) {
      int[] ids = history.stream().mapToInt(Integer::intValue).toArray();
      try (MLXScope stepScope = modelScope.newChild()) {
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < layers; i++) {
          caches.add(new KVCache(stepScope, policy));
        }
        MLXArray tokenIds = MLX.array(stepScope, ids, new int[] {1, ids.length});
        MLXArray logits =
            decoder.stepLogits(tokenIds, caches, new int[] {ids.length}, null, null, null);
        MLXArray[] cacheArrays = DecoderModel.stepCacheArrays(caches);
        MLXArray[] all = new MLXArray[1 + cacheArrays.length];
        all[0] = logits;
        System.arraycopy(cacheArrays, 0, all, 1, cacheArrays.length);
        MLX.eval(all);
        float[] last = logits.toFloatArray();
        int best = 0;
        for (int v = 1; v < last.length; v++) {
          if (last[v] > last[best]) {
            best = v;
          }
        }
        generated.add(best);
        history.add(best);
      }
    }
    return generated;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }
}
