package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
    return Phi3VModel.load(scope, dir, TinyCheckpoints.PHI3V_CLIP_LAYERS);
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
  void boundaryCrossingAccountsForThePrefillStep() {
    // originalMax 470, prompt 465: decode step j holds cache position 465 + j - 1, and only
    // maxNewTokens - 1 decode steps exist (step 0 is the prefill). Budget 8 fires at step 6 of
    // 7; budget 7 fires at step 6 of 6; budget 6 ends at position 469 and never fires.
    assertTrue(Phi3VModel.boundaryCrossable(465, 8, 470));
    assertTrue(Phi3VModel.boundaryCrossable(465, 7, 470));
    assertFalse(Phi3VModel.boundaryCrossable(465, 6, 470));
    // A prompt already exactly at the boundary ran the short table in prefill; the first decode
    // step crosses it, so it still fires. A prompt beyond it never fires.
    assertTrue(Phi3VModel.boundaryCrossable(470, 8, 470));
    assertFalse(Phi3VModel.boundaryCrossable(471, 8, 470));
    // One step of slack in either direction moves the answer, pinning the -2 exactly.
    assertTrue(Phi3VModel.boundaryCrossable(465, 8, 471));
    assertFalse(Phi3VModel.boundaryCrossable(465, 7, 471));
    assertFalse(Phi3VModel.boundaryCrossable(10, 0, 470));
  }

  @Test
  void clipDepthMustMatchTheExpectedArchitecture() {
    Phi3VModel.requirePinnedClipDepth(24, 24);
    var error =
        assertThrows(
            IllegalArgumentException.class, () -> Phi3VModel.requirePinnedClipDepth(23, 24));
    assertTrue(error.getMessage().contains("23"), error.getMessage());
    assertTrue(error.getMessage().contains("24"), error.getMessage());
    // The explicit allowance itself is honored.
    Phi3VModel.requirePinnedClipDepth(3, 3);
    assertThrows(IllegalArgumentException.class, () -> Phi3VModel.requirePinnedClipDepth(24, 3));
  }

  @Test
  void thePublicLoadRejectsTheTinyCheckpointDepth(@TempDir Path dir) throws Exception {
    // The production entry points always check against the pinned depth of 24 (no test property
    // or flag reaches the guard), so a checkpoint that dropped CLIP layers is rejected here.
    TinyCheckpoints.randomPhi3V(dir, SEED);
    try (MLXScope scope = new MLXScope()) {
      var error = assertThrows(IllegalArgumentException.class, () -> Phi3VModel.load(scope, dir));
      assertTrue(error.getMessage().contains("24"), error.getMessage());
      assertTrue(error.getMessage().contains("3"), error.getMessage());
    }
  }

  @Test
  void dispatchLoadsPhi3VWithItsMetadata(@TempDir Path dir) throws Exception {
    // The dispatch entry point is production: it always requires the pinned depth, so the tiny
    // checkpoint installs the package-private test override around this single load.
    Phi3VModel.setExpectedClipLayersForTest(TinyCheckpoints.PHI3V_CLIP_LAYERS);
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
    } finally {
      Phi3VModel.setExpectedClipLayersForTest(null);
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
      // The seam-driven reference runs first, so the generate() call and its assertions stay
      // adjacent below: one caller-owned activation scope kept open for the whole request,
      // stepping the retained hook exactly as the production loop does.
      Phi3VPromptPlan plan = model.plan(prompt, List.of(wide));
      int[] history = new int[plan.expandedLength() + maxNew];
      int historyLength = 0;
      for (int id : plan.expandedIds()) {
        history[historyLength++] = Math.max(id, 0);
      }
      List<Integer> viaSeam = new ArrayList<>();
      try (MLXScope generation = scope.newChild()) {
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < model.metadata().numHiddenLayers(); i++) {
          caches.add(new KVCache(generation));
        }
        Phi3VModel.BoundaryState boundary = new Phi3VModel.BoundaryState(true);
        boundary.history = history;
        boundary.historyLength = historyLength;
        try (MLXScope activation = generation.newChild()) {
          MLXArray logits =
              model
                  .prefill(
                      CancellationToken.NONE, List.of(wide), plan, activation, caches, boundary)
                  .logits();
          viaSeam.add(argmax(logits.toFloatArray()));
          for (int step = 1; step < maxNew; step++) {
            MLXArray ids =
                MLX.array(activation, new int[] {viaSeam.get(step - 1)}, new int[] {1, 1});
            MLXArray stepLogits = model.decodeStep(ids, caches, boundary, viaSeam.get(step - 1));
            MLX.eval(DecoderModel.stepCacheArrays(caches));
            viaSeam.add(argmax(stepLogits.toFloatArray()));
          }
        }
      }
      // The seam fired its one re-prefill; generate() must fire its own, exactly once.
      assertEquals(1, events.stream().filter("re-prefill"::equals).count(), "seam re-prefill");
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
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
      assertEquals(2, events.stream().filter("re-prefill"::equals).count(), "generate re-prefill");
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
          assertThrows(
              IllegalArgumentException.class,
              () -> Phi3VModel.load(scope, dir, TinyCheckpoints.PHI3V_CLIP_LAYERS));
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
          assertThrows(
              IllegalArgumentException.class,
              () -> Phi3VModel.load(scope, dir, TinyCheckpoints.PHI3V_CLIP_LAYERS));
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
