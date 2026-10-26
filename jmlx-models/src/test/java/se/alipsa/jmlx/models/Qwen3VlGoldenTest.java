package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.Qwen3VlImageProcessor;
import se.alipsa.jmlx.vision.Qwen3VlProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;

/**
 * The strict Phase 7.3b WP8 acceptance reference for the composite Qwen3-VL model on the tiny WP8
 * checkpoint (three decoder layers with two DeepStack levels, mrope_section [16, 8, 8], and grids
 * that run the position-table interpolation beyond the learned 8 x 8 table): the ID-level prompt
 * expansion, the processed pixel bytes, the three-axis prompt positions and delta, the tower's
 * merged and DeepStack features, the prefill logits, and every cached decode and
 * forced-continuation step against the golden's fresh, full, uncached reference forward. The forced
 * continuations feed every image/video/vision-start/vision-end sentinel plus a complete
 * wrapper-looking generated run; the observer asserts that none of those steps consumes an image,
 * re-runs the tower, re-preprocesses, or injects DeepStack -- the DeepStack injections fire exactly
 * once per level, all during the prefill.
 */
@Tag("full-float32")
@EnabledIfNativeAvailable
class Qwen3VlGoldenTest {

  static Path directory() {
    return Path.of(
        System.getProperty("jmlx.repository.root"),
        "tools/hf-reference/goldens/checkpoints/qwen3vl-wp8");
  }

  static JsonNode golden() throws Exception {
    return JsonFiles.read(directory().getParent().getParent().resolve("qwen3vl-wp8.json"));
  }

  private static int[] ints(JsonNode node) {
    List<Integer> values = new ArrayList<>();
    node.forEach(v -> values.add(v.intValue()));
    return values.stream().mapToInt(Integer::intValue).toArray();
  }

  /** Recursively collects every integer leaf, flattening nested arrays such as the grids. */
  private static int[] flatInts(JsonNode node) {
    List<Integer> values = new ArrayList<>();
    collectInts(node, values);
    return values.stream().mapToInt(Integer::intValue).toArray();
  }

  private static void collectInts(JsonNode node, List<Integer> values) {
    if (node.isArray()) {
      node.forEach(v -> collectInts(v, values));
    } else {
      values.add(node.intValue());
    }
  }

  private static float[] flatten(JsonNode node) {
    List<Float> values = new ArrayList<>();
    collect(node, values);
    float[] result = new float[values.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = values.get(i);
    }
    return result;
  }

  private static void collect(JsonNode node, List<Float> values) {
    if (node.isArray()) {
      node.forEach(v -> collect(v, values));
    } else {
      values.add(node.floatValue());
    }
  }

  private static List<RgbImage> images(JsonNode c) throws Exception {
    List<RgbImage> images = new ArrayList<>();
    for (JsonNode image : c.required("images")) {
      images.add(ImageDecoder.decode(directory().resolve("images").resolve(image.asString())));
    }
    return images;
  }

  /**
   * The SHA-256 of every image's processed values in prompt order: the Java processor's flat {@code
   * pixel_values} rows serialized as little-endian float32, the same byte order the reference
   * generator hashed.
   */
  private static String pixelValuesSha256(Qwen3VlImageProcessor processor, List<RgbImage> images)
      throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    for (RgbImage image : images) {
      float[] values = processor.process(image).values();
      ByteBuffer bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
      for (float value : values) {
        bytes.putFloat(value);
      }
      digest.update(bytes.array());
    }
    return HexFormat.of().formatHex(digest.digest());
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

  /**
   * Runs every WP8 golden case against the model and returns the maximum absolute error over all
   * prefill, decode and forced-step logit vectors and all merged/DeepStack feature rows.
   *
   * @param bound the per-vector error bound checked as the cases run
   * @param assertGreedyTokens when true, each generation case's argmax must equal the golden's
   *     recorded greedy ID (skipped for the reduced-precision run, where the margin is not pinned)
   */
  static float runCases(float bound, boolean assertGreedyTokens) throws Exception {
    float maximum = 0;
    try (MLXScope root = new MLXScope()) {
      Qwen3VlModel model = Qwen3VlModel.load(root, directory());
      Qwen3VlImageProcessor processor =
          new Qwen3VlImageProcessor(
              Qwen3VlProcessorConfig.fromJson(
                  Files.readAllBytes(directory().resolve("preprocessor_config.json"))));
      List<String> events = new ArrayList<>();
      model.setObserverForTest(events::add);
      for (JsonNode c : golden().required("cases")) {
        maximum =
            Math.max(
                maximum, runCase(model, processor, events, c, bound, assertGreedyTokens, root));
      }
    }
    return maximum;
  }

  private static float runCase(
      Qwen3VlModel model,
      Qwen3VlImageProcessor processor,
      List<String> events,
      JsonNode c,
      float bound,
      boolean assertGreedyTokens,
      MLXScope root)
      throws Exception {
    String name = c.required("name").asString();
    int[] ids = ints(c.required("unexpanded_ids"));
    List<RgbImage> caseImages = images(c);
    Qwen3VlPromptPlan plan = model.plan(ids, caseImages);
    assertArrayEquals(ints(c.required("expanded_ids")), plan.expandedIds(), name + " expansion");
    // The Java processor's per-image grids, in prompt order.
    int[] javaGrids = new int[3 * caseImages.size()];
    int at = 0;
    for (RgbImage image : caseImages) {
      System.arraycopy(processor.process(image).gridThw(), 0, javaGrids, at, 3);
      at += 3;
    }
    assertArrayEquals(flatInts(c.required("grids")), javaGrids, name + " grids");
    if (c.has("pixel_values_sha256")) {
      assertEquals(
          c.required("pixel_values_sha256").asString(),
          pixelValuesSha256(processor, caseImages),
          name + " processed pixel bytes");
    }
    // The prompt axes equal the Java plan exactly; the recorded continuation (present on the
    // forced cases, which carry the full-history axes) is logicalPosition + delta on all axes.
    JsonNode positions = c.required("positions");
    int delta = positions.required("delta").intValue();
    assertEquals(delta, plan.positionDelta(), name + " position delta");
    int promptLength = plan.expandedLength();
    int[] t = ints(positions.required("t"));
    int[] h = ints(positions.required("h"));
    int[] w = ints(positions.required("w"));
    assertArrayEquals(
        Arrays.copyOf(t, promptLength), plan.temporalPositions(), name + " temporal axis");
    assertArrayEquals(
        Arrays.copyOf(h, promptLength), plan.heightPositions(), name + " height axis");
    assertArrayEquals(Arrays.copyOf(w, promptLength), plan.widthPositions(), name + " width axis");
    for (int i = promptLength; i < t.length; i++) {
      assertEquals(i + delta, t[i], name + " temporal continuation at " + i);
      assertEquals(i + delta, h[i], name + " height continuation at " + i);
      assertEquals(i + delta, w[i], name + " width continuation at " + i);
      assertEquals(plan.ropePositionAt(i), i + delta, name + " plan continuation at " + i);
    }
    float maximum = 0;
    // The intermediate tower features (the generation cases with images).
    if (c.hasNonNull("visual_features")) {
      JsonNode expectedFeatures = c.required("visual_features");
      try (MLXScope featureScope = root.newChild()) {
        Qwen3VlVisionFeatures features =
            model.features(CancellationToken.NONE, caseImages, featureScope);
        maximum =
            Math.max(
                maximum,
                compareRows(
                    features.merged(),
                    expectedFeatures.required("merged"),
                    name + " merged features",
                    bound));
        JsonNode levels = expectedFeatures.required("deepstack");
        assertEquals(
            levels.size(), features.deepstackLevels().size(), name + " deepstack level count");
        for (int level = 0; level < levels.size(); level++) {
          maximum =
              Math.max(
                  maximum,
                  compareRows(
                      features.deepstackLevels().get(level),
                      levels.get(level),
                      name + " deepstack level " + level,
                      bound));
        }
      }
    }
    JsonNode decodeSteps = c.required("decode_steps");
    try (MLXScope generation = root.newChild()) {
      List<KVCache> caches = Qwen3VlTestSupport.caches(model, generation);
      // The pinned checkpoint's decoder depth exceeds its DeepStack level count, so an
      // image-bearing prefill injects one event per level, and no cached decode step carries a
      // post-block hook at all.
      int expectedInjections =
          caseImages.isEmpty()
              ? 0
              : Math.min(
                  model.metadata().numHiddenLayers(),
                  model.towerForTest().deepstackIndexes().length);
      events.clear();
      for (int step = 0; step < decodeSteps.size(); step++) {
        JsonNode stepGolden = decodeSteps.get(step);
        float[] stepLogits;
        try (MLXScope activation = generation.newChild()) {
          MLXArray logits;
          if (step == 0) {
            logits =
                model
                    .prefill(CancellationToken.NONE, caseImages, plan, activation, caches)
                    .logits();
          } else {
            int token = decodeSteps.get(step - 1).required("token_id").intValue();
            logits = Qwen3VlTestSupport.decode(model, activation, plan, caches, token);
            MLX.eval(DecoderModel.stepCacheArrays(caches));
          }
          stepLogits = logits.toFloatArray();
        }
        maximum =
            Math.max(
                maximum,
                compareStep(
                    stepGolden,
                    stepLogits,
                    name,
                    step,
                    bound,
                    assertGreedyTokens && !"forced".equals(c.required("type").asString())));
        // Pins both the total injection count and its timing: all injections happen inside
        // the prefill (step 0), so the count may never grow on a cached decode step.
        assertEquals(
            expectedInjections,
            events.stream().filter("deepstack"::equals).count(),
            name + " deepstack injections after step " + step);
      }
      // Step 0 is the prefill itself, so the final position is the prompt length plus one
      // decode step per remaining golden entry.
      assertEquals(
          promptLength + decodeSteps.size() - 1,
          caches.getFirst().nextPosition(),
          name + " cache position");
      // Generated markers are ordinary history: no step after the prefill may consume an
      // image, run the tower, or re-preprocess -- each image's pipeline stages fire exactly
      // once, at the prefill.
      assertEquals(
          caseImages.size(),
          events.stream().filter("vision"::equals).count(),
          name + " tower runs");
      assertEquals(
          caseImages.size(),
          events.stream().filter("preprocess"::equals).count(),
          name + " preprocess runs");
    }
    return maximum;
  }

  private static float compareStep(
      JsonNode stepGolden,
      float[] actual,
      String name,
      int step,
      float bound,
      boolean assertGreedy) {
    float[] expected = flatten(stepGolden.required("logits"));
    assertEquals(expected.length, actual.length, name + " step " + step + " vocabulary size");
    float maximum = 0;
    for (int i = 0; i < actual.length; i++) {
      assertTrue(Float.isFinite(actual[i]), name + " step " + step + " logit " + i);
      maximum = Math.max(maximum, Math.abs(actual[i] - expected[i]));
    }
    if (assertGreedy) {
      assertEquals(
          stepGolden.required("token_id").intValue(),
          argmax(actual),
          name + " greedy ID at step " + step);
    }
    assertTrue(
        maximum <= bound, name + " step " + step + " logit error " + maximum + " exceeds " + bound);
    return maximum;
  }

  private static float compareRows(
      MLXArray actualArray, JsonNode expectedNode, String label, float bound) {
    float[] actual = actualArray.toFloatArray();
    float[] expected = flatten(expectedNode);
    assertEquals(expected.length, actual.length, label + " length");
    float maximum = 0;
    for (int i = 0; i < actual.length; i++) {
      assertTrue(Float.isFinite(actual[i]), label + " row " + i);
      maximum = Math.max(maximum, Math.abs(actual[i] - expected[i]));
    }
    assertTrue(maximum <= bound, label + " error " + maximum + " exceeds " + bound);
    return maximum;
  }

  @Test
  void baseAndForcedCasesMatchStrictHfGolden() throws Exception {
    float maximum = runCases(1e-4f, true);
    System.out.println("Qwen3-VL WP8 strict maximum logit/feature error: " + maximum);
  }
}
