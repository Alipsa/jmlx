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
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.Phi3VGeometryPlan;
import se.alipsa.jmlx.vision.Phi3VImageProcessor;
import se.alipsa.jmlx.vision.Phi3VImageProcessorResult;
import se.alipsa.jmlx.vision.Phi3VProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;

/**
 * The strict Phase 7.3c WP6 acceptance reference for the composite Phi-3.5-Vision model on the tiny
 * WP6 checkpoint (21-token vocabulary, two GQA text layers with NTK-Su scaling 470/2048 and uniform
 * short factor 1.0 / long factor 2.0, and the fixed-width 1024 CLIP tower with two executed
 * layers): the ID-level prompt expansion, the per-image geometry and processed pixel bytes, the
 * tower's projected rows, the prefill logits, and every cached decode step against the golden's
 * fresh, full, uncached reference forward. The wide-cross-boundary case (465 prompt tokens) is the
 * single case whose generation crosses the 470 boundary: its sixth step first reaches total width
 * 471 and re-prefills the whole history under the long factor table, re-injecting the projected
 * rows retained from the prefill (golden {@code recompute_step 6}); the other cases pin the regimes
 * around it (short table throughout, long table from the prefill, and the prompt length -- not the
 * image rows alone -- selecting the table).
 */
@Tag("full-float32")
@EnabledIfNativeAvailable
class Phi3VGoldenTest {

  private static final int CLIP_IMAGE = 336;

  static Path directory() {
    return Path.of(
        System.getProperty("jmlx.repository.root"),
        "tools/hf-reference/goldens/checkpoints/phi3v-wp6");
  }

  static JsonNode golden() throws Exception {
    return JsonFiles.read(directory().getParent().getParent().resolve("phi3v-wp6.json"));
  }

  private static int[] ints(JsonNode node) {
    List<Integer> values = new ArrayList<>();
    node.forEach(v -> values.add(v.intValue()));
    return values.stream().mapToInt(Integer::intValue).toArray();
  }

  /** Expands the golden's compact run form back to the plain expanded ID list. */
  private static int[] expandCompactIds(JsonNode node) {
    List<Integer> values = new ArrayList<>();
    for (JsonNode v : node) {
      if (v.isObject()) {
        int id = v.required("neg_run").intValue();
        for (int i = 0; i < v.required("count").intValue(); i++) {
          values.add(id);
        }
      } else {
        values.add(v.intValue());
      }
    }
    return values.stream().mapToInt(Integer::intValue).toArray();
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
   * The SHA-256 of one image's processed values: the Java processor's flat 5-slot {@code
   * pixel_values} row serialized as little-endian float32, the same bytes the reference generator
   * hashed (zero-padded slots included).
   */
  private static String pixelValuesSha256(Phi3VImageProcessorResult prepared) throws Exception {
    float[] values = prepared.values();
    ByteBuffer bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      bytes.putFloat(value);
    }
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    return HexFormat.of().formatHex(digest.digest(bytes.array()));
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
   * Runs every WP6 golden case against the model and returns the maximum absolute error over all
   * prefill and decode logit vectors and all projected rows.
   *
   * @param bound the per-vector error bound checked as the cases run
   * @param assertGreedyTokens when true, each generation case's argmax must equal the golden's
   *     recorded greedy ID (skipped for the reduced-precision run, where the margin is not pinned)
   */
  static float runCases(float bound, boolean assertGreedyTokens) throws Exception {
    float maximum = 0;
    try (MLXScope root = new MLXScope()) {
      Phi3VModel model = Phi3VModel.load(root, directory());
      Phi3VImageProcessor processor =
          new Phi3VImageProcessor(
              Phi3VProcessorConfig.fromJson(
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
      Phi3VModel model,
      Phi3VImageProcessor processor,
      List<String> events,
      JsonNode c,
      float bound,
      boolean assertGreedyTokens,
      MLXScope root)
      throws Exception {
    String name = c.required("name").asString();
    List<RgbImage> caseImages = images(c);
    Phi3VPromptPlan plan = model.plan(c.required("prompt").asString(), caseImages);
    assertArrayEquals(
        expandCompactIds(c.required("expanded_ids")), plan.expandedIds(), name + " expansion");
    assertEquals(
        c.required("expanded_length").intValue(), plan.expandedLength(), name + " expanded length");
    float maximum = 0;
    // The per-image pipeline pins: the Java processor's geometry and processed bytes against
    // the golden, and the tower's projected rows through the model's own tower seam (the same
    // real-slot pixels, hdRows and shared projector call the prefill uses).
    if (!caseImages.isEmpty()) {
      try (MLXScope featureScope = root.newChild()) {
        List<MLXArray> hdBlocks = new ArrayList<>();
        for (int i = 0; i < caseImages.size(); i++) {
          Phi3VImageProcessorResult prepared = processor.process(caseImages.get(i));
          Phi3VGeometryPlan geometry = prepared.plan();
          int[] size = geometry.imageSize();
          JsonNode expectedSize = c.required("image_sizes").get(i);
          assertEquals(expectedSize.get(0).intValue(), size[0], name + " image " + i + " height");
          assertEquals(expectedSize.get(1).intValue(), size[1], name + " image " + i + " width");
          assertEquals(
              c.required("num_img_tokens").get(i).intValue(),
              geometry.numImgTokens(),
              name + " image " + i + " token count");
          assertEquals(
              c.required("pixel_values_sha256").get(i).asString(),
              pixelValuesSha256(prepared),
              name + " image " + i + " processed pixel bytes");
          int realSlots = geometry.cropBlocks() + 1;
          float[] channelMajor =
              Arrays.copyOf(prepared.values(), realSlots * 3 * CLIP_IMAGE * CLIP_IMAGE);
          MLXArray pixels =
              MLXShape.transpose(
                  MLX.array(
                      featureScope, channelMajor, new int[] {realSlots, 3, CLIP_IMAGE, CLIP_IMAGE}),
                  new int[] {0, 2, 3, 1});
          MLXArray features = model.towerForTest().forward(pixels);
          MLX.eval(features);
          hdBlocks.add(model.towerForTest().hdRows(features, geometry.gridH(), geometry.gridW()));
        }
        MLXArray hd = MLXShape.concatenate(hdBlocks.toArray(MLXArray[]::new), 0);
        MLXArray rows =
            model
                .towerForTest()
                .projector()
                .forward(MLXShape.reshape(hd, new int[] {1, hd.shape()[0], hd.shape()[1]}));
        MLX.eval(rows);
        maximum =
            Math.max(
                maximum,
                compareRows(rows, c.required("projected_rows"), name + " projected rows", bound));
      }
    }
    JsonNode decodeSteps = c.required("decode_steps");
    assertEquals(
        c.required("max_new_tokens").intValue(), decodeSteps.size(), name + " decode step count");
    Integer recomputeStep =
        c.get("recompute_step") == null || c.get("recompute_step").isNull()
            ? null
            : c.get("recompute_step").intValue();
    // The re-prefill history: the clamped expanded prompt (sentinel rows embed as row 0; their
    // rows are replaced by the hook anyway), exactly as the production request seeds it.
    List<Integer> history = new ArrayList<>();
    for (int id : plan.expandedIds()) {
      history.add(Math.max(id, 0));
    }
    try (MLXScope generation = root.newChild()) {
      List<KVCache> caches = new ArrayList<>();
      for (int i = 0; i < model.metadata().numHiddenLayers(); i++) {
        caches.add(new KVCache(generation));
      }
      Phi3VModel.BoundaryState boundary = new Phi3VModel.BoundaryState(true);
      events.clear();
      for (int step = 0; step < decodeSteps.size(); step++) {
        JsonNode stepGolden = decodeSteps.get(step);
        float[] stepLogits;
        try (MLXScope activation = generation.newChild()) {
          MLXArray logits;
          if (step == 0) {
            logits =
                model
                    .prefill(CancellationToken.NONE, caseImages, plan, activation, caches, boundary)
                    .logits();
          } else {
            int token = decodeSteps.get(step - 1).required("token_id").intValue();
            MLXArray ids = MLX.array(activation, new int[] {token}, new int[] {1, 1});
            logits = model.decodeStep(ids, caches, history, boundary);
            MLX.eval(DecoderModel.stepCacheArrays(caches));
          }
          stepLogits = logits.toFloatArray();
        }
        maximum =
            Math.max(
                maximum,
                compareStep(stepGolden, stepLogits, name, step, bound, assertGreedyTokens));
        // Pins the re-prefill's timing: it fires exactly once, at the step that first reaches
        // total width 471 (the golden's recompute step), and never again.
        int expectedRecomputes = recomputeStep == null || step < recomputeStep ? 0 : 1;
        assertEquals(
            expectedRecomputes,
            events.stream().filter("re-prefill"::equals).count(),
            name + " re-prefill count after step " + step);
      }
      // Step 0 is the prefill itself, so the final position is the prompt length plus one
      // decode step per remaining golden entry (the re-prefill refills the same width).
      assertEquals(
          plan.expandedLength() + decodeSteps.size() - 1,
          caches.getFirst().nextPosition(),
          name + " cache position");
      // The visual pipeline stages fire once per image, all during the prefill; no decode
      // step consumes an image, runs the tower, or re-preprocesses.
      assertEquals(
          caseImages.size(),
          events.stream().filter("vision"::equals).count(),
          name + " tower runs");
      assertEquals(
          caseImages.size(),
          events.stream().filter("preprocess"::equals).count(),
          name + " preprocess runs");
      assertEquals(
          caseImages.isEmpty() ? 0 : 1,
          events.stream().filter("project"::equals).count(),
          name + " projection runs");
      assertEquals(
          1, events.stream().filter("prefill-closed"::equals).count(), name + " prefill close");
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
  void casesMatchStrictHfGolden() throws Exception {
    float maximum = runCases(1e-4f, true);
    System.out.println("Phi-3.5-Vision WP6 strict maximum logit/row error: " + maximum);
  }
}
