package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Slow PIL, tower and multimodal full-stack HF checks, including the real learned grid size. */
@EnabledIfNativeAvailable
@Tag("full-float32")
class Qwen35VisionGoldenTest {
  private static final Path ROOT = Path.of("..", "tools", "qwen35-reference");
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void processorInterpolationPatchBlockAndMergerMatchHfAtBothLearnedGridSizes() throws Exception {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> golden = golden(weights);
      for (String fixture : new String[] {"hf", "vlm-float", "hf-grid48"}) {
        Qwen35Model model = Qwen35Model.load(weights, ROOT.resolve("fixtures/" + fixture));
        assertEquals(fixture.equals("hf-grid48") ? 48 : 8, model.towerForTest().gridSide());
        assertArrayEquals(new int[0], model.towerForTest().deepstackIndexes());
        for (JsonNode item : cases()) {
          if (!item.get("fixture")
              .asString()
              .equals(fixture.equals("vlm-float") ? "hf" : fixture)) {
            continue;
          }
          String name = item.get("name").asString();
          List<RgbImage> images = images(item);
          Qwen3VlPromptPlan plan = model.plan(ids(item.get("original_ids")), images);
          assertArrayEquals(golden.get(name + ".ids").toIntArray(), plan.expandedIds());
          assertEquals(golden.get(name + ".delta").toIntArray()[0], plan.positionDelta());
          int[] positions = golden.get(name + ".positions").toIntArray();
          int length = plan.expandedLength();
          assertArrayEquals(
              java.util.Arrays.copyOfRange(positions, 0, length), plan.temporalPositions());
          assertArrayEquals(
              java.util.Arrays.copyOfRange(positions, length, 2 * length), plan.heightPositions());
          assertArrayEquals(
              java.util.Arrays.copyOfRange(positions, 2 * length, 3 * length),
              plan.widthPositions());
          for (int i = 0; i < images.size(); i++) {
            String p = name + ".image" + i;
            var processed = model.processorForTest().process(images.get(i));
            assertArrayEquals(golden.get(p + ".grid").toIntArray(), processed.gridThw());
            assertArrayEquals(golden.get(p + ".pixels").toFloatArray(), processed.values(), 1e-6f);
            try (MLXScope request = weights.newChild()) {
              var captured = model.towerForTest().forwardForTest(request, List.of(processed));
              assertTrue(captured.features().deepstackLevels().isEmpty());
              assertNear(golden.get(p + ".merged"), captured.features().merged());
              var image = captured.images().getFirst();
              assertNear(golden.get(p + ".position"), image.posEmbedInterpolated());
              assertNear(golden.get(p + ".patch"), image.patchEmbed());
              assertNear(golden.get(p + ".block0"), image.blockOutputs()[0]);
            }
          }
        }
      }
    }
  }

  @Test
  void singleAndMultiImagePrefillCachedDecodeAndFreshLogitsMatchHf() throws Exception {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> golden = golden(weights);
      for (String fixture : new String[] {"hf", "vlm-float", "hf-grid48"}) {
        Qwen35Model model = Qwen35Model.load(weights, ROOT.resolve("fixtures/" + fixture));
        for (JsonNode item : cases()) {
          if (!item.get("fixture")
              .asString()
              .equals(fixture.equals("vlm-float") ? "hf" : fixture)) {
            continue;
          }
          String name = item.get("name").asString();
          List<RgbImage> images = images(item);
          Qwen3VlPromptPlan plan = model.plan(ids(item.get("original_ids")), images);
          try (MLXScope request = weights.newChild();
              Qwen35Decoder.Session session =
                  model.decoderForTest().newSession(request, KVCachePolicy.full())) {
            try (MLXScope activation = request.newChild()) {
              MLXArray features = model.features(CancellationToken.NONE, images, activation);
              MLXArray input =
                  MLX.array(activation, plan.expandedIds(), new int[] {1, plan.expandedLength()});
              var rotation =
                  loadedText(fixture)
                      .rope()
                      .prepare(
                          activation,
                          1,
                          plan.expandedLength(),
                          new int[][] {
                            plan.temporalPositions(), plan.heightPositions(), plan.widthPositions()
                          });
              MLXArray logits =
                  session.build(
                      input,
                      rotation,
                      false,
                      embedded ->
                          VisualFeatureInjection.replaceImageRows(
                              embedded,
                              MLXShape.reshape(
                                  features, new int[] {1, plan.totalFeatureRows(), 128}),
                              plan.featureDestinations()));
              session.evaluateAndCommit(logits, StepBoundaryEvaluator.NATIVE);
              assertNear(golden.get(name + ".logits"), logits);
            }
            for (int step = 0; step < 2; step++) {
              try (MLXScope activation = request.newChild()) {
                int position = plan.ropePositionAt(session.position());
                var rotation =
                    loadedText(fixture)
                        .rope()
                        .prepare(
                            activation, 1, 1, new int[][] {{position}, {position}, {position}});
                MLXArray input = MLX.array(activation, new int[] {6 + step}, new int[] {1, 1});
                MLXArray logits = session.build(input, rotation, false);
                session.evaluateAndCommit(logits, StepBoundaryEvaluator.NATIVE);
                assertNear(golden.get(name + ".decode" + step + ".logits"), logits);
                assertNear(golden.get(name + ".decode" + step + ".fresh_logits"), logits);
              }
            }
          }
        }
      }
    }
  }

  @Test
  void publicImageGenerationMatchesHfIdsAndEvaluatesEachTokenAtOneBoundary() throws Exception {
    try (MLXScope weights = new MLXScope()) {
      for (String fixture : new String[] {"hf", "vlm-float", "hf-grid48"}) {
        Qwen35Model model = Qwen35Model.load(weights, ROOT.resolve("fixtures/" + fixture));
        for (JsonNode item : cases()) {
          if (!item.get("fixture")
              .asString()
              .equals(fixture.equals("vlm-float") ? "hf" : fixture)) {
            continue;
          }
          List<GenerationEvent> events = new ArrayList<>();
          final int[] boundaries = {0};
          model.evaluatorForTest(
              arrays -> {
                boundaries[0]++;
                MLX.eval(arrays);
              });
          var request =
              new GenerationRequest(
                      ids(item.get("original_ids")),
                      GenerationConfig.greedyDefaults(3, Set.of()),
                      CancellationToken.NONE)
                  .withImages(images(item));
          GenerationResult result = model.generate(request, events::add);
          assertEquals(
              java.util.Arrays.stream(ids(item.get("greedy_ids"))).boxed().toList(),
              result.generatedTokenIds());
          assertEquals(
              java.util.Arrays.stream(ids(item.get("original_ids"))).boxed().toList(),
              result.promptTokenIds());
          assertEquals(3, boundaries[0]);
          assertEquals(4, events.size());
          assertEquals(FinishReason.MAX_TOKENS, result.finishReason());
          assertEquals(
              model.plan(ids(item.get("original_ids")), images(item)).expandedLength(),
              result.promptPositions());
        }
      }
    }
  }

  private static Qwen35Descriptor.Text loadedText(String fixture) throws Exception {
    return Qwen35Descriptor.parse(
            MAPPER.readTree(ROOT.resolve("fixtures/" + fixture + "/config.json").toFile()))
        .text();
  }

  private static JsonNode cases() throws Exception {
    return MAPPER.readTree(ROOT.resolve("goldens/model-vision.json").toFile()).get("cases");
  }

  private static List<RgbImage> images(JsonNode item) throws Exception {
    List<RgbImage> images = new ArrayList<>();
    for (JsonNode image : item.get("images")) {
      images.add(ImageDecoder.decode(ROOT.resolve(image.get("file").asString())));
    }
    return images;
  }

  private static int[] ids(JsonNode array) {
    int[] result = new int[array.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = array.get(i).asInt();
    }
    return result;
  }

  private static Map<String, MLXArray> golden(MLXScope scope) {
    return MLXIO
        .loadSafetensors(scope, ROOT.resolve("goldens/model-vision.safetensors").toString())
        .tensors();
  }

  private static void assertNear(MLXArray expected, MLXArray actual) {
    assertArrayEquals(expected.shape(), actual.shape());
    assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), 1e-4f);
  }
}
