package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;

/** Strict CPU-HF float32 references on identical oracle pixels, independent of preprocessing. */
@Tag("full-float32")
@EnabledIfNativeAvailable
class SmolVlmGoldenTest {
  static Path directory() {
    return Path.of(
        System.getProperty("jmlx.repository.root"),
        "tools/hf-reference/goldens/checkpoints/idefics3");
  }

  static JsonNode golden() throws Exception {
    return JsonFiles.read(directory().getParent().getParent().resolve("idefics3.json"));
  }

  static int[] ints(JsonNode node) {
    List<Integer> values = new ArrayList<>();
    node.forEach(v -> values.add(v.intValue()));
    return values.stream().mapToInt(Integer::intValue).toArray();
  }

  static float[] floats(JsonNode node) {
    List<Float> values = new ArrayList<>();
    flatten(node, values);
    float[] result = new float[values.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = values.get(i);
    }
    return result;
  }

  private static void flatten(JsonNode node, List<Float> values) {
    if (node.isArray()) {
      node.forEach(v -> flatten(v, values));
    } else {
      values.add(node.floatValue());
    }
  }

  static List<RgbImage> images(JsonNode c) throws Exception {
    List<RgbImage> images = new ArrayList<>();
    for (JsonNode image : c.required("images")) {
      images.add(
          ImageDecoder.decode(
              directory().resolve("images").resolve(image.required("file").asString())));
    }
    return images;
  }

  static GenerationRequest request(JsonNode c, int count) throws Exception {
    return new GenerationRequest(
            ints(c.required("unexpanded_ids")),
            GenerationConfig.greedyDefaults(count, Set.of()),
            CancellationToken.NONE)
        .withImages(images(c));
  }

  @Test
  void intermediatePrefillAndCachedLogitsMatchHf() throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, directory());
      for (JsonNode c : golden().required("cases")) {
        String name = c.required("name").asString();
        GenerationRequest request = request(c, 5);
        SmolVlmPromptPlan plan = model.plan(request.promptTokenIds(), request.images());
        assertArrayEquals(ints(c.required("expanded_ids")), plan.expandedIds(), name);
        try (MLXScope generation = root.newChild()) {
          List<MLXArray> projected = new ArrayList<>();
          for (int i = 0; i < c.required("tiles").size(); i++) {
            JsonNode expected = c.required("tiles").get(i);
            MLXArray pixels =
                MLX.array(
                    generation,
                    floats(c.required("pixel_values").get(i)),
                    new int[] {1, 32, 32, 3});
            int[] mask =
                Idefics3VisionTower.patchMaskFromPixelMask(
                    intsFlatten(c.required("pixel_attention_masks").get(i)), 32, 32, 8);
            assertArrayEquals(ints(expected.required("patch_mask")), mask);
            assertArrayEquals(
                ints(expected.required("position_ids")),
                Idefics3VisionTower.positionIds(mask, 4, 4, 4));
            assertArrayEquals(
                floats(expected.required("patch_embed")),
                model.towerForTest().patchTokens(pixels).toFloatArray(),
                1e-4f,
                name + " patch");
            MLXArray patch = model.towerForTest().patchTokens(pixels);
            MLXArray positions =
                se.alipsa.jmlx.core.MLXShape.takeAxis(
                    model.towerForTest().parameters().get("positionEmbedding.weight"),
                    MLX.array(
                        generation, ints(expected.required("position_ids")), new int[] {1, 16}),
                    0);
            assertArrayEquals(
                floats(expected.required("embeddings")),
                se.alipsa.jmlx.core.MLXOps.add(patch, positions).toFloatArray(),
                1e-4f,
                name + " embeddings");
            MLXArray tower = model.towerForTest().forward(pixels, mask);
            assertArrayEquals(
                floats(expected.required("tower_out")),
                tower.toFloatArray(),
                1e-4f,
                name + " tower");
            MLXArray feature = model.towerForTest().connector().forward(tower);
            assertArrayEquals(
                floats(expected.required("connector_out")),
                feature.toFloatArray(),
                1e-4f,
                name + " connector");
            projected.add(feature);
          }
          MLXArray features =
              projected.isEmpty()
                  ? null
                  : se.alipsa.jmlx.core.MLXShape.concatenate(projected.toArray(MLXArray[]::new), 1);
          List<KVCache> caches = List.of(new KVCache(generation), new KVCache(generation));
          int[] expanded = plan.expandedIds();
          MLXArray ids = MLX.array(generation, expanded, new int[] {1, expanded.length});
          MLXArray logits =
              model
                  .decoderForTest()
                  .stepLogits(
                      ids,
                      caches,
                      new int[] {expanded.length},
                      features == null
                          ? embedded -> {
                            assertArrayEquals(
                                floats(c.required("merged_embeddings")),
                                embedded.toFloatArray(),
                                1e-4f);
                            return embedded;
                          }
                          : embedded -> {
                            MLXArray merged = SmolVlmModel.merge(embedded, features, plan);
                            assertArrayEquals(
                                floats(c.required("merged_embeddings")),
                                merged.toFloatArray(),
                                1e-4f);
                            return merged;
                          });
          assertArrayEquals(
              floats(c.required("prefill_logits")),
              logits.toFloatArray(),
              1e-4f,
              name + " prefill");
          int step = 0;
          for (JsonNode decode : c.required("decode_steps")) {
            MLXArray cached = logits;
            if (step > 0) {
              int token = c.required("decode_steps").get(step - 1).required("token_id").asInt();
              MLXArray next = MLX.array(generation, new int[] {token}, new int[] {1, 1});
              cached = model.decoderForTest().stepLogits(next, caches, new int[] {1}, null);
            }
            assertArrayEquals(
                floats(decode.required("logits")),
                cached.toFloatArray(),
                1e-4f,
                name + " decode " + step);
            // Rebuild the entire history, merging the images only at their original destinations.
            int[] full = Arrays.copyOf(expanded, expanded.length + step);
            for (int j = 0; j < step; j++) {
              full[expanded.length + j] =
                  c.required("decode_steps").get(j).required("token_id").asInt();
            }
            List<KVCache> uncached = List.of(new KVCache(generation), new KVCache(generation));
            MLXArray fullIds = MLX.array(generation, full, new int[] {1, full.length});
            SmolVlmPromptPlan fullPlan =
                new SmolVlmPromptPlan(
                    plan.unexpandedIds(), full, plan.tokensPerTile(), plan.images());
            MLXArray fullLogits =
                model
                    .decoderForTest()
                    .stepLogits(
                        fullIds,
                        uncached,
                        new int[] {full.length},
                        features == null
                            ? null
                            : embedded -> SmolVlmModel.merge(embedded, features, fullPlan));
            assertArrayEquals(
                fullLogits.toFloatArray(), cached.toFloatArray(), 1e-4f, name + " cached/full");
            step++;
          }
          int last = c.required("generated_token_ids").get(2).asInt();
          MLXArray third =
              model
                  .decoderForTest()
                  .stepLogits(
                      MLX.array(generation, new int[] {last}, new int[] {1, 1}),
                      caches,
                      new int[] {1},
                      null);
          int[] all = Arrays.copyOf(expanded, expanded.length + 3);
          for (int j = 0; j < 3; j++) {
            all[expanded.length + j] = c.required("generated_token_ids").get(j).asInt();
          }
          SmolVlmPromptPlan allPlan =
              new SmolVlmPromptPlan(plan.unexpandedIds(), all, plan.tokensPerTile(), plan.images());
          MLXArray fullThird =
              model
                  .decoderForTest()
                  .stepLogits(
                      MLX.array(generation, all, new int[] {1, all.length}),
                      List.of(new KVCache(generation), new KVCache(generation)),
                      new int[] {all.length},
                      features == null
                          ? null
                          : embedded -> SmolVlmModel.merge(embedded, features, allPlan));
          assertArrayEquals(
              fullThird.toFloatArray(), third.toFloatArray(), 1e-4f, name + " third cached/full");
          try (MLXScope prefill = generation.newChild()) {
            MLXArray javaFeatures = model.features(request, prefill);
            MLXArray javaIds = MLX.array(prefill, expanded, new int[] {1, expanded.length});
            MLXArray javaLogits =
                model
                    .decoderForTest()
                    .stepLogits(
                        javaIds,
                        List.of(new KVCache(generation), new KVCache(generation)),
                        new int[] {expanded.length},
                        javaFeatures == null
                            ? null
                            : embedded -> SmolVlmModel.merge(embedded, javaFeatures, plan));
            assertArrayEquals(
                floats(c.required("prefill_logits")),
                javaLogits.toFloatArray(),
                1e-4f,
                name + " Java preprocessing end-to-end");
          }
          assertEquals(3, step);
        }
        GenerationResult result = model.generate(request, ignored -> {});
        assertEquals(
            Arrays.stream(ints(c.required("greedy_ids"))).boxed().toList(),
            result.generatedTokenIds(),
            name);
        assertEquals(plan.expandedLength(), result.promptPositions());
        assertTrue(result.promptTokenIds().size() <= result.promptPositions());
      }
    }
  }

  private static int[] intsFlatten(JsonNode node) {
    float[] values = floats(node);
    int[] result = new int[values.length];
    for (int i = 0; i < values.length; i++) {
      result[i] = (int) values[i];
    }
    return result;
  }
}
