package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

@EnabledIfNativeAvailable
class DecoderSlidingWindowTest {
  @Test
  void matchesIndependentWindowBoundaryAndLongDecodeGoldens() throws Exception {
    Path root =
        Path.of(System.getProperty("jmlx.repository.root"), "tools", "hf-reference", "goldens");
    JsonNode reference = new ObjectMapper().readTree(root.resolve("mistral-window.json").toFile());
    try (MLXScope modelScope = new MLXScope()) {
      DecoderModel model = MistralModel.load(modelScope, root.resolve("checkpoints/mistral"));
      for (JsonNode item : reference.path("cases")) {
        try (MLXScope inference = modelScope.newChild()) {
          List<KVCache> caches = slidingCaches(inference, model);
          int[] prompt = ints(item.path("prompt_ids"));
          int offset = 0;
          MLXArray logits = null;
          for (JsonNode chunk : item.path("chunk_lengths")) {
            int length = chunk.asInt();
            int[] ids = java.util.Arrays.copyOfRange(prompt, offset, offset + length);
            logits = model.forward(MLX.array(inference, ids, new int[] {1, length}), caches);
            offset += length;
          }
          int[] shape = logits.shape();
          MLXArray last =
              se.alipsa.jmlx.core.MLXShape.slice(
                  logits, new int[] {0, shape[1] - 1, 0}, new int[] {1, shape[1], shape[2]});
          assertArrayEquals(
              floats(item.path("last_logits")),
              last.toFloatArray(),
              1e-4f,
              item.path("name").asText());
        }
      }
      try (MLXScope inference = modelScope.newChild()) {
        List<KVCache> caches = slidingCaches(inference, model);
        int[] prompt = {1, 7, 42, 3, 19, 5};
        model.forward(MLX.array(inference, prompt, new int[] {1, prompt.length}), caches);
        for (JsonNode step : reference.path("long_decode")) {
          int token = step.path("token_id").asInt();
          MLXArray logits =
              model.forward(MLX.array(inference, new int[] {token}, new int[] {1, 1}), caches);
          assertArrayEquals(floats(step.path("logits")), logits.toFloatArray(), 1e-4f);
        }
        assertEquals(3, caches.getFirst().length());
      }
    }
  }

  @Test
  void explicitSlidingRetentionMatchesFullMaskedAttention(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        Files.readString(config)
            .replace(
                "\"model_type\":\"llama\"", "\"model_type\":\"mistral\",\"sliding_window\":4"));
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = MistralModel.load(scope, dir);
      List<KVCache> full = new ArrayList<>();
      List<KVCache> sliding = new ArrayList<>();
      for (int i = 0; i < model.config().numHiddenLayers(); i++) {
        full.add(new KVCache(scope));
        sliding.add(new KVCache(scope, KVCachePolicy.slidingWindow(4)));
      }
      int[][] inputs = {{1, 2, 3, 4, 5, 6}, {7}, {8}, {9}};
      for (int[] input : inputs) {
        MLXArray ids = MLX.array(scope, input, new int[] {1, input.length});
        assertArrayEquals(
            model.forward(ids, full).toFloatArray(),
            model.forward(ids, sliding).toFloatArray(),
            1e-4f);
      }
      assertEquals(9, sliding.getFirst().nextPosition());
      assertEquals(3, sliding.getFirst().length());
      assertEquals(0, full.getFirst().startPosition());
    }
  }

  @Test
  void slidingGenerationCancellationStopsBetweenSteps(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        Files.readString(config)
            .replace(
                "\"model_type\":\"llama\"", "\"model_type\":\"mistral\",\"sliding_window\":4"));
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = MistralModel.load(scope, dir);
      AtomicBoolean cancelled = new AtomicBoolean();
      GenerationRequest request =
          new GenerationRequest(
                  new int[] {1, 2, 3, 4, 5},
                  GenerationConfig.greedyDefaults(8, Set.of()),
                  cancelled::get)
              .withCachePolicy(GenerationCachePolicy.slidingWindowFromModel());
      GenerationResult result =
          model.generate(
              request,
              event -> {
                if (event.tokenId() != null) {
                  cancelled.set(true);
                }
              });
      assertEquals(FinishReason.CANCELLED, result.finishReason());
      assertEquals(1, result.generatedTokenIds().size());
    }
  }

  private static List<KVCache> slidingCaches(MLXScope scope, DecoderModel model) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.config().numHiddenLayers(); i++) {
      caches.add(new KVCache(scope, KVCachePolicy.slidingWindow(4)));
    }
    return caches;
  }

  private static int[] ints(JsonNode node) {
    int[] result = new int[node.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = node.get(i).asInt();
    }
    return result;
  }

  private static float[] floats(JsonNode node) {
    float[] result = new float[node.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = (float) node.get(i).asDouble();
    }
    return result;
  }
}
