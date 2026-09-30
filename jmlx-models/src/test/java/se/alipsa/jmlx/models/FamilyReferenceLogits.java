package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Runs a committed Hugging Face prefill and decode reference against the same safetensors. */
final class FamilyReferenceLogits {
  private FamilyReferenceLogits() {}

  static void assertMatches(String family, Class<? extends DecoderModel> modelClass)
      throws Exception {
    Path goldens =
        Path.of(System.getProperty("jmlx.repository.root"), "tools", "hf-reference", "goldens");
    JsonNode reference = new ObjectMapper().readTree(goldens.resolve(family + ".json").toFile());
    try (MLXScope modelScope = new MLXScope();
        MLXScope inference = modelScope.newChild()) {
      DecoderModel model =
          modelClass.cast(
              TextGenerationModels.load(
                  modelScope, goldens.resolve("checkpoints").resolve(family)));
      List<KVCache> caches = new ArrayList<>();
      for (int i = 0; i < model.config().numHiddenLayers(); i++) {
        caches.add(new KVCache(inference));
      }
      JsonNode prompt = reference.path("prompt_ids");
      int[] ids = new int[prompt.size()];
      for (int i = 0; i < ids.length; i++) {
        ids[i] = prompt.get(i).asInt();
      }
      try (MLXScope activation = inference.newChild()) {
        MLXArray logits =
            model.forward(MLX.array(activation, ids, new int[] {1, ids.length}), caches);
        assertArrayEquals(flatten(reference.path("prefill_logits")), logits.toFloatArray(), 1e-4f);
      }
      for (JsonNode step : reference.path("decode_steps")) {
        try (MLXScope activation = inference.newChild()) {
          MLXArray logits =
              model.forward(
                  MLX.array(
                      activation, new int[] {step.path("token_id").asInt()}, new int[] {1, 1}),
                  caches);
          assertArrayEquals(flatten(step.path("logits")), logits.toFloatArray(), 1e-4f);
        }
      }
      assertEquals(ids.length + reference.path("decode_steps").size(), caches.getFirst().offset());
    }
  }

  private static float[] flatten(JsonNode values) {
    List<Float> result = new ArrayList<>();
    append(values, result);
    float[] data = new float[result.size()];
    for (int i = 0; i < data.length; i++) {
      data[i] = result.get(i);
    }
    return data;
  }

  private static void append(JsonNode value, List<Float> result) {
    if (value.isArray()) {
      for (JsonNode child : value) {
        append(child, result);
      }
    } else {
      result.add((float) value.asDouble());
    }
  }
}
