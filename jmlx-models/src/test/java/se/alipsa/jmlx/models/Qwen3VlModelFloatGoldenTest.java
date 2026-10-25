package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * The strict CPU-HF float32 reference for the composite Qwen3-VL model on the tiny WP6 checkpoint:
 * the ID-level prompt expansion and the evaluated logits of the four golden cases (the text-only
 * prefill at every position, the one-image wrapper-grammar prefill at the last position, one
 * KV-cached decode step after that prefill, and the two-image prefill at the last position).
 */
@Tag("full-float32")
@EnabledIfNativeAvailable
class Qwen3VlModelFloatGoldenTest {

  static Path directory() {
    return Path.of(
        System.getProperty("jmlx.repository.root"),
        "tools/hf-reference/goldens/checkpoints/qwen3vl-model");
  }

  static JsonNode golden() throws Exception {
    return JsonFiles.read(directory().getParent().getParent().resolve("qwen3vl-model-wp6.json"));
  }

  private static int[] ints(JsonNode node) {
    List<Integer> values = new ArrayList<>();
    node.forEach(v -> values.add(v.intValue()));
    return values.stream().mapToInt(Integer::intValue).toArray();
  }

  private static float[] floats(JsonNode node) {
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

  private static List<RgbImage> images(JsonNode c) throws Exception {
    List<RgbImage> images = new ArrayList<>();
    for (JsonNode image : c.required("images")) {
      images.add(ImageDecoder.decode(directory().resolve("images").resolve(image.asString())));
    }
    return images;
  }

  /** Reconstructs the HF generator's asymmetric, exactly BF16-representable projection. */
  private static MLXArray projection(MLXScope scope, int rows, int columns, int salt) {
    float[] values = new float[rows * columns];
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < columns; col++) {
        values[row * columns + col] =
            ((row * 17 + col * 31 + row * col * 7 + salt * 13) % 251 - 125) / 4096f;
      }
    }
    return MLX.astype(MLX.array(scope, values, new int[] {rows, columns}), DType.BFLOAT16);
  }

  @Test
  void wideAttentionPrefillAndCachedDecodeMatchHf(@TempDir Path dir) throws Exception {
    ObjectNode config = (ObjectNode) JsonFiles.read(directory().resolve("config.json"));
    ((ObjectNode) config.get("text_config")).put("num_attention_heads", 8);
    Files.writeString(dir.resolve("config.json"), config.toString());
    for (String file : List.of("preprocessor_config.json", "tokenizer.json")) {
      Files.copy(directory().resolve(file), dir.resolve(file));
    }
    JsonNode golden =
        JsonFiles.read(directory().getParent().getParent().resolve("qwen3vl-wide-attention.json"));
    try (MLXScope root = new MLXScope()) {
      Map<String, MLXArray> tensors =
          new LinkedHashMap<>(
              MLXIO
                  .loadSafetensors(root, directory().resolve("model.safetensors").toString())
                  .tensors());
      for (int layer = 0; layer < 2; layer++) {
        String prefix = "model.language_model.layers." + layer + ".self_attn.";
        tensors.put(prefix + "q_proj.weight", projection(root, 512, 256, 11 + 2 * layer));
        tensors.put(prefix + "o_proj.weight", projection(root, 256, 512, 12 + 2 * layer));
      }
      MLXIO.saveSafetensors(
          dir.resolve("model.safetensors").toString(), tensors, Map.of("format", "pt"));
      Qwen3VlModel model = Qwen3VlModel.load(root, dir);
      for (JsonNode c : golden.required("cases")) {
        List<RgbImage> images = images(c);
        int[] ids = ints(c.required("unexpanded_ids"));
        try (MLXScope generation = root.newChild()) {
          float[] logits =
              c.has("token_id")
                  ? model.decodeStepLogitsForTest(
                      generation, ids, images, c.required("token_id").asInt())
                  : model.prefillLogitsForTest(generation, ids, images);
          assertArrayEquals(
              floats(c.required("logits")),
              logits,
              1e-4f,
              c.required("name").asString() + " wide-attention logits");
        }
      }
    }
  }

  @Test
  void prefillAndDecodeLogitsMatchHf() throws Exception {
    try (MLXScope root = new MLXScope()) {
      Qwen3VlModel model = Qwen3VlModel.load(root, directory());
      for (JsonNode c : golden().required("cases")) {
        String name = c.required("name").asString();
        int[] ids = ints(c.required("unexpanded_ids"));
        List<RgbImage> images = images(c);
        Qwen3VlPromptPlan plan = model.plan(ids, images);
        assertArrayEquals(
            ints(c.required("expanded_ids")), plan.expandedIds(), name + " expansion");
        try (MLXScope generation = root.newChild()) {
          float[] logits =
              c.has("token_id")
                  ? model.decodeStepLogitsForTest(
                      generation, ids, images, c.required("token_id").asInt())
                  : model.prefillLogitsForTest(generation, ids, images);
          assertArrayEquals(floats(c.required("logits")), logits, 1e-4f, name + " logits");
        }
      }
    }
  }
}
