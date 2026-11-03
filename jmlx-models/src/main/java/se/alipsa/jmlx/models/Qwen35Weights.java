package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.databind.JsonNode;

/** Loads only a fully preflighted parameter inventory, normalizing layout exactly once. */
final class Qwen35Weights {
  record Loaded(Qwen35Checkpoint.Checked checkpoint, Map<String, MLXArray> tensors) {
    Loaded {
      tensors = Map.copyOf(tensors);
    }
  }

  private Qwen35Weights() {}

  static Loaded load(MLXScope target, Path directory, JsonNode config) throws IOException {
    // Deliberately precedes even the first native scope/array operation.
    Qwen35Checkpoint.Checked checked = Qwen35Checkpoint.inspect(directory, config);
    Map<String, MLXArray> result = new LinkedHashMap<>();
    Map<String, String> canonical = new HashMap<>();
    checked.names().forEach((key, raw) -> canonical.put(raw, key));
    try (MLXScope loading = target.newChild()) {
      for (Path shard : checked.shards()) {
        Set<String> selected =
            canonical.keySet().stream()
                .filter(k -> checked.inventory().get(k).shard().equals(shard))
                .collect(Collectors.toSet());
        if (selected.isEmpty()) {
          continue;
        }
        try (MLXScope batch = loading.newChild()) {
          Map<String, MLXArray> raw =
              MLXIO.loadSafetensors(batch, shard.toString(), selected).tensors();
          Map<String, MLXArray> transformed = new LinkedHashMap<>();
          for (var entry : raw.entrySet()) {
            String name = canonical.get(entry.getKey());
            MLXArray value = entry.getValue();
            if (value.dtype() != DType.UINT32) {
              value = MLX.astype(value, DType.FLOAT32);
            }
            if (checked.shiftedNorms().contains(name)) {
              value = MLXOps.add(value, MLX.array(batch, new float[] {1f}, new int[] {1}));
            }
            if (checked.transposedConvolutions().contains(name)) {
              value = MLXShape.transpose(value, new int[] {0, 2, 1});
            }
            if (checked.transposePatch() && name.equals("vision.patch_embed.proj.weight")) {
              value = MLXShape.transpose(value, new int[] {0, 2, 3, 4, 1});
            }
            transformed.put(name, value);
          }
          MLX.eval(transformed.values().toArray(MLXArray[]::new));
          transformed.forEach((key, value) -> result.put(key, MLX.hoist(value, loading)));
        }
      }
      validateTie(checked.descriptor().text(), result, loading);
      Map<String, MLXArray> retained = new LinkedHashMap<>();
      result.forEach((key, value) -> retained.put(key, MLX.hoist(value, target)));
      return new Loaded(checked, retained);
    }
  }

  private static void validateTie(
      Qwen35Descriptor.Text text, Map<String, MLXArray> weights, MLXScope scope) {
    if (!text.tied() || !weights.containsKey("lm_head.weight")) {
      return;
    }
    for (String suffix : new String[] {"weight", "scales", "biases"}) {
      MLXArray embedding = weights.get("text.embed_tokens." + suffix);
      MLXArray head = weights.get("lm_head." + suffix);
      if (embedding == null && head == null) {
        continue;
      }
      if (embedding == null
          || head == null
          || embedding.dtype() != head.dtype()
          || !java.util.Arrays.equals(embedding.shape(), head.shape())) {
        throw new IllegalArgumentException(
            "Qwen3.5 tied lm_head has a different storage convention");
      }
      try (MLXScope comparison = scope.newChild()) {
        MLXArray left =
            MLXShape.reshape(MLXShape.expandDims(embedding, comparison, 0), embedding.shape());
        MLXArray equal = MLX.astype(MLXOps.all(MLXOps.equal(left, head)), DType.INT32);
        if (equal.toIntArray()[0] != 1) {
          throw new IllegalArgumentException(
              "Qwen3.5 tied lm_head differs from embedding: " + suffix);
        }
      }
    }
  }
}
