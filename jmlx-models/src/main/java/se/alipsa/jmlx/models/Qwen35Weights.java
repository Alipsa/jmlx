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
    return load(target, checked);
  }

  static Loaded load(MLXScope target, Qwen35Checkpoint.Checked checked) {
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
      // Tied-head storage is verified at the byte level by Qwen35Checkpoint.inspect (which also
      // drops the aliased lm_head names from the load inventory), so no loaded lm_head weights can
      // reach this path when tied: there is nothing to re-check here.
      Map<String, MLXArray> retained = new LinkedHashMap<>();
      result.forEach((key, value) -> retained.put(key, MLX.hoist(value, target)));
      return new Loaded(checked, retained);
    }
  }
}
