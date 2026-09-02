package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.memory.MLXScope;

final class CheckpointLoader {
  private CheckpointLoader() {}

  static Map<String, MLXArray> load(MLXScope scope, Path directory) throws IOException {
    Map<String, MLXArray> tensors = new LinkedHashMap<>();
    try (var files = Files.list(directory)) {
      for (Path file :
          files
              .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
              .sorted()
              .toList()) {
        for (var entry : MLXIO.loadSafetensors(scope, file.toString()).tensors().entrySet()) {
          if (tensors.putIfAbsent(entry.getKey(), entry.getValue()) != null) {
            throw new IllegalArgumentException(
                "duplicate tensor in checkpoint shards: " + entry.getKey());
          }
        }
      }
    }
    if (tensors.isEmpty())
      throw new IllegalArgumentException("no .safetensors files in " + directory);
    return tensors;
  }
}
