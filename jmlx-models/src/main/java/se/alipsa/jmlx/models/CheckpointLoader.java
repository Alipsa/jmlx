package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

final class CheckpointLoader {
  private CheckpointLoader() {}

  static Map<String, MLXArray> load(MLXScope scope, Path directory) throws IOException {
    Map<String, MLXArray> tensors = new LinkedHashMap<>();
    for (Path file : checkpointFiles(directory)) {
      for (var entry : MLXIO.loadSafetensors(scope, file.toString()).tensors().entrySet()) {
        if (tensors.putIfAbsent(entry.getKey(), entry.getValue()) != null) {
          throw new IllegalArgumentException(
              "duplicate tensor in checkpoint shards: " + entry.getKey());
        }
      }
    }
    if (tensors.isEmpty())
      throw new IllegalArgumentException("no .safetensors files in " + directory);
    return tensors;
  }

  private static java.util.List<Path> checkpointFiles(Path directory) throws IOException {
    Path index = directory.resolve("model.safetensors.index.json");
    if (Files.isRegularFile(index)) {
      JsonNode weights = new ObjectMapper().readTree(index.toFile()).path("weight_map");
      if (!weights.isObject())
        throw new IllegalArgumentException("invalid safetensors index: missing weight_map");
      java.util.List<Path> files = new java.util.ArrayList<>();
      java.util.HashSet<String> names = new java.util.HashSet<>();
      weights.properties().forEach(entry -> names.add(entry.getValue().asString()));
      for (String name : names.stream().sorted().toList()) {
        Path file = directory.resolve(name).normalize();
        if (!file.startsWith(directory.normalize()) || !Files.isRegularFile(file)) {
          throw new IllegalArgumentException("safetensors index references missing shard " + name);
        }
        files.add(file);
      }
      return files;
    }
    try (var files = Files.list(directory)) {
      return files
          .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
          .sorted()
          .toList();
    }
  }
}
