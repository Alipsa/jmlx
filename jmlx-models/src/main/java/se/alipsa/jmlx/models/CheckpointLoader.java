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
    java.util.List<Path> checkpointFiles = checkpointFiles(directory);
    for (Path file : checkpointFiles) {
      for (var entry : MLXIO.loadSafetensors(scope, file.toString()).tensors().entrySet()) {
        if (tensors.putIfAbsent(entry.getKey(), entry.getValue()) != null) {
          throw new IllegalArgumentException(
              "duplicate tensor in checkpoint shards: " + entry.getKey());
        }
      }
    }
    if (tensors.isEmpty()) {
      throw new IllegalArgumentException(
          "checkpoint contained no tensors"
              + (checkpointFiles.isEmpty() ? " or safetensors files" : "")
              + " in "
              + directory.toAbsolutePath().normalize());
    }
    return tensors;
  }

  private static java.util.List<Path> checkpointFiles(Path directory) throws IOException {
    Path root = directory.toAbsolutePath().normalize();
    Path index = root.resolve("model.safetensors.index.json");
    if (Files.isRegularFile(index)) {
      JsonNode weights = new ObjectMapper().readTree(index.toFile()).path("weight_map");
      if (!weights.isObject())
        throw new IllegalArgumentException("invalid safetensors index: missing weight_map");
      java.util.List<Path> files = new java.util.ArrayList<>();
      java.util.HashSet<String> names = new java.util.HashSet<>();
      weights
          .properties()
          .forEach(
              entry -> {
                if (!entry.getValue().isString()) {
                  throw new IllegalArgumentException(
                      "invalid safetensors index: weight_map values must be strings");
                }
                names.add(entry.getValue().asString());
              });
      for (String name : names.stream().sorted().toList()) {
        Path file = root.resolve(name).normalize();
        if (!file.startsWith(root)) {
          throw new IllegalArgumentException(
              "safetensors index shard escapes checkpoint directory: " + name);
        }
        if (!Files.isRegularFile(file)) {
          throw new IllegalArgumentException(
              "safetensors index references missing regular shard " + name);
        }
        files.add(file);
      }
      return files;
    }
    try (var files = Files.list(root)) {
      return files
          .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
          .sorted()
          .toList();
    }
  }
}
