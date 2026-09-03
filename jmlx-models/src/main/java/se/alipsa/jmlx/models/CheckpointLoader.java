package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

final class CheckpointLoader {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private CheckpointLoader() {}

  static Map<String, MLXArray> load(MLXScope scope, Path directory) throws IOException {
    Map<String, MLXArray> tensors = new LinkedHashMap<>();
    List<Path> checkpointFiles = checkpointFiles(directory);
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

  private static List<Path> checkpointFiles(Path directory) throws IOException {
    Path root = directory.toAbsolutePath().normalize();
    Path index = root.resolve("model.safetensors.index.json");
    if (Files.isRegularFile(index)) {
      JsonNode indexRoot;
      try {
        indexRoot = MAPPER.readTree(index.toFile());
      } catch (JacksonException e) {
        throw new IOException("failed to read " + index, e);
      }
      JsonNode weights = indexRoot.path("weight_map");
      if (!weights.isObject()) {
        throw new IllegalArgumentException("invalid safetensors index: missing weight_map");
      }
      List<Path> files = new ArrayList<>();
      HashSet<String> names = new HashSet<>();
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
          .filter(Files::isRegularFile)
          .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
          .sorted()
          .toList();
    }
  }
}
