package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

final class CheckpointLoader {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final System.Logger LOGGER = System.getLogger(CheckpointLoader.class.getName());

  @FunctionalInterface
  interface Loader {
    MLXIO.SafetensorsResult load(MLXScope scope, Path shard);
  }

  record Preflight(List<Path> shards, Set<String> tensorNames) {}

  /** Shards to read plus the validated index {@code weight_map}, or null without an index. */
  private record CheckpointFiles(List<Path> shards, JsonNode weightMap) {}

  private CheckpointLoader() {}

  static Map<String, MLXArray> load(MLXScope scope, Path directory, TensorPlan plan)
      throws IOException {
    return load(scope, directory, plan, (s, shard) -> MLXIO.loadSafetensors(s, shard.toString()));
  }

  static Map<String, MLXArray> load(MLXScope scope, Path directory, TensorPlan plan, Loader loader)
      throws IOException {
    Preflight checked = preflight(directory, plan);
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(loader, "loader");
    Map<String, MLXArray> tensors = new LinkedHashMap<>();
    for (Path shard : checked.shards()) {
      for (var entry : loader.load(scope, shard).tensors().entrySet()) {
        if (tensors.putIfAbsent(entry.getKey(), entry.getValue()) != null) {
          throw new IllegalArgumentException(
              "duplicate tensor in checkpoint shards: " + entry.getKey());
        }
      }
    }
    return tensors;
  }

  static Preflight preflight(Path directory, TensorPlan plan) throws IOException {
    Objects.requireNonNull(directory, "directory");
    Objects.requireNonNull(plan, "plan");
    Path root = directory.toAbsolutePath().normalize();
    CheckpointFiles files = checkpointFiles(directory);
    List<Path> shards = files.shards();
    Map<Path, Set<String>> byShard = SafetensorsHeaders.tensorNames(shards);
    Map<String, Path> actual = new LinkedHashMap<>();
    for (var entry : byShard.entrySet()) {
      for (String key : entry.getValue()) {
        Path previous = actual.putIfAbsent(key, entry.getKey());
        if (previous != null) {
          throw new IllegalArgumentException(
              "duplicate tensor '" + key + "' in " + previous + " and " + entry.getKey());
        }
      }
    }
    JsonNode weights = files.weightMap();
    if (weights != null) {
      for (var entry : weights.properties()) {
        String key = entry.getKey();
        Path found = actual.get(key);
        if (found == null) {
          throw new IllegalArgumentException("index tensor '" + key + "' absent from every shard");
        }
        Path expected = root.resolve(entry.getValue().asString()).normalize();
        if (!found.equals(expected)) {
          throw new IllegalArgumentException(
              "index tensor '" + key + "' maps to " + expected + " but is in " + found);
        }
      }
      for (String key : actual.keySet()) {
        if (!weights.has(key)) {
          throw new IllegalArgumentException("shard tensor '" + key + "' absent from index");
        }
      }
    }
    plan.validate(actual.keySet());
    return new Preflight(List.copyOf(shards), Set.copyOf(actual.keySet()));
  }

  private static CheckpointFiles checkpointFiles(Path directory) throws IOException {
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
      return new CheckpointFiles(files, weights);
    }
    try (var files = Files.list(root)) {
      List<Path> selected =
          files
              .filter(Files::isRegularFile)
              .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
              .sorted()
              .toList();
      boolean hasModel =
          selected.stream().anyMatch(p -> p.getFileName().toString().startsWith("model"));
      if (hasModel) {
        Path consolidated = root.resolve("consolidated.safetensors");
        if (selected.contains(consolidated)) {
          LOGGER.log(System.Logger.Level.INFO, "ignoring " + consolidated);
          return new CheckpointFiles(
              selected.stream().filter(p -> !p.equals(consolidated)).toList(), null);
        }
      }
      return new CheckpointFiles(selected, null);
    }
  }
}
