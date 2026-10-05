package se.alipsa.jmlx.core;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Shared internal oracle JSON reader with explicit nonfinite sentinels. */
public final class OracleFixtureReader {
  private OracleFixtureReader() {}

  /** Reads a fixture relative to the repository root configured by the Test task. */
  public static JsonNode read(String name) {
    return new ObjectMapper()
        .readTree(
            Path.of(
                    System.getProperty("jmlx.repository.root"),
                    "tools",
                    "mlx-oracle",
                    "fixtures",
                    name)
                .toFile());
  }

  /** Flattens nested arrays and decodes finite values and NaN/Infinity/-Infinity. */
  public static float[] floats(JsonNode node) {
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
      for (JsonNode child : node) {
        flatten(child, values);
      }
    } else if (node.isNumber()) {
      values.add((float) node.doubleValue());
    } else if (node.isTextual()) {
      values.add(
          switch (node.asString()) {
            case "NaN" -> Float.NaN;
            case "Infinity" -> Float.POSITIVE_INFINITY;
            case "-Infinity" -> Float.NEGATIVE_INFINITY;
            default ->
                throw new IllegalArgumentException("unknown oracle sentinel: " + node.asString());
          });
    } else {
      throw new IllegalArgumentException("invalid oracle numeric value: " + node);
    }
  }
}
