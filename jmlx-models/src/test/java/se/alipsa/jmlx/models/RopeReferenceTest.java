package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.RopeSpec;
import tools.jackson.databind.ObjectMapper;

class RopeReferenceTest {
  private static RopeSpec spec(String type) {
    return switch (type) {
      case "default" -> new RopeSpec.Base(10_000);
      case "linear" -> new RopeSpec.Linear(10_000, 8);
      case "dynamic" -> new RopeSpec.DynamicNtk(10_000, 8, 128);
      case "llama3" -> new RopeSpec.Llama3(10_000, 8, 1, 4, 64);
      case "yarn" -> new RopeSpec.Yarn(10_000, 8, 64, 32, 1, 0, 0, null, true);
      default -> throw new AssertionError(type);
    };
  }

  @Test
  void periodsAndAttentionScalingMatchHuggingFace() throws Exception {
    Path golden =
        Path.of(
            System.getProperty("jmlx.repository.root"),
            "tools",
            "hf-reference",
            "goldens",
            "rope.json");
    var samples = new ObjectMapper().readTree(golden.toFile()).path("samples");
    for (var sample : samples) {
      String type = sample.path("rope_type").asString();
      int offset = sample.path("offset").asInt();
      RopeSpec spec = spec(type);
      double[] periods = spec.frequencies(16, Math.max(128, offset + 1));
      for (int i = 0; i < periods.length; i++) {
        double expected = sample.path("inv_freq").get(i).asDouble();
        if (spec instanceof RopeSpec.Linear) {
          expected *= 8;
        }
        double expectedPeriod = 1 / expected;
        assertEquals(
            expectedPeriod,
            periods[i],
            Math.max(1e-5, expectedPeriod * 1e-6),
            type + " offset=" + offset + " i=" + i);
      }
      assertEquals(
          sample.path("attention_scaling").asDouble(), spec.attentionScaling(), 1e-6, type);
    }
  }

  @Test
  @EnabledIfNativeAvailable
  void rotatedOutputsMatchHuggingFace() throws Exception {
    Path golden =
        Path.of(
            System.getProperty("jmlx.repository.root"),
            "tools",
            "hf-reference",
            "goldens",
            "rope.json");
    var samples = new ObjectMapper().readTree(golden.toFile()).path("samples");
    try (MLXScope scope = new MLXScope()) {
      for (var sample : samples) {
        float[] input = new float[16];
        float[] expected = new float[16];
        for (int i = 0; i < 16; i++) {
          input[i] = (float) sample.path("input").get(i).asDouble();
          expected[i] = (float) sample.path("rotated").get(i).asDouble();
        }
        MLXArray x = MLX.array(scope, input, new int[] {1, 1, 1, 16});
        float[] actual =
            spec(sample.path("rope_type").asString())
                .apply(x, 16, sample.path("offset").asInt(), null)
                .toFloatArray();
        assertArrayEquals(
            expected,
            actual,
            2e-4f,
            sample.path("rope_type").asString() + " offset=" + sample.path("offset").asInt());
      }
    }
  }
}
