package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.databind.ObjectMapper;

/** Actual converter parameters qualify one-time shifts/transposes and packed storage retention. */
@EnabledIfNativeAvailable
@Tag("full-float32")
class Qwen35WeightsTest {
  private static final Path FIXTURES = Path.of("..", "tools", "qwen35-reference", "fixtures");
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void actualFloatConverterOutputsNormalizeToBitIdenticalHfParameters() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> hf = load(scope, "hf").tensors();
      for (String kind : Set.of("vlm-float", "lm-float")) {
        Map<String, MLXArray> converted = load(scope, kind).tensors();
        for (var entry : converted.entrySet()) {
          MLXArray expected = hf.get(entry.getKey());
          assertEquals(DType.FLOAT32, entry.getValue().dtype());
          assertArrayEquals(expected.shape(), entry.getValue().shape(), entry.getKey());
          assertArrayEquals(
              expected.toFloatArray(), entry.getValue().toFloatArray(), entry.getKey());
        }
        assertEquals(
            hf.size(),
            converted.size()
                + (kind.startsWith("lm")
                    ? (int) hf.keySet().stream().filter(k -> k.startsWith("vision.")).count()
                    : 0));
      }
    }
  }

  @Test
  void actualAffineOutputsRetainPackedBitsAndUpcastOnlyFloatingParameters() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      for (String kind : Set.of("vlm-affine", "lm-affine")) {
        Qwen35Weights.Loaded loaded = load(scope, kind);
        Qwen35Checkpoint.Checked checked = loaded.checkpoint();
        assertTrue(checked.descriptor().affine());
        int packed = 0;
        for (Path shard : checked.shards()) {
          try (MLXScope rawScope = scope.newChild()) {
            Map<String, MLXArray> raw = MLXIO.loadSafetensors(rawScope, shard.toString()).tensors();
            for (var entry : checked.names().entrySet()) {
              MLXArray source = raw.get(entry.getValue());
              if (source == null) {
                continue;
              }
              MLXArray actual = loaded.tensors().get(entry.getKey());
              if (source.dtype() == DType.UINT32) {
                packed++;
                assertEquals(DType.UINT32, actual.dtype());
                assertArrayEquals(source.shape(), actual.shape());
                // Exact equality preserves every bit, including the high sign bit of packed U32.
                assertEquals(
                    1,
                    MLX.astype(
                            se.alipsa.jmlx.core.MLXOps.all(
                                se.alipsa.jmlx.core.MLXOps.equal(source, actual)),
                            DType.INT32)
                        .toIntArray()[0]);
              } else {
                assertEquals(DType.FLOAT32, actual.dtype());
                if (!entry.getKey().equals("vision.patch_embed.proj.weight")) {
                  assertArrayEquals(
                      MLX.astype(source, DType.FLOAT32).toFloatArray(), actual.toFloatArray());
                }
              }
            }
          }
        }
        assertTrue(packed > 0);
        assertFalse(loaded.tensors().keySet().stream().anyMatch(k -> k.contains("mtp.")));
      }
    }
  }

  private static Qwen35Weights.Loaded load(MLXScope scope, String kind) throws Exception {
    Path directory = FIXTURES.resolve(kind);
    return Qwen35Weights.load(
        scope, directory, MAPPER.readTree(directory.resolve("config.json").toFile()));
  }
}
