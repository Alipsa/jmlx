package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.QuantizedLinear;

/** Native probe for the MLX-community weight/scales/biases safetensors representation. */
@EnabledIfNativeAvailable
class QuantizedCheckpointProbeTest {
  private static final String KEY = "model.layers.0.self_attn.q_proj";

  @Test
  void packedProjectionRoundTripsThroughSafetensors(@TempDir Path directory) throws Exception {
    Path floatCheckpoint = directory.resolve("float");
    TinyCheckpoints.randomLlama(floatCheckpoint, 63L, 2, false, false);
    try (MLXScope modelScope = new MLXScope();
        MLXScope activation = modelScope.newChild()) {
      MLXArray floatWeight =
          MLXIO
              .loadSafetensors(modelScope, floatCheckpoint.resolve("model.safetensors").toString())
              .tensors()
              .get(KEY + ".weight");
      MLXArray[] packed = MLXQuant.quantize(floatWeight, 64, 4, "affine", null);
      Path file = directory.resolve("quantized.safetensors");
      MLXIO.saveSafetensors(
          file.toString(),
          Map.of(
              KEY + ".weight", packed[0],
              KEY + ".scales", packed[1],
              KEY + ".biases", packed[2]),
          Map.of());
      Map<String, MLXArray> loaded = MLXIO.loadSafetensors(modelScope, file.toString()).tensors();
      assertEquals(3, loaded.size());
      QuantizedLinear quantized =
          new QuantizedLinear(
              modelScope,
              loaded.get(KEY + ".weight"),
              loaded.get(KEY + ".scales"),
              loaded.get(KEY + ".biases"),
              null,
              64,
              4);
      Linear floatLayer = new Linear(modelScope, floatWeight, null);
      // Nonzero input exposes packing and orientation errors; an all-zero input would not.
      float[] values = new float[64];
      for (int i = 0; i < values.length; i++) {
        values[i] = (i % 7 - 3) / 7.0f;
      }
      MLXArray input = MLX.array(activation, values, new int[] {1, 64});
      float[] actual = quantized.forward(input).toFloatArray();
      float[] expected = floatLayer.forward(input).toFloatArray();
      assertArrayEquals(expected, actual, 0.08f);
      for (float value : actual) {
        assertTrue(Float.isFinite(value));
      }
    }
  }
}
