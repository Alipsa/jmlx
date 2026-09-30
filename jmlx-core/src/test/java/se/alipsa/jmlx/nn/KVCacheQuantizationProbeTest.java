package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Records cache-shaped affine quantization capabilities on the pinned native runtime. */
@EnabledIfNativeAvailable
class KVCacheQuantizationProbeTest {
  @Test
  void cacheShapesAndAttention() {
    int successes = 0;
    for (int headDim : new int[] {16, 32, 64}) {
      for (int bits : new int[] {4, 8}) {
        for (int groupSize : new int[] {16, 32, 64}) {
          if (groupSize > headDim) {
            continue;
          }
          try (MLXScope scope = new MLXScope()) {
            int length = 64;
            float[] data = new float[length * headDim];
            float[] queryData = new float[headDim];
            for (int i = 0; i < data.length; i++) {
              data[i] = (float) Math.sin(i * 0.031) * 0.25f;
            }
            for (int i = 0; i < headDim; i++) {
              queryData[i] = (float) Math.cos(i * 0.07) * 0.2f;
            }
            MLXArray source = MLX.array(scope, data, new int[] {1, 1, length, headDim});
            MLXArray query = MLX.array(scope, queryData, new int[] {1, 1, 1, headDim});
            try {
              MLXArray[] packed = MLXQuant.quantize(source, groupSize, bits, "affine", null);
              MLX.eval(packed);
              final long packedActive = MLXMemory.activeBytes();
              final long packedBytes =
                  Arrays.stream(packed).mapToLong(KVCacheQuantizationProbeTest::bytes).sum();
              MLXArray restored =
                  MLXQuant.dequantize(
                      packed[0], packed[1], packed[2], groupSize, bits, "affine", null, null);
              MLX.eval(restored);
              float maxError = 0;
              float[] roundTrip = restored.toFloatArray();
              for (int i = 0; i < data.length; i++) {
                maxError = Math.max(maxError, Math.abs(data[i] - roundTrip[i]));
              }
              MLXMemory.resetPeak();
              MLXArray output =
                  MLXFast.scaledDotProductAttention(
                      query,
                      restored,
                      restored,
                      1f / (float) Math.sqrt(headDim),
                      false,
                      null,
                      null);
              MLX.eval(output);
              long dequantAttentionPeak = MLXMemory.peakBytes();
              String direct;
              try {
                MLXArray directOutput =
                    MLXFast.scaledDotProductAttention(
                        query,
                        packed[0],
                        packed[0],
                        1f / (float) Math.sqrt(headDim),
                        false,
                        null,
                        null);
                MLX.eval(directOutput);
                direct = "accepted";
              } catch (RuntimeException ex) {
                direct = "rejected:" + ex.getClass().getSimpleName() + ":" + ex.getMessage();
              }
              System.out.printf(
                  "KV_QUANT_PROBE headDim=%d bits=%d group=%d packedShape=%s scaleShape=%s"
                      + " packedBytes=%d floatBytes=%d maxError=%g packedActive=%d"
                      + " dequantAttentionPeak=%d directSdpa=%s%n",
                  headDim,
                  bits,
                  groupSize,
                  Arrays.toString(packed[0].shape()),
                  Arrays.toString(packed[1].shape()),
                  packedBytes,
                  bytes(source),
                  maxError,
                  packedActive,
                  dequantAttentionPeak,
                  direct.replace('\n', ' '));
              successes++;
            } catch (RuntimeException ex) {
              System.out.printf(
                  "KV_QUANT_PROBE headDim=%d bits=%d group=%d unsupported=%s:%s%n",
                  headDim,
                  bits,
                  groupSize,
                  ex.getClass().getSimpleName(),
                  ex.getMessage().replace('\n', ' '));
            }
          }
        }
      }
    }
    assertTrue(successes > 0, "pinned runtime has no usable cache-shaped affine quantization");
  }

  @Test
  void paddedSixteenDimensionalHead() {
    int headDim = 16;
    int length = 64;
    float[] values = new float[length * headDim];
    for (int i = 0; i < values.length; i++) {
      values[i] = (float) Math.sin(i * 0.031) * 0.25f;
    }
    try (MLXScope scope = new MLXScope()) {
      MLXArray source = MLX.array(scope, values, new int[] {1, 1, length, headDim});
      MLXArray zeroPad = MLX.zeros(scope, new int[] {1, 1, length, headDim}, source.dtype());
      MLXArray padded = MLXShape.concatenate(new MLXArray[] {source, zeroPad}, 3);
      MLXArray[] packed = MLXQuant.quantize(padded, 32, 4, "affine", null);
      MLX.eval(packed);
      final long packedBytes =
          Arrays.stream(packed).mapToLong(KVCacheQuantizationProbeTest::bytes).sum();
      MLXMemory.resetPeak();
      MLXArray unpacked =
          MLXQuant.dequantize(packed[0], packed[1], packed[2], 32, 4, "affine", null, null);
      MLXArray restored =
          MLXShape.slice(unpacked, new int[] {0, 0, 0, 0}, new int[] {1, 1, length, 16});
      MLX.eval(restored);
      long peak = MLXMemory.peakBytes();
      float maxError = 0;
      float[] roundTrip = restored.toFloatArray();
      for (int i = 0; i < values.length; i++) {
        maxError = Math.max(maxError, Math.abs(values[i] - roundTrip[i]));
      }
      System.out.printf(
          "KV_QUANT_PADDED headDim=16 bits=4 group=32 packedBytes=%d floatBytes=%d maxError=%g"
              + " dequantPeak=%d%n",
          packedBytes, bytes(source), maxError, peak);
    }
  }

  private static long bytes(MLXArray array) {
    long count = 1;
    for (int dimension : array.shape()) {
      count *= dimension;
    }
    return count * (array.dtype().name().contains("16") ? 2 : 4);
  }
}
