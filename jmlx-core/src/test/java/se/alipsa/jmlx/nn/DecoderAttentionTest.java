package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXFast;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class DecoderAttentionTest {

  @Test
  void fullAttentionMatchesGroupedQueryAndParameterNames() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray q = MLX.array(scope, identity(4), new int[] {4, 4});
      MLXArray k = MLX.array(scope, new float[] {1, 0, 0, 0, 0, 1, 0, 0}, new int[] {2, 4});
      MLXArray v = MLX.array(scope, new float[] {0, 0, 1, 0, 0, 0, 0, 1}, new int[] {2, 4});
      MLXArray out = MLX.array(scope, identity(4), new int[] {4, 4});
      GroupedQueryAttention original =
          new GroupedQueryAttention(scope, 2, 1, 10000f, q, null, k, null, v, null, out, null);
      DecoderAttention configurable =
          new DecoderAttention(
              scope,
              2,
              1,
              2,
              new RopeSpec.Base(10000f),
              2,
              null,
              null,
              new Linear(scope, q, null),
              new Linear(scope, k, null),
              new Linear(scope, v, null),
              new Linear(scope, out, null));
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4, 2, 1, 4, 3}, new int[] {1, 2, 4});
      assertArrayEquals(
          original.forward(x, null).toFloatArray(),
          configurable.forward(x, null).toFloatArray(),
          1e-5f);
      assertEquals(original.parameters().keySet(), configurable.parameters().keySet());
    }
  }

  @Test
  void explicitHeadDimAndWindowPreflight() {
    try (MLXScope scope = new MLXScope()) {
      DecoderAttention attention =
          new DecoderAttention(
              scope,
              2,
              1,
              4,
              new RopeSpec.Base(10000f),
              4,
              null,
              2,
              new Linear(scope, MLX.zeros(scope, new int[] {8, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {4, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {4, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {4, 8}, DType.FLOAT32), null));
      KVCache cache = new KVCache(scope);
      MLXArray prompt = MLX.ones(scope, new int[] {1, 3, 4}, DType.FLOAT32);
      IllegalArgumentException error =
          assertThrows(IllegalArgumentException.class, () -> attention.forward(prompt, cache));
      assertTrue(error.getMessage().contains("slidingWindow"));
      assertEquals(0, cache.offset());
      MLXArray mask = AttentionMask.slidingWindow(scope, 3, 3, 2);
      assertArrayEquals(new int[] {1, 3, 4}, attention.forward(prompt, cache, mask).shape());
      assertEquals(3, cache.offset());
    }
  }

  @Test
  void explicitHeadDimensionCanDifferFromHiddenPerHeadWidth() {
    try (MLXScope scope = new MLXScope()) {
      DecoderAttention attention =
          new DecoderAttention(
              scope,
              2,
              1,
              3,
              new RopeSpec.Base(10000f),
              2,
              null,
              null,
              new Linear(scope, MLX.zeros(scope, new int[] {6, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {3, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {3, 4}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {4, 6}, DType.FLOAT32), null));
      MLXArray x = MLX.ones(scope, new int[] {1, 2, 4}, DType.FLOAT32);
      assertArrayEquals(new int[] {1, 2, 4}, attention.forward(x, null).shape());
    }
  }

  @Test
  @Tag("full-float32")
  void windowedLastRowMatchesDirectAttentionAtPrefillAndDecodeOffsets() {
    try (MLXScope scope = new MLXScope()) {
      DecoderAttention attention =
          new DecoderAttention(
              scope,
              1,
              1,
              2,
              new RopeSpec.Base(10000f),
              2,
              null,
              2,
              new Linear(scope, MLX.zeros(scope, new int[] {2, 2}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {2, 2}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {2, 2}, DType.FLOAT32), null),
              new Linear(scope, MLX.zeros(scope, new int[] {2, 2}, DType.FLOAT32), null));
      MLXArray queries =
          MLX.array(
              scope, new float[] {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, new int[] {1, 1, 6, 2});
      MLXArray keys =
          MLX.array(
              scope, new float[] {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1}, new int[] {1, 1, 6, 2});
      MLXArray values =
          MLX.array(
              scope, new float[] {0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8}, new int[] {1, 1, 6, 2});
      RopeSpec rope = new RopeSpec.Base(10000f);
      queries = rope.apply(queries, 2, 0, null);
      keys = rope.apply(keys, 2, 0, null);
      MLXArray lastQuery = MLXShape.slice(queries, new int[] {0, 0, 5, 0}, new int[] {1, 1, 6, 2});
      MLXArray lastKeys = MLXShape.slice(keys, new int[] {0, 0, 4, 0}, new int[] {1, 1, 6, 2});
      MLXArray lastValues = MLXShape.slice(values, new int[] {0, 0, 4, 0}, new int[] {1, 1, 6, 2});
      MLXArray direct =
          MLXFast.scaledDotProductAttention(
              lastQuery, lastKeys, lastValues, (float) (1 / Math.sqrt(2)), false, null, null);
      MLXArray prefill = attention.attend(queries, keys, values, 0);
      MLXArray prefillLast =
          MLXShape.slice(prefill, new int[] {0, 0, 5, 0}, new int[] {1, 1, 6, 2});
      assertArrayEquals(direct.toFloatArray(), prefillLast.toFloatArray(), 1e-5f);
      assertArrayEquals(
          direct.toFloatArray(),
          attention.attend(lastQuery, keys, values, 5).toFloatArray(),
          1e-5f);
    }
  }

  private static DecoderAttention qkNormAttention(
      MLXScope scope, float queryScale, float keyScale) {
    float[] queryWeight = new float[] {queryScale, queryScale, queryScale, queryScale};
    float[] keyWeight = new float[] {keyScale, keyScale, keyScale, keyScale};
    return new DecoderAttention(
        scope,
        1,
        1,
        4,
        new RopeSpec.Base(10000f),
        4,
        null,
        null,
        new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
        new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
        new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
        new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
        new RMSNorm(scope, MLX.array(scope, queryWeight, new int[] {4}), 1e-6f),
        new RMSNorm(scope, MLX.array(scope, keyWeight, new int[] {4}), 1e-6f));
  }

  private static float rmsRms(float[] v, float eps) {
    double sum = 0;
    for (float e : v) {
      sum += (double) e * e;
    }
    return (float) Math.sqrt(sum / v.length + eps);
  }

  @Test
  @Tag("full-float32")
  void qkNormNormalizesProjectionOutputBeforeRope() {
    try (MLXScope scope = new MLXScope()) {
      DecoderAttention attention = qkNormAttention(scope, 2f, 4f);
      assertTrue(attention.parameters().containsKey("queryNorm.weight"));
      assertTrue(attention.parameters().containsKey("keyNorm.weight"));
      // Identity projections: each head equals the input row before normalization.
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4, 2, 1, 4, 3}, new int[] {1, 2, 4});
      float[][] rows = {{1, 2, 3, 4}, {2, 1, 4, 3}};
      float[] qn = new float[8];
      float[] kn = new float[8];
      for (int t = 0; t < 2; t++) {
        float rms = rmsRms(rows[t], 1e-6f);
        for (int d = 0; d < 4; d++) {
          qn[t * 4 + d] = 2f * rows[t][d] / rms;
          kn[t * 4 + d] = 4f * rows[t][d] / rms;
        }
      }
      RopeSpec rope = new RopeSpec.Base(10000f);
      MLXArray q = rope.apply(MLX.array(scope, qn, new int[] {1, 1, 2, 4}), 4, 0, null);
      MLXArray k = rope.apply(MLX.array(scope, kn, new int[] {1, 1, 2, 4}), 4, 0, null);
      MLXArray v = MLXShape.reshape(x, new int[] {1, 1, 2, 4});
      MLXArray expected = MLXFast.scaledDotProductAttention(q, k, v, 0.5f, true, null, null);
      assertArrayEquals(expected.toFloatArray(), attention.forward(x, null).toFloatArray(), 1e-5f);
    }
  }

  @Test
  @Tag("full-float32")
  void qkNormNormalizedKeysEnterTheCache() {
    try (MLXScope scope = new MLXScope()) {
      // Decoding from the cache must equal a full prefill: keys are normalized once, at creation,
      // so a second normalization on later steps would change the numbers.
      DecoderAttention attention = qkNormAttention(scope, 2f, 4f);
      MLXArray prompt = MLX.array(scope, new float[] {1, 2, 3, 4, 2, 1, 4, 3}, new int[] {1, 2, 4});
      float[] full = attention.forward(prompt, null).toFloatArray();
      KVCache cache = new KVCache(scope);
      MLXArray row0 = MLXShape.slice(prompt, new int[] {0, 0, 0}, new int[] {1, 1, 4});
      attention.forward(row0, cache);
      MLXArray row1 = MLXShape.slice(prompt, new int[] {0, 1, 0}, new int[] {1, 2, 4});
      float[] decoded = attention.forward(row1, cache).toFloatArray();
      assertArrayEquals(new float[] {full[4], full[5], full[6], full[7]}, decoded, 1e-5f);
    }
  }

  @Test
  void qkNormRejectsWrongWeightDimension() {
    try (MLXScope scope = new MLXScope()) {
      RMSNorm wrongDim =
          new RMSNorm(scope, MLX.array(scope, new float[] {1, 1, 1}, new int[] {3}), 1e-6f);
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new DecoderAttention(
                  scope,
                  1,
                  1,
                  4,
                  new RopeSpec.Base(10000f),
                  4,
                  null,
                  null,
                  new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
                  new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
                  new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
                  new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
                  wrongDim,
                  wrongDim));
    }
  }

  private static float[] identity(int size) {
    float[] data = new float[size * size];
    for (int i = 0; i < size; i++) {
      data[i * size + i] = 1;
    }
    return data;
  }

  private static DecoderAttention identityAttention(MLXScope scope, RopeSpec rope) {
    return new DecoderAttention(
        scope,
        1,
        1,
        4,
        rope,
        4,
        null,
        null,
        new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
        new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
        new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null),
        new Linear(scope, MLX.array(scope, identity(4), new int[] {4, 4}), null));
  }

  @Test
  void callerMaskIsAppliedWithoutASlidingWindow() {
    try (MLXScope scope = new MLXScope()) {
      DecoderAttention attention = identityAttention(scope, new RopeSpec.Base(10000f));
      MLXArray x = MLX.array(scope, new float[] {4, 0, 0, 0, 0, 4, 0, 0}, new int[] {1, 2, 4});
      // Causal attention lets query 1 prefer its own key, so row 1 is about [0, 4, 0, 0]; a mask
      // allowing only key 0 forces row 1 to the value of key 0, [4, 0, 0, 0].
      MLXArray onlyKeyZero =
          MLX.astype(MLX.array(scope, new float[] {1, 0, 1, 0}, new int[] {2, 2}), DType.BOOL);
      float[] causal = attention.forward(x, null, null).toFloatArray();
      float[] masked = attention.forward(x, null, onlyKeyZero).toFloatArray();
      assertTrue(causal[4] < 0.5f, "causal row 1 col 0 was " + causal[4]);
      assertEquals(4f, masked[4], 1e-3f);
    }
  }

  @Test
  void malformedMaskIsRejectedWithoutASlidingWindow() {
    try (MLXScope scope = new MLXScope()) {
      DecoderAttention attention = identityAttention(scope, new RopeSpec.Base(10000f));
      MLXArray x = MLX.ones(scope, new int[] {1, 2, 4}, DType.FLOAT32);
      MLXArray wrongShape = MLX.full(scope, new int[] {3, 3}, 1f, DType.BOOL);
      assertThrows(IllegalArgumentException.class, () -> attention.forward(x, null, wrongShape));
    }
  }

  @Test
  void staticScalingSpecsShareOneFrequencyUploadAcrossDirectConstruction() {
    try (MLXScope scope = new MLXScope()) {
      final RopeSpec llama3 = new RopeSpec.Llama3(10000f, 8f, 1f, 4f, 64);
      assertNull(new RopeSpec.Base(10000f).staticFrequencies(scope, 4));
      assertNull(new RopeSpec.Linear(10000f, 2f).staticFrequencies(scope, 4));
      assertNull(new RopeSpec.DynamicNtk(10000f, 2f, 8).staticFrequencies(scope, 4));
      assertArrayEquals(new int[] {2}, llama3.staticFrequencies(scope, 4).shape());
      MLXArray x = MLX.ones(scope, new int[] {1, 2, 4}, DType.FLOAT32);
      assertArrayEquals(
          new int[] {1, 2, 4}, identityAttention(scope, llama3).forward(x, null).shape());
    }
  }

  @Test
  void batchedForwardRejectsWrongSequenceWidthAndCacheBatchSize() {
    try (MLXScope scope = new MLXScope()) {
      DecoderAttention attention = identityAttention(scope, new RopeSpec.Base(10000f));
      MLXArray x = MLX.ones(scope, new int[] {2, 2, 4}, DType.FLOAT32);
      assertThrows(
          IllegalArgumentException.class,
          () -> attention.forward(x, null, null, null, new int[] {1, 1}));
      KVCache oneRow = new KVCache(scope);
      MLXArray row = MLX.ones(scope, new int[] {1, 1, 2, 4}, DType.FLOAT32);
      oneRow.append(row, row);
      assertThrows(
          IllegalArgumentException.class,
          () -> attention.forward(x, oneRow, null, null, new int[] {2, 2}));
    }
  }
}
