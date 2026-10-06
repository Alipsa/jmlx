package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
@Tag("full-float32")
class EncoderAttentionTest {
  @Test
  void staticCacheEnforcesOwnerThreadDimensionsAndLifetime() throws Exception {
    try (MLXScope model = new MLXScope()) {
      MLXScope request = model.newChild();
      StaticKVCache cache = new StaticKVCache(request);
      assertThrows(IllegalStateException.class, cache::keys);
      cache.initialize(
          MLX.array(request, new float[] {1}, new int[] {1, 1, 1, 1}),
          MLX.array(request, new float[] {2}, new int[] {1, 1, 1, 1}));
      assertThrows(IllegalArgumentException.class, () -> cache.validate(1, 1, 2, 1));
      java.util.concurrent.atomic.AtomicReference<Throwable> failure =
          new java.util.concurrent.atomic.AtomicReference<>();
      Thread thread =
          Thread.ofPlatform()
              .start(
                  () -> {
                    try {
                      cache.keys();
                    } catch (Throwable exception) {
                      failure.set(exception);
                    }
                  });
      thread.join();
      assertInstanceOf(IllegalStateException.class, failure.get());
      request.close();
      assertThrows(IllegalStateException.class, cache::keys);
    }
  }

  @Test
  void paddedQueriesUseEveryValidKey() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray mask = AttentionMask.bidirectional(scope, new int[][] {{1, 1, 0}}, 3);
      assertArrayEquals(
          new float[] {1, 1, 0, 1, 1, 0, 1, 1, 0},
          MLX.astype(mask, se.alipsa.jmlx.core.DType.FLOAT32).toFloatArray());
      assertThrows(
          IllegalArgumentException.class,
          () -> AttentionMask.bidirectional(scope, new int[][] {{0, 0}}, 2));
    }
  }

  @Test
  void rectangularCrossAttentionSurvivesProducerClosure() {
    try (MLXScope model = new MLXScope();
        MLXScope request = model.newChild()) {
      Linear query =
          new Linear(model, MLX.array(model, new float[] {1, 0}, new int[] {1, 2}), null);
      Linear key = new Linear(model, MLX.array(model, new float[] {0, 1}, new int[] {1, 2}), null);
      Linear value =
          new Linear(model, MLX.array(model, new float[] {1, 0}, new int[] {1, 2}), null);
      Linear out = new Linear(model, MLX.array(model, new float[] {1, 2}, new int[] {2, 1}), null);
      BidirectionalAttention attention =
          new BidirectionalAttention(model, 1, 1, 1f, query, key, value, out);
      StaticKVCache cache = new StaticKVCache(request);
      try (MLXScope stage = request.newChild()) {
        MLXArray source = MLX.array(stage, new float[] {3, 1, 7, 2, 99, 3}, new int[] {1, 3, 2});
        attention.initialize(source, cache);
        assertThrows(IllegalStateException.class, () -> attention.initialize(source, cache));
      }
      cache.validate(1, 1, 3, 1);
      try (MLXScope step = request.newChild()) {
        MLXArray input = MLX.array(step, new float[] {1, 0}, new int[] {1, 1, 2});
        MLXArray mask = AttentionMask.bidirectional(step, new int[][] {{1, 1, 0}}, 1);
        float probability = (float) (Math.E / (1 + Math.E));
        float expected = 3 * (1 - probability) + 7 * probability;
        assertArrayEquals(
            new float[] {expected, 2 * expected},
            attention.forward(input, cache, mask).toFloatArray(),
            1e-4f);
        MLXArray bias =
            MLX.array(step, new float[] {0, (float) Math.log(2), 0}, new int[] {1, 1, 1, 3});
        probability = (float) (2 * Math.E / (1 + 2 * Math.E));
        expected = 3 * (1 - probability) + 7 * probability;
        assertArrayEquals(
            new float[] {expected, 2 * expected},
            attention.forward(input, cache, mask, bias).toFloatArray(),
            1e-4f);
      }
    }
  }

  @Test
  void halfPrecisionInputsStayHalfPrecision() {
    try (MLXScope model = new MLXScope();
        MLXScope request = model.newChild()) {
      for (DType dtype : List.of(DType.FLOAT16, DType.BFLOAT16)) {
        Linear query =
            new Linear(
                model,
                MLX.astype(MLX.array(model, new float[] {1, 0}, new int[] {1, 2}), dtype),
                null);
        Linear key =
            new Linear(
                model,
                MLX.astype(MLX.array(model, new float[] {0, 1}, new int[] {1, 2}), dtype),
                null);
        Linear value =
            new Linear(
                model,
                MLX.astype(MLX.array(model, new float[] {1, 0}, new int[] {1, 2}), dtype),
                null);
        Linear out =
            new Linear(
                model,
                MLX.astype(MLX.array(model, new float[] {1, 2}, new int[] {2, 1}), dtype),
                null);
        BidirectionalAttention attention =
            new BidirectionalAttention(model, 1, 1, 1f, query, key, value, out);
        StaticKVCache cache = new StaticKVCache(request);
        try (MLXScope stage = request.newChild()) {
          MLXArray source =
              MLX.astype(
                  MLX.array(stage, new float[] {3, 1, 7, 2, 99, 3}, new int[] {1, 3, 2}), dtype);
          attention.initialize(source, cache);
        }
        cache.validate(1, 1, 3, 1);
        try (MLXScope step = request.newChild()) {
          MLXArray input =
              MLX.astype(MLX.array(step, new float[] {1, 0}, new int[] {1, 1, 2}), dtype);
          MLXArray mask = AttentionMask.bidirectional(step, new int[][] {{1, 1, 0}}, 1);
          MLXArray output = attention.forward(input, cache, mask);
          assertEquals(dtype, output.dtype());
          float probability = (float) (Math.E / (1 + Math.E));
          float expected = 3 * (1 - probability) + 7 * probability;
          assertArrayEquals(new float[] {expected, 2 * expected}, output.toFloatArray(), 0.05f);
        }
      }
    }
  }
}
