package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Cached decode after visual-scope closure versus fresh, fully recomputed image/history logits. */
@Tag("batch-equivalence")
@EnabledIfNativeAvailable
class Qwen3VlCacheEquivalenceTest {
  @Test
  void cachedAndFullHistoryAgreeWithRequestLocalPositionsAndGeneratedMarkers() throws Exception {
    boolean strict = "0".equals(System.getenv("MLX_ENABLE_TF32"));
    // Default TF32 cached/full maximum measured at 0.0021870732 on the pinned checkpoint.
    // Keep the strict float32 gate independent of that reduced-precision family bound.
    float bound = strict ? 1e-4f : 0.003f;
    float maximum = 0;
    // Includes every special token and a complete wrapper-looking continuation.
    int[] forced = {7, 10, 11, 12, 13, 12, 10, 13, 6};
    try (MLXScope root = new MLXScope()) {
      Qwen3VlModel model = Qwen3VlModel.load(root, Qwen3VlModelFloatGoldenTest.directory());
      for (var request : Qwen3VlTestSupport.requests(10)) {
        var plan = model.plan(request.promptTokenIds(), request.images());
        for (boolean greedy : List.of(true, false)) {
          try (MLXScope generation = root.newChild()) {
            var caches = Qwen3VlTestSupport.caches(model, generation);
            float[] cached;
            try (MLXScope activation = generation.newChild()) {
              cached =
                  model
                      .prefill(CancellationToken.NONE, request.images(), plan, activation, caches)
                      .logits()
                      .toFloatArray();
            }
            int[] history = new int[0];
            for (int step = 0; step <= forced.length; step++) {
              float[] full;
              try (MLXScope reference = root.newChild()) {
                full = Qwen3VlTestSupport.fullLogits(model, reference, request, history);
              }
              assertEquals(full.length, cached.length);
              for (int i = 0; i < full.length; i++) {
                assertTrue(Float.isFinite(cached[i]));
                maximum = Math.max(maximum, Math.abs(full[i] - cached[i]));
              }
              assertEquals(argmax(full), argmax(cached), "greedy ID at step " + step);
              assertEquals(plan.expandedLength() + step, caches.getFirst().nextPosition());
              if (step == forced.length) {
                break;
              }
              int token = greedy ? argmax(full) : forced[step];
              history = Arrays.copyOf(history, history.length + 1);
              history[history.length - 1] = token;
              try (MLXScope activation = generation.newChild()) {
                cached =
                    Qwen3VlTestSupport.decode(model, activation, plan, caches, token)
                        .toFloatArray();
                MLX.eval(DecoderModel.stepCacheArrays(caches));
              }
            }
          }
        }
      }
    }
    System.out.println("Qwen3-VL cached/full TF32=" + !strict + " maximum_logit_error=" + maximum);
    assertTrue(maximum <= bound, "cached/full logit error " + maximum + " exceeds " + bound);
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }
}
