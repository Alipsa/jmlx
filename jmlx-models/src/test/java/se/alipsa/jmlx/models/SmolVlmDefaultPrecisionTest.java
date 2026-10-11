package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import tools.jackson.databind.JsonNode;

/** Default precision evidence through Java preprocessing, prefill and three cached steps. */
@EnabledIfNativeAvailable
class SmolVlmDefaultPrecisionTest {
  // Separate from the strict 1e-4 float32 reference; measured errors are recorded in the WP6
  // report.
  private static final float TF32_LOGIT_BOUND = 0.002f;

  @Test
  void javaPreprocessingAndCachedLogitsStayWithinMeasuredTf32Bound() throws Exception {
    float maximum = 0;
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      for (JsonNode c : SmolVlmGoldenTest.golden().required("cases")) {
        GenerationRequest request = SmolVlmGoldenTest.request(c, 5);
        SmolVlmPromptPlan plan = model.plan(request.promptTokenIds(), request.images());
        try (MLXScope generation = root.newChild()) {
          List<KVCache> caches = List.of(new KVCache(generation), new KVCache(generation));
          MLXArray logits;
          try (MLXScope prefill = generation.newChild()) {
            MLXArray features = model.features(request, prefill);
            int[] expanded = plan.expandedIds();
            logits =
                model
                    .decoderForTest()
                    .stepLogits(
                        MLX.array(prefill, expanded, new int[] {1, expanded.length}),
                        caches,
                        new int[] {expanded.length},
                        features == null
                            ? null
                            : embedded -> SmolVlmModel.merge(embedded, features, plan));
            List<MLXArray> arrays = new ArrayList<>();
            arrays.add(logits);
            arrays.addAll(java.util.Arrays.asList(DecoderModel.stepCacheArrays(caches)));
            MLX.eval(arrays.toArray(MLXArray[]::new));
            maximum = Math.max(maximum, error(logits, c.required("prefill_logits")));
            logits = MLX.hoist(logits, generation);
          }
          // Retained KV works after the entire visual/prefill child scope has closed.
          for (int step = 1; step < c.required("decode_steps").size(); step++) {
            int token = c.required("decode_steps").get(step - 1).required("token_id").asInt();
            try (MLXScope activation = generation.newChild()) {
              logits =
                  model
                      .decoderForTest()
                      .stepLogits(
                          MLX.array(activation, new int[] {token}, new int[] {1, 1}),
                          caches,
                          new int[] {1},
                          null);
              maximum =
                  Math.max(
                      maximum,
                      error(logits, c.required("decode_steps").get(step).required("logits")));
            }
          }
        }
      }
    }
    System.out.println("SmolVLM Java-preprocessor TF32 maximum logit error: " + maximum);
    assertTrue(maximum <= TF32_LOGIT_BOUND, "TF32 logit error " + maximum);
  }

  private static float error(MLXArray array, JsonNode reference) {
    float[] actual = array.toFloatArray();
    float[] expected = SmolVlmGoldenTest.floats(reference);
    org.junit.jupiter.api.Assertions.assertEquals(expected.length, actual.length);
    float maximum = 0;
    for (int i = 0; i < actual.length; i++) {
      assertTrue(Float.isFinite(actual[i]));
      maximum = Math.max(maximum, Math.abs(actual[i] - expected[i]));
    }
    return maximum;
  }
}
