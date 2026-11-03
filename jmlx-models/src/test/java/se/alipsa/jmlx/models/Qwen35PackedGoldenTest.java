package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCachePolicy;
import tools.jackson.databind.ObjectMapper;

/**
 * Stock MLX packed-weight acceptance with independent HF arithmetic and fixed attribution bounds.
 */
@EnabledIfNativeAvailable
@Tag("full-float32")
class Qwen35PackedGoldenTest {
  private static final Path ROOT = Path.of("..", "tools", "qwen35-reference");
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void actualAffineLayoutsMatchIndependentHfAndStockMlxUnderPredeclaredBounds() throws Exception {
    Path evidence = ROOT.resolve("evidence/packed-text");
    var record = MAPPER.readTree(evidence.resolve("acceptance.json").toFile());
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> golden =
          MLXIO
              .loadSafetensors(weights, evidence.resolve("acceptance.safetensors").toString())
              .tensors();
      for (String family : new String[] {"vlm", "lm"}) {
        final float[] maximumErrors = new float[2];
        Path directory = ROOT.resolve("fixtures/" + family + "-affine");
        Qwen35Weights.Loaded loaded =
            Qwen35Weights.load(
                weights, directory, MAPPER.readTree(directory.resolve("config.json").toFile()));
        Qwen35Decoder decoder =
            new Qwen35Decoder(weights, loaded.checkpoint().descriptor(), loaded.tensors());
        int eligible = 0;
        for (var item : record.get("cases")) {
          if (!item.get("family").asString().equals(family)) {
            continue;
          }
          if (item.get("eligible").asBoolean()) {
            eligible++;
          }
          try (MLXScope request = weights.newChild();
              Qwen35Decoder.Session session = decoder.newSession(request, KVCachePolicy.full())) {
            for (var step : item.get("steps")) {
              String prefix = item.get("name").asString() + "." + step.get("phase").asString();
              try (MLXScope activation = request.newChild()) {
                MLXArray source = golden.get(prefix + ".ids");
                MLXArray ids =
                    MLXShape.reshape(MLXShape.expandDims(source, activation, 0), source.shape());
                MLXArray logits = session.build(ids, null, true);
                session.evaluateAndCommit(logits, StepBoundaryEvaluator.NATIVE);
                float[] actual = logits.toFloatArray();
                float[] hf = golden.get(prefix + ".hf_logits").toFloatArray();
                float[] stock = golden.get(prefix + ".stock_logits").toFloatArray();
                for (int i = 0; i < actual.length; i++) {
                  maximumErrors[0] = Math.max(maximumErrors[0], Math.abs(hf[i] - actual[i]));
                  maximumErrors[1] = Math.max(maximumErrors[1], Math.abs(stock[i] - actual[i]));
                }
                assertArrayEquals(
                    hf,
                    actual,
                    (float) record.get("arithmetic_epsilon").asDouble(),
                    prefix + " independent HF");
                assertArrayEquals(
                    stock,
                    actual,
                    (float) step.get("stock_epsilon").asDouble(),
                    prefix + " stock MLX");
                // Ineligible measurements still check arithmetic, but never count as exact-ID
                // proof.
                if (item.get("eligible").asBoolean()) {
                  int best = 0;
                  for (int i = 1; i < actual.length; i++) {
                    if (actual[i] > actual[best]) {
                      best = i;
                    }
                  }
                  assertEquals(step.get("greedy").asInt(), best, prefix);
                }
              }
            }
          }
        }
        assertTrue(eligible > 0, family + " requires eligible exact-ID cases");
        System.out.println(
            family
                + ": eligible="
                + eligible
                + "/16, maximum Java/HF="
                + maximumErrors[0]
                + ", maximum Java/stock="
                + maximumErrors[1]);
      }
    }
  }
}
