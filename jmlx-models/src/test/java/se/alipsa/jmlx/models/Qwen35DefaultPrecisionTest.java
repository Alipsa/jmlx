package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCachePolicy;
import tools.jackson.databind.ObjectMapper;

/** Default TF32 mode has its own 0.005 HF bound; strict FLOAT32 remains independently at 1e-4. */
@EnabledIfNativeAvailable
class Qwen35DefaultPrecisionTest {
  private static final float DEFAULT_MODE_EPSILON = .005f;
  private static final Path ROOT = Path.of("..", "tools", "qwen35-reference");

  @Test
  void defaultModePrefillAndCachedDecodeUseAnExplicitIndependentBound() throws Exception {
    assertEquals("1", System.getenv("MLX_ENABLE_TF32"));
    float maximum = 0;
    try (MLXScope weights = new MLXScope()) {
      var golden =
          MLXIO
              .loadSafetensors(weights, ROOT.resolve("goldens/model-text.safetensors").toString())
              .tensors();
      for (String fixture : new String[] {"hf", "hf-tied-equal", "hf-eight"}) {
        Path directory = ROOT.resolve("fixtures/" + fixture);
        var loaded =
            Qwen35Weights.load(
                weights,
                directory,
                new ObjectMapper().readTree(directory.resolve("config.json").toFile()));
        var decoder =
            new Qwen35Decoder(weights, loaded.checkpoint().descriptor(), loaded.tensors());
        for (int length : new int[] {4, 65, 129}) {
          String stem = fixture + ".t" + length;
          try (MLXScope request = weights.newChild();
              var session = decoder.newSession(request, KVCachePolicy.full())) {
            for (int step = -1; step < 2; step++) {
              String prefix = stem + (step < 0 ? "" : ".decode" + step);
              try (MLXScope activation = request.newChild()) {
                MLXArray source = golden.get(prefix + ".ids");
                MLXArray ids =
                    MLXShape.reshape(MLXShape.expandDims(source, activation, 0), source.shape());
                MLXArray logits = session.build(ids, null, true);
                session.evaluateAndCommit(logits, StepBoundaryEvaluator.NATIVE);
                float[] actual = logits.toFloatArray();
                float[] all = golden.get(prefix + ".logits").toFloatArray();
                float[] expected = Arrays.copyOfRange(all, all.length - 128, all.length);
                assertArrayEquals(expected, actual, DEFAULT_MODE_EPSILON, prefix);
                for (int i = 0; i < actual.length; i++) {
                  maximum = Math.max(maximum, Math.abs(expected[i] - actual[i]));
                }
              }
            }
          }
        }
      }
    }
    System.out.println(
        "Default TF32 maximum HF logit error="
            + maximum
            + "; declared bound="
            + DEFAULT_MODE_EPSILON);
  }
}
