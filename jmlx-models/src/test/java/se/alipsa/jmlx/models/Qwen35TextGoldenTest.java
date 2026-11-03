package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
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

/** Independent HF hybrid-stack logits and complete retained state at fixed FLOAT32 precision. */
@EnabledIfNativeAvailable
@Tag("full-float32")
class Qwen35TextGoldenTest {
  private static final float EPSILON = 1e-4f;
  private static final Path ROOT = Path.of("..", "tools", "qwen35-reference");
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void floatLayoutsTiedEqualHeadsAndEightLayerStacksMatchHfPrefillCachedAndFresh()
      throws Exception {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> golden = goldens(weights);
      for (String kind :
          new String[] {"hf", "vlm-float", "lm-float", "hf-tied-equal", "hf-eight"}) {
        Path directory = ROOT.resolve("fixtures").resolve(kind);
        Qwen35Weights.Loaded loaded =
            Qwen35Weights.load(
                weights, directory, MAPPER.readTree(directory.resolve("config.json").toFile()));
        Qwen35Decoder decoder =
            new Qwen35Decoder(weights, loaded.checkpoint().descriptor(), loaded.tensors());
        for (int length : new int[] {4, 65, 129}) {
          String prefix = (kind.endsWith("float") ? "hf" : kind) + ".t" + length;
          try (MLXScope request = weights.newChild();
              Qwen35Decoder.Session session = decoder.newSession(request, KVCachePolicy.full())) {
            for (int step = -1; step < 2; step++) {
              String p = prefix + (step < 0 ? "" : ".decode" + step);
              try (MLXScope activation = request.newChild()) {
                MLXArray ids = inScope(golden.get(p + ".ids"), activation);
                MLXArray logits = session.build(ids, null, false);
                assertEquals(step < 0 ? 0 : length + step, session.position());
                session.evaluateAndCommit(logits, StepBoundaryEvaluator.NATIVE);
                assertNear(golden.get(p + ".logits"), logits, kind + p);
                if (step >= 0) {
                  assertNear(golden.get(p + ".fresh_logits"), logits, kind + " fresh " + p);
                }
              }
              assertEquals(length + Math.max(0, step + 1), session.position());
              assertState(golden, p, session, loaded.checkpoint().descriptor().text().layers());
            }
          }
        }
      }
    }
  }

  @Test
  void splitPrefillRetainsWholeHybridStateAcrossConvolutionAndChunkBoundaries() throws Exception {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> golden = goldens(weights);
      Path directory = ROOT.resolve("fixtures/hf-eight");
      Qwen35Weights.Loaded loaded =
          Qwen35Weights.load(
              weights, directory, MAPPER.readTree(directory.resolve("config.json").toFile()));
      Qwen35Decoder decoder =
          new Qwen35Decoder(weights, loaded.checkpoint().descriptor(), loaded.tensors());
      for (int split : new int[] {1, 3, 4, 63, 64, 65, 128}) {
        try (MLXScope request = weights.newChild();
            Qwen35Decoder.Session session = decoder.newSession(request, KVCachePolicy.full())) {
          for (int[] range : new int[][] {{0, split}, {split, 129}}) {
            try (MLXScope activation = request.newChild()) {
              MLXArray ids =
                  MLXShape.slice(
                      inScope(golden.get("hf-eight.t129.ids"), activation),
                      new int[] {0, range[0]},
                      new int[] {1, range[1]});
              MLXArray logits = session.build(ids, null, false);
              session.evaluateAndCommit(logits, StepBoundaryEvaluator.NATIVE);
              assertNear(
                  MLXShape.slice(
                      golden.get("hf-eight.t129.logits"),
                      new int[] {0, range[0], 0},
                      new int[] {1, range[1], 128}),
                  logits,
                  "split " + split);
            }
          }
          assertState(golden, "hf-eight.t129", session, 8);
          assertEquals(129, session.position());
        }
      }
    }
  }

  @Test
  void lateEvaluationFailurePoisonsEveryCacheWithoutCommittingGlobalPosition() throws Exception {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> golden = goldens(weights);
      Path directory = ROOT.resolve("fixtures/hf");
      Qwen35Weights.Loaded loaded =
          Qwen35Weights.load(
              weights, directory, MAPPER.readTree(directory.resolve("config.json").toFile()));
      Qwen35Decoder decoder =
          new Qwen35Decoder(weights, loaded.checkpoint().descriptor(), loaded.tensors());
      try (MLXScope request = weights.newChild();
          Qwen35Decoder.Session session = decoder.newSession(request, KVCachePolicy.full(4));
          MLXScope activation = request.newChild()) {
        MLXArray ids = inScope(golden.get("hf.t4.ids"), activation);
        final MLXArray logits = session.build(ids, null, true);
        assertEquals(0, session.position());
        assertThrows(IllegalStateException.class, () -> session.build(ids, null, true));
        assertFalse(session.isPoisoned());
        assertThrows(
            IllegalStateException.class,
            () ->
                session.evaluateAndCommit(
                    logits,
                    roots -> {
                      throw new IllegalStateException("injected evaluation failure");
                    }));
        assertTrue(session.isPoisoned());
        assertEquals(0, session.position());
        for (int i = 0; i < 4; i++) {
          assertTrue(
              session.recurrent(i) == null
                  ? session.full(i).isPoisoned()
                  : session.recurrent(i).isPoisoned());
        }
        assertThrows(IllegalStateException.class, () -> session.build(ids, null, true));
      }
    }
  }

  private static Map<String, MLXArray> goldens(MLXScope scope) {
    return MLXIO
        .loadSafetensors(scope, ROOT.resolve("goldens/model-text.safetensors").toString())
        .tensors();
  }

  private static MLXArray inScope(MLXArray array, MLXScope scope) {
    return MLXShape.reshape(MLXShape.expandDims(array, scope, 0), array.shape());
  }

  private static void assertState(
      Map<String, MLXArray> golden, String prefix, Qwen35Decoder.Session session, int layers) {
    for (int i = 0; i < layers; i++) {
      String p = prefix + ".layer" + i;
      if (session.recurrent(i) != null) {
        assertNear(golden.get(p + ".state"), session.recurrent(i).state(), p);
        assertNear(golden.get(p + ".convolution"), session.recurrent(i).convolution(), p);
      } else {
        assertNear(golden.get(p + ".keys"), session.full(i).keys(), p);
        assertNear(golden.get(p + ".values"), session.full(i).values(), p);
      }
    }
  }

  private static void assertNear(MLXArray expected, MLXArray actual, String label) {
    assertArrayEquals(expected.shape(), actual.shape(), label);
    assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), EPSILON, label);
  }
}
