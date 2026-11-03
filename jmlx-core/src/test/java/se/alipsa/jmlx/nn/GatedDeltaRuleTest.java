package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Independent HF delta-rule outputs/state, including grouped heads and nonzero initial state. */
@EnabledIfNativeAvailable
@Tag("full-float32")
class GatedDeltaRuleTest {
  private static final float EPSILON = 1e-4f;

  @Test
  void bothPrototypesMatchHfAcrossChunkAndConvolutionBoundaries() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> fixture = load(scope);
      for (int length : new int[] {1, 3, 4, 5, 63, 64, 65, 127, 128, 129}) {
        for (int heads : new int[] {2, 6}) {
          String stem = "t" + length + "h" + heads + ".";
          try (MLXScope request = scope.newChild()) {
            MLXArray q = inScope(fixture.get(stem + "q"), request);
            MLXArray k = fixture.get(stem + "k");
            MLXArray v = fixture.get(stem + "v");
            MLXArray g = fixture.get(stem + "g");
            MLXArray beta = fixture.get(stem + "beta");
            MLXArray initial = fixture.get(stem + "initial");
            float[] original = initial.toFloatArray();
            GatedDeltaRule.Result recurrent = GatedDeltaRule.recurrent(q, k, v, g, beta, initial);
            GatedDeltaRule.Result chunked = GatedDeltaRule.chunked(q, k, v, g, beta, initial);
            assertResult(fixture, stem, recurrent);
            assertResult(fixture, stem, chunked);
            assertArrayEquals(original, initial.toFloatArray(), 0f, "initial state mutated");
          }
        }
      }
    }
  }

  @Test
  void splitPrefillsCarryTheCompleteState() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> fixture = load(scope);
      String stem = "t129h6.";
      for (int split : new int[] {1, 3, 63, 64, 65, 128}) {
        try (MLXScope request = scope.newChild()) {
          MLXArray q = inScope(fixture.get(stem + "q"), request);
          MLXArray k = inScope(fixture.get(stem + "k"), request);
          MLXArray v = inScope(fixture.get(stem + "v"), request);
          MLXArray g = inScope(fixture.get(stem + "g"), request);
          MLXArray beta = inScope(fixture.get(stem + "beta"), request);
          GatedDeltaRule.Result first =
              GatedDeltaRule.chunked(
                  positions(q, 0, split),
                  positions(k, 0, split),
                  positions(v, 0, split),
                  positions(g, 0, split),
                  positions(beta, 0, split),
                  fixture.get(stem + "initial"));
          GatedDeltaRule.Result second =
              GatedDeltaRule.chunked(
                  positions(q, split, 129),
                  positions(k, split, 129),
                  positions(v, split, 129),
                  positions(g, split, 129),
                  positions(beta, split, 129),
                  first.state());
          MLXArray joined =
              MLXShape.concatenate(new MLXArray[] {first.output(), second.output()}, 1);
          assertArrayEquals(
              fixture.get(stem + "output").toFloatArray(), joined.toFloatArray(), EPSILON);
          assertArrayEquals(
              fixture.get(stem + "state").toFloatArray(), second.state().toFloatArray(), EPSILON);
        }
      }
    }
  }

  @Test
  void invalidStateOrientationAndHeadGroupingFailBeforeExecution() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> fixture = load(scope);
      String stem = "t3h6.";
      MLXArray q = fixture.get(stem + "q");
      MLXArray k = fixture.get(stem + "k");
      MLXArray v = fixture.get(stem + "v");
      MLXArray g = fixture.get(stem + "g");
      MLXArray beta = fixture.get(stem + "beta");
      MLXArray initial = fixture.get(stem + "initial");
      MLXArray transposed = MLXShape.transpose(initial, new int[] {0, 1, 3, 2});
      assertThrows(
          IllegalArgumentException.class,
          () -> GatedDeltaRule.chunked(q, k, v, g, beta, transposed));
      MLXArray badHeads = MLXShape.slice(v, new int[] {0, 0, 0, 0}, new int[] {1, 3, 3, 64});
      assertThrows(
          IllegalArgumentException.class,
          () -> GatedDeltaRule.recurrent(q, k, badHeads, g, beta, initial));
    }
  }

  @Test
  void correlatedKeysDecayExtremesAndPreprocessingBoundariesMatchHf() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> fixture =
          MLXIO
              .loadSafetensors(
                  scope,
                  Path.of("..", "tools", "qwen35-reference", "goldens", "delta-stress.safetensors")
                      .toString())
              .tensors();
      for (String name :
          new String[] {
            "correlated", "near_correlated", "strong_decay", "batch_boundary", "long_boundary"
          }) {
        String stem = name + ".";
        try (MLXScope request = scope.newChild()) {
          MLXArray q = inScope(fixture.get(stem + "q"), request);
          GatedDeltaRule.Result result =
              GatedDeltaRule.chunked(
                  q,
                  fixture.get(stem + "k"),
                  fixture.get(stem + "v"),
                  fixture.get(stem + "g"),
                  fixture.get(stem + "beta"),
                  fixture.get(stem + "initial"));
          assertResult(fixture, stem, result);
        }
      }
    }
  }

  private static Map<String, MLXArray> load(MLXScope scope) {
    return MLXIO
        .loadSafetensors(
            scope,
            Path.of("..", "tools", "qwen35-reference", "goldens", "delta.safetensors").toString())
        .tensors();
  }

  private static void assertResult(
      Map<String, MLXArray> fixture, String stem, GatedDeltaRule.Result result) {
    assertArrayEquals(fixture.get(stem + "output").shape(), result.output().shape());
    assertArrayEquals(fixture.get(stem + "state").shape(), result.state().shape());
    assertArrayEquals(
        fixture.get(stem + "output").toFloatArray(),
        result.output().toFloatArray(),
        EPSILON,
        stem + "output");
    assertArrayEquals(
        fixture.get(stem + "state").toFloatArray(),
        result.state().toFloatArray(),
        EPSILON,
        stem + "state");
  }

  private static MLXArray inScope(MLXArray a, MLXScope scope) {
    return MLXShape.reshape(MLXShape.expandDims(a, scope, 0), a.shape());
  }

  private static MLXArray positions(MLXArray a, int start, int end) {
    int[] starts = new int[a.ndim()];
    int[] stops = a.shape();
    starts[1] = start;
    stops[1] = end;
    return MLXShape.slice(a, starts, stops);
  }
}
