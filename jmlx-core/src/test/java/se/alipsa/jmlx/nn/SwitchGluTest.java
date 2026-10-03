package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.NativeMemoryProbe;
import se.alipsa.jmlx.memory.MLXScope;

/** Gathered expert evaluation checked against one dense {@link GatedMlp} per expert. */
@EnabledIfNativeAvailable
class SwitchGluTest {

  private static final float EPS = 1e-4f;
  // Package-private: MoeMlpTest reuses these fixtures.
  static final int E = 4;
  static final int H = 6;
  static final int F = 5;

  /** Deterministic, non-symmetric fill so a transposed weight cannot pass by accident. */
  static float[] pattern(int size, int salt) {
    float[] out = new float[size];
    for (int i = 0; i < size; i++) {
      out[i] = (float) (0.5 * Math.sin(0.37 * i + 1.3 * salt));
    }
    return out;
  }

  /** Per-expert weights: {@code [expert][0=gate,1=up,2=down]}. */
  static MLXArray[][] expertWeights(MLXScope scope) {
    MLXArray[][] w = new MLXArray[E][3];
    for (int e = 0; e < E; e++) {
      w[e][0] = MLX.array(scope, pattern(F * H, 10 * e + 1), new int[] {F, H});
      w[e][1] = MLX.array(scope, pattern(F * H, 10 * e + 2), new int[] {F, H});
      w[e][2] = MLX.array(scope, pattern(H * F, 10 * e + 3), new int[] {H, F});
    }
    return w;
  }

  static SwitchGlu switchGlu(MLXScope scope, MLXArray[][] w, Activation activation) {
    MLXArray[][] byKind = new MLXArray[3][E];
    for (int e = 0; e < E; e++) {
      for (int kind = 0; kind < 3; kind++) {
        byKind[kind][e] = w[e][kind];
      }
    }
    return new SwitchGlu(
        scope,
        MLXShape.stack(byKind[0], 0),
        MLXShape.stack(byKind[1], 0),
        MLXShape.stack(byKind[2], 0),
        activation);
  }

  static List<GatedMlp> denseExperts(MLXScope scope, MLXArray[][] w, Activation activation) {
    List<GatedMlp> out = new ArrayList<>();
    for (MLXArray[] e : w) {
      out.add(
          new GatedMlp(
              scope,
              new Linear(scope, e[0], null),
              new Linear(scope, e[1], null),
              new Linear(scope, e[2], null),
              activation));
    }
    return out;
  }

  /** Row-major {@code [B, T, K]} indices where slot k of token t picks {@code (t + 3k) % E}. */
  static int[] indices(int b, int t, int k) {
    int[] out = new int[b * t * k];
    for (int i = 0; i < b * t; i++) {
      for (int j = 0; j < k; j++) {
        out[i * k + j] = (i + 3 * j) % E;
      }
    }
    return out;
  }

  /** Expected {@code [B, T, K, H]} assembled from each dense expert's full-batch output. */
  static float[] expected(List<GatedMlp> dense, MLXArray x, int[] idx, int k) {
    float[][] perExpert = new float[E][];
    for (int e = 0; e < E; e++) {
      perExpert[e] = dense.get(e).forward(x).toFloatArray();
    }
    int tokens = idx.length / k;
    float[] out = new float[tokens * k * H];
    for (int token = 0; token < tokens; token++) {
      for (int slot = 0; slot < k; slot++) {
        int e = idx[token * k + slot];
        System.arraycopy(perExpert[e], token * H, out, (token * k + slot) * H, H);
      }
    }
    return out;
  }

  private static void assertMatchesDense(int b, int t, int k, boolean sort, Activation act) {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray[][] w = expertWeights(model);
      SwitchGlu glu = switchGlu(model, w, act);
      MLXArray x = MLX.array(step, pattern(b * t * H, 99), new int[] {b, t, H});
      int[] idx = indices(b, t, k);
      MLXArray actual = glu.forward(x, MLX.array(step, idx, new int[] {b, t, k}), sort);
      assertArrayEquals(new int[] {b, t, k, H}, actual.shape());
      assertArrayEquals(
          expected(denseExperts(model, w, act), x, idx, k), actual.toFloatArray(), EPS);
    }
  }

  @Test
  @Tag("full-float32")
  void unsortedPathMatchesDenseExperts() {
    assertMatchesDense(2, 3, 2, false, Activation.SILU);
  }

  @Test
  void sortedPathMatchesDenseExperts() {
    assertMatchesDense(2, 3, 2, true, Activation.SILU);
  }

  @Test
  @Tag("full-float32")
  void topKEqualToExpertCountMatchesDenseExperts() {
    assertMatchesDense(1, 5, E, false, Activation.SILU);
    assertMatchesDense(1, 5, E, true, Activation.SILU);
  }

  @Test
  @Tag("full-float32")
  void geluTanhActivationMatchesDenseExperts() {
    assertMatchesDense(1, 4, 2, false, Activation.GELU_TANH);
  }

  @Test
  @Tag("full-float32")
  void sortedAndUnsortedPathsAgreeAboveThreshold() {
    // 1 * 40 * 2 = 80 >= SORT_THRESHOLD: the public overload takes the sorted path.
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      SwitchGlu glu = switchGlu(model, expertWeights(model), Activation.SILU);
      MLXArray x = MLX.array(step, pattern(40 * H, 7), new int[] {1, 40, H});
      MLXArray idx = MLX.array(step, indices(1, 40, 2), new int[] {1, 40, 2});
      assertEquals(true, idx.size() >= SwitchGlu.SORT_THRESHOLD);
      assertArrayEquals(
          glu.forward(x, idx, false).toFloatArray(), glu.forward(x, idx).toFloatArray(), EPS);
    }
  }

  @Test
  void bf16StaysBf16() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[][] w = expertWeights(scope);
      for (MLXArray[] e : w) {
        for (int kind = 0; kind < 3; kind++) {
          e[kind] = MLX.astype(e[kind], DType.BFLOAT16);
        }
      }
      SwitchGlu glu = switchGlu(scope, w, Activation.SILU);
      MLXArray x =
          MLX.astype(MLX.array(scope, pattern(2 * H, 5), new int[] {1, 2, H}), DType.BFLOAT16);
      MLXArray idx = MLX.array(scope, indices(1, 2, 2), new int[] {1, 2, 2});
      assertEquals(DType.BFLOAT16, glu.forward(x, idx, false).dtype());
      assertEquals(DType.BFLOAT16, glu.forward(x, idx, true).dtype());
    }
  }

  @Test
  void registersStackedWeightsAsParameters() {
    try (MLXScope scope = new MLXScope()) {
      SwitchGlu glu = switchGlu(scope, expertWeights(scope), Activation.SILU);
      assertEquals(
          List.of("gateWeight", "upWeight", "downWeight"), List.copyOf(glu.parameters().keySet()));
      assertEquals(E, glu.experts());
    }
  }

  @Test
  void rejectsInconsistentWeightShapes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray gate = MLX.zeros(scope, new int[] {E, F, H}, DType.FLOAT32);
      MLXArray down = MLX.zeros(scope, new int[] {E, H, F}, DType.FLOAT32);
      MLXArray wrongUp = MLX.zeros(scope, new int[] {E, F + 1, H}, DType.FLOAT32);
      MLXArray wrongDown = MLX.zeros(scope, new int[] {E, F, H}, DType.FLOAT32);
      MLXArray rank2 = MLX.zeros(scope, new int[] {F, H}, DType.FLOAT32);
      assertThrows(
          IllegalArgumentException.class,
          () -> new SwitchGlu(scope, gate, wrongUp, down, Activation.SILU));
      assertThrows(
          IllegalArgumentException.class,
          () -> new SwitchGlu(scope, gate, gate, wrongDown, Activation.SILU));
      assertThrows(
          IllegalArgumentException.class,
          () -> new SwitchGlu(scope, rank2, rank2, rank2, Activation.SILU));
    }
  }

  @Test
  void rejectsMalformedInputs() {
    try (MLXScope scope = new MLXScope()) {
      SwitchGlu glu = switchGlu(scope, expertWeights(scope), Activation.SILU);
      MLXArray x = MLX.zeros(scope, new int[] {1, 2, H}, DType.FLOAT32);
      MLXArray wrongHidden = MLX.zeros(scope, new int[] {1, 2, H + 1}, DType.FLOAT32);
      MLXArray idx = MLX.array(scope, new int[] {0, 1}, new int[] {1, 2, 1});
      MLXArray floatIdx = MLX.zeros(scope, new int[] {1, 2, 1}, DType.FLOAT32);
      MLXArray wrongTokens = MLX.array(scope, new int[] {0, 1, 2}, new int[] {1, 3, 1});
      assertThrows(IllegalArgumentException.class, () -> glu.forward(wrongHidden, idx));
      assertThrows(IllegalArgumentException.class, () -> glu.forward(x, floatIdx));
      assertThrows(IllegalArgumentException.class, () -> glu.forward(x, wrongTokens));
    }
  }

  /**
   * Per-step leak guard, following {@code LinearTest}'s structure: each iteration runs in its own
   * child scope under a long-lived model scope. Limitation (verified on MLX 0.31.2): a retained
   * {@code swapaxes} view shares its source buffer, so active memory does not detect a view leaked
   * into the model scope. This only guards against data-owning intermediates (gather outputs,
   * stacked copies) accumulating; the {@code x.scope()} rule for the views stays a review item.
   */
  @Test
  void activeMemoryDoesNotGrowAcrossPerStepScopes() {
    try (MLXScope model = new MLXScope()) {
      SwitchGlu glu = switchGlu(model, expertWeights(model), Activation.SILU);
      Runnable step =
          () -> {
            try (MLXScope s = model.newChild()) {
              MLXArray x = MLX.array(s, pattern(40 * H, 2), new int[] {1, 40, H});
              MLXArray idx = MLX.array(s, indices(1, 40, 2), new int[] {1, 40, 2});
              MLX.eval(glu.forward(x, idx));
            }
          };
      for (int i = 0; i < 50; i++) {
        step.run();
      }
      long baseline = NativeMemoryProbe.activeMemoryBytes();
      for (int i = 0; i < 200; i++) {
        step.run();
      }
      long growth = NativeMemoryProbe.activeMemoryBytes() - baseline;
      assertTrue(growth <= 2_000_000, "grew " + growth + " B");
    }
  }

  @Test
  void rejectsIndicesLivingInAnAncestorOfTheInputScope() {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      SwitchGlu glu = switchGlu(model, expertWeights(model), Activation.SILU);
      MLXArray x = MLX.array(step, pattern(2 * H, 5), new int[] {1, 2, H});
      MLXArray modelIdx = MLX.array(model, indices(1, 2, 2), new int[] {1, 2, 2});
      assertThrows(IllegalArgumentException.class, () -> glu.forward(x, modelIdx));
      assertThrows(IllegalArgumentException.class, () -> glu.forward(x, modelIdx, true));
      // The same indices in the step scope are fine.
      glu.forward(x, MLX.array(step, indices(1, 2, 2), new int[] {1, 2, 2}));
    }
  }

  @Test
  void rejectsWeightsOfDifferentDtypes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray gate =
          MLX.astype(MLX.zeros(scope, new int[] {E, F, H}, DType.FLOAT32), DType.BFLOAT16);
      MLXArray up = MLX.zeros(scope, new int[] {E, F, H}, DType.FLOAT32);
      MLXArray down =
          MLX.astype(MLX.zeros(scope, new int[] {E, H, F}, DType.FLOAT32), DType.BFLOAT16);
      assertThrows(
          IllegalArgumentException.class,
          () -> new SwitchGlu(scope, gate, up, down, Activation.SILU));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new SwitchGlu(
                  scope,
                  gate,
                  gate,
                  MLX.zeros(scope, new int[] {E, H, F}, DType.FLOAT32),
                  Activation.SILU));
    }
  }
}
