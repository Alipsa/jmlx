package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Gathered MoE routing checked against the dense Phase 6.3 oracle. */
@EnabledIfNativeAvailable
class MoeMlpTest {

  private static final float EPS = 1e-4f;
  private static final int E = SwitchGluTest.E;
  private static final int H = SwitchGluTest.H;

  /** Zero-weight router whose logits are exactly {@code biases} for every token. */
  private static Linear biasRouter(MLXScope scope, float... biases) {
    return new Linear(
        scope,
        MLX.zeros(scope, new int[] {biases.length, H}, DType.FLOAT32),
        MLX.array(scope, biases, new int[] {biases.length}));
  }

  /** Input-dependent router, so different tokens pick different experts. */
  private static Linear patternRouter(MLXScope scope) {
    return new Linear(
        scope, MLX.array(scope, SwitchGluTest.pattern(E * H, 77), new int[] {E, H}), null);
  }

  private static void assertMatchesOracle(int tokens, int topK) {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray[][] w = SwitchGluTest.expertWeights(model);
      MoeMlp moe =
          new MoeMlp(
              model,
              patternRouter(model),
              SwitchGluTest.switchGlu(model, w, Activation.SILU),
              topK);
      DenseMoeReference oracle =
          new DenseMoeReference(
              model,
              patternRouter(model),
              SwitchGluTest.denseExperts(model, w, Activation.SILU),
              topK);
      MLXArray x = MLX.array(step, SwitchGluTest.pattern(tokens * H, 3), new int[] {1, tokens, H});
      assertArrayEquals(oracle.forward(x).toFloatArray(), moe.forward(x).toFloatArray(), EPS);
    }
  }

  @Test
  void rejectsInvalidTopK() {
    try (MLXScope scope = new MLXScope()) {
      SwitchGlu experts =
          SwitchGluTest.switchGlu(scope, SwitchGluTest.expertWeights(scope), Activation.SILU);
      assertThrows(
          IllegalArgumentException.class,
          () -> new MoeMlp(scope, biasRouter(scope, 0, 0, 0, 0), experts, 0));
      assertThrows(
          IllegalArgumentException.class,
          () -> new MoeMlp(scope, biasRouter(scope, 0, 0, 0, 0), experts, E + 1));
    }
  }

  @Test
  void matchesDenseOracleOnBothSidesOfSortThreshold() {
    assertMatchesOracle(3, 2); // 6 slots: unsorted path
    assertMatchesOracle(40, 2); // 80 slots >= SwitchGlu.SORT_THRESHOLD: sorted path
  }

  @Test
  void selectingEveryExpertMatchesDenseOracle() {
    assertMatchesOracle(5, E);
    assertMatchesOracle(40, E);
  }

  @Test
  void exactTiesChooseFirstExpertAcrossTokensAndRuns() {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray[][] w = SwitchGluTest.expertWeights(model);
      MoeMlp moe =
          new MoeMlp(
              model,
              biasRouter(model, 0, 0, 0, 0),
              SwitchGluTest.switchGlu(model, w, Activation.SILU),
              1);
      GatedMlp first = SwitchGluTest.denseExperts(model, w, Activation.SILU).get(0);
      MLXArray x = MLX.array(step, SwitchGluTest.pattern(2 * 6 * H, 4), new int[] {2, 6, H});
      float[] expected = first.forward(x).toFloatArray();
      for (int run = 0; run < 100; run++) {
        assertArrayEquals(expected, moe.forward(x).toFloatArray(), EPS);
      }
    }
  }

  @Test
  void unselectedInfiniteExpertNeverAffectsOutput() {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray[][] w = SwitchGluTest.expertWeights(model);
      w[E - 1][2] = MLX.full(model, new int[] {H, SwitchGluTest.F}, Float.MAX_VALUE, DType.FLOAT32);
      MoeMlp moe =
          new MoeMlp(
              model,
              biasRouter(model, 2, 1, 0, -1),
              SwitchGluTest.switchGlu(model, w, Activation.SILU),
              2);
      DenseMoeReference oracle =
          new DenseMoeReference(
              model,
              biasRouter(model, 2, 1, 0, -1),
              SwitchGluTest.denseExperts(model, w, Activation.SILU),
              2);
      MLXArray x = MLX.array(step, SwitchGluTest.pattern(3 * H, 8), new int[] {1, 3, H});
      float[] actual = moe.forward(x).toFloatArray();
      for (float v : actual) {
        assertTrue(Float.isFinite(v));
      }
      assertArrayEquals(oracle.forward(x).toFloatArray(), actual, EPS);
    }
  }

  @Test
  void bf16OutputUsesHiddenStateDtype() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[][] w = SwitchGluTest.expertWeights(scope);
      for (MLXArray[] e : w) {
        for (int kind = 0; kind < 3; kind++) {
          e[kind] = MLX.astype(e[kind], DType.BFLOAT16);
        }
      }
      Linear router =
          new Linear(
              scope,
              MLX.astype(MLX.zeros(scope, new int[] {E, H}, DType.FLOAT32), DType.BFLOAT16),
              MLX.astype(
                  MLX.array(scope, new float[] {1, 0, 0, 0}, new int[] {E}), DType.BFLOAT16));
      MoeMlp moe = new MoeMlp(scope, router, SwitchGluTest.switchGlu(scope, w, Activation.SILU), 2);
      MLXArray x =
          MLX.astype(
              MLX.array(scope, SwitchGluTest.pattern(H, 1), new int[] {1, 1, H}), DType.BFLOAT16);
      assertEquals(DType.BFLOAT16, moe.forward(x).dtype());
    }
  }

  @Test
  void registersRouterAndStackedExpertParameters() {
    try (MLXScope scope = new MLXScope()) {
      MoeMlp moe =
          new MoeMlp(
              scope,
              biasRouter(scope, 0, 0, 0, 0),
              SwitchGluTest.switchGlu(scope, SwitchGluTest.expertWeights(scope), Activation.SILU),
              2);
      assertEquals(
          List.of(
              "router.weight",
              "router.bias",
              "experts.gateWeight",
              "experts.upWeight",
              "experts.downWeight"),
          List.copyOf(moe.parameters().keySet()));
    }
  }

  /**
   * Review Focus 1: {@code ModuleGrad} through the gathered route must not hit GatherMM's "Cannot
   * calculate VJP with respect to indices" error, must give exactly zero gradient to never-selected
   * experts, and must equal the dense oracle's per-expert gradients -- on the unsorted path (3
   * tokens x top-2 = 6 slots) AND the sorted path (40 x 2 = 80 slots >= SORT_THRESHOLD), since
   * every realistic training step takes the sorted one.
   */
  @Test
  void gradientsFlowOnlyToSelectedExperts() {
    assertGradientsMatchOracle(3);
    assertGradientsMatchOracle(40);
  }

  private static void assertGradientsMatchOracle(int tokens) {
    try (MLXScope model = new MLXScope()) {
      MLXArray[][] w = SwitchGluTest.expertWeights(model);
      // Biases 2, 1, 0, -1 with top-2: experts 0 and 1 always win; 2 and 3 are never selected.
      MoeMlp moe =
          new MoeMlp(
              model,
              biasRouter(model, 2, 1, 0, -1),
              SwitchGluTest.switchGlu(model, w, Activation.SILU),
              2);
      DenseMoeReference oracle =
          new DenseMoeReference(
              model,
              biasRouter(model, 2, 1, 0, -1),
              SwitchGluTest.denseExperts(model, w, Activation.SILU),
              2);
      try (ModuleGrad gathered =
              ModuleGrad.of(moe, (p, in) -> new MLXArray[] {MLXOps.sum(moe.forward(in[0]))});
          ModuleGrad dense =
              ModuleGrad.of(oracle, (p, in) -> new MLXArray[] {MLXOps.sum(oracle.forward(in[0]))});
          MLXScope step = model.newChild()) {
        MLXArray x =
            MLX.array(step, SwitchGluTest.pattern(tokens * H, 6), new int[] {1, tokens, H});
        var gatheredGrads = gathered.apply(step, new MLXArray[] {x}).grads();
        var denseGrads = dense.apply(step, new MLXArray[] {x}).grads();
        assertEquals(
            List.of(
                "router.weight",
                "router.bias",
                "experts.gateWeight",
                "experts.upWeight",
                "experts.downWeight"),
            List.copyOf(gatheredGrads.keySet()));
        for (String router : List.of("router.weight", "router.bias")) {
          assertArrayEquals(
              denseGrads.get(router).toFloatArray(),
              gatheredGrads.get(router).toFloatArray(),
              EPS,
              router);
        }
        for (String kind : List.of("gate", "up", "down")) {
          MLXArray stacked = gatheredGrads.get("experts." + kind + "Weight");
          int[] shape = stacked.shape();
          for (int e = 0; e < E; e++) {
            float[] slice =
                MLXShape.slice(stacked, new int[] {e, 0, 0}, new int[] {e + 1, shape[1], shape[2]})
                    .toFloatArray();
            assertArrayEquals(
                denseGrads.get("expert" + e + "." + kind + "Proj.weight").toFloatArray(),
                slice,
                EPS,
                kind + " expert " + e + " tokens=" + tokens);
            if (e >= 2) {
              assertArrayEquals(new float[slice.length], slice, 0f, kind + " expert " + e);
            }
          }
        }
      }
    }
  }

  // ---- Absolute anchors: exact closed forms, independent of DenseMoeReference -------------

  /** Router picks logits from the first 3 input coordinates: {@code logits = x[..., :3]}. */
  private static Linear selectorRouter(MLXScope scope, DType dtype) {
    float[] w = new float[3 * 4];
    for (int e = 0; e < 3; e++) {
      w[e * 4 + e] = 1;
    }
    return new Linear(scope, MLX.astype(MLX.array(scope, w, new int[] {3, 4}), dtype), null);
  }

  /** 3 experts, H=F=4: gate = up = identity, down = (e + 1) * identity. */
  private static SwitchGlu diagonalExperts(MLXScope scope, DType dtype) {
    MLXArray[] gate = new MLXArray[3];
    MLXArray[] down = new MLXArray[3];
    for (int e = 0; e < 3; e++) {
      float[] identity = new float[16];
      float[] scaled = new float[16];
      for (int i = 0; i < 4; i++) {
        identity[i * 4 + i] = 1;
        scaled[i * 4 + i] = e + 1;
      }
      gate[e] = MLX.astype(MLX.array(scope, identity, new int[] {4, 4}), dtype);
      down[e] = MLX.astype(MLX.array(scope, scaled, new int[] {4, 4}), dtype);
    }
    MLXArray g = MLXShape.stack(gate, 0);
    return new SwitchGlu(scope, g, g, MLXShape.stack(down, 0), Activation.SILU);
  }

  /**
   * Exact expected output: per token, softmax over {@code x[:3]}, successive-argmax top-k (lowest
   * index on ties), renormalised weights w_j, and output {@code sum_j w_j * (j + 1) * silu(x) * x}
   * elementwise. Computed in double from {@code input} as given, so half-precision callers pass the
   * dtype-rounded input.
   */
  private static float[] closedForm(float[] input, int topK) {
    float[] out = new float[input.length];
    for (int token = 0; token < input.length / 4; token++) {
      double[] p = new double[3];
      double max = Double.NEGATIVE_INFINITY;
      for (int e = 0; e < 3; e++) {
        max = Math.max(max, input[token * 4 + e]);
      }
      double z = 0;
      for (int e = 0; e < 3; e++) {
        p[e] = Math.exp(input[token * 4 + e] - max);
        z += p[e];
      }
      boolean[] taken = new boolean[3];
      double selected = 0;
      double weighted = 0;
      for (int k = 0; k < topK; k++) {
        int best = -1;
        for (int e = 0; e < 3; e++) {
          if (!taken[e] && (best < 0 || p[e] > p[best])) {
            best = e;
          }
        }
        taken[best] = true;
        selected += p[best];
        weighted += (best + 1) * p[best];
      }
      double scale = weighted / selected;
      for (int h = 0; h < 4; h++) {
        double v = input[token * 4 + h];
        out[token * 4 + h] = (float) (scale * v * v / (1 + Math.exp(-v)));
      }
    }
    return out;
  }

  /** Inputs in [-2, 2] so routing varies per token. */
  private static float[] anchorInput(int tokens) {
    float[] x = new float[tokens * 4];
    for (int i = 0; i < x.length; i++) {
      x[i] = (float) (2 * Math.sin(0.37 * i + 3.9));
    }
    return x;
  }

  /** Replaces the dropped Phase 6.3 closed-form tests; float32, both paths, top-k 2 and 3. */
  @Test
  void float32MatchesClosedFormOnBothPaths() {
    for (int tokens : new int[] {2, 40}) {
      for (int topK : new int[] {2, 3}) {
        try (MLXScope scope = new MLXScope()) {
          MoeMlp moe =
              new MoeMlp(
                  scope,
                  selectorRouter(scope, DType.FLOAT32),
                  diagonalExperts(scope, DType.FLOAT32),
                  topK);
          float[] input = anchorInput(tokens);
          float[] actual =
              moe.forward(MLX.array(scope, input, new int[] {1, tokens, 4})).toFloatArray();
          assertArrayEquals(
              closedForm(input, topK), actual, 1e-5f, "tokens=" + tokens + " topK=" + topK);
        }
      }
    }
  }

  /**
   * Review Focus 3: bf16/f16 values, not just dtype. Bounds are Verified facts 8's (probe worst
   * case bf16 0.0076, f16 0.0011 on |a - r| / (|r| + 1)). The reference is evaluated on the
   * dtype-rounded input, so input quantisation is not counted as model error.
   */
  @Test
  void halfPrecisionMatchesClosedFormOnBothPaths() {
    for (DType dtype : new DType[] {DType.BFLOAT16, DType.FLOAT16}) {
      double bound = dtype == DType.BFLOAT16 ? 0.02 : 0.005;
      for (int tokens : new int[] {2, 40}) {
        for (int topK : new int[] {2, 3}) {
          try (MLXScope scope = new MLXScope()) {
            MoeMlp moe =
                new MoeMlp(
                    scope, selectorRouter(scope, dtype), diagonalExperts(scope, dtype), topK);
            MLXArray x =
                MLX.astype(MLX.array(scope, anchorInput(tokens), new int[] {1, tokens, 4}), dtype);
            float[] rounded = x.toFloatArray();
            MLXArray y = moe.forward(x);
            assertEquals(dtype, y.dtype());
            float[] actual = y.toFloatArray();
            float[] expected = closedForm(rounded, topK);
            for (int i = 0; i < actual.length; i++) {
              double err = Math.abs(actual[i] - expected[i]) / (Math.abs(expected[i]) + 1);
              assertTrue(
                  err <= bound,
                  dtype
                      + " tokens="
                      + tokens
                      + " topK="
                      + topK
                      + " index "
                      + i
                      + ": "
                      + actual[i]
                      + " vs "
                      + expected[i]);
            }
          }
        }
      }
    }
  }
}
