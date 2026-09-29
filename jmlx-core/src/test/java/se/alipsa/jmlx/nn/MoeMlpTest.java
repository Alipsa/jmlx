package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Routing and dense expert aggregation tests. */
@EnabledIfNativeAvailable
class MoeMlpTest {

  private static final float EPS = 1e-5f;
  private static final float SILU_ONE = (float) (1.0 / (1.0 + Math.exp(-1.0)));

  @Test
  void rejectsInvalidTopK() {
    try (MLXScope scope = new MLXScope()) {
      List<GatedMlp> experts = List.of(constantExpert(scope, 1));
      assertThrows(
          IllegalArgumentException.class, () -> new MoeMlp(scope, router(scope, 0), experts, 0));
      assertThrows(
          IllegalArgumentException.class, () -> new MoeMlp(scope, router(scope, 0), experts, 2));
    }
  }

  @Test
  void selectingEveryExpertEqualsSoftmaxWeightedSum() {
    try (MLXScope scope = new MLXScope()) {
      MoeMlp moe =
          new MoeMlp(
              scope,
              router(scope, 0, 1, 2),
              List.of(constantExpert(scope, 2), constantExpert(scope, 4), constantExpert(scope, 8)),
              3);
      MLXArray input = MLX.array(scope, new float[] {1, 2}, new int[] {1, 2, 1});
      double z = 1 + Math.E + Math.exp(2);
      float expected = (float) ((2 + 4 * Math.E + 8 * Math.exp(2)) / z);
      assertArrayEquals(new float[] {expected, expected}, moe.forward(input).toFloatArray(), EPS);
      assertTrue(moe.parameters().containsKey("expert0.gateProj.weight"));
      assertTrue(moe.parameters().containsKey("router.weight"));
    }
  }

  @Test
  void exactTiesChooseFirstExpertAcrossTokensAndRuns() {
    try (MLXScope scope = new MLXScope()) {
      MoeMlp moe =
          new MoeMlp(
              scope,
              router(scope, 0, 0, 0),
              List.of(constantExpert(scope, 3), constantExpert(scope, 5), constantExpert(scope, 7)),
              1);
      MLXArray input = MLX.array(scope, new float[12], new int[] {2, 6, 1});
      for (int run = 0; run < 100; run++) {
        float[] actual = moe.forward(input).toFloatArray();
        for (float value : actual) {
          assertEquals(3, value, EPS);
        }
      }
    }
  }

  @Test
  void topTwoWeightsRenormaliseAndIgnoreUnselectedInfiniteExpert() {
    try (MLXScope scope = new MLXScope()) {
      List<GatedMlp> experts = new ArrayList<>();
      experts.add(constantExpert(scope, 2));
      experts.add(constantExpert(scope, 4));
      experts.add(constantExpert(scope, 100));
      experts.add(infiniteExpert(scope));
      MoeMlp moe = new MoeMlp(scope, router(scope, 2, 1, 0, -1), experts, 2);
      MLXArray input = MLX.array(scope, new float[] {1}, new int[] {1, 1, 1});
      float expected = (float) ((2 * Math.exp(2) + 4 * Math.E) / (Math.exp(2) + Math.E));
      float actual = moe.forward(input).toFloatArray()[0];
      assertTrue(Float.isFinite(actual));
      assertEquals(expected, actual, EPS);
    }
  }

  @Test
  void denseResultMatchesPerTokenReferenceWithHiddenFour() {
    try (MLXScope scope = new MLXScope()) {
      float[] input = {1, 2, 0.5f, -1, 0.25f, -0.5f, 1.5f, 2};
      float[] routerWeights = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
      Linear router = new Linear(scope, MLX.array(scope, routerWeights, new int[] {3, 4}), null);
      MoeMlp moe =
          new MoeMlp(
              scope,
              router,
              List.of(diagonalExpert(scope, 1), diagonalExpert(scope, 2), diagonalExpert(scope, 3)),
              2);
      float[] actual = moe.forward(MLX.array(scope, input, new int[] {1, 2, 4})).toFloatArray();
      float[] expected = new float[8];
      for (int token = 0; token < 2; token++) {
        float[] logits = {input[token * 4], input[token * 4 + 1], input[token * 4 + 2]};
        int first = 0;
        int second = 1;
        for (int expert = 0; expert < 3; expert++) {
          if (logits[expert] > logits[first]) {
            second = first;
            first = expert;
          } else if (expert != first && (second == first || logits[expert] > logits[second])) {
            second = expert;
          }
        }
        double firstWeight = Math.exp(logits[first]);
        double secondWeight = Math.exp(logits[second]);
        double scale =
            ((first + 1) * firstWeight + (second + 1) * secondWeight)
                / (firstWeight + secondWeight);
        for (int hidden = 0; hidden < 4; hidden++) {
          float value = input[token * 4 + hidden];
          expected[token * 4 + hidden] = (float) (scale * value * value / (1 + Math.exp(-value)));
        }
      }
      assertArrayEquals(expected, actual, EPS);
    }
  }

  @Test
  void bf16OutputUsesHiddenStateDtype() {
    try (MLXScope scope = new MLXScope()) {
      MoeMlp moe =
          new MoeMlp(
              scope, bf16Router(scope), List.of(bf16Expert(scope, 2), bf16Expert(scope, 4)), 2);
      MLXArray input =
          MLX.astype(MLX.array(scope, new float[] {1}, new int[] {1, 1, 1}), DType.BFLOAT16);
      assertEquals(DType.BFLOAT16, moe.forward(input).dtype());
    }
  }

  private static Linear router(MLXScope scope, float... biases) {
    return new Linear(
        scope,
        MLX.array(scope, new float[biases.length], new int[] {biases.length, 1}),
        MLX.array(scope, biases, new int[] {biases.length}));
  }

  private static GatedMlp constantExpert(MLXScope scope, float value) {
    return expert(scope, 1, value / SILU_ONE);
  }

  private static GatedMlp infiniteExpert(MLXScope scope) {
    return expert(scope, 100, Float.MAX_VALUE);
  }

  private static GatedMlp diagonalExpert(MLXScope scope, float multiplier) {
    float[] identity = new float[16];
    float[] scaled = new float[16];
    for (int i = 0; i < 4; i++) {
      identity[i * 4 + i] = 1;
      scaled[i * 4 + i] = multiplier;
    }
    return new GatedMlp(
        scope,
        new Linear(scope, MLX.array(scope, identity, new int[] {4, 4}), null),
        new Linear(scope, MLX.array(scope, identity, new int[] {4, 4}), null),
        new Linear(scope, MLX.array(scope, scaled, new int[] {4, 4}), null),
        Activation.SILU);
  }

  private static Linear bf16Router(MLXScope scope) {
    return new Linear(
        scope,
        MLX.astype(MLX.array(scope, new float[2], new int[] {2, 1}), DType.BFLOAT16),
        MLX.astype(MLX.array(scope, new float[] {1, 0}, new int[] {2}), DType.BFLOAT16));
  }

  private static GatedMlp bf16Expert(MLXScope scope, float value) {
    return new GatedMlp(
        scope,
        bf16Linear(scope, 0, 1),
        bf16Linear(scope, 0, 1),
        bf16Linear(scope, value / SILU_ONE, 0),
        Activation.SILU);
  }

  private static Linear bf16Linear(MLXScope scope, float weight, float bias) {
    return new Linear(
        scope,
        MLX.astype(MLX.array(scope, new float[] {weight}, new int[] {1, 1}), DType.BFLOAT16),
        MLX.astype(MLX.array(scope, new float[] {bias}, new int[] {1}), DType.BFLOAT16));
  }

  private static GatedMlp expert(MLXScope scope, float gateBias, float downWeight) {
    return new GatedMlp(
        scope,
        linear(scope, 0, gateBias),
        linear(scope, 0, 1),
        linear(scope, downWeight, 0),
        Activation.SILU);
  }

  private static Linear linear(MLXScope scope, float weight, float bias) {
    return new Linear(
        scope,
        MLX.array(scope, new float[] {weight}, new int[] {1, 1}),
        MLX.array(scope, new float[] {bias}, new int[] {1}));
  }
}
