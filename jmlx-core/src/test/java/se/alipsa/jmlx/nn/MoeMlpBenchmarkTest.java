package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestReporter;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXRandom;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Opt-in: gathered vs dense MoE wall time. Enable with {@code -Djmlx.benchmark=true}. */
@EnabledIfNativeAvailable
@EnabledIfSystemProperty(named = "jmlx.benchmark", matches = "true")
class MoeMlpBenchmarkTest {

  private static final int H = 1024;
  private static final int F = 3584;
  private static final int E = 8;
  private static final int K = 2;
  private static final int ROUNDS = 7;
  private static final int RUNS_PER_ROUND = 10;

  @Test
  void gatheredDecodeIsFasterThanDense(TestReporter reporter) {
    try (MLXScope model = new MLXScope()) {
      MLXArray[] gate = new MLXArray[E];
      MLXArray[] up = new MLXArray[E];
      MLXArray[] down = new MLXArray[E];
      List<GatedMlp> dense = new ArrayList<>();
      for (int e = 0; e < E; e++) {
        gate[e] = MLXRandom.normal(model, new int[] {F, H}, DType.BFLOAT16, 0, 0.02f);
        up[e] = MLXRandom.normal(model, new int[] {F, H}, DType.BFLOAT16, 0, 0.02f);
        down[e] = MLXRandom.normal(model, new int[] {H, F}, DType.BFLOAT16, 0, 0.02f);
        dense.add(
            new GatedMlp(
                model,
                new Linear(model, gate[e], null),
                new Linear(model, up[e], null),
                new Linear(model, down[e], null),
                Activation.SILU));
      }
      // One weight array, two Linear instances: Module.child forbids registering one module
      // instance under two parents (Module.java, child() javadoc: undefined results). Sharing
      // the underlying MLXArray is fine; Linear only reads it.
      MLXArray routerWeight = MLXRandom.normal(model, new int[] {E, H}, DType.BFLOAT16, 0, 0.02f);
      MoeMlp gathered =
          new MoeMlp(
              model,
              new Linear(model, routerWeight, null),
              new SwitchGlu(
                  model,
                  MLXShape.stack(gate, 0),
                  MLXShape.stack(up, 0),
                  MLXShape.stack(down, 0),
                  Activation.SILU),
              K);
      DenseMoeReference reference =
          new DenseMoeReference(model, new Linear(model, routerWeight, null), dense, K);
      MLX.eval(gathered.parameters().values().toArray(MLXArray[]::new));

      for (int tokens : new int[] {1, 128}) {
        double g = millis(model, gathered, tokens);
        double d = millis(model, reference, tokens);
        reporter.publishEntry(
            "T=" + tokens,
            String.format("dense %.2f ms, gathered %.2f ms, speedup %.2fx", d, g, d / g));
        if (tokens == 1) {
          // Floor, not target: Java measures ~2.1x for the whole forward (plan, Verified facts 7).
          assertTrue(d / g >= 1.5, "decode speedup " + d / g + "x < 1.5x");
        }
      }
    }
  }

  /** Median over {@code ROUNDS} timed rounds, so one throttled or contended round cannot decide. */
  private static double millis(MLXScope model, UnaryLayer moe, int tokens) {
    for (int i = 0; i < 5; i++) {
      run(model, moe, tokens);
    }
    double[] rounds = new double[ROUNDS];
    for (int r = 0; r < ROUNDS; r++) {
      long start = System.nanoTime();
      for (int i = 0; i < RUNS_PER_ROUND; i++) {
        run(model, moe, tokens);
      }
      rounds[r] = (System.nanoTime() - start) / 1e6 / RUNS_PER_ROUND;
    }
    java.util.Arrays.sort(rounds);
    return rounds[ROUNDS / 2];
  }

  private static void run(MLXScope model, UnaryLayer moe, int tokens) {
    try (MLXScope step = model.newChild()) {
      MLXArray x = MLXRandom.normal(step, new int[] {1, tokens, H}, DType.BFLOAT16, 0, 1);
      MLX.eval(moe.forward(x));
    }
  }
}
