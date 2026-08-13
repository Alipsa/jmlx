package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Tests for {@link ModuleGrad}, the module-aware autograd wrapper (req/phase4-plan.md §6,
 * req/plans/phase4-m2-plan.md Task 2).
 */
@EnabledIfNativeAvailable
class ModuleGradTest {

  private static final float EPS = 1e-4f;

  /**
   * {@code params[0]} = weight [1,3], {@code params[1]} = bias [1]; {@code inputs[0]} = x [1,3],
   * {@code inputs[1]} = target [1,1]. {@code loss = sum((x @ weight.T + bias - target)^2)}.
   */
  private static MLXArray[] mseLoss(MLXArray[] params, MLXArray[] inputs) {
    MLXArray weightT = MLXShape.transpose(params[0], inputs[0].scope());
    MLXArray pred = MLXOps.add(MLXOps.matmul(inputs[0], weightT), params[1]);
    MLXArray diff = MLXOps.subtract(pred, inputs[1]);
    return new MLXArray[] {MLXOps.sum(MLXOps.multiply(diff, diff))};
  }

  @Test
  void gradsMatchHandComputedValues() {
    try (MLXScope model = new MLXScope()) {
      MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
      MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
      Linear linear = new Linear(model, weight, bias);
      try (ModuleGrad mg = ModuleGrad.of(linear, ModuleGradTest::mseLoss)) {
        try (MLXScope step = model.newChild()) {
          MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray target = MLX.array(step, new float[] {0}, new int[] {1, 1});
          ModuleGrad.Result r = mg.apply(step, new MLXArray[] {x, target});
          assertArrayEquals(new float[] {36f}, r.value().toFloatArray(), EPS);
          assertEquals(List.of("weight", "bias"), List.copyOf(r.grads().keySet()));
          assertArrayEquals(new float[] {12, 24, 36}, r.grads().get("weight").toFloatArray(), EPS);
          assertArrayEquals(new float[] {12}, r.grads().get("bias").toFloatArray(), EPS);
        }
        // Restore path: after the step scope closed, forward() must still work --
        // proves rebind's finally restored the model-scope arrays.
        try (MLXScope after = model.newChild()) {
          MLXArray x2 = MLX.array(after, new float[] {1, 0, 0}, new int[] {1, 3});
          assertArrayEquals(new float[] {1f}, linear.forward(x2).toFloatArray(), EPS);
        }
      }
    }
  }

  @Test
  void restoreSurvivesAThrowingLoss() {
    try (MLXScope model = new MLXScope()) {
      MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
      MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
      Linear linear = new Linear(model, weight, bias);
      try (ModuleGrad mg =
          ModuleGrad.of(
              linear,
              (params, inputs) -> {
                throw new RuntimeException("loss boom");
              })) {
        try (MLXScope step = model.newChild()) {
          MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray target = MLX.array(step, new float[] {0}, new int[] {1, 1});
          assertThrows(RuntimeException.class, () -> mg.apply(step, new MLXArray[] {x, target}));
        }
      }
      try (MLXScope after = model.newChild()) {
        MLXArray x2 = MLX.array(after, new float[] {1, 0, 0}, new int[] {1, 3});
        assertArrayEquals(new float[] {1f}, linear.forward(x2).toFloatArray(), EPS);
      }
    }
  }

  @Test
  void reusedAcrossDifferentInputsGradsDifferAccordingly() {
    try (MLXScope model = new MLXScope()) {
      MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
      MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
      Linear linear = new Linear(model, weight, bias);
      try (ModuleGrad mg = ModuleGrad.of(linear, ModuleGradTest::mseLoss)) {
        float[] grads1;
        try (MLXScope step1 = model.newChild()) {
          MLXArray x1 = MLX.array(step1, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray t1 = MLX.array(step1, new float[] {0}, new int[] {1, 1});
          grads1 = mg.apply(step1, new MLXArray[] {x1, t1}).grads().get("weight").toFloatArray();
        }
        try (MLXScope step2 = model.newChild()) {
          MLXArray x2 = MLX.array(step2, new float[] {2, 2, 2}, new int[] {1, 3});
          MLXArray t2 = MLX.array(step2, new float[] {0}, new int[] {1, 1});
          float[] grads2 =
              mg.apply(step2, new MLXArray[] {x2, t2}).grads().get("weight").toFloatArray();
          assertNotEquals(grads1[0], grads2[0]);
        }
      }
    }
  }

  @Test
  void updateThenApplyReflectsNewWeights() {
    try (MLXScope model = new MLXScope()) {
      MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
      MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
      Linear linear = new Linear(model, weight, bias);
      try (ModuleGrad mg = ModuleGrad.of(linear, ModuleGradTest::mseLoss)) {
        float[] gradsBefore;
        try (MLXScope step1 = model.newChild()) {
          MLXArray x = MLX.array(step1, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray t = MLX.array(step1, new float[] {0}, new int[] {1, 1});
          gradsBefore = mg.apply(step1, new MLXArray[] {x, t}).grads().get("weight").toFloatArray();
        }
        linear.update(Map.of("weight", MLX.array(model, new float[] {2, 2, 2}, new int[] {1, 3})));
        try (MLXScope step2 = model.newChild()) {
          MLXArray x = MLX.array(step2, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray t = MLX.array(step2, new float[] {0}, new int[] {1, 1});
          float[] gradsAfter =
              mg.apply(step2, new MLXArray[] {x, t}).grads().get("weight").toFloatArray();
          assertNotEquals(gradsBefore[0], gradsAfter[0]);
        }
      }
    }
  }

  @Test
  void treeWithNoParametersThrows() {
    try (MLXScope model = new MLXScope()) {
      Module empty = new Module(model) {};
      assertThrows(
          IllegalStateException.class, () -> ModuleGrad.of(empty, (params, inputs) -> inputs));
    }
  }

  /**
   * Mirrors {@code MLXGradTest.crossThreadApplyThrows}: confinement is enforced transitively
   * through the wrapped {@code MLXGrad.Fn} either way, but the diagnostic must name {@code
   * ModuleGrad} -- the API the caller actually misused -- not the internal {@code Fn} it delegates
   * to.
   */
  @Test
  void crossThreadApplyThrowsNamingModuleGrad() throws InterruptedException {
    try (MLXScope model = new MLXScope()) {
      MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
      MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
      Linear linear = new Linear(model, weight, bias);
      try (ModuleGrad mg = ModuleGrad.of(linear, ModuleGradTest::mseLoss);
          MLXScope step = model.newChild()) {
        MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {1, 3});
        MLXArray target = MLX.array(step, new float[] {0}, new int[] {1, 1});
        Throwable[] caught = new Throwable[1];
        Thread other =
            new Thread(
                () -> {
                  try {
                    mg.apply(step, new MLXArray[] {x, target});
                  } catch (Throwable t) {
                    caught[0] = t;
                  }
                });
        other.start();
        other.join();
        assertInstanceOf(IllegalStateException.class, caught[0]);
        assertTrue(
            caught[0].getMessage().contains("ModuleGrad"),
            "expected message to name ModuleGrad, got: " + caught[0].getMessage());
      }
    }
  }
}
