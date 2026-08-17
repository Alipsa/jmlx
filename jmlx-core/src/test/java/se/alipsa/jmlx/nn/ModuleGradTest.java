package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.ref.WeakReference;
import java.util.List;
import java.util.Map;
import java.util.SequencedMap;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXQuant;
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
   * This PR's round-12 review, low finding 1: the "tree has no parameters to differentiate" message
   * previously fired verbatim for this scenario too, even though the tree DOES have a parameter --
   * just none floating. Pins the corrected message, which must distinguish the two cases and name
   * the excluded parameter(s).
   */
  @Test
  void treeWithOnlyNonFloatingParametersThrowsANamedMessageNotTheEmptyTreeOne() {
    try (MLXScope model = new MLXScope()) {
      MLXArray intParam = MLX.array(model, new int[] {1, 2, 3}, new int[] {3});
      IntOnlyModule intOnly = new IntOnlyModule(model, intParam);
      IllegalStateException ex =
          assertThrows(
              IllegalStateException.class,
              () -> ModuleGrad.of(intOnly, (params, inputs) -> inputs));
      assertTrue(
          ex.getMessage().contains("count"),
          "expected message to name the excluded non-floating parameter, got: " + ex.getMessage());
      assertFalse(
          ex.getMessage().contains("no parameters to differentiate"),
          "message must distinguish 'has non-floating parameters' from 'has none at all', got: "
              + ex.getMessage());
    }
  }

  private static final class IntOnlyModule extends Module {
    IntOnlyModule(MLXScope scope, MLXArray intParam) {
      super(scope);
      param("count", intParam);
    }
  }

  private static final class TwoLayerTree extends Module {
    TwoLayerTree(MLXScope scope, Linear lin, QuantizedLinear ql) {
      super(scope);
      child("lin", lin);
      child("ql", ql);
    }
  }

  /**
   * This PR's round-8 review finding 1 established that a tree containing a {@code QuantizedLinear}
   * does not fail at the first {@code apply} call when the loss graph never reaches that layer's
   * quantized weight; round-10 review finding 3 went further: {@code ModuleGrad} now excludes a
   * non-floating-dtype parameter (in practice, {@code QuantizedLinear}'s {@code UINT32}-packed
   * {@code weight}) from differentiation entirely, by dtype, rather than merely tolerating whatever
   * nonsensical value the native autodiff machinery produced for an untouched one. This pins that:
   * an unused {@code QuantizedLinear} sibling's {@code weight} does not appear in {@code grads()}
   * at all, while its {@code scales}/{@code biases} still do (with real, if untrained-by-this-loss,
   * gradient values), alongside the {@code Linear} sibling's own real gradient.
   */
  @Test
  void applySucceedsWithAnUnusedQuantizedLinearSiblingExcludingItsWeightFromGrads() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {1f, 2f, 3f, 4f}, new int[] {2, 2});
      Linear lin = new Linear(scope, weight, null);

      float[] qw = new float[2 * 64];
      for (int i = 0; i < qw.length; i++) {
        qw[i] = (i % 7 - 3) * 0.3f;
      }
      MLXArray w = MLX.array(scope, qw, new int[] {2, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      QuantizedLinear ql = new QuantizedLinear(scope, q[0], q[1], q[2], null, 32, 4);

      TwoLayerTree tree = new TwoLayerTree(scope, lin, ql);
      try (ModuleGrad mg =
          ModuleGrad.of(
              tree, (params, inputs) -> new MLXArray[] {MLXOps.sum(lin.forward(inputs[0]))})) {
        MLXArray x = MLX.array(scope, new float[] {1f, 1f}, new int[] {1, 2});

        ModuleGrad.Result result = mg.apply(scope, new MLXArray[] {x});

        SequencedMap<String, MLXArray> grads = result.grads();
        assertEquals(DType.FLOAT32, grads.get("lin.weight").dtype());
        assertFalse(grads.containsKey("ql.weight"), "ql.weight must be excluded from grads()");
        assertEquals(DType.FLOAT32, grads.get("ql.scales").dtype());
        assertEquals(DType.FLOAT32, grads.get("ql.biases").dtype());
      }
    }
  }

  /**
   * This PR's round-11 review finding 3: {@code loss}'s {@code params} array is indexed by {@link
   * ModuleGrad#paramPaths()}, not by {@code tree.parameters()}'s own order, once the tree has any
   * excluded (non-floating) parameter -- {@code lin.weight} then {@code ql.weight} (excluded, a
   * {@code UINT32}-packed weight) then {@code ql.scales} then {@code ql.biases} in {@code
   * tree.parameters()}'s own order, but {@link ModuleGrad#paramPaths()} must skip over {@code
   * ql.weight} entirely, landing {@code ql.scales} at index 1 rather than index 2.
   */
  @Test
  void paramPathsSkipsTheExcludedNonFloatingParameterAndPreservesOrderOfTheRest() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray weight = MLX.array(scope, new float[] {1f, 2f, 3f, 4f}, new int[] {2, 2});
      Linear lin = new Linear(scope, weight, null);

      float[] qw = new float[2 * 64];
      for (int i = 0; i < qw.length; i++) {
        qw[i] = (i % 7 - 3) * 0.3f;
      }
      MLXArray w = MLX.array(scope, qw, new int[] {2, 64});
      MLXArray[] q = MLXQuant.quantize(w, 32, 4, "affine", null);
      QuantizedLinear ql = new QuantizedLinear(scope, q[0], q[1], q[2], null, 32, 4);

      TwoLayerTree tree = new TwoLayerTree(scope, lin, ql);
      try (ModuleGrad mg =
          ModuleGrad.of(
              tree, (params, inputs) -> new MLXArray[] {MLXOps.sum(lin.forward(inputs[0]))})) {
        assertEquals(List.of("lin.weight", "ql.scales", "ql.biases"), mg.paramPaths());
      }
    }
  }

  /**
   * This PR's round-11 review finding 2: {@link Module#update}/{@link Module#rebind} run no dtype
   * check of their own, so a parameter that was floating-dtype (and therefore included in {@link
   * ModuleGrad#paramPaths()}) at construction can be replaced afterward with a non-floating value
   * of the same shape -- without this fix, {@code apply} would feed that value straight into the
   * traced primal vector and fail deep inside the wrapped {@code MLXGrad.Fn} with an opaque native
   * error naming neither {@code ModuleGrad} nor the offending parameter. This pins the earlier,
   * named {@code IllegalStateException} this class now throws instead, confirmed to fire before
   * ever reaching the native call (not merely a different exception type for the same native
   * failure): the replacement here is INT32, and native's own "no gradient" error is specific to
   * quantized (UINT32) weights, so a native failure for this exact scenario would look nothing like
   * this test's assertion if this check were removed.
   */
  @Test
  void applyThrowsIfAFormerlyFloatingParameterIsReplacedWithANonFloatingValue() {
    try (MLXScope model = new MLXScope()) {
      MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
      MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
      Linear linear = new Linear(model, weight, bias);
      try (ModuleGrad mg = ModuleGrad.of(linear, ModuleGradTest::mseLoss)) {
        linear.update(Map.of("weight", MLX.array(model, new int[] {2, 2, 2}, new int[] {1, 3})));
        try (MLXScope step = model.newChild()) {
          MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {1, 3});
          MLXArray target = MLX.array(step, new float[] {0}, new int[] {1, 1});
          IllegalStateException ex =
              assertThrows(
                  IllegalStateException.class, () -> mg.apply(step, new MLXArray[] {x, target}));
          assertTrue(
              ex.getMessage().contains("weight"),
              "expected message to name the offending path, got: " + ex.getMessage());
        }
      }
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

  /**
   * Mirrors {@code MLXGradTest.cleanerBackstopRunsForAnEscapedFn}, but with a capturing body:
   * {@code ModuleGrad}'s closure body must not strongly reference the enclosing {@code ModuleGrad}
   * (which owns the very {@code MLXGrad.Fn} the body backs) or this {@code ModuleGrad}, its {@code
   * tree}, and the model's {@link MLXScope} would never become unreachable once a caller drops a
   * {@code ModuleGrad} without {@code close()}.
   *
   * <p>Polls a second, independent {@link WeakReference} on the model scope after the first
   * assertion passes: the release is two-stage (the {@code ModuleGrad} wrapper becoming unreachable
   * is only the first link -- {@code Fn} must then be collected, its {@code Holder} cleaned, and
   * the arena closed before {@code tree}/{@code model} themselves become unreachable and can run
   * their own backstops). Asserting only on the wrapper would pass even if that second stage were
   * broken.
   *
   * <p>Each stage gets its own, freshly-reset 20s deadline rather than sharing one: stage 2 runs
   * strictly after stage 1 (Fn collected, then Holder.closeAll() on the Cleaner thread, then
   * arena.close(), then tree/model released, then MLXScope's own backstop) and is therefore the
   * more GC-timing-sensitive of the two -- a shared deadline would let a slow stage 1 starve stage
   * 2's budget down to whatever was left, turning a working chain into a spurious failure.
   */
  @Test
  @Timeout(value = 45, unit = TimeUnit.SECONDS)
  void cleanerBackstopRunsForADetachedModuleGrad() throws InterruptedException {
    Detached detached = createDetachedModuleGrad();

    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
    while (detached.moduleGrad().get() != null && System.nanoTime() < deadline) {
      System.gc();
      Thread.sleep(50);
    }
    assertNull(detached.moduleGrad().get(), "escaped ModuleGrad was not collected within 20s");

    deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
    while (detached.model().get() != null && System.nanoTime() < deadline) {
      System.gc();
      Thread.sleep(50);
    }
    assertNull(
        detached.model().get(),
        "model scope was not collected within 20s after the ModuleGrad wrapping it was -- Fn's "
            + "Holder/arena chain must release tree/scope too");
  }

  // Both fields are WeakReferences, not the ModuleGrad/MLXScope themselves, so this record can be
  // held by the calling test method across the whole polling loop without itself keeping either
  // referent reachable.
  private record Detached(WeakReference<ModuleGrad> moduleGrad, WeakReference<MLXScope> model) {}

  // Isolated in its own frame so no local variable in the calling test method keeps mg, model, or
  // the tree it wraps reachable after this method returns. model is deliberately left open (never
  // closed), mirroring the failure scenario: a caller dropping a ModuleGrad without close() -- the
  // case the Cleaner backstop exists for.
  private static Detached createDetachedModuleGrad() {
    MLXScope model = new MLXScope();
    MLXArray weight = MLX.array(model, new float[] {1, 1, 1}, new int[] {1, 3});
    MLXArray bias = MLX.array(model, new float[] {0}, new int[] {1});
    Linear linear = new Linear(model, weight, bias);
    ModuleGrad mg = ModuleGrad.of(linear, ModuleGradTest::mseLoss);
    try (MLXScope step = model.newChild()) {
      MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {1, 3});
      MLXArray target = MLX.array(step, new float[] {0}, new int[] {1, 1});
      mg.apply(step, new MLXArray[] {x, target});
    }
    return new Detached(new WeakReference<>(mg), new WeakReference<>(model));
  }
}
