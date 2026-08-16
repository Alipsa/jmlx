package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SequencedMap;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * See req/phase4-plan.md §5-6 for the design {@link Module} implements; the class javadoc there
 * explains why {@link Module#param(String, MLXArray)}'s return value must not be cached by a
 * subclass.
 */
@EnabledIfNativeAvailable
class ModuleTest {

  private static final class Leaf extends Module {
    boolean notified;

    Leaf(MLXScope scope, MLXArray w) {
      super(scope);
      param("w", w);
    }

    @Override
    protected void onParametersUpdated() {
      notified = true;
    }
  }

  private static final class Branch extends Module {
    boolean notified;

    Branch(MLXScope scope, Module child) {
      super(scope);
      child("leaf", child);
    }

    @Override
    protected void onParametersUpdated() {
      notified = true;
    }
  }

  /** Two own parameters ("a", "b") then a child ("c"), for the insertion-order test. */
  private static final class OrderedModule extends Module {
    OrderedModule(MLXScope scope, MLXArray a, MLXArray b, Module c) {
      super(scope);
      param("a", a);
      param("b", b);
      child("c", c);
    }
  }

  /** A leaf whose {@code onParametersUpdated()} always throws, for the rollback tests below. */
  private static final class ThrowingLeaf extends Module {
    boolean notified;

    ThrowingLeaf(MLXScope scope, MLXArray w) {
      super(scope);
      param("w", w);
    }

    @Override
    protected void onParametersUpdated() {
      notified = true;
      throw new IllegalStateException("ThrowingLeaf always rejects an update");
    }
  }

  /** Two named children ("first", "second"), for the multi-module rollback test. */
  private static final class TwoChildBranch extends Module {
    TwoChildBranch(MLXScope scope, Module first, Module second) {
      super(scope);
      child("first", first);
      child("second", second);
    }
  }

  @Test
  void paramWithDuplicateNameThrows() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      IllegalStateException ex =
          assertThrows(
              IllegalStateException.class,
              () -> leaf.param("w", MLX.array(scope, new float[] {2f}, new int[] {1})));
      assertTrue(ex.getMessage().contains("w"));
    }
  }

  @Test
  void childWithDuplicateNameThrows() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Branch branch = new Branch(scope, new Leaf(scope, w));
      IllegalStateException ex =
          assertThrows(IllegalStateException.class, () -> branch.child("leaf", new Leaf(scope, w)));
      assertTrue(ex.getMessage().contains("leaf"));
    }
  }

  @Test
  void paramAfterFreezeThrowsOnTheFrozenModuleDirectly() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      leaf.freeze();
      assertThrows(IllegalStateException.class, () -> leaf.param("other", w));
    }
  }

  @Test
  void childAfterFreezeThrowsOnTheFrozenModuleDirectly() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Branch branch = new Branch(scope, new Leaf(scope, w));
      branch.freeze();
      assertThrows(IllegalStateException.class, () -> branch.child("other", new Leaf(scope, w)));
    }
  }

  @Test
  void freezeCascadesToAChildRegisteredBeforeTheAncestorsFreezeCall() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      Branch branch = new Branch(scope, leaf);

      branch.freeze();

      // leaf was registered as branch's child before branch.freeze() ran;
      // the cascade must still have frozen it.
      assertThrows(IllegalStateException.class, () -> leaf.child("other", new Leaf(scope, w)));
    }
  }

  @Test
  void parametersOnABranchWrappingALeafReturnsExactlyOneDottedEntry() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      Branch branch = new Branch(scope, leaf);

      SequencedMap<String, MLXArray> params = branch.parameters();

      assertEquals(1, params.size());
      assertTrue(params.containsKey("leaf.w"));
      assertSame(w, params.get("leaf.w"));
    }
  }

  @Test
  void parametersOrderIsInsertionOrderOwnParamsThenChildren() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1f}, new int[] {1});
      MLXArray b = MLX.array(scope, new float[] {2f}, new int[] {1});
      MLXArray w = MLX.array(scope, new float[] {3f}, new int[] {1});
      OrderedModule module = new OrderedModule(scope, a, b, new Leaf(scope, w));

      SequencedMap<String, MLXArray> params = module.parameters();

      assertIterableEquals(List.of("a", "b", "c.w"), params.sequencedKeySet());
    }
  }

  @Test
  void updateWritesEverythingBeforeNotifyingAndOnlyNotifiesTouchedModules() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      MLXArray newW = MLX.array(scope, new float[] {2f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      Branch branch = new Branch(scope, leaf);

      branch.update(Map.of("leaf.w", newW));

      assertTrue(leaf.notified);
      assertFalse(branch.notified);
      assertSame(newW, leaf.param("w"));
    }
  }

  @Test
  void updateWithUnknownPathThrowsIllegalArgumentExceptionNamingThePath() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      Branch branch = new Branch(scope, leaf);

      IllegalArgumentException ex =
          assertThrows(
              IllegalArgumentException.class, () -> branch.update(Map.of("leaf.missing", w)));
      assertTrue(ex.getMessage().contains("leaf.missing"));
    }
  }

  @Test
  void rebindAfterFreezeSucceedsAndDoesNotNotify() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      MLXArray newW = MLX.array(scope, new float[] {2f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      Branch branch = new Branch(scope, leaf);
      branch.freeze();

      SequencedMap<String, MLXArray> values = new LinkedHashMap<>();
      values.put("leaf.w", newW);
      branch.rebind(values);

      assertFalse(leaf.notified);
      assertFalse(branch.notified);
      assertSame(newW, leaf.param("w"));
    }
  }

  @Test
  void rebindWithUnknownPathThrowsIllegalArgumentException() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      Branch branch = new Branch(scope, leaf);

      SequencedMap<String, MLXArray> values = new LinkedHashMap<>();
      values.put("leaf.missing", w);
      assertThrows(IllegalArgumentException.class, () -> branch.rebind(values));
    }
  }

  /**
   * Regression test: {@code update} used to write each entry as it resolved it, so a later entry's
   * bad path left earlier entries already written with no notification fired (the notify pass only
   * runs after the whole loop returns). Resolving every path before writing any of them means a bad
   * entry anywhere leaves every parameter -- including ones with a valid path earlier in iteration
   * order -- untouched.
   */
  @Test
  void updateWithAMixOfValidAndInvalidPathsAppliesNoWritesAndDoesNotNotify() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      MLXArray newW = MLX.array(scope, new float[] {2f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      Branch branch = new Branch(scope, leaf);

      SequencedMap<String, MLXArray> values = new LinkedHashMap<>();
      values.put("leaf.w", newW);
      values.put("leaf.missing", newW);

      assertThrows(IllegalArgumentException.class, () -> branch.update(values));

      assertFalse(leaf.notified);
      assertSame(w, leaf.param("w"));
    }
  }

  /**
   * Regression test for PR #11 round-6 review finding 1: {@code onParametersUpdated()} runs after
   * the write, by contract ("called after update writes"), so a throwing override cannot be
   * validated-before-write the way an unknown path or {@code null} value can be. Before this fix,
   * the throw left the new (rejected) value permanently installed with no way to recover the
   * previous binding -- exactly what a real subclass ({@code QuantizedLinear}) started doing in
   * this PR. This pins that {@code update} now rolls the write back to the pre-call value first.
   */
  @Test
  void updateRollsBackTheWriteWhenOnParametersUpdatedThrows() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      MLXArray newW = MLX.array(scope, new float[] {2f}, new int[] {1});
      ThrowingLeaf leaf = new ThrowingLeaf(scope, w);

      assertThrows(IllegalStateException.class, () -> leaf.update(Map.of("w", newW)));

      assertTrue(leaf.notified);
      assertSame(w, leaf.param("w"));
    }
  }

  /**
   * The composed-tree case: two siblings are both touched by one {@code update} call; the first's
   * {@code onParametersUpdated()} succeeds before the second's throws. The whole call is
   * all-writes-and-all-notifies-succeed or none of it takes effect, so the first sibling's already-
   * successful write is rolled back too, not just the one that threw -- and the second sibling's
   * write, made before either notification ran, is rolled back as well.
   */
  @Test
  void updateRollsBackEverySiblingWhenOneOnParametersUpdatedThrows() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray firstW = MLX.array(scope, new float[] {1f}, new int[] {1});
      MLXArray secondW = MLX.array(scope, new float[] {10f}, new int[] {1});
      MLXArray newFirstW = MLX.array(scope, new float[] {2f}, new int[] {1});
      MLXArray newSecondW = MLX.array(scope, new float[] {20f}, new int[] {1});
      Leaf first = new Leaf(scope, firstW);
      ThrowingLeaf second = new ThrowingLeaf(scope, secondW);
      TwoChildBranch branch = new TwoChildBranch(scope, first, second);

      assertThrows(
          IllegalStateException.class,
          () -> branch.update(Map.of("first.w", newFirstW, "second.w", newSecondW)));

      assertTrue(first.notified);
      assertTrue(second.notified);
      assertSame(firstW, first.param("w"));
      assertSame(secondW, second.param("w"));
    }
  }

  /** Same atomicity guarantee as {@code update}, for {@code rebind}. */
  @Test
  void rebindWithAMixOfValidAndInvalidPathsAppliesNoWrites() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      MLXArray newW = MLX.array(scope, new float[] {2f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      Branch branch = new Branch(scope, leaf);

      SequencedMap<String, MLXArray> values = new LinkedHashMap<>();
      values.put("leaf.w", newW);
      values.put("leaf.missing", newW);

      assertThrows(IllegalArgumentException.class, () -> branch.rebind(values));

      assertSame(w, leaf.param("w"));
    }
  }

  @Test
  void paramRejectsANameContainingADot() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);
      MLXArray other = MLX.array(scope, new float[] {2f}, new int[] {1});
      assertThrows(IllegalArgumentException.class, () -> leaf.param("a.b", other));
    }
  }

  @Test
  void childRejectsANameContainingADot() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Branch branch = new Branch(scope, new Leaf(scope, w));
      assertThrows(IllegalArgumentException.class, () -> branch.child("a.b", new Leaf(scope, w)));
    }
  }

  @Test
  void childRejectsANullModule() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Branch branch = new Branch(scope, new Leaf(scope, w));
      assertThrows(NullPointerException.class, () -> branch.child("other", null));
    }
  }

  /**
   * {@code Map.of} rejects a {@code null} key outright, so this uses a {@code LinkedHashMap} -- the
   * kind of map a checkpoint loader would plausibly build -- to reach {@code resolveAll}'s own key
   * null-check rather than one {@code Map.of} would have caught for free.
   */
  @Test
  void updateWithANullKeyThrowsNullPointerException() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);

      Map<String, MLXArray> byPath = new LinkedHashMap<>();
      byPath.put(null, w);

      assertThrows(NullPointerException.class, () -> leaf.update(byPath));
    }
  }

  @Test
  void rebindWithANullKeyThrowsNullPointerException() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray w = MLX.array(scope, new float[] {1f}, new int[] {1});
      Leaf leaf = new Leaf(scope, w);

      SequencedMap<String, MLXArray> values = new LinkedHashMap<>();
      values.put(null, w);

      assertThrows(NullPointerException.class, () -> leaf.rebind(values));
    }
  }
}
