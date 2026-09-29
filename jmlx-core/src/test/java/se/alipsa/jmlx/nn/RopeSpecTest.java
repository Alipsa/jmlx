package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class RopeSpecTest {
  @Test
  void baseMatchesThetaPower() {
    assertArrayEquals(new double[] {1, 100}, new RopeSpec.Base(10_000).frequencies(4, 1), 1e-9);
  }

  @Test
  void linearUsesPositionScale() {
    RopeSpec linear = new RopeSpec.Linear(10_000, 4);
    assertEquals(0.25f, linear.scale());
    assertArrayEquals(new RopeSpec.Base(10_000).frequencies(4, 1), linear.frequencies(4, 1), 1e-9);
  }

  @Test
  void dynamicNtkChangesBeyondOriginalContext() {
    RopeSpec dynamic = new RopeSpec.DynamicNtk(10_000, 2, 8);
    assertArrayEquals(new RopeSpec.Base(10_000).frequencies(4, 1), dynamic.frequencies(4, 8), 1e-9);
    assertArrayEquals(new double[] {1, 300}, dynamic.frequencies(4, 16), 1e-6);
    assertFalse(dynamic.isStatic());
  }

  @Test
  void llama3PreservesHighAndDividesLowFrequencies() {
    RopeSpec base = new RopeSpec.Base(500_000);
    RopeSpec llama3 = new RopeSpec.Llama3(500_000, 8, 1, 4, 8192);
    double[] old = base.frequencies(64, 1);
    double[] scaled = llama3.frequencies(64, 1);
    assertEquals(old[0], scaled[0], 1e-9);
    assertEquals(old[old.length - 1] * 8, scaled[scaled.length - 1], 1e-6);
    assertTrue(llama3.isStatic());
  }

  @Test
  void yarnReportsAttentionFactor() {
    assertEquals(
        (float) (0.1 * Math.log(4) + 1),
        new RopeSpec.Yarn(10_000, 4, 4096, 32, 1, 0, 0, null, true).attentionScaling(),
        1e-6f);
    assertEquals(
        1.5f, new RopeSpec.Yarn(10_000, 4, 4096, 32, 1, 0, 0, 1.5f, true).attentionScaling());
  }
}
