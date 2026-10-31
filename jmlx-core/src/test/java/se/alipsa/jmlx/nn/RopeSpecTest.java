package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
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

  @Test
  void suMultipliesTheBasePeriodsByTheShortTable() {
    // Base periods for theta 10_000 over 8 dims are 10_000^(i/4): 1, 10, 100, 1000.
    RopeSpec su =
        new RopeSpec.Su(10_000, 8192, 4096, new float[] {2, 3, 4, 5}, new float[] {1, 2, 4, 8});
    assertArrayEquals(new double[] {2, 30, 400, 5000}, su.frequencies(8, 1), 1e-6);
    assertEquals(1f, su.scale(), 0f);
  }

  @Test
  void suSelectsTheLongTableOnlyStrictlyBeyondTheOriginalMax() {
    RopeSpec su =
        new RopeSpec.Su(10_000, 8192, 4096, new float[] {2, 3, 4, 5}, new float[] {1, 2, 4, 8});
    assertArrayEquals(new double[] {2, 30, 400, 5000}, su.frequencies(8, 4096), 1e-6);
    assertArrayEquals(new double[] {1, 20, 400, 8000}, su.frequencies(8, 4097), 1e-6);
    assertFalse(su.isStatic());
  }

  @Test
  void suComputesTheStaticAttentionScaling() {
    RopeSpec su =
        new RopeSpec.Su(10_000, 8192, 1024, new float[] {2, 3, 4, 5}, new float[] {1, 2, 4, 8});
    // 8192 / 1024 = 8 and ln(8) / ln(1024) = ln(2^3) / ln(2^10) = 3 / 10.
    assertEquals((float) Math.sqrt(1 + 3 / 10.0), su.attentionScaling(), 1e-6f);
  }

  @Test
  void suScalingIsOneWhenTheContextIsNotExtended() {
    RopeSpec su =
        new RopeSpec.Su(10_000, 4096, 4096, new float[] {2, 3, 4, 5}, new float[] {1, 2, 4, 8});
    assertEquals(1f, su.attentionScaling(), 0f);
  }

  @Test
  void suRejectsInvalidArguments() {
    float[] factors = {2, 3, 4, 5};
    assertThrows(
        IllegalArgumentException.class, () -> new RopeSpec.Su(-1, 8192, 4096, factors, factors));
    assertThrows(
        IllegalArgumentException.class, () -> new RopeSpec.Su(10_000, 0, 4096, factors, factors));
    assertThrows(
        IllegalArgumentException.class, () -> new RopeSpec.Su(10_000, 8192, 0, factors, factors));
    assertThrows(
        IllegalArgumentException.class,
        () -> new RopeSpec.Su(10_000, 8192, 4096, new float[] {2, -3, 4, 5}, factors));
    assertThrows(
        IllegalArgumentException.class,
        () -> new RopeSpec.Su(10_000, 8192, 4096, factors, new float[] {2, 3}));
  }

  @Test
  void suRejectsAFactorTableThatDoesNotFitTheRotaryDims() {
    RopeSpec su = new RopeSpec.Su(10_000, 8192, 4096, new float[] {2, 3}, new float[] {1, 2});
    assertThrows(IllegalArgumentException.class, () -> su.frequencies(8, 1));
  }

  @Test
  void suIsImmutableAndComparesByValue() {
    float[] shorts = {2, 3, 4, 5};
    float[] longs = {1, 2, 4, 8};
    RopeSpec.Su su = new RopeSpec.Su(10_000, 8192, 4096, shorts, longs);
    shorts[0] = 99;
    longs[0] = 99;
    assertArrayEquals(new float[] {2, 3, 4, 5}, su.shortFactors(), 0f);
    assertArrayEquals(new float[] {1, 2, 4, 8}, su.longFactors(), 0f);
    // Accessors hand out copies: mutating a returned table must not change the spec's math.
    su.shortFactors()[1] = 99;
    su.longFactors()[1] = 99;
    assertArrayEquals(new double[] {2, 30, 400, 5000}, su.frequencies(8, 1), 1e-6);
    assertEquals(
        su,
        new RopeSpec.Su(10_000, 8192, 4096, new float[] {2, 3, 4, 5}, new float[] {1, 2, 4, 8}));
    assertEquals(
        su.hashCode(),
        new RopeSpec.Su(10_000, 8192, 4096, new float[] {2, 3, 4, 5}, new float[] {1, 2, 4, 8})
            .hashCode());
    assertNotEquals(
        su,
        new RopeSpec.Su(10_000, 8192, 4096, new float[] {2, 3, 4, 6}, new float[] {1, 2, 4, 8}));
    // Readable tables, not array identity hashes.
    assertTrue(su.toString().contains("shortFactors=[2.0, 3.0, 4.0, 5.0]"));
    assertTrue(su.toString().contains("longFactors=[1.0, 2.0, 4.0, 8.0]"));
  }
}
