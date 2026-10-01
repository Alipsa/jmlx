package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class KVCachePolicyTest {
  @Test
  void validatesResolvedLimits() {
    assertEquals(0, KVCachePolicy.full().limit());
    assertFalse(KVCachePolicy.full(8).evicts());
    assertTrue(KVCachePolicy.slidingWindow(1).evicts());
    assertThrows(IllegalArgumentException.class, () -> KVCachePolicy.full(0));
    assertThrows(IllegalArgumentException.class, () -> KVCachePolicy.slidingWindow(0));
  }

  @Test
  void quantizedRetentionIsNamedUnsupportedCapability() {
    UnsupportedOperationException failure =
        assertThrows(UnsupportedOperationException.class, () -> KVCachePolicy.quantized(4, 32));
    assertTrue(failure.getMessage().contains("SDPA requires float K/V"));
  }

  @Test
  void requireCapacityBoundsFullButNotSliding() {
    KVCachePolicy.full(4).requireCapacity(4, "x");
    KVCachePolicy.full().requireCapacity(Integer.MAX_VALUE, "x");
    KVCachePolicy.slidingWindow(4).requireCapacity(1000, "x");
    assertThrows(
        IllegalArgumentException.class, () -> KVCachePolicy.full(4).requireCapacity(5, "x"));
    assertThrows(
        IllegalArgumentException.class,
        () -> KVCachePolicy.full().requireCapacity(Integer.MAX_VALUE + 1L, "x"));
    assertThrows(
        IllegalArgumentException.class,
        () -> KVCachePolicy.slidingWindow(4).requireCapacity(Integer.MAX_VALUE + 1L, "x"));
  }
}
