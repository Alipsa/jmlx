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
}
