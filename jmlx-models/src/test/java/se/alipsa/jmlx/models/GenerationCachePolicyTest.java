package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.nn.KVCachePolicy;

class GenerationCachePolicyTest {
  @Test
  void resolvesAgainstCheckpointWindowRatherThanFamily() {
    assertEquals(KVCachePolicy.full(), GenerationCachePolicy.full().resolve(null));
    assertEquals(
        KVCachePolicy.slidingWindow(4), GenerationCachePolicy.slidingWindowFromModel().resolve(4));
    assertThrows(
        IllegalArgumentException.class,
        () -> GenerationCachePolicy.slidingWindowFromModel().resolve(null));
    assertThrows(
        IllegalArgumentException.class, () -> GenerationCachePolicy.slidingWindow(8).resolve(4));
  }
}
