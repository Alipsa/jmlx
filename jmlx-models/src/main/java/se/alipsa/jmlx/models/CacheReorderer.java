package se.alipsa.jmlx.models;

import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;

/** Package-private seam over the per-layer cache reorder, so tests can inject a failure. */
@FunctionalInterface
interface CacheReorderer {
  CacheReorderer NATIVE = (old, layer, indices, destination) -> old.reorder(indices, destination);

  KVCache reorder(KVCache old, int layer, int[] indices, MLXScope destination);
}
