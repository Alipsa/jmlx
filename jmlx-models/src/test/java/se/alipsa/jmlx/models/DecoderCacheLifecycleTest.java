package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;

@EnabledIfNativeAvailable
class DecoderCacheLifecycleTest {
  @Test
  void repeatedGenerationOnOneLoadedModelKeepsOutputsStable(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      GenerationRequest request =
          new GenerationRequest(
              new int[] {1, 2},
              GenerationConfig.greedyDefaults(5, Set.of()),
              CancellationToken.NONE);
      var first = model.generate(request, ignored -> {}).generatedTokenIds();
      long firstActive = MLXMemory.activeBytes();
      for (int i = 0; i < 5; i++) {
        assertEquals(first, model.generate(request, ignored -> {}).generatedTokenIds());
      }
      assertTrue(
          MLXMemory.activeBytes() - firstActive <= 8L * 1024 * 1024,
          "repeated requests retained native activations after scope cleanup");
    }
  }

  @Test
  void resettingEveryLayerAllowsTheSamePrefillAgain(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      List<KVCache> caches = new ArrayList<>();
      for (int i = 0; i < model.config().numHiddenLayers(); i++) {
        caches.add(new KVCache(scope));
      }
      var ids = MLX.array(scope, new int[] {1, 2}, new int[] {1, 2});
      float[] first = model.forward(ids, caches).toFloatArray();
      caches.forEach(KVCache::reset);
      assertArrayEquals(first, model.forward(ids, caches).toFloatArray(), 1e-5f);
    }
  }
}
