package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;

/** Tests both preflight atomicity and failures at the lazy evaluation boundary. */
@EnabledIfNativeAvailable
class DecoderForwardPoisoningTest {
  @Test
  void preflightFailureDoesNotMutateOrPoison(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      List<KVCache> caches = caches(scope, model);
      MLXArray ids = MLX.array(scope, new int[] {1}, new int[] {1, 1});
      assertThrows(IllegalArgumentException.class, () -> model.forward(ids, caches.subList(0, 1)));
      for (KVCache cache : caches) {
        assertEquals(0, cache.nextPosition());
        assertFalse(cache.isPoisoned());
      }
    }
  }

  @Test
  void injectedEvalFailurePoisonsAllLayersUntilReset(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      List<KVCache> caches = caches(scope, model);
      MLXArray ids = MLX.array(scope, new int[] {1}, new int[] {1, 1});
      model.setStepBoundaryEvaluatorForTest(
          arrays -> {
            throw new IllegalStateException("injected eval failure");
          });
      assertThrows(IllegalStateException.class, () -> model.forward(ids, caches));
      for (KVCache cache : caches) {
        assertTrue(cache.isPoisoned());
      }
      assertThrows(IllegalStateException.class, () -> model.forward(ids, caches));
      caches.forEach(KVCache::reset);
      GenerationRequest request =
          new GenerationRequest(
              new int[] {1}, GenerationConfig.greedyDefaults(1, Set.of()), CancellationToken.NONE);
      assertThrows(IllegalStateException.class, () -> model.generate(request, ignored -> {}));
      GenerationRequest sampled =
          new GenerationRequest(
              new int[] {1},
              GenerationConfig.samplingDefaults(1, 42L, 1f, Set.of()),
              CancellationToken.NONE);
      assertThrows(IllegalStateException.class, () -> model.generate(sampled, ignored -> {}));
      model.setStepBoundaryEvaluatorForTest(StepBoundaryEvaluator.NATIVE);
      model.forward(ids, caches);
      assertEquals(1, caches.getFirst().nextPosition());
      assertEquals(1, model.generate(request, ignored -> {}).generatedTokenIds().size());
    }
  }

  @Test
  void generationNonFiniteLogitsFailAfterSamplerEval(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    Path weights = dir.resolve("model.safetensors");
    Path replacement = dir.resolve("replacement.safetensors");
    try (MLXScope editing = new MLXScope()) {
      MLXIO.SafetensorsResult loaded = MLXIO.loadSafetensors(editing, weights.toString());
      Map<String, MLXArray> tensors = new LinkedHashMap<>(loaded.tensors());
      int[] shape = tensors.get("lm_head.weight").shape();
      tensors.put("lm_head.weight", MLX.full(editing, shape, Float.NaN, DType.FLOAT32));
      MLXIO.saveSafetensors(replacement.toString(), tensors, loaded.metadata());
    }
    Files.move(replacement, weights, StandardCopyOption.REPLACE_EXISTING);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      GenerationRequest request =
          new GenerationRequest(
              new int[] {1}, GenerationConfig.greedyDefaults(1, Set.of()), CancellationToken.NONE);
      IllegalStateException failure =
          assertThrows(IllegalStateException.class, () -> model.generate(request, ignored -> {}));
      assertTrue(failure.getMessage().contains("finite-logit"));
    }
  }

  private static List<KVCache> caches(MLXScope scope, DecoderModel model) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.config().numHiddenLayers(); i++) {
      caches.add(new KVCache(scope));
    }
    return caches;
  }
}
