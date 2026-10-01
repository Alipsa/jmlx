package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

@EnabledIfNativeAvailable
class DecoderCacheCapacityTest {
  @Test
  void longFullContextFitsExactCapacityAndMatchesReferenceIds() throws Exception {
    Path root =
        Path.of(System.getProperty("jmlx.repository.root"), "tools", "hf-reference", "goldens");
    JsonNode reference = new ObjectMapper().readTree(root.resolve("mistral-window.json").toFile());
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = MistralModel.load(scope, root.resolve("checkpoints/mistral"));
      GenerationRequest request =
          new GenerationRequest(
              new int[] {1, 7, 42, 3, 19, 5},
              GenerationConfig.greedyDefaults(16, Set.of()),
              CancellationToken.NONE);
      List<Integer> expected = new ArrayList<>();
      for (JsonNode step : reference.path("long_decode")) {
        expected.add(step.path("token_id").asInt());
      }
      assertEquals(
          expected,
          model
              .generate(request.withCachePolicy(GenerationCachePolicy.full(21)), ignored -> {})
              .generatedTokenIds());
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  request.withCachePolicy(GenerationCachePolicy.full(20)), ignored -> {}));
    }
  }

  @Test
  void exactGenerationCapacityIncludesOnlyFedBackTokens(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      GenerationRequest request =
          new GenerationRequest(
              new int[] {1, 2, 3},
              GenerationConfig.greedyDefaults(3, Set.of()),
              CancellationToken.NONE);
      assertEquals(
          3,
          model
              .generate(request.withCachePolicy(GenerationCachePolicy.full(5)), ignored -> {})
              .generatedTokenIds()
              .size());
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  request.withCachePolicy(GenerationCachePolicy.full(4)), ignored -> {}));
      GenerationRequest zero =
          new GenerationRequest(
              new int[] {1, 2, 3},
              GenerationConfig.greedyDefaults(0, Set.of()),
              CancellationToken.NONE);
      assertEquals(
          0,
          model
              .generate(zero.withCachePolicy(GenerationCachePolicy.full(1)), ignored -> {})
              .generatedTokenIds()
              .size());
    }
  }

  @Test
  void raggedCapacityRejectsBeforeAnyLayerAdvances(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      List<KVCache> caches = new ArrayList<>();
      for (int i = 0; i < model.config().numHiddenLayers(); i++) {
        caches.add(new KVCache(scope, KVCachePolicy.full(2)));
      }
      MLXArray prefill = MLX.array(scope, new int[] {0, 1, 2, 3}, new int[] {2, 2});
      model.forward(prefill, caches, new int[] {1, 2});
      assertEquals(1, caches.getFirst().nextPosition(0));
      assertEquals(2, caches.getFirst().nextPosition(1));
      MLXArray decode = MLX.array(scope, new int[] {4, 5}, new int[] {2, 1});
      assertThrows(
          IllegalArgumentException.class, () -> model.forward(decode, caches, new int[] {1, 1}));
      for (KVCache cache : caches) {
        assertEquals(2, cache.nextPosition(1));
      }
    }
  }

  @Test
  void raggedCacheOnUnbatchedForwardPointsToValidLengthsOverload(@TempDir Path dir)
      throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      List<KVCache> caches = new ArrayList<>();
      for (int i = 0; i < model.config().numHiddenLayers(); i++) {
        caches.add(new KVCache(scope));
      }
      model.forward(
          MLX.array(scope, new int[] {0, 1, 2, 3}, new int[] {2, 2}), caches, new int[] {1, 2});
      MLXArray decode = MLX.array(scope, new int[] {4, 5}, new int[] {2, 1});
      IllegalArgumentException failure =
          assertThrows(IllegalArgumentException.class, () -> model.forward(decode, caches));
      assertTrue(failure.getMessage().contains("validLengths"), failure.getMessage());
    }
  }

  @Test
  void resolvedPolicyMatchesWhatGenerationUses(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      assertEquals(KVCachePolicy.full(), model.resolveCachePolicy(GenerationCachePolicy.full()));
      assertThrows(
          IllegalArgumentException.class,
          () -> model.resolveCachePolicy(GenerationCachePolicy.slidingWindowFromModel()));
    }
  }

  @Test
  void unboundedTokenBudgetIsNotRejectedUpFront(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      int[] polls = {0};
      GenerationRequest request =
          new GenerationRequest(
              new int[] {1, 2},
              GenerationConfig.greedyDefaults(Integer.MAX_VALUE, Set.of()),
              () -> ++polls[0] > 3);
      GenerationResult result = model.generate(request, ignored -> {});
      assertEquals(FinishReason.CANCELLED, result.finishReason());
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.generate(
                  request.withCachePolicy(GenerationCachePolicy.full(10)), ignored -> {}));
    }
  }

  @Test
  void poisonedPaddedCacheReportsPoisonNotPadding(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = LlamaModel.load(scope, dir);
      List<KVCache> caches = new ArrayList<>();
      for (int i = 0; i < model.config().numHiddenLayers(); i++) {
        caches.add(new KVCache(scope));
      }
      model.forward(
          MLX.array(scope, new int[] {0, 1, 2, 3}, new int[] {2, 2}), caches, new int[] {1, 2});
      caches.forEach(KVCache::poison);
      MLXArray decode = MLX.array(scope, new int[] {4, 5}, new int[] {2, 1});
      IllegalStateException failure =
          assertThrows(IllegalStateException.class, () -> model.forward(decode, caches));
      assertTrue(failure.getMessage().contains("poisoned"), failure.getMessage());
    }
  }
}
