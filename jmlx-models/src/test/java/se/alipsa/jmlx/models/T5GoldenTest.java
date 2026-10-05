package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.StaticKVCache;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

@EnabledIfNativeAvailable
@Tag("full-float32")
class T5GoldenTest {
  private static final ObjectMapper JSON = new ObjectMapper();

  @Test
  void encoderTeacherForcedAndCachedLogitsMatchHf() throws Exception {
    Path goldens = BertGoldenTest.root().resolve("tools/hf-reference/goldens");
    for (String family : List.of("t5-relu", "t5-gated")) {
      JsonNode data = JSON.readTree(goldens.resolve(family + ".json").toFile());
      int[] source =
          BertGoldenTest.integers(data.path("source_ids")).stream()
              .mapToInt(Integer::intValue)
              .toArray();
      int[] mask =
          BertGoldenTest.integers(data.path("source_mask")).stream()
              .mapToInt(Integer::intValue)
              .toArray();
      int[] history =
          BertGoldenTest.integers(data.path("history")).stream()
              .mapToInt(Integer::intValue)
              .toArray();
      try (MLXScope scope = new MLXScope();
          MLXScope request = scope.newChild()) {
        T5Model model = T5Model.load(scope, goldens.resolve("checkpoints/" + family));
        MLXArray encoded;
        try (MLXScope stage = request.newChild()) {
          encoded = MLX.hoist(model.encode(stage, source, mask), request);
          MLX.eval(encoded);
        }
        assertArrayEquals(
            BertGoldenTest.floats(data.path("encoder")), encoded.toFloatArray(), 1e-4f, family);
        List<StaticKVCache> cross = model.project(request, encoded);
        try (MLXScope step = request.newChild()) {
          assertArrayEquals(
              BertGoldenTest.floats(data.path("logits")),
              model.forward(step, history, source.length, mask, cross, null).toFloatArray(),
              1e-4f,
              family);
        }
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < model.metadata().numHiddenLayers(); i++) {
          caches.add(new KVCache(request));
        }
        for (int i = 0; i < history.length; i++) {
          try (MLXScope step = request.newChild()) {
            MLXArray logits =
                model.forward(step, new int[] {history[i]}, source.length, mask, cross, caches);
            assertArrayEquals(
                BertGoldenTest.floats(data.path("cached_logits").get(i)),
                logits.toFloatArray(),
                1e-4f,
                family);
            for (KVCache cache : caches) {
              MLX.eval(cache.keys(), cache.values());
            }
          }
        }
        List<Integer> positions = BertGoldenTest.integers(data.path("relative_positions"));
        for (boolean bidirectional : List.of(true, false)) {
          for (int i = 0; i < positions.size(); i++) {
            assertEquals(
                data.path("buckets").path(bidirectional ? "True" : "False").get(i).intValue(),
                T5Model.relativeBucket(positions.get(i), bidirectional, 8, 16));
          }
        }
      }
    }
  }

  @Test
  void generationMatchesHfAndProjectsOncePerLayerPerRequest() throws Exception {
    Path goldens = BertGoldenTest.root().resolve("tools/hf-reference/goldens");
    for (String family : List.of("t5-relu", "t5-gated")) {
      JsonNode data = JSON.readTree(goldens.resolve(family + ".json").toFile());
      // HF generation fixture uses key-only padding; pretokenized generation attends all IDs.
      int[] source =
          BertGoldenTest.integers(data.path("source_ids")).stream()
              .limit(4)
              .mapToInt(Integer::intValue)
              .toArray();
      try (MLXScope scope = new MLXScope()) {
        T5Model model = T5Model.load(scope, goldens.resolve("checkpoints/" + family));
        List<StaticKVCache> observed = new ArrayList<>();
        model.projectionObserver((layer, cache) -> observed.add(cache));
        GenerationRequest request =
            new GenerationRequest(
                source,
                GenerationConfig.greedyDefaults(5, java.util.Set.of(1)),
                CancellationToken.NONE);
        GenerationResult result = model.generate(request, ignored -> {});
        assertEquals(
            BertGoldenTest.integers(data.path("greedy_ids"))
                .subList(1, data.path("greedy_ids").size()),
            result.generatedTokenIds(),
            family);
        assertEquals(2, observed.size());
        model.generate(
            new GenerationRequest(new int[] {5, 6, 1}, request.config(), CancellationToken.NONE),
            ignored -> {});
        assertEquals(4, observed.size());
        assertEquals(4, observed.stream().distinct().count());
        assertEquals(Arrays.stream(source).boxed().toList(), result.promptTokenIds());
        assertThrows(
            IllegalArgumentException.class,
            () ->
                model.generate(
                    request.withCachePolicy(GenerationCachePolicy.slidingWindowFromModel()),
                    ignored -> {}));
        GenerationRequest bounded =
            new GenerationRequest(
                    source,
                    GenerationConfig.greedyDefaults(1, java.util.Set.of()),
                    CancellationToken.NONE)
                .withCachePolicy(GenerationCachePolicy.full(1));
        assertEquals(1, model.generate(bounded, ignored -> {}).generatedTokenIds().size());
        int count = observed.size();
        GenerationRequest empty =
            new GenerationRequest(
                source,
                GenerationConfig.greedyDefaults(0, java.util.Set.of()),
                CancellationToken.NONE);
        assertEquals(0, model.generate(empty, ignored -> {}).generatedTokenIds().size());
        assertEquals(count, observed.size());
      }
    }
  }
}
